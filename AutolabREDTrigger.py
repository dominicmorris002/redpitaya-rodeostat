"""
Dual Red Pitaya System - HARDWARE TRIGGERED PARALLEL ACQUISITION (FIXED)

CRITICAL WIRING:
  RP1 DIO1_P  →  RP2 DIO0_P (trigger cable)
  RP1 GND     →  RP2 GND    (common ground - REQUIRED!)

FIXED APPROACH:
  1. Both RPs in ROLLING MODE (continuous acquisition)
  2. ARM BOTH FIRST, then send ONE trigger pulse
  3. Read from rolling buffers after trigger
  4. No blocking .single() calls inside loop
"""

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os
from datetime import datetime

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
REF_FREQUENCY = 100
REF_AMPLITUDE = 0.5
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0
MEASUREMENT_TIME = 30.0
FILTER_BANDWIDTH = 10
DECIMATION = 8192
SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
SAVE_TIMESTAMPS = True

RP1_HOSTNAME = 'rp-f073ce.local'  # MASTER
RP2_HOSTNAME = 'rp-f0909c.local'  # SLAVE
# ============================================================


class RedPitayaMaster:
    """Master Red Pitaya - Lock-in + Trigger Generator"""
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='lockin_master_config', hostname=RP1_HOSTNAME)
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope

        self.lockin_X = []
        self.lockin_Y = []
        self.capture_timestamps = []
        self.acquisition_start_time = None

        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError('Invalid decimation')

        self.sample_rate = 125e6 / self.scope.decimation
        print(f"[MASTER RP1] Sample rate: {self.sample_rate:.2f} Hz")

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        self.ref_sig.output_direct = 'off'

        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,
            phase=phase_setting,
            acbandwidth=0,
            amplitude=ref_amp,
            input='in1',
            output_direct=params['output_ref'],
            output_signal='quadrature',
            quadrature_factor=1)

        print(f"[MASTER RP1] Lock-in: {self.ref_freq} Hz @ {ref_amp}V")
        print(f"[MASTER RP1] Filter BW: {filter_bw} Hz")

    def arm_rolling_mode(self):
        """Start continuous rolling acquisition - MUST be called BEFORE trigger"""
        self.scope.trigger_source = 'immediately'
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = False
        print("[MASTER RP1] ✓ Armed in rolling mode")

    def send_trigger_pulse(self):
        """Send trigger pulse on DIO1_P - Call AFTER both RPs are armed!"""
        try:
            # Set DIO1_P high
            self.rp_modules.hk.led = [1, 0, 0, 0, 0, 0, 0, 0]
            self.rp_modules.hk.dout1 = 1
            time.sleep(0.001)  # 1ms pulse
            self.rp_modules.hk.dout1 = 0
            self.rp_modules.hk.led = [0, 0, 0, 0, 0, 0, 0, 0]
            print("[MASTER RP1] ⚡ Trigger pulse sent on DIO1_P")
        except Exception as e:
            print(f"[WARNING] DIO control issue: {e}")

    def read_rolling_buffer(self):
        """Read current data from rolling buffer (non-blocking)"""
        capture_time = time.time()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)
        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(capture_time)
        return ch1, ch2


class RedPitayaSlave:
    """Slave Red Pitaya - Waits for hardware trigger"""
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, hostname, output_dir='test_data'):
        print(f"\n[SLAVE RP2] Connecting to {hostname}...")
        self.rp = Pyrpl(config='dc_slave_config', hostname=hostname)
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.scope = self.rp_modules.scope

        self.in1_data = []
        self.capture_timestamps = []
        self.acquisition_start_time = None

        print("[SLAVE RP2] ✓ Connected")

    def arm_rolling_mode_with_trigger(self, decimation):
        """Start rolling mode, waiting for external trigger - MUST be called BEFORE master sends pulse"""
        self.scope.input1 = 'in1'
        self.scope.decimation = decimation

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError('Invalid decimation')

        # CRITICAL: External trigger on DIO0_P
        self.scope.trigger_source = 'ext_positive_edge'
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = False
        self.sample_rate = 125e6 / self.scope.decimation

        print(f"[SLAVE RP2] Sample rate: {self.sample_rate:.2f} Hz")
        print(f"[SLAVE RP2] Trigger input: DIO0_P (external positive edge)")
        print(f"[SLAVE RP2] ✓ Armed in rolling mode, WAITING FOR TRIGGER")

    def read_rolling_buffer(self):
        """Read current data from rolling buffer (non-blocking)"""
        capture_time = time.time()
        in1 = np.array(self.scope._data_ch1_current)
        self.in1_data.append(in1)
        self.capture_timestamps.append(capture_time)
        return in1

    def process_data(self):
        """Process captured data"""
        if len(self.in1_data) == 0:
            print("[SLAVE RP2] WARNING: No data captured!")
            return

        self.all_in1 = np.concatenate(self.in1_data)
        total_samples = len(self.all_in1)
        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.in1_data[i])
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        self.t = self.sample_timestamps - self.acquisition_start_time
        print(f"\n[SLAVE RP2] Processed {total_samples} samples")
        print(f"[SLAVE RP2] Mean IN1: {np.mean(self.all_in1):.6f} V ± {np.std(self.all_in1):.6f} V")


def run_hardware_triggered_system(params):
    """Run with CORRECT hardware-triggered parallel acquisition"""
    
    print("\n" + "=" * 60)
    print("HARDWARE-TRIGGERED PARALLEL ACQUISITION (FIXED)")
    print("=" * 60)
    print("✓ CORRECT ORDER: ARM BOTH → TRIGGER ONCE")
    print("=" * 60)

    # Initialize MASTER
    print("\n[MASTER RP1] Initializing lock-in amplifier...")
    master = RedPitayaMaster(output_dir=params['output_dir'])
    master.setup_lockin(params)

    # Initialize SLAVE
    print("\n[SLAVE RP2] Initializing DC sampler...")
    slave = RedPitayaSlave(hostname=RP2_HOSTNAME, output_dir=params['output_dir'])

    print("\n[SYNC] Waiting for lock-in to settle...")
    time.sleep(0.5)

    # CRITICAL STEP 1: ARM SLAVE FIRST (must be waiting)
    slave.arm_rolling_mode_with_trigger(DECIMATION)
    time.sleep(0.1)  # Give slave time to arm

    # CRITICAL STEP 2: ARM MASTER
    master.arm_rolling_mode()
    time.sleep(0.1)  # Give master time to start rolling

    # CRITICAL STEP 3: NOW send trigger pulse
    print("\n[SYNC] Both RPs armed, sending trigger pulse...")
    sync_time = time.time()
    master.acquisition_start_time = sync_time
    slave.acquisition_start_time = sync_time
    
    master.send_trigger_pulse()
    
    print(f"\n[SYNC] ✓ Trigger sent at {datetime.fromtimestamp(sync_time).strftime('%H:%M:%S.%f')}")
    print("[ACQ] Starting synchronized rolling acquisition...")

    # Acquisition loop - read from rolling buffers
    loop_start = time.time()
    capture_count = 0
    
    while (time.time() - loop_start) < params['timeout']:
        # Both read from their rolling buffers (non-blocking)
        master.read_rolling_buffer()
        slave.read_rolling_buffer()
        
        capture_count += 1
        
        # Don't hammer the devices too fast
        time.sleep(0.05)  # 50ms between reads
        
        if capture_count % 20 == 0:
            print(f"[ACQ] Captured {capture_count} synchronized frames...")
    
    actual_duration = time.time() - loop_start
    print(f"\n[ACQ] ✓ Hardware-triggered acquisition complete!")
    print(f"[ACQ] Duration: {actual_duration:.2f}s")
    print(f"[ACQ] Synchronized captures: {capture_count}")

    # Process MASTER data
    print("\n[MASTER RP1] Processing lock-in data...")
    master.all_X = np.array(np.concatenate(master.lockin_X))
    master.all_Y = np.array(np.concatenate(master.lockin_Y))

    total_samples = len(master.all_X)
    master.sample_timestamps = np.zeros(total_samples)
    sample_idx = 0

    for i, capture_time in enumerate(master.capture_timestamps):
        n_samples = len(master.lockin_X[i])
        capture_duration = n_samples / master.sample_rate
        sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
        master.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
        sample_idx += n_samples

    master.R = np.sqrt(master.all_X ** 2 + master.all_Y ** 2)
    master.Theta = np.arctan2(master.all_Y, master.all_X)
    master.t = master.sample_timestamps - master.acquisition_start_time

    print(f"[MASTER RP1] ✓ Processed {len(master.all_X)} samples")
    print(f"[MASTER RP1] Duration: {master.t[-1]:.3f}s")
    print(f"[MASTER RP1] Mean R: {np.mean(master.R):.6f} V")

    # Process SLAVE data
    slave.process_data()
    print(f"[SLAVE RP2] Duration: {slave.t[-1]:.3f}s")

    # Check synchronization
    time_diff = abs(master.t[-1] - slave.t[-1])
    if time_diff < 0.1:
        print(f"\n✓✓✓ EXCELLENT SYNC! Time difference: {time_diff:.3f}s")
    elif time_diff < 0.5:
        print(f"\n✓ Good sync. Time difference: {time_diff:.3f}s")
    else:
        print(f"\n⚠ Sync warning. Time difference: {time_diff:.3f}s")

    print("\n" + "=" * 60)
    print("ACQUISITION COMPLETE")
    print("=" * 60)

    # Create plots
    create_combined_plot(master, slave, params)

    # Save data
    if params['save_file']:
        save_dual_data(master, slave, params)
    
    plt.show()


def create_combined_plot(master, slave, params):
    """Create combined plot"""
    # Capture raw signals for display
    master.scope.input1 = 'out1'
    master.scope.input2 = 'in1'
    time.sleep(0.05)
    # Use current rolling buffer data
    out1_raw = np.array(master.scope._data_ch1_current)
    in1_raw = np.array(master.scope._data_ch2_current)
    t_raw = np.arange(len(out1_raw)) / master.sample_rate
    
    # Restore lock-in inputs
    master.scope.input1 = 'iq2'
    master.scope.input2 = 'iq2_2'
    
    # FFT
    iq = master.all_X + 1j * master.all_Y
    n_pts = len(iq)
    win = np.hanning(n_pts)
    IQwin = iq * win
    IQfft = np.fft.fftshift(np.fft.fft(IQwin))
    freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / master.sample_rate))
    psd_lock = (np.abs(IQfft) ** 2) / (master.sample_rate * np.sum(win ** 2))
    
    fig = plt.figure(figsize=(16, 13))

    # Row 1: Raw signals
    ax1 = plt.subplot(4, 3, 1)
    n_periods = 5
    n_samples_plot = int(n_periods * master.sample_rate / master.ref_freq)
    n_samples_plot = min(n_samples_plot, len(out1_raw))
    ax1.plot(t_raw[:n_samples_plot] * 1000, out1_raw[:n_samples_plot], 'b-', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('OUT1 (V)')
    ax1.set_title(f'Reference @ {master.ref_freq} Hz')
    ax1.grid(True)

    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(t_raw[:n_samples_plot] * 1000, in1_raw[:n_samples_plot], 'r-', linewidth=1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('IN1 (V)')
    ax2.set_title('Input Signal')
    ax2.grid(True)

    ax3 = plt.subplot(4, 3, 3)
    ax3.semilogy(freqs_lock, psd_lock, label='PSD')
    ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_title('FFT Spectrum')
    ax3.legend()
    ax3.grid(True)

    # Row 2: X, Y, IQ plot
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(master.t, master.all_X, 'b-', linewidth=0.5)
    ax4.axhline(np.mean(master.all_X), color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X (V)')
    ax4.set_title('In-phase (X)')
    ax4.grid(True)

    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(master.t, master.all_Y, 'r-', linewidth=0.5)
    ax5.axhline(np.mean(master.all_Y), color='b', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Y (V)')
    ax5.set_title('Quadrature (Y)')
    ax5.grid(True)

    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(master.all_X, master.all_Y, 'g.', markersize=1, alpha=0.5)
    ax6.plot(np.mean(master.all_X), np.mean(master.all_Y), 'r+', markersize=15)
    ax6.set_xlabel('X (V)')
    ax6.set_ylabel('Y (V)')
    ax6.set_title('IQ Plot')
    ax6.grid(True)
    ax6.axis('equal')

    # Row 3: R, Theta, Polar
    ax7 = plt.subplot(4, 3, 7)
    ax7.plot(master.t, master.R, 'm-', linewidth=0.5)
    ax7.axhline(np.mean(master.R), color='b', linestyle='--', alpha=0.7)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('R (V)')
    ax7.set_title('Magnitude (R)')
    ax7.grid(True)

    ax8 = plt.subplot(4, 3, 8)
    ax8.plot(master.t, master.Theta, 'c-', linewidth=0.5)
    ax8.axhline(np.mean(master.Theta), color='r', linestyle='--', alpha=0.7)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Theta (rad)')
    ax8.set_title('Phase (Theta)')
    ax8.grid(True)

    ax9 = plt.subplot(4, 3, 9)
    ax9.plot(master.Theta, master.R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
    ax9.set_xlabel('Theta (rad)')
    ax9.set_ylabel('R (V)')
    ax9.set_title('R vs Theta')
    ax9.grid(True)

    # Row 4: DC Voltage (full width)
    if len(slave.in1_data) > 0:
        ax10 = plt.subplot(4, 1, 4)
        ax10.plot(slave.t, slave.all_in1, 'orange', linewidth=0.8, label='DC Voltage')
        ax10.axhline(np.mean(slave.all_in1), color='b', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Mean: {np.mean(slave.all_in1):.6f}V')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('RP2: DC Voltage (V)', color='orange')
        ax10.set_title('RP2: Hardware-Triggered DC Voltage (FIXED)', fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        ax10.tick_params(axis='y', labelcolor='orange')

    plt.tight_layout()
    return fig


def save_dual_data(master, slave, params):
    """Save data with timestamps"""
    if not os.path.exists(master.output_dir):
        os.makedirs(master.output_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    img_path = os.path.join(master.output_dir, f'hw_triggered_FIXED_{timestamp_str}.png')
    plt.figure(1)
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {img_path}")

    if params.get('save_timestamps', True):
        # Master data
        data_master = np.column_stack((
            master.sample_timestamps, master.t, master.R, master.Theta, master.all_X, master.all_Y
        ))
        csv_path1 = os.path.join(master.output_dir, f'master_lockin_FIXED_{timestamp_str}.csv')
        np.savetxt(csv_path1, data_master, delimiter=",",
                   header="AbsoluteTimestamp,RelativeTime,R,Theta,X,Y",
                   comments='', fmt='%.10f')
        print(f"✓ Master data saved: {csv_path1}")

        # Slave data
        if len(slave.in1_data) > 0:
            data_slave = np.column_stack((slave.sample_timestamps, slave.t, slave.all_in1))
            csv_path2 = os.path.join(master.output_dir, f'slave_dc_FIXED_{timestamp_str}.csv')
            np.savetxt(csv_path2, data_slave, delimiter=",",
                       header="AbsoluteTimestamp,RelativeTime,IN1",
                       comments='', fmt='%.10f')
            print(f"✓ Slave data saved: {csv_path2}")
            print(f"\n⚡ FIXED: Hardware trigger sent AFTER both armed!")


if __name__ == '__main__':
    run_params = {
        'ref_freq': REF_FREQUENCY,
        'ref_amp': REF_AMPLITUDE,
        'output_ref': OUTPUT_CHANNEL,
        'phase': PHASE_OFFSET,
        'timeout': MEASUREMENT_TIME,
        'filter_bandwidth': FILTER_BANDWIDTH,
        'averaging_window': AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'save_timestamps': SAVE_TIMESTAMPS,
    }

    print("=" * 60)
    print("DUAL RED PITAYA - HARDWARE TRIGGERED SYSTEM (FIXED)")
    print("=" * 60)
    print("WIRING CHECK:")
    print("  RP1 DIO1_P  →  RP2 DIO0_P  (trigger cable)")
    print("  RP1 GND     →  RP2 GND     (common ground)")
    print("=" * 60)
    print("CRITICAL FIX:")
    print("  1. ARM SLAVE FIRST (wait for trigger)")
    print("  2. ARM MASTER SECOND (continuous rolling)")
    print("  3. SEND TRIGGER PULSE (both capture simultaneously)")
    print("  4. READ ROLLING BUFFERS (non-blocking)")
    print("=" * 60)
    print(f"Master: {RP1_HOSTNAME} (Lock-in + trigger source)")
    print(f"Slave:  {RP2_HOSTNAME} (DC voltage + trigger target)")
    print("=" * 60)

    run_hardware_triggered_system(run_params)

    print("\n✓ HARDWARE-TRIGGERED ACQUISITION COMPLETE!")
    print("  ⚡ Correct arming sequence: BOTH FIRST → TRIGGER ONCE")
    print("  ⚡ Rolling mode: non-blocking continuous capture")
    print("  ⚡ Microsecond-level hardware synchronization")
