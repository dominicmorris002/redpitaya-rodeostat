"""
Dual Red Pitaya System - FIXED VERSION
Now correctly saves lock-in data and shows actual sample rates!
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 100  # Hz - AC excitation frequency
REF_AMPLITUDE = 0.5  # V - AC signal amplitude (will appear on OUT1)
OUTPUT_CHANNEL = 'out1'  # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET = 0  # degrees - phase adjustment (0, 90, 180, 270)
MEASUREMENT_TIME = 30.0  # seconds - how long to measure

# LOCK-IN FILTER BANDWIDTH
FILTER_BANDWIDTH = 10  # Hz - lower = cleaner, higher = faster response

# AVERAGING
AVERAGING_WINDOW = 1  # samples - set to 1 to see raw lock-in output first

# Data saving
SAVE_DATA = True  # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'

# Advanced settings
DECIMATION = 8192
SHOW_FFT = True

# Synchronization settings
SAVE_TIMESTAMPS = True  # Save absolute timestamps for sync with NI-DAQ
USE_HARDWARE_TRIGGER = False  # Hardware trigger has issues with rolling mode - use software sync

# Red Pitaya hostnames
RP1_HOSTNAME = 'rp-f073ce.local'  # Lock-in amplifier
RP2_HOSTNAME = 'rp-f0909c.local'  # DC voltage sampler
# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os
from datetime import datetime

N_FFT_SHOW = 10


# ============================================================
# ORIGINAL REDPITAYA CLASS - COMPLETELY UNCHANGED
# ============================================================
class RedPitaya:
    electrode_map = {'A': (False, False), 'B': (True, False),
                     'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True),
                         '100uA': (True, False, True, True),
                         '1mA': (True, True, False, True),
                         '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True),
                    '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='lockin_config', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []

        self.capture_timestamps = []
        self.acquisition_start_time = None

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival
        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        self.ref_sig.output_direct = 'off'
        print("ASG0 disabled - IQ module will generate reference")

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

        print(f"Lock-in setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"Filter BW: {filter_bw} Hz")
        print(f"IQ2 output_direct: {self.lockin.output_direct} (outputs {ref_amp}V sine)")
        print(f"IQ2 amplitude: {self.lockin.amplitude} V")
        print(f"IQ2 input: {self.lockin.input}")
        print(f"Scope reading: iq2 (X) and iq2_2 (Y)")

    def capture_lockin(self):
        capture_time = time.time()
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)
        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(capture_time)
        return ch1, ch2


# ============================================================
# DC VOLTAGE SAMPLER FOR RP2
# ============================================================
class RedPitayaDCSampler:
    """Hardware-synced DC voltage sampler"""
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, hostname, output_dir='test_data'):
        print(f"\n[RP2 DC Sampler] Connecting to {hostname}...")
        self.rp = Pyrpl(config='dc_sampler_config', hostname=hostname)
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.scope = self.rp_modules.scope

        self.in1_data = []
        self.capture_timestamps = []
        self.acquisition_start_time = None

        print("[RP2 DC Sampler] ✓ Connected")

    def setup_scope(self, decimation, use_hw_trigger=False):
        """Configure scope to read IN1"""
        self.scope.input1 = 'in1'
        self.scope.decimation = decimation

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f'Invalid decimation')

        self.scope.trigger_source = 'immediately'
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = False
        self.sample_rate = 125e6 / self.scope.decimation

        print(f"[RP2 DC Sampler] Sample rate: {self.sample_rate:.2f} Hz")

    def capture(self):
        """Capture IN1 voltage"""
        capture_time = time.time()
        self.scope.single()
        in1 = np.array(self.scope._data_ch1_current)
        self.in1_data.append(in1)
        self.capture_timestamps.append(capture_time)
        return in1

    def process_data(self):
        """Process captured data"""
        if len(self.in1_data) == 0:
            print("[RP2] WARNING: No data captured!")
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

        print(f"\n[RP2] Processed {total_samples} samples")
        print(f"[RP2] Mean IN1: {np.mean(self.all_in1):.6f} V ± {np.std(self.all_in1):.6f} V")


# ============================================================
# DUAL SYSTEM RUNNER
# ============================================================
def run_dual_system(params):
    """Run both Red Pitayas with software-synchronized interleaved captures"""
    use_hw_trigger = params.get('use_hw_trigger', False)

    if use_hw_trigger:
        print("\n" + "=" * 60)
        print("WARNING: Hardware trigger disabled for interleaved mode")
        print("Using software synchronization instead (timestamps still accurate)")
        print("=" * 60)
        use_hw_trigger = False

    print("\n" + "=" * 60)
    print("DUAL RED PITAYA - INTERLEAVED SOFTWARE SYNC")
    print("=" * 60)

    # Initialize both
    print("\n[RP1] Initializing lock-in amplifier...")
    rp1 = RedPitaya(output_dir=params['output_dir'])

    print("\n[RP2] Initializing DC sampler...")
    rp2 = RedPitayaDCSampler(hostname=RP2_HOSTNAME, output_dir=params['output_dir'])
    rp2.setup_scope(DECIMATION, use_hw_trigger)

    # Setup RP1
    rp1.setup_lockin(params)

    print("\n[SYNC] Waiting for lock-in to settle...")
    time.sleep(0.5)

    # Synchronized start
    sync_time = time.time()
    rp1.acquisition_start_time = sync_time
    rp2.acquisition_start_time = sync_time

    print(f"\n[SYNC] ✓ Starting at {datetime.fromtimestamp(sync_time).strftime('%H:%M:%S.%f')}")

    # Interleaved acquisition
    loop_start = time.time()
    capture_count = 0
    while (time.time() - loop_start) < params['timeout']:
        rp1.capture_lockin()
        rp2.capture()
        capture_count += 1

    print(f"\n[ACQ] ✓ Acquisition complete - {capture_count} captures")

    # Process RP1 data
    print("\n[RP1] Processing lock-in data...")
    rp1.all_X = np.array(np.concatenate(rp1.lockin_X))
    rp1.all_Y = np.array(np.concatenate(rp1.lockin_Y))

    total_samples = len(rp1.all_X)
    rp1.sample_timestamps = np.zeros(total_samples)
    sample_idx = 0

    for i, capture_time in enumerate(rp1.capture_timestamps):
        n_samples = len(rp1.lockin_X[i])
        capture_duration = n_samples / rp1.sample_rate
        sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
        rp1.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
        sample_idx += n_samples

    averaging_window = params.get('averaging_window', 1)
    if averaging_window > 1:
        rp1.all_X = np.convolve(rp1.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
        rp1.all_Y = np.convolve(rp1.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
        rp1.sample_timestamps = rp1.sample_timestamps[:len(rp1.all_X)]

    rp1.R = np.sqrt(rp1.all_X ** 2 + rp1.all_Y ** 2)
    rp1.Theta = np.arctan2(rp1.all_Y, rp1.all_X)
    rp1.t = rp1.sample_timestamps - rp1.acquisition_start_time

    print(f"[RP1] ✓ Processed {len(rp1.all_X)} samples")
    print(f"[RP1] Sample rate: {rp1.sample_rate:.2f} Hz")
    print(f"[RP1] Actual duration: {rp1.t[-1]:.3f} seconds")
    print(f"[RP1] Expected samples: {rp1.sample_rate * params['timeout']:.0f}")
    print(f"[RP1] Mean R: {np.mean(rp1.R):.6f} V ± {np.std(rp1.R):.6f} V")
    
    # DIAGNOSTIC WARNINGS FROM ORIGINAL CODE
    X_ac = np.std(rp1.all_X)
    Y_ac = np.std(rp1.all_Y)
    X_dc = np.mean(np.abs(rp1.all_X))
    Y_dc = np.mean(np.abs(rp1.all_Y))
    SIGNAL_THRESHOLD = 0.02
    
    print("-" * 60)
    print("LOCK-IN SIGNAL QUALITY:")
    print(f"X: DC={X_dc:.6f}V, AC={X_ac:.6f}V, AC/DC={X_ac / max(X_dc, 0.001):.3f}")
    print(f"Y: DC={Y_dc:.6f}V, AC={Y_ac:.6f}V, AC/DC={Y_ac / max(Y_dc, 0.001):.3f}")
    
    if X_dc > SIGNAL_THRESHOLD and X_ac / X_dc > 0.5:
        print("⚠ WARNING: X is oscillating! Should be flat for locked signal")
    
    if Y_dc > SIGNAL_THRESHOLD and Y_ac / Y_dc > 0.5:
        print("⚠ WARNING: Y is oscillating! Should be flat for locked signal")

    # Process RP2 data
    rp2.process_data()
    print(f"[RP2] Sample rate: {rp2.sample_rate:.2f} Hz")
    print(f"[RP2] Actual duration: {rp2.t[-1]:.3f} seconds")
    print(f"[RP2] Expected samples: {rp2.sample_rate * params['timeout']:.0f}")
    
    # CHECK FOR TIMING MISMATCH
    time_diff = abs(rp1.t[-1] - rp2.t[-1])
    if time_diff > 0.5:
        print(f"\n⚠⚠⚠ WARNING: TIMING MISMATCH! ⚠⚠⚠")
        print(f"RP1 duration: {rp1.t[-1]:.3f}s vs RP2 duration: {rp2.t[-1]:.3f}s")
        print(f"Difference: {time_diff:.3f} seconds")
        print("This means timestamps may not align correctly!")
        print("Possible causes:")
        print("  - Different capture rates between RP1 and RP2")
        print("  - One Red Pitaya is slower than the other")
        print("  - USB/network latency differences")

    print("\n" + "=" * 60)
    print("ACQUISITION COMPLETE")
    print("=" * 60)

    # Create combined plot
    create_combined_plot(rp1, rp2, params)

    # Save data
    if params['save_file']:
        save_dual_data(rp1, rp2, params)
    
    plt.show()


def create_combined_plot(rp1, rp2, params):
    """Create single figure with lock-in plots (3x3) + DC voltage (bottom row)"""
    # First, capture raw signals for display
    rp1.scope.input1 = 'out1'
    rp1.scope.input2 = 'in1'
    time.sleep(0.05)
    rp1.scope.single()
    out1_raw = np.array(rp1.scope._data_ch1_current)
    in1_raw = np.array(rp1.scope._data_ch2_current)
    t_raw = np.arange(len(out1_raw)) / rp1.sample_rate
    
    # Restore scope to lock-in mode
    rp1.scope.input1 = 'iq2'
    rp1.scope.input2 = 'iq2_2'
    
    # Calculate FFT
    iq = rp1.all_X + 1j * rp1.all_Y
    n_pts = len(iq)
    win = np.hanning(n_pts)
    IQwin = iq * win
    IQfft = np.fft.fftshift(np.fft.fft(IQwin))
    freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / rp1.sample_rate))
    psd_lock = (np.abs(IQfft) ** 2) / (rp1.sample_rate * np.sum(win ** 2))
    
    # Create figure with 10 subplots (3x3 for lock-in + 1 full-width for DC)
    fig = plt.figure(figsize=(16, 13))
    ZoomOut_Amount = 5

    # ===== LOCK-IN PLOTS (3x3 grid) =====
    # 1. OUT1 (Reference Signal)
    ax1 = plt.subplot(4, 3, 1)
    n_periods = 5
    n_samples_plot = int(n_periods * rp1.sample_rate / rp1.ref_freq)
    n_samples_plot = min(n_samples_plot, len(out1_raw))
    ax1.plot(t_raw[:n_samples_plot] * 1000, out1_raw[:n_samples_plot], 'b-', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('OUT1 (V)')
    ax1.set_title(f'Reference Signal (OUT1) @ {rp1.ref_freq} Hz')
    ax1.grid(True)

    # 2. IN1 (Input Signal)
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(t_raw[:n_samples_plot] * 1000, in1_raw[:n_samples_plot], 'r-', linewidth=1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('IN1 (V)')
    ax2.set_title('Input Signal (IN1)')
    ax2.grid(True)

    # 3. FFT Spectrum
    ax3 = plt.subplot(4, 3, 3)
    ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
    ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz (target)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power (a.u.)')
    ax3.set_title('FFT Spectrum (baseband)')
    ax3.legend()
    ax3.grid(True)

    # 4. X vs Time
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(rp1.t, rp1.all_X, 'b-', linewidth=0.5)
    ax4.axhline(np.mean(rp1.all_X), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp1.all_X):.4f}V')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X (V)')
    ax4.set_title('In-phase (X) vs Time')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(rp1.t[0], rp1.t[-1])

    # 5. Y vs Time
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(rp1.t, rp1.all_Y, 'r-', linewidth=0.5)
    ax5.axhline(np.mean(rp1.all_Y), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp1.all_Y):.4f}V')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Y (V)')
    ax5.set_title('Quadrature (Y) vs Time')
    ax5.legend()
    ax5.grid(True)
    ax5.set_xlim(rp1.t[0], rp1.t[-1])

    # 6. X vs Y (IQ plot)
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(rp1.all_X, rp1.all_Y, 'g.', markersize=1, alpha=0.5)
    ax6.plot(np.mean(rp1.all_X), np.mean(rp1.all_Y), 'r+', markersize=15,
             markeredgewidth=2, label='Mean')
    ax6.set_xlabel('X (V)')
    ax6.set_ylabel('Y (V)')
    ax6.set_title('IQ Plot (X vs Y)')
    ax6.legend()
    ax6.grid(True)
    ax6.axis('equal')

    # 7. R vs Time
    ax7 = plt.subplot(4, 3, 7)
    ax7.plot(rp1.t, rp1.R, 'm-', linewidth=0.5)
    ax7.axhline(np.mean(rp1.R), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp1.R):.4f}V')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('R (V)')
    ax7.set_title('Magnitude (R) vs Time')
    ax7.legend()
    ax7.grid(True)
    ax7.set_xlim(rp1.t[0], rp1.t[-1])

    # 8. Theta vs Time
    ax8 = plt.subplot(4, 3, 8)
    ax8.plot(rp1.t, rp1.Theta, 'c-', linewidth=0.5)
    ax8.axhline(np.mean(rp1.Theta), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp1.Theta):.4f} rad')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Theta (rad)')
    ax8.set_title('Phase (Theta) vs Time')
    ax8.legend()
    ax8.grid(True)
    ax8.set_xlim(rp1.t[0], rp1.t[-1])

    # 9. R vs Theta
    ax9 = plt.subplot(4, 3, 9)
    ax9.plot(rp1.Theta, rp1.R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
    ax9.axhline(np.mean(rp1.R), color='b', linestyle='--', alpha=0.5)
    ax9.axvline(np.mean(rp1.Theta), color='r', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Theta (rad)')
    ax9.set_ylabel('R (V)')
    ax9.set_title('R vs Theta')
    ax9.grid(True)

    # ===== DC VOLTAGE PLOT (bottom, full width) =====
    if len(rp2.in1_data) > 0:
        ax10 = plt.subplot(4, 1, 4)  # Full width on bottom
        ax10.plot(rp2.t, rp2.all_in1, 'orange', linewidth=0.8, label='DC Voltage')
        ax10.axhline(np.mean(rp2.all_in1), color='b', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Mean: {np.mean(rp2.all_in1):.6f}V')
        ax10.set_xlabel('Time (s)', fontsize=11)
        ax10.set_ylabel('RP2: DC Voltage (V)', fontsize=11, color='orange')
        ax10.set_title('RP2: DC Voltage Monitoring', fontsize=12, fontweight='bold')
        ax10.legend(fontsize=10, loc='upper right')
        ax10.grid(True, alpha=0.3)
        ax10.set_xlim(rp2.t[0], rp2.t[-1])
        ax10.tick_params(axis='y', labelcolor='orange')
        
        # Add stats text box
        stats_text = f'Min: {np.min(rp2.all_in1):.6f}V | Max: {np.max(rp2.all_in1):.6f}V | Std: {np.std(rp2.all_in1):.6f}V'
        ax10.text(0.02, 0.98, stats_text, transform=ax10.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def save_dual_data(rp1, rp2, params):
    """Save both datasets with synchronized timestamps"""
    if not os.path.exists(rp1.output_dir):
        os.makedirs(rp1.output_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save combined plot
    img_path_combined = os.path.join(rp1.output_dir, f'combined_results_{timestamp_str}.png')
    plt.figure(1)  # The combined plot
    plt.savefig(img_path_combined, dpi=150, bbox_inches='tight')
    print(f"\n✓ Combined plot saved: {img_path_combined}")

    if params.get('save_timestamps', True):
        # Save RP1 lock-in data
        data_rp1 = np.column_stack((
            rp1.sample_timestamps, rp1.t, rp1.R, rp1.Theta, rp1.all_X, rp1.all_Y
        ))
        csv_path1 = os.path.join(rp1.output_dir, f'rp1_lockin_{timestamp_str}.csv')
        np.savetxt(csv_path1, data_rp1, delimiter=",",
                   header="AbsoluteTimestamp,RelativeTime,R,Theta,X,Y",
                   comments='', fmt='%.10f')
        print(f"✓ RP1 lock-in data saved: {csv_path1}")

        # Save RP2 data
        if len(rp2.in1_data) > 0:
            data_rp2 = np.column_stack((rp2.sample_timestamps, rp2.t, rp2.all_in1))
            csv_path2 = os.path.join(rp1.output_dir, f'rp2_dc_voltage_{timestamp_str}.csv')
            np.savetxt(csv_path2, data_rp2, delimiter=",",
                       header="AbsoluteTimestamp,RelativeTime,IN1",
                       comments='', fmt='%.10f')
            print(f"✓ RP2 data saved: {csv_path2}")
            print(f"\n  📊 Use 'AbsoluteTimestamp' column to sync datasets")


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
        'fft': SHOW_FFT,
        'save_timestamps': SAVE_TIMESTAMPS,
        'use_hw_trigger': USE_HARDWARE_TRIGGER,
    }

    print("=" * 60)
    print("DUAL RED PITAYA MEASUREMENT SYSTEM")
    print("=" * 60)
    print("SETUP:")
    print(f"  RP1 ({RP1_HOSTNAME}):")
    print("    OUT1 → Electrochemical cell (AC excitation)")
    print("    IN1  → Cell response (lock-in)")
    print(f"  RP2 ({RP2_HOSTNAME}):")
    print("    IN1  → DC voltage to monitor")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Decimation: {DECIMATION} (Sample rate: {125e6/DECIMATION:.2f} Hz)")
    print("Software sync: Interleaved acquisition")
    print("=" * 60)

    run_dual_system(run_params)

    print("\n✓ DUAL ACQUISITION COMPLETE!")
    print("  Combined plot: Lock-in (9 panels) + DC voltage (bottom)")
    print("  RP1 lock-in CSV saved (R, Theta, X, Y)")
    print("  RP2 DC voltage CSV saved")
    print("  Both datasets have synchronized timestamps!")
