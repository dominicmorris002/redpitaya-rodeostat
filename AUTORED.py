"""
Dual Red Pitaya System - Lock-In Amplifier + DC Voltage Sampler

SETUP:
Red Pitaya #1 (rp-f073ce.local):
  - OUT1 → Electrochemical cell (AC excitation)
  - IN1 → Cell response (lock-in demodulation)

Red Pitaya #2 (rp-f0909c.local):
  - IN1 → DC voltage to monitor (e.g., cell voltage, Autolab signal)

Both RPs run simultaneously with synchronized timestamps for easy data merging.
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
import threading

N_FFT_SHOW = 10


class RedPitayaLockIn:
    """Lock-in amplifier - UNCHANGED from original code"""
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

    def __init__(self, hostname, output_dir='test_data'):
        self.rp = Pyrpl(config='lockin_config', hostname=hostname)
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []

        # Store capture timestamps
        self.capture_timestamps = []
        self.acquisition_start_time = None

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival
        self.scope = self.rp_modules.scope

        print(f"[RP1 Lock-In] Available scope inputs:", self.scope.inputs)

        # For iq2 module, use iq2 (X) and iq2_2 (Y)
        self.scope.input1 = 'iq2'  # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('[RP1 Lock-In] Invalid decimation')
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
        print("[RP1 Lock-In] ASG0 disabled - IQ module will generate reference")

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

        print(f"[RP1 Lock-In] Setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"[RP1 Lock-In] Filter BW: {filter_bw} Hz")

    def capture_lockin(self):
        """Captures scope data and appends to X and Y arrays with timestamps"""
        capture_time = time.time()

        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)  # iq2 = X (in-phase)
        ch2 = np.array(self.scope._data_ch2_current)  # iq2_2 = Y (quadrature)

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(capture_time)

        return ch1, ch2

    def run(self, params, sync_event=None):
        timeout = params['timeout']
        self.setup_lockin(params)

        print("[RP1 Lock-In] Waiting for lock-in to settle...")
        time.sleep(0.5)

        # Wait for sync if provided
        if sync_event:
            print("[RP1 Lock-In] Waiting for sync signal...")
            sync_event.wait()

        self.acquisition_start_time = time.time()
        print(f"[RP1 Lock-In] ✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        # Generate per-sample timestamps
        samples_per_capture = len(self.lockin_X[0])
        total_samples = len(self.all_X)

        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.lockin_X[i])
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)

        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_X)]
            print(f"[RP1 Lock-In] Applied {averaging_window}-sample moving average filter")

        self.R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        self.Theta = np.arctan2(self.all_Y, self.all_X)
        self.t = self.sample_timestamps - self.acquisition_start_time

        print(f"[RP1 Lock-In] ✓ Acquisition complete: {len(self.all_X)} samples")


class RedPitayaDCSampler:
    """DC voltage sampler for RP2"""
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]
    
    def __init__(self, hostname, output_dir='test_data'):
        print(f"[RP2 DC Sampler] Connecting to {hostname}...")
        self.rp = Pyrpl(config='dc_sampler_config', hostname=hostname)
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.scope = self.rp_modules.scope
        
        self.in1_data = []
        self.capture_timestamps = []
        self.acquisition_start_time = None
        
        print("[RP2 DC Sampler] ✓ Connected")
    
    def setup_scope(self, decimation):
        """Configure scope to read IN1"""
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'  # Use same input for both channels
        self.scope.decimation = decimation
        
        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f'Invalid decimation')
        
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = False
        self.sample_rate = 125e6 / self.scope.decimation
        
        print(f"[RP2 DC Sampler] ✓ Configured: {self.sample_rate:.2f} Hz")
    
    def capture(self):
        """Capture IN1 voltage with timestamp"""
        capture_time = time.time()
        
        self.scope.single()
        in1 = np.array(self.scope._data_ch1_current)
        
        self.in1_data.append(in1)
        self.capture_timestamps.append(capture_time)
        
        return in1
    
    def run(self, timeout, decimation, sync_event=None):
        """Run continuous acquisition"""
        self.setup_scope(decimation)
        
        # Wait for sync if provided
        if sync_event:
            print("[RP2 DC Sampler] Waiting for sync signal...")
            sync_event.wait()
        
        self.acquisition_start_time = time.time()
        print(f"[RP2 DC Sampler] ✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        loop_start = time.time()
        capture_count = 0
        
        while (time.time() - loop_start) < timeout:
            self.capture()
            capture_count += 1
        
        # Concatenate all data
        self.all_in1 = np.concatenate(self.in1_data)
        
        # Generate per-sample timestamps
        samples_per_capture = len(self.in1_data[0])
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
        
        print(f"[RP2 DC Sampler] ✓ Acquisition complete: {total_samples} samples")
        print(f"[RP2 DC Sampler]   Mean IN1: {np.mean(self.all_in1):.6f} V ± {np.std(self.all_in1):.6f} V")


def run_dual_system(run_params):
    """Run both Red Pitayas simultaneously"""
    
    print("\n" + "=" * 60)
    print("DUAL RED PITAYA SYSTEM")
    print("=" * 60)
    print(f"RP1 ({RP1_HOSTNAME}): Lock-in amplifier")
    print(f"RP2 ({RP2_HOSTNAME}): DC voltage sampler")
    print("=" * 60)
    
    # Initialize both RPs
    rp1 = RedPitayaLockIn(hostname=RP1_HOSTNAME, output_dir=run_params['output_dir'])
    rp2 = RedPitayaDCSampler(hostname=RP2_HOSTNAME, output_dir=run_params['output_dir'])
    
    # Create sync event for simultaneous start
    sync_event = threading.Event()
    
    # Run both in separate threads
    def run_rp1():
        rp1.run(run_params, sync_event=sync_event)
    
    def run_rp2():
        rp2.run(timeout=run_params['timeout'], 
                decimation=DECIMATION, 
                sync_event=sync_event)
    
    thread1 = threading.Thread(target=run_rp1)
    thread2 = threading.Thread(target=run_rp2)
    
    print("\n[SYNC] Starting both acquisitions...")
    thread1.start()
    thread2.start()
    
    # Brief delay to ensure both are ready
    time.sleep(0.1)
    
    # Signal both to start simultaneously
    print("[SYNC] ✓ Triggering synchronized start NOW!")
    sync_event.set()
    
    # Wait for both to complete
    thread1.join()
    thread2.join()
    
    print("\n" + "=" * 60)
    print("BOTH ACQUISITIONS COMPLETE")
    print("=" * 60)
    
    # Plot combined results
    plot_combined_results(rp1, rp2, run_params)
    
    # Save combined data
    if run_params['save_file']:
        save_combined_data(rp1, rp2, run_params)
    else:
        plt.show()


def plot_combined_results(rp1, rp2, params):
    """Create comprehensive plot with both RP1 and RP2 data"""
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Lock-in X and Y
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(rp1.t, rp1.all_X, 'b-', linewidth=0.5)
    ax1.axhline(np.mean(rp1.all_X), color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X (V)')
    ax1.set_title('RP1: In-phase (X)')
    ax1.grid(True)
    
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(rp1.t, rp1.all_Y, 'r-', linewidth=0.5)
    ax2.axhline(np.mean(rp1.all_Y), color='b', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y (V)')
    ax2.set_title('RP1: Quadrature (Y)')
    ax2.grid(True)
    
    # IQ plot
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(rp1.all_X, rp1.all_Y, 'g.', markersize=1, alpha=0.5)
    ax3.plot(np.mean(rp1.all_X), np.mean(rp1.all_Y), 'r+', markersize=15, markeredgewidth=2)
    ax3.set_xlabel('X (V)')
    ax3.set_ylabel('Y (V)')
    ax3.set_title('RP1: IQ Plot')
    ax3.grid(True)
    ax3.axis('equal')
    
    # Row 2: Lock-in R and Theta
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(rp1.t, rp1.R, 'm-', linewidth=0.5)
    ax4.axhline(np.mean(rp1.R), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp1.R):.4f}V')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('R (V)')
    ax4.set_title('RP1: Magnitude (R)')
    ax4.legend()
    ax4.grid(True)
    
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(rp1.t, rp1.Theta, 'c-', linewidth=0.5)
    ax5.axhline(np.mean(rp1.Theta), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp1.Theta):.4f} rad')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Theta (rad)')
    ax5.set_title('RP1: Phase (Theta)')
    ax5.legend()
    ax5.grid(True)
    
    # R vs Theta
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(rp1.Theta, rp1.R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
    ax6.set_xlabel('Theta (rad)')
    ax6.set_ylabel('R (V)')
    ax6.set_title('RP1: R vs Theta')
    ax6.grid(True)
    
    # Row 3: RP2 DC voltage
    ax7 = plt.subplot(4, 3, 7)
    ax7.plot(rp2.t, rp2.all_in1, 'orange', linewidth=0.5)
    ax7.axhline(np.mean(rp2.all_in1), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp2.all_in1):.4f}V')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('IN1 (V)')
    ax7.set_title('RP2: DC Voltage vs Time')
    ax7.legend()
    ax7.grid(True)
    ax7.set_xlim(rp2.t[0], rp2.t[-1])
    
    # RP2 histogram
    ax8 = plt.subplot(4, 3, 8)
    ax8.hist(rp2.all_in1, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax8.axvline(np.mean(rp2.all_in1), color='b', linestyle='--', linewidth=2)
    ax8.set_xlabel('IN1 (V)')
    ax8.set_ylabel('Count')
    ax8.set_title('RP2: Voltage Distribution')
    ax8.grid(True, alpha=0.3)
    
    # Combined: R vs IN1
    ax9 = plt.subplot(4, 3, 9)
    # Interpolate to match sample counts if needed
    if len(rp1.R) != len(rp2.all_in1):
        # Resample RP2 to match RP1
        from scipy.interpolate import interp1d
        f = interp1d(rp2.t, rp2.all_in1, kind='linear', fill_value='extrapolate')
        in1_interp = f(rp1.t)
        ax9.plot(in1_interp, rp1.R, 'brown', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax9.set_xlabel('RP2 IN1 (V)')
    else:
        ax9.plot(rp2.all_in1, rp1.R, 'brown', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax9.set_xlabel('RP2 IN1 (V)')
    ax9.set_ylabel('RP1 R (V)')
    ax9.set_title('Cross-correlation: R vs IN1')
    ax9.grid(True)
    
    # Row 4: Time-aligned comparison
    ax10 = plt.subplot(4, 1, 4)
    ax10_twin = ax10.twinx()
    
    line1 = ax10.plot(rp1.t, rp1.R, 'm-', linewidth=1, label='RP1: R (magnitude)', alpha=0.7)
    
    # Interpolate RP2 to RP1 time base for overlay
    if len(rp1.R) != len(rp2.all_in1):
        from scipy.interpolate import interp1d
        f = interp1d(rp2.t, rp2.all_in1, kind='linear', fill_value='extrapolate')
        in1_interp = f(rp1.t)
        line2 = ax10_twin.plot(rp1.t, in1_interp, 'orange', linewidth=1, label='RP2: IN1 (DC voltage)', alpha=0.7)
    else:
        line2 = ax10_twin.plot(rp2.t, rp2.all_in1, 'orange', linewidth=1, label='RP2: IN1 (DC voltage)', alpha=0.7)
    
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('RP1: Magnitude R (V)', color='m')
    ax10_twin.set_ylabel('RP2: IN1 Voltage (V)', color='orange')
    ax10.tick_params(axis='y', labelcolor='m')
    ax10_twin.tick_params(axis='y', labelcolor='orange')
    ax10.set_title('Time-Aligned: Lock-in Magnitude vs DC Voltage')
    ax10.grid(True)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax10.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()


def save_combined_data(rp1, rp2, params):
    """Save both datasets with synchronized timestamps"""
    if not os.path.exists(rp1.output_dir):
        os.makedirs(rp1.output_dir)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plot
    img_path = os.path.join(rp1.output_dir, f'dual_rp_results_{timestamp_str}.png')
    plt.savefig(img_path, dpi=150)
    print(f"\n✓ Plot saved: {img_path}")
    
    # Save RP1 data (lock-in)
    if params.get('save_timestamps', False):
        data_rp1 = np.column_stack((
            rp1.sample_timestamps, rp1.t, rp1.R, rp1.Theta, rp1.all_X, rp1.all_Y
        ))
        csv_path1 = os.path.join(rp1.output_dir, f'rp1_lockin_{timestamp_str}.csv')
        np.savetxt(csv_path1, data_rp1, delimiter=",",
                   header="AbsoluteTimestamp,RelativeTime,R,Theta,X,Y",
                   comments='', fmt='%.10f')
        print(f"✓ RP1 data saved: {csv_path1}")
        
        # Save RP2 data (DC voltage)
        data_rp2 = np.column_stack((
            rp2.sample_timestamps, rp2.t, rp2.all_in1
        ))
        csv_path2 = os.path.join(rp1.output_dir, f'rp2_dc_voltage_{timestamp_str}.csv')
        np.savetxt(csv_path2, data_rp2, delimiter=",",
                   header="AbsoluteTimestamp,RelativeTime,IN1",
                   comments='', fmt='%.10f')
        print(f"✓ RP2 data saved: {csv_path2}")
        print(f"\n  Merge files using AbsoluteTimestamp column")


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
    }
    
    print("=" * 60)
    print("DUAL RED PITAYA MEASUREMENT SYSTEM")
    print("=" * 60)
    print("SETUP:")
    print(f"  RP1 ({RP1_HOSTNAME}):")
    print("    OUT1 → Electrochemical cell")
    print("    IN1  → Cell response (lock-in)")
    print(f"  RP2 ({RP2_HOSTNAME}):")
    print("    IN1  → DC voltage to monitor")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Both RPs will start simultaneously")
    print("=" * 60)
    
    run_dual_system(run_params)
    
    print("\n✓ DUAL ACQUISITION COMPLETE!")
    print("  Both datasets saved with synchronized timestamps")
