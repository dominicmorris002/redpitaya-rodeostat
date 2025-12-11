"""
Red Pitaya DC Voltage Monitor - WITH TIMESTAMP SYNCHRONIZATION

SETUP: Connect your DC voltage source to IN1

This version adds precise timestamps to every sample for easy synchronization
with external data acquisition systems (e.g., lock-in amplifier data).

SYNCHRONIZATION:
- Saves CSV with absolute timestamps (Unix time)
- Can be merged with lock-in data using timestamp matching
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
MEASUREMENT_TIME = 30.0  # seconds - how long to measure

# AVERAGING
AVERAGING_WINDOW = 1  # samples - set to 1 to see raw voltage output first

# Data saving
SAVE_DATA = True  # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'

# Advanced settings
DECIMATION = 8192

# Synchronization settings
SAVE_TIMESTAMPS = True  # Save absolute timestamps for sync with other instruments
# ============================================================

from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from datetime import datetime


class RedPitaya:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='dc_monitor_config', hostname='rp-f0909c.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.in1_data = []
        self.all_in1 = []

        # Store capture timestamps
        self.capture_timestamps = []
        self.acquisition_start_time = None

        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        # Read IN1 directly
        self.scope.input1 = 'in1'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = False
        self.sample_rate = 125e6 / self.scope.decimation

    def capture_voltage(self):
        """Captures scope data and appends to voltage array with timestamps"""
        # Record timestamp at moment of capture
        capture_time = time.time()

        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)  # in1 = DC voltage

        self.in1_data.append(ch1)
        self.capture_timestamps.append(capture_time)

        return ch1

    def run(self, params):
        timeout = params['timeout']

        # Let the scope settle
        print("Waiting for scope to settle...")
        time.sleep(0.1)

        # Record absolute start time
        self.acquisition_start_time = time.time()
        print(
            f"\n✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_voltage()

        self.all_in1 = np.array(np.concatenate(self.in1_data))

        # Generate per-sample timestamps
        # Each capture has multiple samples, need to interpolate timestamps
        total_samples = len(self.all_in1)

        # Create timestamp array for each sample
        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.in1_data[i])
            # Interpolate timestamps for samples in this capture
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)

            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)

        if averaging_window > 1:
            self.all_in1 = np.convolve(self.all_in1, np.ones(averaging_window) / averaging_window, mode='valid')
            # Also trim timestamps to match filtered data
            self.sample_timestamps = self.sample_timestamps[:len(self.all_in1)]
            print(f"Applied {averaging_window}-sample moving average filter")

        # Time array (relative to start)
        t = self.sample_timestamps - self.acquisition_start_time

        print("=" * 60)
        print("DC VOLTAGE DIAGNOSTICS")
        print("=" * 60)
        print(f"Expected Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_in1)}")
        print(f"Measurement Duration: {t[-1]:.3f} seconds")

        # Calculate actual sample rate from timestamps
        actual_sample_rate = len(self.all_in1) / (t[-1] - t[0]) if t[-1] > t[0] else 0
        sample_rate_error = abs(actual_sample_rate - self.sample_rate) / self.sample_rate * 100

        print(f"Actual Sample Rate: {actual_sample_rate:.2f} Hz")
        print(f"Sample Rate Error: {sample_rate_error:.3f}%")

        if sample_rate_error > 1.0:
            print(f"⚠ WARNING: Sample rate error is {sample_rate_error:.3f}% (expected < 1%)")
        else:
            print(f"✓ Sample rate verified within {sample_rate_error:.3f}%")

        print("-" * 60)
        print("TIMESTAMP INFORMATION:")
        print(f"Start time: {datetime.fromtimestamp(self.sample_timestamps[0]).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"End time:   {datetime.fromtimestamp(self.sample_timestamps[-1]).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"Duration:   {self.sample_timestamps[-1] - self.sample_timestamps[0]:.3f} seconds")

        print("-" * 60)
        print(f"Mean Voltage: {np.mean(self.all_in1):.6f} V ± {np.std(self.all_in1):.6f} V")
        print("=" * 60)

        # Create single plot - just voltage vs time
        fig = plt.figure(figsize=(14, 6))

        ax = plt.subplot(1, 1, 1)
        ax.plot(t, self.all_in1, 'b-', linewidth=0.8, label='DC Voltage')
        ax.axhline(np.mean(self.all_in1), color='r', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Mean: {np.mean(self.all_in1):.6f} V')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Voltage (V)', fontsize=12)
        ax.set_title(f'DC Voltage vs Time (Sample Rate: {actual_sample_rate:.2f} Hz)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])

        # Add stats text box
        stats_text = f'Samples: {len(self.all_in1):,}\nDuration: {t[-1]:.2f} s\nStd Dev: {np.std(self.all_in1):.6f} V'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save plot
            img_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Plot saved: {img_path}")

            # Save data with timestamps for synchronization
            if params.get('save_timestamps', False):
                # Save with absolute timestamps AND relative time
                data = np.column_stack((
                    self.sample_timestamps,  # Absolute Unix timestamp
                    t,  # Relative time (s)
                    self.all_in1,  # Voltage
                ))
                csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                           header="AbsoluteTimestamp,RelativeTime,Voltage",
                           comments='', fmt='%.10f')
                print(f"✓ Data saved with timestamps: {csv_path}")
                print(f"  Columns: AbsoluteTimestamp (Unix), RelativeTime (s), Voltage (V)")
            else:
                # Original format without timestamps
                data = np.column_stack((t, self.all_in1))
                csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                           header="Time,Voltage", comments='', fmt='%.6f')
                print(f"✓ Data saved: {csv_path}")

        plt.show()


if __name__ == '__main__':
    rp = RedPitaya()

    run_params = {
        'timeout': MEASUREMENT_TIME,
        'averaging_window': AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'save_timestamps': SAVE_TIMESTAMPS,
    }

    print("=" * 60)
    print("RED PITAYA DC VOLTAGE MONITOR - WITH TIMESTAMP SYNC")
    print("=" * 60)
    print("SETUP: Connect your DC voltage source to IN1")
    print("=" * 60)
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Decimation: {DECIMATION}")
    print(f"Expected Sample Rate: {125e6 / DECIMATION:.2f} Hz")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print(f"Save Timestamps: {SAVE_TIMESTAMPS}")
    print("=" * 60)
    print("\nNOTE: Timestamps will be saved for synchronization with lock-in data")
    print("=" * 60)

    rp.run(run_params)
