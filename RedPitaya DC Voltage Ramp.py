"""
Red Pitaya DC Voltage Monitor with Timestamp Sync

Connect your DC voltage source to IN1 and run this to log voltage over time.
Saves timestamps so you can match it up with lock-in data later.
"""

# ============================================================
# SETTINGS - Change these before running
# ============================================================
MEASUREMENT_TIME = 30.0  # how long to record (seconds)
AVERAGING_WINDOW = 1  # set to 1 for raw data, higher for smoothing
SAVE_DATA = True  # save files or just plot?
OUTPUT_DIRECTORY = 'test_data'
DECIMATION = 8192  # lower = faster sampling, higher = slower

# Gain and voltage settings
VOLTAGE_MODE = 'LV'  # 'LV' = ±1V range, 'HV' = ±20V range, 'MANUAL' = custom gain

AUTOLAB_GAIN = 0.0
MANUAL_GAIN = 1.0 + AUTOLAB_GAIN # only used if VOLTAGE_MODE = 'MANUAL'

# Timestamp saving (for syncing with other instruments)
SAVE_TIMESTAMPS = True
# ============================================================

from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from datetime import datetime


class RedPitaya:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', voltage_mode='LV', manual_gain=1.0):
        self.rp = Pyrpl(config='dc_monitor_config', hostname='rp-f0909c.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.in1_data = []
        self.all_in1 = []
        self.capture_timestamps = []
        self.acquisition_start_time = None

        # voltage scaling based on mode
        self.voltage_mode = voltage_mode.upper()
        if self.voltage_mode == 'HV':
            self.scale_factor = 20.0  # HV mode has built-in 1:20 divider
        elif self.voltage_mode == 'MANUAL':
            self.scale_factor = manual_gain  # use whatever gain you specify
        else:
            self.scale_factor = 1.0  # LV mode is 1:1

        self.scope = self.rp_modules.scope
        print("Available inputs:", self.scope.inputs)

        self.scope.input1 = 'in1'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Decimation value not allowed, pick from:', self.allowed_decimations)
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = False
        self.sample_rate = 125e6 / self.scope.decimation

    def capture_voltage(self):
        """Grab voltage data from scope and save timestamp"""
        capture_time = time.time()
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        
        # apply gain based on mode
        ch1 = ch1 * self.scale_factor
        
        self.in1_data.append(ch1)
        self.capture_timestamps.append(capture_time)
        return ch1

    def run(self, params):
        timeout = params['timeout']

        print("Letting scope settle...")
        time.sleep(0.1)

        self.acquisition_start_time = time.time()
        start_time_str = datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')
        print(f"\nStarted recording at: {start_time_str}")

        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_voltage()

        self.all_in1 = np.array(np.concatenate(self.in1_data))

        # make a timestamp for every single sample
        total_samples = len(self.all_in1)
        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.in1_data[i])
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        # smooth if requested
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            self.all_in1 = np.convolve(self.all_in1, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_in1)]
            print(f"Applied {averaging_window}-point moving average")

        # relative time array
        t = self.sample_timestamps - self.acquisition_start_time

        print("\n" + "="*60)
        print("MEASUREMENT SUMMARY")
        print("="*60)
        if self.voltage_mode == 'MANUAL':
            print(f"Voltage Mode: MANUAL (gain = {self.scale_factor}x)")
        else:
            print(f"Voltage Mode: {self.voltage_mode} (±{1.0 * self.scale_factor:.0f}V range)")
        print(f"Expected Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_in1)}")
        print(f"Duration: {t[-1]:.3f} seconds")

        # check if sample rate is what we expect
        actual_sample_rate = len(self.all_in1) / (t[-1] - t[0]) if t[-1] > t[0] else 0
        sample_rate_error = abs(actual_sample_rate - self.sample_rate) / self.sample_rate * 100
        print(f"Actual Sample Rate: {actual_sample_rate:.2f} Hz")
        print(f"Sample Rate Error: {sample_rate_error:.3f}%")

        if sample_rate_error > 1.0:
            print(f"WARNING: Sample rate off by {sample_rate_error:.3f}%")
        else:
            print(f"Sample rate looks good ({sample_rate_error:.3f}% error)")

        print(f"\nMean Voltage: {np.mean(self.all_in1):.6f} V")
        print(f"Std Dev: {np.std(self.all_in1):.6f} V")
        print("="*60)

        # plot it
        fig = plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)
        ax.plot(t, self.all_in1, 'b-', linewidth=0.8, label='DC Voltage')
        ax.axhline(np.mean(self.all_in1), color='r', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Mean: {np.mean(self.all_in1):.6f} V')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Voltage (V)', fontsize=12)
        ax.set_title(f'DC Voltage vs Time ({actual_sample_rate:.2f} Hz)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])

        # add text box with stats
        stats_text = f'Samples: {len(self.all_in1):,}\nDuration: {t[-1]:.2f} s\nStd Dev: {np.std(self.all_in1):.6f} V'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # save if requested
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            img_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved plot: {img_path}")

            # save data
            if params.get('save_timestamps', False):
                # save with timestamps for syncing with other data
                data = np.column_stack((
                    self.sample_timestamps,  # absolute unix timestamp
                    t,  # time relative to start (s)
                    self.all_in1,  # voltage
                ))
                csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                          header="AbsoluteTimestamp,RelativeTime,Voltage",
                          comments='', fmt='%.10f')
                print(f"Saved data with timestamps: {csv_path}")
            else:
                # just time and voltage
                data = np.column_stack((t, self.all_in1))
                csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                          header="Time,Voltage", comments='', fmt='%.6f')
                print(f"Saved data: {csv_path}")

        plt.show()


if __name__ == '__main__':
    rp = RedPitaya(voltage_mode=VOLTAGE_MODE, manual_gain=MANUAL_GAIN)

    run_params = {
        'timeout': MEASUREMENT_TIME,
        'averaging_window': AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'save_timestamps': SAVE_TIMESTAMPS,
    }

    print("="*60)
    print("Red Pitaya DC Voltage Monitor")
    print("="*60)
    print("Connect DC voltage source to IN1")
    print("="*60)
    if VOLTAGE_MODE.upper() == 'MANUAL':
        print(f"Voltage Mode: MANUAL (gain = {MANUAL_GAIN}x)")
    else:
        print(f"Voltage Mode: {VOLTAGE_MODE}")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Decimation: {DECIMATION}")
    print(f"Sample Rate: {125e6/DECIMATION:.2f} Hz")
    print(f"Averaging: {AVERAGING_WINDOW} samples")
    print(f"Save Timestamps: {SAVE_TIMESTAMPS}")
    print("="*60)

    rp.run(run_params)
