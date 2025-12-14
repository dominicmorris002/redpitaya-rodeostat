"""
Red Pitaya DC Voltage Monitor - WITH TIMESTAMP SYNCHRONIZATION

SETUP: Connect your DC voltage source to IN1

This version adds precise timestamps to every sample for easy synchronization
with external data acquisition systems (e.g., lock-in amplifier data).

SYNCHRONIZATION:
- Saves CSV with absolute timestamps (Unix time)
- Can be merged with lock-in data using timestamp matching

AUTOLAB PGSTAT302N SUPPORT:
- The IOUT cable output voltage depends on the current range selected
- This script can compensate for different current range scalings
- Typical IOUT scaling: 1V = max current of selected range
  (e.g., 10mA range: 1V = 10mA, 100µA range: 1V = 100µA)
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

# HV/LV MODE - Set this to match your physical jumper position
VOLTAGE_MODE = 'LV'  # 'LV' for ±1V range (jumper in LV position)
                     # 'HV' for ±20V range (jumper in HV position)

# ============================================================
# AUTOLAB PGSTAT302N CURRENT RANGE COMPENSATION
# ============================================================
# If measuring IOUT from Autolab, enable this and set current range
AUTOLAB_IOUT_MODE = False  # Set True if measuring Autolab IOUT cable

# Current range selected on Autolab (must match your instrument setting!)
AUTOLAB_CURRENT_RANGE = '10mA'  # Options: '10uA', '100uA', '1mA', '10mA'

# IOUT cable scaling: voltage output per unit current
# Default Autolab IOUT scaling (may vary by model - check manual!)
# Typical: 1V output = max current of selected range
AUTOLAB_IOUT_SCALING = {
    '10uA':  0.1,    # V/µA -> 1V = 10µA
    '100uA': 0.01,   # V/µA -> 1V = 100µA
    '1mA':   0.001,  # V/µA -> 1V = 1mA = 1000µA
    '10mA':  0.0001, # V/µA -> 1V = 10mA = 10000µA
}

# Or set custom scaling factor (V to current in µA)
# CUSTOM_IOUT_SCALING = 0.01  # Uncomment and set if your scaling differs
# ============================================================

from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from datetime import datetime


class RedPitaya:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', voltage_mode='LV', 
                 autolab_mode=False, current_range='10mA', iout_scaling=None):
        self.rp = Pyrpl(config='dc_monitor_config', hostname='rp-f0909c.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.in1_data = []
        self.all_in1 = []

        # Store capture timestamps
        self.capture_timestamps = []
        self.acquisition_start_time = None

        # Set voltage mode and scale factor
        self.voltage_mode = voltage_mode.upper()
        if self.voltage_mode == 'HV':
            self.scale_factor = 20.0  # HV mode has 1:20 attenuator
        else:
            self.scale_factor = 1.0   # LV mode is 1:1

        # Autolab IOUT compensation
        self.autolab_mode = autolab_mode
        self.current_range = current_range
        
        if self.autolab_mode:
            if iout_scaling is not None:
                self.iout_scaling = iout_scaling
            else:
                self.iout_scaling = AUTOLAB_IOUT_SCALING.get(current_range, 0.0001)
            
            # Calculate conversion factor: V -> µA
            self.current_conversion = 1.0 / self.iout_scaling
            
            print(f"\n{'='*60}")
            print("AUTOLAB PGSTAT302N IOUT MODE ENABLED")
            print(f"{'='*60}")
            print(f"Current Range: {current_range}")
            print(f"IOUT Scaling: {self.iout_scaling} V/µA")
            print(f"Conversion: 1V = {self.current_conversion:.2f} µA")
            print(f"Red Pitaya Mode: {self.voltage_mode} (±{1.0 * self.scale_factor:.0f}V)")
            
            if self.voltage_mode == 'LV':
                print(f"\n⚠ WARNING: LV mode (±1V) may clip for high current ranges!")
                print(f"   Consider HV mode (±20V) for larger signals")
            
            print(f"{'='*60}\n")

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
        
        # Apply scale factor for HV/LV mode
        ch1 = ch1 * self.scale_factor

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

        # Convert to current if in Autolab mode
        if self.autolab_mode:
            self.all_current = self.all_in1 * self.current_conversion  # µA
        
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
            if self.autolab_mode:
                self.all_current = np.convolve(self.all_current, np.ones(averaging_window) / averaging_window, mode='valid')
            # Also trim timestamps to match filtered data
            self.sample_timestamps = self.sample_timestamps[:len(self.all_in1)]
            print(f"Applied {averaging_window}-sample moving average filter")

        # Time array (relative to start)
        t = self.sample_timestamps - self.acquisition_start_time

        print("=" * 60)
        print("DC VOLTAGE DIAGNOSTICS")
        print("=" * 60)
        print(f"Voltage Mode: {self.voltage_mode} (±{1.0 * self.scale_factor:.0f}V range)")
        
        if self.autolab_mode:
            print(f"Autolab IOUT Mode: ENABLED")
            print(f"Current Range: {self.current_range}")
            print(f"IOUT Scaling: {self.iout_scaling} V/µA")
        
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
        
        if self.autolab_mode:
            print(f"Mean Current: {np.mean(self.all_current):.6f} µA ± {np.std(self.all_current):.6f} µA")
            print(f"Current Range: [{np.min(self.all_current):.6f}, {np.max(self.all_current):.6f}] µA")
        
        print("=" * 60)

        # Create plot
        fig = plt.figure(figsize=(14, 8))

        if self.autolab_mode:
            # Two subplots: voltage and current
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(t, self.all_in1, 'b-', linewidth=0.8, label='Voltage (IOUT)')
            ax1.axhline(np.mean(self.all_in1), color='r', linestyle='--', linewidth=2, alpha=0.7,
                        label=f'Mean: {np.mean(self.all_in1):.6f} V')
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Voltage (V)', fontsize=12)
            ax1.set_title(f'Autolab IOUT Voltage vs Time - {self.voltage_mode} Mode', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(t[0], t[-1])

            # Add stats text box
            stats_text = f'Mode: {self.voltage_mode}\nRange: ±{1.0 * self.scale_factor:.0f}V\nSamples: {len(self.all_in1):,}\nDuration: {t[-1]:.2f} s'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(t, self.all_current, 'g-', linewidth=0.8, label='Current')
            ax2.axhline(np.mean(self.all_current), color='r', linestyle='--', linewidth=2, alpha=0.7,
                        label=f'Mean: {np.mean(self.all_current):.6f} µA')
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Current (µA)', fontsize=12)
            ax2.set_title(f'Converted Current vs Time - Range: {self.current_range}', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(t[0], t[-1])

            # Add stats text box
            stats_text_current = f'Range: {self.current_range}\nScaling: {self.iout_scaling} V/µA\nMean: {np.mean(self.all_current):.3f} µA\nStd Dev: {np.std(self.all_current):.6f} µA'
            ax2.text(0.02, 0.98, stats_text_current, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        else:
            # Single plot: just voltage
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
                if self.autolab_mode:
                    # Save with voltage and current
                    data = np.column_stack((
                        self.sample_timestamps,  # Absolute Unix timestamp
                        t,  # Relative time (s)
                        self.all_in1,  # Voltage
                        self.all_current,  # Current (µA)
                    ))
                    csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
                    
                    # Create header with metadata
                    header_lines = [
                        f"# Autolab IOUT Mode: {self.current_range}",
                        f"# IOUT Scaling: {self.iout_scaling} V/µA",
                        f"# Red Pitaya Mode: {self.voltage_mode}",
                        "AbsoluteTimestamp,RelativeTime,Voltage,Current_uA"
                    ]
                    
                    with open(csv_path, 'w') as f:
                        f.write('\n'.join(header_lines) + '\n')
                        np.savetxt(f, data, delimiter=",", fmt='%.10f')
                    
                    print(f"✓ Data saved with timestamps: {csv_path}")
                    print(f"  Columns: AbsoluteTimestamp (Unix), RelativeTime (s), Voltage (V), Current (µA)")
                else:
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
                if self.autolab_mode:
                    data = np.column_stack((t, self.all_in1, self.all_current))
                    csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
                    np.savetxt(csv_path, data, delimiter=",",
                               header="Time,Voltage,Current_uA", comments='', fmt='%.6f')
                else:
                    data = np.column_stack((t, self.all_in1))
                    csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
                    np.savetxt(csv_path, data, delimiter=",",
                               header="Time,Voltage", comments='', fmt='%.6f')
                print(f"✓ Data saved: {csv_path}")

        plt.show()


if __name__ == '__main__':
    # Determine IOUT scaling to use
    iout_scaling = None
    # Uncomment below to use custom scaling instead of defaults
    # iout_scaling = CUSTOM_IOUT_SCALING
    
    rp = RedPitaya(
        voltage_mode=VOLTAGE_MODE,
        autolab_mode=AUTOLAB_IOUT_MODE,
        current_range=AUTOLAB_CURRENT_RANGE,
        iout_scaling=iout_scaling
    )

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
    print(f"Voltage Mode: {VOLTAGE_MODE}")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Decimation: {DECIMATION}")
    print(f"Expected Sample Rate: {125e6 / DECIMATION:.2f} Hz")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print(f"Save Timestamps: {SAVE_TIMESTAMPS}")
    
    if AUTOLAB_IOUT_MODE:
        print(f"\nAutolab IOUT Mode: ENABLED")
        print(f"Current Range: {AUTOLAB_CURRENT_RANGE}")
        scaling = iout_scaling if iout_scaling else AUTOLAB_IOUT_SCALING.get(AUTOLAB_CURRENT_RANGE, 0.0001)
        print(f"IOUT Scaling: {scaling} V/µA")
        print("\n⚠ IMPORTANT: Make sure the current range matches your Autolab setting!")
        if VOLTAGE_MODE == 'LV':
            print("⚠ Consider HV mode for higher current ranges to avoid clipping")
    
    print("=" * 60)
    print("\nNOTE: Timestamps will be saved for synchronization with lock-in data")
    print("=" * 60)

    rp.run(run_params)
