"""
Red Pitaya DC Voltage Data Logger 
Architecture matches Lock-In logger exactly.
Single plot: Voltage vs Time
Connect your DC signal to IN1.

To do AC CV use the startboth file

Have a Great Day ;)


"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from pyrpl import Pyrpl

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
MEASUREMENT_TIME = 12.0  # seconds
DECIMATION = 1024
AVERAGING_WINDOW = 10

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'
MANUAL_GAIN_FACTOR = 28.93002007*-1   #28.93002007*-1
MANUAL_DC_OFFSET = -0.016903         #-0.016903

AUTO_CALIBRATE = True  # Only used if INPUT_MODE = 'AUTO'
CALIBRATION_TIME = 2.0  # seconds

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
# ============================================================

START_TIME_FILE = "start_time.txt"

# Synchronization with other processes
try:
    with open(START_TIME_FILE, "r") as f:
        START_TIME = datetime.fromisoformat(f.read().strip())
    while datetime.now() < START_TIME:
        time.sleep(0.001)
except FileNotFoundError:
    pass  # No sync file, start immediately


class RedPitayaDCLogger:
    """DC voltage logger using lock-in style acquisition"""

    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO',
                 manual_gain=1.0, manual_offset=0.0):

        self.rp = Pyrpl(config='dc_config5', hostname='rp-f0909c.local')
        self.rp_modules = self.rp.rp
        self.scope = self.rp_modules.scope
        self.asg = self.rp_modules.asg0
        self.output_dir = output_dir

        self.buffers = []
        self.capture_times = []

        # Setup input gain scaling and DC offset
        self.input_gain_factor = manual_gain
        self.input_dc_offset = manual_offset
        self.input_mode_setting = input_mode.upper()
        self.input_mode = "Unknown"

        if self.input_mode_setting == 'MANUAL':
            self.input_gain_factor = manual_gain
            self.input_dc_offset = manual_offset
            self.input_mode = f"MANUAL ({manual_gain}x gain, {manual_offset}V offset)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_dc_offset = 0.0
            self.input_mode = "LV (±1V)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_dc_offset = 0.0
            self.input_mode = "HV (±20V, 20:1 divider)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'AUTO':
            self.input_mode = "AUTO (will calibrate)"
            print("Input mode: AUTO - will auto-detect")

        # Scope setup (matches lock-in style)
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f"Invalid decimation {DECIMATION}")

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True
        self.nominal_sample_rate = 125e6 / self.scope.decimation

        print(f"DC Logger initialized")
        print(f"Nominal sample rate: {self.nominal_sample_rate:.2f} Hz")

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        """
        Measure actual input scaling and DC offset by comparing OUT1 to IN1 directly.
        This measures the physical signal on IN1.
        """
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT SCALING AND DC OFFSET...")
        print("=" * 60)

        # Step 1: Measure DC offset with no signal
        print("Step 1: Measuring DC offset on IN1 (no signal)...")
        self.asg.output_direct = 'off'

        # Configure scope to read raw IN1
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'
        time.sleep(0.3)  # Let signal settle

        offset_samples = []
        for _ in range(10):
            self.scope.single()
            offset_samples.append(np.mean(self.scope._data_ch1_current))

        self.input_dc_offset = np.mean(offset_samples)
        print(f"  Measured DC offset: {self.input_dc_offset:.6f}V")

        # Step 2: Measure gain with calibration signal
        print(f"\nStep 2: Measuring gain with {cal_amp}V @ {cal_freq} Hz...")

        # Configure scope to read OUT1 and IN1
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'

        # Generate calibration signal using ASG
        self.asg.setup(
            frequency=cal_freq,
            amplitude=cal_amp,
            offset=0,
            waveform='sin',
            trigger_source='immediately'
        )
        self.asg.output_direct = 'out1'

        time.sleep(0.5)  # Let signal settle

        cal_out1 = []
        cal_in1 = []
        start_time = time.time()

        while (time.time() - start_time) < cal_time:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # OUT1
            ch2 = np.array(self.scope._data_ch2_current)  # IN1 (raw)
            cal_out1.append(ch1)
            cal_in1.append(ch2)

        # Concatenate all samples
        all_out1 = np.concatenate(cal_out1)
        all_in1 = np.concatenate(cal_in1)

        # Remove DC offset from IN1
        all_in1_corrected = all_in1 - self.input_dc_offset

        # Calculate peak-to-peak amplitudes
        out1_peak = (np.max(all_out1) - np.min(all_out1)) / 2
        in1_peak = (np.max(all_in1_corrected) - np.min(all_in1_corrected)) / 2

        # Also calculate RMS for comparison
        out1_rms = np.sqrt(np.mean(all_out1 ** 2))
        in1_rms = np.sqrt(np.mean(all_in1_corrected ** 2))

        # Gain factor = what we sent / what we measured
        self.input_gain_factor = out1_peak / in1_peak
        gain_rms = out1_rms / in1_rms

        # Classify the mode based on gain
        if self.input_gain_factor < 1.05:
            self.input_mode = "LV (±1V)"
        elif self.input_gain_factor < 2.0:
            self.input_mode = f"LV with loading ({self.input_gain_factor:.2f}x)"
        elif self.input_gain_factor < 15:
            self.input_mode = f"Custom/Unknown mode ({self.input_gain_factor:.2f}x)"
        else:
            self.input_mode = f"HV (±20V, {self.input_gain_factor:.1f}:1 divider)"

        print(f"\n  OUT1 peak: {out1_peak:.4f}V, RMS: {out1_rms:.4f}V")
        print(f"  IN1 peak (after offset correction): {in1_peak:.4f}V, RMS: {in1_rms:.4f}V")
        print(f"  Gain (peak-based): {self.input_gain_factor:.4f}x")
        print(f"  Gain (RMS-based): {gain_rms:.4f}x")
        print(f"  DC offset: {self.input_dc_offset:.6f}V")
        print(f"  Detected mode: {self.input_mode}")
        print("=" * 60 + "\n")

        # Turn off calibration signal
        self.asg.output_direct = 'off'

        # Restore scope to IN1 only
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'

        return self.input_gain_factor

    def capture_buffer(self):
        """Capture one scope buffer"""
        capture_time = time.time()
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        self.buffers.append(ch1)
        self.capture_times.append(capture_time)

    def run(self, params):
        # Auto-calibrate if requested
        if params.get('auto_calibrate', False):
            self.calibrate_input_gain(
                cal_freq=100,  # Use 100 Hz for calibration
                cal_amp=1.0,
                cal_time=params.get('calibration_time', 2.0)
            )

        print("\nAllowing scope to settle...")
        time.sleep(0.3)

        acquisition_start = time.time()
        print(f"Started: {datetime.fromtimestamp(acquisition_start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        loop_start = time.time()
        while (time.time() - loop_start) < params['timeout']:
            self.capture_buffer()

        acquisition_end = time.time()
        actual_duration = acquisition_end - acquisition_start
        capture_count = len(self.buffers)
        print(f"✓ Captured {capture_count} buffers")

        # Concatenate all buffers
        raw = np.concatenate(self.buffers)

        # Apply manual gain/offset
        corrected = (raw - self.input_dc_offset) * self.input_gain_factor

        # Apply averaging if requested
        w = params.get('averaging_window', 1)
        if w > 1:
            corrected = np.convolve(corrected, np.ones(w) / w, mode='valid')
            print(f"Applied {w}-sample moving average")

        n_samples = len(corrected)

        # Use sample-based indexing (matches lock-in, better for sync)
        sample_index = np.arange(n_samples)
        t = sample_index / (n_samples / actual_duration)

        effective_sample_rate = n_samples / actual_duration

        # Buffer timing diagnostics
        samples_per_buffer = n_samples / capture_count
        data_time_per_buffer = samples_per_buffer / self.nominal_sample_rate
        buffer_spacing = actual_duration / capture_count
        dead_time = buffer_spacing - data_time_per_buffer

        # Print summary
        print("\n" + "=" * 60)
        print("DC MEASUREMENT RESULTS")
        print("=" * 60)
        print(f"Mode: {self.input_mode}")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")
        print(f"DC offset correction: {self.input_dc_offset:.6f}V")
        print(f"Duration: {actual_duration:.3f}s")
        print(f"Samples: {n_samples}")
        print(f"Effective sample rate: {effective_sample_rate:.2f} Hz")
        print(f"Mean voltage: {np.mean(corrected):.6f} V")
        print(f"Std dev: {np.std(corrected):.6f} V")
        print("\nBuffer statistics:")
        print(f"  Buffers: {capture_count}")
        print(f"  Samples per buffer: {samples_per_buffer:.0f}")
        print(f"  Data time per buffer: {data_time_per_buffer:.3f}s")
        print(f"  Gap per buffer: {dead_time * 1000:.1f} ms")
        print(
            f"  Dead time: {dead_time * capture_count:.2f}s ({dead_time * capture_count / actual_duration * 100:.1f}%)")
        print("=" * 60)

        # Plot voltage vs time
        plt.figure(figsize=(14, 6))
        plt.plot(t, corrected, linewidth=0.8)
        plt.axhline(np.mean(corrected), color='r', linestyle='--',
                    label=f"Mean {np.mean(corrected):.6f} V")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"DC Voltage vs Time - {self.input_mode}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure and CSV
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = os.path.join(self.output_dir, f"dc_voltage_{ts}.png")
            plt.savefig(png_path, dpi=150)
            print(f"\n✓ Saved plot: {png_path}")

            csv_path = os.path.join(self.output_dir, f"dc_voltage_{ts}.csv")
            with open(csv_path, 'w') as f:
                f.write("# Red Pitaya DC Logger\n")
                f.write(f"# Mode: {self.input_mode}\n")
                f.write(f"# Gain: {self.input_gain_factor:.6f}\n")
                f.write(f"# Offset: {self.input_dc_offset:.6f}\n")
                f.write(f"# Duration: {actual_duration:.6f}\n")
                f.write(f"# Sample rate: {effective_sample_rate:.3f} Hz\n")
                f.write(f"# Samples: {n_samples}\n")
                f.write("Index,Time(s),Voltage(V)\n")
                np.savetxt(f, np.column_stack((sample_index, t, corrected)), delimiter=",", fmt="%.10f")
            print(f"✓ Saved data: {csv_path}")




if __name__ == "__main__":
    rp = RedPitayaDCLogger(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR,
        manual_offset=MANUAL_DC_OFFSET
    )

    run_params = {
        'timeout': MEASUREMENT_TIME,
        'averaging_window': AVERAGING_WINDOW,
        'save_file': SAVE_DATA,
        'auto_calibrate': AUTO_CALIBRATE,
        'calibration_time': CALIBRATION_TIME,
    }

    print("=" * 60)
    print("RED PITAYA DC VOLTAGE DATA LOGGER")
    print("=" * 60)
    print(f"Measurement time: {MEASUREMENT_TIME}s")
    print(f"Input mode: {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"Manual gain: {MANUAL_GAIN_FACTOR}x")
        print(f"Manual DC offset: {MANUAL_DC_OFFSET}V")
    print("=" * 60)

    rp.run(run_params)
