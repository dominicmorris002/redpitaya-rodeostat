"""
Red Pitaya Lock-In Y Component + DC Ramp Logger
Architecture matches Lock-In logger exactly.

Scope setup:
- CH1: iq2_2 (Y component from lock-in)
- CH2: in2 (DC ramp reference signal)

Connect your DC ramp to IN2.

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
REF_FREQUENCY = 500  # Hz
REF_AMPLITUDE = 0.2  # V (not used, Y device doesn't generate reference)
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 12.0  # seconds

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'
MANUAL_GAIN_FACTOR = 1.0  # Y component gain correction
MANUAL_DC_OFFSET = 0.0  # Y component DC offset

FILTER_BANDWIDTH = 10  # Hz
AVERAGING_WINDOW = 10  # samples (moving average on logged data)
DECIMATION = 1024

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

AUTO_CALIBRATE = True  # Only used if INPUT_MODE = 'AUTO'
CALIBRATION_TIME = 2.0  # seconds

# ACQUISITION MODE: 'SINGLE_SHOT' or 'CONTINUOUS'
ACQUISITION_MODE = 'SINGLE_SHOT'
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


class RedPitayaLockInYLogger:
    """Y component logger using lock-in + DC ramp reference"""

    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO',
                 manual_gain=1.0, manual_offset=0.0):

        self.rp = Pyrpl(config='dc_config5', hostname='rp-f0909c.local')
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.scope = self.rp_modules.scope
        self.asg = self.rp_modules.asg0
        self.output_dir = output_dir

        self.lockin_Y = []
        self.dc_ramp = []
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

        # Scope setup - read Y component and DC ramp
        self.scope.input1 = 'iq2_2'  # Y component from lock-in
        self.scope.input2 = 'in2'  # DC ramp reference
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f"Invalid decimation {DECIMATION}")

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True
        self.nominal_sample_rate = 125e6 / self.scope.decimation

        print(f"Y Component Logger initialized")
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

        # Restore scope to Y component and DC ramp
        self.scope.input1 = 'iq2_2'
        self.scope.input2 = 'in2'

        return self.input_gain_factor

    def setup_lockin(self, params):
        """Configure the lock-in (but don't generate reference signal)"""
        self.ref_freq = params['ref_freq']  # Store ref_freq for plotting
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        # Setup lock-in to listen for reference from other Red Pitaya
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,
            phase=phase_setting,
            acbandwidth=0,
            amplitude=0.0,  # Don't generate output
            input='in1',
            output_direct='off',  # No output
            output_signal='quadrature',
            quadrature_factor=1.0)

        actual_freq = self.lockin.frequency

        print(f"\nY Lock-in Configuration:")
        print(f"  Frequency: {self.ref_freq} Hz (actual: {actual_freq:.2f} Hz)")
        print(f"  Bandwidth: {filter_bw} Hz")
        print(f"  Gain correction: {self.input_gain_factor:.4f}x")
        print(f"  DC offset correction: {self.input_dc_offset:.6f}V")
        print(f"  (Listening for reference from other Red Pitaya)")

    def capture_buffer(self):
        """Capture one scope buffer (single shot)"""
        capture_time = time.time()
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)  # Y component
        ch2 = np.array(self.scope._data_ch2_current)  # DC ramp
        self.lockin_Y.append(ch1)
        self.dc_ramp.append(ch2)
        self.capture_times.append(capture_time)

    def capture_buffer_continuous(self):
        """Capture one scope buffer in continuous mode (no blocking single() call)"""
        capture_time = time.time()
        ch1 = np.array(self.scope._data_ch1_current)  # Y component
        ch2 = np.array(self.scope._data_ch2_current)  # DC ramp
        self.lockin_Y.append(ch1)
        self.dc_ramp.append(ch2)
        self.capture_times.append(capture_time)

    def run(self, params):
        # Auto-calibrate if requested
        if params.get('auto_calibrate', False):
            self.calibrate_input_gain(
                cal_freq=100,  # Use 100 Hz for calibration
                cal_amp=1.0,
                cal_time=params.get('calibration_time', 2.0)
            )

        # Setup lock-in
        self.setup_lockin(params)

        print("\nAllowing lock-in to settle...")
        time.sleep(0.5)

        acquisition_start = time.time()
        print(f"Started: {datetime.fromtimestamp(acquisition_start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        acq_mode = params.get('acquisition_mode', 'SINGLE_SHOT')

        # Collect data!
        if acq_mode == 'CONTINUOUS':
            self.scope.continuous()
            time.sleep(0.1)  # let it start
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_buffer_continuous()
                time.sleep(0.001)  # small delay to avoid overwhelming the system
        else:  # SINGLE_SHOT (default)
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_buffer()

        acquisition_end = time.time()
        actual_duration = acquisition_end - acquisition_start
        capture_count = len(self.lockin_Y)
        print(f"✓ Captured {capture_count} buffers")

        # Concatenate all buffers
        all_Y_raw = np.concatenate(self.lockin_Y)
        all_ramp_raw = np.concatenate(self.dc_ramp)

        # Apply gain correction to Y
        all_Y = all_Y_raw * self.input_gain_factor

        # Apply averaging if requested
        w = params.get('averaging_window', 1)
        if w > 1:
            all_Y = np.convolve(all_Y, np.ones(w) / w, mode='valid')
            all_ramp_raw = np.convolve(all_ramp_raw, np.ones(w) / w, mode='valid')
            print(f"Applied {w}-sample moving average")

        n_samples = len(all_Y)

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
        print("LOCK-IN Y COMPONENT + DC RAMP RESULTS")
        print("=" * 60)
        print(f"Mode: {self.input_mode}")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")
        print(f"DC offset correction: {self.input_dc_offset:.6f}V")
        print(f"Duration: {actual_duration:.3f}s")
        print(f"Samples: {n_samples}")
        print(f"Effective sample rate: {effective_sample_rate:.2f} Hz")
        print(f"Mean Y: {np.mean(all_Y):.6f} V")
        print(f"Std dev: {np.std(all_Y):.6f} V")
        print(f"DC Ramp range: {np.min(all_ramp_raw):.6f} to {np.max(all_ramp_raw):.6f} V")
        print("\nBuffer statistics:")
        print(f"  Buffers: {capture_count}")
        print(f"  Samples per buffer: {samples_per_buffer:.0f}")
        print(f"  Data time per buffer: {data_time_per_buffer:.3f}s")
        print(f"  Gap per buffer: {dead_time * 1000:.1f} ms")
        print(
            f"  Dead time: {dead_time * capture_count:.2f}s ({dead_time * capture_count / actual_duration * 100:.1f}%)")
        print("=" * 60)

        # Grab raw signals for plots
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw_uncorrected = np.array(self.scope._data_ch2_current)
        in1_raw = (in1_raw_uncorrected - self.input_dc_offset) * self.input_gain_factor

        # Put scope back to lock-in Y and DC ramp
        self.scope.input1 = 'iq2_2'
        self.scope.input2 = 'in2'

        # Calculate magnitude (just Y for this logger, no X available)
        R_y = np.abs(all_Y)

        # Make plots - 3x3 layout matching original
        fig = plt.figure(figsize=(16, 10))

        t_raw_corrected = np.linspace(0, len(out1_raw) / self.nominal_sample_rate, len(out1_raw))

        # Reference signal (listening only)
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_plot = min(int(n_periods * self.nominal_sample_rate / self.ref_freq), len(out1_raw))
        ax1.plot(t_raw_corrected[:n_plot] * 1000, out1_raw[:n_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference @ {self.ref_freq} Hz (from RP1)')
        ax1.grid(True)

        # Input signal
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw_corrected[:n_plot] * 1000, in1_raw[:n_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V, corrected)')
        ax2.set_title(f'Input - {self.input_mode}')
        ax2.grid(True)

        # DC Ramp vs Time
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(t, all_ramp_raw, 'g-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('DC Ramp (V)')
        ax3.set_title('DC Ramp (IN2) vs Time')
        ax3.grid(True)

        # Y vs Time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, all_Y, 'r-', linewidth=0.5)
        ax4.axhline(np.mean(all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(all_Y):.9f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Y (V)')
        ax4.set_title('Quadrature (Y)')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(t[0], t[-1])
        margin_Y = 5 * (np.max(all_Y) - np.min(all_Y)) if np.max(all_Y) != np.min(all_Y) else 0.001
        ax4.set_ylim(np.min(all_Y) - margin_Y, np.max(all_Y) + margin_Y)

        # Y vs DC Ramp
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(all_ramp_raw, all_Y, 'm-', linewidth=0.5)
        ax5.set_xlabel('DC Ramp (V)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Y vs DC Ramp')
        ax5.grid(True)



        plt.tight_layout()

        # Save figure and CSV
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = os.path.join(self.output_dir, f"lockin_y_ramp_{ts}.png")
            plt.savefig(png_path, dpi=150)
            print(f"\n✓ Saved plot: {png_path}")

            csv_path = os.path.join(self.output_dir, f"lockin_y_ramp_{ts}.csv")
            with open(csv_path, 'w') as f:
                f.write("# Red Pitaya Lock-In Y Component + DC Ramp Logger\n")
                f.write(f"# Mode: {self.input_mode}\n")
                f.write(f"# Gain: {self.input_gain_factor:.6f}\n")
                f.write(f"# Offset: {self.input_dc_offset:.6f}\n")
                f.write(f"# Duration: {actual_duration:.6f}\n")
                f.write(f"# Sample rate: {effective_sample_rate:.3f} Hz\n")
                f.write(f"# Samples: {n_samples}\n")
                f.write("Index,Time(s),Y(V),DCRamp(V)\n")
                np.savetxt(f, np.column_stack((sample_index, t, all_Y, all_ramp_raw)), delimiter=",", fmt="%.10f")
            print(f"✓ Saved data: {csv_path}")


if __name__ == "__main__":
    rp = RedPitayaLockInYLogger(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR,
        manual_offset=MANUAL_DC_OFFSET
    )

    run_params = {
        'ref_freq': REF_FREQUENCY,
        'phase': PHASE_OFFSET,
        'timeout': MEASUREMENT_TIME,
        'filter_bandwidth': FILTER_BANDWIDTH,
        'averaging_window': AVERAGING_WINDOW,
        'save_file': SAVE_DATA,
        'auto_calibrate': AUTO_CALIBRATE,
        'calibration_time': CALIBRATION_TIME,
        'acquisition_mode': ACQUISITION_MODE,
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN Y + DC RAMP LOGGER")
    print("=" * 60)
    print(f"Reference frequency: {REF_FREQUENCY} Hz (listening)")
    print(f"Filter bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement time: {MEASUREMENT_TIME}s")
    print(f"Input mode: {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"Manual gain: {MANUAL_GAIN_FACTOR}x")
        print(f"Manual DC offset: {MANUAL_DC_OFFSET}V")
    print("=" * 60)

    rp.run(run_params)
