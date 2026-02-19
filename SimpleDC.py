"""
Red Pitaya DC Voltage Data Logger
Architecture matches Lock-In logger exactly.
Single plot: Voltage vs Time
Connect your DC signal to IN1.

To do AC CV use the startboth file

Have a Great Day ;)

Dominic Morris
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

# ACQUISITION MODE: 'SINGLE_SHOT' or 'CONTINUOUS'
ACQUISITION_MODE = 'CONTINUOUS'

# ============================================================
# PLOT DOWNSAMPLING
# ============================================================
# CSV always saves full resolution data no matter what.
PLOT_DOWNSAMPLE_ENABLED = True
PLOT_MAX_POINTS = 50_000
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


# ============================================================
# Data Helpers
# ============================================================

def downsample(arrays, n_samples, max_points, enabled=True):
    if not enabled or n_samples <= max_points:
        return arrays, 1, n_samples
    step = max(1, n_samples // max_points)
    ds = [arr[::step] for arr in arrays]
    return ds, step, len(ds[0])


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
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT SCALING AND DC OFFSET...")
        print("=" * 60)

        print("Step 1: Measuring DC offset on IN1 (no signal)...")
        self.asg.output_direct = 'off'
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'
        time.sleep(0.3)

        offset_samples = []
        for _ in range(10):
            self.scope.single()
            offset_samples.append(np.mean(self.scope._data_ch1_current))

        self.input_dc_offset = np.mean(offset_samples)
        print(f"  Measured DC offset: {self.input_dc_offset:.6f}V")

        print(f"\nStep 2: Measuring gain with {cal_amp}V @ {cal_freq} Hz...")
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'

        self.asg.setup(
            frequency=cal_freq,
            amplitude=cal_amp,
            offset=0,
            waveform='sin',
            trigger_source='immediately'
        )
        self.asg.output_direct = 'out1'
        time.sleep(0.5)

        cal_out1 = []
        cal_in1 = []
        start_time = time.time()

        while (time.time() - start_time) < cal_time:
            self.scope.single()
            cal_out1.append(np.array(self.scope._data_ch1_current))
            cal_in1.append(np.array(self.scope._data_ch2_current))

        all_out1 = np.concatenate(cal_out1)
        all_in1 = np.concatenate(cal_in1)
        all_in1_corrected = all_in1 - self.input_dc_offset

        out1_peak = (np.max(all_out1) - np.min(all_out1)) / 2
        in1_peak = (np.max(all_in1_corrected) - np.min(all_in1_corrected)) / 2
        out1_rms = np.sqrt(np.mean(all_out1 ** 2))
        in1_rms = np.sqrt(np.mean(all_in1_corrected ** 2))

        self.input_gain_factor = out1_peak / in1_peak
        gain_rms = out1_rms / in1_rms

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

        self.asg.output_direct = 'off'
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'

        return self.input_gain_factor

    def capture_buffer(self):
        """Single-shot: trigger a capture and wait for it"""
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        self.buffers.append(ch1)
        self.capture_times.append(time.time())

    def capture_buffer_continuous(self):
        """Continuous: just read whatever is currently in the buffer"""
        ch1 = np.array(self.scope._data_ch1_current)
        self.buffers.append(ch1)
        self.capture_times.append(time.time())

    def run(self, params):
        if params.get('auto_calibrate', False):
            self.calibrate_input_gain(
                cal_freq=100,
                cal_amp=1.0,
                cal_time=params.get('calibration_time', 2.0)
            )

        print("\nAllowing scope to settle...")
        time.sleep(0.3)

        acquisition_start = time.time()
        print(f"Started: {datetime.fromtimestamp(acquisition_start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        acq_mode = params.get('acquisition_mode', 'SINGLE_SHOT')

        if acq_mode == 'CONTINUOUS':
            self.scope.continuous()
            time.sleep(0.1)
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_buffer_continuous()
                time.sleep(0.001)
        else:
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_buffer()

        actual_duration = time.time() - acquisition_start
        capture_count = len(self.buffers)
        print(f"Captured {capture_count} buffers")

        # Concatenate and correct
        raw = np.concatenate(self.buffers)
        corrected = (raw - self.input_dc_offset) * self.input_gain_factor

        w = params.get('averaging_window', 1)
        if w > 1:
            corrected = np.convolve(corrected, np.ones(w) / w, mode='valid')
            print(f"Applied {w}-sample moving average")

        n_samples = len(corrected)
        sample_index = np.arange(n_samples)
        t = sample_index / (n_samples / actual_duration)
        effective_sample_rate = n_samples / actual_duration

        # Buffer timing diagnostics
        samples_per_buffer = n_samples / capture_count
        data_time_per_buffer = samples_per_buffer / self.nominal_sample_rate
        buffer_spacing = actual_duration / capture_count
        dead_time = buffer_spacing - data_time_per_buffer

        print("\n" + "=" * 60)
        print("DC MEASUREMENT RESULTS")
        print("=" * 60)
        print(f"Mode:          {self.input_mode}")
        print(f"Acq. mode:     {acq_mode}")
        print(f"Gain:          {self.input_gain_factor:.4f}x")
        print(f"DC offset:     {self.input_dc_offset:.6f}V")
        print(f"Duration:      {actual_duration:.3f}s")
        print(f"Samples:       {n_samples:,}")
        print(f"Sample rate:   {effective_sample_rate:.2f} Hz")
        print(f"Mean voltage:  {np.mean(corrected):.6f} V")
        print(f"Std dev:       {np.std(corrected):.6f} V")
        print(f"\nBuffer stats:")
        print(f"  Buffers: {capture_count}")
        print(f"  Samples/buf: {samples_per_buffer:.0f}")
        print(f"  Gap/buf: {dead_time * 1000:.1f} ms")
        print(f"  Dead time: {dead_time * capture_count:.2f}s "
              f"({dead_time * capture_count / actual_duration * 100:.1f}%)")
        print("=" * 60)

        # ── Downsample for plotting ───────────────────────────────────────────
        ds_on  = params.get('plot_downsample_enabled', True)
        ds_max = params.get('plot_max_points', 50_000)

        ds_arrays, ds_step, n_plot = downsample(
            [t, corrected], n_samples, ds_max, enabled=ds_on)
        t_p, v_p = ds_arrays

        if ds_step > 1:
            print(f"\nDownsampled: {n_samples:,} -> {n_plot:,} pts "
                  f"(step={ds_step}) for plotting. CSV = full res.")
        else:
            print(f"\nNo downsampling needed ({n_samples:,} pts)")

        ds_note = f' [1:{ds_step} for plot]' if ds_step > 1 else ''

        # ── Plot ─────────────────────────────────────────────────────────────
        plt.figure(figsize=(14, 6))
        plt.plot(t_p, v_p, linewidth=0.8, label=f'Voltage{ds_note}')
        plt.axhline(np.mean(corrected), color='r', linestyle='--',
                    label=f"Mean {np.mean(corrected):.6f} V")
        plt.fill_between(t_p,
                         np.mean(corrected) - np.std(corrected),
                         np.mean(corrected) + np.std(corrected),
                         alpha=0.15, color='blue', label=f'+/-1σ ({np.std(corrected):.6f} V)')
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"DC Voltage vs Time - {self.input_mode}\n"
                  f"Acq: {acq_mode}  |  {n_samples:,} samples @ {effective_sample_rate:.1f} Hz")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # ── Save ─────────────────────────────────────────────────────────────
        if params['save_file']:
            os.makedirs(self.output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            png_path = os.path.join(self.output_dir, f"dc_voltage_{ts}.png")
            plt.savefig(png_path, dpi=150)
            print(f"\nPlot: {png_path}")

            csv_path = os.path.join(self.output_dir, f"dc_voltage_{ts}.csv")
            with open(csv_path, 'w', newline='', encoding='ascii') as f:
                f.write("# Red Pitaya DC Logger\n")
                f.write(f"# Mode: {self.input_mode}\n")
                f.write(f"# Acquisition: {acq_mode}\n")
                f.write(f"# Gain: {self.input_gain_factor:.6f}\n")
                f.write(f"# Offset: {self.input_dc_offset:.6f}\n")
                f.write(f"# Duration: {actual_duration:.6f}\n")
                f.write(f"# Sample rate: {effective_sample_rate:.3f} Hz\n")
                f.write(f"# Samples: {n_samples}\n")
                f.write(f"# Plot downsample step: {ds_step}x  (CSV is full resolution)\n")
                f.write("Index,Time(s),Voltage(V)\n")
                np.savetxt(f, np.column_stack((sample_index, t, corrected)),
                           delimiter=",", fmt="%.10f")
            print(f"Data: {csv_path}")
        else:
            plt.show()


if __name__ == "__main__":
    rp = RedPitayaDCLogger(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR,
        manual_offset=MANUAL_DC_OFFSET
    )

    run_params = {
        'timeout':                 MEASUREMENT_TIME,
        'averaging_window':        AVERAGING_WINDOW,
        'save_file':               SAVE_DATA,
        'auto_calibrate':          AUTO_CALIBRATE,
        'calibration_time':        CALIBRATION_TIME,
        'acquisition_mode':        ACQUISITION_MODE,
        'plot_downsample_enabled': PLOT_DOWNSAMPLE_ENABLED,
        'plot_max_points':         PLOT_MAX_POINTS,
    }

    print("=" * 60)
    print("RED PITAYA DC VOLTAGE DATA LOGGER")
    print("=" * 60)
    print(f"Measurement time: {MEASUREMENT_TIME}s")
    print(f"Input mode:       {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"  Gain:   {MANUAL_GAIN_FACTOR}x")
        print(f"  Offset: {MANUAL_DC_OFFSET}V")
    print(f"Acq. mode:        {ACQUISITION_MODE}")
    print(f"Plot DS:          {'ON -- max ' + str(PLOT_MAX_POINTS) + ' pts' if PLOT_DOWNSAMPLE_ENABLED else 'OFF'}")
    print("=" * 60)

    rp.run(run_params)
