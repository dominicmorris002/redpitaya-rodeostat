"""
Red Pitaya DC Voltage Data Logger (Lock-In Style)
Architecture matches Lock-In logger exactly.
Single plot: Voltage vs Time
Connect your DC signal to IN1.
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
MEASUREMENT_TIME = 10.0  # seconds
DECIMATION = 1024
AVERAGING_WINDOW = 1

INPUT_MODE = 'MANUAL'  # 'LV', 'HV', or 'MANUAL'
MANUAL_GAIN_FACTOR = 27.5760
MANUAL_DC_OFFSET = -0.016379

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
# ============================================================


class RedPitayaDCLogger:
    """DC voltage logger using lock-in style acquisition"""

    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='LV',
                 manual_gain=1.0, manual_offset=0.0):

        self.rp = Pyrpl(config='dc_config5', hostname='rp-f0909c.local')
        self.rp_modules = self.rp.rp
        self.scope = self.rp_modules.scope
        self.output_dir = output_dir

        self.buffers = []
        self.capture_times = []

        self.input_mode_setting = input_mode.upper()
        self.input_gain_factor = manual_gain
        self.input_dc_offset = manual_offset

        if self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_dc_offset = 0.0
            self.input_mode = "LV (±1V)"
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_dc_offset = 0.0
            self.input_mode = "HV (±20V)"
        else:
            self.input_mode = f"MANUAL ({manual_gain}x, offset {manual_offset} V)"

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
        print(f"Input mode: {self.input_mode}")
        print(f"Nominal sample rate: {self.nominal_sample_rate:.2f} Hz")

    def capture_buffer(self):
        """Capture one scope buffer"""
        capture_time = time.time()
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        self.buffers.append(ch1)
        self.capture_times.append(capture_time)

    def run(self, params):
        print("\nAllowing scope to settle...")
        time.sleep(0.3)

        acquisition_start = time.time()
        print(f"Started: {datetime.fromtimestamp(acquisition_start)}")

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
            corrected = np.convolve(corrected, np.ones(w)/w, mode='valid')
            print(f"Applied {w}-sample moving average")

        n_samples = len(corrected)
        t = np.linspace(0, actual_duration, n_samples)

        effective_sample_rate = n_samples / actual_duration

        # Buffer timing diagnostics
        samples_per_buffer = n_samples / capture_count
        data_time_per_buffer = samples_per_buffer / self.nominal_sample_rate
        buffer_spacing = actual_duration / capture_count
        dead_time = buffer_spacing - data_time_per_buffer

        # Print summary
        print("\n" + "="*60)
        print("DC MEASUREMENT RESULTS")
        print("="*60)
        print(f"Mode: {self.input_mode}")
        print(f"Duration: {actual_duration:.3f}s")
        print(f"Samples: {n_samples}")
        print(f"Effective sample rate: {effective_sample_rate:.2f} Hz")
        print(f"Mean voltage: {np.mean(corrected):.6f} V")
        print(f"Std dev: {np.std(corrected):.6f} V")
        print("\nBuffer statistics:")
        print(f"  Buffers: {capture_count}")
        print(f"  Samples per buffer: {samples_per_buffer:.0f}")
        print(f"  Data time per buffer: {data_time_per_buffer:.3f}s")
        print(f"  Gap per buffer: {dead_time*1000:.1f} ms")
        print("="*60)

        # Plot voltage vs time
        plt.figure(figsize=(14,6))
        plt.plot(t, corrected, linewidth=0.8)
        plt.axhline(np.mean(corrected), color='r', linestyle='--',
                    label=f"Mean {np.mean(corrected):.6f} V")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("DC Voltage vs Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure and CSV
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = os.path.join(self.output_dir, f"dc_voltage_{ts}.png")
            plt.draw()  # Ensure figure is fully rendered
            plt.savefig(png_path, dpi=150)
            print(f"✓ Saved plot to {png_path}")

            csv_path = os.path.join(self.output_dir, f"dc_voltage_{ts}.csv")
            with open(csv_path, 'w') as f:
                f.write("# Red Pitaya DC Logger\n")
                f.write(f"# Mode: {self.input_mode}\n")
                f.write(f"# Gain: {self.input_gain_factor}\n")
                f.write(f"# Offset: {self.input_dc_offset}\n")
                f.write(f"# Duration: {actual_duration:.6f}\n")
                f.write(f"# Sample rate: {effective_sample_rate:.3f} Hz\n")
                f.write("Time(s),Voltage(V)\n")
                np.savetxt(f, np.column_stack((t, corrected)), delimiter=",", fmt="%.10f")
            print(f"✓ Saved data to {csv_path}")

        plt.show()


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
        'save_file': SAVE_DATA
    }

    rp.run(run_params)
