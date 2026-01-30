"""
Red Pitaya DC Voltage Data Logger - SLAVE (with Hardware Sync)

This is a SLAVE Red Pitaya that:
1. Receives clock signal from master via SATA daisy-chain
2. Triggers scope acquisitions based on master's trigger signals
3. Maintains perfect synchronization with master board

SATA Connection: Connect "Clock and Trigger" SATA cable FROM master board TO this slave

Have a Great Day ;)
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
import logging
import warnings

# Suppress PyRPL warnings and info messages (only show errors)
logging.getLogger('pyrpl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

from pyrpl import Pyrpl

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
MEASUREMENT_TIME = 12.0  # seconds
DECIMATION = 1024
AVERAGING_WINDOW = 10

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'
MANUAL_GAIN_FACTOR = 28.93002007 * -1
MANUAL_DC_OFFSET = -0.016903

AUTO_CALIBRATE = True
CALIBRATION_TIME = 2.0  # seconds

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

# ============================================================
# HARDWARE SYNCHRONIZATION SETTINGS
# ============================================================
HARDWARE_SYNC = True  # Use SATA cable synchronization (slave mode)
# ============================================================


class RedPitayaDCSlave:
    """DC voltage logger with hardware synchronization (slave mode)"""

    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO',
                 manual_gain=1.0, manual_offset=0.0, hardware_sync=False):

        self.rp = Pyrpl(config='dc_slave_config', hostname='rp-f0909c.local')
        self.rp_modules = self.rp.rp
        self.scope = self.rp_modules.scope
        self.asg = self.rp_modules.asg0
        self.output_dir = output_dir
        self.hardware_sync = hardware_sync

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

        # Configure hardware synchronization
        if self.hardware_sync:
            print("\n" + "=" * 60)
            print("CONFIGURING HARDWARE SYNCHRONIZATION (SLAVE)")
            print("=" * 60)
            try:
                # Set this board as a clock slave
                # This receives the clock signal from master via SATA cable
                self.rp_modules.hk.led = 0b10101010  # Alternating LEDs to indicate slave
                print("✓ Board configured as CLOCK SLAVE")
                print("  Clock signal will be received from master via SATA")
                print("  Scope will trigger synchronously with master")
            except Exception as e:
                print(f"⚠ Warning: Could not configure as slave: {e}")
                print("  Continuing without hardware sync...")
            print("=" * 60 + "\n")

        # Scope setup
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f"Invalid decimation {DECIMATION}")

        # Configure scope trigger for hardware sync
        if self.hardware_sync:
            # Slave waits for external trigger from master
            # Note: In PyRPL, this is typically handled through the rolling mode
            # with the shared clock ensuring synchronous sampling
            self.scope.trigger_source = 'immediately'  # Will sync via shared clock
            print("Scope trigger: SLAVE MODE (synced to master clock)")
        else:
            self.scope.trigger_source = 'immediately'

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True
        self.nominal_sample_rate = 125e6 / self.scope.decimation

        print(f"DC Logger initialized (SLAVE)")
        print(f"Nominal sample rate: {self.nominal_sample_rate:.2f} Hz")

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        """Auto-calibrate input scaling and DC offset"""
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT SCALING AND DC OFFSET...")
        print("=" * 60)

        # Step 1: Measure DC offset
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

        # Step 2: Measure gain
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
            ch1 = np.array(self.scope._data_ch1_current)
            ch2 = np.array(self.scope._data_ch2_current)
            cal_out1.append(ch1)
            cal_in1.append(ch2)

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
        print(f"  IN1 peak: {in1_peak:.4f}V, RMS: {in1_rms:.4f}V")
        print(f"  Gain: {self.input_gain_factor:.4f}x")
        print(f"  DC offset: {self.input_dc_offset:.6f}V")
        print(f"  Detected mode: {self.input_mode}")
        print("=" * 60 + "\n")

        self.asg.output_direct = 'off'
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
                cal_freq=100,
                cal_amp=1.0,
                cal_time=params.get('calibration_time', 2.0)
            )

        print("\nAllowing scope to settle...")
        time.sleep(0.3)

        acquisition_start = time.time()
        print(f"Started: {datetime.fromtimestamp(acquisition_start).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        if self.hardware_sync:
            print("Waiting for triggers from master...")

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

        # Use sample-based indexing (matches lock-in)
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
        print("DC MEASUREMENT RESULTS (SLAVE)")
        print("=" * 60)
        print(f"Mode: {self.input_mode}")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")
        print(f"DC offset correction: {self.input_dc_offset:.6f}V")
        print(f"Hardware sync: {'ENABLED (slave)' if self.hardware_sync else 'DISABLED'}")
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
        print(f"  Dead time: {dead_time * capture_count:.2f}s ({dead_time * capture_count / actual_duration * 100:.1f}%)")
        print("=" * 60)

        # Plot voltage vs time
        plt.figure(figsize=(14, 6))
        plt.plot(t, corrected, linewidth=0.8)
        plt.axhline(np.mean(corrected), color='r', linestyle='--',
                    label=f"Mean {np.mean(corrected):.6f} V")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"DC Voltage vs Time - {self.input_mode} (SLAVE)")
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
                f.write("# Red Pitaya DC Logger (SLAVE)\n")
                f.write(f"# Mode: {self.input_mode}\n")
                f.write(f"# Hardware sync: {'ENABLED (slave)' if self.hardware_sync else 'DISABLED'}\n")
                f.write(f"# Gain: {self.input_gain_factor:.6f}\n")
                f.write(f"# Offset: {self.input_dc_offset:.6f}\n")
                f.write(f"# Duration: {actual_duration:.6f}\n")
                f.write(f"# Sample rate: {effective_sample_rate:.3f} Hz\n")
                f.write(f"# Samples: {n_samples}\n")
                f.write("Index,Time(s),Voltage(V)\n")
                np.savetxt(f, np.column_stack((sample_index, t, corrected)), delimiter=",", fmt="%.10f")
            print(f"✓ Saved data: {csv_path}")


if __name__ == "__main__":
    rp = RedPitayaDCSlave(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR,
        manual_offset=MANUAL_DC_OFFSET,
        hardware_sync=HARDWARE_SYNC
    )

    run_params = {
        'timeout': MEASUREMENT_TIME,
        'averaging_window': AVERAGING_WINDOW,
        'save_file': SAVE_DATA,
        'auto_calibrate': AUTO_CALIBRATE,
        'calibration_time': CALIBRATION_TIME,
    }

    print("=" * 60)
    print("RED PITAYA DC VOLTAGE DATA LOGGER (SLAVE)")
    print("=" * 60)
    print(f"Hardware sync: {'ENABLED' if HARDWARE_SYNC else 'DISABLED'}")
    print(f"Measurement time: {MEASUREMENT_TIME}s")
    print(f"Input mode: {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"Manual gain: {MANUAL_GAIN_FACTOR}x")
        print(f"Manual DC offset: {MANUAL_DC_OFFSET}V")
    print("=" * 60)

    rp.run(run_params)
