"""
Red Pitaya Lock-In Amplifier with Auto-Calibration

Connect OUT1 to IN1 with a cable for testing.
Supports AUTO, LV, HV, or MANUAL gain modes.
"""

from datetime import datetime
import time
import math
import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from pyrpl import Pyrpl

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
REF_FREQUENCY = 500  # Hz
REF_AMPLITUDE = 1  # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 30.0  # seconds

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'Manual'

Auto_Lab_Gain = 0.0
MANUAL_GAIN_FACTOR = 1.08 + Auto_Lab_Gain  # Only used if INPUT_MODE = 'MANUAL' 27.80999388

FILTER_BANDWIDTH = 10  # Hz
AVERAGING_WINDOW = 1  # samples
DECIMATION = 8192

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
SAVE_TIMESTAMPS = True

AUTO_CALIBRATE = True  # Only used if INPUT_MODE = 'AUTO'
CALIBRATION_TIME = 2.0  # seconds
# ============================================================

START_TIME_FILE = "start_time.txt"

with open(START_TIME_FILE, "r") as f:
    START_TIME = datetime.fromisoformat(f.read().strip())

while datetime.now() < START_TIME:
    time.sleep(0.001)


class RedPitaya:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO', manual_gain=1.0):
        self.rp = Pyrpl(config='lockin_config', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope
        self.pid = self.rp_modules.pid0

        self.lockin_X = []
        self.lockin_Y = []
        self.capture_timestamps = []
        self.acquisition_start_time = None

        # Setup input gain
        self.input_gain_factor = manual_gain
        self.input_mode_setting = input_mode.upper()
        self.input_mode = "Unknown"

        if self.input_mode_setting == 'MANUAL':
            self.input_gain_factor = manual_gain
            self.input_mode = f"MANUAL ({manual_gain}x gain)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_mode = "LV (±1V)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_mode = "HV (±20V, 20:1 divider)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'AUTO':
            self.input_mode = "AUTO (will calibrate)"
            print("Input mode: AUTO - will auto-detect")

        # Setup scope
        self.scope.input1 = 'iq2'  # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        """Measure actual input gain to detect LV/HV mode"""
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT GAIN...")
        print("=" * 60)

        self.ref_sig.output_direct = 'off'
        self.lockin.setup(
            frequency=cal_freq,
            bandwidth=10,
            gain=0.0,
            phase=0,
            acbandwidth=0,
            amplitude=cal_amp,
            input='in1',
            output_direct='out1',
            output_signal='quadrature',
            quadrature_factor=1.0)

        print(f"Generating {cal_amp}V at {cal_freq} Hz, measuring for {cal_time}s...")
        time.sleep(0.5)

        cal_X = []
        cal_Y = []
        start_time = time.time()

        while (time.time() - start_time) < cal_time:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)
            ch2 = np.array(self.scope._data_ch2_current)
            cal_X.append(ch1)
            cal_Y.append(ch2)

        all_cal_X = np.concatenate(cal_X)
        all_cal_Y = np.concatenate(cal_Y)
        cal_R = np.sqrt(all_cal_X ** 2 + all_cal_Y ** 2)
        measured_amp = np.mean(cal_R)
        expected_amp = cal_amp / 2.0

        self.input_gain_factor = expected_amp / measured_amp

        if self.input_gain_factor < 1.05:
            self.input_mode = "LV (±1V)"
        elif self.input_gain_factor < 1.15:
            self.input_mode = "LV (±1V) with loading"
        elif self.input_gain_factor < 15:
            self.input_mode = f"Unknown ({self.input_gain_factor:.2f}x)"
        else:
            attenuation = 1.0 / self.input_gain_factor
            self.input_mode = f"HV (±20V, {attenuation:.1f}:1)"

        print(f"Output: {cal_amp:.3f}V, Expected: {expected_amp:.3f}V, Measured: {measured_amp:.3f}V")
        print(f"Gain factor: {self.input_gain_factor:.4f}x")
        print(f"Mode: {self.input_mode}")
        print("=" * 60 + "\n")

        return self.input_gain_factor

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        self.ref_sig.output_direct = 'off'

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
            quadrature_factor=1.0)

        print(f"Lock-in: {self.ref_freq} Hz, {ref_amp}V, BW: {filter_bw} Hz")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")

    def capture_lockin(self):
        """Capture scope data and store with timestamp"""
        capture_time = time.time()
        self.scope.single()

        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(capture_time)

        return ch1, ch2

    def run(self, params):
        if params.get('auto_calibrate', False):
            self.calibrate_input_gain(
                cal_freq=params['ref_freq'],
                cal_amp=params['ref_amp'],
                cal_time=params.get('calibration_time', 2.0)
            )

        self.setup_lockin(params)
        print("Settling...")
        time.sleep(0.5)

        self.acquisition_start_time = time.time()
        print(f"Started: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        loop_start = time.time()
        while (time.time() - loop_start) < params['timeout']:
            self.capture_lockin()

        acquisition_end_time = time.time()

        # Concatenate and apply gain correction
        all_X = np.concatenate(self.lockin_X) * self.input_gain_factor
        all_Y = np.concatenate(self.lockin_Y) * self.input_gain_factor

        # Calculate ACTUAL sampling rate
        total_samples = len(all_X)
        actual_duration = acquisition_end_time - self.acquisition_start_time
        actual_sample_rate = total_samples / actual_duration
        sample_rate_error = (actual_sample_rate - self.sample_rate) / self.sample_rate * 100

        print("\n" + "=" * 60)
        print("SAMPLING RATE ANALYSIS")
        print("=" * 60)
        print(f"Nominal sample rate: {self.sample_rate:.2f} Hz")
        print(f"Actual sample rate: {actual_sample_rate:.2f} Hz")
        print(f"Sample rate error: {sample_rate_error:.4f}%")
        print(f"Total samples: {total_samples}")
        print(f"Actual duration: {actual_duration:.3f}s")
        print("=" * 60)

        # Generate timestamps for each sample
        sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.lockin_X[i])
            capture_duration = n_samples / actual_sample_rate  # Use actual rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
            sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        # Apply averaging filter
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            all_X = np.convolve(all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            all_Y = np.convolve(all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            sample_timestamps = sample_timestamps[:len(all_X)]
            print(f"Applied {averaging_window}-sample moving average")

        R = np.sqrt(all_X ** 2 + all_Y ** 2)
        Theta = np.arctan2(all_Y, all_X)
        t = sample_timestamps - self.acquisition_start_time

        # Get raw signals for plotting
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw = np.array(self.scope._data_ch2_current) * self.input_gain_factor
        t_raw = np.arange(len(out1_raw)) / actual_sample_rate  # Use actual rate

        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        # FFT analysis using actual sample rate
        iq = all_X + 1j * all_Y
        iq -= np.mean(iq)  # remove DC offset

        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win

        # Use actual measured sample rate instead of timestamps
        IQfft = np.fft.fft(IQwin)
        freqs_lock = np.fft.fftfreq(n_pts, d=1 / actual_sample_rate)

        IQfft = np.fft.fftshift(IQfft)
        freqs_lock = np.fft.fftshift(freqs_lock)

        psd_lock = (np.abs(IQfft) ** 2) / (actual_sample_rate * np.sum(win ** 2))  # Use actual rate
        idx = np.argmax(psd_lock)

        # Print diagnostics
        print("\n" + "=" * 60)
        print("RESULTS (GAIN-CORRECTED)")
        print("=" * 60)
        print(f"Mode: {self.input_mode}")
        print(f"Gain: {self.input_gain_factor:.4f}x")
        print(f"FFT peak: {freqs_lock[idx]:.2f} Hz (should be ~0 Hz)")

        if abs(freqs_lock[idx]) < 5:
            print("✓ Lock-in is LOCKED")
        else:
            print("✗ WARNING: Not locked! Peak should be at 0 Hz")

        print(f"Samples: {len(all_X)}, Duration: {t[-1]:.3f}s")
        print(f"Mean R: {np.mean(R):.6f} ± {np.std(R):.6f} V")
        print(f"Mean X: {np.mean(all_X):.6f} ± {np.std(all_X):.6f} V")
        print(f"Mean Y: {np.mean(all_Y):.6f} ± {np.std(all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.6f} ± {np.std(Theta):.6f} rad")

        expected_R = params['ref_amp'] / 2
        if abs(np.mean(R) - expected_R) < 0.05:
            print(f"✓ R matches expected {expected_R:.3f}V")
        else:
            print(f"✗ R differs from expected {expected_R:.3f}V by {abs(np.mean(R) - expected_R):.3f}V")

        print("=" * 60)

        # Create plots
        fig = plt.figure(figsize=(16, 10))

        # Reference signal
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_plot = min(int(n_periods * actual_sample_rate / self.ref_freq), len(out1_raw))
        ax1.plot(t_raw[:n_plot] * 1000, out1_raw[:n_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference @ {self.ref_freq} Hz')
        ax1.grid(True)

        # Input signal
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw[:n_plot] * 1000, in1_raw[:n_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title(f'Input - {self.input_mode}')
        ax2.grid(True)

        # FFT
        ax3 = plt.subplot(3, 3, 3)
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.set_title('FFT Spectrum')
        ax3.legend()
        ax3.grid(True)

        # X vs Time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, all_X, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(all_X), color='r', linestyle='--', label=f'Mean: {np.mean(all_X):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-phase (X)')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(t[0], t[-1])
        margin_X = 5 * (np.max(all_X) - np.min(all_X))
        ax4.set_ylim(np.min(all_X) - margin_X, np.max(all_X) + margin_X)

        # Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(all_Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y)')
        ax5.legend()
        ax5.grid(True)
        ax5.set_xlim(t[0], t[-1])
        margin_Y = 5 * (np.max(all_Y) - np.min(all_Y))
        ax5.set_ylim(np.min(all_Y) - margin_Y, np.max(all_Y) + margin_Y)

        # IQ plot (with corrected gain)
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(all_X, all_Y, 'g-', markersize=1, alpha=0.5)
        ax6.plot(np.mean(all_X), np.mean(all_Y), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # R vs Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, R, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R), color='b', linestyle='--', label=f'Mean: {np.mean(R):.4f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R)')
        ax7.legend()
        ax7.grid(True)
        ax7.set_xlim(t[0], t[-1])
        margin_R = 5 * (np.max(R) - np.min(R))
        ax7.set_ylim(np.min(R) - margin_R, np.max(R) + margin_R)

        # Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', label=f'Mean: {np.mean(Theta):.4f} rad')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (rad)')
        ax8.set_title('Phase (Theta)')
        ax8.legend()
        ax8.grid(True)
        ax8.set_xlim(t[0], t[-1])
        margin_Theta = 5 * (np.max(Theta) - np.min(Theta))
        ax8.set_ylim(np.min(Theta) - margin_Theta, np.max(Theta) + margin_Theta)

        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(Theta, R, 'g-', markersize=1, alpha=0.5)
        ax9.plot(np.mean(Theta), np.mean(R), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax9.set_xlabel('Theta')
        ax9.set_ylabel('R')
        ax9.set_title('IQ Plot')
        ax9.legend()
        ax9.grid(True)
        ax9.axis('equal')

        plt.tight_layout()

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"✓ Saved plot: {img_path}")

            if params.get('save_timestamps', False):
                data = np.column_stack((sample_timestamps, t, R, Theta, all_X, all_Y))
                csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')

                with open(csv_path, 'w', newline='') as f:
                    f.write(f"# Mode: {self.input_mode}\n")
                    f.write(f"# Gain: {self.input_gain_factor:.6f}\n")
                    f.write(f"# Ref Freq: {self.ref_freq} Hz\n")
                    f.write(f"# Ref Amp: {params['ref_amp']} V\n")
                    f.write(f"# Nominal Sample Rate: {self.sample_rate:.2f} Hz\n")
                    f.write(f"# Actual Sample Rate: {actual_sample_rate:.2f} Hz\n")
                    f.write("AbsoluteTimestamp,RelativeTime,R,Theta,X,Y\n")
                    np.savetxt(f, data, delimiter=",", fmt='%.10f')

                print(f"✓ Saved data: {csv_path}")
            else:
                data = np.column_stack((t, R, Theta, all_X, all_Y))
                csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                           header="Time,R,Theta,X,Y", comments='', fmt='%.6f')
                print(f"✓ Saved data: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitaya(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR
    )

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
        'save_timestamps': SAVE_TIMESTAMPS,
        'auto_calibrate': AUTO_CALIBRATE,
        'calibration_time': CALIBRATION_TIME,
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN AMPLIFIER")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE}V")
    print(f"Filter BW: {FILTER_BANDWIDTH} Hz")
    print(f"Duration: {MEASUREMENT_TIME}s")
    print(f"Input Mode: {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"Gain: {MANUAL_GAIN_FACTOR}x")
    print("=" * 60)

    rp.run(run_params)
