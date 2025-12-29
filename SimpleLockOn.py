"""
Red Pitaya Lock-In Amplifier with Auto-Calibration
Connect OUT1 to IN1 for testing.
Supports AUTO, LV, HV, or MANUAL gain modes.
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
REF_AMPLITUDE = 1  # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 30.0  # seconds

INPUT_MODE = 'Manual'
AUTOLAB_GAIN = 0.0
MANUAL_GAIN_FACTOR = 1.08 + AUTOLAB_GAIN  # Manual gain

FILTER_BANDWIDTH = 10  # Hz
AVERAGING_WINDOW = 1  # samples
DECIMATION = 8192

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
SAVE_TIMESTAMPS = True

AUTO_CALIBRATE = True
CALIBRATION_TIME = 2.0
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
        
        self.lockin_X = []
        self.lockin_Y = []
        self.capture_timestamps = []
        self.acquisition_start_time = None

        # Setup input gain
        self.input_gain_factor = manual_gain
        self.input_mode_setting = input_mode.upper()
        self.input_mode = "Unknown"

        if self.input_mode_setting == 'MANUAL':
            self.input_mode = f"MANUAL ({manual_gain}x gain)"
        elif self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_mode = "LV (±1V)"
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_mode = "HV (±20V, 20:1 divider)"
        elif self.input_mode_setting == 'AUTO':
            self.input_mode = "AUTO (will calibrate)"

        # Setup scope
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = DECIMATION
        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError('Invalid decimation')
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("CALIBRATING INPUT GAIN...")
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
            quadrature_factor=1.0
        )
        time.sleep(0.5)

        cal_X, cal_Y = [], []
        start_time = time.time()
        while (time.time() - start_time) < cal_time:
            self.scope.single()
            cal_X.append(np.array(self.scope._data_ch1_current))
            cal_Y.append(np.array(self.scope._data_ch2_current))

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

        print(f"Calibration done. Gain factor: {self.input_gain_factor:.4f}x, Mode: {self.input_mode}")
        return self.input_gain_factor

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_sig.output_direct = 'off'
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=params.get('filter_bandwidth', 10),
            gain=0.0,
            phase=params.get('phase', 0),
            acbandwidth=0,
            amplitude=params['ref_amp'],
            input='in1',
            output_direct=params['output_ref'],
            output_signal='quadrature',
            quadrature_factor=1.0
        )
        print(f"Lock-in setup: {self.ref_freq} Hz, {params['ref_amp']} V, Gain: {self.input_gain_factor:.4f}x")

    def capture_lockin(self):
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)
        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(time.time())
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
        loop_start = time.time()
        while (time.time() - loop_start) < params['timeout']:
            self.capture_lockin()

        acquisition_end_time = time.time()
        all_X = np.concatenate(self.lockin_X) * self.input_gain_factor
        all_Y = np.concatenate(self.lockin_Y) * self.input_gain_factor

        total_samples = len(all_X)
        actual_duration = acquisition_end_time - self.acquisition_start_time
        actual_sample_rate = total_samples / actual_duration

        # Generate timestamps
        sample_timestamps = np.zeros(total_samples)
        sample_idx = 0
        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.lockin_X[i])
            capture_duration = n_samples / actual_sample_rate
            sample_timestamps[sample_idx:sample_idx+n_samples] = capture_time + np.linspace(0, capture_duration, n_samples, endpoint=False)
            sample_idx += n_samples

        # Averaging
        avg_window = params.get('averaging_window', 1)
        if avg_window > 1:
            all_X = np.convolve(all_X, np.ones(avg_window)/avg_window, mode='valid')
            all_Y = np.convolve(all_Y, np.ones(avg_window)/avg_window, mode='valid')
            sample_timestamps = sample_timestamps[:len(all_X)]

        R = np.sqrt(all_X**2 + all_Y**2)
        Theta = np.arctan2(all_Y, all_X)
        t = sample_timestamps - self.acquisition_start_time

        # FFT for diagnostics
        iq = all_X + 1j*all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQfft = np.fft.fftshift(np.fft.fft(iq*win))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0/actual_sample_rate))
        psd_lock = (np.abs(IQfft)**2) / (actual_sample_rate * np.sum(win**2))
        idx = np.argmax(psd_lock)

        print(f"FFT peak: {freqs_lock[idx]:.2f} Hz (should be ~0 Hz)")

        # === PLOTS ===
        fig = plt.figure(figsize=(16, 10))

        # FFT
        ax1 = plt.subplot(3, 3, 1)
        ax1.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax1.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power')
        ax1.set_title('FFT Spectrum')
        ax1.legend()
        ax1.grid(True)

        # X vs Time
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t, all_X, 'b-', linewidth=0.5)
        ax2.axhline(np.mean(all_X), color='r', linestyle='--', label=f'Mean: {np.mean(all_X):.4f}V')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('X (V)')
        ax2.set_title('In-phase (X)')
        ax2.legend()
        ax2.grid(True)

        # Y vs Time
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(t, all_Y, 'r-', linewidth=0.5)
        ax3.axhline(np.mean(all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(all_Y):.4f}V')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Y (V)')
        ax3.set_title('Quadrature (Y)')
        ax3.legend()
        ax3.grid(True)

        # IQ Plot
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(all_X, all_Y, 'g.', markersize=1, alpha=0.5)
        ax4.plot(np.mean(all_X), np.mean(all_Y), 'r+', markersize=15, markeredgewidth=2)
        ax4.set_xlabel('X (V)')
        ax4.set_ylabel('Y (V)')
        ax4.set_title('IQ Plot')
        ax4.grid(True)
        ax4.axis('equal')

        # R vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, R, 'm-', linewidth=0.5)
        ax5.axhline(np.mean(R), color='b', linestyle='--', label=f'Mean: {np.mean(R):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('R (V)')
        ax5.set_title('Magnitude (R)')
        ax5.legend()
        ax5.grid(True)

        # Theta vs Time
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(t, Theta, 'c-', linewidth=0.5)
        ax6.axhline(np.mean(Theta), color='r', linestyle='--', label=f'Mean: {np.mean(Theta):.4f} rad')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Theta (rad)')
        ax6.set_title('Phase (Theta)')
        ax6.legend()
        ax6.grid(True)

        # Theta vs R
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(Theta, R, 'g-', markersize=1, alpha=0.5)
        ax7.plot(np.mean(Theta), np.mean(R), 'r+', markersize=15, markeredgewidth=2)
        ax7.set_xlabel('Theta')
        ax7.set_ylabel('R')
        ax7.set_title('IQ Plot')
        ax7.grid(True)
        ax7.axis('equal')

        plt.tight_layout()

        # Save or show
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"Saved plot: {img_path}")

            data = np.column_stack((t, R, Theta, all_X, all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
            np.savetxt(csv_path, data, delimiter=",", header="Time,R,Theta,X,Y", comments='', fmt='%.6f')
            print(f"Saved data: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitaya(output_dir=OUTPUT_DIRECTORY, input_mode=INPUT_MODE, manual_gain=MANUAL_GAIN_FACTOR)
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

    rp.run(run_params)
