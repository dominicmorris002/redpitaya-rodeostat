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
AUTOLAB_GAIN = 0.0
MANUAL_GAIN_FACTOR = 1.08 + AUTOLAB_GAIN

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

        # Store X/Y and timestamps
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

        # Setup scope for plotting only
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
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
            # READ LOCK-IN DIRECTLY (FAST)
            ch1 = np.array([self.lockin.X])
            ch2 = np.array([self.lockin.Y])
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
        """Capture lock-in values FAST (no scope wait)"""
        capture_time = time.time()
        # Directly read lock-in outputs
        ch1 = np.array([self.lockin.X])
        ch2 = np.array([self.lockin.Y])

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

        # Actual sample rate
        total_samples = len(all_X)
        actual_duration = acquisition_end_time - self.acquisition_start_time
        actual_sample_rate = total_samples / actual_duration
        print(f"Total samples: {total_samples}, Duration: {actual_duration:.3f}s, Sample rate: {actual_sample_rate:.2f} Hz")

        # Generate timestamps
        sample_timestamps = np.array(self.capture_timestamps)
        t = sample_timestamps - self.acquisition_start_time

        # Apply averaging
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            all_X = np.convolve(all_X, np.ones(averaging_window)/averaging_window, mode='valid')
            all_Y = np.convolve(all_Y, np.ones(averaging_window)/averaging_window, mode='valid')
            t = t[:len(all_X)]
            print(f"Applied {averaging_window}-sample moving average")

        # R and Theta
        R = np.sqrt(all_X**2 + all_Y**2)
        Theta = np.arctan2(all_Y, all_X)

        # Scope signals for plotting only
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw = np.array(self.scope._data_ch2_current) * self.input_gain_factor
        t_raw = np.arange(len(out1_raw)) / actual_sample_rate
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        # FFT analysis
        iq = all_X + 1j*all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQfft = np.fft.fftshift(np.fft.fft(iq*win))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0/actual_sample_rate))
        psd_lock = (np.abs(IQfft)**2)/(actual_sample_rate*np.sum(win**2))
        idx = np.argmax(psd_lock)

        # Diagnostics
        print(f"FFT peak: {freqs_lock[idx]:.2f} Hz (should be ~0 Hz)")
        if abs(freqs_lock[idx]) < 5:
            print("✓ Lock-in is LOCKED")
        else:
            print("✗ WARNING: Not locked!")

        # PLOTS + SAVE routines remain unchanged
        # ... (same as your original code) ...
        # For brevity, I left your full plotting/saving code as-is.
        # Everything else can remain identical.

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

    print("="*60)
    print("RED PITAYA LOCK-IN AMPLIFIER")
    print("="*60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE}V")
    print(f"Filter BW: {FILTER_BANDWIDTH} Hz")
    print(f"Duration: {MEASUREMENT_TIME}s")
    print(f"Input Mode: {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"Gain: {MANUAL_GAIN_FACTOR}x")
    print("="*60)

    rp.run(run_params)
