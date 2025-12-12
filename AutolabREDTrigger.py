"""
Combined Red Pitaya Lock-In Amplifier and DC Voltage Monitor

SETUP:
- Red Pitaya 1 (rp-f073ce.local): Lock-in amplifier with OUT1 connected to IN1
- Red Pitaya 2 (rp-f0909c.local): DC voltage monitor on IN1

This version runs both instruments simultaneously and displays all plots in one window.
"""

# ============================================================
# LOCK-IN AMPLIFIER PARAMETERS
# ============================================================
LOCKIN_REF_FREQUENCY = 100
LOCKIN_REF_AMPLITUDE = 1
LOCKIN_OUTPUT_CHANNEL = 'out1'
LOCKIN_PHASE_OFFSET = 0
LOCKIN_MEASUREMENT_TIME = 30.0
LOCKIN_INPUT_MODE = 'AUTO'
LOCKIN_MANUAL_GAIN_FACTOR = 1.0
LOCKIN_FILTER_BANDWIDTH = 10
LOCKIN_AVERAGING_WINDOW = 1
LOCKIN_DECIMATION = 8192
LOCKIN_SHOW_FFT = True
LOCKIN_AUTO_CALIBRATE = True
LOCKIN_CALIBRATION_TIME = 2.0

# ============================================================
# DC MONITOR PARAMETERS
# ============================================================
DC_MEASUREMENT_TIME = 30.0
DC_AVERAGING_WINDOW = 1
DC_DECIMATION = 8192

# ============================================================
# COMMON PARAMETERS
# ============================================================
SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os
from datetime import datetime

N_FFT_SHOW = 10

import subprocess
import threading


def run_dc_in_process(result_queue, params):
    """Run DC monitor in separate process with its own event loop"""
    try:
        # Re-initialize RP2 in this process (objects can't be shared between processes)
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())

        rp_dc_proc = RedPitayaDCMonitor(output_dir=OUTPUT_DIRECTORY)
        rp_dc_proc.scope.trigger_source = 'ext_positive_edge'

        print("[RP2 Process] Waiting for hardware trigger on DIO0_P...")
        # Notify main process that RP2 is armed
        result_queue.put({'armed': True})

        # <-- USE params, not dc_params
        rp_dc_proc.run(params)

        print("[RP2 Process] ✓ DC monitor measurement complete")

        # Send results back through queue
        result_queue.put({
            'success': True,
            'sample_timestamps': rp_dc_proc.sample_timestamps.tolist(),
            'acquisition_start_time': rp_dc_proc.acquisition_start_time,
            't': rp_dc_proc.t.tolist(),
            'all_in1': rp_dc_proc.all_in1.tolist()
        })
    except Exception as e:
        import traceback
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        print(f"[RP2 Process] ERROR: {e}")


def send_trigger_pulse(hostname='rp-f073ce.local'):
    """Send a trigger pulse from RP1's DIO1_P pin using SSH"""
    try:
        # SSH commands to control DIO pins via Red Pitaya's monitor utility
        # Set DIO1_P (pin 1) HIGH
        subprocess.run(['ssh', f'root@{hostname}',
                        'monitor 0x40000010 0x2'],
                       capture_output=True, timeout=2, check=False)

        time.sleep(0.00001)  # 10 microsecond pulse

        # Set DIO1_P LOW
        subprocess.run(['ssh', f'root@{hostname}',
                        'monitor 0x40000010 0x0'],
                       capture_output=True, timeout=2, check=False)

        print("✓ Hardware trigger pulse sent via DIO1_P")
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not send trigger pulse: {e}")
        return False




class RedPitayaLockIn:
    electrode_map = {'A': (False, False), 'B': (True, False),
                     'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True),
                         '100uA': (True, False, True, True),
                         '1mA': (True, True, False, True),
                         '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True),
                    '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO', manual_gain=1.0):
        self.rp = Pyrpl(config='lockin_config', hostname='rp-f073ce.local')

        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []

        self.capture_timestamps = []
        self.acquisition_start_time = None

        self.input_gain_factor = manual_gain
        self.input_mode_setting = input_mode.upper()
        self.input_mode = "Unknown"

        if self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_mode = "LV (±1V) - Manual"
            print(f"⚙ Input mode set to: {self.input_mode}")
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_mode = "HV (±20V) - Manual (20:1 divider)"
            print(f"⚙ Input mode set to: {self.input_mode}")
        elif self.input_mode_setting == 'AUTO':
            self.input_mode = "AUTO - Will calibrate"
            print(f"⚙ Input mode set to: AUTO (will auto-detect)")
        else:
            print(f"⚠ Warning: Unknown input mode '{input_mode}', defaulting to AUTO")
            self.input_mode_setting = 'AUTO'
            self.input_mode = "AUTO - Will calibrate"

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival
        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = LOCKIN_DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        if not force and self.input_mode_setting != 'AUTO':
            print(f"\n⚙ Skipping calibration - using manual mode: {self.input_mode}")
            print(f"   Gain factor: {self.input_gain_factor:.4f}x")
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

        print(f"Generating {cal_amp}V sine at {cal_freq} Hz on OUT1")
        print(f"Measuring response on IN1 for {cal_time} seconds...")

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
            self.input_mode = "LV (±1V) - Direct"
            mode_detail = "Jumpers set to Low Voltage mode (no divider)"
        elif 1.05 <= self.input_gain_factor < 1.15:
            self.input_mode = "LV (±1V) - with loading"
            mode_detail = "Jumpers in LV mode with minor impedance loading (~8%)"
        elif 1.15 <= self.input_gain_factor < 15:
            attenuation_ratio = 1.0 / self.input_gain_factor
            self.input_mode = f"Unknown attenuation - {attenuation_ratio:.2f}:1"
            mode_detail = f"Unexpected attenuation detected (gain factor {self.input_gain_factor:.2f}x)"
        else:
            attenuation_ratio = 1.0 / self.input_gain_factor
            self.input_mode = f"HV (±20V) - {attenuation_ratio:.1f}:1 divider"
            mode_detail = "Jumpers set to High Voltage mode with 20:1 voltage divider"

        print("-" * 60)
        print("CALIBRATION RESULTS:")
        print(f"  Output amplitude: {cal_amp:.3f} V")
        print(f"  Expected measured: {expected_amp:.3f} V")
        print(f"  Actually measured: {measured_amp:.3f} V")
        print(f"  Input gain factor: {self.input_gain_factor:.4f}x")
        print(f"  Input mode: {self.input_mode}")
        print(f"  Details: {mode_detail}")
        print("-" * 60)
        print("✓ All measurements will be corrected using this gain factor")
        print("=" * 60 + "\n")

        return self.input_gain_factor

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        self.ref_sig.output_direct = 'off'
        print("ASG0 disabled - IQ module will generate reference")

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

        print(f"Lock-in setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"Filter BW: {filter_bw} Hz")
        print(f"IQ2 output_direct: {self.lockin.output_direct} (outputs {ref_amp}V sine)")
        print(f"IQ2 amplitude: {self.lockin.amplitude} V")
        print(f"IQ2 input: {self.lockin.input}")
        print(f"Scope reading: iq2 (X) and iq2_2 (Y)")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")

    def capture_lockin(self):
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
            cal_time = params.get('calibration_time', 2.0)
            self.calibrate_input_gain(
                cal_freq=params['ref_freq'],
                cal_amp=params['ref_amp'],
                cal_time=cal_time
            )

        timeout = params['timeout']
        self.setup_lockin(params)

        print("Waiting for lock-in to settle...")
        time.sleep(0.5)

        self.acquisition_start_time = time.time()
        print(
            f"\n✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        self.all_X *= self.input_gain_factor
        self.all_Y *= self.input_gain_factor

        samples_per_capture = len(self.lockin_X[0])
        total_samples = len(self.all_X)

        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.lockin_X[i])
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        averaging_window = params.get('averaging_window', 1)

        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_X)]
            print(f"Applied {averaging_window}-sample moving average filter")

        self.R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        self.Theta = np.arctan2(self.all_Y, self.all_X)

        self.t = self.sample_timestamps - self.acquisition_start_time

        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        self.out1_raw = np.array(self.scope._data_ch1_current)
        self.in1_raw = np.array(self.scope._data_ch2_current) * self.input_gain_factor
        self.t_raw = np.arange(len(self.out1_raw)) / self.sample_rate

        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        iq = self.all_X + 1j * self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        self.freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        self.psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))
        idx = np.argmax(self.psd_lock)

        print("=" * 60)
        print("LOCK-IN DIAGNOSTICS (GAIN-CORRECTED)")
        print("=" * 60)
        print(f"Input Mode: {self.input_mode}")
        print(f"Gain Correction Factor: {self.input_gain_factor:.4f}x")
        print("-" * 60)
        print(f"Reference Frequency Set: {self.ref_freq} Hz")
        print(f"FFT Peak Found at: {self.freqs_lock[idx]:.2f} Hz")
        print(f"Peak Offset from 0 Hz: {abs(self.freqs_lock[idx]):.2f} Hz")

        if abs(self.freqs_lock[idx]) < 5:
            print("✓ Lock-in is LOCKED (peak near 0 Hz)")
        else:
            print("✗ WARNING: Lock-in NOT locked! Peak should be at 0 Hz!")

        print(f"Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_X)}")
        print(f"Measurement Duration: {self.t[-1]:.3f} seconds")

        print("-" * 60)
        print(f"Mean R: {np.mean(self.R):.6f} V ± {np.std(self.R):.6f} V")
        print(f"SNR (R): {np.mean(self.R) / (np.std(self.R) + 1e-9):.2f} (mean/std)")
        print(f"R range: [{np.min(self.R):.6f}, {np.max(self.R):.6f}] V")

        expected_R = params['ref_amp'] / 2
        if abs(np.mean(self.R) - expected_R) < 0.05:
            print(f"✓ R matches expected {expected_R:.3f}V (corrected)")
        else:
            print(f"✗ R differs from expected {expected_R:.3f}V")
            print(f"  Difference: {abs(np.mean(self.R) - expected_R):.3f}V")

        print("-" * 60)
        print(f"Mean X: {np.mean(self.all_X):.6f} V ± {np.std(self.all_X):.6f} V")
        print(f"Mean Y: {np.mean(self.all_Y):.6f} V ± {np.std(self.all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(self.Theta):.6f} rad ± {np.std(self.Theta):.6f} rad")
        print(f"Theta range: [{np.min(self.Theta):.6f}, {np.max(self.Theta):.6f}] rad")
        print(f"Phase stability: {np.std(self.Theta):.3f} rad (lower is better)")

        X_ac = np.std(self.all_X)
        Y_ac = np.std(self.all_Y)
        X_dc = np.mean(np.abs(self.all_X))
        Y_dc = np.mean(np.abs(self.all_Y))

        print("-" * 60)
        print("Signal characteristics:")
        print(f"X: DC={X_dc:.6f}V, AC={X_ac:.6f}V, AC/DC={X_ac / max(X_dc, 0.001):.3f}")
        print(f"Y: DC={Y_dc:.6f}V, AC={Y_ac:.6f}V, AC/DC={Y_ac / max(Y_dc, 0.001):.3f}")

        SIGNAL_THRESHOLD = 0.02

        if X_dc > SIGNAL_THRESHOLD and X_ac / X_dc > 0.5:
            print("⚠ WARNING: X is oscillating! Should be flat for locked signal")

        if Y_dc > SIGNAL_THRESHOLD and Y_ac / Y_dc > 0.5:
            print("⚠ WARNING: Y is oscillating! Should be flat for locked signal")

        print("=" * 60)

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            data = np.column_stack((self.sample_timestamps, self.t, self.R, self.Theta, self.all_X, self.all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
            np.savetxt(csv_path, data, delimiter=",",
                       header="AbsoluteTime,RelativeTime,R,Theta,X,Y", comments='', fmt='%.6f')
            print(f"✓ Lock-in data saved with absolute and relative timestamps: {csv_path}")


class RedPitayaDCMonitor:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='dc_monitor_config4', hostname='rp-f0909c.local', gui=False)

        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.in1_data = []
        self.all_in1 = []

        self.capture_timestamps = []
        self.acquisition_start_time = None

        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        self.scope.input1 = 'in1'
        self.scope.decimation = DC_DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = False
        self.sample_rate = 125e6 / self.scope.decimation

    def capture_voltage(self):
        capture_time = time.time()

        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)

        self.in1_data.append(ch1)
        self.capture_timestamps.append(capture_time)

        return ch1

    def run(self, params):
        timeout = params['timeout']

        print("Waiting for scope to settle...")
        time.sleep(0.1)

        self.acquisition_start_time = time.time()
        print(
            f"\n✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_voltage()

        self.all_in1 = np.array(np.concatenate(self.in1_data))

        total_samples = len(self.all_in1)

        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.in1_data[i])
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)

            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        averaging_window = params.get('averaging_window', 1)

        if averaging_window > 1:
            self.all_in1 = np.convolve(self.all_in1, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_in1)]
            print(f"Applied {averaging_window}-sample moving average filter")

        self.t = self.sample_timestamps - self.acquisition_start_time

        print("=" * 60)
        print("DC VOLTAGE DIAGNOSTICS")
        print("=" * 60)
        print(f"Expected Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_in1)}")
        print(f"Measurement Duration: {self.t[-1]:.3f} seconds")

        actual_sample_rate = len(self.all_in1) / (self.t[-1] - self.t[0]) if self.t[-1] > self.t[0] else 0
        sample_rate_error = abs(actual_sample_rate - self.sample_rate) / self.sample_rate * 100

        print(f"Actual Sample Rate: {actual_sample_rate:.2f} Hz")
        print(f"Sample Rate Error: {sample_rate_error:.3f}%")

        if sample_rate_error > 1.0:
            print(f"⚠ WARNING: Sample rate error is {sample_rate_error:.3f}% (expected < 1%)")
        else:
            print(f"✓ Sample rate verified within {sample_rate_error:.3f}%")

        print("-" * 60)
        print(f"Mean Voltage: {np.mean(self.all_in1):.6f} V ± {np.std(self.all_in1):.6f} V")
        print("=" * 60)

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            data = np.column_stack((self.sample_timestamps, self.t, self.all_in1))
            csv_path = os.path.join(self.output_dir, f'dc_voltage_{timestamp_str}.csv')
            np.savetxt(csv_path, data, delimiter=",",
                       header="AbsoluteTime,RelativeTime,Voltage", comments='', fmt='%.6f')
            print(f"✓ DC voltage data saved with absolute and relative timestamps: {csv_path}")



if __name__ == '__main__':
    print("=" * 60)
    print("COMBINED RED PITAYA MONITOR")
    print("=" * 60)
    print("Initializing Lock-In Amplifier (rp-f073ce.local)...")
    print("=" * 60)

    rp_lockin = RedPitayaLockIn(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=LOCKIN_INPUT_MODE,
        manual_gain=LOCKIN_MANUAL_GAIN_FACTOR
    )

    lockin_params = {
        'ref_freq': LOCKIN_REF_FREQUENCY,
        'ref_amp': LOCKIN_REF_AMPLITUDE,
        'output_ref': LOCKIN_OUTPUT_CHANNEL,
        'phase': LOCKIN_PHASE_OFFSET,
        'timeout': LOCKIN_MEASUREMENT_TIME,
        'filter_bandwidth': LOCKIN_FILTER_BANDWIDTH,
        'averaging_window': LOCKIN_AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'fft': LOCKIN_SHOW_FFT,
        'auto_calibrate': LOCKIN_AUTO_CALIBRATE,
        'calibration_time': LOCKIN_CALIBRATION_TIME,
    }

    print("\n" + "=" * 60)
    print("Initializing DC Voltage Monitor (rp-f0909c.local)...")
    print("=" * 60)

    rp_dc = RedPitayaDCMonitor(output_dir=OUTPUT_DIRECTORY)

    dc_params = {
        'timeout': DC_MEASUREMENT_TIME,
        'averaging_window': DC_AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
    }

    print("\n" + "=" * 60)
    print("HARDWARE TRIGGER SYNCHRONIZATION")
    print("=" * 60)

    # Configure RP2 to wait for external trigger on DIO0_P (EXT input)
    print("Configuring RP2 to wait for trigger on EXT input (DIO0_P)...")
    rp_dc.scope.trigger_source = 'ext_positive_edge'
    print("✓ RP2 armed and waiting for trigger pulse")

    print("\n" + "=" * 60)
    print("STARTING SYNCHRONIZED MEASUREMENTS")
    print("=" * 60)

    # Use multiprocessing instead of threading to avoid asyncio event loop issues
    from multiprocessing import Process, Queue
    import queue as queue_module




    # Create queue for inter-process communication
    result_queue = Queue()

    # Start RP2 in a separate process
    print("\nStarting RP2 process (will wait for trigger)...")
    dc_process = Process(target=run_dc_in_process, args=(result_queue, dc_params))

    dc_process.start()

    # Wait for RP2 to report it is armed
    print("Waiting for RP2 to arm trigger...")
    while True:
        try:
            msg = result_queue.get(timeout=0.1)
            if 'armed' in msg:
                print("✓ RP2 armed and ready for trigger")
                break
        except queue_module.Empty:
            continue

    # Small delay to ensure RP2 is fully armed
    time.sleep(1.5)

    # Send hardware trigger pulse from RP1 DIO1_P to RP2 DIO0_P
    trigger_sent = send_trigger_pulse('rp-f073ce.local')


    if not trigger_sent:
        print("⚠ WARNING: Trigger may not have been sent properly")
        print("⚠ Continuing anyway - check SSH keys and DIO connections")
        print("⚠ TIP: Run 'ssh-keygen' and 'ssh-copy-id root@rp-f073ce.local' to fix SSH timeout")

    # Small delay to ensure trigger is processed
    time.sleep(0.1)

    # Now run RP1 measurement
    print("\nRunning Lock-In Amplifier on RP1...")
    rp_lockin.run(lockin_params)

    # Wait for RP2 process to complete
    print("\nWaiting for RP2 DC monitor to complete...")
    dc_process.join(timeout=DC_MEASUREMENT_TIME + 10)

    # Get results from queue
    dc_result = None
    try:
        dc_result = result_queue.get(timeout=1)
    except queue_module.Empty:
        print("✗ ERROR: No result from RP2 process")

    if dc_process.is_alive():
        print("✗ ERROR: RP2 process did not complete (still waiting for trigger?)")
        print("✗ Check hardware connections: RP1 DIO1_P -> RP2 DIO0_P")
        print("✗ Terminating RP2 process...")
        dc_process.terminate()
        dc_process.join()
        exit(1)
    elif dc_result and not dc_result['success']:
        print(f"✗ ERROR: RP2 encountered an error: {dc_result['error']}")
        if 'traceback' in dc_result:
            print("Traceback:")
            print(dc_result['traceback'])
        exit(1)
    elif dc_result and dc_result['success']:
        print("✓ Both measurements completed successfully")

        # Reconstruct RP2 data from process results
        rp_dc.sample_timestamps = np.array(dc_result['sample_timestamps'])
        rp_dc.acquisition_start_time = dc_result['acquisition_start_time']
        rp_dc.t = np.array(dc_result['t'])
        rp_dc.all_in1 = np.array(dc_result['all_in1'])

        # Calculate synchronization offset
        time_offset = rp_lockin.acquisition_start_time - rp_dc.acquisition_start_time
        print(f"\nSynchronization check:")
        print(f"  RP1 started: {datetime.fromtimestamp(rp_lockin.acquisition_start_time).strftime('%H:%M:%S.%f')}")
        print(f"  RP2 started: {datetime.fromtimestamp(rp_dc.acquisition_start_time).strftime('%H:%M:%S.%f')}")
        print(f"  Time offset: {abs(time_offset) * 1000:.3f} ms")

        if abs(time_offset) < 0.01:  # < 10ms
            print(f"  ✓✓✓ EXCELLENT! Synchronized within 10 ms")
        elif abs(time_offset) < 0.1:  # < 100ms
            print(f"  ✓ GOOD: Synchronized within 100 ms")
        else:
            print(f"  ⚠ WARNING: Devices started {abs(time_offset) * 1000:.1f} ms apart")
    else:
        print("✗ ERROR: Could not get results from RP2")
        print("✗ Skipping data combination and plotting")
        exit(1)

    print("\n" + "=" * 60)
    print("Combining CSV data...")
    print("=" * 60)

    # ----- Get all timestamps and measurements -----
    t_lock_abs = rp_lockin.sample_timestamps  # RP1 absolute timestamps
    t_dc_abs = rp_dc.sample_timestamps  # RP2 absolute timestamps

    t_lock_rel = rp_lockin.t  # RP1 relative time
    t_dc_rel = rp_dc.sample_timestamps - rp_dc.acquisition_start_time  # RP2 relative time

    R = rp_lockin.R
    Theta = rp_lockin.Theta
    X = rp_lockin.all_X
    Y = rp_lockin.all_Y
    Vdc = rp_dc.all_in1

    # ----- Build combined CSV -----
    # Note: if arrays have different lengths, pad the shorter with NaNs
    len_lock = len(t_lock_abs)
    len_dc = len(t_dc_abs)
    max_len = max(len_lock, len_dc)


    def pad(arr, target_len):
        if len(arr) < target_len:
            return np.concatenate([arr, np.full(target_len - len(arr), np.nan)])
        else:
            return arr


    combined = np.column_stack([
        pad(t_lock_abs, max_len),
        pad(t_dc_abs, max_len),
        pad(t_lock_rel, max_len),
        pad(t_dc_rel, max_len),
        pad(R, max_len),
        pad(Theta, max_len),
        pad(X, max_len),
        pad(Y, max_len),
        pad(Vdc, max_len)
    ])

    # ----- Save CSV -----
    if SAVE_DATA:
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(OUTPUT_DIRECTORY, f'combined_lockin_dc_{timestamp_str}.csv')
        header = "Time_RP1_Abs,Time_RP2_Abs,Time_RP1_Rel,Time_RP2_Rel,R,Theta,X,Y,DC_voltage"
        np.savetxt(csv_path, combined, delimiter=",", header=header, comments='', fmt="%.6f")
        print(f"✓ Combined CSV saved: {csv_path}")

    # Create combined plot
    print("\n" + "=" * 60)
    print("Creating combined plot...")
    print("=" * 60)

    fig = plt.figure(figsize=(20, 12))
    ZoomOut_Amount = 5

    # Lock-in plots (3x3 grid on left/center)
    # 1. OUT1 (Reference Signal)
    ax1 = plt.subplot(3, 4, 1)
    n_periods = 5
    n_samples_plot = int(n_periods * rp_lockin.sample_rate / rp_lockin.ref_freq)
    n_samples_plot = min(n_samples_plot, len(rp_lockin.out1_raw))
    ax1.plot(rp_lockin.t_raw[:n_samples_plot] * 1000, rp_lockin.out1_raw[:n_samples_plot], 'b-', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('OUT1 (V)')
    ax1.set_title(f'Reference Signal (OUT1) @ {rp_lockin.ref_freq} Hz')
    ax1.grid(True)

    # 2. IN1 (Input Signal - CORRECTED)
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(rp_lockin.t_raw[:n_samples_plot] * 1000, rp_lockin.in1_raw[:n_samples_plot], 'r-', linewidth=1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('IN1 (V, corrected)')
    ax2.set_title(f'Input Signal (IN1) - {rp_lockin.input_mode}')
    ax2.grid(True)

    # 3. FFT Spectrum
    ax3 = plt.subplot(3, 4, 3)
    ax3.semilogy(rp_lockin.freqs_lock, rp_lockin.psd_lock, label='Lock-in PSD')
    ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz (target)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power (a.u.)')
    ax3.set_title('FFT Spectrum (baseband)')
    ax3.legend()
    ax3.grid(True)

    # 4. X vs Time
    ax4 = plt.subplot(3, 4, 5)
    ax4.plot(rp_lockin.t, rp_lockin.all_X, 'b-', linewidth=0.5)
    ax4.axhline(np.mean(rp_lockin.all_X), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp_lockin.all_X):.4f}V')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X (V, corrected)')
    ax4.set_title('In-phase (X) vs Time [iq2] - Corrected')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(rp_lockin.t[0], rp_lockin.t[-1])
    margin_X = ZoomOut_Amount * (np.max(rp_lockin.all_X) - np.min(rp_lockin.all_X))
    ax4.set_ylim(np.min(rp_lockin.all_X) - margin_X, np.max(rp_lockin.all_X) + margin_X)

    # 5. Y vs Time
    ax5 = plt.subplot(3, 4, 6)
    ax5.plot(rp_lockin.t, rp_lockin.all_Y, 'r-', linewidth=0.5)
    ax5.axhline(np.mean(rp_lockin.all_Y), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp_lockin.all_Y):.4f}V')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Y (V, corrected)')
    ax5.set_title('Quadrature (Y) vs Time [iq2_2] - Corrected')
    ax5.legend()
    ax5.grid(True)
    ax5.set_xlim(rp_lockin.t[0], rp_lockin.t[-1])
    margin_Y = ZoomOut_Amount * (np.max(rp_lockin.all_Y) - np.min(rp_lockin.all_Y))
    ax5.set_ylim(np.min(rp_lockin.all_Y) - margin_Y, np.max(rp_lockin.all_Y) + margin_Y)

    # 6. X vs Y (IQ plot)
    ax6 = plt.subplot(3, 4, 7)
    ax6.plot(rp_lockin.all_X, rp_lockin.all_Y, 'g.', markersize=1, alpha=0.5)
    ax6.plot(np.mean(rp_lockin.all_X), np.mean(rp_lockin.all_Y), 'r+', markersize=15,
             markeredgewidth=2, label='Mean')
    ax6.set_xlabel('X (V, corrected)')
    ax6.set_ylabel('Y (V, corrected)')
    ax6.set_title('IQ Plot (X vs Y) - Corrected')
    ax6.legend()
    ax6.grid(True)
    ax6.axis('equal')

    # 7. R vs Time
    ax7 = plt.subplot(3, 4, 9)
    ax7.plot(rp_lockin.t, rp_lockin.R, 'm-', linewidth=0.5)
    ax7.axhline(np.mean(rp_lockin.R), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp_lockin.R):.4f}V')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('R (V, corrected)')
    ax7.set_title('Magnitude (R) vs Time - Corrected')
    ax7.legend()
    ax7.grid(True)
    ax7.set_xlim(rp_lockin.t[0], rp_lockin.t[-1])
    margin_R = ZoomOut_Amount * (np.max(rp_lockin.R) - np.min(rp_lockin.R))
    ax7.set_ylim(np.min(rp_lockin.R) - margin_R, np.max(rp_lockin.R) + margin_R)

    # 8. Theta vs Time
    ax8 = plt.subplot(3, 4, 10)
    ax8.plot(rp_lockin.t, rp_lockin.Theta, 'c-', linewidth=0.5)
    ax8.axhline(np.mean(rp_lockin.Theta), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(rp_lockin.Theta):.4f} rad')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Theta (rad)')
    ax8.set_title('Phase (Theta) vs Time')
    ax8.legend()
    ax8.grid(True)
    ax8.set_xlim(rp_lockin.t[0], rp_lockin.t[-1])
    margin_Theta = ZoomOut_Amount * (np.max(rp_lockin.Theta) - np.min(rp_lockin.Theta))
    ax8.set_ylim(np.min(rp_lockin.Theta) - margin_Theta, np.max(rp_lockin.Theta) + margin_Theta)

    # 9. R vs Theta
    ax9 = plt.subplot(3, 4, 11)
    ax9.plot(rp_lockin.Theta, rp_lockin.R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
    ax9.axhline(np.mean(rp_lockin.R), color='b', linestyle='--', alpha=0.5)
    ax9.axvline(np.mean(rp_lockin.Theta), color='r', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Theta (rad)')
    ax9.set_ylabel('R (V, corrected)')
    ax9.set_title('R vs Theta - Corrected')
    ax9.grid(True)

    # DC Monitor plot (right column, spanning multiple rows)
    ax_dc = plt.subplot(3, 4, (4, 12))
    ax_dc.plot(rp_dc.t, rp_dc.all_in1, 'b-', linewidth=0.8, label='DC Voltage')
    ax_dc.axhline(np.mean(rp_dc.all_in1), color='r', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Mean: {np.mean(rp_dc.all_in1):.6f} V')
    ax_dc.set_xlabel('Time (s)', fontsize=12)
    ax_dc.set_ylabel('Voltage (V)', fontsize=12)
    ax_dc.set_title('DC Voltage Monitor vs Time', fontsize=14, fontweight='bold')
    ax_dc.legend(fontsize=10)
    ax_dc.grid(True, alpha=0.3)
    ax_dc.set_xlim(rp_dc.t[0], rp_dc.t[-1])

    stats_text = f'Samples: {len(rp_dc.all_in1):,}\nDuration: {rp_dc.t[-1]:.2f} s\nStd Dev: {np.std(rp_dc.all_in1):.6f} V'
    ax_dc.text(0.02, 0.98, stats_text, transform=ax_dc.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if SAVE_DATA:
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.png')
        plt.savefig(img_path, dpi=150)
        print(f"\n✓ Combined plot saved: {img_path}")

    plt.show()

    print("\n" + "=" * 60)
    print("MEASUREMENT COMPLETE")
    print("=" * 60)
