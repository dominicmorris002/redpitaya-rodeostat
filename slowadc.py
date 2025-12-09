"""
Red Pitaya Lock-In Amplifier - WITH SLOW ADC FOR SIMULTANEOUS DC MEASUREMENT

SETUP:
- Connect OUT1 directly to IN1 (for lock-in amplifier)
- Connect DC voltage source to XADC pins (0-3.3V range)

XADC PHYSICAL CONNECTIONS (on extension connector):
- AIN0: Pin AI0 (DIO0_P)
- AIN1: Pin AI1 (DIO1_P)
- AIN2: Pin AI2 (DIO2_P)
- AIN3: Pin AI3 (DIO3_P)
- Ground: Any GND pin

VOLTAGE RANGE: 0-3.3V

IQ MODULE OUTPUTS:
- For iq2 module: iq2 = X (in-phase), iq2_2 = Y (quadrature)

EXPECTED RESULTS (OUT1 → IN1, 0.5V sine @ 100Hz):
- X ≈ 0.25V (flat line) - half of amplitude
- Y ≈ 0V (flat line)
- R ≈ 0.25V (flat line) - half of amplitude
- Theta ≈ 0 rad (flat line)
- FFT peak at 0 Hz (locked!)

SYNCHRONIZATION FIX:
- Lock-in sample rate: 125 MHz / DECIMATION = 15,258 Hz
- Slow ADC now MATCHES lock-in at 15,258 Hz (was 100 Hz)
"""

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
# Red Pitaya hostname
HOSTNAME = 'rp-f073ce.local'

# Lock-in settings
REF_FREQUENCY = 100  # Hz - AC excitation frequency
REF_AMPLITUDE = 0.5  # V - AC signal amplitude
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0
MEASUREMENT_TIME = 30.0  # seconds
FILTER_BANDWIDTH = 10  # Hz
AVERAGING_WINDOW = 1

# Advanced settings
DECIMATION = 8192  # Lock-in sample rate = 125 MHz / 8192 = 15,258 Hz
SHOW_FFT = True

# Slow ADC settings - NOW SYNCHRONIZED
SLOW_ADC_CHANNELS = [0]  # Which XADC channels to read (0, 1, 2, 3 for AIN0-AIN3)
# AUTO-CALCULATE to match lock-in sample rate
SLOW_ADC_SAMPLE_RATE = int(125e6 / DECIMATION)  # 15,258 Hz (was 100 Hz)

# Data saving
SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
SAVE_TIMESTAMPS = True
# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os
from datetime import datetime
import threading
import socket
import re

N_FFT_SHOW = 10


class SCPIConnection:
    """Simple SCPI connection for Red Pitaya - no external dependencies"""

    def __init__(self, hostname, port=5000):
        self.hostname = hostname
        self.port = port
        self.sock = None
        self._connect()

    def _connect(self):
        """Connect to Red Pitaya"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(1.0)  # Add timeout
        self.sock.connect((self.hostname, self.port))
        print(f"✓ SCPI connected to {self.hostname}:{self.port}")

    def tx_txt(self, msg):
        """Send SCPI command"""
        if not msg.endswith('\r\n'):
            msg += '\r\n'
        self.sock.sendall(msg.encode('utf-8'))

    def rx_txt(self):
        """Receive SCPI response - read until newline"""
        response = b''
        while True:
            try:
                chunk = self.sock.recv(1)
                if not chunk:
                    break
                response += chunk
                if chunk == b'\n':
                    break
            except socket.timeout:
                break
        return response.decode('utf-8').strip()

    def query(self, msg):
        """Send command and receive response"""
        self.tx_txt(msg)
        time.sleep(0.001)  # Small delay for Red Pitaya to process
        return self.rx_txt()

    def close(self):
        """Close connection"""
        if self.sock:
            self.sock.close()


class RedPitaya:
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

    def __init__(self, output_dir='test_data', hostname='rp-f073ce.local'):
        self.hostname = hostname
        self.output_dir = output_dir

        # Initialize Pyrpl for lock-in
        print("Initializing Pyrpl...")
        self.rp = Pyrpl(config='lockin_config', hostname=hostname)
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0

        # Initialize SCPI connection for XADC
        print(f"Connecting to SCPI for XADC access...")
        self.scpi_xadc = SCPIConnection(hostname)

        # Lock-in data storage
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []
        self.capture_timestamps = []
        self.acquisition_start_time = None

        # Slow ADC data storage
        self.slow_adc_data = {}
        self.slow_adc_thread = None
        self.slow_adc_running = False

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival
        self.scope = self.rp_modules.scope

        # Setup scope for lock-in
        print("Available scope inputs:", self.scope.inputs)
        self.scope.input1 = 'iq2'  # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

        print(f"\n{'=' * 60}")
        print("SYNCHRONIZED SAMPLING MODE")
        print(f"{'=' * 60}")
        print(f"Lock-in sample rate: {self.sample_rate:.2f} Hz")
        print(f"Slow ADC will match: {SLOW_ADC_SAMPLE_RATE} Hz")
        print(f"{'=' * 60}\n")

        # Test XADC channels
        print("Testing Slow ADC (XADC) channels:")
        for i in range(4):
            try:
                value = self.read_xadc_channel(i)
                print(f"  AIN{i} (Pin AI{i}): {value:.6f} V")
            except Exception as e:
                print(f"  AIN{i}: Error - {e}")

    def read_xadc_channel(self, channel):
        """Read XADC channel using SCPI"""
        try:
            response = self.scpi_xadc.query(f'ANALOG:PIN? AIN{channel}')
            return float(response)
        except ValueError as e:
            # If conversion fails, try to extract first number
            match = re.search(r'[-+]?\d*\.?\d+', response)
            if match:
                return float(match.group())
            raise ValueError(f"Could not parse XADC response: '{response}'")

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        # Turn OFF ASG0
        self.ref_sig.output_direct = 'off'
        print("ASG0 disabled - IQ module will generate reference")

        # IQ module setup
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
            quadrature_factor=1)

        print(f"Lock-in setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"Filter BW: {filter_bw} Hz")
        print(f"IQ2 output_direct: {self.lockin.output_direct} (outputs {ref_amp}V sine)")
        print(f"IQ2 amplitude: {self.lockin.amplitude} V")
        print(f"IQ2 input: {self.lockin.input}")
        print(f"Scope reading: iq2 (X) and iq2_2 (Y)")

    def start_slow_adc_sampling(self, channels, sample_rate):
        """Start background thread to continuously sample slow ADC channels at HIGH SPEED"""
        self.slow_adc_running = True
        self.slow_adc_channels = channels
        self.slow_adc_sample_rate = sample_rate

        # Initialize data storage
        for channel in channels:
            self.slow_adc_data[channel] = []

        def sample_loop():
            sample_interval = 1.0 / sample_rate
            next_sample_time = time.time()
            sample_count = 0
            missed_samples = 0

            print(f"⚡ Starting HIGH-SPEED ADC @ {sample_rate} Hz (interval: {sample_interval * 1e6:.1f} µs)")

            while self.slow_adc_running:
                timestamp = time.time()

                # Check if falling behind
                if timestamp > next_sample_time + sample_interval:
                    missed_samples += 1

                for channel in self.slow_adc_channels:
                    try:
                        value = self.read_xadc_channel(channel)
                        self.slow_adc_data[channel].append((timestamp, value))
                    except Exception as e:
                        print(f"Error reading AIN{channel}: {e}")

                sample_count += 1

                # Progress update every 1000 samples
                if sample_count % 1000 == 0:
                    print(f"  Samples: {sample_count}, Missed: {missed_samples}")

                # Sleep until next sample time
                next_sample_time += sample_interval
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if missed_samples > 0:
                print(
                    f"⚠ ADC missed {missed_samples}/{sample_count} samples ({100 * missed_samples / sample_count:.1f}%)")

        self.slow_adc_thread = threading.Thread(target=sample_loop, daemon=True)
        self.slow_adc_thread.start()
        channel_names = ', '.join([f'AIN{ch}' for ch in channels])
        print(f"\n✓ Slow ADC sampling started: {channel_names} @ {sample_rate} Hz")

    def stop_slow_adc_sampling(self):
        """Stop slow ADC sampling"""
        self.slow_adc_running = False
        if self.slow_adc_thread:
            self.slow_adc_thread.join(timeout=2.0)
        print("✓ Slow ADC sampling stopped")

    def capture_lockin(self):
        """Captures scope data and appends to X and Y arrays with timestamps"""
        capture_time = time.time()

        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)  # iq2 = X
        ch2 = np.array(self.scope._data_ch2_current)  # iq2_2 = Y

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(capture_time)

        return ch1, ch2

    def see_fft(self):
        iq = self.all_X + 1j * self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))
        idx = np.argmax(psd_lock)
        print("Peak at", freqs_lock[idx], "Hz")

        plt.figure(1, figsize=(12, 4))
        plt.semilogy(freqs_lock, psd_lock, label='Lock-in R')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('Lock-in Output Spectrum (baseband)')
        plt.legend()
        plt.grid(True)

    def run(self, params):
        timeout = params['timeout']
        slow_adc_channels = params.get('slow_adc_channels', [])
        slow_adc_rate = params.get('slow_adc_rate', 100)

        self.setup_lockin(params)

        # Start slow ADC sampling if requested
        if slow_adc_channels:
            # Verify rates match
            if abs(slow_adc_rate - self.sample_rate) > 1:
                print(f"⚠ WARNING: Rate mismatch detected!")
                print(f"  Lock-in: {self.sample_rate:.2f} Hz")
                print(f"  Slow ADC: {slow_adc_rate} Hz")
                print(f"  Using lock-in rate for ADC...")
                slow_adc_rate = int(self.sample_rate)

            self.start_slow_adc_sampling(slow_adc_channels, slow_adc_rate)

        # Let the lock-in settle
        print("Waiting for lock-in to settle...")
        time.sleep(0.5)

        # Record absolute start time
        self.acquisition_start_time = time.time()
        print(
            f"\n✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Main acquisition loop
        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        # Stop slow ADC sampling
        if slow_adc_channels:
            self.stop_slow_adc_sampling()

        # Process lock-in data
        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        # Generate per-sample timestamps for lock-in data
        total_samples = len(self.all_X)
        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.lockin_X[i])
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_X)]
            print(f"Applied {averaging_window}-sample moving average filter")

        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)
        t = self.sample_timestamps - self.acquisition_start_time

        # Capture raw signals for plotting
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw = np.array(self.scope._data_ch2_current)
        t_raw = np.arange(len(out1_raw)) / self.sample_rate

        # Switch back to lock-in outputs
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        # FFT calculations
        iq = self.all_X + 1j * self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))
        idx = np.argmax(psd_lock)

        print("=" * 60)
        print("LOCK-IN DIAGNOSTICS")
        print("=" * 60)
        print(f"Reference Frequency Set: {self.ref_freq} Hz")
        print(f"FFT Peak Found at: {freqs_lock[idx]:.2f} Hz")
        print(f"Peak Offset from 0 Hz: {abs(freqs_lock[idx]):.2f} Hz")

        if abs(freqs_lock[idx]) < 5:
            print("✓ Lock-in is LOCKED (peak near 0 Hz)")
        else:
            print("✗ WARNING: Lock-in NOT locked! Peak should be at 0 Hz!")

        print(f"Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_X)}")
        print(f"Measurement Duration: {t[-1]:.3f} seconds")

        # Timestamp info
        print("-" * 60)
        print("TIMESTAMP INFORMATION:")
        print(f"Start time: {datetime.fromtimestamp(self.sample_timestamps[0]).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"End time:   {datetime.fromtimestamp(self.sample_timestamps[-1]).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"Duration:   {self.sample_timestamps[-1] - self.sample_timestamps[0]:.3f} seconds")

        print("-" * 60)
        print(f"Mean R: {np.mean(R):.6f} V ± {np.std(R):.6f} V")
        print(f"SNR (R): {np.mean(R) / (np.std(R) + 1e-9):.2f} (mean/std)")
        print(f"R range: [{np.min(R):.6f}, {np.max(R):.6f}] V")

        expected_R = params['ref_amp'] / 2
        if abs(np.mean(R) - expected_R) < 0.05:
            print(f"✓ R close to expected {expected_R:.3f}V")
        else:
            print(f"✗ R differs from expected {expected_R:.3f}V")
            print(f"  Difference: {abs(np.mean(R) - expected_R):.3f}V")

        print("-" * 60)
        print(f"Mean X: {np.mean(self.all_X):.6f} V ± {np.std(self.all_X):.6f} V")
        print(f"Mean Y: {np.mean(self.all_Y):.6f} V ± {np.std(self.all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.6f} rad ± {np.std(Theta):.6f} rad")
        print(f"Theta range: [{np.min(Theta):.6f}, {np.max(Theta):.6f}] rad")
        print(f"Phase stability: {np.std(Theta):.3f} rad (lower is better)")

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

        # Print slow ADC diagnostics
        if slow_adc_channels:
            print("\n" + "=" * 60)
            print(f"SLOW ADC DIAGNOSTICS - SYNCHRONIZED @ {slow_adc_rate} Hz")
            print("=" * 60)
            for channel in slow_adc_channels:
                data = self.slow_adc_data[channel]
                if data:
                    values = np.array([v for _, v in data])
                    timestamps = np.array([t for t, _ in data])
                    rel_times = timestamps - self.acquisition_start_time

                    # Calculate actual sample rate
                    if len(timestamps) > 1:
                        time_diffs = np.diff(timestamps)
                        actual_rate = 1.0 / np.mean(time_diffs)
                        rate_jitter = np.std(time_diffs) * 1e6  # microseconds
                    else:
                        actual_rate = 0
                        rate_jitter = 0

                    print(f"\nAIN{channel} (Physical Pin AI{channel} = DIO{channel}_P):")
                    print(f"  Samples collected: {len(values):,}")
                    print(f"  Target rate: {slow_adc_rate} Hz")
                    print(f"  Actual rate: {actual_rate:.2f} Hz")
                    print(f"  Rate jitter: {rate_jitter:.1f} µs")
                    print(f"  Time range: {rel_times[0]:.3f} to {rel_times[-1]:.3f} s")
                    print(f"  Mean: {np.mean(values):.6f} V")
                    print(f"  Std: {np.std(values):.6f} V")
                    print(f"  Min/Max: {np.min(values):.6f} / {np.max(values):.6f} V")

                    # Check synchronization
                    rate_match = abs(actual_rate - self.sample_rate) / self.sample_rate * 100
                    if rate_match < 1:
                        print(f"  ✓ PERFECTLY SYNCHRONIZED (<1% rate difference)")
                    elif rate_match < 5:
                        print(f"  ✓ Well synchronized ({rate_match:.1f}% difference)")
                    else:
                        print(f"  ⚠ Rate mismatch ({rate_match:.1f}% difference)")

        print("=" * 60)

        # Create comprehensive plot
        n_subplots = 9 + len(slow_adc_channels)
        n_rows = (n_subplots + 2) // 3
        fig = plt.figure(figsize=(16, 4 * n_rows))
        ZoomOut_Amount = 5

        plot_idx = 1

        # 1. OUT1 (Reference Signal)
        ax1 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        n_periods = 5
        n_samples_plot = int(n_periods * self.sample_rate / self.ref_freq)
        n_samples_plot = min(n_samples_plot, len(out1_raw))
        ax1.plot(t_raw[:n_samples_plot] * 1000, out1_raw[:n_samples_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference Signal (OUT1) @ {self.ref_freq} Hz')
        ax1.grid(True)

        # 2. IN1 (Input Signal)
        ax2 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax2.plot(t_raw[:n_samples_plot] * 1000, in1_raw[:n_samples_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title('Input Signal (IN1)')
        ax2.grid(True)

        # 3. FFT Spectrum
        ax3 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz (target)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power (a.u.)')
        ax3.set_title('FFT Spectrum (baseband)')
        ax3.legend()
        ax3.grid(True)

        # 4. X vs Time
        ax4 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax4.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(self.all_X), color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_X):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-phase (X) vs Time [iq2]')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(t[0], t[-1])
        margin_X = ZoomOut_Amount * (np.max(self.all_X) - np.min(self.all_X))
        ax4.set_ylim(np.min(self.all_X) - margin_X, np.max(self.all_X) + margin_X)

        # 5. Y vs Time
        ax5 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax5.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(self.all_Y), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y) vs Time [iq2_2]')
        ax5.legend()
        ax5.grid(True)
        ax5.set_xlim(t[0], t[-1])
        margin_Y = ZoomOut_Amount * (np.max(self.all_Y) - np.min(self.all_Y))
        ax5.set_ylim(np.min(self.all_Y) - margin_Y, np.max(self.all_Y) + margin_Y)

        # 6. X vs Y (IQ plot)
        ax6 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax6.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
        ax6.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r+', markersize=15,
                 markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot (X vs Y)')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # 7. R vs Time
        ax7 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax7.plot(t, R, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(R):.4f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R) vs Time')
        ax7.legend()
        ax7.grid(True)
        ax7.set_xlim(t[0], t[-1])
        margin_R = ZoomOut_Amount * (np.max(R) - np.min(R))
        ax7.set_ylim(np.min(R) - margin_R, np.max(R) + margin_R)

        # 8. Theta vs Time
        ax8 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(Theta):.4f} rad')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (rad)')
        ax8.set_title('Phase (Theta) vs Time')
        ax8.legend()
        ax8.grid(True)
        ax8.set_xlim(t[0], t[-1])
        margin_Theta = ZoomOut_Amount * (np.max(Theta) - np.min(Theta))
        ax8.set_ylim(np.min(Theta) - margin_Theta, np.max(Theta) + margin_Theta)

        # 9. R vs Theta
        ax9 = plt.subplot(n_rows, 3, plot_idx)
        plot_idx += 1
        ax9.plot(Theta, R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax9.axhline(np.mean(R), color='b', linestyle='--', alpha=0.5)
        ax9.axvline(np.mean(Theta), color='r', linestyle='--', alpha=0.5)
        ax9.set_xlabel('Theta (rad)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('R vs Theta')
        ax9.grid(True)

        # 10+: Slow ADC channels
        for channel in slow_adc_channels:
            data = self.slow_adc_data[channel]
            if data:
                timestamps = np.array([t for t, _ in data])
                values = np.array([v for _, v in data])
                rel_times = timestamps - self.acquisition_start_time

                ax = plt.subplot(n_rows, 3, plot_idx)
                plot_idx += 1
                ax.plot(rel_times, values, 'b-', linewidth=1)
                ax.axhline(np.mean(values), color='r', linestyle='--', alpha=0.7,
                           label=f'Mean: {np.mean(values):.4f}V')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (V)')
                ax.set_title(f'Slow ADC - AIN{channel} @ {slow_adc_rate} Hz')
                ax.legend()
                ax.grid(True)

        plt.tight_layout()

        # Save data
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save plot
            img_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Plot saved: {img_path}")

            # Save lock-in data with timestamps
            if params.get('save_timestamps', False):
                data = np.column_stack((
                    self.sample_timestamps,
                    t,
                    R,
                    Theta,
                    self.all_X,
                    self.all_Y
                ))
                csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                           header="AbsoluteTimestamp,RelativeTime,R,Theta,X,Y",
                           comments='', fmt='%.10f')
                print(f"✓ Lock-in data saved: {csv_path}")
                print(f"  Columns: AbsoluteTimestamp (Unix), RelativeTime (s), R, Theta, X, Y")
                print(f"  Sample rate: {self.sample_rate:.2f} Hz")
            else:
                data = np.column_stack((t, R, Theta, self.all_X, self.all_Y))
                csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                           header="Time,R,Theta,X,Y", comments='', fmt='%.6f')
                print(f"✓ Lock-in data saved: {csv_path}")

            # Save slow ADC data
            for channel in slow_adc_channels:
                data = self.slow_adc_data[channel]
                if data:
                    timestamps = np.array([t for t, _ in data])
                    values = np.array([v for _, v in data])
                    rel_times = timestamps - self.acquisition_start_time

                    adc_data = np.column_stack((timestamps, rel_times, values))
                    csv_path = os.path.join(self.output_dir,
                                            f'slowadc_AIN{channel}_{timestamp_str}.csv')
                    np.savetxt(csv_path, adc_data, delimiter=",",
                               header="AbsoluteTimestamp,RelativeTime,Voltage",
                               comments='', fmt='%.10f')
                    print(f"✓ Slow ADC AIN{channel} saved: {csv_path}")
                    print(f"  Pin: AI{channel} (DIO{channel}_P)")
                    print(f"  Sample rate: {slow_adc_rate} Hz")
                    print(f"  Samples: {len(values):,}")
        else:
            plt.show()

        # Clean up
        self.scpi_xadc.close()


if __name__ == '__main__':
    rp = RedPitaya(hostname=HOSTNAME)

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
        'fft': SHOW_FFT,
        'save_timestamps': SAVE_TIMESTAMPS,
        'slow_adc_channels': SLOW_ADC_CHANNELS,
        'slow_adc_rate': SLOW_ADC_SAMPLE_RATE,
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN + SLOW ADC SIMULTANEOUS ACQUISITION")
    print("=" * 60)
    print("SETUP: Connect OUT1 directly to IN1")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print(f"Decimation: {DECIMATION}")
    print(f"Lock-in Sample Rate: {125e6 / DECIMATION:.2f} Hz")
    channel_names = ', '.join([f'AIN{ch}' for ch in SLOW_ADC_CHANNELS])
    print(f"Slow ADC Channels: {channel_names}")
    print(f"Slow ADC Sample Rate: {SLOW_ADC_SAMPLE_RATE} Hz")
    print(f"✓✓✓ SYNCHRONIZED: Both @ {SLOW_ADC_SAMPLE_RATE} Hz ✓✓✓")
    print(f"Save Timestamps: {SAVE_TIMESTAMPS}")
    print("=" * 60)
    print("\nXADC Pin Mapping:")
    for ch in SLOW_ADC_CHANNELS:
        print(f"  AIN{ch} → Physical Pin AI{ch} (DIO{ch}_P) [0-3.3V]")
    print("=" * 60)
    print("Expected for direct OUT1→IN1 connection:")
    print(f"  X = {REF_AMPLITUDE / 2:.3f} V (in-phase)")
    print("  Y = 0.000 V (quadrature)")
    print(f"  R = {REF_AMPLITUDE / 2:.3f} V (magnitude)")
    print("  Theta = 0.000 rad (phase)")
    print("  FFT peak at 0 Hz")
    print("=" * 60)
    print(f"\n⚡ HIGH-SPEED MODE: {SLOW_ADC_SAMPLE_RATE} Hz synchronized sampling!")
    print("=" * 60)

    rp.run(run_params)
