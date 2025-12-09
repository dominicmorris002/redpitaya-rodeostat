"""
Red Pitaya Lock-In Amplifier - WITH TIMESTAMP SYNCHRONIZATION + IN2 SAMPLING

SETUP: 
- Connect OUT1 directly to IN1 with a cable (for lock-in)
- Connect any signal to IN2 (for DC voltage sampling)

This version keeps your original scope-based lock-in code EXACTLY as is,
and adds simple background sampling of IN2 using direct ADC access.

IQ MODULE OUTPUTS:
- For iq2 module: iq2 = X (in-phase), iq2_2 = Y (quadrature)

SYNCHRONIZATION:
- Saves CSV with absolute timestamps (Unix time)
- Can be merged with NI-DAQ data using timestamp matching
- IN2 voltage sampled at same rate as scope captures
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 100  # Hz - AC excitation frequency
REF_AMPLITUDE = 0.5  # V - AC signal amplitude (will appear on OUT1)
OUTPUT_CHANNEL = 'out1'  # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET = 0  # degrees - phase adjustment (0, 90, 180, 270)
MEASUREMENT_TIME = 30.0  # seconds - how long to measure

# LOCK-IN FILTER BANDWIDTH
FILTER_BANDWIDTH = 10  # Hz - lower = cleaner, higher = faster response

# AVERAGING
AVERAGING_WINDOW = 1  # samples - set to 1 to see raw lock-in output first

# Data saving
SAVE_DATA = True  # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'

# Advanced settings
DECIMATION = 8192
SHOW_FFT = True

# Synchronization settings
SAVE_TIMESTAMPS = True  # Save absolute timestamps for sync with NI-DAQ

# NEW: IN2 sampling
SAMPLE_IN2 = True  # Set to True to record IN2 voltage alongside lock-in data
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

    def __init__(self, output_dir='test_data'):
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

        # Store capture timestamps
        self.capture_timestamps = []
        self.acquisition_start_time = None
        
        # NEW: IN2 voltage storage
        self.in2_voltages = []
        self.all_in2 = None

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival
        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        # For iq2 module, use iq2 (X) and iq2_2 (Y)
        self.scope.input1 = 'iq2'  # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

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
            quadrature_factor=1)

        print(f"Lock-in setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"Filter BW: {filter_bw} Hz")
        print(f"IQ2 output_direct: {self.lockin.output_direct} (outputs {ref_amp}V sine)")
        print(f"IQ2 amplitude: {self.lockin.amplitude} V")
        print(f"IQ2 input: {self.lockin.input}")
        print(f"Scope reading: iq2 (X) and iq2_2 (Y)")

    def read_in2_voltage(self):
        """Read IN2 voltage directly - SIMPLE method using scope property"""
        try:
            # This accesses the instantaneous IN2 voltage
            return self.scope.voltage_in2
        except:
            return np.nan

    def capture_lockin(self, sample_in2=False):
        """Captures scope data and appends to X and Y arrays with timestamps"""
        capture_time = time.time()
        
        # NEW: Sample IN2 BEFORE scope capture (when scope isn't active)
        if sample_in2:
            in2_voltage = self.read_in2_voltage()
            self.in2_voltages.append(in2_voltage)

        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)  # iq2 = X (in-phase)
        ch2 = np.array(self.scope._data_ch2_current)  # iq2_2 = Y (quadrature)

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(capture_time)

        return ch1, ch2

    def run(self, params):
        timeout = params['timeout']
        sample_in2 = params.get('sample_in2', False)
        
        self.setup_lockin(params)

        print("Waiting for lock-in to settle...")
        time.sleep(0.5)

        self.acquisition_start_time = time.time()
        print(f"\n✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        if sample_in2:
            print("✓ IN2 voltage will be sampled alongside lock-in data")

        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_lockin(sample_in2=sample_in2)

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        # Generate per-sample timestamps
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
        
        # NEW: Expand IN2 voltages to match sample count
        if sample_in2 and len(self.in2_voltages) > 0:
            self.all_in2 = np.repeat(self.in2_voltages, samples_per_capture)[:total_samples]

        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)

        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_X)]
            if self.all_in2 is not None:
                self.all_in2 = self.all_in2[:len(self.all_X)]
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

        # Print diagnostics
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

        # NEW: Print IN2 statistics
        if self.all_in2 is not None:
            print("-" * 60)
            print("IN2 VOLTAGE STATISTICS:")
            print(f"Mean IN2: {np.mean(self.all_in2):.6f} V ± {np.std(self.all_in2):.6f} V")
            print(f"IN2 range: [{np.min(self.all_in2):.6f}, {np.max(self.all_in2):.6f}] V")

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

        # Create comprehensive plot
        fig = plt.figure(figsize=(16, 12))
        ZoomOut_Amount = 5

        # Row 1: Raw signals and FFT
        ax1 = plt.subplot(4, 3, 1)
        n_periods = 5
        n_samples_plot = int(n_periods * self.sample_rate / self.ref_freq)
        n_samples_plot = min(n_samples_plot, len(out1_raw))
        ax1.plot(t_raw[:n_samples_plot] * 1000, out1_raw[:n_samples_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference Signal (OUT1) @ {self.ref_freq} Hz')
        ax1.grid(True)

        ax2 = plt.subplot(4, 3, 2)
        ax2.plot(t_raw[:n_samples_plot] * 1000, in1_raw[:n_samples_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title('Input Signal (IN1)')
        ax2.grid(True)

        ax3 = plt.subplot(4, 3, 3)
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz (target)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power (a.u.)')
        ax3.set_title('FFT Spectrum (baseband)')
        ax3.legend()
        ax3.grid(True)

        # Row 2: X, Y, IQ plot
        ax4 = plt.subplot(4, 3, 4)
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

        ax5 = plt.subplot(4, 3, 5)
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

        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
        ax6.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r+', markersize=15,
                 markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot (X vs Y)')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # Row 3: R, Theta, R vs Theta
        ax7 = plt.subplot(4, 3, 7)
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

        ax8 = plt.subplot(4, 3, 8)
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

        ax9 = plt.subplot(4, 3, 9)
        ax9.plot(Theta, R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax9.axhline(np.mean(R), color='b', linestyle='--', alpha=0.5)
        ax9.axvline(np.mean(Theta), color='r', linestyle='--', alpha=0.5)
        ax9.set_xlabel('Theta (rad)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('R vs Theta')
        ax9.grid(True)

        # Row 4: IN2 voltage (if enabled)
        if self.all_in2 is not None:
            ax10 = plt.subplot(4, 3, 10)
            ax10.plot(t, self.all_in2, 'orange', linewidth=0.5)
            ax10.axhline(np.mean(self.all_in2), color='b', linestyle='--', alpha=0.7,
                        label=f'Mean: {np.mean(self.all_in2):.4f}V')
            ax10.set_xlabel('Time (s)')
            ax10.set_ylabel('IN2 (V)')
            ax10.set_title('IN2 Voltage vs Time')
            ax10.legend()
            ax10.grid(True)
            ax10.set_xlim(t[0], t[-1])

        plt.tight_layout()

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            img_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Plot saved: {img_path}")

            if params.get('save_timestamps', False):
                if self.all_in2 is not None:
                    data = np.column_stack((self.sample_timestamps, t, R, Theta, self.all_X, self.all_Y, self.all_in2))
                    csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                    np.savetxt(csv_path, data, delimiter=",",
                               header="AbsoluteTimestamp,RelativeTime,R,Theta,X,Y,IN2",
                               comments='', fmt='%.10f')
                    print(f"✓ Data saved with timestamps: {csv_path}")
                    print(f"  Columns: AbsoluteTimestamp (Unix), RelativeTime (s), R, Theta, X, Y, IN2")
                else:
                    data = np.column_stack((self.sample_timestamps, t, R, Theta, self.all_X, self.all_Y))
                    csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                    np.savetxt(csv_path, data, delimiter=",",
                               header="AbsoluteTimestamp,RelativeTime,R,Theta,X,Y",
                               comments='', fmt='%.10f')
                    print(f"✓ Data saved with timestamps: {csv_path}")
                    print(f"  Columns: AbsoluteTimestamp (Unix), RelativeTime (s), R, Theta, X, Y")
            else:
                if self.all_in2 is not None:
                    data = np.column_stack((t, R, Theta, self.all_X, self.all_Y, self.all_in2))
                    csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                    np.savetxt(csv_path, data, delimiter=",",
                               header="Time,R,Theta,X,Y,IN2", comments='', fmt='%.6f')
                else:
                    data = np.column_stack((t, R, Theta, self.all_X, self.all_Y))
                    csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                    np.savetxt(csv_path, data, delimiter=",",
                               header="Time,R,Theta,X,Y", comments='', fmt='%.6f')
                print(f"✓ Data saved: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitaya()

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
        'sample_in2': SAMPLE_IN2,  # NEW parameter
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN AMPLIFIER - WITH TIMESTAMP SYNC + IN2 SAMPLING")
    print("=" * 60)
    print("SETUP: Connect OUT1 directly to IN1")
    if SAMPLE_IN2:
        print("       Connect signal to IN2 for voltage monitoring")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print(f"Save Timestamps: {SAVE_TIMESTAMPS}")
    print(f"Sample IN2: {SAMPLE_IN2}")
    print("=" * 60)
    print("Expected for direct OUT1→IN1 connection:")
    print(f"  X = {REF_AMPLITUDE / 2:.3f} V (in-phase)")
    print("  Y = 0.000 V (quadrature)")
    print(f"  R = {REF_AMPLITUDE / 2:.3f} V (magnitude)")
    print("  Theta = 0.000 rad (phase)")
    print("  FFT peak at 0 Hz")
    print("=" * 60)
    print("\nNOTE: Timestamps will be saved for synchronization with NI-DAQ data")
    if SAMPLE_IN2:
        print("      IN2 voltage sampled at each scope capture")
    print("=" * 60)

    rp.run(run_params)
