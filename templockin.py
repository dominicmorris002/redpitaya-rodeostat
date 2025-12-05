"""
Red Pitaya Lock-In Amplifier - FIXED IN2 CAPTURE & PROPER ZOOM

SETUP: Connect OUT1 directly to IN1 with a cable
       Connect your DC voltage ramp to IN2

SOLUTION: Use PID as passthrough to capture IN2 as a scope signal!
- Scope CH1: iq2 (X - in-phase)
- Scope CH2: iq2_2 (Y - quadrature)
Then switch to:
- Scope CH1: iq2 (X - in-phase)
- Scope CH2: pid0 (IN2 via passthrough)
And interleave rapidly to get all three signals.

WAIT - BETTER: Capture X+Y together, then X+IN2 together, match by X values.

BEST: Rapid alternation between [X,Y] and [X,IN2] so timestamps are close.
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
SAVE_DATA = False  # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'

# Advanced settings
DECIMATION = 8192
SHOW_FFT = True
# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os

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

        # Check for second scope
        try:
            self.scope2 = self.rp_modules.scope2
            print("✓ Second scope detected! You can use it for truly synchronous X,Y,IN2 capture.")
            print("Scope2 inputs:", self.scope2.inputs)
        except AttributeError:
            self.scope2 = None
            print("✗ No second scope detected. Will use interleaving [X+Y] / [X+IN2].")

        # Storage arrays
        self.X_Y_data = []  # Stores (X, Y) tuples
        self.X_IN2_data = []  # Stores (X, IN2) tuples

        self.all_X = []
        self.all_Y = []
        self.all_in2 = []

        # Configure PID0 as IN2 passthrough
        self.pid0 = self.rp_modules.pid0
        self.pid0.input = 'in2'
        self.pid0.output_direct = 'off'
        self.pid0.setpoint = 0
        self.pid0.p = 1.0  # Unity gain
        self.pid0.i = 0
        self.pid0.d = 0
        self.pid0.ival = 0
        print("PID0 configured as IN2 passthrough")

        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        # Start with X + Y configuration
        self.scope.input1 = 'iq2'    # X (in-phase)
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

        # Turn OFF ASG0 - IQ module generates the reference
        self.ref_sig.output_direct = 'off'
        print("ASG0 disabled - IQ module will generate reference")

        # IQ MODULE setup
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,  # No feedback
            phase=phase_setting,
            acbandwidth=0,  # DC-coupled input
            amplitude=ref_amp,  # Output amplitude
            input='in1',
            output_direct=params['output_ref'],  # Send sine wave to OUT1/OUT2
            output_signal='quadrature',
            quadrature_factor=1)

        print(f"Lock-in setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"Filter BW: {filter_bw} Hz")
        print(f"Decimation: {DECIMATION}, Sample rate: {self.sample_rate:.2f} Hz")

    def run(self, params):
        timeout = params['timeout']
        self.setup_lockin(params)

        # Let the lock-in settle
        print("Waiting for lock-in to settle...")
        time.sleep(0.5)

        loop_start = time.time()

        print(f"Starting RAPID ALTERNATION capture:")
        print(f"  Alternating: [X+Y] -> [X+IN2] -> [X+Y] -> [X+IN2] ...")
        print(f"  This keeps timestamps very close together!")

        capture_count = 0
        while (time.time() - loop_start) < timeout:
            if capture_count % 2 == 0:
                # Config 1: Capture X + Y
                self.scope.input1 = 'iq2'
                self.scope.input2 = 'iq2_2'
                self.scope.single()
                X = np.array(self.scope._data_ch1_current)
                Y = np.array(self.scope._data_ch2_current)
                self.X_Y_data.append((X, Y))
            else:
                # Config 2: Capture X + IN2
                self.scope.input1 = 'iq2'
                self.scope.input2 = 'pid0'  # IN2 via passthrough
                self.scope.single()
                X = np.array(self.scope._data_ch1_current)
                IN2 = np.array(self.scope._data_ch2_current)
                self.X_IN2_data.append((X, IN2))

            capture_count += 1

        print(f"Captured {len(self.X_Y_data)} [X+Y] traces and {len(self.X_IN2_data)} [X+IN2] traces")

        # Process data: interleave X+Y and X+IN2
        # Take the minimum count
        n_pairs = min(len(self.X_Y_data), len(self.X_IN2_data))

        X_list = []
        Y_list = []
        IN2_list = []

        for i in range(n_pairs):
            # Add X and Y from X_Y capture
            X_list.append(self.X_Y_data[i][0])
            Y_list.append(self.X_Y_data[i][1])
            # Add IN2 from X_IN2 capture (same length as X)
            IN2_list.append(self.X_IN2_data[i][1])

        self.all_X = np.concatenate(X_list)
        self.all_Y = np.concatenate(Y_list)
        self.all_in2 = np.concatenate(IN2_list)

        # Match lengths
        min_len = min(len(self.all_X), len(self.all_Y), len(self.all_in2))
        self.all_X = self.all_X[:min_len]
        self.all_Y = self.all_Y[:min_len]
        self.all_in2 = self.all_in2[:min_len]

        print(f"Synchronized data: {min_len} samples each")
        print("Note: X,Y from one timebase; IN2 from interleaved timebase (very close!)")

        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)

        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_in2 = np.convolve(self.all_in2, np.ones(averaging_window) / averaging_window, mode='valid')
            print(f"Applied {averaging_window}-sample moving average filter")

        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)

        # Time array
        t = np.arange(len(self.all_X)) / self.sample_rate

        # Capture raw signals for plotting
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        time.sleep(0.05)
        self.scope.single()
        in1_raw = np.array(self.scope._data_ch1_current)
        out1_raw = np.array(self.scope._data_ch2_current)
        t_raw = np.arange(len(out1_raw)) / self.sample_rate

        # Restore scope configuration
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

        # IN2 statistics
        print("-" * 60)
        print("IN2 DC Voltage Statistics:")
        print(f"IN2 samples: {len(self.all_in2)} (interleaved with X, Y)")
        print(f"Mean IN2: {np.mean(self.all_in2):.6f} V ± {np.std(self.all_in2):.6f} V")
        print(f"IN2 range: [{np.min(self.all_in2):.6f}, {np.max(self.all_in2):.6f}] V")
        print(f"IN2 start: {self.all_in2[0]:.6f} V")
        print(f"IN2 end: {self.all_in2[-1]:.6f} V")
        print(f"Total IN2 change: {self.all_in2[-1] - self.all_in2[0]:.6f} V")
        if len(self.all_in2) > 1:
            ramp_rate = (self.all_in2[-1] - self.all_in2[0]) / t[-1]
            print(f"Average ramp rate: {ramp_rate:.6f} V/s")

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

        # Create comprehensive plot - WITH PROPER ZOOM (like old code)
        fig = plt.figure(figsize=(16, 12))

        # 1. OUT1 (Reference Signal)
        ax1 = plt.subplot(4, 3, 1)
        n_periods = 5
        n_samples_plot = int(n_periods * self.sample_rate / self.ref_freq)
        n_samples_plot = min(n_samples_plot, len(out1_raw))
        ax1.plot(t_raw[:n_samples_plot] * 1000, out1_raw[:n_samples_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference Signal @ {self.ref_freq} Hz')
        ax1.grid(True)

        # 2. IN1 (Input Signal)
        ax2 = plt.subplot(4, 3, 2)
        ax2.plot(t_raw[:n_samples_plot] * 1000, in1_raw[:n_samples_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title('Input Signal')
        ax2.grid(True)

        # 3. IN2 DC Voltage vs Time
        ax3 = plt.subplot(4, 3, 3)
        ax3.plot(t, self.all_in2, 'orange', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('IN2 (V)')
        ax3.set_title('IN2 Voltage (full waveform)')
        ax3.grid(True)
        ax3.set_xlim(t[0], t[-1])

        # 4. FFT Spectrum
        ax4 = plt.subplot(4, 3, 4)
        ax4.semilogy(freqs_lock, psd_lock, 'b-', linewidth=1, label='Lock-in PSD')
        ax4.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz (target)')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Power (a.u.)')
        ax4.set_title('FFT Spectrum (baseband)')
        ax4.legend()
        ax4.grid(True)

        # 5. X vs Time - PROPER ZOOM (like old code)
        ax5 = plt.subplot(4, 3, 5)
        ax5.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax5.axhline(np.mean(self.all_X), color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_X):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('X (V)')
        ax5.set_title('In-phase (X) vs Time')
        ax5.legend()
        ax5.grid(True)
        # Zoom out like old code
        ax5.set_xlim(t[0], t[-1])
        margin_X = 0.5 * (np.max(self.all_X) - np.min(self.all_X))
        ax5.set_ylim(np.min(self.all_X) - margin_X, np.max(self.all_X) + margin_X)

        # 6. Y vs Time - PROPER ZOOM
        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax6.axhline(np.mean(self.all_Y), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('Quadrature (Y) vs Time')
        ax6.legend()
        ax6.grid(True)
        ax6.set_xlim(t[0], t[-1])
        margin_Y = 0.5 * (np.max(self.all_Y) - np.min(self.all_Y))
        ax6.set_ylim(np.min(self.all_Y) - margin_Y, np.max(self.all_Y) + margin_Y)

        # 7. X vs Y (IQ plot)
        ax7 = plt.subplot(4, 3, 7)
        ax7.plot(self.all_X, self.all_Y, 'g.', markersize=0.5, alpha=0.3)
        ax7.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r+', markersize=15,
                 markeredgewidth=2, label='Mean')
        ax7.set_xlabel('X (V)')
        ax7.set_ylabel('Y (V)')
        ax7.set_title('IQ Plot (X vs Y)')
        ax7.legend()
        ax7.grid(True)
        ax7.axis('equal')

        # 8. R vs Time - PROPER ZOOM
        ax8 = plt.subplot(4, 3, 8)
        ax8.plot(t, R, 'm-', linewidth=0.5)
        ax8.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(R):.4f}V')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('R (V)')
        ax8.set_title('Magnitude (R) vs Time')
        ax8.legend()
        ax8.grid(True)
        ax8.set_xlim(t[0], t[-1])
        margin_R = 0.5 * (np.max(R) - np.min(R))
        ax8.set_ylim(np.min(R) - margin_R, np.max(R) + margin_R)

        # 9. Theta vs Time - PROPER ZOOM
        ax9 = plt.subplot(4, 3, 9)
        ax9.plot(t, Theta, 'c-', linewidth=0.5)
        ax9.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(Theta):.4f} rad')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Theta (rad)')
        ax9.set_title('Phase (Theta) vs Time')
        ax9.legend()
        ax9.grid(True)
        ax9.set_xlim(t[0], t[-1])
        margin_Theta = 0.5 * (np.max(Theta) - np.min(Theta))
        ax9.set_ylim(np.min(Theta) - margin_Theta, np.max(Theta) + margin_Theta)

        # 10. R vs IN2 - CRITICAL FOR AC CV!
        ax10 = plt.subplot(4, 3, 10)
        scatter = ax10.scatter(self.all_in2, R, c=t, cmap='viridis',
                              s=1, alpha=0.5)
        ax10.set_xlabel('IN2 Voltage (V)')
        ax10.set_ylabel('R - Magnitude (V)')
        ax10.set_title('Lock-in Magnitude vs IN2')
        ax10.grid(True)
        plt.colorbar(scatter, ax=ax10, label='Time (s)')

        # 11. Theta vs IN2 - CRITICAL FOR AC CV!
        ax11 = plt.subplot(4, 3, 11)
        scatter2 = ax11.scatter(self.all_in2, Theta, c=t, cmap='viridis',
                               s=1, alpha=0.5)
        ax11.set_xlabel('IN2 Voltage (V)')
        ax11.set_ylabel('Theta - Phase (rad)')
        ax11.set_title('Phase vs IN2')
        ax11.grid(True)
        plt.colorbar(scatter2, ax=ax11, label='Time (s)')

        # 12. X vs IN2
        ax12 = plt.subplot(4, 3, 12)
        scatter3 = ax12.scatter(self.all_in2, self.all_X, c=t, cmap='viridis',
                               s=1, alpha=0.5)
        ax12.set_xlabel('IN2 Voltage (V)')
        ax12.set_ylabel('X - In-phase (V)')
        ax12.set_title('In-phase vs IN2')
        ax12.grid(True)
        plt.colorbar(scatter3, ax=ax12, label='Time (s)')

        plt.tight_layout()

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            img_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.png')

            # Save synchronized data
            data = np.column_stack((t, R, Theta, self.all_X, self.all_Y, self.all_in2))
            csv_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.csv')
            np.savetxt(csv_path, data, delimiter=",",
                      header="Time,R,Theta,X,Y,IN2", comments='', fmt='%.6f')

            plt.savefig(img_path, dpi=150)
            print(f"✓ Data saved to {csv_path}")
            print(f"✓ Plot saved to {img_path}")
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
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN - RAPID ALTERNATION FOR AC CV")
    print("=" * 60)
    print("SETUP: Connect OUT1 directly to IN1")
    print("       Connect voltage signal to IN2")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print("=" * 60)
    print("Acquisition strategy:")
    print("  - Rapid alternation: [X+Y] -> [X+IN2] -> [X+Y] -> [X+IN2]")
    print("  - Captures full IN2 waveform (not just one sample)")
    print("  - Very close timestamps for X, Y, and IN2")
    print("=" * 60)
    print("Expected for direct OUT1→IN1 connection:")
    print(f"  X = {REF_AMPLITUDE / 2:.3f} V (in-phase)")
    print("  Y = 0.000 V (quadrature)")
    print(f"  R = {REF_AMPLITUDE / 2:.3f} V (magnitude)")
    print("  Theta = 0.000 rad (phase)")
    print("=" * 60)

    rp.run(run_params)
