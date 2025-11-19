"""
Red Pitaya Lock-In Amplifier - FULLY CORRECTED VERSION

EXPECTED RESULTS (OUT1 → IN1, 1V sine @ 100Hz):
- X ≈ 0.5V (flat line)
- Y ≈ 0V (flat line)
- R ≈ 0.5V (flat line)
- Theta ≈ 0 rad (flat line)
- FFT peak at 0 Hz

OSCILLOSCOPE CHECKS (for debugging):
- OUT1: sine wave (your reference from ASG0)
- OUT2: should be OFF (no signal)
- IN1: sine wave (same as OUT1 if directly connected)
- iq2: sine wave (this is the IQ module's internal sine reference - NOT a demodulated output!)
- iq2_2: sine wave (this is the IQ module's internal cosine reference - NOT a demodulated output!)
- iq0: This is quadrature output Q (use for scope viewing)
- iq1: This is in-phase output I (use for scope viewing)

CRITICAL FIX: The IQ module has FOUR outputs per module:
- iq0, iq1, iq2, iq2_2 are NOT the demodulated X and Y!
- The actual scope signals depend on which IQ module (iq0, iq1, iq2) you're using
- For iq2 module: Check the available scope input names to find actual quadrature outputs

TROUBLESHOOTING:
- If FFT peak is NOT at 0 Hz: Lock-in frequency doesn't match signal frequency
- If X oscillates wildly: Try different PHASE_OFFSET values (0, 90, 180, 270)
- If R is wrong: Check that IN1 is actually receiving the signal
- If OUT2 has a signal: output_direct should be 'off'
- Start with AVERAGING_WINDOW = 1 to see raw lock-in output first
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 100  # Hz - AC excitation frequency (typical: 50-1000 Hz)
REF_AMPLITUDE = 1  # V - AC signal amplitude (typical: 0.1-0.5 V)
OUTPUT_CHANNEL = 'out1'  # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET = 0  # degrees - phase adjustment (0, 90, 180, 270)
MEASUREMENT_TIME = 5.0  # seconds - how long to measure (longer = more accurate)

# LOCK-IN FILTER BANDWIDTH - THIS IS CRITICAL!
# Narrower = cleaner signal but slower response
# For 1V sin wave at 100Hz, expect: X=0.5V, Y=0V, R=0.5V, Theta=0
FILTER_BANDWIDTH = 10  # Hz - INCREASED from 2 to 10 for faster settling
# Try: 1 Hz (very clean, slow), 10 Hz (good balance), 50 Hz (fast)

AVERAGING_WINDOW = 1  # samples - moving average for noise reduction
# HIGHER = MORE ACCURATE but slower response
# For science: 50-100, For testing: 10, For fast response: 1
# NOTE: Start with 1 (no averaging) to see if lock-in is working first!

# Data saving
SAVE_DATA = False  # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'  # where to save results

# Advanced settings (usually don't need to change)
DECIMATION = 64  # sample rate = 125MHz/decimation (64 = good default)
SHOW_FFT = True  # whether to compute and show FFT

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
    electrode_map = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True), '100uA': (True, False, True, True),
                         '1mA': (True, True, False, True), '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True), '2X': (True, False), '10X': (True, True)}
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

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival

        self.scope = self.rp_modules.scope

        # CRITICAL: Need to identify the correct quadrature output signals
        # Print available scope inputs to see what's available
        print("Available scope inputs:", self.scope.inputs)

        # According to pyrpl docs, the quadrature outputs for iq2 should be accessible
        # Try to use the actual quadrature signals, not the internal sine/cosine references
        self.scope.input1 = 'iq2'  # This should be one quadrature
        self.scope.input2 = 'iq2_2'  # This should be the other quadrature

        self.scope.decimation = 64

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
        filter_bw = params.get('filter_bandwidth', 10)  # Increased default

        # Get phase setting - can be 'auto' or a specific value in degrees
        phase_setting = params.get('phase', 0)

        self.ref_sig.setup(waveform='sin',
                           amplitude=ref_amp,
                           frequency=self.ref_freq)

        self.ref_start_t = time.time()

        if params['output_ref'] == 'out1' or params['output_ref'] == 'out2':
            self.ref_sig.output_direct = params['output_ref']
        else:
            self.ref_sig.output_direct = 'off'

        # CRITICAL FIXES:
        # 1. output_direct should be 'off' - we don't want IQ2 sending anything to analog outputs
        # 2. bandwidth should be wider for faster settling
        self.lockin.setup(frequency=self.ref_freq,
                          bandwidth=filter_bw,  # Single value, not list (pyrpl will make it symmetric)
                          gain=0.0,  # IMPORTANT: gain=0 means no feedback from demod to mod
                          phase=phase_setting,
                          acbandwidth=0,  # Set to 0 to disable high-pass (for DC-coupled input)
                          amplitude=0,  # IMPORTANT: amplitude=0 means no modulation output
                          input='in1',
                          output_direct='off',  # CRITICAL: Don't send anything to OUT2!
                          output_signal='quadrature',  # Make quadratures available
                          quadrature_factor=1)  # CRITICAL: 1 = no amplification (was 10 = 10x gain!)

        print(f"Lock-in setup: {self.ref_freq} Hz, Filter BW: {filter_bw} Hz")
        print(f"IQ2 output_direct: {self.lockin.output_direct} (should be 'off')")
        print(f"IQ2 amplitude: {self.lockin.amplitude} (should be 0)")
        print(f"IQ2 quadrature_factor: {self.lockin.quadrature_factor} (should be 1 for no gain)")

    def capture_lockin(self):
        """
        Captures scope data and appends to X and Y arrays
        """
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)

        # Append data - we'll figure out which is X and which is Y from the results
        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)

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

        self.setup_lockin(params)

        # Let the lock-in settle before starting acquisition
        print("Waiting for lock-in to settle...")
        time.sleep(0.5)  # Increased settling time

        loop_start = time.time()

        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        # Check if we need to swap X and Y based on which has larger oscillations
        # The one oscillating at 100Hz is probably wrong!
        X_fft = np.fft.rfft(self.all_X)
        Y_fft = np.fft.rfft(self.all_Y)
        X_peak = np.max(np.abs(X_fft[1:]))  # Ignore DC
        Y_peak = np.max(np.abs(Y_fft[1:]))

        if X_peak > Y_peak * 2:  # X is oscillating much more than Y
            print("WARNING: Swapping X and Y - X was oscillating!")
            self.all_X, self.all_Y = self.all_Y, self.all_X

        # Apply moving average filter for more accurate measurements
        averaging_window = params.get('averaging_window', 1)

        # Store unfiltered data for diagnostics
        X_raw = self.all_X
        Y_raw = self.all_Y

        if averaging_window > 1:
            # Smooth X and Y with moving average
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            print(f"Applied {averaging_window}-sample moving average filter")

        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)

        # Time array
        t = np.arange(start=0, stop=len(self.all_X) / self.sample_rate, step=1 / self.sample_rate)

        # Capture raw signals for plotting
        scope_backup_in1 = self.scope.input1
        scope_backup_in2 = self.scope.input2

        self.scope.input1 = 'out1'  # Reference signal
        self.scope.input2 = 'in1'  # Input signal
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw = np.array(self.scope._data_ch2_current)
        t_raw = np.arange(len(out1_raw)) / self.sample_rate

        # Switch back to lock-in outputs
        self.scope.input1 = scope_backup_in1
        self.scope.input2 = scope_backup_in2

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

        # Check if lock-in is working properly
        if abs(freqs_lock[idx]) < 5:
            print("✓ Lock-in is LOCKED (peak near 0 Hz)")
        else:
            print("✗ WARNING: Lock-in NOT locked! Peak should be at 0 Hz!")
            print("  Peak at:", freqs_lock[idx], "Hz")
            print("  This might indicate wrong scope channel assignment")

        print(f"Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_X)}")
        print(f"Measurement Duration: {len(self.all_X) / self.sample_rate:.3f} seconds")
        print("-" * 60)
        print(f"Mean R: {np.mean(R):.6f} V ± {np.std(R):.6f} V")
        print(f"SNR (R): {np.mean(R) / np.std(R):.2f} (mean/std)")
        print(f"R range: [{np.min(R):.6f}, {np.max(R):.6f}] V")

        # Check if R is close to expected value
        expected_R = params['ref_amp'] / 2
        if abs(np.mean(R) - expected_R) < 0.1:
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

        # Additional diagnostic: check if X or Y are oscillating
        X_ac = np.std(self.all_X)
        Y_ac = np.std(self.all_Y)
        X_dc = np.mean(np.abs(self.all_X))
        Y_dc = np.mean(np.abs(self.all_Y))
        print("-" * 60)
        print("Signal characteristics:")
        print(f"X: DC={X_dc:.6f}V, AC={X_ac:.6f}V, AC/DC={X_ac / max(X_dc, 0.001):.3f}")
        print(f"Y: DC={Y_dc:.6f}V, AC={Y_ac:.6f}V, AC/DC={Y_ac / max(Y_dc, 0.001):.3f}")
        if X_ac / max(X_dc, 0.001) > 0.5:
            print("⚠ WARNING: X is oscillating! Should be flat for locked signal")
        if Y_ac / max(Y_dc, 0.001) > 0.5:
            print("⚠ WARNING: Y is oscillating! Should be flat for locked signal")
        print("=" * 60)

        # Create comprehensive plot
        fig = plt.figure(figsize=(16, 10))

        # 1. OUT1 (Reference Signal)
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_samples_plot = int(n_periods * self.sample_rate / self.ref_freq)
        n_samples_plot = min(n_samples_plot, len(out1_raw))
        ax1.plot(t_raw[:n_samples_plot] * 1000, out1_raw[:n_samples_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference Signal (OUT1) @ {self.ref_freq} Hz')
        ax1.grid(True)

        # 2. IN1 (Input Signal)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw[:n_samples_plot] * 1000, in1_raw[:n_samples_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title('Input Signal (IN1)')
        ax2.grid(True)

        # 3. FFT Spectrum
        ax3 = plt.subplot(3, 3, 3)
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz (target)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power (a.u.)')
        ax3.set_title('FFT Spectrum (baseband)')
        ax3.legend()
        ax3.grid(True)

        # 4. X vs Time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(self.all_X), color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_X):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-phase (X) vs Time')
        ax4.legend()
        ax4.grid(True)

        # 5. Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(self.all_Y), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y) vs Time')
        ax5.legend()
        ax5.grid(True)

        # 6. X vs Y (IQ plot)
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
        ax6.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot (X vs Y)')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # 7. R vs Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, R, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(R):.4f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R) vs Time')
        ax7.legend()
        ax7.grid(True)

        # 8. Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(Theta):.4f} rad')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (rad)')
        ax8.set_title('Phase (Theta) vs Time')
        ax8.legend()
        ax8.grid(True)

        # 9. R vs Theta
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(Theta, R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax9.axhline(np.mean(R), color='b', linestyle='--', alpha=0.5)
        ax9.axvline(np.mean(Theta), color='r', linestyle='--', alpha=0.5)
        ax9.set_xlabel('Theta (rad)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('R vs Theta')
        ax9.grid(True)

        plt.tight_layout()

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            img_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.png')
            data = np.column_stack((R, Theta, self.all_X, self.all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.csv')
            np.savetxt(csv_path, data, delimiter=",", header="R,Theta,X,Y", comments='', fmt='%.6f')
            plt.savefig(img_path, dpi=150)
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
    print("RED PITAYA LOCK-IN AMPLIFIER")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print("=" * 60)
    print("Expected for 1V sin wave perfectly locked:")
    print(f"  X = {REF_AMPLITUDE / 2:.3f} V (in-phase)")
    print("  Y = 0.000 V (quadrature)")
    print(f"  R = {REF_AMPLITUDE / 2:.3f} V (magnitude)")
    print("  Theta = 0.000 rad (phase)")
    print("=" * 60)

    rp.run(run_params)
