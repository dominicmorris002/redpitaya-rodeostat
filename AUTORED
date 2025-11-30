"""
Red Pitaya Lock-In Amplifier - CORRECTED VERSION

SETUP: Connect OUT1 as Output Voltage Connect IN1 to the current TIA.

IQ MODULE OUTPUTS:
- For iq2 module: iq2 = X (in-phase), iq2_2 = Y (quadrature)
- For iq0 module: iq0 = X (in-phase), iq0_2 = Y (quadrature)
- For iq1 module: iq1 = X (in-phase), iq1_2 = Y (quadrature)


"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 500        # Hz - AC excitation frequency
REF_AMPLITUDE = 0.02        # V - AC signal amplitude (will appear on OUT1)
OUTPUT_CHANNEL = 'out1'    # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET = 0           # degrees - phase adjustment (0, 90, 180, 270)
MEASUREMENT_TIME = 12.0     # seconds - how long to measure

# LOCK-IN FILTER BANDWIDTH
FILTER_BANDWIDTH = 10      # Hz - lower = cleaner, higher = faster response

# AVERAGING
AVERAGING_WINDOW = 1       # samples - set to 1 to see raw lock-in output first

# Data saving
SAVE_DATA = True          # True = save to files, False = just show plots
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

def csvGenerate(dat, output_dir, base_filename):
    """Create CSV files for each data type, matching old SEED system format"""
    # Create a subfolder with the base filename
    folder_path = os.path.join(output_dir, base_filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save each array as a separate CSV file
    for key in dat.keys():
        csv_path = os.path.join(folder_path, f'{key}.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(dat[key])
    
    print(f"CSV files saved to: {folder_path}")

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
        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival
        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        # CORRECTED: For iq2 module, use iq2 (X) and iq2_2 (Y)
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

        # CRITICAL: Turn OFF ASG0 - we don't need it!
        # The IQ module will generate and output the reference signal
        self.ref_sig.output_direct = 'off'
        print("ASG0 disabled - IQ module will generate reference")

        # IQ MODULE DOES EVERYTHING:
        # - Generates sine wave at ref_freq with amplitude ref_amp
        # - Outputs it to OUT1 (or OUT2)
        # - Demodulates signal from IN1
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,              # No feedback
            phase=phase_setting,
            acbandwidth=0,         # DC-coupled input
            amplitude=ref_amp,     # THIS IS THE OUTPUT AMPLITUDE!
            input='in1',
            output_direct=params['output_ref'],  # Send sine wave to OUT1/OUT2
            output_signal='quadrature',
            quadrature_factor=1)   # No extra gain

        print(f"Lock-in setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"Filter BW: {filter_bw} Hz")
        print(f"IQ2 output_direct: {self.lockin.output_direct} (outputs {ref_amp}V sine)")
        print(f"IQ2 amplitude: {self.lockin.amplitude} V")
        print(f"IQ2 input: {self.lockin.input}")
        print(f"Scope reading: iq2 (X) and iq2_2 (Y)")

    def capture_lockin(self):
        """Captures scope data and appends to X and Y arrays"""
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)  # iq2 = X (in-phase)
        ch2 = np.array(self.scope._data_ch2_current)  # iq2_2 = Y (quadrature)

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

        # Let the lock-in settle
        print("Waiting for lock-in to settle...")
        time.sleep(0.5)

        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)

        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            print(f"Applied {averaging_window}-sample moving average filter")

        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)

        # Time array
        t = np.arange(start=0, stop=len(self.all_X) / self.sample_rate, step=1 / self.sample_rate)

        # Capture raw signals for plotting
        self.scope.input1 = 'out1'  # Reference signal from IQ module
        self.scope.input2 = 'in1'   # Input signal
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
        print(f"Measurement Duration: {len(self.all_X) / self.sample_rate:.3f} seconds")
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
        ax4.set_title('In-phase (X) vs Time [iq2]')
        ax4.legend()
        ax4.grid(True)

        # 5. Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(self.all_Y), color='b', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y) vs Time [iq2_2]')
        ax5.legend()
        ax5.grid(True)

        # 6. X vs Y (IQ plot)
        ax6 = plt.subplot(3, 3, 6)
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
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, R, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(R):.4f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R) vs Time')
        ax7.legend()
        ax7.grid(True)

        # 8. Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(Theta):.4f} rad')
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
            
            base_filename = f'lockin_results_rf_{self.ref_freq}'
            
            # Save NPZ file (like old system)
            npz_path = os.path.join(self.output_dir, f'{base_filename}.npz')
            np.savez(npz_path, 
                     magnitude=R, 
                     phase=Theta, 
                     time=t,
                     signal=self.all_X,
                     dcRamp=np.zeros_like(t),
                     X=self.all_X,
                     Y=self.all_Y)
            print(f"NPZ file saved: {npz_path}")
            
            # Save individual CSV files in subfolder (like old system)
            dat = {
                'magnitude': R,
                'phase': Theta,
                'time': t,
                'signal': self.all_X,
                'dcRamp': np.zeros_like(t),
                'X': self.all_X,
                'Y': self.all_Y
            }
            csvGenerate(dat, self.output_dir, base_filename)
            
            # Save the plot
            img_path = os.path.join(self.output_dir, f'{base_filename}.png')
            plt.savefig(img_path, dpi=150)
            print(f"Plot saved: {img_path}")
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
    print("SETUP: Connect OUT1 directly to IN1")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print("=" * 60)
    print("Expected for direct OUT1→IN1 connection:")
    print(f"  X = {REF_AMPLITUDE / 2:.3f} V (in-phase)")
    print("  Y = 0.000 V (quadrature)")
    print(f"  R = {REF_AMPLITUDE / 2:.3f} V (magnitude)")
    print("  Theta = 0.000 rad (phase)")
    print("  FFT peak at 0 Hz")
    print("=" * 60)

    rp.run(run_params)
