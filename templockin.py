"""
Red Pitaya Lock-In Amplifier
Replaces SR7280 Lock-In Amplifier for ACV Measurements

EASY ACCESS PARAMETERS - MODIFY THESE FOR YOUR EXPERIMENT
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 100         # Hz - AC excitation frequency (typical: 50-1000 Hz)
REF_AMPLITUDE = 0.4         # V - AC signal amplitude (typical: 0.1-0.5 V)
OUTPUT_CHANNEL = 'out1'     # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET = 0            # degrees - phase adjustment (0, 90, 180, 270)

MEASUREMENT_TIME = 5.0      # seconds - how long to measure
AVERAGING_WINDOW = 10       # samples - moving average for noise reduction (higher = smoother)

# Data saving
SAVE_DATA = False           # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'  # where to save results

# Advanced settings (usually don't need to change)
DECIMATION = 64             # sample rate = 125MHz/decimation (64 = good default)
SHOW_FFT = True             # whether to compute and show FFT

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
    current_range_map = {'10uA': (False, True, True, True), '100uA': (True, False, True, True), '1mA': (True, True, False, True), '10mA': (True, True, True, False)}
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
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = 64

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6/self.scope.decimation

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1/self.ref_freq
        ref_amp = params['ref_amp']
        
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

        self.lockin.setup(frequency=self.ref_freq,
                       bandwidth=[-self.ref_freq * 2, -self.ref_freq, self.ref_freq, self.ref_freq * 2],  # Hz
                       gain=1.0,
                       phase=phase_setting,  # Use user-specified phase or 0
                       acbandwidth=0,
                       amplitude=ref_amp,
                       input='in1',
                       output_direct='out2',
                       output_signal='output_direct',
                       quadrature_factor=10)

    def capture_lockin(self):
        """
        captures a self.scope.decimation length capture and appends them to the X and Y arrays
        :return:
        """
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)

        if self.scope.input1 == 'iq2' and self.scope.input2 == 'iq2_2':
            self.lockin_X.append(ch1)
            self.lockin_Y.append(ch2)

        return ch1, ch2

    def see_fft(self):
        iq = self.all_X + 1j*self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))

        idx = np.argmax(psd_lock)  # PSD computed as above
        print("Peak at", freqs_lock[idx], "Hz")

        plt.figure(1, figsize=(12, 4))

        plt.semilogy(freqs_lock, psd_lock, label='Lock-in R')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('Lock-in Output Spectrum (baseband)')
        plt.legend()
        plt.grid(True)

        # Add figure title with parameters
        fig.suptitle(f'Lock-In Results | {self.ref_freq} Hz, {params["ref_amp"]} V | Avg Window: {params.get("averaging_window", 1)} | SNR: {np.mean(R)/np.std(R):.2f}', 
                     fontsize=12, fontweight='bold', y=0.995)
        
        # Add figure title with parameters
        fig.suptitle(f'Lock-In Amplifier Results | Freq: {self.ref_freq} Hz | Amp: {params["ref_amp"]} V | Phase: {params.get("phase", 0)}°', 
                     fontsize=14, fontweight='bold', y=0.995)

    def run(self, params):
        timeout = params['timeout']

        self.setup_lockin(params)
        time.sleep(0.01)

        loop_start = time.time()

        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))
        
        # Apply moving average filter for more accurate measurements
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            # Smooth X and Y with moving average
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window)/averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window)/averaging_window, mode='valid')
        
        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)

        # Time array
        t = np.arange(start=0, stop=len(self.all_X) / self.sample_rate, step=1 / self.sample_rate)

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
        print(f"Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_X)}")
        print(f"Measurement Duration: {len(self.all_X)/self.sample_rate:.3f} seconds")
        print("-" * 60)
        print(f"Mean R: {np.mean(R):.6f} V ± {np.std(R):.6f} V")
        print(f"SNR (R): {np.mean(R)/np.std(R):.2f} (mean/std)")
        print(f"R range: [{np.min(R):.6f}, {np.max(R):.6f}] V")
        print("-" * 60)
        print(f"Mean Theta: {np.mean(Theta):.6f} rad ± {np.std(Theta):.6f} rad")
        print(f"Theta range: [{np.min(Theta):.6f}, {np.max(Theta):.6f}] rad")
        print(f"Phase stability: {np.std(Theta):.3f} rad (lower is better)")
        print("-" * 60)
        print(f"Mean X: {np.mean(self.all_X):.6f} V ± {np.std(self.all_X):.6f} V")
        print(f"Mean Y: {np.mean(self.all_Y):.6f} V ± {np.std(self.all_Y):.6f} V")
        print("=" * 60)

        # Create comprehensive plot with all lock-in outputs
        fig = plt.figure(figsize=(16, 10))

        # 1. FFT Spectrum
        # Shows frequency content of demodulated signal. Peak at 0 Hz = locked correctly
        # If you see harmonics or peaks away from 0 Hz, lock-in isn't working properly
        ax1 = plt.subplot(3, 3, 1)
        ax1.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (a.u.)')
        ax1.set_title('FFT Spectrum (baseband)')
        ax1.legend()
        ax1.grid(True)

        # 2. X vs Time
        # In-phase component - signal component aligned with reference
        # Oscillations = phase drift, flat = stable lock, spikes = noise
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('X (V)')
        ax2.set_title('In-phase (X) vs Time')
        ax2.grid(True)

        # 3. Y vs Time
        # Quadrature component - signal 90° out of phase with reference
        # Together with X, gives complete signal information
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Y (V)')
        ax3.set_title('Quadrature (Y) vs Time')
        ax3.grid(True)

        # 4. X vs Y (IQ plot) - Full data
        # Shows signal trajectory in complex plane over entire measurement
        # Circle = perfectly locked, infinity symbol = phase drift, cloud = unlocked/noisy
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
        ax4.set_xlabel('X (V)')
        ax4.set_ylabel('Y (V)')
        ax4.set_title('IQ Plot (X vs Y) - Full Trace')
        ax4.grid(True)
        ax4.axis('equal')

        # 5. R vs Time
        # Signal magnitude (amplitude) over time - this is your main measurement!
        # Use mean(R) as your measurement value, std(R) as uncertainty
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, R, 'm-', linewidth=0.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('R (V)')
        ax5.set_title('Magnitude (R) vs Time')
        ax5.grid(True)

        # 6. Theta vs Time
        # Phase angle over time - shows phase stability
        # Constant = stable phase, drifting = phase not locked, jumping = noise
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(t, Theta, 'c-', linewidth=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Theta (rad)')
        ax6.set_title('Phase (Theta) vs Time')
        ax6.grid(True)

        # 7. R vs Theta (polar plot)
        # Magnitude vs phase correlation - shows if amplitude depends on phase
        # Random scatter = no correlation, pattern = phase-amplitude coupling
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(Theta, R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax7.set_xlabel('Theta (rad)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('R vs Theta (Magnitude vs Phase)')
        ax7.grid(True)

        # 8. Phasor plot - Short window (like Mason's circle plots)
        # Shows instantaneous lock state over 0.1 seconds
        # Clean circle = perfectly locked, oval = slight drift, flower = unlocked
        # This matches Mason's presentation plots showing lock quality
        ax8 = plt.subplot(3, 3, 8)
        # Take a short window (first 0.1 seconds) to see the lock pattern
        window_samples = int(0.1 * self.sample_rate)
        if len(self.all_X) > window_samples:
            X_window = self.all_X[:window_samples]
            Y_window = self.all_Y[:window_samples]
        else:
            X_window = self.all_X
            Y_window = self.all_Y
        
        ax8.plot(X_window, Y_window, 'b-', linewidth=1.5, alpha=0.8)
        ax8.set_xlabel('X (V)')
        ax8.set_ylabel('Y (V)')
        ax8.set_title('Phasor Plot (0.1s window)')
        ax8.grid(True)
        ax8.axis('equal')
        
        # Add circle to show if locked
        max_r = np.max(np.sqrt(X_window**2 + Y_window**2))
        circle = plt.Circle((0, 0), max_r, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax8.add_patch(circle)

        # 9. Phasor plot - Mid window
        # Another phasor snapshot from middle of measurement - compare with plot 8
        # If both show circles, lock is stable throughout measurement
        # If patterns differ, lock quality changed over time
        ax9 = plt.subplot(3, 3, 9)
        # Take middle 0.1 seconds
        mid_start = len(self.all_X) // 2
        mid_end = mid_start + int(0.1 * self.sample_rate)
        if mid_end <= len(self.all_X):
            X_mid = self.all_X[mid_start:mid_end]
            Y_mid = self.all_Y[mid_start:mid_end]
        else:
            X_mid = self.all_X[mid_start:]
            Y_mid = self.all_Y[mid_start:]
        
        ax9.plot(X_mid, Y_mid, 'r-', linewidth=1.5, alpha=0.8)
        ax9.set_xlabel('X (V)')
        ax9.set_ylabel('Y (V)')
        ax9.set_title('Phasor Plot (mid 0.1s window)')
        ax9.grid(True)
        ax9.axis('equal')
        
        # Add circle to show if locked
        max_r_mid = np.max(np.sqrt(X_mid**2 + Y_mid**2))
        circle_mid = plt.Circle((0, 0), max_r_mid, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax9.add_patch(circle_mid)

        plt.tight_layout()

                
        plt.tight_layout(rect=[0, 0, 1, 0.99])  # Make room for title

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
        'ref_freq': 100,            # Hz, reference signal frequency for lock-in
        'ref_amp': 0.4,             # V, amplitude of reference signal
        'output_ref': 'out1',       # where to output the ref_signal
        'phase': 0,                 # Phase in degrees - try 0, 90, 180, 270 to find best circle

        'timeout': 5.0,             # seconds, how long to run acquisition loop

        'output_dir': 'test_data',  # where to save FFT and waveform plots
        'save_file': False,         # whether to save plots instead of showing them
        'fft': True,                # whether to perform FFT after run
    }

    # To find best phase for circular phasor plot:
    # Try running with phase = 0, 45, 90, 135, 180, 225, 270, 315
    # Pick the one that gives the cleanest circle in phasor plots
    
    rp.run(run_params)
