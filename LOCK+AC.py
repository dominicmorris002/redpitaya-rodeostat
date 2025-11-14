"""
Red Pitaya Outputs Ac Wave and Lockin Amplifier
"""
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
                       phase=((time.time() - self.ref_start_t)/self.ref_period)*360,       #initial phase is in degrees (delta t[ns])-> delta t [s]/(1/f) * 360
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

        plt.tight_layout()

    def run(self, params):
        timeout = params['timeout']

        self.setup_lockin(params)
        time.sleep(0.01)

        loop_start = time.time()

        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))
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
        print("Peak at", freqs_lock[idx], "Hz")

        # Create comprehensive plot with all lock-in outputs
        fig = plt.figure(figsize=(16, 10))

        # 1. FFT Spectrum
        ax1 = plt.subplot(3, 3, 1)
        ax1.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (a.u.)')
        ax1.set_title('FFT Spectrum (baseband)')
        ax1.legend()
        ax1.grid(True)

        # 2. X vs Time
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('X (V)')
        ax2.set_title('In-phase (X) vs Time')
        ax2.grid(True)

        # 3. Y vs Time
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Y (V)')
        ax3.set_title('Quadrature (Y) vs Time')
        ax3.grid(True)

        # 4. X vs Y (IQ plot)
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
        ax4.set_xlabel('X (V)')
        ax4.set_ylabel('Y (V)')
        ax4.set_title('IQ Plot (X vs Y)')
        ax4.grid(True)
        ax4.axis('equal')

        # 5. R vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, R, 'm-', linewidth=0.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('R (V)')
        ax5.set_title('Magnitude (R) vs Time')
        ax5.grid(True)

        # 6. Theta vs Time
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(t, Theta, 'c-', linewidth=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Theta (rad)')
        ax6.set_title('Phase (Theta) vs Time')
        ax6.grid(True)

        # 7. All signals (X, Y, R, Theta) normalized
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, self.all_X / np.max(np.abs(self.all_X)), label='X (norm)', alpha=0.7)
        ax7.plot(t, self.all_Y / np.max(np.abs(self.all_Y)), label='Y (norm)', alpha=0.7)
        ax7.plot(t, R / np.max(R), label='R (norm)', alpha=0.7)
        ax7.plot(t, Theta / np.max(np.abs(Theta)), label='Theta (norm)', alpha=0.7)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Normalized Amplitude')
        ax7.set_title('All Signals (Normalized)')
        ax7.legend()
        ax7.grid(True)

        # 8. R histogram
        ax8 = plt.subplot(3, 3, 8)
        ax8.hist(R, bins=50, edgecolor='black', alpha=0.7)
        ax8.set_xlabel('R (V)')
        ax8.set_ylabel('Count')
        ax8.set_title('Magnitude Distribution')
        ax8.grid(True)

        # 9. Theta histogram
        ax9 = plt.subplot(3, 3, 9)
        ax9.hist(Theta, bins=50, edgecolor='black', alpha=0.7)
        ax9.set_xlabel('Theta (rad)')
        ax9.set_ylabel('Count')
        ax9.set_title('Phase Distribution')
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
        'ref_freq': 100,            # Hz, reference signal frequency for lock-in
        'ref_amp': 0.4,             # V, amplitude of reference signal
        'output_ref': 'out1',       # where to output the ref_signal

        'timeout': 5.0,             # seconds, how long to run acquisition loop

        'output_dir': 'test_data',  # where to save FFT and waveform plots
        'save_file': False,         # whether to save plots instead of showing them
        'fft': True,                # whether to perform FFT after run
    }

    rp.run(run_params)


