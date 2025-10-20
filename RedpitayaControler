"""
Red Pitaya EIS Control System
============================

PYTHON VERSION: 3.8 (Required - PyRPL doesn't work with 3.12)

PIP INSTALL COMMANDS (run these in order):
pip install --upgrade pip setuptools wheel
pip install PyQt5==5.15.7
pip install pyqtgraph==0.12.4
pip install quamash==0.6.0
pip install pyrpl
pip install numpy matplotlib pandas scipy h5py

SETUP INSTRUCTIONS:
1. Use Python 3.8 (not 3.12)
2. Create new PyCharm project with Python 3.8
3. Run the pip install commands above
4. Update IP address in code to match your Red Pitaya
5. Connect Red Pitaya to network and power on

TROUBLESHOOTING:
- If PyRPL fails: Use redpitaya_scpi instead
- If Qt errors: Install PyQt5==5.15.7
- If pyqtgraph errors: Install pyqtgraph==0.12.4
- If connection fails: Check Red Pitaya IP address

ALTERNATIVE (if PyRPL has issues):
pip install redpitaya_scpi numpy matplotlib pandas scipy
"""

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

    def __init__(self, output_dir='test_data'):
        # UPDATE THIS IP ADDRESS TO MATCH YOUR RED PITAYA
        self.rp = Pyrpl(config='lockin_config', hostname='192.168.1.100')  # Change IP here!
        self.output_dir = output_dir

        self.rp_modules = self.rp.rp
        self.lia_scope = self.rp_modules.scope
        self.lia_scope.input1 = 'iq2'
        self.lia_scope.input2 = 'iq2_2'
        self.lia_scope.decimation = 8192
        self.lia_scope._start_acquisition_rolling_mode()
        self.lia_scope.average = 'true'
        self.sample_rate = 125e6/self.lia_scope.decimation

        self.iq2 = self.rp_modules.iq2

    def setup_test_sig(self, params):
        test_freq = params['test_freq']
        test_amp = params['test_amp']
        noise_freq = params['noise_freq']
        noise_amp = params['noise_amp']

        self.test_sig = self.rp_modules.asg0
        self.test_noise = self.rp_modules.asg1

        self.test_sig.setup(waveform='sin',
                            frequency=test_freq,
                            amplitude=test_amp,
                            offset=0.00,
                            output_direct='out1',
                            trigger_source='immediately')
        self.test_noise.setup(waveform='sin',
                              frequency=noise_freq,
                              amplitude=noise_amp,
                              offset=0.00,
                              output_direct='out1',
                              trigger_source='immediately')

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        ref_amp = params['ref_amp']

        self.iq2.setup(frequency=self.ref_freq,
                       bandwidth=[-self.ref_freq * 2, -self.ref_freq, self.ref_freq, self.ref_freq * 2],  # Hz
                       gain=1.0,
                       phase=0,
                       acbandwidth=0,
                       amplitude=ref_amp,
                       input='in1',
                       output_direct='out2',
                       output_signal='quadrature',
                       quadrature_factor=20)

    def capture(self):
        self.lia_scope.single()
        ch1 = np.array(self.lia_scope._data_ch1_current)
        ch2 = np.array(self.lia_scope._data_ch2_current)
        return ch1, ch2

    def see_fft(self, save_file=False):
        # --- Capture input (e.g. in2) ---
        self.lia_scope.input1 = 'in2'
        self.lia_scope.single()
        data_in = np.array(self.lia_scope._data_ch1_current)
        N = len(data_in)
        freqs = np.fft.rfftfreq(N, 1 / self.sample_rate)
        fft_in = np.fft.rfft(data_in * np.hanning(N))
        psd_in = (np.abs(fft_in) ** 2) / (self.sample_rate * N)

        # --- Capture lock-in outputs (X = iq2.i, Y = iq2.q) ---
        self.lia_scope.input1 = 'iq2'
        time.sleep(0.01)
        X, Y = self.capture()

        # Combine into magnitude
        iq = X + 1j*Y
        N_lock = len(iq)
        win = np.hanning(N)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(N_lock, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))

        idx = np.argmax(psd_lock)  # PSD computed as above
        print("Peak at", freqs[idx], "Hz   (expected difference =", abs(self.test_freq - self.ref_freq), "Hz)")
        # --- Plot ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].semilogy(freqs, psd_in, label='Input IN2')
        ax[0].axvline(self.ref_freq, color='orange', linestyle='--', label='Reference')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Power (a.u.)')
        ax[0].set_title('Input Signal Spectrum')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].semilogy(freqs_lock, psd_lock, label='Lock-in R')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power (a.u.)')
        ax[1].set_title('Lock-in Output Spectrum (baseband)')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        if save_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            path = os.path.join(self.output_dir, f'lockin_FFT_tf_{self.test_freq}_rf_{self.ref_freq}_w_noise.png')
            plt.savefig(path)
        else:
            plt.show()

    def run(self, params, save_file=False, test=True, fft=True):
        if test:
            self.test_freq = params['test_freq']
            self.setup_test_sig(params)
        self.setup_lockin(params)
        time.sleep(0.01)

        if fft:
            self.see_fft(save_file=save_file)
        else:
            X, Y = self.capture()
            t = np.arange(start=0, stop=len(X)/self.sample_rate, step=1/self.sample_rate)
            plt.plot(t, X)
            plt.plot(t, Y)
            plt.title('Lockin Results')
            plt.xlabel('Time (s)')
            plt.ylabel('X and Y outputs')

            if save_file:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

                path = os.path.join(self.output_dir, f'lockin_results_tf_{self.test_freq}_rf_{self.ref_freq}_w_noise.png')
                plt.savefig(path)
            else:
                plt.show()


if __name__ == '__main__':

    rp = RedPitaya()

    run_params= {
        'test_freq': 100,
        'test_amp': 0.4,
        'noise_freq': 10000,
        'noise_amp': 0.1,
        'ref_freq': 100,
        'ref_amp': 0.4
    }

    rp.run(run_params, save_file=False)
