"""
Created on 10/28/2025
@author: Dominic
RedPitaya raw scope capture (no lock-in)
"""

from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class RedPitayaScope:

    def __init__(self, hostname='rp-f073ce.local', output_dir='scope_data', config_file=None):
        if config_file is None:
            root = Tk()
            root.withdraw()  # hide root window
            config_file = askopenfilename(title="Select PyRPL config file",
                                          filetypes=[("YAML files", "*.yml *.yaml")])
            if not config_file:
                raise ValueError("No config file selected")

        self.rp = Pyrpl(config=config_file, hostname=hostname)
        self.output_dir = output_dir

        self.scope = self.rp.rp.scope
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in2'
        self.scope.decimation = 8192
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

    def capture(self):
        """Grab a single capture of both channels."""
        self.scope.single()
        time.sleep(0.01)
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)
        if ch1.size == 0 or ch2.size == 0:
            raise ValueError("Scope returned empty data arrays")
        return ch1, ch2

    def plot_time(self, ch1, ch2, save_file=False, filename='scope_time.png'):
        t = np.arange(len(ch1)) / self.sample_rate
        plt.figure(figsize=(10,4))
        plt.plot(t, ch1, label='CH1')
        plt.plot(t, ch2, label='CH2')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V or A)')
        plt.title('RedPitaya Scope Capture')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_file:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, filename))
            print(f"Saved {filename} in {self.output_dir}")
        else:
            plt.show()

    def plot_fft(self, ch1, ch2, save_file=False, filename='scope_fft.png'):
        N = len(ch1)
        window = np.hanning(N)
        freqs = np.fft.rfftfreq(N, 1 / self.sample_rate)
        fft_ch1 = np.fft.rfft(ch1 * window)
        fft_ch2 = np.fft.rfft(ch2 * window)
        psd_ch1 = (np.abs(fft_ch1) ** 2) / (self.sample_rate * np.sum(window**2) + 1e-20)
        psd_ch2 = (np.abs(fft_ch2) ** 2) / (self.sample_rate * np.sum(window**2) + 1e-20)

        plt.figure(figsize=(10,4))
        plt.semilogy(freqs, psd_ch1, label='CH1')
        plt.semilogy(freqs, psd_ch2, label='CH2')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('RedPitaya Scope FFT')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_file:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, filename))
            print(f"Saved {filename} in {self.output_dir}")
        else:
            plt.show()

    def run(self, show_fft=False, save_file=False):
        ch1, ch2 = self.capture()
        if show_fft:
            self.plot_fft(ch1, ch2, save_file=save_file)
        else:
            self.plot_time(ch1, ch2, save_file=save_file)


if __name__ == '__main__':
    rp_scope = RedPitayaScope()

    # Grab one capture and show time-domain plot
    rp_scope.run(show_fft=False, save_file=False)

    # Uncomment below to show FFT instead
    # rp_scope.run(show_fft=True, save_file=False)
