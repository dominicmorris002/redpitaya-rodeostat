"""
Created on 10/29/2025
@author: Dominic
RedPitaya Scope with automatic YAML creation
Captures out1 -> in1 and overlays signals
"""

import os
import time
import numpy as np
import yaml

from matplotlib import pyplot as plt
from pyrpl import Pyrpl


class RedPitayaScope:
    def __init__(self, hostname='rp-f073ce.local', output_dir='scope_data', yaml_file='scope_config.yml'):
        self.output_dir = output_dir
        self.yaml_file = yaml_file

        # Create YAML config automatically if missing
        self.create_yaml()

        # Connect to RedPitaya
        self.rp = Pyrpl(config=self.yaml_file)

        # Access hardware modules
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Basic scope setup
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 8192
        self.scope.average = True
        self.sample_rate = 125e6 / self.scope.decimation

        # Default output waveform
        self.test_freq = 1000
        self.test_amp = 0.5
        self.setup_output()

    def create_yaml(self):
        """Automatically create a basic YAML config if missing."""
        if not os.path.exists(self.yaml_file):
            config = {
                'redpitaya_hostname': 'rp-f073ce.local',
                'modules': ['scope', 'asg0'],
                'scope': {
                    'ch1_active': True,
                    'ch2_active': True,
                    'input1': 'in1',
                    'input2': 'out1',
                    'threshold': 0.0,
                    'hysteresis': 0.0,
                    'duration': 0.01,
                    'trigger_delay': 0.0,
                    'trigger_source': 'ch1_positive_edge',  # <-- fixed
                    'trigger_mode': 'normal',
                    'average': True,
                    'decimation': 8192
                },
                'asg0': {
                    'waveform': 'sin',
                    'frequency': 1000,
                    'amplitude': 0.5,
                    'offset': 0.0,
                    'output_direct': 'out1',
                    'trigger_source': 'immediately'
                }
            }
            with open(self.yaml_file, 'w') as f:
                yaml.dump(config, f)
            print(f"✅ Created full Scope YAML config: {self.yaml_file}")
        else:
            print(f"ℹ️ YAML already exists: {self.yaml_file}")

    def setup_output(self, freq=None, amp=None):
        """Set up a sine wave on out1."""
        if freq is not None:
            self.test_freq = freq
        if amp is not None:
            self.test_amp = amp

        self.asg.setup(
            waveform='sin',
            frequency=self.test_freq,
            amplitude=self.test_amp,
            offset=0.0,
            output_direct='out1',
            trigger_source='immediately'
        )
        print(f"🔊 Output configured: {self.test_freq} Hz, {self.test_amp} V")

    def capture(self):
        """Grab a single capture of both channels (in1 and out1)."""
        self.scope.single()  # arm scope

        # Poll until data is captured
        timeout = 2.0  # seconds
        dt = 0.01
        elapsed = 0
        while elapsed < timeout:
            ch1 = np.array(self.scope._data_ch1_current)
            ch2 = np.array(self.scope._data_ch2_current)
            if ch1.size > 0 and ch2.size > 0:
                return ch1, ch2
            time.sleep(dt)
            elapsed += dt

        print("⚠️ Acquisition timed out")
        return None, None

    def plot_time(self, ch_in, ch_out, save_file=False, filename='scope_time.png', time_window=None):
        """Plot IN1 and OUT1 signals. Zoom with time_window (seconds)."""
        t = np.arange(len(ch_in)) / self.sample_rate

        if time_window is not None:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in = ch_in[:max_samples]
            ch_out = ch_out[:max_samples]

        plt.figure(figsize=(10, 4))
        plt.plot(t, ch_in, label='IN1')
        plt.plot(t, ch_out, label='OUT1')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.title('RedPitaya Scope Capture (IN1 vs OUT1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_file:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, filename))
        else:
            plt.show()

    def plot_fft(self, ch_in, ch_out, save_file=False, filename='scope_fft.png'):
        N = len(ch_in)
        freqs = np.fft.rfftfreq(N, 1 / self.sample_rate)
        fft_in = np.fft.rfft(ch_in * np.hanning(N))
        fft_out = np.fft.rfft(ch_out * np.hanning(N))
        psd_in = (np.abs(fft_in) ** 2) / (self.sample_rate * N)
        psd_out = (np.abs(fft_out) ** 2) / (self.sample_rate * N)

        plt.figure(figsize=(10, 4))
        plt.semilogy(freqs, psd_in, label='IN1')
        plt.semilogy(freqs, psd_out, label='OUT1')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('RedPitaya Scope FFT (IN1 vs OUT1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_file:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, filename))
        else:
            plt.show()

    def run(self, show_fft=False, save_file=False, freq=None, amp=None, time_window=None):
        if freq or amp:
            self.setup_output(freq=freq, amp=amp)
        ch_in, ch_out = self.capture()
        if ch_in is None or ch_out is None:
            print("⚠️ No data captured.")
            return
        if show_fft:
            self.plot_fft(ch_in, ch_out, save_file=save_file)
        else:
            self.plot_time(ch_in, ch_out, save_file=save_file, time_window=time_window)


if __name__ == '__main__':
    rp_scope = RedPitayaScope()

    # Show zoomed-in time-domain overlay (first 5 ms)
    rp_scope.run(show_fft=False, save_file=False, time_window=0.005)

    # Show FFT (optional)
    # rp_scope.run(show_fft=True, save_file=False)

