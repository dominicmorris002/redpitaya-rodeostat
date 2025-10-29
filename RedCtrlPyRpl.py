import os
import time
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyrpl import Pyrpl

# ------------------------- Waveform Parameters -------------------------
HOSTNAME = 'rp-f073ce.local'
OUTPUT_DIR = 'scope_data'
YAML_FILE = 'scope_config.yml'

# Output waveform settings
WAVEFORM_FREQ = 1000       # Hz
WAVEFORM_AMP = 0.5         # V peak-to-peak
WAVEFORM_OFFSET = 0.0      # DC offset
TIME_WINDOW = 0.005        # Seconds for time-domain plot
SHOW_FFT = False           # True to plot FFT, False for time-domain
SAVE_FILE = False           # True to save plots, False to show interactively
# ----------------------------------------------------------------------

class RedPitayaScope:
    def __init__(self, hostname=HOSTNAME, output_dir=OUTPUT_DIR, yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file

        # Ensure YAML exists
        self.create_yaml()

        # Connect to RedPitaya
        self.rp = Pyrpl(config=self.yaml_file)

        # Access modules
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Scope setup
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.average = False
        self.scope.trigger_mode = 'auto'
        self.sample_rate = 125e6 / self.scope.decimation

        # Default output waveform
        self.test_freq = WAVEFORM_FREQ
        self.test_amp = WAVEFORM_AMP
        self.test_offset = WAVEFORM_OFFSET
        self.setup_output()

    def create_yaml(self):
        if not os.path.exists(self.yaml_file):
            config = {
                'redpitaya_hostname': HOSTNAME,
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
                    'trigger_source': 'ch1_positive_edge',
                    'trigger_mode': 'auto',
                    'average': False,
                    'decimation': 128
                },
                'asg0': {
                    'waveform': 'sin',
                    'frequency': WAVEFORM_FREQ,
                    'amplitude': WAVEFORM_AMP,
                    'offset': WAVEFORM_OFFSET,
                    'output_direct': 'out1',
                    'trigger_source': 'immediately'
                }
            }
            with open(self.yaml_file, 'w') as f:
                yaml.dump(config, f)
            print(f"✅ Created fixed YAML: {self.yaml_file}")
        else:
            print(f"ℹ️ YAML exists: {self.yaml_file}")

    def setup_output(self, freq=None, amp=None, offset=None):
        if freq is not None:
            self.test_freq = freq
        if amp is not None:
            self.test_amp = amp
        if offset is not None:
            self.test_offset = offset
        self.asg.setup(
            waveform='sin',
            frequency=self.test_freq,
            amplitude=self.test_amp,
            offset=self.test_offset,
            output_direct='out1',
            trigger_source='immediately'
        )
        print(f"🔊 Output: {self.test_freq} Hz, {self.test_amp} V, Offset: {self.test_offset} V")

    def capture(self):
        self.scope.single()
        timeout = 2.0
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

    def plot_time(self, ch_in, ch_out, save_file=SAVE_FILE, filename='scope_time.png', time_window=TIME_WINDOW):
        t = np.arange(len(ch_in)) / self.sample_rate
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in = ch_in[:max_samples]
            ch_out = ch_out[:max_samples]

        # Calculate phase difference at the dominant frequency
        N = len(ch_in)
        fft_in = np.fft.rfft(ch_in * np.hanning(N))
        fft_out = np.fft.rfft(ch_out * np.hanning(N))
        peak_idx = np.argmax(np.abs(fft_in))
        phase_diff_rad = np.angle(fft_out[peak_idx]) - np.angle(fft_in[peak_idx])
        phase_diff_deg = np.degrees(phase_diff_rad)

        plt.figure(figsize=(10, 4))
        plt.plot(t, ch_in, label='IN1', color='tab:blue')
        plt.plot(t, ch_out, label='OUT1', color='tab:orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'RedPitaya Scope Capture — Phase: {phase_diff_deg:.1f}°')
        plt.grid(True)
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_file:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, filename))
        else:
            plt.show()

    def plot_fft(self, ch_in, ch_out, save_file=SAVE_FILE, filename='scope_fft.png'):
        N = len(ch_in)
        freqs = np.fft.rfftfreq(N, 1 / self.sample_rate)
        fft_in = np.fft.rfft(ch_in * np.hanning(N))
        fft_out = np.fft.rfft(ch_out * np.hanning(N))
        psd_in = (np.abs(fft_in) ** 2) / (self.sample_rate * N)
        psd_out = (np.abs(fft_out) ** 2) / (self.sample_rate * N)
        plt.figure(figsize=(10, 4))
        plt.semilogy(freqs, psd_in, label='IN1', color='tab:blue')
        plt.semilogy(freqs, psd_out, label='OUT1', color='tab:orange')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('RedPitaya Scope FFT')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_file:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, filename))
        else:
            plt.show()

    def run(self, show_fft=SHOW_FFT, save_file=SAVE_FILE, freq=None, amp=None, offset=None, time_window=TIME_WINDOW):
        if freq or amp or offset:
            self.setup_output(freq=freq, amp=amp, offset=offset)
        ch_in, ch_out = self.capture()
        if ch_in is None or ch_out is None:
            print("⚠️ No data captured")
            return
        if show_fft:
            self.plot_fft(ch_in, ch_out, save_file=save_file)
        else:
            self.plot_time(ch_in, ch_out, save_file=save_file, time_window=time_window)


if __name__ == '__main__':
    rp_scope = RedPitayaScope()
    rp_scope.setup_output(freq=WAVEFORM_FREQ, amp=WAVEFORM_AMP, offset=WAVEFORM_OFFSET)
    rp_scope.run(show_fft=SHOW_FFT, save_file=SAVE_FILE, time_window=TIME_WINDOW)
