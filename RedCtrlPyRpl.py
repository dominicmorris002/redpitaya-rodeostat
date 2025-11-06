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
WAVEFORM_FREQ = 1000  # Hz
WAVEFORM_AMP = 0.5  # V peak-to-peak
WAVEFORM_OFFSET = 0.0  # DC offset
TIME_WINDOW = 0.005  # Seconds for time-domain plot
SHOW_FFT = False  # True to plot FFT, False for time-domain
SAVE_FILE = False  # True to save plots, False to show interactively


# ----------------------------------------------------------------------

class RedPitayaScope:
    def __init__(self, hostname=HOSTNAME, output_dir=OUTPUT_DIR, yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file

        # Ensure YAML exists
        self.create_yaml()

        # Connect to RedPitaya
        self.rp = Pyrpl(modules=['scope', 'asg0'], config=self.yaml_file)

        # Access modules
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Scope setup - KEY CHANGES HERE
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.duration = 0.01  # 10 ms window
        self.scope.average = False
        self.scope.trigger_source = 'immediately'  # Changed from auto edge trigger
        self.scope.running_state = 'running_continuous'  # Force continuous mode

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
                    'trigger_source': 'immediately',  # Changed from edge trigger
                    'running_state': 'running_continuous',  # Added this
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
            print(f"Created YAML: {self.yaml_file}")
        else:
            print(f"YAML exists: {self.yaml_file}")

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
        """Capture current data in continuous mode"""
        try:
            # Use the internal data attributes that PyRPL provides
            ch1 = np.array(self.scope._data_ch1)
            ch2 = np.array(self.scope._data_ch2)

            if ch1.size > 0 and ch2.size > 0:
                return ch1, ch2
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Error during capture: {e}")
            return None, None

    def plot_time(self, ch_in, ch_out, ax=None, time_window=TIME_WINDOW):
        t = np.arange(len(ch_in)) / self.sample_rate
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in = ch_in[:max_samples]
            ch_out = ch_out[:max_samples]

        # Phase difference at dominant frequency
        N = len(ch_in)
        fft_in = np.fft.rfft(ch_in * np.hanning(N))
        fft_out = np.fft.rfft(ch_out * np.hanning(N))
        peak_idx = np.argmax(np.abs(fft_in))
        phase_diff_rad = np.angle(fft_out[peak_idx]) - np.angle(fft_in[peak_idx])
        phase_diff_deg = np.degrees(phase_diff_rad)

        if ax is None:
            plt.figure(figsize=(10, 4))
            plt.plot(t, ch_in, label='IN1', color='tab:blue')
            plt.plot(t, ch_out, label='OUT1', color='tab:orange')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.title(f'RedPitaya Scope Capture — Phase: {phase_diff_deg:.1f}°')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.pause(0.001)
        else:
            ax.clear()
            ax.plot(t, ch_in, label='IN1', color='tab:blue')
            ax.plot(t, ch_out, label='OUT1', color='tab:orange')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage (V)')
            ax.set_title(f'Phase: {phase_diff_deg:.1f}°')
            ax.grid(True)
            ax.legend()
            plt.pause(0.001)

    def run_continuous(self, time_window=TIME_WINDOW):
        """Continuously acquire and plot scope data"""
        print("Starting continuous capture. Press Ctrl+C to stop.")

        # Ensure we're in continuous mode
        self.scope.running_state = 'running_continuous'

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))

        try:
            while True:
                ch_in, ch_out = self.capture()
                if ch_in is None or ch_out is None:
                    time.sleep(0.05)
                    continue

                self.plot_time(ch_in, ch_out, ax=ax, time_window=time_window)
                time.sleep(0.05)  # Update rate ~20 Hz

        except KeyboardInterrupt:
            print("\nStopped continuous capture")
            plt.ioff()
            plt.show()


if __name__ == '__main__':
    rp_scope = RedPitayaScope()
    rp_scope.setup_output(freq=WAVEFORM_FREQ, amp=WAVEFORM_AMP, offset=WAVEFORM_OFFSET)
    rp_scope.run_continuous(time_window=TIME_WINDOW)
