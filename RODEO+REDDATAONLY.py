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

# Acquisition settings
RUN_TIME = 25  # Seconds to run acquisition before stopping and saving

# ----------------------------------------------------------------------

class RedPitayaScope:
    def __init__(self, hostname=HOSTNAME, output_dir=OUTPUT_DIR, yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file

        # Connect to RedPitaya
        self.rp = Pyrpl(modules=['scope', 'asg0'], config=self.yaml_file)

        # Access modules
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Scope setup
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.duration = 0.01  # 10 ms window
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'

        self.sample_rate = 125e6 / self.scope.decimation

        # Default output waveform
        self.test_freq = WAVEFORM_FREQ
        self.test_amp = WAVEFORM_AMP
        self.test_offset = WAVEFORM_OFFSET
        self.setup_output()

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
        """Capture data in continuous mode"""
        try:
            ch1 = np.array(self.scope._data_ch1)
            ch2 = np.array(self.scope._data_ch2)

            if ch1.size > 0 and ch2.size > 0:
                return ch1, ch2
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Error during capture: {e}")
            return None, None

    def plot_all_signals(self, ch_in1, ch_out1, axes, time_window=TIME_WINDOW):
        """Plot OUT1 and IN1 voltages"""
        t = np.arange(len(ch_in1)) / self.sample_rate

        # Limit to time window
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in1 = ch_in1[:max_samples]
            ch_out1 = ch_out1[:max_samples]

        # Clear all axes
        for ax in axes:
            ax.clear()

        # Plot 1: OUT1 Voltage
        axes[0].plot(t, ch_out1, color='tab:orange', linewidth=1.5)
        axes[0].set_ylabel('Voltage (V)')
        axes[0].set_title('OUT1 Voltage')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: IN1 Voltage
        axes[1].plot(t, ch_in1, color='tab:blue', linewidth=1.5)
        axes[1].set_ylabel('Voltage (V)')
        axes[1].set_title('IN1 Voltage')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.001)

    def save_to_csv(self, ch_in1, ch_out1, time_window=TIME_WINDOW):
        """Save captured data to CSV file"""
        t = np.arange(len(ch_in1)) / self.sample_rate

        # Limit to time window
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in1 = ch_in1[:max_samples]
            ch_out1 = ch_out1[:max_samples]

        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'scope_data_{timestamp}.csv'

        # Prepare data
        data = np.column_stack((t, ch_out1, ch_in1))
        header = 'Time(s),OUT1_Voltage(V),IN1_Voltage(V)'

        # Save to CSV
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        print(f"💾 Data saved to: {filename}")
        return filename

    def run_continuous(self, time_window=TIME_WINDOW, run_time=None):
        """Continuously acquire and plot scope data"""
        if run_time is None:
            print("Starting continuous capture...")
            print("Press Ctrl+C to stop.")
        else:
            print(f"Starting capture for {run_time} seconds...")

        # Ensure we're in continuous mode
        self.scope.running_state = 'running_continuous'

        plt.ion()
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))  # Only 2 plots now

        start_time = time.time()
        last_ch_in1 = None
        last_ch_out1 = None

        try:
            while True:
                # Check if run_time has elapsed
                if run_time is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= run_time:
                        print(f"\n✅ Completed {run_time} second acquisition")
                        break

                ch_in1, ch_out1 = self.capture()
                if ch_in1 is None or ch_out1 is None:
                    time.sleep(0.05)
                    continue

                last_ch_in1 = ch_in1
                last_ch_out1 = ch_out1

                self.plot_all_signals(ch_in1, ch_out1, axes, time_window=time_window)
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")

        if last_ch_in1 is not None and last_ch_out1 is not None:
            self.save_to_csv(last_ch_in1, last_ch_out1, time_window=time_window)

        plt.ioff()
        plt.show()


if __name__ == '__main__':
    rp_scope = RedPitayaScope()
    rp_scope.setup_output(freq=WAVEFORM_FREQ, amp=WAVEFORM_AMP, offset=WAVEFORM_OFFSET)
    rp_scope.run_continuous(time_window=TIME_WINDOW, run_time=RUN_TIME)
