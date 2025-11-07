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

# Shunt resistor for current calculation
SHUNT_RESISTOR = 10000  # Ohms (10kΩ) - adjustable

# Acquisition settings
RUN_TIME = 25  # Seconds to run acquisition before stopping and saving


# ----------------------------------------------------------------------

class RedPitayaScope:
    def __init__(self, hostname=HOSTNAME, output_dir=OUTPUT_DIR, yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file
        self.shunt_resistor = SHUNT_RESISTOR

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Ensure YAML exists
        self.create_yaml()

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
                    'trigger_source': 'immediately',
                    'running_state': 'running_continuous',
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

    def set_shunt_resistor(self, resistance):
        """Update shunt resistor value (in Ohms)"""
        self.shunt_resistor = resistance
        print(f"🔧 Shunt resistor set to: {resistance} Ω ({resistance / 1000:.1f} kΩ)")

    def calculate_current(self, voltage):
        """Calculate current from voltage across shunt resistor using Ohm's law"""
        # I = V / R
        return voltage / self.shunt_resistor

    def capture(self):
        """Capture current data in continuous mode"""
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

    def calculate_phase_difference(self, signal1, signal2):
        """Calculate phase difference between two signals using FFT"""
        N = len(signal1)
        fft1 = np.fft.rfft(signal1 * np.hanning(N))
        fft2 = np.fft.rfft(signal2 * np.hanning(N))
        peak_idx = np.argmax(np.abs(fft1))
        phase_diff_rad = np.angle(fft2[peak_idx]) - np.angle(fft1[peak_idx])
        phase_diff_deg = np.degrees(phase_diff_rad)
        return phase_diff_deg

    def plot_all_signals(self, ch_in1, ch_out1, axes, ax4_twin, time_window=TIME_WINDOW):
        """Plot all 4 graphs: OUT1, IN1, Current, and Phase overlay"""
        t = np.arange(len(ch_in1)) / self.sample_rate

        # Limit to time window
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in1 = ch_in1[:max_samples]
            ch_out1 = ch_out1[:max_samples]

        # Calculate current from IN1 voltage
        current = self.calculate_current(ch_in1)

        # Calculate phase differences
        phase_v_i = self.calculate_phase_difference(ch_in1, current)

        # Clear all axes
        for ax in axes[:3]:  # Only clear first 3 axes
            ax.clear()

        # Clear the 4th axis and its twin
        axes[3].clear()
        ax4_twin.clear()

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

        # Plot 3: Calculated Current from IN1
        axes[2].plot(t, current * 1000, color='tab:green', linewidth=1.5)  # Convert to mA
        axes[2].set_ylabel('Current (mA)')
        axes[2].set_title(f'IN1 Current (Shunt: {self.shunt_resistor / 1000:.1f} kΩ)')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Phase overlay - IN1 Voltage and Current
        line1 = axes[3].plot(t, ch_in1, color='tab:blue', linewidth=1.5, label='IN1 Voltage', alpha=0.8)
        line2 = ax4_twin.plot(t, current * 1000, color='tab:green', linewidth=1.5, label='IN1 Current', alpha=0.8)

        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Voltage (V)', color='tab:blue')
        ax4_twin.set_ylabel('Current (mA)', color='tab:green')
        axes[3].tick_params(axis='y', labelcolor='tab:blue')
        ax4_twin.tick_params(axis='y', labelcolor='tab:green')
        axes[3].set_title(f'IN1 Voltage & Current Phase Overlay (Δφ = {phase_v_i:.1f}°)')
        axes[3].grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[3].legend(lines, labels, loc='upper right')

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

        # Calculate current
        current = self.calculate_current(ch_in1)
        current_ma = current * 1000  # Convert to mA

        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f'scope_data_{timestamp}.csv')

        # Prepare data
        data = np.column_stack((t, ch_out1, ch_in1, current_ma))
        header = 'Time(s),OUT1_Voltage(V),IN1_Voltage(V),IN1_Current(mA)'

        # Save to CSV
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        print(f"💾 Data saved to: {filename}")
        return filename

    def run_continuous(self, time_window=TIME_WINDOW, run_time=None):
        """Continuously acquire and plot scope data with all 4 graphs"""
        if run_time is None:
            print("Starting continuous capture with 4 plots...")
            print("Press Ctrl+C to stop.")
        else:
            print(f"Starting capture for {run_time} seconds...")

        print(f"Shunt resistor: {self.shunt_resistor / 1000:.1f} kΩ")

        # Ensure we're in continuous mode
        self.scope.running_state = 'running_continuous'

        plt.ion()
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        ax4_twin = axes[3].twinx()  # Create twin axis once

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

                # Store last valid capture
                last_ch_in1 = ch_in1
                last_ch_out1 = ch_out1

                self.plot_all_signals(ch_in1, ch_out1, axes, ax4_twin, time_window=time_window)
                time.sleep(0.05)  # Update rate ~20 Hz

        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")

        # Save data to CSV if we have valid data
        if last_ch_in1 is not None and last_ch_out1 is not None:
            self.save_to_csv(last_ch_in1, last_ch_out1, time_window=time_window)

        plt.ioff()
        plt.show()


if __name__ == '__main__':
    rp_scope = RedPitayaScope()

    # You can adjust the shunt resistor value here
    # rp_scope.set_shunt_resistor(5000)  # Change to 5kΩ if needed

    rp_scope.setup_output(freq=WAVEFORM_FREQ, amp=WAVEFORM_AMP, offset=WAVEFORM_OFFSET)

    # Run for set time (in seconds) - change RUN_TIME at top of file
    rp_scope.run_continuous(time_window=TIME_WINDOW, run_time=RUN_TIME)
