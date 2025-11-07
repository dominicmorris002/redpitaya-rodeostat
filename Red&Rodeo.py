"""
Red&Rodeo ACCV - Synchronized AC Cyclic Voltammetry
Dominic Morris

Combines Red Pitaya (AC measurements) with Rodeostat (CV measurements)
to run synchronized AC cyclic voltammetry experiments.

Have A Great Day!! :)
-Dominic
"""

import os
import time
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyrpl import Pyrpl
from potentiostat import Potentiostat
import serial.tools.list_ports
import traceback
from datetime import datetime
import threading

# ------------------------- Red Pitaya Parameters -------------------------
RP_HOSTNAME = 'rp-f073ce.local'  # Red Pitaya network address
RP_OUTPUT_DIR = 'accv_data'
RP_YAML_FILE = 'red_rodeo_config.yml'

# AC signal settings (Red Pitaya)
AC_FREQ = 1000  # Hz (1 Hz - 62.5 MHz)
AC_AMP = 0.5  # V peak-to-peak (0 - 2 V, limited by output range ±1V)
AC_OFFSET = 0.0  # DC offset (-1 to +1 V)
TIME_WINDOW = 0.005  # Seconds for time-domain plot (0.001 - 1 s typical)
SHUNT_RESISTOR = 10000  # Ohms (10 - 100k typical for low current measurements)

# ------------------------- Rodeostat Parameters -------------------------
# CV settings (Rodeostat)
CV_MODE = 'CV'  # Options: 'DC', 'RAMP', 'CV'
CURR_RANGE = '1000uA'  # Options: '1uA', '10uA', '100uA', '1000uA'
SAMPLE_RATE = 1000.0  # Hz (10 - 1000 Hz typical)
QUIET_TIME = 0  # Seconds (0 - 10 s)
QUIET_VALUE = 0.0  # V (-1.5 to +1.5 V typical)

# CV sweep parameters
VOLT_MIN = 0.5  # V (-1.5 to +1.5 V, must be < VOLT_MAX)
VOLT_MAX = 1.0  # V (-1.5 to +1.5 V, must be > VOLT_MIN)
VOLT_PER_SEC = 0.2  # V/s (0.001 - 1.0 V/s typical)
NUM_CYCLES = 1  # Number of CV cycles (1 - 100)

# DC / Ramp settings (if needed)
V_START = 0.8  # V (-1.5 to +1.5 V)
V_END = 1.0  # V (-1.5 to +1.5 V, can be > or < V_START for ramp)
DC_RUNTIME = 30  # seconds (1 - 3600 s typical)

# ----------------------------------------------------------------------


class RedPitayaACMeasurement:
    """Handles Red Pitaya AC measurements"""

    def __init__(self, hostname=RP_HOSTNAME, yaml_file=RP_YAML_FILE):
        self.yaml_file = yaml_file
        self.shunt_resistor = SHUNT_RESISTOR
        self.is_running = False
        self.latest_data = None

        # Ensure YAML exists
        self.create_yaml()

        # Connect to RedPitaya
        print("Connecting to Red Pitaya...")
        self.rp = Pyrpl(config=self.yaml_file, gui=False, hostname=RP_HOSTNAME)
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Scope setup
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.duration = 0.01
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'

        self.sample_rate = 125e6 / self.scope.decimation
        print(f"Red Pitaya connected (Sample rate: {self.sample_rate/1e3:.1f} kHz)")

    def create_yaml(self):
        # Always recreate config to ensure correct hostname
        config = {
            'redpitaya_hostname': RP_HOSTNAME,
            'gui_config': {
                'show_gui': False
            },
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
                'frequency': AC_FREQ,
                'amplitude': AC_AMP,
                'offset': AC_OFFSET,
                'output_direct': 'out1',
                'trigger_source': 'immediately'
            }
        }
        with open(self.yaml_file, 'w') as f:
            yaml.dump(config, f)

    def setup_output(self, freq=AC_FREQ, amp=AC_AMP, offset=AC_OFFSET):
        self.asg.setup(
            waveform='sin',
            frequency=freq,
            amplitude=amp,
            offset=offset,
            output_direct='out1',
            trigger_source='immediately'
        )
        print(f"AC Output: {freq} Hz, {amp} V, Offset: {offset} V")

    def capture(self):
        """Capture current data"""
        try:
            ch1 = np.array(self.scope._data_ch1)
            ch2 = np.array(self.scope._data_ch2)
            if ch1.size > 0 and ch2.size > 0:
                return ch1, ch2
            return None, None
        except Exception as e:
            print(f"Red Pitaya capture error: {e}")
            return None, None

    def calculate_current(self, voltage):
        return voltage / self.shunt_resistor

    def calculate_phase_difference(self, signal1, signal2):
        N = len(signal1)
        fft1 = np.fft.rfft(signal1 * np.hanning(N))
        fft2 = np.fft.rfft(signal2 * np.hanning(N))
        peak_idx = np.argmax(np.abs(fft1))
        phase_diff_rad = np.angle(fft2[peak_idx]) - np.angle(fft1[peak_idx])
        return np.degrees(phase_diff_rad)

    def collect_data_continuously(self):
        """Continuously collect AC data in background thread"""
        self.is_running = True
        while self.is_running:
            ch_in1, ch_out1 = self.capture()
            if ch_in1 is not None and ch_out1 is not None:
                self.latest_data = (ch_in1, ch_out1, time.time())
            time.sleep(0.05)

    def stop(self):
        self.is_running = False


class RodeostatCVMeasurement:
    """Handles Rodeostat CV measurements"""

    def __init__(self, port):
        print(f"Connecting to Rodeostat on {port}...")
        self.dev = Potentiostat(port)

        # Patch for unknown firmware
        try:
            _ = self.dev.get_all_curr_range()
        except KeyError:
            print("Unknown firmware. Adding current range list manually.")
            self.dev.hw_variant = 'manual_patch'
            self.dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

        print("Rodeostat connected")

    def configure(self, mode, volt_min, volt_max, volt_per_sec, num_cycles,
                  v_start, v_end, dc_runtime, curr_range, sample_rate,
                  quiet_time, quiet_value):
        """Configure CV parameters"""

        if mode.upper() == 'CV':
            amplitude = (volt_max - volt_min) / 2
            offset = (volt_max + volt_min) / 2
            period_ms = int(1000 * 4 * amplitude / volt_per_sec) if amplitude != 0 else 1000
            shift = 0.0
            cycles = num_cycles

        elif mode.upper() == 'DC':
            amplitude = 0
            offset = v_start
            period_ms = 1000
            shift = 0.0
            cycles = max(1, int(dc_runtime * 1000 / period_ms))

        elif mode.upper() == 'RAMP':
            if v_start < v_end:
                volt_min = v_start
                volt_max = v_end
                shift = 0.0
            else:
                volt_min = v_end
                volt_max = v_start
                shift = 0.5
            amplitude = (volt_max - volt_min) / 2
            offset = (volt_max + volt_min) / 2
            period_ms = int(1000 * 4 * amplitude / volt_per_sec) if amplitude != 0 else 1000
            cycles = 1
        else:
            raise ValueError("Invalid MODE")

        test_param = {
            'quietValue': quiet_value,
            'quietTime': quiet_time,
            'amplitude': amplitude,
            'offset': offset,
            'period': period_ms,
            'numCycles': cycles,
            'shift': shift
        }

        self.dev.set_curr_range(curr_range)
        self.dev.set_sample_rate(sample_rate)
        self.dev.set_param('cyclic', test_param)

        print(f"CV configured: {volt_min}V to {volt_max}V, {num_cycles} cycles")

    def run_test(self):
        """Run the CV test"""
        print("Starting CV sweep...")
        t, volt, curr = self.dev.run_test('cyclic', display='data', filename=None)
        print("CV sweep complete")
        return t, volt, curr


class ACCVExperiment:
    """Main class for synchronized ACCV experiment"""

    def __init__(self):
        self.output_dir = RP_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize devices
        self.red_pitaya = None
        self.rodeostat = None
        self.ac_thread = None

    def setup_devices(self, rodeostat_port):
        """Initialize both devices"""
        # Setup Red Pitaya
        self.red_pitaya = RedPitayaACMeasurement()
        self.red_pitaya.setup_output(AC_FREQ, AC_AMP, AC_OFFSET)

        # Setup Rodeostat
        self.rodeostat = RodeostatCVMeasurement(rodeostat_port)
        self.rodeostat.configure(
            CV_MODE, VOLT_MIN, VOLT_MAX, VOLT_PER_SEC, NUM_CYCLES,
            V_START, V_END, DC_RUNTIME, CURR_RANGE, SAMPLE_RATE,
            QUIET_TIME, QUIET_VALUE
        )

    def run_synchronized_test(self):
        """Run synchronized ACCV measurement"""
        print("="*60)
        print("Starting Synchronized ACCV Experiment")
        print("="*60)

        # Start Red Pitaya AC measurement in background thread
        self.ac_thread = threading.Thread(target=self.red_pitaya.collect_data_continuously)
        self.ac_thread.daemon = True
        self.ac_thread.start()

        # Wait a moment for AC to stabilize
        time.sleep(0.5)

        # Run Rodeostat CV (this blocks until complete)
        t_cv, volt_cv, curr_cv = self.rodeostat.run_test()

        # Stop Red Pitaya collection
        self.red_pitaya.stop()
        self.ac_thread.join(timeout=1)

        # Get final AC data
        if self.red_pitaya.latest_data:
            ch_in1, ch_out1, _ = self.red_pitaya.latest_data
        else:
            ch_in1, ch_out1 = None, None

        return t_cv, volt_cv, curr_cv, ch_in1, ch_out1

    def save_data(self, t_cv, volt_cv, curr_cv, ch_in1, ch_out1):
        """Save all data to CSV files"""

        # Save CV data
        cv_filename = os.path.join(self.output_dir, f'accv_cv_{self.timestamp}.csv')
        cv_data = np.column_stack((t_cv, volt_cv, curr_cv))
        cv_header = 'Time(s),Voltage(V),Current(uA)'
        np.savetxt(cv_filename, cv_data, delimiter=',', header=cv_header, comments='')
        print(f"CV data saved to: {cv_filename}")

        # Save AC data if available
        if ch_in1 is not None and ch_out1 is not None:
            # Limit to time window
            max_samples = int(TIME_WINDOW * self.red_pitaya.sample_rate)
            t_ac = np.arange(min(len(ch_in1), max_samples)) / self.red_pitaya.sample_rate
            ch_in1 = ch_in1[:max_samples]
            ch_out1 = ch_out1[:max_samples]

            current_ac = self.red_pitaya.calculate_current(ch_in1) * 1000  # mA

            ac_filename = os.path.join(self.output_dir, f'accv_ac_{self.timestamp}.csv')
            ac_data = np.column_stack((t_ac, ch_out1, ch_in1, current_ac))
            ac_header = 'Time(s),OUT1_Voltage(V),IN1_Voltage(V),IN1_Current(mA)'
            np.savetxt(ac_filename, ac_data, delimiter=',', header=ac_header, comments='')
            print(f"AC data saved to: {ac_filename}")

    def plot_results(self, t_cv, volt_cv, curr_cv, ch_in1, ch_out1):
        """Plot comprehensive results"""

        fig = plt.figure(figsize=(14, 10))

        # CV plots (left column)
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(t_cv, volt_cv, linewidth=1.5, color='tab:blue')
        ax1.set_ylabel('CV Voltage (V)')
        ax1.set_title('Rodeostat CV - Voltage vs Time')
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 2, 3)
        ax2.plot(t_cv, curr_cv, linewidth=1.5, color='tab:red')
        ax2.set_ylabel('CV Current (μA)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Rodeostat CV - Current vs Time')
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(3, 2, 5)
        ax3.plot(volt_cv, curr_cv, linewidth=1.5, color='tab:purple')
        ax3.set_xlabel('Voltage (V)')
        ax3.set_ylabel('Current (μA)')
        ax3.set_title('Rodeostat CV - I-V Curve')
        ax3.grid(True, alpha=0.3)

        # AC plots (right column)
        if ch_in1 is not None and ch_out1 is not None:
            max_samples = int(TIME_WINDOW * self.red_pitaya.sample_rate)
            t_ac = np.arange(min(len(ch_in1), max_samples)) / self.red_pitaya.sample_rate
            ch_in1 = ch_in1[:max_samples]
            ch_out1 = ch_out1[:max_samples]
            current_ac = self.red_pitaya.calculate_current(ch_in1) * 1000  # mA
            phase = self.red_pitaya.calculate_phase_difference(ch_in1, current_ac)

            ax4 = plt.subplot(3, 2, 2)
            ax4.plot(t_ac, ch_out1, linewidth=1.5, color='tab:orange')
            ax4.set_ylabel('OUT1 Voltage (V)')
            ax4.set_title('Red Pitaya AC - Output Voltage')
            ax4.grid(True, alpha=0.3)

            ax5 = plt.subplot(3, 2, 4)
            ax5.plot(t_ac, ch_in1, linewidth=1.5, color='tab:blue')
            ax5.set_ylabel('IN1 Voltage (V)')
            ax5.set_title('Red Pitaya AC - Input Voltage')
            ax5.grid(True, alpha=0.3)

            ax6 = plt.subplot(3, 2, 6)
            ax6_twin = ax6.twinx()
            line1 = ax6.plot(t_ac, ch_in1, color='tab:blue', linewidth=1.5, label='Voltage', alpha=0.8)
            line2 = ax6_twin.plot(t_ac, current_ac, color='tab:green', linewidth=1.5, label='Current', alpha=0.8)
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Voltage (V)', color='tab:blue')
            ax6_twin.set_ylabel('Current (mA)', color='tab:green')
            ax6.tick_params(axis='y', labelcolor='tab:blue')
            ax6_twin.tick_params(axis='y', labelcolor='tab:green')
            ax6.set_title(f'Red Pitaya AC - Phase Overlay (Δφ = {phase:.1f}°)')
            ax6.grid(True, alpha=0.3)
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax6.legend(lines, labels, loc='upper right')

        plt.suptitle(f'ACCV Experiment Results - {self.timestamp}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""

    # Select Rodeostat port
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise SystemExit("No serial ports found. Connect your Rodeostat and try again.")

    print("Available COM ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")
    choice = int(input("Select Rodeostat port number: "))
    rodeostat_port = ports[choice].device

    try:
        # Create experiment
        experiment = ACCVExperiment()

        # Setup devices
        experiment.setup_devices(rodeostat_port)

        # Run synchronized test
        t_cv, volt_cv, curr_cv, ch_in1, ch_out1 = experiment.run_synchronized_test()

        # Save data
        experiment.save_data(t_cv, volt_cv, curr_cv, ch_in1, ch_out1)

        # Plot results
        experiment.plot_results(t_cv, volt_cv, curr_cv, ch_in1, ch_out1)

        print("\n" + "="*60)
        print("ACCV Experiment Complete!")
        print("="*60)

    except Exception as e:
        print(f"\nError during experiment: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
