"""
Red&Rodeo ACCV - Synchronized AC Cyclic Voltammetry
Dominic Morris

Combines Red Pitaya (AC measurements) with Rodeostat (CV measurements)
to run synchronized AC cyclic voltammetry experiments.
Collects AC waveform data during CV sweep.

Have A Great Day!! :)
-Dominic
"""

import os
import time
import numpy as np
from matplotlib import pyplot as plt
from pyrpl import Pyrpl
from potentiostat import Potentiostat
import serial.tools.list_ports
import traceback
from datetime import datetime
import threading
import yaml

# ------------------------- Red Pitaya Parameters -------------------------
RP_HOSTNAME = 'rp-f073ce.local'  # Red Pitaya network address
RP_OUTPUT_DIR = 'accv_data'
RP_YAML_FILE = 'red_rodeo_config.yml'

# AC signal settings (Red Pitaya)
AC_FREQ = 1000  # Hz
AC_AMP = 0.05  # V amplitude
AC_OFFSET = 0.0  # DC offset
SHUNT_RESISTOR = 10000  # Ohms for current calculation

# ------------------------- Rodeostat Parameters -------------------------
CV_MODE = 'CV'
CURR_RANGE = '1000uA'
SAMPLE_RATE = 100.0
QUIET_TIME = 0
QUIET_VALUE = 0.0

VOLT_MIN = 0.5
VOLT_MAX = 1.0
VOLT_PER_SEC = 0.05
NUM_CYCLES = 1

V_START = 0.8
V_END = 1.0
DC_RUNTIME = 30


# ----------------------------------------------------------------------

class RedPitayaACMeasurement:
    """Handles Red Pitaya AC waveform measurements"""

    def __init__(self, hostname=RP_HOSTNAME, yaml_file=RP_YAML_FILE):
        self.yaml_file = yaml_file
        self.shunt_resistor = SHUNT_RESISTOR
        self.is_running = False
        self.data_buffer = []
        self.lock = threading.Lock()

        # Create YAML config
        self.create_yaml()

        # Connect to Red Pitaya
        print("Connecting to Red Pitaya...")
        self.rp = Pyrpl(config=self.yaml_file, gui=False, hostname=hostname)

        # Direct access to scope and ASG
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Configure ASG first (output signal)
        print("Configuring AC output signal...")
        self.asg.setup(
            waveform='sin',
            frequency=AC_FREQ,
            amplitude=AC_AMP,
            offset=AC_OFFSET,
            output_direct='out1',
            trigger_source='immediately'
        )
        self.asg.output_direct = 'out1'  # Ensure output is enabled

        # Configure Scope for continuous acquisition
        print("Configuring oscilloscope...")
        self.scope.input1 = 'in1'  # Current measurement (through shunt)
        self.scope.input2 = 'out1'  # Voltage output (direct from ASG)
        self.scope.duration = 0.01  # 10ms capture window
        self.scope.decimation = 64  # Faster sampling
        self.scope.trigger_source = 'immediately'  # Continuous trigger
        self.scope.trigger_delay = 0
        self.scope.ch1_active = True
        self.scope.ch2_active = True
        self.scope.average = False  # No averaging for real-time

        # Calculate actual sample rate
        self.sample_rate = 125e6 / self.scope.decimation
        print(f"Sample rate: {self.sample_rate / 1e6:.2f} MHz ({self.sample_rate / 1e3:.1f} kHz)")
        print(f"Samples per waveform: {int(self.sample_rate / AC_FREQ)}")

        # Start continuous acquisition
        self.scope.running_state = 'running_continuous'

        print(f"Red Pitaya ready - AC Output: {AC_FREQ} Hz, {AC_AMP} V amplitude")

    def create_yaml(self):
        """Create minimal YAML config that pre-loads modules"""
        config = {
            'name': 'red_rodeo_accv',
            'redpitaya_hostname': RP_HOSTNAME,
            'gui_config': {
                'show_gui': False
            },
            'scope': {
                'input1': 'in1',
                'input2': 'out1',
                'duration': 0.01,
                'trigger_source': 'immediately',
                'decimation': 64,
                'ch1_active': True,
                'ch2_active': True,
                'running_state': 'running_continuous'
            },
            'asg0': {
                'frequency': AC_FREQ,
                'amplitude': AC_AMP,
                'offset': AC_OFFSET,
                'waveform': 'sin',
                'output_direct': 'out1',
                'trigger_source': 'immediately'
            }
        }
        with open(self.yaml_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created config: {self.yaml_file}")

    def capture(self):
        """Capture latest voltage from Red Pitaya scope"""
        try:
            # Correct way to access scope data in PyRPL - use the private data attributes
            ch1_data = np.array(self.scope._data_ch1)  # Current (through shunt)
            ch2_data = np.array(self.scope._data_ch2)  # Voltage (from ASG)

            if ch1_data.size > 0 and ch2_data.size > 0:
                return ch1_data, ch2_data
            return None, None
        except Exception as e:
            print(f"Red Pitaya capture error: {e}")
            return None, None

    def calculate_current(self, voltage):
        """Convert voltage across shunt to current"""
        return voltage / self.shunt_resistor

    def collect_data_continuously(self):
        """Continuously collect waveform data in background thread"""
        self.is_running = True
        print("Red Pitaya continuous acquisition started")

        capture_count = 0
        while self.is_running:
            ch1, ch2 = self.capture()
            if ch1 is not None and ch2 is not None:
                current = self.calculate_current(ch1)
                timestamp = time.time()

                with self.lock:
                    self.data_buffer.append({
                        'timestamp': timestamp,
                        'voltage': ch2.copy(),
                        'current': current.copy()
                    })
                    # Keep only last 100 captures to prevent memory overflow
                    if len(self.data_buffer) > 100:
                        self.data_buffer.pop(0)

                capture_count += 1
                if capture_count % 20 == 0:
                    print(f"Captured {capture_count} waveforms, buffer size: {len(self.data_buffer)}")

            time.sleep(0.01)  # 100 Hz capture rate for waveforms

        print(f"Red Pitaya acquisition stopped after {capture_count} captures")

    def get_data(self):
        """Thread-safe data retrieval"""
        with self.lock:
            return self.data_buffer.copy()

    def stop(self):
        """Stop continuous acquisition"""
        self.is_running = False
        time.sleep(0.1)  # Allow thread to finish

    def cleanup(self):
        """Clean shutdown"""
        self.stop()
        try:
            self.scope.running_state = 'stopped'
            self.asg.output_direct = 'off'
        except:
            pass


# ------------------------- Rodeostat CV -------------------------
class RodeostatCVMeasurement:
    def __init__(self, port):
        print(f"Connecting to Rodeostat on {port}...")
        self.dev = Potentiostat(port)
        try:
            _ = self.dev.get_all_curr_range()
        except KeyError:
            print("Unknown firmware patch - setting manual configuration")
            self.dev.hw_variant = 'manual_patch'
            self.dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']
        print("Rodeostat connected")

    def configure(self, mode, volt_min, volt_max, volt_per_sec, num_cycles,
                  v_start, v_end, dc_runtime, curr_range, sample_rate,
                  quiet_time, quiet_value):
        amplitude = (volt_max - volt_min) / 2
        offset = (volt_max + volt_min) / 2
        period_ms = int(1000 * 4 * amplitude / volt_per_sec) if amplitude != 0 else 1000

        test_param = {
            'quietValue': quiet_value,
            'quietTime': quiet_time,
            'amplitude': amplitude,
            'offset': offset,
            'period': period_ms,
            'numCycles': num_cycles,
            'shift': 0.0
        }

        self.dev.set_curr_range(curr_range)
        self.dev.set_sample_rate(sample_rate)
        self.dev.set_param('cyclic', test_param)

        print(f"CV configured: {volt_min}V to {volt_max}V @ {volt_per_sec}V/s, {num_cycles} cycles")
        print(f"Period: {period_ms}ms, Sample rate: {sample_rate}Hz")

    def run_test(self):
        print("Starting CV sweep...")
        t, volt, curr = self.dev.run_test('cyclic', display='data', filename=None)
        print(f"CV sweep complete - {len(t)} data points")
        return t, volt, curr


# ------------------------- ACCV Experiment -------------------------
class ACCVExperiment:
    def __init__(self):
        self.output_dir = RP_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.red_pitaya = None
        self.rodeostat = None
        self.ac_thread = None

    def setup_devices(self, rodeostat_port):
        # Red Pitaya setup
        print("\n=== Setting up Red Pitaya ===")
        self.red_pitaya = RedPitayaACMeasurement()

        # Rodeostat setup
        print("\n=== Setting up Rodeostat ===")
        self.rodeostat = RodeostatCVMeasurement(rodeostat_port)
        self.rodeostat.configure(
            CV_MODE, VOLT_MIN, VOLT_MAX, VOLT_PER_SEC, NUM_CYCLES,
            V_START, V_END, DC_RUNTIME, CURR_RANGE, SAMPLE_RATE,
            QUIET_TIME, QUIET_VALUE
        )

        print("\n=== Devices ready ===\n")

    def run_synchronized_test(self):
        """Run CV while recording Red Pitaya AC waveform"""
        print("Starting synchronized ACCV measurement...")

        # Start Red Pitaya continuous acquisition thread
        self.ac_thread = threading.Thread(target=self.red_pitaya.collect_data_continuously)
        self.ac_thread.daemon = True
        self.ac_thread.start()

        # Wait for stable acquisition
        print("Waiting for Red Pitaya to stabilize...")
        time.sleep(2.0)

        # Check if data is being collected
        initial_data = self.red_pitaya.get_data()
        print(f"Red Pitaya collecting data: {len(initial_data)} waveforms captured")

        # Run Rodeostat CV sweep
        print("\n--- Running Rodeostat CV ---")
        t_cv, volt_cv, curr_cv = self.rodeostat.run_test()
        print("--- CV complete ---\n")

        # Stop Red Pitaya and collect all data
        print("Stopping Red Pitaya acquisition...")
        self.red_pitaya.stop()
        if self.ac_thread.is_alive():
            self.ac_thread.join(timeout=3)

        rp_data = self.red_pitaya.get_data()
        print(f"Red Pitaya data collected: {len(rp_data)} total waveforms")

        return t_cv, volt_cv, curr_cv, rp_data

    def save_data(self, t_cv, volt_cv, curr_cv, rp_data):
        # Save CV data
        cv_filename = os.path.join(self.output_dir, f'cv_{self.timestamp}.csv')
        np.savetxt(cv_filename, np.column_stack((t_cv, volt_cv, curr_cv)),
                   delimiter=',', header='Time(s),Voltage(V),Current(uA)', comments='')
        print(f"CV data saved to {cv_filename}")

        # Save Red Pitaya waveforms (last few captures)
        if rp_data:
            # Save last capture
            last = rp_data[-1]
            rp_filename = os.path.join(self.output_dir, f'redpitaya_last_{self.timestamp}.csv')
            np.savetxt(rp_filename, np.column_stack((last['voltage'], last['current'])),
                       delimiter=',', header='Voltage(V),Current(A)', comments='')
            print(f"Red Pitaya last waveform saved to {rp_filename}")

            # Save all timestamps
            timestamps = [d['timestamp'] for d in rp_data]
            ts_filename = os.path.join(self.output_dir, f'redpitaya_timestamps_{self.timestamp}.csv')
            np.savetxt(ts_filename, timestamps, delimiter=',',
                       header='Timestamp', comments='')
            print(f"Red Pitaya timestamps saved to {ts_filename}")

    def plot_results(self, t_cv, volt_cv, curr_cv, rp_data):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'ACCV Results - {self.timestamp}', fontsize=14)

        # Rodeostat CV plots
        axes[0, 0].plot(t_cv, volt_cv, color='tab:blue', linewidth=1.5)
        axes[0, 0].set_title('Rodeostat: Voltage vs Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Voltage (V)')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(t_cv, curr_cv, color='tab:red', linewidth=1.5)
        axes[0, 1].set_title('Rodeostat: Current vs Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Current (µA)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(volt_cv, curr_cv, color='tab:purple', linewidth=1.5)
        axes[0, 2].set_title('Rodeostat: CV Curve')
        axes[0, 2].set_xlabel('Voltage (V)')
        axes[0, 2].set_ylabel('Current (µA)')
        axes[0, 2].grid(True, alpha=0.3)

        # Red Pitaya AC waveform plots (last measurement)
        if rp_data:
            last = rp_data[-1]
            t_rp = np.arange(len(last['voltage'])) / self.red_pitaya.sample_rate * 1000  # ms

            axes[1, 0].plot(t_rp, last['voltage'], color='tab:blue', linewidth=1)
            axes[1, 0].set_title('Red Pitaya: AC Voltage Waveform')
            axes[1, 0].set_xlabel('Time (ms)')
            axes[1, 0].set_ylabel('Voltage (V)')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(t_rp, last['current'] * 1e6, color='tab:red', linewidth=1)
            axes[1, 1].set_title('Red Pitaya: AC Current Waveform')
            axes[1, 1].set_xlabel('Time (ms)')
            axes[1, 1].set_ylabel('Current (µA)')
            axes[1, 1].grid(True, alpha=0.3)

            axes[1, 2].plot(last['voltage'], last['current'] * 1e6,
                            color='tab:purple', linewidth=1, alpha=0.6)
            axes[1, 2].set_title('Red Pitaya: AC I-V Curve')
            axes[1, 2].set_xlabel('Voltage (V)')
            axes[1, 2].set_ylabel('Current (µA)')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_filename = os.path.join(self.output_dir, f'accv_plot_{self.timestamp}.png')
        plt.savefig(plot_filename, dpi=150)
        print(f"Plot saved to {plot_filename}")
        plt.show()

    def cleanup(self):
        """Clean shutdown of devices"""
        if self.red_pitaya:
            self.red_pitaya.cleanup()


# ------------------------- Main -------------------------
def main():
    # Find Rodeostat port
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise SystemExit("No serial ports found.")

    print("Available COM ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")

    choice = int(input("Select Rodeostat port number: "))
    rodeostat_port = ports[choice].device

    exp = None
    try:
        exp = ACCVExperiment()
        exp.setup_devices(rodeostat_port)

        input("\nPress Enter to start synchronized ACCV measurement...")

        t_cv, volt_cv, curr_cv, rp_data = exp.run_synchronized_test()
        exp.save_data(t_cv, volt_cv, curr_cv, rp_data)
        exp.plot_results(t_cv, volt_cv, curr_cv, rp_data)

        print("\n=== Experiment complete! ===")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if exp:
            exp.cleanup()
        print("Cleanup complete")


if __name__ == '__main__':
    main()
