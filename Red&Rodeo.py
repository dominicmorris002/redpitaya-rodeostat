import os
import time
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyrpl import Pyrpl
from potentiostat import Potentiostat
import serial.tools.list_ports
import traceback

# ================== REDPITAYA SCOPE CLASS ==================
class RedPitayaScope:
    def __init__(self, hostname='rp-f073ce.local', output_dir='scope_data', yaml_file='scope_config.yml'):
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
        self.test_freq = 1000
        self.test_amp = 0.5
        self.setup_output()

    def create_yaml(self):
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
                    'trigger_source': 'ch1_positive_edge',
                    'trigger_mode': 'auto',
                    'average': False,
                    'decimation': 128
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
            print(f"✅ Created fixed YAML: {self.yaml_file}")
        else:
            print(f"ℹ️ YAML exists: {self.yaml_file}")

    def setup_output(self, freq=None, amp=None):
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
        print(f"🔊 Output: {self.test_freq} Hz, {self.test_amp} V")

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

    def plot_time(self, ch_in, ch_out, save_file=False, filename='scope_time.png', time_window=None):
        t = np.arange(len(ch_in)) / self.sample_rate
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in = ch_in[:max_samples]
            ch_out = ch_out[:max_samples]

        # Phase calculation
        N = len(ch_in)
        fft_in = np.fft.rfft(ch_in * np.hanning(N))
        fft_out = np.fft.rfft(ch_out * np.hanning(N))
        peak_idx = np.argmax(np.abs(fft_in))
        phase_diff_rad = np.angle(fft_out[peak_idx]) - np.angle(fft_in[peak_idx])
        phase_diff_deg = np.degrees(phase_diff_rad)

        plt.figure(figsize=(10, 4))
        plt.plot(t, ch_in, label='AC Input (IN1)', color='tab:blue')
        plt.plot(t, ch_out, label='AC Output (OUT1)', color='tab:orange')
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

    def run(self, show_fft=False, save_file=False, freq=None, amp=None, time_window=None):
        if freq or amp:
            self.setup_output(freq=freq, amp=amp)
        ch_in, ch_out = self.capture()
        if ch_in is None or ch_out is None:
            print("⚠️ No data captured")
            return
        if show_fft:
            # Optional FFT plot
            pass
        else:
            self.plot_time(ch_in, ch_out, save_file=save_file, time_window=time_window)

# ================== RODEOSTAT SETUP ==================
def setup_rodeostat():
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise SystemExit("No serial ports found. Connect your Rodeostat.")

    print("Available COM ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")

    # Automatically select COM5 if available
    choice = next((i for i, p in enumerate(ports) if p.device.upper() == 'COM5'), None)
    if choice is None:
        raise SystemExit("COM5 not found. Connect Rodeostat.")
    port = ports[choice].device
    print("Using port:", port)

    # Test parameters
    MODE = 'RAMP'  # Options: DC, RAMP, CV
    CURR_RANGE = '100uA'
    SAMPLE_RATE = 100.0
    QUIET_TIME = 0
    QUIET_VALUE = 0.0
    V_START = 0
    V_END = 0.5
    VOLT_MIN = -0.1
    VOLT_MAX = 1.0
    VOLT_PER_SEC = 0.05
    NUM_CYCLES = 1

    if MODE.upper() == 'CV':
        volt_min = VOLT_MIN
        volt_max = VOLT_MAX
        num_cycles = NUM_CYCLES
    elif MODE.upper() == 'DC':
        volt_min = V_START
        volt_max = V_START
        num_cycles = 1
    elif MODE.upper() == 'RAMP':
        volt_min = V_START
        volt_max = V_END
        num_cycles = 1
    else:
        raise ValueError("Invalid MODE")

    amplitude = (volt_max - volt_min) / 2
    offset = (volt_max + volt_min) / 2
    period_ms = int(1000 * 4 * amplitude / VOLT_PER_SEC) if amplitude != 0 else 1000
    shift = 0.0
    test_param = {
        'quietValue': QUIET_VALUE,
        'quietTime': QUIET_TIME,
        'amplitude': amplitude,
        'offset': offset,
        'period': period_ms,
        'numCycles': num_cycles,
        'shift': shift
    }

    dev = Potentiostat(port)
    try:
        _ = dev.get_all_curr_range()
    except KeyError:
        dev.hw_variant = 'manual_patch'
        dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

    try:
        dev.set_curr_range(CURR_RANGE)
        dev.set_sample_rate(SAMPLE_RATE)
        dev.set_param('cyclic', test_param)
    except Exception as e:
        print("Error configuring Rodeostat:", e)
        traceback.print_exc()
        raise SystemExit

    return dev, MODE

# ================== RUN COMBINED TEST ==================
def run_combined():
    # Setup Rodeostat
    rodeostat, MODE = setup_rodeostat()

    # Setup RedPitaya
    rp_scope = RedPitayaScope()
    rp_scope.setup_output(freq=1000, amp=0.5)

    # Start Rodeostat measurement
    print(f"Running {MODE} test on Rodeostat...")
    try:
        t, volt, curr = rodeostat.run_test('cyclic', display='data', filename='data.txt')
    except Exception as e:
        print("Error running Rodeostat:", e)
        traceback.print_exc()
        return

    print("Rodeostat measurement complete. Now capturing RedPitaya AC signals...")

    # Capture RedPitaya signals
    ch_in, ch_out = rp_scope.capture()
    if ch_in is None or ch_out is None:
        print("⚠️ No RedPitaya data captured")
        return

    # ================== PLOTS ==================
    # Rodeostat: Voltage vs Time
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, volt, label='DC Voltage (Rodeostat)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(t, curr, label='DC Current (Rodeostat)')
    plt.ylabel('Current (uA)')
    plt.xlabel('Time (s)')
    plt.grid(True)

    # Rodeostat: Current vs Voltage
    plt.figure(2)
    plt.plot(volt, curr, label=f'{MODE} Test')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title(f'Rodeostat {MODE} Test')
    plt.grid(True)

    # RedPitaya: AC signals with phase
    rp_scope.plot_time(ch_in, ch_out, save_file=False, time_window=0.005)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_combined()
