import os
import time
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyrpl import Pyrpl
from potentiostat import Potentiostat
import serial.tools.list_ports
import traceback

# ================== CONFIGURABLE PARAMETERS ==================

# ---------- RedPitaya AC waveform ----------
RP_HOSTNAME = 'rp-f073ce.local'
RP_OUTPUT_DIR = 'scope_data'
RP_YAML_FILE = 'scope_config.yml'
RP_FREQ = 1000            # Hz
RP_AMP = 0.5              # V peak-to-peak
RP_OFFSET = 0.0           # V
RP_TIME_WINDOW = 0.005    # seconds for plotting
RP_SHOW_FFT = False
RP_SAVE_FILE = False

# ---------- Rodeostat ----------
RODEO_COM = 'COM5'        # default COM port
RODEO_MODE = 'RAMP'       # DC / RAMP / CV
RODEO_CURR_RANGE = '100uA'
RODEO_SAMPLE_RATE = 100.0 # Hz
RODEO_QUIET_TIME = 0
RODEO_QUIET_VALUE = 0.0
RODEO_V_START = 0
RODEO_V_END = 0.5
RODEO_VOLT_MIN = -0.1
RODEO_VOLT_MAX = 1.0
RODEO_VOLT_PER_SEC = 0.05
RODEO_NUM_CYCLES = 1

# ================== REDPITAYA SCOPE CLASS ==================
class RedPitayaScope:
    def __init__(self, hostname=RP_HOSTNAME, output_dir=RP_OUTPUT_DIR, yaml_file=RP_YAML_FILE):
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
        self.test_freq = RP_FREQ
        self.test_amp = RP_AMP
        self.test_offset = RP_OFFSET
        self.setup_output()

    def create_yaml(self):
        if not os.path.exists(self.yaml_file):
            config = {
                'redpitaya_hostname': RP_HOSTNAME,
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
                    'frequency': RP_FREQ,
                    'amplitude': RP_AMP,
                    'offset': RP_OFFSET,
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

    def plot_time(self, ch_in, ch_out, save_file=RP_SAVE_FILE, filename='scope_time.png', time_window=RP_TIME_WINDOW):
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

    def run(self, show_fft=RP_SHOW_FFT, save_file=RP_SAVE_FILE, freq=None, amp=None, offset=None, time_window=RP_TIME_WINDOW):
        if freq or amp or offset:
            self.setup_output(freq=freq, amp=amp, offset=offset)
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

    # Auto-select specified COM port
    choice = next((i for i, p in enumerate(ports) if p.device.upper() == RODEO_COM.upper()), None)
    if choice is None:
        raise SystemExit(f"{RODEO_COM} not found. Connect Rodeostat.")
    port = ports[choice].device
    print("Using port:", port)

    # Configure waveform parameters
    if RODEO_MODE.upper() == 'CV':
        volt_min = RODEO_VOLT_MIN
        volt_max = RODEO_VOLT_MAX
        num_cycles = RODEO_NUM_CYCLES
    elif RODEO_MODE.upper() == 'DC':
        volt_min = RODEO_V_START
        volt_max = RODEO_V_START
        num_cycles = 1
    elif RODEO_MODE.upper() == 'RAMP':
        volt_min = RODEO_V_START
        volt_max = RODEO_V_END
        num_cycles = 1
    else:
        raise ValueError("Invalid RODEO_MODE")

    amplitude = (volt_max - volt_min) / 2
    offset = (volt_max + volt_min) / 2
    period_ms = int(1000 * 4 * amplitude / RODEO_VOLT_PER_SEC) if amplitude != 0 else 1000
    shift = 0.0
    test_param = {
        'quietValue': RODEO_QUIET_VALUE,
        'quietTime': RODEO_QUIET_TIME,
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
        dev.set_curr_range(RODEO_CURR_RANGE)
        dev.set_sample_rate(RODEO_SAMPLE_RATE)
        dev.set_param('cyclic', test_param)
    except Exception as e:
        print("Error configuring Rodeostat:", e)
        traceback.print_exc()
        raise SystemExit

    return dev, RODEO_MODE

# ================== RUN COMBINED TEST ==================
def run_combined():
    rodeostat, MODE = setup_rodeostat()

    rp_scope = RedPitayaScope()
    rp_scope.setup_output(freq=RP_FREQ, amp=RP_AMP, offset=RP_OFFSET)

    print(f"Running {MODE} test on Rodeostat...")
    try:
        t, volt, curr = rodeostat.run_test('cyclic', display='data', filename='data.txt')
    except Exception as e:
        print("Error running Rodeostat:", e)
        traceback.print_exc()
        return

    print("Rodeostat measurement complete. Capturing RedPitaya AC signals...")

    ch_in, ch_out = rp_scope.capture()
    if ch_in is None or ch_out is None:
        print("⚠️ No RedPitaya data captured")
        return

    # Rodeostat plots
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

    plt.figure(2)
    plt.plot(volt, curr, label=f'{MODE} Test')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title(f'Rodeostat {MODE} Test')
    plt.grid(True)

    # RedPitaya AC signals
    rp_scope.plot_time(ch_in, ch_out, save_file=RP_SAVE_FILE, time_window=RP_TIME_WINDOW)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_combined()
