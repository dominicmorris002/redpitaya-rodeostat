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

        self.create_yaml()
        self.rp = Pyrpl(config=self.yaml_file)
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.average = False
        self.scope.trigger_mode = 'auto'
        self.sample_rate = 125e6 / self.scope.decimation

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

    def run(self):
        return self.capture()

# ================== RODEOSTAT SETUP ==================
def setup_rodeostat():
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise SystemExit("No serial ports found. Connect your Rodeostat.")

    print("Available COM ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")

    choice = next((i for i, p in enumerate(ports) if p.device.upper() == RODEO_COM.upper()), None)
    if choice is None:
        raise SystemExit(f"{RODEO_COM} not found. Connect Rodeostat.")
    port = ports[choice].device
    print("Using port:", port)

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
    ch_in, ch_out = rp_scope.run()
    if ch_in is None or ch_out is None:
        print("⚠️ No RedPitaya data captured")
        return

    # ================== PLOT ALL IN ONE FIGURE ==================
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Rodeostat & RedPitaya Combined Measurement', fontsize=16)

    # Rodeostat: Voltage vs Time
    axs[0,0].plot(t, volt, color='tab:blue')
    axs[0,0].set_ylabel('Voltage (V)')
    axs[0,0].set_title('Rodeostat Voltage vs Time')
    axs[0,0].grid(True)

    # Rodeostat: Current vs Time
    axs[0,1].plot(t, curr, color='tab:orange')
    axs[0,1].set_ylabel('Current (uA)')
    axs[0,1].set_title('Rodeostat Current vs Time')
    axs[0,1].grid(True)

    # Rodeostat: Current vs Voltage
    axs[1,0].plot(volt, curr, color='tab:green')
    axs[1,0].set_xlabel('Voltage (V)')
    axs[1,0].set_ylabel('Current (uA)')
    axs[1,0].set_title('Rodeostat I-V Curve')
    axs[1,0].grid(True)

    # RedPitaya AC signals
    N = len(ch_in)
    fft_in = np.fft.rfft(ch_in * np.hanning(N))
    fft_out = np.fft.rfft(ch_out * np.hanning(N))
    peak_idx = np.argmax(np.abs(fft_in))
    phase_diff_rad = np.angle(fft_out[peak_idx]) - np.angle(fft_in[peak_idx])
    phase_diff_deg = np.degrees(phase_diff_rad)

    t_rp = np.arange(len(ch_in)) / rp_scope.sample_rate
    if RP_TIME_WINDOW:
        max_samples = int(RP_TIME_WINDOW * rp_scope.sample_rate)
        t_rp = t_rp[:max_samples]
        ch_in = ch_in[:max_samples]
        ch_out = ch_out[:max_samples]

    axs[1,1].plot(t_rp, ch_in, label='AC IN1', color='tab:blue')
    axs[1,1].plot(t_rp, ch_out, label='AC OUT1', color='tab:orange')
    axs[1,1].set_xlabel('Time (s)')
    axs[1,1].set_ylabel('Voltage (V)')
    axs[1,1].set_title(f'RedPitaya AC Signals — Phase: {phase_diff_deg:.1f}°')
    axs[1,1].legend()
    axs[1,1].grid(True)

    # Empty placeholders for aesthetics
    axs[2,0].axis('off')
    axs[2,1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    run_combined()
