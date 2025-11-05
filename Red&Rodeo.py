"""
Rodeostat + RedPitaya Simultaneous Acquisition
Dominic Morris

Collects, saves, and plots voltage/current from Rodeostat while capturing RedPitaya scope data.
Stops automatically after RUN_TIME_SEC.
"""

import os
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
import traceback
import yaml
from pyrpl import Pyrpl
from potentiostat import Potentiostat

# ------------------------- User Parameters -------------------------
COM_PORT = 'COM5'           # Rodeostat COM port
RUN_TIME_SEC = 30           # Total runtime in seconds
DATAFILE = 'data.txt'       # Rodeostat data file
MODE = 'CV'                 # Options: 'DC', 'RAMP', 'CV'
CURR_RANGE = '1000uA'
SAMPLE_RATE = 1000.0        # Hz
QUIET_TIME = 0
QUIET_VALUE = 0.0
V_START = 1
V_END = 0.8
DC_RUNTIME = 30             # seconds
VOLT_MIN = 0.5
VOLT_MAX = 1.0
VOLT_PER_SEC = 0.1
NUM_CYCLES = 1
SHUNT_R = 1000              # Ohms for converting voltage to current in RedPitaya

# RedPitaya waveform settings
HOSTNAME = 'rp-f073ce.local'
YAML_FILE = 'scope_config.yml'
WAVEFORM_FREQ = 1000        # Hz
WAVEFORM_AMP = 0.5          # V
WAVEFORM_OFFSET = 0.0       # V
TIME_WINDOW = 0.005         # seconds for plotting

# ------------------------- Rodeostat Setup -------------------------
rodeostat_data = {'t': [], 'volt': [], 'curr': []}

def setup_rodeostat():
    ports = serial.tools.list_ports.comports()
    print("Available COM ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")

    # Use COM_PORT directly
    port = COM_PORT
    print("Using port:", port)
    dev = Potentiostat(port)

    # Patch for unknown firmware
    try:
        _ = dev.get_all_curr_range()
    except KeyError:
        print("Unknown firmware. Using fallback ranges.")
        dev.hw_variant = 'manual_patch'
        dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

    # Convert mode to cyclic parameters
    if MODE.upper() == 'CV':
        volt_min = VOLT_MIN
        volt_max = VOLT_MAX
        num_cycles = NUM_CYCLES
    elif MODE.upper() == 'DC':
        volt_min = V_START
        volt_max = V_START
        period_ms = 1000
        num_cycles = max(1, int(DC_RUNTIME * 1000 / period_ms))
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

    try:
        dev.set_curr_range(CURR_RANGE)
        dev.set_sample_rate(SAMPLE_RATE)
        dev.set_param('cyclic', test_param)
    except Exception as e:
        print("Error configuring Rodeostat:", e)
        traceback.print_exc()
        raise SystemExit

    return dev

def run_rodeostat(dev, stop_event):
    print(f"Running {MODE.upper()} test")
    try:
        t, volt, curr = dev.run_test('cyclic', display='data', filename=DATAFILE)
        idx = 0
        while not stop_event.is_set() and idx < len(t):
            rodeostat_data['t'].append(t[idx])
            rodeostat_data['volt'].append(volt[idx])
            rodeostat_data['curr'].append(curr[idx])
            idx += 1
            time.sleep(0.001)
    except Exception as e:
        print("Error running Rodeostat:", e)
        traceback.print_exc()

# ------------------------- RedPitaya Setup -------------------------
class RedPitayaScope:
    def __init__(self):
        self.create_yaml()
        self.rp = Pyrpl(modules=['scope', 'asg0'], config=YAML_FILE)
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.duration = 0.01
        self.scope.average = False
        self.scope.trigger_mode = 'auto'
        self.scope.setup()

        self.sample_rate = 125e6 / self.scope.decimation
        self.setup_output()

    def create_yaml(self):
        if not os.path.exists(YAML_FILE):
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
            with open(YAML_FILE, 'w') as f:
                yaml.dump(config, f)

    def setup_output(self, freq=None, amp=None, offset=None):
        freq = freq or WAVEFORM_FREQ
        amp = amp or WAVEFORM_AMP
        offset = offset or WAVEFORM_OFFSET
        self.asg.setup(
            waveform='sin',
            frequency=freq,
            amplitude=amp,
            offset=offset,
            output_direct='out1',
            trigger_source='immediately'
        )

    def capture(self, timeout=1.0):
        try:
            self.scope.single()
            dt = 0.01
            elapsed = 0
            while elapsed < timeout:
                ch1 = np.array(self.scope._data_ch1_current)
                ch2 = np.array(self.scope._data_ch2_current)
                if ch1.size > 0 and ch2.size > 0:
                    return ch1, ch2
                time.sleep(dt)
                elapsed += dt
        except Exception as e:
            print("Error during RedPitaya capture:", e)
        return None, None

# ------------------------- Main -------------------------
if __name__ == '__main__':
    stop_event = threading.Event()

    # Start devices
    rodeostat_dev = setup_rodeostat()
    rp_scope = RedPitayaScope()

    rodeostat_thread = threading.Thread(target=run_rodeostat, args=(rodeostat_dev, stop_event))
    rodeostat_thread.start()

    plt.ion()
    fig, ax = plt.subplots(2,1, figsize=(12,6))

    start_time = time.time()
    try:
        while time.time() - start_time < RUN_TIME_SEC:
            # Rodeostat plotting
            if rodeostat_data['t']:
                ax[0].clear()
                ax[0].plot(rodeostat_data['t'], rodeostat_data['volt'], label='Voltage')
                ax[0].plot(rodeostat_data['t'], rodeostat_data['curr'], label='Current')
                ax[0].set_ylabel('V / uA')
                ax[0].set_title('Rodeostat')
                ax[0].grid(True)
                ax[0].legend()

            # RedPitaya plotting
            ch_in, ch_out = rp_scope.capture(timeout=0.05)
            if ch_in is not None and ch_out is not None:
                t_rp = np.arange(len(ch_in)) / rp_scope.sample_rate
                current_in = ch_in / SHUNT_R * 1e6
                N = len(ch_in)
                fft_in = np.fft.rfft(ch_in * np.hanning(N))
                fft_out = np.fft.rfft(ch_out * np.hanning(N))
                peak_idx = np.argmax(np.abs(fft_in))
                phase_deg = np.degrees(np.angle(fft_out[peak_idx]) - np.angle(fft_in[peak_idx]))

                ax[1].clear()
                ax[1].plot(t_rp, ch_in, label='IN1 Voltage')
                ax[1].plot(t_rp, current_in, label='IN1 Current (uA)')
                ax[1].plot(t_rp, ch_out, label='OUT1 Voltage')
                ax[1].set_ylabel('V / uA')
                ax[1].set_xlabel('Time (s)')
                ax[1].set_title(f'RedPitaya — Phase: {phase_deg:.1f}°')
                ax[1].grid(True)
                ax[1].legend()

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Acquisition stopped by user.")
    finally:
        stop_event.set()
        rodeostat_thread.join()
        plt.ioff()
        plt.show()
