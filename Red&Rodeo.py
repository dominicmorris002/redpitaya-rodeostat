"""
Rodeostat + RedPitaya Simultaneous Acquisition
Dominic Morris

- Rodeostat: CV measurement
- RedPitaya: OUT1/IN1 capture, calculates current from IN1 via shunt resistor
- Both run simultaneously
- Plots update live
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
COM_PORT = 'COM5'
RUN_TIME_SEC = 30
MODE = 'CV'
CURR_RANGE = '1000uA'
SAMPLE_RATE = 1000.0
VOLT_MIN = 0.5
VOLT_MAX = 1.0
VOLT_PER_SEC = 0.1
NUM_CYCLES = 1
QUIET_TIME = 0
QUIET_VALUE = 0.0
SHUNT_R = 1000              # Ohms for RedPitaya current calculation

# RedPitaya waveform
HOSTNAME = 'rp-f073ce.local'
YAML_FILE = 'scope_config.yml'
WAVEFORM_FREQ = 1000
WAVEFORM_AMP = 0.5
WAVEFORM_OFFSET = 0.0
TIME_WINDOW = 0.005

# ------------------------- Rodeostat Setup -------------------------
rodeostat_data = {'t': [], 'volt': [], 'curr': []}

def setup_rodeostat():
    print("Using port:", COM_PORT)
    dev = Potentiostat(COM_PORT)

    # Patch unknown hardware
    try:
        _ = dev.get_all_curr_range()
    except KeyError:
        print("Unknown hardware variant. Using fallback ranges.")
        dev.hw_variant = 'manual_patch'
        dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

    dev.set_curr_range(CURR_RANGE)
    dev.set_sample_rate(SAMPLE_RATE)

    # Setup cyclic parameters
    volt_min = VOLT_MIN
    volt_max = VOLT_MAX
    amplitude = (volt_max - volt_min) / 2
    offset = (volt_max + volt_min) / 2
    period_ms = int(1000 * 4 * amplitude / VOLT_PER_SEC) if amplitude != 0 else 1000
    test_param = {
        'quietValue': QUIET_VALUE,
        'quietTime': QUIET_TIME,
        'amplitude': amplitude,
        'offset': offset,
        'period': period_ms,
        'numCycles': NUM_CYCLES,
        'shift': 0.0
    }
    dev.set_param('cyclic', test_param)
    return dev

def run_rodeostat(dev, stop_event):
    print("Starting Rodeostat test...")
    try:
        t, volt, curr = dev.run_test('cyclic', display='data', filename=None)
        idx = 0
        while not stop_event.is_set() and idx < len(t):
            rodeostat_data['t'].append(t[idx])
            rodeostat_data['volt'].append(volt[idx])
            rodeostat_data['curr'].append(curr[idx])
            idx += 1
            time.sleep(0.001)
    except Exception as e:
        print("Error running Rodeostat test:", e)
        traceback.print_exc()

# ------------------------- RedPitaya Setup -------------------------
class RedPitayaScope:
    def __init__(self):
        self.create_yaml()
        self.rp = Pyrpl(modules=['scope', 'asg0'], config=YAML_FILE)
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Scope setup
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.duration = TIME_WINDOW
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
                    'duration': TIME_WINDOW,
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

    def capture(self, timeout=0.05):
        try:
            self.scope.single()
            elapsed = 0.0
            dt = 0.01
            while elapsed < timeout:
                ch1 = np.array(self.scope._data_ch1_current)
                ch2 = np.array(self.scope._data_ch2_current)
                if ch1.size > 0 and ch2.size > 0:
                    current = ch1 / SHUNT_R * 1e6  # uA
                    return ch1, ch2, current
                time.sleep(dt)
                elapsed += dt
        except Exception as e:
            print("RedPitaya capture error:", e)
        return None, None, None

# ------------------------- Main -------------------------
if __name__ == '__main__':
    stop_event = threading.Event()

    # Devices
    rodeostat_dev = setup_rodeostat()
    rp_scope = RedPitayaScope()

    rodeostat_thread = threading.Thread(target=run_rodeostat, args=(rodeostat_dev, stop_event))
    rodeostat_thread.start()

    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    start_time = time.time()
    try:
        while time.time() - start_time < RUN_TIME_SEC:
            # Rodeostat plots
            if rodeostat_data['t']:
                ax[0].clear()
                ax[0].plot(rodeostat_data['t'], rodeostat_data['volt'], label='Voltage')
                ax[0].plot(rodeostat_data['t'], rodeostat_data['curr'], label='Current')
                ax[0].set_ylabel('V / uA')
                ax[0].set_title('Rodeostat')
                ax[0].legend()
                ax[0].grid(True)

            # RedPitaya plots
            ch_in, ch_out, current = rp_scope.capture(timeout=0.05)
            if ch_in is not None and ch_out is not None:
                t_vec = np.arange(len(ch_in)) / rp_scope.sample_rate
                ax[1].clear()
                ax[1].plot(t_vec, ch_out, label='OUT1 Voltage')
                ax[1].set_ylabel('V')
                ax[1].set_title('RedPitaya OUT1')
                ax[1].grid(True)
                ax[1].legend()

                ax[2].clear()
                ax[2].plot(t_vec, ch_in, label='IN1 Voltage')
                ax[2].plot(t_vec, current, label='IN1 Current (uA)')
                ax[2].set_ylabel('V / uA')
                ax[2].set_xlabel('Time (s)')
                ax[2].set_title('RedPitaya IN1 / Calculated Current')
                ax[2].grid(True)
                ax[2].legend()

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Acquisition stopped by user.")
    finally:
        stop_event.set()
        rodeostat_thread.join()
        plt.ioff()
        plt.show()
