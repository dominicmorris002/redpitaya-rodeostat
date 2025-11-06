"""
AC Cyclic Voltammetry - Step Measurements
Dominic Morris

Takes RedPitaya AC measurements at each voltage step during Rodeostat CV
No live plotting - all plots shown at end
"""

import os
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import serial.tools.list_ports
import traceback
from pyrpl import Pyrpl
from potentiostat import Potentiostat

# ------------------------- User Parameters -------------------------
COM_PORT = None
MODE = 'RAMP'
CURR_RANGE = '1000uA'
SAMPLE_RATE = 1000.0
QUIET_TIME = 0
QUIET_VALUE = 0.0

# RAMP/DC settings
V_START = 0.8
V_END = 1.0
DC_RUNTIME = 30

# CV settings
VOLT_MIN = 0.5
VOLT_MAX = 1.0
VOLT_PER_SEC = 0.2
NUM_CYCLES = 1

# RedPitaya settings
HOSTNAME = 'rp-f073ce.local'
YAML_FILE = 'scope_config.yml'
WAVEFORM_FREQ = 1000
WAVEFORM_AMP = 0.5
WAVEFORM_OFFSET = 0.0
TIME_WINDOW = 0.005
SHUNT_R = 10_000

# How many AC measurements to take per voltage step
AC_MEASUREMENTS_PER_STEP = 5


# ------------------------- Rodeostat Setup -------------------------
def setup_rodeostat():
    global COM_PORT
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise SystemExit("No serial ports found.")

    print("Available COM ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")

    if COM_PORT is None:
        choice = int(input("Select port number: "))
        COM_PORT = ports[choice].device

    print("Using port:", COM_PORT)

    dev = Potentiostat(COM_PORT)

    try:
        _ = dev.get_all_curr_range()
    except KeyError:
        print("Unknown firmware. Adding current range list manually.")
        dev.hw_variant = 'manual_patch'
        dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

    dev.set_curr_range(CURR_RANGE)
    dev.set_sample_rate(SAMPLE_RATE)

    # Convert mode to cyclic parameters
    if MODE.upper() == 'CV':
        volt_min = VOLT_MIN
        volt_max = VOLT_MAX
        num_cycles = NUM_CYCLES
    elif MODE.upper() == 'DC':
        volt_min = V_START
        volt_max = V_START
        amplitude = 0
        offset = V_START
        period_ms = 1000
        num_cycles = max(1, int(DC_RUNTIME * 1000 / period_ms))
    elif MODE.upper() == 'RAMP':
        volt_min = V_START
        volt_max = V_END
        num_cycles = 1
    else:
        raise ValueError("Invalid MODE, choose 'DC', 'RAMP', or 'CV'")

    if MODE.upper() != 'DC':
        amplitude = (volt_max - volt_min) / 2
        offset = (volt_max + volt_min) / 2
        period_ms = int(1000 * 4 * amplitude / VOLT_PER_SEC) if amplitude != 0 else 1000

    test_param = {
        'quietValue': QUIET_VALUE,
        'quietTime': QUIET_TIME,
        'amplitude': amplitude,
        'offset': offset,
        'period': period_ms,
        'numCycles': num_cycles,
        'shift': 0.0
    }

    dev.set_param('cyclic', test_param)
    return dev


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
        self.scope.duration = TIME_WINDOW
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'

        self.sample_rate = 125e6 / self.scope.decimation
        self.setup_output()
        print("RedPitaya initialized")

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
                    'duration': TIME_WINDOW,
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
            with open(YAML_FILE, 'w') as f:
                yaml.dump(config, f)

    def setup_output(self):
        self.asg.setup(
            waveform='sin',
            frequency=WAVEFORM_FREQ,
            amplitude=WAVEFORM_AMP,
            offset=WAVEFORM_OFFSET,
            output_direct='out1',
            trigger_source='immediately'
        )

    def capture(self):
        try:
            ch1 = np.array(self.scope._data_ch1)
            ch2 = np.array(self.scope._data_ch2)

            if ch1.size > 0 and ch2.size > 0:
                current = ch1 / SHUNT_R * 1e6
                return ch1, ch2, current
        except Exception as e:
            print(f"RedPitaya capture error: {e}")
        return None, None, None


# ------------------------- Main Measurement -------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("AC Cyclic Voltammetry - Step-by-Step Measurement")
    print("=" * 60)

    # Setup devices
    rodeostat_dev = setup_rodeostat()
    rp_scope = RedPitayaScope()

    # Storage for synchronized data
    accv_data = {
        'rodeo_t': [],
        'rodeo_volt': [],
        'rodeo_curr': [],
        'ac_voltage_amplitude': [],
        'ac_current_amplitude': [],
        'ac_phase': []
    }

    print(f"\nRunning {MODE.upper()} test with AC measurements...")
    print("This will take the full measurement time.\n")

    # Run Rodeostat test
    try:
        t, volt, curr = rodeostat_dev.run_test('cyclic', display='data', filename='data.txt')
    except Exception as e:
        print("Error running test:", e)
        traceback.print_exc()
        raise SystemExit

    print(f"\nRodeostat measurement complete: {len(t)} samples")
    print("Test complete. Data saved to data.txt")

    # Store Rodeostat data
    accv_data['rodeo_t'] = t
    accv_data['rodeo_volt'] = volt
    accv_data['rodeo_curr'] = curr

    # Now take AC measurements at key voltage points
    print("\nTaking AC measurements at voltage steps...")

    # Determine sampling points (every N samples)
    total_samples = len(t)
    step_interval = max(1, total_samples // 50)  # Take ~50 measurements across sweep

    for i in range(0, total_samples, step_interval):
        voltage = volt[i]
        dc_current = curr[i]

        # Take multiple AC measurements and average
        ac_volt_amps = []
        ac_curr_amps = []
        phases = []

        for _ in range(AC_MEASUREMENTS_PER_STEP):
            ch_in, ch_out, current = rp_scope.capture()
            if ch_in is not None:
                # Calculate AC amplitude (peak-to-peak / 2)
                ac_volt_amp = (np.max(ch_in) - np.min(ch_in)) / 2
                ac_curr_amp = (np.max(current) - np.min(current)) / 2

                # Calculate phase difference via FFT
                N = len(ch_in)
                if N > 0:
                    fft_in = np.fft.rfft(ch_in * np.hanning(N))
                    fft_out = np.fft.rfft(ch_out * np.hanning(N))
                    peak_idx = np.argmax(np.abs(fft_in))
                    phase_rad = np.angle(fft_in[peak_idx]) - np.angle(fft_out[peak_idx])
                    phase_deg = np.degrees(phase_rad)

                    ac_volt_amps.append(ac_volt_amp)
                    ac_curr_amps.append(ac_curr_amp)
                    phases.append(phase_deg)

            time.sleep(0.02)

        if ac_volt_amps:
            accv_data['ac_voltage_amplitude'].append(np.mean(ac_volt_amps))
            accv_data['ac_current_amplitude'].append(np.mean(ac_curr_amps))
            accv_data['ac_phase'].append(np.mean(phases))

        if (i // step_interval) % 10 == 0:
            print(f"  Progress: {i}/{total_samples} samples")

    print("\nAC measurements complete!")

    # ------------------------- Plot Results -------------------------
    print("\nGenerating plots...")

    fig = plt.figure(figsize=(14, 10))

    # Rodeostat voltage vs time
    plt.subplot(3, 3, 1)
    plt.plot(accv_data['rodeo_t'], accv_data['rodeo_volt'], 'b-')
    plt.ylabel('Voltage (V)')
    plt.title('Rodeostat Voltage')
    plt.grid(True)

    # Rodeostat current vs time
    plt.subplot(3, 3, 4)
    plt.plot(accv_data['rodeo_t'], accv_data['rodeo_curr'], 'g-')
    plt.ylabel('Current (uA)')
    plt.title('Rodeostat DC Current')
    plt.grid(True)

    # IV Curve (CV)
    plt.subplot(3, 3, 7)
    plt.plot(accv_data['rodeo_volt'], accv_data['rodeo_curr'], 'r-')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title(f'{MODE.upper()} Curve')
    plt.grid(True)

    # AC voltage amplitude vs DC voltage
    plt.subplot(3, 3, 2)
    sample_voltages = [accv_data['rodeo_volt'][i] for i in range(0, len(accv_data['rodeo_volt']), step_interval)][
                      :len(accv_data['ac_voltage_amplitude'])]
    plt.plot(sample_voltages, accv_data['ac_voltage_amplitude'], 'o-', color='orange')
    plt.ylabel('AC Voltage Amplitude (V)')
    plt.title('AC Voltage Response')
    plt.grid(True)

    # AC current amplitude vs DC voltage
    plt.subplot(3, 3, 5)
    plt.plot(sample_voltages, accv_data['ac_current_amplitude'], 'o-', color='purple')
    plt.ylabel('AC Current Amplitude (uA)')
    plt.title('AC Current Response')
    plt.grid(True)

    # Phase vs DC voltage
    plt.subplot(3, 3, 8)
    plt.plot(sample_voltages, accv_data['ac_phase'], 'o-', color='brown')
    plt.xlabel('DC Voltage (V)')
    plt.ylabel('Phase (degrees)')
    plt.title('AC Phase Shift')
    plt.grid(True)

    # AC impedance magnitude vs DC voltage
    plt.subplot(3, 3, 3)
    impedance = np.array(accv_data['ac_voltage_amplitude']) / (
                np.array(accv_data['ac_current_amplitude']) / 1e6)  # Convert uA to A
    plt.plot(sample_voltages, impedance, 'o-', color='teal')
    plt.ylabel('Impedance (Ohms)')
    plt.title('AC Impedance vs DC Bias')
    plt.grid(True)

    # Nyquist-like plot (Re vs Im of impedance)
    plt.subplot(3, 3, 6)
    Z_real = impedance * np.cos(np.radians(accv_data['ac_phase']))
    Z_imag = impedance * np.sin(np.radians(accv_data['ac_phase']))
    plt.plot(Z_real, Z_imag, 'o-', color='navy')
    plt.xlabel('Z Real (Ohms)')
    plt.ylabel('Z Imaginary (Ohms)')
    plt.title('Impedance Components')
    plt.grid(True)

    # Summary text
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
    ACCV Measurement Summary

    Mode: {MODE.upper()}
    Rodeostat Samples: {len(accv_data['rodeo_t'])}
    AC Measurements: {len(accv_data['ac_voltage_amplitude'])}

    DC Voltage Range: 
      {min(accv_data['rodeo_volt']):.3f} to {max(accv_data['rodeo_volt']):.3f} V

    DC Current Range:
      {min(accv_data['rodeo_curr']):.1f} to {max(accv_data['rodeo_curr']):.1f} uA

    AC Frequency: {WAVEFORM_FREQ} Hz
    AC Amplitude: {WAVEFORM_AMP} V
    Shunt R: {SHUNT_R} Ohms
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.show()

    print("\nMeasurement complete!")
    print("=" * 60)
