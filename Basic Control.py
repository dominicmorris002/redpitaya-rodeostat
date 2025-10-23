"""
Stepwise DC + AC Frequency Sweep Measurement System
===================================================

Performs measurements:
- Applies DC bias from Rodeostat in steps
- Outputs AC sine from Red Pitaya with multiple frequency steps per DC bias
- Measures DC voltage/current, AC voltage/current for each DC+AC step
- Saves CSV and plots results
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
RED_PITAYA_IP = 'rp-f073ce.local'
OUTPUT_DIR = 'dc_ac_sweep_data'

# DC sweep
DC_SWEEP = {
    'start_bias': -0.2,
    'end_bias': 0.8,
    'num_steps': 6
}

# AC frequency sweep parameters
AC_SWEEP = {
    'start_freq': 10,    # Hz
    'end_freq': 1000,    # Hz
    'num_points': 5,     # Frequencies per DC step
    'amplitude': 0.01    # V
}

SETTLING_TIME = 0.5  # seconds for stabilization

# =============================================================================
# IMPORTS
# =============================================================================
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
from potentiostat import Potentiostat
from pyrpl import Pyrpl
from datetime import datetime

# =============================================================================
# DEVICE CLASSES
# =============================================================================
class RodeostatController:
    """Handles DC bias and current measurement"""
    def __init__(self):
        self.dev = None
        self.connected = False

    def connect(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found. Connect Rodeostat.")
            return False
        port = ports[0].device
        print(f"Connecting Rodeostat on {port} ...")
        self.dev = Potentiostat(port)
        self.connected = True
        print("Rodeostat connected.")
        return True

    def set_dc_voltage(self, voltage):
        if self.connected:
            print(f"Setting DC bias: {voltage:.3f} V")
            # Apply DC bias here
            return True
        return False

    def measure_dc_current(self):
        if self.connected:
            return np.random.random() * 1e-6
        return None

    def measure_dc_voltage(self):
        if self.connected:
            return np.random.random()
        return None

    def disconnect(self):
        if self.connected:
            self.connected = False
            print("Rodeostat disconnected.")

class RedPitaya:
    """Outputs AC sine and measures AC voltage/current"""
    def __init__(self, hostname='169.254.131.37'):
        self.rp = Pyrpl(config='lockin_config', hostname=hostname)
        self.asg = self.rp.rp.asg0

    def output_ac(self, frequency, amplitude):
        self.asg.setup(waveform='sin', frequency=frequency, amplitude=amplitude,
                       offset=0.0, output_direct='out1', trigger_source='immediately')
        print(f"Red Pitaya output AC: {frequency} Hz, {amplitude} V")

    def stop_ac(self):
        self.asg.stop()
        print("Red Pitaya AC stopped.")

    def measure_ac_voltage_current(self):
        # Replace with actual measurements
        ac_voltage = AC_SWEEP['amplitude'] + np.random.random()*0.001
        ac_current = 1e-6 + np.random.random()*1e-7
        return ac_voltage, ac_current

# =============================================================================
# COMBINED SYSTEM
# =============================================================================
class DCACFrequencySweepSystem:
    def __init__(self):
        self.rodeostat = RodeostatController()
        self.redpitaya = RedPitaya(hostname=RED_PITAYA_IP)
        self.data = []

    def run(self):
        if not self.rodeostat.connect():
            print("Rodeostat connection failed. Exiting.")
            return

        dc_steps = np.linspace(DC_SWEEP['start_bias'], DC_SWEEP['end_bias'], DC_SWEEP['num_steps'])
        ac_freqs = np.logspace(np.log10(AC_SWEEP['start_freq']), np.log10(AC_SWEEP['end_freq']), AC_SWEEP['num_points'])

        for dc in dc_steps:
            self.rodeostat.set_dc_voltage(dc)
            time.sleep(SETTLING_TIME)
            dc_current = self.rodeostat.measure_dc_current()
            dc_voltage = self.rodeostat.measure_dc_voltage()

            for freq in ac_freqs:
                self.redpitaya.output_ac(freq, AC_SWEEP['amplitude'])
                time.sleep(SETTLING_TIME)
                ac_voltage, ac_current = self.redpitaya.measure_ac_voltage_current()
                self.redpitaya.stop_ac()

                print(f"DC={dc:.3f} V | I_DC={dc_current*1e6:.3f} uA | "
                      f"AC {freq:.1f} Hz | V_AC={ac_voltage:.3f} V | I_AC={ac_current*1e6:.3f} uA")

                self.data.append({
                    'dc_bias': dc,
                    'dc_voltage': dc_voltage,
                    'dc_current': dc_current,
                    'ac_frequency': freq,
                    'ac_voltage': ac_voltage,
                    'ac_current': ac_current
                })

        self.save_csv()
        self.plot_results()
        self.rodeostat.disconnect()

    def save_csv(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fn = os.path.join(OUTPUT_DIR, f"dc_ac_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(fn, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
        print(f"Data saved: {fn}")

    def plot_results(self):
        dc = [d['dc_bias'] for d in self.data]
        ac_freq = [d['ac_frequency'] for d in self.data]
        i_dc = [d['dc_current']*1e6 for d in self.data]
        v_ac = [d['ac_voltage'] for d in self.data]
        i_ac = [d['ac_current']*1e6 for d in self.data]

        plt.figure(figsize=(10,6))
        plt.subplot(3,1,1)
        plt.plot(dc, i_dc, 'o-', label='DC Current (uA)')
        plt.ylabel("DC Current (uA)")
        plt.grid(True)
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(dc, v_ac, 's-', label='AC Voltage (V)')
        plt.ylabel("AC Voltage (V)")
        plt.grid(True)
        plt.legend()

        plt.subplot(3,1,3)
        plt.semilogx(ac_freq, i_ac, 'o-', label='AC Current (uA)')
        plt.xlabel("AC Frequency (Hz)")
        plt.ylabel("AC Current (uA)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    system = DCACFrequencySweepSystem()
    system.run()
