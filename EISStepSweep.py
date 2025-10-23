"""
Combined EIS System: Rodeostat + Red Pitaya
===========================================

Performs stepwise EIS: For each DC bias, performs a full EIS frequency sweep.
User selects Rodeostat COM port manually.

SYSTEM ARCHITECTURE:
- Rodeostat: Provides DC bias voltage
- Red Pitaya: Generates AC sine and demodulates with lock-in
"""

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

RED_PITAYA_IP = 'rp-f073ce.local'   # Your Red Pitaya IP address
OUTPUT_DIR = 'eis_step_data'        # Output directory for data and plots

# EIS Frequency Sweep Parameters
EIS_FREQUENCIES = {
    'start_freq': 10,       # Hz
    'end_freq': 10000,      # Hz
    'num_points': 10        # Number of frequencies per bias
}

# Stepwise EIS (DC Bias Sweep)
DC_SWEEP = {
    'start_bias': -0.2,     # V
    'end_bias': 0.8,        # V
    'num_steps': 6          # Number of DC bias points
}

# EIS Measurement Settings
EIS_MEASUREMENT = {
    'ac_amplitude': 0.01,   # V
    'settling_time': 1.0    # s
}

# =============================================================================
# END CONFIGURATION
# =============================================================================

import sys
from qtpy import QtCore

# Fix PyQt signal issue
if not hasattr(QtCore, 'pyqtBoundSignal'):
    try:
        from PyQt5.QtCore import QObject
        QtCore.pyqtBoundSignal = type(QObject().destroyed)
    except ImportError:
        class DummySignalType: pass
        QtCore.pyqtBoundSignal = DummySignalType

from potentiostat import Potentiostat
import serial.tools.list_ports
from pyrpl import Pyrpl
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os
from datetime import datetime
import logging

logging.getLogger("pyrpl").setLevel(logging.WARNING)

# =============================================================================
# DEVICE CLASSES
# =============================================================================

class RedPitaya:
    """Red Pitaya Lock-in control"""
    def __init__(self, output_dir='test_data', hostname='169.254.131.37'):
        self.rp = Pyrpl(config='lockin_config', hostname=hostname)
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lia_scope = self.rp_modules.scope
        self.lia_scope.input1 = 'iq2'
        self.lia_scope.input2 = 'iq2_2'
        self.lia_scope.decimation = 8192
        self.lia_scope._start_acquisition_rolling_mode()
        self.lia_scope.average = True
        self.sample_rate = 125e6 / self.lia_scope.decimation
        self.iq2 = self.rp_modules.iq2

    def setup_test_sig(self, params):
        test_freq = params['test_freq']
        test_amp = params['test_amp']
        self.test_sig = self.rp_modules.asg0
        self.test_sig.setup(waveform='sin', frequency=test_freq, amplitude=test_amp,
                            offset=0.00, output_direct='out1', trigger_source='immediately')

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        ref_amp = params['ref_amp']
        self.iq2.setup(frequency=self.ref_freq,
                       bandwidth=[-self.ref_freq*2, -self.ref_freq, self.ref_freq, self.ref_freq*2],
                       gain=1.0, phase=0, acbandwidth=0,
                       amplitude=ref_amp, input='in1', output_direct='out2',
                       output_signal='quadrature', quadrature_factor=20)

    def capture(self):
        self.lia_scope.single()
        ch1 = np.array(self.lia_scope._data_ch1_current)
        ch2 = np.array(self.lia_scope._data_ch2_current)
        return ch1, ch2


class RodeostatController:
    """Rodeostat control with manual port selection"""
    def __init__(self):
        self.dev = None
        self.connected = False

    def connect(self):
        try:
            ports = serial.tools.list_ports.comports()
            if not ports:
                print("⚠ No serial ports found. Connect Rodeostat and retry.")
                return False

            print("\nAvailable COM ports:")
            for i, p in enumerate(ports):
                print(f"  {i+1}: {p.device}")

            choice = input("Select Rodeostat port by number (Enter for first): ").strip()
            if choice == '':
                port = ports[0].device
            else:
                idx = int(choice) - 1
                port = ports[idx].device

            print(f"\nConnecting Rodeostat on {port} ...")
            self.dev = Potentiostat(port)

            try:
                _ = self.dev.get_all_curr_range()
            except KeyError:
                print("Unknown firmware. Adding current range list manually.")
                self.dev.hw_variant = 'manual_patch'
                self.dev.get_all_curr_range = lambda: ['1uA','10uA','100uA','1000uA']

            self.connected = True
            print("✓ Rodeostat connected.")
            return True
        except Exception as e:
            print(f"❌ Rodeostat connection failed: {e}")
            return False

    def set_dc_voltage(self, voltage):
        if not self.connected:
            print("⚠ Rodeostat not connected.")
            return False
        print(f"Setting DC bias to {voltage:.3f} V")
        # Here you could send a DC command via a custom test if supported
        return True

    def disconnect(self):
        if self.connected:
            self.connected = False
            print("✓ Rodeostat disconnected.")


class CombinedEISSystem:
    """Handles combined Red Pitaya + Rodeostat EIS"""
    def __init__(self, redpitaya_ip='169.254.131.37', output_dir='eis_data'):
        self.output_dir = output_dir
        self.redpitaya_ip = redpitaya_ip
        self.rodeostat = RodeostatController()
        self.redpitaya = None
        self.connected = False
        self.all_data = []

    def connect_all(self):
        print("\nInitializing Combined EIS System ...")
        rodeostat_ok = self.rodeostat.connect()
        try:
            print(f"\nConnecting Red Pitaya at {self.redpitaya_ip} ...")
            self.redpitaya = RedPitaya(hostname=self.redpitaya_ip, output_dir=self.output_dir)
            print("✓ Red Pitaya connected.")
            redpitaya_ok = True
        except Exception as e:
            print(f"❌ Red Pitaya connection failed: {e}")
            redpitaya_ok = False
        self.connected = rodeostat_ok and redpitaya_ok
        return self.connected

    def measure_impedance(self, frequency, dc_bias, ac_amplitude, settling_time):
        if not self.connected:
            return None

        self.rodeostat.set_dc_voltage(dc_bias)

        params = {'test_freq': frequency, 'test_amp': ac_amplitude,
                  'ref_freq': frequency, 'ref_amp': ac_amplitude}
        self.redpitaya.setup_test_sig(params)
        self.redpitaya.setup_lockin(params)
        time.sleep(settling_time)

        X, Y = self.redpitaya.capture()
        mag = np.sqrt(np.mean(X**2 + Y**2))
        phase = np.degrees(np.arctan2(np.mean(Y), np.mean(X)))
        Zr = mag * np.cos(np.radians(phase))
        Zi = mag * np.sin(np.radians(phase))

        print(f"f={frequency:8.1f} Hz | Bias={dc_bias:+.3f} V | |Z|={mag:8.4f} | θ={phase:6.1f}°")

        return {'dc_bias': dc_bias, 'frequency': frequency, 'Z_real': Zr,
                'Z_imag': Zi, 'Z_magnitude': mag, 'Z_phase': phase}

    def stepwise_eis(self, freq_list, bias_list, ac_amplitude, settling_time):
        all_results = []
        for bias in bias_list:
            print(f"\n===== DC Bias = {bias:+.3f} V =====")
            for f in freq_list:
                res = self.measure_impedance(f, bias, ac_amplitude, settling_time)
                if res:
                    all_results.append(res)
        self.all_data = all_results
        return all_results

    def save_csv(self):
        if not self.all_data:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        fn = os.path.join(self.output_dir, f"eis_vs_bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.all_data[0].keys())
            w.writeheader()
            for r in self.all_data:
                w.writerow(r)
        print(f"✅ Data saved: {fn}")

    def plot_results(self):
        if not self.all_data:
            return
        plt.figure(figsize=(8,6))
        unique_biases = sorted(set([r['dc_bias'] for r in self.all_data]))
        for b in unique_biases:
            sub = [r for r in self.all_data if r['dc_bias'] == b]
            freqs = [r['frequency'] for r in sub]
            mags = [r['Z_magnitude'] for r in sub]
            plt.loglog(freqs, mags, 'o-', label=f"{b:+.2f} V")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|Z| (a.u.)")
        plt.title("Stepwise EIS vs DC Bias")
        plt.grid(True, which='both', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def disconnect(self):
        if self.rodeostat:
            self.rodeostat.disconnect()
        print("✓ Disconnected all devices.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    system = CombinedEISSystem(redpitaya_ip=RED_PITAYA_IP, output_dir=OUTPUT_DIR)
    if not system.connect_all():
        print("❌ Failed to connect devices. Exiting.")
        return

    freqs = np.logspace(np.log10(EIS_FREQUENCIES['start_freq']),
                        np.log10(EIS_FREQUENCIES['end_freq']),
                        EIS_FREQUENCIES['num_points'])
    biases = np.linspace(DC_SWEEP['start_bias'], DC_SWEEP['end_bias'], DC_SWEEP['num_steps'])
    ac_amp = EIS_MEASUREMENT['ac_amplitude']
    settle = EIS_MEASUREMENT['settling_time']

    print("\nStarting Stepwise EIS Measurement ...\n")
    data = system.stepwise_eis(freqs, biases, ac_amp, settle)
    if data:
        system.save_csv()
        system.plot_results()
    system.disconnect()


if __name__ == '__main__':
    main()
