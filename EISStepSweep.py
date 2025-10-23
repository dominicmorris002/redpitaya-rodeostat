"""
Combined EIS System: Rodeostat + Red Pitaya
===========================================

Performs stepwise EIS: For each DC bias, performs a full EIS frequency sweep.
User selects Rodeostat COM port manually.

SYSTEM ARCHITECTURE:
- Rodeostat: Provides DC bias and measures total current
- Red Pitaya: Generates AC sine and demodulates with lock-in
"""

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

RED_PITAYA_IP = 'rp-f073ce.local'   # Your Red Pitaya IP address
OUTPUT_DIR = 'eis_step_data'        # Output directory

# EIS Frequency Sweep Parameters
EIS_FREQUENCIES = {'start_freq': 10, 'end_freq': 10000, 'num_points': 10}

# Stepwise DC Bias Sweep
DC_SWEEP = {'start_bias': -0.2, 'end_bias': 0.8, 'num_steps': 6}

# Measurement Settings
EIS_MEASUREMENT = {'ac_amplitude': 0.01, 'settling_time': 1.0}

# =============================================================================
# IMPORTS
# =============================================================================

import os, csv, time, logging
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
from datetime import datetime
from pyrpl import Pyrpl
from potentiostat import Potentiostat
from qtpy import QtCore

# Patch pyqtBoundSignal for PyQt5 >=5.15
if not hasattr(QtCore, 'pyqtBoundSignal'):
    try:
        from PyQt5.QtCore import QObject
        QtCore.pyqtBoundSignal = type(QObject().destroyed)
    except ImportError:
        class DummySignalType: pass
        QtCore.pyqtBoundSignal = DummySignalType

logging.getLogger("pyrpl").setLevel(logging.WARNING)

# =============================================================================
# DEVICE CLASSES
# =============================================================================

class RedPitaya:
    """Red Pitaya lock-in interface"""
    def __init__(self, hostname, output_dir):
        self.rp = Pyrpl(config='lockin_config', hostname=hostname)
        self.output_dir = output_dir
        self.scope = self.rp.rp.scope
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = 8192
        self.scope.average = True
        self.scope._start_acquisition_rolling_mode()
        self.sample_rate = 125e6 / self.scope.decimation
        self.iq2 = self.rp.rp.iq2

    def setup(self, freq, amp):
        self.iq2.setup(frequency=freq,
                       amplitude=amp,
                       input='in1',
                       output_direct='out2',
                       output_signal='quadrature',
                       quadrature_factor=20)

    def capture(self):
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)
        X = np.mean(ch1)
        Y = np.mean(ch2)
        mag = np.sqrt(X**2 + Y**2)
        phase = np.degrees(np.arctan2(Y, X))
        return mag, phase


class RodeostatController:
    """Handles DC bias and current measurement"""
    def __init__(self):
        self.dev = None
        self.connected = False

    def connect(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("⚠ No serial ports found.")
            return False
        print("\nAvailable COM ports:")
        for i, p in enumerate(ports):
            print(f"  {i+1}: {p.device}")
        choice = input("Select Rodeostat port by number (Enter for first): ").strip()
        port = ports[int(choice)-1].device if choice else ports[0].device
        print(f"Connecting to {port} ...")
        self.dev = Potentiostat(port)
        self.connected = True
        print("✓ Rodeostat connected.")
        return True

    def set_dc_bias(self, bias_v):
        if not self.connected:
            raise Exception("Rodeostat not connected.")
        try:
            # Apply DC potential
            self.dev.set_volt(bias_v)
            time.sleep(0.2)  # allow voltage to stabilize
        except Exception as e:
            print(f"⚠ Failed to set DC bias: {e}")

    def measure_dc_current(self):
        """Measure current at the current DC bias."""
        data = self.dev.get_curr()
        return float(data)

    def disconnect(self):
        if self.connected:
            self.connected = False
            print("✓ Rodeostat disconnected.")


# =============================================================================
# MAIN COMBINED EIS SYSTEM
# =============================================================================

class CombinedEISSystem:
    def __init__(self, redpitaya_ip, output_dir):
        self.redpitaya_ip = redpitaya_ip
        self.output_dir = output_dir
        self.rodeostat = RodeostatController()
        self.redpitaya = None
        self.data = []

    def connect_all(self):
        r_ok = self.rodeostat.connect()
        try:
            self.redpitaya = RedPitaya(hostname=self.redpitaya_ip, output_dir=self.output_dir)
            print("✓ Red Pitaya connected.")
            p_ok = True
        except Exception as e:
            print("❌ Red Pitaya connection failed:", e)
            p_ok = False
        return r_ok and p_ok

    def measure_point(self, bias, freq, ac_amp, settle):
        """At each DC bias and AC freq, measure |Z| and phase."""
        self.rodeostat.set_dc_bias(bias)
        self.redpitaya.setup(freq, ac_amp)
        time.sleep(settle)
        mag, phase = self.redpitaya.capture()
        dc_current = self.rodeostat.measure_dc_current()

        # Compute complex impedance
        Z = ac_amp / mag if mag != 0 else np.inf
        Zr = Z * np.cos(np.radians(phase))
        Zi = Z * np.sin(np.radians(phase))
        print(f"Bias={bias:+.3f} V  Freq={freq:7.1f} Hz  |Z|={Z:8.3f}  Phase={phase:6.2f}°  IDC={dc_current:9.3e}")
        return dict(bias=bias, freq=freq, Zmag=Z, phase=phase, Zreal=Zr, Zimag=Zi, Idc=dc_current)

    def run_stepwise_eis(self, freq_list, bias_list, ac_amp, settle):
        for bias in bias_list:
            print(f"\n===== DC Bias = {bias:+.3f} V =====")
            for freq in freq_list:
                result = self.measure_point(bias, freq, ac_amp, settle)
                self.data.append(result)

    def save_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        fn = os.path.join(self.output_dir, f"eis_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(fn, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
        print("✅ Data saved to", fn)

    def plot_bode_nyquist(self):
        if not self.data:
            return
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title("Bode Magnitude")
        plt.loglog([d['freq'] for d in self.data], [d['Zmag'] for d in self.data], 'o-')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|Z| (Ohm)")
        plt.grid(True, which='both', ls='--')

        plt.subplot(1,2,2)
        plt.title("Nyquist Plot")
        plt.plot([d['Zreal'] for d in self.data], [-d['Zimag'] for d in self.data], 'o-')
        plt.xlabel("Zreal (Ohm)")
        plt.ylabel("-Zimag (Ohm)")
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def disconnect(self):
        self.rodeostat.disconnect()
        print("✓ All devices disconnected.")


# =============================================================================
# RUN SCRIPT
# =============================================================================

def main():
    sys = CombinedEISSystem(RED_PITAYA_IP, OUTPUT_DIR)
    if not sys.connect_all():
        print("Connection failed. Exiting.")
        return

    freqs = np.logspace(np.log10(EIS_FREQUENCIES['start_freq']),
                        np.log10(EIS_FREQUENCIES['end_freq']),
                        EIS_FREQUENCIES['num_points'])
    biases = np.linspace(DC_SWEEP['start_bias'], DC_SWEEP['end_bias'], DC_SWEEP['num_steps'])
    ac_amp = EIS_MEASUREMENT['ac_amplitude']
    settle = EIS_MEASUREMENT['settling_time']

    sys.run_stepwise_eis(freqs, biases, ac_amp, settle)
    sys.save_csv()
    sys.plot_bode_nyquist()
    sys.disconnect()


if __name__ == "__main__":
    main()
