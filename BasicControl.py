"""
Flexible Rodeostat + Red Pitaya Measurement
===========================================

- Rodeostat provides DC bias control and measurement
- Red Pitaya handles AC injection and waveform measurement
- Combined DC + AC data analysis and visualization

Plots:
------
DC:
  • Input Voltage (Rodeostat) over Time
  • Output Current over Time
  • Output Current vs Output Voltage
AC:
  • Input Voltage over Time
  • Output Current + Output Voltage overlayed (phase shown)
Combined:
  • Total Output Current over Time
  • Total I–V(T)
  • Total Output Current vs Total Output Voltage
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'measurement_data'  # Directory to save CSV data and plots

# Rodeostat (DC bias) configuration
DC_ENABLE = True
DC_SWEEP = True
DC_FIXED = 0.0
DC_START = -0.2
DC_END = 0.8
DC_STEPS = 5

# Red Pitaya (AC injection) configuration
AC_ENABLE = True
AC_SWEEP = True
AC_FIXED_FREQ = 100
AC_FIXED_AMP = 0.01
AC_START_FREQ = 10
AC_END_FREQ = 1000
AC_NUM_POINTS = 5
SAMPLE_RATE = 125e3      # Red Pitaya ADC sample rate (samples/sec)
SAMPLES = 4096           # Samples to capture per waveform
SETTLE_TIME = 0.5        # Delay after setting new conditions

REDPITAYA_IP = "192.168.1.100"  # <-- change to your Red Pitaya IP address

# =============================================================================
# IMPORTS
# =============================================================================

import os, csv, time, socket
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
from datetime import datetime
from potentiostat import Potentiostat

# =============================================================================
# RODEOSTAT CLASS
# =============================================================================

class RodeostatController:
    """Handles DC bias control via Rodeostat"""

    def __init__(self):
        self.dev = None
        self.connected = False

    def connect(self):
        """Select and connect to Rodeostat via serial port"""
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

    def set_dc_bias(self, bias):
        """Set DC bias voltage"""
        if not self.connected:
            raise Exception("Rodeostat not connected.")
        self.dev.set_volt(bias)
        time.sleep(SETTLE_TIME)

    def measure_dc(self):
        """Measure DC voltage and current"""
        v = float(self.dev.get_volt())
        i = float(self.dev.get_curr())
        return v, i

    def disconnect(self):
        """Disconnect Rodeostat"""
        if self.connected:
            self.connected = False
            print("✓ Rodeostat disconnected.")

# =============================================================================
# RED PITAYA CLASS
# =============================================================================

class RedPitayaController:
    """Handles AC waveform generation and acquisition via SCPI commands"""

    def __init__(self, ip):
        self.ip = ip
        self.sock = None

    def connect(self):
        """Connect to Red Pitaya via TCP SCPI socket"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, 5000))
            self.sock.settimeout(2)
            print(f"✓ Connected to Red Pitaya at {self.ip}")
            return True
        except Exception as e:
            print(f"❌ Red Pitaya connection failed: {e}")
            return False

    def send(self, cmd):
        """Send SCPI command"""
        self.sock.send((cmd + '\n').encode())

    def query(self, cmd):
        """Send SCPI command and read response"""
        self.send(cmd)
        return self.sock.recv(16384).decode().strip()

    def configure_sine(self, freq, amp):
        """Configure Red Pitaya to output a sine wave"""
        self.send("GEN:RST")
        self.send(f"SOUR1:FUNC SINE")
        self.send(f"SOUR1:FREQ:FIX {freq}")
        self.send(f"SOUR1:VOLT {amp}")
        self.send("OUTPUT1:STATE ON")

    def acquire_waveforms(self):
        """Acquire waveform data from Red Pitaya analog inputs"""
        self.send("ACQ:RST")
        self.send("ACQ:DEC 1")
        self.send("ACQ:TRIG:LEV 0.05")
        self.send("ACQ:TRIG:DLY 8192")
        self.send("ACQ:START")
        time.sleep(SETTLE_TIME)
        self.send("ACQ:TRIG NOW")
        time.sleep(SETTLE_TIME)

        vin = np.array(self._get_data("SOUR1:DATA? CH1"), dtype=float)
        vout = np.array(self._get_data("SOUR1:DATA? CH2"), dtype=float)
        return vin, vout

    def _get_data(self, cmd):
        """Helper to fetch and parse waveform data"""
        self.send(cmd)
        raw = self.sock.recv(65536).decode().strip('{}\n\r').replace(' ', '').split(',')
        return raw

    def disconnect(self):
        """Close connection"""
        if self.sock:
            self.send("OUTPUT1:STATE OFF")
            self.sock.close()
            print("✓ Red Pitaya disconnected.")

# =============================================================================
# MEASUREMENT SYSTEM
# =============================================================================

class MeasurementSystem:
    def __init__(self):
        self.rodeostat = RodeostatController()
        self.redpitaya = RedPitayaController(REDPITAYA_IP)
        self.data = []

    def connect_all(self):
        """Connect both devices"""
        ok1 = self.rodeostat.connect()
        ok2 = self.redpitaya.connect()
        return ok1 and ok2

    def run_measurement(self):
        """Run DC + AC sweep measurements"""

        dc_list = np.linspace(DC_START, DC_END, DC_STEPS) if DC_SWEEP else [DC_FIXED]
        ac_list = np.logspace(np.log10(AC_START_FREQ), np.log10(AC_END_FREQ), AC_NUM_POINTS) if AC_SWEEP else [AC_FIXED_FREQ]

        for bias in dc_list:
            self.rodeostat.set_dc_bias(bias)
            vdc, idc = self.rodeostat.measure_dc()
            print(f"Set DC bias {bias:.3f} V → Vdc={vdc:.3f} V, Idc={idc:.6f} A")

            for freq in ac_list:
                print(f"  Injecting {freq:.1f} Hz AC signal...")
                self.redpitaya.configure_sine(freq, AC_FIXED_AMP)
                vin, vout = self.redpitaya.acquire_waveforms()

                # Compute instantaneous current (example with R=100Ω)
                R_LOAD = 100.0
                iout = vout / R_LOAD
                t = np.arange(len(vin)) / SAMPLE_RATE

                self.data.append({
                    'DC_Bias': bias,
                    'AC_Freq': freq,
                    'Time': t,
                    'Vin': vin,
                    'Vout': vout,
                    'Iout': iout,
                    'V_DC': vdc,
                    'I_DC': idc
                })

        print("✅ Measurement complete")

    def save_csv(self):
        """Save measurement data summary"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fn = os.path.join(OUTPUT_DIR, f"measurement_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(fn, "w", newline="") as f:
            fieldnames = ['DC_Bias', 'AC_Freq', 'V_DC', 'I_DC']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for d in self.data:
                w.writerow({k: d[k] for k in fieldnames})
        print("✅ Saved:", fn)

    def plot_results(self):
        """Generate requested plots"""
        if not self.data:
            return

        for d in self.data:
            t = d['Time']
            vin, vout, iout = d['Vin'], d['Vout'], d['Iout']

            fig, axs = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle(f"DC={d['DC_Bias']} V  AC={d['AC_Freq']} Hz", fontsize=14)

            # DC plots
            axs[0,0].plot(t, np.full_like(t, d['V_DC']))
            axs[0,0].set_title("DC: Input Voltage (Rodeostat) over Time")
            axs[0,0].set_xlabel("Time (s)")
            axs[0,0].set_ylabel("V_DC (V)")

            axs[0,1].plot(t, np.full_like(t, d['I_DC']))
            axs[0,1].set_title("DC: Output Current over Time")
            axs[0,1].set_xlabel("Time (s)")
            axs[0,1].set_ylabel("I_DC (A)")

            axs[1,0].plot([d['V_DC']], [d['I_DC']], 'ro')
            axs[1,0].set_title("DC: Output Current vs Output Voltage")
            axs[1,0].set_xlabel("V_DC (V)")
            axs[1,0].set_ylabel("I_DC (A)")

            # AC plots
            axs[1,1].plot(t, vin, label="Vin")
            axs[1,1].set_title("AC: Input Voltage over Time")
            axs[1,1].set_xlabel("Time (s)")
            axs[1,1].set_ylabel("Vin (V)")
            axs[1,1].legend()

            axs[2,0].plot(t, vout, label="Vout")
            axs[2,0].plot(t, iout, label="Iout")
            axs[2,0].set_title("AC: Output Current & Voltage (Phase Overlay)")
            axs[2,0].set_xlabel("Time (s)")
            axs[2,0].legend()

            axs[2,1].plot(vout, iout)
            axs[2,1].set_title("Combined: Output Current vs Output Voltage")
            axs[2,1].set_xlabel("Vout (V)")
            axs[2,1].set_ylabel("Iout (A)")

            plt.tight_layout(rect=[0,0,1,0.96])
            plt.show()

    def disconnect(self):
        self.rodeostat.disconnect()
        self.redpitaya.disconnect()

# =============================================================================
# MAIN
# =============================================================================

def main():
    system = MeasurementSystem()
    if not system.connect_all():
        print("❌ Connection failed.")
        return
    system.run_measurement()
    system.save_csv()
    system.plot_results()
    system.disconnect()

if __name__ == "__main__":
    main()
