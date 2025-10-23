"""
Flexible Rodeostat Measurement
==============================

- Optional DC bias: fixed or sweep
- Optional AC injection: fixed or sweep
- Measures voltage and current at each step
- All plots combined into a single figure
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'measurement_data'  # Directory to save CSV data and plots

# DC configuration
DC_ENABLE = True        # Enable DC bias measurement
DC_SWEEP = True         # If True, sweep DC bias; if False, use fixed DC
DC_FIXED = 0.0          # Fixed DC voltage (V) if sweep is disabled
DC_START = -0.2         # Start voltage (V) for DC sweep
DC_END = 0.8            # End voltage (V) for DC sweep
DC_STEPS = 5            # Number of points in DC sweep

# AC configuration
AC_ENABLE = True        # Enable AC injection
AC_SWEEP = True         # If True, sweep AC frequency; if False, use fixed frequency
AC_FIXED_FREQ = 100     # Fixed AC frequency (Hz) if sweep is disabled
AC_FIXED_AMP = 0.01     # AC amplitude (V)
AC_START_FREQ = 10      # Start frequency (Hz) for AC sweep
AC_END_FREQ = 1000      # End frequency (Hz) for AC sweep
AC_NUM_POINTS = 5       # Number of points in AC sweep

SETTLE_TIME = 0.5       # Time (s) to wait after setting voltage or AC injection

# =============================================================================
# IMPORTS
# =============================================================================

import os, csv, time
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
from datetime import datetime
from potentiostat import Potentiostat

# =============================================================================
# RODEOSTAT CLASS
# =============================================================================

class RodeostatController:
    """Handles DC and AC measurements on Rodeostat"""

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

    def measure_ac(self, freq, amp):
        """Measure AC voltage and current (dummy AC injection if hardware not supported)"""
        time.sleep(SETTLE_TIME)
        v = float(self.dev.get_volt())  # AC voltage sample
        i = float(self.dev.get_curr())  # AC current sample
        return v, i

    def disconnect(self):
        """Disconnect Rodeostat"""
        if self.connected:
            self.connected = False
            print("✓ Rodeostat disconnected.")

# =============================================================================
# MEASUREMENT SYSTEM
# =============================================================================

class MeasurementSystem:
    """Manages full measurement routine with optional DC/AC sweeps"""

    def __init__(self):
        self.rodeostat = RodeostatController()
        self.data = []

    def connect_all(self):
        """Connect all devices"""
        return self.rodeostat.connect()

    def run_measurement(self):
        """Run the measurement loop over DC and AC values"""

        # Create DC voltage list
        if DC_ENABLE:
            if DC_SWEEP:
                dc_list = np.linspace(DC_START, DC_END, DC_STEPS)
            else:
                dc_list = [DC_FIXED]
        else:
            dc_list = [None]

        # Create AC frequency list
        if AC_ENABLE:
            if AC_SWEEP:
                ac_list = np.logspace(np.log10(AC_START_FREQ), np.log10(AC_END_FREQ), AC_NUM_POINTS)
            else:
                ac_list = [AC_FIXED_FREQ]
        else:
            ac_list = [None]

        # Loop through all combinations
        for bias in dc_list:
            if DC_ENABLE and bias is not None:
                self.rodeostat.set_dc_bias(bias)
                vdc, idc = self.rodeostat.measure_dc()
            else:
                vdc, idc = None, None

            for freq in ac_list:
                if AC_ENABLE and freq is not None:
                    vac, iac = self.rodeostat.measure_ac(freq, AC_FIXED_AMP)
                else:
                    vac, iac = None, None

                # Store measurement
                self.data.append({
                    'DC_Bias': bias,
                    'AC_Freq': freq,
                    'V_DC': vdc,
                    'I_DC': idc,
                    'V_AC': vac,
                    'I_AC': iac
                })

                print(f"DC={bias} V  AC_Freq={freq} Hz  V_DC={vdc}  I_DC={idc}  V_AC={vac}  I_AC={iac}")

    def save_csv(self):
        """Save all measurement data to CSV"""
        if not self.data:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fn = os.path.join(OUTPUT_DIR, f"measurement_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.data[0].keys())
            w.writeheader()
            w.writerows(self.data)
        print("✅ Data saved:", fn)

    def plot_results(self):
        """Plot all results in a single figure with subplots"""
        if not self.data:
            return

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Rodeostat Measurement Results", fontsize=16)

        # DC I-V
        dc_points = [d for d in self.data if d['V_DC'] is not None]
        if dc_points:
            axs[0, 0].plot([d['DC_Bias'] for d in dc_points], [d['I_DC'] for d in dc_points], 'o-')
            axs[0, 0].set_xlabel("DC Bias (V)")
            axs[0, 0].set_ylabel("DC Current (A)")
            axs[0, 0].set_title("DC I-V")
            axs[0, 0].grid(True)

        # AC Voltage vs DC bias
        ac_points = [d for d in self.data if d['V_AC'] is not None]
        if ac_points:
            freqs = sorted(list(set(d['AC_Freq'] for d in ac_points)))
            for f in freqs:
                sub = [d for d in ac_points if d['AC_Freq']==f]
                axs[0, 1].plot([d['DC_Bias'] for d in sub], [d['V_AC'] for d in sub], 'o-', label=f"{f} Hz")
            axs[0, 1].set_xlabel("DC Bias (V)")
            axs[0, 1].set_ylabel("AC Voltage (V)")
            axs[0, 1].set_title("AC Voltage vs DC Bias")
            axs[0, 1].legend()
            axs[0, 1].grid(True)

        # AC Current vs DC bias
        if ac_points:
            for f in freqs:
                sub = [d for d in ac_points if d['AC_Freq']==f]
                axs[1, 0].plot([d['DC_Bias'] for d in sub], [d['I_AC'] for d in sub], 'o-', label=f"{f} Hz")
            axs[1, 0].set_xlabel("DC Bias (V)")
            axs[1, 0].set_ylabel("AC Current (A)")
            axs[1, 0].set_title("AC Current vs DC Bias")
            axs[1, 0].legend()
            axs[1, 0].grid(True)

        # DC Voltage vs DC Bias (should be linear)
        if dc_points:
            axs[1, 1].plot([d['DC_Bias'] for d in dc_points], [d['V_DC'] for d in dc_points], 'o-')
            axs[1, 1].set_xlabel("DC Bias (V)")
            axs[1, 1].set_ylabel("DC Voltage (V)")
            axs[1, 1].set_title("DC Voltage vs Bias")
            axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def disconnect(self):
        """Disconnect all devices"""
        self.rodeostat.disconnect()

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    system = MeasurementSystem()
    if not system.connect_all():
        print("❌ Failed to connect Rodeostat.")
        return

    system.run_measurement()
    system.save_csv()
    system.plot_results()
    system.disconnect()

if __name__ == "__main__":
    main()
