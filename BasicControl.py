"""
Hybrid Rodeostat (DC) + Red Pitaya (AC + measurement) measurement script
- Rodeostat (via potentiostat.Potentiostat) sets DC bias
- Red Pitaya (SCPI over TCP) generates AC and measures channels
- Save CSV and plot combined figure

Configure constants below for your hardware, IPs, and shunt resistor.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'measurement_data'  # Directory to save CSV data and plots

# Rodeostat / DC configuration
DC_ENABLE = True        # Enable DC bias control
DC_SWEEP = True         # If True, sweep DC bias; if False, use fixed DC
DC_FIXED = 0.0          # Fixed DC voltage (V) if sweep is disabled
DC_START = -0.2         # Start voltage (V) for DC sweep
DC_END = 0.8            # End voltage (V) for DC sweep
DC_STEPS = 5            # Number of points in DC sweep

# Red Pitaya / AC configuration
AC_ENABLE = True        # Enable AC injection
AC_SWEEP = True         # Sweep frequency?
AC_FIXED_FREQ = 100     # Fixed AC frequency if not sweeping (Hz)
AC_FIXED_AMP = 0.01     # AC amplitude (peak) in V (set amplitude of generator)

AC_START_FREQ = 10      # Start freq for AC sweep (Hz)
AC_END_FREQ = 1000      # End freq for AC sweep (Hz)
AC_NUM_POINTS = 5       # Points in AC sweep

SETTLE_TIME = 0.5       # Seconds to wait after setting voltages/waveforms

# Red Pitaya network config
RP_IP = '192.168.1.100'  # <-- change to your Red Pitaya IP
RP_PORT = 5000           # default SCPI/TCP port for many Red Pitaya images

# Red Pitaya channel mapping
# (adjust according to your wiring: which channel sees the output, which sees the shunt)
RP_CHANNEL_OUT = 1   # channel where Red Pitaya output (AC source) is wired (1 or 2)
RP_CHANNEL_SHUNT = 2 # channel where the shunt resistor voltage is measured (1 or 2)

SHUNT_RESISTANCE = 10.0   # ohms: voltage across shunt / R = current (A)

# Other run settings
SAMPLE_AC_POINTS = 2048   # number of acquisition samples to request (depends on RP firmware)
AC_MEAS_WINDOW = 0.1      # seconds to acquire for AC measurement (used for wait/timing)

# =============================================================================
# IMPORTS
# =============================================================================

import os
import csv
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports
from datetime import datetime
from potentiostat import Potentiostat   # existing Rodeostat wrapper you already use

# =============================================================================
# RODEOSTAT CONTROLLER (DC bias only)
# =============================================================================

class RodeostatController:
    """Controls Rodeostat for DC bias only (uses existing Potentiostat wrapper)"""

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
        """Set DC bias voltage using Rodeostat"""
        if not self.connected:
            raise Exception("Rodeostat not connected.")
        # Use the Potentiostat API you already have:
        self.dev.set_volt(bias)
        time.sleep(SETTLE_TIME)

    def measure_dc(self):
        """Measure DC voltage and current from Rodeostat (if desired)"""
        if not self.connected:
            raise Exception("Rodeostat not connected.")
        v = float(self.dev.get_volt())
        i = float(self.dev.get_curr())
        return v, i

    def disconnect(self):
        if self.connected:
            # If Potentiostat class has a close/disconnect method, call it (best-effort)
            try:
                self.dev.close()
            except Exception:
                pass
            self.connected = False
            print("✓ Rodeostat disconnected.")

# =============================================================================
# RED PITAYA CONTROLLER (SCPI over TCP)
# =============================================================================

class RedPitayaController:
    """
    Controls a Red Pitaya over SCPI/TCP for:
      - setting AC waveform (sine)
      - turning output on/off
      - acquiring samples from channels

    NOTE: SCPI command strings may vary by Red Pitaya firmware.
    Adjust commands in _send() and dataset parsing if your unit uses different calls.
    """

    def __init__(self, ip=RP_IP, port=RP_PORT, timeout=2.0):
        self.ip = ip
        self.port = port
        self.sock = None
        self.timeout = timeout

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect((self.ip, self.port))
        except Exception as e:
            print("❌ Could not connect to Red Pitaya at", self.ip, "port", self.port, "-", e)
            return False
        self.sock = s
        # flush any initial banner
        try:
            self.sock.settimeout(0.2)
            _ = self.sock.recv(4096)
        except Exception:
            pass
        self.sock.settimeout(self.timeout)
        print(f"✓ Connected to Red Pitaya {self.ip}:{self.port}")
        return True

    def _send(self, cmd, expect_reply=True):
        """
        Send SCPI command and optionally return reply as str.
        IMPORTANT: adjust string commands here to match your Red Pitaya firmware if needed.
        """
        if not self.sock:
            raise Exception("Red Pitaya not connected.")
        if not cmd.endswith('\n'):
            cmd = cmd + '\n'
        # print("SCPI ->", cmd.strip())
        try:
            self.sock.sendall(cmd.encode('ascii'))
        except Exception as e:
            raise Exception("Failed to send SCPI command: " + str(e))

        if not expect_reply:
            return None

        # read reply (best-effort)
        try:
            data = b''
            # Wait briefly for data then read what is available
            time.sleep(0.02)
            while True:
                try:
                    chunk = self.sock.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                    # if small reply, break (SCPI generally returns together)
                    if len(chunk) < 4096:
                        break
                except socket.timeout:
                    break
            if not data:
                return ''
            return data.decode('ascii', errors='ignore').strip()
        except Exception as e:
            # Return empty string on read issues
            return ''

    def setup_ac(self, freq, amp, channel_out=RP_CHANNEL_OUT):
        """
        Configure the Red Pitaya to output a sine wave with given freq (Hz) and amp (V peak).
        May need command adjustments depending on your RP image.
        """
        # Example SCPI-like commands commonly accepted by many RP firmwares:
        # - set function to sine
        # - set frequency
        # - set amplitude (peak or peak-to-peak depends on implementation)
        # - enable output

        # NOTE: Some firmwares use 'SOUR1:FUNC SIN', 'SOUR1:FREQ', 'SOUR1:VOLT', 'OUTP:STATE ON'
        # If your RP uses a different prefix (SOUR: or GENERATOR:), adapt here.

        # target source text (SOUR1 or SOUR2)
        src = f"SOUR{channel_out}"

        # Set function
        self._send(f"{src}:FUNC SIN", expect_reply=False)
        # Set frequency
        self._send(f"{src}:FREQ {float(freq)}", expect_reply=False)
        # Set amplitude. Many implementations expect amplitude as peak (V) or Vrms — check your system.
        self._send(f"{src}:VOLT {float(amp)}", expect_reply=False)
        # Turn output on
        # Some firmwares use "OUTP:STATE ON" or "SOUR1:OUTP ON". We try both (harmless if one fails).
        self._send(f"{src}:OUTP:STATE ON", expect_reply=False)
        self._send(f"OUTP:STATE ON", expect_reply=False)
        # A tiny pause so output settles
        time.sleep(0.02)

    def stop_ac(self, channel_out=RP_CHANNEL_OUT):
        src = f"SOUR{channel_out}"
        self._send(f"{src}:OUTP:STATE OFF", expect_reply=False)
        self._send("OUTP:STATE OFF", expect_reply=False)

    def acquire_raw(self, channel, num_points=SAMPLE_AC_POINTS, timeout_s=2.0):
        """
        Acquire raw waveform data from `channel` (1 or 2). Returns numpy array of floats.
        Adjust the SCPI acquisition request below if your firmware uses a different syntax.
        """
        # Common commands:
        # ACQ:START
        # ACQ:TRIG:LEV 0
        # ACQ:DATA? CH1  (or ACQ:DATA? CH1,CH2)
        #
        # We'll try a common pattern and parse whitespace/comma-separated numbers.

        # start acquisition
        self._send("ACQ:START", expect_reply=False)
        # Give it time to collect
        time.sleep(min(timeout_s, AC_MEAS_WINDOW + 0.05))

        # Request channel data. Try a few variants until we get a reply
        variants = [
            f"ACQ:DATA? CH{channel}",        # some firmwares
            f"ACQ:DATA? CH{channel},COUNT {num_points}",
            f"ACQ:DATA? {channel}",          # some firmwares accept numeric channel
            f"ACQ:DATA:RAW? CH{channel}",
            f"ACQ:DATA? CH{channel},FORMAT ASCII"
        ]

        resp = ''
        for cmd in variants:
            try:
                resp = self._send(cmd)
            except Exception:
                resp = ''
            if resp and any(ch.isdigit() for ch in resp):
                break

        if not resp:
            # fallback: try generic 'ACQ:DATA?'
            resp = self._send("ACQ:DATA?")
        if not resp:
            print("⚠ No data received from Red Pitaya for channel", channel)
            return np.array([])

        # Parse response: remove common prefixes and split by commas/whitespace
        # Response may include header/metadata; extract numeric tokens
        tokens = []
        for part in resp.replace('\r', ' ').replace('\n', ' ').replace(',', ' ').split():
            try:
                tokens.append(float(part))
            except Exception:
                continue
        if not tokens:
            print("⚠ Could not parse numeric ADC data from Red Pitaya response (raw follows):")
            print(resp[:500])
            return np.array([])

        arr = np.array(tokens, dtype=float)
        return arr

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            print("✓ Red Pitaya disconnected.")

# =============================================================================
# MEASUREMENT SYSTEM (Hybrid)
# =============================================================================

class MeasurementSystem:
    def __init__(self):
        self.rodeostat = RodeostatController()
        self.redpitaya = RedPitayaController(ip=RP_IP, port=RP_PORT)
        self.data = []

    def connect_all(self):
        # Connect Rodeostat first (interactive)
        ok = self.rodeostat.connect()
        if not ok:
            return False
        # Connect Red Pitaya
        ok2 = self.redpitaya.connect()
        return ok2

    def run_measurement(self):
        # Build DC list
        if DC_ENABLE:
            if DC_SWEEP:
                dc_list = np.linspace(DC_START, DC_END, DC_STEPS)
            else:
                dc_list = [DC_FIXED]
        else:
            dc_list = [None]

        # Build AC list
        if AC_ENABLE:
            if AC_SWEEP:
                ac_list = np.logspace(np.log10(AC_START_FREQ), np.log10(AC_END_FREQ), AC_NUM_POINTS)
            else:
                ac_list = [AC_FIXED_FREQ]
        else:
            ac_list = [None]

        for bias in dc_list:
            # Set DC bias on Rodeostat
            if DC_ENABLE and bias is not None:
                print(f"\n--- Setting DC bias to {bias:.6f} V on Rodeostat ---")
                try:
                    self.rodeostat.set_dc_bias(float(bias))
                except Exception as e:
                    print("❌ Rodeostat error:", e)
                # Optionally read Rodeostat DC measurement (V/I)
                try:
                    vdc, idc = self.rodeostat.measure_dc()
                except Exception:
                    vdc, idc = None, None
            else:
                vdc, idc = None, None

            for freq in ac_list:
                if AC_ENABLE and freq is not None:
                    print(f"  -> AC: {freq:.3f} Hz, amp {AC_FIXED_AMP} V (peak)")

                    # Configure Red Pitaya for AC injection
                    try:
                        self.redpitaya.setup_ac(freq, AC_FIXED_AMP, channel_out=RP_CHANNEL_OUT)
                    except Exception as e:
                        print("❌ Red Pitaya setup error:", e)
                        vac, iac = None, None
                        # store and continue
                        self.data.append({
                            'DC_Bias': bias,
                            'AC_Freq': freq,
                            'V_DC': vdc,
                            'I_DC': idc,
                            'V_AC_out_rms': vac,
                            'I_AC_rms': iac
                        })
                        continue

                    # Wait for settle
                    time.sleep(SETTLE_TIME)

                    # Acquire from output channel (what RP is driving) and shunt channel (measures current)
                    out_samples = self.redpitaya.acquire_raw(RP_CHANNEL_OUT, num_points=SAMPLE_AC_POINTS, timeout_s=AC_MEAS_WINDOW + 0.5)
                    shunt_samples = self.redpitaya.acquire_raw(RP_CHANNEL_SHUNT, num_points=SAMPLE_AC_POINTS, timeout_s=AC_MEAS_WINDOW + 0.5)

                    vac = None
                    iac = None
                    if out_samples.size:
                        # Estimate RMS of output channel (assuming waveform centered at 0)
                        vac = np.sqrt(np.mean(np.square(out_samples)))
                    if shunt_samples.size:
                        v_shunt_rms = np.sqrt(np.mean(np.square(shunt_samples)))
                        iac = v_shunt_rms / SHUNT_RESISTANCE

                    # Stop AC output after measurement
                    try:
                        self.redpitaya.stop_ac(channel_out=RP_CHANNEL_OUT)
                    except Exception:
                        pass

                else:
                    vac, iac = None, None
                    freq = None

                # Store measurement
                self.data.append({
                    'DC_Bias': float(bias) if bias is not None else None,
                    'AC_Freq': float(freq) if freq is not None else None,
                    'V_DC': float(vdc) if vdc is not None else None,
                    'I_DC': float(idc) if idc is not None else None,
                    'V_AC_out_rms': float(vac) if vac is not None else None,
                    'I_AC_rms': float(iac) if iac is not None else None
                })

                print(f"    stored: DC={bias} V, AC={freq} Hz, V_AC_rms={vac}, I_AC_rms={iac}")

    def save_csv(self):
        if not self.data:
            print("No data to save.")
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fn = os.path.join(OUTPUT_DIR, f"measurement_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(fn, "w", newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.data[0].keys())
            w.writeheader()
            for row in self.data:
                w.writerow(row)
        print("✅ Data saved:", fn)

    def plot_results(self):
        if not self.data:
            print("No data to plot.")
            return

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Hybrid Rodeostat (DC) + Red Pitaya (AC) Measurement Results", fontsize=16)

        # DC I-V
        dc_points = [d for d in self.data if d['V_DC'] is not None and d['I_DC'] is not None]
        if dc_points:
            axs[0, 0].plot([d['DC_Bias'] for d in dc_points], [d['I_DC'] for d in dc_points], 'o-')
            axs[0, 0].set_xlabel("DC Bias (V)")
            axs[0, 0].set_ylabel("DC Current (A)")
            axs[0, 0].set_title("DC I-V (Rodeostat)")
            axs[0, 0].grid(True)
        else:
            axs[0,0].text(0.5,0.5,"No DC measurements", ha='center', va='center')

        # AC Voltage vs DC bias (for each frequency)
        ac_points = [d for d in self.data if d['V_AC_out_rms'] is not None]
        if ac_points:
            freqs = sorted(list(set(d['AC_Freq'] for d in ac_points)))
            for f in freqs:
                sub = [d for d in ac_points if d['AC_Freq'] == f]
                axs[0, 1].plot([d['DC_Bias'] for d in sub], [d['V_AC_out_rms'] for d in sub], 'o-', label=f"{f:.1f} Hz")
            axs[0, 1].set_xlabel("DC Bias (V)")
            axs[0, 1].set_ylabel("AC V_out (RMS, V)")
            axs[0, 1].set_title("AC V_out (RMS) vs DC Bias")
            axs[0, 1].legend()
            axs[0, 1].grid(True)
        else:
            axs[0,1].text(0.5,0.5,"No AC voltage data", ha='center', va='center')

        # AC Current vs DC bias
        if ac_points:
            for f in freqs:
                sub = [d for d in ac_points if d['AC_Freq'] == f]
                axs[1, 0].plot([d['DC_Bias'] for d in sub], [d['I_AC_rms'] for d in sub], 'o-', label=f"{f:.1f} Hz")
            axs[1, 0].set_xlabel("DC Bias (V)")
            axs[1, 0].set_ylabel("AC Current (RMS, A)")
            axs[1, 0].set_title("AC I (RMS) vs DC Bias")
            axs[1, 0].legend()
            axs[1, 0].grid(True)
        else:
            axs[1,0].text(0.5,0.5,"No AC current data", ha='center', va='center')

        # DC Voltage vs DC Bias (sanity)
        if dc_points:
            axs[1, 1].plot([d['DC_Bias'] for d in dc_points], [d['V_DC'] for d in dc_points], 'o-')
            axs[1, 1].set_xlabel("DC Bias (V)")
            axs[1, 1].set_ylabel("DC Voltage (V)")
            axs[1, 1].set_title("DC Voltage vs Bias")
            axs[1, 1].grid(True)
        else:
            axs[1,1].text(0.5,0.5,"No DC voltage data", ha='center', va='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def disconnect(self):
        try:
            self.rodeostat.disconnect()
        except Exception:
            pass
        try:
            self.redpitaya.close()
        except Exception:
            pass

# =============================================================================
# MAIN
# =============================================================================

def main():
    system = MeasurementSystem()
    if not system.connect_all():
        print("❌ Failed to connect to all devices. Aborting.")
        return
    try:
        system.run_measurement()
        system.save_csv()
        system.plot_results()
    finally:
        system.disconnect()

if __name__ == "__main__":
    main()
