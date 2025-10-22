"""
Combined EIS System: Rodeostat + Red Pitaya
==========================================

PYTHON VERSION: 3.8 (Required for PyRpl)

PIP INSTALL COMMANDS:
pip install --upgrade pip setuptools wheel
pip install bcrypt==5.0.0 cffi==1.17.1 contourpy==1.1.1 cryptography==46.0.3 cycler==0.12.1 fonttools==4.57.0 futures==3.0.5 h5py==3.11.0 importlib_resources==6.4.5 kiwisolver==1.4.7 matplotlib==3.7.5 nose==1.3.7 numpy==1.23.5 packaging==25.0 pandas==2.0.3 paramiko==3.5.1 pillow==10.4.0 potentiostat==0.0.4 progressbar2==4.5.0 pycparser==2.23 PyNaCl==1.6.0 pyparsing==3.1.4 PyQt5==5.15.7 PyQt5-Qt5==5.15.2 PyQt5_sip==12.15.0 pyqtgraph==0.12.4 pyrpl==0.9.3.6 pyserial==3.5 python-dateutil==2.9.0.post0 python-utils==3.8.2 pytz==2025.2 PyYAML==6.0.3 qasync==0.28.0 QtPy==2.4.3 Quamash==0.6.1 scipy==1.10.1 scp==0.15.0 six==1.17.0 typing_extensions==4.13.2 tzdata==2025.2 zipp==3.20.2

SETUP:
1. Connect Rodeostat via USB
2. Connect Red Pitaya via Ethernet
3. Update IP address in code
4. Run EIS measurements

SYSTEM ARCHITECTURE:
- Rodeostat: Provides DC bias voltage
- Red Pitaya: Generates AC signal + Lock-in measurement
- Combined: Full EIS measurement
"""

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Red Pitaya Configuration
RED_PITAYA_IP = 'rp-f073ce.local'  # Your Red Pitaya IP
RED_PITAYA_OUTPUT_DIR = 'eis_data'

# Rodeostat Configuration
RODEOSTAT_PORT = None  # Leave None for auto-detect or set specific port (e.g., 'COM3')

# EIS Measurement Parameters
EIS_FREQUENCIES = {
    'start_freq': 10,      # Starting frequency (Hz)
    'end_freq': 10000,     # Ending frequency (Hz)
    'num_points': 15       # Number of frequency points
}

EIS_MEASUREMENT = {
    'dc_bias': 0.0,        # DC bias voltage (V)
    'ac_amplitude': 0.01,  # AC amplitude (V)
    'settling_time': 1.0   # Settling time per frequency (s)
}

# Red Pitaya Test Parameters (for original functionality test)
RED_PITAYA_TEST = {
    'test_freq': 1000,     # Test frequency (Hz)
    'test_amp': 0.1,       # Test amplitude (V)
    'noise_freq': 10000,   # Noise frequency (Hz)
    'noise_amp': 0.01,     # Noise amplitude (V)
    'ref_freq': 1000,      # Reference frequency (Hz)
    'ref_amp': 0.1         # Reference amplitude (V)
}

# Data Output Configuration
OUTPUT_DIR = 'eis_data'
SAVE_PLOTS = True
SAVE_DATA = True

# =============================================================================
# END CONFIGURATION - DO NOT MODIFY CODE BELOW THIS LINE
# =============================================================================

import sys
from qtpy import QtCore

# Monkeypatch QtCore to add pyqtBoundSignal attribute if missing
if not hasattr(QtCore, 'pyqtBoundSignal'):
    try:
        # Attempt to get pyqtBoundSignal from PyQt5.QtCore
        from PyQt5.QtCore import QObject
        QtCore.pyqtBoundSignal = type(QObject().destroyed)  # A real Qt signal type
    except ImportError:
        # Fallback: create dummy type to avoid error (not ideal but stops crash)
        class DummySignalType:
            pass
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
import pandas as pd
import logging

# Suppress PyRPL debug logs
logging.getLogger("pyrpl").setLevel(logging.WARNING)


class RedPitaya:
    """Red Pitaya controller using original code structure"""

    electrode_map = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True), '100uA': (True, False, True, True),
                         '1mA': (True, True, False, True), '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True), '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}

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
        noise_freq = params['noise_freq']
        noise_amp = params['noise_amp']

        self.test_sig = self.rp_modules.asg0
        self.test_noise = self.rp_modules.asg1

        self.test_sig.setup(waveform='sin',
                            frequency=test_freq,
                            amplitude=test_amp,
                            offset=0.00,
                            output_direct='out1',
                            trigger_source='immediately')
        self.test_noise.setup(waveform='sin',
                              frequency=noise_freq,
                              amplitude=noise_amp,
                              offset=0.00,
                              output_direct='out1',
                              trigger_source='immediately')

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        ref_amp = params['ref_amp']

        self.iq2.setup(frequency=self.ref_freq,
                       bandwidth=[-self.ref_freq * 2, -self.ref_freq, self.ref_freq, self.ref_freq * 2],
                       gain=1.0,
                       phase=0,
                       acbandwidth=0,
                       amplitude=ref_amp,
                       input='in1',
                       output_direct='out2',
                       output_signal='quadrature',
                       quadrature_factor=20)

    def capture(self):
        self.lia_scope.single()
        ch1 = np.array(self.lia_scope._data_ch1_current)
        ch2 = np.array(self.lia_scope._data_ch2_current)
        return ch1, ch2

    def see_fft(self, save_file=False):
        self.lia_scope.input1 = 'in2'
        self.lia_scope.single()
        data_in = np.array(self.lia_scope._data_ch1_current)
        N = len(data_in)
        freqs = np.fft.rfftfreq(N, 1 / self.sample_rate)
        fft_in = np.fft.rfft(data_in * np.hanning(N))
        psd_in = (np.abs(fft_in) ** 2) / (self.sample_rate * N)

        self.lia_scope.input1 = 'iq2'
        time.sleep(0.01)
        X, Y = self.capture()

        iq = X + 1j * Y
        N_lock = len(iq)
        win = np.hanning(N)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(N_lock, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))

        idx = np.argmax(psd_lock)
        print("Peak at", freqs_lock[idx], "Hz   (expected difference =", abs(self.test_freq - self.ref_freq), "Hz)")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].semilogy(freqs, psd_in, label='Input IN2')
        ax[0].axvline(self.ref_freq, color='orange', linestyle='--', label='Reference')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Power (a.u.)')
        ax[0].set_title('Input Signal Spectrum')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].semilogy(freqs_lock, psd_lock, label='Lock-in R')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power (a.u.)')
        ax[1].set_title('Lock-in Output Spectrum (baseband)')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        if save_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            path = os.path.join(self.output_dir, f'lockin_FFT_tf_{self.test_freq}_rf_{self.ref_freq}_w_noise.png')
            plt.savefig(path)
        else:
            plt.show()

    def run(self, params, save_file=False, test=True, fft=True):
        if test:
            self.test_freq = params['test_freq']
            self.setup_test_sig(params)
        self.setup_lockin(params)
        time.sleep(0.01)

        if fft:
            self.see_fft(save_file=save_file)
        else:
            X, Y = self.capture()
            t = np.arange(start=0, stop=len(X) / self.sample_rate, step=1 / self.sample_rate)
            plt.plot(t, X)
            plt.plot(t, Y)
            plt.title('Lockin Results')
            plt.xlabel('Time (s)')
            plt.ylabel('X and Y outputs')

            if save_file:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                path = os.path.join(self.output_dir,
                                    f'lockin_results_tf_{self.test_freq}_rf_{self.ref_freq}_w_noise.png')
                plt.savefig(path)
            else:
                plt.show()


class RodeostatController:
    """Rodeostat controller using original code structure"""

    def __init__(self, port=None):
        self.dev = None
        self.connected = False
        self.port = port

    def connect(self):
        """Connect to Rodeostat"""
        try:
            ports = serial.tools.list_ports.comports()
            if not ports:
                print("Warning: No serial ports found. Rodeostat will not be connected.")
                return False

            if self.port is None:
                print("Available COM ports:")
                for i, p in enumerate(ports):
                    print(f"{i+1}: {p.device}")
                choice = input("Select Rodeostat port by number (or press Enter for first port): ")
                if choice.strip() == '':
                    port = ports[0].device
                else:
                    idx = int(choice) - 1
                    port = ports[idx].device
            else:
                port = self.port

            print("Connecting to Rodeostat on", port)
            self.dev = Potentiostat(port)

            try:
                _ = self.dev.get_all_curr_range()
            except KeyError:
                print("Unknown firmware. Adding current range list manually.")
                self.dev.hw_variant = 'manual_patch'
                self.dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

            self.connected = True
            print("✓ Rodeostat connected successfully!")
            return True

        except Exception as e:
            print(f"Rodeostat connection failed: {e}")
            return False

    def set_dc_voltage(self, voltage):
        if not self.connected:
            print("Rodeostat not connected")
            return False
        print(f"Setting DC voltage to {voltage} V")
        # Implement the actual command to set DC voltage here if supported
        return True

    def disconnect(self):
        if self.connected:
            self.connected = False
            print("✓ Disconnected from Rodeostat")


class CombinedEISSystem:
    """Combined EIS system using both original code structures"""

    def __init__(self, redpitaya_ip='169.254.131.37', output_dir='eis_data'):
        self.output_dir = output_dir
        self.redpitaya_ip = redpitaya_ip

        self.rodeostat = RodeostatController(port=RODEOSTAT_PORT)
        self.redpitaya = None
        self.connected = False
        self.eis_data = []

    def connect_all(self):
        print("Initializing Combined EIS System...")
        print("=" * 50)

        rodeostat_ok = self.rodeostat.connect()

        try:
            print(f"Connecting to Red Pitaya at {self.redpitaya_ip}...")
            self.redpitaya = RedPitaya(hostname=self.redpitaya_ip, output_dir=self.output_dir)
            print("✓ Red Pitaya connected successfully!")
            redpitaya_ok = True
        except Exception as e:
            print(f"Red Pitaya connection failed: {e}")
            redpitaya_ok = False

        self.connected = rodeostat_ok and redpitaya_ok
        return self.connected

    def measure_impedance(self, frequency, dc_bias=0.0, ac_amplitude=0.01, settling_time=1.0):
        if not self.connected:
            print("System not connected")
            return None

        self.rodeostat.set_dc_voltage(dc_bias)
        params = {
            'test_freq': frequency,
            'test_amp': ac_amplitude,
            'noise_freq': frequency * 10,
            'noise_amp': ac_amplitude * 0.1,
            'ref_freq': frequency,
            'ref_amp': ac_amplitude
        }
        self.redpitaya.setup_test_sig(params)
        self.redpitaya.setup_lockin(params)
        time.sleep(settling_time)

        X, Y = self.redpitaya.capture()
        magnitude = np.sqrt(np.mean(X ** 2 + Y ** 2))
        phase = np.degrees(np.arctan2(np.mean(Y), np.mean(X)))
        Z_real = magnitude * np.cos(np.radians(phase))
        Z_imag = magnitude * np.sin(np.radians(phase))

        result = {'frequency': frequency, 'dc_bias': dc_bias, 'ac_amplitude': ac_amplitude,
                  'Z_real': Z_real, 'Z_imag': Z_imag, 'Z_magnitude': magnitude, 'Z_phase': phase,
                  'X': np.mean(X), 'Y': np.mean(Y)}
        print(f"f={frequency:8.1f} Hz | |Z|={magnitude:8.4f} | θ={phase:6.1f}°")
        return result

    def frequency_sweep(self, frequencies, dc_bias=0.0, ac_amplitude=0.01, settling_time=1.0):
        if not self.connected:
            print("System not connected")
            return []

        results = []
        for freq in frequencies:
            res = self.measure_impedance(freq, dc_bias, ac_amplitude, settling_time)
            if res:
                results.append(res)
        self.eis_data = results
        return results

    def save_data_csv(self, filename=None):
        if not self.eis_data:
            print("No data to save")
            return False

        if filename is None:
            filename = f'eis_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filepath = os.path.join(self.output_dir, filename)

        keys = list(self.eis_data[0].keys())
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.eis_data:
                writer.writerow(row)

        print(f"Data saved to {filepath}")
        return True

    def plot_results(self):
        if not self.eis_data:
            print("No data to plot")
            return

        frequencies = [d['frequency'] for d in self.eis_data]
        magnitudes = [d['Z_magnitude'] for d in self.eis_data]
        phases = [d['Z_phase'] for d in self.eis_data]

        fig, ax1 = plt.subplots()
        ax1.set_xscale('log')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('|Z| (Ohm)', color='blue')
        ax1.plot(frequencies, magnitudes, 'b.-')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, which='both', ls='--')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Phase (degrees)', color='red')
        ax2.plot(frequencies, phases, 'r.-')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('EIS Measurement Results')
        plt.show()

    def disconnect(self):
        if self.redpitaya:
            try:
                self.redpitaya.rp_modules.asg0.output_direct = False
                self.redpitaya.rp_modules.asg1.output_direct = False
            except Exception:
                pass
        if self.rodeostat:
            self.rodeostat.disconnect()
        print("✓ Disconnected from all devices")


def main():
    system = CombinedEISSystem(redpitaya_ip=RED_PITAYA_IP, output_dir=OUTPUT_DIR)
    if not system.connect_all():
        print("Failed to connect all devices. Exiting.")
        return

    start_freq = EIS_FREQUENCIES['start_freq']
    end_freq = EIS_FREQUENCIES['end_freq']
    num_points = EIS_FREQUENCIES['num_points']

    frequencies = np.logspace(np.log10(start_freq), np.log10(end_freq), num_points)

    dc_bias = EIS_MEASUREMENT['dc_bias']
    ac_amplitude = EIS_MEASUREMENT['ac_amplitude']
    settling_time = EIS_MEASUREMENT['settling_time']

    print(f"Starting frequency sweep from {start_freq} Hz to {end_freq} Hz ({num_points} points)")

    results = system.frequency_sweep(frequencies, dc_bias=dc_bias, ac_amplitude=ac_amplitude, settling_time=settling_time)

    if results:
        system.save_data_csv()
        system.plot_results()

    system.disconnect()


if __name__ == '__main__':
    main()
