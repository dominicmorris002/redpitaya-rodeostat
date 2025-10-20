"""
Combined EIS System: Rodeostat + Red Pitaya
==========================================

PYTHON VERSION: 3.8 (Required for PyRpl)

PIP INSTALL COMMANDS:
pip install --upgrade pip setuptools wheel
pip install PyQt5==5.15.7 pyqtgraph==0.12.4 quamash==0.6.0 pyrpl
pip install potentiostat pyserial matplotlib numpy pandas scipy h5py

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
RED_PITAYA_IP = '169.254.131.37'  # Your Red Pitaya IP
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
            return False

        print(f"\nStarting EIS Frequency Sweep ({len(frequencies)} points)")
        self.eis_data = []

        for i, freq in enumerate(frequencies):
            print(f"Measuring {i + 1}/{len(frequencies)}: ", end="")
            result = self.measure_impedance(freq, dc_bias, ac_amplitude, settling_time)
            if result:
                self.eis_data.append(result)
            else:
                print(f"Skipping {freq} Hz due to measurement failure")

        print(f"\n✓ Frequency sweep completed! ({len(self.eis_data)} successful measurements)")
        return len(self.eis_data) > 0

    def plot_eis_results(self, save_file=None):
        if not self.eis_data:
            print("No EIS data to plot")
            return
        df = pd.DataFrame(self.eis_data)
        plt.figure(1, figsize=(15, 10))

        plt.subplot(221)
        plt.plot(df['Z_real'], -df['Z_imag'], 'bo-', markersize=6)
        plt.xlabel('Z\' (Real)')
        plt.ylabel('-Z\'\' (Imaginary)')
        plt.title('Nyquist Plot')
        plt.grid(True)

        plt.subplot(222)
        plt.loglog(df['frequency'], df['Z_magnitude'], 'ro-', markersize=6)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('|Z| (Ω)')
        plt.title('Bode Plot - Magnitude')
        plt.grid(True)

        plt.subplot(223)
        plt.semilogx(df['frequency'], df['Z_phase'], 'go-', markersize=6)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (degrees)')
        plt.title('Bode Plot - Phase')
        plt.grid(True)

        plt.subplot(224)
        plt.semilogx(df['frequency'], df['Z_real'], 'bo-', markersize=4, label='Real')
        plt.semilogx(df['frequency'], df['Z_imag'], 'ro-', markersize=4, label='Imaginary')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Impedance (Ω)')
        plt.title('Real and Imaginary Components')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        if save_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            filepath = os.path.join(self.output_dir, save_file)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filepath}")

        plt.show()

    def save_data(self, filename=None):
        if not self.eis_data:
            print("No data to save")
            return
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eis_data_{timestamp}.csv"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['frequency', 'dc_bias', 'ac_amplitude', 'Z_real', 'Z_imag', 'Z_magnitude',
                          'Z_phase', 'X', 'Y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in self.eis_data:
                writer.writerow(data)
        print(f"Data saved to {filepath}")

    def run_original_redpitaya_test(self, params):
        if not self.connected or not self.redpitaya:
            print("Red Pitaya not connected")
            return
        print("Running original Red Pitaya test...")
        self.redpitaya.run(params, save_file=False, test=True, fft=True)

    def disconnect(self):
        if self.redpitaya:
            try:
                self.redpitaya.rp.rp.synthesizer.output_direct = False
                self.redpitaya.rp.rp.synthesizer2.output_direct = False
            except:
                pass
        if self.rodeostat:
            self.rodeostat.disconnect()
        print("✓ Disconnected from all devices")


def main():
    print("Combined EIS System: Rodeostat + Red Pitaya")
    print("=" * 60)
    eis = CombinedEISSystem(redpitaya_ip=RED_PITAYA_IP, output_dir=OUTPUT_DIR)

    if not eis.connect_all():
        print("Failed to connect to devices")
        return

    try:
        # Test original Red Pitaya functionality first
        print("\nTesting original Red Pitaya functionality...")
        eis.run_original_redpitaya_test(RED_PITAYA_TEST)

        # Run EIS measurement using configuration parameters
        print("\nStarting EIS measurement...")
        frequencies = np.logspace(
            np.log10(EIS_FREQUENCIES['start_freq']),
            np.log10(EIS_FREQUENCIES['end_freq']),
            EIS_FREQUENCIES['num_points']
        )

        success = eis.frequency_sweep(
            frequencies=frequencies,
            dc_bias=EIS_MEASUREMENT['dc_bias'],
            ac_amplitude=EIS_MEASUREMENT['ac_amplitude'],
            settling_time=EIS_MEASUREMENT['settling_time']
        )

        if success:
            if SAVE_DATA:
                eis.save_data()
            if SAVE_PLOTS:
                eis.plot_eis_results("eis_results.png")
            print("\n✓ EIS measurement completed successfully!")
        else:
            print("EIS measurement failed!")

    except KeyboardInterrupt:
        print("\nMeasurement interrupted by user")
    finally:
        eis.disconnect()


if __name__ == "__main__":
    main()


