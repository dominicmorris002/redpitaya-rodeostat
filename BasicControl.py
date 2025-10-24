"""
CV-EIS System: Rodeostat + Red Pitaya
======================================

Rodeostat: DC voltage output with CV (ramp) functionality
Red Pitaya: AC wave generation and comprehensive measurements
No lock-in amplifier - all measurements via Red Pitaya oscilloscope

REQUIRED PLOTS:
- Input DC over time (from Rodeostat)
- Input AC wave over time (from Red Pitaya)
- DC output current vs DC output voltage (from Red Pitaya)
- DC output current over time (from Red Pitaya)
- AC output current and AC output voltage overlapped over time (from Red Pitaya)
- Combined output current over time (from Red Pitaya)
- Combined voltage over time (from Red Pitaya)
- Combined AC wave superimposed on DC wave (from Red Pitaya)
"""

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

RED_PITAYA_IP = 'rp-f073ce.local'   # Your Red Pitaya IP address
OUTPUT_DIR = 'cv_eis_data'          # Output directory

# CV Parameters
CV_PARAMS = {
    'start_voltage': -0.2,          # Start voltage (V)
    'end_voltage': 0.8,             # End voltage (V)
    'scan_rate': 0.01,              # Scan rate (V/s)
    'num_cycles': 1                 # Number of CV cycles
}

# AC Wave Parameters
AC_PARAMS = {
    'frequency': 1000,              # AC frequency (Hz)
    'amplitude': 0.01,              # AC amplitude (V)
    'phase': 0                      # AC phase (degrees)
}

# Measurement Parameters
MEASUREMENT_PARAMS = {
    'sample_rate': 125000,          # Red Pitaya sample rate (Hz)
    'measurement_time': 10,         # Total measurement time (s)
    'settling_time': 0.1            # Settling time between measurements (s)
}

# =============================================================================
# IMPORTS
# =============================================================================

import os
import csv
import time
import logging
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

class RodeostatController:
    """Handles DC voltage output with CV (ramp) functionality"""
    def __init__(self):
        self.dev = None
        self.connected = False
        self.current_voltage = 0.0

    def connect(self):
        """Connect to Rodeostat via serial port"""
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
        try:
            self.dev = Potentiostat(port)
            self.connected = True
            print("✓ Rodeostat connected.")
            return True
        except Exception as e:
            print(f"❌ Rodeostat connection failed: {e}")
            return False

    def set_voltage(self, voltage):
        """Set DC voltage output"""
        if not self.connected:
            raise Exception("Rodeostat not connected.")
        try:
            self.dev.set_volt(voltage)
            self.current_voltage = voltage
            time.sleep(0.01)  # Allow voltage to stabilize
        except Exception as e:
            print(f"⚠ Failed to set voltage: {e}")

    def measure_current(self):
        """Measure current from Rodeostat"""
        if not self.connected:
            raise Exception("Rodeostat not connected.")
        try:
            data = self.dev.get_curr()
            return float(data)
        except Exception as e:
            print(f"⚠ Failed to measure current: {e}")
            return 0.0

    def disconnect(self):
        """Disconnect from Rodeostat"""
        if self.connected:
            self.connected = False
            print("✓ Rodeostat disconnected.")


class RedPitayaController:
    """Handles AC wave generation and comprehensive measurements"""
    def __init__(self, hostname, output_dir):
        self.hostname = hostname
        self.output_dir = output_dir
        self.rp = None
        self.connected = False
        self.scope = None
        self.asg = None
        self.sample_rate = MEASUREMENT_PARAMS['sample_rate']

    def connect(self):
        """Connect to Red Pitaya"""
        try:
            self.rp = Pyrpl(hostname=self.hostname)
            self.scope = self.rp.rp.scope
            self.asg = self.rp.rp.asg0
            
            # Configure oscilloscope
            self.scope.input1 = 'in1'  # Input 1
            self.scope.input2 = 'in2'  # Input 2
            self.scope.decimation = 1  # Maximum resolution
            self.scope.average = False
            self.scope.trigger_source = 'ch1_positive_edge'
            self.scope.trigger_level = 0.1
            
            self.connected = True
            print("✓ Red Pitaya connected.")
            return True
        except Exception as e:
            print(f"❌ Red Pitaya connection failed: {e}")
            return False

    def generate_ac_wave(self, frequency, amplitude, phase=0):
        """Generate AC sine wave"""
        if not self.connected:
            raise Exception("Red Pitaya not connected.")
        try:
            self.asg.setup(frequency=frequency, amplitude=amplitude, 
                          phase=phase, output_direct='out1', 
                          output_signal='sine')
            print(f"✓ AC wave generated: {frequency} Hz, {amplitude} V")
        except Exception as e:
            print(f"⚠ Failed to generate AC wave: {e}")

    def stop_ac_wave(self):
        """Stop AC wave generation"""
        if self.connected:
            try:
                self.asg.output_direct = 'off'
            except Exception as e:
                print(f"⚠ Failed to stop AC wave: {e}")

    def capture_waveforms(self, duration):
        """Capture waveforms from both channels"""
        if not self.connected:
            raise Exception("Red Pitaya not connected.")
        
        try:
            # Configure scope for capture
            self.scope.duration = duration
            self.scope.single()
            
            # Wait for acquisition to complete
            time.sleep(duration + 0.1)
            
            # Get data
            ch1_data = np.array(self.scope._data_ch1_current)
            ch2_data = np.array(self.scope._data_ch2_current)
            time_data = np.linspace(0, duration, len(ch1_data))
            
            return time_data, ch1_data, ch2_data
        except Exception as e:
            print(f"⚠ Failed to capture waveforms: {e}")
            return None, None, None

    def measure_impedance(self, frequency, amplitude):
        """Measure impedance using Red Pitaya"""
        if not self.connected:
            raise Exception("Red Pitaya not connected.")
        
        try:
            # Generate AC wave
            self.generate_ac_wave(frequency, amplitude)
            time.sleep(0.1)  # Allow settling
            
            # Capture waveforms
            time_data, ch1_data, ch2_data = self.capture_waveforms(0.1)
            
            if ch1_data is not None and ch2_data is not None:
                # Calculate RMS values
                ch1_rms = np.sqrt(np.mean(ch1_data**2))
                ch2_rms = np.sqrt(np.mean(ch2_data**2))
                
                # Calculate impedance (assuming ch1 is voltage, ch2 is current)
                if ch2_rms > 0:
                    impedance = ch1_rms / ch2_rms
                else:
                    impedance = np.inf
                
                return impedance, ch1_rms, ch2_rms
            else:
                return np.inf, 0, 0
        except Exception as e:
            print(f"⚠ Failed to measure impedance: {e}")
            return np.inf, 0, 0

    def disconnect(self):
        """Disconnect from Red Pitaya"""
        if self.connected:
            self.stop_ac_wave()
            self.connected = False
            print("✓ Red Pitaya disconnected.")


# =============================================================================
# MAIN CV-EIS SYSTEM
# =============================================================================

class CVEISSystem:
    def __init__(self, redpitaya_ip, output_dir):
        self.redpitaya_ip = redpitaya_ip
        self.output_dir = output_dir
        self.rodeostat = RodeostatController()
        self.redpitaya = RedPitayaController(redpitaya_ip, output_dir)
        self.data = []
        self.measurement_data = {}

    def connect_all(self):
        """Connect to all devices"""
        print("Connecting to devices...")
        r_ok = self.rodeostat.connect()
        p_ok = self.redpitaya.connect()
        return r_ok and p_ok

    def run_cv_scan(self):
        """Run CV scan with AC wave generation"""
        print("\n===== Starting CV-EIS Measurement =====")
        
        # Calculate CV parameters
        start_v = CV_PARAMS['start_voltage']
        end_v = CV_PARAMS['end_voltage']
        scan_rate = CV_PARAMS['scan_rate']
        num_cycles = CV_PARAMS['num_cycles']
        
        # Calculate time points
        voltage_range = end_v - start_v
        scan_time = voltage_range / scan_rate
        total_time = scan_time * num_cycles
        
        # Generate time array
        dt = 1.0 / MEASUREMENT_PARAMS['sample_rate']
        time_points = np.arange(0, total_time, dt)
        
        # Generate voltage ramp
        voltage_ramp = np.linspace(start_v, end_v, len(time_points))
        
        # Initialize data arrays
        dc_voltage_data = []
        dc_current_data = []
        ac_voltage_data = []
        ac_current_data = []
        combined_current_data = []
        combined_voltage_data = []
        time_data = []
        
        print(f"CV Scan: {start_v}V to {end_v}V at {scan_rate} V/s")
        print(f"Total time: {total_time:.2f} s")
        
        # Start AC wave generation
        self.redpitaya.generate_ac_wave(
            AC_PARAMS['frequency'], 
            AC_PARAMS['amplitude'], 
            AC_PARAMS['phase']
        )
        
        try:
            # Run CV scan
            for i, (t, v) in enumerate(zip(time_points, voltage_ramp)):
                # Set DC voltage
                self.rodeostat.set_voltage(v)
                
                # Measure DC current
                dc_current = self.rodeostat.measure_current()
                
                # Capture waveforms from Red Pitaya
                time_wf, ch1_wf, ch2_wf = self.redpitaya.capture_waveforms(0.01)
                
                if ch1_wf is not None and ch2_wf is not None:
                    # Calculate AC components
                    ac_voltage = np.sqrt(np.mean(ch1_wf**2))  # RMS AC voltage
                    ac_current = np.sqrt(np.mean(ch2_wf**2))  # RMS AC current
                    
                    # Combined signals
                    combined_current = dc_current + ac_current
                    combined_voltage = v + ac_voltage
                    
                    # Store data
                    dc_voltage_data.append(v)
                    dc_current_data.append(dc_current)
                    ac_voltage_data.append(ac_voltage)
                    ac_current_data.append(ac_current)
                    combined_current_data.append(combined_current)
                    combined_voltage_data.append(combined_voltage)
                    time_data.append(t)
                    
                    # Progress update
                    if i % 1000 == 0:
                        progress = (i / len(time_points)) * 100
                        print(f"Progress: {progress:.1f}% - V={v:.3f}V, I_DC={dc_current:.3e}A")
        
        finally:
            # Stop AC wave generation
            self.redpitaya.stop_ac_wave()
        
        # Store measurement data
        self.measurement_data = {
            'time': np.array(time_data),
            'dc_voltage': np.array(dc_voltage_data),
            'dc_current': np.array(dc_current_data),
            'ac_voltage': np.array(ac_voltage_data),
            'ac_current': np.array(ac_current_data),
            'combined_current': np.array(combined_current_data),
            'combined_voltage': np.array(combined_voltage_data)
        }
        
        print("✓ CV-EIS measurement completed.")

    def save_data(self):
        """Save measurement data to CSV"""
        os.makedirs(self.output_dir, exist_ok=True)
        filename = os.path.join(self.output_dir, f"cv_eis_{datetime.now():%Y%m%d_%H%M%S}.csv")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time (s)', 'DC Voltage (V)', 'DC Current (A)', 
                           'AC Voltage (V)', 'AC Current (A)', 
                           'Combined Current (A)', 'Combined Voltage (V)'])
            
            for i in range(len(self.measurement_data['time'])):
                writer.writerow([
                    self.measurement_data['time'][i],
                    self.measurement_data['dc_voltage'][i],
                    self.measurement_data['dc_current'][i],
                    self.measurement_data['ac_voltage'][i],
                    self.measurement_data['ac_current'][i],
                    self.measurement_data['combined_current'][i],
                    self.measurement_data['combined_voltage'][i]
                ])
        
        print(f"✓ Data saved to {filename}")

    def plot_all_graphs(self):
        """Create all required plots"""
        if not self.measurement_data:
            print("No data to plot.")
            return
        
        print("Creating plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle('CV-EIS Measurement Results', fontsize=16)
        
        # 1. Input DC over time (from Rodeostat)
        axes[0, 0].plot(self.measurement_data['time'], self.measurement_data['dc_voltage'])
        axes[0, 0].set_title('Input DC Voltage over Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('DC Voltage (V)')
        axes[0, 0].grid(True)
        
        # 2. Input AC wave over time (from Red Pitaya)
        axes[0, 1].plot(self.measurement_data['time'], self.measurement_data['ac_voltage'])
        axes[0, 1].set_title('Input AC Wave over Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('AC Voltage (V)')
        axes[0, 1].grid(True)
        
        # 3. DC output current vs DC output voltage
        axes[1, 0].plot(self.measurement_data['dc_voltage'], self.measurement_data['dc_current'])
        axes[1, 0].set_title('DC Output Current vs DC Output Voltage')
        axes[1, 0].set_xlabel('DC Voltage (V)')
        axes[1, 0].set_ylabel('DC Current (A)')
        axes[1, 0].grid(True)
        
        # 4. DC output current over time
        axes[1, 1].plot(self.measurement_data['time'], self.measurement_data['dc_current'])
        axes[1, 1].set_title('DC Output Current over Time')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('DC Current (A)')
        axes[1, 1].grid(True)
        
        # 5. AC output current and AC output voltage overlapped over time
        ax5 = axes[2, 0]
        ax5.plot(self.measurement_data['time'], self.measurement_data['ac_voltage'], 
                label='AC Voltage', color='blue')
        ax5.plot(self.measurement_data['time'], self.measurement_data['ac_current'], 
                label='AC Current', color='red')
        ax5.set_title('AC Output Current and Voltage over Time')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Amplitude')
        ax5.legend()
        ax5.grid(True)
        
        # 6. Combined output current over time
        axes[2, 1].plot(self.measurement_data['time'], self.measurement_data['combined_current'])
        axes[2, 1].set_title('Combined Output Current over Time')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Combined Current (A)')
        axes[2, 1].grid(True)
        
        # 7. Combined voltage over time
        axes[3, 0].plot(self.measurement_data['time'], self.measurement_data['combined_voltage'])
        axes[3, 0].set_title('Combined Voltage over Time')
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 0].set_ylabel('Combined Voltage (V)')
        axes[3, 0].grid(True)
        
        # 8. Combined AC wave superimposed on DC wave
        axes[3, 1].plot(self.measurement_data['time'], self.measurement_data['dc_voltage'], 
                       label='DC Component', color='blue', alpha=0.7)
        axes[3, 1].plot(self.measurement_data['time'], self.measurement_data['ac_voltage'], 
                       label='AC Component', color='red', alpha=0.7)
        axes[3, 1].plot(self.measurement_data['time'], self.measurement_data['combined_voltage'], 
                       label='Combined', color='green', linewidth=2)
        axes[3, 1].set_title('Combined AC Wave Superimposed on DC Wave')
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 1].set_ylabel('Voltage (V)')
        axes[3, 1].legend()
        axes[3, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Save plots
        plot_filename = os.path.join(self.output_dir, f"cv_eis_plots_{datetime.now():%Y%m%d_%H%M%S}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {plot_filename}")

    def disconnect_all(self):
        """Disconnect from all devices"""
        self.rodeostat.disconnect()
        self.redpitaya.disconnect()
        print("✓ All devices disconnected.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("CV-EIS System: Rodeostat + Red Pitaya")
    print("=" * 50)
    
    # Create system
    system = CVEISSystem(RED_PITAYA_IP, OUTPUT_DIR)
    
    # Connect to devices
    if not system.connect_all():
        print("❌ Connection failed. Exiting.")
        return
    
    try:
        # Run CV-EIS measurement
        system.run_cv_scan()
        
        # Save data
        system.save_data()
        
        # Create plots
        system.plot_all_graphs()
        
        print("\n✅ CV-EIS measurement completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Measurement interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during measurement: {e}")
    finally:
        # Disconnect from all devices
        system.disconnect_all()


if __name__ == "__main__":
    main()
