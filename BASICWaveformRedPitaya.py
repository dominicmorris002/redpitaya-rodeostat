"""
Simple Red Pitaya Control - Fixed Version
=========================================

Basic Red Pitaya control without configuration files
- Direct connection without PyRPL config
- Fixed freezing issue
- Simple AC generation and measurement
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import csv
import os
import socket

# Red Pitaya configuration
RED_PITAYA_IP = 'rp-f073ce.local'  # Change this to your Red Pitaya IP
OUTPUT_DIR = 'redpitaya_data'

class SimpleRedPitaya:
    def __init__(self, ip_address):
        self.ip = ip_address
        self.rp = None
        self.connected = False
        self.data = []
        
    def connect(self):
        """Connect to Red Pitaya without configuration file"""
        try:
            from pyrpl import Pyrpl
            print(f"Connecting to Red Pitaya at {self.ip}...")
            
            # Connect without config file to avoid the configuration window
            self.rp = Pyrpl(hostname=self.ip, config=None)
            self.connected = True
            print("✓ Connected to Red Pitaya")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("Trying alternative connection method...")
            return self.connect_alternative()
    
    def connect_alternative(self):
        """Alternative connection method using direct socket"""
        try:
            # Test if Red Pitaya is reachable
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.ip, 22))  # SSH port
            sock.close()
            
            if result == 0:
                print("✓ Red Pitaya is reachable")
                # Try to connect with minimal config
                from pyrpl import Pyrpl
                self.rp = Pyrpl(hostname=self.ip, config={})
                self.connected = True
                print("✓ Connected to Red Pitaya (alternative method)")
                return True
            else:
                print("❌ Red Pitaya not reachable")
                return False
        except Exception as e:
            print(f"❌ Alternative connection failed: {e}")
            return False
    
    def setup_scope(self):
        """Setup oscilloscope for measurements"""
        if not self.connected:
            print("Not connected to Red Pitaya")
            return False
        
        try:
            # Configure oscilloscope with minimal settings
            scope = self.rp.rp.scope
            scope.input1 = 'in1'
            scope.input2 = 'in2'
            scope.decimation = 1
            scope.average = False
            print("✓ Oscilloscope configured")
            return True
        except Exception as e:
            print(f"❌ Scope setup failed: {e}")
            return False
    
    def generate_ac_wave(self, frequency, amplitude, phase=0):
        """Generate AC sine wave on output 1"""
        if not self.connected:
            print("Not connected to Red Pitaya")
            return False
        
        try:
            # Use ASG for AC wave generation
            asg = self.rp.rp.asg0
            asg.setup(
                frequency=frequency,
                amplitude=amplitude,
                phase=phase,
                output_direct='out1',
                output_signal='sine'
            )
            print(f"✓ AC wave generated: {frequency} Hz, {amplitude} V")
            return True
        except Exception as e:
            print(f"❌ AC wave generation failed: {e}")
            return False
    
    def stop_ac_wave(self):
        """Stop AC wave generation"""
        if self.connected:
            try:
                self.rp.rp.asg0.output_direct = 'off'
                print("✓ AC wave stopped")
            except Exception as e:
                print(f"⚠ Failed to stop AC wave: {e}")
    
    def capture_measurement(self, duration=0.1):
        """Capture waveforms and calculate measurements"""
        if not self.connected:
            print("Not connected to Red Pitaya")
            return None
        
        try:
            # Configure scope for capture
            scope = self.rp.rp.scope
            scope.duration = duration
            scope.single()
            
            # Wait for acquisition with timeout
            max_wait = duration + 1.0
            start_wait = time.time()
            
            while time.time() - start_wait < max_wait:
                try:
                    # Check if data is ready
                    if hasattr(scope, '_data_ch1_current') and scope._data_ch1_current:
                        break
                except:
                    pass
                time.sleep(0.01)
            
            # Get waveform data
            ch1_data = np.array(scope._data_ch1_current)
            ch2_data = np.array(scope._data_ch2_current)
            
            if len(ch1_data) == 0 or len(ch2_data) == 0:
                print("⚠ No data captured")
                return None
            
            time_data = np.linspace(0, duration, len(ch1_data))
            
            # Calculate measurements
            timestamp = datetime.now()
            
            # RMS values
            ch1_rms = np.sqrt(np.mean(ch1_data**2))
            ch2_rms = np.sqrt(np.mean(ch2_data**2))
            
            # Peak-to-peak values
            ch1_pp = np.max(ch1_data) - np.min(ch1_data)
            ch2_pp = np.max(ch2_data) - np.min(ch2_data)
            
            # Simple phase calculation
            if len(ch1_data) > 10 and len(ch2_data) > 10:
                # Find zero crossings for phase calculation
                ch1_zeros = np.where(np.diff(np.sign(ch1_data)))[0]
                ch2_zeros = np.where(np.diff(np.sign(ch2_data)))[0]
                
                if len(ch1_zeros) > 0 and len(ch2_zeros) > 0:
                    # Calculate phase difference
                    phase_samples = ch2_zeros[0] - ch1_zeros[0] if len(ch2_zeros) > 0 else 0
                    phase_rad = 2 * np.pi * phase_samples / len(ch1_data)
                    phase_deg = np.degrees(phase_rad)
                else:
                    phase_deg = 0
            else:
                phase_deg = 0
            
            # Create measurement record
            measurement = {
                'timestamp': timestamp,
                'time': time_data,
                'ch1_voltage': ch1_data,
                'ch2_voltage': ch2_data,
                'ch1_rms': ch1_rms,
                'ch2_rms': ch2_rms,
                'ch1_pp': ch1_pp,
                'ch2_pp': ch2_pp,
                'phase_deg': phase_deg,
                'duration': duration
            }
            
            return measurement
            
        except Exception as e:
            print(f"❌ Measurement failed: {e}")
            return None
    
    def run_continuous_measurement(self, frequency, amplitude, duration=10, sample_interval=0.5):
        """Run continuous measurements with better error handling"""
        print(f"Starting continuous measurement for {duration} seconds...")
        print(f"AC Wave: {frequency} Hz, {amplitude} V")
        print(f"Sample interval: {sample_interval} s")
        
        # Start AC wave
        if not self.generate_ac_wave(frequency, amplitude):
            return False
        
        try:
            start_time = time.time()
            measurement_count = 0
            last_measurement_time = 0
            
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Check if it's time for next measurement
                if current_time - last_measurement_time >= sample_interval:
                    print(f"Taking measurement {measurement_count + 1}...")
                    
                    # Capture measurement
                    measurement = self.capture_measurement(0.1)
                    
                    if measurement:
                        self.data.append(measurement)
                        measurement_count += 1
                        
                        # Print current status
                        print(f"  CH1 RMS={measurement['ch1_rms']:.4f}V, "
                              f"CH2 RMS={measurement['ch2_rms']:.4f}V, "
                              f"Phase={measurement['phase_deg']:.1f}°")
                    else:
                        print("  ⚠ Measurement failed, retrying...")
                    
                    last_measurement_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
            
            print(f"✓ Completed {measurement_count} measurements")
            return True
            
        except KeyboardInterrupt:
            print("\n⚠ Measurement interrupted by user.")
            return False
        except Exception as e:
            print(f"❌ Measurement error: {e}")
            return False
        finally:
            # Stop AC wave
            self.stop_ac_wave()
    
    def save_data(self):
        """Save measurement data to CSV"""
        if not self.data:
            print("No data to save")
            return
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"redpitaya_data_{datetime.now():%Y%m%d_%H%M%S}.csv")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'CH1_RMS_V', 'CH2_RMS_V', 'CH1_PP_V', 'CH2_PP_V', 'Phase_Deg'])
            
            for measurement in self.data:
                writer.writerow([
                    measurement['timestamp'],
                    measurement['ch1_rms'],
                    measurement['ch2_rms'],
                    measurement['ch1_pp'],
                    measurement['ch2_pp'],
                    measurement['phase_deg']
                ])
        
        print(f"✓ Data saved to {filename}")
        return filename
    
    def plot_data(self):
        """Create plots of the measurement data"""
        if not self.data:
            print("No data to plot")
            return
        
        # Extract data for plotting
        timestamps = [m['timestamp'] for m in self.data]
        ch1_rms = [m['ch1_rms'] for m in self.data]
        ch2_rms = [m['ch2_rms'] for m in self.data]
        phase_deg = [m['phase_deg'] for m in self.data]
        
        # Create time axis (seconds from start)
        start_time = timestamps[0]
        time_axis = [(t - start_time).total_seconds() for t in timestamps]
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: RMS voltages over time
        axes[0].plot(time_axis, ch1_rms, 'b-', label='CH1 (Input)', linewidth=2)
        axes[0].plot(time_axis, ch2_rms, 'r-', label='CH2 (Output)', linewidth=2)
        axes[0].set_title('RMS Voltages over Time')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('RMS Voltage (V)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Phase over time
        axes[1].plot(time_axis, phase_deg, 'g-', linewidth=2)
        axes[1].set_title('Phase Difference over Time')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Phase (degrees)')
        axes[1].grid(True)
        
        # Plot 3: CH1 vs CH2 (scatter plot)
        axes[2].scatter(ch1_rms, ch2_rms, c=phase_deg, cmap='viridis', alpha=0.7)
        axes[2].set_title('CH2 vs CH1 (colored by phase)')
        axes[2].set_xlabel('CH1 RMS (V)')
        axes[2].set_ylabel('CH2 RMS (V)')
        axes[2].grid(True)
        
        # Add colorbar for phase
        cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
        cbar.set_label('Phase (degrees)')
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plot_filename = os.path.join(OUTPUT_DIR, f"redpitaya_plots_{datetime.now():%Y%m%d_%H%M%S}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {plot_filename}")
    
    def disconnect(self):
        """Disconnect from Red Pitaya"""
        if self.connected:
            self.stop_ac_wave()
            self.connected = False
            print("✓ Disconnected from Red Pitaya")


def main():
    """Main function"""
    print("Simple Red Pitaya Control - Fixed Version")
    print("=" * 40)
    
    # Create Red Pitaya instance
    rp = SimpleRedPitaya(RED_PITAYA_IP)
    
    # Connect
    if not rp.connect():
        print("Failed to connect. Exiting.")
        return
    
    # Setup scope
    if not rp.setup_scope():
        print("Failed to setup scope. Exiting.")
        return
    
    try:
        # Configuration
        frequency = 1000  # Hz
        amplitude = 0.1   # V
        duration = 10     # seconds
        sample_interval = 1.0  # seconds (increased to prevent freezing)
        
        print(f"\nConfiguration:")
        print(f"  Frequency: {frequency} Hz")
        print(f"  Amplitude: {amplitude} V")
        print(f"  Duration: {duration} s")
        print(f"  Sample interval: {sample_interval} s")
        
        # Run measurement
        rp.run_continuous_measurement(frequency, amplitude, duration, sample_interval)
        
        # Save and plot data
        rp.save_data()
        rp.plot_data()
        
        print("\n✅ Measurement completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Measurement interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        rp.disconnect()


if __name__ == "__main__":
    main()
