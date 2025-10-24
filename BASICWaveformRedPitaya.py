"""
Simple Red Pitaya Control
=========================

Basic Red Pitaya control for:
- AC wave generation
- Input/output measurement
- Phase measurement
- Timestamps and data logging

No lock-in amplifier - just basic oscilloscope measurements
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import csv
import os

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
        """Connect to Red Pitaya"""
        try:
            from pyrpl import Pyrpl
            print(f"Connecting to Red Pitaya at {self.ip}...")
            self.rp = Pyrpl(hostname=self.ip)
            self.connected = True
            print("✓ Connected to Red Pitaya")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def setup_scope(self):
        """Setup oscilloscope for measurements"""
        if not self.connected:
            print("Not connected to Red Pitaya")
            return False
        
        try:
            # Configure oscilloscope
            self.rp.rp.scope.input1 = 'in1'  # Input 1
            self.rp.rp.scope.input2 = 'in2'  # Input 2
            self.rp.rp.scope.decimation = 1  # Maximum resolution
            self.rp.rp.scope.average = False
            self.rp.rp.scope.trigger_source = 'ch1_positive_edge'
            self.rp.rp.scope.trigger_level = 0.1
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
            # Use ASG (Arbitrary Signal Generator) for AC wave
            self.rp.rp.asg0.setup(
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
            self.rp.rp.scope.duration = duration
            self.rp.rp.scope.single()
            
            # Wait for acquisition
            time.sleep(duration + 0.1)
            
            # Get waveform data
            ch1_data = np.array(self.rp.rp.scope._data_ch1_current)
            ch2_data = np.array(self.rp.rp.scope._data_ch2_current)
            time_data = np.linspace(0, duration, len(ch1_data))
            
            # Calculate measurements
            timestamp = datetime.now()
            
            # RMS values
            ch1_rms = np.sqrt(np.mean(ch1_data**2))
            ch2_rms = np.sqrt(np.mean(ch2_data**2))
            
            # Peak-to-peak values
            ch1_pp = np.max(ch1_data) - np.min(ch1_data)
            ch2_pp = np.max(ch2_data) - np.min(ch2_data)
            
            # Phase calculation (simple cross-correlation method)
            if len(ch1_data) > 0 and len(ch2_data) > 0:
                # Normalize signals
                ch1_norm = ch1_data / np.max(np.abs(ch1_data)) if np.max(np.abs(ch1_data)) > 0 else ch1_data
                ch2_norm = ch2_data / np.max(np.abs(ch2_data)) if np.max(np.abs(ch2_data)) > 0 else ch2_data
                
                # Calculate phase difference
                correlation = np.correlate(ch1_norm, ch2_norm, mode='full')
                max_corr_idx = np.argmax(correlation)
                phase_samples = max_corr_idx - len(ch1_data) + 1
                phase_rad = 2 * np.pi * phase_samples / len(ch1_data)
                phase_deg = np.degrees(phase_rad)
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
    
    def run_continuous_measurement(self, frequency, amplitude, duration=10, sample_interval=0.1):
        """Run continuous measurements"""
        print(f"Starting continuous measurement for {duration} seconds...")
        print(f"AC Wave: {frequency} Hz, {amplitude} V")
        print(f"Sample interval: {sample_interval} s")
        
        # Start AC wave
        if not self.generate_ac_wave(frequency, amplitude):
            return False
        
        try:
            start_time = time.time()
            measurement_count = 0
            
            while time.time() - start_time < duration:
                # Capture measurement
                measurement = self.capture_measurement(0.1)
                
                if measurement:
                    self.data.append(measurement)
                    measurement_count += 1
                    
                    # Print current status
                    print(f"Measurement {measurement_count}: "
                          f"CH1 RMS={measurement['ch1_rms']:.4f}V, "
                          f"CH2 RMS={measurement['ch2_rms']:.4f}V, "
                          f"Phase={measurement['phase_deg']:.1f}°")
                
                # Wait for next sample
                time.sleep(sample_interval)
            
            print(f"✓ Completed {measurement_count} measurements")
            return True
            
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
    print("Simple Red Pitaya Control")
    print("=" * 30)
    
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
        sample_interval = 0.5  # seconds
        
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
