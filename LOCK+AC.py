import time
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyrpl import Pyrpl

# ------------------------- Waveform Parameters -------------------------
HOSTNAME = 'rp-f073ce.local'
OUTPUT_DIR = 'lockin_data'
YAML_FILE = 'lockin_config.yml'

# Output waveform settings (reference signal)
WAVEFORM_FREQ = 1000  # Hz
WAVEFORM_AMP = 0.5  # V peak-to-peak
WAVEFORM_OFFSET = 0.0  # DC offset

# Lock-in settings
LOCKIN_BANDWIDTH = 10  # Hz, bandwidth for lock-in filter
LOCKIN_GAIN = 1.0  # Lock-in gain

# Acquisition settings
RUN_TIME = 25  # Seconds to run acquisition
UPDATE_INTERVAL = 0.1  # Seconds between plot updates
TIME_WINDOW = 5.0  # Seconds of data to display in time-domain plots

# ----------------------------------------------------------------------

class RedPitayaLockinScope:
    def __init__(self, hostname=HOSTNAME, output_dir=OUTPUT_DIR, yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file

        # Connect to RedPitaya
        self.rp = Pyrpl(modules=['scope', 'asg0', 'iq0'], config=self.yaml_file)

        # Access modules
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0  # Reference signal generator
        self.lockin = self.rp.rp.iq0  # Lock-in amplifier

        # Scope setup - capture raw signals (fastest acquisition)
        self.scope.input1 = 'in1'  # Raw input signal
        self.scope.input2 = 'out1'  # Reference output signal
        self.scope.decimation = 1  # Minimum decimation for fastest sampling (125 MHz)
        self.scope.duration = 0.001  # 1 ms window for fast updates
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'

        self.sample_rate = 125e6 / self.scope.decimation

        # Data buffers for continuous acquisition
        self.time_buffer = []
        self.in1_buffer = []
        self.out1_buffer = []
        self.lockin_X_buffer = []
        self.lockin_Y_buffer = []
        self.lockin_R_buffer = []
        self.lockin_Theta_buffer = []
        
        self.start_time = None

        # Default output waveform
        self.ref_freq = WAVEFORM_FREQ
        self.ref_amp = WAVEFORM_AMP
        self.ref_offset = WAVEFORM_OFFSET
        self.ref_start_t = 0.0

    def setup_output_and_lockin(self, freq=None, amp=None, offset=None, 
                                 bandwidth=LOCKIN_BANDWIDTH, gain=LOCKIN_GAIN):
        """Setup both the reference signal and lock-in amplifier"""
        if freq is not None:
            self.ref_freq = freq
        if amp is not None:
            self.ref_amp = amp
        if offset is not None:
            self.ref_offset = offset

        # Setup reference signal output
        self.asg.setup(
            waveform='sin',
            frequency=self.ref_freq,
            amplitude=self.ref_amp,
            offset=self.ref_offset,
            output_direct='out1',
            trigger_source='immediately'
        )
        
        self.ref_start_t = time.time()
        ref_period = 1 / self.ref_freq

        # Setup lock-in amplifier
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=[-self.ref_freq * 2, -self.ref_freq, self.ref_freq, self.ref_freq * 2],
            gain=gain,
            phase=((time.time() - self.ref_start_t) / ref_period) * 360,
            acbandwidth=0,
            amplitude=self.ref_amp,
            input='in1',  # Measure from IN1
            output_direct='off',  # Don't output lock-in signal
            quadrature_factor=10
        )

        print(f"🔊 Reference Output (OUT1): {self.ref_freq} Hz, {self.ref_amp} V, Offset: {self.ref_offset} V")
        print(f"🔒 Lock-in configured: Freq={self.ref_freq} Hz, Bandwidth={bandwidth} Hz, Gain={gain}")

    def capture(self):
        """Capture data in continuous mode - fastest possible"""
        try:
            in1 = np.array(self.scope._data_ch1)
            out1 = np.array(self.scope._data_ch2)
            
            # Get lock-in X and Y directly from the lock-in module
            lockin_X = self.lockin.quadrature_values[0]
            lockin_Y = self.lockin.quadrature_values[1]
            
            if in1.size > 0 and out1.size > 0:
                # Calculate R and Theta from X and Y
                lockin_R = np.sqrt(lockin_X**2 + lockin_Y**2)
                lockin_Theta = np.arctan2(lockin_Y, lockin_X)
                
                # Replicate lock-in values to match array length
                lockin_X_array = np.full_like(in1, lockin_X)
                lockin_Y_array = np.full_like(in1, lockin_Y)
                lockin_R_array = np.full_like(in1, lockin_R)
                lockin_Theta_array = np.full_like(in1, lockin_Theta)
                
                return in1, out1, lockin_X_array, lockin_Y_array, lockin_R_array, lockin_Theta_array
            else:
                return None, None, None, None, None, None
        except Exception as e:
            print(f"⚠️ Error during capture: {e}")
            return None, None, None, None, None, None

    def update_buffers(self, in1, out1, lockin_X, lockin_Y, lockin_R, lockin_Theta):
        """Add new data to buffers"""
        current_time = time.time() - self.start_time
        n_samples = len(in1)
        time_array = np.linspace(current_time, current_time + n_samples/self.sample_rate, n_samples)
        
        self.time_buffer.extend(time_array)
        self.in1_buffer.extend(in1)
        self.out1_buffer.extend(out1)
        self.lockin_X_buffer.extend(lockin_X)
        self.lockin_Y_buffer.extend(lockin_Y)
        self.lockin_R_buffer.extend(lockin_R)
        self.lockin_Theta_buffer.extend(lockin_Theta)

    def plot_all_signals(self, axes, time_window=TIME_WINDOW):
        """Plot all signals: OUT1, IN1, Lock-in X/Y, R, and Theta"""
        if len(self.time_buffer) == 0:
            return

        # Convert to arrays
        t = np.array(self.time_buffer)
        out1 = np.array(self.out1_buffer)
        in1 = np.array(self.in1_buffer)
        X = np.array(self.lockin_X_buffer)
        Y = np.array(self.lockin_Y_buffer)
        R = np.array(self.lockin_R_buffer)
        Theta = np.array(self.lockin_Theta_buffer)

        # Limit to time window (show last N seconds)
        if time_window and len(t) > 0:
            latest_time = t[-1]
            mask = t >= (latest_time - time_window)
            t = t[mask]
            out1 = out1[mask]
            in1 = in1[mask]
            X = X[mask]
            Y = Y[mask]
            R = R[mask]
            Theta = Theta[mask]

        # Clear all axes
        for ax in axes:
            ax.clear()

        # Plot 1: OUT1 Reference Signal
        axes[0].plot(t, out1, color='tab:orange', linewidth=1, alpha=0.8)
        axes[0].set_ylabel('Voltage (V)')
        axes[0].set_title('OUT1 Reference Signal')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Raw IN1 signal
        axes[1].plot(t, in1, color='tab:blue', linewidth=1, alpha=0.7)
        axes[1].set_ylabel('Voltage (V)')
        axes[1].set_title('IN1 Measured Signal')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Lock-in X and Y components
        axes[2].plot(t, X, color='tab:green', linewidth=1.5, label='X (in-phase)', alpha=0.8)
        axes[2].plot(t, Y, color='tab:cyan', linewidth=1.5, label='Y (quadrature)', alpha=0.8)
        axes[2].set_ylabel('Voltage (V)')
        axes[2].set_title('Lock-in Demodulated Components')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Lock-in Amplitude (R)
        axes[3].plot(t, R, color='tab:red', linewidth=1.5)
        axes[3].set_ylabel('Amplitude (V)')
        axes[3].set_title('Lock-in Amplitude (R)')
        axes[3].grid(True, alpha=0.3)

        # Plot 5: Lock-in Phase (Theta)
        axes[4].plot(t, np.degrees(Theta), color='tab:purple', linewidth=1.5)
        axes[4].set_ylabel('Phase (degrees)')
        axes[4].set_xlabel('Time (s)')
        axes[4].set_title('Lock-in Phase (θ)')
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.001)

    def save_to_csv(self):
        """Save all captured data to CSV file"""
        if len(self.time_buffer) == 0:
            print("⚠️ No data to save")
            return None

        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'{self.output_dir}/lockin_data_{self.ref_freq}Hz_{timestamp}.csv'

        # Prepare data
        data = np.column_stack((
            self.time_buffer,
            self.out1_buffer,
            self.in1_buffer,
            self.lockin_X_buffer,
            self.lockin_Y_buffer,
            self.lockin_R_buffer,
            np.degrees(self.lockin_Theta_buffer)
        ))
        
        header = 'Time(s),OUT1_Ref(V),IN1_Measured(V),Lockin_X(V),Lockin_Y(V),Lockin_R(V),Lockin_Theta(deg)'

        # Create output directory if needed
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Save to CSV
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        print(f"💾 Data saved to: {filename}")
        
        # Also save a summary
        print(f"📊 Summary:")
        print(f"   Mean R: {np.mean(self.lockin_R_buffer):.6f} V")
        print(f"   Std R: {np.std(self.lockin_R_buffer):.6f} V")
        print(f"   Mean Theta: {np.degrees(np.mean(self.lockin_Theta_buffer)):.2f}°")
        
        return filename

    def run_continuous(self, time_window=TIME_WINDOW, run_time=RUN_TIME, 
                       update_interval=UPDATE_INTERVAL):
        """Continuously acquire and plot lock-in data"""
        print(f"Starting continuous lock-in acquisition for {run_time} seconds...")
        print(f"Reference: {self.ref_freq} Hz, {self.ref_amp} V")
        print("Press Ctrl+C to stop early.")

        # Ensure we're in continuous mode
        self.scope.running_state = 'running_continuous'

        plt.ion()
        fig, axes = plt.subplots(5, 1, figsize=(12, 12))

        self.start_time = time.time()
        last_update = self.start_time

        try:
            while True:
                current_time = time.time()
                elapsed = current_time - self.start_time

                # Check if run_time has elapsed
                if elapsed >= run_time:
                    print(f"\n✅ Completed {run_time} second acquisition")
                    break

                # Capture data
                in1, out1, X, Y, R, Theta = self.capture()
                if in1 is None:
                    time.sleep(0.001)  # Minimal sleep for fastest acquisition
                    continue

                # Update buffers
                self.update_buffers(in1, out1, X, Y, R, Theta)

                # Update plot at specified interval
                if (current_time - last_update) >= update_interval:
                    self.plot_all_signals(axes, time_window=time_window)
                    last_update = current_time
                    print(f"\rTime: {elapsed:.1f}s / {run_time}s", end='', flush=True)

                # No sleep here for maximum speed

        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")

        # Save data
        self.save_to_csv()

        plt.ioff()
        plt.show()


if __name__ == '__main__':
    rp_lockin = RedPitayaLockinScope(output_dir=OUTPUT_DIR)
    
    # Setup reference signal and lock-in amplifier
    rp_lockin.setup_output_and_lockin(
        freq=WAVEFORM_FREQ,
        amp=WAVEFORM_AMP,
        offset=WAVEFORM_OFFSET,
        bandwidth=LOCKIN_BANDWIDTH,
        gain=LOCKIN_GAIN
    )
    
    # Run continuous acquisition
    rp_lockin.run_continuous(
        time_window=TIME_WINDOW,
        run_time=RUN_TIME,
        update_interval=UPDATE_INTERVAL
    )
