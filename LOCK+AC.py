"""
Combined RedPitaya Lock-in Amplifier and AC Emitter
Combines scope acquisition with lock-in amplifier functionality
Created on 09/25/2025
@author: mason mandernach
"""
import time
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyrpl import Pyrpl
import csv
import os

# ------------------------- Waveform Parameters -------------------------
HOSTNAME = 'rp-f073ce.local'
OUTPUT_DIR = 'scope_data'
YAML_FILE = 'scope_config.yml'

# Output waveform settings
WAVEFORM_FREQ = 1000  # Hz
WAVEFORM_AMP = 0.5  # V peak-to-peak
WAVEFORM_OFFSET = 0.0  # DC offset
TIME_WINDOW = 0.005  # Seconds for time-domain plot

# Acquisition settings
RUN_TIME = 25  # Seconds to run acquisition before stopping and saving

# Lock-in parameters
N_FFT_SHOW = 10

# ----------------------------------------------------------------------

class RedPitayaCombined:
    """
    Combined class for RedPitaya that includes:
    - AC emitter (ASG output on OUT1)
    - Lock-in amplifier (IQ2)
    - Scope acquisition
    """
    
    electrode_map = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True), '100uA': (True, False, True, True), 
                        '1mA': (True, True, False, True), '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True), '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, hostname=HOSTNAME, output_dir=OUTPUT_DIR, yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file

        # Connect to RedPitaya with all needed modules
        self.rp = Pyrpl(modules=['scope', 'asg0', 'iq2', 'pid0'], config=self.yaml_file)

        # Access modules
        self.rp_modules = self.rp.rp
        self.scope = self.rp_modules.scope
        self.asg = self.rp_modules.asg0  # AC emitter
        self.lockin = self.rp_modules.iq2  # Lock-in amplifier
        self.pid = self.rp_modules.pid0

        # Lock-in data storage
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []

        # PID parameters
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival

        # Scope setup for general acquisition (default)
        self.scope.input1 = 'in1'
        self.scope.input2 = 'out1'
        self.scope.decimation = 128
        self.scope.duration = 0.01  # 10 ms window
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'

        self.sample_rate = 125e6 / self.scope.decimation

        # Default output waveform
        self.test_freq = WAVEFORM_FREQ
        self.test_amp = WAVEFORM_AMP
        self.test_offset = WAVEFORM_OFFSET

    def setup_output(self, freq=None, amp=None, offset=None):
        """Setup AC emitter output on OUT1"""
        if freq is not None:
            self.test_freq = freq
        if amp is not None:
            self.test_amp = amp
        if offset is not None:
            self.test_offset = offset
        self.asg.setup(
            waveform='sin',
            frequency=self.test_freq,
            amplitude=self.test_amp,
            offset=self.test_offset,
            output_direct='out1',
            trigger_source='immediately'
        )
        print(f"🔊 AC Emitter Output: {self.test_freq} Hz, {self.test_amp} V, Offset: {self.test_offset} V")

    def setup_lockin(self, params):
        """Setup lock-in amplifier with reference signal"""
        self.ref_freq = params['ref_freq']
        self.ref_period = 1/self.ref_freq
        ref_amp = params['ref_amp']

        # Setup reference signal (can use same ASG or different output)
        self.asg.setup(waveform='sin',
                       amplitude=ref_amp,
                       frequency=self.ref_freq)

        self.ref_start_t = time.time()

        if params['output_ref'] == 'out1' or params['output_ref'] == 'out2':
            self.asg.output_direct = params['output_ref']
        else:
            self.asg.output_direct = 'off'

        # Setup lock-in amplifier
        self.lockin.setup(frequency=self.ref_freq,
                       bandwidth=[-self.ref_freq * 2, -self.ref_freq, self.ref_freq, self.ref_freq * 2],  # Hz
                       gain=1.0,
                       phase=((time.time() - self.ref_start_t)/self.ref_period)*360,  # initial phase in degrees
                       acbandwidth=0,
                       amplitude=ref_amp,
                       input='in1',
                       output_direct='out2',
                       output_signal='output_direct',
                       quadrature_factor=10)
        
        print(f"🔒 Lock-in setup: {self.ref_freq} Hz, {ref_amp} V")

    def capture(self):
        """Capture data in continuous mode (for scope signals)"""
        try:
            ch1 = np.array(self.scope._data_ch1)
            ch2 = np.array(self.scope._data_ch2)

            if ch1.size > 0 and ch2.size > 0:
                return ch1, ch2
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Error during capture: {e}")
            return None, None

    def capture_lockin(self):
        """
        Captures lock-in X and Y data in continuous mode
        Uses continuous mode like capture() - no scope.single() call
        """
        try:
            ch1 = np.array(self.scope._data_ch1)
            ch2 = np.array(self.scope._data_ch2)

            if ch1.size > 0 and ch2.size > 0:
                self.lockin_X.append(ch1)
                self.lockin_Y.append(ch2)
                return ch1, ch2
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Error during lock-in capture: {e}")
            return None, None

    def plot_all_signals(self, ch_in1, ch_out1, axes, time_window=TIME_WINDOW):
        """Plot OUT1 and IN1 voltages"""
        t = np.arange(len(ch_in1)) / self.sample_rate

        # Limit to time window
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in1 = ch_in1[:max_samples]
            ch_out1 = ch_out1[:max_samples]

        # Clear all axes
        for ax in axes:
            ax.clear()

        # Plot 1: OUT1 Voltage
        axes[0].plot(t, ch_out1, color='tab:orange', linewidth=1.5)
        axes[0].set_ylabel('Voltage (V)')
        axes[0].set_title('OUT1 Voltage (AC Emitter)')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: IN1 Voltage
        axes[1].plot(t, ch_in1, color='tab:blue', linewidth=1.5)
        axes[1].set_ylabel('Voltage (V)')
        axes[1].set_title('IN1 Voltage')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.001)

    def plot_lockin_xy(self, axes=None):
        """Plot lock-in X vs Y (should be a circle when locked)"""
        if len(self.all_X) == 0 or len(self.all_Y) == 0:
            print("No lock-in data to plot")
            return

        if axes is None:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
        else:
            ax = axes

        ax.plot(self.all_X, self.all_Y, 'b-', linewidth=1.5, alpha=0.7)
        ax.scatter(self.all_X[0], self.all_Y[0], color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(self.all_X[-1], self.all_Y[-1], color='red', s=100, marker='x', label='End', zorder=5)
        ax.set_xlabel('X (Lock-in)')
        ax.set_ylabel('Y (Lock-in)')
        ax.set_title('Lock-in X vs Y (Circle = Locked)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

    def see_fft(self):
        """Plot FFT of lock-in data"""
        iq = self.all_X + 1j*self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))

        idx = np.argmax(psd_lock)
        print("Peak at", freqs_lock[idx], "Hz")

        plt.figure(figsize=(12, 4))
        plt.semilogy(freqs_lock, psd_lock, label='Lock-in R')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.title('Lock-in Output Spectrum (baseband)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def save_to_csv(self, ch_in1, ch_out1, time_window=TIME_WINDOW):
        """Save captured scope data to CSV file"""
        t = np.arange(len(ch_in1)) / self.sample_rate

        # Limit to time window
        if time_window:
            max_samples = int(time_window * self.sample_rate)
            t = t[:max_samples]
            ch_in1 = ch_in1[:max_samples]
            ch_out1 = ch_out1[:max_samples]

        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'scope_data_{timestamp}.csv'

        # Prepare data
        data = np.column_stack((t, ch_out1, ch_in1))
        header = 'Time(s),OUT1_Voltage(V),IN1_Voltage(V)'

        # Save to CSV
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        print(f"💾 Scope data saved to: {filename}")
        return filename

    def run_continuous(self, time_window=TIME_WINDOW, run_time=None):
        """Continuously acquire and plot scope data"""
        if run_time is None:
            print("Starting continuous capture...")
            print("Press Ctrl+C to stop.")
        else:
            print(f"Starting capture for {run_time} seconds...")

        # Ensure we're in continuous mode
        self.scope.running_state = 'running_continuous'

        plt.ion()
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        start_time = time.time()
        last_ch_in1 = None
        last_ch_out1 = None

        try:
            while True:
                # Check if run_time has elapsed
                if run_time is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= run_time:
                        print(f"\n✅ Completed {run_time} second acquisition")
                        break

                ch_in1, ch_out1 = self.capture()
                if ch_in1 is None or ch_out1 is None:
                    time.sleep(0.05)
                    continue

                last_ch_in1 = ch_in1
                last_ch_out1 = ch_out1

                self.plot_all_signals(ch_in1, ch_out1, axes, time_window=time_window)
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")

        if last_ch_in1 is not None and last_ch_out1 is not None:
            self.save_to_csv(last_ch_in1, last_ch_out1, time_window=time_window)

        plt.ioff()
        plt.show()

    def run_lockin(self, params):
        """Run lock-in acquisition using continuous mode"""
        timeout = params['timeout']

        self.setup_lockin(params)
        time.sleep(0.01)

        # Save original scope settings
        old_input1 = self.scope.input1
        old_input2 = self.scope.input2
        old_decimation = self.scope.decimation
        old_duration = self.scope.duration
        old_average = self.scope.average

        # Configure scope for lock-in acquisition
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        
        if params.get('lockin_decimation'):
            if params['lockin_decimation'] in self.allowed_decimations:
                self.scope.decimation = params['lockin_decimation']
                self.sample_rate = 125e6 / self.scope.decimation
            else:
                print(f"Invalid decimation {params['lockin_decimation']}, using default 64")
                self.scope.decimation = 64
                self.sample_rate = 125e6 / 64
        else:
            # Default lock-in decimation
            self.scope.decimation = 64
            self.sample_rate = 125e6 / 64

        # Set up for continuous mode (like the scope file)
        self.scope.duration = 0.01  # 10 ms window
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'

        # Clear previous data
        self.lockin_X = []
        self.lockin_Y = []

        loop_start = time.time()

        print(f"🔒 Starting lock-in acquisition for {timeout} seconds...")
        print(f"   Using continuous mode (decimation: {self.scope.decimation}, sample rate: {self.sample_rate:.0f} Hz)")
        
        try:
            while (time.time() - loop_start) < timeout:
                ch1, ch2 = self.capture_lockin()
                if ch1 is None or ch2 is None:
                    time.sleep(0.05)
                    continue
                time.sleep(0.05)  # Small delay like in run_continuous

        except KeyboardInterrupt:
            print("\n⏹️ Lock-in acquisition stopped by user")

        # Restore original scope settings
        self.scope.input1 = old_input1
        self.scope.input2 = old_input2
        self.scope.decimation = old_decimation
        self.scope.duration = old_duration
        self.scope.average = old_average
        self.sample_rate = 125e6 / old_decimation

        if len(self.lockin_X) == 0:
            print("⚠️ No lock-in data captured!")
            return

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))
        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)

        print(f"✅ Lock-in acquisition complete. Captured {len(self.all_X)} points")

        # Plotting
        if params.get('plot_xy', True):  # Default to X/Y plot (circle)
            plt.figure(figsize=(12, 5))
            
            # Plot 1: X vs Y (circle plot)
            plt.subplot(1, 2, 1)
            self.plot_lockin_xy(plt.gca())
            
            # Plot 2: R and Theta vs time
            plt.subplot(1, 2, 2)
            t = np.arange(start=0, stop=len(self.all_X)/self.sample_rate, step=1/self.sample_rate)
            if len(t) > len(R):
                t = t[:len(R)]
            elif len(t) < len(R):
                R = R[:len(t)]
                Theta = Theta[:len(t)]
            plt.plot(t, R, label='R (Magnitude)')
            plt.plot(t, Theta, label='Theta (Phase)')
            plt.title('Lock-in R and Theta vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('R and Theta')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
        elif params.get('fft', False):
            self.see_fft()
        else:
            t = np.arange(start=0, stop=len(self.all_X)/self.sample_rate, step=1/self.sample_rate)
            if len(t) > len(R):
                t = t[:len(R)]
            elif len(t) < len(R):
                R = R[:len(R)]
                Theta = Theta[:len(R)]
            plt.figure(figsize=(10, 6))
            plt.plot(t, R, label='R')
            plt.plot(t, Theta, label='Theta')
            plt.title('Lockin Results')
            plt.xlabel('Time (s)')
            plt.ylabel('R and Theta')
            plt.legend()
            plt.grid(True, alpha=0.3)

        if params.get('save_file', False):
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            img_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.png')
            data = np.column_stack((R, Theta, self.all_X, self.all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.csv')
            np.savetxt(csv_path, data, delimiter=",", header="R,Theta,X,Y", comments='', fmt='%.6f')
            plt.savefig(img_path)
            print(f"💾 Lock-in data saved to: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    # Initialize combined RedPitaya
    rp = RedPitayaCombined()

    # Option 1: Run AC emitter with scope acquisition
    print("=" * 60)
    print("Option 1: AC Emitter + Scope Acquisition")
    print("=" * 60)
    rp.setup_output(freq=WAVEFORM_FREQ, amp=WAVEFORM_AMP, offset=WAVEFORM_OFFSET)
    # Uncomment to run:
    # rp.run_continuous(time_window=TIME_WINDOW, run_time=RUN_TIME)

    # Option 2: Run lock-in amplifier
    print("\n" + "=" * 60)
    print("Option 2: Lock-in Amplifier")
    print("=" * 60)
    run_params = {
        'ref_freq': 100,            # Hz, reference signal frequency for lock-in
        'ref_amp': 0.4,             # V, amplitude of reference signal
        'output_ref': 'out1',       # where to output the ref_signal (AC emitter)

        'timeout': 5.0,             # seconds, how long to run acquisition loop
        'lockin_decimation': 64,    # decimation for lock-in (must be in allowed_decimations)

        'output_dir': 'test_data',  # where to save FFT and waveform plots
        'save_file': False,         # whether to save plots instead of showing them
        'fft': False,               # whether to perform FFT after run
        'plot_xy': True,            # whether to plot X vs Y (circle plot) - recommended!
    }

    # Uncomment to run:
    # rp.run_lockin(run_params)
