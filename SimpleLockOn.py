"""
Red Pitaya Lock-In Amplifier - FIXED & SIMPLE VERSION
This actually reads from the lock-in properly at high speed

Connect OUT1 to IN1 with a cable for testing.
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from pyrpl import Pyrpl

# ============================================================
# MEASUREMENT PARAMETERS - KEEP THESE THE SAME
# ============================================================
REF_FREQUENCY = 500  # Hz
REF_AMPLITUDE = 1  # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 30.0  # seconds

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'  # Fixed to uppercase
AUTOLAB_GAIN = 0.0
MANUAL_GAIN_FACTOR = 1.08 + AUTOLAB_GAIN

# Lock-in settings
FILTER_BANDWIDTH = 100  # Hz - Higher = faster response, more noise. Lower = slower, cleaner
SAMPLE_RATE = 10000  # Hz - How fast to read lock-in output (10 kHz as requested)

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
SAVE_TIMESTAMPS = True

AUTO_CALIBRATE = True
CALIBRATION_TIME = 2.0  # seconds
# ============================================================

START_TIME_FILE = "start_time.txt"

with open(START_TIME_FILE, "r") as f:
    START_TIME = datetime.fromisoformat(f.read().strip())

while datetime.now() < START_TIME:
    time.sleep(0.001)


class RedPitaya:
    def __init__(self, output_dir='test_data', input_mode='AUTO', manual_gain=1.0):
        print("Connecting to Red Pitaya...")
        self.rp = Pyrpl(config='lockin_config', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope
        
        self.lockin_X = []
        self.lockin_Y = []
        self.lockin_R = []
        self.lockin_Theta = []
        self.timestamps = []
        self.acquisition_start_time = None

        # Setup input gain
        self.input_gain_factor = manual_gain
        self.input_mode_setting = input_mode.upper()
        self.input_mode = "Unknown"

        if self.input_mode_setting == 'MANUAL':
            self.input_gain_factor = manual_gain
            self.input_mode = f"MANUAL ({manual_gain}x gain)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_mode = "LV (±1V)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_mode = "HV (±20V, 20:1 divider)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'AUTO':
            self.input_mode = "AUTO (will calibrate)"
            print("Input mode: AUTO - will auto-detect")

    def calibrate_input_gain(self, cal_freq, cal_amp, cal_time):
        """Auto-detect if you're in LV or HV mode"""
        if self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("AUTO-CALIBRATING INPUT GAIN...")
        print("=" * 60)

        # Setup lock-in for calibration
        self.ref_sig.output_direct = 'off'
        self.lockin.setup(
            frequency=cal_freq,
            bandwidth=100,  # Use faster bandwidth for calibration
            gain=0.0,
            phase=0,
            acbandwidth=0,
            amplitude=cal_amp,
            input='in1',
            output_direct='out1',
            output_signal='quadrature',
            quadrature_factor=1.0)

        print(f"Generating {cal_amp}V at {cal_freq} Hz...")
        print("Waiting for lock-in to settle...")
        time.sleep(1.0)

        # Read lock-in directly
        cal_readings = []
        start_time = time.time()
        
        while (time.time() - start_time) < cal_time:
            R = self.lockin.amplitude
            cal_readings.append(R)
            time.sleep(0.01)  # 100 Hz during calibration

        measured_amp = np.mean(cal_readings)
        expected_amp = cal_amp / 2.0  # Lock-in theory: output is half input for pure sine
        
        self.input_gain_factor = expected_amp / measured_amp

        # Determine mode
        if self.input_gain_factor < 1.05:
            self.input_mode = "LV (±1V)"
        elif self.input_gain_factor < 1.15:
            self.input_mode = "LV (±1V) with loading"
        elif self.input_gain_factor < 15:
            self.input_mode = f"Unknown ({self.input_gain_factor:.2f}x)"
        else:
            attenuation = 1.0 / self.input_gain_factor
            self.input_mode = f"HV (±20V, {attenuation:.1f}:1)"

        print(f"Expected: {expected_amp:.4f}V, Measured: {measured_amp:.4f}V")
        print(f"Gain factor: {self.input_gain_factor:.4f}x")
        print(f"Detected mode: {self.input_mode}")
        print("=" * 60 + "\n")

        return self.input_gain_factor

    def setup_lockin(self, ref_freq, ref_amp, filter_bw, phase):
        """Configure the lock-in amplifier"""
        self.ref_freq = ref_freq
        
        self.ref_sig.output_direct = 'off'
        
        self.lockin.setup(
            frequency=ref_freq,
            bandwidth=filter_bw,
            gain=0.0,
            phase=phase,
            acbandwidth=0,
            amplitude=ref_amp,
            input='in1',
            output_direct='out1',
            output_signal='quadrature',
            quadrature_factor=1.0)

        print(f"\nLock-in configured:")
        print(f"  Reference: {ref_freq} Hz @ {ref_amp}V")
        print(f"  Filter bandwidth: {filter_bw} Hz")
        print(f"  Gain correction: {self.input_gain_factor:.4f}x")

    def run(self, params):
        """Main acquisition loop"""
        
        # Calibrate if needed
        if params['auto_calibrate']:
            self.calibrate_input_gain(
                cal_freq=params['ref_freq'],
                cal_amp=params['ref_amp'],
                cal_time=params['calibration_time']
            )

        # Setup lock-in
        self.setup_lockin(
            ref_freq=params['ref_freq'],
            ref_amp=params['ref_amp'],
            filter_bw=params['filter_bandwidth'],
            phase=params['phase']
        )

        print("\nWaiting for lock-in to settle...")
        time.sleep(2.0)  # Important: let the filter settle

        # Calculate target interval between samples
        sample_interval = 1.0 / params['sample_rate']
        
        print(f"\nStarting acquisition at {params['sample_rate']} Hz for {params['timeout']}s...")
        self.acquisition_start_time = time.time()
        print(f"Started: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        loop_start = time.time()
        next_sample_time = loop_start
        samples_taken = 0
        
        # MAIN ACQUISITION LOOP - Read lock-in as fast as requested
        while (time.time() - loop_start) < params['timeout']:
            current_time = time.time()
            
            if current_time >= next_sample_time:
                # READ LOCK-IN DIRECTLY (this is the key fix!)
                X = self.lockin.x
                Y = self.lockin.y
                
                # Store raw values
                self.lockin_X.append(X)
                self.lockin_Y.append(Y)
                self.timestamps.append(current_time)
                
                samples_taken += 1
                next_sample_time += sample_interval
            else:
                # Sleep briefly to avoid hammering the CPU
                sleep_time = min(0.0001, (next_sample_time - current_time) * 0.5)
                time.sleep(sleep_time)

        acquisition_end_time = time.time()

        # Convert to numpy arrays
        all_X = np.array(self.lockin_X) * self.input_gain_factor
        all_Y = np.array(self.lockin_Y) * self.input_gain_factor
        all_timestamps = np.array(self.timestamps)
        
        # Calculate R and Theta from X and Y
        R = np.sqrt(all_X**2 + all_Y**2)
        Theta = np.arctan2(all_Y, all_X)
        t = all_timestamps - self.acquisition_start_time

        # Calculate actual sample rate
        total_samples = len(all_X)
        actual_duration = acquisition_end_time - self.acquisition_start_time
        actual_sample_rate = total_samples / actual_duration
        
        print("\n" + "=" * 60)
        print("ACQUISITION COMPLETE")
        print("=" * 60)
        print(f"Requested: {params['sample_rate']} Hz")
        print(f"Actual: {actual_sample_rate:.2f} Hz")
        print(f"Samples: {total_samples}")
        print(f"Duration: {actual_duration:.3f}s")
        print("=" * 60)

        # Get raw signals for diagnostic plots (using scope briefly)
        print("\nCapturing raw waveforms for plots...")
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        self.scope.decimation = 8  # Fast decimation to see waveform
        time.sleep(0.1)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw = np.array(self.scope._data_ch2_current) * self.input_gain_factor
        t_raw = np.arange(len(out1_raw)) / (125e6 / self.scope.decimation)

        # Print results
        print("\n" + "=" * 60)
        print("LOCK-IN RESULTS (GAIN-CORRECTED)")
        print("=" * 60)
        print(f"Mode: {self.input_mode}")
        print(f"Gain factor: {self.input_gain_factor:.4f}x")
        print(f"\nMean R: {np.mean(R):.6f} V (±{np.std(R):.6f})")
        print(f"Mean X: {np.mean(all_X):.6f} V (±{np.std(all_X):.6f})")
        print(f"Mean Y: {np.mean(all_Y):.6f} V (±{np.std(all_Y):.6f})")
        print(f"Mean Phase: {np.mean(Theta):.4f} rad (±{np.std(Theta):.4f})")
        
        expected_R = params['ref_amp'] / 2.0
        error = abs(np.mean(R) - expected_R)
        print(f"\nExpected R: {expected_R:.3f}V")
        print(f"Error: {error:.4f}V ({error/expected_R*100:.2f}%)")
        
        if error < 0.05:
            print("✓ GOOD: R is close to expected value")
        else:
            print("✗ WARNING: R differs significantly from expected")
            print("  This might indicate gain calibration issues")
        
        print("=" * 60)

        # Create plots
        self.create_plots(t, all_X, all_Y, R, Theta, 
                         out1_raw, in1_raw, t_raw,
                         expected_R, actual_sample_rate, params)

        # Save data
        if params['save_file']:
            self.save_data(all_timestamps, t, R, Theta, all_X, all_Y, 
                          actual_sample_rate, params)

    def create_plots(self, t, X, Y, R, Theta, out1_raw, in1_raw, t_raw, 
                     expected_R, sample_rate, params):
        """Create diagnostic plots"""
        fig = plt.figure(figsize=(16, 10))

        # Raw reference signal
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_plot = min(int(n_periods * (125e6/8) / self.ref_freq), len(out1_raw))
        ax1.plot(t_raw[:n_plot] * 1000, out1_raw[:n_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference Signal @ {self.ref_freq} Hz')
        ax1.grid(True)

        # Raw input signal
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw[:n_plot] * 1000, in1_raw[:n_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title(f'Input Signal - {self.input_mode}')
        ax2.grid(True)

        # Spectrum of lock-in output
        ax3 = plt.subplot(3, 3, 3)
        freqs = np.fft.fftfreq(len(X), 1.0/sample_rate)
        fft_mag = np.abs(np.fft.fft(X + 1j*Y))
        positive = freqs > 0
        ax3.semilogy(freqs[positive], fft_mag[positive])
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Lock-in Output Spectrum')
        ax3.grid(True)
        ax3.set_xlim([0, min(100, sample_rate/2)])

        # X vs Time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, X, 'b-', linewidth=0.5, alpha=0.7)
        ax4.axhline(np.mean(X), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(X):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-Phase (X)')
        ax4.legend()
        ax4.grid(True)

        # Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, Y, 'r-', linewidth=0.5, alpha=0.7)
        ax5.axhline(np.mean(Y), color='b', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y)')
        ax5.legend()
        ax5.grid(True)

        # IQ plot
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(X, Y, 'g.', markersize=1, alpha=0.3)
        ax6.plot(np.mean(X), np.mean(Y), 'r+', markersize=20, markeredgewidth=3, 
                label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot (Constellation)')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # R vs Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, R, 'm-', linewidth=0.5, alpha=0.7)
        ax7.axhline(np.mean(R), color='b', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(R):.4f}V')
        ax7.axhline(expected_R, color='g', linestyle=':', linewidth=2,
                   label=f'Expected: {expected_R:.3f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R)')
        ax7.legend()
        ax7.grid(True)

        # Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5, alpha=0.7)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(Theta):.4f} rad')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (rad)')
        ax8.set_title('Phase (Theta)')
        ax8.legend()
        ax8.grid(True)

        # Polar plot
        ax9 = plt.subplot(3, 3, 9, projection='polar')
        ax9.plot(Theta, R, 'g.', markersize=1, alpha=0.3)
        ax9.plot(np.mean(Theta), np.mean(R), 'r+', markersize=20, 
                markeredgewidth=3, label='Mean')
        ax9.set_title('Polar Plot')
        ax9.legend()
        ax9.grid(True)

        plt.tight_layout()

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(self.output_dir, f'lockin_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Saved plot: {img_path}")
        else:
            plt.show()

    def save_data(self, abs_time, rel_time, R, Theta, X, Y, sample_rate, params):
        """Save data to CSV"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f'lockin_{timestamp_str}.csv')

        if params['save_timestamps']:
            data = np.column_stack((abs_time, rel_time, R, Theta, X, Y))
            header_cols = "AbsoluteTime,RelativeTime,R,Theta,X,Y"
        else:
            data = np.column_stack((rel_time, R, Theta, X, Y))
            header_cols = "Time,R,Theta,X,Y"

        with open(csv_path, 'w') as f:
            f.write(f"# Red Pitaya Lock-In Amplifier Data\n")
            f.write(f"# Mode: {self.input_mode}\n")
            f.write(f"# Gain Factor: {self.input_gain_factor:.6f}\n")
            f.write(f"# Reference: {self.ref_freq} Hz @ {params['ref_amp']} V\n")
            f.write(f"# Filter BW: {params['filter_bandwidth']} Hz\n")
            f.write(f"# Sample Rate: {sample_rate:.2f} Hz\n")
            f.write(f"# {header_cols}\n")
            np.savetxt(f, data, delimiter=',', fmt='%.10f')

        print(f"✓ Saved data: {csv_path}")


if __name__ == '__main__':
    rp = RedPitaya(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR
    )

    run_params = {
        'ref_freq': REF_FREQUENCY,
        'ref_amp': REF_AMPLITUDE,
        'phase': PHASE_OFFSET,
        'timeout': MEASUREMENT_TIME,
        'filter_bandwidth': FILTER_BANDWIDTH,
        'sample_rate': SAMPLE_RATE,
        'save_file': SAVE_DATA,
        'save_timestamps': SAVE_TIMESTAMPS,
        'auto_calibrate': AUTO_CALIBRATE,
        'calibration_time': CALIBRATION_TIME,
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN AMPLIFIER - FIXED VERSION")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE}V")
    print(f"Filter bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Sample rate: {SAMPLE_RATE} Hz (reading lock-in output)")
    print(f"Duration: {MEASUREMENT_TIME}s")
    print(f"Input mode: {INPUT_MODE}")
    if INPUT_MODE == 'MANUAL':
        print(f"Manual gain: {MANUAL_GAIN_FACTOR}x")
    print("=" * 60)

    rp.run(run_params)
