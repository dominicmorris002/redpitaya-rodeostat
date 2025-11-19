"""
Red Pitaya Lock-In Amplifier - CORRECTED VERSION

SETUP: Connect OUT1 directly to IN1 with a cable

The IQ module demodulated outputs (X and Y) are NOT available as scope inputs!
We must read them directly from the IQ module's internal registers.
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 100        # Hz - AC excitation frequency
REF_AMPLITUDE = 0.4        # V - AC signal amplitude (will appear on OUT1)
OUTPUT_CHANNEL = 'out1'    # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET = 0           # degrees - phase adjustment (0, 90, 180, 270)
MEASUREMENT_TIME = 5.0     # seconds - how long to measure

# LOCK-IN FILTER BANDWIDTH
FILTER_BANDWIDTH = 10      # Hz - lower = cleaner, higher = faster response

# AVERAGING
AVERAGING_WINDOW = 1       # samples - set to 1 to see raw lock-in output first

# Data saving
SAVE_DATA = False          # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'

# Advanced settings
DECIMATION = 64
SHOW_FFT = True

# Red Pitaya connection
HOSTNAME = 'rp-f073ce.local'
YAML_FILE = 'lockin_scope_config.yml'
# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os
import yaml

N_FFT_SHOW = 10

class RedPitaya:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file
        
        # Create YAML config first
        self.create_yaml()
        
        # Connect to RedPitaya with proper modules
        self.rp = Pyrpl(config=self.yaml_file, hostname=HOSTNAME)
        self.rp_modules = self.rp.rp
        
        # Get modules
        self.lockin = self.rp_modules.iq2
        self.asg = self.rp_modules.asg0
        self.scope = self.rp_modules.scope
        self.pid = self.rp_modules.pid0
        
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []
        self.timestamps = []

        print("Available scope inputs:", self.scope.inputs)
        
        # Setup scope to view raw signals
        self.scope.input1 = 'out1'   # Reference from ASG
        self.scope.input2 = 'in1'    # Input signal
        self.scope.decimation = DECIMATION
        self.scope.duration = 0.01
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'
        
        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()
        
        self.sample_rate = 50  # We'll sample at ~50 Hz from IQ module registers
        print(f"Lock-in sample rate: {self.sample_rate:.2f} Hz")

    def create_yaml(self):
        """Create YAML config file similar to working example"""
        if not os.path.exists(self.yaml_file):
            config = {
                'redpitaya_hostname': HOSTNAME,
                'modules': ['scope', 'asg0', 'iq2'],
                'scope': {
                    'ch1_active': True,
                    'ch2_active': True,
                    'input1': 'out1',
                    'input2': 'in1',
                    'threshold': 0.0,
                    'hysteresis': 0.0,
                    'duration': 0.01,
                    'trigger_delay': 0.0,
                    'trigger_source': 'immediately',
                    'running_state': 'running_continuous',
                    'average': False,
                    'decimation': DECIMATION
                },
                'asg0': {
                    'waveform': 'sin',
                    'frequency': REF_FREQUENCY,
                    'amplitude': REF_AMPLITUDE,
                    'offset': 0.0,
                    'output_direct': OUTPUT_CHANNEL,
                    'trigger_source': 'immediately'
                },
                'iq2': {
                    'input': 'in1',
                    'frequency': REF_FREQUENCY,
                    'bandwidth': FILTER_BANDWIDTH,
                    'gain': 0.0,
                    'phase': PHASE_OFFSET,
                    'acbandwidth': 0,
                    'amplitude': 0.0,
                    'output_direct': 'off',
                    'output_signal': 'quadrature',
                    'quadrature_factor': 1
                }
            }
            with open(self.yaml_file, 'w') as f:
                yaml.dump(config, f)
            print(f"Created YAML: {self.yaml_file}")
        else:
            print(f"YAML exists: {self.yaml_file}")

    def setup_output(self, freq, amp, offset=0.0):
        """Setup ASG0 output - same as working example"""
        self.asg.setup(
            waveform='sin',
            frequency=freq,
            amplitude=amp,
            offset=offset,
            output_direct=OUTPUT_CHANNEL,
            trigger_source='immediately'
        )
        print(f"🔊 ASG0 Output: {freq} Hz, {amp} V, Offset: {offset} V on {OUTPUT_CHANNEL}")

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)
        
        # Setup ASG0 to output reference signal (like the working example)
        self.setup_output(freq=self.ref_freq, amp=ref_amp, offset=0.0)
        
        # Setup IQ2 for demodulation ONLY (no output)
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,              # No feedback
            phase=phase_setting,
            acbandwidth=0,         # DC-coupled input
            amplitude=0.0,         # NO OUTPUT from IQ module
            input='in1',
            output_direct='off',   # IQ module outputs NOTHING
            output_signal='quadrature',
            quadrature_factor=1)   # No extra gain
        
        print(f"Lock-in demodulation setup: {self.ref_freq} Hz")
        print(f"Filter BW: {filter_bw} Hz")
        print(f"IQ2 input: {self.lockin.input}")
        print(f"IQ2 output_direct: {self.lockin.output_direct} (should be 'off')")
        print("Reading X and Y directly from IQ module registers")

    def capture_lockin(self):
        """Read X and Y directly from IQ module registers"""
        try:
            # Access the quadrature values directly from the IQ module
            # These are the actual demodulated outputs!
            data = self.lockin._nadata()
            
            if data is not None and len(data) == 2:
                X_value = data[0]  # In-phase
                Y_value = data[1]  # Quadrature
                
                self.lockin_X.append(X_value)
                self.lockin_Y.append(Y_value)
                self.timestamps.append(time.time())
                return X_value, Y_value
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Error during capture: {e}")
            return None, None

    def run(self, params):
        timeout = params['timeout']
        self.setup_lockin(params)
        
        # Let the lock-in settle
        print("Waiting for lock-in to settle...")
        time.sleep(1.0)  # Give it time to stabilize
        
        loop_start = time.time()
        capture_count = 0
        while (time.time() - loop_start) < timeout:
            X, Y = self.capture_lockin()
            if X is not None:
                capture_count += 1
            time.sleep(0.02)  # ~50 Hz capture rate
        
        print(f"Captured {capture_count} samples")
        
        self.all_X = np.array(self.lockin_X)
        self.all_Y = np.array(self.lockin_Y)
        
        print(f"Total X samples: {len(self.all_X)}")
        print(f"Total Y samples: {len(self.all_Y)}")
        
        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)
        
        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            print(f"Applied {averaging_window}-sample moving average filter")
        
        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)
        
        # Time array
        t = np.arange(len(self.all_X)) / self.sample_rate
        
        # Capture raw signals for plotting
        out1_raw = np.array(self.scope._data_ch1)
        in1_raw = np.array(self.scope._data_ch2)
        scope_sample_rate = 125e6 / self.scope.decimation
        t_raw = np.arange(len(out1_raw)) / scope_sample_rate
        
        # FFT calculations
        iq = self.all_X + 1j * self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))
        idx = np.argmax(psd_lock)
        
        print("=" * 60)
        print("LOCK-IN DIAGNOSTICS")
        print("=" * 60)
        print(f"Reference Frequency Set: {self.ref_freq} Hz")
        print(f"FFT Peak Found at: {freqs_lock[idx]:.2f} Hz")
        print(f"Peak Offset from 0 Hz: {abs(freqs_lock[idx]):.2f} Hz")
        
        if abs(freqs_lock[idx]) < 5:
            print("✓ Lock-in is LOCKED (peak near 0 Hz)")
        else:
            print("✗ WARNING: Lock-in NOT locked! Peak should be at 0 Hz!")
        
        print(f"Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_X)}")
        print(f"Measurement Duration: {len(self.all_X) / self.sample_rate:.3f} seconds")
        print("-" * 60)
        print(f"Mean R: {np.mean(R):.6f} V ± {np.std(R):.6f} V")
        print(f"SNR (R): {np.mean(R) / (np.std(R) + 1e-9):.2f} (mean/std)")
        print(f"R range: [{np.min(R):.6f}, {np.max(R):.6f}] V")
        
        expected_R = params['ref_amp'] / 2
        if abs(np.mean(R) - expected_R) < 0.05:
            print(f"✓ R close to expected {expected_R:.3f}V")
        else:
            print(f"✗ R differs from expected {expected_R:.3f}V")
            print(f"  Difference: {abs(np.mean(R) - expected_R):.3f}V")
        
        print("-" * 60)
        print(f"Mean X: {np.mean(self.all_X):.6f} V ± {np.std(self.all_X):.6f} V")
        print(f"Mean Y: {np.mean(self.all_Y):.6f} V ± {np.std(self.all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.6f} rad ± {np.std(Theta):.6f} rad")
        print(f"Theta range: [{np.min(Theta):.6f}, {np.max(Theta):.6f}] rad")
        print(f"Phase stability: {np.std(Theta):.3f} rad (lower is better)")
        
        X_ac = np.std(self.all_X)
        Y_ac = np.std(self.all_Y)
        X_dc = np.mean(np.abs(self.all_X))
        Y_dc = np.mean(np.abs(self.all_Y))
        
        print("-" * 60)
        print("Signal characteristics:")
        print(f"X: DC={X_dc:.6f}V, AC={X_ac:.6f}V, AC/DC={X_ac / max(X_dc, 0.001):.3f}")
        print(f"Y: DC={Y_dc:.6f}V, AC={Y_ac:.6f}V, AC/DC={Y_ac / max(Y_dc, 0.001):.3f}")
        
        if X_ac / max(X_dc, 0.001) > 0.5:
            print("⚠ WARNING: X is oscillating! Should be flat for locked signal")
        if Y_ac / max(Y_dc, 0.001) > 0.5:
            print("⚠ WARNING: Y is oscillating! Should be flat for locked signal")
        
        print("=" * 60)
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(16, 10))
        
        # 1. OUT1 (Reference Signal from ASG0)
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_samples_plot = int(n_periods * scope_sample_rate / self.ref_freq)
        n_samples_plot = min(n_samples_plot, len(out1_raw))
        ax1.plot(t_raw[:n_samples_plot] * 1000, out1_raw[:n_samples_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference Signal (OUT1 from ASG0) @ {self.ref_freq} Hz')
        ax1.grid(True)
        
        # 2. IN1 (Input Signal)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw[:n_samples_plot] * 1000, in1_raw[:n_samples_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title('Input Signal (IN1)')
        ax2.grid(True)
        
        # 3. FFT Spectrum
        ax3 = plt.subplot(3, 3, 3)
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz (target)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power (a.u.)')
        ax3.set_title('FFT Spectrum (baseband)')
        ax3.legend()
        ax3.grid(True)
        
        # 4. X vs Time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(self.all_X), color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(self.all_X):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-phase (X) vs Time')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(self.all_Y), color='b', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y) vs Time')
        ax5.legend()
        ax5.grid(True)
        
        # 6. X vs Y (IQ plot)
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(self.all_X, self.all_Y, 'g.', markersize=2, alpha=0.5)
        ax6.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r+', markersize=15, 
                markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot (X vs Y)')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')
        
        # 7. R vs Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, R, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(R):.4f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R) vs Time')
        ax7.legend()
        ax7.grid(True)
        
        # 8. Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(Theta):.4f} rad')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (rad)')
        ax8.set_title('Phase (Theta) vs Time')
        ax8.legend()
        ax8.grid(True)
        
        # 9. R vs Theta
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(Theta, R, 'purple', marker='.', markersize=2, linestyle='', alpha=0.5)
        ax9.axhline(np.mean(R), color='b', linestyle='--', alpha=0.5)
        ax9.axvline(np.mean(Theta), color='r', linestyle='--', alpha=0.5)
        ax9.set_xlabel('Theta (rad)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('R vs Theta')
        ax9.grid(True)
        
        plt.tight_layout()
        
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            img_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.png')
            data = np.column_stack((R, Theta, self.all_X, self.all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.csv')
            np.savetxt(csv_path, data, delimiter=",", header="R,Theta,X,Y", comments='', fmt='%.6f')
            plt.savefig(img_path, dpi=150)
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitaya(output_dir=OUTPUT_DIRECTORY, yaml_file=YAML_FILE)
    
    run_params = {
        'ref_freq': REF_FREQUENCY,
        'ref_amp': REF_AMPLITUDE,
        'output_ref': OUTPUT_CHANNEL,
        'phase': PHASE_OFFSET,
        'timeout': MEASUREMENT_TIME,
        'filter_bandwidth': FILTER_BANDWIDTH,
        'averaging_window': AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'fft': SHOW_FFT,
    }
    
    print("=" * 60)
    print("RED PITAYA LOCK-IN AMPLIFIER")
    print("=" * 60)
    print("SETUP: Connect OUT1 directly to IN1")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print("=" * 60)
    print("Expected for direct OUT1→IN1 connection:")
    print(f"  X = {REF_AMPLITUDE / 2:.3f} V (in-phase)")
    print("  Y = 0.000 V (quadrature)")
    print(f"  R = {REF_AMPLITUDE / 2:.3f} V (magnitude)")
    print("  Theta = 0.000 rad (phase)")
    print("  FFT peak at 0 Hz")
    print("=" * 60)
    
    rp.run(run_params)
