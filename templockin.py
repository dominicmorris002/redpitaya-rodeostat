"""
Red Pitaya Lock-In Amplifier - CORRECTED

SETUP: Connect OUT1 directly to IN1 with a cable

Based on official PyRPL documentation example.
The IQ module does BOTH: outputs modulation to OUT1 AND demodulates IN1.
"""

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
REF_FREQUENCY = 100        # Hz
REF_AMPLITUDE = 0.4        # V - will appear on OUT1  
OUTPUT_CHANNEL = 'out1'    # 'out1' or 'out2'
PHASE_OFFSET = 0           # degrees
MEASUREMENT_TIME = 5.0     # seconds

FILTER_BANDWIDTH = 10      # Hz
AVERAGING_WINDOW = 1       # samples

SAVE_DATA = False
OUTPUT_DIRECTORY = 'test_data'

HOSTNAME = 'rp-f073ce.local'
YAML_FILE = 'lockin_scope_config.yml'
# ============================================================

from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import yaml

class RedPitaya:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', yaml_file=YAML_FILE):
        self.output_dir = output_dir
        self.yaml_file = yaml_file
        
        # Create YAML
        self.create_yaml()
        
        # Connect
        self.rp = Pyrpl(config=self.yaml_file, hostname=HOSTNAME)
        self.rp_modules = self.rp.rp
        
        # Get modules - IQ module will do EVERYTHING
        self.lockin = self.rp_modules.iq2
        self.scope = self.rp_modules.scope
        
        self.lockin_X = []
        self.lockin_Y = []
        self.timestamps = []

        print("Available scope inputs:", self.scope.inputs)
        
        # Setup scope to view OUT1 and IN1
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        self.scope.decimation = 64
        self.scope.duration = 0.01
        self.scope.average = False
        self.scope.trigger_source = 'immediately'
        self.scope.running_state = 'running_continuous'
        
        self.sample_rate = 50  # Lock-in sampling rate

    def create_yaml(self):
        """Create YAML config - same style as working example"""
        if not os.path.exists(self.yaml_file):
            config = {
                'redpitaya_hostname': HOSTNAME,
                'modules': ['scope', 'iq2'],
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
                    'decimation': 64
                },
                'iq2': {
                    'input': 'in1',
                    'frequency': REF_FREQUENCY,
                    'bandwidth': FILTER_BANDWIDTH,
                    'gain': 0.0,
                    'phase': PHASE_OFFSET,
                    'acbandwidth': 0,
                    'amplitude': REF_AMPLITUDE,
                    'output_direct': OUTPUT_CHANNEL,
                    'output_signal': 'quadrature',
                    'quadrature_factor': 1
                }
            }
            with open(self.yaml_file, 'w') as f:
                yaml.dump(config, f)
            print(f"Created YAML: {self.yaml_file}")
        else:
            print(f"YAML exists: {self.yaml_file}")

    def setup_lockin(self, params):
        """Setup IQ module - it does BOTH modulation output AND demodulation"""
        self.ref_freq = params['ref_freq']
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)
        
        # IQ module setup - following official PyRPL documentation
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,
            phase=phase_setting,
            acbandwidth=0,
            amplitude=ref_amp,                    # Modulation amplitude to OUT1
            input='in1',                          # Demodulate signal from IN1
            output_direct=params['output_ref'],   # Send modulation to OUT1
            output_signal='quadrature',
            quadrature_factor=1)
        
        print(f"🔊 IQ2 Module Setup:")
        print(f"  Frequency: {self.ref_freq} Hz")
        print(f"  Amplitude: {ref_amp} V → {params['output_ref']}")
        print(f"  Input: in1")
        print(f"  Filter BW: {filter_bw} Hz")
        print(f"  Reading X and Y from IQ module registers")

    def capture_lockin(self):
        """Read X and Y directly from IQ module"""
        try:
            # The IQ module stores demodulated values internally
            # Access them via _nadata() method
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
            print(f"⚠️ Error: {e}")
            return None, None

    def run(self, params):
        timeout = params['timeout']
        self.setup_lockin(params)
        
        print("Waiting for lock-in to settle...")
        time.sleep(1.0)
        
        loop_start = time.time()
        capture_count = 0
        while (time.time() - loop_start) < timeout:
            X, Y = self.capture_lockin()
            if X is not None:
                capture_count += 1
            time.sleep(0.02)  # ~50 Hz
        
        print(f"Captured {capture_count} samples")
        
        self.all_X = np.array(self.lockin_X)
        self.all_Y = np.array(self.lockin_Y)
        
        # Apply averaging if requested
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            print(f"Applied {averaging_window}-sample moving average")
        
        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)
        t = np.arange(len(self.all_X)) / self.sample_rate
        
        # Capture scope traces
        out1_raw = np.array(self.scope._data_ch1)
        in1_raw = np.array(self.scope._data_ch2)
        scope_rate = 125e6 / 64
        t_raw = np.arange(len(out1_raw)) / scope_rate
        
        # FFT
        iq = self.all_X + 1j * self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))
        idx = np.argmax(psd_lock)
        
        # Print diagnostics
        print("=" * 60)
        print("LOCK-IN DIAGNOSTICS")
        print("=" * 60)
        print(f"Reference: {self.ref_freq} Hz")
        print(f"FFT Peak: {freqs_lock[idx]:.2f} Hz")
        print(f"Peak Offset from 0 Hz: {abs(freqs_lock[idx]):.2f} Hz")
        
        if abs(freqs_lock[idx]) < 5:
            print("✓ LOCKED (peak near 0 Hz)")
        else:
            print("✗ NOT LOCKED!")
        
        print(f"Samples: {len(self.all_X)}")
        print(f"Duration: {len(self.all_X) / self.sample_rate:.3f} s")
        print("-" * 60)
        print(f"Mean R: {np.mean(R):.6f} V ± {np.std(R):.6f} V")
        print(f"SNR: {np.mean(R) / (np.std(R) + 1e-9):.2f}")
        
        expected_R = params['ref_amp'] / 2
        if abs(np.mean(R) - expected_R) < 0.05:
            print(f"✓ R close to expected {expected_R:.3f}V")
        else:
            print(f"✗ R = {np.mean(R):.3f}V, expected {expected_R:.3f}V")
        
        print("-" * 60)
        print(f"Mean X: {np.mean(self.all_X):.6f} V ± {np.std(self.all_X):.6f} V")
        print(f"Mean Y: {np.mean(self.all_Y):.6f} V ± {np.std(self.all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.6f} rad ± {np.std(Theta):.6f} rad")
        print("=" * 60)
        
        # Plot
        fig = plt.figure(figsize=(16, 10))
        
        # 1. OUT1
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_plot = int(min(n_periods * scope_rate / self.ref_freq, len(out1_raw)))
        ax1.plot(t_raw[:n_plot] * 1000, out1_raw[:n_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'OUT1 (from IQ2) @ {self.ref_freq} Hz')
        ax1.grid(True)
        
        # 2. IN1
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw[:n_plot] * 1000, in1_raw[:n_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V)')
        ax2.set_title('IN1')
        ax2.grid(True)
        
        # 3. FFT
        ax3 = plt.subplot(3, 3, 3)
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax3.axvline(0, color='r', linestyle='--', alpha=0.5, label='0 Hz')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.set_title('FFT Spectrum')
        ax3.legend()
        ax3.grid(True)
        
        # 4. X vs Time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(self.all_X), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(self.all_X):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-phase (X)')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(self.all_Y), color='b', linestyle='--', 
                   label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y)')
        ax5.legend()
        ax5.grid(True)
        
        # 6. IQ Plot
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(self.all_X, self.all_Y, 'g.', markersize=2, alpha=0.5)
        ax6.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r+', 
                markersize=15, markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')
        
        # 7. R vs Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, R, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R), color='b', linestyle='--', 
                   label=f'Mean: {np.mean(R):.4f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R)')
        ax7.legend()
        ax7.grid(True)
        
        # 8. Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(Theta):.4f} rad')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (rad)')
        ax8.set_title('Phase (Theta)')
        ax8.legend()
        ax8.grid(True)
        
        # 9. R vs Theta
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(Theta, R, 'purple', marker='.', markersize=2, alpha=0.5)
        ax9.axhline(np.mean(R), color='b', linestyle='--', alpha=0.5)
        ax9.axvline(np.mean(Theta), color='r', linestyle='--', alpha=0.5)
        ax9.set_xlabel('Theta (rad)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('R vs Theta')
        ax9.grid(True)
        
        plt.tight_layout()
        
        if params['save_file']:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, f'lockin_{self.ref_freq}Hz.png'), dpi=150)
            data = np.column_stack((R, Theta, self.all_X, self.all_Y))
            np.savetxt(os.path.join(self.output_dir, f'lockin_{self.ref_freq}Hz.csv'), 
                      data, delimiter=",", header="R,Theta,X,Y", comments='', fmt='%.6f')
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
        'save_file': SAVE_DATA,
    }
    
    print("=" * 60)
    print("RED PITAYA LOCK-IN AMPLIFIER")
    print("=" * 60)
    print("SETUP: Connect OUT1 to IN1")
    print("=" * 60)
    print(f"IQ2 generates: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"IQ2 demodulates: IN1")
    print(f"Filter BW: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement: {MEASUREMENT_TIME} s")
    print("=" * 60)
    print(f"Expected: X ≈ {REF_AMPLITUDE/2:.3f}V, Y ≈ 0V, R ≈ {REF_AMPLITUDE/2:.3f}V")
    print("=" * 60)
    
    rp.run(run_params)
