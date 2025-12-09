"""
Red Pitaya DUAL-CHANNEL Lock-In Amplifier - WITH TIMESTAMP SYNCHRONIZATION

SETUP: 
- OUT1 connected to your electrochemical cell (via Red Pitaya's Autolab connector or directly)
- IN1 measures electrochemical response
- IN2 measures Autolab current (or any other signal at the same frequency)

This version uses TWO IQ modules to simultaneously demodulate both channels
WITHOUT using the scope! Direct reading from IQ modules = no bandwidth limits!

IQ MODULE OUTPUTS:
- iq2: Demodulates IN1 (electrochemistry) → outputs X1, Y1
- iq0: Demodulates IN2 (Autolab current) → outputs X2, Y2

SYNCHRONIZATION:
- Both channels perfectly synchronized (same timestamp for each sample pair)
- Saves CSV with absolute timestamps for merging with external data
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 100  # Hz - AC excitation frequency
REF_AMPLITUDE = 0.5  # V - AC signal amplitude (will appear on OUT1)
OUTPUT_CHANNEL = 'out1'  # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET_CH1 = 0  # degrees - phase adjustment for IN1
PHASE_OFFSET_CH2 = 0  # degrees - phase adjustment for IN2
MEASUREMENT_TIME = 30.0  # seconds - how long to measure

# LOCK-IN FILTER BANDWIDTH
FILTER_BANDWIDTH = 10  # Hz - lower = cleaner, higher = faster response

# SAMPLING RATE (direct IQ reading)
SAMPLE_RATE = 100  # Hz - how often to read IQ module outputs (1-1000 Hz typical)

# AVERAGING
AVERAGING_WINDOW = 1  # samples - set to 1 to see raw lock-in output first

# Data saving
SAVE_DATA = True  # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'

# Advanced settings
SHOW_FFT = True
SAVE_TIMESTAMPS = True  # Save absolute timestamps for sync with NI-DAQ
# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os
from datetime import datetime


class DualChannelRedPitaya:
    
    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='dual_lockin_config', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        
        # TWO lock-in amplifiers!
        self.lockin1 = self.rp_modules.iq2  # IN1 (electrochemistry)
        self.lockin2 = self.rp_modules.iq0  # IN2 (Autolab current)
        
        # Data storage for Channel 1 (IN1)
        self.all_X1 = []
        self.all_Y1 = []
        
        # Data storage for Channel 2 (IN2)
        self.all_X2 = []
        self.all_Y2 = []
        
        # Timestamp storage
        self.sample_timestamps = []
        self.acquisition_start_time = None
        
        print("✓ Dual-channel lock-in initialized")
        print("  Channel 1: iq2 → IN1")
        print("  Channel 2: iq0 → IN2")
    
    def setup_dual_lockin(self, params):
        self.ref_freq = params['ref_freq']
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_ch1 = params.get('phase_ch1', 0)
        phase_ch2 = params.get('phase_ch2', 0)
        
        print("\n" + "=" * 60)
        print("CONFIGURING DUAL LOCK-IN AMPLIFIERS")
        print("=" * 60)
        
        # CHANNEL 1 (IN1): Generates reference AND demodulates
        self.lockin1.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,  # No feedback
            phase=phase_ch1,
            acbandwidth=0,  # DC-coupled input
            amplitude=ref_amp,  # Output amplitude
            input='in1',
            output_direct=params['output_ref'],  # Send sine wave to OUT1/OUT2
            output_signal='quadrature',
            quadrature_factor=1)
        
        print(f"✓ Channel 1 (iq2):")
        print(f"  Input: IN1")
        print(f"  Generates: {ref_amp}V sine @ {self.ref_freq}Hz on {params['output_ref']}")
        print(f"  Filter BW: {filter_bw} Hz")
        print(f"  Phase: {phase_ch1}°")
        
        # CHANNEL 2 (IN2): Only demodulates (no output)
        self.lockin2.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,  # No feedback
            phase=phase_ch2,
            acbandwidth=0,  # DC-coupled input
            amplitude=0.0,  # NO output from this channel
            input='in2',
            output_direct='off',  # Don't output anything
            output_signal='quadrature',
            quadrature_factor=1)
        
        print(f"✓ Channel 2 (iq0):")
        print(f"  Input: IN2")
        print(f"  Demodulates at: {self.ref_freq}Hz")
        print(f"  Filter BW: {filter_bw} Hz")
        print(f"  Phase: {phase_ch2}°")
        print("=" * 60)
    
    def read_iq_direct(self):
        """Read IQ module outputs directly - NO SCOPE NEEDED!"""
        timestamp = time.time()
        
        # Read Channel 1 (IN1) - iq2 module
        X1 = self.lockin1.iq  # In-phase (X)
        Y1 = self.lockin1.iq2  # Quadrature (Y) - might be .iq_2 depending on PyRPL version
        
        # Read Channel 2 (IN2) - iq0 module
        X2 = self.lockin2.iq  # In-phase (X)
        Y2 = self.lockin2.iq2  # Quadrature (Y)
        
        return timestamp, X1, Y1, X2, Y2
    
    def run(self, params):
        timeout = params['timeout']
        sample_rate = params.get('sample_rate', 100)  # Hz
        sample_period = 1.0 / sample_rate
        
        self.setup_dual_lockin(params)
        
        # Let the lock-ins settle
        print("\nWaiting for lock-ins to settle...")
        time.sleep(0.5)
        
        # Record absolute start time
        self.acquisition_start_time = time.time()
        print(f"✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"  Sampling at {sample_rate} Hz")
        print(f"  Duration: {timeout} seconds")
        print(f"  Expected samples: ~{int(timeout * sample_rate)}")
        
        loop_start = time.time()
        sample_count = 0
        
        # MAIN ACQUISITION LOOP
        while (time.time() - loop_start) < timeout:
            # Read both channels simultaneously
            timestamp, X1, Y1, X2, Y2 = self.read_iq_direct()
            
            # Store data
            self.all_X1.append(X1)
            self.all_Y1.append(Y1)
            self.all_X2.append(X2)
            self.all_Y2.append(Y2)
            self.sample_timestamps.append(timestamp)
            
            sample_count += 1
            
            # Wait for next sample
            time.sleep(sample_period)
        
        print(f"✓ Acquisition complete: {sample_count} samples collected")
        
        # Convert to numpy arrays
        self.all_X1 = np.array(self.all_X1)
        self.all_Y1 = np.array(self.all_Y1)
        self.all_X2 = np.array(self.all_X2)
        self.all_Y2 = np.array(self.all_Y2)
        self.sample_timestamps = np.array(self.sample_timestamps)
        
        # Apply moving average filter if requested
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            self.all_X1 = np.convolve(self.all_X1, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y1 = np.convolve(self.all_Y1, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_X2 = np.convolve(self.all_X2, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y2 = np.convolve(self.all_Y2, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_X1)]
            print(f"✓ Applied {averaging_window}-sample moving average filter")
        
        # Calculate R and Theta for both channels
        R1 = np.sqrt(self.all_X1**2 + self.all_Y1**2)
        Theta1 = np.arctan2(self.all_Y1, self.all_X1)
        
        R2 = np.sqrt(self.all_X2**2 + self.all_Y2**2)
        Theta2 = np.arctan2(self.all_Y2, self.all_X2)
        
        # Time array (relative to start)
        t = self.sample_timestamps - self.acquisition_start_time
        
        # Print diagnostics
        self.print_diagnostics(t, R1, Theta1, R2, Theta2, params)
        
        # Plot results
        self.plot_results(t, R1, Theta1, R2, Theta2)
        
        # Save data
        if params['save_file']:
            self.save_data(t, R1, Theta1, R2, Theta2, params)
        else:
            plt.show()
    
    def print_diagnostics(self, t, R1, Theta1, R2, Theta2, params):
        print("\n" + "=" * 60)
        print("DUAL-CHANNEL LOCK-IN DIAGNOSTICS")
        print("=" * 60)
        print(f"Reference Frequency: {self.ref_freq} Hz")
        print(f"Measurement Duration: {t[-1]:.3f} seconds")
        print(f"Total Samples: {len(self.all_X1)}")
        print(f"Actual Sample Rate: {len(self.all_X1) / t[-1]:.2f} Hz")
        
        print("\n" + "-" * 60)
        print("TIMESTAMP INFORMATION:")
        print(f"Start: {datetime.fromtimestamp(self.sample_timestamps[0]).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"End:   {datetime.fromtimestamp(self.sample_timestamps[-1]).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        print("\n" + "-" * 60)
        print("CHANNEL 1 (IN1 - Electrochemistry):")
        print(f"  Mean R1: {np.mean(R1):.6f} V ± {np.std(R1):.6f} V")
        print(f"  SNR (R1): {np.mean(R1) / (np.std(R1) + 1e-9):.2f}")
        print(f"  Mean X1: {np.mean(self.all_X1):.6f} V ± {np.std(self.all_X1):.6f} V")
        print(f"  Mean Y1: {np.mean(self.all_Y1):.6f} V ± {np.std(self.all_Y1):.6f} V")
        print(f"  Mean Theta1: {np.mean(Theta1):.6f} rad ± {np.std(Theta1):.6f} rad")
        
        print("\n" + "-" * 60)
        print("CHANNEL 2 (IN2 - Autolab/External):")
        print(f"  Mean R2: {np.mean(R2):.6f} V ± {np.std(R2):.6f} V")
        print(f"  SNR (R2): {np.mean(R2) / (np.std(R2) + 1e-9):.2f}")
        print(f"  Mean X2: {np.mean(self.all_X2):.6f} V ± {np.std(self.all_X2):.6f} V")
        print(f"  Mean Y2: {np.mean(self.all_Y2):.6f} V ± {np.std(self.all_Y2):.6f} V")
        print(f"  Mean Theta2: {np.mean(Theta2):.6f} rad ± {np.std(Theta2):.6f} rad")
        
        print("\n" + "-" * 60)
        print("CROSS-CHANNEL ANALYSIS:")
        phase_diff = Theta1 - Theta2
        print(f"  Phase difference (Θ1 - Θ2): {np.mean(phase_diff):.6f} rad ± {np.std(phase_diff):.6f} rad")
        print(f"  R1/R2 ratio: {np.mean(R1) / (np.mean(R2) + 1e-9):.3f}")
        print("=" * 60)
    
    def plot_results(self, t, R1, Theta1, R2, Theta2):
        fig = plt.figure(figsize=(18, 12))
        
        # Row 1: Channel 1 (IN1)
        # X1 vs Time
        ax1 = plt.subplot(4, 3, 1)
        ax1.plot(t, self.all_X1, 'b-', linewidth=0.5)
        ax1.axhline(np.mean(self.all_X1), color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_X1):.4f}V')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('X1 (V)')
        ax1.set_title('Channel 1 (IN1): In-phase (X)')
        ax1.legend()
        ax1.grid(True)
        
        # Y1 vs Time
        ax2 = plt.subplot(4, 3, 2)
        ax2.plot(t, self.all_Y1, 'r-', linewidth=0.5)
        ax2.axhline(np.mean(self.all_Y1), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_Y1):.4f}V')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Y1 (V)')
        ax2.set_title('Channel 1 (IN1): Quadrature (Y)')
        ax2.legend()
        ax2.grid(True)
        
        # R1 vs Time
        ax3 = plt.subplot(4, 3, 3)
        ax3.plot(t, R1, 'm-', linewidth=0.5)
        ax3.axhline(np.mean(R1), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(R1):.4f}V')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('R1 (V)')
        ax3.set_title('Channel 1 (IN1): Magnitude')
        ax3.legend()
        ax3.grid(True)
        
        # Row 2: Channel 2 (IN2)
        # X2 vs Time
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(t, self.all_X2, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(self.all_X2), color='r', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_X2):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X2 (V)')
        ax4.set_title('Channel 2 (IN2): In-phase (X)')
        ax4.legend()
        ax4.grid(True)
        
        # Y2 vs Time
        ax5 = plt.subplot(4, 3, 5)
        ax5.plot(t, self.all_Y2, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(self.all_Y2), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(self.all_Y2):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y2 (V)')
        ax5.set_title('Channel 2 (IN2): Quadrature (Y)')
        ax5.legend()
        ax5.grid(True)
        
        # R2 vs Time
        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(t, R2, 'm-', linewidth=0.5)
        ax6.axhline(np.mean(R2), color='b', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(R2):.4f}V')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('R2 (V)')
        ax6.set_title('Channel 2 (IN2): Magnitude')
        ax6.legend()
        ax6.grid(True)
        
        # Row 3: IQ Plots
        # Channel 1 IQ plot
        ax7 = plt.subplot(4, 3, 7)
        ax7.plot(self.all_X1, self.all_Y1, 'g.', markersize=1, alpha=0.5)
        ax7.plot(np.mean(self.all_X1), np.mean(self.all_Y1), 'r+', markersize=15,
                 markeredgewidth=2, label='Mean')
        ax7.set_xlabel('X1 (V)')
        ax7.set_ylabel('Y1 (V)')
        ax7.set_title('Channel 1: IQ Plot')
        ax7.legend()
        ax7.grid(True)
        ax7.axis('equal')
        
        # Channel 2 IQ plot
        ax8 = plt.subplot(4, 3, 8)
        ax8.plot(self.all_X2, self.all_Y2, 'c.', markersize=1, alpha=0.5)
        ax8.plot(np.mean(self.all_X2), np.mean(self.all_Y2), 'r+', markersize=15,
                 markeredgewidth=2, label='Mean')
        ax8.set_xlabel('X2 (V)')
        ax8.set_ylabel('Y2 (V)')
        ax8.set_title('Channel 2: IQ Plot')
        ax8.legend()
        ax8.grid(True)
        ax8.axis('equal')
        
        # R1 vs R2
        ax9 = plt.subplot(4, 3, 9)
        ax9.plot(R1, R2, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax9.set_xlabel('R1 (V)')
        ax9.set_ylabel('R2 (V)')
        ax9.set_title('R1 vs R2 (Cross-channel)')
        ax9.grid(True)
        
        # Row 4: Phase analysis
        # Theta1 vs Time
        ax10 = plt.subplot(4, 3, 10)
        ax10.plot(t, Theta1, 'c-', linewidth=0.5)
        ax10.axhline(np.mean(Theta1), color='r', linestyle='--', alpha=0.7,
                     label=f'Mean: {np.mean(Theta1):.4f} rad')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Theta1 (rad)')
        ax10.set_title('Channel 1: Phase')
        ax10.legend()
        ax10.grid(True)
        
        # Theta2 vs Time
        ax11 = plt.subplot(4, 3, 11)
        ax11.plot(t, Theta2, 'c-', linewidth=0.5)
        ax11.axhline(np.mean(Theta2), color='r', linestyle='--', alpha=0.7,
                     label=f'Mean: {np.mean(Theta2):.4f} rad')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Theta2 (rad)')
        ax11.set_title('Channel 2: Phase')
        ax11.legend()
        ax11.grid(True)
        
        # Phase difference
        ax12 = plt.subplot(4, 3, 12)
        phase_diff = Theta1 - Theta2
        ax12.plot(t, phase_diff, 'orange', linewidth=0.5)
        ax12.axhline(np.mean(phase_diff), color='r', linestyle='--', alpha=0.7,
                     label=f'Mean: {np.mean(phase_diff):.4f} rad')
        ax12.set_xlabel('Time (s)')
        ax12.set_ylabel('Theta1 - Theta2 (rad)')
        ax12.set_title('Phase Difference (Θ1 - Θ2)')
        ax12.legend()
        ax12.grid(True)
        
        plt.tight_layout()
    
    def save_data(self, t, R1, Theta1, R2, Theta2, params):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save plot
        img_path = os.path.join(self.output_dir, f'dual_lockin_results_{timestamp_str}.png')
        plt.savefig(img_path, dpi=150)
        print(f"\n✓ Plot saved: {img_path}")
        
        # Save data with timestamps
        if params.get('save_timestamps', False):
            data = np.column_stack((
                self.sample_timestamps,  # Absolute Unix timestamp
                t,  # Relative time (s)
                # Channel 1 (IN1)
                R1, Theta1, self.all_X1, self.all_Y1,
                # Channel 2 (IN2)
                R2, Theta2, self.all_X2, self.all_Y2
            ))
            csv_path = os.path.join(self.output_dir, f'dual_lockin_results_{timestamp_str}.csv')
            np.savetxt(csv_path, data, delimiter=",",
                       header="AbsoluteTimestamp,RelativeTime,R1,Theta1,X1,Y1,R2,Theta2,X2,Y2",
                       comments='', fmt='%.10f')
            print(f"✓ Data saved with timestamps: {csv_path}")
            print(f"  Columns: AbsoluteTimestamp, RelativeTime, R1, Theta1, X1, Y1, R2, Theta2, X2, Y2")
        else:
            data = np.column_stack((
                t,
                R1, Theta1, self.all_X1, self.all_Y1,
                R2, Theta2, self.all_X2, self.all_Y2
            ))
            csv_path = os.path.join(self.output_dir, f'dual_lockin_results_{timestamp_str}.csv')
            np.savetxt(csv_path, data, delimiter=",",
                       header="Time,R1,Theta1,X1,Y1,R2,Theta2,X2,Y2",
                       comments='', fmt='%.6f')
            print(f"✓ Data saved: {csv_path}")


if __name__ == '__main__':
    rp = DualChannelRedPitaya()
    
    run_params = {
        'ref_freq': REF_FREQUENCY,
        'ref_amp': REF_AMPLITUDE,
        'output_ref': OUTPUT_CHANNEL,
        'phase_ch1': PHASE_OFFSET_CH1,
        'phase_ch2': PHASE_OFFSET_CH2,
        'timeout': MEASUREMENT_TIME,
        'filter_bandwidth': FILTER_BANDWIDTH,
        'sample_rate': SAMPLE_RATE,
        'averaging_window': AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'save_timestamps': SAVE_TIMESTAMPS,
    }
    
    print("=" * 60)
    print("RED PITAYA DUAL-CHANNEL LOCK-IN AMPLIFIER")
    print("=" * 60)
    print("SETUP:")
    print("  OUT1 → Electrochemical cell")
    print("  IN1  → Electrochemical response")
    print("  IN2  → Autolab current (or other signal)")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Sampling Rate: {SAMPLE_RATE} Hz (direct IQ reading)")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print(f"Averaging Window: {AVERAGING_WINDOW} samples")
    print("=" * 60)
    print("\n✓ Both channels will be sampled simultaneously")
    print("✓ Timestamps synchronized for both channels")
    print("✓ No scope limitations - direct IQ module access!")
    print("=" * 60)
    
    rp.run(run_params)
