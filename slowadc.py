"""
Red Pitaya Lock-In Amplifier - WITH SLOW ADC FOR SIMULTANEOUS DC MEASUREMENT

SETUP: 
- Connect OUT1 directly to IN1 (for lock-in amplifier)
- Connect DC voltage source to XADC pins (0-1V range)

This version uses:
- FAST ADC (IN1/IN2): Lock-in amplifier (X, Y outputs)
- SLOW ADC (XADC): Simultaneous DC voltage measurement

XADC PHYSICAL CONNECTIONS (on extension connector):
- XADC_0: Pin AI0 (DIO0_P)
- XADC_1: Pin AI1 (DIO1_P) 
- XADC_2: Pin AI2 (DIO2_P)
- XADC_3: Pin AI3 (DIO3_P)
- Ground: Any GND pin

VOLTAGE RANGE: 0-1V (use voltage divider if your signal exceeds 1V)
"""

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
# Lock-in settings
REF_FREQUENCY = 100  # Hz - AC excitation frequency
REF_AMPLITUDE = 0.5  # V - AC signal amplitude
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0
MEASUREMENT_TIME = 30.0  # seconds
FILTER_BANDWIDTH = 10  # Hz
AVERAGING_WINDOW = 1

# Slow ADC settings
SLOW_ADC_CHANNELS = ['xadc_0']  # Which XADC channels to read (xadc_0, xadc_1, xadc_2, xadc_3)
SLOW_ADC_SAMPLE_RATE = 100  # Hz - how often to sample slow ADC (1-1000 Hz recommended)

# Data saving
SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'
SAVE_TIMESTAMPS = True

# Advanced settings
DECIMATION = 8192
SHOW_FFT = True
# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os
from datetime import datetime
import threading


class RedPitaya:
    electrode_map = {'A': (False, False), 'B': (True, False),
                     'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True),
                         '100uA': (True, False, True, True),
                         '1mA': (True, True, False, True),
                         '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True),
                    '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='lockin_config', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        
        # Slow ADC sampler module
        self.sampler = self.rp_modules.sampler
        
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []

        # Store capture timestamps
        self.capture_timestamps = []
        self.acquisition_start_time = None

        # NEW: Slow ADC data storage
        self.slow_adc_data = {}  # {channel_name: [(timestamp, value), ...]}
        self.slow_adc_thread = None
        self.slow_adc_running = False

        self.pid = self.rp_modules.pid0
        self.scope = self.rp_modules.scope

        print("Available scope inputs:", self.scope.inputs)

        # Fast ADC for lock-in
        self.scope.input1 = 'iq2'  # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation
        
        # Check available XADC channels
        print("\nAvailable Slow ADC (XADC) channels:")
        for i in range(4):
            channel = f'xadc_{i}'
            try:
                val = getattr(self.sampler, channel)
                print(f"  {channel}: {val:.6f} V")
            except Exception as e:
                print(f"  {channel}: Error - {e}")

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        # Turn OFF ASG0
        self.ref_sig.output_direct = 'off'
        print("ASG0 disabled - IQ module will generate reference")

        # IQ module setup
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,
            phase=phase_setting,
            acbandwidth=0,
            amplitude=ref_amp,
            input='in1',
            output_direct=params['output_ref'],
            output_signal='quadrature',
            quadrature_factor=1)

        print(f"Lock-in setup: {self.ref_freq} Hz, Amplitude: {ref_amp}V")
        print(f"Filter BW: {filter_bw} Hz")

    def start_slow_adc_sampling(self, channels, sample_rate):
        """Start background thread to continuously sample slow ADC channels"""
        self.slow_adc_running = True
        self.slow_adc_channels = channels
        self.slow_adc_sample_rate = sample_rate
        
        # Initialize data storage
        for channel in channels:
            self.slow_adc_data[channel] = []
        
        def sample_loop():
            sample_interval = 1.0 / sample_rate
            while self.slow_adc_running:
                timestamp = time.time()
                for channel in self.slow_adc_channels:
                    try:
                        value = getattr(self.sampler, channel)
                        self.slow_adc_data[channel].append((timestamp, value))
                    except Exception as e:
                        print(f"Error reading {channel}: {e}")
                
                time.sleep(sample_interval)
        
        self.slow_adc_thread = threading.Thread(target=sample_loop, daemon=True)
        self.slow_adc_thread.start()
        print(f"\n✓ Slow ADC sampling started: {channels} @ {sample_rate} Hz")

    def stop_slow_adc_sampling(self):
        """Stop slow ADC sampling"""
        self.slow_adc_running = False
        if self.slow_adc_thread:
            self.slow_adc_thread.join(timeout=2.0)
        print("✓ Slow ADC sampling stopped")

    def capture_lockin(self):
        """Captures scope data and appends to X and Y arrays with timestamps"""
        capture_time = time.time()

        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)  # iq2 = X
        ch2 = np.array(self.scope._data_ch2_current)  # iq2_2 = Y

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_timestamps.append(capture_time)

        return ch1, ch2

    def run(self, params):
        timeout = params['timeout']
        slow_adc_channels = params.get('slow_adc_channels', [])
        slow_adc_rate = params.get('slow_adc_rate', 100)
        
        self.setup_lockin(params)

        # Start slow ADC sampling if requested
        if slow_adc_channels:
            self.start_slow_adc_sampling(slow_adc_channels, slow_adc_rate)

        # Let the lock-in settle
        print("Waiting for lock-in to settle...")
        time.sleep(0.5)

        # Record absolute start time
        self.acquisition_start_time = time.time()
        print(f"\n✓ Acquisition started at: {datetime.fromtimestamp(self.acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Main acquisition loop
        loop_start = time.time()
        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        # Stop slow ADC sampling
        if slow_adc_channels:
            self.stop_slow_adc_sampling()

        # Process lock-in data
        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        # Generate per-sample timestamps for lock-in data
        total_samples = len(self.all_X)
        self.sample_timestamps = np.zeros(total_samples)
        sample_idx = 0

        for i, capture_time in enumerate(self.capture_timestamps):
            n_samples = len(self.lockin_X[i])
            capture_duration = n_samples / self.sample_rate
            sample_times = np.linspace(0, capture_duration, n_samples, endpoint=False)
            self.sample_timestamps[sample_idx:sample_idx + n_samples] = capture_time + sample_times
            sample_idx += n_samples

        # Apply moving average filter
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            self.sample_timestamps = self.sample_timestamps[:len(self.all_X)]
            print(f"Applied {averaging_window}-sample moving average filter")

        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)
        t = self.sample_timestamps - self.acquisition_start_time

        # Print diagnostics
        print("=" * 60)
        print("LOCK-IN DIAGNOSTICS")
        print("=" * 60)
        print(f"Sample Rate: {self.sample_rate:.2f} Hz")
        print(f"Total Samples: {len(self.all_X)}")
        print(f"Measurement Duration: {t[-1]:.3f} seconds")
        print(f"Mean R: {np.mean(R):.6f} V ± {np.std(R):.6f} V")
        print(f"Mean X: {np.mean(self.all_X):.6f} V ± {np.std(self.all_X):.6f} V")
        print(f"Mean Y: {np.mean(self.all_Y):.6f} V ± {np.std(self.all_Y):.6f} V")
        
        # Print slow ADC diagnostics
        if slow_adc_channels:
            print("\n" + "=" * 60)
            print("SLOW ADC DIAGNOSTICS")
            print("=" * 60)
            for channel in slow_adc_channels:
                data = self.slow_adc_data[channel]
                if data:
                    values = np.array([v for _, v in data])
                    timestamps = np.array([t for t, _ in data])
                    rel_times = timestamps - self.acquisition_start_time
                    
                    print(f"\n{channel}:")
                    print(f"  Samples collected: {len(values)}")
                    print(f"  Time range: {rel_times[0]:.3f} to {rel_times[-1]:.3f} s")
                    print(f"  Mean: {np.mean(values):.6f} V")
                    print(f"  Std: {np.std(values):.6f} V")
                    print(f"  Min/Max: {np.min(values):.6f} / {np.max(values):.6f} V")
        
        print("=" * 60)

        # Create comprehensive plot
        n_subplots = 8 + len(slow_adc_channels)  # Base plots + one per XADC channel
        fig = plt.figure(figsize=(16, 4 * ((n_subplots + 2) // 3)))
        
        plot_idx = 1

        # 1. X vs Time
        ax = plt.subplot(3, 3, plot_idx)
        plot_idx += 1
        ax.plot(t, self.all_X, 'b-', linewidth=0.5)
        ax.axhline(np.mean(self.all_X), color='r', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(self.all_X):.4f}V')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('X (V)')
        ax.set_title('Lock-in In-phase (X)')
        ax.legend()
        ax.grid(True)

        # 2. Y vs Time
        ax = plt.subplot(3, 3, plot_idx)
        plot_idx += 1
        ax.plot(t, self.all_Y, 'r-', linewidth=0.5)
        ax.axhline(np.mean(self.all_Y), color='b', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Y (V)')
        ax.set_title('Lock-in Quadrature (Y)')
        ax.legend()
        ax.grid(True)

        # 3. R vs Time
        ax = plt.subplot(3, 3, plot_idx)
        plot_idx += 1
        ax.plot(t, R, 'm-', linewidth=0.5)
        ax.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(R):.4f}V')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('R (V)')
        ax.set_title('Lock-in Magnitude (R)')
        ax.legend()
        ax.grid(True)

        # 4. Theta vs Time
        ax = plt.subplot(3, 3, plot_idx)
        plot_idx += 1
        ax.plot(t, Theta, 'c-', linewidth=0.5)
        ax.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(Theta):.4f} rad')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Theta (rad)')
        ax.set_title('Lock-in Phase (Theta)')
        ax.legend()
        ax.grid(True)

        # 5. X vs Y (IQ plot)
        ax = plt.subplot(3, 3, plot_idx)
        plot_idx += 1
        ax.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
        ax.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r+', markersize=15,
                markeredgewidth=2, label='Mean')
        ax.set_xlabel('X (V)')
        ax.set_ylabel('Y (V)')
        ax.set_title('IQ Plot (X vs Y)')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # 6. R vs Theta
        ax = plt.subplot(3, 3, plot_idx)
        plot_idx += 1
        ax.plot(Theta, R, 'purple', marker='.', markersize=1, linestyle='', alpha=0.5)
        ax.axhline(np.mean(R), color='b', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(Theta), color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Theta (rad)')
        ax.set_ylabel('R (V)')
        ax.set_title('R vs Theta')
        ax.grid(True)

        # 7-N: Slow ADC channels
        for channel in slow_adc_channels:
            data = self.slow_adc_data[channel]
            if data:
                timestamps = np.array([t for t, _ in data])
                values = np.array([v for _, v in data])
                rel_times = timestamps - self.acquisition_start_time
                
                ax = plt.subplot(3, 3, plot_idx)
                plot_idx += 1
                ax.plot(rel_times, values, 'b-', linewidth=1)
                ax.axhline(np.mean(values), color='r', linestyle='--', alpha=0.7,
                          label=f'Mean: {np.mean(values):.4f}V')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (V)')
                ax.set_title(f'Slow ADC - {channel}')
                ax.legend()
                ax.grid(True)

        plt.tight_layout()

        # Save data
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save plot
            img_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Plot saved: {img_path}")

            # Save lock-in data
            if params.get('save_timestamps', False):
                data = np.column_stack((
                    self.sample_timestamps,
                    t,
                    R,
                    Theta,
                    self.all_X,
                    self.all_Y
                ))
                csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')
                np.savetxt(csv_path, data, delimiter=",",
                          header="AbsoluteTimestamp,RelativeTime,R,Theta,X,Y",
                          comments='', fmt='%.10f')
                print(f"✓ Lock-in data saved: {csv_path}")

            # Save slow ADC data
            for channel in slow_adc_channels:
                data = self.slow_adc_data[channel]
                if data:
                    timestamps = np.array([t for t, _ in data])
                    values = np.array([v for _, v in data])
                    rel_times = timestamps - self.acquisition_start_time
                    
                    adc_data = np.column_stack((timestamps, rel_times, values))
                    csv_path = os.path.join(self.output_dir, 
                                          f'slowadc_{channel}_{timestamp_str}.csv')
                    np.savetxt(csv_path, adc_data, delimiter=",",
                              header="AbsoluteTimestamp,RelativeTime,Voltage",
                              comments='', fmt='%.10f')
                    print(f"✓ Slow ADC data saved: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitaya()

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
        'save_timestamps': SAVE_TIMESTAMPS,
        'slow_adc_channels': SLOW_ADC_CHANNELS,  # NEW
        'slow_adc_rate': SLOW_ADC_SAMPLE_RATE,   # NEW
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN + SLOW ADC SIMULTANEOUS ACQUISITION")
    print("=" * 60)
    print(f"Lock-in: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V on {OUTPUT_CHANNEL}")
    print(f"Slow ADC: {SLOW_ADC_CHANNELS} @ {SLOW_ADC_SAMPLE_RATE} Hz")
    print(f"Measurement Time: {MEASUREMENT_TIME} s")
    print("=" * 60)

    rp.run(run_params)
