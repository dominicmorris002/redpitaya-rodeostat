"""
Red Pitaya Lock-In Amplifier Data Logger

Uses PyRPL's built-in lock-in amplifier (iq2 module) and logs the results.
Connect OUT1 to IN1 with a cable for testing.
Supports AUTO, LV, HV, or MANUAL gain modes for scaling.
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from pyrpl import Pyrpl

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
REF_FREQUENCY = 500  # Hz
REF_AMPLITUDE = 1  # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 10.0  # seconds

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'
AUTOLAB_GAIN = 1e-3  # Based on Autolab "Current Scale" if Scale = 1mA : Set to 1e-3
MANUAL_GAIN_FACTOR = 28.78857669 * AUTOLAB_GAIN  # Only used if INPUT_MODE = 'MANUAL'
MANUAL_DC_OFFSET = -0.011800  # Only used if INPUT_MODE = 'MANUAL'

FILTER_BANDWIDTH = 10  # Hz
AVERAGING_WINDOW = 1  # samples (moving average on logged data)
DECIMATION = 1024

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

AUTO_CALIBRATE = True  # Only used if INPUT_MODE = 'AUTO'
CALIBRATION_TIME = 2.0  # seconds
# ============================================================

START_TIME_FILE = "start_time.txt"

# Synchronization with other processes
try:
    with open(START_TIME_FILE, "r") as f:
        START_TIME = datetime.fromisoformat(f.read().strip())
    while datetime.now() < START_TIME:
        time.sleep(0.001)
except FileNotFoundError:
    pass  # No sync file, start immediately


class RedPitayaLockInLogger:
    """Data logger for PyRPL's built-in lock-in amplifier"""

    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO', manual_gain=1.0, manual_offset=0.0):
        self.rp = Pyrpl(config='lockin_config5', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope

        self.lockin_X = []
        self.lockin_Y = []
        self.capture_times = []

        # Setup input gain scaling and DC offset
        self.input_gain_factor = manual_gain
        self.input_dc_offset = manual_offset
        self.input_mode_setting = input_mode.upper()
        self.input_mode = "Unknown"

        if self.input_mode_setting == 'MANUAL':
            self.input_gain_factor = manual_gain
            self.input_dc_offset = manual_offset
            self.input_mode = f"MANUAL ({manual_gain}x gain, {manual_offset}V offset)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_dc_offset = 0.0
            self.input_mode = "LV (±1V)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_dc_offset = 0.0
            self.input_mode = "HV (±20V, 20:1 divider)"
            print(f"Input mode: {self.input_mode}")
        elif self.input_mode_setting == 'AUTO':
            self.input_mode = "AUTO (will calibrate)"
            print("Input mode: AUTO - will auto-detect")

        # Setup scope to read lock-in outputs
        self.scope.input1 = 'iq2'  # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f'Invalid decimation. Must be one of {self.allowed_decimations}')

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True

        # Nominal sample rate (actual rate may vary slightly)
        self.nominal_sample_rate = 125e6 / self.scope.decimation

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        """
        Measure actual input scaling and DC offset by comparing OUT1 to IN1 directly.
        This measures the physical signal on IN1, not the lock-in output.
        """
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT SCALING AND DC OFFSET...")
        print("=" * 60)

        # Step 1: Measure DC offset with no signal
        print("Step 1: Measuring DC offset on IN1 (no signal)...")
        self.ref_sig.output_direct = 'off'
        self.lockin.output_direct = 'off'

        # Configure scope to read raw IN1
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'
        time.sleep(0.3)  # Let signal settle

        offset_samples = []
        for _ in range(10):
            self.scope.single()
            offset_samples.append(np.mean(self.scope._data_ch1_current))

        self.input_dc_offset = np.mean(offset_samples)
        print(f"  Measured DC offset: {self.input_dc_offset:.6f}V")

        # Step 2: Measure gain with calibration signal
        print(f"\nStep 2: Measuring gain with {cal_amp}V @ {cal_freq} Hz...")

        # Configure scope to read OUT1 and IN1
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'

        # Generate calibration signal using ASG
        self.ref_sig.setup(
            frequency=cal_freq,
            amplitude=cal_amp,
            offset=0,
            waveform='sin',
            trigger_source='immediately'
        )
        self.ref_sig.output_direct = 'out1'

        time.sleep(0.5)  # Let signal settle

        cal_out1 = []
        cal_in1 = []
        start_time = time.time()

        while (time.time() - start_time) < cal_time:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # OUT1
            ch2 = np.array(self.scope._data_ch2_current)  # IN1 (raw)
            cal_out1.append(ch1)
            cal_in1.append(ch2)

        # Concatenate all samples
        all_out1 = np.concatenate(cal_out1)
        all_in1 = np.concatenate(cal_in1)

        # Remove DC offset from IN1
        all_in1_corrected = all_in1 - self.input_dc_offset

        # Calculate peak-to-peak amplitudes
        out1_peak = (np.max(all_out1) - np.min(all_out1)) / 2
        in1_peak = (np.max(all_in1_corrected) - np.min(all_in1_corrected)) / 2

        # Also calculate RMS for comparison
        out1_rms = np.sqrt(np.mean(all_out1 ** 2))
        in1_rms = np.sqrt(np.mean(all_in1_corrected ** 2))

        # Gain factor = what we sent / what we measured
        self.input_gain_factor = out1_peak / in1_peak
        gain_rms = out1_rms / in1_rms

        # Classify the mode based on gain
        if self.input_gain_factor < 1.05:
            self.input_mode = "LV (±1V)"
        elif self.input_gain_factor < 2.0:
            self.input_mode = f"LV with loading ({self.input_gain_factor:.2f}x)"
        elif self.input_gain_factor < 15:
            self.input_mode = f"Custom/Unknown mode ({self.input_gain_factor:.2f}x)"
        else:
            self.input_mode = f"HV (±20V, {self.input_gain_factor:.1f}:1 divider)"

        print(f"\n  OUT1 peak: {out1_peak:.4f}V, RMS: {out1_rms:.4f}V")
        print(f"  IN1 peak (after offset correction): {in1_peak:.4f}V, RMS: {in1_rms:.4f}V")
        print(f"  Gain (peak-based): {self.input_gain_factor:.4f}x")
        print(f"  Gain (RMS-based): {gain_rms:.4f}x")
        print(f"  DC offset: {self.input_dc_offset:.6f}V")
        print(f"  Detected mode: {self.input_mode}")
        print("=" * 60 + "\n")

        # Restore scope to lock-in outputs
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        return self.input_gain_factor

    def setup_lockin(self, params):
        """Configure PyRPL's lock-in amplifier"""
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        self.ref_sig.output_direct = 'off'

        # Configure PyRPL's iq2 module (the actual lock-in)
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
            quadrature_factor=1.0)

        # Verify what PyRPL actually set
        actual_freq = self.lockin.frequency
        actual_amp = self.lockin.amplitude

        print(f"Lock-in frequency: {self.ref_freq} Hz (actual: {actual_freq:.2f} Hz)")
        print(f"Lock-in bandwidth: {filter_bw} Hz")
        print(f"Reference amplitude: {ref_amp}V on {params['output_ref']} (actual: {actual_amp:.3f}V)")

        if abs(actual_freq - self.ref_freq) > 0.1:
            print(f"⚠ WARNING: Requested {self.ref_freq} Hz but PyRPL set {actual_freq:.2f} Hz!")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")
        print(f"DC offset correction: {self.input_dc_offset:.6f}V")

    def capture_lockin(self):
        """Capture one scope trace of lock-in outputs"""
        capture_time = time.time()
        self.scope.single()

        ch1 = np.array(self.scope._data_ch1_current)  # X from PyRPL
        ch2 = np.array(self.scope._data_ch2_current)  # Y from PyRPL

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_times.append(capture_time)

        return ch1, ch2

    def run(self, params):
        """Run the lock-in data logging"""

        # Auto-calibrate if requested
        if params.get('auto_calibrate', False):
            self.calibrate_input_gain(
                cal_freq=params['ref_freq'],
                cal_amp=params['ref_amp'],
                cal_time=params.get('calibration_time', 2.0)
            )

        # Setup and start acquisition
        self.setup_lockin(params)
        print("Allowing lock-in to settle...")
        time.sleep(0.5)

        acquisition_start_time = time.time()
        capture_count = 0
        print(f"Started: {datetime.fromtimestamp(acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Data collection loop
        loop_start = time.time()
        while (time.time() - loop_start) < params['timeout']:
            self.capture_lockin()
            capture_count += 1

        acquisition_end_time = time.time()
        actual_duration = acquisition_end_time - acquisition_start_time

        print(f"✓ Captured {capture_count} scope buffers")

        # Concatenate all captured data
        all_X_raw = np.concatenate(self.lockin_X)
        all_Y_raw = np.concatenate(self.lockin_Y)

        # Apply gain correction (lock-in already handles the DC offset removal)
        all_X = all_X_raw * self.input_gain_factor
        all_Y = all_Y_raw * self.input_gain_factor

        # Apply averaging filter if requested
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            all_X = np.convolve(all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            all_Y = np.convolve(all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            print(f"Applied {averaging_window}-sample moving average")

        # Calculate magnitude and phase
        R = np.sqrt(all_X ** 2 + all_Y ** 2)
        Theta = np.arctan2(all_Y, all_X)

        # Create time vector and sample index
        n_samples = len(all_X)
        sample_index = np.arange(n_samples)
        t = sample_index / (n_samples / actual_duration)

        # Get raw signals for diagnostic plots
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw_uncorrected = np.array(self.scope._data_ch2_current)
        in1_raw = (in1_raw_uncorrected - self.input_dc_offset) * self.input_gain_factor
        t_raw = np.linspace(0, len(out1_raw) / self.nominal_sample_rate, len(out1_raw))

        # Restore scope to lock-in outputs
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        # FFT analysis of lock-in output (should be centered at DC)
        iq = all_X + 1j * all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, t[1] - t[0]))
        psd_lock = (np.abs(IQfft) ** 2) / (len(t) * np.sum(win ** 2))

        # Find the DC peak (should be the strongest)
        idx_dc = np.argmin(np.abs(freqs_lock))
        dc_power = psd_lock[idx_dc]

        # Find peaks EXCLUDING DC region (±50 Hz)
        dc_exclusion_mask = np.abs(freqs_lock) > 50
        psd_no_dc = psd_lock.copy()
        psd_no_dc[~dc_exclusion_mask] = 0

        idx_max_no_dc = np.argmax(psd_no_dc)
        freq_max_no_dc = freqs_lock[idx_max_no_dc]
        power_max_no_dc = psd_no_dc[idx_max_no_dc]

        # Find top peaks above threshold
        threshold = 0.01 * dc_power
        peak_indices = np.where(psd_lock > threshold)[0]
        peak_freqs = freqs_lock[peak_indices]
        peak_powers = psd_lock[peak_indices]

        # Sort by power
        sort_idx = np.argsort(peak_powers)[::-1]
        top_10_freqs = peak_freqs[sort_idx[:min(10, len(peak_freqs))]]
        top_10_powers = peak_powers[sort_idx[:min(10, len(peak_powers))]]

        print(f"\n⚠ FFT ANALYSIS:")
        print(f"DC peak (0 Hz): {dc_power:.2e}")
        print(f"Largest non-DC peak: {freq_max_no_dc:+.2f} Hz (power: {power_max_no_dc:.2e})")
        print(f"Ratio (non-DC/DC): {power_max_no_dc / dc_power * 100:.1f}%")

        print(f"\nTop 10 frequency peaks:")
        for i, (f, p) in enumerate(zip(top_10_freqs, top_10_powers)):
            marker = " ← DC" if abs(f) < 5 else ""
            marker += " ⚠ SPURIOUS!" if abs(f) > 50 and p > 0.01 * dc_power else ""
            print(f"  {i + 1}. {f:+8.2f} Hz (power: {p:.2e}){marker}")

        # Determine lock status
        if power_max_no_dc > 0.1 * dc_power:
            print(f"\n✗ WARNING: Large spurious peak at {freq_max_no_dc:+.2f} Hz!")
            print(f"   This peak is {power_max_no_dc / dc_power * 100:.1f}% of DC power")
            freq_offset = freq_max_no_dc
        else:
            print(f"\n✓ Spectrum dominated by DC (properly locked)")
            freq_offset = 0

        # Print results
        print("\n" + "=" * 60)
        print("LOCK-IN RESULTS (CORRECTED)")
        print("=" * 60)
        print(f"Mode: {self.input_mode}")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")
        print(f"DC offset correction: {self.input_dc_offset:.6f}V")
        print(f"Duration: {actual_duration:.3f}s")
        print(f"Samples collected: {n_samples}")
        print(f"Effective sample rate: {n_samples / actual_duration:.2f} Hz")

        # Calculate buffer statistics
        samples_per_buffer = n_samples / capture_count
        time_per_buffer = actual_duration / capture_count
        data_time_per_buffer = samples_per_buffer / self.nominal_sample_rate
        gap_per_buffer = time_per_buffer - data_time_per_buffer

        print(f"\nBuffer statistics:")
        print(f"  Buffers captured: {capture_count}")
        print(f"  Samples per buffer: {samples_per_buffer:.0f}")
        print(f"  Data time per buffer: {data_time_per_buffer:.3f}s")
        print(f"  Gap between buffers: {gap_per_buffer * 1000:.1f}ms")
        print(
            f"  Dead time: {gap_per_buffer * capture_count:.2f}s ({gap_per_buffer * capture_count / actual_duration * 100:.1f}%)")

        print(f"\nMean R: {np.mean(R):.6f} ± {np.std(R):.6f} V")
        print(f"Mean X: {np.mean(all_X):.6f} ± {np.std(all_X):.6f} V")
        print(f"Mean Y: {np.mean(all_Y):.6f} ± {np.std(all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.6f} ± {np.std(Theta):.6f} rad")

        # Expected DC output for lock-in: A/2 for amplitude A
        expected_R = params['ref_amp'] / 2.0
        R_error = abs(np.mean(R) - expected_R)
        R_ratio = np.mean(R) / expected_R

        print(f"\nExpected R (lock-in DC): {expected_R:.3f}V")
        print(f"Measured R: {np.mean(R):.3f}V")
        print(f"Ratio: {R_ratio:.3f} ({R_ratio * 100:.1f}%)")

        if R_error < 0.05:
            print(f"✓ Magnitude matches expected (loopback test)")
        elif abs(R_ratio - 1.0) < 0.1:
            print(f"✓ Magnitude close to expected (within 10%)")
        else:
            print(f"✗ Magnitude differs by {R_error:.3f}V ({(R_ratio - 1) * 100:.1f}% error)")

        print("=" * 60)

        # Create plots
        fig = plt.figure(figsize=(16, 10))

        # Reference signal
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_plot = min(int(n_periods * self.nominal_sample_rate / self.ref_freq), len(out1_raw))
        ax1.plot(t_raw[:n_plot] * 1000, out1_raw[:n_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference @ {self.ref_freq} Hz')
        ax1.grid(True)

        # Input signal (corrected)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw[:n_plot] * 1000, in1_raw[:n_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V, corrected)')
        ax2.set_title(f'Input - {self.input_mode}')
        ax2.grid(True)

        # FFT
        ax3 = plt.subplot(3, 3, 3)
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        if power_max_no_dc > 0.01 * dc_power:
            ax3.axvline(freq_max_no_dc, color='orange', linestyle='--', alpha=0.7,
                        label=f'Spurious ({freq_max_no_dc:.0f} Hz)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.set_title('FFT of Demodulated Signal')
        ax3.legend()
        ax3.grid(True)

        # X vs Time
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t, all_X, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(all_X), color='r', linestyle='--', label=f'Mean: {np.mean(all_X):.4f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-phase (X) - Offset Corrected')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(t[0], t[-1])
        margin_X = 5 * (np.max(all_X) - np.min(all_X))
        ax4.set_ylim(np.min(all_X) - margin_X, np.max(all_X) + margin_X)

        # Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(all_Y):.4f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y) - Offset Corrected')
        ax5.legend()
        ax5.grid(True)
        ax5.set_xlim(t[0], t[-1])
        margin_Y = 5 * (np.max(all_Y) - np.min(all_Y))
        ax5.set_ylim(np.min(all_Y) - margin_Y, np.max(all_Y) + margin_Y)

        # IQ plot
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(all_X, all_Y, 'g.', markersize=1, alpha=0.5)
        ax6.plot(np.mean(all_X), np.mean(all_Y), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # R vs Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t, R, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R), color='b', linestyle='--', label=f'Mean: {np.mean(R):.4f}V')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (V)')
        ax7.set_title('Magnitude (R)')
        ax7.legend()
        ax7.grid(True)
        ax7.set_xlim(t[0], t[-1])
        margin_R = 5 * (np.max(R) - np.min(R))
        ax7.set_ylim(np.min(R) - margin_R, np.max(R) + margin_R)

        # Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', label=f'Mean: {np.mean(Theta):.4f} rad')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (rad)')
        ax8.set_title('Phase (Theta)')
        ax8.legend()
        ax8.grid(True)
        ax8.set_xlim(t[0], t[-1])
        margin_Theta = 5 * (np.max(Theta) - np.min(Theta))
        ax8.set_ylim(np.min(Theta) - margin_Theta, np.max(Theta) + margin_Theta)

        # Theta vs R
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(Theta, R, 'g.', markersize=1, alpha=0.5)
        ax9.plot(np.mean(Theta), np.mean(R), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax9.set_xlabel('Theta (rad)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('Phase vs Magnitude')
        ax9.legend()
        ax9.grid(True)

        plt.tight_layout()

        # Save data and plots
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save plot
            img_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Saved plot: {img_path}")

            # Save CSV data
            data = np.column_stack((sample_index, t, R, Theta, all_X, all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.csv')

            with open(csv_path, 'w', newline='') as f:
                f.write(f"# Red Pitaya Lock-In Amplifier Data Logger\n")
                f.write(f"# Mode: {self.input_mode}\n")
                f.write(f"# Gain correction: {self.input_gain_factor:.6f}\n")
                f.write(f"# DC offset correction: {self.input_dc_offset:.6f} V\n")
                f.write(f"# Reference frequency: {self.ref_freq} Hz\n")
                f.write(f"# Reference amplitude: {params['ref_amp']} V\n")
                f.write(f"# Filter bandwidth: {params.get('filter_bandwidth', 10)} Hz\n")
                f.write(f"# Duration: {actual_duration:.3f} s\n")
                f.write(f"# Samples: {n_samples}\n")
                f.write(f"# Sample rate: {n_samples / actual_duration:.2f} Hz\n")
                f.write("Index,Time(s),R(V),Theta(rad),X(V),Y(V)\n")
                np.savetxt(f, data, delimiter=",", fmt='%.10f')

            print(f"✓ Saved data: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitayaLockInLogger(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR,
        manual_offset=MANUAL_DC_OFFSET
    )

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
        'auto_calibrate': AUTO_CALIBRATE,
        'calibration_time': CALIBRATION_TIME,
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN DATA LOGGER")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE}V")
    print(f"Filter bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement time: {MEASUREMENT_TIME}s")
    print(f"Input mode: {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"Manual gain: {MANUAL_GAIN_FACTOR}x")
        print(f"Manual DC offset: {MANUAL_DC_OFFSET}V")
    print("=" * 60)

    rp.run(run_params)
