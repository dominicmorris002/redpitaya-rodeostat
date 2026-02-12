"""
Red Pitaya Lock-In Amplifier - FAST ACQUISITION VERSIONS

Three acquisition modes to test:
1. SINGLE_SHOT - Your current method (baseline)
2. CONTINUOUS - Uses scope.continuous() for potentially faster acquisition
3. CURVE_BUFFER - Uses Red Pitaya's internal buffer (fastest, but less flexible)

For testing: connect OUT1 to IN1 with a cable.
Supports AUTO, LV, HV, or MANUAL gain modes.
Phase is in degrees!

Have a great day :)
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
REF_AMPLITUDE = 1.0  # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 12  # seconds

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'
AUTOLAB_GAIN = 1  # Based on Autolab "Current Scale" if Scale = 1mA : Set to 1e-3
MANUAL_GAIN_FACTOR = 1 * AUTOLAB_GAIN  # Only used if INPUT_MODE = 'MANUAL'
MANUAL_DC_OFFSET = 0  # Only used if INPUT_MODE = 'MANUAL'

FILTER_BANDWIDTH = 100  # Hz
AVERAGING_WINDOW = 1  # samples (moving average on logged data)
DECIMATION = 8  # Try: 1, 8, 64, 1024, 8192, 65536

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

AUTO_CALIBRATE = False  # Only used if INPUT_MODE = 'AUTO'
CALIBRATION_TIME = 2.0  # seconds

# ============================================================
# ACQUISITION MODE SELECTION
# Choose one: 'SINGLE_SHOT', 'CONTINUOUS', or 'CURVE_BUFFER'
# ============================================================
ACQUISITION_MODE = 'CONTINUOUS'
# ============================================================

START_TIME_FILE = "start_time.txt"

# Sync with other processes if needed
try:
    with open(START_TIME_FILE, "r") as f:
        START_TIME = datetime.fromisoformat(f.read().strip())
    while datetime.now() < START_TIME:
        time.sleep(0.001)
except FileNotFoundError:
    pass  # no sync file, just start now


class RedPitayaLockInLogger:
    """Data logger for PyRPL's built-in lock-in with multiple acquisition modes"""

    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO', manual_gain=1.0, manual_offset=0.0):
        self.rp = Pyrpl(config='lockin_config10', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope

        self.lockin_X = []
        self.lockin_Y = []
        self.capture_times = []

        # Setup input gain and DC offset
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

        # Point scope at lock-in outputs
        self.scope.input1 = 'iq2'  # X component
        self.scope.input2 = 'iq2_2'  # Y component
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f'Invalid decimation. Must be one of {self.allowed_decimations}')

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True

        # Nominal sample rate
        self.nominal_sample_rate = 125e6 / self.scope.decimation

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        """
        Figure out input scaling and DC offset by comparing OUT1 to IN1.
        Measures the actual physical signal on IN1.
        """
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT SCALING AND DC OFFSET...")
        print("=" * 60)

        # Step 1: Check DC offset with nothing connected
        print("Step 1: Measuring DC offset on IN1 (no signal)...")
        self.ref_sig.output_direct = 'off'
        self.lockin.output_direct = 'off'

        # Point scope at raw IN1
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'
        time.sleep(0.3)  # wait for things to settle

        offset_samples = []
        for _ in range(10):
            self.scope.single()
            offset_samples.append(np.mean(self.scope._data_ch1_current))

        self.input_dc_offset = np.mean(offset_samples)
        print(f"  Measured DC offset: {self.input_dc_offset:.6f}V")

        # Step 2: Send a signal and see what comes back
        print(f"\nStep 2: Measuring gain with {cal_amp}V @ {cal_freq} Hz...")

        # Now look at both OUT1 and IN1
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'

        # Generate test signal
        self.ref_sig.setup(
            frequency=cal_freq,
            amplitude=cal_amp,
            offset=0,
            waveform='sin',
            trigger_source='immediately'
        )
        self.ref_sig.output_direct = 'out1'

        time.sleep(0.5)  # let it settle

        cal_out1 = []
        cal_in1 = []
        start_time = time.time()

        while (time.time() - start_time) < cal_time:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # what we sent
            ch2 = np.array(self.scope._data_ch2_current)  # what we got back
            cal_out1.append(ch1)
            cal_in1.append(ch2)

        # Put all the data together
        all_out1 = np.concatenate(cal_out1)
        all_in1 = np.concatenate(cal_in1)

        # Remove DC offset
        all_in1_corrected = all_in1 - self.input_dc_offset

        # Find peak amplitudes
        out1_peak = (np.max(all_out1) - np.min(all_out1)) / 2
        in1_peak = (np.max(all_in1_corrected) - np.min(all_in1_corrected)) / 2

        # Also check RMS
        out1_rms = np.sqrt(np.mean(all_out1 ** 2))
        in1_rms = np.sqrt(np.mean(all_in1_corrected ** 2))

        # Calculate gain: what we sent / what we got
        self.input_gain_factor = out1_peak / in1_peak
        gain_rms = out1_rms / in1_rms

        # Try to guess what mode we're in
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

        # Put scope back to lock-in outputs
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        return self.input_gain_factor

    def setup_lockin(self, params):
        """Configure the lock-in"""
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        self.ref_sig.output_direct = 'off'

        # Setup PyRPL's iq2 module
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

        # Double-check what PyRPL actually did
        actual_freq = self.lockin.frequency
        actual_amp = self.lockin.amplitude

        print(f"Lock-in frequency: {self.ref_freq} Hz (actual: {actual_freq:.2f} Hz)")
        print(f"Lock-in bandwidth: {filter_bw} Hz")
        print(f"Reference amplitude: {ref_amp}V on {params['output_ref']} (actual: {actual_amp:.3f}V)")

        if abs(actual_freq - self.ref_freq) > 0.1:
            print(f"⚠ WARNING: Requested {self.ref_freq} Hz but PyRPL set {actual_freq:.2f} Hz!")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")
        print(f"DC offset correction: {self.input_dc_offset:.6f}V")

    # ============================================================
    # OPTION 1: SINGLE SHOT MODE (Your current method)
    # ============================================================
    def capture_lockin_single_shot(self):
        """Grab one buffer of lock-in data using single() mode"""
        capture_time = time.time()
        self.scope.single()

        ch1 = np.array(self.scope._data_ch1_current)  # X from lock-in
        ch2 = np.array(self.scope._data_ch2_current)  # Y from lock-in

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_times.append(capture_time)

        return ch1, ch2

    # ============================================================
    # OPTION 2: CONTINUOUS MODE (Potentially faster)
    # ============================================================
    def capture_lockin_continuous(self):
        """Grab lock-in data using continuous mode"""
        capture_time = time.time()

        # Read current data (scope is already running in continuous mode)
        ch1 = np.array(self.scope._data_ch1_current)  # X from lock-in
        ch2 = np.array(self.scope._data_ch2_current)  # Y from lock-in

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_times.append(capture_time)

        return ch1, ch2

    # ============================================================
    # OPTION 3: CURVE BUFFER MODE (Fastest - collects internally)
    # ============================================================
    def acquire_with_curve_buffer(self, params):
        """Use Red Pitaya's curve buffer for fastest acquisition"""
        print("Using CURVE BUFFER mode - acquiring all data internally...")

        curve = self.rp_modules.curve

        # Calculate how many points we can get
        duration = params['timeout']
        sample_rate = self.nominal_sample_rate

        print(f"Curve buffer acquiring for {duration}s at {sample_rate / 1e3:.1f} kHz...")

        # Setup curve to record both lock-in outputs
        curve.setup(
            duration=duration,
            trigger_source='immediately',
            input1='iq2',  # X component
            input2='iq2_2',  # Y component
        )

        # Wait for acquisition to complete
        time.sleep(duration + 0.5)

        # Get the data
        all_X_raw = np.array(curve.data_ch1)
        all_Y_raw = np.array(curve.data_ch2)

        print(f"✓ Curve buffer collected {len(all_X_raw)} samples")

        return all_X_raw, all_Y_raw

    def run(self, params):
        """Main acquisition loop"""

        # Run calibration if AUTO mode
        if params.get('auto_calibrate', False):
            self.calibrate_input_gain(
                cal_freq=params['ref_freq'],
                cal_amp=params['ref_amp'],
                cal_time=params.get('calibration_time', 2.0)
            )

        # Get everything running
        self.setup_lockin(params)
        print("Allowing lock-in to settle...")
        time.sleep(0.5)

        acquisition_start_time = time.time()
        capture_count = 0

        acq_mode = params.get('acquisition_mode', 'SINGLE_SHOT')
        print(f"\n{'=' * 60}")
        print(f"ACQUISITION MODE: {acq_mode}")
        print(f"{'=' * 60}")

        # ============================================================
        # Choose acquisition method based on mode
        # ============================================================

        if acq_mode == 'CURVE_BUFFER':
            # OPTION 3: Use curve buffer
            print(f"Started: {datetime.fromtimestamp(acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
            all_X_raw, all_Y_raw = self.acquire_with_curve_buffer(params)

        elif acq_mode == 'CONTINUOUS':
            # OPTION 2: Continuous mode
            print(f"Started: {datetime.fromtimestamp(acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
            self.scope.continuous()  # Start continuous acquisition
            time.sleep(0.1)  # Let it start

            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_lockin_continuous()
                capture_count += 1
                time.sleep(0.001)  # Small delay to prevent overwhelming the system

            all_X_raw = np.concatenate(self.lockin_X)
            all_Y_raw = np.concatenate(self.lockin_Y)

        else:  # SINGLE_SHOT (default)
            # OPTION 1: Single shot mode (your original)
            print(f"Started: {datetime.fromtimestamp(acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_lockin_single_shot()
                capture_count += 1

            all_X_raw = np.concatenate(self.lockin_X)
            all_Y_raw = np.concatenate(self.lockin_Y)

        acquisition_end_time = time.time()
        actual_duration = acquisition_end_time - acquisition_start_time

        if acq_mode != 'CURVE_BUFFER':
            print(f"✓ Captured {capture_count} scope buffers")

        # Apply gain correction
        all_X = all_X_raw * self.input_gain_factor
        all_Y = all_Y_raw * self.input_gain_factor

        # Smooth it out if requested
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            all_X = np.convolve(all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            all_Y = np.convolve(all_Y, np.ones(averaging_window) / averaging_window, mode='valid')
            print(f"Applied {averaging_window}-sample moving average")

        # Convert to polar coordinates (magnitude and phase in degrees!)
        R = np.sqrt(all_X ** 2 + all_Y ** 2)
        Theta = np.degrees(np.arctan2(all_Y, all_X))

        # Make time axis
        n_samples = len(all_X)
        sample_index = np.arange(n_samples)
        t = sample_index / (n_samples / actual_duration)

        # Grab raw signals for plots
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw_uncorrected = np.array(self.scope._data_ch2_current)
        in1_raw = (in1_raw_uncorrected - self.input_dc_offset) * self.input_gain_factor
        t_raw = np.linspace(0, len(out1_raw) / self.nominal_sample_rate, len(out1_raw))

        # Put scope back
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        # FFT to check if we're actually locked
        iq = all_X + 1j * all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, t[1] - t[0]))
        psd_lock = (np.abs(IQfft) ** 2) / (len(t) * np.sum(win ** 2))

        # Find DC peak (should be biggest if locked properly)
        idx_dc = np.argmin(np.abs(freqs_lock))
        dc_power = psd_lock[idx_dc]

        # Look for spurious peaks away from DC
        dc_exclusion_mask = np.abs(freqs_lock) > 50
        psd_no_dc = psd_lock.copy()
        psd_no_dc[~dc_exclusion_mask] = 0

        idx_max_no_dc = np.argmax(psd_no_dc)
        freq_max_no_dc = freqs_lock[idx_max_no_dc]
        power_max_no_dc = psd_no_dc[idx_max_no_dc]

        # Find all significant peaks
        threshold = 0.01 * dc_power
        peak_indices = np.where(psd_lock > threshold)[0]
        peak_freqs = freqs_lock[peak_indices]
        peak_powers = psd_lock[peak_indices]

        # Sort by power and grab top 10
        sort_idx = np.argsort(peak_powers)[::-1]
        top_10_freqs = peak_freqs[sort_idx[:min(10, len(peak_freqs))]]
        top_10_powers = peak_powers[sort_idx[:min(10, len(peak_powers))]]

        print(f"\nFFT ANALYSIS:")
        print(f"DC peak (0 Hz): {dc_power:.2e}")
        print(f"Largest non-DC peak: {freq_max_no_dc:+.2f} Hz (power: {power_max_no_dc:.2e})")
        print(f"Ratio (non-DC/DC): {power_max_no_dc / dc_power * 100:.1f}%")

        print(f"\nTop 10 frequency peaks:")
        for i, (f, p) in enumerate(zip(top_10_freqs, top_10_powers)):
            marker = " ← DC" if abs(f) < 5 else ""
            marker += " ⚠ SPURIOUS!" if abs(f) > 50 and p > 0.01 * dc_power else ""
            print(f"  {i + 1}. {f:+8.2f} Hz (power: {p:.2e}){marker}")

        # Check lock quality
        if power_max_no_dc > 0.1 * dc_power:
            print(f"\n✗ WARNING: Large spurious peak at {freq_max_no_dc:+.2f} Hz!")
            print(f"   This peak is {power_max_no_dc / dc_power * 100:.1f}% of DC power")
            freq_offset = freq_max_no_dc
        else:
            print(f"\n✓ Spectrum dominated by DC (properly locked)")
            freq_offset = 0

        # Show results
        print("\n" + "=" * 60)
        print("LOCK-IN RESULTS")
        print("=" * 60)
        print(f"Acquisition mode: {acq_mode}")
        print(f"Input mode: {self.input_mode}")
        print(f"Gain correction: {self.input_gain_factor:.4f}x")
        print(f"DC offset correction: {self.input_dc_offset:.6f}V")
        print(f"Duration: {actual_duration:.3f}s")
        print(f"Samples collected: {n_samples}")
        print(f"Effective sample rate: {n_samples / actual_duration:.2f} Hz")

        # Real statistics that matter
        lock_in_sample_rate = 125e6  # Lock-in processes at full ADC rate
        output_bandwidth = params.get('filter_bandwidth', 10)
        time_constant = 1 / (2 * np.pi * output_bandwidth)

        print(f"\nLock-in Processing:")
        print(f"  ADC sample rate: {lock_in_sample_rate / 1e6:.0f} MHz (continuous)")
        print(f"  Filter bandwidth: {output_bandwidth} Hz")
        print(f"  Time constant: {time_constant * 1000:.2f} ms")
        if acq_mode != 'CURVE_BUFFER':
            print(f"  Scope buffers captured: {capture_count}")
        print(f"  Data points logged: {n_samples}")
        print(f"  Logging rate: {n_samples / actual_duration:.2f} samples/sec")

        print(f"\nMean R: {np.mean(R):.6f} ± {np.std(R):.6f} V")
        print(f"Mean X: {np.mean(all_X):.6f} ± {np.std(all_X):.6f} V")
        print(f"Mean Y: {np.mean(all_Y):.6f} ± {np.std(all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.3f} ± {np.std(Theta):.3f}°")
        print(f"\nMeasured R: {np.mean(R):.3f}V")
        print("=" * 60)

        # Downsample for plotting if we have too many points
        max_plot_points = 50000
        downsample_factor = 1
        if len(t) > max_plot_points:
            downsample_factor = len(t) // max_plot_points
            print(f"\n⚠ Downsampling from {len(t)} to ~{len(t) // downsample_factor} points for plotting...")
            t_plot = t[::downsample_factor]
            X_plot = all_X[::downsample_factor]
            Y_plot = all_Y[::downsample_factor]
            R_plot = R[::downsample_factor]
            Theta_plot = Theta[::downsample_factor]
        else:
            t_plot = t
            X_plot = all_X
            Y_plot = all_Y
            R_plot = R
            Theta_plot = Theta

        # Make plots
        fig = plt.figure(figsize=(16, 10))

        t_raw_corrected = np.linspace(0, len(out1_raw) / self.nominal_sample_rate, len(out1_raw))

        # Reference signal
        ax1 = plt.subplot(3, 3, 1)
        n_periods = 5
        n_plot = min(int(n_periods * self.nominal_sample_rate / self.ref_freq), len(out1_raw))
        ax1.plot(t_raw_corrected[:n_plot] * 1000, out1_raw[:n_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference @ {self.ref_freq} Hz')
        ax1.grid(True)

        # Input signal
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t_raw_corrected[:n_plot] * 1000, in1_raw[:n_plot], 'r-', linewidth=1)
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
        ax4.plot(t_plot, X_plot, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(all_X), color='r', linestyle='--', label=f'Mean: {np.mean(all_X):.9f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title(f'In-phase (X) - {acq_mode}')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(t_plot[0], t_plot[-1])
        if len(X_plot) > 0:
            margin_X = 5 * (np.max(X_plot) - np.min(X_plot))
            ax4.set_ylim(np.min(X_plot) - margin_X, np.max(X_plot) + margin_X)

        # Y vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t_plot, Y_plot, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(all_Y):.9f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title(f'Quadrature (Y) - {acq_mode}')
        ax5.legend()
        ax5.grid(True)
        ax5.set_xlim(t_plot[0], t_plot[-1])
        if len(Y_plot) > 0:
            margin_Y = 5 * (np.max(Y_plot) - np.min(Y_plot))
            ax5.set_ylim(np.min(Y_plot) - margin_Y, np.max(Y_plot) + margin_Y)

        # IQ plot
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(X_plot, Y_plot, 'g.', markersize=1, alpha=0.5)
        ax6.plot(np.mean(all_X), np.mean(all_Y), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # R vs Time
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(t_plot, R_plot * 1e6, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R) * 1e6, color='b', linestyle='--', label=f'Mean: {np.mean(R) * 1e6:.6f}μA')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (μA)')
        ax7.set_title(f'Magnitude (R) - {acq_mode}')
        ax7.legend()
        ax7.grid(True)
        ax7.set_xlim(t_plot[0], t_plot[-1])
        if len(R_plot) > 0:
            margin_R = 5 * (np.max(R_plot * 1e6) - np.min(R_plot * 1e6))
            ax7.set_ylim(np.min(R_plot * 1e6) - margin_R, np.max(R_plot * 1e6) + margin_R)

        # Theta vs Time
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(t_plot, Theta_plot, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--', label=f'Mean: {np.mean(Theta):.3f}°')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (°)')
        ax8.set_title(f'Phase (Theta) - {acq_mode}')
        ax8.legend()
        ax8.grid(True)
        ax8.set_xlim(t_plot[0], t_plot[-1])
        if len(Theta_plot) > 0:
            margin_Theta = 5 * (np.max(Theta_plot) - np.min(Theta_plot))
            ax8.set_ylim(np.min(Theta_plot) - margin_Theta, np.max(Theta_plot) + margin_Theta)

        # Theta vs R
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(Theta_plot, R_plot, 'g.', markersize=1, alpha=0.5)
        ax9.plot(np.mean(Theta), np.mean(R), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax9.set_xlabel('Theta (°)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('Phase vs Magnitude')
        ax9.legend()
        ax9.grid(True)

        plt.tight_layout()

        # Save everything
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save plot
            img_path = os.path.join(self.output_dir, f'lockin_{acq_mode}_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Saved plot: {img_path}")

            # Save CSV (use full resolution, not downsampled)
            data = np.column_stack((sample_index, t, R, Theta, all_X, all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_{acq_mode}_{timestamp_str}.csv')

            with open(csv_path, 'w', newline='') as f:
                f.write(f"# Red Pitaya Lock-In Amplifier Data Logger\n")
                f.write(f"# Acquisition mode: {acq_mode}\n")
                f.write(f"# Input mode: {self.input_mode}\n")
                f.write(f"# Gain correction: {self.input_gain_factor:.6f}\n")
                f.write(f"# DC offset correction: {self.input_dc_offset:.6f} V\n")
                f.write(f"# Reference frequency: {self.ref_freq} Hz\n")
                f.write(f"# Reference amplitude: {params['ref_amp']} V\n")
                f.write(f"# Filter bandwidth: {params.get('filter_bandwidth', 10)} Hz\n")
                f.write(f"# Time constant: {time_constant * 1000:.2f} ms\n")
                f.write(f"# Duration: {actual_duration:.3f} s\n")
                f.write(f"# Samples: {n_samples}\n")
                f.write(f"# Sample rate: {n_samples / actual_duration:.2f} Hz\n")
                f.write("Index,Time(s),R(V),Theta(°),X(V),Y(V)\n")
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
        'acquisition_mode': ACQUISITION_MODE,  # NEW!
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN DATA LOGGER - MULTI-MODE VERSION")
    print("=" * 60)
    print(f"Acquisition mode: {ACQUISITION_MODE}")
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE}V")
    print(f"Filter bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Decimation: {DECIMATION}")
    print(f"Measurement time: {MEASUREMENT_TIME}s")
    print(f"Input mode: {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"Manual gain: {MANUAL_GAIN_FACTOR}x")
        print(f"Manual DC offset: {MANUAL_DC_OFFSET}V")
    print("=" * 60)

    rp.run(run_params)
