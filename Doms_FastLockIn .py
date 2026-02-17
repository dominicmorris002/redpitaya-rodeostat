"""
Red Pitaya Lock-In Amplifier

Uses PyRPL's built-in lock-in (iq2 and iq2_2 module) to log data of X(In-Phase) and Y(Out-Phase).
For testing: connect OUT1 to IN1 with a cable.
Supports AUTO, LV, HV, or MANUAL gain modes.
Phase is in degrees!

Have a great day :)
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
from pyrpl import Pyrpl
from scipy.signal import find_peaks

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
REF_FREQUENCY = 500  # Hz
REF_AMPLITUDE = 0.2  # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 12  # seconds

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'
AUTOLAB_GAIN = 1  # Based on Autolab "Current Scale" if Scale = 1mA : Set to 1e-3
MANUAL_GAIN_FACTOR = 1 * AUTOLAB_GAIN  # Only used if INPUT_MODE = 'MANUAL'
MANUAL_DC_OFFSET = 0   # Only used if INPUT_MODE = 'MANUAL'

FILTER_BANDWIDTH = 100  # Hz
AVERAGING_WINDOW = 1  # samples (moving average on logged data)
DECIMATION = 8

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

AUTO_CALIBRATE = True  # Only used if INPUT_MODE = 'AUTO'
CALIBRATION_TIME = 2.0  # seconds

# ACQUISITION MODE: 'SINGLE_SHOT' or 'CONTINUOUS'
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


def detect_peaks_and_annotate(ax, t, signal, unit_str, color_peak='red',
                               color_trough='blue', min_prominence_frac=0.1,
                               scale=1.0, label_prefix=''):
    """
    Detect peaks and troughs in signal, annotate the zoomed axes with:
      - numbered markers at each peak/trough
      - peak-to-peak time arrows between consecutive peaks
      - a legend entry summarising mean peak-to-peak time and amplitude

    Returns True if peaks were found, False otherwise.
    scale: multiply signal values by this for display (e.g. 1e6 for μA)
    """
    sig = signal * scale
    sig_range = np.max(sig) - np.min(sig)
    if sig_range == 0:
        return False

    prominence = min_prominence_frac * sig_range

    # Find peaks (local maxima)
    peak_idx, peak_props = find_peaks(sig, prominence=prominence,
                                       distance=max(1, len(sig) // 30))
    # Find troughs (local minima)
    trough_idx, trough_props = find_peaks(-sig, prominence=prominence,
                                           distance=max(1, len(sig) // 30))

    has_peaks = len(peak_idx) >= 2
    has_troughs = len(trough_idx) >= 1

    if not has_peaks and not has_troughs:
        return False

    # ── Annotate peaks ────────────────────────────────────────────────────────
    peak_times = []
    peak_vals = []
    for i, idx in enumerate(peak_idx):
        pt = t[idx]
        pv = sig[idx]
        peak_times.append(pt)
        peak_vals.append(pv)
        ax.plot(pt, pv, 'v', color=color_peak, markersize=8, zorder=5)
        ax.annotate(f'P{i+1}\n{pv:.0f}{unit_str}',
                    xy=(pt, pv), xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5,
                    color=color_peak, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec=color_peak, lw=0.8))

    # ── Annotate troughs ─────────────────────────────────────────────────────
    trough_times = []
    trough_vals = []
    for i, idx in enumerate(trough_idx):
        tt = t[idx]
        tv = sig[idx]
        trough_times.append(tt)
        trough_vals.append(tv)
        ax.plot(tt, tv, '^', color=color_trough, markersize=8, zorder=5)
        ax.annotate(f'T{i+1}\n{tv:.0f}{unit_str}',
                    xy=(tt, tv), xytext=(0, -14), textcoords='offset points',
                    ha='center', va='top', fontsize=6.5,
                    color=color_trough, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec=color_trough, lw=0.8))

    # ── Peak-to-peak time arrows between consecutive peaks ────────────────────
    pp_times = []
    pp_amps = []
    if len(peak_idx) >= 2:
        arrow_y = np.max(sig) + 0.08 * sig_range  # draw arrows just above max
        for i in range(len(peak_idx) - 1):
            t0 = peak_times[i]
            t1 = peak_times[i + 1]
            dt = t1 - t0
            pp_times.append(dt)
            mid_t = (t0 + t1) / 2
            # Double-headed arrow
            ax.annotate('', xy=(t1, arrow_y), xytext=(t0, arrow_y),
                        arrowprops=dict(arrowstyle='<->', color='purple',
                                        lw=1.3, shrinkA=0, shrinkB=0))
            ax.text(mid_t, arrow_y + 0.01 * sig_range,
                    f'Δt={dt*1000:.1f}ms',
                    ha='center', va='bottom', fontsize=6, color='purple',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8, ec='purple', lw=0.6))

        # Peak-to-peak amplitude (max - min across all detected peaks/troughs)
        if has_troughs:
            all_peak_vals = np.array(peak_vals)
            all_trough_vals = np.array(trough_vals)
            pp_amp = np.mean(all_peak_vals) - np.mean(all_trough_vals)
            pp_amps.append(pp_amp)

        mean_pp_time = np.mean(pp_times)
        std_pp_time = np.std(pp_times)
        mean_pp_freq = 1.0 / mean_pp_time if mean_pp_time > 0 else 0.0

        # Summary legend entry
        summary_lines = [
            f'{label_prefix}Peak-to-peak:',
            f'  Δt = {mean_pp_time*1000:.1f} ± {std_pp_time*1000:.1f} ms',
            f'  f  = {mean_pp_freq:.3f} Hz',
        ]
        if pp_amps:
            summary_lines.append(f'  ΔR = {pp_amps[0]:.0f} {unit_str}')
        summary_str = '\n'.join(summary_lines)

        # Add as an invisible line so it appears in legend
        ax.plot([], [], color='purple', lw=0, marker='', label=summary_str)

    return True


class RedPitayaLockInLogger:
    """Data logger for PyRPL's built-in lock-in"""

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

    def capture_lockin(self):
        """Grab one buffer of lock-in data (single shot - blocks until buffer is full)"""
        capture_time = time.time()
        self.scope.single()

        ch1 = np.array(self.scope._data_ch1_current)  # X from lock-in
        ch2 = np.array(self.scope._data_ch2_current)  # Y from lock-in

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_times.append(capture_time)

        return ch1, ch2

    def capture_lockin_continuous(self):
        """Grab lock-in data in continuous mode (no blocking single() call)"""
        capture_time = time.time()

        ch1 = np.array(self.scope._data_ch1_current)  # X from lock-in
        ch2 = np.array(self.scope._data_ch2_current)  # Y from lock-in

        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_times.append(capture_time)

        return ch1, ch2

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
        print(f"Started: {datetime.fromtimestamp(acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        acq_mode = params.get('acquisition_mode', 'SINGLE_SHOT')

        # Collect data!
        if acq_mode == 'CONTINUOUS':
            self.scope.continuous()
            time.sleep(0.1)
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_lockin_continuous()
                capture_count += 1
                time.sleep(0.001)
        else:  # SINGLE_SHOT (default)
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_lockin()
                capture_count += 1

        acquisition_end_time = time.time()
        actual_duration = acquisition_end_time - acquisition_start_time

        print(f"✓ Captured {capture_count} scope buffers")

        # Combine all the data
        all_X_raw = np.concatenate(self.lockin_X)
        all_Y_raw = np.concatenate(self.lockin_Y)

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
        win = np.hanning(n_pts)  # New way (NumPy 2.0+)
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
        print(f"Mode: {self.input_mode}")
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
        print(f"  Scope buffers captured: {capture_count}")
        print(f"  Data points logged: {n_samples}")
        print(f"  Logging rate: {n_samples / actual_duration:.2f} samples/sec")

        samples_per_buffer = n_samples / capture_count
        time_per_buffer = actual_duration / capture_count
        data_time_per_buffer = samples_per_buffer / self.nominal_sample_rate
        gap_per_buffer = time_per_buffer - data_time_per_buffer

        print(f"\nBuffer statistics:")
        print(f"  Buffers captured: {capture_count}")
        print(f"  Samples per buffer: {samples_per_buffer:.0f}")
        print(f"  Data time per buffer: {data_time_per_buffer:.3f}s")
        print(f"  Gap between buffers: {gap_per_buffer * 1000:.1f}ms")
        print(f"  Dead time: {gap_per_buffer * capture_count:.2f}s "
              f"({gap_per_buffer * capture_count / actual_duration * 100:.1f}%)")

        print(f"\nMean R: {np.mean(R):.6f} ± {np.std(R):.6f} V")
        print(f"Mean X: {np.mean(all_X):.6f} ± {np.std(all_X):.6f} V")
        print(f"Mean Y: {np.mean(all_Y):.6f} ± {np.std(all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.3f} ± {np.std(Theta):.3f}°")
        print(f"\nMeasured R: {np.mean(R):.3f}V")
        print("=" * 60)

        # ── Zoomed window helper ──────────────────────────────────────────────
        ZOOM_FRACTION = 0.2
        n_zoom = max(2, int(n_samples * ZOOM_FRACTION))
        zoom_center = n_samples // 2
        zoom_start = zoom_center - n_zoom // 2
        zoom_end = zoom_center + n_zoom // 2
        t_zoom = t[zoom_start:zoom_end]
        R_zoom = R[zoom_start:zoom_end]
        Theta_zoom = Theta[zoom_start:zoom_end]
        X_zoom = all_X[zoom_start:zoom_end]
        Y_zoom = all_Y[zoom_start:zoom_end]

        # ── Build figure: 4 rows × 3 cols ─────────────────────────────────────
        fig = plt.figure(figsize=(18, 16))

        t_raw_corrected = np.linspace(0, len(out1_raw) / self.nominal_sample_rate, len(out1_raw))

        # ── Row 1: raw signals + FFT ──────────────────────────────────────────
        ax1 = plt.subplot(4, 3, 1)
        n_periods = 5
        n_plot = min(int(n_periods * self.nominal_sample_rate / self.ref_freq), len(out1_raw))
        ax1.plot(t_raw_corrected[:n_plot] * 1000, out1_raw[:n_plot], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('OUT1 (V)')
        ax1.set_title(f'Reference @ {self.ref_freq} Hz')
        ax1.grid(True)

        ax2 = plt.subplot(4, 3, 2)
        ax2.plot(t_raw_corrected[:n_plot] * 1000, in1_raw[:n_plot], 'r-', linewidth=1)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('IN1 (V, corrected)')
        ax2.set_title(f'Input - {self.input_mode}')
        ax2.grid(True)

        ax3 = plt.subplot(4, 3, 3)
        ax3.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        if power_max_no_dc > 0.01 * dc_power:
            ax3.axvline(freq_max_no_dc, color='orange', linestyle='--', alpha=0.7,
                        label=f'Spurious ({freq_max_no_dc:.0f} Hz)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.set_title('FFT of Demodulated Signal')
        ax3.legend()
        ax3.grid(True)

        # ── Row 2: X, Y, IQ (full) ───────────────────────────────────────────
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(t, all_X, 'b-', linewidth=0.5)
        ax4.axhline(np.mean(all_X), color='r', linestyle='--', label=f'Mean: {np.mean(all_X):.9f}V')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('X (V)')
        ax4.set_title('In-phase (X)')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(t[0], t[-1])
        margin_X = 5 * (np.max(all_X) - np.min(all_X))
        ax4.set_ylim(np.min(all_X) - margin_X, np.max(all_X) + margin_X)

        ax5 = plt.subplot(4, 3, 5)
        ax5.plot(t, all_Y, 'r-', linewidth=0.5)
        ax5.axhline(np.mean(all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(all_Y):.9f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Y (V)')
        ax5.set_title('Quadrature (Y)')
        ax5.legend()
        ax5.grid(True)
        ax5.set_xlim(t[0], t[-1])
        margin_Y = 5 * (np.max(all_Y) - np.min(all_Y))
        ax5.set_ylim(np.min(all_Y) - margin_Y, np.max(all_Y) + margin_Y)

        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(all_X, all_Y, 'g.', markersize=1, alpha=0.5)
        ax6.plot(np.mean(all_X), np.mean(all_Y), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # ── Row 3: R and Theta (full) ─────────────────────────────────────────
        ax7 = plt.subplot(4, 3, 7)
        ax7.plot(t, R * 1e6, 'm-', linewidth=0.5)
        ax7.axhline(np.mean(R) * 1e6, color='b', linestyle='--',
                    label=f'Mean: {np.mean(R) * 1e6:.4f} μA')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('R (μA)')
        ax7.set_title('Magnitude (R) — Full')
        ax7.legend()
        ax7.grid(True)
        ax7.set_xlim(t[0], t[-1])
        margin_R = 5 * (np.max(R * 1e6) - np.min(R * 1e6))
        ax7.set_ylim(np.min(R * 1e6) - margin_R, np.max(R * 1e6) + margin_R)
        ax7.axvspan(t_zoom[0], t_zoom[-1], color='yellow', alpha=0.25,
                    label=f'Zoom ({ZOOM_FRACTION*100:.0f}%)')
        ax7.legend(fontsize=7)

        ax8 = plt.subplot(4, 3, 8)
        ax8.plot(t, Theta, 'c-', linewidth=0.5)
        ax8.axhline(np.mean(Theta), color='r', linestyle='--',
                    label=f'Mean: {np.mean(Theta):.3f}°')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Theta (°)')
        ax8.set_title('Phase (Theta) — Full')
        ax8.legend()
        ax8.grid(True)
        ax8.set_xlim(t[0], t[-1])
        margin_Theta = 5 * (np.max(Theta) - np.min(Theta))
        ax8.set_ylim(np.min(Theta) - margin_Theta, np.max(Theta) + margin_Theta)
        ax8.axvspan(t_zoom[0], t_zoom[-1], color='yellow', alpha=0.25,
                    label=f'Zoom ({ZOOM_FRACTION*100:.0f}%)')
        ax8.legend(fontsize=7)

        ax9 = plt.subplot(4, 3, 9)
        ax9.plot(Theta, R, 'g.', markersize=1, alpha=0.5)
        ax9.plot(np.mean(Theta), np.mean(R), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax9.set_xlabel('Theta (°)')
        ax9.set_ylabel('R (V)')
        ax9.set_title('Phase vs Magnitude')
        ax9.legend()
        ax9.grid(True)

        # ── Row 4: Zoomed R, Theta, IQ — with peak detection ─────────────────
        zoom_label = (f'Zoomed: {t_zoom[0]:.1f}–{t_zoom[-1]:.1f}s '
                      f'({ZOOM_FRACTION*100:.0f}% of data)')

        # ── Zoomed R ──────────────────────────────────────────────────────────
        ax10 = plt.subplot(4, 3, 10)
        ax10.plot(t_zoom, R_zoom * 1e6, 'm-', linewidth=1.0)
        R_zoom_mean = np.mean(R_zoom) * 1e6
        R_zoom_std = np.std(R_zoom) * 1e6
        ax10.axhline(R_zoom_mean, color='b', linestyle='--',
                     label=f'Mean: {R_zoom_mean:.2f} μA\nStd: {R_zoom_std:.2f} μA')
        ax10.fill_between(t_zoom,
                          R_zoom_mean - R_zoom_std,
                          R_zoom_mean + R_zoom_std,
                          alpha=0.15, color='blue', label='±1σ')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('R (μA)')
        ax10.set_title(f'Magnitude (R) — Zoomed\n{zoom_label}')
        ax10.grid(True)
        ax10.set_xlim(t_zoom[0], t_zoom[-1])

        # Peak detection on R
        peaks_found_R = detect_peaks_and_annotate(
            ax10, t_zoom, R_zoom,
            unit_str=' μA',
            color_peak='darkred', color_trough='navy',
            min_prominence_frac=0.15,
            scale=1e6,
            label_prefix='R: '
        )
        ax10.legend(fontsize=7, loc='upper right')

        # ── Zoomed Theta ──────────────────────────────────────────────────────
        ax11 = plt.subplot(4, 3, 11)
        ax11.plot(t_zoom, Theta_zoom, 'c-', linewidth=1.0)
        Theta_zoom_mean = np.mean(Theta_zoom)
        Theta_zoom_std = np.std(Theta_zoom)
        ax11.axhline(Theta_zoom_mean, color='r', linestyle='--',
                     label=f'Mean: {Theta_zoom_mean:.4f}°\nStd: {Theta_zoom_std:.4f}°')
        ax11.fill_between(t_zoom,
                          Theta_zoom_mean - Theta_zoom_std,
                          Theta_zoom_mean + Theta_zoom_std,
                          alpha=0.15, color='red', label='±1σ')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Theta (°)')
        ax11.set_title(f'Phase (Theta) — Zoomed\n{zoom_label}')
        ax11.grid(True)
        ax11.set_xlim(t_zoom[0], t_zoom[-1])

        # Peak detection on Theta
        peaks_found_Theta = detect_peaks_and_annotate(
            ax11, t_zoom, Theta_zoom,
            unit_str='°',
            color_peak='darkgreen', color_trough='darkorange',
            min_prominence_frac=0.15,
            scale=1.0,
            label_prefix='θ: '
        )
        ax11.legend(fontsize=7, loc='upper right')

        # ── Zoomed IQ scatter ─────────────────────────────────────────────────
        ax12 = plt.subplot(4, 3, 12)
        ax12.plot(X_zoom, Y_zoom, 'g.', markersize=2, alpha=0.6)
        ax12.plot(np.mean(X_zoom), np.mean(Y_zoom), 'r+', markersize=15,
                  markeredgewidth=2, label='Mean')
        ax12.set_xlabel('X (V)')
        ax12.set_ylabel('Y (V)')
        ax12.set_title(f'IQ Plot — Zoomed\n{zoom_label}')
        ax12.legend(fontsize=7)
        ax12.grid(True)
        ax12.axis('equal')

        plt.tight_layout()

        # ── Save everything ───────────────────────────────────────────────────
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            img_path = os.path.join(self.output_dir, f'lockin_results_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Saved plot: {img_path}")

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
                f.write(f"# Acquisition mode: {acq_mode}\n")
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
        'acquisition_mode': ACQUISITION_MODE,
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
    print(f"Acquisition mode: {ACQUISITION_MODE}")
    print("=" * 60)

    rp.run(run_params)
