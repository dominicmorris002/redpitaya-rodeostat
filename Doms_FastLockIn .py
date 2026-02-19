"""
Red Pitaya Lock-In Amplifier

Uses PyRPL's built-in lock-in (iq2 and iq2_2 module) to log data of X(In-Phase) and Y(Out-Phase).
For testing: connect OUT1 to IN1 with a cable.
Supports AUTO, LV, HV, or MANUAL gain modes.
Phase is in degrees!

I recommend CONTINUOUS Mode for good Data.

Have a great day :)
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from pyrpl import Pyrpl
from scipy.signal import find_peaks

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
REF_FREQUENCY = 500   # Hz
REF_AMPLITUDE = 0.2   # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0      # degrees
MEASUREMENT_TIME = 12 # seconds

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE = 'MANUAL'
AUTOLAB_GAIN = 1  # e.g. Based on AUTOLAB Device Scale of TIA, 1mA scale -> 1e-3
MANUAL_GAIN_FACTOR = 1 * AUTOLAB_GAIN
MANUAL_DC_OFFSET = 0

# ── FILTER BANDWIDTH ──────────────────────────────────────────────────────────
#   10 Hz  -> tau = 16 ms This is Best
#   100 Hz -> tau = 1.6 ms
#   500 Hz -> tau = 0.3 ms
FILTER_BANDWIDTH = 10  # Hz
# ─────────────────────────────────────────────────────────────────────────────

# ── DECIMATION ────────────────────────────────────────────────────────────────
# Controls scope sample rate: sample_rate = 125e6 / DECIMATION
DECIMATION = 8
# ─────────────────────────────────────────────────────────────────────────────

AVERAGING_WINDOW = 1    # samples (moving average on logged data)

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

AUTO_CALIBRATE = True
CALIBRATION_TIME = 2.0  # seconds

# ACQUISITION MODE: 'SINGLE_SHOT' or 'CONTINUOUS'
ACQUISITION_MODE = 'CONTINUOUS'

# ============================================================
# PLOT DOWNSAMPLING
# ============================================================

# CSV always saves full resolution data no matter what.
PLOT_DOWNSAMPLE_ENABLED = True
PLOT_MAX_POINTS = 50_000



START_TIME_FILE = "start_time.txt"
try:
    with open(START_TIME_FILE, "r") as f:
        START_TIME = datetime.fromisoformat(f.read().strip())
    while datetime.now() < START_TIME:
        time.sleep(0.001)
except FileNotFoundError:
    pass


# ============================================================
# Data Helpers
# ============================================================

def downsample(arrays, n_samples, max_points, enabled=True):

    if not enabled or n_samples <= max_points:
        return arrays, 1, n_samples
    step = max(1, n_samples // max_points)
    ds = [arr[::step] for arr in arrays]
    return ds, step, len(ds[0])


def detect_peaks_and_annotate(ax, t, signal, unit_str, color_peak='red',
                               color_trough='blue', min_prominence_frac=0.1,
                               scale=1.0, label_prefix=''):
    sig = signal * scale
    sig_range = np.max(sig) - np.min(sig)
    if sig_range == 0:
        return False

    prominence = min_prominence_frac * sig_range
    peak_idx, _   = find_peaks( sig, prominence=prominence, distance=max(1, len(sig) // 30))
    trough_idx, _ = find_peaks(-sig, prominence=prominence, distance=max(1, len(sig) // 30))

    if len(peak_idx) < 2 and len(trough_idx) < 1:
        return False

    peak_times, peak_vals = [], []
    for i, idx in enumerate(peak_idx):
        pt, pv = t[idx], sig[idx]
        peak_times.append(pt); peak_vals.append(pv)
        ax.plot(pt, pv, 'v', color=color_peak, markersize=8, zorder=5)
        ax.annotate(f'P{i+1}\n{pv:.0f}{unit_str}',
                    xy=(pt, pv), xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5, color=color_peak,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                              ec=color_peak, lw=0.8))

    trough_vals = []
    for i, idx in enumerate(trough_idx):
        tt, tv = t[idx], sig[idx]
        trough_vals.append(tv)
        ax.plot(tt, tv, '^', color=color_trough, markersize=8, zorder=5)
        ax.annotate(f'T{i+1}\n{tv:.0f}{unit_str}',
                    xy=(tt, tv), xytext=(0, -14), textcoords='offset points',
                    ha='center', va='top', fontsize=6.5, color=color_trough,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                              ec=color_trough, lw=0.8))

    if len(peak_idx) >= 2:
        arrow_y = np.max(sig) + 0.08 * sig_range
        pp_times = []
        for i in range(len(peak_idx) - 1):
            t0, t1 = peak_times[i], peak_times[i + 1]
            dt = t1 - t0
            pp_times.append(dt)
            ax.annotate('', xy=(t1, arrow_y), xytext=(t0, arrow_y),
                        arrowprops=dict(arrowstyle='<->', color='purple',
                                        lw=1.3, shrinkA=0, shrinkB=0))
            ax.text((t0 + t1) / 2, arrow_y + 0.01 * sig_range,
                    f'dt={dt*1000:.1f}ms', ha='center', va='bottom',
                    fontsize=6, color='purple',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8,
                              ec='purple', lw=0.6))

        mean_dt = np.mean(pp_times)
        std_dt  = np.std(pp_times)
        mean_f  = 1.0 / mean_dt if mean_dt > 0 else 0.0
        lines = [f'{label_prefix}Peak-to-peak:',
                 f'  dt = {mean_dt*1000:.1f} +/- {std_dt*1000:.1f} ms',
                 f'  f  = {mean_f:.3f} Hz']
        if trough_vals:
            lines.append(f'  dR = {np.mean(peak_vals) - np.mean(trough_vals):.0f} {unit_str}')
        ax.plot([], [], color='purple', lw=0, marker='', label='\n'.join(lines))

    return True


# ============================================================
# MAIN CLASS
# ============================================================

class RedPitayaLockInLogger:
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data', input_mode='AUTO',
                 manual_gain=1.0, manual_offset=0.0):
        self.rp = Pyrpl(config='lockin_config10', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin  = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope   = self.rp_modules.scope

        self.lockin_X = []
        self.lockin_Y = []
        self.capture_times = []

        self.input_gain_factor  = manual_gain
        self.input_dc_offset    = manual_offset
        self.input_mode_setting = input_mode.upper()
        self.input_mode         = "Unknown"

        if self.input_mode_setting == 'MANUAL':
            self.input_mode = f"MANUAL ({manual_gain}x gain, {manual_offset}V offset)"
        elif self.input_mode_setting == 'LV':
            self.input_gain_factor = 1.0
            self.input_dc_offset   = 0.0
            self.input_mode = "LV (+-1V)"
        elif self.input_mode_setting == 'HV':
            self.input_gain_factor = 20.0
            self.input_dc_offset   = 0.0
            self.input_mode = "HV (+-20V, 20:1 divider)"
        elif self.input_mode_setting == 'AUTO':
            self.input_mode = "AUTO (will calibrate)"
        print(f"Input mode: {self.input_mode}")

        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = DECIMATION

        if self.scope.decimation not in self.allowed_decimations:
            raise ValueError(f'Invalid decimation. Must be one of {self.allowed_decimations}')

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True
        self.nominal_sample_rate = 125e6 / self.scope.decimation

    def calibrate_input_gain(self, cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        if not force and self.input_mode_setting != 'AUTO':
            print(f"Skipping calibration - using {self.input_mode}")
            return self.input_gain_factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT SCALING AND DC OFFSET...")
        print("=" * 60)

        print("Step 1: Measuring DC offset on IN1 (no signal)...")
        self.ref_sig.output_direct = 'off'
        self.lockin.output_direct  = 'off'
        self.scope.input1 = 'in1'
        self.scope.input2 = 'in1'
        time.sleep(0.3)

        offset_samples = []
        for _ in range(10):
            self.scope.single()
            offset_samples.append(np.mean(self.scope._data_ch1_current))
        self.input_dc_offset = np.mean(offset_samples)
        print(f"  DC offset: {self.input_dc_offset:.6f}V")

        print(f"\nStep 2: Gain measurement {cal_amp}V @ {cal_freq} Hz...")
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        self.ref_sig.setup(frequency=cal_freq, amplitude=cal_amp, offset=0,
                           waveform='sin', trigger_source='immediately')
        self.ref_sig.output_direct = 'out1'
        time.sleep(0.5)

        cal_out1, cal_in1 = [], []
        t0 = time.time()
        while (time.time() - t0) < cal_time:
            self.scope.single()
            cal_out1.append(np.array(self.scope._data_ch1_current))
            cal_in1.append(np.array(self.scope._data_ch2_current))

        all_out1 = np.concatenate(cal_out1)
        all_in1  = np.concatenate(cal_in1) - self.input_dc_offset
        out1_peak = (np.max(all_out1) - np.min(all_out1)) / 2
        in1_peak  = (np.max(all_in1)  - np.min(all_in1))  / 2
        out1_rms  = np.sqrt(np.mean(all_out1 ** 2))
        in1_rms   = np.sqrt(np.mean(all_in1  ** 2))
        self.input_gain_factor = out1_peak / in1_peak

        if   self.input_gain_factor < 1.05: self.input_mode = "LV (+-1V)"
        elif self.input_gain_factor < 2.0:  self.input_mode = f"LV with loading ({self.input_gain_factor:.2f}x)"
        elif self.input_gain_factor < 15:   self.input_mode = f"Custom ({self.input_gain_factor:.2f}x)"
        else:                               self.input_mode = f"HV ({self.input_gain_factor:.1f}:1)"

        print(f"  Gain (peak): {self.input_gain_factor:.4f}x  RMS: {out1_rms/in1_rms:.4f}x")
        print(f"  Mode: {self.input_mode}")
        print("=" * 60 + "\n")

        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        return self.input_gain_factor

    def setup_lockin(self, params):
        self.ref_freq   = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp   = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)

        self.ref_sig.output_direct = 'off'
        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,
            phase=params.get('phase', 0),
            acbandwidth=0,
            amplitude=ref_amp,
            input='in1',
            output_direct=params['output_ref'],
            output_signal='quadrature',
            quadrature_factor=1.0)

        actual_freq = self.lockin.frequency
        actual_amp  = self.lockin.amplitude
        tc_ms = 1e3 / (2 * np.pi * filter_bw)
        print(f"Lock-in frequency: {self.ref_freq} Hz (actual: {actual_freq:.2f} Hz)")
        print(f"Lock-in bandwidth: {filter_bw} Hz  (time constant: {tc_ms:.2f} ms)")
        print(f"Reference: {ref_amp}V on {params['output_ref']} (actual: {actual_amp:.3f}V)")
        if abs(actual_freq - self.ref_freq) > 0.1:
            print(f"WARNING: Requested {self.ref_freq} Hz but got {actual_freq:.2f} Hz!")
        if filter_bw < 50:
            print(f"NOTE: Filter BW is {filter_bw} Hz (tau = {tc_ms:.1f} ms) -- peaks will be "
                  f"rounded. Consider raising FILTER_BANDWIDTH to 100+ Hz for sharper peaks.")

    def capture_lockin(self):
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)
        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_times.append(time.time())
        return ch1, ch2

    def capture_lockin_continuous(self):
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)
        self.lockin_X.append(ch1)
        self.lockin_Y.append(ch2)
        self.capture_times.append(time.time())
        return ch1, ch2

    def run(self, params):

        if params.get('auto_calibrate', False):
            self.calibrate_input_gain(
                cal_freq=params['ref_freq'],
                cal_amp=params['ref_amp'],
                cal_time=params.get('calibration_time', 2.0))

        self.setup_lockin(params)

        # ── WAIT FOR USER TO TRIGGER ──────────────────────────────────────────
        print()
        print("=" * 60)
        print("AC signal is running. Press ENTER to start measurement.")
        print("=" * 60)
        input()
        # ─────────────────────────────────────────────────────────────────────

        print("Allowing lock-in to settle...")
        time.sleep(0.5)

        acq_start = time.time()
        capture_count = 0
        print(f"Started: {datetime.fromtimestamp(acq_start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        acq_mode = params.get('acquisition_mode', 'SINGLE_SHOT')
        if acq_mode == 'CONTINUOUS':
            self.scope.continuous()
            time.sleep(0.1)
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_lockin_continuous()
                capture_count += 1
                time.sleep(0.001)
        else:
            loop_start = time.time()
            while (time.time() - loop_start) < params['timeout']:
                self.capture_lockin()
                capture_count += 1

        actual_duration = time.time() - acq_start
        print(f"Captured {capture_count} buffers")

        # ── Combine and correct ───────────────────────────────────────────────
        all_X = np.concatenate(self.lockin_X) * self.input_gain_factor
        all_Y = np.concatenate(self.lockin_Y) * self.input_gain_factor

        avg_win = params.get('averaging_window', 1)
        if avg_win > 1:
            k = np.ones(avg_win) / avg_win
            all_X = np.convolve(all_X, k, mode='valid')
            all_Y = np.convolve(all_Y, k, mode='valid')
            print(f"Applied {avg_win}-sample moving average")

        R     = np.sqrt(all_X ** 2 + all_Y ** 2)
        Theta = np.degrees(np.arctan2(all_Y, all_X))

        n_samples    = len(all_X)
        sample_index = np.arange(n_samples)
        t = sample_index / (n_samples / actual_duration)

        filter_bw = params.get('filter_bandwidth', 10)
        tc_ms = 1e3 / (2 * np.pi * filter_bw)

        # ── Downsample first  ───────────────

        ds_on  = params.get('plot_downsample_enabled', True)
        ds_max = params.get('plot_max_points', 50_000)

        ds_arrays, ds_step, n_plot = downsample(
            [t, R, Theta, all_X, all_Y],
            n_samples, ds_max, enabled=ds_on)
        t_p, R_p, Theta_p, X_p, Y_p = ds_arrays

        if ds_step > 1:
            print(f"\nDownsampled: {n_samples:,} -> {n_plot:,} pts "
                  f"(step={ds_step}) for plotting & sharpening. CSV = full res.")
        else:
            print(f"\nNo downsampling needed ({n_samples:,} pts)")

        ds_sample_rate = n_plot / actual_duration
        print(f"Downsampled sample rate: {ds_sample_rate:.1f} Hz")

        # ── Grab raw scope signals for Row 1 ──────────────────────────────────
        print("\nCapturing scope snapshot...")
        self.scope.input1 = 'out1'
        self.scope.input2 = 'in1'
        time.sleep(0.05)
        self.scope.single()
        out1_raw = np.array(self.scope._data_ch1_current)
        in1_raw  = (np.array(self.scope._data_ch2_current) - self.input_dc_offset) * self.input_gain_factor
        t_raw    = np.linspace(0, len(out1_raw) / self.nominal_sample_rate, len(out1_raw))
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'

        # ── FFT on downsampled data ───────────────────────────────────────────
        print("Computing FFT...")
        n_pts = len(X_p)
        win   = np.hanning(n_pts)
        IQfft = np.fft.fftshift(np.fft.fft((X_p + 1j * Y_p) * win))
        freqs = np.fft.fftshift(np.fft.fftfreq(n_pts, t_p[1] - t_p[0]))
        psd   = (np.abs(IQfft) ** 2) / (n_pts * np.sum(win ** 2))

        dc_power = psd[np.argmin(np.abs(freqs))]
        psd_no_dc = psd.copy()
        psd_no_dc[np.abs(freqs) <= 50] = 0
        idx_spur   = np.argmax(psd_no_dc)
        freq_spur  = freqs[idx_spur]
        power_spur = psd_no_dc[idx_spur]

        sig_idx    = np.where(psd > 0.01 * dc_power)[0]
        sort_order = np.argsort(psd[sig_idx])[::-1]
        top_freqs  = freqs[sig_idx][sort_order[:10]]
        top_powers = psd[sig_idx][sort_order[:10]]

        print(f"\nFFT:  DC={dc_power:.2e}   Spurious={freq_spur:+.1f} Hz "
              f"({power_spur/dc_power*100:.1f}% of DC)")
        for i, (f, p) in enumerate(zip(top_freqs, top_powers)):
            flag = " <- DC" if abs(f) < 5 else (" SPURIOUS!" if abs(f) > 50 and p > 0.01*dc_power else "")
            print(f"  {i+1:2d}. {f:+8.2f} Hz  {p:.2e}{flag}")
        if power_spur > 0.1 * dc_power:
            print(f"WARNING: Spurious at {freq_spur:+.2f} Hz is {power_spur/dc_power*100:.1f}% of DC!")
        else:
            print("Spectrum dominated by DC (properly locked)")

        # ── Summary ───────────────────────────────────────────────────────────
        spb = n_samples / capture_count
        gap = (actual_duration / capture_count) - (spb / self.nominal_sample_rate)

        print("\n" + "=" * 60)
        print("LOCK-IN RESULTS")
        print("=" * 60)
        print(f"Mode:         {self.input_mode}")
        print(f"Duration:     {actual_duration:.3f}s   Samples: {n_samples:,}")
        print(f"Logging rate: {n_samples/actual_duration:.2f} Hz")
        print(f"Filter BW:    {filter_bw} Hz  (time constant: {tc_ms:.2f} ms)")
        print(f"Decimation:   {DECIMATION}  (scope sample rate: {self.nominal_sample_rate/1e3:.1f} kHz)")
        print(f"Buffers:      {capture_count}   Samples/buf: {spb:.0f}   Gap/buf: {gap*1000:.1f} ms")
        print(f"\nMean R:     {np.mean(R):.6f} +/- {np.std(R):.6f} V")
        print(f"Mean X:     {np.mean(all_X):.6f} +/- {np.std(all_X):.6f} V")
        print(f"Mean Y:     {np.mean(all_Y):.6f} +/- {np.std(all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.3f} +/- {np.std(Theta):.3f} deg")
        print("=" * 60)

        # ── Zoom window ───────────────────────────────────────────────────────
        ZOOM_FRAC = 0.2
        n_z = max(2, int(n_plot * ZOOM_FRAC))
        zs  = n_plot // 2 - n_z // 2
        ze  = zs + n_z

        t_z   = t_p[zs:ze];   R_z   = R_p[zs:ze];   Th_z  = Theta_p[zs:ze]
        X_z   = X_p[zs:ze];   Y_z   = Y_p[zs:ze]

        ds_note   = f' [1:{ds_step} for plot]' if ds_step > 1 else ''
        z_raw_lbl = f'Zoomed {t_z[0]:.1f}-{t_z[-1]:.1f}s ({ZOOM_FRAC*100:.0f}%) RAW{ds_note}'

        # ── Plot ──────────────────────────────────────────────────────────────
        # Layout: 4 rows x 3 cols
        #   Row 1: Reference signal | Input signal | FFT
        #   Row 2: X (full)        | Y (full)      | IQ plot (full)
        #   Row 3: R (full)        | Theta (full)  | Phase vs Magnitude
        #   Row 4: R zoomed        | Theta zoomed  | IQ zoomed

        print("\nRendering plots...")
        fig = plt.figure(figsize=(18, 18))

        def _margin(arr):
            return 5 * max(np.max(arr) - np.min(arr), 1e-12)

        # Row 1
        ax1 = plt.subplot(4, 3, 1)
        n_raw = min(int(5 * self.nominal_sample_rate / self.ref_freq), len(out1_raw))
        ax1.plot(t_raw[:n_raw] * 1000, out1_raw[:n_raw], 'b-', lw=1)
        ax1.set(xlabel='Time (ms)', ylabel='OUT1 (V)', title=f'Reference @ {self.ref_freq} Hz')
        ax1.grid(True)

        ax2 = plt.subplot(4, 3, 2)
        ax2.plot(t_raw[:n_raw] * 1000, in1_raw[:n_raw], 'r-', lw=1)
        ax2.set(xlabel='Time (ms)', ylabel='IN1 (V, corrected)', title=f'Input -- {self.input_mode}')
        ax2.grid(True)

        ax3 = plt.subplot(4, 3, 3)
        ax3.semilogy(freqs, psd, label='Lock-in PSD')
        if power_spur > 0.01 * dc_power:
            ax3.axvline(freq_spur, color='orange', ls='--', alpha=0.7,
                        label=f'Spurious ({freq_spur:.0f} Hz)')
        ax3.set(xlabel='Frequency (Hz)', ylabel='Power', title='FFT of Demodulated Signal')
        ax3.legend(); ax3.grid(True)

        # Row 2
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(t_p, X_p, 'b-', lw=0.5)
        ax4.axhline(np.mean(all_X), color='r', ls='--', label=f'Mean: {np.mean(all_X):.6f}V')
        ax4.set(xlabel='Time (s)', ylabel='X (V)', title=f'In-phase (X){ds_note}')
        ax4.legend(); ax4.grid(True); ax4.set_xlim(t_p[0], t_p[-1])
        m = _margin(X_p); ax4.set_ylim(np.min(X_p)-m, np.max(X_p)+m)

        ax5 = plt.subplot(4, 3, 5)
        ax5.plot(t_p, Y_p, 'r-', lw=0.5)
        ax5.axhline(np.mean(all_Y), color='b', ls='--', label=f'Mean: {np.mean(all_Y):.6f}V')
        ax5.set(xlabel='Time (s)', ylabel='Y (V)', title=f'Quadrature (Y){ds_note}')
        ax5.legend(); ax5.grid(True); ax5.set_xlim(t_p[0], t_p[-1])
        m = _margin(Y_p); ax5.set_ylim(np.min(Y_p)-m, np.max(Y_p)+m)

        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(X_p, Y_p, 'g.', ms=1, alpha=0.5)
        ax6.plot(np.mean(all_X), np.mean(all_Y), 'r+', ms=15, mew=2, label='Mean')
        ax6.set(xlabel='X (V)', ylabel='Y (V)', title=f'IQ Plot{ds_note}')
        ax6.legend(); ax6.grid(True); ax6.axis('equal')

        # Row 3
        ax7 = plt.subplot(4, 3, 7)
        R_p_uA = R_p * 1e6
        ax7.plot(t_p, R_p_uA, 'm-', lw=0.5)
        ax7.axhline(np.mean(R)*1e6, color='b', ls='--', label=f'Mean: {np.mean(R)*1e6:.4f} uA')
        ax7.set(xlabel='Time (s)', ylabel='R (uA)', title=f'Magnitude (R) -- Full{ds_note}')
        ax7.grid(True); ax7.set_xlim(t_p[0], t_p[-1])
        m = _margin(R_p_uA); ax7.set_ylim(np.min(R_p_uA)-m, np.max(R_p_uA)+m)
        ax7.axvspan(t_z[0], t_z[-1], color='yellow', alpha=0.25, label=f'Zoom ({ZOOM_FRAC*100:.0f}%)')
        ax7.legend(fontsize=7)

        ax8 = plt.subplot(4, 3, 8)
        ax8.plot(t_p, Theta_p, 'c-', lw=0.5)
        ax8.axhline(np.mean(Theta), color='r', ls='--', label=f'Mean: {np.mean(Theta):.3f} deg')
        ax8.set(xlabel='Time (s)', ylabel='Theta (deg)', title=f'Phase (Theta) -- Full{ds_note}')
        ax8.grid(True); ax8.set_xlim(t_p[0], t_p[-1])
        m = _margin(Theta_p); ax8.set_ylim(np.min(Theta_p)-m, np.max(Theta_p)+m)
        ax8.axvspan(t_z[0], t_z[-1], color='yellow', alpha=0.25, label=f'Zoom ({ZOOM_FRAC*100:.0f}%)')
        ax8.legend(fontsize=7)

        ax9 = plt.subplot(4, 3, 9)
        ax9.plot(Theta_p, R_p, 'g.', ms=1, alpha=0.5)
        ax9.plot(np.mean(Theta), np.mean(R), 'r+', ms=15, mew=2, label='Mean')
        ax9.set(xlabel='Theta (deg)', ylabel='R (V)', title='Phase vs Magnitude')
        ax9.legend(); ax9.grid(True)

        # Row 4: Zoomed
        def _zoom_plot(ax, t_z, sig, scale, ylabel, title, color, unit_str, lp):
            s = sig * scale
            mn, sd = np.mean(s), np.std(s)
            ax.plot(t_z, s, color=color, lw=1.0)
            ax.axhline(mn, color='b', ls='--', label=f'Mean: {mn:.2f}\nStd: {sd:.2f}')
            ax.fill_between(t_z, mn-sd, mn+sd, alpha=0.15, color='blue', label='+/-1 sigma')
            ax.set(xlabel='Time (s)', ylabel=ylabel, title=title)
            ax.grid(True); ax.set_xlim(t_z[0], t_z[-1])
            detect_peaks_and_annotate(ax, t_z, sig, unit_str=unit_str,
                                      color_peak='darkred', color_trough='navy',
                                      min_prominence_frac=0.15, scale=scale, label_prefix=lp)
            ax.legend(fontsize=7, loc='upper right')

        _zoom_plot(plt.subplot(4,3,10), t_z, R_z,  1e6, 'R (uA)',
                   f'R -- Zoomed\n{z_raw_lbl}',      'm', ' uA',  'R: ')
        _zoom_plot(plt.subplot(4,3,11), t_z, Th_z, 1.0, 'Theta (deg)',
                   f'Theta -- Zoomed\n{z_raw_lbl}',  'c', ' deg', 'Th: ')

        ax12 = plt.subplot(4, 3, 12)
        ax12.plot(X_z, Y_z, 'g.', ms=2, alpha=0.6)
        ax12.plot(np.mean(X_z), np.mean(Y_z), 'r+', ms=15, mew=2, label='Mean')
        ax12.set(xlabel='X (V)', ylabel='Y (V)', title=f'IQ -- Zoomed\n{z_raw_lbl}')
        ax12.legend(fontsize=7); ax12.grid(True); ax12.axis('equal')

        plt.tight_layout()

        # ── Save ──────────────────────────────────────────────────────────────
        if params['save_file']:
            os.makedirs(self.output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            img_path = os.path.join(self.output_dir, f'lockin_results_{ts}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\nPlot:  {img_path}")

            # CSV: always full resolution, ASCII-only to avoid Windows cp1252 errors
            csv_path = os.path.join(self.output_dir, f'lockin_results_{ts}.csv')
            data = np.column_stack((sample_index, t, R, Theta, all_X, all_Y))
            with open(csv_path, 'w', newline='', encoding='ascii') as f:
                f.write("# Red Pitaya Lock-In Amplifier\n")
                f.write(f"# Mode: {self.input_mode}\n")
                f.write(f"# Gain: {self.input_gain_factor:.6f}  DC offset: {self.input_dc_offset:.6f} V\n")
                f.write(f"# Ref: {self.ref_freq} Hz @ {params['ref_amp']} V\n")
                f.write(f"# Filter BW: {filter_bw} Hz  (time constant: {tc_ms:.2f} ms)\n")
                f.write(f"# Decimation: {DECIMATION}  (scope rate: {self.nominal_sample_rate/1e3:.1f} kHz)\n")
                f.write(f"# Duration: {actual_duration:.3f} s  Samples: {n_samples}\n")
                f.write(f"# Sample rate: {n_samples/actual_duration:.2f} Hz\n")
                f.write(f"# Acquisition: {acq_mode}\n")
                f.write(f"# Plot downsample step: {ds_step}x  (CSV is full resolution)\n")
                f.write("Index,Time(s),R(V),Theta(deg),X(V),Y(V)\n")
                np.savetxt(f, data, delimiter=",", fmt='%.10f')
            print(f"Data:  {csv_path}")
        else:
            plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    rp = RedPitayaLockInLogger(
        output_dir=OUTPUT_DIRECTORY,
        input_mode=INPUT_MODE,
        manual_gain=MANUAL_GAIN_FACTOR,
        manual_offset=MANUAL_DC_OFFSET
    )

    run_params = {
        'ref_freq':                REF_FREQUENCY,
        'ref_amp':                 REF_AMPLITUDE,
        'output_ref':              OUTPUT_CHANNEL,
        'phase':                   PHASE_OFFSET,
        'timeout':                 MEASUREMENT_TIME,
        'filter_bandwidth':        FILTER_BANDWIDTH,
        'averaging_window':        AVERAGING_WINDOW,
        'output_dir':              OUTPUT_DIRECTORY,
        'save_file':               SAVE_DATA,
        'auto_calibrate':          AUTO_CALIBRATE,
        'calibration_time':        CALIBRATION_TIME,
        'acquisition_mode':        ACQUISITION_MODE,
        'plot_downsample_enabled': PLOT_DOWNSAMPLE_ENABLED,
        'plot_max_points':         PLOT_MAX_POINTS,
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN DATA LOGGER")
    print("=" * 60)
    print(f"Reference:    {REF_FREQUENCY} Hz @ {REF_AMPLITUDE}V")
    print(f"Filter BW:    {FILTER_BANDWIDTH} Hz")
    print(f"Decimation:   {DECIMATION}  (scope rate: {125e6/DECIMATION/1e3:.1f} kHz, "
          f"buf covers {16384/(125e6/DECIMATION)*1000:.1f} ms)")
    print(f"Meas. time:   {MEASUREMENT_TIME}s")
    print(f"Input mode:   {INPUT_MODE}")
    if INPUT_MODE.upper() == 'MANUAL':
        print(f"  Gain:   {MANUAL_GAIN_FACTOR}x")
        print(f"  Offset: {MANUAL_DC_OFFSET}V")
    print(f"Acq. mode:    {ACQUISITION_MODE}")
    print(f"Plot DS:      {'ON -- max ' + str(PLOT_MAX_POINTS) + ' pts/signal' if PLOT_DOWNSAMPLE_ENABLED else 'OFF'}")
    print("=" * 60)

    rp.run(run_params)
