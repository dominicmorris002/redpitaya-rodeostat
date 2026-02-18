import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d
from scipy import signal as scipy_signal
from scipy.signal import find_peaks

# ============================================================
# FILE PATHS
# ============================================================
sr_folder = r"C:\Users\lab\Downloads\Dominic (1)\LIC_EC 10uM ssDNA run 1"
sr_files = {
    "Magnitude": "signal.csv",
    "Phase":     "phase.csv",
    "Time":      "time.csv",
}

rp_combined_file = r"C:\Users\lab\PycharmProjects\dominic\redpitaya-rodeostat\test_data\lockin_results_20260217_172404.csv"

MAX_TIME      = 10.0      # seconds to compare
RP_MAX_POINTS = 50_000    # downsample RP to this before any processing

# ============================================================
# HELPERS
# ============================================================
def load_sr_data(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, 'r') as f:
        lines = f.readlines()
    all_values = []
    for line in lines:
        line = line.strip()
        if line:
            values = [float(x.strip()) for x in line.split(',') if x.strip()]
            all_values.extend(values)
    return np.array(all_values)

def align_signals(reference, sig, max_shift_fraction=0.1):
    max_shift = int(len(sig) * max_shift_fraction)
    ref_norm = (reference - np.mean(reference)) / (np.std(reference) + 1e-10)
    sig_norm = (sig      - np.mean(sig))      / (np.std(sig)      + 1e-10)
    correlation = scipy_signal.correlate(ref_norm, sig_norm, mode='full')
    lags        = scipy_signal.correlation_lags(len(ref_norm), len(sig_norm), mode='full')
    valid_idx   = np.where(np.abs(lags) <= max_shift)[0]
    if len(valid_idx) == 0:
        return sig, 0
    max_corr_idx = valid_idx[np.argmax(correlation[valid_idx])]
    shift = lags[max_corr_idx]
    if shift > 0:
        aligned = sig[shift:]
    elif shift < 0:
        aligned = sig[:shift]
    else:
        aligned = sig
    return aligned, shift

# ============================================================
# LOAD SIGNAL RECOVERY
# ============================================================
sr_data = {}
print("=== Loading Signal Recovery Data ===")
for label, fname in sr_files.items():
    try:
        sr_data[label] = load_sr_data(sr_folder, fname)
        print(f"Loaded {label} ({fname}): {len(sr_data[label])} samples")
    except Exception as e:
        print(f"Error loading {label} ({fname}): {e}")

# ============================================================
# LOAD RED PITAYA LOCK-IN
# ============================================================
print("\n=== Loading Red Pitaya Lock-in Data ===")
with open(rp_combined_file, 'r', encoding='ascii', errors='replace') as f:
    lines = f.readlines()

# Parse gain
gain = 1.0
for line in lines:
    if line.lower().startswith("# gain"):
        try:
            gain = float(line.split(":")[1].strip().split()[0])
        except Exception:
            pass
        break
print(f"Gain correction: {gain}")

# Header row
header_idx = next(i for i, l in enumerate(lines) if l.startswith("Index"))
print(f"Header: {lines[header_idx].strip()}")
header_cols = [c.strip() for c in lines[header_idx].strip().split(",")]
col_time  = header_cols.index("Time(s)")
col_R     = next(i for i, c in enumerate(header_cols) if c.startswith("R("))
col_Theta = next(i for i, c in enumerate(header_cols) if c.startswith("Theta"))

rp_raw = np.genfromtxt(rp_combined_file, delimiter=',', skip_header=header_idx + 1)

time_rp_full  = rp_raw[:, col_time]
R_rp_full     = rp_raw[:, col_R]    * gain
Theta_rp_full = rp_raw[:, col_Theta]

print(f"Red Pitaya samples loaded: {len(time_rp_full):,}")

# Truncate to MAX_TIME
mask          = time_rp_full <= MAX_TIME
time_rp_full  = time_rp_full[mask]
R_rp_full     = R_rp_full[mask]
Theta_rp_full = Theta_rp_full[mask]
print(f"After truncation at {MAX_TIME}s: {len(time_rp_full):,} samples")

# Downsample
n_rp = len(time_rp_full)
if n_rp > RP_MAX_POINTS:
    step     = n_rp // RP_MAX_POINTS
    time_rp  = time_rp_full[::step]
    R_rp     = R_rp_full[::step]
    Theta_rp = Theta_rp_full[::step]
    print(f"Downsampled RP: {n_rp:,} -> {len(time_rp):,} pts (step={step})")
else:
    time_rp  = time_rp_full
    R_rp     = R_rp_full
    Theta_rp = Theta_rp_full
    print(f"No downsampling needed ({n_rp:,} pts)")

# ============================================================
# RESAMPLE SIGNAL RECOVERY TO RP LENGTH
# ============================================================
rp_len = len(time_rp)
sr_magnitude_interp = np.interp(
    np.linspace(0, len(sr_data["Magnitude"]) - 1, rp_len),
    np.arange(len(sr_data["Magnitude"])),
    sr_data["Magnitude"]
)
sr_phase_interp = np.interp(
    np.linspace(0, len(sr_data["Phase"]) - 1, rp_len),
    np.arange(len(sr_data["Phase"])),
    sr_data["Phase"]
)

print(f"\nSignal Recovery data (resampled to {rp_len} pts):")
print(f"  Magnitude: {np.min(sr_magnitude_interp):.6f} to {np.max(sr_magnitude_interp):.6f} V")
print(f"  Phase:     {np.min(sr_phase_interp):.1f} to {np.max(sr_phase_interp):.1f} deg")

# ============================================================
# DOUBLE PEAK ANALYSIS
# ============================================================
peak_indices, _ = find_peaks(R_rp, distance=10, prominence=0.01)
peak_times = time_rp[peak_indices]
peak_mags  = R_rp[peak_indices]
print(f"\nDetected {len(peak_indices)} peaks in Red Pitaya magnitude")

# Group peaks into double peak cycles
double_peak_pairs = []
double_peak_mags  = []
double_peak_time_diffs = []
max_double_peak_gap = 0.1  # seconds

i = 0
while i < len(peak_indices) - 1:
    t1 = peak_times[i]
    t2 = peak_times[i + 1]
    dt = t2 - t1
    if dt <= max_double_peak_gap:
        double_peak_pairs.append((peak_indices[i], peak_indices[i + 1]))
        double_peak_time_diffs.append(dt)
        double_peak_mags.append((peak_mags[i] + peak_mags[i + 1]) / 2)
        i += 2
    else:
        i += 1

dp_start_times = [peak_times[pair[0]] for pair in double_peak_pairs]
double_peak_cycle_times = np.diff(dp_start_times)

print(f"Found {len(double_peak_pairs)} double peak cycles")
print("\nDouble peak time differences (within pairs):", double_peak_time_diffs)
print("Double peak cycle times (between cycles):", double_peak_cycle_times)
print("Average magnitude per double peak:", double_peak_mags)

# Plot detected peaks
plt.figure(figsize=(14, 6))
plt.plot(time_rp, R_rp, label='Red Pitaya Magnitude', alpha=0.7)
plt.plot(peak_times, peak_mags, 'x', label='Detected Peaks', color='orange')
for idx, ((p1, p2), dt, mag) in enumerate(zip(double_peak_pairs, double_peak_time_diffs, double_peak_mags)):
    t1, t2 = peak_times[p1], peak_times[p2]
    m1, m2 = peak_mags[p1], peak_mags[p2]
    plt.plot([t1, t2], [m1, m2], 'g-', alpha=0.5)
    plt.text((t1+t2)/2, (m1+m2)/2, f"{dt:.3f}s\nMag:{mag:.3f}", fontsize=8, color='green', ha='center', va='bottom')
plt.xlabel('Time (s)')
plt.ylabel('AC Magnitude (V)')
plt.title('Red Pitaya Magnitude with Detected Double Peak Cycles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================
# PLOT 1: Time-domain overlaps
# ============================================================
plt.figure(figsize=(14, 10), constrained_layout=True)
plt.subplot(2, 2, 1)
plt.plot(time_rp, sr_magnitude_interp, 'b-', label='Signal Recovery', linewidth=1.2, alpha=0.7)
plt.plot(time_rp, R_rp, 'r-', label='Red Pitaya', linewidth=1.2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('AC Magnitude (V)')
plt.title('Magnitude vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(time_rp, sr_phase_interp, 'b-', label='Signal Recovery', linewidth=1.2, alpha=0.7)
plt.plot(time_rp, Theta_rp, 'r-', label='Red Pitaya', linewidth=1.2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Phase (degrees)')
plt.title('Phase vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Time-Domain Signal Comparison (Signal Recovery vs Red Pitaya)', fontsize=14, fontweight='bold')
plt.savefig('comparison_time_domain.png', dpi=150)
plt.show()
print("Plot 1 saved: comparison_time_domain.png")

# ============================================================
# PLOT 2: Gain-corrected overlaps
# ============================================================
max_sr_mag = np.max(np.abs(sr_magnitude_interp))
max_rp_mag = np.max(np.abs(R_rp))
mag_scale  = max_sr_mag / max_rp_mag if max_rp_mag != 0 else 1.0

plt.figure(figsize=(14, 10), constrained_layout=True)
plt.subplot(2, 2, 1)
plt.plot(time_rp, sr_magnitude_interp, 'b-', label='Signal Recovery', linewidth=1.2, alpha=0.7)
plt.plot(time_rp, R_rp * mag_scale, 'r-', label=f'Red Pitaya (x{mag_scale:.3f})', linewidth=1.2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('AC Magnitude (V)')
plt.title('Gain-Corrected Magnitude vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(time_rp, sr_phase_interp, 'b-', label='Signal Recovery', linewidth=1.2, alpha=0.7)
plt.plot(time_rp, Theta_rp, 'r-', label='Red Pitaya', linewidth=1.2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Phase (degrees)')
plt.title('Phase vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Gain-Corrected Time-Domain Comparison', fontsize=14, fontweight='bold')
plt.savefig('comparison_gain_corrected.png', dpi=150)
plt.show()
print(f"Plot 2 saved: comparison_gain_corrected.png  (mag scale factor: {mag_scale:.4f})")

# ============================================================
# PLOT 3: Smoothed signal overlaps
# ============================================================
window_size = max(10, rp_len // 100)
R_rp_smooth     = uniform_filter1d(R_rp, size=window_size, mode='nearest')
Theta_rp_smooth = uniform_filter1d(Theta_rp, size=window_size, mode='nearest')

plt.figure(figsize=(14, 5), constrained_layout=True)
plt.subplot(1, 2, 1)
plt.plot(time_rp, sr_magnitude_interp, 'b-', label='Signal Recovery', linewidth=2, alpha=0.7)
plt.plot(time_rp, R_rp_smooth, 'r-', label='Red Pitaya (smoothed)', linewidth=2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('AC Magnitude (V)')
plt.title('Smoothed Magnitude vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(time_rp, sr_phase_interp, 'b-', label='Signal Recovery', linewidth=2, alpha=0.7)
plt.plot(time_rp, Theta_rp_smooth, 'r-', label='Red Pitaya (smoothed)', linewidth=2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Phase (degrees)')
plt.title('Smoothed Phase vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Smoothed Comparison (Signal Recovery vs Red Pitaya)', fontsize=14, fontweight='bold')
plt.savefig('comparison_smoothed.png', dpi=150)
plt.show()
print("Plot 3 saved: comparison_smoothed.png")

# ============================================================
# PLOT 4: Gain-corrected smoothed overlaps
# ============================================================
plt.figure(figsize=(14, 5), constrained_layout=True)
plt.subplot(1, 2, 1)
plt.plot(time_rp, sr_magnitude_interp, 'b-', label='Signal Recovery', linewidth=2, alpha=0.7)
plt.plot(time_rp, R_rp_smooth * mag_scale, 'r-', label=f'Red Pitaya (x{mag_scale:.3f})', linewidth=2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('AC Magnitude (V)')
plt.title('Gain-Corrected Smoothed Magnitude vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(time_rp, sr_phase_interp, 'b-', label='Signal Recovery', linewidth=2, alpha=0.7)
plt.plot(time_rp, Theta_rp_smooth, 'r-', label='Red Pitaya (smoothed)', linewidth=2, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Phase (degrees)')
plt.title('Gain-Corrected Smoothed Phase vs Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Gain-Corrected Smoothed Comparison (Signal Recovery vs Red Pitaya)', fontsize=14, fontweight='bold')
plt.savefig('comparison_gain_corrected_smoothed.png', dpi=150)
plt.show()
print(f"Plot 4 saved: comparison_gain_corrected_smoothed.png")

print(f"\nScaling factors applied:")
print(f"  Magnitude: {mag_scale:.4f}")
print("\n=== Comparison Complete ===")
