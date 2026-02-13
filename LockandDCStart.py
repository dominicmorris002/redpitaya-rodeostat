"""
Synchronous dual Red Pitaya data acquisition - DC Ramp Synchronized
Launches X+Ramp and Y+Ramp loggers, then merges data using DC ramp as common reference.

Hardware Setup:
- DC Ramp (triangle wave) → IN2 on BOTH Red Pitayas
- Lock-in X output → RP1 (reads on iq2)
- Lock-in Y output → RP2 (reads on iq2_2)

Merging strategy:
  - Detects individual triangle sweep segments (up and down) from both ramps
  - Matches X and Y samples within each sweep by nearest DC voltage
  - Handles timing offsets between the two devices
  - Averages across repeated cycles for noise reduction
"""

import subprocess
from datetime import datetime
import time
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.image import imread
import glob
from scipy import interpolate
from scipy.signal import find_peaks

# ============================================================
# SYNCHRONIZATION PARAMETERS
# ============================================================
START_DELAY = 5  # seconds
OUTPUT_DIRECTORY = 'test_data'

# IN2 DC Ramp scaling - must match LockXandDC.py and LockYandDC.py
IN2_GAIN_FACTOR = 1.0    # same as IN2_GAIN_FACTOR in both logger scripts
IN2_DC_OFFSET = 0.0      # same as IN2_DC_OFFSET in both logger scripts

# Triangle wave detection tuning
# Minimum fraction of the ramp voltage range that counts as a full sweep
MIN_SWEEP_FRACTION = 0.7
# How many voltage bins to use when averaging a sweep
N_VOLTAGE_BINS = 500
# Minimum number of samples a segment needs to be kept
MIN_SEGMENT_SAMPLES = 50

# ============================================================

START_TIME_FILE = "start_time.txt"
START_TIME = datetime.now() + pd.Timedelta(seconds=START_DELAY)

with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())

print("=" * 60)
print("DUAL RED PITAYA - DC RAMP SYNCHRONIZED ACQUISITION")
print("=" * 60)
print(f"Start time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Waiting {(START_TIME - datetime.now()).total_seconds():.1f}s...")
print(f"IN2 ramp scaling: gain={IN2_GAIN_FACTOR:.4f}x, offset={IN2_DC_OFFSET:.6f}V")
print("=" * 60)

python_exe = sys.executable
x_script = os.path.join(os.path.dirname(__file__), "LockXandDC.py")
y_script = os.path.join(os.path.dirname(__file__), "LockYandDC.py")

if not os.path.exists(x_script):
    print(f"❌ Error: {x_script} not found!")
    sys.exit(1)
if not os.path.exists(y_script):
    print(f"❌ Error: {y_script} not found!")
    sys.exit(1)

while datetime.now() < START_TIME:
    time.sleep(0.001)

print("\n🚀 Launching both acquisitions...")
proc_x = subprocess.Popen([python_exe, x_script])
proc_y = subprocess.Popen([python_exe, y_script])

print("⏳ Waiting for acquisitions to complete...")
proc_x.wait()
proc_y.wait()
print("\n✓ Both acquisitions finished")

try:
    os.remove(START_TIME_FILE)
except:
    pass

time.sleep(0.5)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_ramp_scaling(raw_voltage, gain, offset):
    """Apply the same IN2 correction used in the logger scripts."""
    return (raw_voltage - offset) * gain


def detect_sweep_segments(ramp, min_sweep_fraction=0.7, min_samples=50):
    """
    Split a triangle-wave ramp into individual up/down sweep segments.

    The ramp may have millions of samples but only a handful of triangle cycles.
    Running find_peaks on the raw array fails because the required 'distance'
    parameter would need to be in the millions.  Strategy:

      1. Downsample to ~10 000 points so find_peaks sees a clean triangle.
      2. Detect turning-point indices in the downsampled array.
      3. Scale those indices back to the original sample grid.

    Returns a list of dicts:
        {'indices': array, 'direction': 'up'|'down',
         'v_start': float, 'v_end': float,
         'v_min': float, 'v_max': float}
    """
    ramp = np.asarray(ramp, dtype=float)
    n = len(ramp)

    # ── Downsample for peak detection ─────────────────────────────────────
    TARGET_DS_LEN = 10_000
    ds_factor = max(1, n // TARGET_DS_LEN)
    ramp_ds = ramp[::ds_factor]
    n_ds = len(ramp_ds)

    v_range = np.max(ramp_ds) - np.min(ramp_ds)
    if v_range < 1e-6:
        print("  WARNING: Ramp has no voltage variation - cannot detect segments.")
        return []

    min_sweep_height = v_range * min_sweep_fraction
    prominence = min_sweep_height * 0.4
    # Each half-cycle must span at least 1% of the downsampled array
    min_dist_ds = max(5, n_ds // 200)

    peaks_ds,   _ = find_peaks( ramp_ds, prominence=prominence, distance=min_dist_ds)
    valleys_ds, _ = find_peaks(-ramp_ds, prominence=prominence, distance=min_dist_ds)

    # Scale back to original indices
    peaks   = peaks_ds   * ds_factor
    valleys = valleys_ds * ds_factor

    print(f"    downsample: {ds_factor}x  ({n} -> {n_ds} pts)  "
          f"peaks: {len(peaks)}  valleys: {len(valleys)}")

    # Build sorted turning-point list including array boundaries
    turning_points = np.unique(np.sort(
        np.concatenate(([0], peaks, valleys, [n - 1]))
    ))

    segments = []
    for i in range(len(turning_points) - 1):
        i0 = int(turning_points[i])
        i1 = int(turning_points[i + 1])
        seg_indices = np.arange(i0, i1 + 1)
        if len(seg_indices) < min_samples:
            continue
        v_start = ramp[i0]
        v_end   = ramp[i1]
        sweep_height = abs(v_end - v_start)
        if sweep_height < min_sweep_height:
            continue
        direction = 'up' if v_end > v_start else 'down'
        segments.append({
            'indices':   seg_indices,
            'direction': direction,
            'v_start':   v_start,
            'v_end':     v_end,
            'v_min':     min(v_start, v_end),
            'v_max':     max(v_start, v_end),
        })

    return segments


def bin_sweep(ramp_seg, signal_seg, n_bins, v_min, v_max):
    """
    Bin a single sweep onto a uniform voltage grid (fully vectorized).
    Returns (bin_centers, binned_signal, bin_counts).
    """
    bin_edges   = np.linspace(v_min, v_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # digitize gives 1-based bin numbers; clip to valid range
    idx = np.clip(np.digitize(ramp_seg, bin_edges) - 1, 0, n_bins - 1)

    bin_sum    = np.bincount(idx, weights=signal_seg.astype(float), minlength=n_bins)
    bin_counts = np.bincount(idx,                                    minlength=n_bins)

    binned = np.full(n_bins, np.nan)
    mask = bin_counts > 0
    binned[mask] = bin_sum[mask] / bin_counts[mask]

    return bin_centers, binned, bin_counts


def merge_segments_by_voltage(x_segs, y_segs, ramp_x, ramp_y,
                               x_signal, y_signal,
                               n_bins=500, min_sweep_fraction=0.7):
    """
    For each pair of matching (direction, approximate voltage range) segments,
    bin both onto the same voltage grid and pair them up.

    Returns arrays: (voltage, X_merged, Y_merged) already averaged over cycles.
    """
    # Global voltage overlap
    global_v_min = max(np.min(ramp_x), np.min(ramp_y))
    global_v_max = min(np.max(ramp_x), np.max(ramp_y))
    print(f"  Global ramp overlap: {global_v_min:.4f} → {global_v_max:.4f} V")

    # Bucket segments by direction
    x_up   = [s for s in x_segs if s['direction'] == 'up']
    x_down = [s for s in x_segs if s['direction'] == 'down']
    y_up   = [s for s in y_segs if s['direction'] == 'up']
    y_down = [s for s in y_segs if s['direction'] == 'down']

    print(f"  X sweeps — up: {len(x_up)}, down: {len(x_down)}")
    print(f"  Y sweeps — up: {len(y_up)}, down: {len(y_down)}")

    bin_edges = np.linspace(global_v_min, global_v_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    results = {}  # direction → {'X_stack': [], 'Y_stack': []}

    for direction, x_list, y_list in [('up', x_up, y_up), ('down', x_down, y_down)]:
        if not x_list or not y_list:
            continue

        x_stack = []
        y_stack = []

        # Bin every X sweep onto the common grid
        for seg in x_list:
            idx = seg['indices']
            _, binned, counts = bin_sweep(
                ramp_x[idx], x_signal[idx], n_bins, global_v_min, global_v_max)
            if np.sum(counts > 0) > n_bins * 0.3:   # at least 30% coverage
                x_stack.append(binned)

        # Bin every Y sweep onto the common grid
        for seg in y_list:
            idx = seg['indices']
            _, binned, counts = bin_sweep(
                ramp_y[idx], y_signal[idx], n_bins, global_v_min, global_v_max)
            if np.sum(counts > 0) > n_bins * 0.3:
                y_stack.append(binned)

        if not x_stack or not y_stack:
            continue

        # Average across cycles (nanmean ignores empty bins)
        X_avg = np.nanmean(np.array(x_stack), axis=0)
        Y_avg = np.nanmean(np.array(y_stack), axis=0)

        results[direction] = {
            'voltage': bin_centers,
            'X': X_avg,
            'Y': Y_avg,
            'n_x_sweeps': len(x_stack),
            'n_y_sweeps': len(y_stack),
        }

    return results


# ============================================================
# LOAD DATA
# ============================================================
try:
    x_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_x_ramp_*.csv')), key=os.path.getctime)
    y_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_y_ramp_*.csv')), key=os.path.getctime)
    x_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_x_ramp_*.png')), key=os.path.getctime)
    y_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_y_ramp_*.png')), key=os.path.getctime)
except ValueError:
    print("❌ Error: Could not find output files!")
    sys.exit(1)

print("\n" + "=" * 60)
print("MERGING DATA USING DC RAMP REFERENCE")
print("=" * 60)
print(f"X file: {os.path.basename(x_csv)}")
print(f"Y file: {os.path.basename(y_csv)}")

x_data = pd.read_csv(x_csv, comment='#', encoding='latin-1')
y_data = pd.read_csv(y_csv, comment='#', encoding='latin-1')

print(f"X samples: {len(x_data)},  Y samples: {len(y_data)}")

# Apply IN2 scaling (same parameters as in the logger scripts)
ramp_x = apply_ramp_scaling(x_data['DCRamp(V)'].values, IN2_GAIN_FACTOR, IN2_DC_OFFSET)
ramp_y = apply_ramp_scaling(y_data['DCRamp(V)'].values, IN2_GAIN_FACTOR, IN2_DC_OFFSET)

x_signal = x_data['X(V)'].values
y_signal = y_data['Y(V)'].values

print(f"\nX ramp range: {np.min(ramp_x):.4f} → {np.max(ramp_x):.4f} V")
print(f"Y ramp range: {np.min(ramp_y):.4f} → {np.max(ramp_y):.4f} V")

# ============================================================
# DETECT TRIANGLE SWEEP SEGMENTS
# ============================================================
print("\nDetecting triangle sweep segments...")
x_segs = detect_sweep_segments(ramp_x, MIN_SWEEP_FRACTION, MIN_SEGMENT_SAMPLES)
y_segs = detect_sweep_segments(ramp_y, MIN_SWEEP_FRACTION, MIN_SEGMENT_SAMPLES)

print(f"  X: found {len(x_segs)} segments  "
      f"({sum(1 for s in x_segs if s['direction']=='up')} up, "
      f"{sum(1 for s in x_segs if s['direction']=='down')} down)")
print(f"  Y: found {len(y_segs)} segments  "
      f"({sum(1 for s in y_segs if s['direction']=='up')} up, "
      f"({sum(1 for s in y_segs if s['direction']=='down')} down)")

if not x_segs or not y_segs:
    print("⚠ WARNING: No sweep segments detected. "
          "Check MIN_SWEEP_FRACTION or MIN_SEGMENT_SAMPLES.")
    sys.exit(1)

# ============================================================
# MERGE BY VOLTAGE BINNING
# ============================================================
print("\nMerging sweeps by voltage binning...")
merged = merge_segments_by_voltage(
    x_segs, y_segs, ramp_x, ramp_y,
    x_signal, y_signal,
    n_bins=N_VOLTAGE_BINS,
    min_sweep_fraction=MIN_SWEEP_FRACTION
)

if not merged:
    print("❌ No matching sweep pairs found! Check your ramp data.")
    sys.exit(1)

# Combine up and down sweeps into one dataset (or keep separate if preferred)
all_voltages = []
all_X        = []
all_Y        = []
all_dirs     = []

for direction, res in merged.items():
    v   = res['voltage']
    X   = res['X']
    Y   = res['Y']
    # Only keep bins that have valid data in BOTH X and Y
    valid = ~np.isnan(X) & ~np.isnan(Y)
    all_voltages.append(v[valid])
    all_X.append(X[valid])
    all_Y.append(Y[valid])
    all_dirs.extend([direction] * np.sum(valid))
    print(f"  {direction.upper()} sweep: {np.sum(valid)} valid voltage bins, "
          f"averaged over {res['n_x_sweeps']} X / {res['n_y_sweeps']} Y cycles")

voltage_merged = np.concatenate(all_voltages)
X_merged       = np.concatenate(all_X)
Y_merged       = np.concatenate(all_Y)

# Sort by voltage for clean plots
sort_idx       = np.argsort(voltage_merged)
voltage_merged = voltage_merged[sort_idx]
X_merged       = X_merged[sort_idx]
Y_merged       = Y_merged[sort_idx]

R_merged     = np.sqrt(X_merged**2 + Y_merged**2)
Theta_merged = np.degrees(np.arctan2(Y_merged, X_merged))
n_merged     = len(voltage_merged)

print(f"\nTotal merged data points: {n_merged}")
print(f"Mean R:     {np.nanmean(R_merged):.6f} ± {np.nanstd(R_merged):.6f} V")
print(f"Mean X:     {np.nanmean(X_merged):.6f} ± {np.nanstd(X_merged):.6f} V")
print(f"Mean Y:     {np.nanmean(Y_merged):.6f} ± {np.nanstd(Y_merged):.6f} V")
print(f"Mean Theta: {np.nanmean(Theta_merged):.3f} ± {np.nanstd(Theta_merged):.3f}°")

# ============================================================
# SAVE MERGED CSV
# ============================================================
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

merged_df = pd.DataFrame({
    'Index':      np.arange(n_merged),
    'DC_Voltage': voltage_merged,
    'R':          R_merged,
    'Theta':      Theta_merged,
    'X':          X_merged,
    'Y':          Y_merged,
})

merged_csv = os.path.join(OUTPUT_DIRECTORY, f'ramp_sync_combined_{timestamp_str}.csv')
merged_df.to_csv(merged_csv, index=False)
print(f"\n✓ Saved merged CSV: {merged_csv}")

# Also save per-direction CSVs if both directions exist
for direction, res in merged.items():
    v = res['voltage']
    X = res['X']
    Y = res['Y']
    valid = ~np.isnan(X) & ~np.isnan(Y)
    R_dir = np.sqrt(X[valid]**2 + Y[valid]**2)
    T_dir = np.degrees(np.arctan2(Y[valid], X[valid]))
    df_dir = pd.DataFrame({
        'DC_Voltage': v[valid],
        'R': R_dir, 'Theta': T_dir,
        'X': X[valid], 'Y': Y[valid]
    })
    dir_csv = os.path.join(OUTPUT_DIRECTORY,
                           f'ramp_sync_{direction}_{timestamp_str}.csv')
    df_dir.to_csv(dir_csv, index=False)
    print(f"✓ Saved {direction} sweep CSV: {dir_csv}")

# ============================================================
# PLOTS
# ============================================================
fig = plt.figure(figsize=(20, 12))

# ── Row 0: original subplot images ────────────────────────
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax1.imshow(imread(x_png))
ax1.axis('off')
ax1.set_title('Lock-in X + DC Ramp (RP1)', fontsize=13, fontweight='bold')

ax2 = plt.subplot2grid((3, 3), (0, 1))
ax2.imshow(imread(y_png))
ax2.axis('off')
ax2.set_title('Lock-in Y + DC Ramp (RP2)', fontsize=13, fontweight='bold')

# ── Segment detection diagnostic ──────────────────────────
ax_diag = plt.subplot2grid((3, 3), (0, 2))
# Downsample both ramps to ~10k points so the triangle shape is visible
diag_ds = max(1, min(len(ramp_x), len(ramp_y)) // 10_000)
rx_ds = ramp_x[::diag_ds]
ry_ds = ramp_y[::diag_ds]
t_x = np.arange(len(rx_ds)) * diag_ds
t_y = np.arange(len(ry_ds)) * diag_ds
ax_diag.plot(t_x, rx_ds, 'b-', linewidth=0.6, alpha=0.8, label='X ramp')
ax_diag.plot(t_y, ry_ds, 'r-', linewidth=0.6, alpha=0.8, label='Y ramp')
# Mark segment boundaries
for seg in x_segs:
    ax_diag.axvline(seg['indices'][0],  color='b', linewidth=0.8, alpha=0.5, linestyle='--')
    ax_diag.axvline(seg['indices'][-1], color='b', linewidth=0.8, alpha=0.5, linestyle=':')
for seg in y_segs:
    ax_diag.axvline(seg['indices'][0],  color='r', linewidth=0.8, alpha=0.5, linestyle='--')
    ax_diag.axvline(seg['indices'][-1], color='r', linewidth=0.8, alpha=0.5, linestyle=':')
ax_diag.set_xlabel('Sample index', fontsize=10)
ax_diag.set_ylabel('DC Ramp (V)', fontsize=10)
n_x_segs = len(x_segs)
n_y_segs = len(y_segs)
ax_diag.set_title(f'Ramp Segments  (X: {n_x_segs}, Y: {n_y_segs})', fontsize=11, fontweight='bold')
ax_diag.legend(fontsize=9)
ax_diag.grid(True, alpha=0.3)

# ── Row 1: IQ, R vs Voltage, Theta vs Voltage ─────────────
ax4 = plt.subplot2grid((3, 3), (1, 0))
# colour by direction if both present
if len(merged) == 2:
    for direction, res in merged.items():
        v = res['voltage']
        X = res['X']
        Y = res['Y']
        valid = ~np.isnan(X) & ~np.isnan(Y)
        col = 'steelblue' if direction == 'up' else 'tomato'
        ax4.plot(X[valid], Y[valid], '.', color=col, markersize=1,
                 alpha=0.5, label=direction)
    ax4.legend(fontsize=9, markerscale=5)
else:
    ax4.plot(X_merged, Y_merged, 'g.', markersize=1, alpha=0.5)
ax4.plot(np.nanmean(X_merged), np.nanmean(Y_merged), 'r+',
         markersize=15, markeredgewidth=2, label='Mean')
ax4.set_xlabel('X (V)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Y (V)', fontsize=11, fontweight='bold')
ax4.set_title('IQ Plot', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

ax5 = plt.subplot2grid((3, 3), (1, 1))
ax5.plot(voltage_merged, R_merged * 1e6, 'm-', linewidth=1.0)
ax5.set_xlabel('DC Ramp (V)', fontsize=11, fontweight='bold')
ax5.set_ylabel('R (μA)', fontsize=11, fontweight='bold')
ax5.set_title('Magnitude R vs DC Voltage', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot2grid((3, 3), (1, 2))
ax6.plot(voltage_merged, Theta_merged, 'c-', linewidth=1.0)
ax6.set_xlabel('DC Ramp (V)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Theta (°)', fontsize=11, fontweight='bold')
ax6.set_title('Phase vs DC Voltage', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# ── Row 2: up/down overlaid, R coloured by phase, X and Y ──
ax7 = plt.subplot2grid((3, 3), (2, 0))
colors_dir = {'up': 'steelblue', 'down': 'tomato'}
for direction, res in merged.items():
    v   = res['voltage']
    X   = res['X']
    Y   = res['Y']
    valid = ~np.isnan(X) & ~np.isnan(Y)
    R_d = np.sqrt(X[valid]**2 + Y[valid]**2)
    ax7.plot(v[valid], R_d * 1e6,
             color=colors_dir.get(direction, 'gray'),
             linewidth=1.2, label=f'{direction} sweep')
ax7.set_xlabel('DC Ramp (V)', fontsize=11, fontweight='bold')
ax7.set_ylabel('R (μA)', fontsize=11, fontweight='bold')
ax7.set_title('R — up vs down sweeps', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot2grid((3, 3), (2, 1))
sc = ax8.scatter(voltage_merged, R_merged * 1e6,
                 c=Theta_merged, s=3, alpha=0.7,
                 cmap='viridis', marker='.')
ax8.set_xlabel('DC Potential (V)', fontsize=11, fontweight='bold')
ax8.set_ylabel('R (μA)', fontsize=11, fontweight='bold')
ax8.set_title('AC Response Map (colour = phase)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax8)
cbar.set_label('Phase (°)', fontsize=10)

ax9 = plt.subplot2grid((3, 3), (2, 2))
ax9.plot(voltage_merged, X_merged, 'b-', linewidth=0.8, label='X', alpha=0.8)
ax9.plot(voltage_merged, Y_merged, 'r-', linewidth=0.8, label='Y', alpha=0.8)
ax9.set_xlabel('DC Ramp (V)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Signal (V)', fontsize=11, fontweight='bold')
ax9.set_title('X and Y vs DC Voltage', fontsize=12, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)

fig.suptitle(
    f'AC Cyclic Voltammetry — Triangle-Wave Ramp Synchronized\n'
    f'{n_merged} merged voltage bins  |  '
    f'{sum(len(v["indices"]) for v in x_segs)} X samples  |  '
    f'{sum(len(v["indices"]) for v in y_segs)} Y samples',
    fontsize=15, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

combined_png = os.path.join(OUTPUT_DIRECTORY, f'ramp_sync_accv_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved ACCV plot: {combined_png}")

print("\n" + "=" * 60)
print("AC CYCLIC VOLTAMMETRY STATISTICS")
print("=" * 60)
print(f"R:      {np.nanmean(R_merged)*1e6:.3f} ± {np.nanstd(R_merged)*1e6:.3f} μA")
print(f"Theta:  {np.nanmean(Theta_merged):.3f} ± {np.nanstd(Theta_merged):.3f}°")
print(f"X:      {np.nanmean(X_merged):.6f} ± {np.nanstd(X_merged):.6f} V")
print(f"Y:      {np.nanmean(Y_merged):.6f} ± {np.nanstd(Y_merged):.6f} V")
print(f"DC span: {np.min(voltage_merged):.4f} → {np.max(voltage_merged):.4f} V")
print("=" * 60)

plt.show()
print("\n✓ COMPLETE")
print("=" * 60)
