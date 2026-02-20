"""
Synchronous dual Red Pitaya data acquisition - ACCV Style
Plots styled for AC Cyclic Voltammetry visualization

Press Enter ONCE here to launch both acquisitions simultaneously.

SPEED IMPROVEMENTS vs old version:
- No longer embeds PNG images (was the main bottleneck)
- All combined plots drawn directly from CSV data
- Parallel CSV loading with threads
- Butterworth LP filter AC removal applied to DC during merge

Have a Great Day :)

Dominic Morris
"""

import subprocess
import threading
from datetime import datetime
import time
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
import glob

# ============================================================
# PARAMETERS
# ============================================================
OUTPUT_DIRECTORY = 'test_data'
START_TIME_FILE  = "start_time.txt"

ENTER_FEED_DELAY = 30  # seconds — increase if RP connection is slow

# ── AC bias removal on merged DC ─────────────────────────────────────────────
AC_REMOVAL_FS     = 1_000_000  # your actual sampling rate (Hz)
AC_REMOVAL_CUTOFF = 500        # choose well below LIA freq (Hz)
AC_REMOVAL_ORDER  = 4

# ============================================================

def butter_lowpass_filter(signal, fs, cutoff, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, signal)


print("=" * 60)
print("DUAL RED PITAYA SYNCHRONIZED ACQUISITION")
print("=" * 60)

python_exe = sys.executable

lockin_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lockin_with_timestamp.py")
dc_script     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dc_monitor_with_timestamp.py")

for path, name in [(lockin_script, "lock-in"), (dc_script, "DC")]:
    if not os.path.exists(path):
        print(f"Error: {path} not found!")
        sys.exit(1)

print(f"Lock-in script: {lockin_script}")
print(f"DC script:      {dc_script}")
print(f"AC removal:     Butterworth LP  fs={AC_REMOVAL_FS} Hz  cutoff={AC_REMOVAL_CUTOFF} Hz  order={AC_REMOVAL_ORDER}")
print("=" * 60)

input("\nPress Enter to start both acquisitions...")
print("")

START_TIME = datetime.now() + pd.Timedelta(seconds=2)
with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())

print(f"Start time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Launching (auto-start in ~{ENTER_FEED_DELAY}s once RPs connect)...")

proc_lockin = subprocess.Popen([python_exe, lockin_script], stdin=subprocess.PIPE)
proc_dc     = subprocess.Popen([python_exe, dc_script],     stdin=subprocess.PIPE)

def feed_enter(proc, delay, name):
    time.sleep(delay)
    try:
        proc.stdin.write(b"\n")
        proc.stdin.flush()
        print(f"  -> Sent Enter to {name}")
    except Exception as e:
        print(f"  -> Could not send Enter to {name}: {e}")

t_lockin = threading.Thread(target=feed_enter, args=(proc_lockin, ENTER_FEED_DELAY, "lock-in"), daemon=True)
t_dc     = threading.Thread(target=feed_enter, args=(proc_dc,     ENTER_FEED_DELAY, "DC"),      daemon=True)
t_lockin.start()
t_dc.start()

print("Waiting for acquisitions to complete...")
proc_lockin.wait()
proc_dc.wait()
print("\nBoth acquisitions finished")

try:
    os.remove(START_TIME_FILE)
except Exception:
    pass

time.sleep(0.3)

# ============================================================
# FAST CSV LOADING (parallel)
# ============================================================
print("\n" + "=" * 60)
print("LOADING & MERGING DATA")
print("=" * 60)

t_load_start = time.time()

try:
    lockin_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.csv')),  key=os.path.getctime)
    dc_csv     = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')),      key=os.path.getctime)
except ValueError:
    print("Error: Could not find output CSV files!")
    sys.exit(1)

print(f"Lock-in CSV: {lockin_csv}")
print(f"DC CSV:      {dc_csv}")

lockin_data, dc_data = [None], [None]

def load_lockin():
    lockin_data[0] = pd.read_csv(lockin_csv, comment='#', encoding='latin-1')

def load_dc():
    dc_data[0] = pd.read_csv(dc_csv, comment='#', encoding='latin-1')

t1 = threading.Thread(target=load_lockin)
t2 = threading.Thread(target=load_dc)
t1.start(); t2.start()
t1.join();  t2.join()

lockin_data = lockin_data[0]
dc_data     = dc_data[0]

print(f"Lock-in samples: {len(lockin_data):,}")
print(f"DC samples:      {len(dc_data):,}")
print(f"Load time: {time.time() - t_load_start:.2f}s")

n_samples = min(len(lockin_data), len(dc_data))

# Handle theta column name variants
if   'Theta(deg)' in lockin_data.columns: theta_col, theta_unit = 'Theta(deg)', 'deg'
elif 'Theta(rad)' in lockin_data.columns: theta_col, theta_unit = 'Theta(rad)', 'rad'
else:                                      theta_col, theta_unit = lockin_data.columns[3], 'deg'

dc_col  = 'DC_Voltage' if 'DC_Voltage' in dc_data.columns else 'Voltage(V)'
dc_raw  = dc_data[dc_col].values[:n_samples]
t_RP2   = dc_data['Time(s)'].values[:n_samples]

# ── Butterworth low-pass: remove AC bias from DC ramp ────────────────────────
print(f"\nApplying AC bias removal "
      f"(Butterworth LP: fs={AC_REMOVAL_FS} Hz, cutoff={AC_REMOVAL_CUTOFF} Hz, order={AC_REMOVAL_ORDER})...")
t_filt   = time.time()
dc_clean = butter_lowpass_filter(dc_raw.astype(float),
                                 AC_REMOVAL_FS, AC_REMOVAL_CUTOFF, AC_REMOVAL_ORDER)
print(f"Filter done in {time.time() - t_filt:.2f}s")

R     = lockin_data['R(V)'].values[:n_samples]
Theta = lockin_data[theta_col].values[:n_samples]
X     = lockin_data['X(V)'].values[:n_samples]
Y     = lockin_data['Y(V)'].values[:n_samples]
t_RP1 = lockin_data['Time(s)'].values[:n_samples]

# ============================================================
# SAVE MERGED CSV
# ============================================================
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv    = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')

merged_df = pd.DataFrame({
    'Index':      np.arange(n_samples),
    'Time_RP1':   t_RP1,
    'Time_RP2':   t_RP2,
    'R':          R,
    'Theta':      Theta,
    'X':          X,
    'Y':          Y,
    'DC_Voltage': dc_clean,
})
merged_df.to_csv(merged_csv, index=False)
print(f"\nSaved merged CSV: {merged_csv}")

time_diff = t_RP1 - t_RP2
print(f"Sync quality: mean={np.mean(time_diff)*1000:.1f}ms  max={np.max(np.abs(time_diff))*1000:.1f}ms")

# ============================================================
# FAST COMBINED PLOT (direct from data, no PNG embedding)
# ============================================================
print("\nRendering combined plot...")
t_plot = time.time()

fig = plt.figure(figsize=(18, 12))
fig.suptitle(f'AC Cyclic Voltammetry -- Synchronized Dual Red Pitaya\n'
             f'{n_samples:,} samples  |  AC removed: Butterworth LP @ {AC_REMOVAL_CUTOFF} Hz',
             fontsize=14, fontweight='bold')

# Layout: top row = 3 time series, bottom row = 2 vs-DC plots
ax_r   = fig.add_subplot(2, 3, 1)
ax_th  = fig.add_subplot(2, 3, 2)
ax_dc  = fig.add_subplot(2, 3, 3)
ax_rdc = fig.add_subplot(2, 3, 4)
ax_tdc = fig.add_subplot(2, 3, 5)
# centre the bottom row by hiding the 6th slot
fig.add_subplot(2, 3, 6).set_visible(False)

# Downsample for plotting only
MAX_PLOT_PTS = 50_000
step = max(1, n_samples // MAX_PLOT_PTS)
sl   = slice(None, None, step)

dc_p = dc_clean[sl]; R_p = R[sl]; Th_p = Theta[sl]
t_p  = t_RP1[sl]

# --- R vs Time ---
ax_r.plot(t_p, R_p, lw=0.7, color='steelblue')
ax_r.set_xlabel('Time (s)', fontweight='bold')
ax_r.set_ylabel('R (V)', fontweight='bold')
ax_r.set_title('R vs Time')
ax_r.grid(True, alpha=0.3)

# --- Theta vs Time ---
ax_th.plot(t_p, Th_p, lw=0.7, color='tomato')
ax_th.set_xlabel('Time (s)', fontweight='bold')
ax_th.set_ylabel(f'Theta ({theta_unit})', fontweight='bold')
ax_th.set_title('Theta vs Time')
ax_th.grid(True, alpha=0.3)

# --- DC vs Time ---
ax_dc.plot(t_p, dc_p, lw=0.7, color='green')
ax_dc.set_xlabel('Time (s)', fontweight='bold')
ax_dc.set_ylabel('DC Potential (V)', fontweight='bold')
ax_dc.set_title('DC Voltage vs Time')
ax_dc.grid(True, alpha=0.3)

# --- R vs DC ---
ax_rdc.scatter(dc_p, R_p, s=2, alpha=0.5, c='steelblue')
ax_rdc.set_xlabel('DC Potential (V)', fontweight='bold')
ax_rdc.set_ylabel('R (V)', fontweight='bold')
ax_rdc.set_title('R vs DC Potential')
ax_rdc.grid(True, alpha=0.3)

# --- Theta vs DC ---
ax_tdc.scatter(dc_p, Th_p, s=2, alpha=0.5, c='tomato')
ax_tdc.set_xlabel('DC Potential (V)', fontweight='bold')
ax_tdc.set_ylabel(f'Theta ({theta_unit})', fontweight='bold')
ax_tdc.set_title('Theta vs DC Potential')
ax_tdc.grid(True, alpha=0.3)

plt.tight_layout()
print(f"Plot rendered in {time.time() - t_plot:.2f}s")

combined_png = os.path.join(OUTPUT_DIRECTORY, f'accv_combined_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"Saved: {combined_png}")

# ============================================================
# STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("ACCV STATISTICS")
print("=" * 60)
print(f"AC Magnitude (R): {np.mean(R):.6f} +/- {np.std(R):.6f} V")
print(f"Phase Angle:      {np.mean(Theta):.6f} +/- {np.std(Theta):.6f} {theta_unit}")
print(f"DC Potential:     {np.mean(dc_clean):.6f} +/- {np.std(dc_clean):.6f} V")
print(f"\nCorrelations:")
print(f"  R vs DC:     {np.corrcoef(R, dc_clean[:len(R)])[0,1]:.4f}")
print(f"  Phase vs DC: {np.corrcoef(Theta, dc_clean[:len(Theta)])[0,1]:.4f}")
print("=" * 60)

plt.show()
print("\nCOMPLETE")
print("=" * 60)
