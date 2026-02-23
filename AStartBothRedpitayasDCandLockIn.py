"""
Synchronous dual Red Pitaya data acquisition - ACCV Style
Plots styled for AC Cyclic Voltammetry visualization

Press Enter ONCE here to launch both acquisitions simultaneously.
Then press Enter a SECOND time when both RPs are connected and ready
to fire the measurement start to both at the same instant.

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
from matplotlib.image import imread
import glob

# ============================================================
# PARAMETERS
# ============================================================
OUTPUT_DIRECTORY = 'test_data'
START_TIME_FILE  = "start_time.txt"

# ============================================================
# COMBINED PLOT DOWNSAMPLING
# ============================================================
# Individual CSVs are always saved at full resolution by each script.
# This only affects the merged combined_results CSV and the ACCV PNG.
# 10_000 pts is plenty for a smooth CV curve at any sweep rate.
COMBINED_MAX_POINTS = 10_000
# ============================================================

# ============================================================
# STARTUP BANNER
# ============================================================
print("=" * 60)
print("DUAL RED PITAYA SYNCHRONIZED ACQUISITION")
print("=" * 60)

python_exe = sys.executable

lockin_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lockin_with_timestamp.py")
dc_script     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dc_monitor_with_timestamp.py")

if not os.path.exists(lockin_script):
    print(f"Error: {lockin_script} not found!")
    sys.exit(1)
if not os.path.exists(dc_script):
    print(f"Error: {dc_script} not found!")
    sys.exit(1)

print(f"Lock-in script:  {lockin_script}")
print(f"DC script:       {dc_script}")
print("=" * 60)

# ── First Enter: launch subprocesses ─────────────────────────────────────────
input("\nPress Enter to start both acquisitions...")
print("")
# ─────────────────────────────────────────────────────────────────────────────

# Write a start_time file that both subprocess scripts will read and wait for.
# 2 seconds gives them time to connect to the Red Pitaya before the clock hits.
START_TIME = datetime.now() + pd.Timedelta(seconds=2)
with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())

print(f"Start time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print("Launching both subprocesses -- watch for both RPs to connect below...")
print("")

# Launch both subprocesses with stdin=PIPE so we can feed them a newline
proc_lockin = subprocess.Popen([python_exe, lockin_script], stdin=subprocess.PIPE)
proc_dc     = subprocess.Popen([python_exe, dc_script],     stdin=subprocess.PIPE)

# ── Second Enter: fires both the instant YOU choose ──────────────────────────
input("Press Enter when both RPs are connected and ready to start measurement...")
print("")
proc_lockin.stdin.write(b"\n")
proc_lockin.stdin.flush()
proc_dc.stdin.write(b"\n")
proc_dc.stdin.flush()
print("  -> Sent Enter to lock-in")
print("  -> Sent Enter to DC")
# ─────────────────────────────────────────────────────────────────────────────

print("Waiting for acquisitions to complete...")
proc_lockin.wait()
proc_dc.wait()
print("\nBoth acquisitions finished")

try:
    os.remove(START_TIME_FILE)
except Exception:
    pass

time.sleep(0.5)

# ============================================================
# INDEX-BASED MERGING  (downsampled for speed)
# ============================================================
try:
    lockin_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.csv')),  key=os.path.getctime)
    dc_csv     = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')),      key=os.path.getctime)
    lockin_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.png')),  key=os.path.getctime)
    dc_png     = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.png')),      key=os.path.getctime)
except ValueError:
    print("Error: Could not find output files!")
    sys.exit(1)

print("\n" + "=" * 60)
print("MERGING DATA (INDEX-BASED)")
print("=" * 60)

# usecols skips unused columns so pandas doesn't load every column from a 12M-row file
print("Reading lock-in CSV...")
lockin_data = pd.read_csv(lockin_csv, comment='#', encoding='latin-1',
                          usecols=lambda c: c in ('Time(s)', 'R(V)', 'Theta(deg)',
                                                   'Theta(°)', 'Theta(rad)', 'X(V)', 'Y(V)'))
print("Reading DC CSV...")
dc_data     = pd.read_csv(dc_csv,     comment='#', encoding='latin-1',
                          usecols=lambda c: c in ('Time(s)', 'Voltage(V)'))

n_lockin = len(lockin_data)
n_dc     = len(dc_data)

print(f"Lock-in samples: {n_lockin:,}")
print(f"DC samples:      {n_dc:,}")

n_samples = min(n_lockin, n_dc)
print(f"\nUsing index-based merge")
print(f"  Common samples: {n_samples:,}")
if n_lockin != n_dc:
    diff = abs(n_lockin - n_dc)
    pct  = diff / max(n_lockin, n_dc) * 100
    print(f"  Sample count mismatch: {diff:,} samples ({pct:.1f}%)")

# Handle all theta column name variants
if   'Theta(deg)' in lockin_data.columns: theta_col, theta_unit = 'Theta(deg)', 'deg'
elif 'Theta(°)'   in lockin_data.columns: theta_col, theta_unit = 'Theta(°)',   '°'
else:                                      theta_col, theta_unit = 'Theta(rad)', 'rad'

# ── Downsample before building merged DataFrame ───────────────────────────────
# Slice numpy arrays directly at step -- never build a 12M-row merged frame
step = max(1, n_samples // COMBINED_MAX_POINTS)
idx  = np.arange(0, n_samples, step)
n_ds = len(idx)
print(f"\nDownsampling for combined plot/CSV: {n_samples:,} -> {n_ds:,} pts (step={step})")

li_time  = lockin_data['Time(s)'].values[:n_samples][idx]
li_R     = lockin_data['R(V)'].values[:n_samples][idx]
li_Theta = lockin_data[theta_col].values[:n_samples][idx]
li_X     = lockin_data['X(V)'].values[:n_samples][idx]
li_Y     = lockin_data['Y(V)'].values[:n_samples][idx]
dc_time  = dc_data['Time(s)'].values[:n_samples][idx]
dc_V     = dc_data['Voltage(V)'].values[:n_samples][idx]

time_diff = li_time - dc_time
print(f"\nTime synchronization quality:")
print(f"  Mean time difference: {np.mean(time_diff)*1000:.3f} ms")
print(f"  Max time difference:  {np.max(np.abs(time_diff))*1000:.3f} ms")

if   np.max(np.abs(time_diff)) < 0.1: print("  Excellent sync (< 100 ms)")
elif np.max(np.abs(time_diff)) < 0.5: print("  Good sync (< 500 ms)")
else:                                  print("  Poor sync (> 500 ms)")

# Save downsampled merged CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv    = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')

merged_df = pd.DataFrame({
    'Index':      np.arange(n_ds),
    'Time_RP1':   li_time,
    'Time_RP2':   dc_time,
    'R':          li_R,
    'Theta':      li_Theta,
    'X':          li_X,
    'Y':          li_Y,
    'DC_Voltage': dc_V,
})
merged_df.to_csv(merged_csv, index=False)
print(f"\nSaved merged CSV ({n_ds:,} rows, step={step}): {merged_csv}")

# ============================================================
# CREATE ACCV-STYLE COMBINED PLOT
# ============================================================
ds_note = f' [1:{step} ds]' if step > 1 else ''

fig = plt.figure(figsize=(20, 12))

# Top row: original individual plots
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax1.imshow(imread(lockin_png))
ax1.axis('off')
ax1.set_title('Lock-in Amplifier Results (AC Signal Demodulation)',
              fontsize=14, fontweight='bold', pad=10)

ax2 = plt.subplot2grid((3, 2), (0, 1))
ax2.imshow(imread(dc_png))
ax2.axis('off')
ax2.set_title('DC Voltage Monitor Results (Potentiostat Output)',
              fontsize=14, fontweight='bold', pad=10)

# Middle row: ACCV plots
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax3.plot(dc_V, li_R, 'b-', linewidth=1.5, alpha=0.8)
ax3.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
ax3.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.set_title(f'AC Magnitude vs DC Potential{ds_note}', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot2grid((3, 2), (1, 1))
ax4.plot(dc_V, li_Theta, 'r-', linewidth=1.5, alpha=0.8)
ax4.set_xlabel('DC Potential (V)',            fontsize=12, fontweight='bold')
ax4.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold', color='r')
ax4.tick_params(axis='y', labelcolor='r')
ax4.set_title(f'Phase Angle vs DC Potential{ds_note}', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Bottom row: time series + scatter map
ax5   = plt.subplot2grid((3, 2), (2, 0))
ax5_r = ax5.twinx()
ax5.plot(  li_time, dc_V, 'g-', linewidth=1.0, alpha=0.7, label='DC Potential')
ax5_r.plot(li_time, li_R, 'b-', linewidth=1.0, alpha=0.7, label='AC Magnitude')
ax5.set_xlabel('Time (s)',           fontsize=12, fontweight='bold')
ax5.set_ylabel('DC Potential (V)',   fontsize=11, fontweight='bold', color='g')
ax5_r.set_ylabel('AC Magnitude (V)', fontsize=11, fontweight='bold', color='b')
ax5.tick_params(axis='y',  labelcolor='g')
ax5_r.tick_params(axis='y', labelcolor='b')
ax5.set_title(f'Time Series: DC Potential & AC Response{ds_note}', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

ax6     = plt.subplot2grid((3, 2), (2, 1))
scatter = ax6.scatter(dc_V, li_R, c=li_Theta, s=2, alpha=0.6,
                      cmap='viridis', marker='.')
ax6.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
ax6.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax6.set_title(f'AC Response Map (colored by phase){ds_note}', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label(f'Phase ({theta_unit})', fontsize=10, fontweight='bold')

effective_rate = n_samples / li_time[-1] if li_time[-1] > 0 else 0
fig.suptitle(f'AC Cyclic Voltammetry -- Synchronized Dual Red Pitaya Measurements\n'
             f'{n_samples:,} raw samples -> {n_ds:,} plotted (step={step}) @ {effective_rate:.1f} Hz',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

combined_png = os.path.join(OUTPUT_DIRECTORY, f'accv_combined_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"Saved ACCV-style plot: {combined_png}")

# ============================================================
# STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("ACCV STATISTICS")
print("=" * 60)
print(f"AC Magnitude (R):  {np.mean(li_R):.6f} +/- {np.std(li_R):.6f} V")
print(f"Phase Angle:       {np.mean(li_Theta):.6f} +/- {np.std(li_Theta):.6f} {theta_unit}")
print(f"DC Potential:      {np.mean(dc_V):.6f} +/- {np.std(dc_V):.6f} V")
print(f"\nCorrelations:")
print(f"  R vs DC:     {np.corrcoef(li_R,     dc_V)[0,1]:.4f}")
print(f"  Phase vs DC: {np.corrcoef(li_Theta, dc_V)[0,1]:.4f}")
print("=" * 60)

plt.show()

print("\nCOMPLETE")
print("=" * 60)
