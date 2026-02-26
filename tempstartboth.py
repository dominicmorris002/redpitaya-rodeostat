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
from datetime import datetime
import time
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
# Higher = better cycle averaging resolution (more points per cycle).
# 100_000 gives ~10k pts/cycle for a 10-cycle experiment -- recommended.
COMBINED_MAX_POINTS = 100_000
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

# First Enter: launch subprocesses
input("\nPress Enter to start both acquisitions...")
print("")

START_TIME = datetime.now() + pd.Timedelta(seconds=2)
with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())

print(f"Start time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print("Launching both subprocesses -- watch for both RPs to connect below...")
print("")

proc_lockin = subprocess.Popen([python_exe, lockin_script], stdin=subprocess.PIPE)
proc_dc     = subprocess.Popen([python_exe, dc_script],     stdin=subprocess.PIPE)

# Second Enter: fires both the instant YOU choose
input("Press Enter when both RPs are connected and ready to start measurement...")
print("")
proc_lockin.stdin.write(b"\n")
proc_lockin.stdin.flush()
proc_dc.stdin.write(b"\n")
proc_dc.stdin.flush()
print("  -> Sent Enter to lock-in")
print("  -> Sent Enter to DC")

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
except ValueError:
    print("Error: Could not find output files!")
    sys.exit(1)

print("\n" + "=" * 60)
print("MERGING DATA (INDEX-BASED)")
print("=" * 60)

print("Reading lock-in CSV...")
lockin_data = pd.read_csv(lockin_csv, comment='#', encoding='latin-1',
                          usecols=lambda c: c in ('Time(s)', 'R(V)', 'Theta(deg)',
                                                   'Theta(\xb0)', 'Theta(rad)', 'X(V)', 'Y(V)'))
print("Reading DC CSV...")
dc_data = pd.read_csv(dc_csv, comment='#', encoding='latin-1',
                      usecols=lambda c: c in ('Time(s)', 'Voltage(V)'))

n_lockin = len(lockin_data)
n_dc     = len(dc_data)
n_samples = min(n_lockin, n_dc)

print(f"Lock-in samples: {n_lockin:,}")
print(f"DC samples:      {n_dc:,}")
print(f"Common samples:  {n_samples:,}")
if n_lockin != n_dc:
    diff = abs(n_lockin - n_dc)
    print(f"Sample count mismatch: {diff:,} ({diff/max(n_lockin,n_dc)*100:.1f}%)")

if   'Theta(deg)' in lockin_data.columns:  theta_col, theta_unit = 'Theta(deg)', 'deg'
elif 'Theta(\xb0)' in lockin_data.columns: theta_col, theta_unit = 'Theta(\xb0)', '\xb0'
else:                                       theta_col, theta_unit = 'Theta(rad)', 'rad'

step = max(1, n_samples // COMBINED_MAX_POINTS)
idx  = np.arange(0, n_samples, step)
n_ds = len(idx)
print(f"\nDownsampling: {n_samples:,} -> {n_ds:,} pts (step={step})")

li_time  = lockin_data['Time(s)'].values[:n_samples][idx]
li_R     = lockin_data['R(V)'].values[:n_samples][idx]
li_Theta = lockin_data[theta_col].values[:n_samples][idx]
li_X     = lockin_data['X(V)'].values[:n_samples][idx]
li_Y     = lockin_data['Y(V)'].values[:n_samples][idx]
dc_time  = dc_data['Time(s)'].values[:n_samples][idx]
dc_V     = dc_data['Voltage(V)'].values[:n_samples][idx]

time_diff = li_time - dc_time
print(f"\nTime sync -- mean: {np.mean(time_diff)*1000:.3f} ms  max: {np.max(np.abs(time_diff))*1000:.3f} ms")
if   np.max(np.abs(time_diff)) < 0.1: print("  Excellent sync (< 100 ms)")
elif np.max(np.abs(time_diff)) < 0.5: print("  Good sync (< 500 ms)")
else:                                  print("  Poor sync (> 500 ms)")

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv    = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
pd.DataFrame({
    'Index': np.arange(n_ds), 'Time_RP1': li_time, 'Time_RP2': dc_time,
    'R': li_R, 'Theta': li_Theta, 'X': li_X, 'Y': li_Y, 'DC_Voltage': dc_V,
}).to_csv(merged_csv, index=False)
print(f"\nSaved merged CSV ({n_ds:,} rows, step={step}): {merged_csv}")

# ============================================================
# PLOT -- 7 panels (5 + scatter versions of R vs DC, Theta vs DC)
# ============================================================
ds_note = f' [1:{step} ds]' if step > 1 else ''

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
ax_R_t        = axes[0, 0]
ax_Th_t       = axes[0, 1]
ax_DC_t       = axes[0, 2]
ax_R_DC       = axes[1, 0]
ax_Th_DC      = axes[1, 1]
axes[1, 2].set_visible(False)
ax_R_DC_sc    = axes[2, 0]
ax_Th_DC_sc   = axes[2, 1]
axes[2, 2].set_visible(False)

# R vs Time
ax_R_t.plot(li_time, li_R, 'm-', linewidth=1.0)
ax_R_t.set_xlabel('Time (s)',           fontsize=12, fontweight='bold')
ax_R_t.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax_R_t.set_title(f'R vs Time{ds_note}', fontsize=13, fontweight='bold')
ax_R_t.grid(True, alpha=0.3)

# Theta vs Time
ax_Th_t.plot(li_time, li_Theta, 'c-', linewidth=1.0)
ax_Th_t.set_xlabel('Time (s)',                    fontsize=12, fontweight='bold')
ax_Th_t.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold')
ax_Th_t.set_title(f'Theta vs Time{ds_note}',      fontsize=13, fontweight='bold')
ax_Th_t.grid(True, alpha=0.3)

# DC vs Time
ax_DC_t.plot(li_time, dc_V, 'g-', linewidth=1.0)
ax_DC_t.set_xlabel('Time (s)',         fontsize=12, fontweight='bold')
ax_DC_t.set_ylabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax_DC_t.set_title(f'DC vs Time{ds_note}', fontsize=13, fontweight='bold')
ax_DC_t.grid(True, alpha=0.3)

# R vs DC (line) -- colour by sweep direction using gradient
# Build a simple forward/reverse mask from the DC signal derivative
_dc_diff = np.gradient(dc_V)
_fwd = _dc_diff >= 0   # forward = DC rising
_rev = ~_fwd           # reverse = DC falling

ax_R_DC.plot(dc_V[_fwd], li_R[_fwd], '.', color='#1565C0', markersize=1.5, alpha=0.6, label='Forward (→)')
ax_R_DC.plot(dc_V[_rev], li_R[_rev], '.', color='#B71C1C', markersize=1.5, alpha=0.6, label='Reverse (←)')
ax_R_DC.legend(fontsize=9)
ax_R_DC.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
ax_R_DC.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax_R_DC.set_title(f'R vs DC Potential -- Raw{ds_note}', fontsize=13, fontweight='bold')
ax_R_DC.grid(True, alpha=0.3)

# Theta vs DC (line) -- same sweep colouring
ax_Th_DC.plot(dc_V[_fwd], li_Theta[_fwd], '.', color='#1565C0', markersize=1.5, alpha=0.6, label='Forward (→)')
ax_Th_DC.plot(dc_V[_rev], li_Theta[_rev], '.', color='#B71C1C', markersize=1.5, alpha=0.6, label='Reverse (←)')
ax_Th_DC.legend(fontsize=9)
ax_Th_DC.set_xlabel('DC Potential (V)',            fontsize=12, fontweight='bold')
ax_Th_DC.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold')
ax_Th_DC.set_title(f'Theta vs DC Potential -- Raw{ds_note}', fontsize=13, fontweight='bold')
ax_Th_DC.grid(True, alpha=0.3)

# R vs DC (scatter) -- sweep coloured
ax_R_DC_sc.scatter(dc_V[_fwd], li_R[_fwd], s=3, alpha=0.5, color='#1565C0', label='Forward (→)')
ax_R_DC_sc.scatter(dc_V[_rev], li_R[_rev], s=3, alpha=0.5, color='#B71C1C', label='Reverse (←)')
ax_R_DC_sc.legend(fontsize=9)
ax_R_DC_sc.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
ax_R_DC_sc.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax_R_DC_sc.set_title(f'R vs DC Potential -- Scatter{ds_note}', fontsize=13, fontweight='bold')
ax_R_DC_sc.grid(True, alpha=0.3)

# Theta vs DC (scatter) -- sweep coloured
ax_Th_DC_sc.scatter(dc_V[_fwd], li_Theta[_fwd], s=3, alpha=0.5, color='#1565C0', label='Forward (→)')
ax_Th_DC_sc.scatter(dc_V[_rev], li_Theta[_rev], s=3, alpha=0.5, color='#B71C1C', label='Reverse (←)')
ax_Th_DC_sc.legend(fontsize=9)
ax_Th_DC_sc.set_xlabel('DC Potential (V)',            fontsize=12, fontweight='bold')
ax_Th_DC_sc.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold')
ax_Th_DC_sc.set_title(f'Theta vs DC Potential -- Scatter{ds_note}', fontsize=13, fontweight='bold')
ax_Th_DC_sc.grid(True, alpha=0.3)

effective_rate = n_samples / li_time[-1] if li_time[-1] > 0 else 0
fig.suptitle(f'AC Cyclic Voltammetry -- Synchronized Dual Red Pitaya Measurements\n'
             f'{n_samples:,} raw samples -> {n_ds:,} plotted (step={step}) @ {effective_rate:.1f} Hz',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

combined_png = os.path.join(OUTPUT_DIRECTORY, f'accv_combined_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"Saved plot: {combined_png}")

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

# ============================================================
# CYCLE-AVERAGED ACCV PLOTS
# ============================================================
# Produces a separate crisp figure with forward/reverse sweeps
# averaged across all cycles -- eliminates the "fat line" problem.
# Requires accv_cycle_average.py in the same directory.
try:
    from accv_cycle_average import plot_accv_cycle_averaged
    plot_accv_cycle_averaged(
        dc_v          = dc_V,
        li_R          = li_R,
        li_Theta      = li_Theta,
        theta_unit    = theta_unit,
        timestamp_str = timestamp_str,
        output_dir    = OUTPUT_DIRECTORY,
        ds_note       = ds_note,
        n_samples     = n_samples,
        n_ds          = n_ds,
    )
except ImportError:
    print("\nNote: accv_cycle_average.py not found -- skipping cycle-averaged plots.")
    print("      Place accv_cycle_average.py in the same folder as startboth.py.")

# ============================================================
plt.show()

print("\nCOMPLETE")
print("=" * 60)
