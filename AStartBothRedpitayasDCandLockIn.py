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
COMBINED_MAX_POINTS = 10_000
# ============================================================

# ============================================================
# CYCLE AVERAGING ANALYSIS  (runs after measurement)
# ============================================================
# Set to False to skip the interactive cycle-averaged plot
RUN_CYCLE_AVERAGING = True
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

# R vs Time  (with DC ramp overlay on secondary y-axis)
ax_R_t.plot(li_time, li_R, 'm-', linewidth=1.0, label='R (lock-in)')
ax_R_t.set_xlabel('Time (s)',           fontsize=12, fontweight='bold')
ax_R_t.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold', color='m')
ax_R_t.set_title(f'R vs Time  |  DC Ramp overlay{ds_note}', fontsize=13, fontweight='bold')
ax_R_t.grid(True, alpha=0.3)
ax_R_t_dc = ax_R_t.twinx()
ax_R_t_dc.plot(li_time, dc_V, color='#E65100', linewidth=0.8, alpha=0.7, label='DC Ramp')
ax_R_t_dc.set_ylabel('DC Ramp (V)', fontsize=11, color='#E65100')
ax_R_t_dc.tick_params(axis='y', labelcolor='#E65100')

# Theta vs Time  (with DC ramp overlay on secondary y-axis)
ax_Th_t.plot(li_time, li_Theta, 'c-', linewidth=1.0, label='Theta')
ax_Th_t.set_xlabel('Time (s)',                    fontsize=12, fontweight='bold')
ax_Th_t.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold', color='c')
ax_Th_t.set_title(f'Theta vs Time  |  DC Ramp overlay{ds_note}', fontsize=13, fontweight='bold')
ax_Th_t.grid(True, alpha=0.3)
ax_Th_t_dc = ax_Th_t.twinx()
ax_Th_t_dc.plot(li_time, dc_V, color='#E65100', linewidth=0.8, alpha=0.7, label='DC Ramp')
ax_Th_t_dc.set_ylabel('DC Ramp (V)', fontsize=11, color='#E65100')
ax_Th_t_dc.tick_params(axis='y', labelcolor='#E65100')

# DC vs Time
ax_DC_t.plot(li_time, dc_V, 'g-', linewidth=1.0)
ax_DC_t.set_xlabel('Time (s)',         fontsize=12, fontweight='bold')
ax_DC_t.set_ylabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax_DC_t.set_title(f'DC vs Time{ds_note}', fontsize=13, fontweight='bold')
ax_DC_t.grid(True, alpha=0.3)

# R vs DC (line)
ax_R_DC.plot(dc_V, li_R, 'b-', linewidth=1.5, alpha=0.8)
ax_R_DC.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
ax_R_DC.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax_R_DC.set_title(f'R vs DC Potential -- Line{ds_note}', fontsize=13, fontweight='bold')
ax_R_DC.grid(True, alpha=0.3)

# Theta vs DC (line)
ax_Th_DC.plot(dc_V, li_Theta, 'r-', linewidth=1.5, alpha=0.8)
ax_Th_DC.set_xlabel('DC Potential (V)',            fontsize=12, fontweight='bold')
ax_Th_DC.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold')
ax_Th_DC.set_title(f'Theta vs DC Potential -- Line{ds_note}', fontsize=13, fontweight='bold')
ax_Th_DC.grid(True, alpha=0.3)

# R vs DC (scatter)
ax_R_DC_sc.scatter(dc_V, li_R, s=3, alpha=0.5, color='b')
ax_R_DC_sc.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
ax_R_DC_sc.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax_R_DC_sc.set_title(f'R vs DC Potential -- Scatter{ds_note}', fontsize=13, fontweight='bold')
ax_R_DC_sc.grid(True, alpha=0.3)

# Theta vs DC (scatter)
ax_Th_DC_sc.scatter(dc_V, li_Theta, s=3, alpha=0.5, color='r')
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

plt.show()

# ============================================================
# CYCLE AVERAGING ANALYSIS  (NEW - runs automatically after)
# ============================================================
if RUN_CYCLE_AVERAGING:
    print("\n" + "=" * 60)
    print("LAUNCHING CYCLE AVERAGING ANALYSIS...")
    print("=" * 60)
    try:
        cycle_avg_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "postdataplotcreate.py")

        if os.path.exists(cycle_avg_script):
            import importlib.util
            spec = importlib.util.spec_from_file_location("postdataplotcreate", cycle_avg_script)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            print(f"Loading full merged CSV for cycle analysis: {merged_csv}")
            df_full   = pd.read_csv(merged_csv)
            dc_full   = df_full['DC_Voltage'].values
            R_full    = df_full['R'].values
            Theta_full = df_full['Theta'].values
            time_full  = (df_full['Time_RP1'].values - df_full['Time_RP1'].values[0]
                          if 'Time_RP1' in df_full.columns else None)

            fig_ca = mod.plot_accv_cycle_averaged(
                dc_full, R_full, Theta_full,
                time_s=time_full,
                theta_unit=theta_unit,
                timestamp_str=timestamp_str,
                output_dir=OUTPUT_DIRECTORY,
                csv_path=merged_csv,
                n_samples=len(df_full),
                n_ds=len(df_full),
            )
            if fig_ca is not None:
                fig_ca.show()

        else:
            print(f"  (postdataplotcreate.py not found at {cycle_avg_script})")
            print(f"  Trying to run via subprocess with merged CSV as argument...")
            candidates = glob.glob(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "*cycle*avg*.py"))
            if not candidates:
                candidates = glob.glob(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "*accv*.py"))
                candidates = [c for c in candidates
                              if os.path.abspath(c) != os.path.abspath(__file__)]
            if candidates:
                cycle_avg_script = candidates[0]
                print(f"  Found: {cycle_avg_script}")
                subprocess.run([python_exe, cycle_avg_script, merged_csv])
            else:
                print("  Could not find cycle averaging script automatically.")
                print(f"  Run manually:  python postdataplotcreate.py  {merged_csv}")

    except Exception as e:
        print(f"  Cycle averaging failed: {e}")
        print(f"  You can still run it manually:")
        print(f"  python postdataplotcreate.py  {merged_csv}")

print("\nCOMPLETE")
print("=" * 60)
