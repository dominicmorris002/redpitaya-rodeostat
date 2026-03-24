"""
Synchronous dual Red Pitaya data acquisition - ACCV Style
Plots styled for AC Cyclic Voltammetry visualization

Press Enter ONCE to launch both acquisitions simultaneously.
Then press Enter a SECOND time when both RPs are connected and ready.

Have a Great Day :)

Dominic Morris

If R and Theta are inversed try running 10 Hz and then 500 Hz to see if data inverts again
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
Output_Directory = 'test_data'
Start_Time_File  = "start_time.txt"

# Only affects the merged combined_results CSV and the ACCV PNG.
# Individual CSVs are always saved at full resolution.
Combined_Max_Points = 10_000

# Set to False to skip the cycle-averaged plot
Run_Cycle_Averaging = True

print("=" * 60)
print("DUAL RED PITAYA SYNCHRONIZED ACQUISITION")
print("=" * 60)

Python_Exe = sys.executable

Lockin_Script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALockInAmplifier.py")
Dc_Script     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ADCMonitorWithACBiasRemover.py")

if not os.path.exists(Lockin_Script):
    print(f"Error: {Lockin_Script} not found!")
    sys.exit(1)
if not os.path.exists(Dc_Script):
    print(f"Error: {Dc_Script} not found!")
    sys.exit(1)

print(f"Lock-in script:  {Lockin_Script}")
print(f"DC script:       {Dc_Script}")
print("=" * 60)

input("\nPress Enter to start both acquisitions...")
print("")

Start_Time = datetime.now() + pd.Timedelta(seconds=2)
with open(Start_Time_File, "w") as F:
    F.write(Start_Time.isoformat())

print(f"Start time: {Start_Time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print("Launching both subprocesses -- watch for both RPs to connect below...")
print("")

Proc_Lockin = subprocess.Popen([Python_Exe, Lockin_Script], stdin=subprocess.PIPE)
Proc_Dc     = subprocess.Popen([Python_Exe, Dc_Script],     stdin=subprocess.PIPE)

input("Press Enter when both RPs are connected and ready to start measurement...")
print("")
Proc_Lockin.stdin.write(b"\n")
Proc_Lockin.stdin.flush()
Proc_Dc.stdin.write(b"\n")
Proc_Dc.stdin.flush()
print("  -> Sent Enter to lock-in")
print("  -> Sent Enter to DC")

print("Waiting for acquisitions to complete...")
Proc_Lockin.wait()
Proc_Dc.wait()
print("\nBoth acquisitions finished")

try:
    os.remove(Start_Time_File)
except Exception:
    pass

time.sleep(0.5)

# ============================================================
# INDEX-BASED MERGING
# ============================================================
try:
    Lockin_Csv = max(glob.glob(os.path.join(Output_Directory, 'lockin_results_*.csv')),  key=os.path.getctime)
    Dc_Csv     = max(glob.glob(os.path.join(Output_Directory, 'dc_voltage_*.csv')),      key=os.path.getctime)
except ValueError:
    print("Error: Could not find output files!")
    sys.exit(1)

print("\n" + "=" * 60)
print("MERGING DATA (INDEX-BASED)")
print("=" * 60)

print("Reading lock-in CSV...")
Lockin_Data = pd.read_csv(Lockin_Csv, comment='#', encoding='latin-1',
                          usecols=lambda C: C in ('Time(s)', 'R(V)', 'Theta(deg)',
                                                   'Theta(\xb0)', 'Theta(rad)', 'X(V)', 'Y(V)'))
print("Reading DC CSV...")
Dc_Data = pd.read_csv(Dc_Csv, comment='#', encoding='latin-1',
                      usecols=lambda C: C in ('Time(s)', 'Voltage(V)'))

N_Lockin  = len(Lockin_Data)
N_Dc      = len(Dc_Data)
N_Samples = min(N_Lockin, N_Dc)

print(f"Lock-in samples: {N_Lockin:,}")
print(f"DC samples:      {N_Dc:,}")
print(f"Common samples:  {N_Samples:,}")
if N_Lockin != N_Dc:
    Diff = abs(N_Lockin - N_Dc)
    print(f"Sample count mismatch: {Diff:,} ({Diff/max(N_Lockin,N_Dc)*100:.1f}%)")

if   'Theta(deg)' in Lockin_Data.columns:  Theta_Col, Theta_Unit = 'Theta(deg)', 'deg'
elif 'Theta(\xb0)' in Lockin_Data.columns: Theta_Col, Theta_Unit = 'Theta(\xb0)', '\xb0'
else:                                       Theta_Col, Theta_Unit = 'Theta(rad)', 'rad'

Step = max(1, N_Samples // Combined_Max_Points)
Idx  = np.arange(0, N_Samples, Step)
N_Ds = len(Idx)
print(f"\nDownsampling: {N_Samples:,} -> {N_Ds:,} pts (step={Step})")

Li_Time  = Lockin_Data['Time(s)'].values[:N_Samples][Idx]
Li_R     = Lockin_Data['R(V)'].values[:N_Samples][Idx]
Li_Theta = Lockin_Data[Theta_Col].values[:N_Samples][Idx]
Li_X     = Lockin_Data['X(V)'].values[:N_Samples][Idx]
Li_Y     = Lockin_Data['Y(V)'].values[:N_Samples][Idx]
Dc_Time  = Dc_Data['Time(s)'].values[:N_Samples][Idx]
Dc_V     = Dc_Data['Voltage(V)'].values[:N_Samples][Idx]

Time_Diff = Li_Time - Dc_Time
print(f"\nTime sync -- mean: {np.mean(Time_Diff)*1000:.3f} ms  max: {np.max(np.abs(Time_Diff))*1000:.3f} ms")
if   np.max(np.abs(Time_Diff)) < 0.1: print("  Excellent sync (< 100 ms)")
elif np.max(np.abs(Time_Diff)) < 0.5: print("  Good sync (< 500 ms)")
else:                                  print("  Poor sync (> 500 ms)")

Timestamp_Str = datetime.now().strftime("%Y%m%d_%H%M%S")
Merged_Csv    = os.path.join(Output_Directory, f'combined_results_{Timestamp_Str}.csv')
pd.DataFrame({
    'Index': np.arange(N_Ds), 'Time_RP1': Li_Time, 'Time_RP2': Dc_Time,
    'R': Li_R, 'Theta': Li_Theta, 'X': Li_X, 'Y': Li_Y, 'DC_Voltage': Dc_V,
}).to_csv(Merged_Csv, index=False)
print(f"\nSaved merged CSV ({N_Ds:,} rows, step={Step}): {Merged_Csv}")

# ============================================================
# PLOT -- 7 panels (time traces + line/scatter of R and Theta vs DC)
# ============================================================
Ds_Note = f' [1:{Step} ds]' if Step > 1 else ''

Fig, Axes = plt.subplots(3, 3, figsize=(20, 15))
Ax_R_T      = Axes[0, 0]
Ax_Th_T     = Axes[0, 1]
Ax_Dc_T     = Axes[0, 2]
Ax_R_Dc     = Axes[1, 0]
Ax_Th_Dc    = Axes[1, 1]
Axes[1, 2].set_visible(False)
Ax_R_Dc_Sc  = Axes[2, 0]
Ax_Th_Dc_Sc = Axes[2, 1]
Axes[2, 2].set_visible(False)

# R vs Time with DC ramp overlay
Ax_R_T.plot(Li_Time, Li_R, 'm-', linewidth=1.0, label='R (lock-in)')
Ax_R_T.set_xlabel('Time (s)',           fontsize=12, fontweight='bold')
Ax_R_T.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold', color='m')
Ax_R_T.set_title(f'R vs Time  |  DC Ramp overlay{Ds_Note}', fontsize=13, fontweight='bold')
Ax_R_T.grid(True, alpha=0.3)
Ax_R_T_Dc = Ax_R_T.twinx()
Ax_R_T_Dc.plot(Li_Time, Dc_V, color='#E65100', linewidth=0.8, alpha=0.7, label='DC Ramp')
Ax_R_T_Dc.set_ylabel('DC Ramp (V)', fontsize=11, color='#E65100')
Ax_R_T_Dc.tick_params(axis='y', labelcolor='#E65100')

# Theta vs Time with DC ramp overlay
Ax_Th_T.plot(Li_Time, Li_Theta, 'c-', linewidth=1.0, label='Theta')
Ax_Th_T.set_xlabel('Time (s)',                      fontsize=12, fontweight='bold')
Ax_Th_T.set_ylabel(f'Phase Angle ({Theta_Unit})',   fontsize=12, fontweight='bold', color='c')
Ax_Th_T.set_title(f'Theta vs Time  |  DC Ramp overlay{Ds_Note}', fontsize=13, fontweight='bold')
Ax_Th_T.grid(True, alpha=0.3)
Ax_Th_T_Dc = Ax_Th_T.twinx()
Ax_Th_T_Dc.plot(Li_Time, Dc_V, color='#E65100', linewidth=0.8, alpha=0.7, label='DC Ramp')
Ax_Th_T_Dc.set_ylabel('DC Ramp (V)', fontsize=11, color='#E65100')
Ax_Th_T_Dc.tick_params(axis='y', labelcolor='#E65100')

# DC vs Time
Ax_Dc_T.plot(Li_Time, Dc_V, 'g-', linewidth=1.0)
Ax_Dc_T.set_xlabel('Time (s)',         fontsize=12, fontweight='bold')
Ax_Dc_T.set_ylabel('DC Potential (V)', fontsize=12, fontweight='bold')
Ax_Dc_T.set_title(f'DC vs Time{Ds_Note}', fontsize=13, fontweight='bold')
Ax_Dc_T.grid(True, alpha=0.3)

# R vs DC (line)
Ax_R_Dc.plot(Dc_V, Li_R, 'b-', linewidth=1.5, alpha=0.8)
Ax_R_Dc.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
Ax_R_Dc.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
Ax_R_Dc.set_title(f'R vs DC Potential -- Line{Ds_Note}', fontsize=13, fontweight='bold')
Ax_R_Dc.grid(True, alpha=0.3)

# Theta vs DC (line)
Ax_Th_Dc.plot(Dc_V, Li_Theta, 'r-', linewidth=1.5, alpha=0.8)
Ax_Th_Dc.set_xlabel('DC Potential (V)',            fontsize=12, fontweight='bold')
Ax_Th_Dc.set_ylabel(f'Phase Angle ({Theta_Unit})', fontsize=12, fontweight='bold')
Ax_Th_Dc.set_title(f'Theta vs DC Potential -- Line{Ds_Note}', fontsize=13, fontweight='bold')
Ax_Th_Dc.grid(True, alpha=0.3)

# R vs DC (scatter)
Ax_R_Dc_Sc.scatter(Dc_V, Li_R, s=3, alpha=0.5, color='b')
Ax_R_Dc_Sc.set_xlabel('DC Potential (V)',    fontsize=12, fontweight='bold')
Ax_R_Dc_Sc.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
Ax_R_Dc_Sc.set_title(f'R vs DC Potential -- Scatter{Ds_Note}', fontsize=13, fontweight='bold')
Ax_R_Dc_Sc.grid(True, alpha=0.3)

# Theta vs DC (scatter)
Ax_Th_Dc_Sc.scatter(Dc_V, Li_Theta, s=3, alpha=0.5, color='r')
Ax_Th_Dc_Sc.set_xlabel('DC Potential (V)',            fontsize=12, fontweight='bold')
Ax_Th_Dc_Sc.set_ylabel(f'Phase Angle ({Theta_Unit})', fontsize=12, fontweight='bold')
Ax_Th_Dc_Sc.set_title(f'Theta vs DC Potential -- Scatter{Ds_Note}', fontsize=13, fontweight='bold')
Ax_Th_Dc_Sc.grid(True, alpha=0.3)

Effective_Rate = N_Samples / Li_Time[-1] if Li_Time[-1] > 0 else 0
Fig.suptitle(f'AC Cyclic Voltammetry -- Synchronized Dual Red Pitaya Measurements\n'
             f'{N_Samples:,} raw samples -> {N_Ds:,} plotted (step={Step}) @ {Effective_Rate:.1f} Hz',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

Combined_Png = os.path.join(Output_Directory, f'accv_combined_{Timestamp_Str}.png')
plt.savefig(Combined_Png, dpi=150, bbox_inches='tight')
print(f"Saved plot: {Combined_Png}")

# ============================================================
# STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("ACCV STATISTICS")
print("=" * 60)
print(f"AC Magnitude (R):  {np.mean(Li_R):.6f} +/- {np.std(Li_R):.6f} V")
print(f"Phase Angle:       {np.mean(Li_Theta):.6f} +/- {np.std(Li_Theta):.6f} {Theta_Unit}")
print(f"DC Potential:      {np.mean(Dc_V):.6f} +/- {np.std(Dc_V):.6f} V")
print(f"\nCorrelations:")
print(f"  R vs DC:     {np.corrcoef(Li_R,     Dc_V)[0,1]:.4f}")
print(f"  Phase vs DC: {np.corrcoef(Li_Theta, Dc_V)[0,1]:.4f}")
print("=" * 60)

plt.show()

# ============================================================
# CYCLE AVERAGING ANALYSIS
# ============================================================
if Run_Cycle_Averaging:
    print("\n" + "=" * 60)
    print("LAUNCHING CYCLE AVERAGING ANALYSIS...")
    print("=" * 60)
    try:
        Cycle_Avg_Script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "postdataplotcreate.py")

        if os.path.exists(Cycle_Avg_Script):
            import importlib.util
            Spec = importlib.util.spec_from_file_location("postdataplotcreate", Cycle_Avg_Script)
            Mod  = importlib.util.module_from_spec(Spec)
            Spec.loader.exec_module(Mod)

            print(f"Loading full merged CSV for cycle analysis: {Merged_Csv}")
            Df_Full    = pd.read_csv(Merged_Csv)
            Dc_Full    = Df_Full['DC_Voltage'].values
            R_Full     = Df_Full['R'].values
            Theta_Full = Df_Full['Theta'].values
            Time_Full  = (Df_Full['Time_RP1'].values - Df_Full['Time_RP1'].values[0]
                          if 'Time_RP1' in Df_Full.columns else None)

            Fig_Ca = Mod.plot_accv_cycle_averaged(
                Dc_Full, R_Full, Theta_Full,
                time_s=Time_Full,
                theta_unit=Theta_Unit,
                timestamp_str=Timestamp_Str,
                output_dir=Output_Directory,
                csv_path=Merged_Csv,
                n_samples=len(Df_Full),
                n_ds=len(Df_Full),
            )
            if Fig_Ca is not None:
                Fig_Ca.show()

        else:
            print(f"  postdataplotcreate.py not found at {Cycle_Avg_Script}")
            Candidates = glob.glob(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "*cycle*avg*.py"))
            if not Candidates:
                Candidates = glob.glob(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "*accv*.py"))
                Candidates = [C for C in Candidates
                              if os.path.abspath(C) != os.path.abspath(__file__)]
            if Candidates:
                Cycle_Avg_Script = Candidates[0]
                print(f"  Found: {Cycle_Avg_Script}")
                subprocess.run([Python_Exe, Cycle_Avg_Script, Merged_Csv])
            else:
                print(f"  Run manually:  python postdataplotcreate.py  {Merged_Csv}")

    except Exception as E:
        print(f"  Cycle averaging failed: {E}")
        print(f"  Run manually:  python postdataplotcreate.py  {Merged_Csv}")

print("\nCOMPLETE")
print("=" * 60)
