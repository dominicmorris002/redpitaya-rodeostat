"""
Synchronous dual Red Pitaya data acquisition - ACCV Style
Plots styled for AC Cyclic Voltammetry visualization
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

# ============================================================
# SYNCHRONIZATION PARAMETERS
# ============================================================
START_DELAY = 10  # seconds
OUTPUT_DIRECTORY = 'test_data'

# Create synchronized start time
START_TIME = datetime.now() + pd.Timedelta(seconds=START_DELAY)
START_TIME_FILE = "start_time.txt"

# Write start time to file for both scripts to read
with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())

print("=" * 60)
print("DUAL RED PITAYA SYNCHRONIZED ACQUISITION")
print("=" * 60)
print(f"Start time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Waiting {(START_TIME - datetime.now()).total_seconds():.1f}s...")
print("\n⚠ IMPORTANT:")
print("  Make sure AUTO_CALIBRATE=False in both scripts!")
print("=" * 60)

# Get Python executable from virtual environment
python_exe = sys.executable

# Paths to scripts
lockin_script = os.path.join(os.path.dirname(__file__), "SimpleLockOn.py")
dc_script = os.path.join(os.path.dirname(__file__), "dc_monitor_with_timestamp.py")

# Check that scripts exist
if not os.path.exists(lockin_script):
    print(f"❌ Error: {lockin_script} not found!")
    sys.exit(1)
if not os.path.exists(dc_script):
    print(f"❌ Error: {dc_script} not found!")
    sys.exit(1)

# Wait until start time
while datetime.now() < START_TIME:
    time.sleep(0.001)

print("\n🚀 Launching both acquisitions...")

# Launch both scripts simultaneously
proc_lockin = subprocess.Popen([python_exe, lockin_script])
proc_dc = subprocess.Popen([python_exe, dc_script])

# Wait for both to finish
print("⏳ Waiting for acquisitions to complete...")
proc_lockin.wait()
proc_dc.wait()
print("\n✓ Both acquisitions finished")

# Clean up sync file
try:
    os.remove(START_TIME_FILE)
except:
    pass

time.sleep(0.5)

# ============================================================
# INDEX-BASED MERGING
# ============================================================

# Find latest files
try:
    lockin_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.csv')), key=os.path.getctime)
    dc_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')), key=os.path.getctime)
    lockin_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.png')), key=os.path.getctime)
    dc_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.png')), key=os.path.getctime)
except ValueError:
    print("❌ Error: Could not find output files!")
    sys.exit(1)

print("\n" + "=" * 60)
print("MERGING DATA (INDEX-BASED)")
print("=" * 60)

# Load CSVs
lockin_data = pd.read_csv(lockin_csv, comment='#')
dc_data = pd.read_csv(dc_csv, comment='#')

n_lockin = len(lockin_data)
n_dc = len(dc_data)

print(f"Lock-in samples: {n_lockin}")
print(f"DC samples: {n_dc}")

# Index-based merge
n_samples = min(n_lockin, n_dc)
print(f"\n✓ Using index-based merge")
print(f"  Common samples: {n_samples}")
if n_lockin != n_dc:
    sample_diff = abs(n_lockin - n_dc)
    sample_diff_percent = (sample_diff / max(n_lockin, n_dc)) * 100
    print(f"  ⚠ Sample count mismatch: {sample_diff} samples ({sample_diff_percent:.1f}%)")

# Create merged dataframe
merged_df = pd.DataFrame({
    'Index': np.arange(n_samples),
    'Time_RP1': lockin_data['Time(s)'].values[:n_samples],
    'Time_RP2': dc_data['Time(s)'].values[:n_samples],
    'R': lockin_data['R(V)'].values[:n_samples],
    'Theta': lockin_data['Theta(rad)'].values[:n_samples],
    'X': lockin_data['X(V)'].values[:n_samples],
    'Y': lockin_data['Y(V)'].values[:n_samples],
    'DC_Voltage': dc_data['Voltage(V)'].values[:n_samples],
})

# Time synchronization quality
time_diff = merged_df['Time_RP1'] - merged_df['Time_RP2']
print(f"\nTime synchronization quality:")
print(f"  Mean time difference: {np.mean(time_diff)*1000:.3f} ms")
print(f"  Max time difference: {np.max(np.abs(time_diff))*1000:.3f} ms")

if np.max(np.abs(time_diff)) < 0.1:
    print(f"  ✓ Excellent sync (< 100 ms)")
elif np.max(np.abs(time_diff)) < 0.5:
    print(f"  ✓ Good sync (< 500 ms)")
else:
    print(f"  ⚠ Poor sync (> 500 ms)")

# Save merged CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
merged_df.to_csv(merged_csv, index=False)
print(f"\n✓ Saved merged CSV: {merged_csv}")

# ============================================================
# CREATE ACCV-STYLE COMBINED PLOT
# ============================================================
fig = plt.figure(figsize=(20, 12))

# Top row: Original plots side by side (wider)
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
img_lockin = imread(lockin_png)
ax1.imshow(img_lockin)
ax1.axis('off')
ax1.set_title('Lock-in Amplifier Results (AC Signal Demodulation)',
              fontsize=14, fontweight='bold', pad=10)

ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
img_dc = imread(dc_png)
ax2.imshow(img_dc)
ax2.axis('off')
ax2.set_title('DC Voltage Monitor Results (Potentiostat Output)',
              fontsize=14, fontweight='bold', pad=10)

# Middle row: ACCV-style plots
# AC Magnitude (R) vs DC Potential
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax3.plot(merged_df['DC_Voltage'], merged_df['R'], 'b-', linewidth=1.5, alpha=0.8)
ax3.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax3.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.set_title('AC Magnitude vs DC Potential', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Phase vs DC Potential
ax4 = plt.subplot2grid((3, 2), (1, 1))
ax4.plot(merged_df['DC_Voltage'], merged_df['Theta'], 'r-', linewidth=1.5, alpha=0.8)
ax4.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Phase Angle (rad)', fontsize=12, fontweight='bold', color='r')
ax4.tick_params(axis='y', labelcolor='r')
ax4.set_title('Phase Angle vs DC Potential', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Bottom row: Time series and combined view
# Time series overlay
ax5 = plt.subplot2grid((3, 2), (2, 0))
ax5_r = ax5.twinx()
ax5.plot(merged_df['Time_RP1'], merged_df['DC_Voltage'], 'g-',
         linewidth=1.0, alpha=0.7, label='DC Potential')
ax5_r.plot(merged_df['Time_RP1'], merged_df['R'], 'b-',
           linewidth=1.0, alpha=0.7, label='AC Magnitude')
ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax5.set_ylabel('DC Potential (V)', fontsize=11, fontweight='bold', color='g')
ax5_r.set_ylabel('AC Magnitude (V)', fontsize=11, fontweight='bold', color='b')
ax5.tick_params(axis='y', labelcolor='g')
ax5_r.tick_params(axis='y', labelcolor='b')
ax5.set_title('Time Series: DC Potential & AC Response', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Combined 3D-style plot (DC vs R vs Phase)
ax6 = plt.subplot2grid((3, 2), (2, 1))
scatter = ax6.scatter(merged_df['DC_Voltage'], merged_df['R'],
                     c=merged_df['Theta'], s=2, alpha=0.6,
                     cmap='viridis', marker='.')
ax6.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax6.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax6.set_title('AC Response Map (colored by phase)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Phase (rad)', fontsize=10, fontweight='bold')

# Overall title
effective_rate = n_samples / merged_df['Time_RP1'].iloc[-1]
fig.suptitle(f'AC Cyclic Voltammetry - Synchronized Dual Red Pitaya Measurements\n'
             f'{n_samples} samples @ {effective_rate:.1f} Hz',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save combined plot
combined_png = os.path.join(OUTPUT_DIRECTORY, f'accv_combined_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved ACCV-style plot: {combined_png}")

# Print statistics
print("\n" + "=" * 60)
print("ACCV STATISTICS")
print("=" * 60)
print(f"AC Magnitude (R):  {np.mean(merged_df['R']):.6f} ± {np.std(merged_df['R']):.6f} V")
print(f"Phase Angle:       {np.mean(merged_df['Theta']):.6f} ± {np.std(merged_df['Theta']):.6f} rad")
print(f"DC Potential:      {np.mean(merged_df['DC_Voltage']):.6f} ± {np.std(merged_df['DC_Voltage']):.6f} V")
print(f"\nCorrelations:")
print(f"  R vs DC:         {np.corrcoef(merged_df['R'], merged_df['DC_Voltage'])[0,1]:.4f}")
print(f"  Phase vs DC:     {np.corrcoef(merged_df['Theta'], merged_df['DC_Voltage'])[0,1]:.4f}")
print("=" * 60)

plt.show()

print("\n✓ COMPLETE")
print("=" * 60)
