"""
Synchronous dual Red Pitaya data acquisition
Uses index-based synchronization instead of timestamps for perfect alignment
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
# Create synchronized start time
START_TIME = datetime.now() + pd.Timedelta(seconds=5)  # 5 seconds from now
START_TIME_FILE = "start_time.txt"

# Write start time to file for both scripts to read
with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())

print("=" * 60)
print("DUAL RED PITAYA SYNCHRONIZED ACQUISITION")
print("=" * 60)
print(f"Start time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Waiting {(START_TIME - datetime.now()).total_seconds():.1f}s...")
print("=" * 60)

# Get venv Python executable
python_exe = sys.executable

# Paths to scripts
lockin_script = os.path.join(os.path.dirname(__file__), "SimpleLockOn.py")
dc_script = os.path.join(os.path.dirname(__file__), "dc_monitor_with_timestamp.py")

# Wait until just before start time
while datetime.now() < START_TIME:
    time.sleep(0.001)

# Launch both scripts simultaneously
proc_lockin = subprocess.Popen([python_exe, lockin_script])
proc_dc = subprocess.Popen([python_exe, dc_script])

# Wait for both to finish
proc_lockin.wait()
proc_dc.wait()
print("\n✓ Both acquisitions finished")

# ============================================================
# INDEX-BASED MERGING
# ============================================================
OUTPUT_DIRECTORY = 'test_data'

time.sleep(0.5)

# Find latest files
lockin_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.csv')), key=os.path.getctime)
dc_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')), key=os.path.getctime)
lockin_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.png')), key=os.path.getctime)
dc_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.png')), key=os.path.getctime)

print("\n" + "=" * 60)
print("MERGING DATA (INDEX-BASED)")
print("=" * 60)

# Load CSVs
lockin_data = pd.read_csv(lockin_csv, comment='#')
dc_data = pd.read_csv(dc_csv)

n_lockin = len(lockin_data)
n_dc = len(dc_data)

print(f"Lock-in samples: {n_lockin}")
print(f"DC samples: {n_dc}")

# Index-based merge: use the shorter length
n_samples = min(n_lockin, n_dc)

print(f"\n✓ Using index-based merge")
print(f"  Common samples: {n_samples}")
if n_lockin != n_dc:
    print(f"  ⚠ Sample count mismatch: {abs(n_lockin - n_dc)} samples discarded")
    print(f"    This is normal due to slight timing differences in scope.single() calls")

# Create merged dataframe by index
merged_df = pd.DataFrame({
    'Index': np.arange(n_samples),
    'Time_RP1': lockin_data['Time(s)'].values[:n_samples],
    'Time_RP2': dc_data['RelativeTime'].values[:n_samples],
    'R': lockin_data['R(V)'].values[:n_samples],
    'Theta': lockin_data['Theta(rad)'].values[:n_samples],
    'X': lockin_data['X(V)'].values[:n_samples],
    'Y': lockin_data['Y(V)'].values[:n_samples],
    'DC_Voltage': dc_data['Voltage'].values[:n_samples],
})

# Calculate time synchronization quality
time_diff = merged_df['Time_RP1'] - merged_df['Time_RP2']
print(f"\nTime synchronization quality:")
print(f"  Mean time difference: {np.mean(time_diff)*1000:.3f} ms")
print(f"  Std time difference: {np.std(time_diff)*1000:.3f} ms")
print(f"  Max time difference: {np.max(np.abs(time_diff))*1000:.3f} ms")

if np.max(np.abs(time_diff)) < 0.1:
    print(f"  ✓ Excellent sync (< 100 ms)")
elif np.max(np.abs(time_diff)) < 0.5:
    print(f"  ✓ Good sync (< 500 ms)")
else:
    print(f"  ⚠ Poor sync (> 500 ms) - check system clocks")

# Save merged CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
merged_df.to_csv(merged_csv, index=False)
print(f"\n✓ Saved merged CSV: {merged_csv}")

# ============================================================
# CREATE COMBINED PLOT
# ============================================================
fig = plt.figure(figsize=(20, 12))

# Load the individual plots
img_lockin = imread(lockin_png)
img_dc = imread(dc_png)

# Top: Lock-in plot
ax1 = plt.subplot(3, 1, 1)
ax1.imshow(img_lockin)
ax1.axis('off')
ax1.set_title('Lock-in Amplifier Results', fontsize=16, fontweight='bold', pad=10)

# Middle: DC Monitor plot
ax2 = plt.subplot(3, 1, 2)
ax2.imshow(img_dc)
ax2.axis('off')
ax2.set_title('DC Voltage Monitor Results', fontsize=16, fontweight='bold', pad=10)

# Bottom: Correlation plots
ax3 = plt.subplot(3, 3, 7)
ax3.plot(merged_df['Index'], merged_df['R'], 'b-', linewidth=0.5, label='Lock-in R')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('R (V)', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.grid(True, alpha=0.3)

ax3_twin = ax3.twinx()
ax3_twin.plot(merged_df['Index'], merged_df['DC_Voltage'], 'r-', linewidth=0.5, label='DC Voltage')
ax3_twin.set_ylabel('DC Voltage (V)', color='r')
ax3_twin.tick_params(axis='y', labelcolor='r')
ax3.set_title('Lock-in R vs DC Voltage (Time Series)')

# DC vs R scatter
ax4 = plt.subplot(3, 3, 8)
ax4.scatter(merged_df['R'], merged_df['DC_Voltage'], s=1, alpha=0.5, c='green')
ax4.set_xlabel('Lock-in R (V)')
ax4.set_ylabel('DC Voltage (V)')
ax4.set_title('DC Voltage vs Lock-in R')
ax4.grid(True, alpha=0.3)

# Time sync quality
ax5 = plt.subplot(3, 3, 9)
ax5.plot(merged_df['Index'], time_diff * 1000, 'purple', linewidth=0.5)
ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Sample Index')
ax5.set_ylabel('Time Difference (ms)')
ax5.set_title('Synchronization Quality')
ax5.grid(True, alpha=0.3)
ax5.set_ylim(-100, 100)  # ±100 ms range

fig.suptitle(f'Synchronized Dual Red Pitaya Measurements\n'
             f'{n_samples} samples @ {1/np.mean(np.diff(merged_df["Time_RP1"])):.1f} Hz',
             fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save combined plot
combined_png = os.path.join(OUTPUT_DIRECTORY, f'combined_plot_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved combined plot: {combined_png}")

print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)
print(f"Lock-in R:    {np.mean(merged_df['R']):.6f} ± {np.std(merged_df['R']):.6f} V")
print(f"DC Voltage:   {np.mean(merged_df['DC_Voltage']):.6f} ± {np.std(merged_df['DC_Voltage']):.6f} V")
print(f"Correlation:  {np.corrcoef(merged_df['R'], merged_df['DC_Voltage'])[0,1]:.4f}")
print("=" * 60)

plt.show()

print("✓ COMPLETE")
print("=" * 60)
