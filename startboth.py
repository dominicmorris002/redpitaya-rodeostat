"""
Synchronous dual Red Pitaya data acquisition - ACCV Style
Better synchronization with subprocess communication
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
START_DELAY = 8  # seconds - longer delay for better sync
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
print(f"Waiting {(START_TIME - datetime.now()).total_seconds():.1f}s for sync...")
print("=" * 60)

# Get Python executable
python_exe = sys.executable

# Paths to scripts
lockin_script = os.path.join(os.path.dirname(__file__), "lockin_with_timestamp.py")
dc_script = os.path.join(os.path.dirname(__file__), "dc_monitor_with_timestamp.py")

# Check scripts exist
if not os.path.exists(lockin_script):
    print(f"❌ Error: {lockin_script} not found!")
    sys.exit(1)
if not os.path.exists(dc_script):
    print(f"❌ Error: {dc_script} not found!")
    sys.exit(1)

# Launch both scripts with output suppression for cleaner logs
print("\n🚀 Launching both acquisitions...")
print("(Script outputs suppressed for clarity)")

proc_lockin = subprocess.Popen(
    [python_exe, lockin_script],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
proc_dc = subprocess.Popen(
    [python_exe, dc_script],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for both to finish
print("⏳ Waiting for acquisitions to complete...")
lockin_out, lockin_err = proc_lockin.communicate()
dc_out, dc_err = proc_dc.communicate()

if proc_lockin.returncode != 0:
    print("❌ Lock-in script failed!")
    print(lockin_err.decode())
    sys.exit(1)
if proc_dc.returncode != 0:
    print("❌ DC script failed!")
    print(dc_err.decode())
    sys.exit(1)

print("✓ Both acquisitions finished")

# Clean up sync file
try:
    os.remove(START_TIME_FILE)
except:
    pass

time.sleep(0.5)

# ============================================================
# SMART DATA MERGING
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
print("MERGING DATA")
print("=" * 60)

# Load CSVs with proper encoding
lockin_data = pd.read_csv(lockin_csv, comment='#', encoding='latin-1')
dc_data = pd.read_csv(dc_csv, comment='#', encoding='latin-1')

n_lockin = len(lockin_data)
n_dc = len(dc_data)

print(f"Lock-in samples: {n_lockin}")
print(f"DC samples: {n_dc}")

# Get time vectors
t_lockin = lockin_data['Time(s)'].values
t_dc = dc_data['Time(s)'].values

# Find common time range
t_start = max(t_lockin[0], t_dc[0])
t_end = min(t_lockin[-1], t_dc[-1])

print(f"\nTime ranges:")
print(f"  Lock-in: {t_lockin[0]:.3f}s to {t_lockin[-1]:.3f}s")
print(f"  DC:      {t_dc[0]:.3f}s to {t_dc[-1]:.3f}s")
print(f"  Overlap: {t_start:.3f}s to {t_end:.3f}s ({t_end - t_start:.3f}s)")

# Trim both datasets to common time range
lockin_mask = (t_lockin >= t_start) & (t_lockin <= t_end)
dc_mask = (t_dc >= t_start) & (t_dc <= t_end)

lockin_trimmed = lockin_data[lockin_mask].copy()
dc_trimmed = dc_data[dc_mask].copy()

n_lockin_trim = len(lockin_trimmed)
n_dc_trim = len(dc_trimmed)

print(f"\nAfter trimming to overlap:")
print(f"  Lock-in: {n_lockin_trim} samples")
print(f"  DC:      {n_dc_trim} samples")

# Downsample to match sample counts
n_samples = min(n_lockin_trim, n_dc_trim)

# Evenly sample from each dataset
lockin_indices = np.linspace(0, n_lockin_trim - 1, n_samples, dtype=int)
dc_indices = np.linspace(0, n_dc_trim - 1, n_samples, dtype=int)

lockin_resampled = lockin_trimmed.iloc[lockin_indices].reset_index(drop=True)
dc_resampled = dc_trimmed.iloc[dc_indices].reset_index(drop=True)

print(f"\nAfter resampling to match:")
print(f"  Common samples: {n_samples}")

# Check column names to handle both rad and degrees
theta_col = 'Theta(°)' if 'Theta(°)' in lockin_resampled.columns else 'Theta(rad)'

# Create merged dataframe
merged_df = pd.DataFrame({
    'Index': np.arange(n_samples),
    'Time_RP1': lockin_resampled['Time(s)'].values,
    'Time_RP2': dc_resampled['Time(s)'].values,
    'Time_Avg': (lockin_resampled['Time(s)'].values + dc_resampled['Time(s)'].values) / 2,
    'R': lockin_resampled['R(V)'].values,
    'Theta': lockin_resampled[theta_col].values,
    'X': lockin_resampled['X(V)'].values,
    'Y': lockin_resampled['Y(V)'].values,
    'DC_Voltage': dc_resampled['Voltage(V)'].values,
})

# Time synchronization quality
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
    print(f"  ⚠ Moderate sync - using averaged time axis for plots")

# Save merged CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
merged_df.to_csv(merged_csv, index=False)
print(f"\n✓ Saved merged CSV: {merged_csv}")

# ============================================================
# CREATE ACCV-STYLE COMBINED PLOT
# ============================================================
fig = plt.figure(figsize=(20, 12))

# Top row: Original plots side by side
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
img_lockin = imread(lockin_png)
ax1.imshow(img_lockin)
ax1.axis('off')
ax1.set_title('Lock-in Amplifier Results',
              fontsize=14, fontweight='bold', pad=10)

ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
img_dc = imread(dc_png)
ax2.imshow(img_dc)
ax2.axis('off')
ax2.set_title('DC Voltage Monitor Results',
              fontsize=14, fontweight='bold', pad=10)

# Middle row: ACCV-style plots
# AC Magnitude vs DC Potential
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax3.plot(merged_df['DC_Voltage'], merged_df['R'], 'b-', linewidth=1.2, alpha=0.7)
ax3.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax3.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.set_title('AC Magnitude vs DC Potential', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Phase vs DC Potential
ax4 = plt.subplot2grid((3, 2), (1, 1))
theta_unit = '°' if 'Theta(°)' in lockin_data.columns else 'rad'
ax4.plot(merged_df['DC_Voltage'], merged_df['Theta'], 'r-', linewidth=1.2, alpha=0.7)
ax4.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax4.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold', color='r')
ax4.tick_params(axis='y', labelcolor='r')
ax4.set_title('Phase Angle vs DC Potential', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Bottom row: Time series and combined view
# Time series overlay (using averaged time)
ax5 = plt.subplot2grid((3, 2), (2, 0))
ax5_r = ax5.twinx()
ax5.plot(merged_df['Time_Avg'], merged_df['DC_Voltage'], 'g-',
         linewidth=1.0, alpha=0.7, label='DC Potential')
ax5_r.plot(merged_df['Time_Avg'], merged_df['R'], 'b-',
           linewidth=1.0, alpha=0.7, label='AC Magnitude')
ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax5.set_ylabel('DC Potential (V)', fontsize=11, fontweight='bold', color='g')
ax5_r.set_ylabel('AC Magnitude (V)', fontsize=11, fontweight='bold', color='b')
ax5.tick_params(axis='y', labelcolor='g')
ax5_r.tick_params(axis='y', labelcolor='b')
ax5.set_title('Time Series: DC Potential & AC Response', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Combined plot with smoothing for better visualization
ax6 = plt.subplot2grid((3, 2), (2, 1))

# Apply light smoothing to reduce noise in visualization
from scipy.ndimage import uniform_filter1d
smooth_window = max(1, n_samples // 1000)  # Smooth over ~0.1% of data
dc_smooth = uniform_filter1d(merged_df['DC_Voltage'], smooth_window)
r_smooth = uniform_filter1d(merged_df['R'], smooth_window)
theta_smooth = uniform_filter1d(merged_df['Theta'], smooth_window)

scatter = ax6.scatter(dc_smooth, r_smooth,
                     c=theta_smooth, s=3, alpha=0.6,
                     cmap='viridis', marker='.')
ax6.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax6.set_ylabel('AC Magnitude R (V)', fontsize=12, fontweight='bold')
ax6.set_title('AC Response Map (colored by phase)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label(f'Phase ({theta_unit})', fontsize=10, fontweight='bold')

# Overall title
effective_rate = n_samples / (t_end - t_start)
fig.suptitle(f'AC Cyclic Voltammetry - Synchronized Dual Red Pitaya\n'
             f'{n_samples} samples @ {effective_rate:.1f} Hz | Time sync: ±{np.max(np.abs(time_diff))*1000:.1f}ms',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save combined plot
combined_png = os.path.join(OUTPUT_DIRECTORY, f'accv_combined_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved ACCV plot: {combined_png}")

# Print statistics
print("\n" + "=" * 60)
print("ACCV STATISTICS")
print("=" * 60)
print(f"AC Magnitude (R):  {np.mean(merged_df['R']):.6f} ± {np.std(merged_df['R']):.6f} V")
print(f"Phase Angle:       {np.mean(merged_df['Theta']):.6f} ± {np.std(merged_df['Theta']):.6f} {theta_unit}")
print(f"DC Potential:      {np.mean(merged_df['DC_Voltage']):.6f} ± {np.std(merged_df['DC_Voltage']):.6f} V")
print(f"\nCorrelations:")
print(f"  R vs DC:         {np.corrcoef(merged_df['R'], merged_df['DC_Voltage'])[0,1]:.4f}")
print(f"  Phase vs DC:     {np.corrcoef(merged_df['Theta'], merged_df['DC_Voltage'])[0,1]:.4f}")
print("=" * 60)

plt.show()

print("\n✓ COMPLETE")
print("=" * 60)
