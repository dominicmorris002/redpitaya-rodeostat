"""
Synchronized Dual Red Pitaya Measurement System

Launches lock-in and DC voltage monitor scripts simultaneously,
then combines and visualizes the results.

This script handles synchronization internally - no changes needed to other files.
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
import tempfile
import shutil

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIRECTORY = 'test_data'
SYNC_TOLERANCE = 0.01  # 10 ms tolerance for timestamp matching
START_DELAY = 2.0  # seconds from now to start both measurements

# ============================================================
# SETUP
# ============================================================

# Get venv Python executable
python_exe = sys.executable

# Calculate start time
START_TIME = datetime.now() + pd.Timedelta(seconds=START_DELAY)
print("=" * 60)
print("SYNCHRONIZED DUAL RED PITAYA MEASUREMENT")
print("=" * 60)
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Start time scheduled: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Waiting {START_DELAY} seconds...")
print("=" * 60)

# Write start time to file for lock-in script to read
START_TIME_FILE = "start_time.txt"
with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())
print(f"✓ Wrote start time to {START_TIME_FILE}")

# Paths to scripts
lockin_script = os.path.join(os.path.dirname(__file__), "lockin_with_timestamp.py")
dc_script_orig = os.path.join(os.path.dirname(__file__), "dc_monitor_with_timestamp.py")

# Verify scripts exist
if not os.path.exists(lockin_script):
    print(f"ERROR: Lock-in script not found: {lockin_script}")
    sys.exit(1)
if not os.path.exists(dc_script_orig):
    print(f"ERROR: DC monitor script not found: {dc_script_orig}")
    sys.exit(1)

# ============================================================
# CREATE MODIFIED DC SCRIPT WITH SYNCHRONIZATION
# ============================================================
# Read original DC script
with open(dc_script_orig, 'r') as f:
    dc_script_content = f.read()

# Add synchronization code after imports
sync_code = f'''
# ============================================================
# SYNCHRONIZATION CODE (added by launcher)
# ============================================================
START_TIME_FILE = "start_time.txt"

with open(START_TIME_FILE, "r") as f:
    START_TIME = datetime.fromisoformat(f.read().strip())

print(f"Synchronized start time: {{START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}}")
print("Waiting for start time...")

while datetime.now() < START_TIME:
    time.sleep(0.001)

print("✓ Starting DC acquisition NOW")
# ============================================================

'''

# Find where to insert (after imports, before class definition)
class_pos = dc_script_content.find('class RedPitaya:')
if class_pos == -1:
    print("ERROR: Could not find 'class RedPitaya:' in DC script")
    sys.exit(1)

# Insert sync code before the class
modified_dc_content = dc_script_content[:class_pos] + sync_code + dc_script_content[class_pos:]

# Write to temporary file
temp_dir = tempfile.gettempdir()
dc_script_temp = os.path.join(temp_dir, "dc_monitor_synced_temp.py")
with open(dc_script_temp, 'w') as f:
    f.write(modified_dc_content)

print(f"✓ Created synchronized DC script: {dc_script_temp}")

# Wait until start time
while datetime.now() < START_TIME:
    time.sleep(0.001)

# ============================================================
# LAUNCH BOTH SCRIPTS
# ============================================================
print("\n" + "=" * 60)
print("LAUNCHING MEASUREMENTS")
print("=" * 60)

proc_lockin = subprocess.Popen([python_exe, lockin_script])
print("✓ Launched lock-in amplifier script")

proc_dc = subprocess.Popen([python_exe, dc_script_temp])
print("✓ Launched DC monitor script (synchronized)")

# Wait for both to finish
print("\nWaiting for measurements to complete...")
proc_lockin.wait()
print("✓ Lock-in measurement finished")

proc_dc.wait()
print("✓ DC monitor measurement finished")

# Clean up temp file
try:
    os.remove(dc_script_temp)
except:
    pass

print("\n" + "=" * 60)
print("PROCESSING RESULTS")
print("=" * 60)

# Give file system a moment to finish writing
time.sleep(0.5)

# ============================================================
# FIND LATEST FILES
# ============================================================
try:
    lockin_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.csv')), 
                     key=os.path.getctime)
    dc_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')), 
                 key=os.path.getctime)
    lockin_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.png')), 
                     key=os.path.getctime)
    dc_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.png')), 
                 key=os.path.getctime)
    
    print(f"✓ Found lock-in CSV: {os.path.basename(lockin_csv)}")
    print(f"✓ Found DC CSV: {os.path.basename(dc_csv)}")
    print(f"✓ Found lock-in plot: {os.path.basename(lockin_png)}")
    print(f"✓ Found DC plot: {os.path.basename(dc_png)}")
except ValueError as e:
    print(f"ERROR: Could not find output files in {OUTPUT_DIRECTORY}")
    print("Make sure both scripts completed successfully")
    sys.exit(1)

# ============================================================
# LOAD AND MERGE DATA
# ============================================================
print("\n" + "=" * 60)
print("MERGING DATASETS")
print("=" * 60)

# Load CSVs
lockin_data = pd.read_csv(lockin_csv, comment='#')
dc_data = pd.read_csv(dc_csv)

print(f"Lock-in samples: {len(lockin_data):,}")
print(f"DC monitor samples: {len(dc_data):,}")

# Get timestamp columns
lockin_times = lockin_data['AbsoluteTimestamp'].values
dc_times = dc_data['AbsoluteTimestamp'].values

# Find overlapping time range
start_time = max(lockin_times[0], dc_times[0])
end_time = min(lockin_times[-1], dc_times[-1])
overlap_duration = end_time - start_time

print(f"\nTime overlap: {overlap_duration:.3f} seconds")
print(f"Start: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S.%f')}")
print(f"End: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')}")

# Filter to overlapping region
lockin_mask = (lockin_times >= start_time) & (lockin_times <= end_time)
dc_mask = (dc_times >= start_time) & (dc_times <= end_time)

lockin_overlap = lockin_data[lockin_mask].copy()
dc_overlap = dc_data[dc_mask].copy()

print(f"\nOverlapping lock-in samples: {len(lockin_overlap):,}")
print(f"Overlapping DC samples: {len(dc_overlap):,}")

# Merge datasets by matching timestamps
print(f"\nMatching timestamps (tolerance: {SYNC_TOLERANCE*1000:.1f} ms)...")
merged_data = []

for idx, row in lockin_overlap.iterrows():
    t_lockin = row['AbsoluteTimestamp']
    
    # Find nearest DC timestamp
    time_diffs = np.abs(dc_overlap['AbsoluteTimestamp'].values - t_lockin)
    nearest_idx = np.argmin(time_diffs)
    
    # Only merge if within tolerance
    if time_diffs[nearest_idx] <= SYNC_TOLERANCE:
        dc_row = dc_overlap.iloc[nearest_idx]
        merged_data.append({
            'AbsoluteTimestamp_Lockin': t_lockin,
            'AbsoluteTimestamp_DC': dc_row['AbsoluteTimestamp'],
            'RelativeTime_Lockin': row['RelativeTime'],
            'RelativeTime_DC': dc_row['RelativeTime'],
            'R': row['R'],
            'Theta': row['Theta'],
            'X': row['X'],
            'Y': row['Y'],
            'DC_Voltage': dc_row['Voltage'],
            'TimestampDiff_ms': time_diffs[nearest_idx] * 1000
        })

merged_df = pd.DataFrame(merged_data)

print(f"✓ Successfully matched {len(merged_df):,} samples")
print(f"  Match rate: {len(merged_df)/len(lockin_overlap)*100:.1f}%")
print(f"  Mean timestamp difference: {merged_df['TimestampDiff_ms'].mean():.3f} ms")
print(f"  Max timestamp difference: {merged_df['TimestampDiff_ms'].max():.3f} ms")

# ============================================================
# SAVE MERGED DATA
# ============================================================
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
merged_df.to_csv(merged_csv, index=False, float_format='%.10f')
print(f"\n✓ Saved merged CSV: {merged_csv}")

# ============================================================
# CREATE COMBINED VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("CREATING COMBINED VISUALIZATION")
print("=" * 60)

fig = plt.figure(figsize=(20, 10))

# Load the individual plots
img_lockin = imread(lockin_png)
img_dc = imread(dc_png)

# Top: Lock-in plot (takes up 60% of height)
ax1 = plt.subplot(2, 1, 1)
ax1.imshow(img_lockin)
ax1.axis('off')
ax1.set_title('Lock-in Amplifier Results', fontsize=16, fontweight='bold', pad=10)

# Bottom: DC Monitor plot (takes up 40% of height)
ax2 = plt.subplot(2, 1, 2)
ax2.imshow(img_dc)
ax2.axis('off')
ax2.set_title('DC Voltage Monitor Results', fontsize=16, fontweight='bold', pad=10)

# Overall title with sync info
sync_info = f'Synchronized Measurements | {len(merged_df):,} matched samples | {overlap_duration:.2f}s overlap'
fig.suptitle(sync_info, fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save combined plot
combined_png = os.path.join(OUTPUT_DIRECTORY, f'combined_plot_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved combined plot: {combined_png}")

# ============================================================
# DISPLAY SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print("\nLock-in Amplifier:")
print(f"  Mean R: {merged_df['R'].mean():.6f} ± {merged_df['R'].std():.6f} V")
print(f"  Mean X: {merged_df['X'].mean():.6f} ± {merged_df['X'].std():.6f} V")
print(f"  Mean Y: {merged_df['Y'].mean():.6f} ± {merged_df['Y'].std():.6f} V")
print(f"  Mean Theta: {merged_df['Theta'].mean():.6f} ± {merged_df['Theta'].std():.6f} rad")

print("\nDC Voltage Monitor:")
print(f"  Mean Voltage: {merged_df['DC_Voltage'].mean():.6f} ± {merged_df['DC_Voltage'].std():.6f} V")

print("\nSynchronization:")
print(f"  Matched samples: {len(merged_df):,}")
print(f"  Time span: {merged_df['RelativeTime_Lockin'].max():.3f} seconds")
print(f"  Timestamp sync quality: {merged_df['TimestampDiff_ms'].mean():.3f} ± {merged_df['TimestampDiff_ms'].std():.3f} ms")

# ============================================================
# SHOW PLOTS
# ============================================================
plt.show()

print("\n" + "=" * 60)
print("✓ COMPLETE")
print("=" * 60)
print(f"\nOutput files saved in: {OUTPUT_DIRECTORY}/")
print(f"  - Merged data: {os.path.basename(merged_csv)}")
print(f"  - Combined plot: {os.path.basename(combined_png)}")
print(f"  - Lock-in plot: {os.path.basename(lockin_png)}")
print(f"  - DC plot: {os.path.basename(dc_png)}")
print("=" * 60)
