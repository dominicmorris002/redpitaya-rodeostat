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

# --- Get venv Python executable ---
python_exe = sys.executable  # ensures it uses the same Python interpreter as this script

# --- Scheduled start time ---
START_TIME = datetime.fromisoformat("2025-12-12 17:13:40.497741")
print("Start time scheduled:", START_TIME)

# Wait until start time
while datetime.now() < START_TIME:
    time.sleep(0.001)  # 1 ms resolution

# --- Paths to scripts ---
lockin_script = os.path.join(os.path.dirname(__file__), "lockin_with_timestamp.py")
dc_script = os.path.join(os.path.dirname(__file__), "dc_monitor_with_timestamp.py")

# --- Launch both scripts using the venv Python ---
proc_lockin = subprocess.Popen([python_exe, lockin_script])
proc_dc = subprocess.Popen([python_exe, dc_script])

# --- Wait for both to finish ---
proc_lockin.wait()
proc_dc.wait()
print("✓ Both acquisitions finished")

# ============================================================
# COMBINE RESULTS
# ============================================================
OUTPUT_DIRECTORY = 'test_data'
SYNC_TOLERANCE = 0.01  # 10 ms

time.sleep(0.5)

# Find latest files
lockin_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.csv')), key=os.path.getctime)
dc_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')), key=os.path.getctime)
lockin_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_results_*.png')), key=os.path.getctime)
dc_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'dc_voltage_*.png')), key=os.path.getctime)

print("\n" + "=" * 60)
print("COMBINING RESULTS")
print("=" * 60)

# Load CSVs
lockin_data = pd.read_csv(lockin_csv, comment='#')
dc_data = pd.read_csv(dc_csv)

print(f"Lock-in samples: {len(lockin_data)}")
print(f"DC samples: {len(dc_data)}")

# Merge by timestamp
lockin_times = lockin_data['AbsoluteTimestamp'].values
dc_times = dc_data['AbsoluteTimestamp'].values

start_time = max(lockin_times[0], dc_times[0])
end_time = min(lockin_times[-1], dc_times[-1])

lockin_mask = (lockin_times >= start_time) & (lockin_times <= end_time)
dc_mask = (dc_times >= start_time) & (dc_times <= end_time)

lockin_overlap = lockin_data[lockin_mask].copy()
dc_overlap = dc_data[dc_mask].copy()

merged_data = []
for idx, row in lockin_overlap.iterrows():
    t_lockin = row['AbsoluteTimestamp']
    time_diffs = np.abs(dc_overlap['AbsoluteTimestamp'].values - t_lockin)
    nearest_idx = np.argmin(time_diffs)
    
    if time_diffs[nearest_idx] <= SYNC_TOLERANCE:
        dc_row = dc_overlap.iloc[nearest_idx]
        merged_data.append({
            'AbsoluteTimestamp_RP1': t_lockin,
            'AbsoluteTimestamp_RP2': dc_row['AbsoluteTimestamp'],
            'RelativeTime_RP1': row['RelativeTime'],
            'RelativeTime_RP2': dc_row['RelativeTime'],
            'R': row['R'],
            'Theta': row['Theta'],
            'X': row['X'],
            'Y': row['Y'],
            'DC_Voltage': dc_row['Voltage'],
            'TimestampDiff': time_diffs[nearest_idx]
        })

merged_df = pd.DataFrame(merged_data)
print(f"✓ Matched {len(merged_df)}/{len(lockin_overlap)} samples")

# Save merged CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
merged_df.to_csv(merged_csv, index=False)
print(f"✓ Merged CSV: {merged_csv}")

# Combine plots into one figure
fig = plt.figure(figsize=(20, 10))

# Load the individual plots
img_lockin = imread(lockin_png)
img_dc = imread(dc_png)

# Top: Lock-in plot
ax1 = plt.subplot(2, 1, 1)
ax1.imshow(img_lockin)
ax1.axis('off')
ax1.set_title('Lock-in Amplifier Results', fontsize=16, fontweight='bold', pad=10)

# Bottom: DC Monitor plot
ax2 = plt.subplot(2, 1, 2)
ax2.imshow(img_dc)
ax2.axis('off')
ax2.set_title('DC Voltage Monitor Results', fontsize=16, fontweight='bold', pad=10)

fig.suptitle('Combined Red Pitaya Measurements', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save combined plot
combined_png = os.path.join(OUTPUT_DIRECTORY, f'combined_plot_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Combined plot: {combined_png}")

plt.show()

print("=" * 60)
print("✓ COMPLETE")
print("=" * 60)
