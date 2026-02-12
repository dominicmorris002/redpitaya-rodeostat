"""
Synchronous dual Red Pitaya data acquisition - DC Ramp Synchronized
Launches X+Ramp and Y+Ramp loggers, then merges data using DC ramp as common reference

Hardware Setup:
- DC Ramp → IN2 on BOTH Red Pitayas
- Lock-in X output → RP1 (reads on iq2)
- Lock-in Y output → RP2 (reads on iq2_2)
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
from scipy import interpolate

# ============================================================
# SYNCHRONIZATION PARAMETERS
# ============================================================
START_DELAY = 5  # seconds
OUTPUT_DIRECTORY = 'test_data'

# DC Ramp scaling (must match what's in the individual scripts)
RAMP_GAIN = -28.93002007
RAMP_OFFSET = -0.016903

# Create synchronized start time
START_TIME = datetime.now() + pd.Timedelta(seconds=START_DELAY)
START_TIME_FILE = "start_time.txt"

# Write start time to file for both scripts to read
with open(START_TIME_FILE, "w") as f:
    f.write(START_TIME.isoformat())

print("=" * 60)
print("DUAL RED PITAYA - DC RAMP SYNCHRONIZED ACQUISITION")
print("=" * 60)
print(f"Start time: {START_TIME.strftime('%Y-%m-%d %H:%M:%S.%f')}")
print(f"Waiting {(START_TIME - datetime.now()).total_seconds():.1f}s...")
print(f"Ramp scaling: {RAMP_GAIN:.4f}x gain, {RAMP_OFFSET:.6f}V offset")
print("=" * 60)

# Get Python executable from virtual environment
python_exe = sys.executable

# Paths to scripts
x_script = os.path.join(os.path.dirname(__file__), "LockXandDC.py")
y_script = os.path.join(os.path.dirname(__file__), "LockYandDC.py")

# Check that scripts exist
if not os.path.exists(x_script):
    print(f"❌ Error: {x_script} not found!")
    sys.exit(1)
if not os.path.exists(y_script):
    print(f"❌ Error: {y_script} not found!")
    sys.exit(1)

# Wait until start time
while datetime.now() < START_TIME:
    time.sleep(0.001)

print("\n🚀 Launching both acquisitions...")

# Launch both scripts simultaneously
proc_x = subprocess.Popen([python_exe, x_script])
proc_y = subprocess.Popen([python_exe, y_script])

# Wait for both to finish
print("⏳ Waiting for acquisitions to complete...")
proc_x.wait()
proc_y.wait()
print("\n✓ Both acquisitions finished")

# Clean up sync file
try:
    os.remove(START_TIME_FILE)
except:
    pass

time.sleep(0.5)

# ============================================================
# DC RAMP-BASED MERGING
# ============================================================

# Find latest files
try:
    x_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_x_ramp_*.csv')), key=os.path.getctime)
    y_csv = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_y_ramp_*.csv')), key=os.path.getctime)
    x_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_x_ramp_*.png')), key=os.path.getctime)
    y_png = max(glob.glob(os.path.join(OUTPUT_DIRECTORY, 'lockin_y_ramp_*.png')), key=os.path.getctime)
except ValueError:
    print("❌ Error: Could not find output files!")
    sys.exit(1)

print("\n" + "=" * 60)
print("MERGING DATA USING DC RAMP REFERENCE")
print("=" * 60)

# Load CSVs
x_data = pd.read_csv(x_csv, comment='#', encoding='latin-1')
y_data = pd.read_csv(y_csv, comment='#', encoding='latin-1')

n_x = len(x_data)
n_y = len(y_data)

print(f"X component samples: {n_x}")
print(f"Y component samples: {n_y}")

# Apply ramp scaling to both
ramp_x = (x_data['DCRamp(V)'].values - RAMP_OFFSET) * RAMP_GAIN
ramp_y = (y_data['DCRamp(V)'].values - RAMP_OFFSET) * RAMP_GAIN

print(f"\nX ramp range: {np.min(ramp_x):.6f} to {np.max(ramp_x):.6f} V")
print(f"Y ramp range: {np.min(ramp_y):.6f} to {np.max(ramp_y):.6f} V")

# Find overlapping ramp region
ramp_min = max(np.min(ramp_x), np.min(ramp_y))
ramp_max = min(np.max(ramp_x), np.max(ramp_y))

print(f"Overlapping ramp range: {ramp_min:.6f} to {ramp_max:.6f} V")

# Filter to overlapping region
mask_x = (ramp_x >= ramp_min) & (ramp_x <= ramp_max)
mask_y = (ramp_y >= ramp_min) & (ramp_y <= ramp_max)

ramp_x_overlap = ramp_x[mask_x]
x_overlap = x_data['X(V)'].values[mask_x]

ramp_y_overlap = ramp_y[mask_y]
y_overlap = y_data['Y(V)'].values[mask_y]

print(f"\nX samples in overlap: {len(ramp_x_overlap)}")
print(f"Y samples in overlap: {len(ramp_y_overlap)}")

# Decide which has fewer samples - use that as reference
if len(ramp_x_overlap) <= len(ramp_y_overlap):
    # Use X ramp as reference, interpolate Y onto it
    print("\n✓ Using X ramp as reference, interpolating Y")
    
    # Sort X by ramp voltage
    sort_idx_x = np.argsort(ramp_x_overlap)
    ramp_ref = ramp_x_overlap[sort_idx_x]
    x_ref = x_overlap[sort_idx_x]
    
    # Sort Y by ramp voltage
    sort_idx_y = np.argsort(ramp_y_overlap)
    ramp_y_sorted = ramp_y_overlap[sort_idx_y]
    y_sorted = y_overlap[sort_idx_y]
    
    # Interpolate Y values onto X's ramp positions
    interp_func = interpolate.interp1d(ramp_y_sorted, y_sorted, 
                                      kind='linear', 
                                      bounds_error=False,
                                      fill_value='extrapolate')
    y_ref = interp_func(ramp_ref)
    
else:
    # Use Y ramp as reference, interpolate X onto it
    print("\n✓ Using Y ramp as reference, interpolating X")
    
    # Sort Y by ramp voltage
    sort_idx_y = np.argsort(ramp_y_overlap)
    ramp_ref = ramp_y_overlap[sort_idx_y]
    y_ref = y_overlap[sort_idx_y]
    
    # Sort X by ramp voltage
    sort_idx_x = np.argsort(ramp_x_overlap)
    ramp_x_sorted = ramp_x_overlap[sort_idx_x]
    x_sorted = x_overlap[sort_idx_x]
    
    # Interpolate X values onto Y's ramp positions
    interp_func = interpolate.interp1d(ramp_x_sorted, x_sorted,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value='extrapolate')
    x_ref = interp_func(ramp_ref)

# Calculate R and Theta
R = np.sqrt(x_ref**2 + y_ref**2)
Theta = np.degrees(np.arctan2(y_ref, x_ref))

n_merged = len(ramp_ref)

print(f"\nMerged data points: {n_merged}")
print(f"\nMerged Statistics:")
print(f"  Mean R: {np.mean(R):.6f} ± {np.std(R):.6f} V")
print(f"  Mean X: {np.mean(x_ref):.6f} ± {np.std(x_ref):.6f} V")
print(f"  Mean Y: {np.mean(y_ref):.6f} ± {np.std(y_ref):.6f} V")
print(f"  Mean Theta: {np.mean(Theta):.3f} ± {np.std(Theta):.3f}°")
print(f"  DC Ramp span: {np.max(ramp_ref) - np.min(ramp_ref):.6f} V")

# Create merged dataframe
merged_df = pd.DataFrame({
    'Index': np.arange(n_merged),
    'DC_Voltage': ramp_ref,
    'R': R,
    'Theta': Theta,
    'X': x_ref,
    'Y': y_ref
})

# Save merged CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv = os.path.join(OUTPUT_DIRECTORY, f'ramp_sync_combined_{timestamp_str}.csv')
merged_df.to_csv(merged_csv, index=False)
print(f"\n✓ Saved merged CSV: {merged_csv}")

# ============================================================
# CREATE ACCV-STYLE COMBINED PLOT
# ============================================================
fig = plt.figure(figsize=(20, 12))

# Top row: Original plots side by side
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
img_x = imread(x_png)
ax1.imshow(img_x)
ax1.axis('off')
ax1.set_title('Lock-in X Component + DC Ramp (RP1)',
              fontsize=14, fontweight='bold', pad=10)

ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
img_y = imread(y_png)
ax2.imshow(img_y)
ax2.axis('off')
ax2.set_title('Lock-in Y Component + DC Ramp (RP2)',
              fontsize=14, fontweight='bold', pad=10)

# Middle row: ACCV-style plots
# AC Magnitude (R) vs DC Potential
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax3.plot(merged_df['DC_Voltage'], merged_df['R']*1e6, 'b-', linewidth=1.5, alpha=0.8)
ax3.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax3.set_ylabel('AC Magnitude R (μA)', fontsize=12, fontweight='bold', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.set_title('AC Magnitude vs DC Potential', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Phase vs DC Potential
ax4 = plt.subplot2grid((3, 2), (1, 1))
ax4.plot(merged_df['DC_Voltage'], merged_df['Theta'], 'r-', linewidth=1.5, alpha=0.8)
ax4.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Phase Angle (°)', fontsize=12, fontweight='bold', color='r')
ax4.tick_params(axis='y', labelcolor='r')
ax4.set_title('Phase Angle vs DC Potential', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Bottom row: IQ plot and combined view
# IQ Plot
ax5 = plt.subplot2grid((3, 2), (2, 0))
ax5.plot(merged_df['X'], merged_df['Y'], 'g-', linewidth=1.0, alpha=0.7)
ax5.plot(np.mean(merged_df['X']), np.mean(merged_df['Y']), 'r+', 
         markersize=15, markeredgewidth=2, label='Mean')
ax5.set_xlabel('X (In-Phase) (V)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Y (Quadrature) (V)', fontsize=12, fontweight='bold')
ax5.set_title('IQ Plot (Demodulated Components)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.axis('equal')

# Combined 3D-style plot (DC vs R vs Phase)
ax6 = plt.subplot2grid((3, 2), (2, 1))
scatter = ax6.scatter(merged_df['DC_Voltage'], merged_df['R']*1e6,
                     c=merged_df['Theta'], s=2, alpha=0.6,
                     cmap='viridis', marker='.')
ax6.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax6.set_ylabel('AC Magnitude R (μA)', fontsize=12, fontweight='bold')
ax6.set_title('AC Response Map (colored by phase)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Phase (°)', fontsize=10, fontweight='bold')

# Overall title
fig.suptitle(f'AC Cyclic Voltammetry - DC Ramp Synchronized Dual Red Pitaya\n'
             f'{n_merged} merged samples',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save combined plot
combined_png = os.path.join(OUTPUT_DIRECTORY, f'ramp_sync_accv_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved ACCV-style plot: {combined_png}")

# Print statistics
print("\n" + "=" * 60)
print("AC CYCLIC VOLTAMMETRY STATISTICS")
print("=" * 60)
print(f"AC Magnitude (R):  {np.mean(merged_df['R'])*1e6:.3f} ± {np.std(merged_df['R'])*1e6:.3f} μA")
print(f"Phase Angle:       {np.mean(merged_df['Theta']):.3f} ± {np.std(merged_df['Theta']):.3f}°")
print(f"X Component:       {np.mean(merged_df['X']):.6f} ± {np.std(merged_df['X']):.6f} V")
print(f"Y Component:       {np.mean(merged_df['Y']):.6f} ± {np.std(merged_df['Y']):.6f} V")
print(f"DC Potential:      {np.mean(merged_df['DC_Voltage']):.6f} ± {np.std(merged_df['DC_Voltage']):.6f} V")
print(f"\nCorrelations:")
print(f"  R vs DC:         {np.corrcoef(merged_df['R'], merged_df['DC_Voltage'])[0,1]:.4f}")
print(f"  Phase vs DC:     {np.corrcoef(merged_df['Theta'], merged_df['DC_Voltage'])[0,1]:.4f}")
print(f"  X vs DC:         {np.corrcoef(merged_df['X'], merged_df['DC_Voltage'])[0,1]:.4f}")
print(f"  Y vs DC:         {np.corrcoef(merged_df['Y'], merged_df['DC_Voltage'])[0,1]:.4f}")
print("=" * 60)

plt.show()

print("\n✓ COMPLETE - DC Ramp Synchronized Acquisition")
print("=" * 60)
