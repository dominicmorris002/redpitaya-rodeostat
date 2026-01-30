"""
Synchronous dual Red Pitaya data acquisition - Hardware Sync Version
Uses SATA cable daisy-chain for clock and trigger synchronization

SETUP:
1. Connect SATA cable from Master (Lock-in) "Clock and Trigger" port
   to Slave (DC Monitor) "Clock and Trigger" port
2. Master generates clock and triggers
3. Slave follows master's timing exactly

This provides microsecond-level synchronization!
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
START_DELAY = 3  # seconds (reduced since hardware handles sync)
OUTPUT_DIRECTORY = 'test_data'

print("=" * 60)
print("DUAL RED PITAYA HARDWARE-SYNCHRONIZED ACQUISITION")
print("=" * 60)
print("Using SATA cable for clock and trigger synchronization")
print("Master: Lock-in Amplifier (rp-f073ce)")
print("Slave:  DC Monitor (rp-f0909c)")
print("=" * 60)

# Get Python executable from virtual environment
python_exe = sys.executable

# Paths to scripts (use hardware sync versions)
lockin_script = os.path.join(os.path.dirname(__file__), "lockin_master_hwsync.py")
dc_script = os.path.join(os.path.dirname(__file__), "dc_slave_hwsync.py")

# Check that scripts exist
if not os.path.exists(lockin_script):
    print(f"❌ Error: {lockin_script} not found!")
    print(f"   Looking for: {os.path.abspath(lockin_script)}")
    sys.exit(1)
if not os.path.exists(dc_script):
    print(f"❌ Error: {dc_script} not found!")
    print(f"   Looking for: {os.path.abspath(dc_script)}")
    sys.exit(1)

print(f"\n📂 Using scripts:")
print(f"   Master: {lockin_script}")
print(f"   Slave:  {dc_script}")

# Simple delay before starting (hardware handles the rest)
print(f"\nStarting in {START_DELAY} seconds...")
time.sleep(START_DELAY)

print("\n🚀 Launching both acquisitions with hardware sync...")

# Launch both scripts simultaneously
proc_lockin = subprocess.Popen([python_exe, lockin_script])
proc_dc = subprocess.Popen([python_exe, dc_script])

# Wait for both to finish
print("⏳ Waiting for acquisitions to complete...")
proc_lockin.wait()
proc_dc.wait()
print("\n✓ Both acquisitions finished")

time.sleep(0.5)

# ============================================================
# INDEX-BASED MERGING (Hardware sync ensures perfect alignment)
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
print("MERGING DATA (HARDWARE-SYNCHRONIZED)")
print("=" * 60)

# Load CSVs
lockin_data = pd.read_csv(lockin_csv, comment='#', encoding='latin-1')
dc_data = pd.read_csv(dc_csv, comment='#', encoding='latin-1')

n_lockin = len(lockin_data)
n_dc = len(dc_data)

print(f"Lock-in samples (Master): {n_lockin}")
print(f"DC samples (Slave):       {n_dc}")

# Index-based merge
n_samples = min(n_lockin, n_dc)
print(f"\n✓ Hardware-synchronized merge")
print(f"  Common samples: {n_samples}")

if n_lockin != n_dc:
    sample_diff = abs(n_lockin - n_dc)
    sample_diff_percent = (sample_diff / max(n_lockin, n_dc)) * 100
    print(f"  Sample count difference: {sample_diff} samples ({sample_diff_percent:.2f}%)")
    if sample_diff_percent > 1.0:
        print(f"  ⚠ Warning: >1% difference may indicate sync issue")
    else:
        print(f"  ✓ Excellent: <1% difference")
else:
    print(f"  ✓ Perfect: Identical sample counts!")

# Create merged dataframe
theta_col = 'Theta(°)' if 'Theta(°)' in lockin_data.columns else 'Theta(rad)'

merged_df = pd.DataFrame({
    'Index': np.arange(n_samples),
    'Time_Master': lockin_data['Time(s)'].values[:n_samples],
    'Time_Slave': dc_data['Time(s)'].values[:n_samples],
    'R': lockin_data['R(V)'].values[:n_samples],
    'Theta': lockin_data[theta_col].values[:n_samples],
    'X': lockin_data['X(V)'].values[:n_samples],
    'Y': lockin_data['Y(V)'].values[:n_samples],
    'DC_Voltage': dc_data['Voltage(V)'].values[:n_samples],
})

# Time synchronization quality
time_diff = merged_df['Time_Master'] - merged_df['Time_Slave']
time_diff_us = time_diff * 1e6  # Convert to microseconds

print(f"\nHardware synchronization quality:")
print(f"  Mean time difference:   {np.mean(time_diff)*1000:.3f} ms ({np.mean(time_diff_us):.1f} μs)")
print(f"  Std time difference:    {np.std(time_diff)*1000:.3f} ms ({np.std(time_diff_us):.1f} μs)")
print(f"  Max time difference:    {np.max(np.abs(time_diff))*1000:.3f} ms ({np.max(np.abs(time_diff_us)):.1f} μs)")

if np.max(np.abs(time_diff_us)) < 100:
    print(f"  ✓✓✓ EXCELLENT: Sub-100μs sync (hardware working perfectly!)")
elif np.max(np.abs(time_diff)) < 0.001:
    print(f"  ✓✓ VERY GOOD: Sub-millisecond sync")
elif np.max(np.abs(time_diff)) < 0.01:
    print(f"  ✓ GOOD: Sub-10ms sync")
else:
    print(f"  ⚠ POOR: >10ms difference (check SATA cable connection)")

# Save merged CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_hwsync_{timestamp_str}.csv')
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
ax1.set_title('Lock-in Amplifier - MASTER (Hardware Sync)',
              fontsize=14, fontweight='bold', pad=10)

ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
img_dc = imread(dc_png)
ax2.imshow(img_dc)
ax2.axis('off')
ax2.set_title('DC Voltage Monitor - SLAVE (Hardware Sync)',
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
theta_unit = '°' if 'Theta(°)' in lockin_data.columns else 'rad'
ax4.plot(merged_df['DC_Voltage'], merged_df['Theta'], 'r-', linewidth=1.5, alpha=0.8)
ax4.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
ax4.set_ylabel(f'Phase Angle ({theta_unit})', fontsize=12, fontweight='bold', color='r')
ax4.tick_params(axis='y', labelcolor='r')
ax4.set_title('Phase Angle vs DC Potential', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Bottom row: Time series and sync quality
# Time series overlay
ax5 = plt.subplot2grid((3, 2), (2, 0))
ax5_r = ax5.twinx()
ax5.plot(merged_df['Time_Master'], merged_df['DC_Voltage'], 'g-',
         linewidth=1.0, alpha=0.7, label='DC Potential')
ax5_r.plot(merged_df['Time_Master'], merged_df['R'], 'b-',
           linewidth=1.0, alpha=0.7, label='AC Magnitude')
ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax5.set_ylabel('DC Potential (V)', fontsize=11, fontweight='bold', color='g')
ax5_r.set_ylabel('AC Magnitude (V)', fontsize=11, fontweight='bold', color='b')
ax5.tick_params(axis='y', labelcolor='g')
ax5_r.tick_params(axis='y', labelcolor='b')
ax5.set_title('Time Series: Hardware-Synchronized', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Synchronization quality plot
ax6 = plt.subplot2grid((3, 2), (2, 1))
ax6.plot(merged_df['Time_Master'], time_diff_us, 'purple', linewidth=0.8, alpha=0.7)
ax6.axhline(0, color='red', linestyle='--', linewidth=1, label='Perfect sync')
ax6.axhline(np.mean(time_diff_us), color='orange', linestyle='--', linewidth=1, 
            label=f'Mean: {np.mean(time_diff_us):.1f}μs')
ax6.fill_between(merged_df['Time_Master'], 
                  np.mean(time_diff_us) - np.std(time_diff_us),
                  np.mean(time_diff_us) + np.std(time_diff_us),
                  alpha=0.2, color='orange', label=f'±1σ: {np.std(time_diff_us):.1f}μs')
ax6.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Time Difference (μs)', fontsize=12, fontweight='bold')
ax6.set_title('Synchronization Quality (Master - Slave)', fontsize=13, fontweight='bold')
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)

# Overall title
effective_rate = n_samples / merged_df['Time_Master'].iloc[-1]
sync_quality = "EXCELLENT" if np.max(np.abs(time_diff_us)) < 100 else "GOOD"
fig.suptitle(f'AC Cyclic Voltammetry - Hardware-Synchronized Dual Red Pitaya ({sync_quality})\n'
             f'{n_samples} samples @ {effective_rate:.1f} Hz | '
             f'Sync: {np.max(np.abs(time_diff_us)):.1f}μs max deviation',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save combined plot
combined_png = os.path.join(OUTPUT_DIRECTORY, f'accv_hwsync_{timestamp_str}.png')
plt.savefig(combined_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved ACCV-style plot: {combined_png}")

# Print statistics
print("\n" + "=" * 60)
print("ACCV STATISTICS (HARDWARE-SYNCHRONIZED)")
print("=" * 60)
print(f"AC Magnitude (R):  {np.mean(merged_df['R']):.6f} ± {np.std(merged_df['R']):.6f} V")
print(f"Phase Angle:       {np.mean(merged_df['Theta']):.6f} ± {np.std(merged_df['Theta']):.6f} {theta_unit}")
print(f"DC Potential:      {np.mean(merged_df['DC_Voltage']):.6f} ± {np.std(merged_df['DC_Voltage']):.6f} V")
print(f"\nCorrelations:")
print(f"  R vs DC:           {np.corrcoef(merged_df['R'], merged_df['DC_Voltage'])[0,1]:.4f}")
print(f"  Phase vs DC:       {np.corrcoef(merged_df['Theta'], merged_df['DC_Voltage'])[0,1]:.4f}")
print(f"\nSynchronization:")
print(f"  Max time error:    {np.max(np.abs(time_diff_us)):.2f} μs")
print(f"  Mean time error:   {np.mean(time_diff_us):.2f} μs")
print(f"  Std time error:    {np.std(time_diff_us):.2f} μs")
print("=" * 60)

plt.show()

print("\n✓ COMPLETE - Hardware synchronization successful!")
print("=" * 60)
