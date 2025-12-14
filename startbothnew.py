"""
Synchronized Red Pitaya Lock-In + DC Monitor with Combined Output

Launches both instruments simultaneously and combines:
- Plots into a single comprehensive figure
- CSV data merged by timestamp alignment
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
# CONFIGURATION
# ============================================================
# Scheduled start time (set to None for immediate start)
START_TIME = None  # datetime.fromisoformat("2025-12-12 17:13:40.497741")

# Output directory (should match both scripts)
OUTPUT_DIRECTORY = 'test_data'

# Timestamp alignment tolerance (seconds)
# Samples within this window will be considered synchronized
SYNC_TOLERANCE = 0.01  # 10 ms tolerance
# ============================================================

def wait_until_start_time(start_time):
    """Wait until scheduled start time"""
    if start_time is None:
        print("Starting immediately...")
        return
    
    print(f"Start time scheduled: {start_time}")
    while datetime.now() < start_time:
        time.sleep(0.001)  # 1 ms resolution
    print(f"✓ Starting at: {datetime.now()}")

def find_latest_files(output_dir, pattern):
    """Find the most recent files matching pattern"""
    files = glob.glob(os.path.join(output_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_csv_data(filepath):
    """Load CSV data with metadata handling"""
    # Check if file has metadata header
    with open(filepath, 'r') as f:
        first_line = f.readline()
    
    if first_line.startswith('#'):
        # Has metadata, skip lines starting with #
        data = pd.read_csv(filepath, comment='#')
    else:
        data = pd.read_csv(filepath)
    
    return data

def merge_data_by_timestamp(lockin_data, dc_data, tolerance=0.01):
    """
    Merge lock-in and DC data based on timestamp alignment
    
    Uses nearest-neighbor matching within tolerance window
    """
    print("\nMerging data by timestamp...")
    print(f"Lock-in samples: {len(lockin_data)}")
    print(f"DC samples: {len(dc_data)}")
    print(f"Sync tolerance: {tolerance * 1000:.1f} ms")
    
    # Use AbsoluteTimestamp for alignment
    lockin_times = lockin_data['AbsoluteTimestamp'].values
    dc_times = dc_data['AbsoluteTimestamp'].values
    
    # Find overlapping time range
    start_time = max(lockin_times[0], dc_times[0])
    end_time = min(lockin_times[-1], dc_times[-1])
    
    print(f"Overlap period: {end_time - start_time:.3f} seconds")
    
    # Filter to overlapping region
    lockin_mask = (lockin_times >= start_time) & (lockin_times <= end_time)
    dc_mask = (dc_times >= start_time) & (dc_times <= end_time)
    
    lockin_overlap = lockin_data[lockin_mask].copy()
    dc_overlap = dc_data[dc_mask].copy()
    
    # For each lock-in sample, find nearest DC sample
    merged_data = []
    matched_count = 0
    
    for idx, row in lockin_overlap.iterrows():
        t_lockin = row['AbsoluteTimestamp']
        
        # Find nearest DC sample
        time_diffs = np.abs(dc_overlap['AbsoluteTimestamp'].values - t_lockin)
        nearest_idx = np.argmin(time_diffs)
        time_diff = time_diffs[nearest_idx]
        
        if time_diff <= tolerance:
            dc_row = dc_overlap.iloc[nearest_idx]
            
            merged_row = {
                'AbsoluteTimestamp': t_lockin,
                'RelativeTime': row['RelativeTime'],
                'R': row['R'],
                'Theta': row['Theta'],
                'X': row['X'],
                'Y': row['Y'],
                'DC_Voltage': dc_row['Voltage'],
                'TimestampDiff': time_diff
            }
            merged_data.append(merged_row)
            matched_count += 1
    
    merged_df = pd.DataFrame(merged_data)
    
    match_rate = (matched_count / len(lockin_overlap)) * 100
    print(f"✓ Matched {matched_count}/{len(lockin_overlap)} samples ({match_rate:.1f}%)")
    
    if match_rate < 80:
        print(f"⚠ WARNING: Low match rate ({match_rate:.1f}%). Check timing synchronization.")
    
    avg_time_diff = merged_df['TimestampDiff'].mean() * 1000
    max_time_diff = merged_df['TimestampDiff'].max() * 1000
    print(f"Average time difference: {avg_time_diff:.3f} ms")
    print(f"Maximum time difference: {max_time_diff:.3f} ms")
    
    return merged_df

def create_combined_plot(merged_df, lockin_file, dc_file):
    """Create comprehensive combined plot"""
    
    fig = plt.figure(figsize=(18, 12))
    
    t = merged_df['RelativeTime'].values
    X = merged_df['X'].values
    Y = merged_df['Y'].values
    R = merged_df['R'].values
    Theta = merged_df['Theta'].values
    DC_V = merged_df['DC_Voltage'].values
    
    # 1. Lock-in X (In-phase)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, X, 'b-', linewidth=0.5)
    ax1.axhline(np.mean(X), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(X):.4f}V')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X (V)')
    ax1.set_title('Lock-in: In-phase (X)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Lock-in Y (Quadrature)
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, Y, 'r-', linewidth=0.5)
    ax2.axhline(np.mean(Y), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(Y):.4f}V')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y (V)')
    ax2.set_title('Lock-in: Quadrature (Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Lock-in R (Magnitude)
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(t, R, 'm-', linewidth=0.5)
    ax3.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(R):.4f}V')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('R (V)')
    ax3.set_title('Lock-in: Magnitude (R)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Lock-in Theta (Phase)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(t, Theta, 'c-', linewidth=0.5)
    ax4.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(Theta):.4f} rad')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Theta (rad)')
    ax4.set_title('Lock-in: Phase (Theta)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. DC Voltage
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(t, DC_V, 'g-', linewidth=0.5)
    ax5.axhline(np.mean(DC_V), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(DC_V):.6f}V')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Voltage (V)')
    ax5.set_title('DC Voltage Monitor')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. X vs Y (IQ plot)
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(X, Y, 'g.', markersize=1, alpha=0.5)
    ax6.plot(np.mean(X), np.mean(Y), 'r+', markersize=15,
             markeredgewidth=2, label='Mean')
    ax6.set_xlabel('X (V)')
    ax6.set_ylabel('Y (V)')
    ax6.set_title('Lock-in: IQ Plot (X vs Y)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    # 7. R vs DC Voltage
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(DC_V, R, c=t, cmap='viridis', s=1, alpha=0.6)
    ax7.set_xlabel('DC Voltage (V)')
    ax7.set_ylabel('Lock-in R (V)')
    ax7.set_title('R vs DC Voltage (colored by time)')
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Time (s)')
    ax7.grid(True, alpha=0.3)
    
    # 8. All signals overlaid (normalized)
    ax8 = plt.subplot(3, 3, 8)
    # Normalize each signal to [0, 1] for comparison
    R_norm = (R - np.min(R)) / (np.max(R) - np.min(R) + 1e-9)
    DC_norm = (DC_V - np.min(DC_V)) / (np.max(DC_V) - np.min(DC_V) + 1e-9)
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-9)
    
    ax8.plot(t, R_norm, 'm-', linewidth=1, alpha=0.7, label='R (normalized)')
    ax8.plot(t, DC_norm, 'g-', linewidth=1, alpha=0.7, label='DC (normalized)')
    ax8.plot(t, X_norm, 'b-', linewidth=1, alpha=0.7, label='X (normalized)')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Normalized Value')
    ax8.set_title('All Signals Overlaid (Normalized)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Timestamp synchronization quality
    ax9 = plt.subplot(3, 3, 9)
    time_diffs_ms = merged_df['TimestampDiff'].values * 1000
    ax9.hist(time_diffs_ms, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax9.axvline(np.mean(time_diffs_ms), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(time_diffs_ms):.3f} ms')
    ax9.set_xlabel('Timestamp Difference (ms)')
    ax9.set_ylabel('Count')
    ax9.set_title('Synchronization Quality')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Add overall title with file info
    fig.suptitle('Combined Lock-in + DC Monitor Results\n' + 
                 f'Lock-in: {os.path.basename(lockin_file)} | DC: {os.path.basename(dc_file)}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

def main():
    # Get venv Python executable
    python_exe = sys.executable
    
    # Wait until start time
    wait_until_start_time(START_TIME)
    
    # Paths to scripts
    script_dir = os.path.dirname(__file__)
    lockin_script = os.path.join(script_dir, "lockin_with_timestamp.py")
    dc_script = os.path.join(script_dir, "dc_monitor_with_timestamp.py")
    
    # Check scripts exist
    if not os.path.exists(lockin_script):
        print(f"ERROR: Lock-in script not found: {lockin_script}")
        return
    if not os.path.exists(dc_script):
        print(f"ERROR: DC monitor script not found: {dc_script}")
        return
    
    print("=" * 60)
    print("SYNCHRONIZED ACQUISITION")
    print("=" * 60)
    print(f"Launch time: {datetime.now()}")
    print("=" * 60)
    
    # Launch both scripts
    print("\n🚀 Launching lock-in amplifier...")
    proc_lockin = subprocess.Popen([python_exe, lockin_script])
    
    print("🚀 Launching DC monitor...")
    proc_dc = subprocess.Popen([python_exe, dc_script])
    
    # Wait for both to finish
    print("\n⏳ Waiting for acquisitions to complete...")
    proc_lockin.wait()
    print("✓ Lock-in acquisition finished")
    
    proc_dc.wait()
    print("✓ DC monitor acquisition finished")
    
    print("\n" + "=" * 60)
    print("COMBINING RESULTS")
    print("=" * 60)
    
    # Find most recent output files
    time.sleep(0.5)  # Brief wait to ensure files are written
    
    lockin_csv = find_latest_files(OUTPUT_DIRECTORY, 'lockin_results_*.csv')
    dc_csv = find_latest_files(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')
    
    if lockin_csv is None:
        print("ERROR: No lock-in CSV file found!")
        return
    if dc_csv is None:
        print("ERROR: No DC monitor CSV file found!")
        return
    
    print(f"\n📁 Lock-in data: {os.path.basename(lockin_csv)}")
    print(f"📁 DC data: {os.path.basename(dc_csv)}")
    
    # Load data
    print("\nLoading data...")
    lockin_data = load_csv_data(lockin_csv)
    dc_data = load_csv_data(dc_csv)
    
    print(f"✓ Lock-in: {len(lockin_data)} samples")
    print(f"✓ DC monitor: {len(dc_data)} samples")
    
    # Merge data
    merged_df = merge_data_by_timestamp(lockin_data, dc_data, SYNC_TOLERANCE)
    
    if len(merged_df) == 0:
        print("ERROR: No synchronized samples found! Check timestamp alignment.")
        return
    
    # Save merged CSV
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
    merged_df.to_csv(merged_csv, index=False)
    print(f"\n✓ Merged data saved: {merged_csv}")
    print(f"  Columns: {', '.join(merged_df.columns)}")
    
    # Create combined plot
    print("\n📊 Creating combined plot...")
    fig = create_combined_plot(merged_df, lockin_csv, dc_csv)
    
    # Save plot
    plot_file = os.path.join(OUTPUT_DIRECTORY, f'combined_plot_{timestamp_str}.png')
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Combined plot saved: {plot_file}")
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 60)
    print("✓ ALL PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Individual lock-in: {os.path.basename(lockin_csv)}")
    print(f"  - Individual DC: {os.path.basename(dc_csv)}")
    print(f"  - Combined CSV: {os.path.basename(merged_csv)}")
    print(f"  - Combined plot: {os.path.basename(plot_file)}")
    print("=" * 60)

if __name__ == '__main__':
    main()
