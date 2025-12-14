import subprocess
from datetime import datetime
import time
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
SYNC_TOLERANCE = 0.01  # 10 ms tolerance for timestamp matching

def find_latest_files(output_dir, pattern):
    """Find the most recent files matching pattern"""
    files = glob.glob(os.path.join(output_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_csv_data(filepath):
    """Load CSV data with metadata handling"""
    with open(filepath, 'r') as f:
        first_line = f.readline()
    
    if first_line.startswith('#'):
        data = pd.read_csv(filepath, comment='#')
    else:
        data = pd.read_csv(filepath)
    
    return data

def merge_data_by_timestamp(lockin_data, dc_data, tolerance=0.01):
    """Merge lock-in and DC data based on timestamp alignment"""
    print("\nMerging data by timestamp...")
    print(f"Lock-in samples: {len(lockin_data)}")
    print(f"DC samples: {len(dc_data)}")
    
    lockin_times = lockin_data['AbsoluteTimestamp'].values
    dc_times = dc_data['AbsoluteTimestamp'].values
    
    start_time = max(lockin_times[0], dc_times[0])
    end_time = min(lockin_times[-1], dc_times[-1])
    
    lockin_mask = (lockin_times >= start_time) & (lockin_times <= end_time)
    dc_mask = (dc_times >= start_time) & (dc_times <= end_time)
    
    lockin_overlap = lockin_data[lockin_mask].copy()
    dc_overlap = dc_data[dc_mask].copy()
    
    merged_data = []
    matched_count = 0
    
    for idx, row in lockin_overlap.iterrows():
        t_lockin = row['AbsoluteTimestamp']
        time_diffs = np.abs(dc_overlap['AbsoluteTimestamp'].values - t_lockin)
        nearest_idx = np.argmin(time_diffs)
        time_diff = time_diffs[nearest_idx]
        
        if time_diff <= tolerance:
            dc_row = dc_overlap.iloc[nearest_idx]
            merged_row = {
                'AbsoluteTimestamp_RP1': t_lockin,
                'AbsoluteTimestamp_RP2': dc_row['AbsoluteTimestamp'],
                'RelativeTime_RP1': row['RelativeTime'],
                'RelativeTime_RP2': dc_row['RelativeTime'],
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
    print(f"✓ Matched {matched_count}/{len(lockin_overlap)} samples ({(matched_count/len(lockin_overlap)*100):.1f}%)")
    
    return merged_df

def create_combined_plot(merged_df):
    """Create comprehensive combined plot"""
    fig = plt.figure(figsize=(18, 12))
    
    # Use RP1 (lock-in) relative time for x-axis
    t = merged_df['RelativeTime_RP1'].values
    X = merged_df['X'].values
    Y = merged_df['Y'].values
    R = merged_df['R'].values
    Theta = merged_df['Theta'].values
    DC_V = merged_df['DC_Voltage'].values
    
    # 1. Lock-in X
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, X, 'b-', linewidth=0.5)
    ax1.axhline(np.mean(X), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(X):.4f}V')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X (V)')
    ax1.set_title('Lock-in: In-phase (X)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Lock-in Y
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, Y, 'r-', linewidth=0.5)
    ax2.axhline(np.mean(Y), color='b', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(Y):.4f}V')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y (V)')
    ax2.set_title('Lock-in: Quadrature (Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Lock-in R
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(t, R, 'm-', linewidth=0.5)
    ax3.axhline(np.mean(R), color='b', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(R):.4f}V')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('R (V)')
    ax3.set_title('Lock-in: Magnitude (R)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Lock-in Theta
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(t, Theta, 'c-', linewidth=0.5)
    ax4.axhline(np.mean(Theta), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(Theta):.4f} rad')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Theta (rad)')
    ax4.set_title('Lock-in: Phase (Theta)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. DC Voltage
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(t, DC_V, 'g-', linewidth=0.5)
    ax5.axhline(np.mean(DC_V), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(DC_V):.6f}V')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Voltage (V)')
    ax5.set_title('DC Voltage Monitor')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. IQ plot
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(X, Y, 'g.', markersize=1, alpha=0.5)
    ax6.plot(np.mean(X), np.mean(Y), 'r+', markersize=15, markeredgewidth=2, label='Mean')
    ax6.set_xlabel('X (V)')
    ax6.set_ylabel('Y (V)')
    ax6.set_title('Lock-in: IQ Plot')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    # 7. R vs DC Voltage
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(DC_V, R, c=t, cmap='viridis', s=1, alpha=0.6)
    ax7.set_xlabel('DC Voltage (V)')
    ax7.set_ylabel('Lock-in R (V)')
    ax7.set_title('R vs DC Voltage')
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Time (s)')
    ax7.grid(True, alpha=0.3)
    
    # 8. All signals overlaid (normalized)
    ax8 = plt.subplot(3, 3, 8)
    R_norm = (R - np.min(R)) / (np.max(R) - np.min(R) + 1e-9)
    DC_norm = (DC_V - np.min(DC_V)) / (np.max(DC_V) - np.min(DC_V) + 1e-9)
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-9)
    
    ax8.plot(t, R_norm, 'm-', linewidth=1, alpha=0.7, label='R')
    ax8.plot(t, DC_norm, 'g-', linewidth=1, alpha=0.7, label='DC')
    ax8.plot(t, X_norm, 'b-', linewidth=1, alpha=0.7, label='X')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Normalized Value')
    ax8.set_title('All Signals (Normalized)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Sync quality
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
    
    fig.suptitle('Combined Lock-in + DC Monitor Results', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

print("\n" + "=" * 60)
print("COMBINING RESULTS")
print("=" * 60)

time.sleep(0.5)

lockin_csv = find_latest_files(OUTPUT_DIRECTORY, 'lockin_results_*.csv')
dc_csv = find_latest_files(OUTPUT_DIRECTORY, 'dc_voltage_*.csv')

if lockin_csv and dc_csv:
    print(f"Lock-in: {os.path.basename(lockin_csv)}")
    print(f"DC: {os.path.basename(dc_csv)}")
    
    lockin_data = load_csv_data(lockin_csv)
    dc_data = load_csv_data(dc_csv)
    
    merged_df = merge_data_by_timestamp(lockin_data, dc_data, SYNC_TOLERANCE)
    
    if len(merged_df) > 0:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_csv = os.path.join(OUTPUT_DIRECTORY, f'combined_results_{timestamp_str}.csv')
        merged_df.to_csv(merged_csv, index=False)
        print(f"✓ Merged CSV saved: {merged_csv}")
        
        fig = create_combined_plot(merged_df)
        plot_file = os.path.join(OUTPUT_DIRECTORY, f'combined_plot_{timestamp_str}.png')
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Combined plot saved: {plot_file}")
        plt.show()
        print("=" * 60)
    else:
        print("ERROR: No synchronized samples found!")
else:
    print("ERROR: Could not find output files!")
