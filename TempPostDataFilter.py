"""
ACCV Cycle Averaging Module
Drop-in addition for startboth.py

Replaces the "fat" R vs DC and Theta vs DC plots with clean, thin,
cycle-averaged traces by:
  1. Detecting triangle-wave cycles from the DC ramp
  2. Splitting both DC and lock-in signals into per-cycle segments
  3. Interpolating each cycle onto a shared voltage grid
  4. Averaging (or plotting individually with transparency)

Usage: paste plot_accv_cycle_averaged() call at the bottom of startboth.py
       after the merged CSV is created, or run this file standalone
       pointing at an existing combined_results CSV.
"""

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os
import glob
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors startboth.py settings)
# ─────────────────────────────────────────────────────────────────────────────

# *** SET YOUR CSV PATH HERE -- change this to your combined_results CSV ***
CSV_PATH = r"C:\Users\Owner\Downloads\combined_results_20260223_182526.csv"

# Output PNG is saved next to the CSV automatically (no need to change this)
OUTPUT_DIRECTORY  = None      # None = auto: saves PNG in same folder as CSV
N_GRID_POINTS     = 500       # voltage grid resolution for interpolation
MIN_CYCLES        = 2         # need at least this many cycles to average
SHOW_INDIVIDUAL   = False     # plot individual cycles faintly behind average
INDIVIDUAL_ALPHA  = 0.12      # transparency for individual cycle traces
SAVE_PNG          = True
SMOOTH_WINDOW     = 51        # Savitzky-Golay window for mean line smoothing
                              # must be odd; bigger = smoother (try 21-51)
                              # set to 0 to disable smoothing
# ─────────────────────────────────────────────────────────────────────────────


def detect_triangle_cycles(dc_v, min_prominence_frac=0.3):
    """
    Find the indices of DC ramp peaks and troughs to segment cycles.

    Returns
    -------
    peaks   : array of peak indices  (top of ramp)
    troughs : array of trough indices (bottom of ramp)
    """
    sig_range   = np.max(dc_v) - np.min(dc_v)
    prominence  = min_prominence_frac * sig_range
    min_dist    = max(1, len(dc_v) // 100)   # at least 100 cycles between peaks

    peaks,   _ = find_peaks( dc_v, prominence=prominence, distance=min_dist)
    troughs, _ = find_peaks(-dc_v, prominence=prominence, distance=min_dist)

    return peaks, troughs


def build_cycle_boundaries(peaks, troughs, n_samples):
    """
    Interleave peaks and troughs in time order to get cycle start/end indices.
    A full cycle = trough -> peak -> trough  (or peak -> trough -> peak).
    We define cycles as trough-to-trough for consistency.

    Returns list of (start_idx, end_idx) tuples.
    """
    # Sort all turning points
    all_tp = np.sort(np.concatenate([peaks, troughs]))

    # Find troughs in the sorted list
    trough_set = set(troughs.tolist())
    trough_positions = [i for i, idx in enumerate(all_tp) if idx in trough_set]

    if len(trough_positions) < 2:
        return []

    cycles = []
    for i in range(len(trough_positions) - 1):
        s = all_tp[trough_positions[i]]
        e = all_tp[trough_positions[i + 1]]
        cycles.append((int(s), int(e)))

    return cycles


def interpolate_to_voltage_grid(dc_seg, signal_seg, v_grid, sweep='up'):
    """
    Interpolate signal_seg onto v_grid for either the up or down sweep.
    Within one trough-to-trough cycle:
        up sweep   = first half  (trough -> peak)
        down sweep = second half (peak -> trough)
    """
    # find the peak within this segment
    peak_local = np.argmax(dc_seg)

    if sweep == 'up':
        dc_half  = dc_seg[:peak_local + 1]
        sig_half = signal_seg[:peak_local + 1]
    else:
        dc_half  = dc_seg[peak_local:]
        sig_half = signal_seg[peak_local:]

    if len(dc_half) < 3:
        return None

    # Sort by DC voltage (handles minor non-monotonicity)
    sort_idx = np.argsort(dc_half)
    dc_s     = dc_half[sort_idx]
    sig_s    = sig_half[sort_idx]

    # Only interpolate within the valid DC range of this sweep
    v_min, v_max = dc_s[0], dc_s[-1]
    mask = (v_grid >= v_min) & (v_grid <= v_max)
    if mask.sum() < 3:
        return None

    result = np.full(len(v_grid), np.nan)
    result[mask] = np.interp(v_grid[mask], dc_s, sig_s)
    return result


def cycle_average(dc_v, signal, cycles, v_grid, sweep='up'):
    """
    For each cycle, extract the requested sweep and interpolate onto v_grid.
    Returns (individual_traces, mean_trace, std_trace).
    """
    traces = []
    for (s, e) in cycles:
        dc_seg  = dc_v[s:e]
        sig_seg = signal[s:e]
        tr = interpolate_to_voltage_grid(dc_seg, sig_seg, v_grid, sweep)
        if tr is not None:
            traces.append(tr)

    if not traces:
        return [], np.full(len(v_grid), np.nan), np.full(len(v_grid), np.nan)

    stack   = np.vstack(traces)
    # Only average columns that have at least one real value
    all_nan = np.all(np.isnan(stack), axis=0)
    mean_tr = np.where(all_nan, np.nan, np.nanmean(stack, axis=0))
    std_tr  = np.where(all_nan, np.nan, np.nanstd(stack,  axis=0))

    # Smooth the mean trace with Savitzky-Golay to remove residual jitter
    # while preserving peak shape
    from scipy.signal import savgol_filter
    valid = ~np.isnan(mean_tr)
    if SMOOTH_WINDOW and valid.sum() > SMOOTH_WINDOW:
        smoothed = mean_tr.copy()
        smoothed[valid] = savgol_filter(mean_tr[valid], window_length=SMOOTH_WINDOW, polyorder=3)
        mean_tr = smoothed
    return traces, mean_tr, std_tr


def plot_accv_cycle_averaged(dc_v, li_R, li_Theta, theta_unit='deg',
                              timestamp_str=None, output_dir=None,
                              ds_note='', n_samples=None, n_ds=None,
                              csv_path=None):
    """
    Main entry point.  Call this with the merged, (optionally downsampled)
    arrays that startboth.py already builds.

    Parameters
    ----------
    dc_v        : 1-D array  DC potential (V)
    li_R        : 1-D array  lock-in magnitude R (V)
    li_Theta    : 1-D array  lock-in phase Theta
    theta_unit  : str        'deg' or 'rad'
    timestamp_str: str       for filename; auto-generated if None
    output_dir  : str
    ds_note     : str        e.g. ' [1:1250 ds]' for plot titles
    n_samples   : int        raw sample count for title
    n_ds        : int        plotted sample count for title
    """

    # Resolve output directory: use folder of CSV if not specified
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_path)) if csv_path else 'test_data'

    print("\n" + "=" * 60)
    print("CYCLE AVERAGING")
    print("=" * 60)

    # ── Detect cycles ──────────────────────────────────────────────────────
    peaks, troughs = detect_triangle_cycles(dc_v)
    print(f"Detected {len(peaks)} peaks, {len(troughs)} troughs")

    cycles = build_cycle_boundaries(peaks, troughs, len(dc_v))
    print(f"Usable trough-to-trough cycles: {len(cycles)}")

    if len(cycles) < MIN_CYCLES:
        print(f"Not enough cycles (need >= {MIN_CYCLES}). Skipping cycle averaging.")
        return

    # ── Shared voltage grid ────────────────────────────────────────────────
    v_min = np.min(dc_v)
    v_max = np.max(dc_v)
    v_grid = np.linspace(v_min, v_max, N_GRID_POINTS)

    # ── Average both sweeps ────────────────────────────────────────────────
    R_up_traces,  R_up_mean,  R_up_std  = cycle_average(dc_v, li_R,     cycles, v_grid, 'up')
    R_dn_traces,  R_dn_mean,  R_dn_std  = cycle_average(dc_v, li_R,     cycles, v_grid, 'down')
    Th_up_traces, Th_up_mean, Th_up_std = cycle_average(dc_v, li_Theta, cycles, v_grid, 'up')
    Th_dn_traces, Th_dn_mean, Th_dn_std = cycle_average(dc_v, li_Theta, cycles, v_grid, 'down')

    n_cyc_R  = len(R_up_traces)
    n_cyc_Th = len(Th_up_traces)
    print(f"Averaged {n_cyc_R} cycles for R,  {n_cyc_Th} cycles for Theta")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_R_line   = axes[0, 0]
    ax_Th_line  = axes[0, 1]
    ax_R_sc     = axes[1, 0]
    ax_Th_sc    = axes[1, 1]

    UP_COLOR   = '#1565C0'   # deep blue  -- anodic
    DN_COLOR   = '#B71C1C'   # deep red   -- cathodic
    UP_LABEL   = 'Forward sweep (→)'
    DN_LABEL   = 'Reverse sweep (←)'

    def _plot_panel(ax, v_grid,
                    up_traces, up_mean, up_std,
                    dn_traces, dn_mean, dn_std,
                    ylabel, title, scatter=False):

        if SHOW_INDIVIDUAL and not scatter:
            for tr in up_traces:
                mask = ~np.isnan(tr)
                ax.plot(v_grid[mask], tr[mask],
                        color=UP_COLOR, alpha=INDIVIDUAL_ALPHA, lw=0.6)
            for tr in dn_traces:
                mask = ~np.isnan(tr)
                ax.plot(v_grid[mask], tr[mask],
                        color=DN_COLOR, alpha=INDIVIDUAL_ALPHA, lw=0.6)

        if scatter:
            # scatter each averaged point
            mask_u = ~np.isnan(up_mean)
            mask_d = ~np.isnan(dn_mean)
            ax.scatter(v_grid[mask_u], up_mean[mask_u],
                       s=4, color=UP_COLOR, alpha=0.8, label=UP_LABEL, zorder=3)
            ax.scatter(v_grid[mask_d], dn_mean[mask_d],
                       s=4, color=DN_COLOR, alpha=0.8, label=DN_LABEL, zorder=3)
        else:
            mask_u = ~np.isnan(up_mean)
            mask_d = ~np.isnan(dn_mean)
            # ±1σ band
            ax.fill_between(v_grid[mask_u],
                             (up_mean - up_std)[mask_u],
                             (up_mean + up_std)[mask_u],
                             color=UP_COLOR, alpha=0.15, zorder=2)
            ax.fill_between(v_grid[mask_d],
                             (dn_mean - dn_std)[mask_d],
                             (dn_mean + dn_std)[mask_d],
                             color=DN_COLOR, alpha=0.15, zorder=2)
            # Mean lines
            ax.plot(v_grid[mask_u], up_mean[mask_u],
                    color=UP_COLOR, lw=2.0, label=UP_LABEL, zorder=4)
            ax.plot(v_grid[mask_d], dn_mean[mask_d],
                    color=DN_COLOR, lw=2.0, label=DN_LABEL, zorder=4)

        ax.set_xlabel('DC Potential (V)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel,             fontsize=12, fontweight='bold')
        ax.set_title(title,               fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # R -- line
    _plot_panel(
        ax_R_line, v_grid,
        R_up_traces, R_up_mean, R_up_std,
        R_dn_traces, R_dn_mean, R_dn_std,
        'AC Magnitude R (V)',
        f'R vs DC Potential — Cycle Average ({n_cyc_R} cycles)\n'
        f'Shaded band = ±1σ across cycles{ds_note}',
        scatter=False)

    # Theta -- line
    _plot_panel(
        ax_Th_line, v_grid,
        Th_up_traces, Th_up_mean, Th_up_std,
        Th_dn_traces, Th_dn_mean, Th_dn_std,
        f'Phase Angle ({theta_unit})',
        f'Theta vs DC Potential — Cycle Average ({n_cyc_Th} cycles)\n'
        f'Shaded band = ±1σ across cycles{ds_note}',
        scatter=False)

    # R -- scatter of averaged values
    _plot_panel(
        ax_R_sc, v_grid,
        R_up_traces, R_up_mean, R_up_std,
        R_dn_traces, R_dn_mean, R_dn_std,
        'AC Magnitude R (V)',
        f'R vs DC Potential — Averaged Scatter{ds_note}',
        scatter=True)

    # Theta -- scatter of averaged values
    _plot_panel(
        ax_Th_sc, v_grid,
        Th_up_traces, Th_up_mean, Th_up_std,
        Th_dn_traces, Th_dn_mean, Th_dn_std,
        f'Phase Angle ({theta_unit})',
        f'Theta vs DC Potential — Averaged Scatter{ds_note}',
        scatter=True)

    # Title
    title_parts = ['AC Cyclic Voltammetry — Cycle-Averaged Plots']
    if n_samples is not None and n_ds is not None:
        title_parts.append(f'{n_samples:,} raw samples  |  {n_ds:,} used for averaging  |  {n_cyc_R} cycles')
    fig.suptitle('\n'.join(title_parts), fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    if SAVE_PNG:
        os.makedirs(output_dir, exist_ok=True)
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(output_dir, f'accv_cycle_averaged_{timestamp_str}.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved cycle-averaged plot: {png_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE: run directly -- uses CSV_PATH set at the top of this file
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Command-line argument overrides the CSV_PATH config above
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    elif CSV_PATH:
        csv_path = CSV_PATH
    else:
        # Fallback: find most recent combined_results CSV in test_data/
        try:
            csv_path = max(
                glob.glob(os.path.join('test_data', 'combined_results_*.csv')),
                key=os.path.getctime)
        except ValueError:
            print("No CSV found. Set CSV_PATH at the top of this file.")
            sys.exit(1)

    print(f"Loading: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        print("Check that CSV_PATH at the top of this file is correct.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    dc_v     = df['DC_Voltage'].values
    li_R     = df['R'].values
    li_Theta = df['Theta'].values
    n_ds     = len(df)
    n_samples = n_ds

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Output PNG saves next to the CSV (OUTPUT_DIRECTORY=None triggers this)
    out_dir = OUTPUT_DIRECTORY if OUTPUT_DIRECTORY else None

    fig = plot_accv_cycle_averaged(
        dc_v, li_R, li_Theta,
        theta_unit='deg',
        timestamp_str=ts,
        output_dir=out_dir,
        csv_path=csv_path,
        ds_note='',
        n_samples=n_samples,
        n_ds=n_ds)

    if fig is not None:
        plt.show()
    print("Done.")
