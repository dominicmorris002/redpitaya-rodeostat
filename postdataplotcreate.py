"""
ACCV Cycle Averaging Module  - drop-in for startboth.py

Layout: 4 columns x 4 rows
  Col 1: R magnitude plots
  Col 2: Theta plots  
  Col 3: R magnitude plots (continued)
  Col 4: Theta plots (continued)

Row 1: R vs Time (DC overlay) | Theta vs Time (DC overlay) | R vs Time (clean) | Theta vs Time (clean)
Row 2: R raw scatter           | Theta raw scatter           | R cycle-avg line  | Theta cycle-avg line
Row 3: R bin-avg scatter       | Theta bin-avg scatter       | R smooth line only | Theta smooth line only
Row 4: (spare / expandable)
"""

import numpy as np
import pandas as pd
import warnings
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import os, glob
from datetime import datetime

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
CSV_PATH         = r"C:\Users\lab\PycharmProjects\dominic\redpitaya-rodeostat\Important_Data\combined_results_20260223_182526.csv"
OUTPUT_DIRECTORY = None
N_GRID_POINTS    = 500
MIN_CYCLES       = 2
MIN_PTS          = 3
SAVE_PNG         = True
SMOOTH_WINDOW    = 51
RAW_DS           = 5
DC_FILTER_ORDER  = 3
DC_FILTER_CUTOFF = 0.015
# -----------------------------------------------------------------------------


# =============================================================================
#  METHOD A - cycle interpolation (smooth line + sigma band)
# =============================================================================

def detect_triangle_cycles(dc_v, min_prominence_frac=0.3):
    sig_range  = np.max(dc_v) - np.min(dc_v)
    prominence = min_prominence_frac * sig_range
    min_dist   = max(1, len(dc_v) // 100)
    peaks,   _ = find_peaks( dc_v, prominence=prominence, distance=min_dist)
    troughs, _ = find_peaks(-dc_v, prominence=prominence, distance=min_dist)
    return peaks, troughs


def build_cycle_triples(peaks, troughs, dc_v):
    n = len(dc_v)

    def snap(indices, neighbours, mode):
        out = []
        for idx in indices:
            left  = neighbours[neighbours < idx]
            right = neighbours[neighbours > idx]
            lo = (int(left[-1])  + idx) // 2 if len(left)  else max(0,   idx - 50)
            hi = (idx + int(right[0])) // 2  if len(right) else min(n-1, idx + 50)
            lo, hi = max(0, lo), min(n-1, hi)
            seg = dc_v[lo:hi+1]
            out.append(lo + int(np.argmin(seg) if mode == 'min' else np.argmax(seg)))
        return np.array(out, dtype=int)

    s_troughs = snap(troughs, peaks,     'min')
    s_peaks   = snap(peaks,   s_troughs, 'max')

    cycles = []
    for pk in s_peaks:
        lt = s_troughs[s_troughs < pk]
        rt = s_troughs[s_troughs > pk]
        if len(lt) and len(rt):
            cycles.append((int(lt[-1]), int(pk), int(rt[0])))
    return cycles


def interp_half(dc_h, sig_h, v_grid):
    if len(dc_h) < 3:
        return None
    order = np.argsort(dc_h)
    dc_s, sig_s = dc_h[order], sig_h[order]
    _, keep = np.unique(dc_s, return_index=True)
    dc_s, sig_s = dc_s[keep], sig_s[keep]
    if len(dc_s) < 3:
        return None
    v_lo, v_hi = dc_s[0], dc_s[-1]
    mask = (v_grid >= v_lo) & (v_grid <= v_hi)
    if mask.sum() < 3:
        return None
    result = np.full(len(v_grid), np.nan)
    result[mask] = np.interp(v_grid[mask], dc_s, sig_s)
    return result


def cycle_average_line(dc_v, signal, cycles, v_grid, sweep='up'):
    traces = []
    for t0, pk, t1 in cycles:
        dc_h  = dc_v[t0:pk+1]   if sweep == 'up' else dc_v[pk:t1+1]
        sig_h = signal[t0:pk+1] if sweep == 'up' else signal[pk:t1+1]
        tr = interp_half(dc_h, sig_h, v_grid)
        if tr is not None:
            traces.append(tr)
    if not traces:
        return np.full(len(v_grid), np.nan), np.full(len(v_grid), np.nan)
    stack = np.vstack(traces)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_nan = np.all(np.isnan(stack), axis=0)
        mean_tr = np.where(all_nan, np.nan, np.nanmean(stack, axis=0))
        std_tr  = np.where(all_nan, np.nan, np.nanstd(stack,  axis=0))
    valid = ~np.isnan(mean_tr)
    if SMOOTH_WINDOW and valid.sum() > SMOOTH_WINDOW:
        smoothed = mean_tr.copy()
        smoothed[valid] = savgol_filter(mean_tr[valid],
                                        window_length=SMOOTH_WINDOW, polyorder=3)
        mean_tr = smoothed
    return mean_tr, std_tr


# =============================================================================
#  METHOD B - MATLAB bin-average (scatter, full coverage)
# =============================================================================

def filter_dc_ramp(dc_v):
    b, a = butter(DC_FILTER_ORDER, DC_FILTER_CUTOFF, btype='low')
    padlen = max(300, len(dc_v) // 10)
    return filtfilt(b, a, dc_v.astype(float), padlen=padlen)


def split_sweeps(dc_filtered):
    derV      = np.diff(dc_filtered)
    up_mask   = np.zeros(len(dc_filtered), dtype=bool)
    down_mask = np.zeros(len(dc_filtered), dtype=bool)
    up_mask[:-1]   = derV > 0
    down_mask[:-1] = derV <= 0
    up_mask[-1]    = up_mask[-2]
    down_mask[-1]  = down_mask[-2]
    return up_mask, down_mask


def bin_average(dc_pts, sig_pts, v_grid):
    bin_edges = np.linspace(v_grid[0], v_grid[-1], len(v_grid) + 1)
    bin_idx   = np.clip(np.digitize(dc_pts, bin_edges) - 1, 0, len(v_grid)-1)
    means = np.full(len(v_grid), np.nan)
    for b in range(len(v_grid)):
        pts = sig_pts[bin_idx == b]
        if len(pts) >= MIN_PTS:
            means[b] = np.mean(pts)
    valid = ~np.isnan(means)
    return v_grid[valid], means[valid]


# =============================================================================
#  MAIN PLOT  -  4 columns x 3 rows
# =============================================================================

def plot_accv_cycle_averaged(dc_v, li_R, li_Theta, time_s=None, theta_unit='deg',
                              timestamp_str=None, output_dir=None,
                              ds_note='', n_samples=None, n_ds=None,
                              csv_path=None):

    if output_dir is None:
        output_dir = (os.path.dirname(os.path.abspath(csv_path))
                      if csv_path else 'test_data')

    print("\n" + "="*60)
    print("CYCLE AVERAGING  (hybrid method)")
    print("="*60)

    # -- Method A --
    peaks, troughs = detect_triangle_cycles(dc_v)
    print(f"Detected {len(peaks)} peaks, {len(troughs)} troughs")
    cycles = build_cycle_triples(peaks, troughs, dc_v)
    print(f"Cycles: {len(cycles)}")

    v_grid = np.linspace(dc_v.min(), dc_v.max(), N_GRID_POINTS)

    R_up_mn, R_up_sd = cycle_average_line(dc_v, li_R,     cycles, v_grid, 'up')
    R_dn_mn, R_dn_sd = cycle_average_line(dc_v, li_R,     cycles, v_grid, 'down')
    T_up_mn, T_up_sd = cycle_average_line(dc_v, li_Theta, cycles, v_grid, 'up')
    T_dn_mn, T_dn_sd = cycle_average_line(dc_v, li_Theta, cycles, v_grid, 'down')

    # -- Method B --
    dc_filt            = filter_dc_ramp(dc_v)
    up_mask, down_mask = split_sweeps(dc_filt)
    print(f"Up pts: {up_mask.sum():,}   Down pts: {down_mask.sum():,}")

    v_Rs_up, Rs_up = bin_average(dc_v[up_mask],   li_R[up_mask],       v_grid)
    v_Rs_dn, Rs_dn = bin_average(dc_v[down_mask], li_R[down_mask],     v_grid)
    v_Ts_up, Ts_up = bin_average(dc_v[up_mask],   li_Theta[up_mask],   v_grid)
    v_Ts_dn, Ts_dn = bin_average(dc_v[down_mask], li_Theta[down_mask], v_grid)

    # t_axis used for DC-overlay plots (sample index)
    # t_seconds used for clean time plots (real seconds from Time_RP1)
    t_axis    = np.arange(len(dc_v))
    t_seconds = time_s if time_s is not None else t_axis  # real seconds if available

    UP_C  = '#1565C0'
    DN_C  = '#B71C1C'
    UP_CT = 'rgba(21,101,192,0.18)'
    DN_CT = 'rgba(183,28,28,0.18)'
    UP_L  = 'Forward sweep'
    DN_L  = 'Reverse sweep'
    DC_C  = '#E65100'
    R_C   = '#1565C0'
    TH_C  = '#6A1B9A'

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("pip install plotly")

    # =========================================================================
    # 4 cols x 3 rows
    #
    # Col:    1                        2                        3                          4
    # Row 1: R vs Time (DC overlay)   Theta vs Time (DC ovly)  R vs Time (clean)          Theta vs Time (clean)
    # Row 2: R raw scatter            Theta raw scatter         R cycle-avg line +/-1sig   Theta cycle-avg line +/-1sig
    # Row 3: R bin-avg scatter        Theta bin-avg scatter     R smooth line only         Theta smooth line only
    # =========================================================================

    n_cyc = len(cycles)
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=(
            # Row 1
            'R vs Time  |  DC Ramp overlay',
            'Theta vs Time  |  DC Ramp overlay',
            'R vs Time  (no overlay)',
            'Theta vs Time  (no overlay)',
            # Row 2
            f'R vs DC  -  Raw Data  ({len(dc_v):,} pts)',
            f'Theta vs DC  -  Raw Data  ({len(dc_v):,} pts)',
            f'R vs DC  -  Cycle-Averaged  ({n_cyc} cycles)  +/-1sig',
            f'Theta vs DC  -  Cycle-Averaged  ({n_cyc} cycles)  +/-1sig',
            # Row 3
            'R vs DC  -  Bin-Averaged Scatter',
            'Theta vs DC  -  Bin-Averaged Scatter',
            'R vs DC  -  Smooth Averaged Line',
            'Theta vs DC  -  Smooth Averaged Line',
        ),
        specs=[
            # Row 1: cols 1&2 have dual y for DC overlay, cols 3&4 plain
            [{"secondary_y": True}, {"secondary_y": True},
             {"secondary_y": False}, {"secondary_y": False}],
            # Row 2: all plain
            [{"secondary_y": False}, {"secondary_y": False},
             {"secondary_y": False}, {"secondary_y": False}],
            # Row 3: all plain
            [{"secondary_y": False}, {"secondary_y": False},
             {"secondary_y": False}, {"secondary_y": False}],
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    title_str = 'AC Cyclic Voltammetry  -  Full Analysis'
    if n_samples is not None:
        title_str += (f'<br><sup>{n_samples:,} raw samples  |  '
                      f'{n_cyc} cycles  |  '
                      f'up: {up_mask.sum():,}  down: {down_mask.sum():,}</sup>')

    # -- helpers ---------------------------------------------------------------
    def _tline(row, col, x, y, color, name, secondary=False, showlegend=True):
        fig.add_trace(go.Scattergl(
            x=x.tolist(), y=y.tolist(),
            mode='lines', line=dict(color=color, width=1.0),
            name=name, showlegend=showlegend,
            hovertemplate='t=%{x}  val=%{y:.6f}<extra>' + name + '</extra>',
        ), row=row, col=col, secondary_y=secondary)

    def _raw(row, col, mask, signal, color, name, showlegend=True):
        idx = np.where(mask)[0][::RAW_DS]
        fig.add_trace(go.Scattergl(
            x=dc_v[idx].tolist(), y=signal[idx].tolist(),
            mode='markers', marker=dict(color=color, size=3, opacity=0.45),
            name=name, showlegend=showlegend,
            hovertemplate='V=%{x:.4f}  val=%{y:.6f}<extra>' + name + '</extra>',
        ), row=row, col=col)

    def _band(row, col, mean, std, color_t):
        msk = ~np.isnan(mean)
        v   = v_grid[msk].tolist()
        hi  = (mean + std)[msk].tolist()
        lo  = (mean - std)[msk].tolist()
        fig.add_trace(go.Scatter(
            x=v + v[::-1], y=hi + lo[::-1],
            fill='toself', fillcolor=color_t,
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip', showlegend=False,
        ), row=row, col=col)

    def _line(row, col, mean, color, name, showlegend=True):
        msk = ~np.isnan(mean)
        fig.add_trace(go.Scatter(
            x=v_grid[msk].tolist(), y=mean[msk].tolist(),
            mode='lines', line=dict(color=color, width=2.5),
            name=name, showlegend=showlegend,
            hovertemplate='V=%{x:.4f}  val=%{y:.6f}<extra>' + name + '</extra>',
        ), row=row, col=col)

    def _scatter(row, col, vx, vy, color, name, showlegend=False):
        fig.add_trace(go.Scattergl(
            x=vx.tolist(), y=vy.tolist(),
            mode='markers', marker=dict(color=color, size=5, opacity=0.80),
            name=name, showlegend=showlegend,
            hovertemplate='V=%{x:.4f}  val=%{y:.6f}<extra>' + name + '</extra>',
        ), row=row, col=col)

    # =========================================================================
    # ROW 1
    # =========================================================================
    # Col 1: R vs Time WITH DC overlay
    _tline(1, 1, t_axis[::RAW_DS], li_R[::RAW_DS],     R_C,  'R (lock-in)',  secondary=False)
    _tline(1, 1, t_axis[::RAW_DS], dc_v[::RAW_DS],     DC_C, 'DC Ramp (V)', secondary=True)
    # Col 2: Theta vs Time WITH DC overlay
    _tline(1, 2, t_axis[::RAW_DS], li_Theta[::RAW_DS], TH_C, 'Theta',        secondary=False, showlegend=False)
    _tline(1, 2, t_axis[::RAW_DS], dc_v[::RAW_DS],     DC_C, 'DC Ramp (V)', secondary=True,  showlegend=False)
    # Col 3: R vs Time NO overlay
    _tline(1, 3, t_seconds[::RAW_DS], li_R[::RAW_DS],     R_C,  'R (lock-in)',  showlegend=False)
    # Col 4: Theta vs Time NO overlay
    _tline(1, 4, t_seconds[::RAW_DS], li_Theta[::RAW_DS], TH_C, 'Theta',        showlegend=False)

    # =========================================================================
    # ROW 2
    # =========================================================================
    # Col 1&2: raw scatter coloured by sweep
    _raw(2, 1, up_mask,   li_R,     UP_C, UP_L)
    _raw(2, 1, down_mask, li_R,     DN_C, DN_L)
    _raw(2, 2, up_mask,   li_Theta, UP_C, UP_L, showlegend=False)
    _raw(2, 2, down_mask, li_Theta, DN_C, DN_L, showlegend=False)
    # Col 3&4: smooth cycle-averaged line + sigma band
    _band(2, 3, R_up_mn, R_up_sd, UP_CT);  _band(2, 3, R_dn_mn, R_dn_sd, DN_CT)
    _line(2, 3, R_up_mn, UP_C, UP_L, showlegend=False)
    _line(2, 3, R_dn_mn, DN_C, DN_L, showlegend=False)
    _band(2, 4, T_up_mn, T_up_sd, UP_CT);  _band(2, 4, T_dn_mn, T_dn_sd, DN_CT)
    _line(2, 4, T_up_mn, UP_C, UP_L, showlegend=False)
    _line(2, 4, T_dn_mn, DN_C, DN_L, showlegend=False)

    # =========================================================================
    # ROW 3
    # =========================================================================
    # Col 1&2: bin-averaged scatter
    _scatter(3, 1, v_Rs_up, Rs_up, UP_C, UP_L)
    _scatter(3, 1, v_Rs_dn, Rs_dn, DN_C, DN_L)
    _scatter(3, 2, v_Ts_up, Ts_up, UP_C, UP_L, showlegend=False)
    _scatter(3, 2, v_Ts_dn, Ts_dn, DN_C, DN_L, showlegend=False)
    # Col 3&4: pure smooth averaged line
    _line(3, 3, R_up_mn, UP_C, UP_L, showlegend=False)
    _line(3, 3, R_dn_mn, DN_C, DN_L, showlegend=False)
    _line(3, 4, T_up_mn, UP_C, UP_L, showlegend=False)
    _line(3, 4, T_dn_mn, DN_C, DN_L, showlegend=False)

    # =========================================================================
    # AXIS LABELS
    # =========================================================================
    # Row 1, cols 1&2 dual-y
    for col in (1, 2):
        fig.update_xaxes(title_text='Sample Index', row=1, col=col)
    fig.update_yaxes(title_text='AC Magnitude R (V)',          row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='DC Ramp (V)',                 row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text='DC Ramp (V)',                 row=1, col=2, secondary_y=True, showgrid=False)

    # Row 1, cols 3&4 plain time
    fig.update_xaxes(title_text='Time (s)', row=1, col=3)
    fig.update_xaxes(title_text='Time (s)', row=1, col=4)
    fig.update_yaxes(title_text='AC Magnitude R (V)',          row=1, col=3)
    fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=1, col=4)

    # Rows 2&3: DC potential on x
    for row in (2, 3):
        for col in (1, 2, 3, 4):
            fig.update_xaxes(title_text='DC Potential (V)', row=row, col=col)
        fig.update_yaxes(title_text='AC Magnitude R (V)',          row=row, col=1)
        fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=row, col=2)
        fig.update_yaxes(title_text='AC Magnitude R (V)',          row=row, col=3)
        fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=row, col=4)

    fig.update_layout(
        title=dict(text=title_str, x=0.5, xanchor='center', font=dict(size=15)),
        height=1100, width=2000,
        template='plotly_white',
        hovermode='closest',
        legend=dict(x=0.0, y=1.05, orientation='h',
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='grey', borderwidth=1),
    )

    # -- Save ------------------------------------------------------------------
    if SAVE_PNG:
        os.makedirs(output_dir, exist_ok=True)
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = os.path.join(output_dir, f'accv_cycle_averaged_{timestamp_str}.html')
        fig.write_html(html_path)
        print(f"Saved interactive HTML: {html_path}")
        try:
            png_path = os.path.join(output_dir, f'accv_cycle_averaged_{timestamp_str}.png')
            fig.write_image(png_path, scale=2)
            print(f"Saved static PNG:       {png_path}")
        except Exception:
            print("Note: static PNG skipped (pip install kaleido to enable)")

    return fig


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    elif CSV_PATH:
        csv_path = CSV_PATH
    else:
        try:
            csv_path = max(
                glob.glob(os.path.join('test_data', 'combined_results_*.csv')),
                key=os.path.getctime)
        except ValueError:
            print("No CSV found. Set CSV_PATH at the top.")
            sys.exit(1)

    print(f"Loading: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df       = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    dc_v     = df['DC_Voltage'].values
    li_R     = df['R'].values
    li_Theta = df['Theta'].values
    # Use Time_RP1 if available, normalised to start at 0
    if 'Time_RP1' in df.columns:
        time_s = df['Time_RP1'].values
        time_s = time_s - time_s[0]
    elif 'Time_RP2' in df.columns:
        time_s = df['Time_RP2'].values
        time_s = time_s - time_s[0]
    else:
        time_s = None
    n_ds     = len(df)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig = plot_accv_cycle_averaged(
        dc_v, li_R, li_Theta,
        time_s=time_s,
        theta_unit='deg',
        timestamp_str=ts,
        output_dir=OUTPUT_DIRECTORY or None,
        csv_path=csv_path,
        n_samples=n_ds,
        n_ds=n_ds)

    if fig is not None:
        fig.show()
    print("Done.")
