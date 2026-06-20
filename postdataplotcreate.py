"""
ACCV Cycle Averaging Module - drop-in for startboth.py
Editor panel built in - no external dependencies needed.

Layout: 4 columns x 4 rows
  Col 1: R magnitude plots
  Col 2: Theta plots
  Col 3: R magnitude plots (continued)
  Col 4: Theta plots (continued)

Row 1: R vs Time (DC overlay) | Theta vs Time (DC overlay) | R vs Time (clean) | Theta vs Time (clean)
Row 2: R raw scatter           | Theta raw scatter           | R cycle-avg line  | Theta cycle-avg line
Row 3: R bin-avg scatter       | Theta bin-avg scatter       | R smooth line only | Theta smooth line only
Row 4: DC vs Time (raw)        | DC vs Time (filtered)       | DC derivative      | DC histogram
"""

import numpy as np
import pandas as pd
import warnings
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import os, glob, sys, webbrowser
from datetime import datetime

# Config
Csv_Path         = None                        #For manual File Selection type the path here ex.) r"C:\dominic\redpitaya-rodeostat\test_data\combined_results_20260601_173448.csv"
Output_Directory = None
N_Grid_Points    = 500
Min_Cycles       = 2
Min_Pts          = 3
Save_Png         = True
Smooth_Window    = 51
Raw_Ds           = 3
Dc_Filter_Order  = 3
Dc_Filter_Cutoff = 0.015

# ---------------------------------------------------------------------------
# Inline editor panel - injected into HTML after saving
# ---------------------------------------------------------------------------

EDITOR_JS = r"""
<style>
#accv-edit-toggle {
  position: fixed; top: 12px; right: 12px; z-index: 999999;
  background: #1565C0; color: #fff; border: none; border-radius: 6px;
  padding: 8px 16px; font-size: 13px; font-weight: 600;
  cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  transition: background 0.2s;
}
#accv-edit-toggle:hover { background: #0d47a1; }
#accv-edit-toggle.active { background: #B71C1C; }
#accv-panel {
  position: fixed; top: 0; right: -520px; width: 500px; height: 100vh;
  background: #fafafa; border-left: 2px solid #1565C0;
  box-shadow: -4px 0 16px rgba(0,0,0,0.15);
  z-index: 999998; overflow-y: auto;
  transition: right 0.3s ease; padding: 16px;
  font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px;
}
#accv-panel.open { right: 0; }
#accv-panel h2 { font-size: 15px; color: #1565C0; margin: 0 0 12px; }
#accv-panel h3 { font-size: 12px; color: #555; margin: 14px 0 6px;
  text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid #ddd; padding-bottom:3px; }
.accv-row { display: flex; align-items: center; gap: 6px; margin: 5px 0; }
.accv-row label { width: 110px; flex-shrink: 0; color: #333; }
.accv-row input[type=range] { flex: 1; }
.accv-row input[type=number] { width: 72px; padding: 2px 4px; border: 1px solid #bbb; border-radius: 4px; }
.accv-btn {
  display: inline-block; margin: 4px 4px 0 0;
  padding: 6px 12px; border: none; border-radius: 4px;
  font-size: 12px; font-weight: 600; cursor: pointer;
}
.accv-btn.blue  { background:#1565C0; color:#fff; }
.accv-btn.red   { background:#B71C1C; color:#fff; }
.accv-btn.grey  { background:#607D8B; color:#fff; }
.accv-btn.green  { background:#2E7D32; color:#fff; }
.accv-btn.orange { background:#E65100; color:#fff; }
.accv-btn:hover { opacity:0.85; }
#accv-removed-count { color:#B71C1C; font-weight:700; }
#accv-undo-count    { color:#555; font-size:11px; }
#accv-status        { margin-top:8px; color:#2E7D32; font-weight:600; min-height:18px; }
.accv-signal-block { background:#fff; border:1px solid #e0e0e0;
  border-radius:6px; padding:10px; margin-bottom:10px; }
.accv-signal-block .sig-title { font-weight:700; color:#333; margin-bottom:6px; font-size:12px; }
</style>

<button id="accv-edit-toggle" onclick="accvTogglePanel()">&#9881; Edit Mode</button>

<div id="accv-panel">
  <h2>&#9881; ACCV Edit Mode</h2>
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
    <button class="accv-btn grey"  onclick="accvUndo()">&#8617; Undo</button>
    <button class="accv-btn red"   onclick="accvReset()">&#10005; Reset All</button>
    <span id="accv-undo-count">0 actions in stack</span>
  </div>
  <h3>Global Crop (samples)</h3>
  <div class="accv-row">
    <label>Trim Left</label>
    <input type="range" id="crop-left"  min="0" max="5000" step="1" value="0"
           oninput="accvSetCrop('left', this.value)">
    <input type="number" id="crop-left-n"  min="0" max="5000" step="1" value="0"
           onchange="accvSetCrop('left', this.value)">
  </div>
  <div class="accv-row">
    <label>Trim Right</label>
    <input type="range" id="crop-right" min="0" max="5000" step="1" value="0"
           oninput="accvSetCrop('right', this.value)">
    <input type="number" id="crop-right-n" min="0" max="5000" step="1" value="0"
           onchange="accvSetCrop('right', this.value)">
  </div>
  <h3>Per-Signal Controls</h3>
  <div id="accv-signals"></div>
  <h3>Sweep Direction</h3>
  <p style="font-size:11px;color:#666;margin:0 0 6px;">
    After shifting DC to align cycles, press this to recalculate
    which points are forward (blue) vs reverse (red) sweeps.
  </p>
  <button class="accv-btn orange" onclick="accvRecalcSweeps(false)">
    &#8635; Recalculate Forward / Reverse Sweeps
  </button>
  <div id="accv-sweep-info" style="font-size:11px;color:#555;margin-top:4px;"></div>
  <h3>Click-to-Remove Points</h3>
  <div style="margin-bottom:6px;">
    <button class="accv-btn blue" id="accv-pick-btn" onclick="accvTogglePick()">&#127919; Enable Click-Remove</button>
    <button class="accv-btn grey" onclick="accvRestoreRemoved()">&#8617; Restore Removed</button>
  </div>
  <div>Removed: <span id="accv-removed-count">0</span> points</div>
  <h3>Export</h3>
  <div style="display:flex;gap:6px;flex-wrap:wrap;">
    <button class="accv-btn green" onclick="accvExportCSV()">&#11015; CSV</button>
    <button class="accv-btn green" onclick="accvExportHTML()">&#11015; HTML</button>
  </div>
  <div id="accv-status"></div>
</div>

<script id="accv-editor-logic">
(function(){
"use strict";
const SIGNALS = ['R','Theta','DC'];
const MAX_UNDO = 30;

const state = {
  shift : { R: 0, Theta: 0, DC: 0 },
  gain  : { R: 1, Theta: 1, DC: 1 },
  clipLo: { R: null, Theta: null, DC: null },
  clipHi: { R: null, Theta: null, DC: null },
  cropL : 0, cropR : 0,
  removed: new Set(),
};
const undoStack = [];
let pickMode = false;

// rawData[i] = { x: Float64Array, y: Float64Array, name: str,
//                sig: 'R'|'Theta'|'DC'|null,
//                xtype: 'time'|'dc_voltage'|null,
//                plotlyIdx: i }
//
// For xtype === 'dc_voltage' (the scatter plots), we also store
// rawTimeY — the *original* time-ordered signal values (same length as rawDC_V)
// and rawDC_V — the raw DC voltage array, so we can re-apply shift in time
// and rebuild the V vs signal pair on the fly.
let rawData = null;

// Parallel full-length time arrays for each signal (extracted once)
// Keys: 'R', 'Theta', 'DC'
// Value: { x: Float64Array (time), y: Float64Array (signal) }
let rawTimeSeries = {};

// The full-length DC voltage array (time-ordered)
let rawDC_V_full = null;

let currentUpMask = null;
let currentDownMask = null;
let _lastMaskShift = null;
let _dcRange = 1;
let _dcFiltered = null;
let _dragSig = null;
let _forceRecalc = false;
let gd = null;
let _isDragging = false;
let _rafId = null;

function getPlotlyDiv() {
  if (gd && gd.data) return gd;
  gd = document.querySelector('.js-plotly-plot') ||
       document.querySelector('.plotly-graph-div') ||
       document.querySelector('[id^="plotly"]');
  return gd;
}

// ------------------------------------------------------------
// classifyTraceObj — determine signal and x-axis type
// ------------------------------------------------------------
function classifyTraceObj(tr) {
  const nm = (tr.name || '').toLowerCase().trim();

  if (nm === 'dc raw' || nm.startsWith('dc ramp') || nm.startsWith('dc (v)')) {
    return { sig: 'DC', xtype: 'time' };
  }
  if (nm === 'r (lock-in)') return { sig: 'R',     xtype: 'time' };
  if (nm === 'theta')        return { sig: 'Theta', xtype: 'time' };

  if (nm === 'forward sweep' || nm === 'reverse sweep') {
    const yref = tr.yaxis || 'y';
    const ynum = parseInt(yref.replace('y','') || '1');
    const sig = (ynum % 2 === 1) ? 'R' : 'Theta';
    const dir = nm.startsWith('forward') ? 'up' : 'down';
    const mode = (tr.mode || '').toLowerCase();
    const xtype = (mode.indexOf('line') >= 0 && mode.indexOf('marker') < 0)
      ? 'dc_voltage_line' : 'dc_voltage';
    return { sig, xtype, dir };
  }

  return { sig: null, xtype: null };
}

// ------------------------------------------------------------
// extractRaw — snapshot all trace data before any edits
// ------------------------------------------------------------
function extractRaw() {
  if (rawData) return;
  const div = getPlotlyDiv();
  if (!div || !div.data || !div.data.length) {
    console.warn('accv: Plotly div not ready yet');
    return;
  }

  rawData = div.data.map((tr, i) => {
    const cls = classifyTraceObj(tr);
    return {
      x   : Float64Array.from(tr.x || []),
      y   : Float64Array.from(tr.y || []),
      name: tr.name || '',
      sig : cls.sig,
      xtype: cls.xtype,
      dir : cls.dir || null,
      plotlyIdx: i,
    };
  });

  // Determine full-length N from the longest time-domain trace
  let N = 0;
  rawData.forEach(t => {
    if (t.xtype === 'time' && t.y.length > N) N = t.y.length;
  });
  if (!N) N = rawData.reduce((m, t) => Math.max(m, t.y.length), 0);
  rawData._N = N;

  // Cache per-signal full-length time series
  rawData.forEach(t => {
    if (t.xtype === 'time' && t.sig && !rawTimeSeries[t.sig]) {
      rawTimeSeries[t.sig] = { x: t.x, y: t.y };
    }
  });

  // Cache the DC voltage time series (used to rebuild scatter plots after shift)
  if (rawTimeSeries['DC']) {
    rawDC_V_full = rawTimeSeries['DC'].y;  // voltage indexed by sample
  } else {
    // Fallback: look for the DC raw trace directly
    for (const t of rawData) {
      if (t.sig === 'DC' && t.xtype === 'time') {
        rawDC_V_full = t.y; break;
      }
    }
  }

  if (rawDC_V_full) {
    let vmin = rawDC_V_full[0], vmax = rawDC_V_full[0];
    for (let i = 1; i < rawDC_V_full.length; i++) {
      if (rawDC_V_full[i] < vmin) vmin = rawDC_V_full[i];
      if (rawDC_V_full[i] > vmax) vmax = rawDC_V_full[i];
    }
    _dcRange = Math.max(vmax - vmin, 1e-6);
    _dcFiltered = movingAverageDC(rawDC_V_full, Math.max(3, Math.round(rawDC_V_full.length / 200)));
    syncMasksToDcShift(0, true);
  }

  // Update crop slider maxima
  const half = Math.floor(N / 2);
  ['crop-left','crop-right','crop-left-n','crop-right-n'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.max = half; }
  });

  console.log('accv: extracted', rawData.length, 'traces, N =', N);
}

function movingAverageDC(src, halfWin) {
  const n = src.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    const lo = Math.max(0, i - halfWin), hi = Math.min(n - 1, i + halfWin);
    for (let j = lo; j <= hi; j++) { s += src[j]; c++; }
    out[i] = s / c;
  }
  return out;
}

function computeMasksFromDC(N, dcShift) {
  const up = new Uint8Array(N);
  const down = new Uint8Array(N);
  const srcArr = _dcFiltered || rawDC_V_full;
  const len = srcArr.length;
  for (let i = 0; i < N - 1; i++) {
    const a = i - dcShift, b = i + 1 - dcShift;
    if (a < 0 || b < 0 || a >= len || b >= len) {
      if (i > 0) { up[i] = up[i - 1]; down[i] = down[i - 1]; }
      else { up[i] = 1; }
      continue;
    }
    if (srcArr[b] > srcArr[a]) up[i] = 1;
    else down[i] = 1;
  }
  up[N - 1] = N > 1 ? up[N - 2] : 0;
  down[N - 1] = N > 1 ? down[N - 2] : 0;
  return { upMask: up, downMask: down };
}

function syncMasksToDcShift(forceShift, force) {
  if (!rawData || !rawDC_V_full) return;
  const dcShift = (forceShift !== undefined) ? (forceShift | 0) : (state.shift['DC'] | 0);
  if (!force && !_forceRecalc && _lastMaskShift === dcShift && currentUpMask) return;
  const m = computeMasksFromDC(rawData._N, dcShift);
  currentUpMask = m.upMask;
  currentDownMask = m.downMask;
  _lastMaskShift = dcShift;
}

function isPristine() {
  if (state.cropL || state.cropR || state.removed.size) return false;
  for (const s of SIGNALS) {
    if (state.shift[s] !== 0 || state.gain[s] !== 1) return false;
    if (state.clipLo[s] !== null || state.clipHi[s] !== null) return false;
  }
  return true;
}

function getDcV(i, dcShift) {
  const dcSrc = i - dcShift;
  if (dcSrc < 0 || dcSrc >= rawDC_V_full.length) return NaN;
  return rawDC_V_full[dcSrc];
}

function collectDcPts(sig, dir, sh, g, lo, hi, L, end, N, dcShift, step) {
  const sigY = rawTimeSeries[sig].y;
  const mask = (dir === 'up') ? currentUpMask : currentDownMask;
  const dcX = [], sigVals = [];
  for (let i = L; i < Math.min(end, N); i += step) {
    if (!mask[i] || state.removed.has(i)) continue;
    const src = i - sh;
    if (src < 0 || src >= sigY.length) continue;
    const ry = sigY[src];
    if (ry === undefined || isNaN(ry)) continue;
    const dcV = getDcV(i, dcShift);
    if (isNaN(dcV)) continue;
    let yv = ry * g;
    if ((lo !== null && yv < lo) || (hi !== null && yv > hi)) continue;
    dcX.push(dcV); sigVals.push(yv);
  }
  return { dcX, sigVals };
}

function interpSegToGrid(dc, sig, nGrid) {
  const n = dc.length;
  if (n < 2) return { x: [], y: [] };
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => dc[a] - dc[b]);
  const dcs = [], sigs = [];
  for (const i of order) {
    if (dcs.length && Math.abs(dc[i] - dcs[dcs.length - 1]) < 1e-9) continue;
    dcs.push(dc[i]); sigs.push(sig[i]);
  }
  if (dcs.length < 2) return { x: [], y: [] };
  const vLo = dcs[0], vHi = dcs[dcs.length - 1];
  const x = [], y = [];
  const ng = Math.max(2, nGrid);
  for (let gi = 0; gi < ng; gi++) {
    const v = vLo + gi * (vHi - vLo) / (ng - 1);
    let j = 0;
    while (j < dcs.length - 1 && dcs[j + 1] < v) j++;
    if (j >= dcs.length - 1) { x.push(v); y.push(sigs[dcs.length - 1]); continue; }
    const t = (v - dcs[j]) / (dcs[j + 1] - dcs[j]);
    x.push(v);
    y.push(sigs[j] + t * (sigs[j + 1] - sigs[j]));
  }
  return { x, y };
}

function collectDcLine(sig, dir, sh, g, lo, hi, L, end, N, dcShift, step, gridPts) {
  const sigY = rawTimeSeries[sig].y;
  const mask = (dir === 'up') ? currentUpMask : currentDownMask;
  const vBreak = Math.max(0.015, _dcRange * 0.04);
  const segs = [];
  let cur = { dc: [], sig: [] };
  let prevI = -999999, prevV = NaN;

  for (let i = L; i < Math.min(end, N); i += step) {
    if (!mask[i] || state.removed.has(i)) continue;
    const src = i - sh;
    if (src < 0 || src >= sigY.length) continue;
    const ry = sigY[src];
    if (ry === undefined || isNaN(ry)) continue;
    const dcV = getDcV(i, dcShift);
    if (isNaN(dcV)) continue;
    let yv = ry * g;
    if ((lo !== null && yv < lo) || (hi !== null && yv > hi)) continue;

    let breakSeg = false;
    if (cur.dc.length) {
      if (i - prevI > step * 4) breakSeg = true;
      else if (dir === 'up' && dcV < prevV - vBreak) breakSeg = true;
      else if (dir === 'down' && dcV > prevV + vBreak) breakSeg = true;
    }
    if (breakSeg) {
      if (cur.dc.length >= 2) segs.push(cur);
      cur = { dc: [], sig: [] };
    }
    cur.dc.push(dcV); cur.sig.push(yv);
    prevV = dcV; prevI = i;
  }
  if (cur.dc.length >= 2) segs.push(cur);

  const dcX = [], sigVals = [];
  for (let s = 0; s < segs.length; s++) {
    if (s > 0) { dcX.push(null); sigVals.push(null); }
    const interp = interpSegToGrid(segs[s].dc, segs[s].sig, gridPts);
    for (let k = 0; k < interp.x.length; k++) {
      dcX.push(interp.x[k]); sigVals.push(interp.y[k]);
    }
  }
  return { dcX, sigVals };
}

function traceNeedsUpdate(sig, xtype, light) {
  if (!light || !_dragSig) return true;
  if (xtype === 'time') {
    if (_dragSig === 'DC') return true;
    return sig === _dragSig || sig === 'DC';
  }
  return _dragSig === 'DC' || sig === _dragSig;
}

window.accvStartDrag = function(sig, el) {
  _dragSig = sig;
  accvPushUndoOnce(el);
  _isDragging = true;
};

window.accvOnShiftRelease = function() {
  _isDragging = false;
  _dragSig = null;
  scheduleRerender(true);
};

function updateSweepInfo() {
  const el = document.getElementById('accv-sweep-info');
  if (!el || !currentUpMask) return;
  let upCount = 0, downCount = 0;
  for (let i = 0; i < currentUpMask.length; i++) {
    upCount += currentUpMask[i];
    downCount += currentDownMask[i];
  }
  el.textContent = 'Forward: ' + upCount.toLocaleString() + ' pts  |  Reverse: ' + downCount.toLocaleString() + ' pts';
}

window.accvRecalcSweeps = function(skipUndo) {
  if (!rawData || !rawDC_V_full) { alert('No data loaded yet'); return; }
  if (skipUndo !== true) pushUndo();
  _forceRecalc = true;
  _lastMaskShift = null;
  syncMasksToDcShift(undefined, true);
  updateSweepInfo();
  rerender(true);
  _forceRecalc = false;
  document.getElementById('accv-status').textContent = 'Sweeps recalculated \u2713';
};

function scheduleRerender(full) {
  if (_rafId) return;
  _rafId = requestAnimationFrame(() => {
    _rafId = null;
    rerender(full !== false);
  });
}

function rerender(fullUpdate) {
  if (!rawData) { extractRaw(); if (!rawData) return; }
  const div = getPlotlyDiv();
  if (!div || !div.data) return;

  const N   = rawData._N;
  const L   = Math.max(0, Math.min(state.cropL, N - 2));
  const R   = Math.max(0, Math.min(state.cropR, N - L - 1));
  const end = N - R;
  const light = _isDragging || !fullUpdate;
  const forceAll = _forceRecalc;

  const dcShift = state.shift['DC'] | 0;
  const timeStep = light ? 3 : 1;
  const scatterStep = light ? 10 : 3;
  const lineStep = light ? 5 : 2;
  const lineGrid = light ? 35 : 70;
  const linePristine = !forceAll && isPristine();

  const batchIdx = [];
  const batchX   = [];
  const batchY   = [];

  rawData.forEach(raw => {
    if (!raw.sig) return;
    if ((raw.name || '').startsWith('__band__')) return;

    const { sig, xtype, dir, plotlyIdx } = raw;
    if (!forceAll && !traceNeedsUpdate(sig, xtype, light)) return;

    const sh = state.shift[sig] | 0;
    const g  = state.gain[sig];
    const lo = state.clipLo[sig];
    const hi = state.clipHi[sig];

    let newX, newY;

    if (xtype === 'time') {
      const len = raw.y.length;
      newX = [];
      newY = [];

      for (let i = L; i < Math.min(end, len); i += timeStep) {
        const xv = raw.x[i] !== undefined ? raw.x[i] : i;

        if (state.removed.has(i)) {
          newX.push(xv); newY.push(null); continue;
        }

        // Shift: read from raw.y[i - sh]
        const src = i - sh;
        if (src < 0 || src >= len) {
          newX.push(xv); newY.push(null); continue;
        }

        const rawY = raw.y[src];
        if (rawY === undefined || isNaN(rawY)) {
          newX.push(xv); newY.push(null); continue;
        }

        let yv = rawY * g;
        if ((lo !== null && yv < lo) || (hi !== null && yv > hi)) {
          newX.push(xv); newY.push(null); continue;
        }

        newX.push(xv);
        newY.push(yv);
      }

    } else if (xtype === 'dc_voltage_line') {
      if (linePristine) {
        newX = Array.from(raw.x);
        newY = Array.from(raw.y);
      } else {
        if (!rawDC_V_full || !rawTimeSeries[sig] || !currentUpMask) return;
        const pts = collectDcLine(sig, dir, sh, g, lo, hi, L, end, N, dcShift, lineStep, lineGrid);
        newX = pts.dcX; newY = pts.sigVals;
      }

    } else if (xtype === 'dc_voltage') {
      if (!rawDC_V_full || !rawTimeSeries[sig] || !currentUpMask) return;
      const { dcX, sigVals } = collectDcPts(sig, dir, sh, g, lo, hi, L, end, N, dcShift, scatterStep);
      newX = dcX; newY = sigVals;

    } else {
      return;  // unknown xtype
    }

    batchIdx.push(plotlyIdx);
    batchX.push(newX);
    batchY.push(newY);
  });

  // Single batched restyle call — much faster than one per trace
  if (batchIdx.length) {
    try {
      Plotly.restyle(div, { x: batchX, y: batchY, connectgaps: batchIdx.map(() => false) }, batchIdx);
    } catch(e) {
      console.error('accv batch restyle error:', e);
    }
  }

  document.getElementById('accv-removed-count').textContent = state.removed.size;
  if (light) return;
  document.getElementById('accv-status').textContent =
    'Updated ' + batchIdx.length + ' traces \u00b7 ' + state.removed.size + ' pts removed';
}

// ------------------------------------------------------------
// Undo / Reset
// ------------------------------------------------------------
function pushUndo() {
  if (undoStack.length >= MAX_UNDO) undoStack.shift();
  undoStack.push(JSON.stringify({
    shift: {...state.shift}, gain: {...state.gain},
    clipLo: {...state.clipLo}, clipHi: {...state.clipHi},
    cropL: state.cropL, cropR: state.cropR,
    removed: [...state.removed],
  }));
  updateUndoCount();
}
window.pushUndo = pushUndo;
window.accvPushUndoOnce = function(el) {
  if (el._accvUndoPushed) return;
  el._accvUndoPushed = true;
  pushUndo();
  el.addEventListener('pointerup', () => { el._accvUndoPushed = false; }, { once: true });
};

function updateUndoCount() {
  const n = undoStack.length;
  document.getElementById('accv-undo-count').textContent =
    n + ' action' + (n !== 1 ? 's' : '') + ' in stack';
}

window.accvUndo = function() {
  if (!undoStack.length) return;
  const s = JSON.parse(undoStack.pop());
  Object.assign(state.shift,  s.shift);
  Object.assign(state.gain,   s.gain);
  Object.assign(state.clipLo, s.clipLo);
  Object.assign(state.clipHi, s.clipHi);
  state.cropL   = s.cropL;
  state.cropR   = s.cropR;
  state.removed = new Set(s.removed);
  _lastMaskShift = null;
  syncMasksToDcShift(undefined, true);
  syncUI();
  updateSweepInfo();
  rerender(true);
  updateUndoCount();
};

window.accvReset = function() {
  pushUndo();
  SIGNALS.forEach(s => {
    state.shift[s]  = 0;
    state.gain[s]   = 1;
    state.clipLo[s] = null;
    state.clipHi[s] = null;
  });
  state.cropL = 0;
  state.cropR = 0;
  state.removed.clear();
  _lastMaskShift = null;
  syncUI();
  syncMasksToDcShift(0, true);
  updateSweepInfo();
  rerender(true);
};

// ------------------------------------------------------------
// syncUI
// ------------------------------------------------------------
function syncUI() {
  SIGNALS.forEach(sig => {
    const sl = document.getElementById('shift-' + sig);
    const sn = document.getElementById('shift-' + sig + '-n');
    const gl = document.getElementById('gain-'  + sig);
    const gn = document.getElementById('gain-'  + sig + '-n');
    if (sl) sl.value = state.shift[sig];
    if (sn) sn.value = state.shift[sig];
    if (gl) gl.value = state.gain[sig];
    if (gn) gn.value = state.gain[sig];
    const clo = document.getElementById('cliplo-' + sig);
    const chi = document.getElementById('cliphi-' + sig);
    if (clo) clo.value = state.clipLo[sig] !== null ? state.clipLo[sig] : '';
    if (chi) chi.value = state.clipHi[sig] !== null ? state.clipHi[sig] : '';
  });
  document.getElementById('crop-left').value    = state.cropL;
  document.getElementById('crop-left-n').value  = state.cropL;
  document.getElementById('crop-right').value   = state.cropR;
  document.getElementById('crop-right-n').value = state.cropR;
  document.getElementById('accv-removed-count').textContent = state.removed.size;
}

// ------------------------------------------------------------
// Per-signal controls
// ------------------------------------------------------------
window.accvSetShift = function(sig, val) {
  state.shift[sig] = parseInt(val) || 0;
  document.getElementById('shift-' + sig).value     = state.shift[sig];
  document.getElementById('shift-' + sig + '-n').value = state.shift[sig];
  scheduleRerender(_isDragging ? false : true);
};
window.accvSetGain = function(sig, val) {
  state.gain[sig] = parseFloat(val) || 1;
  document.getElementById('gain-' + sig).value     = state.gain[sig];
  document.getElementById('gain-' + sig + '-n').value = state.gain[sig];
  scheduleRerender(_isDragging ? false : true);
};
window.accvSetClipLo = function(sig, val) {
  pushUndo();
  state.clipLo[sig] = (val === '' || val === null) ? null : parseFloat(val);
  scheduleRerender(true);
};
window.accvSetClipHi = function(sig, val) {
  pushUndo();
  state.clipHi[sig] = (val === '' || val === null) ? null : parseFloat(val);
  scheduleRerender(true);
};
window.accvSetCrop = function(side, val) {
  pushUndo();
  const v = parseInt(val) || 0;
  if (side === 'left') {
    state.cropL = v;
    document.getElementById('crop-left').value   = v;
    document.getElementById('crop-left-n').value = v;
  } else {
    state.cropR = v;
    document.getElementById('crop-right').value   = v;
    document.getElementById('crop-right-n').value = v;
  }
  scheduleRerender(true);
};

// ------------------------------------------------------------
// Click-to-remove (time-domain only)
// ------------------------------------------------------------
window.accvTogglePick = function() {
  pickMode = !pickMode;
  const btn = document.getElementById('accv-pick-btn');
  btn.textContent = pickMode
    ? '\uD83D\uDD34 Click-Remove ON (click again to stop)'
    : '\uD83C\uDFAF Enable Click-Remove';
  btn.style.background = pickMode ? '#B71C1C' : '#1565C0';
  const div = getPlotlyDiv();
  if (!div) return;
  if (pickMode) div.on('plotly_click', onPlotlyClick);
  else          div.removeAllListeners('plotly_click');
};

function onPlotlyClick(data) {
  if (!data.points || !data.points.length || !rawData) return;
  pushUndo();

  data.points.forEach(pt => {
    const trName  = (pt.data && pt.data.name) ? pt.data.name : '';
    const cls = { xtype: null };
    for (const rt of rawData) {
      if (rt.name === trName) { cls.xtype = rt.xtype; break; }
    }
    if (cls.xtype !== 'time') return;

    const clickedX = pt.x;
    let absIdx = -1;

    for (const rt of rawData) {
      if (rt.name !== trName || rt.xtype !== 'time') continue;
      let best = Infinity, bestI = -1;
      const lo = state.cropL, hi = rawData._N - state.cropR;
      for (let i = lo; i < hi; i++) {
        const d = Math.abs(rt.x[i] - clickedX);
        if (d < best) { best = d; bestI = i; }
      }
      if (bestI >= 0) { absIdx = bestI; break; }
    }

    if (absIdx < 0) absIdx = (pt.pointIndex || 0) + state.cropL;
    if (absIdx >= 0 && absIdx < rawData._N) state.removed.add(absIdx);
  });

  document.getElementById('accv-removed-count').textContent = state.removed.size;
  rerender(true);
}

window.accvRestoreRemoved = function() {
  pushUndo();
  state.removed.clear();
  rerender(true);
};

// ------------------------------------------------------------
// Panel toggle
// ------------------------------------------------------------
window.accvTogglePanel = function() {
  const panel = document.getElementById('accv-panel');
  const btn   = document.getElementById('accv-edit-toggle');
  const open  = panel.classList.toggle('open');
  btn.classList.toggle('active', open);
  btn.textContent = open ? '\u2715 Close Editor' : '\u2699 Edit Mode';
  if (open) setTimeout(() => { extractRaw(); buildSignalUI(); }, 400);
};

function buildSignalUI() {
  const container = document.getElementById('accv-signals');
  if (!container) return;
  const colors   = { R: '#1565C0', Theta: '#6A1B9A', DC: '#E65100' };
  const shiftMax = rawData ? Math.floor(rawData._N / 4) : 5000;
  container.innerHTML = '';
  SIGNALS.forEach(sig => {
    const block = document.createElement('div');
    block.className = 'accv-signal-block';
    block.innerHTML = `
      <div class="sig-title" style="color:${colors[sig]||'#333'}">&#9616; ${sig}</div>
      <div class="accv-row">
        <label>Shift (samples)</label>
        <input type="range"  id="shift-${sig}"   min="${-shiftMax}" max="${shiftMax}" step="1" value="0"
               onpointerdown="accvStartDrag('${sig}', this)"
               onpointerup="accvOnShiftRelease()"
               oninput="accvSetShift('${sig}', this.value)">
        <input type="number" id="shift-${sig}-n" min="${-shiftMax}" max="${shiftMax}" step="1" value="0"
               onchange="pushUndo();accvSetShift('${sig}', this.value)">
      </div>
      <div class="accv-row">
        <label>Gain (\u00d7)</label>
        <input type="range"  id="gain-${sig}"   min="0.01" max="10" step="0.01" value="1"
               onpointerdown="accvStartDrag('${sig}', this)"
               onpointerup="accvOnShiftRelease()"
               oninput="accvSetGain('${sig}', this.value)">
        <input type="number" id="gain-${sig}-n" min="0.01" max="100" step="0.001" value="1"
               onchange="accvSetGain('${sig}', this.value)">
      </div>
      <div class="accv-row">
        <label>Y-Clip Low</label>
        <input type="number" id="cliplo-${sig}" placeholder="none" step="any"
               onchange="accvSetClipLo('${sig}', this.value)">
      </div>
      <div class="accv-row">
        <label>Y-Clip High</label>
        <input type="number" id="cliphi-${sig}" placeholder="none" step="any"
               onchange="accvSetClipHi('${sig}', this.value)">
      </div>`;
    container.appendChild(block);
  });
}

// ------------------------------------------------------------
// Export
// ------------------------------------------------------------
window.accvExportCSV = function() {
  if (!rawData) { alert('No data available'); return; }
  const div = getPlotlyDiv();
  if (!div || !div.data) { alert('No plot data'); return; }

  const rendered = {};
  div.data.forEach(tr => {
    const cls = classifyTraceObj(tr);
    if (cls.sig && cls.xtype === 'time' && !rendered[cls.sig]) {
      rendered[cls.sig] = { x: tr.x, y: tr.y };
    }
  });

  const ref = rendered.R || rendered.Theta || rendered.DC;
  if (!ref) { alert('No time-domain traces found'); return; }
  const rows = ['Time,R,Theta,DC'];
  for (let i = 0; i < ref.x.length; i++) {
    const t  = ref.x[i];
    const rv = (rendered.R     && rendered.R.y[i]     != null) ? rendered.R.y[i]     : '';
    const tv = (rendered.Theta && rendered.Theta.y[i] != null) ? rendered.Theta.y[i] : '';
    const dv = (rendered.DC    && rendered.DC.y[i]    != null) ? rendered.DC.y[i]    : '';
    if (rv === '' && tv === '' && dv === '') continue;
    rows.push(`${t},${rv},${tv},${dv}`);
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'accv_edited_' + new Date().toISOString().replace(/[:.]/g, '-') + '.csv';
  a.click();
  document.getElementById('accv-status').textContent = 'CSV exported \u2713';
};

window.accvExportHTML = function() {
  document.getElementById('accv-status').textContent = 'Saving HTML\u2026';
  const div = getPlotlyDiv();
  if (!div || !div.data) { alert('No plot data'); return; }
  const plotId = div.id || ('accv_plot_' + Date.now());
  const layout = div.layout || {};
  const h = layout.height || 1450;
  const w = layout.width || '100%';
  const wCss = (typeof w === 'number') ? (w + 'px') : w;
  let styleHtml = '';
  document.querySelectorAll('style').forEach(s => {
    if (s.textContent && s.textContent.indexOf('accv-edit-toggle') >= 0) styleHtml = s.outerHTML;
  });
  const btn = document.getElementById('accv-edit-toggle');
  const panel = document.getElementById('accv-panel');
  const editorHtml = (btn ? btn.outerHTML : '') + (panel ? panel.outerHTML : '');
  const logicEl = document.getElementById('accv-editor-logic');
  const editorScript = logicEl ? logicEl.textContent : '';
  if (!editorScript) { alert('Editor script missing'); return; }
  const esc = s => JSON.stringify(s).replace(/<\//g, '\\u003c/');
  const dataStr = esc(div.data);
  const layoutStr = esc(layout);
  const cfgStr = esc({ responsive: true });
  const safeScript = editorScript.replace(/<\/script/gi, '<\\/script');
  const html = '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
    + styleHtml + '\n'
    + '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"><\/script>\n'
    + '</head>\n<body style="margin:0;">\n'
    + '<div id="' + plotId + '" class="plotly-graph-div" style="width:' + wCss + ';height:' + h + 'px;"></div>\n'
    + '<script>Plotly.newPlot("' + plotId + '",' + dataStr + ',' + layoutStr + ',' + cfgStr + ');<\/script>\n'
    + editorHtml + '\n'
    + '<script id="accv-editor-logic">\n' + safeScript + '\n<\/script>\n'
    + '</body>\n</html>';
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'accv_edited_' + new Date().toISOString().replace(/[:.]/g, '-') + '.html';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
  document.getElementById('accv-status').textContent = 'HTML exported \u2713';
};

// ------------------------------------------------------------
// Boot — wait for Plotly to finish rendering before snapshotting
// ------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function() {
  function poll() {
    const div = getPlotlyDiv();
    if (div && div.data && div.data.length > 0) {
        extractRaw();
        buildSignalUI();
        accvRecalcSweeps(true);
    } else {
      setTimeout(poll, 600);
    }
  }
  setTimeout(poll, 1000);
});

})();
</script>
"""


def inject_editor(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    if 'accv-edit-toggle' in html:
        print(f"Editor already present in {html_path}, skipping injection.")
        return
    if '</body>' in html:
        html = html.replace('</body>', EDITOR_JS + '\n</body>', 1)
    else:
        html += EDITOR_JS
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Editor injected → {html_path}")


# ---------------------------------------------------------------------------
# Method A - cycle interpolation (smooth line + sigma band)
# ---------------------------------------------------------------------------

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
    if Smooth_Window and valid.sum() > Smooth_Window:
        smoothed = mean_tr.copy()
        smoothed[valid] = savgol_filter(mean_tr[valid], window_length=Smooth_Window, polyorder=3)
        mean_tr = smoothed
    return mean_tr, std_tr


# ---------------------------------------------------------------------------
# Method B - MATLAB bin-average (scatter, full coverage)
# ---------------------------------------------------------------------------

def filter_dc_ramp(dc_v):
    b, a = butter(Dc_Filter_Order, Dc_Filter_Cutoff, btype='low')
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
        if len(pts) >= Min_Pts:
            means[b] = np.mean(pts)
    valid = ~np.isnan(means)
    return v_grid[valid], means[valid]


# ---------------------------------------------------------------------------
# Main plot - 4 columns x 4 rows
# ---------------------------------------------------------------------------

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

    peaks, troughs = detect_triangle_cycles(dc_v)
    print(f"Detected {len(peaks)} peaks, {len(troughs)} troughs")
    cycles = build_cycle_triples(peaks, troughs, dc_v)
    print(f"Cycles: {len(cycles)}")

    v_grid = np.linspace(dc_v.min(), dc_v.max(), N_Grid_Points)

    R_up_mn, R_up_sd = cycle_average_line(dc_v, li_R,     cycles, v_grid, 'up')
    R_dn_mn, R_dn_sd = cycle_average_line(dc_v, li_R,     cycles, v_grid, 'down')
    T_up_mn, T_up_sd = cycle_average_line(dc_v, li_Theta, cycles, v_grid, 'up')
    T_dn_mn, T_dn_sd = cycle_average_line(dc_v, li_Theta, cycles, v_grid, 'down')

    dc_filt            = filter_dc_ramp(dc_v)
    up_mask, down_mask = split_sweeps(dc_filt)
    print(f"Up pts: {up_mask.sum():,}   Down pts: {down_mask.sum():,}")

    v_Rs_up, Rs_up = bin_average(dc_v[up_mask],   li_R[up_mask],       v_grid)
    v_Rs_dn, Rs_dn = bin_average(dc_v[down_mask], li_R[down_mask],     v_grid)
    v_Ts_up, Ts_up = bin_average(dc_v[up_mask],   li_Theta[up_mask],   v_grid)
    v_Ts_dn, Ts_dn = bin_average(dc_v[down_mask], li_Theta[down_mask], v_grid)

    t_axis    = np.arange(len(dc_v))
    t_seconds = time_s if time_s is not None else t_axis

    Up_C  = '#1565C0'
    Dn_C  = '#B71C1C'
    Up_Ct = 'rgba(21,101,192,0.18)'
    Dn_Ct = 'rgba(183,28,28,0.18)'
    Up_L  = 'Forward sweep'
    Dn_L  = 'Reverse sweep'
    Dc_C  = '#E65100'
    R_C   = '#1565C0'
    Th_C  = '#6A1B9A'

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("pip install plotly")

    n_cyc = len(cycles)
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=(
            'R vs Time  |  DC Ramp overlay',
            'Theta vs Time  |  DC Ramp overlay',
            'R vs Time  (no overlay)',
            'Theta vs Time  (no overlay)',
            f'R vs DC  -  Raw Data  ({len(dc_v):,} pts)',
            f'Theta vs DC  -  Raw Data  ({len(dc_v):,} pts)',
            f'R vs DC  -  Cycle-Averaged  ({n_cyc} cycles)  +/-1sig',
            f'Theta vs DC  -  Cycle-Averaged  ({n_cyc} cycles)  +/-1sig',
            'R vs DC  -  Bin-Averaged Scatter',
            'Theta vs DC  -  Bin-Averaged Scatter',
            'R vs DC  -  Smooth Averaged Line',
            'Theta vs DC  -  Smooth Averaged Line',
            'DC Voltage vs Time  (raw)',
            '', '', '',
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": True},
             {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False},
             {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False},
             {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False},
             {"secondary_y": False}, {"secondary_y": False}],
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.07,
    )

    title_str = 'AC Cyclic Voltammetry  -  Full Analysis'
    if n_samples is not None:
        title_str += (f'<br><sup>{n_samples:,} raw samples  |  '
                      f'{n_cyc} cycles  |  '
                      f'up: {up_mask.sum():,}  down: {down_mask.sum():,}</sup>')

    def _tline(row, col, x, y, color, name, secondary=False, showlegend=True):
        fig.add_trace(go.Scattergl(
            x=x.tolist(), y=y.tolist(),
            mode='lines', line=dict(color=color, width=1.0),
            name=name, showlegend=showlegend,
            hovertemplate='t=%{x}  val=%{y:.6f}<extra>' + name + '</extra>',
        ), row=row, col=col, secondary_y=secondary)

    def _raw(row, col, mask, signal, color, name, showlegend=True):
        idx = np.where(mask)[0][::Raw_Ds]
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
            hoverinfo='skip', showlegend=False, name='__band__',
        ), row=row, col=col)

    def _line(row, col, mean, color, name, showlegend=True):
        msk = ~np.isnan(mean)
        fig.add_trace(go.Scattergl(
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

    # Row 1
    _tline(1, 1, t_axis[::Raw_Ds], li_R[::Raw_Ds],        R_C,  'R (lock-in)',  secondary=False)
    _tline(1, 1, t_axis[::Raw_Ds], dc_v[::Raw_Ds],        Dc_C, 'DC Ramp (V)', secondary=True)
    _tline(1, 2, t_axis[::Raw_Ds], li_Theta[::Raw_Ds],    Th_C, 'Theta',        secondary=False, showlegend=False)
    _tline(1, 2, t_axis[::Raw_Ds], dc_v[::Raw_Ds],        Dc_C, 'DC Ramp (V)', secondary=True,  showlegend=False)
    _tline(1, 3, t_seconds[::Raw_Ds], li_R[::Raw_Ds],     R_C,  'R (lock-in)',  showlegend=False)
    _tline(1, 4, t_seconds[::Raw_Ds], li_Theta[::Raw_Ds], Th_C, 'Theta',        showlegend=False)

    # Row 2 — raw scatter (dc_voltage xtype); name must be 'Forward sweep'/'Reverse sweep'
    # so the JS can re-pair them with the time-series after shift.
    _raw(2, 1, up_mask,   li_R,     Up_C, Up_L)
    _raw(2, 1, down_mask, li_R,     Dn_C, Dn_L)
    _raw(2, 2, up_mask,   li_Theta, Up_C, Up_L, showlegend=False)
    _raw(2, 2, down_mask, li_Theta, Dn_C, Dn_L, showlegend=False)
    _band(2, 3, R_up_mn, R_up_sd, Up_Ct);  _band(2, 3, R_dn_mn, R_dn_sd, Dn_Ct)
    _line(2, 3, R_up_mn, Up_C, Up_L, showlegend=False)
    _line(2, 3, R_dn_mn, Dn_C, Dn_L, showlegend=False)
    _band(2, 4, T_up_mn, T_up_sd, Up_Ct);  _band(2, 4, T_dn_mn, T_dn_sd, Dn_Ct)
    _line(2, 4, T_up_mn, Up_C, Up_L, showlegend=False)
    _line(2, 4, T_dn_mn, Dn_C, Dn_L, showlegend=False)

    # Row 3
    _scatter(3, 1, v_Rs_up, Rs_up, Up_C, Up_L)
    _scatter(3, 1, v_Rs_dn, Rs_dn, Dn_C, Dn_L)
    _scatter(3, 2, v_Ts_up, Ts_up, Up_C, Up_L, showlegend=False)
    _scatter(3, 2, v_Ts_dn, Ts_dn, Dn_C, Dn_L, showlegend=False)
    _line(3, 3, R_up_mn, Up_C, Up_L, showlegend=False)
    _line(3, 3, R_dn_mn, Dn_C, Dn_L, showlegend=False)
    _line(3, 4, T_up_mn, Up_C, Up_L, showlegend=False)
    _line(3, 4, T_dn_mn, Dn_C, Dn_L, showlegend=False)

    # Row 4
    t_ds = t_seconds[::Raw_Ds].tolist()
    fig.add_trace(go.Scattergl(
        x=t_ds, y=dc_v[::Raw_Ds].tolist(),
        mode='lines', line=dict(color=Dc_C, width=1.0),
        name='DC raw', showlegend=True,
        hovertemplate='t=%{x:.4f}  V=%{y:.6f}<extra>DC raw</extra>',
    ), row=4, col=1)

    for col in (1, 2):
        fig.update_xaxes(title_text='Sample Index', row=1, col=col)
    fig.update_yaxes(title_text='AC Magnitude R (V)',          row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='DC Ramp (V)',                 row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text='DC Ramp (V)',                 row=1, col=2, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text='Time (s)', row=1, col=3)
    fig.update_xaxes(title_text='Time (s)', row=1, col=4)
    fig.update_yaxes(title_text='AC Magnitude R (V)',          row=1, col=3)
    fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=1, col=4)
    for row in (2, 3):
        for col in (1, 2, 3, 4):
            fig.update_xaxes(title_text='DC Potential (V)', row=row, col=col)
        fig.update_yaxes(title_text='AC Magnitude R (V)',          row=row, col=1)
        fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=row, col=2)
        fig.update_yaxes(title_text='AC Magnitude R (V)',          row=row, col=3)
        fig.update_yaxes(title_text=f'Phase Angle ({theta_unit})', row=row, col=4)
    t_label = 'Time (s)' if time_s is not None else 'Sample Index'
    fig.update_xaxes(title_text=t_label,          row=4, col=1)
    fig.update_yaxes(title_text='DC Voltage (V)', row=4, col=1)

    fig.update_layout(
        title=dict(text=title_str, x=0.5, xanchor='center', font=dict(size=15)),
        height=1450, width=2000,
        template='plotly_white',
        hovermode='closest',
        margin=dict(t=90, b=70),
        legend=dict(x=0.5, y=0.01, xanchor='center', yanchor='top', orientation='h',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='grey', borderwidth=1),
    )

    html_path = None
    if Save_Png:
        os.makedirs(output_dir, exist_ok=True)
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = os.path.join(output_dir, f'accv_cycle_averaged_{timestamp_str}.html')
        fig.write_html(html_path)
        inject_editor(html_path)
        print(f"Saved interactive HTML: {html_path}")
        try:
            png_path = os.path.join(output_dir, f'accv_cycle_averaged_{timestamp_str}.png')
            fig.write_image(png_path, scale=2)
            print(f"Saved static PNG:       {png_path}")
        except Exception:
            print("Note: static PNG skipped (pip install kaleido to enable)")

    return fig, html_path


if __name__ == '__main__':

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    elif Csv_Path:
        csv_path = Csv_Path
    else:
        try:
            csv_path = max(
                glob.glob(os.path.join('test_data', 'combined_results_*.csv')),
                key=os.path.getctime)
        except ValueError:
            print("No CSV found. Set Csv_Path at the top.")
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

    if 'Time_RP1' in df.columns:
        time_s = df['Time_RP1'].values
        time_s = time_s - time_s[0]
    elif 'Time_RP2' in df.columns:
        time_s = df['Time_RP2'].values
        time_s = time_s - time_s[0]
    else:
        time_s = None

    n_ds = len(df)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, html_path = plot_accv_cycle_averaged(
        dc_v, li_R, li_Theta,
        time_s=time_s,
        theta_unit='deg',
        timestamp_str=ts,
        output_dir=Output_Directory or None,
        csv_path=csv_path,
        n_samples=n_ds,
        n_ds=n_ds)

    if html_path and os.path.exists(html_path):
        url = 'file:///' + html_path.replace('\\', '/')
        print(f"Opening in browser: {url}")
        webbrowser.open(url)
    elif fig is not None:
        print("Warning: HTML not saved; falling back to fig.show() — editor button will not appear.")
        fig.show()

    print("Done.")
