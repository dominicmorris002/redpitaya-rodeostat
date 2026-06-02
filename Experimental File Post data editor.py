"""
ACCV HTML Editor Injector
=========================
Injects a full interactive Edit Mode panel into a Plotly-generated ACCV HTML file.

Usage (standalone post-processor):
    python accv_html_editor.py path/to/accv_cycle_averaged_TIMESTAMP.html

Or import and call from your existing code:
    from accv_html_editor import inject_editor
    inject_editor("path/to/output.html")          # overwrites in-place
    inject_editor("path/to/output.html", "path/to/output_editable.html")

Edit-mode features (all update all 16 subplots live):
  - Per-signal SHIFT  (sample-index offset, positive = shift right)
  - Per-signal GAIN   (multiply signal values)
  - Per-signal Y-CLIP min/max thresholds
  - Global CROP       (trim N samples from left / right of dataset)
  - Click-to-remove points on any subplot
  - UNDO stack (unlimited)
  - Export corrected CSV
  - Export new baked HTML with edits applied
"""

import os
import sys
import re

# ---------------------------------------------------------------------------
# The JavaScript + CSS edit panel (injected before </body>)
# ---------------------------------------------------------------------------

EDITOR_JS = r"""
<style>
/* ── Edit Mode Panel ──────────────────────────────────────────── */
#accv-edit-toggle {
  position: fixed; top: 12px; right: 12px; z-index: 9999;
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
  z-index: 9998; overflow-y: auto;
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
.accv-row span.val { width: 58px; text-align: right; color: #1565C0; font-weight: 600; font-size: 11px; }

.accv-btn {
  display: inline-block; margin: 4px 4px 0 0;
  padding: 6px 12px; border: none; border-radius: 4px;
  font-size: 12px; font-weight: 600; cursor: pointer;
}
.accv-btn.blue  { background:#1565C0; color:#fff; }
.accv-btn.red   { background:#B71C1C; color:#fff; }
.accv-btn.grey  { background:#607D8B; color:#fff; }
.accv-btn.green { background:#2E7D32; color:#fff; }
.accv-btn:hover { opacity:0.85; }

#accv-removed-count { color:#B71C1C; font-weight:700; }
#accv-undo-count    { color:#555; font-size:11px; }
#accv-status        { margin-top:8px; color:#2E7D32; font-weight:600; min-height:18px; }

.accv-signal-block { background:#fff; border:1px solid #e0e0e0;
  border-radius:6px; padding:10px; margin-bottom:10px; }
.accv-signal-block .sig-title { font-weight:700; color:#333; margin-bottom:6px; font-size:12px; }
</style>

<button id="accv-edit-toggle" onclick="accvTogglePanel()">⚙ Edit Mode</button>

<div id="accv-panel">
  <h2>⚙ ACCV Edit Mode</h2>

  <!-- Undo / Reset -->
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
    <button class="accv-btn grey"  onclick="accvUndo()">↩ Undo</button>
    <button class="accv-btn red"   onclick="accvReset()">✕ Reset All</button>
    <span id="accv-undo-count">0 actions in stack</span>
  </div>

  <!-- Global crop -->
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

  <!-- Per-signal controls -->
  <h3>Per-Signal Controls</h3>
  <div id="accv-signals"></div>

  <!-- Click-to-remove -->
  <h3>Click-to-Remove Points</h3>
  <div style="margin-bottom:6px;">
    <button class="accv-btn blue"  id="accv-pick-btn" onclick="accvTogglePick()">🎯 Enable Click-Remove</button>
    <button class="accv-btn grey"  onclick="accvRestoreRemoved()">↩ Restore Removed</button>
  </div>
  <div>Removed: <span id="accv-removed-count">0</span> points</div>

  <!-- Export -->
  <h3>Export</h3>
  <div style="display:flex;gap:6px;flex-wrap:wrap;">
    <button class="accv-btn green" onclick="accvExportCSV()">⬇ CSV</button>
    <button class="accv-btn green" onclick="accvExportHTML()">⬇ HTML</button>
  </div>

  <div id="accv-status"></div>
</div>

<script>
(function(){
"use strict";

// ── State ─────────────────────────────────────────────────────────
const SIGNALS = ['R','Theta','DC'];
const state = {
  shift : { R: 0, Theta: 0, DC: 0 },
  gain  : { R: 1, Theta: 1, DC: 1 },
  clipLo: { R: null, Theta: null, DC: null },
  clipHi: { R: null, Theta: null, DC: null },
  cropL : 0,
  cropR : 0,
  removed: new Set(),
};
const undoStack = [];
let pickMode = false;

// ── Raw data extracted from Plotly ────────────────────────────────
let rawData = null;   // { R: Float64Array, Theta: Float64Array, DC: Float64Array, t: Float64Array }
let gd      = null;   // the plotly div

function getPlotlyDiv() {
  if (gd) return gd;
  gd = document.querySelector('.plotly-graph-div') ||
       document.querySelector('[id^="plotly"]') ||
       document.getElementById('chart') ||
       document.querySelector('div[data-plotly]');
  return gd;
}

// Extract raw arrays from Plotly traces on first run
function extractRaw() {
  if (rawData) return;
  const div = getPlotlyDiv();
  if (!div || !div.data) { console.warn('Plotly div not ready'); return; }

  // Find time-series traces (row 1 col 1: R vs index, DC overlay)
  // We rely on trace names set in the Python code
  let R_arr=null, Th_arr=null, DC_arr=null, t_arr=null;

  div.data.forEach(tr => {
    const nm = (tr.name||'').toLowerCase();
    if (!R_arr  && nm.includes('r (lock')) { R_arr  = Float64Array.from(tr.y); t_arr = Float64Array.from(tr.x); }
    if (!Th_arr && nm.includes('theta'))   { Th_arr = Float64Array.from(tr.y); }
    if (!DC_arr && nm.includes('dc ramp')) { DC_arr = Float64Array.from(tr.y); }
  });

  if (!R_arr || !Th_arr || !DC_arr) {
    console.warn('Could not find all signal traces by name. Trying by index.');
    // fallback: grab first 3 traces
    if (div.data.length >= 2) {
      R_arr  = R_arr  || Float64Array.from(div.data[0].y||[]);
      Th_arr = Th_arr || Float64Array.from(div.data[1].y||[]);
      DC_arr = DC_arr || Float64Array.from(div.data[2]?.y||div.data[1].y||[]);
      t_arr  = t_arr  || Float64Array.from(div.data[0].x||[]);
    }
  }

  const N = R_arr ? R_arr.length : 0;
  if (!t_arr || t_arr.length !== N) {
    t_arr = new Float64Array(N); for(let i=0;i<N;i++) t_arr[i]=i;
  }
  rawData = { R: R_arr, Theta: Th_arr, DC: DC_arr, t: t_arr, N };
  // Update crop slider max
  ['crop-left','crop-right','crop-left-n','crop-right-n'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.max = Math.floor(N/2);
  });
  console.log(`ACCV Editor: extracted ${N} samples.`);
}

// ── Apply state → derive working arrays ───────────────────────────
function derive() {
  if (!rawData) { extractRaw(); if (!rawData) return null; }
  const { R, Theta, DC, t, N } = rawData;
  const L = state.cropL, Rr = state.cropR;
  const end = N - Rr;

  function shifted(arr, sig) {
    const sh = state.shift[sig]|0;
    const out = new Float64Array(end - L);
    for (let i=L; i<end; i++) {
      const src = i - sh;
      out[i-L] = (src>=0 && src<N) ? arr[src] : NaN;
    }
    return out;
  }
  function gained(arr, sig) {
    const g = state.gain[sig];
    const lo = state.clipLo[sig], hi = state.clipHi[sig];
    return arr.map(v => {
      const gv = v * g;
      if (lo !== null && gv < lo) return NaN;
      if (hi !== null && gv > hi) return NaN;
      return gv;
    });
  }

  const tW   = t.slice(L, end);
  const rW   = gained(shifted(R,     'R'),     'R');
  const thW  = gained(shifted(Theta, 'Theta'), 'Theta');
  const dcW  = gained(shifted(DC,    'DC'),    'DC');

  // mask removed points
  const rem = state.removed;
  const mask = new Uint8Array(tW.length);
  rem.forEach(gi => { const li = gi-L; if(li>=0 && li<mask.length) mask[li]=1; });

  return { t: tW, R: rW, Theta: thW, DC: dcW, mask, L, N: tW.length };
}

// ── Re-render all Plotly traces ───────────────────────────────────
function rerender() {
  const d = derive();
  if (!d) return;
  const div = getPlotlyDiv();
  if (!div || !div.data) return;

  const { t, R, Theta, DC, mask } = d;
  function masked(arr) {
    return Array.from(arr).map((v,i)=>mask[i]?null:v);
  }

  const updates = {};
  div.data.forEach((tr, idx) => {
    const nm = (tr.name||'').toLowerCase();
    let xArr=null, yArr=null;
    if (nm.includes('r (lock') || nm.includes('r vs')) {
      xArr = Array.from(t); yArr = masked(R);
    } else if (nm.includes('theta')) {
      xArr = Array.from(t); yArr = masked(Theta);
    } else if (nm.includes('dc ramp') || nm.includes('dc raw')) {
      xArr = Array.from(t); yArr = masked(DC);
    }
    if (xArr) {
      updates[idx] = { x: [xArr], y: [yArr] };
    }
  });

  const idxs = Object.keys(updates).map(Number);
  if (!idxs.length) return;
  Plotly.restyle(div, {
    x: idxs.map(i => updates[i].x),
    y: idxs.map(i => updates[i].y),
  }, idxs);

  document.getElementById('accv-status').textContent =
    `Updated ${idxs.length} traces  ·  ${state.removed.size} pts removed`;
}

// ── Undo ─────────────────────────────────────────────────────────
function pushUndo() {
  undoStack.push(JSON.stringify({
    shift:   {...state.shift},
    gain:    {...state.gain},
    clipLo:  {...state.clipLo},
    clipHi:  {...state.clipHi},
    cropL:   state.cropL,
    cropR:   state.cropR,
    removed: [...state.removed],
  }));
  document.getElementById('accv-undo-count').textContent =
    undoStack.length + ' action' + (undoStack.length!==1?'s':'') + ' in stack';
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
  syncUI();
  rerender();
  document.getElementById('accv-undo-count').textContent =
    undoStack.length + ' action' + (undoStack.length!==1?'s':'') + ' in stack';
};

window.accvReset = function() {
  pushUndo();
  SIGNALS.forEach(s => {
    state.shift[s]=0; state.gain[s]=1;
    state.clipLo[s]=null; state.clipHi[s]=null;
  });
  state.cropL=0; state.cropR=0;
  state.removed.clear();
  syncUI(); rerender();
};

// ── UI sync ───────────────────────────────────────────────────────
function syncUI() {
  SIGNALS.forEach(sig => {
    const sl = document.getElementById('shift-'+sig);
    const sn = document.getElementById('shift-'+sig+'-n');
    const gl = document.getElementById('gain-'+sig);
    const gn = document.getElementById('gain-'+sig+'-n');
    if(sl) sl.value = state.shift[sig];
    if(sn) sn.value = state.shift[sig];
    if(gl) gl.value = state.gain[sig];
    if(gn) gn.value = state.gain[sig];
    const clo = document.getElementById('cliplo-'+sig);
    const chi = document.getElementById('cliphi-'+sig);
    if(clo) clo.value = state.clipLo[sig]??'';
    if(chi) chi.value = state.clipHi[sig]??'';
  });
  document.getElementById('crop-left').value   = state.cropL;
  document.getElementById('crop-left-n').value  = state.cropL;
  document.getElementById('crop-right').value  = state.cropR;
  document.getElementById('crop-right-n').value = state.cropR;
  document.getElementById('accv-removed-count').textContent = state.removed.size;
}

// ── Control callbacks ─────────────────────────────────────────────
window.accvSetShift = function(sig, val) {
  pushUndo();
  state.shift[sig] = parseInt(val)||0;
  document.getElementById('shift-'+sig).value   = state.shift[sig];
  document.getElementById('shift-'+sig+'-n').value = state.shift[sig];
  rerender();
};
window.accvSetGain = function(sig, val) {
  pushUndo();
  state.gain[sig] = parseFloat(val)||1;
  document.getElementById('gain-'+sig).value   = state.gain[sig];
  document.getElementById('gain-'+sig+'-n').value = state.gain[sig];
  rerender();
};
window.accvSetClipLo = function(sig, val) {
  pushUndo();
  state.clipLo[sig] = val===''?null:parseFloat(val);
  rerender();
};
window.accvSetClipHi = function(sig, val) {
  pushUndo();
  state.clipHi[sig] = val===''?null:parseFloat(val);
  rerender();
};
window.accvSetCrop = function(side, val) {
  pushUndo();
  const v = parseInt(val)||0;
  if (side==='left')  { state.cropL=v; document.getElementById('crop-left').value=v;  document.getElementById('crop-left-n').value=v;  }
  else                { state.cropR=v; document.getElementById('crop-right').value=v; document.getElementById('crop-right-n').value=v; }
  rerender();
};

// ── Click-to-remove ───────────────────────────────────────────────
window.accvTogglePick = function() {
  pickMode = !pickMode;
  const btn = document.getElementById('accv-pick-btn');
  btn.textContent = pickMode ? '🔴 Click-Remove ON (click again to stop)' : '🎯 Enable Click-Remove';
  btn.style.background = pickMode ? '#B71C1C' : '#1565C0';
  const div = getPlotlyDiv();
  if (!div) return;
  if (pickMode) {
    div.on('plotly_click', onPlotlyClick);
  } else {
    div.removeAllListeners('plotly_click');
  }
};

function onPlotlyClick(data) {
  if (!data.points || !data.points.length) return;
  pushUndo();
  data.points.forEach(pt => {
    const xi = pt.pointIndex;
    if (xi !== undefined) state.removed.add(xi + state.cropL);
  });
  document.getElementById('accv-removed-count').textContent = state.removed.size;
  rerender();
}

window.accvRestoreRemoved = function() {
  pushUndo();
  state.removed.clear();
  document.getElementById('accv-removed-count').textContent = 0;
  rerender();
};

// ── Toggle panel ─────────────────────────────────────────────────
window.accvTogglePanel = function() {
  const panel = document.getElementById('accv-panel');
  const btn   = document.getElementById('accv-edit-toggle');
  const open  = panel.classList.toggle('open');
  btn.classList.toggle('active', open);
  btn.textContent = open ? '✕ Close Editor' : '⚙ Edit Mode';
  if (open && !rawData) {
    setTimeout(()=>{ extractRaw(); buildSignalUI(); }, 300);
  }
};

// ── Build signal UI ───────────────────────────────────────────────
function buildSignalUI() {
  const container = document.getElementById('accv-signals');
  if (!container) return;
  const colors = { R:'#1565C0', Theta:'#6A1B9A', DC:'#E65100' };
  const shiftMax = rawData ? Math.floor(rawData.N/4) : 5000;
  container.innerHTML = '';
  SIGNALS.forEach(sig => {
    const block = document.createElement('div');
    block.className = 'accv-signal-block';
    block.innerHTML = `
      <div class="sig-title" style="color:${colors[sig]||'#333'}">▐ ${sig}</div>
      <div class="accv-row">
        <label>Shift (samples)</label>
        <input type="range"  id="shift-${sig}"   min="${-shiftMax}" max="${shiftMax}" step="1" value="0"
               oninput="accvSetShift('${sig}', this.value)">
        <input type="number" id="shift-${sig}-n" min="${-shiftMax}" max="${shiftMax}" step="1" value="0"
               onchange="accvSetShift('${sig}', this.value)">
      </div>
      <div class="accv-row">
        <label>Gain (×)</label>
        <input type="range"  id="gain-${sig}"   min="0.01" max="10" step="0.01" value="1"
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
      </div>
    `;
    container.appendChild(block);
  });
}

// ── Export CSV ────────────────────────────────────────────────────
window.accvExportCSV = function() {
  const d = derive();
  if (!d) { alert('No data available'); return; }
  const { t, R, Theta, DC, mask } = d;
  const rows = ['Time,R,Theta,DC'];
  for (let i=0; i<t.length; i++) {
    if (mask[i]) continue;
    rows.push(`${t[i]},${R[i]},${Theta[i]},${DC[i]}`);
  }
  const blob = new Blob([rows.join('\n')], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'accv_edited_' + new Date().toISOString().replace(/[:.]/g,'-') + '.csv';
  a.click();
  document.getElementById('accv-status').textContent = 'CSV exported ✓';
};

// ── Export HTML ───────────────────────────────────────────────────
window.accvExportHTML = function() {
  const d = derive();
  if (!d) { alert('No data available'); return; }
  document.getElementById('accv-status').textContent = 'Baking HTML…';

  // Snapshot current Plotly state as JSON then rebuild
  const div = getPlotlyDiv();
  if (!div) { alert('Plotly div not found'); return; }

  const dataJSON = JSON.stringify(div.data);
  const layoutJSON = JSON.stringify(div.layout);

  const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ACCV Edited Export</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"><\/script>
</head><body>
<div id="chart" style="width:100%;height:100vh;"></div>
<script>
Plotly.newPlot('chart', ${dataJSON}, ${layoutJSON}, {responsive:true});
<\/script>
</body></html>`;

  const blob = new Blob([html], {type:'text/html'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'accv_edited_' + new Date().toISOString().replace(/[:.]/g,'-') + '.html';
  a.click();
  document.getElementById('accv-status').textContent = 'HTML exported ✓';
};

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  // Wait for Plotly to finish rendering
  setTimeout(function poll() {
    const div = getPlotlyDiv();
    if (div && div.data && div.data.length > 0) {
      extractRaw();
      buildSignalUI();
    } else {
      setTimeout(poll, 500);
    }
  }, 800);
});

})();
</script>
"""


# ---------------------------------------------------------------------------
# inject_editor()
# ---------------------------------------------------------------------------

def inject_editor(input_path: str, output_path: str = None) -> str:
    """
    Read a Plotly HTML file and inject the ACCV edit panel.

    Parameters
    ----------
    input_path  : path to the source HTML file
    output_path : where to write the result (default = overwrite input)

    Returns
    -------
    output_path actually written
    """
    if output_path is None:
        output_path = input_path

    with open(input_path, 'r', encoding='utf-8') as f:
        html = f.read()

    if 'accv-edit-toggle' in html:
        print("Editor already injected – skipping.")
        return output_path

    # Inject before </body>
    if '</body>' in html:
        html = html.replace('</body>', EDITOR_JS + '\n</body>', 1)
    else:
        html += EDITOR_JS

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Editor injected → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Monkey-patch helper: call this in your plot_accv_cycle_averaged() after
# fig.write_html() to auto-inject the editor every time.
# ---------------------------------------------------------------------------

def auto_inject_after_save(html_path: str):
    """Drop-in after fig.write_html(html_path) in startboth.py"""
    inject_editor(html_path)


# ---------------------------------------------------------------------------
# Integration snippet (copy into startboth.py / accv module)
# ---------------------------------------------------------------------------
INTEGRATION_SNIPPET = '''
# ── Add this near the top of your file ────────────────────────────────────
from accv_html_editor import auto_inject_after_save

# ── Then immediately after fig.write_html(html_path) add: ─────────────────
auto_inject_after_save(html_path)
# ──────────────────────────────────────────────────────────────────────────
'''


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python accv_html_editor.py <input.html> [output.html]")
        sys.exit(0)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(src):
        print(f"ERROR: File not found: {src}")
        sys.exit(1)

    result = inject_editor(src, dst)
    print(f"Done. Open in browser: {result}")
    print("\nIntegration snippet for startboth.py:")
    print(INTEGRATION_SNIPPET)
