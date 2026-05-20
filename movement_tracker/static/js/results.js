/* Results page: distance traces, movement parameters, group comparison */

let subjects = [];
let currentTab = 'distances';
let currentSubjectId = null;

// Cached data
let cachedTraces = null;
let cachedMovements = null;
let cachedGroup = null;
let cachedSequenceAssignments = null; // { byTrial: { trialIdx: {sequences, seq_r2} }, totalSeqs, totalR2 }

// Y-range slider state for the distance/velocity trace plots.
//   _yFull[kind]       → [min, max] data extent (slider bounds)
//   _yLocked[kind]     → {min, max} shared range when Lock Y-axis is on
//   _yTrial[kind][idx] → {min, max} per-trial range when unlocked
// kind ∈ {'dist','vel'}.  Reset whenever new traces are fetched.
let _yFull = { dist: null, vel: null };
let _yLocked = { dist: null, vel: null };
let _yTrial = { dist: {}, vel: {} };

function _resetYOverrides() {
    _yFull = { dist: null, vel: null };
    _yLocked = { dist: null, vel: null };
    _yTrial = { dist: {}, vel: {} };
}

const PARAM_LABELS = {
    peak_dist: 'Peak Distance (mm)',
    amplitude: 'Amplitude (mm)',
    imi: 'Inter-Movement Interval (s)',
    peak_open_vel: 'Peak Opening Velocity (mm/s)',
    peak_close_vel: 'Peak Closing Velocity (mm/s)',
    rel_amplitude: 'Relative Amplitude',
    power: 'Power (mm\u00b2/s)',
    mean_open_vel: 'Mean Opening Velocity (mm/s)',
    mean_close_vel: 'Mean Closing Velocity (mm/s)',
};

// Short Y-axis labels for the movement scatter plots.
const PARAM_YLABELS = {
    peak_dist: 'Distance (mm)',
    amplitude: 'Amplitude (mm)',
    imi: 'IMI (s)',
    peak_open_vel: 'Velocity (mm/s)',
    peak_close_vel: 'Velocity (mm/s)',
    rel_amplitude: 'Relative Amplitude',
    power: 'Power (mm\u00b2/s)',
    mean_open_vel: 'Velocity (mm/s)',
    mean_close_vel: 'Velocity (mm/s)',
};

// Single shared color for all trials on the movement scatters.
const MOVEMENT_DOT_COLOR = '#2196F3';

const TRIAL_COLORS = [
    '#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4',
];


// ── Subject loading ────────────────────────────────────────────

async function loadSubjects() {
    try {
        subjects = await API.get('/api/subjects');
        const sel = document.getElementById('subjectSelect');
        subjects.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load subjects:', e);
    }
}

function navigateSubject(delta) {
    const sel = document.getElementById('subjectSelect');
    const idx = sel.selectedIndex;
    const newIdx = idx + delta;
    if (newIdx >= 1 && newIdx < sel.options.length) {
        sel.selectedIndex = newIdx;
        sel.dispatchEvent(new Event('change'));
    }
}

function updateNavLinks() {
    const labelingLink = document.getElementById('labelingLink');
    if (labelingLink && currentSubjectId) {
        labelingLink.href = `/labeling-select?subject=${currentSubjectId}`;
    }
}

// ── Tab switching ──────────────────────────────────────────────

function switchTab(tab) {
    currentTab = tab;
    // Remember per-subject tab preference (skip 'group' — that's a page-level default)
    if (currentSubjectId && tab !== 'group') {
        sessionStorage.setItem(`dlc_resultsTab_${currentSubjectId}`, tab);
    }

    // Update tab buttons
    document.querySelectorAll('#tabSwitcher .btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Show/hide tab content
    document.querySelectorAll('.results-tab').forEach(el => {
        el.classList.toggle('active', el.id === 'tab' + tab.charAt(0).toUpperCase() + tab.slice(1));
    });

    // Grey out subject selector in group tab
    const isGroup = tab === 'group';
    document.getElementById('subjectSelect').disabled = isGroup;
    document.getElementById('prevSubjectBtn').disabled = isGroup;
    document.getElementById('nextSubjectBtn').disabled = isGroup;

    // Load data for the active tab
    if ((tab === 'distances' || tab === 'movements') && currentSubjectId) {
        loadDistances(currentSubjectId);
    } else if (tab === 'group') {
        loadGroup();
    }
}

// ── Distances Tab ──────────────────────────────────────────────

async function loadDistances(subjectId) {
    const container = document.getElementById('distancePlots');
    const movContainer = document.getElementById('distMovementPlots');
    const movControls = document.getElementById('distMovementControls');

    if (!subjectId) {
        container.innerHTML = '<div class="results-no-data">Select a subject to view distance traces</div>';
        if (movContainer) movContainer.innerHTML = '';
        if (movControls) movControls.style.display = 'none';
        return;
    }

    container.innerHTML = '<div class="results-no-data">Loading...</div>';

    const selectedSource = document.getElementById('resultsSourceSelect')?.value
        || document.getElementById('distSourceSelect')?.value || 'auto';

    try {
        // Load traces and movements in parallel
        const [traceData, movData] = await Promise.all([
            API.get(`/api/results/${subjectId}/traces?source=${selectedSource}`),
            API.get(`/api/results/${subjectId}/movements?source=${selectedSource}`).catch(() => null),
        ]);

        cachedTraces = traceData;
        _resetYOverrides();   // new subject/source → fresh full-range sliders

        // Update source selector with available sources
        _updateSourceSelector(traceData.available_sources || [], traceData.data_source);

        if (!traceData.trials || traceData.trials.length === 0) {
            container.innerHTML = '<div class="results-no-data">No distance data available for this subject</div>';
            if (movControls) movControls.style.display = 'none';
            return;
        }

        renderAllDistancePlots();

        // Render movement parameters below distance traces
        if (movData && movData.movements && movData.movements.length > 0) {
            cachedMovements = movData;
            cachedSequenceAssignments = null;
            if (movControls) movControls.style.display = '';
            renderDistMovementPlots();
        } else {
            if (movControls) movControls.style.display = 'none';
            if (movContainer) movContainer.innerHTML = '';
        }
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="color:#d32f2f;">${e.message}</div>`;
    }
}

function getDistCheckedParams() {
    const checks = document.querySelectorAll('#distMovementControls input[data-dparam]');
    const params = [];
    checks.forEach(cb => { if (cb.checked) params.push(cb.dataset.dparam); });
    return params;
}

function renderDistMovementPlots() {
    const container = document.getElementById('distMovementPlots');
    const data = cachedMovements;
    if (!container || !data || !data.movements || data.movements.length === 0) return;

    const params = getDistCheckedParams();
    if (params.length === 0) {
        container.innerHTML = '';
        return;
    }

    const seqMode = document.getElementById('distSequenceMode').value;

    // Pre-compute sequence assignments if multi-seq mode.  (The
    // summary text after the dropdown was removed as redundant \u2014 the
    // per-fit R\u00b2 is shown on each plot.)
    if (seqMode.endsWith('_multi') && data.movements.length > 0) {
        if (!cachedSequenceAssignments) {
            cachedSequenceAssignments = computeSequenceAssignments(data);
        }
    }

    container.innerHTML = '';

    params.forEach(param => {
        const div = document.createElement('div');
        div.id = `distMovPlot_${param}`;
        div.style.height = '320px';
        container.appendChild(div);
        renderMovementScatter(div.id, data, param, seqMode);
    });
}

function _updateSourceSelector(availableSources, activeSource) {
    // No-op: the page-level #resultsSourceSelect is now the single
    // source of truth for distance source.  The per-subject auto
    // result is surfaced via the badge in renderAllDistancePlots.
}

// When "Auto" is selected, append the actually-resolved source to the
// dropdown's Auto option (e.g. "Auto — DLC (corrected)").  On an
// explicit selection it reverts to plain "Auto".
function _updateAutoSourceLabel(dataSource) {
    const sel = document.getElementById('resultsSourceSelect');
    if (!sel) return;
    const autoOpt = sel.querySelector('option[value="auto"]');
    if (!autoOpt) return;
    const NAMES = {
        corrections: 'DLC (corrected)',
        mp_combined: 'MP combined',
        mp_forward: 'MP forward',
        mediapipe: 'MP forward',
    };
    autoOpt.textContent = (sel.value === 'auto' && dataSource && NAMES[dataSource])
        ? `Auto — ${NAMES[dataSource]}`
        : 'Auto';
}

function renderAllDistancePlots() {
    const container = document.getElementById('distancePlots');
    const data = cachedTraces;
    if (!data || !data.trials) return;

    container.innerHTML = '';

    // Reflect the auto-resolved source in the dropdown's Auto label.
    _updateAutoSourceLabel(data.data_source);

    // Show active source badge
    if (data.data_source && data.data_source !== 'none') {
        const badge = document.createElement('div');
        badge.style.cssText = 'padding:4px 10px;margin-bottom:8px;border-radius:4px;font-size:11px;display:inline-block;background:rgba(100,180,255,0.15);color:var(--accent);';
        const labels = { mediapipe: 'MediaPipe', vision: 'Vision', skeleton_v1: 'Skeleton v1', skeleton_v2: 'Skeleton v2', skeleton_v3: 'Skeleton v3', dlc: 'DLC', corrections: 'Corrections' };
        badge.textContent = `Source: ${labels[data.data_source] || data.data_source}`;
        container.appendChild(badge);
    }

    const yRange = data.y_range;

    // Establish full data extents (slider bounds) once per dataset.
    // A 5% pad keeps the trace off the very edge at full range.
    if (yRange && _yFull.dist === null) {
        const dPad = (yRange.dist_max - yRange.dist_min) * 0.05 || 1;
        _yFull.dist = [yRange.dist_min - dPad, yRange.dist_max + dPad];
        const vPad = (yRange.vel_max - yRange.vel_min) * 0.05 || 1;
        _yFull.vel = [yRange.vel_min - vPad, yRange.vel_max + vPad];
    }

    // Build per-trial overlay data from movements
    const overlayPeakDist = document.getElementById('overlayPeakDist').checked;
    const overlayPeakOpenVel = document.getElementById('overlayPeakOpenVel').checked;
    const overlayPeakCloseVel = document.getElementById('overlayPeakCloseVel').checked;
    // Sequence shading on the distance/velocity traces was removed with
    // the "Sequences" checkbox; sequence fits now live only in the
    // Movement-plot scatters below.
    const showSequences = false;
    let seqAssignments = null;
    const _r2El = document.getElementById('overlayR2');
    if (_r2El) _r2El.textContent = '';

    // Group movements by trial
    const movByTrial = {};
    if (cachedMovements && cachedMovements.movements) {
        cachedMovements.movements.forEach((m, mi) => {
            const ti = m.trial_idx;
            if (!movByTrial[ti]) movByTrial[ti] = [];
            movByTrial[ti].push({ ...m, _idx: mi });
        });
    }

    // Ensure container has a valid width before rendering
    const containerWidth = container.clientWidth || container.parentElement?.clientWidth || 1200;

    // Plot heights + Plotly margins, kept in sync with the layout
    // objects in renderDistancePlot / renderVelocityPlot so the
    // slider travel lines up with the y-axis data area.
    const DIST_H = 220, DIST_MT = 30, DIST_MB = 5;
    const VEL_H = 150, VEL_MT = 5, VEL_MB = 35;

    data.trials.forEach((trial, idx) => {
        // Trial block: fixed slider column + horizontally-scrolling plots.
        const block = document.createElement('div');
        block.style.display = 'flex';
        block.style.alignItems = 'flex-start';
        block.style.marginBottom = '16px';

        // Slider column (does NOT scroll with the plots)
        const sliderCol = document.createElement('div');
        sliderCol.style.display = 'flex';
        sliderCol.style.flexDirection = 'column';
        sliderCol.style.flex = '0 0 auto';
        sliderCol.appendChild(_buildYSliderCol('dist', idx, DIST_H, DIST_MT, DIST_MB));
        sliderCol.appendChild(_buildYSliderCol('vel', idx, VEL_H, VEL_MT, VEL_MB));
        block.appendChild(sliderCol);

        // Scrolling wrapper holds both plots
        const wrapper = document.createElement('div');
        wrapper.style.overflowX = 'auto';
        wrapper.style.flex = '1 1 auto';

        // Distance plot
        const distDiv = document.createElement('div');
        distDiv.id = `distPlot_${idx}`;
        distDiv.style.height = DIST_H + 'px';
        wrapper.appendChild(distDiv);

        // Velocity plot
        const velDiv = document.createElement('div');
        velDiv.id = `velPlot_${idx}`;
        velDiv.style.height = VEL_H + 'px';
        wrapper.appendChild(velDiv);

        block.appendChild(wrapper);
        container.appendChild(block);

        // Compute plot width based on 20s per screen width
        const fps = trial.fps || 60;
        const durationSec = trial.distances.length / fps;
        const plotWidth = Math.max(containerWidth, (durationSec / 20) * containerWidth);

        // Set wrapper width to enable scrolling
        distDiv.style.width = plotWidth + 'px';
        velDiv.style.width = plotWidth + 'px';

        const trialStart = data.trials.slice(0, idx).reduce((acc, t) => acc + (t.distances ? t.distances.length : 0), 0);
        const trialMovs = movByTrial[idx] || [];

        // Build overlay traces for distance plot (peak distance markers on the curve)
        const distOverlays = [];
        const distShapes = [];
        if (overlayPeakDist && trialMovs.length > 0) {
            const peakTimes = [], peakVals = [];
            trialMovs.forEach(m => {
                if (m.peak_dist != null && m.peak_frame != null) {
                    const localFrame = m.peak_frame - trialStart;
                    peakTimes.push(+(localFrame / fps).toFixed(3));
                    peakVals.push(m.peak_dist);
                }
            });
            if (peakTimes.length > 0) {
                distOverlays.push({
                    x: peakTimes, y: peakVals,
                    type: 'scatter', mode: 'markers',
                    marker: { color: '#FF9800', size: 7, symbol: 'diamond' },
                    name: 'Peak Distance',
                    hovertemplate: '%{x:.2f}s<br>Peak: %{y:.1f} mm<extra></extra>',
                });
            }
        }

        // Sequence shading
        if (showSequences && seqAssignments && seqAssignments.byTrial[idx]) {
            const seqs = seqAssignments.byTrial[idx].sequences || [];
            seqs.forEach((seq, si) => {
                // seq has start/end indices within the trial's movement list
                const startMov = trialMovs[seq.start];
                const endMov = trialMovs[seq.end];
                if (startMov && endMov && startMov.peak_frame != null && endMov.peak_frame != null) {
                    const x0 = (startMov.peak_frame - trialStart) / fps;
                    const x1 = (endMov.peak_frame - trialStart) / fps;
                    distShapes.push({
                        type: 'rect', xref: 'x', yref: 'paper',
                        x0, x1, y0: 0, y1: 1,
                        fillcolor: 'rgba(156, 39, 176, 0.08)',
                        line: { color: 'rgba(156, 39, 176, 0.3)', width: 1 },
                    });
                }
            });
        }

        // Build overlay traces for velocity plot
        const velOverlays = [];
        if (overlayPeakOpenVel && trialMovs.length > 0) {
            const ts = [], vs = [];
            trialMovs.forEach(m => {
                // Position the marker at the opening-velocity peak frame
                // (falls back to the distance-peak frame if absent).
                const vf = m.peak_open_vel_frame != null ? m.peak_open_vel_frame : m.peak_frame;
                if (m.peak_open_vel != null && vf != null) {
                    ts.push(+((vf - trialStart) / fps).toFixed(3));
                    vs.push(m.peak_open_vel);
                }
            });
            if (ts.length > 0) {
                velOverlays.push({
                    x: ts, y: vs,
                    type: 'scatter', mode: 'markers',
                    marker: { color: '#2196F3', size: 6, symbol: 'triangle-up' },
                    name: 'Peak Open Vel',
                    hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm/s<extra></extra>',
                });
            }
        }
        if (overlayPeakCloseVel && trialMovs.length > 0) {
            const ts = [], vs = [];
            trialMovs.forEach(m => {
                // Position the marker at the closing-velocity peak frame
                // (falls back to the distance-peak frame if absent).
                const vf = m.peak_close_vel_frame != null ? m.peak_close_vel_frame : m.peak_frame;
                if (m.peak_close_vel != null && vf != null) {
                    ts.push(+((vf - trialStart) / fps).toFixed(3));
                    vs.push(m.peak_close_vel);
                }
            });
            if (ts.length > 0) {
                velOverlays.push({
                    x: ts, y: vs,
                    type: 'scatter', mode: 'markers',
                    marker: { color: '#f44336', size: 6, symbol: 'triangle-down' },
                    name: 'Peak Close Vel',
                    hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm/s<extra></extra>',
                });
            }
        }

        renderDistancePlot(distDiv.id, trial, _effYRange('dist', idx), plotWidth, distOverlays, distShapes);
        renderVelocityPlot(velDiv.id, trial, _effYRange('vel', idx), plotWidth, velOverlays, distShapes);
    });
}

// ── Y-range slider helpers ─────────────────────────────────────

// Effective [min,max] for a plot: the locked shared range, the
// per-trial range, or the full data extent when nothing's been
// adjusted yet.
function _effYRange(kind, idx) {
    if (!_yFull[kind]) return null;
    const locked = document.getElementById('lockYAxis').checked;
    const ov = locked ? _yLocked[kind] : _yTrial[kind][idx];
    if (ov && ov.min != null && ov.max != null) return [ov.min, ov.max];
    return _yFull[kind].slice();
}

// Build one dual-thumb vertical Y-range slider (shared track with a
// min thumb + max thumb + fill bar), matching the Labels page Y-lim
// sliders.  ``mTop``/``mBot`` are the Plotly plot margins so the
// slider travel lines up with the y-axis data area.
function _buildYSliderCol(kind, idx, h, mTop, mBot) {
    const wrap = document.createElement('div');
    wrap.className = 'y-range-wrap' + (kind === 'vel' ? ' vel' : '');
    wrap.style.height = Math.max(20, h - mTop - mBot) + 'px';
    wrap.style.marginTop = mTop + 'px';
    wrap.style.marginBottom = mBot + 'px';
    wrap.dataset.kind = kind; wrap.dataset.trial = idx;

    const full = _yFull[kind];
    if (!full) return wrap;

    const eff = _effYRange(kind, idx) || full;
    const step = ((full[1] - full[0]) / 1000) || 0.1;

    const track = document.createElement('div'); track.className = 'y-track';
    const fill = document.createElement('div'); fill.className = 'y-fill';
    wrap.appendChild(track); wrap.appendChild(fill);

    const mk = (bound, val) => {
        const s = document.createElement('input');
        s.type = 'range';
        s.className = 'ysl ' + bound;
        s.min = full[0]; s.max = full[1]; s.step = step; s.value = val;
        s.dataset.kind = kind; s.dataset.bound = bound; s.dataset.trial = idx;
        s.setAttribute('orient', 'vertical');
        s.style.cssText = 'writing-mode:vertical-lr;direction:rtl;height:100%;';
        s.title = bound === 'max' ? 'Y max' : 'Y min';
        s.addEventListener('input', () => _onYSlider(s));
        return s;
    };
    wrap.appendChild(mk('min', eff[0]));
    wrap.appendChild(mk('max', eff[1]));

    // Fill needs the element's measured height — update once laid out.
    requestAnimationFrame(() => _updateYFill(wrap));
    return wrap;
}

// Position the fill bar between the two thumbs.
function _updateYFill(wrap) {
    const fill = wrap.querySelector('.y-fill');
    const mn = wrap.querySelector('.ysl.min');
    const mx = wrap.querySelector('.ysl.max');
    if (!fill || !mn || !mx) return;
    const lo = parseFloat(mn.min), hi = parseFloat(mn.max), rng = hi - lo;
    if (!(rng > 0)) return;
    const h = wrap.clientHeight || 100;
    const pctMin = (parseFloat(mn.value) - lo) / rng;
    const pctMax = (parseFloat(mx.value) - lo) / rng;
    fill.style.bottom = (pctMin * h) + 'px';
    fill.style.height = ((pctMax - pctMin) * h) + 'px';
}

function _onYSlider(el) {
    const kind = el.dataset.kind;
    const idx = +el.dataset.trial;
    const bound = el.dataset.bound;
    const full = _yFull[kind];
    if (!full) return;
    const wrap = el.closest('.y-range-wrap');
    const mn = wrap.querySelector('.ysl.min');
    const mx = wrap.querySelector('.ysl.max');

    const minGap = (full[1] - full[0]) * 0.02 || 0.1;
    let lo = parseFloat(mn.value), hi = parseFloat(mx.value);
    // Keep the thumbs from crossing.
    if (bound === 'min' && lo > hi - minGap) { lo = hi - minGap; mn.value = lo; }
    if (bound === 'max' && hi < lo + minGap) { hi = lo + minGap; mx.value = hi; }

    const cur = { min: lo, max: hi };
    const locked = document.getElementById('lockYAxis').checked;
    if (locked) _yLocked[kind] = cur; else _yTrial[kind][idx] = cur;

    _updateYFill(wrap);
    _applyYRange(kind, locked ? null : idx);
}

// Push the chosen range to the plot(s) via Plotly.relayout (no full
// re-render, so it's instant).  When locked (idx === null) it hits
// every plot of this kind and syncs all sibling sliders + fills.
function _applyYRange(kind, idx) {
    const prefix = kind === 'dist' ? 'distPlot_' : 'velPlot_';
    if (idx === null) {
        const range = _yLocked[kind];
        const r = [range.min, range.max];
        document.querySelectorAll(`[id^="${prefix}"]`).forEach(div => {
            if (window.Plotly) { try { Plotly.relayout(div, { 'yaxis.range': r }); } catch {} }
        });
        document.querySelectorAll(`.y-range-wrap[data-kind="${kind}"]`).forEach(w => {
            const mn = w.querySelector('.ysl.min');
            const mx = w.querySelector('.ysl.max');
            if (mn) mn.value = range.min;
            if (mx) mx.value = range.max;
            _updateYFill(w);
        });
    } else {
        const range = _yTrial[kind][idx];
        const div = document.getElementById(`${prefix}${idx}`);
        if (div && window.Plotly) { try { Plotly.relayout(div, { 'yaxis.range': [range.min, range.max] }); } catch {} }
    }
}

function renderDistancePlot(divId, trial, yRange, width, overlayTraces, shapes) {
    const fps = trial.fps || 60;
    const n = trial.distances.length;
    const times = Array.from({ length: n }, (_, i) => +(i / fps).toFixed(3));

    const distTrace = {
        x: times,
        y: trial.distances,
        type: 'scatter',
        mode: 'lines',
        name: 'Distance (mm)',
        line: { color: '#2196F3', width: 1.2 },
        hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm<extra></extra>',
    };

    const traces = [distTrace, ...(overlayTraces || [])];
    const hasOverlays = (overlayTraces || []).length > 0;

    const layout = {
        title: { text: trial.name, font: { size: 13, color: '#666' } },
        margin: { t: 30, b: 5, l: 55, r: 20 },
        xaxis: { showticklabels: false, color: '#666', gridcolor: '#eee' },
        yaxis: {
            title: { text: 'Distance (mm)', font: { size: 11, color: '#2196F3' } },
            color: '#2196F3',
            gridcolor: '#f0f0f0',
            zeroline: false,
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: hasOverlays,
        legend: { x: 1, xanchor: 'right', y: 1, font: { size: 10 } },
        hovermode: 'x unified',
        width: width,
        shapes: shapes || [],
    };

    if (yRange) layout.yaxis.range = yRange;

    Plotly.newPlot(divId, traces, layout, {
        responsive: false,
        displayModeBar: false,
    });
}

function renderVelocityPlot(divId, trial, yRange, width, overlayTraces, shapes) {
    const fps = trial.fps || 60;
    const n = trial.velocities.length;
    const times = Array.from({ length: n }, (_, i) => +(i / fps).toFixed(3));

    const velTrace = {
        x: times,
        y: trial.velocities,
        type: 'scatter',
        mode: 'lines',
        name: 'Velocity (mm/s)',
        line: { color: '#4CAF50', width: 1 },
        hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm/s<extra></extra>',
    };

    const traces = [velTrace, ...(overlayTraces || [])];
    const hasOverlays = (overlayTraces || []).length > 0;

    const layout = {
        margin: { t: 5, b: 35, l: 55, r: 20 },
        xaxis: { title: { text: 'Time (s)', font: { size: 11 } }, color: '#666', gridcolor: '#eee' },
        yaxis: {
            title: { text: 'Velocity (mm/s)', font: { size: 11, color: '#4CAF50' } },
            color: '#4CAF50',
            gridcolor: '#f0f0f0',
            zeroline: true,
            zerolinecolor: '#ddd',
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: hasOverlays,
        legend: { x: 1, xanchor: 'right', y: 1, font: { size: 10 } },
        hovermode: 'x unified',
        width: width,
        shapes: shapes || [],
    };

    if (yRange) layout.yaxis.range = yRange;

    Plotly.newPlot(divId, traces, layout, {
        responsive: false,
        displayModeBar: false,
    });
}

// ── Movements Tab ──────────────────────────────────────────────

async function loadMovements(subjectId) {
    const container = document.getElementById('movementPlots');

    if (!subjectId) {
        container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">Select a subject to view movement parameters</div>';
        return;
    }

    container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">Loading...</div>';

    try {
        cachedSequenceAssignments = null;
        const data = await API.get(`/api/results/${subjectId}/movements`);
        cachedMovements = data;

        // Show auto-detect banner if applicable
        const banner = document.getElementById('movementEventBanner');
        if (data.event_source === 'auto') {
            banner.innerHTML = '<div class="auto-detect-banner">Using auto-detected events (no saved events found for this subject)</div>';
        } else {
            banner.innerHTML = '';
        }

        renderMovementPlots();
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="grid-column:1/-1;color:#d32f2f;">${e.message}</div>`;
    }
}

function getCheckedParams() {
    const checks = document.querySelectorAll('#movementControls input[data-param]');
    const params = [];
    checks.forEach(cb => {
        if (cb.checked) params.push(cb.dataset.param);
    });
    return params;
}

function renderMovementPlots() {
    const container = document.getElementById('movementPlots');
    const data = cachedMovements;

    if (!data || !data.movements || data.movements.length === 0) {
        container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">No movement data available</div>';
        return;
    }

    const params = getCheckedParams();
    if (params.length === 0) {
        container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">Select at least one parameter</div>';
        return;
    }

    const seqMode = document.getElementById('sequenceMode').value;

    // Pre-compute sequence assignments if multi-seq mode (based on amplitude)
    if (seqMode.endsWith('_multi') && data.movements.length > 0) {
        if (!cachedSequenceAssignments) {
            cachedSequenceAssignments = computeSequenceAssignments(data);
        }
        updateSeqInfo(cachedSequenceAssignments);
    } else {
        hideSeqInfo();
    }

    container.innerHTML = '';

    params.forEach(param => {
        const div = document.createElement('div');
        div.id = `movPlot_${param}`;
        div.style.height = '320px';
        container.appendChild(div);

        renderMovementScatter(div.id, data, param, seqMode);
    });
}

function computeSequenceAssignments(data) {
    const movements = data.movements;
    // Group by trial
    const byTrial = {};
    movements.forEach((m, i) => {
        const ti = m.trial_idx;
        if (!byTrial[ti]) byTrial[ti] = [];
        byTrial[ti].push({ ...m, _globalIdx: i });
    });

    const result = { byTrial: {}, totalSeqs: 0, totalR2: 0 };
    let totalSS = 0, totalSSExplained = 0;

    // Need global amplitude stats for overall R²
    const allAmps = movements.map(m => m.amplitude).filter(a => a != null && isFinite(a));
    let globalAmpMean = 0;
    for (const a of allAmps) globalAmpMean += a;
    globalAmpMean /= allAmps.length || 1;
    for (const a of allAmps) totalSS += (a - globalAmpMean) ** 2;

    Object.keys(byTrial).sort((a, b) => +a - +b).forEach(ti => {
        const ms = byTrial[ti];
        const amps = ms.map(m => m.amplitude).filter(a => a != null && isFinite(a));

        if (amps.length < 5) {
            result.byTrial[ti] = { sequences: [], seq_r2: 0 };
            return;
        }

        const opt = optimizeSequences(amps, 5, 0.3);
        result.byTrial[ti] = opt;
        result.totalSeqs += opt.sequences.length;

        // Accumulate explained variance for overall R²
        for (const seq of opt.sequences) {
            totalSSExplained += seq.ss_reg;
        }
    });

    result.totalR2 = totalSS > 0 ? totalSSExplained / totalSS : 0;
    return result;
}

function updateSeqInfo(assignments) {
    const el = document.getElementById('seqInfo');
    if (!el) return;
    if (assignments.totalSeqs > 0) {
        el.textContent = `${assignments.totalSeqs} sequence${assignments.totalSeqs !== 1 ? 's' : ''} detected | R\u00b2 = ${assignments.totalR2.toFixed(3)}`;
    } else {
        el.textContent = 'No sequences detected';
    }
}

function hideSeqInfo() {
    const el = document.getElementById('seqInfo');
    if (el) el.textContent = '';
}

function renderMovementScatter(divId, data, param, seqMode) {
    const movements = data.movements;
    const trialNames = data.trial_names || [];

    // Group by trial
    const byTrial = {};
    movements.forEach(m => {
        const ti = m.trial_idx;
        if (!byTrial[ti]) byTrial[ti] = [];
        byTrial[ti].push(m);
    });

    const traces = [];
    const shapes = []; // for sequence spans
    const annotations = [];
    const isMulti = seqMode.endsWith('_multi');
    const isExp = seqMode.startsWith('exp_');
    const isFirst10 = seqMode.endsWith('_first10');

    // Peak velocity params are timestamped at their own velocity peak,
    // not the distance peak, so their sequence-effect fits use the
    // correct time axis.  Everything else stays on the distance-peak
    // time (``peak_time``).
    const xField = param === 'peak_open_vel' ? 'peak_open_vel_time'
                 : param === 'peak_close_vel' ? 'peak_close_vel_time'
                 : 'peak_time';

    // Same color for every trial — trial identity is shown by the
    // labels above each trial's segment instead of a color key.
    const color = MOVEMENT_DOT_COLOR;

    // Pre-compute each trial's local time span so trials can be laid
    // out left-to-right, each restarting at t=0, with a small gap.
    const trialKeys = Object.keys(byTrial).sort((a, b) => +a - +b);
    const trialInfo = trialKeys.map(ti => {
        const ms = byTrial[ti];
        const rawX = ms.map(m => (m[xField] != null ? m[xField] : m.peak_time));
        const valid = rawX.filter(v => v != null && isFinite(v));
        const t0 = valid.length ? Math.min(...valid) : 0;
        const span = valid.length ? (Math.max(...valid) - t0) : 0;
        return { ti, ms, rawX, t0, span };
    });
    const maxSpan = Math.max(1, ...trialInfo.map(t => t.span));
    const gap = maxSpan * 0.08;
    // Cumulative left-edge offset per trial (precomputed so the early
    // returns in the fit branches below can't skip the bookkeeping).
    let _acc = 0;
    trialInfo.forEach(t => { t.offset = _acc; _acc += t.span + gap; });

    trialInfo.forEach(({ ti, ms, rawX, t0, offset: xOffset }) => {
        // Local time restarts at 0 per trial; shift by the running
        // offset so trials sit side-by-side without overlapping.
        const x = rawX.map(v => (v != null && isFinite(v)) ? (v - t0 + xOffset) : v);
        const y = ms.map(m => m[param]);
        const trialLabel = trialNames[ti] || `Trial ${+ti + 1}`;

        traces.push({
            x, y,
            type: 'scatter',
            mode: 'markers',
            name: trialLabel,
            marker: { color, size: 7, opacity: 0.8 },
            hovertemplate: `${trialLabel}<br>t=%{x:.2f}s<br>${PARAM_LABELS[param]}: %{y:.2f}<extra></extra>`,
            showlegend: false,
        });

        // Trial label above the plot, left-aligned with the trial start.
        annotations.push({
            x: xOffset, y: 1.0,
            xref: 'x', yref: 'paper',
            xanchor: 'left', yanchor: 'bottom',
            text: trialLabel,
            showarrow: false,
            font: { size: 11, color: '#444' },
        });

        // Vertical line marking t=0 for this trial, matching the y-axis
        // line drawn at t=0 of the first trial.
        shapes.push({
            type: 'line', xref: 'x', yref: 'paper',
            x0: xOffset, x1: xOffset, y0: 0, y1: 1,
            line: { color: '#888', width: 1 },
            layer: 'below',
        });

        if (seqMode === 'none') return;

        // Closing velocities are negative; negate before fitting so we model
        // the magnitude decay (positive values) and negate predictions back.
        const flipSign = (param === 'peak_close_vel' || param === 'mean_close_vel');

        // Build valid pairs (non-null x and y)
        const validPairs = [];
        for (let i = 0; i < x.length; i++) {
            if (x[i] != null && y[i] != null && isFinite(x[i]) && isFinite(y[i])) {
                validPairs.push({ x: x[i], y: flipSign ? -y[i] : y[i], idx: i });
            }
        }

        if (isMulti) {
            // Multi-sequence mode: use cached sequence assignments from amplitude
            const trialSeqs = cachedSequenceAssignments && cachedSequenceAssignments.byTrial[ti];
            if (!trialSeqs || trialSeqs.sequences.length === 0) return;

            const sign = flipSign ? -1 : 1;

            // Map the trial's valid-amplitude indices to movement indices
            // (seq.start/end index into the amplitude array).
            const ampValid = [];
            for (let i = 0; i < ms.length; i++) {
                if (ms[i].amplitude != null && isFinite(ms[i].amplitude)) {
                    ampValid.push(i);
                }
            }

            // Collected per sequence so the leftover (non-sequence)
            // movements can be modeled afterwards.
            const seqWindows = [];   // {xMin, xMax, baselineMag}
            const coveredIdx = new Set();

            // For each sequence window (start/end are indices into the trial's amplitude array,
            // but we need to filter for valid param values within those windows)
            trialSeqs.sequences.forEach((seq, si) => {
                const seqMsIndices = ampValid.slice(seq.start, seq.end);

                // Get x, y for this param within the sequence window
                // For closing velocities, negate so we fit on positive magnitudes
                const seqX = [], seqY = [];
                for (const mi of seqMsIndices) {
                    if (x[mi] != null && y[mi] != null && isFinite(x[mi]) && isFinite(y[mi])) {
                        seqX.push(x[mi]);
                        seqY.push(flipSign ? -y[mi] : y[mi]);
                        coveredIdx.add(mi);
                    }
                }
                if (seqX.length < 2) return;

                const xMin = Math.min(...seqX);
                const xMax = Math.max(...seqX);

                // Background span — use trial color so shading matches the dots
                shapes.push({
                    type: 'rect', xref: 'x', yref: 'paper',
                    x0: xMin, x1: xMax, y0: 0, y1: 1,
                    fillcolor: color, opacity: 0.07,
                    line: { width: 0 },
                    layer: 'below',
                });

                let baselineMag;
                if (isExp) {
                    // Exponential fit within sequence
                    const fit = exponentialFit(seqX, seqY);
                    if (!fit) return;
                    baselineMag = fit.predict(xMin);
                    const nPts = 50;
                    const step = (xMax - xMin) / (nPts - 1);
                    const curveX = [], curveY = [];
                    for (let k = 0; k < nPts; k++) {
                        const xv = xMin + k * step;
                        curveX.push(xv);
                        curveY.push(sign * fit.predict(xv));
                    }
                    traces.push({
                        x: curveX, y: curveY,
                        type: 'scatter', mode: 'lines',
                        name: `Seq ${si + 1}`,
                        line: { color, width: 2.5 },
                        showlegend: false, hoverinfo: 'skip',
                    });
                    // R² annotation
                    const midX = seqX[Math.floor(seqX.length / 2)];
                    const midY = sign * fit.predict(midX);
                    annotations.push({
                        x: midX, y: midY,
                        text: `R\u00b2=${fit.r2.toFixed(2)}`,
                        showarrow: false,
                        font: { size: 9, color },
                        yshift: 18,
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: color,
                        borderpad: 2,
                    });
                } else {
                    // Linear fit within sequence
                    const reg = linearRegressionFull(seqX, seqY);
                    if (!reg) return;
                    baselineMag = reg.slope * xMin + reg.intercept;
                    traces.push({
                        x: [xMin, xMax],
                        y: [sign * (reg.slope * xMin + reg.intercept), sign * (reg.slope * xMax + reg.intercept)],
                        type: 'scatter', mode: 'lines',
                        name: `Seq ${si + 1}`,
                        line: { color, width: 2.5 },
                        showlegend: false, hoverinfo: 'skip',
                    });
                    // R² annotation
                    const midX = (xMin + xMax) / 2;
                    const midY = sign * (reg.slope * midX + reg.intercept);
                    annotations.push({
                        x: midX, y: midY,
                        text: `R\u00b2=${reg.r2.toFixed(2)}`,
                        showarrow: false,
                        font: { size: 9, color },
                        yshift: 18,
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: color,
                        borderpad: 2,
                    });
                }
                seqWindows.push({ xMin, xMax, baselineMag });
            });

            // Model every movement NOT in a sequence as a constant equal
            // to the largest sequence baseline (the fit value at a
            // sequence start).  Drawn as a flat dotted line over the
            // non-sequence regions so trailing low movements aren't left
            // unmodeled / cut off the ends of the sequences.
            if (seqWindows.length > 0) {
                const maxBaselineMag = Math.max(...seqWindows.map(w => w.baselineMag));
                const constY = sign * maxBaselineMag;

                const allX = [], nonSeqX = [];
                for (let i = 0; i < x.length; i++) {
                    if (x[i] == null || !isFinite(x[i]) || y[i] == null || !isFinite(y[i])) continue;
                    allX.push(x[i]);
                    if (!coveredIdx.has(i)) nonSeqX.push(x[i]);
                }

                if (nonSeqX.length > 0 && allX.length > 0) {
                    const trialXmin = Math.min(...allX);
                    const trialXmax = Math.max(...allX);
                    // Merged sequence coverage, ascending.
                    const cov = seqWindows
                        .map(w => [w.xMin, w.xMax])
                        .sort((a, b) => a[0] - b[0]);
                    // Complement intervals within [trialXmin, trialXmax].
                    const segs = [];
                    let cursor = trialXmin;
                    for (const [a, b] of cov) {
                        if (a > cursor) segs.push([cursor, Math.min(a, trialXmax)]);
                        cursor = Math.max(cursor, b);
                    }
                    if (cursor < trialXmax) segs.push([cursor, trialXmax]);

                    // Keep only gap segments that actually contain a
                    // non-sequence movement; draw one broken polyline.
                    const cx = [], cy = [];
                    let firstSeg = true;
                    for (const [a, b] of segs) {
                        if (b <= a) continue;
                        if (!nonSeqX.some(xv => xv >= a - 1e-6 && xv <= b + 1e-6)) continue;
                        if (!firstSeg) { cx.push(null); cy.push(null); }
                        cx.push(a, b); cy.push(constY, constY);
                        firstSeg = false;
                    }
                    if (cx.length > 0) {
                        traces.push({
                            x: cx, y: cy,
                            type: 'scatter', mode: 'lines',
                            name: 'Baseline (non-seq)',
                            line: { color, width: 2, dash: 'dot' },
                            showlegend: false, hoverinfo: 'skip',
                        });
                    }
                }
            }

        } else {
            // Full or first-10 modes
            let fitPairs = validPairs;
            if (isFirst10) {
                fitPairs = validPairs.slice(0, 10);
            }
            if (fitPairs.length < 2) return;

            const fitX = fitPairs.map(p => p.x);
            const fitY = fitPairs.map(p => p.y);
            const xMin = Math.min(...fitX);
            const xMax = Math.max(...fitX);
            const sign = flipSign ? -1 : 1;

            if (isExp) {
                // Exponential fit
                const fit = exponentialFit(fitX, fitY);
                if (!fit) return;
                const nPts = 80;
                const step = (xMax - xMin) / (nPts - 1);
                const curveX = [], curveY = [];
                for (let k = 0; k < nPts; k++) {
                    const xv = xMin + k * step;
                    curveX.push(xv);
                    curveY.push(sign * fit.predict(xv));
                }
                traces.push({
                    x: curveX, y: curveY,
                    type: 'scatter', mode: 'lines',
                    name: `Exp. fit (Trial ${+ti + 1})`,
                    line: { color, width: 2, dash: 'dash' },
                    showlegend: false, hoverinfo: 'skip',
                });
                // R² annotation
                annotations.push({
                    x: (xMin + xMax) / 2, y: sign * fit.predict((xMin + xMax) / 2),
                    text: `R\u00b2=${fit.r2.toFixed(2)}`,
                    showarrow: false,
                    font: { size: 9, color },
                    yshift: 18,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: color,
                    borderpad: 2,
                });
            } else {
                // Linear fit (full or first-10) — use the R²-bearing
                // regression so we can annotate the fit like the other
                // modes.
                const reg = linearRegressionFull(fitX, fitY);
                if (!reg) return;
                const { slope, intercept } = reg;
                traces.push({
                    x: [xMin, xMax],
                    y: [sign * (slope * xMin + intercept), sign * (slope * xMax + intercept)],
                    type: 'scatter', mode: 'lines',
                    name: `Trend (Trial ${+ti + 1})`,
                    line: { color, width: 2, dash: 'dash' },
                    showlegend: false, hoverinfo: 'skip',
                });
                // R² annotation
                const midX = (xMin + xMax) / 2;
                annotations.push({
                    x: midX, y: sign * (slope * midX + intercept),
                    text: `R²=${reg.r2.toFixed(2)}`,
                    showarrow: false,
                    font: { size: 9, color },
                    yshift: 18,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: color,
                    borderpad: 2,
                });
            }
        }
    });

    // Closing-velocity params are negative; reverse the Y axis so the
    // curve points up like the opening-velocity plots for easy visual
    // comparison.
    const reverseY = (param === 'peak_close_vel' || param === 'mean_close_vel');

    const layout = {
        title: { text: PARAM_LABELS[param], font: { size: 13, color: '#444' } },
        margin: { t: 35, b: 40, l: 60, r: 20 },
        xaxis: { title: { text: 'Time (s)', font: { size: 11 } }, color: '#666', gridcolor: '#eee' },
        yaxis: {
            title: { text: PARAM_YLABELS[param] || '', font: { size: 11 } },
            color: '#666', gridcolor: '#f0f0f0',
            autorange: reverseY ? 'reversed' : true,
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: false,
        shapes,
        annotations,
    };

    Plotly.newPlot(divId, traces, layout, { responsive: true, displayModeBar: false });
}

function linearRegression(x, y) {
    const n = x.length;
    let sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (let i = 0; i < n; i++) {
        sx += x[i]; sy += y[i]; sxy += x[i] * y[i]; sx2 += x[i] * x[i];
    }
    const denom = n * sx2 - sx * sx;
    if (denom === 0) return { slope: 0, intercept: sy / n };
    const slope = (n * sxy - sx * sy) / denom;
    const intercept = (sy - slope * sx) / n;
    return { slope, intercept };
}

function linearRegressionFull(x, y) {
    // Filter to valid (non-null, finite) pairs
    const pairs = [];
    for (let i = 0; i < x.length; i++) {
        if (x[i] != null && y[i] != null && isFinite(x[i]) && isFinite(y[i])) {
            pairs.push([x[i], y[i]]);
        }
    }
    if (pairs.length < 2) return null;
    const xs = pairs.map(p => p[0]);
    const ys = pairs.map(p => p[1]);
    const n = xs.length;
    let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    for (let i = 0; i < n; i++) {
        sx += xs[i]; sy += ys[i]; sxy += xs[i] * ys[i];
        sx2 += xs[i] * xs[i]; sy2 += ys[i] * ys[i];
    }
    const xMean = sx / n;
    const yMean = sy / n;
    const denom = n * sx2 - sx * sx;
    if (denom === 0) return null;
    const slope = (n * sxy - sx * sy) / denom;
    const intercept = (sy - slope * sx) / n;

    // SS_total and SS_reg
    let ssTot = 0, ssReg = 0;
    for (let i = 0; i < n; i++) {
        ssTot += (ys[i] - yMean) ** 2;
        ssReg += (xs[i] - xMean) ** 2;
    }
    ssReg *= slope * slope;
    const r2 = ssTot > 0 ? ssReg / ssTot : 0;

    return { slope, intercept, r2, ss_reg: ssReg, n };
}

/**
 * DP-optimal non-overlapping negative-slope sequence detection.
 * Ported from bradykinesia_analysis.ipynb `optimize_sequences()`.
 *
 * @param {number[]} amplitudes - amplitude values for movements in one trial
 * @param {number} minMoves - minimum movements per sequence window (default 5)
 * @param {number} minR2 - minimum R² per window (default 0.3)
 * @returns {{ sequences: Array<{start,end,slope,r2,ss_reg}>, seq_r2: number }}
 */
function optimizeSequences(amplitudes, minMoves = 5, minR2 = 0.3) {
    const n = amplitudes.length;
    const result = { sequences: [], seq_r2: 0 };
    if (n < minMoves) return result;

    // Total SS (for computing seq_r2)
    let ampMean = 0;
    for (let i = 0; i < n; i++) ampMean += amplitudes[i];
    ampMean /= n;
    let totalSS = 0;
    for (let i = 0; i < n; i++) totalSS += (amplitudes[i] - ampMean) ** 2;
    if (totalSS === 0) return result;

    // Enumerate all negative-slope windows of length >= minMoves
    // Each window: [start, end, ss_reg, slope, r2]  (end is exclusive)
    const windows = [];
    for (let i = 0; i <= n - minMoves; i++) {
        for (let j = i + minMoves; j <= n; j++) {
            // Linear regression on (index, amplitude) for window [i, j)
            const reg = linearRegressionFull(
                Array.from({ length: j - i }, (_, k) => k),
                amplitudes.slice(i, j)
            );
            if (!reg || reg.slope >= 0) continue;
            if (reg.r2 < minR2) continue;
            windows.push({ start: i, end: j, ss_reg: reg.ss_reg, slope: reg.slope, r2: reg.r2 });
        }
    }

    if (windows.length === 0) return result;

    // Sort windows by end position
    windows.sort((a, b) => a.end - b.end);

    // DP: find max-SS non-overlapping set
    const dp = new Float64Array(n + 1);    // dp[pos] = best total SS using windows ending at or before pos
    const choice = new Array(n + 1);        // choice[pos] = list of chosen windows
    for (let pos = 0; pos <= n; pos++) choice[pos] = [];

    let wi = 0; // window index
    for (let pos = 1; pos <= n; pos++) {
        // Carry forward
        dp[pos] = dp[pos - 1];
        choice[pos] = choice[pos - 1];

        // Try all windows ending at this position
        while (wi < windows.length && windows[wi].end === pos) {
            const w = windows[wi];
            const val = dp[w.start] + w.ss_reg;
            if (val > dp[pos]) {
                dp[pos] = val;
                choice[pos] = [...choice[w.start], w];
            }
            wi++;
        }
    }

    result.sequences = choice[n];
    result.seq_r2 = dp[n] / totalSS;
    return result;
}

/**
 * Exponential decay fit: y = a * exp(b * x)
 * Computed by log-transforming y and fitting linear regression.
 * R² is computed on original scale.
 *
 * @param {number[]} x
 * @param {number[]} y
 * @returns {{ a: number, b: number, r2: number, predict: function }|null}
 */
function exponentialFit(x, y) {
    // Filter to valid pairs with positive y
    const pairs = [];
    for (let i = 0; i < x.length; i++) {
        if (x[i] != null && y[i] != null && isFinite(x[i]) && isFinite(y[i]) && y[i] > 0) {
            pairs.push([x[i], y[i]]);
        }
    }
    if (pairs.length < 2) return null;

    const xs = pairs.map(p => p[0]);
    const logYs = pairs.map(p => Math.log(p[1]));
    const ys = pairs.map(p => p[1]);

    // Linear regression on (x, log(y))
    const reg = linearRegression(xs, logYs);
    const a = Math.exp(reg.intercept);
    const b = reg.slope;

    // R² on original scale
    let yMean = 0;
    for (let i = 0; i < ys.length; i++) yMean += ys[i];
    yMean /= ys.length;
    let ssTot = 0, ssRes = 0;
    for (let i = 0; i < xs.length; i++) {
        const pred = a * Math.exp(b * xs[i]);
        ssTot += (ys[i] - yMean) ** 2;
        ssRes += (ys[i] - pred) ** 2;
    }
    const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

    return { a, b, r2, predict: (xVal) => a * Math.exp(b * xVal) };
}

// ── Group Comparison Tab ───────────────────────────────────────

let highlightedSubject = null;

// Per-subject checked state for the group tab.  Map<subjectName, bool>.
// Saved-event subjects default ON; auto-only subjects default to the
// current state of the "Include auto-detected events" toggle.  The
// user can override either way via the subject-list checkboxes.
let _groupSubjectChecked = {};

async function loadGroup() {
    const container = document.getElementById('groupPlots');

    // Always fetch with include_auto=true so we see the full subject
    // list regardless of toggle state.  ``has_saved_events`` tags
    // which ones have saved CSVs; the toggle and per-subject
    // checkboxes decide who actually contributes to the plots.
    if (cachedGroup) {
        renderGroupPlots();
        return;
    }

    container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">Loading group data...</div>';

    try {
        const src = document.getElementById('resultsSourceSelect')?.value || 'auto';
        cachedGroup = await API.get(`/api/results/group?include_auto=true&source=${src}`);
        _initGroupSubjectChecked();
        renderGroupPlots();
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="grid-column:1/-1;color:#d32f2f;">${e.message}</div>`;
    }
}

function _initGroupSubjectChecked() {
    if (!cachedGroup || !cachedGroup.subjects) return;
    const includeAuto = document.getElementById('includeAutoToggle').checked;
    _groupSubjectChecked = {};
    cachedGroup.subjects.forEach(s => {
        _groupSubjectChecked[s.name] = s.has_saved_events ? true : includeAuto;
    });
}

// Toggle handler — flip default-state of non-saved subjects to match
// the "Include auto-detected events" toggle, without disturbing
// saved-event subjects or explicit overrides the user made on saved
// subjects.  Auto-only subjects always follow the toggle.
function _onIncludeAutoChanged() {
    const includeAuto = document.getElementById('includeAutoToggle').checked;
    if (cachedGroup && cachedGroup.subjects) {
        cachedGroup.subjects.forEach(s => {
            if (!s.has_saved_events) _groupSubjectChecked[s.name] = includeAuto;
        });
    }
    renderGroupPlots();
}

function _activeGroupSubjects() {
    if (!cachedGroup || !cachedGroup.subjects) return [];
    return cachedGroup.subjects.filter(s => _groupSubjectChecked[s.name]);
}

function _renderGroupSubjectList() {
    const host = document.getElementById('groupSubjectList');
    if (!host) return;
    if (!cachedGroup || !cachedGroup.subjects) { host.innerHTML = ''; return; }

    // Group subjects by diagnosis so the chips stay visually clustered.
    const groups = cachedGroup.groups || [];
    const byGroup = {};
    groups.forEach(g => { byGroup[g] = []; });
    cachedGroup.subjects.forEach(s => {
        const g = s.diagnosis || 'Control';
        (byGroup[g] = byGroup[g] || []).push(s);
    });

    let html = '';
    groups.forEach(g => {
        const color = (typeof GROUP_COLORS !== 'undefined' && GROUP_COLORS[g]) || '#999';
        (byGroup[g] || []).forEach(s => {
            const checked = !!_groupSubjectChecked[s.name];
            const dim = !s.has_saved_events ? ' dim' : '';
            const safe = s.name.replace(/"/g, '&quot;');
            html += `<label class="${dim.trim()}" title="${s.has_saved_events ? 'Saved events' : 'Auto-detected events'} (${g})">
                <input type="checkbox" data-subject="${safe}" ${checked ? 'checked' : ''}>
                <span class="gsl-dot" style="background:${color};"></span>
                ${s.name}${s.has_saved_events ? '' : ' *'}
            </label>`;
        });
    });
    host.innerHTML = html;

    host.querySelectorAll('input[type=checkbox]').forEach(cb => {
        cb.addEventListener('change', (e) => {
            const name = e.target.getAttribute('data-subject');
            _groupSubjectChecked[name] = e.target.checked;
            renderGroupPlots();
        });
    });
}

window._groupSelectAll = function (state) {
    if (!cachedGroup || !cachedGroup.subjects) return;
    const includeAuto = document.getElementById('includeAutoToggle').checked;
    cachedGroup.subjects.forEach(s => {
        // "None" always unchecks everything.  "All" respects the
        // include-auto toggle so auto-only subjects don't get pulled
        // in while the toggle is off — matches the dim/deactivated
        // semantics the user asked for.
        if (state) {
            _groupSubjectChecked[s.name] = s.has_saved_events ? true : includeAuto;
        } else {
            _groupSubjectChecked[s.name] = false;
        }
    });
    renderGroupPlots();
};

// Metrics organized as columns; rows = Mean, Variance (CV), Sequence Effect
const GROUP_METRICS = [
    {
        name: 'Amplitude', id: 'amplitude', defaultOn: true,
        mean: { key: 'mean_amplitude', unit: 'mm' },
        cv: { key: 'cv_amplitude', unit: '' },
        seq: { key: 'seq_amplitude', unit: '' },
    },
    {
        name: 'Peak Open Vel', id: 'peak_open_vel', defaultOn: true,
        mean: { key: 'mean_peak_open_vel', unit: 'mm/s' },
        cv: { key: 'cv_peak_open_vel', unit: '' },
        seq: { key: 'seq_peak_open_vel', unit: '' },
    },
    {
        name: 'Peak Close Vel', id: 'peak_close_vel', defaultOn: true,
        mean: { key: 'mean_peak_close_vel', unit: 'mm/s' },
        cv: { key: 'cv_peak_close_vel', unit: '' },
        seq: { key: 'seq_peak_close_vel', unit: '' },
    },
    {
        name: 'Power', id: 'power', defaultOn: true,
        mean: { key: 'mean_power', unit: 'mm\u00b2/s' },
        cv: { key: 'cv_power', unit: '' },
        seq: { key: 'seq_power', unit: '' },
    },
    {
        name: 'IMI', id: 'imi', defaultOn: true,
        mean: { key: 'mean_imi', unit: 's' },
        cv: { key: 'cv_imi', unit: '' },
        seq: { key: 'seq_imi', unit: '' },
    },
    {
        name: 'Frequency', id: 'frequency', defaultOn: false,
        mean: { key: 'frequency', unit: 'Hz' },
        cv: { key: 'cv_frequency', unit: '' },
        seq: { key: 'seq_frequency', unit: '' },
    },
    {
        name: 'Rel. Amplitude', id: 'rel_amplitude', defaultOn: false,
        mean: { key: 'mean_rel_amplitude', unit: '' },
        cv: { key: 'cv_rel_amplitude', unit: '' },
        seq: { key: 'seq_rel_amplitude', unit: '' },
    },
    {
        name: 'Avg Open Vel', id: 'mean_open_vel', defaultOn: false,
        mean: { key: 'mean_mean_open_vel', unit: 'mm/s' },
        cv: { key: 'cv_mean_open_vel', unit: '' },
        seq: { key: 'seq_mean_open_vel', unit: '' },
    },
    {
        name: 'Avg Close Vel', id: 'mean_close_vel', defaultOn: false,
        mean: { key: 'mean_mean_close_vel', unit: 'mm/s' },
        cv: { key: 'cv_mean_close_vel', unit: '' },
        seq: { key: 'seq_mean_close_vel', unit: '' },
    },
];

// Backward compat: flat list for any code that still references GROUP_PARAMS
const GROUP_PARAMS = GROUP_METRICS.flatMap(m => {
    const out = [];
    if (m.mean) out.push({ key: m.mean.key, label: `${m.name} (${m.mean.unit || ''})`.trim() });
    if (m.cv) out.push({ key: m.cv.key, label: `CV ${m.name}` });
    if (m.seq) out.push({ key: m.seq.key, label: `Seq. Effect: ${m.name}` });
    return out;
});

const GROUP_COLORS = {
    Control: '#4CAF50',
    MSA: '#FF5722',
    PD: '#2196F3',
    PSP: '#9C27B0',
};

let _groupMetricVisible = {};

function renderGroupPlots() {
    const container = document.getElementById('groupPlots');
    const data = cachedGroup;

    if (!data || !data.subjects || data.subjects.length === 0) {
        container.innerHTML = '<div class="results-no-data">No group data available</div>';
        return;
    }

    _renderGroupSubjectList();

    // Initialize visibility from defaults
    if (Object.keys(_groupMetricVisible).length === 0) {
        GROUP_METRICS.forEach(m => { _groupMetricVisible[m.id] = m.defaultOn; });
    }

    const visibleMetrics = GROUP_METRICS.filter(m => _groupMetricVisible[m.id]);

    // Build checkbox bar
    let html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;align-items:center;">';
    html += '<span style="font-size:12px;color:var(--text-muted);font-weight:600;">Show:</span>';
    GROUP_METRICS.forEach(m => {
        const checked = _groupMetricVisible[m.id] ? 'checked' : '';
        html += `<label style="display:flex;align-items:center;gap:3px;font-size:11px;cursor:pointer;">
            <input type="checkbox" ${checked} onchange="_toggleGroupMetric('${m.id}', this.checked)">
            ${m.name}
        </label>`;
    });
    html += '</div>';

    // Scrollable grid: 3 rows (Mean, Variance, Sequence) × N metric columns
    const ROW_DEFS = [
        { label: 'Mean', field: 'mean', height: 200 },
        { label: 'Variance', field: 'cv', height: 180 },
        { label: 'Sequence Effect', field: 'seq', height: 180 },
    ];

    const colW = Math.max(200, Math.min(280, (container.clientWidth - 80) / visibleMetrics.length));

    html += '<div style="overflow-x:auto;">';
    html += `<div style="display:grid;grid-template-columns:60px repeat(${visibleMetrics.length}, ${colW}px);gap:0;">`;

    ROW_DEFS.forEach((row, ri) => {
        // Row label (first column)
        html += `<div style="display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--text-muted);writing-mode:vertical-rl;text-orientation:mixed;transform:rotate(180deg);padding:4px;">${row.label}</div>`;

        visibleMetrics.forEach((m, ci) => {
            const spec = m[row.field];
            const divId = `grpPlot_${m.id}_${row.field}`;
            if (spec) {
                html += `<div style="height:${row.height}px;">
                    ${ri === 0 ? `<div style="text-align:center;font-size:13px;font-weight:700;padding:4px 0 0;">${m.name}</div>` : ''}
                    <div id="${divId}" style="height:${row.height - (ri === 0 ? 24 : 0)}px;"></div>
                </div>`;
            } else {
                html += `<div style="height:${row.height}px;"></div>`;
            }
        });
    });

    html += '</div></div>';

    // ── Dose-response section ───────────────────────────────
    // Same Y variables as the Mean row, plotted against time
    // since last levodopa dose (parsed from each subject's
    // ``last_dose`` clinical field).  Subjects without a
    // parseable last-dose value are excluded.
    html += `<div style="margin-top:24px;border-top:1px solid var(--border, #e0e0e0);padding-top:12px;">`;
    html += `<div style="font-size:13px;font-weight:700;color:var(--text-muted);margin-bottom:4px;">
        Time since last levodopa dose</div>`;
    html += '<div style="overflow-x:auto;">';
    html += `<div style="display:grid;grid-template-columns:60px repeat(${visibleMetrics.length}, ${colW}px);gap:0;">`;
    ROW_DEFS.forEach((row, ri) => {
        html += `<div style="display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:var(--text-muted);writing-mode:vertical-rl;text-orientation:mixed;transform:rotate(180deg);padding:4px;">${row.label}</div>`;
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            const divId = `grpPlotDose_${m.id}_${row.field}`;
            if (spec) {
                html += `<div style="height:${row.height}px;">
                    ${ri === 0 ? `<div style="text-align:center;font-size:13px;font-weight:700;padding:4px 0 0;">${m.name}</div>` : ''}
                    <div id="${divId}" style="height:${row.height - (ri === 0 ? 24 : 0)}px;"></div>
                </div>`;
            } else {
                html += `<div style="height:${row.height}px;"></div>`;
            }
        });
    });
    html += '</div></div></div>';

    container.innerHTML = html;

    // Render each chart
    ROW_DEFS.forEach(row => {
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            if (!spec) return;
            const divId = `grpPlot_${m.id}_${row.field}`;
            const unit = spec.unit ? ` (${spec.unit})` : '';
            renderGroupBar(divId, data, spec.key, '');
        });
    });

    // Dose-response scatters — one per (metric × row) like the bars
    // above.  Row labels still come from ROW_DEFS so the rows stay
    // visually aligned with the corresponding bar-chart rows.
    ROW_DEFS.forEach((row, ri) => {
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            if (!spec) return;
            const divId = `grpPlotDose_${m.id}_${row.field}`;
            const label = ri === 0
                ? `${m.name}${spec.unit ? ' (' + spec.unit + ')' : ''}`
                : '';  // title only on the top (Mean) row, matching the bar grid
            renderDoseScatter(divId, data, spec.key, label);
        });
    });
}

function renderDoseScatter(divId, data, paramKey, paramLabel) {
    const groups = data.groups;
    // Active subjects with both a parseable last-dose value AND a
    // non-null parameter value.  Drop everyone else from the scatter
    // so the X axis isn't padded with missing points.
    const subjects = _activeGroupSubjects().filter(s =>
        s.time_since_dose_min != null
        && isFinite(s.time_since_dose_min)
        && s[paramKey] != null
        && isFinite(s[paramKey])
    );

    // One trace per group so the legend shows the diagnosis colors.
    const traces = groups.map(g => {
        const subs = subjects.filter(s => (s.diagnosis || 'Control') === g);
        return {
            x: subs.map(s => s.time_since_dose_min / 60.0),
            y: subs.map(s => s[paramKey]),
            text: subs.map(s => `${s.name}<br>${s.last_dose_raw || ''}`),
            type: 'scatter',
            mode: 'markers',
            name: g,
            marker: {
                color: GROUP_COLORS[g] || '#999',
                size: subs.map(s => highlightedSubject === s.name ? 10 : 7),
                opacity: 0.8,
                line: { color: '#333', width: 0.5 },
            },
            hovertemplate: '%{text}<br>%{x:.2f} h<br>%{y:.3f}<extra></extra>',
            showlegend: false,
        };
    });

    const layout = {
        title: { text: paramLabel, font: { size: 11, color: '#444' } },
        margin: { t: 28, b: 36, l: 45, r: 10 },
        xaxis: {
            title: { text: 'Hours since dose', font: { size: 10, color: '#666' } },
            color: '#666', gridcolor: '#f0f0f0', tickfont: { size: 9 },
            rangemode: 'tozero',
        },
        yaxis: { title: '', color: '#666', gridcolor: '#f0f0f0', tickfont: { size: 9 } },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
    };

    Plotly.newPlot(divId, traces, layout, {
        responsive: true,
        displayModeBar: false,
    });

    const plotDiv = document.getElementById(divId);
    plotDiv.on('plotly_click', (eventData) => {
        if (eventData.points && eventData.points.length > 0) {
            const pt = eventData.points[0];
            if (pt.text) {
                highlightSubject(String(pt.text).split('<br>')[0]);
            }
        }
    });
}

function _toggleGroupMetric(id, checked) {
    _groupMetricVisible[id] = checked;
    renderGroupPlots();
}

// Expose for inline onclick
window._toggleGroupMetric = _toggleGroupMetric;

function renderGroupBar(divId, data, paramKey, paramLabel) {
    const groups = data.groups;
    // Filter to the subjects whose checkbox is currently on.  This
    // is what makes unchecking a noisy subject immediately drop it
    // from every plot and re-scale the Y axes.
    const subjects = _activeGroupSubjects();

    // Group subjects
    const byGroup = {};
    groups.forEach(g => { byGroup[g] = []; });
    subjects.forEach(s => {
        const g = s.diagnosis || 'Control';
        if (byGroup[g]) byGroup[g].push(s);
    });

    // Compute group means and SEM
    const groupMeans = [];
    const groupSems = [];
    const groupLabels = [];

    groups.forEach(g => {
        const vals = byGroup[g].map(s => s[paramKey]).filter(v => v != null && isFinite(v));
        if (vals.length > 0) {
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const std = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length);
            const sem = std / Math.sqrt(vals.length);
            groupMeans.push(mean);
            groupSems.push(sem);
        } else {
            groupMeans.push(0);
            groupSems.push(0);
        }
        groupLabels.push(g);
    });

    const barTrace = {
        x: groupLabels.map((_, i) => i),
        y: groupMeans,
        customdata: groupLabels,
        type: 'bar',
        textposition: 'none',
        marker: { color: groupLabels.map(g => GROUP_COLORS[g] || '#999'), opacity: 0.3 },
        error_y: { type: 'data', array: groupSems, visible: true, color: '#666', thickness: 1.5 },
        hovertemplate: '%{customdata}<br>Mean: %{y:.3f}<extra></extra>',
        showlegend: false,
        width: 0.6,
    };

    // Individual dots with jitter — differentiate saved vs auto
    const dotX = [];
    const dotY = [];
    const dotText = [];
    const dotColors = [];
    const dotSizes = [];
    const dotOpacities = [];
    const dotLineWidths = [];
    const dotLineColors = [];

    groups.forEach((g, gi) => {
        const subs = byGroup[g];
        subs.forEach((s, si) => {
            const val = s[paramKey];
            if (val == null || !isFinite(val)) return;

            const jitter = subs.length > 1
                ? -0.25 + (si / (subs.length - 1)) * 0.5
                : 0;
            dotX.push(gi + jitter);
            dotY.push(val);
            dotText.push(s.name + (s.event_source === 'auto' ? ' (auto)' : ''));

            const groupColor = GROUP_COLORS[g] || '#999';
            const isHighlighted = highlightedSubject === s.name;
            const isAuto = s.event_source === 'auto';

            // Auto-detected: open circle (white fill, colored border)
            // Saved: filled circle
            dotColors.push(isAuto ? '#ffffff' : groupColor);
            dotLineColors.push(isAuto ? groupColor : '#333');
            dotLineWidths.push(isAuto ? 2 : (isHighlighted ? 2 : 0.5));
            dotSizes.push(isHighlighted ? 10 : 6);
            dotOpacities.push(isHighlighted ? 1.0 : 0.7);
        });
    });

    const dotTrace = {
        x: dotX,
        y: dotY,
        text: dotText,
        type: 'scatter',
        mode: 'markers',
        marker: {
            color: dotColors,
            size: dotSizes,
            opacity: dotOpacities,
            line: { color: dotLineColors, width: dotLineWidths },
        },
        hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>',
        showlegend: false,
    };

    const layout = {
        title: { text: paramLabel, font: { size: 11, color: '#444' } },
        margin: { t: 28, b: 30, l: 45, r: 10 },
        xaxis: {
            tickvals: groups.map((_, i) => i),
            ticktext: groups,
            color: '#666',
            tickfont: { size: 10 },
        },
        yaxis: { title: '', color: '#666', gridcolor: '#f0f0f0', tickfont: { size: 9 } },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        bargap: 0.5,
    };

    Plotly.newPlot(divId, [barTrace, dotTrace], layout, {
        responsive: true,
        displayModeBar: false,
    });

    // Click handler on dots
    const plotDiv = document.getElementById(divId);
    plotDiv.on('plotly_click', (eventData) => {
        if (eventData.points && eventData.points.length > 0) {
            const pt = eventData.points[0];
            if (pt.curveNumber === 1 && pt.text) {
                highlightSubject(pt.text.replace(' (auto)', ''));
            }
        }
    });
}

function highlightSubject(name) {
    highlightedSubject = highlightedSubject === name ? null : name;
    document.getElementById('highlightedSubject').textContent =
        highlightedSubject ? `Highlighted: ${highlightedSubject}` : '';

    // Re-render all group plots to update highlighting
    renderGroupPlots();
}

// ── Event listeners ────────────────────────────────────────────

document.getElementById('subjectSelect').addEventListener('change', (e) => {
    currentSubjectId = e.target.value;
    sessionStorage.setItem('dlc_lastSubjectId', String(currentSubjectId));
    if (typeof setLastSubject === 'function') setLastSubject(currentSubjectId);
    if (typeof setNavState === 'function') setNavState({ subjectId: parseInt(currentSubjectId) });
    cachedTraces = null;
    cachedMovements = null;
    cachedSequenceAssignments = null;
    updateNavLinks();

    // Purge any Plotly plots from the previous subject so there's no
    // flash of stale data while the new subject's API calls are in-flight.
    const _purge = (el) => {
        if (!el) return;
        if (window.Plotly) {
            el.querySelectorAll('.js-plotly-plot').forEach(p => {
                try { Plotly.purge(p); } catch {}
            });
        }
        el.innerHTML = '';
    };
    _purge(document.getElementById('distancePlots'));
    _purge(document.getElementById('distMovementPlots'));
    _purge(document.getElementById('movementPlots'));

    // Always reload distances for the new subject — regardless of which
    // tab was last active — so switching subjects never leaves stale
    // content behind.  Force the tab to 'distances' for the visible
    // individual-subject view.
    if (currentTab !== 'group') {
        currentTab = 'distances';
        // Update tab-button highlighting to match.
        document.querySelectorAll('#tabSwitcher .btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === 'distances');
        });
        document.querySelectorAll('.results-tab').forEach(el => {
            el.classList.toggle('active', el.id === 'tabDistances');
        });
        loadDistances(currentSubjectId);
    }
});

document.getElementById('prevSubjectBtn').addEventListener('click', () => navigateSubject(-1));
document.getElementById('nextSubjectBtn').addEventListener('click', () => navigateSubject(1));

document.querySelectorAll('#tabSwitcher .btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Movement controls: re-render on checkbox or sequence mode change
document.querySelectorAll('#movementControls input[data-param]').forEach(cb => {
    cb.addEventListener('change', () => {
        if (cachedMovements) renderMovementPlots();
    });
});
document.getElementById('sequenceMode').addEventListener('change', () => {
    if (cachedMovements) renderMovementPlots();
});

// Distance tab movement controls: re-render on checkbox or fit mode change
document.querySelectorAll('#distMovementControls input[data-dparam]').forEach(cb => {
    cb.addEventListener('change', () => {
        if (cachedMovements) renderDistMovementPlots();
    });
});
document.getElementById('distSequenceMode').addEventListener('change', () => {
    cachedSequenceAssignments = null;
    if (cachedMovements) renderDistMovementPlots();
});

// Overlay controls: re-render distance/velocity plots
['overlayPeakDist', 'overlayPeakOpenVel', 'overlayPeakCloseVel'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => {
        if (cachedTraces) renderAllDistancePlots();
    });
});

// Lock Y-axis toggle: switch between shared (locked) and per-trial
// (unlocked) Y-range sliders.  Seed the shared range from trial 0's
// current per-trial range when locking, and vice-versa, so the view
// doesn't jump.  Then re-render to rebuild sliders in the new mode.
document.getElementById('lockYAxis').addEventListener('change', () => {
    const locked = document.getElementById('lockYAxis').checked;
    if (locked) {
        // Adopt trial 0's range (or full) as the new shared range.
        ['dist', 'vel'].forEach(kind => {
            const t0 = _yTrial[kind][0];
            _yLocked[kind] = t0 ? { ...t0 }
                : (_yFull[kind] ? { min: _yFull[kind][0], max: _yFull[kind][1] } : null);
        });
    } else {
        // Seed every trial's range from the shared one so unlocking
        // keeps the current view until the user diverges them.
        ['dist', 'vel'].forEach(kind => {
            if (!_yLocked[kind]) return;
            const n = (cachedTraces && cachedTraces.trials) ? cachedTraces.trials.length : 0;
            for (let i = 0; i < n; i++) _yTrial[kind][i] = { ..._yLocked[kind] };
        });
    }
    if (cachedTraces) renderAllDistancePlots();
});

// Group comparison: auto-detect toggle.  No re-fetch — the data is
// the union of saved + auto already.  Toggle only changes the
// default-checked state of auto-only subjects.
document.getElementById('includeAutoToggle').addEventListener('change', _onIncludeAutoChanged);

document.getElementById('groupSelectAllBtn').addEventListener('click',
    () => window._groupSelectAll(true));
document.getElementById('groupSelectNoneBtn').addEventListener('click',
    () => window._groupSelectAll(false));

// Page-level distance source dropdown — applies to both Individual
// and Group Comparison tabs.  Auto = corrections → mp_combined →
// mp_forward (per the backend's _load_distances_and_trials).
document.getElementById('resultsSourceSelect')?.addEventListener('change', () => {
    cachedTraces = null;
    cachedMovements = null;
    cachedGroup = null;
    const activeTab = document.querySelector('.results-tab.active')?.id;
    if (activeTab === 'tabGroup') {
        loadGroup();
    } else if (currentSubjectId) {
        loadDistances(currentSubjectId);
    }
});

// ── Init ───────────────────────────────────────────────────────

const params = new URLSearchParams(window.location.search);
const initSubject = params.get('subject');
const initFrom = params.get('from');
const initTab = params.get('tab');

loadSubjects().then(() => {
    // 1. Determine which subject to show
    let subjectId = initSubject;
    if (!subjectId) {
        // No subject in URL — check nav state, then sessionStorage
        const _nav = typeof getNavState === 'function' ? getNavState() : {};
        subjectId = (_nav.subjectId ? String(_nav.subjectId) : null)
            || sessionStorage.getItem('dlc_lastSubjectId') || null;
    }
    if (subjectId) {
        document.getElementById('subjectSelect').value = subjectId;
        currentSubjectId = subjectId;
        sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
        if (typeof setNavState === 'function') setNavState({ subjectId: parseInt(subjectId) });
        updateNavLinks();
    }

    // 2. Determine which tab to show
    // 'individual' is a legacy alias used by the dashboard — map it to
    // 'distances' so the individual-subject view renders.
    const _tabAlias = { individual: 'distances' };
    const _normTab = _tabAlias[initTab] || initTab;
    if (_normTab && ['distances', 'movements', 'group'].includes(_normTab)) {
        // Explicit tab param takes highest priority
        switchTab(_normTab);
    } else if (!initSubject && !initFrom) {
        // No subject param and no source context → group comparison
        switchTab('group');
    } else if (currentSubjectId) {
        // Have a subject — check for a remembered tab.  The legacy
        // 'movements' tab is hidden (display:none) so never restore it;
        // map it (and any unknown value) to the visible 'distances'
        // tab.  Restoring 'movements' rendered a blank page when
        // arriving via the nav prev/next buttons (full reload to
        // /results?subject=X with no tab param).
        const lastTab = sessionStorage.getItem(`dlc_resultsTab_${currentSubjectId}`);
        const restoreTab = (lastTab === 'distances' || lastTab === 'group')
            ? lastTab : 'distances';
        switchTab(restoreTab);
    } else {
        // No subject at all → group comparison
        switchTab('group');
    }
});
