/* Results page: distance traces, movement parameters, group comparison */

// Static-site mode: the SE-based hand groupings aren't pre-computed
// into JSON files (they require live sequence-effect math per subject),
// so hide them from the dropdown when STATIC_RESULTS is set.
(function _trimStaticResultsControls() {
    if (!window.STATIC_RESULTS) return;
    const sel = document.getElementById('groupHandSelect');
    if (!sel) return;
    Array.from(sel.options).forEach(opt => {
        if (['larger_se', 'smaller_se'].includes(opt.value)) opt.remove();
    });
})();

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

    // On the group tab the subject selector acts as a "jump to an
    // individual subject" control: keep it enabled, reset it to the
    // placeholder, and hide the prev/next buttons (they only make
    // sense within the individual view).
    const isGroup = tab === 'group';
    _setSubjectNavVisible(!isGroup);
    const sel = document.getElementById('subjectSelect');
    if (sel) {
        sel.disabled = false;
        if (isGroup) {
            sel.value = '';
            const navSel = document.getElementById('navSubjectSelect');
            if (navSel) navSel.value = '';
        }
    }

    // Load data for the active tab
    if ((tab === 'distances' || tab === 'movements') && currentSubjectId) {
        loadDistances(currentSubjectId);
    } else if (tab === 'group') {
        loadGroup();
    }
}

// Show/hide the prev/next subject buttons (both the local header copy
// and the nav-bar copy created by nav.js).
function _setSubjectNavVisible(visible) {
    ['prevSubjectBtn', 'nextSubjectBtn', 'navPrevSubjectBtn', 'navNextSubjectBtn'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = visible ? '' : 'none';
    });
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
            renderShapeOverlayPlots();
        } else {
            if (movControls) movControls.style.display = 'none';
            if (movContainer) movContainer.innerHTML = '';
            const shapeSec = document.getElementById('shapeOverlaySection');
            if (shapeSec) shapeSec.style.display = 'none';
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

    // X-scale slider: seconds-of-trace-per-screen-width.  Slider is
    // reversed (right = stretched): raw 5..120 maps to secPerWidth =
    // 125 - raw (120..5).  Default value=95 → 30s/window.  Matches the
    // distance/velocity slider behavior.
    const _mxRaw = parseFloat(document.getElementById('movXScaleSlider')?.value);
    const secPerWidth = isFinite(_mxRaw) ? (125 - _mxRaw) : 30;
    const containerW = container.clientWidth || container.parentElement?.clientWidth || 1000;

    // Total time spanned by all trials concatenated (matches the way
    // the movement plot lays out per-trial subplots side-by-side).
    let totalSec = 0;
    (data.trials || []).forEach(t => {
        const fps = t.fps || 60;
        const n = (t.distances && t.distances.length) || 0;
        totalSec += n / fps;
    });
    if (totalSec <= 0) totalSec = 1;
    const plotW = Math.max(containerW, (totalSec / secPerWidth) * containerW);

    // Header row above each plot: label on the left, copy-to-clipboard
    // button on the right.  Anchored above the scroll wrapper so neither
    // moves with horizontal scrolling.
    const _titleHeader = (param) => {
        const h = document.createElement('div');
        h.style.cssText = 'display:flex;align-items:center;justify-content:space-between;'
                        + 'padding:2px 8px 2px 4px;';
        const label = document.createElement('span');
        label.textContent = PARAM_LABELS[param] || param;
        label.style.cssText = 'font-size:13px;font-weight:600;color:#444;';
        h.appendChild(label);
        h.appendChild(_makeCopyBtn(btn => _copyMovementPlot(param, btn)));
        return h;
    };

    {
        // One plot per row; width derived from seconds/window.  Wider
        // than the container → horizontal-scroll wrapper (scroll synced
        // across all params).
        container.style.display = 'block';
        const wraps = [];
        params.forEach(param => {
            const block = document.createElement('div');
            block.style.marginBottom = '16px';
            block.appendChild(_titleHeader(param));   // anchored title
            const wrap = document.createElement('div');
            wrap.className = 'mov-scroll-wrap';
            wrap.style.cssText = 'overflow-x:auto;width:100%;';
            const div = document.createElement('div');
            div.id = `distMovPlot_${param}`;
            div.style.height = '300px';
            div.style.width = plotW + 'px';
            wrap.appendChild(div);
            block.appendChild(wrap);
            container.appendChild(block);
            wraps.push(wrap);
            renderMovementScatter(div.id, data, param, seqMode, plotW);
        });
        // Synchronize horizontal scrolling — scrolling any movement
        // plot scrolls them all to the same offset.
        let _syncing = false;
        wraps.forEach(w => {
            w.addEventListener('scroll', () => {
                if (_syncing) return;
                _syncing = true;
                const sl = w.scrollLeft;
                for (const o of wraps) { if (o !== w) o.scrollLeft = sl; }
                _syncing = false;
            });
        });
    }
}

function _updateSourceSelector(availableSources, activeSource) {
    // No-op: the page-level #resultsSourceSelect is now the single
    // source of truth for distance source.  The per-subject auto
    // result is surfaced via the badge in renderAllDistancePlots.
}

// Movement-shape overlay: per-trial superimposed opening→closing
// distance segments with t=0 at the opening event.  X-axis range,
// highlighted-movement index and mean-trace visibility are driven
// from the shape-control bar above the plots.
let _shapeData = null;   // {trials:[{name, fps, segments:[{xs,ys}], maxT}], xMaxDefault, maxMov}

function _buildShapeData() {
    const trData = cachedTraces, mvData = cachedMovements;
    if (!trData || !trData.trials || !mvData || !mvData.movements) return null;
    const movsByTrial = {};
    mvData.movements.forEach(m => {
        if (m.open_frame_local == null || m.close_frame_local == null) return;
        (movsByTrial[m.trial_idx] ||= []).push(m);
    });
    if (!Object.keys(movsByTrial).length) return null;

    let globalMaxT = 0, maxMov = 0;
    const trials = trData.trials.map((trial, idx) => {
        const movs = movsByTrial[idx] || [];
        const fps = trial.fps || 60;
        const dist = trial.distances || [];
        const segments = movs.map(m => {
            const o = Math.max(0, m.open_frame_local | 0);
            const c = Math.min(dist.length - 1, m.close_frame_local | 0);
            const p = (m.peak_frame_local != null)
                ? Math.min(c, Math.max(o, m.peak_frame_local | 0))
                : o;
            const xs = [], ys = [];
            for (let f = o; f <= c; f++) {
                const v = dist[f];
                if (v == null) continue;
                xs.push((f - o) / fps);   // raw: open at 0
                ys.push(v);
            }
            // Peak / close offsets in the segment's own time axis.
            const peakT = (p - o) / fps;
            const closeT = (c - o) / fps;
            const peakY = (dist[p] != null) ? dist[p] : null;
            return { xs, ys, peakT, closeT, peakY };
        }).filter(s => s.xs.length >= 2);
        const maxT = segments.reduce((a, s) => Math.max(a, s.xs[s.xs.length - 1]), 0);
        if (maxT > globalMaxT) globalMaxT = maxT;
        if (segments.length > maxMov) maxMov = segments.length;
        return { name: trial.name, fps, segments, maxT };
    });

    // Sensible default x-max: covers the longest movement plus a touch
    // of headroom, snapped to the nearest 0.1 s.
    const xMaxDefault = Math.max(0.5, Math.ceil((globalMaxT * 1.05) * 10) / 10);
    return { trials, xMaxDefault, maxMov };
}

// Resample (xs, ys) to nPts samples spanning [xLo..xHi] using linear
// interpolation.  Returns NaN for grid points outside the segment.
function _resampleXY(xs, ys, xLo, xHi, nPts) {
    const out = new Array(nPts);
    const xStart = xs[0], xEnd = xs[xs.length - 1];
    let j = 0;
    for (let i = 0; i < nPts; i++) {
        const t = xLo + (i / (nPts - 1)) * (xHi - xLo);
        if (t < xStart || t > xEnd) { out[i] = NaN; continue; }
        while (j < xs.length - 2 && xs[j + 1] < t) j++;
        while (j > 0 && xs[j] > t) j--;
        const x0 = xs[j], x1 = xs[j + 1];
        const f = (x1 > x0) ? (t - x0) / (x1 - x0) : 0;
        out[i] = ys[j] + f * (ys[j + 1] - ys[j]);
    }
    return out;
}

// Per-trial highlight index (1-indexed, 0 = none).  Reset on each
// renderShapeOverlayPlots() call.
let _shapeHighlight = {};
// Per-trial highlight setter — populated by _buildShapeOverlayCells so
// click handlers in _redrawOneTrial can update the slider in place.
let _shapeSetHighlight = {};

function renderShapeOverlayPlots() {
    const section = document.getElementById('shapeOverlaySection');
    const container = document.getElementById('shapeOverlayPlots');
    if (!section || !container) return;
    _shapeData = _buildShapeData();
    if (!_shapeData) {
        section.style.display = 'none';
        return;
    }
    section.style.display = '';

    // Configure x-scale slider from the per-subject default.
    const xSl = document.getElementById('shapeXScaleSlider');
    const xVal = document.getElementById('shapeXScaleVal');
    if (xSl) {
        const xMaxCap = Math.max(5, Math.ceil(_shapeData.xMaxDefault * 1.5));
        xSl.max = String(xMaxCap);
        xSl.value = String(_shapeData.xMaxDefault.toFixed(2));
        if (xVal) xVal.textContent = `${(+xSl.value).toFixed(2)} s`;
    }
    _shapeHighlight = {};

    // Force a fresh cell build (new subject — sliders / titles differ).
    const cont = document.getElementById('shapeOverlayPlots');
    if (cont) cont.innerHTML = '';
    _drawShapeOverlayPlots();
}

// Cross-correlation alignment, peak-aligned.  Each segment is
// resampled onto a peak-aligned grid, a mean reference is built from
// those peak-aligned traces, and per-segment lags maximizing the
// cross-correlation against the mean are refined with parabolic
// interpolation.  After the first pass, the mean is rebuilt from the
// newly-shifted segments and a second pass refines the lags.  The
// returned shifts are applied to the open-aligned xs, so the final
// shift includes both the peak offset and the correlation correction.
function _computeCorrShifts(segments, xMax) {
    const N = segments.length;
    if (!N) return [];
    const dt = 1 / 240;
    const xLoRef = -xMax, xHiRef = xMax;
    const nGrid = Math.round((xHiRef - xLoRef) / dt) + 1;

    // Peak-aligned resamples: shift each segment so its peak sits at
    // t=0 in the reference grid.
    const peakAligned = segments.map(s =>
        _resampleXY(s.xs.map(x => x - s.peakT), s.ys, xLoRef, xHiRef, nGrid));

    // Build a mean reference, optionally weighting samples by an
    // existing per-segment correction (used on the second iteration).
    // The extent uses the same density threshold as the displayed
    // average trace so the per-peak constraint matches what the user
    // sees.
    const minCount = Math.max(2, Math.ceil(N * 0.25));
    const buildMean = (extraCorrSamples) => {
        const sums = new Float64Array(nGrid), counts = new Float64Array(nGrid);
        for (let si = 0; si < N; si++) {
            const arr = peakAligned[si];
            const k = extraCorrSamples ? extraCorrSamples[si] : 0;
            // Shift arr by k samples (positive k → arr lands later in grid).
            for (let i = 0; i < nGrid; i++) {
                const j = i - Math.round(k);
                if (j < 0 || j >= nGrid) continue;
                const v = arr[j];
                if (isFinite(v)) { sums[i] += v; counts[i] += 1; }
            }
        }
        const ref = new Float64Array(nGrid);
        for (let i = 0; i < nGrid; i++) ref[i] = counts[i] >= minCount ? sums[i] / counts[i] : NaN;
        let first = -1, last = -1;
        for (let i = 0; i < nGrid; i++) {
            if (isFinite(ref[i])) { if (first < 0) first = i; last = i; }
        }
        return { ref, meanXmin: first >= 0 ? xLoRef + first * dt : -Infinity,
                      meanXmax: last  >= 0 ? xLoRef + last  * dt :  Infinity };
    };

    const maxLagSamples = Math.round(xMax / 2 / dt);

    // One pass: given a reference (and its [meanXmin, meanXmax]),
    // return the per-segment best lags (in samples, sub-frame refined).
    const onePass = (ref, meanXmin, meanXmax) => {
        const lags = new Array(N).fill(0);
        for (let si = 0; si < N; si++) {
            const seg = peakAligned[si];
            // Constraint: after lag k, the segment's peak (at t=0 in
            // peak-aligned coords) lands at t = -k*dt in display coords.
            // Require -k*dt ∈ [meanXmin, meanXmax].
            const kLoConstr = Math.ceil(-meanXmax / dt);
            const kHiConstr = Math.floor(-meanXmin / dt);
            const kLo = Math.max(-maxLagSamples, kLoConstr);
            const kHi = Math.min( maxLagSamples, kHiConstr);
            if (kLo > kHi) { lags[si] = 0; continue; }
            const corr = new Float64Array(2 * maxLagSamples + 1);
            for (let i = 0; i < corr.length; i++) corr[i] = -Infinity;
            let bestLag = 0, bestC = -Infinity;
            for (let k = kLo; k <= kHi; k++) {
                let s = 0, n = 0, sx = 0, sy = 0, sxx = 0, syy = 0;
                for (let i = 0; i < nGrid; i++) {
                    const j = i - k;
                    if (j < 0 || j >= nGrid) continue;
                    const a = seg[i], b = ref[j];
                    if (!isFinite(a) || !isFinite(b)) continue;
                    sx += a; sy += b; sxx += a * a; syy += b * b; s += a * b; n += 1;
                }
                if (n < 5) continue;
                const mx = sx / n, my = sy / n;
                const num = s - n * mx * my;
                const den = Math.sqrt(Math.max(1e-12, (sxx - n * mx * mx) * (syy - n * my * my)));
                const c = num / den;
                corr[k + maxLagSamples] = c;
                if (c > bestC) { bestC = c; bestLag = k; }
            }
            let refined = bestLag;
            if (bestLag > kLo && bestLag < kHi) {
                const c0 = corr[bestLag - 1 + maxLagSamples];
                const c1 = corr[bestLag + maxLagSamples];
                const c2 = corr[bestLag + 1 + maxLagSamples];
                const denom = c0 - 2 * c1 + c2;
                if (isFinite(c0) && isFinite(c2) && denom < 0) {
                    refined = bestLag + 0.5 * (c0 - c2) / denom;
                    if (refined < kLo) refined = kLo;
                    if (refined > kHi) refined = kHi;
                }
            }
            lags[si] = refined;
        }
        return lags;
    };

    // Pass 1 — peak-aligned mean as reference.
    let mean1 = buildMean(null);
    const lags1 = onePass(mean1.ref, mean1.meanXmin, mean1.meanXmax);

    // Pass 2 — rebuild the mean from the corr-shifted traces, refine.
    const lagsRounded = lags1.map(l => Math.round(l));
    const mean2 = buildMean(lagsRounded);
    let lags2 = onePass(mean2.ref, mean2.meanXmin, mean2.meanXmax);

    // Final guard: build the mean *with the actual pass-2 shifts* and
    // clamp any per-segment lag whose post-shift peak falls outside
    // that displayed-mean extent.  Catches edge cases where the pass-2
    // constraint (computed against mean2) lets the peak escape the
    // mean that ends up on screen.
    const finalMean = buildMean(lags2.map(l => Math.round(l)));
    const lo = finalMean.meanXmin, hi = finalMean.meanXmax;
    if (isFinite(lo) && isFinite(hi)) {
        lags2 = lags2.map(l => {
            const peakOut = -l * dt;        // peak's display time
            if (peakOut < lo) return -lo / dt;
            if (peakOut > hi) return -hi / dt;
            return l;
        });
    }

    // Final shift applied to open-aligned xs: align peak to 0 then
    // apply the (sub-frame) correlation lag.  shift_t = -peakT - lag*dt.
    return segments.map((s, si) => -s.peakT - lags2[si] * dt);
}

// Build cells once (title + per-trial slider + plot div).  Re-used by
// _drawShapeOverlayPlots — DOM is left in place between draws so the
// movement sliders don't get destroyed mid-drag.
function _buildShapeOverlayCells() {
    const container = document.getElementById('shapeOverlayPlots');
    if (!container || !_shapeData) return;
    container.innerHTML = '';
    _shapeData.trials.forEach((trial, idx) => {
        const cell = document.createElement('div');
        cell.className = 'results-plot-cell';
        // Fixed-width column so all trials sit in one horizontally
        // scrollable row.
        cell.style.cssText = 'flex:0 0 auto;width:380px;min-width:0;';

        // Title with an inline per-trial movement-highlight slider.
        const title = document.createElement('div');
        title.style.cssText = 'font-size:12px;font-weight:600;margin-bottom:4px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;';
        const tn = document.createElement('span');
        tn.textContent = trial.name;
        title.appendChild(tn);

        const moveLabel = document.createElement('label');
        moveLabel.style.cssText = 'display:flex;align-items:center;gap:4px;font-size:11px;font-weight:400;color:var(--text-muted);';
        moveLabel.title = 'Highlight one movement.  0 = none.';
        moveLabel.appendChild(document.createTextNode('Movement #:'));
        const moveSlider = document.createElement('input');
        moveSlider.type = 'range'; moveSlider.min = '0';
        moveSlider.max = String(trial.segments.length);
        moveSlider.step = '1';
        moveSlider.value = String(_shapeHighlight[idx] || 0);
        moveSlider.style.cssText = 'width:140px;';
        const moveVal = document.createElement('span');
        moveVal.style.cssText = 'min-width:28px;text-align:right;';
        moveVal.textContent = moveSlider.value === '0' ? 'All' : moveSlider.value;
        moveLabel.appendChild(moveSlider);
        moveLabel.appendChild(moveVal);

        const _setHighlight = (v) => {
            const N = trial.segments.length;
            let n = parseInt(v, 10);
            if (!isFinite(n)) n = 0;
            if (n < 0) n = 0;
            if (n > N) n = N;
            _shapeHighlight[idx] = n;
            moveSlider.value = String(n);
            moveVal.textContent = n === 0 ? 'All' : String(n);
            _redrawOneTrial(idx);
        };
        _shapeSetHighlight[idx] = _setHighlight;
        moveSlider.addEventListener('input', () => _setHighlight(moveSlider.value));

        // Prev / next arrow buttons.
        const prevBtn = document.createElement('button');
        prevBtn.type = 'button';
        prevBtn.className = 'btn btn-sm';
        prevBtn.textContent = '◀';
        prevBtn.title = 'Previous movement';
        prevBtn.style.cssText = 'padding:0 6px;font-size:11px;line-height:1.4;';
        prevBtn.addEventListener('click', () => _setHighlight((_shapeHighlight[idx] || 0) - 1));
        const nextBtn = document.createElement('button');
        nextBtn.type = 'button';
        nextBtn.className = 'btn btn-sm';
        nextBtn.textContent = '▶';
        nextBtn.title = 'Next movement';
        nextBtn.style.cssText = 'padding:0 6px;font-size:11px;line-height:1.4;';
        nextBtn.addEventListener('click', () => _setHighlight((_shapeHighlight[idx] || 0) + 1));
        moveLabel.appendChild(prevBtn);
        moveLabel.appendChild(nextBtn);

        title.appendChild(moveLabel);
        cell.appendChild(title);

        const plotDiv = document.createElement('div');
        plotDiv.id = `shapeOverlayPlot_${idx}`;
        plotDiv.style.cssText = 'width:100%;height:240px;';
        cell.appendChild(plotDiv);

        // Correlation matrix (single heatmap, optionally clustered).
        // Always uses the dendrogram subplot's domain so unsorted and
        // clustered modes render at the same physical size.  Width
        // matches the shape-overlay plot above it.
        const corrDiv = document.createElement('div');
        corrDiv.id = `shapeCorrPlot_${idx}`;
        corrDiv.style.cssText = 'width:100%;margin-top:6px;';
        cell.appendChild(corrDiv);

        container.appendChild(cell);
    });
}

// Read all global control values used by every per-trial draw.
function _shapeOverlayState() {
    const xMax = parseFloat(document.getElementById('shapeXScaleSlider')?.value)
              || (_shapeData ? _shapeData.xMaxDefault : 2.5);
    const showMean = !!document.getElementById('shapeShowMean')?.checked;
    const alignR = document.querySelector('input[name="shapeAlign"]:checked');
    const align = alignR ? alignR.value : 'open';
    let xLo, xHi;
    if (align === 'peak' || align === 'corr') { xLo = -xMax / 2; xHi = xMax / 2; }
    else if (align === 'close') { xLo = -xMax; xHi = 0; }
    else { xLo = 0; xHi = xMax; }
    return { xMax, showMean, align, xLo, xHi };
}

function _drawShapeOverlayPlots() {
    if (!_shapeData) return;
    // First draw of a new subject: build the cells.  Subsequent draws
    // reuse them so the sliders stay alive during interaction.
    const container = document.getElementById('shapeOverlayPlots');
    const haveCells = container && container.children.length === _shapeData.trials.length;
    if (!haveCells) _buildShapeOverlayCells();
    _shapeData.trials.forEach((_t, idx) => _redrawOneTrial(idx));
}

function _redrawOneTrial(idx) {
    if (!_shapeData) return;
    const trial = _shapeData.trials[idx];
    if (!trial) return;
    const plotDiv = document.getElementById(`shapeOverlayPlot_${idx}`);
    if (!plotDiv) return;
    const { xMax, showMean, align, xLo, xHi } = _shapeOverlayState();

    if (!trial.segments.length) {
        plotDiv.innerHTML = '<div class="results-no-data">No movements</div>';
        return;
    }

    const hiIdx = _shapeHighlight[idx] || 0;

        // Per-segment alignment shift (subtracted from xs at draw
        // time so the raw segment data can stay in open-aligned form).
        // For 'corr' we precompute optimal shifts via cross-corr.
        let corrShifts = null;
        if (align === 'corr') {
            corrShifts = _computeCorrShifts(trial.segments, xMax);
        }
        const shiftOf = (s, si) => align === 'peak' ? -s.peakT
                            : align === 'close' ? -s.closeT
                            : align === 'corr' ? (corrShifts ? corrShifts[si] : 0)
                            : 0;
        const shiftedX = (s, si) => s.xs.map(x => x + shiftOf(s, si));

        // Dim the rest when a movement is highlighted.
        const dimmed = hiIdx > 0;
        const baseColor = dimmed
            ? 'rgba(31,119,180,0.15)'
            : 'rgba(31,119,180,0.45)';
        const traces = trial.segments.map((s, mi) => ({
            x: shiftedX(s, mi), y: s.ys, type: 'scatter', mode: 'lines',
            line: { width: 1, color: baseColor },
            customdata: new Array(s.xs.length).fill(mi),
            hoverinfo: 'skip', showlegend: false,
        }));

        // Average trace (resampled to the displayed time grid).  Drawn
        // before any highlighted movement so the highlight sits on top.
        if (showMean) {
            const N = 200;
            const sums = new Float64Array(N), counts = new Float64Array(N);
            for (let mi = 0; mi < trial.segments.length; mi++) {
                const s = trial.segments[mi];
                const sx = shiftedX(s, mi);
                const ys = _resampleXY(sx, s.ys, xLo, xHi, N);
                for (let i = 0; i < N; i++) {
                    if (isFinite(ys[i])) { sums[i] += ys[i]; counts[i] += 1; }
                }
            }
            const meanXs = [], meanYs = [];
            const minCount = Math.max(2, trial.segments.length * 0.25);
            for (let i = 0; i < N; i++) {
                if (counts[i] >= minCount) {
                    meanXs.push(xLo + (i / (N - 1)) * (xHi - xLo));
                    meanYs.push(sums[i] / counts[i]);
                }
            }
            if (meanXs.length >= 2) {
                traces.push({
                    x: meanXs, y: meanYs, type: 'scatter', mode: 'lines',
                    line: { width: 3.5, color: '#000' },
                    hoverinfo: 'skip', showlegend: false,
                });
            }
        }

        // Highlighted movement (1-indexed in the UI).  Skip if this
        // trial doesn't have that many movements.
        if (hiIdx > 0 && hiIdx <= trial.segments.length) {
            const s = trial.segments[hiIdx - 1];
            traces.push({
                x: shiftedX(s, hiIdx - 1), y: s.ys, type: 'scatter', mode: 'lines',
                line: { width: 2.5, color: '#d32f2f' },
                hoverinfo: 'skip', showlegend: false,
            });
        }

        // Peak-event markers.  Gate on the global checkbox.  When the
        // movement slider is at "All" (0) mark every movement; otherwise
        // mark just the highlighted one.
        const showPeaks = !!document.getElementById('shapeShowPeaks')?.checked;
        if (showPeaks) {
            const markIdxs = (hiIdx === 0)
                ? trial.segments.map((_, i) => i)
                : (hiIdx > 0 && hiIdx <= trial.segments.length ? [hiIdx - 1] : []);
            const pxs = [], pys = [];
            for (const mi of markIdxs) {
                const s = trial.segments[mi];
                if (s.peakY == null) continue;
                pxs.push(s.peakT + shiftOf(s, mi));
                pys.push(s.peakY);
            }
            if (pxs.length) {
                traces.push({
                    x: pxs, y: pys, type: 'scatter', mode: 'markers',
                    marker: { size: 9, color: '#d32f2f',
                              line: { color: '#000', width: 1 }, symbol: 'circle' },
                    customdata: markIdxs.slice(),
                    hoverinfo: 'skip', showlegend: false,
                });
            }
        }

        const xTitle = align === 'peak'
            ? 'Time from peak (s)'
            : align === 'corr' ? 'Time from correlation peak (s)'
            : align === 'close' ? 'Time from closing (s)'
                                : 'Time from opening (s)';
        const layout = {
            margin: { t: 6, b: 36, l: 48, r: 8 },
            xaxis: {
                title: { text: xTitle, font: { size: 11 } },
                range: [xLo, xHi], showline: true, linecolor: '#666',
                zeroline: (align !== 'open'), zerolinecolor: '#bbb',
                tickfont: { size: 10 },
            },
            yaxis: {
                title: { text: 'Distance (mm)', font: { size: 11 } },
                showline: true, linecolor: '#666', zeroline: false,
                tickfont: { size: 10 }, automargin: true,
            },
            plot_bgcolor: '#fff', paper_bgcolor: '#fff',
            hovermode: false, showlegend: false,
        };
        Plotly.react(plotDiv, traces, layout,
                     { responsive: true, displayModeBar: false });

        // Click on a movement line or peak marker → set the highlight.
        if (!plotDiv._clickBound) {
            plotDiv._clickBound = true;
            plotDiv.on('plotly_click', (ev) => {
                const p = ev?.points?.[0];
                if (!p || p.customdata == null) return;
                const mi = (typeof p.customdata === 'number')
                    ? p.customdata
                    : parseInt(p.customdata, 10);
                if (!isFinite(mi)) return;
                _shapeSetHighlight[idx]?.(mi + 1);
            });
        }

        // ── Pairwise correlation matrix ──────────────────────────
        _redrawOneTrialCorr(idx, trial, xLo, xHi, shiftedX, hiIdx);
}

// Pairwise Pearson correlations between every movement at the
// currently-displayed alignment.  Resamples segments to a 120-pt grid
// over [xLo, xHi] and ignores grid points where either side is NaN.
function _pairwiseCorrMatrix(trial, xLo, xHi, shiftedX) {
    const N = trial.segments.length;
    const GRID = 120;
    const resampled = trial.segments.map((s, mi) =>
        _resampleXY(shiftedX(s, mi), s.ys, xLo, xHi, GRID));
    const mat = [];
    for (let i = 0; i < N; i++) {
        const row = new Array(N);
        for (let j = 0; j < N; j++) {
            if (j < i) { row[j] = mat[j][i]; continue; }
            if (i === j) { row[j] = 1; continue; }
            const a = resampled[i], b = resampled[j];
            let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            for (let k = 0; k < GRID; k++) {
                const u = a[k], v = b[k];
                if (!isFinite(u) || !isFinite(v)) continue;
                sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v; n += 1;
            }
            if (n < 5) { row[j] = null; continue; }
            const mx = sx / n, my = sy / n;
            const num = sxy - n * mx * my;
            const den = Math.sqrt(Math.max(1e-12, (sxx - n * mx * mx) * (syy - n * my * my)));
            row[j] = num / den;
        }
        mat.push(row);
    }
    return mat;
}

// Hierarchical agglomerative clustering with average linkage on the
// supplied distance matrix.  Returns the root node of the binary
// merge tree; each node has {id, members[], left, right, height}.
function _hacAverage(distMat) {
    const N = distMat.length;
    const nodes = [];
    for (let i = 0; i < N; i++) nodes.push({ id: i, members: [i], left: null, right: null, height: 0 });
    // Active distances stored as a Map keyed by "min,max".
    const dist = new Map();
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const d = distMat[i][j];
        dist.set(i + ',' + j, isFinite(d) ? d : 2);
    }
    const active = new Set();
    for (let i = 0; i < N; i++) active.add(i);
    let nextId = N;
    while (active.size > 1) {
        let bestA = -1, bestB = -1, bestD = Infinity;
        const acts = [...active];
        for (let i = 0; i < acts.length; i++) {
            for (let j = i + 1; j < acts.length; j++) {
                const a = Math.min(acts[i], acts[j]);
                const b = Math.max(acts[i], acts[j]);
                const d = dist.get(a + ',' + b);
                if (d == null) continue;
                if (d < bestD) { bestD = d; bestA = a; bestB = b; }
            }
        }
        if (bestA < 0) break;
        const ca = nodes.find(n => n.id === bestA);
        const cb = nodes.find(n => n.id === bestB);
        const merged = { id: nextId, members: [...ca.members, ...cb.members],
                         left: ca, right: cb, height: bestD };
        const na = ca.members.length, nb = cb.members.length;
        for (const k of active) {
            if (k === bestA || k === bestB) continue;
            const dak = dist.get(Math.min(bestA, k) + ',' + Math.max(bestA, k));
            const dbk = dist.get(Math.min(bestB, k) + ',' + Math.max(bestB, k));
            const newD = (na * dak + nb * dbk) / (na + nb);
            dist.set(Math.min(nextId, k) + ',' + Math.max(nextId, k), newD);
        }
        active.delete(bestA); active.delete(bestB); active.add(nextId);
        nodes.push(merged);
        nextId++;
    }
    return nodes[nodes.length - 1];
}

// Cut the merge tree at height `cutH`; return the leaf order
// (contiguous within each cluster) and the size of each cluster.
function _cutAndOrderHAC(root, cutH) {
    const order = [];
    const sizes = [];
    function leaves(node) {
        if (!node) return;
        if (node.left && node.right) { leaves(node.left); leaves(node.right); }
        else order.push(node.id);
    }
    function walk(node) {
        if (!node) return;
        if (node.height > cutH && node.left && node.right) {
            walk(node.left); walk(node.right);
        } else {
            const before = order.length;
            leaves(node);
            sizes.push(order.length - before);
        }
    }
    walk(root);
    return { order, sizes };
}

// Build dendrogram line segments for an HAC tree.  `leafToY` maps
// each leaf node id → its y position (matching the heatmap's
// reordered axis).  Returns an array of {x0, x1, y0, y1} segments
// drawn as horizontal "U" connectors at each merge.
function _buildDendroLines(root, leafToY) {
    const lines = [];
    function walk(node) {
        if (!node) return null;
        if (!node.left || !node.right) {
            return { reprX: 0, reprY: leafToY[node.id] };
        }
        const L = walk(node.left);
        const R = walk(node.right);
        const h = node.height;
        lines.push({ x0: L.reprX, x1: h, y0: L.reprY, y1: L.reprY });
        lines.push({ x0: h, x1: h, y0: L.reprY, y1: R.reprY });
        lines.push({ x0: h, x1: R.reprX, y0: R.reprY, y1: R.reprY });
        return { reprX: h, reprY: (L.reprY + R.reprY) / 2 };
    }
    walk(root);
    return lines;
}

// Clustered heatmap that includes a horizontal dendrogram to the
// left.  Uses linear axes (with manual tick labels) for both the
// heatmap and the dendrogram so they can share a single yaxis.
function _renderClusteredCorrHeatmap(targetDiv, mat, labels, titleText, hiIdx, hiPos, boundaries, dendroLines, maxH, cutH) {
    const N = mat.length;
    const nums = Array.from({ length: N }, (_, i) => i);
    const dx = [], dy = [];
    for (const s of dendroLines) {
        dx.push(s.x0, s.x1, null);
        dy.push(s.y0, s.y1, null);
    }
    const dendroTrace = {
        x: dx, y: dy, type: 'scatter', mode: 'lines',
        line: { color: '#666', width: 1 }, hoverinfo: 'skip', showlegend: false,
        xaxis: 'x', yaxis: 'y',
    };
    const heatTrace = {
        z: mat, x: nums, y: nums, type: 'heatmap',
        xaxis: 'x2', yaxis: 'y',
        zmin: -1, zmax: 1, zmid: 0,
        colorscale: [
            [0.0, 'rgb(33,102,172)'], [0.25, 'rgb(146,197,222)'],
            [0.5, 'rgb(247,247,247)'],
            [0.75, 'rgb(244,165,130)'], [1.0, 'rgb(178,24,43)'],
        ],
        hovertemplate: 'r = %{z:.2f}<extra></extra>',
        // Colorbar length is set later from the actual matrix height.
        colorbar: { thickness: 8, lenmode: 'fraction',
                    y: 0.5, yanchor: 'middle', outlinewidth: 0, ypad: 0,
                    tickvals: [-1, 0, 1], tickfont: { size: 10 } },
    };
    const shapes = [];
    // Highlight: vertical pair for the column, horizontal pair for the
    // row.  Plain line shapes (no rect outline) so there are no short
    // perpendicular caps where the rect's top/bottom edges used to be.
    if (hiIdx > 0 && hiPos >= 0) {
        shapes.push(
            { type: 'line', xref: 'x2', yref: 'y',
              x0: hiPos - 0.5, x1: hiPos - 0.5, y0: -0.5, y1: N - 0.5,
              line: { color: '#000', width: 2 } },
            { type: 'line', xref: 'x2', yref: 'y',
              x0: hiPos + 0.5, x1: hiPos + 0.5, y0: -0.5, y1: N - 0.5,
              line: { color: '#000', width: 2 } },
            { type: 'line', xref: 'x2', yref: 'y',
              x0: -0.5, x1: N - 0.5, y0: hiPos - 0.5, y1: hiPos - 0.5,
              line: { color: '#000', width: 2 } },
            { type: 'line', xref: 'x2', yref: 'y',
              x0: -0.5, x1: N - 0.5, y0: hiPos + 0.5, y1: hiPos + 0.5,
              line: { color: '#000', width: 2 } },
        );
    }
    if (boundaries && boundaries.length) {
        for (const p of boundaries) {
            const pos = p - 0.5;
            shapes.push(
                { type: 'line', xref: 'x2', yref: 'y',
                  x0: pos, x1: pos, y0: -0.5, y1: N - 0.5,
                  line: { color: '#fff', width: 3 } },
                { type: 'line', xref: 'x2', yref: 'y',
                  x0: -0.5, x1: N - 0.5, y0: pos, y1: pos,
                  line: { color: '#fff', width: 3 } },
            );
        }
    }
    // Vertical dashed line at the dendrogram cutoff — only on the
    // dendrogram subplot (xref='x'), constrained to matrix y range.
    if (cutH != null && Number.isFinite(cutH)) {
        shapes.push({
            type: 'line', xref: 'x', yref: 'y',
            x0: cutH, x1: cutH, y0: -0.5, y1: N - 0.5,
            line: { color: '#d32f2f', width: 1, dash: 'dash' },
        });
    }
    const xMaxDendro = Math.max(maxH * 1.05, 0.05);
    // Dendrogram sits in [0, 0.15]; row-number tick labels live in the
    // gap [0.15, 0.20]; matrix in [0.20, 1.0].  The yaxis is anchored
    // to x2 so the labels render right next to the matrix.  Tight
    // bottom margin pulls the column numbers up against the matrix.
    const MARGIN = { t: 18, b: 14, l: 8, r: 50 };
    const layout = {
        margin: MARGIN,
        title: { text: titleText, font: { size: 11 }, x: 0, xanchor: 'left', y: 0.98 },
        xaxis: {
            domain: [0, 0.15],
            range: [xMaxDendro, 0],
            showticklabels: false, zeroline: false, showgrid: false,
            showline: false, ticks: '', fixedrange: true, automargin: false,
        },
        xaxis2: {
            domain: [0.20, 1.0], anchor: 'y', automargin: false,
            tickfont: { size: 9 }, side: 'bottom',
            tickmode: 'array', tickvals: nums, ticktext: labels,
            range: [-0.5, N - 0.5], constrain: 'domain',
            showline: false, showgrid: false, zeroline: false, ticks: '',
        },
        yaxis: {
            tickfont: { size: 9 }, automargin: false, anchor: 'x2',
            tickmode: 'array', tickvals: nums, ticktext: labels,
            range: [N - 0.5, -0.5],   // reversed so movement 1 is at top
            scaleanchor: 'x2', scaleratio: 1,
            showline: false, showgrid: false, zeroline: false, ticks: '',
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        shapes, showlegend: false,
    };
    // Keep height = width so the matrix renders square from the very
    // first paint.  Also set the colorbar's length to match the
    // matrix height (heatmap is square — width set by x2's domain).
    const _sizeToSquare = () => {
        const w = targetDiv.clientWidth || targetDiv.offsetWidth || 0;
        if (w <= 0) return;
        if (targetDiv.style.height !== `${w}px`) {
            targetDiv.style.height = `${w}px`;
        }
        const plotW = Math.max(1, w - MARGIN.l - MARGIN.r);
        const plotH = Math.max(1, w - MARGIN.t - MARGIN.b);
        const x2Frac = 1.0 - 0.20;   // matrix domain [0.20, 1.0]
        const matrixSide = Math.min(x2Frac * plotW, plotH);
        const cbLen = Math.max(0.1, Math.min(1, matrixSide / plotH));
        if (window.Plotly) {
            if (targetDiv._fullLayout) {
                Plotly.restyle(targetDiv, { 'colorbar.len': cbLen }, [1])
                      .catch(() => {});
                Plotly.Plots.resize(targetDiv);
            } else {
                // First paint: bake the len into the next Plotly.react
                // by mutating the trace before plot reads it.
                if (targetDiv.data && targetDiv.data[1] && targetDiv.data[1].colorbar) {
                    targetDiv.data[1].colorbar.len = cbLen;
                }
            }
        }
    };
    // Bake the initial len into the trace passed to Plotly.react so
    // there's no first-paint with len:1 followed by a shrink.
    {
        const w0 = targetDiv.clientWidth || targetDiv.offsetWidth || 380;
        const plotW0 = Math.max(1, w0 - MARGIN.l - MARGIN.r);
        const plotH0 = Math.max(1, w0 - MARGIN.t - MARGIN.b);
        const matrixSide0 = Math.min((1 - 0.20) * plotW0, plotH0);
        heatTrace.colorbar.len = Math.max(0.1, Math.min(1, matrixSide0 / plotH0));
    }
    _sizeToSquare();
    if (!targetDiv._squareObserver && window.ResizeObserver) {
        targetDiv._squareObserver = new ResizeObserver(_sizeToSquare);
        targetDiv._squareObserver.observe(targetDiv);
    }
    // Stash the row labels so the click handler resolves a clicked
    // position (e.g. row 0 in a sorted matrix) to the original
    // movement number rather than just the row index.
    targetDiv._yLabels = labels;
    Plotly.react(targetDiv, [dendroTrace, heatTrace], layout,
                 { responsive: true, displayModeBar: false });
    // Click on any heatmap cell → set the highlighted movement to that row.
    if (!targetDiv._clickBound) {
        targetDiv._clickBound = true;
        targetDiv.on('plotly_click', (ev) => {
            const p = ev?.points?.[0];
            if (!p || p.data?.type !== 'heatmap') return;
            const yi = (typeof p.y === 'number') ? p.y : parseInt(p.y, 10);
            if (!isFinite(yi)) return;
            const lbl = targetDiv._yLabels?.[yi];
            const mov = parseInt(lbl, 10);
            if (!isFinite(mov)) return;
            const m = /shapeCorrPlot_(\d+)/.exec(targetDiv.id || '');
            if (!m) return;
            _shapeSetHighlight[parseInt(m[1], 10)]?.(mov);
        });
    }
}

// Render a single correlation-matrix heatmap into `targetDiv`.
// Optional `boundaries` is an array of category indices (in the
// reordered axis) where vertical/horizontal separator lines are drawn
// to mark cluster boundaries.
function _renderCorrHeatmap(targetDiv, mat, labels, titleText, hiIdx, hiPos, boundaries) {
    const N = mat.length;
    const shapes = [];
    if (hiIdx > 0 && hiPos >= 0) {
        shapes.push(
            { type: 'rect', xref: 'x', yref: 'paper',
              x0: hiPos - 0.5, x1: hiPos + 0.5, y0: 0, y1: 1,
              line: { color: '#000', width: 2 }, fillcolor: 'rgba(0,0,0,0)' },
            { type: 'rect', xref: 'paper', yref: 'y',
              x0: 0, x1: 1, y0: hiPos - 0.5, y1: hiPos + 0.5,
              line: { color: '#000', width: 2 }, fillcolor: 'rgba(0,0,0,0)' },
        );
    }
    if (boundaries && boundaries.length) {
        for (const p of boundaries) {
            const pos = p - 0.5;
            shapes.push(
                { type: 'line', xref: 'x', yref: 'paper',
                  x0: pos, x1: pos, y0: 0, y1: 1,
                  line: { color: '#fff', width: 3 } },
                { type: 'line', xref: 'paper', yref: 'y',
                  x0: 0, x1: 1, y0: pos, y1: pos,
                  line: { color: '#fff', width: 3 } },
            );
        }
    }
    const trace = {
        z: mat, x: labels, y: labels, type: 'heatmap',
        zmin: -1, zmax: 1, zmid: 0,
        colorscale: [
            [0.0, 'rgb(33,102,172)'], [0.25, 'rgb(146,197,222)'],
            [0.5, 'rgb(247,247,247)'],
            [0.75, 'rgb(244,165,130)'], [1.0, 'rgb(178,24,43)'],
        ],
        hovertemplate: 'mov %{x} × mov %{y}<br>r = %{z:.2f}<extra></extra>',
        colorbar: { thickness: 8, tickvals: [-1, 0, 1], tickfont: { size: 10 } },
    };
    const layout = {
        margin: { t: 18, b: 36, l: 36, r: 50 },
        title: { text: titleText, font: { size: 11 }, x: 0, xanchor: 'left', y: 0.98 },
        xaxis: { title: { text: 'Movement #', font: { size: 10 } },
                 tickfont: { size: 9 }, side: 'bottom', automargin: true,
                 type: 'category', constrain: 'domain' },
        yaxis: { title: { text: 'Movement #', font: { size: 10 } },
                 tickfont: { size: 9 }, autorange: 'reversed', automargin: true,
                 type: 'category', scaleanchor: 'x', scaleratio: 1 },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        shapes,
    };
    requestAnimationFrame(() => {
        const w = targetDiv.clientWidth || targetDiv.offsetWidth || 0;
        if (w > 0) {
            targetDiv.style.height = `${w}px`;
            if (window.Plotly && targetDiv._fullLayout) Plotly.Plots.resize(targetDiv);
        }
    });
    Plotly.react(targetDiv, [trace], layout,
                 { responsive: true, displayModeBar: false });
}

function _redrawOneTrialCorr(idx, trial, xLo, xHi, shiftedX, hiIdx) {
    const corrDiv = document.getElementById(`shapeCorrPlot_${idx}`);
    const N = trial.segments.length;
    if (!corrDiv) return;
    if (N < 2) { corrDiv.innerHTML = ''; return; }

    const mat = _pairwiseCorrMatrix(trial, xLo, xHi, shiftedX);
    const labels = trial.segments.map((_, i) => String(i + 1));
    const clusterOn = !!document.getElementById('shapeClusterOn')?.checked;

    if (!clusterOn) {
        // Unsorted matrix, but rendered with the same two-subplot
        // layout (empty dendrogram region) so the heatmap size matches
        // the clustered view.
        _renderClusteredCorrHeatmap(
            corrDiv, mat, labels, 'Pairwise correlation',
            hiIdx, hiIdx > 0 ? hiIdx - 1 : -1,
            null, [], 1, null,
        );
        return;
    }

    // Clustered: HAC with average linkage on 1 − r, cut at slider height.
    const dist = mat.map(row => row.map(v => (v == null ? 2 : 1 - v)));
    const root = _hacAverage(dist);
    if (!root) { corrDiv.innerHTML = ''; return; }
    const cutH = parseFloat(document.getElementById('shapeClusterCutoff')?.value);
    const useCut = isFinite(cutH) ? cutH : 0.5;
    const { order, sizes } = _cutAndOrderHAC(root, useCut);
    const reord = order.map(i => order.map(j => mat[i][j]));
    const reordLabels = order.map(i => String(i + 1));
    const boundaries = [];
    let acc = 0;
    for (let i = 0; i < sizes.length - 1; i++) { acc += sizes[i]; boundaries.push(acc); }
    const hiPosClu = hiIdx > 0 ? order.indexOf(hiIdx - 1) : -1;
    const k = sizes.length;
    const leafToY = {};
    order.forEach((id, pos) => { leafToY[id] = pos; });
    const dendroLines = _buildDendroLines(root, leafToY);
    const maxH = root.height || 1;
    _renderClusteredCorrHeatmap(
        corrDiv, reord, reordLabels,
        `Clustered (HAC, avg linkage, 1−r) — ${k} group${k === 1 ? '' : 's'}`,
        hiIdx, hiPosClu, boundaries, dendroLines, maxH, useCut,
    );
}

// Live-update the shape plots when the controls change.
document.addEventListener('DOMContentLoaded', () => {
    const xSl = document.getElementById('shapeXScaleSlider');
    const xVal = document.getElementById('shapeXScaleVal');
    const meanCb = document.getElementById('shapeShowMean');
    if (xSl) xSl.addEventListener('input', () => {
        if (xVal) xVal.textContent = `${(+xSl.value).toFixed(2)} s`;
        _drawShapeOverlayPlots();
    });
    if (meanCb) meanCb.addEventListener('change', _drawShapeOverlayPlots);
    const peaksCb = document.getElementById('shapeShowPeaks');
    if (peaksCb) peaksCb.addEventListener('change', _drawShapeOverlayPlots);
    document.querySelectorAll('input[name="shapeAlign"]').forEach(r =>
        r.addEventListener('change', _drawShapeOverlayPlots));
    const cutSl = document.getElementById('shapeClusterCutoff');
    const cutVal = document.getElementById('shapeClusterCutoffVal');
    if (cutSl) cutSl.addEventListener('input', () => {
        if (cutVal) cutVal.textContent = (+cutSl.value).toFixed(2);
        _drawShapeOverlayPlots();
    });
    const cluCb = document.getElementById('shapeClusterOn');
    if (cluCb) cluCb.addEventListener('change', _drawShapeOverlayPlots);
});

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
    // (The standalone "Source: …" badge was removed as redundant — the
    // dropdown already shows the resolved source.)
    _updateAutoSourceLabel(data.data_source);

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
    // Reduced top margin (was 30) — the trial title used to sit in
    // Plotly's title space; it's now rendered in an HTML header above
    // the scrolling wrapper instead.
    const DIST_H = 220, DIST_MT = 10, DIST_MB = 5;
    const VEL_H = 150, VEL_MT = 5, VEL_MB = 35;

    // Short trial label (e.g. "R1" from "PD03_R1").
    const _trialPartOf = t => String(t.name || '').split('_').pop();

    data.trials.forEach((trial, idx) => {
        // Trial block: fixed slider column on the left + a horizontally-
        // scrolling area on the right.  Both the trial-name header and
        // the copy button live in a NON-scrolling row above the wrapper
        // so they stay anchored when the user scrolls long plots.
        const block = document.createElement('div');
        block.style.display = 'grid';
        block.style.gridTemplateColumns = 'auto 1fr';
        block.style.gridTemplateAreas = '"_ header" "slider wrapper"';
        block.style.alignItems = 'flex-start';
        block.style.marginBottom = '16px';

        // Locked header: trial name (left) + Copy button (right).
        const header = document.createElement('div');
        header.style.gridArea = 'header';
        header.style.display = 'flex';
        header.style.alignItems = 'center';
        header.style.justifyContent = 'space-between';
        header.style.padding = '2px 8px 4px';
        const titleSpan = document.createElement('span');
        titleSpan.textContent = `Trial: ${_trialPartOf(trial)}`;
        titleSpan.style.cssText = 'font-size:13px;color:#666;font-weight:600;';
        const copyBtn = _makeCopyBtn(b => _copyTrialPlots(idx, trial.name || '', b));
        header.appendChild(titleSpan);
        header.appendChild(copyBtn);
        block.appendChild(header);

        // Slider column (does NOT scroll with the plots)
        const sliderCol = document.createElement('div');
        sliderCol.style.gridArea = 'slider';
        sliderCol.style.display = 'flex';
        sliderCol.style.flexDirection = 'column';
        sliderCol.appendChild(_buildYSliderCol('dist', idx, DIST_H, DIST_MT, DIST_MB));
        sliderCol.appendChild(_buildYSliderCol('vel', idx, VEL_H, VEL_MT, VEL_MB));
        block.appendChild(sliderCol);

        // Scrolling wrapper holds both plots
        const wrapper = document.createElement('div');
        wrapper.style.gridArea = 'wrapper';
        wrapper.style.overflowX = 'auto';
        // CSS grid would otherwise let the wrapper grow to fit its
        // content; min-width:0 lets the 1fr track stay 1fr.
        wrapper.style.minWidth = '0';

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

        // Compute plot width based on the X-scale slider (seconds shown
        // per container width; default 20).  When the trial is shorter
        // than the chosen scale it occupies only a fraction of the
        // window rather than being stretched to the full width.
        // Slider is reversed (right = stretched): raw 5..60 maps to secPerWidth = 65 - raw (60..5).
        const _xRaw = parseFloat(document.getElementById('xScaleSlider')?.value);
        const secPerWidth = isFinite(_xRaw) ? (65 - _xRaw) : 15;
        const fps = trial.fps || 60;
        const durationSec = trial.distances.length / fps;
        const plotWidth = Math.max(120, (durationSec / secPerWidth) * containerWidth);

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

function _loadImg(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

/** Inline SVG for the copy-to-clipboard icon (two rounded rects, offset). */
const COPY_ICON_HTML =
    `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ` +
    `stroke="currentColor" stroke-width="2" stroke-linecap="round" ` +
    `stroke-linejoin="round" style="display:block;">` +
    `<rect x="9" y="9" width="13" height="13" rx="2"/>` +
    `<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>` +
    `</svg>`;

/** Build a "copy to clipboard" icon-button.  onClick receives the
 *  button so the click handler can show feedback on it. */
function _makeCopyBtn(onClick) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-sm';
    btn.innerHTML = COPY_ICON_HTML;
    btn.title = 'Copy to clipboard';
    btn.style.padding = '4px 6px';
    btn.style.lineHeight = '0';
    btn.addEventListener('click', () => onClick(btn));
    return btn;
}

/** Copy a vertical stack of Plotly plot divs to the clipboard as one
 *  composed PNG.  Each plot's full content (with axes, grid, tick
 *  labels, axis titles) is captured at 2× scale via Plotly.toImage.
 *  The clipboard entry is a File so its `name` carries the suggested
 *  filename through to Finder paste on macOS. */
async function _copyPlotsAsPng(plotDivs, filename, btn, opts) {
    plotDivs = plotDivs.filter(Boolean);
    if (!plotDivs.length || !window.Plotly) return;
    if (btn && btn.disabled) return;
    if (btn) {
        btn.disabled = true;
        btn.style.opacity = '0.4';
        btn.style.filter = 'brightness(0.7)';
    }
    const title = (opts && opts.title) || '';
    const minHold = new Promise(r => setTimeout(r, 1000));
    try {
        const SCALE = 2;
        const urls = await Promise.all(plotDivs.map(d =>
            Plotly.toImage(d, {
                format: 'png',
                width:  d.clientWidth  || 800,
                height: d.clientHeight || 220,
                scale:  SCALE,
            })));
        const imgs = await Promise.all(urls.map(_loadImg));
        const W = Math.max(...imgs.map(i => i.width));
        // Optional header band with the column title.
        const TITLE_H = title ? 32 * SCALE : 0;
        const H = TITLE_H + imgs.reduce((s, i) => s + i.height, 0);
        const cv = document.createElement('canvas');
        cv.width = W; cv.height = H;
        const ctx = cv.getContext('2d');
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, W, H);
        if (title) {
            ctx.fillStyle = '#222';
            ctx.font = `700 ${16 * SCALE}px system-ui, -apple-system, "Segoe UI", sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(title, W / 2, TITLE_H / 2);
        }
        let y = TITLE_H;
        for (const img of imgs) {
            ctx.drawImage(img, Math.round((W - img.width) / 2), y);
            y += img.height;
        }
        const blob = await new Promise(r => cv.toBlob(r, 'image/png'));
        const stem = (filename || 'plot').replace(/[^A-Za-z0-9_-]+/g, '_');
        const file = new File([blob], `${stem}.png`, { type: 'image/png' });
        await navigator.clipboard.write([
            new ClipboardItem({ 'image/png': file }),
        ]);
    } catch (err) {
        console.error('Copy failed:', err);
    } finally {
        await minHold;
        if (btn) {
            btn.disabled = false;
            btn.style.opacity = '';
            btn.style.filter = '';
        }
    }
}

/** Copy one whole column on the Group Comparison page (title + Mean
 *  + Variance + Sequence-Effect plots).  Called from inline onclick
 *  handlers attached to the per-column copy buttons. */
window._copyGroupColumn = function(paramId, kind, btn) {
    const prefix = kind === 'dose' ? 'grpPlotDose_' : 'grpPlot_';
    const divs = ['mean', 'cv', 'seq']
        .map(f => document.getElementById(prefix + paramId + '_' + f))
        .filter(Boolean);
    const m = (typeof GROUP_METRICS !== 'undefined')
        ? GROUP_METRICS.find(x => x.id === paramId) : null;
    const titleText = m
        ? m.name + (m.mean && m.mean.unit ? ' (' + m.mean.unit + ')' : '')
        : paramId;
    const stem = (kind === 'dose' ? 'levodopa_' : 'group_') + paramId;
    return _copyPlotsAsPng(divs, stem, btn, { title: titleText });
};

/** Copy the distance + velocity plots for one trial.  Filename
 *  pattern: {subject}_{trial}_trace.png (trial.name already includes
 *  the subject prefix, e.g. "PD03_R1"). */
async function _copyTrialPlots(idx, trialFullName, btn) {
    const dist = document.getElementById(`distPlot_${idx}`);
    const vel  = document.getElementById(`velPlot_${idx}`);
    if (!dist || !vel) return;
    const stem = trialFullName || `trial${idx}`;
    return _copyPlotsAsPng([dist, vel], `${stem}_trace`, btn);
}

/** Currently loaded subject's name, for movement-plot filenames. */
function _currentSubjectName() {
    if (Array.isArray(subjects) && currentSubjectId != null) {
        const s = subjects.find(s => s.id === currentSubjectId);
        if (s && s.name) return s.name;
    }
    return 'subject';
}

/** Copy a single movement-parameter plot (all trials laid out
 *  side-by-side).  Filename pattern: {subject}_{param}.png. */
async function _copyMovementPlot(param, btn) {
    const div = document.getElementById(`distMovPlot_${param}`);
    if (!div) return;
    return _copyPlotsAsPng([div], `${_currentSubjectName()}_${param}`, btn);
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

    // Trial name is rendered in an HTML header above the wrapper now
    // (so it stays anchored when the user scrolls long plots).  No
    // Plotly title here — top margin shrunk to reclaim the space.
    const layout = {
        margin: { t: 10, b: 5, l: 55, r: 20 },
        // Pin the X range to the trace extent so toggling the peak
        // overlays (which adds marker traces) can't rescale the axis.
        xaxis: {
            showticklabels: false, color: '#666', gridcolor: '#eee',
            range: [times[0] || 0, times[n - 1] || 0], autorange: false,
        },
        yaxis: {
            title: { text: 'Distance (mm)', font: { size: 11, color: '#2196F3' } },
            color: '#2196F3',
            gridcolor: '#f0f0f0',
            zeroline: false,
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: false,
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
        // Pin the X range to the trace extent so toggling the peak
        // overlays can't rescale the axis.
        xaxis: {
            title: { text: 'Time (s)', font: { size: 11 } }, color: '#666', gridcolor: '#eee',
            range: [times[0] || 0, times[n - 1] || 0], autorange: false,
        },
        yaxis: {
            title: { text: 'Velocity (mm/s)', font: { size: 11, color: '#4CAF50' } },
            color: '#4CAF50',
            gridcolor: '#f0f0f0',
            zeroline: true,
            zerolinecolor: '#ddd',
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: false,
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

// Per-trial frame offset + fps, derived from the loaded distance
// traces, so movement-plot times match the distance/velocity plots.
function _trialFrameMeta() {
    const meta = {};
    const trials = (cachedTraces && cachedTraces.trials) || [];
    let acc = 0;
    trials.forEach((t, i) => {
        meta[i] = { start: acc, fps: t.fps || 60 };
        acc += (t.distances ? t.distances.length : 0);
    });
    return meta;
}

function renderMovementScatter(divId, data, param, seqMode, widthPx) {
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
    // not the distance peak.  Use the FRAME fields and convert to
    // trial-local time (frame relative to the trial's first frame /
    // fps) — the same time values the distance/velocity plots use, so
    // the markers line up.  No per-trial first-movement offset.
    const frameField = param === 'peak_open_vel' ? 'peak_open_vel_frame'
                     : param === 'peak_close_vel' ? 'peak_close_vel_frame'
                     : 'peak_frame';
    const frameMeta = _trialFrameMeta();

    // Same color for every trial — trial identity is shown by the
    // labels above each trial's segment instead of a color key.
    const color = MOVEMENT_DOT_COLOR;

    // Each trial becomes its own subplot (own X axis restarting at 0),
    // all sharing the single Y axis.  Domains are sized in proportion
    // to each trial's time span, separated by a small gap.
    const trialKeys = Object.keys(byTrial).sort((a, b) => +a - +b);
    const trialInfo = trialKeys.map(ti => {
        const ms = byTrial[ti];
        const meta = frameMeta[ti] || { start: 0, fps: 60 };
        const rawX = ms.map(m => {
            const f = (m[frameField] != null ? m[frameField] : m.peak_frame);
            return (f != null && isFinite(f)) ? (f - meta.start) / meta.fps : null;
        });
        const valid = rawX.filter(v => v != null && isFinite(v));
        // Local time is measured from the trial's first frame (0), so
        // the span runs up to the last movement's time.
        const span = valid.length ? Math.max(...valid) : 0;
        return { ti, ms, rawX, span };
    });
    const N = trialInfo.length;
    const maxSpan = Math.max(1, ...trialInfo.map(t => t.span));
    const gapFrac = N > 1 ? 0.04 : 0;
    // Single-movement trials get a minimum width so they're visible.
    const weights = trialInfo.map(t => Math.max(t.span, maxSpan * 0.15));
    const wsum = weights.reduce((a, b) => a + b, 0) || 1;
    const avail = 1 - gapFrac * (N - 1);
    let _dcur = 0;
    trialInfo.forEach((t, i) => {
        const w = avail * (weights[i] / wsum);
        t.domain = [_dcur, Math.min(1, _dcur + w)];
        _dcur += w + gapFrac;
        t.axId = i === 0 ? 'x' : 'x' + (i + 1);
        t.axKey = i === 0 ? 'xaxis' : 'xaxis' + (i + 1);
    });

    trialInfo.forEach(({ ti, ms, rawX, axId, domain }) => {
        // Trial-local time (frame-from-trial-start / fps), same values
        // as the distance/velocity plots.
        const x = rawX;
        const y = ms.map(m => m[param]);
        const trialLabel = trialNames[ti] || `Trial ${+ti + 1}`;
        // Short label: just the trial suffix (e.g. "R1" from "PD03_R1").
        const trialShort = String(trialLabel).split('_').pop();

        traces.push({
            x, y,
            type: 'scatter',
            mode: 'markers',
            name: trialLabel,
            xaxis: axId, yaxis: 'y',
            marker: { color, size: 7, opacity: 0.8 },
            hovertemplate: `${trialShort}<br>t=%{x:.2f}s<br>${PARAM_LABELS[param]}: %{y:.2f}<extra></extra>`,
            showlegend: false,
        });

        // Trial label above this subplot, left-aligned with its start.
        // The total segmentation R² for this trial gets appended once
        // the fits below are computed (patched via labelAnnIdx).
        const labelAnnIdx = annotations.length;
        annotations.push({
            x: domain[0], y: 1.0,
            xref: 'paper', yref: 'paper',
            xanchor: 'left', yanchor: 'bottom',
            text: trialShort,
            showarrow: false,
            font: { size: 11, color: '#444' },
        });

        // Accumulate explained / total variance for this trial's total R².
        let trialSSReg = 0;
        const _ssTot = (arr) => {
            const v = arr.filter(a => a != null && isFinite(a));
            if (v.length < 2) return 0;
            const m = v.reduce((a, b) => a + b, 0) / v.length;
            return v.reduce((a, b) => a + (b - m) ** 2, 0);
        };
        const trialSSTot = _ssTot(y);
        const _finishTrialR2 = () => {
            if (seqMode !== 'none' && trialSSTot > 0 && trialSSReg > 0) {
                const r2 = Math.min(1, trialSSReg / trialSSTot);
                annotations[labelAnnIdx].text = `${trialShort}  (R²=${r2.toFixed(2)})`;
            }
        };

        // Vertical line at this trial's t=0 (its subplot's left edge),
        // matching the y-axis line at the first trial's start.
        shapes.push({
            type: 'line', xref: axId, yref: 'paper',
            x0: 0, x1: 0, y0: 0, y1: 1,
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
                    type: 'rect', xref: axId, yref: 'paper',
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
                    trialSSReg += fit.r2 * _ssTot(seqY);
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
                        xaxis: axId, yaxis: 'y',
                        line: { color, width: 2.5 },
                        showlegend: false, hoverinfo: 'skip',
                    });
                    // R² annotation
                    const midX = seqX[Math.floor(seqX.length / 2)];
                    annotations.push({
                        x: midX, y: 0.98,
                        xref: axId, yref: 'paper', yanchor: 'top',
                        text: `R\u00b2=${fit.r2.toFixed(2)}<br>Slope=${fit.b.toFixed(3)}`,
                        showarrow: false,
                        font: { size: 9, color },
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: color,
                        borderpad: 2,
                    });
                } else {
                    // Linear fit within sequence
                    const reg = linearRegressionFull(seqX, seqY);
                    if (!reg) return;
                    baselineMag = reg.slope * xMin + reg.intercept;
                    trialSSReg += reg.ss_reg;
                    traces.push({
                        x: [xMin, xMax],
                        y: [sign * (reg.slope * xMin + reg.intercept), sign * (reg.slope * xMax + reg.intercept)],
                        type: 'scatter', mode: 'lines',
                        name: `Seq ${si + 1}`,
                        xaxis: axId, yaxis: 'y',
                        line: { color, width: 2.5 },
                        showlegend: false, hoverinfo: 'skip',
                    });
                    // R² annotation
                    const midX = (xMin + xMax) / 2;
                    annotations.push({
                        x: midX, y: 0.98,
                        xref: axId, yref: 'paper', yanchor: 'top',
                        text: `R\u00b2=${reg.r2.toFixed(2)}<br>Slope=${reg.slope.toFixed(3)}`,
                        showarrow: false,
                        font: { size: 9, color },
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
                            xaxis: axId, yaxis: 'y',
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
                trialSSReg += fit.r2 * _ssTot(fitY);
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
                    xaxis: axId, yaxis: 'y',
                    line: { color, width: 2, dash: 'dash' },
                    showlegend: false, hoverinfo: 'skip',
                });
                // R² annotation
                annotations.push({
                    x: (xMin + xMax) / 2, y: 0.98,
                    xref: axId, yref: 'paper', yanchor: 'top',
                    text: `R\u00b2=${fit.r2.toFixed(2)}<br>Slope=${fit.b.toFixed(3)}`,
                    showarrow: false,
                    font: { size: 9, color },
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
                trialSSReg += reg.ss_reg;
                const { slope, intercept } = reg;
                traces.push({
                    x: [xMin, xMax],
                    y: [sign * (slope * xMin + intercept), sign * (slope * xMax + intercept)],
                    type: 'scatter', mode: 'lines',
                    name: `Trend (Trial ${+ti + 1})`,
                    xaxis: axId, yaxis: 'y',
                    line: { color, width: 2, dash: 'dash' },
                    showlegend: false, hoverinfo: 'skip',
                });
                // R² label at the top of the plot.
                const midX = (xMin + xMax) / 2;
                annotations.push({
                    x: midX, y: 0.98,
                    xref: axId, yref: 'paper', yanchor: 'top',
                    text: `R²=${reg.r2.toFixed(2)}<br>Slope=${reg.slope.toFixed(3)}`,
                    showarrow: false,
                    font: { size: 9, color },
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: color,
                    borderpad: 2,
                });
            }
        }

        _finishTrialR2();   // append the trial's total R² to its label
    });

    // Closing-velocity params are negative; reverse the Y axis so the
    // curve points up like the opening-velocity plots for easy visual
    // comparison.
    const reverseY = (param === 'peak_close_vel' || param === 'mean_close_vel');

    const layout = {
        // Title rendered as an external HTML header (left-aligned and
        // anchored during horizontal scroll), so none here.
        margin: { t: 24, b: 40, l: 60, r: 20 },
        // Single shared Y axis (anchored to the first subplot's X axis).
        yaxis: {
            title: { text: PARAM_YLABELS[param] || '', font: { size: 11 } },
            color: '#666', gridcolor: '#f0f0f0',
            autorange: reverseY ? 'reversed' : true,
            anchor: 'x',
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: false,
        shapes,
        annotations,
    };

    // One X axis per trial, each restarting at 0, sharing the Y axis.
    trialInfo.forEach((t, i) => {
        layout[t.axKey] = {
            domain: t.domain,
            anchor: 'y',
            color: '#666',
            gridcolor: '#eee',
            rangemode: 'tozero',
            zeroline: false,
            title: (i === 0) ? { text: 'Time (s)', font: { size: 11 } } : undefined,
        };
    });

    // When an explicit width is given (X-scale > 1) the plot is fixed
    // and the wrapper scrolls; otherwise it stays responsive to its
    // grid cell.
    const config = { displayModeBar: false };
    if (widthPx) { layout.width = widthPx; config.responsive = false; }
    else { config.responsive = true; }

    Plotly.newPlot(divId, traces, layout, config);
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
        const src = document.getElementById('groupSourceSelect')?.value || 'auto';
        const seqMode = document.getElementById('groupSeqModeSelect')?.value || 'linear_full';
        const hand = document.getElementById('groupHandSelect')?.value || 'more';
        const trial = document.getElementById('groupTrialSelect')?.value || 'last';
        cachedGroup = await API.get(
            `/api/results/group?include_auto=true&source=${src}&seq_mode=${seqMode}` +
            `&hand=${hand}&trial=${trial}`);
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
        _groupSubjectChecked[s.name] = s.has_complete_events ? true : includeAuto;
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
            if (!s.has_complete_events) _groupSubjectChecked[s.name] = includeAuto;
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
        const subs = byGroup[g] || [];
        if (subs.length === 0) return;
        const color = (typeof GROUP_COLORS !== 'undefined' && GROUP_COLORS[g]) || '#999';
        // Each diagnosis group on its own row, led by a group label.
        html += `<div class="gsl-group"><span class="gsl-group-label">${g}</span><div class="gsl-tags">`;
        subs.forEach(s => {
            const checked = !!_groupSubjectChecked[s.name];
            // Dim subjects without a complete saved event set (must
            // have ≥1 open, peak, and close).
            const complete = !!s.has_complete_events;
            const dim = complete ? '' : ' dim';
            const title = complete ? 'Saved events (open/peak/close)'
                : (s.has_saved_events ? 'Incomplete saved events' : 'No saved events');
            const safe = s.name.replace(/"/g, '&quot;');
            // Laterality (more-affected side) tag, when known.
            const tag = s.laterality_side
                ? ` <span style="font-weight:700;font-size:9px;opacity:0.75;">${s.laterality_side}</span>`
                : '';
            // Tab background = group color, black text for readability.
            html += `<label class="${dim.trim()}" style="background:${color};color:#000;border-color:${color};"
                title="${title} (${g})${s.laterality_side ? ' · more-affected ' + s.laterality_side : ''}">
                <input type="checkbox" data-subject="${safe}" ${checked ? 'checked' : ''}>
                ${s.name}${complete ? '' : ' *'}${tag}
            </label>`;
        });
        html += '</div></div>';
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
            _groupSubjectChecked[s.name] = s.has_complete_events ? true : includeAuto;
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
    // The prev/next subject buttons don't apply to the group view —
    // hide them here too in case nav.js created them after switchTab.
    _setSubjectNavVisible(false);

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

    const colW = Math.max(200, Math.min(280, container.clientWidth / visibleMetrics.length));

    // Row labels now live on each plot's Y axis (see renderGroupBar's
    // yLabel arg) — no left label column needed.
    const _columnTitle = (m, kind) => {
        const titleText = m.name + (m.mean && m.mean.unit ? ' (' + m.mean.unit + ')' : '');
        return `<div style="display:flex;align-items:center;justify-content:center;gap:6px;padding:4px 0 0;">
            <span style="font-size:13px;font-weight:700;">${titleText}</span>
            <button class="btn btn-sm" title="Copy to clipboard"
                style="padding:2px 5px;line-height:0;"
                onclick="_copyGroupColumn('${m.id}', '${kind}', this)">${COPY_ICON_HTML}</button>
        </div>`;
    };

    html += '<div style="overflow-x:auto;">';
    html += `<div style="display:grid;grid-template-columns:repeat(${visibleMetrics.length}, ${colW}px);gap:0;">`;

    ROW_DEFS.forEach((row, ri) => {
        visibleMetrics.forEach((m) => {
            const spec = m[row.field];
            const divId = `grpPlot_${m.id}_${row.field}`;
            if (spec) {
                html += `<div style="height:${row.height}px;">
                    ${ri === 0 ? _columnTitle(m, 'bar') : ''}
                    <div id="${divId}" style="height:${row.height - (ri === 0 ? 28 : 0)}px;"></div>
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
        Levodopa motor response</div>`;
    html += '<div style="overflow-x:auto;">';
    html += `<div style="display:grid;grid-template-columns:repeat(${visibleMetrics.length}, ${colW}px);gap:0;">`;
    ROW_DEFS.forEach((row, ri) => {
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            const divId = `grpPlotDose_${m.id}_${row.field}`;
            if (spec) {
                html += `<div style="height:${row.height}px;">
                    ${ri === 0 ? _columnTitle(m, 'dose') : ''}
                    <div id="${divId}" style="height:${row.height - (ri === 0 ? 28 : 0)}px;"></div>
                </div>`;
            } else {
                html += `<div style="height:${row.height}px;"></div>`;
            }
        });
    });
    html += '</div></div></div>';

    container.innerHTML = html;

    // Closing-velocity mean values are negative; reverse that row's Y
    // axis so the bars/points point up like the opening-velocity plots.
    const _reverseY = (m, row) =>
        row.field === 'mean' && (m.id === 'peak_close_vel' || m.id === 'mean_close_vel');

    // Sequence-Effect row uses R² (seq_*) or slope (seqslope_*) per the
    // radio next to the Sequence-effect dropdown.
    const seqMetric = document.querySelector('#groupSeqMetric input:checked')?.value || 'r2';
    const _key = (spec, row) =>
        (row.field === 'seq' && seqMetric === 'slope')
            ? spec.key.replace(/^seq_/, 'seqslope_')
            : spec.key;

    // Per-row Y-axis label.  Sequence Effect row picks up "(R²)" or
    // "(slope)" depending on the radio next to the seq-effect dropdown.
    const _yLabelFor = (row) => {
        if (row.field === 'seq') {
            return 'Sequence Effect (' + (seqMetric === 'slope' ? 'slope' : 'R²') + ')';
        }
        return row.label;       // 'Mean' or 'Variance'
    };

    // Render each chart.  Column titles live in the HTML header above
    // each column; per-plot Y-axis labels tell which row (Mean / Var /
    // Sequence Effect) the chart belongs to.
    ROW_DEFS.forEach(row => {
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            if (!spec) return;
            const divId = `grpPlot_${m.id}_${row.field}`;
            renderGroupBar(divId, data, _key(spec, row), _reverseY(m, row), _yLabelFor(row));
        });
    });

    // Dose-response scatters — one per (metric × row), aligned with the
    // bar grid.
    ROW_DEFS.forEach((row) => {
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            if (!spec) return;
            const divId = `grpPlotDose_${m.id}_${row.field}`;
            renderDoseScatter(divId, data, _key(spec, row), _reverseY(m, row), _yLabelFor(row));
        });
    });
}

function renderDoseScatter(divId, data, paramKey, reverseY, yLabel) {
    // Levodopa plots include PD subjects only.
    const color = GROUP_COLORS['PD'] || '#2196F3';
    const pd = _activeGroupSubjects().filter(s =>
        (s.diagnosis || 'Control') === 'PD'
        && s[paramKey] != null && isFinite(s[paramKey]));

    // On-levodopa: plotted at hours since last dose.
    const onPD = pd.filter(s => s.time_since_dose_min != null && isFinite(s.time_since_dose_min));
    // Off-levodopa: clustered to the right, after a gap.
    const offPD = pd.filter(s =>
        (s.time_since_dose_min == null || !isFinite(s.time_since_dose_min)) && s.levodopa_off);

    const maxH = onPD.length ? Math.max(...onPD.map(s => s.time_since_dose_min / 60)) : 0;
    const GAP = 2;            // hours of empty space before the Off cluster
    const OFF_STEP = 0.6;     // spacing between off subjects
    const offStart = (onPD.length ? maxH : 0) + GAP;

    const _sz = s => (highlightedSubject === s.name ? 10 : 7);
    const traces = [];
    if (onPD.length) {
        traces.push({
            x: onPD.map(s => s.time_since_dose_min / 60.0),
            y: onPD.map(s => s[paramKey]),
            text: onPD.map(s => `${s.name}<br>${s.last_dose_raw || ''}`),
            type: 'scatter', mode: 'markers',
            marker: { color, size: onPD.map(_sz), opacity: 0.8, line: { color: '#333', width: 0.5 } },
            hovertemplate: '%{text}<br>%{x:.2f} h<br>%{y:.3f}<extra></extra>',
            showlegend: false,
        });
    }
    let offCenter = null;
    if (offPD.length) {
        const offX = offPD.map((s, i) => offStart + i * OFF_STEP);
        offCenter = offStart + ((offPD.length - 1) / 2) * OFF_STEP;
        traces.push({
            x: offX,
            y: offPD.map(s => s[paramKey]),
            text: offPD.map(s => `${s.name}<br>Off levodopa`),
            type: 'scatter', mode: 'markers',
            marker: { color, size: offPD.map(_sz), opacity: 0.8, line: { color: '#333', width: 0.5 } },
            hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>',
            showlegend: false,
        });
    }

    // X ticks: numeric hours (0,2,4,…) plus an "Off" tick at the cluster.
    const tickvals = [], ticktext = [];
    const hiHour = Math.ceil(maxH);
    for (let h = 0; h <= hiHour; h += 2) { tickvals.push(h); ticktext.push(String(h)); }
    const shapes = [];
    if (offCenter != null) {
        tickvals.push(offCenter); ticktext.push('Not taking');
        // Dashed separator in the gap between on- and off-levodopa.
        const sepX = ((onPD.length ? maxH : 0) + offStart) / 2;
        shapes.push({
            type: 'line', xref: 'x', yref: 'paper',
            x0: sepX, x1: sepX, y0: 0, y1: 1,
            line: { color: '#bbb', width: 1, dash: 'dot' },
        });
    }
    const xMax = (offPD.length ? offStart + offPD.length * OFF_STEP : Math.max(1, maxH * 1.05)) + 0.3;

    const layout = {
        margin: { t: 12, b: 40, l: 60, r: 10 },
        xaxis: {
            title: { text: 'Time since last dose (h)', font: { size: 10, color: '#666' } },
            color: '#666', gridcolor: '#f0f0f0', tickfont: { size: 9 },
            tickmode: 'array', tickvals, ticktext,
            range: [-0.3, xMax],
        },
        yaxis: {
            title: { text: yLabel || '', font: { size: 11, color: '#444' }, standoff: 8 },
            color: '#666', gridcolor: '#f0f0f0', tickfont: { size: 9 },
            autorange: reverseY ? 'reversed' : true,
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        shapes,
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

function renderGroupBar(divId, data, paramKey, reverseY, yLabel) {
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
        margin: { t: 12, b: 30, l: 60, r: 10 },
        xaxis: {
            tickvals: groups.map((_, i) => i),
            ticktext: groups,
            color: '#666',
            tickfont: { size: 10 },
        },
        yaxis: {
            title: { text: yLabel || '', font: { size: 11, color: '#444' }, standoff: 8 },
            color: '#666', gridcolor: '#f0f0f0', tickfont: { size: 9 },
            autorange: reverseY ? 'reversed' : true,
        },
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
    // Empty value = the "Select a subject" placeholder (used on the
    // group tab); nothing to show.
    if (!currentSubjectId) return;

    sessionStorage.setItem('dlc_lastSubjectId', String(currentSubjectId));
    if (typeof setLastSubject === 'function') setLastSubject(currentSubjectId);
    if (typeof setNavState === 'function') setNavState({ subjectId: parseInt(currentSubjectId) });
    // Keep the URL in sync so a refresh resolves to the current subject
    // instead of whichever subject was in ?subject=… at first load.
    try {
        const u = new URL(window.location.href);
        u.searchParams.set('subject', String(currentSubjectId));
        history.replaceState(null, '', u.toString());
    } catch {}
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

    // Selecting a subject always switches to the individual view —
    // including from the Group Comparison tab.
    switchTab('distances');
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

// X-scale slider: seconds of trace shown per screen width, applied to
// every distance/velocity plot.
(() => {
    const sl = document.getElementById('xScaleSlider');
    if (!sl) return;
    sl.addEventListener('input', () => {
        if (cachedTraces) renderAllDistancePlots();
    });
})();

// X-scale slider for the movement plots (width multiplier).
(() => {
    const sl = document.getElementById('movXScaleSlider');
    if (!sl) return;
    sl.addEventListener('input', () => {
        if (cachedMovements) renderDistMovementPlots();
    });
})();

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

// Group-tab source + sequence-effect dropdowns — re-fetch the group
// data with the new parameters.
document.getElementById('groupSourceSelect')?.addEventListener('change', () => {
    cachedGroup = null;
    loadGroup();
});
document.getElementById('groupSeqModeSelect')?.addEventListener('change', () => {
    cachedGroup = null;
    loadGroup();
});
['groupHandSelect', 'groupTrialSelect'].forEach(id =>
    document.getElementById(id)?.addEventListener('change', () => {
        cachedGroup = null;
        loadGroup();
    }));
// R²/Slope for the Sequence-Effect row — both values are already in the
// loaded data, so just re-render (no re-fetch).
document.querySelectorAll('#groupSeqMetric input').forEach(r =>
    r.addEventListener('change', () => { if (cachedGroup) renderGroupPlots(); }));

// Page-level distance source dropdown — applies to the Individual tab.
// Auto = corrections → mp_combined → mp_forward (per the backend's
// _load_distances_and_trials).
document.getElementById('resultsSourceSelect')?.addEventListener('change', () => {
    cachedTraces = null;
    cachedMovements = null;
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
