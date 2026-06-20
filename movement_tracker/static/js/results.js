/* Results page: distance traces, movement parameters, group comparison */

// Keep the hidden combined groupSeqModeSelect in sync with the two
// visible Type + Window selects (the same split the Explore page uses).
// Individual-tab Sequence Calculation: two sets of radios (Type +
// Window) keep the hidden combined #distSequenceMode select in sync
// so renderDistMovementPlots and the sequence-effect math (which
// already read .value off the select) don't need to change.
(function _wireSplitDistSeqMode() {
    const typeRadios = document.querySelectorAll('input[name="distSeqType"]');
    const winRadios  = document.querySelectorAll('input[name="distSeqWindow"]');
    const combined   = document.getElementById('distSequenceMode');
    if (!typeRadios.length || !winRadios.length || !combined) return;
    const getChecked = (rs, fallback) =>
        Array.from(rs).find(r => r.checked)?.value || fallback;
    const sync = () => {
        const t = getChecked(typeRadios, 'exp');
        const w = getChecked(winRadios, 'multi');
        // When Type = None, the window doesn't matter — mirror that
        // visually by dimming + disabling the window radios.
        const isNone = (t === 'none');
        winRadios.forEach(r => {
            r.disabled = isNone;
            const lbl = r.closest('label');
            if (lbl) lbl.style.opacity = isNone ? '0.4' : '';
        });
        combined.value = isNone ? 'none' : `${t}_${w}`;
        combined.dispatchEvent(new Event('change', { bubbles: true }));
    };
    typeRadios.forEach(r => r.addEventListener('change', sync));
    winRadios.forEach(r  => r.addEventListener('change', sync));
    // Initialise the radios from whatever the hidden select started
    // with so a saved/default 'exp_multi' lights up the right buttons.
    const cur = combined.value || 'exp_multi';
    if (cur === 'none') {
        typeRadios.forEach(r => r.checked = (r.value === 'none'));
    } else {
        const m = cur.match(/^(linear|exp)_(full|first10|multi)$/);
        if (m) {
            typeRadios.forEach(r => r.checked = (r.value === m[1]));
            winRadios.forEach(r  => r.checked = (r.value === m[2]));
        }
    }
    sync();
})();

(function _wireSplitGroupSeqMode() {
    const typeSel = document.getElementById('groupSeqTypeSelect');
    const winSel  = document.getElementById('groupSeqWindowSelect');
    const combined = document.getElementById('groupSeqModeSelect');
    if (!typeSel || !winSel || !combined) return;
    const sync = () => {
        const t = typeSel.value;
        const w = winSel.value;
        if (t === 'none') {
            combined.value = 'none';
            winSel.disabled = true;
            winSel.style.opacity = '0.4';
        } else {
            combined.value = `${t}_${w}`;
            winSel.disabled = false;
            winSel.style.opacity = '';
        }
        combined.dispatchEvent(new Event('change', { bubbles: true }));
    };
    typeSel.addEventListener('change', sync);
    winSel.addEventListener('change',  sync);
    const cur = combined.value || 'exp_multi';
    if (cur === 'none') {
        typeSel.value = 'none';
    } else {
        const m = cur.match(/^(linear|exp)_(full|first10|multi)$/);
        if (m) { typeSel.value = m[1]; winSel.value = m[2]; }
    }
    sync();
})();

// Static-site mode: only a subset of source × (hand, trial) combos
// is exported as JSON, so hide what we don't ship.  seq_mode is no
// longer a cache axis — every model's seq fields are embedded in
// each cached file — so the seq-mode dropdown stays unrestricted
// on the public site.
(function _trimStaticResultsControls() {
    if (!window.STATIC_RESULTS) return;
    const drop = (selectId, values) => {
        const el = document.getElementById(selectId);
        if (!el) return;
        Array.from(el.options).forEach(opt => {
            if (values.includes(opt.value)) opt.remove();
        });
    };
    const rename = (selectId, map) => {
        const el = document.getElementById(selectId);
        if (!el) return;
        Array.from(el.options).forEach(opt => {
            if (map[opt.value]) opt.textContent = map[opt.value];
        });
    };
    drop('groupHandSelect',        ['L', 'R', 'larger_se', 'smaller_se']);
    // Skeleton-fit v1 is now part of the default static cache, so
    // keep it in the dropdown; mp_forward still isn't exported and
    // stays trimmed in static mode.
    drop('groupSourceSelect',      ['mp_forward']);
    drop('resultsSourceSelect',    ['mp_forward']);
    rename('groupSourceSelect',    { corrections: 'DLC', mp_combined: 'MediaPipe', skeleton_v1: 'Skel fit v1', skeleton_v2: 'Skel fit v2' });
    rename('resultsSourceSelect',  { corrections: 'DLC', mp_combined: 'MediaPipe', skeleton_v1: 'Skel fit v1', skeleton_v2: 'Skel fit v2' });
    // Pair the Group hand+trial selects so they only allow the
    // exported (more,last) / (less,last) / (average,average) combos.
    const allowed = {
        'more':    ['last'],
        'less':    ['last'],
        'average': ['average'],
    };
    const handSel  = document.getElementById('groupHandSelect');
    const trialSel = document.getElementById('groupTrialSelect');
    const applyPair = () => {
        if (!handSel || !trialSel) return;
        const ok = allowed[handSel.value] || [];
        Array.from(trialSel.options).forEach(o => { o.hidden = !ok.includes(o.value); });
        if (!ok.includes(trialSel.value)) trialSel.value = ok[0] || trialSel.value;
    };
    handSel?.addEventListener('change', applyPair);
    applyPair();
})();

let subjects = [];

// Trials whose distance + velocity plots are currently collapsed
// (hidden) by user click on the trial title.  Keyed by trial.name so
// the state survives re-renders and subject reloads of the same
// subject.  Cleared when the subject changes.
let _collapsedTrials = new Set();

// ── Click-anywhere time-cursor across plots ──────────────────────
// When the user clicks any per-trial plot (distance, velocity, any
// of the movement-parameter scatters), we draw a faint vertical line
// at that x value on every plot showing the same trial.  Stored
// here as a module-level state object so a re-render of a single
// plot still picks the line back up.
let _clickHL = null;            // { trialIdx, time }  or  null
// Drag-to-mark intervals on the distance / velocity / IMI plots.
// _intervals[trialIdx] is an array of { x0, x1, marched } in seconds
// (trial-local).  Cleared on subject change; preserved across
// source / overlay / X-scale changes.
let _intervals = {};
let _intervalHover = null;      // { trialIdx, intervalIdx }  or  null
const _INTERVAL_PLOT_PREFIXES = ['distPlot_', 'imiPlot_', 'velPlot_'];
// Saved tapping events for the current subject — { open, peak, close,
// pause } each a list of GLOBAL video-frame indices.  Used by the
// Pause overlay on per-trial distance plots.
let _savedEvents = null;
const _baseShapeCount = {};     // divId → shapes.length right after newPlot
function _registerClickPlot(divId) {
    const div = document.getElementById(divId);
    if (!div || div._clickHLBound) return;
    div._clickHLBound = true;
    div.on('plotly_click', ev => _handlePlotClick(divId, ev));
    // Native-click fallback.  plotly_click only fires when the
    // cursor hits a data marker — the per-parameter movement
    // scatters are sparse so most clicks miss.  Catch the underlying
    // DOM event and convert pixel coords → data coords via the
    // subplot's xaxis directly.
    div.addEventListener('click', e => _handleNativeClick(divId, e));
    // On dist/imi/vel plots, install the drag-to-mark gesture and
    // hover detection for the per-interval action button.
    if (_isIntervalPlot(divId)) _wireIntervalDrag(divId);
    // Lock in how many shapes the plot started with so the highlight
    // append/strip cycle doesn't accumulate.
    _baseShapeCount[divId] = (div.layout?.shapes || []).length;
    // Re-apply existing intervals + click highlight on a fresh plot.
    _rebuildShapes(divId);
}

function _handlePlotClick(divId, ev) {
    if (!ev || !ev.points || !ev.points.length) return;
    const pt = ev.points[0];
    const x = (typeof pt.x === 'number') ? pt.x : parseFloat(pt.x);
    if (!isFinite(x)) return;
    let trialIdx = null;
    if (divId.startsWith('distPlot_') || divId.startsWith('velPlot_') || divId.startsWith('imiPlot_')) {
        trialIdx = parseInt(divId.split('_')[1]);
    } else if (divId.startsWith('movPlot_') || divId.startsWith('distMovPlot_')) {
        // The movement-plot div bundles every trial's data on its own
        // xaxis (x / x2 / x3 / …).  Position in trialInfo = trial idx,
        // since data.trials is in sorted trial-idx order.
        const axId = pt.xaxis?._id || 'x';
        const m = axId.match(/^x(\d*)$/);
        trialIdx = m ? (m[1] ? parseInt(m[1]) - 1 : 0) : 0;
    } else return;
    if (!Number.isFinite(trialIdx) || trialIdx < 0) return;
    _clickHL = { trialIdx, time: x };
    // Tell the native-click listener to skip the same gesture.
    const div = document.getElementById(divId);
    if (div) div._lastPlotlyClickAt = performance.now();
    _applyHighlightAll();
}

function _handleNativeClick(divId, e) {
    const div = document.getElementById(divId);
    if (!div || !div._fullLayout) return;
    // plotly_click fires before the native click on a marker hit —
    // skip the trailing native handler so we don't re-process.
    if (div._lastPlotlyClickAt && (performance.now() - div._lastPlotlyClickAt) < 200) return;
    // A drag-to-mark gesture just committed an interval — the
    // trailing native click from the same pointer up isn't a
    // user-intended click.
    if (div._suppressClickUntil && performance.now() < div._suppressClickUntil) return;
    const rect = div.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const fl = div._fullLayout;
    // Find which subplot xaxis the click pixel falls inside.
    let chosenAx = null, chosenN = 1;
    Object.keys(fl).forEach(k => {
        const m = k.match(/^xaxis(\d*)$/);
        if (!m) return;
        const ax = fl[k];
        if (!ax || ax._offset == null || ax._length == null) return;
        if (px >= ax._offset && px <= ax._offset + ax._length) {
            chosenAx = ax;
            chosenN = m[1] ? parseInt(m[1]) : 1;
        }
    });
    if (!chosenAx || typeof chosenAx.p2c !== 'function') return;
    // Bound on yaxis so clicks in title / margin area don't count.
    const yax = fl.yaxis;
    if (yax && yax._offset != null
            && (py < yax._offset || py > yax._offset + yax._length)) return;
    const dataX = chosenAx.p2c(px - chosenAx._offset);
    if (!isFinite(dataX)) return;
    let trialIdx = null;
    if (divId.startsWith('distPlot_') || divId.startsWith('velPlot_') || divId.startsWith('imiPlot_')) {
        trialIdx = parseInt(divId.split('_')[1]);
    } else if (divId.startsWith('movPlot_') || divId.startsWith('distMovPlot_')) {
        trialIdx = chosenN - 1;
    } else return;
    if (!Number.isFinite(trialIdx) || trialIdx < 0) return;
    _clickHL = { trialIdx, time: dataX };
    _applyHighlightAll();
}

function _applyHighlightAll() {
    if (!_clickHL) return;
    const { trialIdx } = _clickHL;
    // Repaint dist/imi/vel for the clicked trial AND every other
    // trial (so stale lines on previously-clicked trials clear).
    document.querySelectorAll('[id^="distPlot_"], [id^="imiPlot_"], [id^="velPlot_"]')
        .forEach(d => _rebuildShapes(d.id));
    // Movement plots: repaint all so the trial's subplot gets the
    // line and others stay clean.
    document.querySelectorAll('[id^="movPlot_"], [id^="distMovPlot_"]')
        .forEach(d => _rebuildShapes(d.id));
}

// Back-compat aliases — interval handlers and older call sites
// route through the single shape composer.
function _clearHighlightOn(divId) { _rebuildShapes(divId); }
function _applyHighlightTo(divId) { _rebuildShapes(divId); }

// Single source of truth for a plot div's `layout.shapes`:
//   base (sliced) + drag-mark intervals (for dist/imi/vel only)
//   + click-highlight vertical line (if any).
function _rebuildShapes(divId) {
    const div = document.getElementById(divId);
    if (!div || !div.layout) return;
    const base = _baseShapeCount[divId] || 0;
    const shapes = (div.layout.shapes || []).slice(0, base);

    // Drag-marked intervals (dist / imi / vel plots only; one trial).
    const trialIdx = _trialIdxOfDiv(divId);
    if (trialIdx != null && _isIntervalPlot(divId)) {
        const list = _intervals[trialIdx] || [];
        list.forEach(iv => {
            if (iv.marched) {
                shapes.push(..._marchedShapes(iv, div));
            } else {
                shapes.push(_bandShape(iv.x0, iv.x1, 'rgba(33,150,243,0.20)'));
            }
            // Thin edge lines so the boundaries stay readable even
            // when the band fill is light.
            shapes.push(_edgeLine(iv.x0));
            shapes.push(_edgeLine(iv.x1));
        });
    }

    // Click-highlight vertical line.
    if (_clickHL) {
        const lineStyle = { color: 'rgba(0,0,0,0.45)', width: 1, dash: 'dot' };
        if (divId.startsWith('movPlot_') || divId.startsWith('distMovPlot_')) {
            // Only the clicked trial's subplot gets a line.
            const axId = _clickHL.trialIdx === 0 ? 'x' : 'x' + (_clickHL.trialIdx + 1);
            shapes.push({
                type: 'line', xref: axId, yref: 'paper',
                x0: _clickHL.time, x1: _clickHL.time, y0: 0, y1: 1,
                line: lineStyle, layer: 'above',
            });
        } else if (trialIdx === _clickHL.trialIdx) {
            shapes.push({
                type: 'line', xref: 'x', yref: 'paper',
                x0: _clickHL.time, x1: _clickHL.time, y0: 0, y1: 1,
                line: lineStyle, layer: 'above',
            });
        }
    }

    try { Plotly.relayout(divId, { shapes }); } catch (_) {}
}

function _isIntervalPlot(divId) {
    return _INTERVAL_PLOT_PREFIXES.some(p => divId.startsWith(p));
}
function _trialIdxOfDiv(divId) {
    for (const p of _INTERVAL_PLOT_PREFIXES) {
        if (divId.startsWith(p)) return parseInt(divId.slice(p.length));
    }
    return null;
}
function _bandShape(x0, x1, fill) {
    return {
        type: 'rect', xref: 'x', yref: 'paper',
        x0, x1, y0: 0, y1: 1,
        fillcolor: fill, line: { width: 0 }, layer: 'below',
    };
}
function _edgeLine(x) {
    return {
        type: 'line', xref: 'x', yref: 'paper',
        x0: x, x1: x, y0: 0, y1: 1,
        line: { color: 'rgba(33,150,243,0.55)', width: 1 }, layer: 'below',
    };
}
// Tile the trial in BOTH directions from [iv.x0, iv.x1] with bands of
// width dt = iv.x1 - iv.x0, alternating two fills so each repetition
// is visually distinct.  k = 0 is the source band (centered at the
// original interval); k = +1, +2, ... extend forward; k = -1, -2, …
// extend backward to the start of the trial.
function _marchedShapes(iv, div) {
    const dt = iv.x1 - iv.x0;
    if (!(dt > 0)) return [];
    const fl = div._fullLayout;
    const ax = fl && fl.xaxis;
    const xMin = (ax && ax.range) ? ax.range[0] : 0;
    const xMax = (ax && ax.range) ? ax.range[1] : (iv.x1 + dt * 50);
    // Compute the smallest k whose band's RIGHT edge sits above xMin
    // and the largest k whose band's LEFT edge sits below xMax.
    const kMin = Math.ceil((xMin - iv.x1) / dt);
    const kMax = Math.floor((xMax - iv.x0) / dt);
    const out = [];
    for (let k = kMin; k <= kMax; k++) {
        const s = Math.max(xMin, iv.x0 + k * dt);
        const e = Math.min(xMax, iv.x0 + (k + 1) * dt);
        if (e <= s) continue;
        // JS modulo is signed — Math.abs keeps the parity stable
        // across negative k so adjacent bands always alternate.
        const col = (Math.abs(k) % 2 === 0)
            ? 'rgba(33,150,243,0.22)'
            : 'rgba(33,150,243,0.06)';
        out.push(_bandShape(s, e, col));
    }
    return out;
}

// ── Drag-to-mark intervals on dist/imi/vel plots ──────────────────
//
// Native pointer handlers convert pixel coords → trial-local
// seconds via the plot's xaxis.  A drag of >4 px commits an
// interval into _intervals[trialIdx].  Without shift/ctrl the new
// interval REPLACES previous ones on that trial; with shift OR
// ctrl held the new one is appended.  Sub-threshold drags fall
// through to the click-cursor handler (suppressed via
// _suppressNextClickOn so we don't double-fire).
function _wireIntervalDrag(divId) {
    const div = document.getElementById(divId);
    if (!div || div._intervalDragBound) return;
    div._intervalDragBound = true;
    const trialIdx = _trialIdxOfDiv(divId);

    // Drag state for this plot.  Captured at pointerdown and read at
    // pointermove / pointerup.
    let down = null;

    div.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        const ax = div._fullLayout && div._fullLayout.xaxis;
        if (!ax || ax._offset == null) return;
        const rect = div.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        // Only react inside the plot area (avoid axis labels).
        if (px < ax._offset || px > ax._offset + ax._length) return;
        const yax = div._fullLayout.yaxis;
        if (yax && yax._offset != null
                && (py < yax._offset || py > yax._offset + yax._length)) return;
        // Edge → resize.  Inside (but not edge) → move the whole
        // interval.  Empty space → start a new drag-mark gesture.
        const hit = _hitTestIntervalEdge(trialIdx, ax, px);
        if (hit) {
            down = {
                mode: 'resize', clientX0: e.clientX,
                intervalIdx: hit.intervalIdx, edge: hit.edge, ax,
            };
        } else {
            const inside = _hitTestIntervalInside(trialIdx, ax, px);
            if (inside) {
                const iv = (_intervals[trialIdx] || [])[inside.intervalIdx];
                down = {
                    mode: 'move', clientX0: e.clientX,
                    intervalIdx: inside.intervalIdx,
                    startData: ax.p2c(px - ax._offset),
                    origX0: iv ? iv.x0 : 0,
                    origX1: iv ? iv.x1 : 0,
                    ax,
                };
            } else {
                down = {
                    mode: 'new', clientX0: e.clientX,
                    startData: ax.p2c(px - ax._offset), ax,
                    additive: !!(e.shiftKey || e.ctrlKey || e.metaKey),
                };
            }
        }
        try { div.setPointerCapture(e.pointerId); } catch (_) {}
        e.preventDefault();
    });

    div.addEventListener('pointermove', (e) => {
        if (!down) {
            // No drag in progress — update cursor: col-resize on
            // edges (drag to resize), move inside (drag to slide
            // the whole interval), default elsewhere (drag to
            // make a new interval).
            const ax = div._fullLayout && div._fullLayout.xaxis;
            if (!ax || ax._offset == null) return;
            const rect = div.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const edge = _hitTestIntervalEdge(trialIdx, ax, px);
            if (edge) { _setPlotCursor(div, 'col-resize'); return; }
            const inside = _hitTestIntervalInside(trialIdx, ax, px);
            _setPlotCursor(div, inside ? 'move' : '');
            return;
        }
        const rect = div.getBoundingClientRect();
        const cur = down.ax.p2c(e.clientX - rect.left - down.ax._offset);
        if (down.mode === 'resize') {
            const list = _intervals[trialIdx] || [];
            const iv = list[down.intervalIdx];
            if (!iv) return;
            if (down.edge === 'x0') {
                // Keep x0 strictly below x1 with a 1 px floor on width.
                const minW = down.ax.p2c(1) - down.ax.p2c(0);
                iv.x0 = Math.min(cur, iv.x1 - Math.abs(minW || 0.01));
            } else {
                const minW = down.ax.p2c(1) - down.ax.p2c(0);
                iv.x1 = Math.max(cur, iv.x0 + Math.abs(minW || 0.01));
            }
            _rebuildShapesForTrial(trialIdx);
            _updateHoverButtonContent(trialIdx, down.intervalIdx);
        } else if (down.mode === 'move') {
            const list = _intervals[trialIdx] || [];
            const iv = list[down.intervalIdx];
            if (!iv) return;
            const shift = cur - down.startData;
            iv.x0 = down.origX0 + shift;
            iv.x1 = down.origX1 + shift;
            _rebuildShapesForTrial(trialIdx);
            _showHoverButton(trialIdx, down.intervalIdx);
        } else {
            const x0 = Math.min(down.startData, cur);
            const x1 = Math.max(down.startData, cur);
            _previewInterval(trialIdx, x0, x1);
        }
    });

    const finish = (e) => {
        if (!down) return;
        const ax = down.ax;
        const rect = div.getBoundingClientRect();
        const cur = ax.p2c(e.clientX - rect.left - ax._offset);
        const movedPx = Math.abs(e.clientX - down.clientX0);
        if (down.mode === 'resize' || down.mode === 'move') {
            down = null;
            // Suppress the trailing native-click for this gesture
            // (a drag that ended inside an interval would otherwise
            // be re-interpreted as a click on the cursor handler).
            div._suppressClickUntil = performance.now() + 250;
            // Final rebuild — pointermove already updated bands live.
            _rebuildShapesForTrial(trialIdx);
            return;
        }
        const additive = down.additive;
        const startData = down.startData;
        down = null;
        _clearPreview(trialIdx);
        if (movedPx < 4) return;        // sub-threshold → click handlers
        const x0 = Math.min(startData, cur);
        const x1 = Math.max(startData, cur);
        if (!(x1 - x0 > 0)) return;
        // Commit.  Replace existing intervals on the trial unless
        // shift/ctrl/cmd is held.
        const list = additive ? (_intervals[trialIdx] || []) : [];
        list.push({ x0, x1, marched: false });
        _intervals[trialIdx] = list;
        div._suppressClickUntil = performance.now() + 250;
        _rebuildShapesForTrial(trialIdx);
    };
    div.addEventListener('pointerup', finish);
    div.addEventListener('pointercancel', () => {
        down = null; _clearPreview(trialIdx);
        _setPlotCursor(div, '');
    });

    // Hover detection for the per-interval action button.
    div.addEventListener('mousemove', (e) => _onIntervalHover(divId, e));
    div.addEventListener('mouseleave', () => {
        _scheduleHoverHide(trialIdx);
        if (!down) _setPlotCursor(div, '');
    });
}

// Plotly's draglayer (``<rect class="nsewdrag drag">``) sits on top
// of the plot div with its own ``cursor: crosshair`` style, which
// beats ``div.style.cursor`` until the user starts dragging.  To
// surface the resize / move hint on hover (not just during drag),
// stamp the cursor on the drag layer too.
function _setPlotCursor(div, cur) {
    div.style.cursor = cur;
    // Plotly draws several "drag" rects (nsewdrag for the plot
    // area, plus n/s/e/w/edge handles for axis-edge drags).
    // Targeting them all keeps the cursor consistent no matter
    // which sub-region the pointer is over.
    div.querySelectorAll('.drag').forEach(el => {
        el.style.cursor = cur;
    });
}

// Return { intervalIdx, edge } if the plot-pixel `px` lies within
// the 6-px hit zone of any interval edge on this trial; otherwise
// null.  Used by the drag handler and the cursor-hint move handler.
function _hitTestIntervalEdge(trialIdx, ax, px) {
    const list = _intervals[trialIdx] || [];
    if (!list.length) return null;
    const TOL = 6;
    let best = null, bestDist = TOL + 1;
    list.forEach((iv, i) => {
        const px0 = ax.d2p(iv.x0) + ax._offset;
        const px1 = ax.d2p(iv.x1) + ax._offset;
        const d0 = Math.abs(px - px0);
        const d1 = Math.abs(px - px1);
        if (d0 <= TOL && d0 < bestDist) { best = { intervalIdx: i, edge: 'x0' }; bestDist = d0; }
        if (d1 <= TOL && d1 < bestDist) { best = { intervalIdx: i, edge: 'x1' }; bestDist = d1; }
    });
    return best;
}

// Return { intervalIdx } if the plot-pixel ``px`` lies STRICTLY
// inside an interval (excluding the 6-px edge hit zone) on this
// trial; otherwise null.  Edge hits win first (resize); inside
// hits drive the whole-interval drag.
function _hitTestIntervalInside(trialIdx, ax, px) {
    const list = _intervals[trialIdx] || [];
    if (!list.length) return null;
    const TOL = 6;
    const xData = ax.p2c(px - ax._offset);
    for (let i = 0; i < list.length; i++) {
        const iv = list[i];
        if (xData < iv.x0 || xData > iv.x1) continue;
        const px0 = ax.d2p(iv.x0) + ax._offset;
        const px1 = ax.d2p(iv.x1) + ax._offset;
        if (Math.abs(px - px0) <= TOL || Math.abs(px - px1) <= TOL) continue;
        return { intervalIdx: i };
    }
    return null;
}

function _hideHoverButton(trialIdx) {
    const wrapper = document.getElementById(`trialWrap_${trialIdx}`);
    const btn = wrapper && wrapper.querySelector(':scope > .interval-actions');
    if (btn) btn.style.display = 'none';
}

function _previewInterval(trialIdx, x0, x1) {
    _INTERVAL_PLOT_PREFIXES.forEach(prefix => {
        const div = document.getElementById(prefix + trialIdx);
        if (!div || !div.layout) return;
        const base = _baseShapeCount[prefix + trialIdx] || 0;
        const list = _intervals[trialIdx] || [];
        const shapes = (div.layout.shapes || []).slice(0, base);
        list.forEach(iv => {
            if (iv.marched) shapes.push(..._marchedShapes(iv, div));
            else shapes.push(_bandShape(iv.x0, iv.x1, 'rgba(33,150,243,0.20)'));
            shapes.push(_edgeLine(iv.x0));
            shapes.push(_edgeLine(iv.x1));
        });
        shapes.push(_bandShape(x0, x1, 'rgba(33,150,243,0.30)'));
        try { Plotly.relayout(prefix + trialIdx, { shapes }); } catch (_) {}
    });
}
function _clearPreview(trialIdx) {
    if (trialIdx == null) return;
    _rebuildShapesForTrial(trialIdx);
}
function _rebuildShapesForTrial(trialIdx) {
    _INTERVAL_PLOT_PREFIXES.forEach(p => _rebuildShapes(p + trialIdx));
}

// ── Per-interval hover button (✕ delete + March across) ──────────
function _onIntervalHover(divId, e) {
    const trialIdx = _trialIdxOfDiv(divId);
    if (trialIdx == null) return;
    const list = _intervals[trialIdx] || [];
    if (!list.length) { _scheduleHoverHide(trialIdx); return; }
    const div = document.getElementById(divId);
    const ax = div._fullLayout && div._fullLayout.xaxis;
    if (!ax || ax._offset == null) return;
    const rect = div.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const xData = ax.p2c(px - ax._offset);
    // Find first interval whose [x0,x1] contains the cursor.  When
    // marched, treat the original [x0,x1] as the anchor (the rest
    // are derived bands) so the button only shows over the source.
    const hit = list.findIndex(iv => xData >= iv.x0 && xData <= iv.x1);
    if (hit < 0) { _scheduleHoverHide(trialIdx); return; }
    _showHoverButton(trialIdx, hit);
}

function _ensureHoverButton(trialIdx) {
    const wrapper = document.getElementById(`trialWrap_${trialIdx}`);
    if (!wrapper) return null;
    let btn = wrapper.querySelector(':scope > .interval-actions');
    if (btn) return btn;
    btn = document.createElement('div');
    btn.className = 'interval-actions';
    btn.style.cssText = [
        'position:absolute', 'top:2px', 'transform:translateX(-50%)',
        'display:none', 'gap:4px', 'z-index:10',
        'background:rgba(255,255,255,0.95)',
        'border:1px solid rgba(33,150,243,0.55)',
        'border-radius:4px', 'padding:2px 4px',
        'box-shadow:0 1px 3px rgba(0,0,0,0.15)',
        'font-size:11px', 'align-items:center',
    ].join(';');
    const march = document.createElement('button');
    march.type = 'button';
    march.className = 'iv-march';
    march.textContent = 'March across';
    march.title = 'Tile this interval forward across the trial';
    march.style.cssText = 'border:none;background:transparent;color:#1976D2;font-weight:600;cursor:pointer;padding:0 2px;font-size:11px;';
    // Live read-out of the interval's duration (and frequency once
    // marched).  Updated every pointermove via _updateHoverButtonContent
    // so the user sees the value tick as they drag.
    const stats = document.createElement('span');
    stats.className = 'iv-stats';
    stats.style.cssText = 'color:#444;font-variant-numeric:tabular-nums;padding:0 4px;font-size:11px;border-left:1px solid rgba(0,0,0,0.15);';
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'iv-del';
    del.textContent = '✕';
    del.title = 'Delete this interval';
    del.style.cssText = 'border:none;background:transparent;color:#c62828;font-weight:700;cursor:pointer;padding:0 2px;font-size:12px;';
    btn.appendChild(march);
    btn.appendChild(stats);
    btn.appendChild(del);
    // Stay open while pointer is over the button itself.
    btn.addEventListener('mouseenter', () => { if (_hoverHideT) clearTimeout(_hoverHideT); });
    btn.addEventListener('mouseleave', () => _scheduleHoverHide(trialIdx));
    march.addEventListener('click', (e) => {
        e.stopPropagation();
        const idx = btn._intervalIdx;
        const list = _intervals[trialIdx] || [];
        if (idx == null || !list[idx]) return;
        list[idx].marched = !list[idx].marched;
        _updateHoverButtonContent(trialIdx, idx);
        _rebuildShapesForTrial(trialIdx);
    });
    del.addEventListener('click', (e) => {
        e.stopPropagation();
        const idx = btn._intervalIdx;
        const list = _intervals[trialIdx] || [];
        if (idx == null || !list[idx]) return;
        list.splice(idx, 1);
        if (!list.length) delete _intervals[trialIdx];
        btn.style.display = 'none';
        _rebuildShapesForTrial(trialIdx);
    });
    wrapper.appendChild(btn);
    return btn;
}

let _hoverHideT = null;
function _scheduleHoverHide(trialIdx) {
    if (_hoverHideT) clearTimeout(_hoverHideT);
    _hoverHideT = setTimeout(() => {
        const wrapper = document.getElementById(`trialWrap_${trialIdx}`);
        const btn = wrapper && wrapper.querySelector(':scope > .interval-actions');
        if (btn) btn.style.display = 'none';
    }, 180);
}

// Update just the readout strings on the hover button (march/stop
// label, duration in ms, frequency in Hz iff marched).  Used during
// drag so the numbers tick live without re-running the positioning
// math every frame.
function _updateHoverButtonContent(trialIdx, intervalIdx) {
    const wrapper = document.getElementById(`trialWrap_${trialIdx}`);
    const btn = wrapper && wrapper.querySelector(':scope > .interval-actions');
    if (!btn) return;
    const list = _intervals[trialIdx] || [];
    const iv = list[intervalIdx];
    if (!iv) return;
    const march = btn.querySelector('.iv-march');
    if (march) march.textContent = iv.marched ? 'Stop march' : 'March across';
    const stats = btn.querySelector('.iv-stats');
    if (stats) {
        const durSec = Math.max(0, iv.x1 - iv.x0);
        const durMs = Math.round(durSec * 1000);
        let txt = `${durMs} ms`;
        if (iv.marched && durSec > 0) {
            txt += ` · ${(1 / durSec).toFixed(2)} Hz`;
        }
        stats.textContent = txt;
    }
}

function _showHoverButton(trialIdx, intervalIdx) {
    if (_hoverHideT) { clearTimeout(_hoverHideT); _hoverHideT = null; }
    const btn = _ensureHoverButton(trialIdx);
    if (!btn) return;
    btn._intervalIdx = intervalIdx;
    const list = _intervals[trialIdx] || [];
    const iv = list[intervalIdx];
    if (!iv) return;
    // Update march/stop label + the duration/frequency readout.
    _updateHoverButtonContent(trialIdx, intervalIdx);
    // Position centered over the interval on the distance plot.
    // The wrapper is the offset parent; the distance plot now sits
    // inside an inner flex row with a sticky y-axis column to its
    // left, so add the dist plot's offsetLeft within the wrapper.
    const dist = document.getElementById(`distPlot_${trialIdx}`);
    const wrapper = document.getElementById(`trialWrap_${trialIdx}`);
    if (!dist || !wrapper || !dist._fullLayout || !dist._fullLayout.xaxis) return;
    const ax = dist._fullLayout.xaxis;
    if (ax._offset == null) return;
    const mid = (iv.x0 + iv.x1) / 2;
    const pxMid = ax.d2p(mid) + ax._offset;
    // Walk offset chain to find dist's x position relative to wrapper.
    let leftWithin = 0, el = dist;
    while (el && el !== wrapper) {
        leftWithin += el.offsetLeft;
        el = el.offsetParent;
    }
    btn.style.left = (leftWithin + pxMid) + 'px';
    btn.style.display = 'inline-flex';
}
function _updateHoverButtonForTrial(trialIdx) {
    // Refresh the button's stored idx after list mutation.
    const wrapper = document.getElementById(`trialWrap_${trialIdx}`);
    const btn = wrapper && wrapper.querySelector(':scope > .interval-actions');
    if (btn && btn.style.display !== 'none') btn.style.display = 'none';
}
let currentTab = 'distances';
let currentSubjectId = null;

// Cached data
let cachedTraces = null;
let cachedMovements = null;
let cachedGroup = null;
let cachedSequenceAssignments = null; // { byTrial: { trialIdx: {sequences, seq_r2} }, totalSeqs, totalR2 }
let cachedPCA = null;
let _resultsViewMode = 'distances';

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
    peak_dist: 'Peak Thumb-Index Distance (mm)',
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

const PC_COLORS = ['#2196F3', '#FF5722', '#4CAF50'];

function _syncViewMode() {
    const mode = _resultsViewMode;
    const distPlots = document.getElementById('distancePlots');
    const pcaPlots  = document.getElementById('pcaPlots');
    if (distPlots) distPlots.style.display = mode === 'pca' ? 'none' : '';
    if (pcaPlots)  pcaPlots.style.display  = mode === 'pca' ? '' : 'none';
    const pcaCompCtl = document.getElementById('pcaCompControls');
    if (pcaCompCtl) pcaCompCtl.style.display = mode === 'pca' ? 'inline-flex' : 'none';

    // IMI row only relevant for distances
    const imiRefRow = document.getElementById('imiRefRow');
    if (imiRefRow) imiRefRow.style.display = mode === 'pca' ? 'none' : '';

    // Lock Y-axis and its preceding separator only make sense for distances
    const lockYLabel = document.getElementById('lockYAxis')?.closest('label');
    if (lockYLabel) {
        lockYLabel.style.display = mode === 'pca' ? 'none' : '';
        const sep = lockYLabel.previousElementSibling;
        if (sep) sep.style.display = mode === 'pca' ? 'none' : '';
    }

    // Velocity-specific overlay labels (Peak Open/Close Vel + R²)
    const velLabels = [
        document.getElementById('overlayPeakOpenVel')?.closest('label'),
        document.getElementById('overlayPeakCloseVel')?.closest('label'),
        document.getElementById('overlayR2'),
    ];
    const velSep = document.getElementById('overlayPeakOpenVel')
        ?.closest('label')?.previousElementSibling;
    velLabels.forEach(el => { if (el) el.style.display = mode === 'pca' ? 'none' : ''; });
    if (velSep) velSep.style.display = mode === 'pca' ? 'none' : '';

    // Lower sections (IP scatters, movement params, shape overlay) not relevant in PCA mode
    const ipCtl   = document.getElementById('ipPlotControls');
    const ipPlots = document.getElementById('ipPlots');
    const movCtl  = document.getElementById('distMovementControls');
    const movPlots = document.getElementById('distMovementPlots');
    const shapeSec = document.getElementById('shapeOverlaySection');

    if (mode === 'pca') {
        if (ipCtl)   ipCtl.style.display   = 'none';
        if (ipPlots) ipPlots.innerHTML      = '';
        if (movCtl)  movCtl.style.display   = 'none';
        if (movPlots) movPlots.innerHTML    = '';
        if (shapeSec) shapeSec.style.display = 'none';
        if (currentSubjectId) {
            if (!cachedPCA) {
                loadFingertipPCA(currentSubjectId);
            } else {
                renderFingertipPCA();
            }
        }
    } else {
        // Restore distance view
        if (cachedTraces && cachedTraces.trials && cachedTraces.trials.length > 0) {
            renderAllDistancePlots();
            if (cachedMovements && cachedMovements.movements && cachedMovements.movements.length > 0) {
                if (ipCtl)  ipCtl.style.display  = '';
                if (movCtl) movCtl.style.display  = '';
                renderIntervalParamPlots();
                renderDistMovementPlots();
                renderShapeOverlayPlots();
            }
        }
    }
}

async function loadFingertipPCA(subjectId) {
    const container = document.getElementById('pcaPlots');
    if (!container) return;
    container.innerHTML = '<div class="results-no-data">Loading PCA…</div>';
    const src = document.getElementById('resultsSourceSelect')?.value || 'auto';
    try {
        const data = await API.get(`/api/results/${subjectId}/fingertip_pca?source=${src}`);
        cachedPCA = data;
        renderFingertipPCA();
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="color:#d32f2f;">PCA failed: ${e.message}</div>`;
    }
}

function renderFingertipPCA() {
    const container = document.getElementById('pcaPlots');
    const data = cachedPCA;
    if (!container) return;
    container.innerHTML = '';

    if (!data || !data.trials || data.trials.length === 0) {
        container.innerHTML = '<div class="results-no-data">No MediaPipe data available for fingertip PCA</div>';
        return;
    }

    const overlayPeak  = document.getElementById('overlayPeakDist')?.checked;
    const overlayOpen  = document.getElementById('overlayOpen')?.checked;
    const overlayClose = document.getElementById('overlayClose')?.checked;
    const overlayPause = document.getElementById('overlayPause')?.checked;

    // PC visibility — default-on when checkbox missing.
    const _pcOn = ci => {
        const el = document.getElementById(`showPC${ci + 1}`);
        return el ? el.checked : true;
    };

    // FFT smoothing window (centered box-car, half-width = slider value).
    const fftSmoothRaw = parseInt(document.getElementById('pcaFftSmooth')?.value, 10);
    const fftSmoothHW = Number.isFinite(fftSmoothRaw) ? Math.max(0, fftSmoothRaw) : 0;
    const _smoothFft = (power) => {
        if (!power || fftSmoothHW === 0) return power;
        const n = power.length;
        const out = new Array(n);
        let sum = 0, count = 0;
        // Prime the leading window.
        for (let i = 0; i <= fftSmoothHW && i < n; i++) {
            const v = power[i];
            if (v != null && Number.isFinite(v)) { sum += v; count++; }
        }
        for (let i = 0; i < n; i++) {
            const lo = i - fftSmoothHW - 1;
            const hi = i + fftSmoothHW;
            if (lo >= 0) {
                const v = power[lo];
                if (v != null && Number.isFinite(v)) { sum -= v; count--; }
            }
            if (hi < n && i > 0) {
                const v = power[hi];
                if (v != null && Number.isFinite(v)) { sum += v; count++; }
            }
            out[i] = count > 0 ? sum / count : null;
        }
        return out;
    };

    const movByTrial = {};
    if (cachedMovements && cachedMovements.movements) {
        cachedMovements.movements.forEach(m => {
            (movByTrial[m.trial_idx] ||= []).push(m);
        });
    }

    const xScaleRaw = parseFloat(document.getElementById('xScaleSlider')?.value);
    const secPerWidth = isFinite(xScaleRaw) ? (125 - xScaleRaw) : 75;
    const containerW = container.clientWidth || container.parentElement?.clientWidth || 1200;

    data.trials.forEach((trial, idx) => {
        const { n_components: n_comp, times, pc_scores, explained_variance: ev,
                fft_freqs, fft_power, fps = 60 } = trial;

        // Trial reference from cachedTraces for event frame→time conversion
        const trTrial   = cachedTraces?.trials?.[idx];
        const trialStart = trTrial?.start_frame ?? 0;
        const trialEnd   = trTrial?.end_frame   ?? Infinity;
        const trialMovs  = movByTrial[idx] || [];

        // Event times (seconds into trial)
        const _evTimes = (field, movs) =>
            movs.map(m => m[field]).filter(f => f != null)
                .map(f => (f - trialStart) / fps);

        const peakTimes  = overlayPeak  ? _evTimes('peak_frame',  trialMovs) : [];
        const openTimes  = overlayOpen  ? _evTimes('open_frame',  trialMovs) : [];
        const closeTimes = overlayClose ? _evTimes('close_frame', trialMovs) : [];
        const pauseTimes = [];
        if (overlayPause && _savedEvents?.pause && trTrial) {
            _savedEvents.pause.forEach(gf => {
                if (gf >= trialStart && gf <= trialEnd)
                    pauseTimes.push((gf - trialStart) / fps);
            });
        }

        const _evShapes = (ts, color) => ts.map(t => ({
            type: 'line', xref: 'x', yref: 'paper',
            x0: t, x1: t, y0: 0, y1: 1,
            line: { color, width: 1, dash: 'dot' },
        }));
        const eventShapes = [
            ..._evShapes(peakTimes,  '#FF9800'),
            ..._evShapes(openTimes,  '#2ca02c'),
            ..._evShapes(closeTimes, '#d62728'),
            ..._evShapes(pauseTimes, '#cc66ff'),
        ];

        const totalSec = times.at(-1) || (times.length / fps);
        const plotW = Math.max(containerW, (totalSec / secPerWidth) * containerW);

        // --- Block container ---
        const block = document.createElement('div');
        block.style.marginBottom = '24px';

        const header = document.createElement('div');
        header.style.cssText = 'padding:2px 8px 6px;';
        const trialPart = String(trial.name || '').split('_').pop();
        const dimLabel  = data.is_3d ? '3D' : '2D';
        header.innerHTML = `<span style="font-size:13px;font-weight:600;color:#666;">` +
            `Trial: ${trialPart} — Index Fingertip ${dimLabel} PCA</span>`;
        block.appendChild(header);

        // Scrollable wrapper for time-series plots
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'overflow-x:auto;width:100%;';

        const timeDivIds = [];
        const nShow = Math.min(n_comp, 3);
        // Index of the last VISIBLE PC's time-series row -- that row
        // gets the X-axis labels and title.  Falls back to the last
        // PC when nothing is visible (the row stays hidden anyway).
        let lastVisibleCi = -1;
        for (let ci = 0; ci < nShow; ci++) {
            if (_pcOn(ci)) lastVisibleCi = ci;
        }
        if (lastVisibleCi < 0) lastVisibleCi = nShow - 1;
        for (let ci = 0; ci < nShow; ci++) {
            const divId = `pcaTime_${idx}_${ci}`;
            timeDivIds.push(divId);
            const div = document.createElement('div');
            div.id = divId;
            // Height shrinks per row; PC1 taller for emphasis.  Row
            // is hidden entirely when its checkbox is off.
            const visible = _pcOn(ci);
            div.style.cssText = `height:${ci === 0 ? 160 : 120}px;width:${plotW}px;` +
                                (visible ? '' : 'display:none;');
            wrapper.appendChild(div);
        }
        block.appendChild(wrapper);

        // FFT row: one plot per PC at 1/3 width so all three fit on a
        // single line.  Hidden PCs collapse their flex item.
        const fftDivIds = [];
        const fftRow = document.createElement('div');
        fftRow.style.cssText = 'display:flex;gap:8px;margin-top:4px;';
        for (let ci = 0; ci < nShow; ci++) {
            const divId = `pcaFft_${idx}_${ci}`;
            fftDivIds.push(divId);
            const div = document.createElement('div');
            div.id = divId;
            const visible = _pcOn(ci);
            // flex:1 splits available width evenly across visible PCs.
            div.style.cssText = `flex:1;min-width:0;height:180px;` +
                                (visible ? '' : 'display:none;');
            fftRow.appendChild(div);
        }
        block.appendChild(fftRow);

        container.appendChild(block);

        // --- Render PC time-series ---
        for (let ci = 0; ci < nShow; ci++) {
            if (!_pcOn(ci)) continue;
            const isLast = ci === lastVisibleCi;
            const pct    = ev[ci] != null ? ` (${(ev[ci] * 100).toFixed(0)}%)` : '';
            Plotly.newPlot(timeDivIds[ci], [{
                x: times, y: pc_scores[ci],
                type: 'scatter', mode: 'lines',
                line: { color: PC_COLORS[ci], width: 1 },
                connectgaps: false,
                hovertemplate: `%{x:.3f}s<br>PC${ci + 1}: %{y:.2f}<extra></extra>`,
            }], {
                margin: { t: ci === 0 ? 8 : 0, r: 20, b: isLast ? 30 : 2, l: 60 },
                plot_bgcolor: '#fff', paper_bgcolor: '#fff',
                showlegend: false,
                shapes: eventShapes,
                xaxis: {
                    showticklabels: isLast,
                    title: isLast ? { text: 'Time (s)', font: { size: 11 } } : undefined,
                    showgrid: true, gridcolor: '#eee', zeroline: false,
                },
                yaxis: {
                    title: { text: `PC${ci + 1}${pct}`, font: { size: 11 } },
                    showgrid: true, gridcolor: '#eee',
                    zeroline: true, zerolinecolor: '#ccc',
                },
            }, { responsive: false, displayModeBar: false });
        }

        // Sync x-axis zoom across PC time plots within this trial
        timeDivIds.forEach(srcId => {
            const el = document.getElementById(srcId);
            if (!el) return;
            el.on('plotly_relayout', ev_r => {
                const r0 = ev_r['xaxis.range[0]'], r1 = ev_r['xaxis.range[1]'];
                timeDivIds.forEach(id => {
                    if (id === srcId) return;
                    const t = document.getElementById(id);
                    if (t && t._fullLayout) {
                        if (r0 != null && r1 != null) {
                            Plotly.relayout(t, { 'xaxis.range': [r0, r1] });
                        } else if (ev_r['xaxis.autorange']) {
                            Plotly.relayout(t, { 'xaxis.autorange': true });
                        }
                    }
                });
            });
        });

        // --- FFT plots (one per PC, all on one line) ---
        for (let ci = 0; ci < nShow; ci++) {
            if (!_pcOn(ci)) continue;
            const divId = fftDivIds[ci];
            if (!fft_freqs || fft_freqs.length === 0 || !fft_power?.[ci]) {
                const d = document.getElementById(divId);
                if (d) d.innerHTML = '<div class="results-no-data" style="font-size:12px;color:#999;padding:8px;">FFT not available</div>';
                continue;
            }
            const pct = ev[ci] != null ? ` (${(ev[ci] * 100).toFixed(0)}%)` : '';
            Plotly.newPlot(divId, [{
                x: fft_freqs, y: _smoothFft(fft_power[ci]),
                type: 'scatter', mode: 'lines',
                line: { color: PC_COLORS[ci], width: 1.5 },
                hovertemplate: `%{x:.1f} Hz<br>%{y:.2e}<extra>PC${ci + 1}</extra>`,
            }], {
                margin: { t: 22, r: 10, b: 38, l: 55 },
                height: 180,
                plot_bgcolor: '#fff', paper_bgcolor: '#fff',
                showlegend: false,
                title: { text: `PC${ci + 1}${pct}`, font: { size: 12 }, x: 0.5, y: 0.97 },
                xaxis: {
                    title: { text: 'Frequency (Hz)', font: { size: 11 } },
                    showgrid: true, gridcolor: '#eee', zeroline: false,
                },
                yaxis: {
                    title: { text: 'Power', font: { size: 11 } },
                    type: 'log', showgrid: true, gridcolor: '#eee',
                },
                // Reference line at 10 Hz (physiologic tremor band)
                shapes: [{ type: 'line', xref: 'x', yref: 'paper',
                    x0: 10, x1: 10, y0: 0, y1: 1,
                    line: { color: '#bbb', width: 1, dash: 'dash' } }],
                annotations: [{ x: 10, y: 1, xref: 'x', yref: 'paper',
                    text: '10 Hz', showarrow: false,
                    font: { size: 10, color: '#999' },
                    xanchor: 'left', yanchor: 'top', xshift: 3 }],
            }, { responsive: true, displayModeBar: false });
        }
    });
}


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
        // Load traces, movements, and saved events in parallel.
        // Saved events power the Pause overlay (pauses aren't tied to
        // a movement object the way Open/Close are).
        const [traceData, movData, evData] = await Promise.all([
            API.get(`/api/results/${subjectId}/traces?source=${selectedSource}`),
            API.get(`/api/results/${subjectId}/movements?source=${selectedSource}`).catch(() => null),
            API.get(`/api/skeleton/${subjectId}/events`).catch(() => null),
        ]);

        cachedTraces = traceData;
        _savedEvents = evData || null;
        _resetYOverrides();   // new subject/source → fresh full-range sliders

        // Update source selector with available sources
        _updateSourceSelector(traceData.available_sources || [], traceData.data_source);

        if (!traceData.trials || traceData.trials.length === 0) {
            container.innerHTML = '<div class="results-no-data">No distance data available for this subject</div>';
            if (movControls) movControls.style.display = 'none';
            return;
        }

        if (_resultsViewMode !== 'pca') {
            renderAllDistancePlots();
        }

        // Render movement parameters below distance traces (distances mode only)
        if (movData && movData.movements && movData.movements.length > 0) {
            cachedMovements = movData;
            cachedSequenceAssignments = null;
            if (_resultsViewMode !== 'pca') {
                if (movControls) movControls.style.display = '';
                renderIntervalParamPlots();
                renderDistMovementPlots();
                renderShapeOverlayPlots();
            }
        } else {
            if (movControls) movControls.style.display = 'none';
            if (movContainer) movContainer.innerHTML = '';
            const ipCtl = document.getElementById('ipPlotControls');
            const ipC = document.getElementById('ipPlots');
            if (ipCtl) ipCtl.style.display = 'none';
            if (ipC) ipC.innerHTML = '';
            const shapeSec = document.getElementById('shapeOverlaySection');
            if (shapeSec) shapeSec.style.display = 'none';
        }

        // PCA mode: load/render fingertip PCA (movements already cached above for events)
        if (_resultsViewMode === 'pca') {
            _syncViewMode();
        }
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="color:#d32f2f;">${e.message}</div>`;
    }
}

function getDistCheckedParams() {
    const checks = document.querySelectorAll('#distMovementControls input[data-dparam]');
    const params = [];
    checks.forEach(cb => { if (cb.checked) params.push(cb.dataset.dparam); });
    // Top-bar IMI ref = 'off' globally hides every IMI plot.
    const imiRef = document.querySelector('input[name="imiRef"]:checked')?.value;
    if (imiRef === 'off') return params.filter(p => p !== 'imi');
    return params;
}

// One scatter plot per trial: chosen interval (x) vs chosen
// movement parameter (y).  Each dot is one movement.  Sequence
// assignments do NOT apply here — every movement is plotted with
// the same marker style.  Driven by the ipXSelect / ipYSelect
// dropdowns in the controls bar.
function renderIntervalParamPlots() {
    const data = cachedMovements;
    const controls = document.getElementById('ipPlotControls');
    const container = document.getElementById('ipPlots');
    if (!container) return;
    container.innerHTML = '';
    if (!data || !data.movements || !data.movements.length) {
        if (controls) controls.style.display = 'none';
        return;
    }
    if (controls) controls.style.display = 'flex';

    const xKey = document.getElementById('ipXSelect')?.value || 'pp';
    const yParam = document.getElementById('ipYSelect')?.value || 'amplitude';
    const keys = _imiKeys(xKey);
    if (!keys) return;
    const frameMeta = _trialFrameMeta();
    const trialNames = data.trial_names || [];

    // Group movements by trial (preserve per-trial insertion order).
    const byTrial = {};
    data.movements.forEach(m => {
        const ti = m.trial_idx;
        (byTrial[ti] = byTrial[ti] || []).push(m);
    });
    const trialKeys = Object.keys(byTrial).sort((a, b) => +a - +b);

    // Two passes: compute per-trial (xs, ys) once, then derive the
    // SHARED axis ranges across every trial so the scatters can be
    // compared on the same scale.  Empty trials still get an
    // empty plot with the shared axes for layout consistency.
    const perTrial = trialKeys.map(ti => {
        const ms = byTrial[ti];
        const meta = frameMeta[ti] || { fps: 60 };
        const xs = [], ys = [];
        ms.forEach((m, i) => {
            // Intra-movement (O-P, P-C): same movement, every index;
            // inter-movement: needs prev movement, skip i=0.
            let dt;
            if (keys.intra) {
                const from = m[keys.from], to = m[keys.to];
                if (from == null || to == null) return;
                if (!isFinite(from) || !isFinite(to)) return;
                dt = (to - from) / (meta.fps || 60);
            } else {
                if (i === 0) return;
                const prev = ms[i - 1][keys.from];
                const cur = m[keys.to];
                if (prev == null || cur == null) return;
                if (!isFinite(prev) || !isFinite(cur)) return;
                dt = (cur - prev) / (meta.fps || 60);
            }
            if (!(dt > 0)) return;
            // Relative amplitude = m[i].amplitude / m[i-1].amplitude.
            // For every other param, just read the field directly.
            let yv;
            if (yParam === 'relative_amplitude') {
                const a = m.amplitude, ap = ms[i - 1]?.amplitude;
                if (a == null || ap == null || !isFinite(a) || !isFinite(ap) || ap === 0) return;
                yv = a / ap;
            } else {
                yv = m[yParam];
                if (yv == null || !isFinite(yv)) return;
            }
            xs.push(+dt.toFixed(4));
            ys.push(yv);
        });
        return { ti, xs, ys };
    });
    // 5 % pad on each side so points don't sit on the axis lines.
    const _pad = (lo, hi) => {
        if (!isFinite(lo) || !isFinite(hi)) return null;
        if (lo === hi) { const e = Math.abs(lo) * 0.05 || 1; return [lo - e, hi + e]; }
        const p = (hi - lo) * 0.05;
        return [lo - p, hi + p];
    };
    const allXs = perTrial.flatMap(t => t.xs);
    const allYs = perTrial.flatMap(t => t.ys);
    const xRange = allXs.length ? _pad(Math.min(...allXs), Math.max(...allXs)) : null;
    const yRange = allYs.length ? _pad(Math.min(...allYs), Math.max(...allYs)) : null;

    perTrial.forEach(({ ti, xs, ys }) => {
        const block = document.createElement('div');
        block.style.cssText = 'flex:0 0 auto;display:flex;flex-direction:column;';
        const head = document.createElement('div');
        const trialName = (trialNames[ti] || '').toString();
        head.textContent = `Trial: ${trialName.split('_').pop() || ti}`;
        head.style.cssText = 'font-size:12px;color:#666;font-weight:600;padding:2px 8px;';
        block.appendChild(head);

        const plotDiv = document.createElement('div');
        plotDiv.id = `ipPlot_${ti}`;
        plotDiv.style.cssText = 'height:260px;width:320px;';
        block.appendChild(plotDiv);
        container.appendChild(block);

        const xLabel = `${xKey.toUpperCase()} (s)`;
        const yLabel = _ipYLabel(yParam);
        const traces = [{
            x: xs, y: ys,
            type: 'scatter', mode: 'markers',
            marker: { color: MOVEMENT_DOT_COLOR, size: 6,
                      line: { color: '#fff', width: 0.5 } },
            hovertemplate: `${xLabel.replace(' (s)', '')}: %{x:.3f}s<br>${yLabel.split(' (')[0]}: %{y:.2f}<extra></extra>`,
        }];
        const annotations = [];
        // When the Slope toggle is on, append a best-fit line trace
        // and a slope / R² / p annotation pinned to the upper-left of
        // each per-trial plot.
        if (_ipSlopeOn && xs.length >= 3) {
            const stats = _linRegStatsP(xs, ys);
            if (stats) {
                const lo = Math.min(...xs), hi = Math.max(...xs);
                traces.push({
                    x: [lo, hi],
                    y: [stats.intercept + stats.slope * lo,
                        stats.intercept + stats.slope * hi],
                    type: 'scatter', mode: 'lines',
                    line: { color: '#c62828', width: 1.5, dash: 'solid' },
                    hoverinfo: 'skip', showlegend: false,
                });
                const pStr = (stats.p < 1e-4)
                    ? stats.p.toExponential(2)
                    : stats.p.toFixed(4);
                annotations.push({
                    xref: 'paper', yref: 'paper', x: 0.02, y: 0.98,
                    xanchor: 'left', yanchor: 'top',
                    align: 'left', showarrow: false,
                    text: `slope = ${stats.slope.toPrecision(3)}<br>`
                        + `R² = ${stats.r2.toFixed(3)}<br>`
                        + `p = ${pStr}<br>n = ${stats.n}`,
                    font: { size: 10, color: '#333' },
                    bgcolor: 'rgba(255,255,255,0.85)',
                    bordercolor: 'rgba(0,0,0,0.15)', borderwidth: 1,
                    borderpad: 4,
                });
            }
        }
        Plotly.newPlot(plotDiv.id, traces, {
            margin: { t: 8, b: 36, l: 52, r: 12 },
            xaxis: { title: { text: xLabel, font: { size: 10 } },
                     gridcolor: '#eee', tickfont: { size: 9 },
                     range: xRange ? xRange.slice() : undefined,
                     autorange: xRange ? false : true },
            yaxis: { title: { text: yLabel, font: { size: 10 } },
                     gridcolor: '#eee', tickfont: { size: 9 },
                     range: yRange ? yRange.slice() : undefined,
                     autorange: yRange ? false : true },
            plot_bgcolor: '#fff', paper_bgcolor: '#fff',
            showlegend: false, hovermode: 'closest', dragmode: false,
            width: 320, height: 260, annotations,
        }, { displayModeBar: false, responsive: false });
    });
}

// Linear regression with R² and t-distribution p-value — port of
// _linRegStats / _studentT2Tail / _betai / _betacf / _lnGamma from
// explore.js so the Slope toggle on the interval scatters can
// surface the same slope / R² / p numbers as the Explore page.
function _linRegStatsP(xs, ys) {
    const n = xs.length;
    if (n < 3) return null;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
    mx /= n; my /= n;
    let sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
        const dx = xs[i] - mx, dy = ys[i] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    if (sxx <= 0 || syy <= 0) return null;
    const slope = sxy / sxx;
    const intercept = my - slope * mx;
    const r = sxy / Math.sqrt(sxx * syy);
    const r2 = r * r;
    const df = n - 2;
    const denom = Math.max(1 - r2, 1e-300);
    const t = r * Math.sqrt(df / denom);
    return { slope, intercept, r2, t, df, n, p: _ipStudentT2Tail(t, df) };
}
function _ipStudentT2Tail(t, df) {
    const x = df / (df + t * t);
    return _ipBetai(df / 2, 0.5, x);
}
function _ipBetai(a, b, x) {
    if (x <= 0 || x >= 1) return x <= 0 ? 0 : 1;
    const bt = Math.exp(_ipLnGamma(a + b) - _ipLnGamma(a) - _ipLnGamma(b)
                       + a * Math.log(x) + b * Math.log(1 - x));
    return (x < (a + 1) / (a + b + 2))
        ? bt * _ipBetacf(a, b, x) / a
        : 1 - bt * _ipBetacf(b, a, 1 - x) / b;
}
function _ipBetacf(a, b, x) {
    const FPMIN = 1e-300, MAXIT = 200, EPS = 3e-7;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
        const m2 = 2 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
        d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        const del = d * c; h *= del;
        if (Math.abs(del - 1) < EPS) break;
    }
    return h;
}
function _ipLnGamma(x) {
    const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
                 -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
    return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function _ipYLabel(param) {
    return ({
        peak_dist:          'Peak Thumb-Index Distance (mm)',
        amplitude:          'Amplitude (mm)',
        relative_amplitude: 'Amplitude ratio (i / i-1)',
        peak_open_vel:      'Peak Open Vel (mm/s)',
        peak_close_vel:     'Peak Close Vel (mm/s)',
        power:              'Power',
    })[param] || param;
}

// Toggle state for the Slope button on the Interval × Parameter
// scatters.  Off by default; toggling on overlays a per-trial
// best-fit line and a slope / R² / p annotation.
let _ipSlopeOn = false;

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
    // cachedMovements doesn't carry trace data, so read totalSec from
    // cachedTraces -- otherwise the slider's secPerWidth had no
    // denominator to scale against and plotW pinned to containerW.
    let totalSec = 0;
    (cachedTraces?.trials || []).forEach(t => {
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
        // distValid(lo, hi): true iff dist[f] is a finite number for
        // every integer f in [lo, hi].  Used by the per-phase validity
        // checks below — any missing or NaN value disqualifies the
        // movement from similarity analysis under that phase.
        const distValid = (lo, hi) => {
            if (lo == null || hi == null || lo > hi) return false;
            if (lo < 0 || hi >= dist.length) return false;
            for (let f = lo; f <= hi; f++) {
                const v = dist[f];
                if (v == null || !isFinite(v)) return false;
            }
            return true;
        };
        const segments = movs.map((m, origMovIdx) => {
            const o_raw = m.open_frame_local;
            const p_raw = m.peak_frame_local;
            const c_raw = m.close_frame_local;
            const o = (o_raw != null) ? Math.max(0, o_raw | 0) : null;
            const c = (c_raw != null) ? Math.min(dist.length - 1, c_raw | 0) : null;
            const p = (p_raw != null && o !== null && c !== null)
                ? Math.min(c, Math.max(o, p_raw | 0)) : null;
            const xs = [], ys = [];
            if (o !== null && c !== null && o <= c) {
                for (let f = o; f <= c; f++) {
                    const v = dist[f];
                    if (v == null) continue;
                    xs.push((f - o) / fps);   // raw: open at 0
                    ys.push(v);
                }
            }
            const peakT = (p !== null && o !== null) ? (p - o) / fps : 0;
            const closeT = (c !== null && o !== null) ? (c - o) / fps : 0;
            const peakY = (p !== null && dist[p] != null) ? dist[p] : null;
            // Per-phase validity flags.  Order is checked on the
            // original (un-clamped) frame indices so a movement with a
            // peak outside [open, close] is flagged invalid.
            const orderOpen  = (o_raw != null && p_raw != null && o_raw < p_raw);
            const orderClose = (p_raw != null && c_raw != null && p_raw < c_raw);
            const orderWhole = orderOpen && orderClose;
            const validWhole = orderWhole && distValid(o, c);
            const validOpen  = orderOpen  && distValid(o, p);
            const validClose = orderClose && distValid(p, c);
            return { xs, ys, peakT, closeT, peakY,
                     peak_frame: m.peak_frame, origMovIdx,
                     validWhole, validOpen, validClose };
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

// Per-trial highlight = the original movement number (1-indexed,
// 0 = none).  Reset on each renderShapeOverlayPlots() call.
let _shapeHighlight = {};
// Per-trial highlight setter — populated by _buildShapeOverlayCells so
// click handlers in _redrawOneTrial can update the slider in place.
let _shapeSetHighlight = {};
// Per-trial sorted leaf order from the most recent HAC pass.  Null
// (or missing) when "Cluster" is off — the slider then falls back to
// the natural 1..N order.
let _shapeClusterOrder = {};
// Per-trial cluster colors keyed by ORIGINAL movement index.  Set
// when the "Colors" checkbox is on so the shape plot and matrix tick
// labels can paint each cluster distinctly.
let _shapeClusterColors = {};
// Per-trial active-cluster filter (set by clicking a tick label):
//   null/undefined = show all clusters, otherwise = the visible cluster index.
let _shapeClusterFilter = {};
// Categorical palette used to color clusters (max 9 visually distinct
// hues; wraps if there are more clusters).
const _SHAPE_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#bcbd22', '#17becf',
];
function _hexToRgba(hex, alpha) {
    if (!hex || hex[0] !== '#' || hex.length !== 7) return hex;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}
// Per-trial slider DOM refs (slider + value span) so cluster-toggle
// can resync the slider position when the order changes.
let _shapeSliderByIdx = {};

// Position (slider value) → original movement number.  The cluster
// order, when present, stores the original movement indices of the
// VISIBLE / VALID movements for the current phase.
function _posToMov(idx, pos, N) {
    if (!(pos > 0)) return 0;
    const ord = _shapeClusterOrder[idx];
    if (ord && ord.length > 0) {
        if (pos > ord.length) return 0;
        return ord[pos - 1] + 1;
    }
    if (pos > N) pos = N;
    return pos;
}
// Original movement number → slider position; 0 if the movement is
// not in the currently-visible / valid set.
function _movToPos(idx, mov, N) {
    if (!(mov > 0)) return 0;
    const ord = _shapeClusterOrder[idx];
    if (ord && ord.length > 0) {
        const p = ord.indexOf(mov - 1);
        return p >= 0 ? p + 1 : 0;
    }
    if (mov > N) mov = N;
    return mov;
}
// Keep the slider position matching the currently-highlighted movement
// after the cluster order changes.
function _syncShapeSlider(idx) {
    const refs = _shapeSliderByIdx[idx];
    if (!refs || !_shapeData) return;
    const N = _shapeData.trials[idx]?.segments.length || 0;
    // When filtering / phase-dependent validity reduces the visible
    // count, cap the slider's max accordingly so dragging only steps
    // through visible movements.
    const ord = _shapeClusterOrder[idx];
    const maxPos = (ord && ord.length > 0) ? ord.length : N;
    refs.slider.max = String(maxPos);
    const mov = _shapeHighlight[idx] || 0;
    refs.slider.value = String(_movToPos(idx, mov, N));
    refs.valSpan.textContent = mov === 0 ? 'All' : String(mov);
}

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

    // Configure x-scale slider.  We always preserve the user's
    // current value across subject changes — only initialize to the
    // per-subject default on the very first load (when the slider
    // hasn't been seeded yet).  Cap is updated for this subject and
    // the value is clamped if it exceeds the new cap.
    const xSl = document.getElementById('shapeXScaleSlider');
    const xVal = document.getElementById('shapeXScaleVal');
    if (xSl) {
        const xMaxCap = Math.max(5, Math.ceil(_shapeData.xMaxDefault * 1.5));
        xSl.max = String(xMaxCap);
        if (!xSl.dataset.seeded) {
            xSl.value = String(_shapeData.xMaxDefault.toFixed(2));
            xSl.dataset.seeded = '1';
        } else {
            const v = parseFloat(xSl.value);
            if (!isFinite(v) || v <= 0) {
                xSl.value = String(_shapeData.xMaxDefault.toFixed(2));
            } else if (v > xMaxCap) {
                xSl.value = String(xMaxCap);
            }
        }
        if (xVal) xVal.textContent = `${(+xSl.value).toFixed(2)} s`;
    }
    _shapeHighlight = {};
    _shapeClusterOrder = {};
    _shapeClusterColors = {};
    _shapeClusterFilter = {};
    _shapeSliderByIdx = {};

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
        cell.style.cssText = 'flex:0 0 auto;width:527px;min-width:0;';

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

        // Highlight is keyed by ORIGINAL movement number.  The slider
        // value is a POSITION (1..N) which maps to a movement via the
        // current cluster order (when "Cluster" is on) or directly
        // (when off).
        const _setByMov = (m) => {
            const N = trial.segments.length;
            let mov = parseInt(m, 10);
            if (!isFinite(mov)) mov = 0;
            if (mov < 0) mov = 0;
            if (mov > N) mov = N;
            _shapeHighlight[idx] = mov;
            moveSlider.value = String(_movToPos(idx, mov, N));
            moveVal.textContent = mov === 0 ? 'All' : String(mov);
            _redrawOneTrial(idx);
        };
        const _setByPos = (p) => {
            const N = trial.segments.length;
            let pos = parseInt(p, 10);
            if (!isFinite(pos)) pos = 0;
            if (pos < 0) pos = 0;
            const ord = _shapeClusterOrder[idx];
            const cap = (ord && ord.length > 0) ? ord.length : N;
            if (pos > cap) pos = cap;
            _setByMov(_posToMov(idx, pos, N));
        };
        _shapeSetHighlight[idx] = _setByMov;
        _shapeSliderByIdx[idx] = { slider: moveSlider, valSpan: moveVal };
        moveSlider.addEventListener('input', () => _setByPos(moveSlider.value));

        // Prev / next arrow buttons.
        const prevBtn = document.createElement('button');
        prevBtn.type = 'button';
        prevBtn.className = 'btn btn-sm';
        prevBtn.textContent = '◀';
        prevBtn.title = 'Previous movement';
        prevBtn.style.cssText = 'padding:0 6px;font-size:11px;line-height:1.4;';
        prevBtn.addEventListener('click', () => _setByPos(parseInt(moveSlider.value, 10) - 1));
        const nextBtn = document.createElement('button');
        nextBtn.type = 'button';
        nextBtn.className = 'btn btn-sm';
        nextBtn.textContent = '▶';
        nextBtn.title = 'Next movement';
        nextBtn.style.cssText = 'padding:0 6px;font-size:11px;line-height:1.4;';
        nextBtn.addEventListener('click', () => _setByPos(parseInt(moveSlider.value, 10) + 1));
        moveLabel.appendChild(prevBtn);
        moveLabel.appendChild(nextBtn);

        title.appendChild(moveLabel);
        // Small readout for RMSE relative to the trial reference (mean or
        // median).  Populated by _redrawOneTrial each draw.
        const rmseSpan = document.createElement('span');
        rmseSpan.id = `shapeRmse_${idx}`;
        rmseSpan.style.cssText = 'font-size:11px;font-weight:400;color:var(--text-muted);margin-left:auto;';
        title.appendChild(rmseSpan);
        cell.appendChild(title);

        const plotDiv = document.createElement('div');
        plotDiv.id = `shapeOverlayPlot_${idx}`;
        plotDiv.style.cssText = 'width:100%;height:312px;';
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
    const phaseR = document.querySelector('input[name="shapePhase"]:checked');
    const phase = phaseR ? phaseR.value : 'whole';
    let xLo, xHi;
    if (align === 'peak' || align === 'corr') { xLo = -xMax / 2; xHi = xMax / 2; }
    else if (align === 'close') { xLo = -xMax; xHi = 0; }
    else { xLo = 0; xHi = xMax; }
    return { xMax, showMean, align, xLo, xHi, phase };
}

// Slice a segment's xs/ys to the requested phase.  Returns a NEW
// object so callers can swap it into a fresh trial without mutating
// the cached _shapeData segments.
function _phaseSliceSegment(s, phase) {
    if (phase === 'whole' || !phase) return s;
    const xs = [], ys = [];
    const peakT = s.peakT;
    if (phase === 'open') {
        for (let i = 0; i < s.xs.length; i++) {
            if (s.xs[i] <= peakT + 1e-9) { xs.push(s.xs[i]); ys.push(s.ys[i]); }
        }
    } else if (phase === 'close') {
        for (let i = 0; i < s.xs.length; i++) {
            if (s.xs[i] >= peakT - 1e-9) { xs.push(s.xs[i]); ys.push(s.ys[i]); }
        }
    }
    return Object.assign({}, s, { xs, ys });
}

function _drawShapeOverlayPlots() {
    if (!_shapeData) return;
    // First draw of a new subject: build the cells.  Subsequent draws
    // reuse them so the sliders stay alive during interaction.
    const container = document.getElementById('shapeOverlayPlots');
    const haveCells = container && container.children.length === _shapeData.trials.length;
    if (!haveCells) _buildShapeOverlayCells();
    _shapeData.trials.forEach((_t, idx) => _redrawOneTrial(idx));
    // Cluster colors may have changed — refresh the movement-parameter
    // plots and the distance/velocity peak overlays so their markers
    // pick up the new palette.
    if (cachedMovements && typeof renderDistMovementPlots === 'function') {
        renderDistMovementPlots();
    }
    if (cachedTraces && typeof renderAllDistancePlots === 'function') {
        renderAllDistancePlots();
    }
}

function _redrawOneTrial(idx) {
    if (!_shapeData) return;
    let trial = _shapeData.trials[idx];
    if (!trial) return;
    const plotDiv = document.getElementById(`shapeOverlayPlot_${idx}`);
    if (!plotDiv) return;
    const _state = _shapeOverlayState();
    const { xMax, showMean, align, phase } = _state;
    let { xLo, xHi } = _state;

    if (!trial.segments.length) {
        plotDiv.innerHTML = '<div class="results-no-data">No movements</div>';
        return;
    }

    // Restrict similarity analysis to movements that have the
    // required events in order with valid distances between them
    // (depends on phase: Whole = open<peak<close + open→close valid;
    // Open = open<peak + open→peak valid; Close = peak<close +
    // peak→close valid).
    const phaseValidKey = phase === 'open'  ? 'validOpen'
                       : phase === 'close' ? 'validClose'
                                            : 'validWhole';
    let phaseSegs = trial.segments.filter(s => !!s[phaseValidKey]);
    if (phase && phase !== 'whole') {
        phaseSegs = phaseSegs.map(s => _phaseSliceSegment(s, phase));
    }
    trial = Object.assign({}, trial, { segments: phaseSegs });

    if (!trial.segments.length) {
        plotDiv.innerHTML = '<div class="results-no-data">No valid movements</div>';
        return;
    }

    const hiIdx = _shapeHighlight[idx] || 0;

        // Per-segment alignment.  Shift-based aligns (open/peak/close/corr)
        // just translate xs.  "Rigid" piecewise-linearly time-warps each
        // movement so its opening and closing phases match the trial's
        // median durations, putting all opens/peaks/closes on top of
        // each other.
        let corrShifts = null;
        if (align === 'corr') {
            corrShifts = _computeCorrShifts(trial.segments, xMax);
        }
        // Median open-peak and peak-close durations for Rigid.
        let medOp = 0, medPc = 0;
        if (align === 'rigid') {
            const _median = (vs) => {
                const a = vs.filter(v => v > 0 && isFinite(v)).sort((p, q) => p - q);
                if (!a.length) return 0;
                const m = Math.floor(a.length / 2);
                return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
            };
            medOp = _median(trial.segments.map(s => s.peakT));
            medPc = _median(trial.segments.map(s => s.closeT - s.peakT));
            const total = (medOp + medPc) * 1.05;
            xLo = 0;
            xHi = total > 0 ? total : (xMax || 1);
        }
        const shiftOf = (s, si) => align === 'peak' ? -s.peakT
                            : align === 'close' ? -s.closeT
                            : align === 'corr' ? (corrShifts ? corrShifts[si] : 0)
                            : 0;
        const warpT = (s, si, t) => {
            if (align !== 'rigid') return t + shiftOf(s, si);
            if (t <= 0) return 0;
            const pT = s.peakT, cT = s.closeT;
            if (t <= pT) return pT > 0 ? t * (medOp / pT) : medOp;
            const pc = cT - pT;
            return pc > 0 ? medOp + (t - pT) * (medPc / pc) : medOp + medPc;
        };
        const shiftedX = (s, si) => s.xs.map(x => warpT(s, si, x));

        // Cluster pre-compute (shared with the corr-matrix draw).
        // Use the already-shifted data so the clustering reflects
        // exactly what the matrix below will compute.
        const _metricR = document.querySelector('input[name="shapeMetric"]:checked');
        const _metric = _metricR ? _metricR.value : 'corr';
        const _subAvg = !!document.getElementById('shapeSubtractAvg')?.checked;
        const N = trial.segments.length;
        const _matNow = (N >= 2)
            ? _pairwiseMatrix(trial, xLo, xHi, shiftedX, _metric, _subAvg)
            : null;
        const _cutH = parseFloat(document.getElementById('shapeClusterCutoff')?.value);
        const _useCut = isFinite(_cutH) ? _cutH : 0.5;
        const _colorsOn = !!document.getElementById('shapeClusterColors')?.checked;
        let _movColors = null;
        if (_matNow) {
            // Distance: 1 - r for correlation, or covariance rescaled
            // to [0, 2] so the same cutoff slider applies in both
            // modes (largest covariance ↦ distance 0, smallest ↦ 2).
            let _dist;
            if (_metric === 'cov') {
                let _lo = +Infinity, _hi = -Infinity;
                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < N; j++) {
                        if (i === j) continue;
                        const v = _matNow[i][j];
                        if (v == null || !isFinite(v)) continue;
                        if (v < _lo) _lo = v;
                        if (v > _hi) _hi = v;
                    }
                }
                const _range = Math.max(1e-12, _hi - _lo);
                _dist = _matNow.map((row, i) => row.map((v, j) => {
                    if (i === j) return 0;
                    if (v == null || !isFinite(v)) return 2;
                    return 2 * (_hi - v) / _range;
                }));
            } else {
                _dist = _matNow.map(row => row.map(v => v == null ? 2 : 1 - v));
            }
            const _root = _hacAverage(_dist);
            if (_root) {
                const _cut = _cutAndOrderHAC(_root, _useCut);
                const _clusterOf = new Array(N).fill(0);
                let _accC = 0;
                for (let kk = 0; kk < _cut.sizes.length; kk++) {
                    for (let j = 0; j < _cut.sizes[kk]; j++) {
                        _clusterOf[_cut.order[_accC + j]] = kk;
                    }
                    _accC += _cut.sizes[kk];
                }
                if (_colorsOn) {
                    _movColors = _clusterOf.map(c => _SHAPE_PALETTE[c % _SHAPE_PALETTE.length]);
                }
                trial._hacCache = {
                    mat: _matNow, root: _root,
                    order: _cut.order, sizes: _cut.sizes,
                    useCut: _useCut, movColors: _movColors,
                    metric: _metric, clusterOf: _clusterOf,
                };
            } else {
                trial._hacCache = null;
            }
        } else {
            trial._hacCache = null;
        }
        // Externally _shapeClusterColors[idx] is a peak_frame → color
        // map so other plots (movement-parameter scatter, distance /
        // velocity peak overlays) can look up the color for any
        // movement by its global peak_frame.  Local code below still
        // uses the _movColors array indexed by the (filtered) segment
        // position `mi`.
        if (_movColors) {
            const _pfMap = {};
            for (let mi = 0; mi < trial.segments.length; mi++) {
                const pf = trial.segments[mi].peak_frame;
                if (pf != null && _movColors[mi]) _pfMap[pf] = _movColors[mi];
            }
            _shapeClusterColors[idx] = _pfMap;
        } else {
            _shapeClusterColors[idx] = null;
        }

        // Dim the rest when a movement is highlighted.
        const dimmed = hiIdx > 0;
        const fallbackBase = dimmed
            ? 'rgba(31,119,180,0.15)'
            : 'rgba(31,119,180,0.45)';
        // Cluster filter: when a tick label was clicked, only show
        // movements in that cluster.  null = show all.
        const activeCluster = _shapeClusterFilter[idx];
        const clusterOfArr = trial._hacCache ? trial._hacCache.clusterOf : null;
        const inFilter = (mi) => {
            if (activeCluster == null) return true;
            if (!clusterOfArr) return true;
            return clusterOfArr[mi] === activeCluster;
        };

        // ── Reference curve + per-segment RMSE ──────────────────
        // Build a per-frame mean (or median) over the visible
        // segments resampled to a shared grid; then each segment's
        // RMSE = sqrt(mean((seg − ref)^2)) over the overlapping
        // valid points.  Used for the readout in the header and for
        // optional "Deviation"-mode coloring.
        const _useMedian = !!document.getElementById('shapeUseMedian')?.checked;
        const _devColorsOn = !!document.getElementById('shapeDeviationColors')?.checked;
        const _refN = 200;
        const _refGridX = (i) => xLo + (i / (_refN - 1)) * (xHi - xLo);
        const _refCurve = new Array(_refN).fill(null);
        const _resampled = new Array(trial.segments.length).fill(null);
        const _inFilterIdx = [];
        for (let mi = 0; mi < trial.segments.length; mi++) {
            if (!inFilter(mi)) continue;
            const s = trial.segments[mi];
            _resampled[mi] = _resampleXY(shiftedX(s, mi), s.ys, xLo, xHi, _refN);
            _inFilterIdx.push(mi);
        }
        if (_inFilterIdx.length >= 2) {
            for (let i = 0; i < _refN; i++) {
                const col = [];
                for (const mi of _inFilterIdx) {
                    const v = _resampled[mi][i];
                    if (isFinite(v)) col.push(v);
                }
                if (col.length >= Math.max(2, _inFilterIdx.length * 0.25)) {
                    if (_useMedian) {
                        col.sort((a, b) => a - b);
                        const m = Math.floor(col.length / 2);
                        _refCurve[i] = col.length % 2 ? col[m] : (col[m - 1] + col[m]) / 2;
                    } else {
                        let sum = 0;
                        for (const v of col) sum += v;
                        _refCurve[i] = sum / col.length;
                    }
                }
            }
        }
        // Per-segment RMSE relative to the reference curve.
        const _rmse = new Array(trial.segments.length).fill(null);
        for (const mi of _inFilterIdx) {
            const r = _resampled[mi];
            let ssum = 0, n = 0;
            for (let i = 0; i < _refN; i++) {
                const v = r[i], ref = _refCurve[i];
                if (!isFinite(v) || ref == null) continue;
                const d = v - ref;
                ssum += d * d;
                n += 1;
            }
            if (n >= 5) _rmse[mi] = Math.sqrt(ssum / n);
        }
        const _rmseVals = _rmse.filter(v => v != null && isFinite(v));
        let _meanRmse = null;
        if (_rmseVals.length) {
            let s = 0;
            for (const v of _rmseVals) s += v;
            _meanRmse = s / _rmseVals.length;
        }
        // Update the per-trial RMSE readout.  Shows mean across all
        // in-filter movements, and — when a movement is highlighted —
        // also that movement's individual RMSE.
        const _rmseEl = document.getElementById(`shapeRmse_${idx}`);
        if (_rmseEl) {
            const refLabel = _useMedian ? 'median' : 'mean';
            const parts = [];
            if (_meanRmse != null) {
                parts.push(`RMSE vs ${refLabel}: <b>${_meanRmse.toFixed(2)} mm</b>`);
            }
            if (hiIdx > 0) {
                const hiSeg = trial.segments.findIndex(s => s.origMovIdx === hiIdx - 1);
                if (hiSeg >= 0 && _rmse[hiSeg] != null) {
                    parts.push(`#${hiIdx}: <b>${_rmse[hiSeg].toFixed(2)} mm</b>`);
                }
            }
            _rmseEl.innerHTML = parts.join('  ·  ');
        }
        // Deviation coloring: green (low) → yellow → red (high).
        // Overrides cluster coloring when both are checked.
        if (_devColorsOn && _rmseVals.length >= 2) {
            const lo = Math.min(..._rmseVals);
            const hi = Math.max(..._rmseVals);
            const rng = Math.max(1e-9, hi - lo);
            const colorAt = (t) => {
                // t in [0,1].  Two-segment ramp: green→yellow→red.
                const tt = Math.max(0, Math.min(1, t));
                let r, g, b;
                if (tt < 0.5) {
                    const u = tt / 0.5;
                    r = Math.round(46 + (255 - 46) * u);
                    g = Math.round(160 + (193 - 160) * u);
                    b = Math.round(67 + (7 - 67) * u);
                } else {
                    const u = (tt - 0.5) / 0.5;
                    r = Math.round(255 + (211 - 255) * u);
                    g = Math.round(193 + (47 - 193) * u);
                    b = Math.round(7 + (47 - 7) * u);
                }
                return `#${[r,g,b].map(x => x.toString(16).padStart(2,'0')).join('')}`;
            };
            _movColors = new Array(trial.segments.length).fill(null);
            for (const mi of _inFilterIdx) {
                if (_rmse[mi] == null) continue;
                _movColors[mi] = colorAt((_rmse[mi] - lo) / rng);
            }
            // Also push these into the peak_frame map so peak markers
            // and other plots pick up the same coloring.
            const _pfMap = {};
            for (let mi = 0; mi < trial.segments.length; mi++) {
                const pf = trial.segments[mi].peak_frame;
                if (pf != null && _movColors[mi]) _pfMap[pf] = _movColors[mi];
            }
            _shapeClusterColors[idx] = _pfMap;
        }
        const lineColor = (mi) => {
            if (_movColors && _movColors[mi]) {
                return _hexToRgba(_movColors[mi], dimmed ? 0.25 : 0.75);
            }
            return fallbackBase;
        };

        const traces = trial.segments.map((s, mi) => ({
            x: shiftedX(s, mi), y: s.ys, type: 'scatter', mode: 'lines',
            line: { width: 1, color: lineColor(mi) },
            // customdata holds the ORIGINAL movement index so click
            // handlers resolve to the user-facing movement number.
            customdata: new Array(s.xs.length).fill(
                s.origMovIdx != null ? s.origMovIdx : mi),
            visible: inFilter(mi) ? true : false,
            hoverinfo: 'skip', showlegend: false,
        }));

        // Average trace(s).  When Colors is on AND median is OFF we
        // draw one mean per visible cluster (each in its cluster
        // color).  When median is on we draw a single black median
        // curve (the same _refCurve used for RMSE).  Otherwise a
        // single global mean in black.  Drawn before any highlighted
        // movement so the highlight sits on top.
        if (showMean && _useMedian) {
            const xs = [], ys = [];
            for (let i = 0; i < _refN; i++) {
                if (_refCurve[i] != null) { xs.push(_refGridX(i)); ys.push(_refCurve[i]); }
            }
            if (xs.length >= 2) {
                traces.push({
                    x: xs, y: ys, type: 'scatter', mode: 'lines',
                    line: { width: 3.5, color: '#000' },
                    hoverinfo: 'skip', showlegend: false,
                });
            }
        } else if (showMean) {
            const N = 200;
            const _gridX = (i) => xLo + (i / (N - 1)) * (xHi - xLo);

            // Build the list of {label, indices, color} groups to average.
            const groupBuckets = [];
            if (_movColors && clusterOfArr) {
                const map = new Map();   // cluster idx → bucket
                trial.segments.forEach((_s, mi) => {
                    if (!inFilter(mi)) return;
                    const c = clusterOfArr[mi];
                    if (c == null) return;
                    if (!map.has(c)) {
                        map.set(c, { color: _movColors[mi] || '#000', indices: [] });
                    }
                    map.get(c).indices.push(mi);
                });
                for (const b of map.values()) groupBuckets.push(b);
            } else {
                const indices = [];
                trial.segments.forEach((_s, mi) => { if (inFilter(mi)) indices.push(mi); });
                if (indices.length) groupBuckets.push({ color: '#000', indices });
            }

            for (const bucket of groupBuckets) {
                const sums = new Float64Array(N), counts = new Float64Array(N);
                for (const mi of bucket.indices) {
                    const s = trial.segments[mi];
                    const sx = shiftedX(s, mi);
                    const ys = _resampleXY(sx, s.ys, xLo, xHi, N);
                    for (let i = 0; i < N; i++) {
                        if (isFinite(ys[i])) { sums[i] += ys[i]; counts[i] += 1; }
                    }
                }
                const minCount = Math.max(2, bucket.indices.length * 0.25);
                const meanXs = [], meanYs = [];
                for (let i = 0; i < N; i++) {
                    if (counts[i] >= minCount) {
                        meanXs.push(_gridX(i));
                        meanYs.push(sums[i] / counts[i]);
                    }
                }
                if (meanXs.length >= 2) {
                    traces.push({
                        x: meanXs, y: meanYs, type: 'scatter', mode: 'lines',
                        line: { width: 3.5, color: bucket.color },
                        hoverinfo: 'skip', showlegend: false,
                    });
                }
            }
        }

        // Highlighted movement.  hiIdx is the ORIGINAL movement number;
        // find the segment whose origMovIdx matches.
        const hiSegIdx = hiIdx > 0
            ? trial.segments.findIndex(s => s.origMovIdx === hiIdx - 1)
            : -1;
        if (hiSegIdx >= 0) {
            const s = trial.segments[hiSegIdx];
            traces.push({
                x: shiftedX(s, hiSegIdx), y: s.ys, type: 'scatter', mode: 'lines',
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
                : (hiSegIdx >= 0 ? [hiSegIdx] : []);
            const pxs = [], pys = [], pcs = [], pcd = [];
            for (const mi of markIdxs) {
                if (!inFilter(mi)) continue;
                const s = trial.segments[mi];
                if (s.peakY == null) continue;
                pxs.push(warpT(s, mi, s.peakT));
                pys.push(s.peakY);
                pcs.push(_movColors ? (_movColors[mi] || '#d32f2f') : '#d32f2f');
                pcd.push(s.origMovIdx != null ? s.origMovIdx : mi);
            }
            if (pxs.length) {
                traces.push({
                    x: pxs, y: pys, type: 'scatter', mode: 'markers',
                    marker: { size: 9, color: _movColors ? pcs : '#d32f2f',
                              line: { color: '#000', width: 1 }, symbol: 'circle' },
                    customdata: pcd,
                    hoverinfo: 'skip', showlegend: false,
                });
            }
        }

        const xTitle = align === 'peak'
            ? 'Time from peak (s)'
            : align === 'corr' ? 'Time from correlation peak (s)'
            : align === 'close' ? 'Time from closing (s)'
            : align === 'rigid' ? 'Canonical time (s)'
                                : 'Time from opening (s)';
        // Pin the y-axis to the range that would be used if every
        // movement were plotted, so the axes don't jump when the
        // cluster filter hides some segments.
        let yMin = Infinity, yMax = -Infinity;
        for (const s of trial.segments) {
            for (const v of s.ys) {
                if (!isFinite(v)) continue;
                if (v < yMin) yMin = v;
                if (v > yMax) yMax = v;
            }
        }
        const yRange = (isFinite(yMin) && isFinite(yMax) && yMax > yMin)
            ? [yMin - (yMax - yMin) * 0.05, yMax + (yMax - yMin) * 0.05]
            : null;
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
                ...(yRange ? { range: yRange, autorange: false } : { autorange: true }),
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

// Pairwise similarity between every movement at the currently-
// displayed alignment.  Resamples segments to a 120-pt grid over
// [xLo, xHi] and ignores grid points where either side is NaN.
// `metric` is 'corr' (Pearson r, default) or 'cov' (sample covariance).
function _pairwiseMatrix(trial, xLo, xHi, shiftedX, metric, subtractMean) {
    const N = trial.segments.length;
    const GRID = 120;
    const resampled = trial.segments.map((s, mi) =>
        _resampleXY(shiftedX(s, mi), s.ys, xLo, xHi, GRID));
    // Optionally regress the global mean out of each time series.
    // For each segment we fit y_i ≈ α + β·mean (least squares, using
    // only grid points where both y_i and the mean are defined), then
    // keep the residual y_i − α − β·mean.  Correlations of the
    // residuals then capture shape variation independent of the
    // shared "average movement".
    let arrs = resampled;
    if (subtractMean) {
        const meanArr = new Float64Array(GRID);
        const counts = new Float64Array(GRID);
        for (const a of resampled) {
            for (let k = 0; k < GRID; k++) {
                if (isFinite(a[k])) { meanArr[k] += a[k]; counts[k] += 1; }
            }
        }
        for (let k = 0; k < GRID; k++) meanArr[k] = counts[k] > 0 ? meanArr[k] / counts[k] : NaN;

        arrs = resampled.map(a => {
            // Centered statistics over the overlap of finite samples.
            let n = 0, sy = 0, sm = 0, sym = 0, smm = 0;
            for (let k = 0; k < GRID; k++) {
                if (isFinite(a[k]) && isFinite(meanArr[k])) {
                    n += 1;
                    sy += a[k]; sm += meanArr[k];
                    sym += a[k] * meanArr[k];
                    smm += meanArr[k] * meanArr[k];
                }
            }
            const out = new Float64Array(GRID);
            if (n < 2) {
                for (let k = 0; k < GRID; k++) out[k] = NaN;
                return out;
            }
            const my = sy / n, mm = sm / n;
            const varM = smm - n * mm * mm;
            const covYM = sym - n * my * mm;
            const beta = (Math.abs(varM) > 1e-12) ? covYM / varM : 0;
            const alpha = my - beta * mm;
            for (let k = 0; k < GRID; k++) {
                out[k] = (isFinite(a[k]) && isFinite(meanArr[k]))
                    ? a[k] - alpha - beta * meanArr[k]
                    : NaN;
            }
            return out;
        });
    }
    const mat = [];
    for (let i = 0; i < N; i++) {
        const row = new Array(N);
        for (let j = 0; j < N; j++) {
            if (j < i) { row[j] = mat[j][i]; continue; }
            const a = arrs[i], b = arrs[j];
            let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            for (let k = 0; k < GRID; k++) {
                const u = a[k], v = b[k];
                if (!isFinite(u) || !isFinite(v)) continue;
                sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v; n += 1;
            }
            if (n < 5) { row[j] = null; continue; }
            const mx = sx / n, my = sy / n;
            const sxxC = sxx - n * mx * mx;
            const syyC = syy - n * my * my;
            const sxyC = sxy - n * mx * my;
            if (metric === 'cov') {
                row[j] = (n > 1) ? sxyC / (n - 1) : 0;
            } else {
                if (i === j) { row[j] = 1; continue; }
                row[j] = sxyC / Math.sqrt(Math.max(1e-12, sxxC * syyC));
            }
        }
        mat.push(row);
    }
    return mat;
}
// Backwards-compatible alias.
const _pairwiseCorrMatrix = (trial, xLo, xHi, shiftedX) =>
    _pairwiseMatrix(trial, xLo, xHi, shiftedX, 'corr');

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
// Annotations standing in for the row/column tick labels when each
// movement is colored by cluster — text on a filled rectangle whose
// fill is the cluster color.
function _buildTickColorAnnotations(N, labels, tickColors, tickClusters) {
    const out = [];
    const click = !!tickClusters;
    for (let i = 0; i < N; i++) {
        const col = tickColors[i] || '#ddd';
        out.push({
            xref: 'x2', yref: 'y',
            x: -0.5, y: i,
            xanchor: 'right', yanchor: 'middle',
            text: ` ${labels[i]} `, showarrow: false,
            font: { size: 9, color: '#000' },
            bgcolor: col, bordercolor: col, borderwidth: 0, borderpad: 1,
            xshift: -2,
            captureevents: click,
        });
        out.push({
            xref: 'x2', yref: 'y',
            x: i, y: N - 0.5,
            xanchor: 'center', yanchor: 'top',
            text: ` ${labels[i]} `, showarrow: false,
            font: { size: 9, color: '#000' },
            bgcolor: col, bordercolor: col, borderwidth: 0, borderpad: 1,
            yshift: -2,
            captureevents: click,
        });
    }
    return out;
}

function _renderClusteredCorrHeatmap(targetDiv, mat, labels, titleText, hiIdx, hiPos, boundaries, dendroLines, maxH, cutH, tickColors, zRange, metric, tickClusters) {
    if (!zRange) zRange = { min: -1, max: 1, mid: 0 };
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
        zmin: zRange.min, zmax: zRange.max, zmid: zRange.mid,
        colorscale: [
            [0.0, 'rgb(33,102,172)'], [0.25, 'rgb(146,197,222)'],
            [0.5, 'rgb(247,247,247)'],
            [0.75, 'rgb(244,165,130)'], [1.0, 'rgb(178,24,43)'],
        ],
        hovertemplate: (metric === 'cov' ? 'cov = %{z:.3g}' : 'r = %{z:.2f}') + '<extra></extra>',
        // Colorbar length is set later from the actual matrix height.
        colorbar: { thickness: 8, lenmode: 'fraction',
                    y: 1, yanchor: 'top', outlinewidth: 0, ypad: 0,
                    tickvals: [zRange.min, zRange.mid, zRange.max],
                    tickformat: (metric === 'cov' ? '.2g' : '.2f'),
                    tickfont: { size: 10 } },
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
    const MARGIN = { t: 28, b: 14, l: 8, r: 50 };
    const layout = {
        margin: MARGIN,
        title: { text: titleText, font: { size: 11 }, x: 0, xanchor: 'left', y: 0.98 },
        xaxis: {
            domain: [0, 0.21],
            range: [xMaxDendro, 0],
            showticklabels: false, zeroline: false, showgrid: false,
            showline: false, ticks: '', fixedrange: true, automargin: false,
        },
        xaxis2: {
            domain: [0.26, 1.0], anchor: 'y', automargin: false,
            tickfont: { size: 9 }, side: 'bottom',
            tickmode: 'array', tickvals: nums, ticktext: labels,
            range: [-0.5, N - 0.5],
            showline: false, showgrid: false, zeroline: false, ticks: '',
            showticklabels: !tickColors,
        },
        // constrain on yaxis (not x2) so the y-domain shrinks to match
        // the square matrix; constraintoward:'top' keeps the matrix at
        // the top of the plot area so the bottom x-axis labels sit
        // right under the matrix instead of at the plot-area bottom.
        yaxis: {
            tickfont: { size: 9 }, automargin: false, anchor: 'x2',
            tickmode: 'array', tickvals: nums, ticktext: labels,
            range: [N - 0.5, -0.5],
            scaleanchor: 'x2', scaleratio: 1,
            constrain: 'domain', constraintoward: 'top',
            showline: false, showgrid: false, zeroline: false, ticks: '',
            showticklabels: !tickColors,
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        shapes, showlegend: false,
        annotations: (tickColors ? _buildTickColorAnnotations(N, labels, tickColors, tickClusters) : []),
    };
    // Cluster index per annotation, in the order returned by
    // _buildTickColorAnnotations (y-tick then x-tick per movement).
    if (tickColors && tickClusters) {
        const meta = [];
        for (let i = 0; i < N; i++) { meta.push(tickClusters[i]); meta.push(tickClusters[i]); }
        targetDiv._annotClusters = meta;
    } else {
        targetDiv._annotClusters = null;
    }
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
        const x2Frac = 1.0 - 0.26;   // matrix domain [0.26, 1.0]
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
        const matrixSide0 = Math.min((1 - 0.26) * plotW0, plotH0);
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
    // Click on a tick-label annotation (Colors mode) → toggle the
    // cluster filter on the shape-overlay plot.
    if (!targetDiv._annotClickBound) {
        targetDiv._annotClickBound = true;
        targetDiv.on('plotly_clickannotation', (ev) => {
            const meta = targetDiv._annotClusters;
            const ai = ev?.index;
            if (!meta || typeof ai !== 'number') return;
            const cluster = meta[ai];
            if (cluster == null) return;
            const m = /shapeCorrPlot_(\d+)/.exec(targetDiv.id || '');
            if (!m) return;
            const tIdx = parseInt(m[1], 10);
            // Toggle: same cluster (or any label in it) → show all; new cluster → switch.
            if (_shapeClusterFilter[tIdx] === cluster) {
                _shapeClusterFilter[tIdx] = null;
            } else {
                _shapeClusterFilter[tIdx] = cluster;
            }
            _redrawOneTrial(tIdx);
        });
    }
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

    // Reuse the HAC cache produced by _redrawOneTrial when possible.
    const cache = trial._hacCache;
    const metric = cache ? cache.metric : 'corr';
    const subAvg = !!document.getElementById('shapeSubtractAvg')?.checked;
    const mat = cache ? cache.mat : _pairwiseMatrix(trial, xLo, xHi, shiftedX, metric, subAvg);
    // Labels use the ORIGINAL movement number (origMovIdx + 1) so a
    // valid segment shown at position 0 in a filtered trial still
    // reads as e.g. movement #7 if movements #1–6 were excluded.
    const labels = trial.segments.map((s, i) =>
        String((s.origMovIdx != null ? s.origMovIdx : i) + 1));
    const clusterOn = !!document.getElementById('shapeClusterOn')?.checked;
    const movColors = cache ? cache.movColors : null;
    // Color-scale bounds: fixed [-1, 1] for correlation; symmetric
    // around 0 with absMax for covariance.
    let zRange = { min: -1, max: 1, mid: 0 };
    if (metric === 'cov') {
        let absMax = 0;
        for (let i = 0; i < mat.length; i++) {
            for (let j = 0; j < mat.length; j++) {
                if (i === j) continue;
                const v = mat[i][j];
                if (v == null || !isFinite(v)) continue;
                const a = Math.abs(v);
                if (a > absMax) absMax = a;
            }
        }
        if (absMax < 1e-12) absMax = 1;
        zRange = { min: -absMax, max: absMax, mid: 0 };
    }
    const titlePrefix = metric === 'cov' ? 'Pairwise covariance' : 'Pairwise correlation';
    const titleClu = (k) => metric === 'cov'
        ? `Clustered (HAC, avg linkage, cov) — ${k} group${k === 1 ? '' : 's'}`
        : `Clustered (HAC, avg linkage, 1−r) — ${k} group${k === 1 ? '' : 's'}`;

    if (!clusterOn) {
        _shapeClusterOrder[idx] = null;
        const tickColors = movColors ? movColors.slice() : null;
        const tickClusters = (movColors && cache && cache.clusterOf)
            ? cache.clusterOf.slice() : null;
        _renderClusteredCorrHeatmap(
            corrDiv, mat, labels, titlePrefix,
            hiIdx, hiIdx > 0 ? hiIdx - 1 : -1,
            null, [], 1, null, tickColors, zRange, metric, tickClusters,
        );
        _syncShapeSlider(idx);
        return;
    }

    // Clustered: reuse cached HAC tree / order if present.
    let root, order, sizes, useCut;
    if (cache) {
        root = cache.root; order = cache.order; sizes = cache.sizes;
        useCut = cache.useCut;
    } else {
        const dist = mat.map(row => row.map(v => (v == null ? 2 : 1 - v)));
        root = _hacAverage(dist);
        if (!root) { corrDiv.innerHTML = ''; return; }
        const cutH = parseFloat(document.getElementById('shapeClusterCutoff')?.value);
        useCut = isFinite(cutH) ? cutH : 0.5;
        const _cut = _cutAndOrderHAC(root, useCut);
        order = _cut.order; sizes = _cut.sizes;
    }
    const reord = order.map(i => order.map(j => mat[i][j]));
    const reordLabels = order.map(i => {
        const s = trial.segments[i];
        return String((s && s.origMovIdx != null ? s.origMovIdx : i) + 1);
    });
    const boundaries = [];
    let acc = 0;
    for (let i = 0; i < sizes.length - 1; i++) { acc += sizes[i]; boundaries.push(acc); }
    const hiPosClu = hiIdx > 0 ? order.indexOf(hiIdx - 1) : -1;
    const k = sizes.length;
    const leafToY = {};
    order.forEach((id, pos) => { leafToY[id] = pos; });
    const dendroLines = _buildDendroLines(root, leafToY);
    const maxH = root.height || 1;
    const tickColors = movColors ? order.map(i => movColors[i]) : null;
    const tickClusters = (movColors && cache && cache.clusterOf)
        ? order.map(i => cache.clusterOf[i]) : null;
    // Store order as ORIGINAL movement indices (origMovIdx values) so
    // _posToMov / _movToPos translate to the user-facing movement
    // number even when the trial has been filtered.
    _shapeClusterOrder[idx] = order.map(pos => {
        const s = trial.segments[pos];
        return (s && s.origMovIdx != null) ? s.origMovIdx : pos;
    });
    _renderClusteredCorrHeatmap(
        corrDiv, reord, reordLabels, titleClu(k),
        hiIdx, hiPosClu, boundaries, dendroLines, maxH, useCut, tickColors,
        zRange, metric, tickClusters,
    );
    _syncShapeSlider(idx);
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
    document.querySelectorAll('input[name="shapePhase"]').forEach(r =>
        r.addEventListener('change', _drawShapeOverlayPlots));
    const cutSl = document.getElementById('shapeClusterCutoff');
    const cutVal = document.getElementById('shapeClusterCutoffVal');
    if (cutSl) cutSl.addEventListener('input', () => {
        if (cutVal) cutVal.textContent = (+cutSl.value).toFixed(2);
        _drawShapeOverlayPlots();
    });
    const cluCb = document.getElementById('shapeClusterOn');
    if (cluCb) cluCb.addEventListener('change', _drawShapeOverlayPlots);
    const colCb = document.getElementById('shapeClusterColors');
    if (colCb) colCb.addEventListener('change', _drawShapeOverlayPlots);
    const devCb = document.getElementById('shapeDeviationColors');
    if (devCb) devCb.addEventListener('change', _drawShapeOverlayPlots);
    const medCb = document.getElementById('shapeUseMedian');
    if (medCb) medCb.addEventListener('change', _drawShapeOverlayPlots);
    const subCb = document.getElementById('shapeSubtractAvg');
    if (subCb) subCb.addEventListener('change', _drawShapeOverlayPlots);
    document.querySelectorAll('input[name="shapeMetric"]').forEach(r =>
        r.addEventListener('change', _drawShapeOverlayPlots));
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
    const overlayOpen = document.getElementById('overlayOpen')?.checked;
    const overlayClose = document.getElementById('overlayClose')?.checked;
    const overlayPause = document.getElementById('overlayPause')?.checked;
    const overlayPeakOpenVel = document.getElementById('overlayPeakOpenVel').checked;
    const overlayPeakCloseVel = document.getElementById('overlayPeakCloseVel').checked;

    // Per-trial peak_frame → cluster color lookup so the peak-distance
    // and peak-velocity markers below can pick up the shape-overlay
    // colors when the "Colors" checkbox is on.  _shapeClusterColors
    // is already keyed by peak_frame.
    const _colorsOn = !!document.getElementById('shapeClusterColors')?.checked;
    const _colByTrial = {};
    if (_colorsOn && typeof _shapeClusterColors !== 'undefined' && _shapeClusterColors) {
        for (const k in _shapeClusterColors) {
            if (_shapeClusterColors[k]) _colByTrial[k] = _shapeClusterColors[k];
        }
    }
    const _peakColor = (trialIdx, peakFrame, fallback) => {
        const map = _colByTrial[trialIdx];
        return (map && peakFrame != null && map[peakFrame]) ? map[peakFrame] : fallback;
    };
    // Sequence shading on the distance/velocity traces is gated by the
    // "Sequences" checkbox in the overlay row.  When on, we fall back
    // to recomputing assignments from cachedMovements (using the
    // currently-selected distSequenceMode) if nothing is cached yet.
    const showSequences = !!document.getElementById('overlaySequences')?.checked;
    let seqAssignments = null;
    if (showSequences && cachedMovements && cachedMovements.movements) {
        if (!cachedSequenceAssignments) {
            cachedSequenceAssignments = computeSequenceAssignments(cachedMovements);
        }
        seqAssignments = cachedSequenceAssignments;
    }
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
    const DIST_H = 220, DIST_MT = 10, DIST_MB = 0;
    // Compressed IMI strip between distance and velocity plots.
    const IMI_H = 70, IMI_MT = 0, IMI_MB = 0;
    const VEL_H = 150, VEL_MT = 0, VEL_MB = 35;

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

        // Locked header: trial name (clickable to collapse) + Copy button.
        const header = document.createElement('div');
        header.style.gridArea = 'header';
        header.style.display = 'flex';
        header.style.alignItems = 'center';
        header.style.justifyContent = 'space-between';
        header.style.padding = '2px 8px 4px';
        const titleSpan = document.createElement('span');
        const trialKey = trial.name || `trial_${idx}`;
        const isCollapsed = _collapsedTrials.has(trialKey);
        // Chevron in a fixed-width slot so '▾' and '▸' (different
        // glyph widths) don't shift the trial-name text horizontally.
        const chevSpan = document.createElement('span');
        chevSpan.style.cssText = 'display:inline-block;width:12px;text-align:center;';
        const labelSpan = document.createElement('span');
        labelSpan.style.marginLeft = '4px';
        const _renderTitle = () => {
            chevSpan.textContent = _collapsedTrials.has(trialKey) ? '▸' : '▾';
            labelSpan.textContent = `Trial: ${_trialPartOf(trial)}`;
        };
        _renderTitle();
        titleSpan.appendChild(chevSpan);
        titleSpan.appendChild(labelSpan);
        titleSpan.style.cssText = 'font-size:13px;color:#666;font-weight:600;cursor:pointer;user-select:none;';
        titleSpan.title = 'Click to collapse/expand this trial\'s distance + velocity plots';
        titleSpan.addEventListener('click', () => {
            if (_collapsedTrials.has(trialKey)) _collapsedTrials.delete(trialKey);
            else _collapsedTrials.add(trialKey);
            _renderTitle();
            const nowCollapsed = _collapsedTrials.has(trialKey);
            sliderCol.style.display = nowCollapsed ? 'none' : 'flex';
            wrapper.style.display   = nowCollapsed ? 'none' : '';
        });
        const copyBtn = _makeCopyBtn(b => _copyTrialPlots(idx, trial.name || '', b));
        header.appendChild(titleSpan);
        header.appendChild(copyBtn);
        block.appendChild(header);

        // Slider column (does NOT scroll with the plots)
        const sliderCol = document.createElement('div');
        sliderCol.style.gridArea = 'slider';
        sliderCol.style.display = isCollapsed ? 'none' : 'flex';
        sliderCol.style.flexDirection = 'column';
        sliderCol.appendChild(_buildYSliderCol('dist', idx, DIST_H, DIST_MT, DIST_MB));
        // Spacer in the slider column so the velocity slider lines up
        // with the velocity plot — the IMI strip in the wrapper has
        // no Y-range slider of its own.
        const imiSpacer = document.createElement('div');
        imiSpacer.style.height = IMI_H + 'px';
        sliderCol.appendChild(imiSpacer);
        sliderCol.appendChild(_buildYSliderCol('vel', idx, VEL_H, VEL_MT, VEL_MB));
        block.appendChild(sliderCol);

        // Scrolling wrapper holds both plots
        const wrapper = document.createElement('div');
        wrapper.style.gridArea = 'wrapper';
        wrapper.style.overflowX = 'auto';
        wrapper.style.position = 'relative';   // anchor for hover button overlay
        wrapper.dataset.trialIdx = String(idx);
        wrapper.id = `trialWrap_${idx}`;
        if (isCollapsed) wrapper.style.display = 'none';
        // CSS grid would otherwise let the wrapper grow to fit its
        // content; min-width:0 lets the 1fr track stay 1fr.
        wrapper.style.minWidth = '0';

        // Static y-axis column INSIDE the wrapper — position:sticky
        // keeps it pinned to the wrapper's left edge while the plot
        // body scrolls horizontally.  Each kind gets its own narrow
        // plotly plot that draws ONLY the y-axis line, ticks, labels,
        // and title.  Width Y_AXIS_W matches what the data plots used
        // to spend on margin.l before the split.
        const Y_AXIS_W = 60;
        const innerFlex = document.createElement('div');
        innerFlex.style.cssText = 'display:flex;min-width:max-content;align-items:flex-start;';
        const stickyY = document.createElement('div');
        stickyY.style.cssText = `position:sticky;left:0;z-index:5;flex:0 0 ${Y_AXIS_W}px;background:#fff;`;
        const yDistDiv = document.createElement('div');
        yDistDiv.id = `yAxisDist_${idx}`;
        yDistDiv.style.cssText = `width:${Y_AXIS_W}px;height:${DIST_H}px;`;
        stickyY.appendChild(yDistDiv);
        const yImiDiv = document.createElement('div');
        yImiDiv.id = `yAxisImi_${idx}`;
        yImiDiv.style.cssText = `width:${Y_AXIS_W}px;height:${IMI_H}px;`;
        stickyY.appendChild(yImiDiv);
        const yVelDiv = document.createElement('div');
        yVelDiv.id = `yAxisVel_${idx}`;
        yVelDiv.style.cssText = `width:${Y_AXIS_W}px;height:${VEL_H}px;`;
        stickyY.appendChild(yVelDiv);

        const plotCol = document.createElement('div');
        plotCol.style.cssText = 'flex:0 0 auto;';

        // Distance plot
        const distDiv = document.createElement('div');
        distDiv.id = `distPlot_${idx}`;
        distDiv.style.height = DIST_H + 'px';
        plotCol.appendChild(distDiv);

        // Compressed IMI strip between distance and velocity.
        const imiDiv = document.createElement('div');
        imiDiv.id = `imiPlot_${idx}`;
        imiDiv.style.height = IMI_H + 'px';
        plotCol.appendChild(imiDiv);

        // Velocity plot
        const velDiv = document.createElement('div');
        velDiv.id = `velPlot_${idx}`;
        velDiv.style.height = VEL_H + 'px';
        plotCol.appendChild(velDiv);

        innerFlex.appendChild(stickyY);
        innerFlex.appendChild(plotCol);
        wrapper.appendChild(innerFlex);
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
        imiDiv.style.width = plotWidth + 'px';
        velDiv.style.width = plotWidth + 'px';

        const trialStart = data.trials.slice(0, idx).reduce((acc, t) => acc + (t.distances ? t.distances.length : 0), 0);
        const trialMovs = movByTrial[idx] || [];

        // Build overlay traces for distance plot (peak distance markers on the curve)
        const distOverlays = [];
        const distShapes = [];
        if (overlayPeakDist && trialMovs.length > 0) {
            const peakTimes = [], peakVals = [], peakColors = [];
            trialMovs.forEach(m => {
                if (m.peak_dist != null && m.peak_frame != null) {
                    const localFrame = m.peak_frame - trialStart;
                    peakTimes.push(+(localFrame / fps).toFixed(3));
                    peakVals.push(m.peak_dist);
                    peakColors.push(_peakColor(idx, m.peak_frame, '#FF9800'));
                }
            });
            if (peakTimes.length > 0) {
                distOverlays.push({
                    x: peakTimes, y: peakVals,
                    type: 'scatter', mode: 'markers',
                    marker: { color: peakColors, size: 7, symbol: 'diamond' },
                    name: 'Peak Distance',
                    hovertemplate: '%{x:.2f}s<br>Peak: %{y:.1f} mm<extra></extra>',
                });
            }
        }

        // Open / close event markers on the distance trace.  Plotted
        // at the open/close frame's distance value, in green (open)
        // and red (close).
        const _evMarkers = (kind, frameField, color, name) => {
            if (!trialMovs.length) return;
            const xs = [], ys = [];
            trialMovs.forEach(m => {
                const f = m[frameField];
                if (f == null || !isFinite(f)) return;
                const idxLocal = f;       // global frame index
                const local = idxLocal - trialStart;
                if (local < 0) return;
                const fpsT = trial.fps || fps;
                if (!trial.distances) return;
                const v = trial.distances[local];
                if (v == null || !isFinite(v)) return;
                xs.push(+(local / fpsT).toFixed(3));
                ys.push(v);
            });
            if (xs.length) {
                distOverlays.push({
                    x: xs, y: ys, type: 'scatter', mode: 'markers',
                    marker: { color, size: 7, symbol: 'circle' },
                    name, hoverinfo: 'skip', showlegend: false,
                });
            }
        };
        if (overlayOpen) _evMarkers('open', 'open_frame', '#2ca02c', 'Open');
        if (overlayClose) _evMarkers('close', 'close_frame', '#d62728', 'Close');

        // Pause overlay — pauses live in saved events (events.csv),
        // not in the per-movement dicts.  Pause frames are GLOBAL
        // video frame indices; convert with trial.start_frame.
        if (overlayPause && _savedEvents && Array.isArray(_savedEvents.pause)
                && trial.distances && trial.start_frame != null) {
            const sf = trial.start_frame;
            const ef = (trial.end_frame != null)
                ? trial.end_frame
                : (sf + trial.distances.length - 1);
            const fpsT = trial.fps || fps;
            const xs = [], ys = [];
            _savedEvents.pause.forEach(gf => {
                if (!Number.isFinite(gf) || gf < sf || gf > ef) return;
                const local = gf - sf;
                const v = trial.distances[local];
                if (v == null || !isFinite(v)) return;
                xs.push(+(local / fpsT).toFixed(3));
                ys.push(v);
            });
            if (xs.length) {
                distOverlays.push({
                    x: xs, y: ys, type: 'scatter', mode: 'markers',
                    marker: { color: '#cc66ff', size: 7, symbol: 'circle' },
                    name: 'Pause', hoverinfo: 'skip', showlegend: false,
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
            const ts = [], vs = [], cs = [];
            trialMovs.forEach(m => {
                // Position the marker at the opening-velocity peak frame
                // (falls back to the distance-peak frame if absent).
                const vf = m.peak_open_vel_frame != null ? m.peak_open_vel_frame : m.peak_frame;
                if (m.peak_open_vel != null && vf != null) {
                    ts.push(+((vf - trialStart) / fps).toFixed(3));
                    vs.push(m.peak_open_vel);
                    cs.push(_peakColor(idx, m.peak_frame, '#2196F3'));
                }
            });
            if (ts.length > 0) {
                velOverlays.push({
                    x: ts, y: vs,
                    type: 'scatter', mode: 'markers',
                    marker: { color: cs, size: 6, symbol: 'triangle-up' },
                    name: 'Peak Open Vel',
                    hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm/s<extra></extra>',
                });
            }
        }
        if (overlayPeakCloseVel && trialMovs.length > 0) {
            const ts = [], vs = [], cs = [];
            trialMovs.forEach(m => {
                // Position the marker at the closing-velocity peak frame
                // (falls back to the distance-peak frame if absent).
                const vf = m.peak_close_vel_frame != null ? m.peak_close_vel_frame : m.peak_frame;
                if (m.peak_close_vel != null && vf != null) {
                    ts.push(+((vf - trialStart) / fps).toFixed(3));
                    vs.push(m.peak_close_vel);
                    cs.push(_peakColor(idx, m.peak_frame, '#f44336'));
                }
            });
            if (ts.length > 0) {
                velOverlays.push({
                    x: ts, y: vs,
                    type: 'scatter', mode: 'markers',
                    marker: { color: cs, size: 6, symbol: 'triangle-down' },
                    name: 'Peak Close Vel',
                    hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm/s<extra></extra>',
                });
            }
        }

        renderDistancePlot(distDiv.id, trial, _effYRange('dist', idx), plotWidth, distOverlays, distShapes);
        renderYAxisOnly(`yAxisDist_${idx}`, _effYRange('dist', idx), 'dist',
                        DIST_H, DIST_MT, DIST_MB, Y_AXIS_W);
        const _imiRefVal = (document.querySelector('input[name="imiRef"]:checked')?.value) || 'pp';
        const _imiOff = _imiRefVal === 'off';
        imiDiv.style.display = _imiOff ? 'none' : '';
        imiSpacer.style.display = _imiOff ? 'none' : '';
        yImiDiv.style.display = _imiOff ? 'none' : '';
        if (!_imiOff) {
            // IMI plot autoranges its y-axis; render the strip first,
            // then mirror the resulting range onto the sticky y-axis
            // so labels match the data.
            renderIMIPlot(imiDiv.id, trial, trialStart, trialMovs, fps, _imiRefVal, plotWidth)
                .then(() => {
                    const r = imiDiv._fullLayout?.yaxis?.range || null;
                    renderYAxisOnly(`yAxisImi_${idx}`, r, 'imi',
                                    IMI_H, IMI_MT, IMI_MB, Y_AXIS_W);
                });
        }
        renderVelocityPlot(velDiv.id, trial, _effYRange('vel', idx), plotWidth, velOverlays, distShapes);
        renderYAxisOnly(`yAxisVel_${idx}`, _effYRange('vel', idx), 'vel',
                        VEL_H, VEL_MT, VEL_MB, Y_AXIS_W);
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
    const prefix  = kind === 'dist' ? 'distPlot_'  : 'velPlot_';
    const yPrefix = kind === 'dist' ? 'yAxisDist_' : 'yAxisVel_';
    const _relayout = (selector, r) => {
        document.querySelectorAll(selector).forEach(div => {
            if (window.Plotly) {
                try { Plotly.relayout(div, { 'yaxis.range': r }); } catch {}
            }
        });
    };
    if (idx === null) {
        const range = _yLocked[kind];
        const r = [range.min, range.max];
        _relayout(`[id^="${prefix}"]`, r);
        // Mirror to the sticky y-axis plots so their labels match.
        _relayout(`[id^="${yPrefix}"]`, r);
        document.querySelectorAll(`.y-range-wrap[data-kind="${kind}"]`).forEach(w => {
            const mn = w.querySelector('.ysl.min');
            const mx = w.querySelector('.ysl.max');
            if (mn) mn.value = range.min;
            if (mx) mx.value = range.max;
            _updateYFill(w);
        });
    } else {
        const range = _yTrial[kind][idx];
        const r = [range.min, range.max];
        const div = document.getElementById(`${prefix}${idx}`);
        if (div && window.Plotly) { try { Plotly.relayout(div, { 'yaxis.range': r }); } catch {} }
        const yDiv = document.getElementById(`${yPrefix}${idx}`);
        if (yDiv && window.Plotly) { try { Plotly.relayout(yDiv, { 'yaxis.range': r }); } catch {} }
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
        let cv = document.createElement('canvas');
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
        // Optional output-height target: rescale the composite so
        // the clipboard PNG matches a fixed pixel height (e.g. for
        // pasting into a Keynote slide at a known size).  Width is
        // scaled proportionally.
        const targetH = (opts && opts.outputHeight) || 0;
        if (targetH > 0 && cv.height !== targetH) {
            const scale = targetH / cv.height;
            const out = document.createElement('canvas');
            out.width = Math.round(cv.width * scale);
            out.height = targetH;
            const octx = out.getContext('2d');
            octx.imageSmoothingEnabled = true;
            octx.imageSmoothingQuality = 'high';
            octx.fillStyle = '#fff';
            octx.fillRect(0, 0, out.width, out.height);
            octx.drawImage(cv, 0, 0, out.width, out.height);
            cv = out;
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
    // Seq-effect div-id suffix tracks the R²/Slope checkboxes —
    // each metric can be visible alone, both, or neither.
    const seqSuffixes = ['seq_r2', 'seq_slope']
        .filter(s => document.getElementById(prefix + paramId + '_' + s));
    const divs = ['mean', 'cv', ...seqSuffixes]
        .map(f => document.getElementById(prefix + paramId + '_' + f))
        .filter(Boolean);
    const m = (typeof GROUP_METRICS !== 'undefined')
        ? GROUP_METRICS.find(x => x.id === paramId) : null;
    const titleText = m
        ? m.name + (m.mean && m.mean.unit ? ' (' + m.mean.unit + ')' : '')
        : paramId;
    const stem = (kind === 'dose' ? 'levodopa_' : 'group_') + paramId;
    // Keynote slides on this project are 1024x768 with the title
    // banner cropped; 963 px is the target height the user pastes
    // into.  Force the clipboard image to that exact pixel height.
    return _copyPlotsAsPng(divs, stem, btn, { title: titleText, outputHeight: 963 });
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

// Tiny plot whose only job is to draw the y-axis (line + ticks +
// labels + title) for one of the data plots beside it.  Lives in
// the sticky column inside the wrapper so the axis stays pinned to
// the wrapper's left edge while the data plot scrolls horizontally.
function renderYAxisOnly(divId, yRange, kind, height, mTop, mBot, width) {
    const styles = {
        dist: { color: '#2196F3', title: 'Distance (mm)', nticks: undefined },
        imi:  { color: '#9C27B0', title: 'Interval (s)',  nticks: 3 },
        vel:  { color: '#4CAF50', title: 'Velocity (mm/s)', nticks: undefined },
    };
    const s = styles[kind] || styles.dist;
    // A single invisible point seeds plotly's coordinate system.
    // y goes within the configured range; x is irrelevant.
    const seedY = yRange ? (yRange[0] + yRange[1]) / 2 : 0;
    const trace = {
        x: [0], y: [seedY],
        type: 'scatter', mode: 'markers',
        marker: { opacity: 0 }, hoverinfo: 'skip',
    };
    const layout = {
        margin: { t: mTop, b: mBot, l: 52, r: 0 },
        xaxis: { visible: false, range: [0, 1], fixedrange: true },
        yaxis: {
            color: s.color, gridcolor: '#f0f0f0', zeroline: false,
            title: { text: s.title, font: { size: kind === 'imi' ? 10 : 11, color: s.color } },
            side: 'left', fixedrange: true,
            tickfont: { size: kind === 'imi' ? 9 : 10 },
            ...(s.nticks ? { nticks: s.nticks } : {}),
            ...(yRange ? { range: yRange, autorange: false } : { autorange: true }),
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        showlegend: false, dragmode: false,
        width, height,
    };
    return Plotly.newPlot(divId, [trace], layout, {
        responsive: false, displayModeBar: false,
    });
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
        // l = 0 because the y-axis is rendered in a separate
        // sticky-positioned column to the left of the wrapper — see
        // renderYAxisOnly.  The data plot's own y-axis is hidden so
        // it doesn't scroll out of view with the trace.
        margin: { t: 10, b: 5, l: 0, r: 20 },
        // Custom drag-to-mark gesture replaces zoom — see _wireIntervalDrag.
        dragmode: false,
        // Pin the X range to the trace extent so toggling the peak
        // overlays (which adds marker traces) can't rescale the axis.
        xaxis: {
            showticklabels: false, color: '#666', gridcolor: '#eee',
            range: [times[0] || 0, times[n - 1] || 0], autorange: false,
        },
        yaxis: {
            // Title / labels / axis line all live on the sticky
            // y-axis plot to the left; keep just the gridcolor.
            showticklabels: false, ticks: '', showline: false,
            title: { text: '' },
            color: '#2196F3', gridcolor: '#f0f0f0', zeroline: false,
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
    }).then(() => _registerClickPlot(divId));
}

// Compressed IMI strip rendered between the distance and velocity
// plots.  X axis is shared trial-local time; Y is inter-movement
// interval (seconds), measured between consecutive `refKind`_frame
// values (peak / open / close).  Each point is plotted at the
// SECOND frame of the pair so it sits over the spot in time where
// that interval ends.
// Resolve the Interval radio value to a {from, to} frame-field pair.
//   pp/oo/cc → same frame on both sides (P-P, O-O, C-C)
//   po       → peak of movement i → open of movement i+1
//   co       → close of movement i → open of movement i+1
//   off / unknown → null (caller should skip rendering).
function _imiKeys(refVal) {
    // ``intra: true`` means the interval is computed WITHIN a single
    // movement (O-P opening-phase, P-C closing-phase).  Everything
    // else is between consecutive movements.
    switch (refVal) {
        case 'pp': return { from: 'peak_frame',  to: 'peak_frame',  intra: false };
        case 'oo': return { from: 'open_frame',  to: 'open_frame',  intra: false };
        case 'cc': return { from: 'close_frame', to: 'close_frame', intra: false };
        case 'po': return { from: 'peak_frame',  to: 'open_frame',  intra: false };
        case 'co': return { from: 'close_frame', to: 'open_frame',  intra: false };
        case 'op': return { from: 'open_frame',  to: 'peak_frame',  intra: true  };
        case 'pc': return { from: 'peak_frame',  to: 'close_frame', intra: true  };
        default:   return null;
    }
}

function renderIMIPlot(divId, trial, trialStart, trialMovs, fps, refVal, width) {
    const fpsT = trial.fps || fps || 60;
    const keys = _imiKeys(refVal);
    if (!keys) return;
    const xs = [], ys = [];
    if (keys.intra) {
        // Intra-movement durations: read ``from`` and ``to`` off the
        // SAME movement.  Skip movements that lack either frame.
        (trialMovs || []).forEach(m => {
            const fFrom = m[keys.from];
            const fTo   = m[keys.to];
            if (fFrom == null || fTo == null || !isFinite(fFrom) || !isFinite(fTo)) return;
            const toLocal = fTo - trialStart;
            if (toLocal < 0) return;
            const dt = (fTo - fFrom) / fpsT;
            if (!isFinite(dt) || dt <= 0) return;
            xs.push(+(toLocal / fpsT).toFixed(3));
            ys.push(+dt.toFixed(3));
        });
    } else {
        // Inter-movement intervals: previous movement's ``from``
        // frame to current movement's ``to`` frame.
        let prevFromLocal = null;
        (trialMovs || []).forEach(m => {
            const fTo = m[keys.to];
            if (prevFromLocal != null && fTo != null && isFinite(fTo)) {
                const toLocal = fTo - trialStart;
                if (toLocal >= 0) {
                    const dt = (toLocal - prevFromLocal) / fpsT;
                    if (isFinite(dt) && dt > 0) {
                        xs.push(+(toLocal / fpsT).toFixed(3));
                        ys.push(+dt.toFixed(3));
                    }
                }
            }
            const fFrom = m[keys.from];
            prevFromLocal = (fFrom != null && isFinite(fFrom))
                ? fFrom - trialStart
                : null;
        });
    }
    const trace = {
        x: xs, y: ys,
        type: 'scatter', mode: 'lines+markers',
        line: { color: '#9C27B0', width: 1 },
        marker: { color: '#9C27B0', size: 4 },
        hovertemplate: `%{x:.2f}s<br>${(refVal || '').toUpperCase()} %{y:.2f}s<extra></extra>`,
        name: 'Interval',
    };
    const n = trial.distances.length;
    const tEnd = n > 0 ? (n - 1) / fpsT : 1;
    const layout = {
        // l = 0 — the y-axis is in the sticky column on the left.
        // Zero top/bottom margins so the strip butts up against the
        // distance plot above and the velocity plot below.
        margin: { t: 0, b: 0, l: 0, r: 20 },
        dragmode: false,
        xaxis: {
            showticklabels: false, color: '#666', gridcolor: '#eee',
            range: [0, tEnd], autorange: false,
        },
        yaxis: {
            showticklabels: false, ticks: '', showline: false,
            title: { text: '' },
            color: '#9C27B0', gridcolor: '#f0f0f0',
            zeroline: false, nticks: 3,
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        showlegend: false, hovermode: 'x unified',
        width: width,
    };
    return Plotly.newPlot(divId, [trace], layout, {
        responsive: false, displayModeBar: false,
    }).then(() => _registerClickPlot(divId));
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
        // l = 0 — the y-axis is in the sticky column on the left.
        margin: { t: 5, b: 35, l: 0, r: 20 },
        dragmode: false,
        // Pin the X range to the trace extent so toggling the peak
        // overlays can't rescale the axis.
        xaxis: {
            title: { text: 'Time (s)', font: { size: 11 } }, color: '#666', gridcolor: '#eee',
            range: [times[0] || 0, times[n - 1] || 0], autorange: false,
        },
        yaxis: {
            showticklabels: false, ticks: '', showline: false,
            title: { text: '' },
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
    }).then(() => _registerClickPlot(divId));
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
    // IMI honors the inline 'peak / open / close' radio so the user
    // can switch which event drives the inter-movement interval.
    const _imiRef = (document.querySelector('input[name="imiRef"]:checked')?.value)
                    || 'pp';
    const _imiKeyPair = _imiKeys(_imiRef);
    const frameField = param === 'peak_open_vel'  ? 'peak_open_vel_frame'
                     : param === 'peak_close_vel' ? 'peak_close_vel_frame'
                     : param === 'imi'            ? (_imiKeyPair?.to || 'peak_frame')
                                                  : 'peak_frame';
    const frameMeta = _trialFrameMeta();

    // Same color for every trial — trial identity is shown by the
    // labels above each trial's segment instead of a color key.
    const color = MOVEMENT_DOT_COLOR;

    // Per-trial peak_frame → cluster color lookup so each movement
    // marker can pick up the same hue as the shape-overlay plot.
    // _shapeClusterColors[ti] is already keyed by peak_frame.
    const colorsOn = !!document.getElementById('shapeClusterColors')?.checked;
    const colorsByTrial = {};
    if (colorsOn && typeof _shapeClusterColors !== 'undefined' && _shapeClusterColors) {
        for (const k in _shapeClusterColors) {
            if (_shapeClusterColors[k]) colorsByTrial[k] = _shapeClusterColors[k];
        }
    }
    const colorForMovement = (m) => {
        const map = colorsByTrial[m.trial_idx];
        if (map && m.peak_frame != null && map[m.peak_frame]) return map[m.peak_frame];
        return color;
    };

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
        let y;
        if (param === 'imi') {
            // Inter-movement interval — see _imiKeys for the mapping
            // from the Interval radio value to {from, to} frame
            // fields.  For pp/oo/cc, from == to (same event on both
            // movements).  For po/co, the previous movement's `from`
            // event pairs with this movement's `to` event.  First
            // movement is null (no prior reference).
            const meta = frameMeta[ti] || { fps: 60 };
            const kp = _imiKeyPair || { from: 'peak_frame', to: 'peak_frame' };
            y = ms.map((m, i) => {
                if (i === 0) return null;
                const cur = m[kp.to];
                const prev = ms[i - 1][kp.from];
                if (cur == null || prev == null) return null;
                if (!isFinite(cur) || !isFinite(prev)) return null;
                return (cur - prev) / meta.fps;
            });
        } else {
            y = ms.map(m => m[param]);
        }
        const trialLabel = trialNames[ti] || `Trial ${+ti + 1}`;
        // Short label: just the trial suffix (e.g. "R1" from "PD03_R1").
        const trialShort = String(trialLabel).split('_').pop();

        const msColors = ms.map(colorForMovement);
        traces.push({
            x, y,
            type: 'scatter',
            mode: 'markers',
            name: trialLabel,
            xaxis: axId, yaxis: 'y',
            marker: { color: msColors, size: 7, opacity: 0.8 },
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

    Plotly.newPlot(divId, traces, layout, config)
        .then(() => _registerClickPlot(divId));
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
        const hand = document.getElementById('groupHandSelect')?.value || 'more';
        const trial = document.getElementById('groupTrialSelect')?.value || 'last';
        // seq_mode is no longer part of the cache key — each cache
        // file embeds every seq-effect model.  Only include it for
        // larger_se / smaller_se hand picks (live-computed; the
        // chosen hand depends on the model).
        let url = `/api/results/group?include_auto=true&source=${src}`
                + `&hand=${hand}&trial=${trial}`;
        if (hand === 'larger_se' || hand === 'smaller_se') {
            const seqMode = document.getElementById('groupSeqModeSelect')?.value || 'linear_full';
            url += `&seq_mode=${seqMode}`;
        }
        cachedGroup = await API.get(url);
        _initGroupSubjectChecked();
        // If the user has the DLC-group filter on for a non-DLC
        // source, pre-fetch the DLC subject set so the first render
        // reflects the filter immediately.
        if (_dlcFilterActive()) await _ensureDlcSubjectSet();
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

// Subjects that the DLC (corrections) source would include for the
// current (seq_mode, hand, trial, include_auto) combination.
// Populated lazily by _ensureDlcSubjectSet() when the "DLC group"
// checkbox is in play.  Cleared on any control change so we re-fetch
// for the new parameter combo.
let _dlcSubjectNames = null;
let _dlcSubjectKey   = null;   // signature of the combo it was fetched for

function _isNonDlcSource(src) {
    return src === 'mp_combined' || src === 'mp_forward'
        || src === 'skeleton_v1' || src === 'skeleton_v2';
}

function _dlcFilterActive() {
    const src = document.getElementById('groupSourceSelect')?.value || 'auto';
    if (!_isNonDlcSource(src)) return false;
    return !!document.getElementById('groupDlcFilter')?.checked;
}

async function _ensureDlcSubjectSet() {
    // Build a per-combo cache key from the controls that affect which
    // subjects the API would return.  seq_mode only matters for the
    // SE-hand picks; the rest of the response is independent of it.
    const hand    = document.getElementById('groupHandSelect')?.value    || 'more';
    const trial   = document.getElementById('groupTrialSelect')?.value   || 'last';
    const includeAuto = document.getElementById('includeAutoToggle')?.checked ? '1' : '0';
    const isSE = (hand === 'larger_se' || hand === 'smaller_se');
    const seqMode = isSE
        ? (document.getElementById('groupSeqModeSelect')?.value || 'linear_full')
        : '';
    const key = [seqMode, hand, trial, includeAuto].join('|');
    if (_dlcSubjectNames && _dlcSubjectKey === key) return _dlcSubjectNames;
    try {
        let url = `/api/results/group?include_auto=true&source=corrections`
                + `&hand=${hand}&trial=${trial}`;
        if (isSE) url += `&seq_mode=${seqMode}`;
        const data = await API.get(url);
        const subs = (data && data.subjects) || [];
        // /api/results/group only returns subjects that produced
        // at least two movements under the requested source — so
        // the response list IS the "subjects with DLC data" set.
        _dlcSubjectNames = new Set(subs.map(s => s.name).filter(Boolean));
        _dlcSubjectKey = key;
    } catch (_) {
        _dlcSubjectNames = new Set();
        _dlcSubjectKey = key;
    }
    return _dlcSubjectNames;
}

function _activeGroupSubjects() {
    if (!cachedGroup || !cachedGroup.subjects) return [];
    const dlcOn = _dlcFilterActive();
    // "Include Clinic Consents" checkbox (default UNCHECKED).  When
    // unchecked we restrict to subjects with ``updated_consent``
    // (the newer "New Consent" flag on the Subjects dashboard);
    // when checked we also include the older Clinic-Consent
    // subjects (no consent filter).
    const includeClinic = !!document.getElementById('updatedConsentFilter')?.checked;
    return cachedGroup.subjects.filter(s => {
        if (!_groupSubjectChecked[s.name]) return false;
        if (dlcOn && _dlcSubjectNames && !_dlcSubjectNames.has(s.name)) return false;
        if (!includeClinic && !s.updated_consent) return false;
        return true;
    });
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

    // Scrollable grid: Mean + Variance + 0..2 sequence-effect rows
    // (R² and/or slope) per the checkboxes next to the seq-effect
    // dropdown.  ``field`` is the lookup key into each metric spec;
    // ``seqMetric`` is set on seq rows to tell ``_key`` which
    // per-mode prefix to use (``seq_…`` for R², ``seqslope_…`` for
    // slope).
    const _seqShow = (v) =>
        !!document.querySelector(`#groupSeqMetric input.groupseqm[value="${v}"]:checked`);
    const _showR2 = _seqShow('r2');
    const _showSlope = _seqShow('slope');
    const ROW_DEFS = [
        { label: 'Mean', field: 'mean', height: 200 },
        { label: 'Variance', field: 'cv', height: 180 },
    ];
    if (_showR2)    ROW_DEFS.push({ label: 'Sequence Effect (R²)',    field: 'seq', seqMetric: 'r2',    height: 180 });
    if (_showSlope) ROW_DEFS.push({ label: 'Sequence Effect (slope)', field: 'seq', seqMetric: 'slope', height: 180 });

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

    // Per-row div-id suffix — two seq rows both have field=='seq',
    // so distinguish them by their seqMetric to avoid id collisions.
    const _rowSuffix = (row) => row.seqMetric ? `${row.field}_${row.seqMetric}` : row.field;
    ROW_DEFS.forEach((row, ri) => {
        visibleMetrics.forEach((m) => {
            const spec = m[row.field];
            const divId = `grpPlot_${m.id}_${_rowSuffix(row)}`;
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
            const divId = `grpPlotDose_${m.id}_${_rowSuffix(row)}`;
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

    // Sequence-Effect rows read R² (seq_<mode>_*) or slope
    // (seqslope_<mode>_*) from each subject's cached fields.  ``row``
    // carries the seqMetric ('r2' or 'slope') so two seq rows can
    // coexist with independent metrics.
    const seqMode = document.getElementById('groupSeqModeSelect')?.value || 'linear_full';
    const _key = (spec, row) => {
        if (row.field !== 'seq') return spec.key;
        if (seqMode === 'none') return null;   // 'none' has no per-mode data
        const param = spec.key.replace(/^seq_/, '');
        const prefix = row.seqMetric === 'slope' ? 'seqslope' : 'seq';
        return `${prefix}_${seqMode}_${param}`;
    };

    // Per-row Y-axis label.  Sequence Effect rows carry their own
    // "(R²)" / "(slope)" suffix from the row label.
    const _yLabelFor = (row) => row.label;

    // Render each chart.  Column titles live in the HTML header above
    // each column; per-plot Y-axis labels tell which row (Mean / Var /
    // Sequence Effect) the chart belongs to.
    ROW_DEFS.forEach(row => {
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            if (!spec) return;
            const k = _key(spec, row);
            // seq_mode === 'none' returns null — the row has nothing
            // to plot, so paint an empty placeholder and move on.
            if (k == null) return;
            const divId = `grpPlot_${m.id}_${_rowSuffix(row)}`;
            renderGroupBar(divId, data, k, _reverseY(m, row), _yLabelFor(row));
        });
    });

    // Dose-response scatters — one per (metric × row), aligned with the
    // bar grid.
    ROW_DEFS.forEach((row) => {
        visibleMetrics.forEach(m => {
            const spec = m[row.field];
            if (!spec) return;
            const k = _key(spec, row);
            if (k == null) return;
            const divId = `grpPlotDose_${m.id}_${_rowSuffix(row)}`;
            renderDoseScatter(divId, data, k, _reverseY(m, row), _yLabelFor(row));
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

    // Persist the highlighted subject (and a sensible trial pick)
    // into nav state so navigating to Labels / Events / Analyze /
    // Individual Results / Oscillations lands on the same one.
    if (highlightedSubject) {
        const subj = (subjects || []).find(s => s.name === highlightedSubject);
        if (subj && subj.id) {
            sessionStorage.setItem('dlc_lastSubjectId', String(subj.id));
            if (typeof setLastSubject === 'function') setLastSubject(subj.id);
            if (typeof setNavState === 'function') {
                // Pick the trial index that matches the current
                // hand/trial selectors: "first"/"average" → first
                // trial in the chosen-hand subset (idx 0 means the
                // landing page falls back to its own default), "last"
                // → leave it null so the landing page picks naturally.
                // We don't have per-subject trial maps here, so just
                // anchor on 0 (= the first trial); the landing page
                // can map it through its own data.
                setNavState({ subjectId: subj.id, trialIdx: 0 });
            }
        }
    }

    // Re-render all group plots to update highlighting
    renderGroupPlots();
}

// ── Event listeners ────────────────────────────────────────────

document.getElementById('subjectSelect').addEventListener('change', (e) => {
    currentSubjectId = e.target.value;
    // Empty value = the "Select a subject" placeholder (used on the
    // group tab); nothing to show.
    if (!currentSubjectId) return;
    // Trial-collapse state is per-subject — drop it on subject change.
    _collapsedTrials = new Set();
    // Drag-marked intervals are per-subject too.  Clear the click
    // highlight too so it doesn't reappear on a stale trial idx.
    _intervals = {};
    _clickHL = null;

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
        if (cb.dataset.dparam === 'imi') _syncImiRefVisibility();
        if (cachedMovements) renderDistMovementPlots();
    });
});
document.getElementById('distSequenceMode').addEventListener('change', () => {
    cachedSequenceAssignments = null;
    if (cachedMovements) renderDistMovementPlots();
});

// IMI reference (peak / open / close): re-render when changed.
document.querySelectorAll('input[name="imiRef"]').forEach(r => {
    r.addEventListener('change', () => {
        if (cachedMovements) renderDistMovementPlots();
        // The compressed IMI strips between dist/vel also depend
        // on the peak/open/close reference — re-render them.
        if (cachedTraces) renderAllDistancePlots();
    });
});

// PC visibility checkboxes (PCA view): re-render the PCA plots.
document.querySelectorAll('#pcaCompControls input[data-pc]').forEach(cb => {
    cb.addEventListener('change', () => {
        if (_resultsViewMode === 'pca' && cachedPCA) renderFingertipPCA();
    });
});

// FFT smoothing slider: update the readout live, re-render on commit.
(() => {
    const slider = document.getElementById('pcaFftSmooth');
    const readout = document.getElementById('pcaFftSmoothVal');
    if (!slider) return;
    slider.addEventListener('input', () => {
        if (readout) readout.textContent = slider.value;
    });
    slider.addEventListener('change', () => {
        if (_resultsViewMode === 'pca' && cachedPCA) renderFingertipPCA();
    });
})();

/** No-op kept for the existing call site — the IMI reference
 *  radios now live in the top overlay bar and also drive the
 *  always-on compressed IMI strip between dist/vel plots, so they
 *  stay visible regardless of the lower-bar IMI checkbox state. */
function _syncImiRefVisibility() { /* intentionally empty */ }

// Overlay controls: re-render distance/velocity plots or PCA plots
['overlayPeakDist', 'overlayOpen', 'overlayClose', 'overlayPause', 'overlayPeakOpenVel', 'overlayPeakCloseVel', 'overlaySequences'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => {
        if (_resultsViewMode === 'pca') { if (cachedPCA) renderFingertipPCA(); }
        else if (cachedTraces) renderAllDistancePlots();
    });
});

// X-scale slider: seconds of trace shown per screen width, applied to
// every distance/velocity plot.
(() => {
    const sl = document.getElementById('xScaleSlider');
    if (!sl) return;
    sl.addEventListener('input', () => {
        if (_resultsViewMode === 'pca') { if (cachedPCA) renderFingertipPCA(); }
        else if (cachedTraces) renderAllDistancePlots();
    });
})();

// Interval × Parameter scatter dropdowns — re-render on change.
['ipXSelect', 'ipYSelect'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', () => {
        if (cachedMovements) renderIntervalParamPlots();
    });
});

// Slope toggle — flips _ipSlopeOn, restyles the button to show
// the active state, and re-renders the scatters with / without
// best-fit lines + stats annotations.
(() => {
    const btn = document.getElementById('ipSlopeBtn');
    if (!btn) return;
    const _paint = () => {
        if (_ipSlopeOn) {
            btn.classList.add('btn-primary');
            btn.setAttribute('aria-pressed', 'true');
        } else {
            btn.classList.remove('btn-primary');
            btn.setAttribute('aria-pressed', 'false');
        }
    };
    _paint();
    btn.addEventListener('click', () => {
        _ipSlopeOn = !_ipSlopeOn;
        _paint();
        if (cachedMovements) renderIntervalParamPlots();
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
document.getElementById('updatedConsentFilter')?.addEventListener('change', () => {
    if (cachedGroup) renderGroupPlots();
});

document.getElementById('groupSelectAllBtn').addEventListener('click',
    () => window._groupSelectAll(true));

document.getElementById('groupRegenerateBtn')?.addEventListener('click', async (e) => {
    const btn = e.currentTarget;
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Regenerating…';
    try {
        const src = document.getElementById('groupSourceSelect')?.value || 'auto';
        const seqMode = document.getElementById('groupSeqModeSelect')?.value || 'linear_full';
        const hand = document.getElementById('groupHandSelect')?.value || 'more';
        const trial = document.getElementById('groupTrialSelect')?.value || 'last';
        cachedGroup = await API.post(
            `/api/results/group/regenerate?include_auto=true&source=${src}` +
            `&seq_mode=${seqMode}&hand=${hand}&trial=${trial}`);
        _initGroupSubjectChecked();
        renderGroupPlots();
        btn.textContent = 'Done';
        setTimeout(() => { btn.textContent = orig; }, 1200);
    } catch (err) {
        btn.textContent = 'Error';
        console.error('Regenerate failed:', err);
        setTimeout(() => { btn.textContent = orig; }, 1800);
    } finally {
        btn.disabled = false;
    }
});

// Group-tab source + sequence-effect dropdowns — re-fetch the group
// data with the new parameters.
function _syncDlcFilterVisibility() {
    const wrap = document.getElementById('groupDlcFilterWrap');
    const src  = document.getElementById('groupSourceSelect')?.value || 'auto';
    if (!wrap) return;
    wrap.style.display = _isNonDlcSource(src) ? 'inline-flex' : 'none';
}
_syncDlcFilterVisibility();

document.getElementById('groupSourceSelect')?.addEventListener('change', () => {
    _syncDlcFilterVisibility();
    cachedGroup = null;
    loadGroup();
});
document.getElementById('groupSeqModeSelect')?.addEventListener('change', () => {
    // seq_mode is no longer part of the cache key — every seq model
    // is embedded in the cached payload, so a model change is a pure
    // re-render.  Exception: SE-hand picks (larger_se/smaller_se)
    // depend on the chosen model for the per-subject hand selection,
    // so refetch in that case.
    const hand = document.getElementById('groupHandSelect')?.value || 'more';
    const isSE = (hand === 'larger_se' || hand === 'smaller_se');
    if (isSE) {
        _dlcSubjectNames = null; _dlcSubjectKey = null;
        cachedGroup = null;
        loadGroup();
    } else if (cachedGroup) {
        renderGroupPlots();
    }
});
['groupHandSelect', 'groupTrialSelect'].forEach(id =>
    document.getElementById(id)?.addEventListener('change', () => {
        _dlcSubjectNames = null; _dlcSubjectKey = null;
        cachedGroup = null;
        loadGroup();
    }));

// "DLC group" filter checkbox — restrict the current non-DLC source
// to subjects that have DLC data available, for head-to-head
// comparison on the same set of subjects.
document.getElementById('groupDlcFilter')?.addEventListener('change', async () => {
    if (_dlcFilterActive()) {
        await _ensureDlcSubjectSet();
    }
    if (cachedGroup) renderGroupPlots();
});
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
    cachedPCA = null;
    const activeTab = document.querySelector('.results-tab.active')?.id;
    if (activeTab === 'tabGroup') {
        loadGroup();
    } else if (currentSubjectId) {
        loadDistances(currentSubjectId);
    }
});

document.getElementById('resultsViewMode')?.addEventListener('change', e => {
    _resultsViewMode = e.target.value;
    _syncViewMode();
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
