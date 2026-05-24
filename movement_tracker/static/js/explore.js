/* Variable Explorer: scatter any pair of clinical/movement variables,
   or a by-group bar chart of one variable. Colors match the Group
   Comparison page. */

const GROUP_COLORS = {
    Control: '#4CAF50',
    MSA: '#FF5722',
    PD: '#2196F3',
    PSP: '#9C27B0',
};

let _data = null;        // { groups, subjects, variables }
let _varMeta = {};       // base key -> { label, aggregatable }
let _subjChecked = {};   // subject name -> bool
let _groupCollapsed = {};   // group name -> { subject_name: bool } snapshot (frozen state)

const AGGS = [
    { v: 'mean',     label: 'Mean' },
    { v: 'variance', label: 'Variance' },
    { v: 'cv',       label: 'CV' },
    { v: 'seq',      label: 'Sequence effect' },
];
const AGG_PREFIX = { mean: 'Mean ', variance: 'Variance ', cv: 'CV ', seq: 'Seq. Effect ' };

const $ = (id) => document.getElementById(id);

// Static-site mode (GitHub Pages export): only a subset of source ×
// seq_mode × hand × trial combinations is exported as JSON, so hide
// the unsupported choices from the UI.
(function _trimStaticControls() {
    if (!window.STATIC_RESULTS) return;
    const drop = (selectId, values) => {
        const el = document.getElementById(selectId);
        if (!el) return;
        Array.from(el.options).forEach(opt => {
            if (values.includes(opt.value)) opt.remove();
        });
    };
    // Source locked to "auto" — hide the whole control.
    const srcSel = document.getElementById('exSourceSelect');
    if (srcSel) {
        srcSel.value = 'auto';
        if (srcSel.parentElement) srcSel.parentElement.style.display = 'none';
    }
    drop('exSeqModeSelect', ['none']);
    drop('exHandSelect',    ['L', 'R', 'larger_se', 'smaller_se']);
})();

async function loadExplore() {
    const plot = $('explorePlot');
    const src = $('exSourceSelect').value || 'auto';
    const sm = $('exSeqModeSelect').value || 'linear_full';
    const hand = $('exHandSelect')?.value || 'more';
    const trial = $('exTrialSelect')?.value || 'last';
    // Always fetch with auto events available; the include-auto toggle
    // only controls which subjects are checked by default (matches the
    // Group Comparison page).
    plot.innerHTML = '<div class="results-no-data">Loading…</div>';
    try {
        _data = await API.get(
            `/api/results/explore?include_auto=true&source=${src}&seq_mode=${sm}` +
            `&hand=${hand}&trial=${trial}`);
        _varMeta = {};
        _data.variables.forEach(v => {
            _varMeta[v.key] = { label: v.label, aggregatable: !!v.aggregatable };
        });
        _initSubjChecked();
        _populateVarSelectors();
        render();
    } catch (e) {
        plot.innerHTML = `<div class="results-no-data" style="color:#d32f2f;">${e.message}</div>`;
    }
}

function _optionsHtml(selectedKey) {
    // Grouped by category, with optgroups.
    const byCat = { clinical: [], movement: [] };
    _data.variables.forEach(v => { (byCat[v.category] || (byCat[v.category] = [])).push(v); });
    const grp = (label, items) => items.length
        ? `<optgroup label="${label}">` +
          items.map(v => `<option value="${v.key}" ${v.key === selectedKey ? 'selected' : ''}>${v.label}</option>`).join('') +
          '</optgroup>'
        : '';
    return grp('Clinical', byCat.clinical) + grp('Movement', byCat.movement);
}

function _populateVarSelectors() {
    const xSel = $('exVarX'), ySel = $('exVarY');
    const keys = _data.variables.map(v => v.key);
    const curX = keys.includes(xSel.value) ? xSel.value
        : (keys.includes('amplitude') ? 'amplitude' : keys[0]);
    const curY = keys.includes(ySel.value) ? ySel.value
        : (keys.includes('peak_open_vel') ? 'peak_open_vel' : keys[Math.min(1, keys.length - 1)]);
    xSel.innerHTML = _optionsHtml(curX);
    ySel.innerHTML = _optionsHtml(curY);
}

// Render the Mean/CV/Sequence-effect radios for a select, only when the
// selected variable is aggregatable (a movement parameter).
function _renderAggRadios(prefix) {
    const sel = $(prefix === 'X' ? 'exVarX' : 'exVarY');
    const host = $(prefix === 'X' ? 'exXAgg' : 'exYAgg');
    const meta = _varMeta[sel.value];
    if (!meta || !meta.aggregatable) { host.innerHTML = ''; return; }
    const name = `agg${prefix}`;
    const cur = host.querySelector(`input[name="${name}"]:checked`)?.value || 'mean';
    const mname = `seqm${prefix}`;
    // Default to slope when the user first picks "Sequence effect".
    const curM = host.querySelector(`input[name="${mname}"]:checked`)?.value || 'slope';
    let html = AGGS.map(a =>
        `<label style="display:inline-flex;align-items:center;gap:2px;cursor:pointer;">
            <input type="radio" name="${name}" value="${a.v}" ${a.v === cur ? 'checked' : ''}> ${a.label}
        </label>`).join('');
    // When "Sequence effect" is chosen, offer R² vs Slope.
    if (cur === 'seq') {
        html += `<span style="opacity:0.5;">|</span>` +
            [{ v: 'r2', label: 'R²' }, { v: 'slope', label: 'Slope' }].map(m =>
            `<label style="display:inline-flex;align-items:center;gap:2px;cursor:pointer;">
                <input type="radio" name="${mname}" value="${m.v}" ${m.v === curM ? 'checked' : ''}> ${m.label}
            </label>`).join('');
    }
    host.innerHTML = html;
    host.querySelectorAll('input').forEach(r => r.addEventListener('change', render));
}

function _agg(prefix) {
    const host = $(prefix === 'X' ? 'exXAgg' : 'exYAgg');
    return host.querySelector(`input[name="agg${prefix}"]:checked`)?.value || 'mean';
}

function _seqMetric(prefix) {
    const host = $(prefix === 'X' ? 'exXAgg' : 'exYAgg');
    return host.querySelector(`input[name="seqm${prefix}"]:checked`)?.value || 'r2';
}

// Effective data key + display label for a select, applying the
// aggregate radio when the variable is aggregatable.
function _resolve(prefix) {
    const sel = $(prefix === 'X' ? 'exVarX' : 'exVarY');
    const base = sel.value;
    const meta = _varMeta[base] || { label: base, aggregatable: false };
    if (!meta.aggregatable) return { key: base, label: meta.label };
    const agg = _agg(prefix);
    if (agg === 'seq') {
        const m = _seqMetric(prefix);
        const key = m === 'slope' ? `seqslope_${base}` : `seq_${base}`;
        const tag = m === 'slope' ? '(slope)' : '(R²)';
        return { key, label: `Seq. Effect ${tag} ${meta.label}` };
    }
    return { key: `${agg}_${base}`, label: `${AGG_PREFIX[agg]}${meta.label}` };
}

// ── Subject selection (group-colored checkboxes) ───────────────
function _initSubjChecked() {
    if (!_data || !_data.subjects) return;
    const inc = $('exIncludeAuto').checked;
    _subjChecked = {};
    _data.subjects.forEach(s => {
        // Complete-event subjects on by default; others follow the
        // include-auto toggle.
        _subjChecked[s.name] = s.has_complete_events ? true : inc;
    });
}

function _renderSubjectList() {
    const host = $('exSubjectList');
    if (!host || !_data || !_data.subjects) { if (host) host.innerHTML = ''; return; }
    const byGroup = {};
    (_data.groups || []).forEach(g => { byGroup[g] = []; });
    _data.subjects.forEach(s => { (byGroup[s.group] = byGroup[s.group] || []).push(s); });

    let html = '';
    (_data.groups || []).forEach(g => {
        const subs = byGroup[g] || [];
        if (!subs.length) return;
        const color = GROUP_COLORS[g] || '#999';
        const collapsed = !!_groupCollapsed[g];
        const safeG = g.replace(/"/g, '&quot;');
        html += `<div class="gsl-group"><span class="gsl-group-label${collapsed ? ' gsl-collapsed' : ''}"
            data-group="${safeG}" title="Click to ${collapsed ? 'restore' : 'uncheck'} all ${g} subjects">${g}</span><div class="gsl-tags">`;
        subs.forEach(s => {
            const checked = !!_subjChecked[s.name];
            const complete = !!s.has_complete_events;
            const dim = complete ? '' : ' dim';
            const title = complete ? 'Saved events (open/peak/close)'
                : (s.has_saved_events ? 'Incomplete saved events' : 'No saved events');
            const safe = s.name.replace(/"/g, '&quot;');
            const tag = s.laterality_side
                ? ` <span style="font-weight:700;font-size:9px;opacity:0.75;">${s.laterality_side}</span>`
                : '';
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
            _subjChecked[e.target.getAttribute('data-subject')] = e.target.checked;
            render();
        });
    });
    // Click a group name → uncheck every subject in it (saving the
    // current state).  Click again → restore the saved state.
    host.querySelectorAll('.gsl-group-label').forEach(lbl => {
        lbl.addEventListener('click', () => _toggleGroupCollapsed(lbl.dataset.group));
    });
}

function _activeSubjects() {
    if (!_data || !_data.subjects) return [];
    return _data.subjects.filter(s => _subjChecked[s.name]);
}

/** Toggle a group's "collapsed" state.  When collapsed we save each
 *  subject's checked state and force them all OFF.  Clicking again
 *  restores whatever was checked before. */
function _toggleGroupCollapsed(g) {
    if (!g || !_data || !_data.subjects) return;
    const subs = _data.subjects.filter(s => s.group === g);
    if (_groupCollapsed[g]) {
        const saved = _groupCollapsed[g];
        subs.forEach(s => {
            if (Object.prototype.hasOwnProperty.call(saved, s.name)) {
                _subjChecked[s.name] = saved[s.name];
            }
        });
        delete _groupCollapsed[g];
    } else {
        const saved = {};
        subs.forEach(s => {
            saved[s.name] = !!_subjChecked[s.name];
            _subjChecked[s.name] = false;
        });
        _groupCollapsed[g] = saved;
    }
    render();
}

function _exPlotMode() {
    const r = document.querySelector('input[name="exPlotType"]:checked');
    return r ? r.value : 'scatter';
}

function render() {
    if (!_data) return;
    _renderSubjectList();
    const mode = _exPlotMode();
    // Y row hides entirely in bar mode.  Must re-apply display:flex
    // (not just empty string) — the row is a <div>, which falls back
    // to display:block, breaking the side-by-side layout of label +
    // agg-radios.
    const yRow = $('exYRow');
    if (yRow) yRow.style.display = (mode === 'scatter') ? 'flex' : 'none';
    $('exXLabel').childNodes[0].textContent = (mode === 'scatter') ? 'X: ' : 'Variable: ';
    // "Slope" + "Legend" only apply to scatter.
    const bestFitLbl = $('exBestFitLabel');
    if (bestFitLbl) bestFitLbl.style.display = (mode === 'scatter') ? '' : 'none';
    const legendLbl  = $('exLegendLabel');
    if (legendLbl)  legendLbl.style.display  = (mode === 'scatter') ? '' : 'none';
    const anovaLbl   = $('exAnovaLabel');
    if (anovaLbl)   anovaLbl.style.display   = (mode === 'bar') ? 'flex' : 'none';
    _renderAggRadios('X');
    if (mode === 'scatter') { _renderAggRadios('Y'); renderScatter(); }
    else { $('exYAgg').innerHTML = ''; renderBar(); }
}

// ── Linear regression + p-value helpers ───────────────────────
function _linRegStats(xs, ys) {
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
    return { slope, intercept, r2, t, df, p: _studentT2Tail(t, df), n };
}
function _studentT2Tail(t, df) {
    const x = df / (df + t * t);
    return _betai(df / 2, 0.5, x);
}
// Upper-tail p-value for the F-distribution: P(F > f | df1, df2).
function _fPValue(f, df1, df2) {
    if (!isFinite(f) || f <= 0 || df1 <= 0 || df2 <= 0) return 1;
    return _betai(df2 / 2, df1 / 2, df2 / (df2 + df1 * f));
}
// One-way ANOVA across groups.  `groups` is an array of value-arrays.
// Returns {F, df1, df2, p, k, N} or null if not enough data.
function _oneWayANOVA(groups) {
    const nonEmpty = groups.filter(g => g.length > 0);
    const k = nonEmpty.length;
    if (k < 2) return null;
    const N = nonEmpty.reduce((a, g) => a + g.length, 0);
    if (N - k < 1) return null;
    const grandMean = nonEmpty.reduce((a, g) => a + g.reduce((x, y) => x + y, 0), 0) / N;
    let SSB = 0, SSW = 0;
    for (const g of nonEmpty) {
        const m = g.reduce((a, v) => a + v, 0) / g.length;
        SSB += g.length * (m - grandMean) ** 2;
        for (const v of g) SSW += (v - m) ** 2;
    }
    const df1 = k - 1, df2 = N - k;
    const MSB = SSB / df1, MSW = SSW / df2;
    if (MSW <= 0) return { F: Infinity, df1, df2, p: 0, k, N };
    const F = MSB / MSW;
    return { F, df1, df2, p: _fPValue(F, df1, df2), k, N };
}
// Welch's unpaired t-test for two samples.  Returns {t, df, p} or null.
function _welchT(a, b) {
    if (a.length < 2 || b.length < 2) return null;
    const ma = a.reduce((x, y) => x + y, 0) / a.length;
    const mb = b.reduce((x, y) => x + y, 0) / b.length;
    const va = a.reduce((x, y) => x + (y - ma) ** 2, 0) / (a.length - 1);
    const vb = b.reduce((x, y) => x + (y - mb) ** 2, 0) / (b.length - 1);
    const se2 = va / a.length + vb / b.length;
    if (se2 <= 0) return { t: 0, df: a.length + b.length - 2, p: 1 };
    const t = (ma - mb) / Math.sqrt(se2);
    const df = (se2 ** 2) /
        ((va / a.length) ** 2 / (a.length - 1) + (vb / b.length) ** 2 / (b.length - 1));
    return { t, df, p: _studentT2Tail(t, df) };
}
function _fmtP(p) {
    if (!isFinite(p)) return 'NaN';
    if (p < 0.001) return 'p < 0.001';
    return `p = ${p.toFixed(3)}`;
}
function _betai(a, b, x) {
    if (x <= 0 || x >= 1) return x <= 0 ? 0 : 1;
    const bt = Math.exp(_lnGamma(a + b) - _lnGamma(a) - _lnGamma(b)
                       + a * Math.log(x) + b * Math.log(1 - x));
    return (x < (a + 1) / (a + b + 2))
        ? bt * _betacf(a, b, x) / a
        : 1 - bt * _betacf(b, a, 1 - x) / b;
}
function _betacf(a, b, x) {
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
function _lnGamma(x) {
    const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
                 -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
    return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function _val(s, key) {
    const v = s.vars[key];
    return (v != null && isFinite(v)) ? v : null;
}

/** Compute the Plotly left-margin so the Y-axis baseline (the plot's
 *  left edge in pixels) lands at 10% along the width of the plot-
 *  controls bar above it.  This pins the data area to a fixed
 *  horizontal position regardless of tick-label / Y-title widths.
 *  Plotly margins are measured inside the plot div, so we subtract
 *  the div's window-x from the desired window-x. */
/** Choose a sensible up/down step for a numeric input based on the
 *  current data span.  1 unit per click for ranges < 20, 5 for < 100,
 *  25 for < 500, etc. */
function _autoStep(range) {
    if (!isFinite(range) || range <= 0) return 1;
    if (range < 0.2)  return 0.01;
    if (range < 2)    return 0.1;
    if (range < 20)   return 1;
    if (range < 100)  return 5;
    if (range < 500)  return 25;
    if (range < 2000) return 100;
    return 500;
}

/** Round a value to a sensible precision for the chosen step. */
function _fmtNum(v, step) {
    if (!isFinite(v)) return '';
    const dec = step < 0.1 ? 3 : step < 1 ? 2 : step < 10 ? 1 : 0;
    return (+v.toFixed(dec)).toString();
}

/** After Plotly has rendered, push the resolved axis ranges into the
 *  min/max textboxes (so the user sees real numbers, not "auto"),
 *  and set each input's `step` to a sensible increment for the
 *  variable's range.  User-typed values are never overwritten — only
 *  empty inputs get filled. */
function _applyAutoRangeInputs() {
    const div = $('explorePlot');
    if (!div || !div._fullLayout) return;
    const mode = _exPlotMode();
    // In bar mode the X variable lives on the y-axis (groups on x).
    const axForX = (mode === 'scatter') ? div._fullLayout.xaxis : div._fullLayout.yaxis;
    const axForY = (mode === 'scatter') ? div._fullLayout.yaxis : null;
    const _fill = (minId, maxId, ax) => {
        const minEl = $(minId), maxEl = $(maxId);
        if (!minEl || !maxEl) return;
        if (!ax || !ax.range) { minEl.step = '1'; maxEl.step = '1'; return; }
        const [lo, hi] = ax.range;
        const step = _autoStep(hi - lo);
        if (minEl.value === '') minEl.value = _fmtNum(lo, step);
        if (maxEl.value === '') maxEl.value = _fmtNum(hi, step);
        minEl.step = String(step);
        maxEl.step = String(step);
    };
    _fill('exXMin', 'exXMax', axForX);
    _fill('exYMin', 'exYMax', axForY);
}

/** Read a (min, max) pair from two numeric inputs, falling back to
 *  the data extremes (`vals`) for whichever bound is left blank.
 *  Returns `null` when both inputs are empty so the caller can use
 *  Plotly's autorange. */
function _readRangeInputs(minId, maxId, vals) {
    const mi = $(minId), ma = $(maxId);
    if (!mi || !ma) return null;
    const a = mi.value === '' ? NaN : parseFloat(mi.value);
    const b = ma.value === '' ? NaN : parseFloat(ma.value);
    if (!isFinite(a) && !isFinite(b)) return null;
    const finite = vals.filter(v => isFinite(v));
    const dMin = finite.length ? Math.min(...finite) : 0;
    const dMax = finite.length ? Math.max(...finite) : 1;
    const lo = isFinite(a) ? a : dMin;
    const hi = isFinite(b) ? b : dMax;
    return [lo, hi];
}

function _leftMarginForFixedYBaseline() {
    const div = document.getElementById('explorePlot');
    if (!div) return 90;
    const ref = document.getElementById('explorePlotControls') || div;
    const refRect = ref.getBoundingClientRect();
    const divRect = div.getBoundingClientRect();
    const desired = refRect.left + refRect.width * 0.10;
    // Floor so the y-title / tick numbers can't get clipped off the
    // plot's left edge in narrow viewports.
    return Math.max(40, Math.round(desired - divRect.left));
}

function renderScatter() {
    const X = _resolve('X'), Y = _resolve('Y');
    const groups = _data.groups;
    let n = 0;
    const active = _activeSubjects();
    const traces = groups.map(g => {
        const subs = active.filter(s =>
            s.group === g && _val(s, X.key) != null && _val(s, Y.key) != null);
        n += subs.length;
        return {
            x: subs.map(s => _val(s, X.key)),
            y: subs.map(s => _val(s, Y.key)),
            text: subs.map(s => s.name),
            type: 'scatter', mode: 'markers', name: g,
            marker: { color: GROUP_COLORS[g] || '#999', size: 18, opacity: 0.8,
                      line: { color: '#333', width: 1 } },
            hovertemplate: `%{text}<br>${X.label}: %{x:.3f}<br>${Y.label}: %{y:.3f}<extra>${g}</extra>`,
        };
    });
    // Always include a (possibly invisible) best-fit trace so legend
    // clicks can re-style it instead of triggering a full re-render
    // (which would reset the user's legend selections).
    traces.push({
        x: [], y: [],
        type: 'scatter', mode: 'lines', name: 'Best fit',
        line: { color: '#000', width: 1.5 },
        hoverinfo: 'skip', showlegend: false,
        visible: false,
    });
    // Optional explicit axis ranges from the min/max textboxes.  If
    // the user only fills in one bound, use the data extreme for the
    // other so the user's bound is honoured.
    const _xs = traces.flatMap(t => (t.x || []).filter(v => v != null && isFinite(v)));
    const _ys = traces.flatMap(t => (t.y || []).filter(v => v != null && isFinite(v)));
    const xRange = _readRangeInputs('exXMin', 'exXMax', _xs);
    const yRange = _readRangeInputs('exYMin', 'exYMax', _ys);
    const layout = {
        // Pin the Y-axis baseline (= plot's left edge in pixels) to
        // 10% of the controls-bar width so the plot doesn't shift
        // when the tick labels or y-title get wider/narrower.
        margin: { t: 20, b: 80, l: _leftMarginForFixedYBaseline(), r: 20 },
        xaxis: {
            title: { text: X.label, font: { size: 24 } },
            color: '#666', gridcolor: '#f0f0f0',
            tickfont: { size: 22 },
            showline: true, linecolor: '#666', linewidth: 2, mirror: false, zeroline: false,
            ...(xRange ? { range: xRange, autorange: false } : { autorange: true }),
        },
        yaxis: {
            title: { text: Y.label, font: { size: 24 }, standoff: 14 },
            color: '#666', gridcolor: '#f0f0f0',
            tickfont: { size: 22 },
            showline: true, linecolor: '#666', linewidth: 2, mirror: false, zeroline: false,
            automargin: false,
            ...(yRange ? { range: yRange, autorange: false } : { autorange: true }),
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        legend: { orientation: 'h', y: 1.08, font: { size: 22 } },
        showlegend: $('exLegend') ? $('exLegend').checked : true,
        hovermode: 'closest',
    };
    Plotly.newPlot('explorePlot', traces, layout, { responsive: true, displayModeBar: false })
        .then(() => {
            _refitBestFitFromVisible();
            _wireBestFitListener();
            _applyAutoRangeInputs();
        });
}

// Recompute the best-fit line and the slope/R²/p annotation using
// only the traces that aren't currently hidden by a legend click.
// Re-styles the existing "Best fit" trace (no full re-render — that
// would reset the user's legend selections).
let _refittingBF = false;
function _refitBestFitFromVisible() {
    const div = $('explorePlot');
    if (!div || !div.data) return;
    const bfIdx = div.data.findIndex(t => t && t.name === 'Best fit');
    if (bfIdx < 0) return;
    const checked = !!($('exBestFit') && $('exBestFit').checked);
    const allX = [], allY = [];
    if (checked) {
        div.data.forEach((t, i) => {
            if (i === bfIdx) return;
            if (t.visible === 'legendonly' || t.visible === false) return;
            const xs = t.x || [], ys = t.y || [];
            for (let k = 0; k < xs.length; k++) {
                allX.push(xs[k]); allY.push(ys[k]);
            }
        });
    }
    // Subject count = visible points if Slope-on, else total over the
    // non-best-fit traces (matches the existing displayed count).
    const visibleCount = div.data.reduce((s, t, i) => i === bfIdx ? s
        : s + (t.visible === 'legendonly' || t.visible === false
               ? 0 : (t.x || []).length), 0);
    const baseText = `n = ${checked ? allX.length : visibleCount}`;
    let lineX = null, lineY = null, fitBoxText = null;
    if (checked && allX.length >= 3) {
        const stats = _linRegStats(allX, allY);
        if (stats) {
            const xMin = Math.min(...allX), xMax = Math.max(...allX);
            lineX = [xMin, xMax];
            lineY = [stats.intercept + stats.slope * xMin,
                     stats.intercept + stats.slope * xMax];
            const pStr = (stats.p < 1e-4)
                ? stats.p.toExponential(2)
                : stats.p.toFixed(4);
            // Plain text with newlines — the fit box uses
            // white-space: pre so newlines render, and copy/paste of
            // selected text comes out cleanly.
            fitBoxText = `slope = ${stats.slope.toPrecision(3)}\n`
                       + `R² = ${stats.r2.toFixed(3)}\n`
                       + `p = ${pStr}`;
        }
    }
    _refittingBF = true;
    const updates = lineX
        ? { x: [lineX], y: [lineY], visible: true }
        : { visible: false };
    // Snapshot current axis ranges so we can restore them — flipping
    // the best-fit visibility (or its endpoints) shouldn't cause the
    // axes to re-fit.
    const fl = div._fullLayout || {};
    const xRange = fl.xaxis && fl.xaxis.range ? fl.xaxis.range.slice() : null;
    const yRange = fl.yaxis && fl.yaxis.range ? fl.yaxis.range.slice() : null;
    Promise.resolve(Plotly.restyle(div, updates, [bfIdx]))
        .then(() => {
            if (xRange && yRange) {
                return Plotly.relayout(div, {
                    'xaxis.range': xRange, 'xaxis.autorange': false,
                    'yaxis.range': yRange, 'yaxis.autorange': false,
                });
            }
        })
        .then(() => { _refittingBF = false; })
        .catch(() => { _refittingBF = false; });
    // Stats live in an HTML sibling of the plot — selectable, and
    // not picked up by Plotly.toImage when copying the plot.
    const box = $('exFitBox');
    if (box) {
        if (fitBoxText) {
            box.style.display = '';
            box.textContent = fitBoxText;
        } else {
            box.style.display = 'none';
            box.textContent = '';
        }
    }
    $('exInfo').textContent = baseText;
}

function _wireBestFitListener() {
    const div = $('explorePlot');
    if (!div || div._bfListenerAttached) return;
    div._bfListenerAttached = true;
    div.on('plotly_restyle', () => {
        if (_refittingBF) return;        // our own restyle — ignore
        _refitBestFitFromVisible();
    });
}

function renderBar() {
    const X = _resolve('X');
    const key = X.key;
    const groups = _data.groups;
    const byGroup = {};
    groups.forEach(g => { byGroup[g] = []; });
    _activeSubjects().forEach(s => {
        if (byGroup[s.group] && _val(s, key) != null) byGroup[s.group].push(s);
    });

    const means = [], sems = [], labels = [], colors = [];
    let n = 0;
    groups.forEach(g => {
        const vals = byGroup[g].map(s => _val(s, key));
        n += vals.length;
        if (vals.length) {
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const sd = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length);
            means.push(mean); sems.push(sd / Math.sqrt(vals.length));
        } else { means.push(0); sems.push(0); }
        labels.push(g); colors.push(GROUP_COLORS[g] || '#999');
    });

    const barTrace = {
        x: labels.map((_, i) => i), y: means, customdata: labels, type: 'bar',
        marker: { color: colors, opacity: 0.3 },
        // Whiskers twice as thick.
        error_y: { type: 'data', array: sems, visible: true, color: '#666', thickness: 3 },
        hovertemplate: '%{customdata}<br>Mean: %{y:.3f}<extra></extra>',
        width: 0.6, showlegend: false,
    };

    const dotX = [], dotY = [], dotText = [], dotColors = [];
    groups.forEach((g, gi) => {
        const subs = byGroup[g];
        subs.forEach((s, si) => {
            const jitter = subs.length > 1 ? -0.22 + (si / (subs.length - 1)) * 0.44 : 0;
            dotX.push(gi + jitter); dotY.push(_val(s, key)); dotText.push(s.name);
            dotColors.push(GROUP_COLORS[g] || '#999');
        });
    });
    const dotTrace = {
        x: dotX, y: dotY, text: dotText, type: 'scatter', mode: 'markers',
        marker: { color: dotColors, size: 16, opacity: 0.85, line: { color: '#333', width: 1 } },
        hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>', showlegend: false,
    };

    $('exInfo').textContent = `n = ${n}`;
    // ANOVA + post-hoc Welch's t pairwise — populates the side text box.
    const fitBox = $('exFitBox');
    const anovaOn = $('exAnova') && $('exAnova').checked;
    if (fitBox) {
        if (anovaOn) {
            const groupVals = groups.map(g => byGroup[g].map(s => _val(s, key)));
            const res = _oneWayANOVA(groupVals);
            if (!res) {
                fitBox.style.display = '';
                fitBox.textContent = 'ANOVA: not enough data';
            } else {
                const lines = [
                    `ANOVA F(${res.df1}, ${res.df2}) = ${res.F.toFixed(2)}`,
                    `      ${_fmtP(res.p)}`,
                ];
                if (res.p < 0.05) {
                    lines.push('', 'Post-hoc (Welch):');
                    const idx = groups.map((g, i) => [g, i]).filter(([g, i]) => groupVals[i].length >= 2);
                    for (let i = 0; i < idx.length; i++) {
                        for (let j = i + 1; j < idx.length; j++) {
                            const [ga, ia] = idx[i], [gb, ib] = idx[j];
                            const t = _welchT(groupVals[ia], groupVals[ib]);
                            if (t) lines.push(`  ${ga} vs ${gb}: ${_fmtP(t.p)}`);
                        }
                    }
                }
                fitBox.style.display = '';
                fitBox.textContent = lines.join('\n');
            }
        } else {
            fitBox.style.display = 'none';
            fitBox.textContent = '';
        }
    }
    // Bar's "X" variable is plotted on the y-axis; honour the X min/max
    // inputs by piping them into yaxis.range.
    const allValues = [];
    for (const g in byGroup) byGroup[g].forEach(s => allValues.push(_val(s, key)));
    const yRange = _readRangeInputs('exXMin', 'exXMax', allValues.filter(v => v != null && isFinite(v)));
    const layout = {
        margin: { t: 20, b: 70, l: _leftMarginForFixedYBaseline(), r: 20 },
        xaxis: {
            tickvals: groups.map((_, i) => i), ticktext: groups, color: '#666',
            tickfont: { size: 22 },
            showline: true, linecolor: '#666', linewidth: 2, zeroline: false,
            automargin: true,
        },
        yaxis: {
            title: { text: X.label, font: { size: 24 }, standoff: 14 },
            color: '#666', gridcolor: '#f0f0f0',
            tickfont: { size: 22 },
            showline: true, linecolor: '#666', linewidth: 2, zeroline: false,
            automargin: false,
            ...(yRange ? { range: yRange, autorange: false } : { autorange: true }),
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff', bargap: 0.5,
    };
    Plotly.newPlot('explorePlot', [barTrace, dotTrace], layout, { responsive: true, displayModeBar: false })
        .then(_applyAutoRangeInputs);
}

// ── Listeners ──────────────────────────────────────────────────
['exSourceSelect', 'exSeqModeSelect', 'exHandSelect', 'exTrialSelect'].forEach(id =>
    $(id).addEventListener('change', loadExplore));

// Include-auto: toggle only the default-checked state of subjects
// without complete saved events (no re-fetch — auto data is already
// loaded).  Matches the Group Comparison page.
$('exIncludeAuto').addEventListener('change', () => {
    const inc = $('exIncludeAuto').checked;
    (_data?.subjects || []).forEach(s => {
        if (!s.has_complete_events) _subjChecked[s.name] = inc;
    });
    render();
});

$('exSelectAllBtn').addEventListener('click', () => {
    const inc = $('exIncludeAuto').checked;
    (_data?.subjects || []).forEach(s => {
        _subjChecked[s.name] = s.has_complete_events ? true : inc;
    });
    render();
});

document.querySelectorAll('input[name="exPlotType"]').forEach(r =>
    r.addEventListener('change', render));
// Variable changes clear the axis-range inputs (units may differ).
['exVarX', 'exVarY'].forEach(id =>
    $(id).addEventListener('change', () => {
        if (id === 'exVarX') { $('exXMin').value = ''; $('exXMax').value = ''; }
        if (id === 'exVarY') { $('exYMin').value = ''; $('exYMax').value = ''; }
        render();
    }));
// Axis-range textboxes — re-render on commit (change fires on blur /
// Enter for number inputs).
['exXMin','exXMax','exYMin','exYMax'].forEach(id =>
    $(id).addEventListener('change', render));
// Toggling Slope doesn't change the underlying data — just restyle
// the already-present best-fit trace in place so axes don't re-fit.
$('exBestFit').addEventListener('change', _refitBestFitFromVisible);
$('exAnova').addEventListener('change', render);
function _resetRange(minId, maxId) {
    const minEl = $(minId), maxEl = $(maxId);
    if (minEl) minEl.value = '';
    if (maxEl) maxEl.value = '';
    render();
}
$('exXReset').addEventListener('click', () => _resetRange('exXMin', 'exXMax'));
$('exYReset').addEventListener('click', () => _resetRange('exYMin', 'exYMax'));
// Legend visibility — show/hide live on the existing plot.  Same
// flag is read at copy time, so the exported PNG matches the screen.
$('exLegend').addEventListener('change', () => {
    const div = $('explorePlot');
    if (div && window.Plotly && div._fullLayout) {
        try {
            Plotly.relayout(div, { showlegend: $('exLegend').checked });
        } catch (_) {}
    }
});

// ── Copy to clipboard ─────────────────────────────────────────
$('exCopyBtn').addEventListener('click', async () => {
    const btn = $('exCopyBtn');
    const div = $('explorePlot');
    if (!div || !window.Plotly || btn.disabled) return;

    // Visual feedback: darken + disable for ≥ 1 s.
    btn.disabled = true;
    btn.style.opacity = '0.4';
    btn.style.filter = 'brightness(0.7)';
    const minHold = new Promise(r => setTimeout(r, 1000));

    try {
        const SCALE = 2;
        // The Legend checkbox controls both the on-screen plot and the
        // copied image, so the live div already reflects the user's
        // chosen view — no layout overrides needed.
        const url = await Plotly.toImage(div, {
            format: 'png',
            width:  div.clientWidth  || 900,
            height: div.clientHeight || 520,
            scale:  SCALE,
        });
        // Build a File with a sensible name so Finder paste names the
        // file something useful.  Pattern: explore_<kind>_<labels>.png
        const plotType = $('exPlotType').value;
        const xVar = $('exVarX').value || 'x';
        const yVar = $('exVarY').value || 'y';
        const stemRaw = (plotType === 'scatter')
            ? `explore_${xVar}_vs_${yVar}`
            : `explore_${xVar}_by_group`;
        const stem = stemRaw.replace(/[^A-Za-z0-9_-]+/g, '_');

        // toImage returns a data URL — convert to blob via fetch().
        const blob = await (await fetch(url)).blob();
        const file = new File([blob], stem + '.png', { type: 'image/png' });
        await navigator.clipboard.write([
            new ClipboardItem({ 'image/png': file }),
        ]);
    } catch (err) {
        console.error('Copy failed:', err);
    } finally {
        await minHold;
        btn.disabled = false;
        btn.style.opacity = '';
        btn.style.filter = '';
    }
});

// Resize the Plotly chart whenever the user drags the bottom-right
// corner of #explorePlot (enabled via CSS `resize: both`).  Plotly's
// fonts/markers are pixel-sized, so they stay constant — only the
// plot area expands.  Redraws live during the drag (ResizeObserver
// fires continuously, not just on mouse-up).
(function _watchExplorePlotResize() {
    const div = $('explorePlot');
    if (!div || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
        // _fullLayout is set on any element Plotly has plotted into;
        // before that, there's nothing to resize.
        if (window.Plotly && div._fullLayout) {
            try { Plotly.Plots.resize(div); } catch (_) {}
        }
    });
    ro.observe(div);
})();

loadExplore();
