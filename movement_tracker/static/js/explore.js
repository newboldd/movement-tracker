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

const AGGS = [
    { v: 'mean', label: 'Mean' },
    { v: 'cv', label: 'CV' },
    { v: 'seq', label: 'Sequence effect' },
];
const AGG_PREFIX = { mean: 'Mean ', cv: 'CV ', seq: 'Seq. Effect ' };

const $ = (id) => document.getElementById(id);

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
    const curM = host.querySelector(`input[name="${mname}"]:checked`)?.value || 'r2';
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
        html += `<div class="gsl-group"><span class="gsl-group-label">${g}</span><div class="gsl-tags">`;
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
}

function _activeSubjects() {
    if (!_data || !_data.subjects) return [];
    return _data.subjects.filter(s => _subjChecked[s.name]);
}

function render() {
    if (!_data) return;
    _renderSubjectList();
    const mode = $('exPlotType').value;
    $('exYLabel').style.display = (mode === 'scatter') ? '' : 'none';
    $('exXLabel').childNodes[0].textContent = (mode === 'scatter') ? 'X: ' : 'Variable: ';
    // "Best fit" only applies to scatter.
    const bestFitLbl = $('exBestFitLabel');
    if (bestFitLbl) bestFitLbl.style.display = (mode === 'scatter') ? '' : 'none';
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
            marker: { color: GROUP_COLORS[g] || '#999', size: 9, opacity: 0.8,
                      line: { color: '#333', width: 0.5 } },
            hovertemplate: `%{text}<br>${X.label}: %{x:.3f}<br>${Y.label}: %{y:.3f}<extra>${g}</extra>`,
        };
    });
    let infoText = `${n} subjects plotted (missing either variable excluded)`;
    // Optional linear best-fit line across all visible points.
    if ($('exBestFit') && $('exBestFit').checked) {
        const allX = [], allY = [];
        traces.forEach(t => { for (let i = 0; i < t.x.length; i++) {
            allX.push(t.x[i]); allY.push(t.y[i]);
        }});
        const stats = _linRegStats(allX, allY);
        if (stats) {
            const xMin = Math.min(...allX), xMax = Math.max(...allX);
            traces.push({
                x: [xMin, xMax],
                y: [stats.intercept + stats.slope * xMin,
                    stats.intercept + stats.slope * xMax],
                type: 'scatter', mode: 'lines', name: 'Best fit',
                line: { color: '#000', width: 1.5 },
                hoverinfo: 'skip', showlegend: false,
            });
            const pStr = (stats.p < 1e-4)
                ? stats.p.toExponential(2)
                : stats.p.toFixed(4);
            infoText += `  ·  slope=${stats.slope.toPrecision(3)},`
                     + `  R²=${stats.r2.toFixed(3)},`
                     + `  p=${pStr}`;
        }
    }
    $('exInfo').textContent = infoText;
    const layout = {
        margin: { t: 20, b: 50, l: 60, r: 20 },
        xaxis: {
            title: { text: X.label, font: { size: 12 } },
            color: '#666', gridcolor: '#f0f0f0',
            showline: true, linecolor: '#666', linewidth: 1, mirror: false, zeroline: false,
        },
        yaxis: {
            title: { text: Y.label, font: { size: 12 } },
            color: '#666', gridcolor: '#f0f0f0',
            showline: true, linecolor: '#666', linewidth: 1, mirror: false, zeroline: false,
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        legend: { orientation: 'h', y: 1.08, font: { size: 11 } },
        hovermode: 'closest',
    };
    Plotly.newPlot('explorePlot', traces, layout, { responsive: true, displayModeBar: false });
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
        error_y: { type: 'data', array: sems, visible: true, color: '#666', thickness: 1.5 },
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
        marker: { color: dotColors, size: 8, opacity: 0.85, line: { color: '#333', width: 0.5 } },
        hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>', showlegend: false,
    };

    $('exInfo').textContent = `${n} subjects (missing variable excluded)`;
    const layout = {
        margin: { t: 20, b: 40, l: 60, r: 20 },
        xaxis: {
            tickvals: groups.map((_, i) => i), ticktext: groups, color: '#666',
            showline: true, linecolor: '#666', linewidth: 1, zeroline: false,
        },
        yaxis: {
            title: { text: X.label, font: { size: 12 } }, color: '#666', gridcolor: '#f0f0f0',
            showline: true, linecolor: '#666', linewidth: 1, zeroline: false,
        },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff', bargap: 0.5,
    };
    Plotly.newPlot('explorePlot', [barTrace, dotTrace], layout, { responsive: true, displayModeBar: false });
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

['exPlotType', 'exVarX', 'exVarY', 'exBestFit'].forEach(id =>
    $(id).addEventListener('change', render));

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
        const includeLegend = !!($('exCopyIncludeLegend') && $('exCopyIncludeLegend').checked);
        // Passing {data, layout} (instead of the div) lets us override
        // `showlegend` for the export without disturbing the displayed
        // plot.  Same is true for the plot data — we copy the live
        // arrays so the snapshot reflects the current selection.
        const source = (!includeLegend && div.data && div.layout)
            ? { data: div.data, layout: { ...div.layout, showlegend: false } }
            : div;
        const url = await Plotly.toImage(source, {
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
