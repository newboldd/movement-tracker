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
    const inc = $('exIncludeAuto').checked ? 'true' : 'false';
    plot.innerHTML = '<div class="results-no-data">Loading…</div>';
    try {
        _data = await API.get(
            `/api/results/explore?include_auto=${inc}&source=${src}&seq_mode=${sm}`);
        _varMeta = {};
        _data.variables.forEach(v => {
            _varMeta[v.key] = { label: v.label, aggregatable: !!v.aggregatable };
        });
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
    const cur = host.querySelector('input:checked')?.value || 'mean';
    host.innerHTML = AGGS.map(a =>
        `<label style="display:inline-flex;align-items:center;gap:2px;cursor:pointer;">
            <input type="radio" name="${name}" value="${a.v}" ${a.v === cur ? 'checked' : ''}> ${a.label}
        </label>`).join('');
    host.querySelectorAll('input').forEach(r => r.addEventListener('change', render));
}

function _agg(prefix) {
    const host = $(prefix === 'X' ? 'exXAgg' : 'exYAgg');
    return host.querySelector('input:checked')?.value || 'mean';
}

// Effective data key + display label for a select, applying the
// aggregate radio when the variable is aggregatable.
function _resolve(prefix) {
    const sel = $(prefix === 'X' ? 'exVarX' : 'exVarY');
    const base = sel.value;
    const meta = _varMeta[base] || { label: base, aggregatable: false };
    if (!meta.aggregatable) return { key: base, label: meta.label };
    const agg = _agg(prefix);
    return { key: `${agg}_${base}`, label: `${AGG_PREFIX[agg]}${meta.label}` };
}

function render() {
    if (!_data) return;
    const mode = $('exPlotType').value;
    $('exYLabel').style.display = (mode === 'scatter') ? '' : 'none';
    $('exXLabel').childNodes[0].textContent = (mode === 'scatter') ? 'X: ' : 'Variable: ';
    _renderAggRadios('X');
    if (mode === 'scatter') { _renderAggRadios('Y'); renderScatter(); }
    else { $('exYAgg').innerHTML = ''; renderBar(); }
}

function _val(s, key) {
    const v = s.vars[key];
    return (v != null && isFinite(v)) ? v : null;
}

function renderScatter() {
    const X = _resolve('X'), Y = _resolve('Y');
    const groups = _data.groups;
    let n = 0;
    const traces = groups.map(g => {
        const subs = _data.subjects.filter(s =>
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
    $('exInfo').textContent = `${n} subjects plotted (missing either variable excluded)`;
    const layout = {
        margin: { t: 20, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: X.label, font: { size: 12 } }, color: '#666', gridcolor: '#f0f0f0' },
        yaxis: { title: { text: Y.label, font: { size: 12 } }, color: '#666', gridcolor: '#f0f0f0' },
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
    _data.subjects.forEach(s => {
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
        xaxis: { tickvals: groups.map((_, i) => i), ticktext: groups, color: '#666' },
        yaxis: { title: { text: X.label, font: { size: 12 } }, color: '#666', gridcolor: '#f0f0f0' },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff', bargap: 0.5,
    };
    Plotly.newPlot('explorePlot', [barTrace, dotTrace], layout, { responsive: true, displayModeBar: false });
}

// ── Listeners ──────────────────────────────────────────────────
['exSourceSelect', 'exSeqModeSelect'].forEach(id =>
    $(id).addEventListener('change', loadExplore));
$('exIncludeAuto').addEventListener('change', loadExplore);
['exPlotType', 'exVarX', 'exVarY'].forEach(id =>
    $(id).addEventListener('change', render));

loadExplore();
