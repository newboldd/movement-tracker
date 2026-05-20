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
let _varLabel = {};      // key -> label

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
        _varLabel = {};
        _data.variables.forEach(v => { _varLabel[v.key] = v.label; });
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
    // Preserve current selections if still valid; else sensible defaults.
    const keys = _data.variables.map(v => v.key);
    const curX = keys.includes(xSel.value) ? xSel.value
        : (keys.includes('mean_amplitude') ? 'mean_amplitude' : keys[0]);
    const curY = keys.includes(ySel.value) ? ySel.value
        : (keys.includes('mean_peak_open_vel') ? 'mean_peak_open_vel' : keys[Math.min(1, keys.length - 1)]);
    xSel.innerHTML = _optionsHtml(curX);
    ySel.innerHTML = _optionsHtml(curY);
}

function render() {
    if (!_data) return;
    const mode = $('exPlotType').value;
    // Show/hide the Y selector for bar mode.
    $('exYLabel').style.display = (mode === 'scatter') ? '' : 'none';
    $('exXLabel').querySelector('select').previousSibling; // no-op
    $('exXLabel').childNodes[0].textContent = (mode === 'scatter') ? 'X: ' : 'Variable: ';
    if (mode === 'scatter') renderScatter();
    else renderBar();
}

function _val(s, key) {
    const v = s.vars[key];
    return (v != null && isFinite(v)) ? v : null;
}

function renderScatter() {
    const xKey = $('exVarX').value, yKey = $('exVarY').value;
    const groups = _data.groups;
    let n = 0;
    const traces = groups.map(g => {
        const subs = _data.subjects.filter(s =>
            s.group === g && _val(s, xKey) != null && _val(s, yKey) != null);
        n += subs.length;
        return {
            x: subs.map(s => _val(s, xKey)),
            y: subs.map(s => _val(s, yKey)),
            text: subs.map(s => s.name),
            type: 'scatter', mode: 'markers', name: g,
            marker: { color: GROUP_COLORS[g] || '#999', size: 9, opacity: 0.8,
                      line: { color: '#333', width: 0.5 } },
            hovertemplate: `%{text}<br>${_varLabel[xKey]}: %{x:.3f}<br>${_varLabel[yKey]}: %{y:.3f}<extra>${g}</extra>`,
        };
    });
    $('exInfo').textContent = `${n} subjects plotted (missing either variable excluded)`;
    const layout = {
        margin: { t: 20, b: 50, l: 60, r: 20 },
        xaxis: { title: { text: _varLabel[xKey], font: { size: 12 } }, color: '#666', gridcolor: '#f0f0f0' },
        yaxis: { title: { text: _varLabel[yKey], font: { size: 12 } }, color: '#666', gridcolor: '#f0f0f0' },
        plot_bgcolor: '#fff', paper_bgcolor: '#fff',
        legend: { orientation: 'h', y: 1.08, font: { size: 11 } },
        hovermode: 'closest',
    };
    Plotly.newPlot('explorePlot', traces, layout, { responsive: true, displayModeBar: false });
}

function renderBar() {
    const key = $('exVarX').value;
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
        yaxis: { title: { text: _varLabel[key], font: { size: 12 } }, color: '#666', gridcolor: '#f0f0f0' },
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
