/* Results page: distance traces, movement parameters, group comparison */

let subjects = [];
let currentTab = 'distances';
let currentSubjectId = null;

// Cached data
let cachedTraces = null;
let cachedMovements = null;
let cachedGroup = null;

const PARAM_LABELS = {
    imi: 'Inter-Movement Interval (s)',
    amplitude: 'Amplitude (mm)',
    peak_open_vel: 'Peak Opening Velocity (mm/s)',
    peak_close_vel: 'Peak Closing Velocity (mm/s)',
    mean_open_vel: 'Mean Opening Velocity (mm/s)',
    mean_close_vel: 'Mean Closing Velocity (mm/s)',
};

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
            opt.textContent = `${s.name} (${s.diagnosis || 'Control'})`;
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

// ── Tab switching ──────────────────────────────────────────────

function switchTab(tab) {
    currentTab = tab;

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
    if (tab === 'distances' && currentSubjectId) {
        loadDistances(currentSubjectId);
    } else if (tab === 'movements' && currentSubjectId) {
        loadMovements(currentSubjectId);
    } else if (tab === 'group') {
        loadGroup();
    }
}

// ── Distances Tab ──────────────────────────────────────────────

async function loadDistances(subjectId) {
    const container = document.getElementById('distancePlots');

    if (!subjectId) {
        container.innerHTML = '<div class="results-no-data">Select a subject to view distance traces</div>';
        return;
    }

    container.innerHTML = '<div class="results-no-data">Loading...</div>';

    try {
        const data = await API.get(`/api/results/${subjectId}/traces`);
        cachedTraces = data;

        if (!data.trials || data.trials.length === 0) {
            container.innerHTML = '<div class="results-no-data">No distance data available for this subject</div>';
            return;
        }

        container.innerHTML = '';
        data.trials.forEach((trial, idx) => {
            const div = document.createElement('div');
            div.id = `distPlot_${idx}`;
            div.style.width = '100%';
            div.style.height = '300px';
            div.style.marginBottom = '8px';
            container.appendChild(div);

            renderDistancePlot(div.id, trial);
        });
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="color:#d32f2f;">${e.message}</div>`;
    }
}

function renderDistancePlot(divId, trial) {
    const fps = trial.fps || 60;
    const n = trial.distances.length;
    const times = Array.from({ length: n }, (_, i) => +(i / fps).toFixed(3));

    const distTrace = {
        x: times,
        y: trial.distances,
        type: 'scattergl',
        mode: 'lines',
        name: 'Distance (mm)',
        line: { color: '#2196F3', width: 1.2 },
        yaxis: 'y',
        hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm<extra></extra>',
    };

    const velTrace = {
        x: times,
        y: trial.velocities,
        type: 'scattergl',
        mode: 'lines',
        name: 'Velocity (mm/s)',
        line: { color: '#4CAF50', width: 1 },
        yaxis: 'y2',
        hovertemplate: '%{x:.2f}s<br>%{y:.1f} mm/s<extra></extra>',
    };

    const layout = {
        title: { text: trial.name, font: { size: 13, color: '#666' } },
        margin: { t: 30, b: 35, l: 55, r: 55 },
        xaxis: { title: '', color: '#666', gridcolor: '#eee' },
        yaxis: {
            title: { text: 'Distance (mm)', font: { size: 11, color: '#2196F3' } },
            color: '#2196F3',
            gridcolor: '#f0f0f0',
            zeroline: false,
        },
        yaxis2: {
            title: { text: 'Velocity (mm/s)', font: { size: 11, color: '#4CAF50' } },
            overlaying: 'y',
            side: 'right',
            color: '#4CAF50',
            gridcolor: 'transparent',
            zeroline: true,
            zerolinecolor: '#ddd',
        },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: false,
        hovermode: 'x unified',
    };

    Plotly.newPlot(divId, [distTrace, velTrace], layout, {
        responsive: true,
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
        const data = await API.get(`/api/results/${subjectId}/movements`);
        cachedMovements = data;
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

    const showSeq = document.getElementById('sequenceToggle').checked;
    container.innerHTML = '';

    params.forEach(param => {
        const div = document.createElement('div');
        div.id = `movPlot_${param}`;
        div.style.height = '320px';
        container.appendChild(div);

        renderMovementScatter(div.id, data, param, showSeq);
    });
}

function renderMovementScatter(divId, data, param, showSequence) {
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
    Object.keys(byTrial).sort((a, b) => +a - +b).forEach(ti => {
        const ms = byTrial[ti];
        const x = ms.map(m => m.peak_time);
        const y = ms.map(m => m[param]);

        const color = TRIAL_COLORS[ti % TRIAL_COLORS.length];

        traces.push({
            x, y,
            type: 'scatter',
            mode: 'markers',
            name: trialNames[ti] || `Trial ${+ti + 1}`,
            marker: { color, size: 7, opacity: 0.8 },
            hovertemplate: `t=%{x:.2f}s<br>${PARAM_LABELS[param]}: %{y:.2f}<extra></extra>`,
        });

        // Sequence effect: linear regression line
        if (showSequence) {
            const pairs = x.map((xi, i) => [xi, y[i]]).filter(p => p[0] != null && p[1] != null);
            if (pairs.length >= 2) {
                const xs = pairs.map(p => p[0]);
                const ys = pairs.map(p => p[1]);
                const { slope, intercept } = linearRegression(xs, ys);
                const xMin = Math.min(...xs);
                const xMax = Math.max(...xs);
                traces.push({
                    x: [xMin, xMax],
                    y: [slope * xMin + intercept, slope * xMax + intercept],
                    type: 'scatter',
                    mode: 'lines',
                    name: `Trend (Trial ${+ti + 1})`,
                    line: { color, width: 2, dash: 'dash' },
                    showlegend: false,
                    hoverinfo: 'skip',
                });
            }
        }
    });

    const layout = {
        title: { text: PARAM_LABELS[param], font: { size: 13, color: '#444' } },
        margin: { t: 35, b: 40, l: 55, r: 20 },
        xaxis: { title: { text: 'Time (s)', font: { size: 11 } }, color: '#666', gridcolor: '#eee' },
        yaxis: { title: '', color: '#666', gridcolor: '#f0f0f0' },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        showlegend: true,
        legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
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

// ── Group Comparison Tab ───────────────────────────────────────

let highlightedSubject = null;

async function loadGroup() {
    const container = document.getElementById('groupPlots');

    if (cachedGroup) {
        renderGroupPlots();
        return;
    }

    container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">Loading group data...</div>';

    try {
        cachedGroup = await API.get('/api/results/group');
        renderGroupPlots();
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="grid-column:1/-1;color:#d32f2f;">${e.message}</div>`;
    }
}

const GROUP_PARAMS = [
    { key: 'mean_imi', label: 'Mean IMI (s)' },
    { key: 'cv_imi', label: 'CV of IMI' },
    { key: 'seq_imi', label: 'Sequence Effect: IMI' },
    { key: 'frequency', label: 'Frequency (Hz)' },
    { key: 'mean_amplitude', label: 'Mean Amplitude (mm)' },
    { key: 'cv_amplitude', label: 'CV of Amplitude' },
    { key: 'seq_amplitude', label: 'Sequence Effect: Amplitude' },
    { key: 'mean_peak_open_vel', label: 'Mean Peak Opening Vel. (mm/s)' },
    { key: 'cv_peak_open_vel', label: 'CV Peak Opening Vel.' },
    { key: 'mean_peak_close_vel', label: 'Mean Peak Closing Vel. (mm/s)' },
    { key: 'cv_peak_close_vel', label: 'CV Peak Closing Vel.' },
    { key: 'mean_mean_open_vel', label: 'Mean Avg Opening Vel. (mm/s)' },
    { key: 'cv_mean_open_vel', label: 'CV Avg Opening Vel.' },
    { key: 'mean_mean_close_vel', label: 'Mean Avg Closing Vel. (mm/s)' },
    { key: 'cv_mean_close_vel', label: 'CV Avg Closing Vel.' },
];

const GROUP_COLORS = {
    Control: '#4CAF50',
    MSA: '#FF5722',
    PD: '#2196F3',
    PSP: '#9C27B0',
};

function renderGroupPlots() {
    const container = document.getElementById('groupPlots');
    const data = cachedGroup;

    if (!data || !data.subjects || data.subjects.length === 0) {
        container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">No group data available</div>';
        return;
    }

    container.innerHTML = '';

    GROUP_PARAMS.forEach(({ key, label }) => {
        const div = document.createElement('div');
        div.id = `grpPlot_${key}`;
        div.style.height = '320px';
        container.appendChild(div);

        renderGroupBar(div.id, data, key, label);
    });
}

function renderGroupBar(divId, data, paramKey, paramLabel) {
    const groups = data.groups;
    const subjects = data.subjects;

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
        x: groupLabels,
        y: groupMeans,
        type: 'bar',
        marker: { color: groupLabels.map(g => GROUP_COLORS[g] || '#999'), opacity: 0.6 },
        error_y: { type: 'data', array: groupSems, visible: true, color: '#666' },
        hovertemplate: '%{x}<br>Mean: %{y:.3f}<extra></extra>',
        showlegend: false,
    };

    // Individual dots with jitter
    const dotX = [];
    const dotY = [];
    const dotText = [];
    const dotColors = [];
    const dotSizes = [];
    const dotOpacities = [];

    groups.forEach((g, gi) => {
        const subs = byGroup[g];
        subs.forEach((s, si) => {
            const val = s[paramKey];
            if (val == null || !isFinite(val)) return;

            // Jitter: spread dots within ±0.25 of the bar center
            const jitter = subs.length > 1
                ? -0.25 + (si / (subs.length - 1)) * 0.5
                : 0;
            dotX.push(gi + jitter);
            dotY.push(val);
            dotText.push(s.name);
            dotColors.push(GROUP_COLORS[g] || '#999');

            const isHighlighted = highlightedSubject === s.name;
            dotSizes.push(isHighlighted ? 12 : 8);
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
            line: { color: '#333', width: dotSizes.map(s => s > 8 ? 2 : 0.5) },
        },
        hovertemplate: '%{text}<br>%{y:.3f}<extra></extra>',
        showlegend: false,
    };

    const layout = {
        title: { text: paramLabel, font: { size: 13, color: '#444' } },
        margin: { t: 35, b: 40, l: 55, r: 20 },
        xaxis: {
            tickvals: groups.map((_, i) => i),
            ticktext: groups,
            color: '#666',
        },
        yaxis: { title: '', color: '#666', gridcolor: '#f0f0f0' },
        plot_bgcolor: '#fff',
        paper_bgcolor: '#fff',
        bargap: 0.4,
    };

    Plotly.newPlot(divId, [barTrace, dotTrace], layout, {
        responsive: true,
        displayModeBar: false,
    });

    // Click handler on dots (curveNumber 1 = dot trace, 0 = bar trace)
    const plotDiv = document.getElementById(divId);
    plotDiv.on('plotly_click', (eventData) => {
        if (eventData.points && eventData.points.length > 0) {
            const pt = eventData.points[0];
            if (pt.curveNumber === 1 && pt.text) {
                highlightSubject(pt.text);
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
    cachedTraces = null;
    cachedMovements = null;

    if (currentTab === 'distances') {
        loadDistances(currentSubjectId);
    } else if (currentTab === 'movements') {
        loadMovements(currentSubjectId);
    }
});

document.getElementById('prevSubjectBtn').addEventListener('click', () => navigateSubject(-1));
document.getElementById('nextSubjectBtn').addEventListener('click', () => navigateSubject(1));

document.querySelectorAll('#tabSwitcher .btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Movement controls: re-render on checkbox change
document.querySelectorAll('#movementControls input').forEach(cb => {
    cb.addEventListener('change', () => {
        if (cachedMovements) renderMovementPlots();
    });
});

// ── Init ───────────────────────────────────────────────────────

const params = new URLSearchParams(window.location.search);
const initSubject = params.get('subject');
const initTab = params.get('tab');

loadSubjects().then(() => {
    if (initSubject) {
        document.getElementById('subjectSelect').value = initSubject;
        currentSubjectId = initSubject;
    }
    if (initTab && ['distances', 'movements', 'group'].includes(initTab)) {
        switchTab(initTab);
    } else if (currentSubjectId) {
        loadDistances(currentSubjectId);
    }
});
