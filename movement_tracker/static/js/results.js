/* Results page: distance traces, movement parameters, group comparison */

let subjects = [];
let currentTab = 'distances';
let currentSubjectId = null;

// Cached data
let cachedTraces = null;
let cachedMovements = null;
let cachedGroup = null;
let cachedSequenceAssignments = null; // { byTrial: { trialIdx: {sequences, seq_r2} }, totalSeqs, totalR2 }

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

    try {
        // Load traces and movements in parallel
        const [traceData, movData] = await Promise.all([
            API.get(`/api/results/${subjectId}/traces`),
            API.get(`/api/results/${subjectId}/movements`).catch(() => null),
        ]);

        cachedTraces = traceData;

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

    // Pre-compute sequence assignments if multi-seq mode
    if (seqMode.endsWith('_multi') && data.movements.length > 0) {
        if (!cachedSequenceAssignments) {
            cachedSequenceAssignments = computeSequenceAssignments(data);
        }
        const el = document.getElementById('distSeqInfo');
        if (el && cachedSequenceAssignments.totalSeqs > 0) {
            el.textContent = `${cachedSequenceAssignments.totalSeqs} sequence${cachedSequenceAssignments.totalSeqs !== 1 ? 's' : ''} | R\u00b2 = ${cachedSequenceAssignments.totalR2.toFixed(3)}`;
        } else if (el) {
            el.textContent = 'No sequences detected';
        }
    } else {
        const el = document.getElementById('distSeqInfo');
        if (el) el.textContent = '';
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

function getYAxisConfig() {
    const locked = document.getElementById('lockYAxis').checked;
    const minInput = document.getElementById('yAxisMin');
    const maxInput = document.getElementById('yAxisMax');
    const customMin = minInput.value !== '' ? parseFloat(minInput.value) : null;
    const customMax = maxInput.value !== '' ? parseFloat(maxInput.value) : null;
    return { locked, customMin, customMax };
}

function renderAllDistancePlots() {
    const container = document.getElementById('distancePlots');
    const data = cachedTraces;
    if (!data || !data.trials) return;

    container.innerHTML = '';

    // Show data source badge if using MediaPipe fallback
    if (data.data_source && data.data_source !== 'dlc') {
        const badge = document.createElement('div');
        badge.style.cssText = 'padding:6px 12px;margin-bottom:8px;border-radius:4px;font-size:12px;display:inline-block;';
        if (data.data_source === 'mediapipe') {
            badge.style.background = 'rgba(255,152,0,0.15)';
            badge.style.color = 'var(--orange)';
            badge.textContent = 'Showing MediaPipe prelabel data (no DLC labels available)';
        }
        container.appendChild(badge);
    }

    const yConfig = getYAxisConfig();
    const yRange = data.y_range;

    // Determine y-axis ranges
    let distRange = null;
    let velRange = null;
    if (yConfig.locked && yRange) {
        const dMin = yConfig.customMin !== null ? yConfig.customMin : yRange.dist_min;
        const dMax = yConfig.customMax !== null ? yConfig.customMax : yRange.dist_max;
        const pad = (dMax - dMin) * 0.05;
        distRange = [dMin - pad, dMax + pad];

        const vPad = (yRange.vel_max - yRange.vel_min) * 0.05;
        velRange = [yRange.vel_min - vPad, yRange.vel_max + vPad];
    }

    // Build per-trial overlay data from movements
    const overlayPeakDist = document.getElementById('overlayPeakDist').checked;
    const overlayPeakOpenVel = document.getElementById('overlayPeakOpenVel').checked;
    const overlayPeakCloseVel = document.getElementById('overlayPeakCloseVel').checked;
    const showSequences = document.getElementById('overlaySequences').checked;

    // Compute sequence assignments if needed
    let seqAssignments = null;
    if (showSequences && cachedMovements && cachedMovements.movements && cachedMovements.movements.length > 0) {
        if (!cachedSequenceAssignments) {
            cachedSequenceAssignments = computeSequenceAssignments(cachedMovements);
        }
        seqAssignments = cachedSequenceAssignments;
        const r2El = document.getElementById('overlayR2');
        if (r2El && seqAssignments.totalSeqs > 0) {
            r2El.textContent = `R\u00b2 = ${seqAssignments.totalR2.toFixed(3)}`;
        } else if (r2El) {
            r2El.textContent = '';
        }
    } else {
        const r2El = document.getElementById('overlayR2');
        if (r2El) r2El.textContent = '';
    }

    // Group movements by trial
    const movByTrial = {};
    if (cachedMovements && cachedMovements.movements) {
        cachedMovements.movements.forEach((m, mi) => {
            const ti = m.trial_idx;
            if (!movByTrial[ti]) movByTrial[ti] = [];
            movByTrial[ti].push({ ...m, _idx: mi });
        });
    }

    data.trials.forEach((trial, idx) => {
        // Wrapper for each trial (enables horizontal scrolling)
        const wrapper = document.createElement('div');
        wrapper.style.marginBottom = '16px';
        wrapper.style.overflowX = 'auto';

        // Distance plot
        const distDiv = document.createElement('div');
        distDiv.id = `distPlot_${idx}`;
        distDiv.style.height = '220px';
        wrapper.appendChild(distDiv);

        // Velocity plot
        const velDiv = document.createElement('div');
        velDiv.id = `velPlot_${idx}`;
        velDiv.style.height = '150px';
        wrapper.appendChild(velDiv);

        container.appendChild(wrapper);

        // Compute plot width based on 20s per screen width
        const fps = trial.fps || 60;
        const durationSec = trial.distances.length / fps;
        const containerWidth = container.clientWidth || 1200;
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
                if (m.peak_open_vel != null && m.peak_frame != null) {
                    ts.push(+((m.peak_frame - trialStart) / fps).toFixed(3));
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
                if (m.peak_close_vel != null && m.peak_frame != null) {
                    ts.push(+((m.peak_frame - trialStart) / fps).toFixed(3));
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

        renderDistancePlot(distDiv.id, trial, distRange, plotWidth, distOverlays, distShapes);
        renderVelocityPlot(velDiv.id, trial, velRange, plotWidth, velOverlays, distShapes);
    });
}

function renderDistancePlot(divId, trial, yRange, width, overlayTraces, shapes) {
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
        type: 'scattergl',
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

            // For each sequence window (start/end are indices into the trial's amplitude array,
            // but we need to filter for valid param values within those windows)
            trialSeqs.sequences.forEach((seq, si) => {
                // seq.start and seq.end are indices into the trial's valid amplitude array
                // Map to the trial's movement indices (ms[])
                // The amplitude array was built from ms.filter(valid amplitude) — need same mapping
                const ampValid = [];
                for (let i = 0; i < ms.length; i++) {
                    if (ms[i].amplitude != null && isFinite(ms[i].amplitude)) {
                        ampValid.push(i);
                    }
                }
                const seqMsIndices = ampValid.slice(seq.start, seq.end);

                // Get x, y for this param within the sequence window
                // For closing velocities, negate so we fit on positive magnitudes
                const seqX = [], seqY = [];
                for (const mi of seqMsIndices) {
                    if (x[mi] != null && y[mi] != null && isFinite(x[mi]) && isFinite(y[mi])) {
                        seqX.push(x[mi]);
                        seqY.push(flipSign ? -y[mi] : y[mi]);
                    }
                }
                if (seqX.length < 2) return;

                const xMin = Math.min(...seqX);
                const xMax = Math.max(...seqX);
                const sign = flipSign ? -1 : 1;

                // Background span — use trial color so shading matches the dots
                shapes.push({
                    type: 'rect', xref: 'x', yref: 'paper',
                    x0: xMin, x1: xMax, y0: 0, y1: 1,
                    fillcolor: color, opacity: 0.07,
                    line: { width: 0 },
                    layer: 'below',
                });

                if (isExp) {
                    // Exponential fit within sequence
                    const fit = exponentialFit(seqX, seqY);
                    if (!fit) return;
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
                        yshift: 12,
                    });
                } else {
                    // Linear fit within sequence
                    const reg = linearRegressionFull(seqX, seqY);
                    if (!reg) return;
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
                        yshift: 12,
                    });
                }
            });

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
                    yshift: 12,
                });
            } else {
                // Linear fit
                const { slope, intercept } = linearRegression(fitX, fitY);
                traces.push({
                    x: [xMin, xMax],
                    y: [sign * (slope * xMin + intercept), sign * (slope * xMax + intercept)],
                    type: 'scatter', mode: 'lines',
                    name: `Trend (Trial ${+ti + 1})`,
                    line: { color, width: 2, dash: 'dash' },
                    showlegend: false, hoverinfo: 'skip',
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

async function loadGroup() {
    const container = document.getElementById('groupPlots');
    const includeAuto = document.getElementById('includeAutoToggle').checked;

    if (cachedGroup && cachedGroup._includeAuto === includeAuto) {
        renderGroupPlots();
        return;
    }

    container.innerHTML = '<div class="results-no-data" style="grid-column:1/-1;">Loading group data...</div>';

    try {
        cachedGroup = await API.get(`/api/results/group?include_auto=${includeAuto}`);
        cachedGroup._includeAuto = includeAuto;
        renderGroupPlots();
    } catch (e) {
        container.innerHTML = `<div class="results-no-data" style="grid-column:1/-1;color:#d32f2f;">${e.message}</div>`;
    }
}

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
}

function _toggleGroupMetric(id, checked) {
    _groupMetricVisible[id] = checked;
    renderGroupPlots();
}

// Expose for inline onclick
window._toggleGroupMetric = _toggleGroupMetric;

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

    if (currentTab === 'distances' || currentTab === 'movements') {
        currentTab = 'distances';
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
['overlayPeakDist', 'overlayPeakOpenVel', 'overlayPeakCloseVel', 'overlaySequences'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => {
        if (cachedTraces) renderAllDistancePlots();
    });
});

// Distance controls: re-render on y-axis changes
document.getElementById('lockYAxis').addEventListener('change', () => {
    const locked = document.getElementById('lockYAxis').checked;
    document.getElementById('yAxisInputs').style.display = locked ? '' : 'none';
    if (cachedTraces) renderAllDistancePlots();
});
document.getElementById('yAxisMin').addEventListener('change', () => {
    if (cachedTraces) renderAllDistancePlots();
});
document.getElementById('yAxisMax').addEventListener('change', () => {
    if (cachedTraces) renderAllDistancePlots();
});

// Group comparison: auto-detect toggle
document.getElementById('includeAutoToggle').addEventListener('change', () => {
    cachedGroup = null;
    loadGroup();
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
    if (initTab && ['distances', 'movements', 'group'].includes(initTab)) {
        // Explicit tab param takes highest priority
        switchTab(initTab);
    } else if (!initSubject && !initFrom) {
        // No subject param and no source context → group comparison
        switchTab('group');
    } else if (currentSubjectId) {
        // Have a subject — check for a remembered tab
        const lastTab = sessionStorage.getItem(`dlc_resultsTab_${currentSubjectId}`);
        switchTab(lastTab || 'movements');
    } else {
        // No subject at all → group comparison
        switchTab('group');
    }
});
