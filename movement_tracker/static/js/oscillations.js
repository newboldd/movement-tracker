/* Oscillation Analysis — fit sine waves to amplitude fluctuations */
const osc = (() => {
    let subjects = [];
    let subjectId = null;
    let traceData = null;   // {distances, velocity, fps, trial_names, trial_ranges}
    let movements = null;   // [{peak_time, amplitude, trial_idx, ...}]
    let trialNames = [];
    let selectedTrial = -1; // -1 = all trials

    let canvas, ctx;

    // Chart layout
    const MARGIN = { top: 20, right: 60, bottom: 40, left: 60 };

    // X-axis zoom/pan state
    let xZoomMin = null;  // null = auto
    let xZoomMax = null;
    let xDragging = false;
    let xDragStart = null;

    async function init() {
        canvas = document.getElementById('chart');
        ctx = canvas.getContext('2d');

        // Load subjects
        try {
            subjects = await API.get('/api/subjects');
        } catch (e) { subjects = []; }

        const sel = document.getElementById('subjectSelect');
        sel.innerHTML = '<option value="">Select subject...</option>';
        subjects.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', () => {
            if (sel.value) loadSubject(parseInt(sel.value));
        });

        window.addEventListener('resize', () => render());

        // Zoom/pan on chart
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const plotW = canvas.width - MARGIN.left - MARGIN.right;
            const frac = (mx - MARGIN.left) / plotW;  // 0-1 position of mouse

            const [curMin, curMax] = _getXRange();
            const span = curMax - curMin;
            const zoomFactor = e.deltaY < 0 ? 0.85 : 1.18;
            const newSpan = span * zoomFactor;
            const pivot = curMin + frac * span;

            xZoomMin = pivot - frac * newSpan;
            xZoomMax = pivot + (1 - frac) * newSpan;
            render();
        }, { passive: false });

        canvas.addEventListener('mousedown', (e) => {
            if (e.button === 0) {
                xDragging = true;
                xDragStart = { x: e.clientX, min: xZoomMin, max: xZoomMax };
                canvas.style.cursor = 'grabbing';
            }
        });
        canvas.addEventListener('mousemove', (e) => {
            if (!xDragging || !xDragStart) return;
            const plotW = canvas.width - MARGIN.left - MARGIN.right;
            const [curMin, curMax] = [xDragStart.min, xDragStart.max];
            if (curMin == null || curMax == null) return;
            const span = curMax - curMin;
            const dx = (xDragStart.x - e.clientX) / plotW * span;
            xZoomMin = curMin + dx;
            xZoomMax = curMax + dx;
            render();
        });
        canvas.addEventListener('mouseup', () => {
            xDragging = false; xDragStart = null; canvas.style.cursor = '';
        });
        canvas.addEventListener('mouseleave', () => {
            xDragging = false; xDragStart = null; canvas.style.cursor = '';
        });
        canvas.addEventListener('dblclick', () => {
            xZoomMin = null; xZoomMax = null; render();  // reset zoom
        });

        // Auto-select from URL or if single subject
        const params = new URLSearchParams(window.location.search);
        const urlSubj = params.get('subject');
        if (urlSubj) { sel.value = urlSubj; loadSubject(parseInt(urlSubj)); }
        else if (subjects.length === 1) { sel.value = subjects[0].id; loadSubject(subjects[0].id); }
    }

    async function loadSubject(sid) {
        subjectId = sid;
        traceData = null;
        movements = null;

        try {
            const [traces, movData] = await Promise.all([
                API.get(`/api/results/${sid}/traces`),
                API.get(`/api/results/${sid}/movements`),
            ]);
            traceData = traces;
            movements = movData.movements || [];
            trialNames = (traces.trials || []).map(t => t.name);
        } catch (e) {
            document.getElementById('fitInfo').textContent = 'Error loading data: ' + e.message;
            return;
        }

        // Build trial buttons
        const btns = document.getElementById('trialBtns');
        btns.innerHTML = '';
        const allBtn = document.createElement('button');
        allBtn.className = 'trial-btn active';
        allBtn.textContent = 'All';
        allBtn.onclick = () => selectTrial(-1);
        btns.appendChild(allBtn);
        trialNames.forEach((name, i) => {
            const btn = document.createElement('button');
            btn.className = 'trial-btn';
            btn.textContent = name;
            btn.onclick = () => selectTrial(i);
            btns.appendChild(btn);
        });

        selectedTrial = -1;
        xZoomMin = null; xZoomMax = null;
        _autoSetWaveDefaults();
        render();
    }

    function selectTrial(idx) {
        selectedTrial = idx;
        xZoomMin = null; xZoomMax = null;
        document.querySelectorAll('.trial-btn').forEach((b, i) => {
            b.classList.toggle('active', i === (idx + 1));
        });
        _autoSetWaveDefaults();
        render();
    }

    function _getVisibleMovements() {
        if (!movements) return [];
        if (selectedTrial < 0) return movements.filter(m => m.amplitude != null);
        return movements.filter(m => m.trial_idx === selectedTrial && m.amplitude != null);
    }

    function _getVisibleDistances() {
        if (!traceData || !traceData.trials || traceData.trials.length === 0) return { times: [], values: [] };

        const trials = traceData.trials;
        const times = [], values = [];

        if (selectedTrial < 0) {
            // All trials concatenated with time offset
            let timeOffset = 0;
            for (const trial of trials) {
                const fps = trial.fps || 30;
                const dist = trial.distances || [];
                for (let i = 0; i < dist.length; i++) {
                    if (dist[i] != null) {
                        times.push(timeOffset + i / fps);
                        values.push(dist[i]);
                    }
                }
                timeOffset += dist.length / (trial.fps || 30);
            }
        } else if (selectedTrial < trials.length) {
            const trial = trials[selectedTrial];
            const fps = trial.fps || 30;
            const dist = trial.distances || [];
            for (let i = 0; i < dist.length; i++) {
                if (dist[i] != null) {
                    times.push(i / fps);
                    values.push(dist[i]);
                }
            }
        }

        return { times, values };
    }

    function _autoSetWaveDefaults() {
        const distData = _getVisibleDistances();
        if (distData.values.length < 10) return;

        const vals = distData.values;
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const range = Math.max(...vals) - Math.min(...vals);
        const totalTime = distData.times[distData.times.length - 1] - distData.times[0];

        // Wave 1: tapping frequency (~3-5 Hz typical)
        document.getElementById('w1Amp').value = Math.min(100, range / 2);
        document.getElementById('w1Offset').value = mean;
        document.getElementById('w1Freq').value = 4;

        // Wave 2: slower modulation
        document.getElementById('w2Amp').value = Math.min(100, range / 6);
        document.getElementById('w2Offset').value = 0;
        document.getElementById('w2Freq').value = 0.3;

        updateWave();
    }

    function updateWave() {
        // Update display values
        document.getElementById('w1FreqVal').textContent = parseFloat(document.getElementById('w1Freq').value).toFixed(2) + ' Hz';
        document.getElementById('w1AmpVal').textContent = parseFloat(document.getElementById('w1Amp').value).toFixed(1) + ' mm';
        document.getElementById('w1PhaseVal').textContent = parseFloat(document.getElementById('w1Phase').value).toFixed(2) + ' rad';
        document.getElementById('w1OffsetVal').textContent = parseFloat(document.getElementById('w1Offset').value).toFixed(1) + ' mm';
        document.getElementById('w2FreqVal').textContent = parseFloat(document.getElementById('w2Freq').value).toFixed(2) + ' Hz';
        document.getElementById('w2AmpVal').textContent = parseFloat(document.getElementById('w2Amp').value).toFixed(1) + ' mm';
        document.getElementById('w2PhaseVal').textContent = parseFloat(document.getElementById('w2Phase').value).toFixed(2) + ' rad';
        document.getElementById('w2OffsetVal').textContent = parseFloat(document.getElementById('w2Offset').value).toFixed(1) + ' mm';

        render();
        _updateFitInfo();
    }

    function _getWaveParams() {
        return {
            w1: {
                freq: parseFloat(document.getElementById('w1Freq').value),
                amp: parseFloat(document.getElementById('w1Amp').value),
                phase: parseFloat(document.getElementById('w1Phase').value),
                offset: parseFloat(document.getElementById('w1Offset').value),
            },
            w2: {
                freq: parseFloat(document.getElementById('w2Freq').value),
                amp: parseFloat(document.getElementById('w2Amp').value),
                phase: parseFloat(document.getElementById('w2Phase').value),
                offset: parseFloat(document.getElementById('w2Offset').value),
            },
        };
    }

    function _evalWave(w, t) {
        return w.amp * Math.sin(2 * Math.PI * w.freq * t + w.phase) + w.offset;
    }

    function _updateFitInfo() {
        const distData = _getVisibleDistances();
        if (distData.values.length < 10) {
            document.getElementById('fitInfo').textContent = 'Need distance trace data.';
            return;
        }

        const params = _getWaveParams();
        // Subsample for speed (every 5th point)
        const step = Math.max(1, Math.floor(distData.times.length / 500));
        let ssRes = 0, ssTot = 0, n = 0;
        const mean = distData.values.reduce((a, b) => a + b, 0) / distData.values.length;

        for (let i = 0; i < distData.times.length; i += step) {
            const predicted = _evalWave(params.w1, distData.times[i]) + _evalWave(params.w2, distData.times[i]);
            ssRes += (distData.values[i] - predicted) ** 2;
            ssTot += (distData.values[i] - mean) ** 2;
            n++;
        }

        const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
        const rmse = Math.sqrt(ssRes / n);

        document.getElementById('fitInfo').textContent =
            `R\u00b2 = ${r2.toFixed(4)} | RMSE = ${rmse.toFixed(2)} mm | n = ${n} samples (${distData.values.length} total)`;
    }

    // ── Auto-fit using grid search on distance trace ──
    function autoFit() {
        const distData = _getVisibleDistances();
        if (distData.values.length < 20) return;

        document.getElementById('fitInfo').textContent = 'Fitting...';

        // Subsample for speed
        const step = Math.max(1, Math.floor(distData.times.length / 200));
        const times = [], vals = [];
        for (let i = 0; i < distData.times.length; i += step) {
            times.push(distData.times[i]);
            vals.push(distData.values[i]);
        }
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const range = Math.max(...vals) - Math.min(...vals);

        let bestR2 = -Infinity;
        let bestParams = null;

        // Grid search: coarser for the large frequency range
        const freqStep = 0.2;
        const phaseStep = 0.8;

        for (let f1 = 0.1; f1 <= 12; f1 += freqStep) {
            for (let f2 = 0.1; f2 <= 12; f2 += freqStep) {
                for (let p1 = 0; p1 < 6.28; p1 += phaseStep) {
                    for (let p2 = 0; p2 < 6.28; p2 += phaseStep) {
                        const w1 = { freq: f1, amp: range / 2, phase: p1, offset: mean };
                        const w2 = { freq: f2, amp: range / 4, phase: p2, offset: 0 };

                        let ssRes = 0, ssTot = 0;
                        for (let i = 0; i < times.length; i++) {
                            const pred = _evalWave(w1, times[i]) + _evalWave(w2, times[i]);
                            ssRes += (vals[i] - pred) ** 2;
                            ssTot += (vals[i] - mean) ** 2;
                        }
                        const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
                        if (r2 > bestR2) {
                            bestR2 = r2;
                            bestParams = { w1: { ...w1 }, w2: { ...w2 } };
                        }
                    }
                }
            }
        }

        if (bestParams) {
            document.getElementById('w1Freq').value = bestParams.w1.freq;
            document.getElementById('w1Amp').value = bestParams.w1.amp;
            document.getElementById('w1Phase').value = bestParams.w1.phase;
            document.getElementById('w1Offset').value = bestParams.w1.offset;
            document.getElementById('w2Freq').value = bestParams.w2.freq;
            document.getElementById('w2Amp').value = bestParams.w2.amp;
            document.getElementById('w2Phase').value = bestParams.w2.phase;
            document.getElementById('w2Offset').value = bestParams.w2.offset;
            updateWave();
        }
    }

    function _getXRange() {
        if (xZoomMin != null && xZoomMax != null) return [xZoomMin, xZoomMax];
        // Auto range from data
        const mvs = _getVisibleMovements();
        const distData = _getVisibleDistances();
        let tMin = Infinity, tMax = -Infinity;
        if (distData.times.length > 0) {
            tMin = Math.min(tMin, distData.times[0]);
            tMax = Math.max(tMax, distData.times[distData.times.length - 1]);
        }
        for (const m of mvs) {
            tMin = Math.min(tMin, m.peak_time);
            tMax = Math.max(tMax, m.peak_time);
        }
        if (tMax <= tMin) { tMin = 0; tMax = 10; }
        return [tMin, tMax];
    }

    // ── Rendering ──
    function render() {
        if (!ctx || !canvas) return;
        const container = canvas.parentElement;
        canvas.width = container.clientWidth;
        canvas.height = Math.max(250, container.clientHeight);
        const cw = canvas.width, ch = canvas.height;

        ctx.clearRect(0, 0, cw, ch);
        ctx.fillStyle = 'var(--bg)';
        ctx.fillRect(0, 0, cw, ch);

        const distData = _getVisibleDistances();
        const showDist = document.getElementById('showDistance').checked;

        // Determine time range (respects zoom/pan)
        let [tMin, tMax] = _getXRange();
        let yMin = Infinity, yMax = -Infinity;

        // Y range from visible data within the current x range
        if (showDist && distData.times.length > 0) {
            for (let i = 0; i < distData.times.length; i++) {
                if (distData.times[i] >= tMin && distData.times[i] <= tMax) {
                    yMin = Math.min(yMin, distData.values[i]);
                    yMax = Math.max(yMax, distData.values[i]);
                }
            }
        }

        // Include wave range in y bounds
        const params = _getWaveParams();
        const showW1 = document.getElementById('showWave1').checked;
        const showW2 = document.getElementById('showWave2').checked;
        const showSum = document.getElementById('showSum').checked;

        if (tMax <= tMin) { tMin = 0; tMax = 10; }
        if (yMax <= yMin) { yMin = 0; yMax = 100; }

        // Pad
        const yPad = (yMax - yMin) * 0.1;
        yMin -= yPad; yMax += yPad;

        const plotW = cw - MARGIN.left - MARGIN.right;
        const plotH = ch - MARGIN.top - MARGIN.bottom;

        function tToX(t) { return MARGIN.left + (t - tMin) / (tMax - tMin) * plotW; }
        function yToY(v) { return MARGIN.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

        // Grid lines
        ctx.strokeStyle = 'rgba(150,150,150,0.15)';
        ctx.lineWidth = 1;
        const nGridY = 5;
        for (let i = 0; i <= nGridY; i++) {
            const v = yMin + (yMax - yMin) * i / nGridY;
            const y = yToY(v);
            ctx.beginPath(); ctx.moveTo(MARGIN.left, y); ctx.lineTo(cw - MARGIN.right, y); ctx.stroke();
            ctx.fillStyle = 'rgba(150,150,150,0.6)';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(v.toFixed(0), MARGIN.left - 5, y + 3);
        }
        const nGridX = 8;
        for (let i = 0; i <= nGridX; i++) {
            const t = tMin + (tMax - tMin) * i / nGridX;
            const x = tToX(t);
            ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, ch - MARGIN.bottom); ctx.stroke();
            ctx.fillStyle = 'rgba(150,150,150,0.6)';
            ctx.textAlign = 'center';
            ctx.fillText(t.toFixed(1) + 's', x, ch - MARGIN.bottom + 14);
        }

        // Axis labels
        ctx.fillStyle = 'rgba(150,150,150,0.8)';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Time (s)', cw / 2, ch - 4);
        ctx.save();
        ctx.translate(12, ch / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Distance / Amplitude (mm)', 0, 0);
        ctx.restore();

        // Distance trace (thin blue line)
        if (showDist && distData.times.length > 1) {
            ctx.strokeStyle = 'rgba(33,150,243,0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < distData.times.length; i++) {
                const x = tToX(distData.times[i]);
                const y = yToY(distData.values[i]);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Wave curves
        const dt = (tMax - tMin) / (plotW * 2);
        if (showW1) {
            _drawWaveCurve(params.w1, tMin, tMax, dt, tToX, yToY, 'rgba(255,152,0,0.7)', 2);
        }
        if (showW2) {
            _drawWaveCurve(params.w2, tMin, tMax, dt, tToX, yToY, 'rgba(76,175,80,0.7)', 2);
        }
        if (showSum) {
            // Combined wave
            ctx.strokeStyle = 'rgba(244,67,54,0.8)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            let first = true;
            for (let t = tMin; t <= tMax; t += dt) {
                const v = _evalWave(params.w1, t) + _evalWave(params.w2, t);
                const x = tToX(t);
                const y = yToY(v);
                if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Legend
        let ly = MARGIN.top + 10;
        ctx.font = '11px sans-serif';
        if (showDist) { _drawLegend(cw - MARGIN.right - 10, ly, 'Distance', 'rgba(33,150,243,0.6)'); ly += 16; }
        if (showW1) { _drawLegend(cw - MARGIN.right - 10, ly, 'Wave 1', 'rgba(255,152,0,0.7)'); ly += 16; }
        if (showW2) { _drawLegend(cw - MARGIN.right - 10, ly, 'Wave 2', 'rgba(76,175,80,0.7)'); ly += 16; }
        if (showSum) { _drawLegend(cw - MARGIN.right - 10, ly, 'Sum', 'rgba(244,67,54,0.8)'); }
    }

    function _drawWaveCurve(w, tMin, tMax, dt, tToX, yToY, color, lineWidth) {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        let first = true;
        for (let t = tMin; t <= tMax; t += dt) {
            const v = _evalWave(w, t);
            const x = tToX(t);
            const y = yToY(v);
            if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    function _drawLegend(x, y, label, color) {
        ctx.textAlign = 'right';
        ctx.fillStyle = color;
        ctx.fillRect(x - 50, y - 4, 16, 3);
        ctx.fillStyle = 'rgba(200,200,200,0.8)';
        ctx.fillText(label, x - 55, y);
    }

    document.addEventListener('DOMContentLoaded', init);

    return { loadSubject, selectTrial, updateWave, autoFit, render };
})();
