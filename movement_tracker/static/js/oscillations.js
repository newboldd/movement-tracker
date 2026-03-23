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
            trialNames = movData.trial_names || [];
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
        _autoSetWaveDefaults();
        render();
    }

    function selectTrial(idx) {
        selectedTrial = idx;
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
        if (!traceData || !traceData.distances) return { times: [], values: [] };
        const fps = traceData.fps || 30;
        const ranges = traceData.trial_ranges || [];
        const dist = traceData.distances;

        if (selectedTrial < 0) {
            // All trials
            const times = [], values = [];
            for (let i = 0; i < dist.length; i++) {
                if (dist[i] != null) {
                    times.push(i / fps);
                    values.push(dist[i]);
                }
            }
            return { times, values };
        }

        // Single trial
        if (selectedTrial >= ranges.length) return { times: [], values: [] };
        const r = ranges[selectedTrial];
        const times = [], values = [];
        for (let i = r.start; i <= r.end; i++) {
            if (i < dist.length && dist[i] != null) {
                times.push((i - r.start) / fps);
                values.push(dist[i]);
            }
        }
        return { times, values };
    }

    function _autoSetWaveDefaults() {
        const mvs = _getVisibleMovements();
        if (mvs.length < 3) return;

        const amps = mvs.map(m => m.amplitude);
        const mean = amps.reduce((a, b) => a + b, 0) / amps.length;
        const range = Math.max(...amps) - Math.min(...amps);
        const totalTime = mvs[mvs.length - 1].peak_time - mvs[0].peak_time;

        // Set wave 1 defaults: approximate dominant frequency
        document.getElementById('w1Amp').value = range / 2;
        document.getElementById('w1Offset').value = mean;
        if (totalTime > 0) {
            document.getElementById('w1Freq').value = Math.min(2, Math.max(0.01, 1 / totalTime));
        }

        // Wave 2: half the frequency, smaller amplitude
        document.getElementById('w2Amp').value = range / 4;
        document.getElementById('w2Offset').value = 0;
        if (totalTime > 0) {
            document.getElementById('w2Freq').value = Math.min(2, Math.max(0.01, 2 / totalTime));
        }

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
        const mvs = _getVisibleMovements();
        if (mvs.length < 2) {
            document.getElementById('fitInfo').textContent = 'Need at least 2 movements with amplitude data.';
            return;
        }

        const params = _getWaveParams();
        let ssRes = 0, ssTot = 0;
        const mean = mvs.reduce((a, m) => a + m.amplitude, 0) / mvs.length;

        for (const m of mvs) {
            const predicted = _evalWave(params.w1, m.peak_time) + _evalWave(params.w2, m.peak_time);
            ssRes += (m.amplitude - predicted) ** 2;
            ssTot += (m.amplitude - mean) ** 2;
        }

        const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
        const rmse = Math.sqrt(ssRes / mvs.length);

        document.getElementById('fitInfo').textContent =
            `R\u00b2 = ${r2.toFixed(4)} | RMSE = ${rmse.toFixed(2)} mm | n = ${mvs.length} movements`;
    }

    // ── Auto-fit using grid search ──
    function autoFit() {
        const mvs = _getVisibleMovements();
        if (mvs.length < 3) return;

        const amps = mvs.map(m => m.amplitude);
        const times = mvs.map(m => m.peak_time);
        const mean = amps.reduce((a, b) => a + b, 0) / amps.length;
        const range = Math.max(...amps) - Math.min(...amps);
        const totalTime = times[times.length - 1] - times[0];

        let bestR2 = -Infinity;
        let bestParams = null;

        // Grid search over frequency combinations
        for (let f1 = 0.05; f1 <= 1.5; f1 += 0.05) {
            for (let f2 = 0.05; f2 <= 1.5; f2 += 0.05) {
                // For each frequency pair, solve for optimal amplitude/phase/offset
                // using least squares (simplified: try a few phases)
                for (let p1 = 0; p1 < 6.28; p1 += 0.5) {
                    for (let p2 = 0; p2 < 6.28; p2 += 0.5) {
                        const w1 = { freq: f1, amp: range / 2, phase: p1, offset: mean };
                        const w2 = { freq: f2, amp: range / 4, phase: p2, offset: 0 };

                        // Quick R2 calc
                        let ssRes = 0, ssTot = 0;
                        for (let i = 0; i < mvs.length; i++) {
                            const pred = _evalWave(w1, times[i]) + _evalWave(w2, times[i]);
                            ssRes += (amps[i] - pred) ** 2;
                            ssTot += (amps[i] - mean) ** 2;
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
            // Refine amplitudes with fixed frequencies/phases using linear regression
            const { w1, w2 } = bestParams;

            // Build design matrix: [sin1, cos1, sin2, cos2, 1]
            // y = a1*sin(2pi*f1*t+p1) + a2*sin(2pi*f2*t+p2) + offset
            // For now just set the sliders to the grid search result
            document.getElementById('w1Freq').value = w1.freq;
            document.getElementById('w1Amp').value = w1.amp;
            document.getElementById('w1Phase').value = w1.phase;
            document.getElementById('w1Offset').value = w1.offset;
            document.getElementById('w2Freq').value = w2.freq;
            document.getElementById('w2Amp').value = w2.amp;
            document.getElementById('w2Phase').value = w2.phase;
            document.getElementById('w2Offset').value = w2.offset;
            updateWave();
        }
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

        const mvs = _getVisibleMovements();
        const distData = _getVisibleDistances();
        const showDist = document.getElementById('showDistance').checked;
        const showAmp = document.getElementById('showAmplitude').checked;

        // Determine time and value ranges
        let tMin = Infinity, tMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;

        if (showDist && distData.times.length > 0) {
            tMin = Math.min(tMin, distData.times[0]);
            tMax = Math.max(tMax, distData.times[distData.times.length - 1]);
            for (const v of distData.values) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
        }
        if (showAmp && mvs.length > 0) {
            for (const m of mvs) {
                tMin = Math.min(tMin, m.peak_time);
                tMax = Math.max(tMax, m.peak_time);
                yMin = Math.min(yMin, m.amplitude);
                yMax = Math.max(yMax, m.amplitude);
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

        // Amplitude scatter (orange diamonds)
        if (showAmp && mvs.length > 0) {
            ctx.fillStyle = 'rgba(255,152,0,0.8)';
            for (const m of mvs) {
                const x = tToX(m.peak_time);
                const y = yToY(m.amplitude);
                ctx.save();
                ctx.translate(x, y);
                ctx.rotate(Math.PI / 4);
                ctx.fillRect(-4, -4, 8, 8);
                ctx.restore();
            }
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
        if (showAmp) { _drawLegend(cw - MARGIN.right - 10, ly, 'Amplitude', 'rgba(255,152,0,0.8)'); ly += 16; }
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
