/* Interactive Deidentify/Blur Page */
const deid = (() => {
    // ── State ──
    let subjects = [];
    let subjectId = null;
    let subjectName = '';
    let trials = [];
    let currentTrialIdx = -1;
    let trialMeta = null;     // current trial metadata

    let currentFrame = 0;
    let totalFrames = 0;
    let fps = 30;
    let playing = false;
    let playTimer = null;

    // Face detection results: [{frame, faces: [{x1,y1,x2,y2,side}]}]
    let faceDetections = [];
    // Active blur spots: [{id, spot_type, x, y, radius, frame_start, frame_end, side}]
    let blurSpots = [];
    let nextSpotId = 1;
    let selectedSpotId = null;
    let addingCustom = false;

    // Hand overlay
    let handOverlayEnabled = false;
    let handLandmarks = [];   // [{x, y, side}]
    let handMaskRadius = 30;
    let handMaskEnabled = true;

    // Canvas
    let canvas, ctx;
    let currentImage = null;
    let imgW = 0, imgH = 0;
    let scale = 1, offsetX = 0, offsetY = 0;

    // Debounce save timer
    let saveTimer = null;

    // ── Init ──
    async function init() {
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');

        // Load subjects
        try {
            subjects = await API.get('/api/subjects');
        } catch (e) {
            subjects = [];
        }

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

        // Canvas events
        canvas.addEventListener('click', onCanvasClick);
        canvas.addEventListener('wheel', onWheel, { passive: false });

        // Keyboard
        document.addEventListener('keydown', onKeyDown);

        // Resize
        window.addEventListener('resize', () => { fitImage(); render(); });

        // Auto-select if only one subject
        if (subjects.length === 1) {
            sel.value = subjects[0].id;
            loadSubject(subjects[0].id);
        }
    }

    // ── Load subject ──
    async function loadSubject(sid) {
        subjectId = sid;
        faceDetections = [];
        blurSpots = [];
        selectedSpotId = null;
        handLandmarks = [];

        try {
            const data = await API.get(`/api/deidentify/${sid}/trials`);
            subjectName = data.subject.name;
            trials = data.trials;
        } catch (e) {
            document.getElementById('statusMsg').textContent = 'Error loading subject: ' + e.message;
            return;
        }

        // Build trial buttons
        const btns = document.getElementById('trialBtns');
        btns.innerHTML = '';
        trials.forEach((t, i) => {
            const btn = document.createElement('button');
            btn.className = 'trial-btn';
            btn.textContent = t.trial_name;
            btn.onclick = () => selectTrial(i);
            btns.appendChild(btn);
        });

        if (trials.length > 0) selectTrial(0);
    }

    // ── Select trial ──
    async function selectTrial(idx) {
        if (playing) togglePlay();
        currentTrialIdx = idx;
        trialMeta = trials[idx];
        currentFrame = trialMeta.start_frame;
        totalFrames = trialMeta.frame_count;
        fps = trialMeta.fps || 30;
        faceDetections = [];
        handLandmarks = [];

        // Update UI
        document.querySelectorAll('.trial-btn').forEach((b, i) => {
            b.classList.toggle('active', i === idx);
        });

        const slider = document.getElementById('frameSlider');
        slider.min = trialMeta.start_frame;
        slider.max = trialMeta.end_frame;
        slider.value = currentFrame;

        // Load saved blur specs
        try {
            const resp = await API.get(`/api/deidentify/${subjectId}/blur-specs?trial_idx=${idx}`);
            blurSpots = (resp.specs || []).map(s => ({ ...s, id: nextSpotId++ }));
        } catch (e) {
            blurSpots = [];
        }

        // Load hand settings
        try {
            const hs = await API.get(`/api/deidentify/${subjectId}/hand-settings?trial_idx=${idx}`);
            handMaskEnabled = hs.enabled;
            handMaskRadius = hs.mask_radius;
            document.getElementById('handMaskEnabled').checked = handMaskEnabled;
            document.getElementById('handRadiusSlider').value = handMaskRadius;
            document.getElementById('handRadiusVal').textContent = handMaskRadius;
        } catch (e) {}

        await loadFrame(currentFrame);
        renderSpotList();
    }

    // ── Frame loading ──
    async function loadFrame(frameNum) {
        currentFrame = frameNum;
        const url = `/api/deidentify/${subjectId}/frame?trial_idx=${currentTrialIdx}&frame_num=${frameNum}&side=full`;

        try {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = url;
            });
            currentImage = img;
            imgW = img.width;
            imgH = img.height;
        } catch (e) {
            return;
        }

        if (scale === 1 && offsetX === 0 && offsetY === 0) fitImage();

        // Update slider + display
        document.getElementById('frameSlider').value = frameNum;
        const localFrame = frameNum - (trialMeta ? trialMeta.start_frame : 0);
        document.getElementById('frameDisplay').textContent =
            `Frame: ${localFrame} / ${totalFrames - 1}`;

        // Load hand landmarks if overlay enabled
        if (handOverlayEnabled) {
            try {
                const res = await API.post(`/api/deidentify/${subjectId}/detect-hands`, {
                    trial_idx: currentTrialIdx, frame_num: frameNum,
                });
                handLandmarks = res.landmarks || [];
            } catch (e) {
                handLandmarks = [];
            }
        }

        render();
    }

    function fitImage() {
        if (!currentImage || !canvas) return;
        const cw = canvas.clientWidth;
        const ch = canvas.clientHeight;
        canvas.width = cw;
        canvas.height = ch;
        scale = Math.min(cw / imgW, ch / imgH);
        offsetX = (cw - imgW * scale) / 2;
        offsetY = (ch - imgH * scale) / 2;
    }

    // ── Render ──
    function render() {
        if (!ctx || !canvas) return;
        const cw = canvas.width;
        const ch = canvas.height;
        ctx.clearRect(0, 0, cw, ch);

        if (!currentImage) return;

        // Draw video frame
        ctx.drawImage(currentImage, offsetX, offsetY, imgW * scale, imgH * scale);

        // Draw face detections (blue dashed rectangles)
        const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
        if (faceDetections.length > localFrame) {
            const entry = faceDetections[localFrame];
            if (entry && entry.faces) {
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = 'rgba(33,150,243,0.8)';
                ctx.lineWidth = 2;
                for (const f of entry.faces) {
                    const sx = offsetX + f.x1 * scale;
                    const sy = offsetY + f.y1 * scale;
                    const sw = (f.x2 - f.x1) * scale;
                    const sh = (f.y2 - f.y1) * scale;
                    ctx.strokeRect(sx, sy, sw, sh);
                }
                ctx.setLineDash([]);
            }
        }

        // Draw blur spots (red semi-transparent circles)
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const sx = offsetX + spot.x * scale;
            const sy = offsetY + spot.y * scale;
            const sr = spot.radius * scale;

            ctx.beginPath();
            ctx.arc(sx, sy, sr, 0, Math.PI * 2);
            ctx.fillStyle = spot.id === selectedSpotId
                ? 'rgba(244,67,54,0.35)'
                : 'rgba(244,67,54,0.2)';
            ctx.fill();
            ctx.strokeStyle = spot.id === selectedSpotId
                ? 'rgba(244,67,54,0.9)'
                : 'rgba(244,67,54,0.5)';
            ctx.lineWidth = spot.id === selectedSpotId ? 2 : 1;
            ctx.stroke();
        }

        // Draw hand landmarks (green dots)
        if (handOverlayEnabled && handLandmarks.length > 0) {
            ctx.fillStyle = 'rgba(76,175,80,0.7)';
            for (const lm of handLandmarks) {
                const sx = offsetX + lm.x * scale;
                const sy = offsetY + lm.y * scale;
                ctx.beginPath();
                ctx.arc(sx, sy, 3, 0, Math.PI * 2);
                ctx.fill();
            }

            // Draw hand mask radius preview
            if (handMaskEnabled) {
                ctx.strokeStyle = 'rgba(76,175,80,0.3)';
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                // Show convex hull area approximation
                for (const lm of handLandmarks) {
                    const sx = offsetX + lm.x * scale;
                    const sy = offsetY + lm.y * scale;
                    ctx.beginPath();
                    ctx.arc(sx, sy, handMaskRadius * scale * 0.3, 0, Math.PI * 2);
                    ctx.stroke();
                }
                ctx.setLineDash([]);
            }
        }

        // "Adding custom" cursor indicator
        if (addingCustom) {
            ctx.fillStyle = 'rgba(255,152,0,0.15)';
            ctx.strokeStyle = 'rgba(255,152,0,0.6)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            // Show a hint circle at center
            ctx.beginPath();
            ctx.arc(cw / 2, ch / 2, 30, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    // ── Canvas click ──
    function onCanvasClick(e) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        // Convert to image coordinates
        const ix = (cx - offsetX) / scale;
        const iy = (cy - offsetY) / scale;

        if (ix < 0 || iy < 0 || ix > imgW || iy > imgH) return;

        // Check if clicking on a face detection → create face spot
        const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
        if (faceDetections.length > localFrame) {
            const entry = faceDetections[localFrame];
            if (entry && entry.faces) {
                for (const f of entry.faces) {
                    if (ix >= f.x1 && ix <= f.x2 && iy >= f.y1 && iy <= f.y2) {
                        // Create a blur spot from this face
                        const spot = {
                            id: nextSpotId++,
                            spot_type: 'face',
                            x: Math.round((f.x1 + f.x2) / 2),
                            y: Math.round((f.y1 + f.y2) / 2),
                            radius: Math.round(Math.max(f.x2 - f.x1, f.y2 - f.y1) / 2),
                            frame_start: trialMeta.start_frame,
                            frame_end: trialMeta.end_frame,
                            side: f.side || 'full',
                        };
                        blurSpots.push(spot);
                        selectedSpotId = spot.id;
                        renderSpotList();
                        updateSpotControls();
                        scheduleSave();
                        render();
                        return;
                    }
                }
            }
        }

        // Check if clicking on an existing blur spot → select it
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const dx = ix - spot.x;
            const dy = iy - spot.y;
            if (Math.sqrt(dx * dx + dy * dy) <= spot.radius) {
                selectedSpotId = spot.id;
                renderSpotList();
                updateSpotControls();
                render();
                return;
            }
        }

        // Adding custom spot mode
        if (addingCustom) {
            const spot = {
                id: nextSpotId++,
                spot_type: 'custom',
                x: Math.round(ix),
                y: Math.round(iy),
                radius: 40,
                frame_start: Math.max(trialMeta.start_frame, currentFrame - 30),
                frame_end: Math.min(trialMeta.end_frame, currentFrame + 30),
                side: 'full',
            };
            blurSpots.push(spot);
            selectedSpotId = spot.id;
            addingCustom = false;
            document.getElementById('addCustomBtn').style.background = '';
            renderSpotList();
            updateSpotControls();
            scheduleSave();
            render();
            return;
        }

        // Click on nothing → deselect
        selectedSpotId = null;
        renderSpotList();
        updateSpotControls();
        render();
    }

    // ── Zoom ──
    function onWheel(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
        const newScale = scale * zoomFactor;

        offsetX = mx - (mx - offsetX) * (newScale / scale);
        offsetY = my - (my - offsetY) * (newScale / scale);
        scale = newScale;
        render();
    }

    // ── Keyboard ──
    function onKeyDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        if (e.key === 'ArrowRight' || e.key === 'd') {
            e.preventDefault();
            seekFrame(Math.min(currentFrame + 1, trialMeta ? trialMeta.end_frame : 0));
        } else if (e.key === 'ArrowLeft' || e.key === 'a') {
            e.preventDefault();
            seekFrame(Math.max(currentFrame - 1, trialMeta ? trialMeta.start_frame : 0));
        } else if (e.key === ' ') {
            e.preventDefault();
            togglePlay();
        } else if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedSpotId !== null) {
                deleteSpot(selectedSpotId);
            }
        } else if (e.key === 'Escape') {
            if (addingCustom) {
                addingCustom = false;
                document.getElementById('addCustomBtn').style.background = '';
                render();
            }
        }
    }

    // ── Frame navigation ──
    function seekFrame(n) {
        if (!trialMeta) return;
        n = Math.max(trialMeta.start_frame, Math.min(trialMeta.end_frame, n));
        loadFrame(n);
    }

    function togglePlay() {
        playing = !playing;
        const btn = document.getElementById('playBtn');
        if (playing) {
            btn.innerHTML = '&#9646;&#9646;';
            const interval = 1000 / fps;
            playTimer = setInterval(() => {
                if (currentFrame >= (trialMeta ? trialMeta.end_frame : 0)) {
                    togglePlay();
                    return;
                }
                seekFrame(currentFrame + 1);
            }, interval);
        } else {
            btn.innerHTML = '&#9654;';
            if (playTimer) { clearInterval(playTimer); playTimer = null; }
        }
    }

    // ── Face detection ──
    async function detectFaces() {
        if (!subjectId || currentTrialIdx < 0) return;

        const btn = document.getElementById('detectFacesBtn');
        const status = document.getElementById('faceDetStatus');
        btn.disabled = true;
        status.textContent = 'Detecting faces...';

        try {
            const result = await API.post(`/api/deidentify/${subjectId}/detect-faces`, {
                trial_idx: currentTrialIdx,
            });
            faceDetections = result.faces || [];
            const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
            status.textContent = `Found faces in ${nFaces}/${faceDetections.length} frames`;
            render();
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
        }
        btn.disabled = false;
    }

    // ── Spot management ──
    function renderSpotList() {
        const list = document.getElementById('spotList');
        if (!list) return;

        if (blurSpots.length === 0) {
            list.innerHTML = '<div style="font-size:11px;color:var(--text-muted);">No blur spots yet. Click a detected face or use + Custom Spot.</div>';
            return;
        }

        list.innerHTML = blurSpots.map(s => `
            <div class="spot-item ${s.id === selectedSpotId ? 'selected' : ''}"
                 onclick="deid.selectSpot(${s.id})">
                <span class="spot-label">${s.spot_type === 'face' ? '👤' : '⊕'}
                    r=${s.radius} [${s.frame_start}-${s.frame_end}]</span>
                <span class="spot-delete" onclick="event.stopPropagation(); deid.deleteSpot(${s.id})">×</span>
            </div>
        `).join('');
    }

    function selectSpot(id) {
        selectedSpotId = id;
        renderSpotList();
        updateSpotControls();
        render();
    }

    function deleteSpot(id) {
        blurSpots = blurSpots.filter(s => s.id !== id);
        if (selectedSpotId === id) {
            selectedSpotId = null;
            updateSpotControls();
        }
        renderSpotList();
        scheduleSave();
        render();
    }

    function updateSpotControls() {
        const controls = document.getElementById('spotControls');
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (!spot) {
            if (controls) controls.style.display = 'none';
            return;
        }
        if (controls) controls.style.display = 'block';
        document.getElementById('radiusSlider').value = spot.radius;
        document.getElementById('radiusVal').textContent = spot.radius;
        document.getElementById('frameStartInput').value = spot.frame_start;
        document.getElementById('frameEndInput').value = spot.frame_end;
    }

    function updateSpotRadius(val) {
        val = parseInt(val);
        document.getElementById('radiusVal').textContent = val;
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (spot) {
            spot.radius = val;
            scheduleSave();
            render();
        }
    }

    function updateSpotFrameRange() {
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (!spot) return;
        spot.frame_start = parseInt(document.getElementById('frameStartInput').value) || 0;
        spot.frame_end = parseInt(document.getElementById('frameEndInput').value) || 0;
        scheduleSave();
        render();
    }

    // ── Custom spot toggle ──
    function toggleAddCustom() {
        addingCustom = !addingCustom;
        const btn = document.getElementById('addCustomBtn');
        btn.style.background = addingCustom ? 'var(--blue)' : '';
        btn.style.color = addingCustom ? '#fff' : '';
        render();
    }

    // ── Hand overlay ──
    async function toggleHandOverlay(enabled) {
        handOverlayEnabled = enabled;
        if (enabled && subjectId) {
            try {
                const res = await API.post(`/api/deidentify/${subjectId}/detect-hands`, {
                    trial_idx: currentTrialIdx, frame_num: currentFrame,
                });
                handLandmarks = res.landmarks || [];
            } catch (e) {
                handLandmarks = [];
            }
        } else {
            handLandmarks = [];
        }
        render();
    }

    function updateHandRadius(val) {
        handMaskRadius = parseInt(val);
        document.getElementById('handRadiusVal').textContent = handMaskRadius;
        updateHandSettings();
        render();
    }

    async function updateHandSettings() {
        handMaskEnabled = document.getElementById('handMaskEnabled').checked;
        if (!subjectId || currentTrialIdx < 0) return;
        try {
            await API.put(`/api/deidentify/${subjectId}/hand-settings`, {
                trial_idx: currentTrialIdx,
                enabled: handMaskEnabled,
                mask_radius: handMaskRadius,
            });
        } catch (e) {}
    }

    // ── Save blur specs (debounced) ──
    function scheduleSave() {
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(saveSpecs, 500);
    }

    async function saveSpecs() {
        if (!subjectId || currentTrialIdx < 0) return;
        const specs = blurSpots.map(s => ({
            spot_type: s.spot_type,
            x: s.x,
            y: s.y,
            radius: s.radius,
            frame_start: s.frame_start,
            frame_end: s.frame_end,
        }));
        try {
            await API.put(`/api/deidentify/${subjectId}/blur-specs`, {
                trial_idx: currentTrialIdx,
                specs: specs,
            });
        } catch (e) {
            console.error('Save blur specs failed:', e);
        }
    }

    // ── Render all trials ──
    async function renderAll() {
        if (!subjectId) return;

        // Save current specs first
        await saveSpecs();

        const btn = document.getElementById('renderBtn');
        const status = document.getElementById('renderStatus');
        btn.disabled = true;
        status.textContent = 'Starting render...';

        try {
            const result = await API.post(`/api/deidentify/${subjectId}/render`);
            const jobId = result.job_id;

            API.streamJob(jobId,
                (data) => {
                    if (data.status === 'running') {
                        const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                        status.textContent = `Rendering... ${pct}%`;
                    }
                },
                (data) => {
                    btn.disabled = false;
                    if (data.status === 'completed') {
                        status.textContent = 'Render complete! Deidentified videos saved.';
                    } else if (data.status === 'failed') {
                        status.textContent = 'Render failed: ' + (data.error_msg || 'unknown');
                    } else {
                        status.textContent = 'Render ' + data.status;
                    }
                },
            );
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
            btn.disabled = false;
        }
    }

    // ── Public API ──
    document.addEventListener('DOMContentLoaded', init);

    return {
        detectFaces,
        togglePlay,
        seekFrame,
        toggleAddCustom,
        selectSpot,
        deleteSpot,
        updateSpotRadius,
        updateSpotFrameRange,
        toggleHandOverlay,
        updateHandRadius,
        updateHandSettings,
        renderAll,
    };
})();
