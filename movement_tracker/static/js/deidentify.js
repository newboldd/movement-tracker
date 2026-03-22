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

    // Camera
    let cameraMode = 'stereo';
    let cameraNames = ['OS', 'OD'];
    let currentSide = 'OS';
    let hasMediapipe = false;

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
    let hasUserZoom = false;

    // Pan state
    let panning = false;
    let panStart = null;

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
        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('mouseleave', onMouseUp);

        // Keyboard
        document.addEventListener('keydown', onKeyDown);

        // Resize
        window.addEventListener('resize', () => { if (!hasUserZoom) fitImage(); render(); });

        // Auto-select subject from URL param or if only one
        const params = new URLSearchParams(window.location.search);
        const urlSubject = params.get('subject');
        if (urlSubject) {
            sel.value = urlSubject;
            loadSubject(parseInt(urlSubject));
        } else if (subjects.length === 1) {
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
            cameraMode = data.subject.camera_mode || 'stereo';
            hasMediapipe = data.has_mediapipe || false;

            // Set camera names
            if (cameraMode === 'single') {
                cameraNames = [data.camera_names ? data.camera_names[0] : 'OS'];
                currentSide = cameraNames[0];
            } else if (cameraMode === 'multicam' && trials.length > 0
                       && trials[0].cameras && trials[0].cameras.length > 0) {
                cameraNames = trials[0].cameras.map(c => c.name);
                currentSide = cameraNames[0];
            } else {
                cameraNames = data.camera_names || ['OS', 'OD'];
                currentSide = cameraNames[0];
            }
        } catch (e) {
            document.getElementById('statusMsg').textContent = 'Error loading subject: ' + e.message;
            return;
        }

        // Update side toggle button
        const sideBtn = document.getElementById('sideToggle');
        if (sideBtn) {
            sideBtn.style.display = (cameraMode === 'single') ? 'none' : '';
            sideBtn.textContent = currentSide;
        }

        // Update hand section availability
        _updateHandSection();

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

    // ── Update hand overlay section availability ──
    function _updateHandSection() {
        const handSection = document.getElementById('handSection');
        if (!handSection) return;

        // Grey out checkboxes if no MP labels, but keep MP button always active
        const checkboxes = handSection.querySelectorAll('input[type=checkbox], input[type=range]');
        checkboxes.forEach(el => { el.disabled = !hasMediapipe; });
        if (!hasMediapipe) {
            handSection.classList.add('hand-disabled');
        } else {
            handSection.classList.remove('hand-disabled');
        }

        // Update button text — always visible
        const mpBtn = document.getElementById('goMediapipeBtn');
        if (mpBtn) {
            mpBtn.textContent = hasMediapipe ? 'Re-run MediaPipe' : 'Run MediaPipe';
        }
    }

    // ── Select trial ──
    async function selectTrial(idx) {
        if (playing) togglePlay();
        currentTrialIdx = idx;
        trialMeta = trials[idx];
        currentFrame = trialMeta.start_frame;
        totalFrames = trialMeta.frame_count;
        fps = trialMeta.fps || 30;
        handLandmarks = [];

        // Load saved face detections from DB
        try {
            const fdResp = await API.get(`/api/deidentify/${subjectId}/face-detections?trial_idx=${idx}`);
            faceDetections = fdResp.faces || [];
            if (faceDetections.length > 0) {
                const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
                document.getElementById('faceDetStatus').textContent =
                    `Loaded ${nFaces} frames with faces (saved)`;
            }
        } catch (e) {
            faceDetections = [];
        }

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

        // Reset zoom for new trial
        hasUserZoom = false;
        scale = 1; offsetX = 0; offsetY = 0;

        await loadFrame(currentFrame);
        renderSpotList();
    }

    // ── Frame loading ──
    async function loadFrame(frameNum) {
        currentFrame = frameNum;
        const url = `/api/deidentify/${subjectId}/frame?trial_idx=${currentTrialIdx}&frame_num=${frameNum}&side=${encodeURIComponent(currentSide)}`;

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

        if (!hasUserZoom) fitImage();

        // Update slider + display
        document.getElementById('frameSlider').value = frameNum;
        const localFrame = frameNum - (trialMeta ? trialMeta.start_frame : 0);
        document.getElementById('frameDisplay').textContent =
            `Frame: ${localFrame} / ${totalFrames - 1}`;

        // Load hand landmarks if overlay enabled
        if (handOverlayEnabled && hasMediapipe) {
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

    function resetZoom() {
        hasUserZoom = false;
        fitImage();
        render();
    }

    // ── Camera toggle ──
    function toggleSide() {
        if (cameraMode === 'single') return;
        const idx = cameraNames.indexOf(currentSide);
        const newIdx = (idx + 1) % cameraNames.length;
        currentSide = cameraNames[newIdx];

        const btn = document.getElementById('sideToggle');
        if (btn) btn.textContent = currentSide;

        loadFrame(currentFrame);
    }

    // ── Go to MediaPipe page for this subject ──
    function goToMediapipe() {
        if (!subjectId) return;
        // Preserve current frame and zoom state via session storage
        sessionStorage.setItem('deid_returnFrame', currentFrame);
        sessionStorage.setItem('deid_returnSubject', subjectId);
        window.location.href = `/mediapipe-select?subject=${subjectId}`;
    }

    // ── Side label mapping (stereo: left/right ↔ camera names) ──
    function _sideLabel() {
        // Map currentSide camera name to the detection side label
        if (cameraMode !== 'stereo') return 'full';
        if (cameraNames.length >= 2 && currentSide === cameraNames[1]) return 'right';
        return 'left';
    }

    function _facesForCurrentSide(entry) {
        if (!entry || !entry.faces) return [];
        if (cameraMode === 'single') return entry.faces;
        const label = _sideLabel();
        return entry.faces.filter(f => (f.side || 'full') === label || (f.side || 'full') === 'full');
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

        // Draw face detections (blue dashed rectangles) — filtered by current side
        const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
        if (faceDetections.length > localFrame) {
            const faces = _facesForCurrentSide(faceDetections[localFrame]);
            if (faces.length > 0) {
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = 'rgba(33,150,243,0.8)';
                ctx.lineWidth = 2;
                for (const f of faces) {
                    const sx = offsetX + f.x1 * scale;
                    const sy = offsetY + f.y1 * scale;
                    const sw = (f.x2 - f.x1) * scale;
                    const sh = (f.y2 - f.y1) * scale;
                    ctx.strokeRect(sx, sy, sw, sh);
                }
                ctx.setLineDash([]);
            }
        }

        // Draw blur spots (red semi-transparent circles) — filtered by current side
        const curSideLabel = _sideLabel();
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const spotSide = spot.side || 'full';
            if (spotSide !== 'full' && spotSide !== curSideLabel) continue;
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

        // Draw hand landmarks (green dots) — filtered by current side
        const sideLabel = _sideLabel();
        const visibleLandmarks = handLandmarks.filter(lm =>
            (lm.side || 'full') === sideLabel || (lm.side || 'full') === 'full'
        );
        if (handOverlayEnabled && visibleLandmarks.length > 0) {
            ctx.fillStyle = 'rgba(76,175,80,0.7)';
            for (const lm of visibleLandmarks) {
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
                for (const lm of visibleLandmarks) {
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
            ctx.beginPath();
            ctx.arc(cw / 2, ch / 2, 30, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    // ── Canvas click ──
    function onCanvasClick(e) {
        // Don't handle clicks after a pan drag
        if (panning) return;

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
        hasUserZoom = true;
        render();
    }

    // ── Pan (right-click drag or middle-click drag) ──
    function onMouseDown(e) {
        // Right-click or middle-click for pan
        if (e.button === 1 || e.button === 2) {
            e.preventDefault();
            panning = true;
            panStart = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
            canvas.style.cursor = 'grabbing';
            return;
        }
        // Left-click pan when zoomed in (hold and drag)
        if (e.button === 0 && hasUserZoom && !addingCustom) {
            panning = true;
            panStart = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
            canvas.style.cursor = 'grabbing';
        }
    }

    function onMouseMove(e) {
        if (!panning || !panStart) return;
        offsetX = panStart.ox + (e.clientX - panStart.x);
        offsetY = panStart.oy + (e.clientY - panStart.y);
        render();
    }

    function onMouseUp(e) {
        if (panning) {
            canvas.style.cursor = '';
            // If mouse barely moved, don't suppress click
            if (panStart) {
                const dx = Math.abs(e.clientX - panStart.x);
                const dy = Math.abs(e.clientY - panStart.y);
                if (dx < 3 && dy < 3) {
                    panning = false;
                    panStart = null;
                    return; // allow click to fire
                }
            }
            // Suppress click after real pan drag
            setTimeout(() => { panning = false; }, 50);
            panStart = null;
        }
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
        } else if (e.key === 'e') {
            toggleSide();
        } else if (e.key === 'r') {
            resetZoom();
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

            // Auto-create blur spots from unique face clusters
            _autoCreateFaceSpots();
            renderSpotList();
            scheduleSave();
            render();
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
        }
        btn.disabled = false;
    }

    // ── Auto-create blur spots from face detections ──
    function _autoCreateFaceSpots() {
        if (!faceDetections.length || !trialMeta) return;

        // Find the most common face position per side (cluster by center)
        const clusters = {}; // key: side → [{cx, cy, w, h, count, firstFrame, lastFrame}]

        for (const entry of faceDetections) {
            for (const f of entry.faces) {
                const side = f.side || 'full';
                if (!clusters[side]) clusters[side] = [];

                const cx = (f.x1 + f.x2) / 2;
                const cy = (f.y1 + f.y2) / 2;
                const w = f.x2 - f.x1;
                const h = f.y2 - f.y1;

                // Find nearest existing cluster
                let matched = false;
                for (const c of clusters[side]) {
                    const dist = Math.sqrt((cx - c.cx) ** 2 + (cy - c.cy) ** 2);
                    if (dist < Math.max(w, h) * 1.5) {
                        // Update running average
                        c.cx = (c.cx * c.count + cx) / (c.count + 1);
                        c.cy = (c.cy * c.count + cy) / (c.count + 1);
                        c.w = Math.max(c.w, w);
                        c.h = Math.max(c.h, h);
                        c.count++;
                        c.firstFrame = Math.min(c.firstFrame, entry.frame);
                        c.lastFrame = Math.max(c.lastFrame, entry.frame);
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    clusters[side].push({
                        cx, cy, w, h, count: 1,
                        firstFrame: entry.frame, lastFrame: entry.frame,
                    });
                }
            }
        }

        // Remove existing auto-generated face spots for this trial
        blurSpots = blurSpots.filter(s => s.spot_type !== 'face');

        // Create a blur spot for each significant cluster
        for (const [side, clusterList] of Object.entries(clusters)) {
            for (const c of clusterList) {
                if (c.count < 3) continue; // Skip noise (< 3 frames)
                blurSpots.push({
                    id: nextSpotId++,
                    spot_type: 'face',
                    x: Math.round(c.cx),
                    y: Math.round(c.cy),
                    radius: Math.round(Math.max(c.w, c.h) / 2 * 1.2), // 20% larger than face
                    frame_start: c.firstFrame,
                    frame_end: c.lastFrame,
                    side: side,
                });
            }
        }
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
                <span class="spot-label">${s.spot_type === 'face' ? '\u{1F464}' : '\u2295'}
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
        if (enabled && subjectId && hasMediapipe) {
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
        toggleSide,
        resetZoom,
        seekFrame,
        toggleAddCustom,
        selectSpot,
        deleteSpot,
        updateSpotRadius,
        updateSpotFrameRange,
        toggleHandOverlay,
        updateHandRadius,
        updateHandSettings,
        goToMediapipe,
        renderAll,
    };
})();
