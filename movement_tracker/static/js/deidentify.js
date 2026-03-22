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

    // Timeline
    let tlCanvas, tlCtx;
    let tlDragSpot = null;    // spot being dragged (blur spot object or 'hand')
    let tlDragEdge = null;    // 'start', 'end', or 'move'
    let tlDragStartX = null;  // mouse X at drag start
    let tlDragOrigRange = null; // {start, end} at drag start
    let handCoverage = [];    // array of frame numbers with MP hand data
    let handProtectStart = 0; // frame range for hand protection
    let handProtectEnd = 0;

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

        // Timeline setup
        tlCanvas = document.getElementById('timelineCanvas');
        tlCtx = tlCanvas ? tlCanvas.getContext('2d') : null;
        if (tlCanvas) {
            setupTimeline();
        }

        // Keyboard
        document.addEventListener('keydown', onKeyDown);

        // Resize
        window.addEventListener('resize', () => {
            if (!hasUserZoom) fitImage();
            render();
            renderTimeline();
        });

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

        // Grey out controls if no MP labels, but keep MP button always active
        const inputs = handSection.querySelectorAll('input[type=checkbox], input[type=range]');
        inputs.forEach(el => { el.disabled = !hasMediapipe; });
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

        // (frame slider removed — timeline handles navigation)

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
            handMaskEnabled = true; // always enabled, controlled by timeline range
            handMaskRadius = hs.mask_radius;
            handProtectStart = hs.frame_start != null ? hs.frame_start : trialMeta.start_frame;
            handProtectEnd = hs.frame_end != null ? hs.frame_end : trialMeta.end_frame;
            document.getElementById('handRadiusSlider').value = handMaskRadius;
            document.getElementById('handRadiusVal').textContent = handMaskRadius;
        } catch (e) {
            handProtectStart = trialMeta.start_frame;
            handProtectEnd = trialMeta.end_frame;
        }

        // Reset zoom for new trial
        hasUserZoom = false;
        scale = 1; offsetX = 0; offsetY = 0;

        // Load hand coverage for timeline
        handCoverage = [];
        if (hasMediapipe) {
            try {
                const hc = await API.get(`/api/deidentify/${subjectId}/hand-coverage?trial_idx=${idx}`);
                handCoverage = hc.frames || [];
            } catch (e) {}
        }

        await loadFrame(currentFrame);
        renderSpotList();
        renderTimeline();
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

        // Update display
        const localFrame = frameNum - (trialMeta ? trialMeta.start_frame : 0);
        document.getElementById('frameDisplay').textContent =
            `Frame: ${localFrame} / ${totalFrames - 1}`;

        // Load hand landmarks if overlay enabled or protection active for this frame
        const needHands = hasMediapipe && (handOverlayEnabled
            || (handMaskEnabled && frameNum >= handProtectStart && frameNum <= handProtectEnd));
        if (needHands) {
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
        renderTimeline();
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

    // ── Get face spot position for current frame (tracks detection centroid) ──
    function _getSpotPosition(spot) {
        // For face spots: track the nearest face detection centroid + offset
        if (spot.spot_type === 'face' && faceDetections.length > 0) {
            const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
            if (localFrame >= 0 && localFrame < faceDetections.length) {
                const faces = _facesForCurrentSide(faceDetections[localFrame]);
                if (faces.length > 0) {
                    // Find closest face to spot's reference position
                    let bestFace = faces[0];
                    let bestDist = Infinity;
                    for (const f of faces) {
                        const cx = (f.x1 + f.x2) / 2;
                        const cy = (f.y1 + f.y2) / 2;
                        const d = Math.sqrt((cx - spot.x) ** 2 + (cy - spot.y) ** 2);
                        if (d < bestDist) {
                            bestDist = d;
                            bestFace = f;
                        }
                    }
                    return {
                        x: (bestFace.x1 + bestFace.x2) / 2 + (spot.offset_x || 0),
                        y: (bestFace.y1 + bestFace.y2) / 2 + (spot.offset_y || 0),
                    };
                }
            }
        }
        // Fallback: static position + offset
        return { x: spot.x + (spot.offset_x || 0), y: spot.y + (spot.offset_y || 0) };
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

        // Compute hand protection state (controlled by timeline range, not checkbox)
        const curSideLabel = _sideLabel();
        const visibleLandmarks = handLandmarks.filter(lm =>
            (lm.side || 'full') === curSideLabel || (lm.side || 'full') === 'full'
        );
        const handProtectActive = handMaskEnabled
            && visibleLandmarks.length > 0
            && currentFrame >= handProtectStart && currentFrame <= handProtectEnd;

        // Draw blur spots with hand protection subtracted via offscreen canvas
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const spotSide = spot.side || 'full';
            if (spotSide !== 'full' && spotSide !== curSideLabel) continue;

            const pos = _getSpotPosition(spot);
            const sx = offsetX + pos.x * scale;
            const sy = offsetY + pos.y * scale;
            const isFace = spot.spot_type === 'face';
            const isSelected = spot.id === selectedSpotId;
            const sw = (spot.width || spot.radius * 2) * scale / 2;
            const sh = (spot.height || spot.radius * 2) * scale / 2;

            const cr = isFace ? 33 : 244;
            const cg = isFace ? 150 : 67;
            const cb = isFace ? 243 : 54;
            const fillAlpha = isSelected ? 0.35 : 0.2;

            if (handProtectActive) {
                // Offscreen canvas: draw blur ellipse, then punch out hand circles
                const offCanvas = document.createElement('canvas');
                offCanvas.width = cw;
                offCanvas.height = ch;
                const off = offCanvas.getContext('2d');

                // Draw the blur ellipse fill
                off.fillStyle = `rgba(${cr},${cg},${cb},${fillAlpha})`;
                off.beginPath();
                off.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
                off.fill();

                // Subtract hand protection circles
                off.globalCompositeOperation = 'destination-out';
                const hr = handMaskRadius * scale;
                for (const lm of visibleLandmarks) {
                    const lx = offsetX + lm.x * scale;
                    const ly = offsetY + lm.y * scale;
                    off.beginPath();
                    off.arc(lx, ly, hr, 0, Math.PI * 2);
                    off.fill();
                }

                // Composite result onto main canvas
                ctx.drawImage(offCanvas, 0, 0);
            } else {
                // Simple fill (no hand subtraction)
                ctx.beginPath();
                ctx.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${cr},${cg},${cb},${fillAlpha})`;
                ctx.fill();
            }

            // Always draw the outline
            ctx.beginPath();
            ctx.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
            ctx.strokeStyle = isSelected
                ? `rgba(${cr},${cg},${cb},0.9)`
                : `rgba(${cr},${cg},${cb},0.5)`;
            ctx.lineWidth = isSelected ? 2 : 1;
            ctx.stroke();
        }

        // Draw hand landmarks (green dots) — controlled by "Show hand landmarks" checkbox
        if (handOverlayEnabled && visibleLandmarks.length > 0) {
            ctx.fillStyle = 'rgba(76,175,80,0.7)';
            for (const lm of visibleLandmarks) {
                const sx = offsetX + lm.x * scale;
                const sy = offsetY + lm.y * scale;
                ctx.beginPath();
                ctx.arc(sx, sy, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw hand protection outline — controlled by timeline range (not checkbox)
        if (handProtectActive) {
            const hr = handMaskRadius * scale;
            // Offscreen canvas to build the union outline
            const hOff = document.createElement('canvas');
            hOff.width = cw;
            hOff.height = ch;
            const hCtx = hOff.getContext('2d');

            // Fill all circles as solid white (natural union)
            hCtx.fillStyle = '#fff';
            for (const lm of visibleLandmarks) {
                const lx = offsetX + lm.x * scale;
                const ly = offsetY + lm.y * scale;
                hCtx.beginPath();
                hCtx.arc(lx, ly, hr, 0, Math.PI * 2);
                hCtx.fill();
            }

            // Erase interior to leave 2px border ring
            hCtx.globalCompositeOperation = 'destination-out';
            for (const lm of visibleLandmarks) {
                const lx = offsetX + lm.x * scale;
                const ly = offsetY + lm.y * scale;
                hCtx.beginPath();
                hCtx.arc(lx, ly, hr - 2, 0, Math.PI * 2);
                hCtx.fill();
            }

            // Tint green
            hCtx.globalCompositeOperation = 'source-in';
            hCtx.fillStyle = 'rgba(76,175,80,0.8)';
            hCtx.fillRect(0, 0, cw, ch);

            ctx.drawImage(hOff, 0, 0);

            // Subtle fill
            ctx.fillStyle = 'rgba(76,175,80,0.06)';
            for (const lm of visibleLandmarks) {
                const lx = offsetX + lm.x * scale;
                const ly = offsetY + lm.y * scale;
                ctx.beginPath();
                ctx.arc(lx, ly, hr, 0, Math.PI * 2);
                ctx.fill();
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
                        const fw = f.x2 - f.x1;
                        const fh = f.y2 - f.y1;
                        const spot = {
                            id: nextSpotId++,
                            spot_type: 'face',
                            x: Math.round((f.x1 + f.x2) / 2),
                            y: Math.round((f.y1 + f.y2) / 2),
                            radius: Math.round(Math.max(fw, fh) / 2 * 1.2),
                            width: Math.round(fw * 1.2),
                            height: Math.round(fh * 1.2),
                            offset_x: 0,
                            offset_y: 0,
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
                width: 80,
                height: 80,
                offset_x: 0,
                offset_y: 0,
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
        const btn = document.getElementById('playBtn');
        if (playing) {
            // Stop
            playing = false;
            if (playTimer) { clearInterval(playTimer); playTimer = null; }
            btn.innerHTML = '&#9654;';
        } else {
            if (!trialMeta) return;
            // Start
            playing = true;
            btn.innerHTML = '&#9616;&#9616;';
            let loading = false; // prevent overlapping loads
            playTimer = setInterval(() => {
                if (!playing) return;
                if (loading) return;
                if (currentFrame >= trialMeta.end_frame) {
                    togglePlay();
                    return;
                }
                loading = true;
                currentFrame++;
                // Simple sync render: just update frame and draw
                const url = `/api/deidentify/${subjectId}/frame?trial_idx=${currentTrialIdx}&frame_num=${currentFrame}&side=${encodeURIComponent(currentSide)}`;
                const img = new Image();
                img.onload = () => {
                    currentImage = img;
                    imgW = img.width;
                    imgH = img.height;
                    const localFrame = currentFrame - trialMeta.start_frame;
                    document.getElementById('frameDisplay').textContent =
                        `Frame: ${localFrame} / ${totalFrames - 1}`;
                    render();
                    loading = false;
                };
                img.onerror = () => { loading = false; };
                img.src = url;
            }, 1000 / fps);
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
                    radius: Math.round(Math.max(c.w, c.h) / 2 * 1.2),
                    width: Math.round(c.w * 1.2),
                    height: Math.round(c.h * 1.2),
                    offset_x: 0,
                    offset_y: 0,
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
        renderTimeline();
    }

    function updateSpotControls() {
        const controls = document.getElementById('spotControls');
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (!spot) {
            if (controls) controls.style.display = 'none';
            return;
        }
        if (controls) controls.style.display = 'block';
        const w = spot.width || spot.radius * 2;
        const h = spot.height || spot.radius * 2;
        document.getElementById('widthSlider').value = w;
        document.getElementById('widthVal').textContent = Math.round(w);
        document.getElementById('heightSlider').value = h;
        document.getElementById('heightVal').textContent = Math.round(h);
        document.getElementById('offsetXSlider').value = spot.offset_x || 0;
        document.getElementById('offsetXVal').textContent = Math.round(spot.offset_x || 0);
        document.getElementById('offsetYSlider').value = spot.offset_y || 0;
        document.getElementById('offsetYVal').textContent = Math.round(spot.offset_y || 0);
        document.getElementById('frameStartInput').value = spot.frame_start;
        document.getElementById('frameEndInput').value = spot.frame_end;
    }

    function updateSpotDim(dim, val) {
        val = parseInt(val);
        const valEl = document.getElementById(
            dim === 'width' ? 'widthVal' :
            dim === 'height' ? 'heightVal' :
            dim === 'offset_x' ? 'offsetXVal' : 'offsetYVal'
        );
        if (valEl) valEl.textContent = val;
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (spot) {
            spot[dim] = val;
            // Keep radius in sync as max(width, height) / 2 for backward compat
            if (dim === 'width' || dim === 'height') {
                spot.radius = Math.round(Math.max(spot.width || 0, spot.height || 0) / 2);
            }
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
        renderTimeline();
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
        // handMaskEnabled is always true — protection range controlled by timeline
        if (!subjectId || currentTrialIdx < 0) return;
        try {
            await API.put(`/api/deidentify/${subjectId}/hand-settings`, {
                trial_idx: currentTrialIdx,
                enabled: handMaskEnabled,
                mask_radius: handMaskRadius,
                frame_start: handProtectStart,
                frame_end: handProtectEnd,
            });
        } catch (e) {}
        renderTimeline();
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
            width: s.width || null,
            height: s.height || null,
            offset_x: s.offset_x || 0,
            offset_y: s.offset_y || 0,
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

    // ── Timeline ──────────────────────────────────────────────────────────

    function setupTimeline() {
        if (!tlCanvas) return;
        const ro = new ResizeObserver(() => renderTimeline());
        ro.observe(tlCanvas.parentElement);

        tlCanvas.addEventListener('mousedown', onTlMouseDown);
        tlCanvas.addEventListener('mousemove', onTlMouseMove);
        tlCanvas.addEventListener('mouseup', onTlMouseUp);
        tlCanvas.addEventListener('mouseleave', onTlMouseUp);
        tlCanvas.addEventListener('click', onTlClick);
    }

    function _tlLayout() {
        // Returns layout metrics for timeline rendering
        if (!trialMeta || !tlCanvas) return null;
        const cw = tlCanvas.width;
        const ch = tlCanvas.height;
        const margin = 4;
        const labelW = 50; // left label area
        const barX = labelW;
        const barW = cw - labelW - margin;
        const start = trialMeta.start_frame;
        const end = trialMeta.end_frame;
        const range = end - start || 1;
        return { cw, ch, margin, labelW, barX, barW, start, end, range };
    }

    function _frameToTlX(frame, layout) {
        return layout.barX + ((frame - layout.start) / layout.range) * layout.barW;
    }

    function _tlXToFrame(x, layout) {
        return Math.round(layout.start + ((x - layout.barX) / layout.barW) * layout.range);
    }

    function renderTimeline() {
        if (!tlCtx || !tlCanvas || !trialMeta) return;

        const container = tlCanvas.parentElement;
        tlCanvas.width = container.clientWidth;
        tlCanvas.height = container.clientHeight;
        const L = _tlLayout();
        if (!L) return;

        tlCtx.clearRect(0, 0, L.cw, L.ch);

        const sideLabel = _sideLabel();
        const visibleSpots = blurSpots.filter(s => {
            const ss = s.side || 'full';
            return ss === 'full' || ss === sideLabel;
        });

        const faceRowH = 35;
        const handRowH = 25;
        const gap = 3;
        let y = 2;

        // ── Face row: dark detection coverage + bright blue blur bars ──
        tlCtx.fillStyle = 'rgba(150,150,150,0.5)';
        tlCtx.font = '10px sans-serif';
        tlCtx.fillText('Faces', 2, y + 12);

        // Dark background showing where faces were detected
        if (faceDetections.length > 0) {
            tlCtx.fillStyle = 'rgba(33,150,243,0.15)';
            const pw = Math.max(1, L.barW / L.range);
            for (let i = 0; i < faceDetections.length; i++) {
                const entry = faceDetections[i];
                if (!entry || !entry.faces || entry.faces.length === 0) continue;
                const frame = entry.frame != null ? entry.frame : (trialMeta.start_frame + i);
                const x = _frameToTlX(frame, L);
                tlCtx.fillRect(x, y, pw, faceRowH);
            }
        }

        // Bright blue blur spot bars on top (matching face detection box color)
        if (visibleSpots.length > 0) {
            const spotBarH = Math.min(14, (faceRowH - 4) / visibleSpots.length);
            let spotY = y + 2;
            for (const spot of visibleSpots) {
                const x1 = _frameToTlX(spot.frame_start, L);
                const x2 = _frameToTlX(spot.frame_end, L);
                const w = Math.max(3, x2 - x1);
                const isSelected = spot.id === selectedSpotId;

                tlCtx.fillStyle = isSelected
                    ? 'rgba(33,150,243,0.6)'
                    : 'rgba(33,150,243,0.35)';
                tlCtx.fillRect(x1, spotY, w, spotBarH - 1);

                tlCtx.strokeStyle = isSelected
                    ? 'rgba(33,150,243,1.0)'
                    : 'rgba(33,150,243,0.7)';
                tlCtx.lineWidth = isSelected ? 1.5 : 0.5;
                tlCtx.strokeRect(x1, spotY, w, spotBarH - 1);

                // Edge handles for selected
                if (isSelected) {
                    tlCtx.fillStyle = 'rgba(33,150,243,1.0)';
                    tlCtx.fillRect(x1 - 1, spotY, 3, spotBarH - 1);
                    tlCtx.fillRect(x2 - 2, spotY, 3, spotBarH - 1);
                }

                spotY += spotBarH;
            }
        }

        y += faceRowH + gap;

        // ── Hand row: green coverage + draggable protection range ──
        if (handCoverage.length > 0 || hasMediapipe) {
            tlCtx.fillStyle = 'rgba(150,150,150,0.5)';
            tlCtx.fillText('Hands', 2, y + 12);

            // Background coverage heatmap
            if (handCoverage.length > 0) {
                tlCtx.fillStyle = 'rgba(76,175,80,0.2)';
                const pw = Math.max(1, L.barW / L.range);
                for (const f of handCoverage) {
                    const x = _frameToTlX(f, L);
                    tlCtx.fillRect(x, y, pw, handRowH);
                }
            }

            // Protection active range bar (draggable)
            if (handMaskEnabled) {
                const hx1 = _frameToTlX(handProtectStart, L);
                const hx2 = _frameToTlX(handProtectEnd, L);
                const hw = Math.max(3, hx2 - hx1);

                tlCtx.fillStyle = 'rgba(76,175,80,0.45)';
                tlCtx.fillRect(hx1, y + 2, hw, handRowH - 4);
                tlCtx.strokeStyle = 'rgba(76,175,80,0.9)';
                tlCtx.lineWidth = 1.5;
                tlCtx.strokeRect(hx1, y + 2, hw, handRowH - 4);

                // Edge handles
                tlCtx.fillStyle = 'rgba(76,175,80,1.0)';
                tlCtx.fillRect(hx1 - 1, y + 2, 3, handRowH - 4);
                tlCtx.fillRect(hx2 - 2, y + 2, 3, handRowH - 4);
            }

            y += handRowH + gap;
        }

        // ── Current frame indicator ──
        const cfx = _frameToTlX(currentFrame, L);
        tlCtx.strokeStyle = '#fff';
        tlCtx.lineWidth = 1;
        tlCtx.beginPath();
        tlCtx.moveTo(cfx, 0);
        tlCtx.lineTo(cfx, L.ch);
        tlCtx.stroke();
    }

    // ── Timeline mouse handlers ──

    function _tlHitTest(e) {
        // Returns { type: 'spot'|'hand', spot?, edge: 'start'|'end'|'move' } or null
        const rect = tlCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const L = _tlLayout();
        if (!L) return null;

        const sideLabel = _sideLabel();
        const visibleSpots = blurSpots.filter(s => {
            const ss = s.side || 'full';
            return ss === 'full' || ss === sideLabel;
        });

        const faceRowH = 35, handRowH = 25, gap = 3;
        let y = 2;

        // Hit test blur spot bars within face row
        if (visibleSpots.length > 0) {
            const spotBarH = Math.min(12, (faceRowH - 4) / visibleSpots.length);
            let spotY = y + 2;
            for (const spot of visibleSpots) {
                const x1 = _frameToTlX(spot.frame_start, L);
                const x2 = _frameToTlX(spot.frame_end, L);

                if (my >= spotY && my <= spotY + spotBarH) {
                    if (Math.abs(mx - x1) < 6) return { type: 'spot', spot, edge: 'start' };
                    if (Math.abs(mx - x2) < 6) return { type: 'spot', spot, edge: 'end' };
                    if (mx > x1 + 4 && mx < x2 - 4) return { type: 'spot', spot, edge: 'move' };
                }
                spotY += spotBarH;
            }
        }

        y += faceRowH + gap;

        // Hit test hand protection bar
        if ((handCoverage.length > 0 || hasMediapipe) && handMaskEnabled) {
            const hx1 = _frameToTlX(handProtectStart, L);
            const hx2 = _frameToTlX(handProtectEnd, L);

            if (my >= y + 2 && my <= y + handRowH - 2) {
                if (Math.abs(mx - hx1) < 6) return { type: 'hand', edge: 'start' };
                if (Math.abs(mx - hx2) < 6) return { type: 'hand', edge: 'end' };
                if (mx > hx1 + 4 && mx < hx2 - 4) return { type: 'hand', edge: 'move' };
            }
        }

        return null;
    }

    function onTlMouseDown(e) {
        const hit = _tlHitTest(e);
        if (!hit) return;
        e.preventDefault();

        if (hit.type === 'spot') {
            tlDragSpot = hit.spot;
            tlDragEdge = hit.edge;
            tlDragStartX = e.clientX;
            tlDragOrigRange = { start: hit.spot.frame_start, end: hit.spot.frame_end };
            selectedSpotId = hit.spot.id;
            renderSpotList();
            updateSpotControls();
        } else if (hit.type === 'hand') {
            tlDragSpot = 'hand';
            tlDragEdge = hit.edge;
            tlDragStartX = e.clientX;
            tlDragOrigRange = { start: handProtectStart, end: handProtectEnd };
        }
        renderTimeline();
    }

    function onTlMouseMove(e) {
        if (tlDragSpot && tlDragEdge) {
            const L = _tlLayout();
            if (!L) return;
            const dx = e.clientX - tlDragStartX;
            const dFrames = Math.round((dx / L.barW) * L.range);

            if (tlDragSpot === 'hand') {
                // Drag hand protection range
                if (tlDragEdge === 'start') {
                    handProtectStart = Math.max(
                        trialMeta.start_frame,
                        Math.min(handProtectEnd - 1, tlDragOrigRange.start + dFrames)
                    );
                } else if (tlDragEdge === 'end') {
                    handProtectEnd = Math.min(
                        trialMeta.end_frame,
                        Math.max(handProtectStart + 1, tlDragOrigRange.end + dFrames)
                    );
                } else if (tlDragEdge === 'move') {
                    const len = tlDragOrigRange.end - tlDragOrigRange.start;
                    let newStart = tlDragOrigRange.start + dFrames;
                    newStart = Math.max(trialMeta.start_frame, Math.min(trialMeta.end_frame - len, newStart));
                    handProtectStart = newStart;
                    handProtectEnd = newStart + len;
                }
                renderTimeline();
                return;
            }

            if (tlDragEdge === 'start') {
                tlDragSpot.frame_start = Math.max(
                    trialMeta.start_frame,
                    Math.min(tlDragSpot.frame_end - 1, tlDragOrigRange.start + dFrames)
                );
            } else if (tlDragEdge === 'end') {
                tlDragSpot.frame_end = Math.min(
                    trialMeta.end_frame,
                    Math.max(tlDragSpot.frame_start + 1, tlDragOrigRange.end + dFrames)
                );
            } else if (tlDragEdge === 'move') {
                const len = tlDragOrigRange.end - tlDragOrigRange.start;
                let newStart = tlDragOrigRange.start + dFrames;
                newStart = Math.max(trialMeta.start_frame, Math.min(trialMeta.end_frame - len, newStart));
                tlDragSpot.frame_start = newStart;
                tlDragSpot.frame_end = newStart + len;
            }

            updateSpotControls();
            render();
            renderTimeline();
            return;
        }

        // Hover cursor
        const hit = _tlHitTest(e);
        if (hit) {
            tlCanvas.style.cursor = hit.edge === 'move' ? 'grab' : 'ew-resize';
        } else {
            tlCanvas.style.cursor = 'pointer';
        }
    }

    function onTlMouseUp(e) {
        if (tlDragSpot) {
            if (tlDragSpot === 'hand') {
                updateHandSettings(); // saves frame range
            } else {
                scheduleSave();
            }
            tlDragSpot = null;
            tlDragEdge = null;
            tlDragStartX = null;
            tlDragOrigRange = null;
        }
    }

    function onTlClick(e) {
        if (tlDragSpot) return; // was a drag, not click
        const rect = tlCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const L = _tlLayout();
        if (!L) return;

        // Always seek to clicked frame
        const frame = Math.max(L.start, Math.min(L.end, _tlXToFrame(mx, L)));
        seekFrame(frame);

        // Also select spot if clicking on one
        const hit = _tlHitTest(e);
        if (hit && hit.type === 'spot') {
            selectedSpotId = hit.spot.id;
            renderSpotList();
            updateSpotControls();
            renderTimeline();
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
        updateSpotDim,
        updateSpotFrameRange,
        toggleHandOverlay,
        updateHandRadius,
        updateHandSettings,
        goToMediapipe,
        renderAll,
    };
})();
