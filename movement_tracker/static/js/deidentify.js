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
    let playbackSpeed = 1.0;

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

    // View mode: 'original' (with overlays), 'preview' (blur mask only), 'deidentified' (rendered video)
    let viewMode = 'original';

    // Hand overlay
    let handOverlayEnabled = false;
    let handLandmarks = [];   // [{x, y, side}] for current frame (after smoothing)
    let handLandmarksBulk = {}; // {frameNum: [{x, y, side}]} all frames
    let handTemporalSmooth = 0; // temporal smoothing window (frames each direction)
    let handMaskRadius = 5;
    let forearmRadius = 10;  // dilation around forearm triangle (separate from circle radius)
    let forearmExtent = 0.4; // 0=wrist, 1=elbow, >1=past elbow
    let handSmooth = 7;  // morphological close on hand circles
    let handSmooth2 = 5;  // morphological close after joining forearm
    let handMaskEnabled = true;
    let dlcRadius = 15;  // separate radius for DLC thumb/index markers
    let hasDlcLabels = false;

    // Canvas
    let canvas, ctx;
    let currentImage = null;
    let imgW = 0, imgH = 0;
    let scale = 1, offsetX = 0, offsetY = 0;
    let hasUserZoom = false;

    // Video element for smooth playback
    let videoEl = null;
    let videoPlaying = false;
    let currentVideoTrialIdx = -1;
    let currentVideoBlurred = false;

    // Pan state
    let panning = false;
    let panStart = null;

    // Spot drag state (for on-canvas resize/move)
    let spotDrag = null; // {spotId, handle, startMx, startMy, origSpot}

    // Timeline
    let tlCanvas, tlCtx;
    let tlDragSpot = null;    // spot being dragged (blur spot object, segment object, or 'newhand')
    let tlDragEdge = null;    // 'start', 'end', 'move', or 'create'
    let tlDragStartX = null;  // mouse X at drag start
    let tlDragOrigRange = null; // {start, end} at drag start
    let tlDragCreateFrame = null; // frame at start of hand segment creation drag
    let handCoverage = [];    // array of frame numbers with MP hand data
    // Multiple hand protection segments: [{id, start, end, radius}]
    let handProtectSegments = [];
    let nextHandSegId = 1;
    let selectedHandSegId = null;

    // Undo/redo stack
    const MAX_UNDO = 50;
    let undoStack = [];
    let redoStack = [];

    function _snapshotState() {
        return {
            blurSpots: blurSpots.map(s => ({ ...s })),
            handProtectSegments: handProtectSegments.map(s => ({ ...s })),
            selectedSpotId,
            selectedHandSegId,
        };
    }

    function _pushUndo() {
        undoStack.push(_snapshotState());
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        redoStack = []; // clear redo on new action
    }

    function undo() {
        if (undoStack.length === 0) return;
        redoStack.push(_snapshotState());
        const state = undoStack.pop();
        blurSpots = state.blurSpots;
        handProtectSegments = state.handProtectSegments;
        selectedSpotId = state.selectedSpotId;
        selectedHandSegId = state.selectedHandSegId;
        renderSpotList();
        updateSpotControls();
        scheduleSave();
        _saveHandSettings();
        render();
        renderTimeline();
    }

    function redo() {
        if (redoStack.length === 0) return;
        undoStack.push(_snapshotState());
        const state = redoStack.pop();
        blurSpots = state.blurSpots;
        handProtectSegments = state.handProtectSegments;
        selectedSpotId = state.selectedSpotId;
        selectedHandSegId = state.selectedHandSegId;
        renderSpotList();
        updateSpotControls();
        scheduleSave();
        _saveHandSettings();
        render();
        renderTimeline();
    }

    // Debounce save timer
    let saveTimer = null;

    // ── Init ──
    async function init() {
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');
        videoEl = document.getElementById('videoPlayer');

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

        // Auto-select subject from URL param, nav state, or first with videos
        const params = new URLSearchParams(window.location.search);
        const urlSubject = params.get('subject');
        const nav = typeof getNavState === 'function' ? getNavState() : {};
        const savedSubject = urlSubject || (nav.subjectId ? String(nav.subjectId) : null)
            || sessionStorage.getItem('dlc_lastSubjectId');
        if (savedSubject && subjects.some(s => String(s.id) === savedSubject)) {
            sel.value = savedSubject;
            await loadSubject(parseInt(savedSubject));
            // Restore trial/frame from nav state
            if (nav.trialIdx != null && nav.trialIdx >= 0 && nav.trialIdx < trials.length) {
                await selectTrial(nav.trialIdx);
                if (nav.frame != null && trialMeta && nav.frame >= trialMeta.start_frame && nav.frame <= trialMeta.end_frame) {
                    await loadFrame(nav.frame);
                }
            }
            if (nav.side && cameraNames.includes(nav.side)) {
                currentSide = nav.side;
                const btn = document.getElementById('sideToggle');
                if (btn) btn.textContent = currentSide;
            }
        } else if (subjects.length >= 1) {
            sel.value = subjects[0].id;
            loadSubject(subjects[0].id);
        }
    }

    // ── Load subject ──
    async function loadSubject(sid) {
        subjectId = sid;
        sessionStorage.setItem('dlc_lastSubjectId', String(sid));
        if (typeof setNavState === 'function') setNavState({ subjectId: sid });
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

        // Update camera label
        const camLabel = document.getElementById('cameraLabel');
        if (camLabel) {
            camLabel.style.display = (cameraMode === 'single') ? 'none' : '';
            camLabel.textContent = currentSide;
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
            // Color: green if no faces or already deidentified, red if needs deident
            if (!t.has_faces || t.has_blurred) {
                btn.style.borderColor = 'var(--green)';
                btn.style.color = 'var(--green)';
            } else {
                btn.style.borderColor = '#e53935';
                btn.style.color = '#e53935';
            }
            btn.onclick = () => selectTrial(i);
            btns.appendChild(btn);
        });

        if (trials.length > 0) selectTrial(0);
    }

    // ── Update hand overlay section availability ──
    function _updateHandSection() {
        const handSection = document.getElementById('handSection');
        if (!handSection) return;

        // Disable controls if no MP labels
        const inputs = handSection.querySelectorAll('input[type=checkbox], input[type=range]');
        inputs.forEach(el => { el.disabled = !hasMediapipe; });
        if (!hasMediapipe) {
            handSection.classList.add('hand-disabled');
        } else {
            handSection.classList.remove('hand-disabled');
        }

        // Show/hide help text vs disabled message
        const helpText = document.getElementById('handHelpText');
        const disabledMsg = document.getElementById('handDisabledMsg');
        if (helpText) helpText.style.display = hasMediapipe ? '' : 'none';
        if (disabledMsg) disabledMsg.style.display = hasMediapipe ? 'none' : 'block';
    }

    // ── Select trial ──
    async function selectTrial(idx) {
        if (playing) togglePlay();
        currentTrialIdx = idx;
        trialMeta = trials[idx];
        currentFrame = trialMeta.start_frame;
        if (typeof setNavState === 'function') setNavState({ trialIdx: idx, frame: currentFrame });
        totalFrames = trialMeta.frame_count;
        fps = trialMeta.fps || 30;
        handLandmarks = [];
        handLandmarksBulk = {}; // clear bulk cache for new trial

        // Load saved face detections from DB
        try {
            const fdResp = await API.get(`/api/deidentify/${subjectId}/face-detections?trial_idx=${idx}`);
            faceDetections = fdResp.faces || [];
            if (faceDetections.length > 0) {
                const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
                document.getElementById('faceDetStatus').textContent =
                    `${nFaces} frames with faces`;
                document.getElementById('detectFacesBtn').style.display = 'block';
                document.getElementById('detectFacesBtn').textContent = 'Re-detect Faces';
            } else {
                // No saved detections — auto-detect
                document.getElementById('faceDetStatus').textContent = 'Detecting faces...';
                // Fire and forget — will update UI when done
                _autoDetectFaces(idx);
            }
        } catch (e) {
            faceDetections = [];
            document.getElementById('faceDetStatus').textContent = 'Detecting faces...';
            _autoDetectFaces(idx);
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
            // Warn about legacy face spots without proper side assignment
            if (cameraMode === 'stereo') {
                const legacy = blurSpots.filter(s => s.spot_type === 'face' && (!s.side || s.side === 'full'));
                if (legacy.length > 0) {
                    document.getElementById('faceDetStatus').textContent =
                        'Re-run Detect Faces to assign blur spots to individual cameras.';
                }
            }
        } catch (e) {
            blurSpots = [];
        }

        // Load hand protection segments
        handProtectSegments = [];
        selectedHandSegId = null;
        try {
            const hs = await API.get(`/api/deidentify/${subjectId}/hand-settings?trial_idx=${idx}`);
            handMaskRadius = hs.mask_radius || 10;
            handSmooth = hs.hand_smooth || 10;
            forearmRadius = hs.forearm_radius || 10;
            forearmExtent = hs.forearm_extent != null ? hs.forearm_extent : 0.5;
            handSmooth2 = hs.hand_smooth2 || 0;
            dlcRadius = hs.dlc_radius || 15;
            document.getElementById('dlcRadiusSlider').value = dlcRadius;
            document.getElementById('dlcRadiusVal').textContent = dlcRadius;
            document.getElementById('handRadiusSlider').value = handMaskRadius;
            document.getElementById('handRadiusVal').textContent = handMaskRadius;
            document.getElementById('handSmoothSlider').value = handSmooth;
            document.getElementById('handSmoothVal').textContent = handSmooth;
            forearmRadius = 10; // hardcoded
            document.getElementById('handExtentSlider').value = forearmExtent;
            document.getElementById('handExtentVal').textContent = forearmExtent.toFixed(1);
            document.getElementById('handSmooth2Slider').value = handSmooth2;
            document.getElementById('handSmooth2Val').textContent = handSmooth2;
            if (hs.segments && hs.segments.length > 0) {
                handProtectSegments = hs.segments.map(s => ({
                    id: nextHandSegId++,
                    start: s.start,
                    end: s.end,
                    radius: s.radius || handMaskRadius,
                    smooth: s.smooth != null ? s.smooth : handSmooth,
                }));
            }
        } catch (e) {}

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

        _updateViewButtons();
        await loadFrame(currentFrame);
        renderSpotList();
        renderTimeline();
    }

    // ── Frame loading ──
    async function loadFrame(frameNum) {
        currentFrame = frameNum;
        if (typeof setNavState === 'function') setNavState({ frame: frameNum, side: currentSide });
        let url = `/api/deidentify/${subjectId}/frame?trial_idx=${currentTrialIdx}&frame_num=${frameNum}&side=${encodeURIComponent(currentSide)}`;
        if (viewMode === 'deidentified') url += '&blurred=true';
        else if (viewMode === 'preview') url += `&preview=true&canvas_w=${canvas ? canvas.width : 700}`;
        // Cache-bust to prevent browser from reusing frames across view modes
        url += `&_=${viewMode}_${Date.now()}`;

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

        // Load hand landmarks from bulk cache with temporal smoothing
        const inProtectSeg = handProtectSegments.some(s => frameNum >= s.start && frameNum <= s.end);
        const needHands = hasMediapipe && (handOverlayEnabled || inProtectSeg);
        if (needHands) {
            await _loadBulkLandmarks();
            _applyTemporalSmoothing();
        } else {
            handLandmarks = [];
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

    // ── View mode ──
    function setViewMode(mode) {
        if (mode === 'deidentified' && (!trialMeta || !trialMeta.has_blurred)) return;
        viewMode = mode;
        _updateViewButtons();
        _updateSidebarState();
        loadFrame(currentFrame);
    }

    function _updateViewButtons() {
        const btns = {
            original: document.getElementById('viewOriginal'),
            preview: document.getElementById('viewPreview'),
            deidentified: document.getElementById('viewDeidentified'),
        };
        for (const [mode, btn] of Object.entries(btns)) {
            if (!btn) continue;
            if (mode === viewMode) {
                btn.style.background = 'var(--blue)';
                btn.style.color = '#fff';
            } else {
                btn.style.background = '';
                btn.style.color = '';
            }
        }
        // Disable deidentified button if no rendered video
        const deidBtn = btns.deidentified;
        if (deidBtn) {
            deidBtn.disabled = !trialMeta || !trialMeta.has_blurred;
            deidBtn.title = (!trialMeta || !trialMeta.has_blurred)
                ? 'Render first to enable' : 'Show deidentified video';
        }
    }

    function _updateSidebarState() {
        const sidebar = document.querySelector('.deid-sidebar');
        if (!sidebar) return;
        if (viewMode === 'deidentified') {
            sidebar.style.opacity = '0.4';
            sidebar.style.pointerEvents = 'none';
        } else {
            sidebar.style.opacity = '';
            sidebar.style.pointerEvents = '';
        }
    }

    function toggleSide() {
        if (cameraMode === 'single') return;
        const idx = cameraNames.indexOf(currentSide);
        const newIdx = (idx + 1) % cameraNames.length;
        currentSide = cameraNames[newIdx];

        const camLabel = document.getElementById('cameraLabel');
        if (camLabel) camLabel.textContent = currentSide;

        renderSpotList();
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

    // ── Blur+threshold helper (morphological close approximation) ──
    function _morphClose(srcCanvas, blurPx) {
        if (blurPx <= 0) return srcCanvas;
        const w = srcCanvas.width, h = srcCanvas.height;
        const c2 = document.createElement('canvas');
        c2.width = w; c2.height = h;
        const ctx2 = c2.getContext('2d');
        ctx2.filter = `blur(${blurPx}px)`;
        ctx2.drawImage(srcCanvas, 0, 0);
        ctx2.filter = 'none';

        const c3 = document.createElement('canvas');
        c3.width = w; c3.height = h;
        const ctx3 = c3.getContext('2d');
        for (let i = 0; i < 8; i++) ctx3.drawImage(c2, 0, 0);

        const imgData = ctx3.getImageData(0, 0, w, h);
        const dd = imgData.data;
        for (let i = 3; i < dd.length; i += 4) {
            dd[i] = dd[i] > 30 ? 255 : 0;
        }
        ctx3.putImageData(imgData, 0, 0);
        return c3;
    }

    // ── Build smoothed hand protection mask ──
    function _buildHandMask(landmarks, radiusPx, forearmPx, smoothPx, smooth2Px, w, h) {
        // Replace MediaPipe fingertips with DLC labels when available
        // DLC thumb=joint 4, index=joint 8 — same as MediaPipe convention
        const dlcLms = landmarks.filter(l => l.type === 'dlc');
        let mergedLandmarks = landmarks.filter(l => l.type !== 'dlc');
        if (dlcLms.length > 0) {
            const dlcByJoint = {};
            for (const lm of dlcLms) dlcByJoint[lm.joint] = lm;
            // Replace MP joints 4 and 8 with DLC versions
            mergedLandmarks = mergedLandmarks.map(lm => {
                if (lm.type === 'hand' && dlcByJoint[lm.joint]) {
                    return { ...lm, x: dlcByJoint[lm.joint].x, y: dlcByJoint[lm.joint].y };
                }
                return lm;
            });
            // Add DLC joints that don't have a MediaPipe equivalent
            for (const [j, dlc] of Object.entries(dlcByJoint)) {
                if (!mergedLandmarks.some(l => l.type === 'hand' && l.joint === parseInt(j))) {
                    mergedLandmarks.push({ ...dlc, type: 'hand' });
                }
            }
        }

        // Step 1: Draw hand circles only (no pose)
        const c1 = document.createElement('canvas');
        c1.width = w; c1.height = h;
        const ctx1 = c1.getContext('2d');
        ctx1.fillStyle = '#fff';

        // Build interpolated hand points: add midpoints along each finger segment
        const fingerChains = [
            [0, 1, 2, 3, 4],     // thumb: wrist→CMC→MCP→IP→TIP
            [0, 5, 6, 7, 8],     // index: wrist→MCP→PIP→DIP→TIP
            [0, 9, 10, 11, 12],  // middle
            [0, 13, 14, 15, 16], // ring
            [0, 17, 18, 19, 20], // pinky
        ];
        const handLms = mergedLandmarks.filter(l => l.type !== 'pose');
        const byJoint = {};
        for (const lm of handLms) byJoint[lm.joint] = lm;

        const allPoints = [...handLms]; // start with original keypoints
        for (const chain of fingerChains) {
            for (let ci = 0; ci < chain.length - 1; ci++) {
                const a = byJoint[chain[ci]];
                const b = byJoint[chain[ci + 1]];
                if (a && b) {
                    allPoints.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, type: 'interp' });
                }
            }
        }

        // Draw circles for all hand keypoints + interpolated midpoints
        for (const lm of allPoints) {
            ctx1.beginPath();
            ctx1.arc(offsetX + lm.x * scale, offsetY + lm.y * scale, radiusPx, 0, Math.PI * 2);
            ctx1.fill();
        }

        // Step 2: First smooth — on hand circles only (fills finger gaps)
        let handSmoothed = _morphClose(c1, smoothPx);

        // Step 3: Add forearm triangle onto smoothed hand mask
        const pinkyMCP = landmarks.find(l => l.type === 'hand' && l.joint === 17);
        const thumbCMC = landmarks.find(l => l.type === 'hand' && l.joint === 1);
        const handWrist = landmarks.find(l => l.type === 'hand' && l.joint === 0);
        const elbows = landmarks.filter(l => l.type === 'pose' && (l.joint === 13 || l.joint === 14));
        let elbow = null;
        if (handWrist && elbows.length >= 2) {
            const d0 = Math.hypot(elbows[0].x - handWrist.x, elbows[0].y - handWrist.y);
            const d1 = Math.hypot(elbows[1].x - handWrist.x, elbows[1].y - handWrist.y);
            elbow = d0 < d1 ? elbows[0] : elbows[1];
        } else if (elbows.length === 1) {
            elbow = elbows[0];
        }

        if (pinkyMCP && thumbCMC && elbow && handWrist) {
            const t = forearmExtent;
            const interpElbow = {
                x: handWrist.x + t * (elbow.x - handWrist.x),
                y: handWrist.y + t * (elbow.y - handWrist.y),
            };
            const pts = [pinkyMCP, interpElbow, thumbCMC].map(p => ({
                sx: offsetX + p.x * scale,
                sy: offsetY + p.y * scale,
            }));

            const sCtx = handSmoothed.getContext('2d');
            sCtx.fillStyle = '#fff';
            sCtx.strokeStyle = '#fff';
            sCtx.lineCap = 'round';

            // Filled triangle
            sCtx.beginPath();
            sCtx.moveTo(pts[0].sx, pts[0].sy);
            sCtx.lineTo(pts[1].sx, pts[1].sy);
            sCtx.lineTo(pts[2].sx, pts[2].sy);
            sCtx.closePath();
            sCtx.fill();

            // Palm side (thumbCMC → elbow): dilate by circle radius
            if (radiusPx > 0) {
                sCtx.lineWidth = radiusPx * 2;
                sCtx.beginPath();
                sCtx.moveTo(pts[2].sx, pts[2].sy);
                sCtx.lineTo(pts[1].sx, pts[1].sy);
                sCtx.stroke();
            }

            // Dorsal side (pinkyMCP → elbow): dilate by forearm slider
            if (forearmPx > 0) {
                sCtx.lineWidth = forearmPx * 2;
                sCtx.beginPath();
                sCtx.moveTo(pts[0].sx, pts[0].sy);
                sCtx.lineTo(pts[1].sx, pts[1].sy);
                sCtx.stroke();
            }
        }

        // Step 4: Second smooth — on combined hand+forearm (smooths the join)
        let finalMask = _morphClose(handSmoothed, smooth2Px);

        return finalMask;
    }

    // ── Preview blur mask (shows exactly what will be blurred) ──
    function _renderPreviewMask(cw, ch) {
        const curSideLabel = _sideLabel();

        // Build blur mask on offscreen canvas
        const blurCanvas = document.createElement('canvas');
        blurCanvas.width = cw;
        blurCanvas.height = ch;
        const bCtx = blurCanvas.getContext('2d');

        // Draw blur ellipses
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const spotSide = spot.side || 'full';
            if (spotSide !== 'full' && spotSide !== curSideLabel) continue;

            const pos = _getSpotPosition(spot);
            const sx = offsetX + pos.x * scale;
            const sy = offsetY + pos.y * scale;
            const sw = (spot.width || spot.radius || 50) * scale / 2;
            const sh = (spot.height || spot.radius || 50) * scale / 2;

            bCtx.fillStyle = spot.spot_type === 'face'
                ? 'rgba(33,150,243,0.45)' : 'rgba(244,67,54,0.45)';
            bCtx.beginPath();
            bCtx.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
            bCtx.fill();
        }

        // Build and subtract hand protection mask
        const landmarks = _getVisibleLandmarks();
        if (landmarks.length > 0) {
            const activeSegment = _getActiveHandSegment();
            if (activeSegment) {
                const radiusPx = (activeSegment.radius || handMaskRadius) * scale;
                const faPx = forearmRadius * scale;
                const smPx = (activeSegment.smooth != null ? activeSegment.smooth : handSmooth) * scale;
                const sm2Px = handSmooth2 * scale;
                const handMask = _buildHandMask(landmarks, radiusPx, faPx, smPx, sm2Px, cw, ch);

                // Subtract hand mask from blur using destination-out
                bCtx.globalCompositeOperation = 'destination-out';
                bCtx.drawImage(handMask, 0, 0);
                bCtx.globalCompositeOperation = 'source-over';
            }
        }

        // Draw result onto main canvas
        ctx.drawImage(blurCanvas, 0, 0);

        // Draw outline of blur areas
        ctx.setLineDash([6, 4]);
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const spotSide = spot.side || 'full';
            if (spotSide !== 'full' && spotSide !== curSideLabel) continue;

            const pos = _getSpotPosition(spot);
            const sx = offsetX + pos.x * scale;
            const sy = offsetY + pos.y * scale;
            const sw = (spot.width || spot.radius || 50) * scale / 2;
            const sh = (spot.height || spot.radius || 50) * scale / 2;

            ctx.strokeStyle = spot.spot_type === 'face'
                ? 'rgba(33,150,243,0.8)' : 'rgba(244,67,54,0.8)';
            ctx.lineWidth = 2;
            if (spot.shape === 'rect') {
                ctx.strokeRect(sx - sw, sy - sh, sw * 2, sh * 2);
            } else {
                ctx.beginPath();
                ctx.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
        ctx.setLineDash([]);

        // No green outline in preview — just the blur mask with cutouts
    }

    function _getVisibleLandmarks() {
        const curSideLabel = _sideLabel();
        return handLandmarks.filter(lm =>
            (lm.side || 'full') === curSideLabel || (lm.side || 'full') === 'full'
        );
    }

    function _getActiveHandSegment() {
        for (const seg of handProtectSegments) {
            if (seg.start <= currentFrame && currentFrame <= seg.end) return seg;
        }
        return null;
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

        // In deidentified or preview mode, the server provides the blurred frame — no overlays
        if (viewMode === 'deidentified' || viewMode === 'preview') return;

        _drawOverlays();
    }

    function _drawOverlays() {
        const cw = canvas.width;
        const ch = canvas.height;

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

        // Find active hand protection segment for current frame
        const curSideLabel = _sideLabel();
        const visibleLandmarks = handLandmarks.filter(lm =>
            (lm.side || 'full') === curSideLabel || (lm.side || 'full') === 'full'
        );
        const activeSeg = handProtectSegments.find(s =>
            currentFrame >= s.start && currentFrame <= s.end
        );
        const handProtectActive = activeSeg && visibleLandmarks.length > 0;
        const activeProtectRadius = activeSeg ? (activeSeg.radius || handMaskRadius) : handMaskRadius;
        const activeSmooth = activeSeg ? (activeSeg.smooth != null ? activeSeg.smooth : handSmooth) : handSmooth;

        // Build smoothed hand mask (morphological close via blur+threshold)
        let handMaskCanvas = null;
        if (handProtectActive) {
            handMaskCanvas = _buildHandMask(
                visibleLandmarks, activeProtectRadius * scale, forearmRadius * scale, activeSmooth * scale, handSmooth2 * scale, cw, ch
            );
        }

        // Draw ALL blur spots on one offscreen canvas, then subtract hand protection
        const hasVisibleSpots = blurSpots.some(s => {
            if (currentFrame < s.frame_start || currentFrame > s.frame_end) return false;
            const ss = s.side || 'full';
            return ss === 'full' || ss === curSideLabel;
        });

        if (hasVisibleSpots) {
            // Offscreen canvas for compositing blur fills with hand subtraction
            const offCanvas = document.createElement('canvas');
            offCanvas.width = cw;
            offCanvas.height = ch;
            const off = offCanvas.getContext('2d');

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

                off.fillStyle = `rgba(${cr},${cg},${cb},${fillAlpha})`;
                if (spot.shape === 'rect') {
                    off.fillRect(sx - sw, sy - sh, sw * 2, sh * 2);
                } else {
                    off.beginPath();
                    off.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
                    off.fill();
                }
            }

            // Subtract hand protection mask from ALL blur fills
            if (handMaskCanvas) {
                off.globalCompositeOperation = 'destination-out';
                off.drawImage(handMaskCanvas, 0, 0);
                off.globalCompositeOperation = 'source-over';
            }

            // Composite result onto main canvas
            ctx.drawImage(offCanvas, 0, 0);

            // Draw outlines on main canvas (not affected by subtraction)
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

                ctx.strokeStyle = isSelected
                    ? `rgba(${cr},${cg},${cb},0.9)`
                    : `rgba(${cr},${cg},${cb},0.5)`;
                ctx.lineWidth = isSelected ? 2 : 1;
                if (spot.shape === 'rect') {
                    ctx.strokeRect(sx - sw, sy - sh, sw * 2, sh * 2);
                } else {
                    ctx.beginPath();
                    ctx.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
                    ctx.stroke();
                }

                // Draw drag handles for selected spot
                if (isSelected) {
                    const handleSize = 5;
                    ctx.fillStyle = '#fff';
                    ctx.strokeStyle = `rgba(${cr},${cg},${cb},0.9)`;
                    ctx.lineWidth = 1.5;
                    // N, S, E, W handles
                    const handles = [
                        { x: sx, y: sy - sh },        // N
                        { x: sx, y: sy + sh },         // S
                        { x: sx + sw, y: sy },         // E
                        { x: sx - sw, y: sy },         // W
                    ];
                    for (const h of handles) {
                        ctx.fillRect(h.x - handleSize, h.y - handleSize, handleSize * 2, handleSize * 2);
                        ctx.strokeRect(h.x - handleSize, h.y - handleSize, handleSize * 2, handleSize * 2);
                    }
                }
            }
        }

        // Draw landmarks — green for hand, orange for pose
        if (handOverlayEnabled && visibleLandmarks.length > 0) {
            for (const lm of visibleLandmarks) {
                const sx = offsetX + lm.x * scale;
                const sy = offsetY + lm.y * scale;
                if (lm.type === 'pose') {
                    // Pose: orange diamond shape to distinguish from hand circles
                    ctx.fillStyle = 'rgba(255,152,0,0.8)';
                    ctx.beginPath();
                    ctx.moveTo(sx, sy - 5);
                    ctx.lineTo(sx + 5, sy);
                    ctx.lineTo(sx, sy + 5);
                    ctx.lineTo(sx - 5, sy);
                    ctx.closePath();
                    ctx.fill();
                } else {
                    // Hand: green circle
                    ctx.fillStyle = 'rgba(76,175,80,0.7)';
                    ctx.beginPath();
                    ctx.arc(sx, sy, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }

        // Draw hand protection outline from the smoothed mask
        if (handMaskCanvas) {
            // Extract outline: erode the mask inward by 2px
            const hOff = document.createElement('canvas');
            hOff.width = cw;
            hOff.height = ch;
            const hCtx = hOff.getContext('2d');

            // Draw the smoothed mask
            hCtx.drawImage(handMaskCanvas, 0, 0);

            // Erode inward by 2px: shrink the mask slightly and erase it
            const innerCanvas = document.createElement('canvas');
            innerCanvas.width = cw;
            innerCanvas.height = ch;
            const ic = innerCanvas.getContext('2d');
            // Shrink by drawing at reduced radius
            const shrinkR = activeProtectRadius * scale - 2;
            if (shrinkR > 0) {
                if (activeSmooth > 0) {
                    // Rebuild mask at slightly smaller radius (shrink forearm too)
                    const shrinkForearm = Math.max(0, forearmRadius * scale - 2);
                    ic.drawImage(_buildHandMask(
                        visibleLandmarks, shrinkR, shrinkForearm, activeSmooth * scale, Math.max(0, handSmooth2 * scale - 2), cw, ch
                    ), 0, 0);
                } else {
                    ic.fillStyle = '#fff';
                    for (const lm of visibleLandmarks) {
                        ic.beginPath();
                        ic.arc(offsetX + lm.x * scale, offsetY + lm.y * scale, shrinkR, 0, Math.PI * 2);
                        ic.fill();
                    }
                }
                hCtx.globalCompositeOperation = 'destination-out';
                hCtx.drawImage(innerCanvas, 0, 0);
                hCtx.globalCompositeOperation = 'source-over';
            }

            // Tint green
            hCtx.globalCompositeOperation = 'source-in';
            hCtx.fillStyle = 'rgba(76,175,80,0.8)';
            hCtx.fillRect(0, 0, cw, ch);

            ctx.drawImage(hOff, 0, 0);
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
        _pushUndo(); // snapshot before any spot creation/selection

        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        // Convert to image coordinates
        const ix = (cx - offsetX) / scale;
        const iy = (cy - offsetY) / scale;

        if (ix < 0 || iy < 0 || ix > imgW || iy > imgH) return;

        // Check if clicking on a face detection → create face spot (if none exists for this face)
        const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
        if (faceDetections.length > localFrame) {
            const entry = faceDetections[localFrame];
            if (entry && entry.faces) {
                for (const f of entry.faces) {
                    if (ix >= f.x1 && ix <= f.x2 && iy >= f.y1 && iy <= f.y2) {
                        const fcx = (f.x1 + f.x2) / 2;
                        const fcy = (f.y1 + f.y2) / 2;
                        const fSide = f.side || 'full';

                        // Check if a blur spot already covers this face
                        const existing = blurSpots.find(s => {
                            if (s.spot_type !== 'face') return false;
                            if (s.side !== fSide) return false;
                            const dx = s.x - fcx;
                            const dy = s.y - fcy;
                            return Math.sqrt(dx * dx + dy * dy) < Math.max(f.x2 - f.x1, f.y2 - f.y1);
                        });
                        if (existing) {
                            // Select the existing spot instead
                            selectedSpotId = existing.id;
                            renderSpotList();
                            render();
                            return;
                        }

                        const fw = f.x2 - f.x1;
                        const fh = f.y2 - f.y1;
                        const spot = {
                            id: nextSpotId++,
                            spot_type: 'face',
                            x: Math.round(fcx),
                            y: Math.round(fcy),
                            radius: Math.round(Math.max(fw, fh) / 2 * 1.2),
                            width: Math.round(fw * 1.2),
                            height: Math.round(fh * 1.2),
                            offset_x: 0,
                            offset_y: 0,
                            frame_start: trialMeta.start_frame,
                            frame_end: trialMeta.end_frame,
                            side: fSide,
                        };
                        blurSpots.push(spot);
                        selectedSpotId = spot.id;
                        renderSpotList();
                        updateSpotControls();
                        scheduleSave();
                        render();
                        renderTimeline();
                        return;
                    }
                }
            }
        }

        // Check if clicking on an existing blur spot → select it
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const pos = _getSpotPosition(spot);
            const dx = ix - pos.x;
            const dy = ix - pos.y;
            // Use the larger dimension for hit testing
            const hitR = Math.max(spot.width || spot.radius, spot.height || spot.radius) / 2;
            if (Math.sqrt(dx * dx + (iy - pos.y) * (iy - pos.y)) <= hitR) {
                selectedSpotId = spot.id;
                renderSpotList();
                updateSpotControls();
                render();
                renderTimeline();
                return;
            }
        }

        // In original view, clicking empty canvas creates a custom spot
        if (viewMode === 'original') {
            const spot = {
                id: nextSpotId++,
                spot_type: 'custom',
                shape: 'oval',
                x: Math.round(ix),
                y: Math.round(iy),
                radius: 40,
                width: 80,
                height: 80,
                offset_x: 0,
                offset_y: 0,
                frame_start: Math.max(trialMeta.start_frame, currentFrame - 30),
                frame_end: Math.min(trialMeta.end_frame, currentFrame + 30),
                side: _sideLabel(),
            };
            blurSpots.push(spot);
            selectedSpotId = spot.id;
            addingCustom = false;
            const customBtn = document.getElementById('addCustomBtn');
            if (customBtn) { customBtn.style.background = ''; customBtn.style.color = ''; }
            renderSpotList();
            updateSpotControls();
            scheduleSave();
            render();
            renderTimeline();
            return;
        }

        // Click on nothing → deselect
        selectedSpotId = null;
        renderSpotList();
        updateSpotControls();
        render();
        renderTimeline();
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
    // ── Hit-test blur spot edges/center for drag handles ──
    function _spotHitTest(ix, iy) {
        const curSideLabel = _sideLabel();
        const threshold = 8 / scale; // 8 screen pixels
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const sSide = spot.side || 'full';
            if (sSide !== 'full' && sSide !== curSideLabel) continue;

            const pos = _getSpotPosition(spot);
            const w2 = (spot.width || spot.radius * 2) / 2;
            const h2 = (spot.height || spot.radius * 2) / 2;
            const cx = pos.x, cy = pos.y;

            // Check edge handles (N, S, E, W)
            if (Math.abs(ix - cx) < threshold && Math.abs(iy - (cy - h2)) < threshold) return { spotId: spot.id, handle: 'n' };
            if (Math.abs(ix - cx) < threshold && Math.abs(iy - (cy + h2)) < threshold) return { spotId: spot.id, handle: 's' };
            if (Math.abs(ix - (cx + w2)) < threshold && Math.abs(iy - cy) < threshold) return { spotId: spot.id, handle: 'e' };
            if (Math.abs(ix - (cx - w2)) < threshold && Math.abs(iy - cy) < threshold) return { spotId: spot.id, handle: 'w' };

            // Check interior for move
            if (Math.abs(ix - cx) <= w2 && Math.abs(iy - cy) <= h2) return { spotId: spot.id, handle: 'move' };
        }
        return null;
    }

    function onMouseDown(e) {
        if (e.button === 0 && viewMode === 'original') {
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const ix = (mx - offsetX) / scale;
            const iy = (my - offsetY) / scale;

            // Check for spot drag handle
            const hit = _spotHitTest(ix, iy);
            if (hit) {
                e.preventDefault();
                const spot = blurSpots.find(s => s.id === hit.spotId);
                if (spot) {
                    _pushUndo(); // snapshot before drag
                    spotDrag = {
                        spotId: hit.spotId,
                        handle: hit.handle,
                        startMx: mx, startMy: my,
                        origW: spot.width || spot.radius * 2,
                        origH: spot.height || spot.radius * 2,
                        origOffX: spot.offset_x || 0,
                        origOffY: spot.offset_y || 0,
                    };
                    selectedSpotId = hit.spotId;
                    renderSpotList();
                    render();
                    return;
                }
            }
        }

        // Right-click or middle-click for pan
        if (e.button === 1 || e.button === 2) {
            e.preventDefault();
            panning = true;
            panStart = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
            canvas.style.cursor = 'grabbing';
            return;
        }
        // Left-click pan when zoomed in (hold and drag)
        if (e.button === 0 && hasUserZoom && !addingCustom && !spotDrag) {
            panning = true;
            panStart = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
            canvas.style.cursor = 'grabbing';
        }
    }

    function onMouseMove(e) {
        // Spot dragging
        if (spotDrag) {
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const dxImg = (mx - spotDrag.startMx) / scale;
            const dyImg = (my - spotDrag.startMy) / scale;
            const spot = blurSpots.find(s => s.id === spotDrag.spotId);
            if (!spot) return;

            if (spotDrag.handle === 'move') {
                spot.offset_x = Math.round(spotDrag.origOffX + dxImg);
                spot.offset_y = Math.round(spotDrag.origOffY + dyImg);
            } else if (spotDrag.handle === 'e') {
                spot.width = Math.max(10, Math.round(spotDrag.origW + dxImg * 2));
            } else if (spotDrag.handle === 'w') {
                spot.width = Math.max(10, Math.round(spotDrag.origW - dxImg * 2));
            } else if (spotDrag.handle === 's') {
                spot.height = Math.max(10, Math.round(spotDrag.origH + dyImg * 2));
            } else if (spotDrag.handle === 'n') {
                spot.height = Math.max(10, Math.round(spotDrag.origH - dyImg * 2));
            }
            render();
            return;
        }

        // Pan
        if (!panning || !panStart) {
            // Update cursor for spot hover
            if (viewMode === 'original') {
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const ix = (mx - offsetX) / scale;
                const iy = (my - offsetY) / scale;
                const hit = _spotHitTest(ix, iy);
                if (hit) {
                    if (hit.handle === 'move') canvas.style.cursor = 'grab';
                    else if (hit.handle === 'n' || hit.handle === 's') canvas.style.cursor = 'ns-resize';
                    else canvas.style.cursor = 'ew-resize';
                } else {
                    canvas.style.cursor = '';
                }
            }
            return;
        }
        offsetX = panStart.ox + (e.clientX - panStart.x);
        offsetY = panStart.oy + (e.clientY - panStart.y);
        render();
    }

    function onMouseUp(e) {
        if (spotDrag) {
            spotDrag = null;
            scheduleSave();
            renderSpotList();
            return;
        }
        if (panning) {
            canvas.style.cursor = '';
            if (panStart) {
                const dx = Math.abs(e.clientX - panStart.x);
                const dy = Math.abs(e.clientY - panStart.y);
                if (dx < 3 && dy < 3) {
                    panning = false;
                    panStart = null;
                    return;
                }
            }
            setTimeout(() => { panning = false; }, 50);
            panStart = null;
        }
    }

    // ── Keyboard ──
    function onKeyDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        if ((e.key === 'z' || e.key === 'Z') && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            if (e.shiftKey) redo();
            else undo();
            return;
        }

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
        } else if (e.key === 'p') {
            // Cycle view modes: original → preview → deidentified → original
            if (viewMode === 'original') setViewMode('preview');
            else if (viewMode === 'preview' && trialMeta && trialMeta.has_blurred) setViewMode('deidentified');
            else setViewMode('original');
        } else if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedHandSegId !== null) {
                deleteSelectedHandSeg();
            } else if (selectedSpotId !== null) {
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
    function setPlaybackSpeed(val) {
        playbackSpeed = val;
        document.getElementById('speedVal').textContent = val + 'x';
        // Restart playback with new speed if playing
        if (playing) {
            togglePlay();
            togglePlay();
        }
    }

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
            videoPlaying = false;
            if (playTimer) { clearInterval(playTimer); playTimer = null; }
            if (videoEl) videoEl.pause();
            btn.innerHTML = '&#9654;';
            // Reload current frame as JPEG for precise overlay rendering
            loadFrame(currentFrame);
        } else {
            if (!trialMeta) return;
            playing = true;
            btn.innerHTML = '&#9616;&#9616;';
            _startVideoPlayback();
        }
    }

    async function _startVideoPlayback() {
        if (!trialMeta || !videoEl) {
            _fallbackPlay();
            return;
        }

        // Load the correct video into the element
        const isBlurred = viewMode === 'deidentified';
        const needReload = (currentVideoTrialIdx !== currentTrialIdx || currentVideoBlurred !== isBlurred);

        if (needReload) {
            let videoUrl = `/api/deidentify/${subjectId}/video?trial_idx=${currentTrialIdx}`;
            if (isBlurred) videoUrl += '&blurred=true';
            videoUrl += `&_=${Date.now()}`;
            videoEl.src = videoUrl;
            currentVideoTrialIdx = currentTrialIdx;
            currentVideoBlurred = isBlurred;

            // Wait for metadata
            const loaded = await new Promise(resolve => {
                if (videoEl.readyState >= 1) { resolve(true); return; }
                const timer = setTimeout(() => resolve(false), 5000);
                videoEl.addEventListener('loadedmetadata', () => { clearTimeout(timer); resolve(true); }, { once: true });
                videoEl.addEventListener('error', () => { clearTimeout(timer); resolve(false); }, { once: true });
            });
            if (!loaded || !playing) {
                _fallbackPlay();
                return;
            }
        }

        // Seek to current frame position
        const localFrame = currentFrame - trialMeta.start_frame;
        const startTime = localFrame / fps;
        videoEl.currentTime = startTime;
        videoPlaying = true;

        try {
            await videoEl.play();
            // Use requestVideoFrameCallback for precise frame sync
            if ('requestVideoFrameCallback' in videoEl) {
                videoEl.requestVideoFrameCallback(_videoDrawLoop);
            } else {
                requestAnimationFrame(_videoDrawLoop);
            }
        } catch (e) {
            console.error('[deid] video play() rejected:', e);
            videoPlaying = false;
            _fallbackPlay();
        }

        videoEl.onended = () => {
            playing = false;
            videoPlaying = false;
            document.getElementById('playBtn').innerHTML = '&#9654;';
            loadFrame(currentFrame);
        };
    }

    function _videoDrawLoop(now, metadata) {
        if (!videoPlaying || !playing) return;
        if (!trialMeta) return;

        // Get the actual media time for frame sync
        const mediaTime = (metadata && metadata.mediaTime != null)
            ? metadata.mediaTime : videoEl.currentTime;
        const localFrame = Math.floor(mediaTime * fps);
        currentFrame = trialMeta.start_frame + Math.min(localFrame, totalFrames - 1);

        if (currentFrame >= trialMeta.end_frame) {
            playing = false;
            videoPlaying = false;
            videoEl.pause();
            document.getElementById('playBtn').innerHTML = '&#9654;';
            loadFrame(currentFrame);
            return;
        }

        // Draw video to canvas (with stereo cropping)
        const cw = canvas.clientWidth;
        const ch = canvas.clientHeight;
        canvas.width = cw;
        canvas.height = ch;

        const vw = videoEl.videoWidth;
        const vh = videoEl.videoHeight;
        const isStereo = cameraMode === 'stereo' && vw > 0 && vh > 0 && (vw / vh) > 1.7;

        let sx = 0, sw = vw;
        if (isStereo) {
            const midline = Math.floor(vw / 2);
            if (cameraNames.length >= 2 && currentSide === cameraNames[1]) {
                sx = midline; sw = vw - midline;
            } else {
                sx = 0; sw = midline;
            }
        }

        imgW = sw;
        imgH = vh;
        if (!hasUserZoom) {
            scale = Math.min(cw / imgW, ch / imgH);
            offsetX = (cw - imgW * scale) / 2;
            offsetY = (ch - imgH * scale) / 2;
        }

        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(videoEl, sx, 0, sw, vh, offsetX, offsetY, imgW * scale, imgH * scale);

        // Draw overlays (except in deidentified mode)
        if (viewMode === 'original') {
            _drawOverlays();
        }

        // Update frame display
        document.getElementById('frameDisplay').textContent =
            `Frame: ${localFrame} / ${totalFrames - 1}`;
        renderTimeline();

        // Schedule next frame
        if ('requestVideoFrameCallback' in videoEl) {
            videoEl.requestVideoFrameCallback(_videoDrawLoop);
        } else {
            requestAnimationFrame(_videoDrawLoop);
        }
    }

    function _fallbackPlay() {
        // Fallback: per-frame JPEG loading (slow but works for preview mode)
        if (playTimer) clearInterval(playTimer);
        let loading = false;
        playTimer = setInterval(() => {
            if (!playing) return;
            if (loading) return;
            if (currentFrame >= trialMeta.end_frame) {
                togglePlay();
                return;
            }
            loading = true;
            currentFrame++;
            let url = `/api/deidentify/${subjectId}/frame?trial_idx=${currentTrialIdx}&frame_num=${currentFrame}&side=${encodeURIComponent(currentSide)}`;
            if (viewMode === 'deidentified') url += '&blurred=true';
            else if (viewMode === 'preview') url += '&preview=true';
            url += `&_=${viewMode}_${Date.now()}`;
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
        document.getElementById('detectFacesBtn').style.display = 'block';
    }

    // ── Auto-detect faces on trial load ──
    async function _autoDetectFaces(trialIdx) {
        try {
            const result = await API.post(`/api/deidentify/${subjectId}/detect-faces`, {
                trial_idx: trialIdx,
            });
            faceDetections = result.faces || [];
            const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
            document.getElementById('faceDetStatus').textContent =
                `${nFaces} frames with faces`;
            document.getElementById('detectFacesBtn').style.display = 'block';
            document.getElementById('detectFacesBtn').textContent = 'Re-detect Faces';

            _autoCreateFaceSpots();
            renderSpotList();
            scheduleSave();
            render();
            renderTimeline();
        } catch (e) {
            document.getElementById('faceDetStatus').textContent = 'Detection failed: ' + e.message;
            document.getElementById('detectFacesBtn').style.display = 'block';
            document.getElementById('detectFacesBtn').textContent = 'Detect Faces';
        }
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
        _pushUndo();
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

        const curSideLabel = _sideLabel();
        const visible = blurSpots.filter(s => {
            const sSide = s.side || 'full';
            return sSide === 'full' || sSide === curSideLabel;
        });

        if (visible.length === 0) {
            list.innerHTML = '<div style="font-size:11px;color:var(--text-muted);">No blur spots for this camera. Use + Custom Spot.</div>';
            return;
        }

        list.innerHTML = visible.map(s => `
            <div class="spot-item ${s.id === selectedSpotId ? 'selected' : ''}"
                 onclick="deid.selectSpot(${s.id})">
                <span class="spot-label">${s.spot_type === 'face' ? '\u{1F464}' : '\u2295'}
                    ${s.width || s.radius}×${s.height || s.radius}</span>
                <span class="spot-delete" onclick="event.stopPropagation(); deid.deleteSpot(${s.id})">×</span>
            </div>
        `).join('');
    }

    function selectSpot(id) {
        selectedSpotId = id;
        renderSpotList();
        _updateShapeToggle();
        updateSpotControls();
        render();
    }

    function _updateShapeToggle() {
        const toggle = document.getElementById('spotShapeToggle');
        const btn = document.getElementById('shapeBtn');
        if (!toggle || !btn) return;
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (spot && spot.spot_type === 'custom') {
            toggle.style.display = 'block';
            const shape = spot.shape || 'oval';
            btn.textContent = `Shape: ${shape === 'rect' ? 'Rectangle' : 'Oval'}`;
        } else {
            toggle.style.display = 'none';
        }
    }

    function toggleSpotShape() {
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (!spot || spot.spot_type !== 'custom') return;
        spot.shape = (spot.shape || 'oval') === 'oval' ? 'rect' : 'oval';
        _updateShapeToggle();
        scheduleSave();
        render();
    }

    function deleteSpot(id) {
        _pushUndo();
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
        // Spot controls are now on-canvas (drag handles) and timeline (frame range)
        // This function just triggers a re-render to show/hide handles
        render();
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

    // ── Copy custom spots from other camera ──
    function copyFromOtherCamera() {
        if (cameraMode === 'single' || cameraNames.length < 2) return;
        _pushUndo();

        const curSide = _sideLabel();
        const otherSide = curSide === 'left' ? 'right' : 'left';

        const otherCustom = blurSpots.filter(s =>
            s.spot_type === 'custom' && s.side === otherSide
        );

        if (otherCustom.length === 0) {
            document.getElementById('renderStatus').textContent = 'No custom spots on the other camera.';
            return;
        }

        let copied = 0;
        for (const s of otherCustom) {
            // Check if a similar spot already exists on this side
            const exists = blurSpots.find(b =>
                b.spot_type === 'custom' && b.side === curSide &&
                Math.abs(b.x - s.x) < 20 && Math.abs(b.y - s.y) < 20
            );
            if (exists) continue;

            blurSpots.push({
                id: nextSpotId++,
                spot_type: 'custom',
                shape: s.shape || 'oval',
                x: s.x,
                y: s.y,
                radius: s.radius,
                width: s.width || s.radius * 2,
                height: s.height || s.radius * 2,
                offset_x: s.offset_x || 0,
                offset_y: s.offset_y || 0,
                frame_start: s.frame_start,
                frame_end: s.frame_end,
                side: curSide,
            });
            copied++;
        }

        if (copied > 0) {
            renderSpotList();
            scheduleSave();
            render();
            renderTimeline();
            document.getElementById('renderStatus').textContent = `Copied ${copied} custom spot(s) from ${otherSide} camera.`;
        } else {
            document.getElementById('renderStatus').textContent = 'All spots already exist on this camera.';
        }
    }

    // ── Hand overlay ──
    async function toggleHandOverlay(enabled) {
        handOverlayEnabled = enabled;
        if (enabled && subjectId && hasMediapipe) {
            await _loadBulkLandmarks();
            _applyTemporalSmoothing();
        } else {
            handLandmarks = [];
        }
        render();
    }

    async function _loadBulkLandmarks() {
        if (Object.keys(handLandmarksBulk).length > 0) return; // already loaded
        try {
            const res = await API.get(
                `/api/deidentify/${subjectId}/hand-landmarks-bulk?trial_idx=${currentTrialIdx}`
            );
            handLandmarksBulk = res.landmarks || {};
            hasDlcLabels = res.has_dlc || false;
            // Show/hide DLC section
            const dlcSec = document.getElementById('dlcSection');
            if (dlcSec) dlcSec.style.display = hasDlcLabels ? 'block' : 'none';
        } catch (e) {
            handLandmarksBulk = {};
            hasDlcLabels = false;
        }
    }

    function _applyTemporalSmoothing() {
        // Average landmark positions across a window of frames
        const win = handTemporalSmooth;
        const cf = currentFrame;
        const sideLabel = _sideLabel();

        if (Object.keys(handLandmarksBulk).length === 0) {
            handLandmarks = [];
            return;
        }

        if (win === 0) {
            // No smoothing — use raw data for current frame (copy to avoid mutation)
            const raw = handLandmarksBulk[String(cf)];
            handLandmarks = raw ? raw.map(lm => ({ ...lm })) : [];
            return;
        }

        // Collect landmarks from nearby frames, grouped by side
        const frameKeys = [];
        for (let f = cf - win; f <= cf + win; f++) {
            const key = String(f);
            if (handLandmarksBulk[key]) frameKeys.push(key);
        }

        if (frameKeys.length === 0) {
            handLandmarks = [];
            return;
        }

        // Group by side, then average positions per keypoint index
        const sides = {};
        for (const key of frameKeys) {
            for (const lm of handLandmarksBulk[key]) {
                const s = lm.side || 'full';
                if (!sides[s]) sides[s] = [];
                sides[s].push(lm);
            }
        }

        // For each side, average the keypoints across frames
        // Landmarks come in groups of 21 (per hand), so we average by index
        const result = [];
        for (const [side, allPts] of Object.entries(sides)) {
            // Count landmarks per frame to know how many hands/keypoints
            const ptsPerFrame = {};
            for (const key of frameKeys) {
                const framePts = (handLandmarksBulk[key] || []).filter(
                    lm => (lm.side || 'full') === side
                );
                ptsPerFrame[key] = framePts;
            }

            // Find the frame with the most points as reference
            let refKey = frameKeys[0];
            let maxPts = 0;
            for (const key of frameKeys) {
                const n = (ptsPerFrame[key] || []).length;
                if (n > maxPts) { maxPts = n; refKey = key; }
            }
            if (maxPts === 0) continue;

            // Average each keypoint position
            const refPts = ptsPerFrame[refKey];
            for (let i = 0; i < refPts.length; i++) {
                let sumX = 0, sumY = 0, count = 0;
                for (const key of frameKeys) {
                    const pts = ptsPerFrame[key] || [];
                    if (i < pts.length) {
                        sumX += pts[i].x;
                        sumY += pts[i].y;
                        count++;
                    }
                }
                if (count > 0) {
                    result.push({
                        x: Math.round((sumX / count) * 10) / 10,
                        y: Math.round((sumY / count) * 10) / 10,
                        side: side,
                        type: refPts[i].type || 'hand',
                        joint: refPts[i].joint,
                    });
                }
            }
        }

        handLandmarks = result;
    }

    function updateHandTemporalSmooth(val) {
        handTemporalSmooth = parseInt(val);
        document.getElementById('handTemporalVal').textContent = handTemporalSmooth;
        _applyTemporalSmoothing();
        render();
    }

    function updateHandRadius(val) {
        handMaskRadius = parseInt(val);
        document.getElementById('handRadiusVal').textContent = handMaskRadius;
        // Update ALL segments if none selected, or just the selected one
        if (selectedHandSegId) {
            const seg = handProtectSegments.find(s => s.id === selectedHandSegId);
            if (seg) seg.radius = handMaskRadius;
        } else {
            // Apply to all segments
            for (const seg of handProtectSegments) {
                seg.radius = handMaskRadius;
            }
        }
        saveHandSettings();
        render();
        renderTimeline();
    }

    function updateDlcRadius(val) {
        dlcRadius = parseInt(val);
        document.getElementById('dlcRadiusVal').textContent = dlcRadius;
        saveHandSettings();
        render();
    }

    function updateForearmRadius(val) {
        forearmRadius = parseInt(val);
        document.getElementById('handForearmVal').textContent = forearmRadius;
        saveHandSettings();
        render();
    }

    function updateForearmExtent(val) {
        forearmExtent = parseFloat(val);
        document.getElementById('handExtentVal').textContent = forearmExtent.toFixed(1);
        saveHandSettings();
        render();
    }

    function updateHandSmooth(val) {
        handSmooth = parseInt(val);
        document.getElementById('handSmoothVal').textContent = handSmooth;
        // Update selected segment's smooth, or all if none selected
        if (selectedHandSegId) {
            const seg = handProtectSegments.find(s => s.id === selectedHandSegId);
            if (seg) seg.smooth = handSmooth;
        } else {
            for (const seg of handProtectSegments) {
                seg.smooth = handSmooth;
            }
        }
        saveHandSettings();
        render();
    }

    function updateHandSmooth2(val) {
        handSmooth2 = parseInt(val);
        document.getElementById('handSmooth2Val').textContent = handSmooth2;
        saveHandSettings();
        render();
    }

    async function saveHandSettings() {
        if (!subjectId || currentTrialIdx < 0) return;
        try {
            await API.put(`/api/deidentify/${subjectId}/hand-settings`, {
                trial_idx: currentTrialIdx,
                enabled: true,
                mask_radius: handMaskRadius,
                hand_smooth: handSmooth,
                forearm_radius: forearmRadius,
                forearm_extent: forearmExtent,
                hand_smooth2: handSmooth2,
                dlc_radius: dlcRadius,
                segments: handProtectSegments.map(s => ({
                    start: s.start, end: s.end, radius: s.radius,
                    smooth: s.smooth != null ? s.smooth : handSmooth,
                })),
            });
        } catch (e) {}
        renderTimeline();
    }

    function deleteSelectedHandSeg() {
        if (!selectedHandSegId) return;
        _pushUndo();
        handProtectSegments = handProtectSegments.filter(s => s.id !== selectedHandSegId);
        selectedHandSegId = null;
        saveHandSettings();
        render();
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
            side: s.side || 'full',
            shape: s.shape || 'oval',
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

    // ── Render current trial ──
    async function renderTrial() {
        if (!subjectId || currentTrialIdx < 0) return;

        // Save current specs first
        await saveSpecs();

        const btn = document.getElementById('renderBtn');
        const status = document.getElementById('renderStatus');
        const trialName = trialMeta ? trialMeta.trial_name : `trial ${currentTrialIdx}`;
        btn.disabled = true;
        status.textContent = `Rendering ${trialName}...`;

        try {
            const result = await API.post(`/api/deidentify/${subjectId}/render`, {
                trial_idx: currentTrialIdx,
            });
            const jobId = result.job_id;

            API.streamJob(jobId,
                (data) => {
                    if (data.status === 'running') {
                        const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                        status.textContent = `Rendering ${trialName}... ${pct}%`;
                    }
                },
                (data) => {
                    btn.disabled = false;
                    if (data.status === 'completed') {
                        status.textContent = `Render complete! ${trialName} saved.`;
                        if (trialMeta) trialMeta.has_blurred = true;
                        _updateViewButtons();
                        // Update trial button color to green
                        const trialBtns = document.querySelectorAll('.trial-btn');
                        if (trialBtns[currentTrialIdx]) {
                            trialBtns[currentTrialIdx].style.borderColor = 'var(--green)';
                            trialBtns[currentTrialIdx].style.color = 'var(--green)';
                        }
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

        const sideLabel = _sideLabel();
        const visibleSpots = blurSpots.filter(s => {
            const ss = s.side || 'full';
            return ss === 'full' || ss === sideLabel;
        });

        const faceRowH = 35;
        const customSpotRowH = 18;
        const handRowH = 25;
        const gap = 3;
        const nCustom = visibleSpots.filter(s => s.spot_type === 'custom').length;
        const hasHands = handCoverage.length > 0 || hasMediapipe;

        // Compute needed height and resize container
        let neededH = 4 + faceRowH + gap;
        if (nCustom > 0) neededH += nCustom * customSpotRowH + gap;
        if (hasHands) neededH += handRowH + gap;
        neededH = Math.max(neededH, 65);
        container.style.height = neededH + 8 + 'px';

        tlCanvas.width = container.clientWidth;
        tlCanvas.height = container.clientHeight;
        const L = _tlLayout();
        if (!L) return;

        tlCtx.clearRect(0, 0, L.cw, L.ch);

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

        // Face blur spot bars (blue) on top of detection coverage — use full row height
        const faceSpots = visibleSpots.filter(s => s.spot_type === 'face');
        if (faceSpots.length > 0) {
            const spotBarH = (faceRowH - 4) / faceSpots.length;
            let spotY = y + 2;
            for (const spot of faceSpots) {
                const x1 = _frameToTlX(spot.frame_start, L);
                const x2 = _frameToTlX(spot.frame_end, L);
                const w = Math.max(3, x2 - x1);
                const isSelected = spot.id === selectedSpotId;

                tlCtx.fillStyle = isSelected
                    ? 'rgba(33,150,243,0.6)' : 'rgba(33,150,243,0.35)';
                tlCtx.fillRect(x1, spotY, w, spotBarH - 1);
                tlCtx.strokeStyle = isSelected
                    ? 'rgba(33,150,243,1.0)' : 'rgba(33,150,243,0.7)';
                tlCtx.lineWidth = isSelected ? 1.5 : 0.5;
                tlCtx.strokeRect(x1, spotY, w, spotBarH - 1);

                if (isSelected) {
                    tlCtx.fillStyle = 'rgba(33,150,243,1.0)';
                    tlCtx.fillRect(x1 - 1, spotY, 3, spotBarH - 1);
                    tlCtx.fillRect(x2 - 2, spotY, 3, spotBarH - 1);
                }
                spotY += spotBarH;
            }
        }

        y += faceRowH + gap;

        // ── Custom spots: one row per spot (red) ──
        const customSpots = visibleSpots.filter(s => s.spot_type === 'custom');
        for (let ci = 0; ci < customSpots.length; ci++) {
            const spot = customSpots[ci];
            const x1 = _frameToTlX(spot.frame_start, L);
            const x2 = _frameToTlX(spot.frame_end, L);
            const w = Math.max(3, x2 - x1);
            const isSelected = spot.id === selectedSpotId;

            // Label
            tlCtx.fillStyle = 'rgba(150,150,150,0.5)';
            tlCtx.font = '10px sans-serif';
            tlCtx.fillText(`C${ci + 1}`, 2, y + 12);

            // Bar
            tlCtx.fillStyle = isSelected
                ? 'rgba(244,67,54,0.6)' : 'rgba(244,67,54,0.35)';
            tlCtx.fillRect(x1, y + 2, w, customSpotRowH - 4);
            tlCtx.strokeStyle = isSelected
                ? 'rgba(244,67,54,1.0)' : 'rgba(244,67,54,0.7)';
            tlCtx.lineWidth = isSelected ? 1.5 : 0.5;
            tlCtx.strokeRect(x1, y + 2, w, customSpotRowH - 4);

            if (isSelected) {
                tlCtx.fillStyle = 'rgba(244,67,54,1.0)';
                tlCtx.fillRect(x1 - 1, y + 2, 3, customSpotRowH - 4);
                tlCtx.fillRect(x2 - 2, y + 2, 3, customSpotRowH - 4);
            }
            y += customSpotRowH;
        }
        if (customSpots.length > 0) y += gap;

        // ── Hand row: green coverage + multiple draggable protection segments ──
        if (handCoverage.length > 0 || hasMediapipe) {
            tlCtx.fillStyle = 'rgba(150,150,150,0.5)';
            tlCtx.fillText('Hands', 2, y + 12);

            // Background coverage heatmap
            if (handCoverage.length > 0) {
                tlCtx.fillStyle = 'rgba(76,175,80,0.15)';
                const pw = Math.max(1, L.barW / L.range);
                for (const f of handCoverage) {
                    const x = _frameToTlX(f, L);
                    tlCtx.fillRect(x, y, pw, handRowH);
                }
            }

            // Protection segment bars (multiple, draggable)
            for (const seg of handProtectSegments) {
                const hx1 = _frameToTlX(seg.start, L);
                const hx2 = _frameToTlX(seg.end, L);
                const hw = Math.max(3, hx2 - hx1);
                const isSel = seg.id === selectedHandSegId;

                tlCtx.fillStyle = isSel ? 'rgba(76,175,80,0.55)' : 'rgba(76,175,80,0.35)';
                tlCtx.fillRect(hx1, y + 2, hw, handRowH - 4);
                tlCtx.strokeStyle = isSel ? 'rgba(76,175,80,1.0)' : 'rgba(76,175,80,0.8)';
                tlCtx.lineWidth = isSel ? 2 : 1;
                tlCtx.strokeRect(hx1, y + 2, hw, handRowH - 4);

                if (isSel) {
                    tlCtx.fillStyle = 'rgba(76,175,80,1.0)';
                    tlCtx.fillRect(hx1 - 1, y + 2, 3, handRowH - 4);
                    tlCtx.fillRect(hx2 - 2, y + 2, 3, handRowH - 4);
                }
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

    // Convert mouse event to canvas-space coordinates (handles CSS vs buffer size mismatch)
    function _tlMouseToCanvas(e) {
        const rect = tlCanvas.getBoundingClientRect();
        const scaleX = tlCanvas.width / rect.width;
        const scaleY = tlCanvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function _tlHitTest(e) {
        // Returns { type: 'spot'|'hand'|'hand_empty', spot?, seg?, edge } or null
        const { x: mx, y: my } = _tlMouseToCanvas(e);
        const L = _tlLayout();
        if (!L) return null;

        const sideLabel = _sideLabel();
        const visibleSpots = blurSpots.filter(s => {
            const ss = s.side || 'full';
            return ss === 'full' || ss === sideLabel;
        });

        const faceRowH = 35, customRowH = 25, handRowH = 25, gap = 3;
        let y = 2;

        // Hit test face blur spot bars within face row
        const faceSpots = visibleSpots.filter(s => s.spot_type === 'face');
        if (faceSpots.length > 0) {
            const spotBarH = (faceRowH - 4) / faceSpots.length;
            let spotY = y + 2;
            for (const spot of faceSpots) {
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

        // Hit test custom spot bars — one row per spot
        const customSpots = visibleSpots.filter(s => s.spot_type === 'custom');
        const customSpotRowH = 18;
        for (const spot of customSpots) {
            const x1 = _frameToTlX(spot.frame_start, L);
            const x2 = _frameToTlX(spot.frame_end, L);

            if (my >= y + 2 && my <= y + customSpotRowH - 2) {
                if (Math.abs(mx - x1) < 6) return { type: 'spot', spot, edge: 'start' };
                if (Math.abs(mx - x2) < 6) return { type: 'spot', spot, edge: 'end' };
                if (mx > x1 + 4 && mx < x2 - 4) return { type: 'spot', spot, edge: 'move' };
            }
            y += customSpotRowH;
        }
        if (customSpots.length > 0) y += gap;

        // Hit test hand protection segments
        if (handCoverage.length > 0 || hasMediapipe) {
            for (const seg of handProtectSegments) {
                const hx1 = _frameToTlX(seg.start, L);
                const hx2 = _frameToTlX(seg.end, L);

                if (my >= y + 2 && my <= y + handRowH - 2) {
                    if (Math.abs(mx - hx1) < 6) return { type: 'hand', seg, edge: 'start' };
                    if (Math.abs(mx - hx2) < 6) return { type: 'hand', seg, edge: 'end' };
                    if (mx > hx1 + 4 && mx < hx2 - 4) return { type: 'hand', seg, edge: 'move' };
                }
            }
            // If in hand row but not on a segment → create new
            if (my >= y + 2 && my <= y + handRowH - 2) {
                return { type: 'hand_empty', edge: 'create' };
            }
        }

        return null;
    }

    function onTlMouseDown(e) {
        const hit = _tlHitTest(e);
        if (!hit) return;
        e.preventDefault();

        if (hit.type === 'spot') {
            _pushUndo(); // snapshot before timeline drag
            tlDragSpot = hit.spot;
            tlDragEdge = hit.edge;
            tlDragStartX = e.clientX;
            tlDragOrigRange = { start: hit.spot.frame_start, end: hit.spot.frame_end };
            selectedSpotId = hit.spot.id;
            renderSpotList();
            updateSpotControls();
        } else if (hit.type === 'hand') {
            _pushUndo(); // snapshot before hand segment drag
            tlDragSpot = hit.seg;
            tlDragEdge = hit.edge;
            tlDragStartX = e.clientX;
            tlDragOrigRange = { start: hit.seg.start, end: hit.seg.end };
            selectedHandSegId = hit.seg.id;
            // Update sliders to show this segment's values
            handMaskRadius = hit.seg.radius || handMaskRadius;
            document.getElementById('handRadiusSlider').value = handMaskRadius;
            document.getElementById('handRadiusVal').textContent = handMaskRadius;
            handSmooth = hit.seg.smooth != null ? hit.seg.smooth : handSmooth;
            document.getElementById('handSmoothSlider').value = handSmooth;
            document.getElementById('handSmoothVal').textContent = handSmooth;
        } else if (hit.type === 'hand_empty') {
            // Start creating a new segment by dragging
            const L = _tlLayout();
            if (!L) return;
            const { x: mx } = _tlMouseToCanvas(e);
            tlDragCreateFrame = _tlXToFrame(mx, L);
            _pushUndo(); // snapshot before new hand segment
            tlDragSpot = 'newhand';
            tlDragEdge = 'create';
            tlDragStartX = e.clientX;
        }
        renderTimeline();
    }

    function onTlMouseMove(e) {
        if (tlDragSpot && tlDragEdge) {
            const L = _tlLayout();
            if (!L) return;
            const dx = e.clientX - tlDragStartX;
            const dFrames = Math.round((dx / L.barW) * L.range);

            // Creating new hand segment by dragging
            if (tlDragSpot === 'newhand') {
                const L2 = _tlLayout();
                if (!L2) return;
                const { x: mx } = _tlMouseToCanvas(e);
                const endFrame = _tlXToFrame(mx, L2);
                // Show preview by temporarily adding a segment
                const existing = handProtectSegments.find(s => s.id === -1);
                const s = Math.min(tlDragCreateFrame, endFrame);
                const en = Math.max(tlDragCreateFrame, endFrame);
                if (existing) {
                    existing.start = s; existing.end = en;
                } else {
                    handProtectSegments.push({ id: -1, start: s, end: en, radius: handMaskRadius, smooth: handSmooth });
                }
                renderTimeline();
                return;
            }

            // Drag existing hand segment
            if (tlDragSpot && typeof tlDragSpot === 'object' && tlDragSpot.start !== undefined) {
                if (tlDragEdge === 'start') {
                    tlDragSpot.start = Math.max(
                        trialMeta.start_frame,
                        Math.min(tlDragSpot.end - 1, tlDragOrigRange.start + dFrames)
                    );
                } else if (tlDragEdge === 'end') {
                    tlDragSpot.end = Math.min(
                        trialMeta.end_frame,
                        Math.max(tlDragSpot.start + 1, tlDragOrigRange.end + dFrames)
                    );
                } else if (tlDragEdge === 'move') {
                    const len = tlDragOrigRange.end - tlDragOrigRange.start;
                    let newStart = tlDragOrigRange.start + dFrames;
                    newStart = Math.max(trialMeta.start_frame, Math.min(trialMeta.end_frame - len, newStart));
                    tlDragSpot.start = newStart;
                    tlDragSpot.end = newStart + len;
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
            if (tlDragSpot === 'newhand') {
                // Finalize the new segment (replace the preview id=-1)
                const preview = handProtectSegments.find(s => s.id === -1);
                if (preview && preview.end > preview.start) {
                    preview.id = nextHandSegId++;
                    selectedHandSegId = preview.id;
                } else {
                    // Too small, remove it
                    handProtectSegments = handProtectSegments.filter(s => s.id !== -1);
                }
                saveHandSettings();
                render();
            } else if (tlDragSpot && typeof tlDragSpot === 'object' && tlDragSpot.start !== undefined) {
                saveHandSettings();
            } else {
                scheduleSave();
            }
            tlDragSpot = null;
            tlDragEdge = null;
            tlDragStartX = null;
            tlDragOrigRange = null;
            tlDragCreateFrame = null;
        }
    }

    function onTlClick(e) {
        if (tlDragSpot) return; // was a drag, not click
        const { x: mx } = _tlMouseToCanvas(e);
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
        setPlaybackSpeed,
        toggleSide,
        setViewMode,
        resetZoom,
        seekFrame,
        toggleAddCustom,
        copyFromOtherCamera,
        toggleSpotShape,
        selectSpot,
        deleteSpot,
        updateSpotDim,
        updateSpotFrameRange,
        toggleHandOverlay,
        updateHandRadius,
        deleteSelectedHandSeg,
        updateForearmRadius,
        updateForearmExtent,
        updateHandSmooth,
        updateHandSmooth2,
        updateHandTemporalSmooth,
        updateDlcRadius,
        goToMediapipe,
        renderTrial,
        undo,
        redo,
    };
})();
