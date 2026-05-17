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

    // Render queue
    let _renderInProgress = false;
    let _renderQueue = []; // [{subjectId, trialIdx, trialName}]

    // Remote server detection
    let _hasRemote = false;
    (async () => {
        try {
            const s = await API.get('/api/settings');
            _hasRemote = !!(s.remote_host && s.remote_host.trim());
        } catch {}
    })();

    /** Show inline Local/Remote buttons inside a Run button.
     *  If no remote server, resolves 'local-cpu' immediately. */
    function _promptTarget(btn) {
        return new Promise(resolve => {
            if (!_hasRemote) { resolve('local-cpu'); return; }
            const origHTML = btn.innerHTML;
            const origOnclick = btn.onclick;
            btn.onclick = null;
            btn.style.padding = '0';
            btn.innerHTML = `<span style="display:flex;gap:2px;width:100%;height:100%;">
                <button class="btn btn-sm btn-success" style="flex:1;font-size:10px;border-radius:3px 0 0 3px;">Local</button>
                <button class="btn btn-sm btn-success" style="flex:1;font-size:10px;border-radius:0 3px 3px 0;">Remote</button>
            </span>`;
            const [localBtn, remoteBtn] = btn.querySelectorAll('button');
            const cleanup = () => { btn.innerHTML = origHTML; btn.onclick = origOnclick; btn.style.padding = ''; };
            localBtn.addEventListener('click', e => { e.stopPropagation(); cleanup(); resolve('local-cpu'); });
            remoteBtn.addEventListener('click', e => { e.stopPropagation(); cleanup(); resolve('remote'); });
        });
    }

    // Camera
    let cameraMode = 'stereo';
    let cameraNames = ['OS', 'OD'];
    let currentSide = 'OS';
    let hasMediapipe = false;

    // Face detection results: [{frame, faces: [{x1,y1,x2,y2,side}]}]
    // faceDetByLocalFrame is a Map<localFrame, faces[]> built whenever faceDetections
    // changes — avoids the dense-index bug (faceDetections[localFrame] is wrong because
    // the array is sparse: only frames with detections are present).
    let faceDetections = [];
    let faceDetByLocalFrame = new Map();
    let _pendingAutoCreateFaceSpots = false;

    function _rebuildFaceDetMap() {
        faceDetByLocalFrame = new Map();
        if (!trialMeta) return;
        const startFrame = trialMeta.start_frame || 0;
        for (const entry of faceDetections) {
            const lf = entry.frame - startFrame;
            faceDetByLocalFrame.set(lf, entry.faces || []);
        }
    }
    // Active blur spots: [{id, spot_type, x, y, radius, frame_start, frame_end, side}]
    let blurSpots = [];
    let nextSpotId = 1;
    let selectedSpotId = null;
    let addingCustom = false;
    let _pendingCustomShape = 'oval';   // shape of the next custom spot

    // View mode: 'original' (with overlays), 'preview' (blur mask only), 'deidentified' (rendered video)
    let viewMode = 'original';

    // Hand overlay
    let handOverlayEnabled = true;
    let handLandmarks = [];   // [{x, y, side}] for current frame (after smoothing)
    let handLandmarksBulk = {}; // {frameNum: [{x, y, side}]} all frames
    let handTemporalSmooth = 0; // deprecated -- kept for back-compat with saved settings
    let handMaskRadius = 5;
    let forearmRadius = 10;  // dilation around forearm triangle (separate from circle radius)
    let forearmExtent = 0.4; // 0=wrist, 1=elbow, >1=past elbow
    let handSmooth = 7;       // hand-region dilation (no longer applies to arm triangle)
    let handSmooth2 = 0;      // deprecated
    let armDorsalDilate = 0;   // extra dilation of dorsal arm edge (elbow→pinky MCP)
    let armVentralDilate = 0;  // extra dilation of ventral arm edge (elbow→thumb CMC)
    let handMaskEnabled = true;
    // Hand-mask source is always MediaPipe -- HRnet hand-mask path removed.
    let handMaskSource = 'mediapipe';
    let hrnetMaskThresh = 0.30;  // unused (kept for back-compat in saved settings)
    let hrnetMaskSmooth = 7;     // unused
    // dlcRadius removed — all hand keypoints use the same marker size
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
    let didDrag = false;  // true if mouse moved during a mousedown — suppresses click

    // Spot drag state (for on-canvas resize/move)
    let spotDrag = null; // {spotId, handle, startMx, startMy, origSpot}

    // Timeline
    let tlCanvas, tlCtx;
    let tlDragSpot = null;    // spot being dragged (blur spot object, segment object, or 'newhand')
    let tlDragEdge = null;    // 'start', 'end', 'move', or 'create'
    let tlDragStartX = null;  // mouse X at drag start
    let tlDragOrigRange = null; // {start, end} at drag start
    let tlDragCreateFrame = null; // frame at start of hand segment creation drag
    let tlDragNewHandSide = null; // camera side for new hand segment being created
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
        saveHandSettings();
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
        saveHandSettings();
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

        // Save on page unload (handles browser refresh, tab close, navigation away)
        window.addEventListener('beforeunload', () => {
            if (!subjectId || currentTrialIdx < 0 || blurSpots.length === 0) return;
            if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
            const specs = blurSpots.map(s => ({
                spot_type: s.spot_type, x: s.x, y: s.y, radius: s.radius,
                width: s.width || null, height: s.height || null,
                offset_x: s.offset_x || 0, offset_y: s.offset_y || 0,
                frame_start: s.frame_start, frame_end: s.frame_end,
                side: s.side || 'full', shape: s.shape || 'oval',
            }));
            const blob = new Blob(
                [JSON.stringify({ trial_idx: currentTrialIdx, specs })],
                { type: 'application/json' }
            );
            navigator.sendBeacon(`/api/deidentify/${subjectId}/blur-specs`, blob);
        });

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
        faceDetections = []; faceDetByLocalFrame = new Map(); _pendingAutoCreateFaceSpots = false;
        blurSpots = [];
        selectedSpotId = null;
        handLandmarks = [];

        try {
            const data = await API.get(`/api/deidentify/${sid}/trials`);
            subjectName = data.subject.name;
            trials = data.trials;
            cameraMode = data.subject.camera_mode || 'stereo';
            hasMediapipe = data.has_mediapipe || false;
            const hasPose = data.has_pose || false;
            // Hide "Detect Pose" button if both hand and pose detection are done
            const goBtn = document.getElementById('goAnalyzeBtn');
            if (goBtn) goBtn.style.display = (hasMediapipe && hasPose) ? 'none' : '';

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
            btn.textContent = t.trial_name.includes('_') ? t.trial_name.split('_').slice(1).join('_') : t.trial_name;
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
        // Flush any pending save for the previous trial before switching
        if (saveTimer) {
            clearTimeout(saveTimer);
            saveTimer = null;
            await saveSpecs();
        }
        currentTrialIdx = idx;
        trialMeta = trials[idx];
        currentFrame = trialMeta.start_frame;
        if (typeof setNavState === 'function') setNavState({ trialIdx: idx, frame: currentFrame });

        // Always start in mask-editing mode when switching trials
        viewMode = 'original';
        _updateViewButtons();
        _updateSidebarState();
        // Sync "has faces" checkbox
        const hfToggle = document.getElementById('hasFacesToggle');
        if (hfToggle) hfToggle.checked = trialMeta.has_faces !== false;
        totalFrames = trialMeta.frame_count;
        fps = trialMeta.fps || 30;
        handLandmarks = [];
        handLandmarksBulk = {}; // clear bulk cache for new trial

        // Load face detections and blur specs in parallel
        {
            let fdResp = { faces: [] }, bsResp = { specs: [] };
            try {
                [fdResp, bsResp] = await Promise.all([
                    API.get(`/api/deidentify/${subjectId}/face-detections?trial_idx=${idx}`),
                    API.get(`/api/deidentify/${subjectId}/blur-specs?trial_idx=${idx}`),
                ]);
            } catch (e) {}

            faceDetections = fdResp.faces || [];
            _rebuildFaceDetMap();
            // Load saved specs and re-anchor their frame range to the
            // current trial.  Saved frame_start / frame_end are global
            // frame numbers from whenever the spec was last saved -- if
            // the trial got re-trimmed in between, the saved range may
            // fall entirely outside the current trial (which is what
            // causes the "spots show in the sidebar but never draw on
            // the canvas" bug: the per-frame range check excludes them).
            const _tStart = trialMeta.start_frame || 0;
            const _tEnd = trialMeta.end_frame != null
                ? trialMeta.end_frame
                : (_tStart + (trialMeta.frame_count || 0) - 1);
            blurSpots = (bsResp.specs || []).map(s => {
                let fs = (s.frame_start != null) ? s.frame_start : _tStart;
                let fe = (s.frame_end   != null) ? s.frame_end   : _tEnd;
                // Re-anchor when the saved range doesn't overlap the
                // current trial at all.
                if (fe < _tStart || fs > _tEnd) {
                    fs = _tStart;
                    fe = _tEnd;
                } else {
                    fs = Math.max(_tStart, fs);
                    fe = Math.min(_tEnd,   fe);
                }
                return { ...s, id: nextSpotId++, frame_start: fs, frame_end: fe };
            });

            // Warn about legacy face spots without proper side assignment
            if (cameraMode === 'stereo') {
                const legacy = blurSpots.filter(s => s.spot_type === 'face' && (!s.side || s.side === 'full'));
                if (legacy.length > 0) {
                    document.getElementById('faceDetStatus').textContent =
                        'Re-run Detect Faces to assign blur spots to individual cameras.';
                }
            }

            if (faceDetections.length > 0) {
                const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
                document.getElementById('faceDetStatus').textContent = `${nFaces} frames with faces`;
                document.getElementById('detectFacesBtn').style.display = 'block';
                document.getElementById('detectFacesBtn').textContent = 'Re-detect Faces';
                // Don't auto-create face spots — respect user's edits.
                // Face spots are only created after explicit "Detect Faces" click.
            } else if (blurSpots.length === 0 && !bsResp._ever_saved) {
                // Trial has never been touched — auto-detect faces on first visit
                document.getElementById('faceDetStatus').textContent = 'Detecting faces...';
                _autoDetectFaces(idx);
            } else {
                // Trial has been edited but detections not saved — show button only
                document.getElementById('detectFacesBtn').style.display = 'block';
                document.getElementById('detectFacesBtn').textContent = 'Detect Faces';
                document.getElementById('faceDetStatus').textContent = '';
            }
        }

        // Update UI
        document.querySelectorAll('.trial-btn').forEach((b, i) => {
            b.classList.toggle('active', i === idx);
        });

        // Load hand protection segments
        handProtectSegments = [];
        selectedHandSegId = null;
        try {
            const hs = await API.get(`/api/deidentify/${subjectId}/hand-settings?trial_idx=${idx}`);
            handMaskRadius = hs.mask_radius || 10;
            handSmooth = hs.hand_smooth || 10;
            forearmRadius = hs.forearm_radius || 10;
            forearmExtent = hs.forearm_extent != null ? hs.forearm_extent : 0.5;
            handSmooth2 = 0;  // deprecated, always 0
            handTemporalSmooth = 0;  // deprecated, always 0
            armDorsalDilate = hs.arm_dorsal_dilate || 0;
            armVentralDilate = hs.arm_ventral_dilate || 0;
            handOverlayEnabled = hs.show_landmarks || false;
            handMaskSource = 'mediapipe';
            document.getElementById('handRadiusSlider').value = handMaskRadius;
            document.getElementById('handRadiusVal').textContent = handMaskRadius;
            document.getElementById('handSmoothSlider').value = handSmooth;
            document.getElementById('handSmoothVal').textContent = handSmooth;
            forearmRadius = 10; // hardcoded
            document.getElementById('handExtentSlider').value = forearmExtent;
            document.getElementById('handExtentVal').textContent = forearmExtent.toFixed(1);
            const dorsalSlider = document.getElementById('handDorsalSlider');
            if (dorsalSlider) {
                dorsalSlider.value = armDorsalDilate;
                document.getElementById('handDorsalVal').textContent = armDorsalDilate;
            }
            const ventralSlider = document.getElementById('handVentralSlider');
            if (ventralSlider) {
                ventralSlider.value = armVentralDilate;
                document.getElementById('handVentralVal').textContent = armVentralDilate;
            }
            const overlayToggle = document.getElementById('handOverlayToggle');
            if (overlayToggle) { overlayToggle.checked = handOverlayEnabled; }
            if (hs.segments && hs.segments.length > 0) {
                handProtectSegments = hs.segments.map(s => ({
                    id: nextHandSegId++,
                    start: s.start,
                    end: s.end,
                    radius: s.radius || handMaskRadius,
                    smooth: s.smooth != null ? s.smooth : handSmooth,
                    side: s.side || null,  // null = legacy (applies to both cameras)
                }));
            } else if (hs.has_row) {
                // DB row exists with no segments — user intentionally cleared, respect that
                handProtectSegments = [];
            } else if (hasMediapipe && trialMeta) {
                // No DB row = never configured — create per-camera defaults
                handProtectSegments = [];
                const sides = cameraMode === 'single' ? ['full'] : ['left', 'right'];
                for (const side of sides) {
                    handProtectSegments.push({
                        id: nextHandSegId++,
                        start: trialMeta.start_frame,
                        end: trialMeta.end_frame,
                        radius: handMaskRadius,
                        smooth: handSmooth,
                        side: side,
                    });
                }
            }
        } catch (e) {
            // No saved settings at all — create default hand protection if MP available
            if (hasMediapipe && trialMeta) {
                handProtectSegments = [];
                const sides = cameraMode === 'single' ? ['full'] : ['left', 'right'];
                for (const side of sides) {
                    handProtectSegments.push({
                        id: nextHandSegId++,
                        start: trialMeta.start_frame,
                        end: trialMeta.end_frame,
                        radius: handMaskRadius,
                        smooth: handSmooth,
                        side: side,
                    });
                }
            }
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

        _updateSidebarState();
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

        // Fit on first load only (scale 1 = never zoomed)
        if (scale === 1 && offsetX === 0 && offsetY === 0) fitImage();

        // Update display
        const localFrame = frameNum - (trialMeta ? trialMeta.start_frame : 0);
        document.getElementById('frameDisplay').textContent =
            `Frame: ${localFrame} / ${totalFrames - 1}`;

        // Load hand landmarks from bulk cache with temporal smoothing
        const curSideForProtect = _sideLabel();
        const inProtectSeg = handProtectSegments.some(s =>
            frameNum >= s.start && frameNum <= s.end &&
            (!s.side || s.side === curSideForProtect || s.side === 'full'));
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
        // Update View Deidentified Video button availability
        const viewDeidBtn = document.getElementById('viewDeidBtn');
        if (viewDeidBtn) {
            const avail = trialMeta && trialMeta.has_blurred;
            viewDeidBtn.disabled = !avail;
            viewDeidBtn.style.opacity = avail ? '' : '0.35';
            viewDeidBtn.title = avail ? 'View deidentified video' : 'Render first to enable';
        }
    }

    function _updateSidebarState() {
        const sidebar = document.querySelector('.deid-sidebar');
        if (!sidebar) return;
        const renderSection = document.getElementById('renderSection');

        // Show the correct workflow button group
        const maskBtns    = document.getElementById('workflowMaskBtns');
        const previewBtns = document.getElementById('workflowPreviewBtns');
        const deidBtns    = document.getElementById('workflowDeidBtns');
        if (maskBtns)    maskBtns.style.display    = viewMode === 'original'      ? '' : 'none';
        if (previewBtns) previewBtns.style.display  = viewMode === 'preview'       ? '' : 'none';
        if (deidBtns)    deidBtns.style.display     = viewMode === 'deidentified'  ? '' : 'none';

        // Keep the View Deidentified Video button in sync
        _updateViewButtons();

        // Dim non-workflow sections when not in mask-editing mode.
        // The renderSection (workflow buttons) always stays interactive.
        const allSections = sidebar.querySelectorAll('.section');
        const shouldDimOthers = (viewMode === 'preview' || viewMode === 'deidentified');
        allSections.forEach(s => {
            if (s === renderSection) {
                s.style.opacity = '';
                s.style.pointerEvents = '';
            } else {
                s.style.opacity    = shouldDimOthers ? '0.4' : '';
                s.style.pointerEvents = shouldDimOthers ? 'none' : '';
            }
        });
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

    function goToAnalyze() {
        if (!subjectId) return;
        sessionStorage.setItem('deid_returnFrame', currentFrame);
        sessionStorage.setItem('deid_returnSubject', subjectId);
        window.location.href = `/mano?subject=${subjectId}`;
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
            if (localFrame >= 0) {
                const faces = _facesForCurrentSide({ faces: faceDetByLocalFrame.get(localFrame) || [] });
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
    // Boundary-pixel extraction: returns a new canvas whose only opaque
    // pixels are the 8-neighbour edge of the source canvas's alpha mask,
    // thickened by drawing the same edge mask at ±1 px offsets.  Used
    // to draw the green hand-protection outline so it traces the exact
    // contour of the blur-cutout mask (which is what the render path
    // actually uses to protect the hand).
    function _alphaEdgeRing(srcCanvas, cr, cg, cb, alpha) {
        const w = srcCanvas.width, h = srcCanvas.height;
        const src = srcCanvas.getContext('2d').getImageData(0, 0, w, h).data;
        const out = document.createElement('canvas');
        out.width = w; out.height = h;
        const octx = out.getContext('2d');
        const edgeImg = octx.createImageData(w, h);
        const ed = edgeImg.data;
        const aByte = Math.round(255 * Math.max(0, Math.min(1, alpha)));
        const inside = (x, y) => x >= 0 && x < w && y >= 0 && y < h
                                  && src[(y * w + x) * 4 + 3] > 128;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = (y * w + x) * 4;
                if (!(src[i + 3] > 128)) continue;
                // Edge if any 4-neighbour is outside the mask.
                if (inside(x - 1, y) && inside(x + 1, y)
                    && inside(x, y - 1) && inside(x, y + 1)) continue;
                ed[i] = cr; ed[i + 1] = cg; ed[i + 2] = cb; ed[i + 3] = aByte;
            }
        }
        octx.putImageData(edgeImg, 0, 0);
        // Thicken the 1-px ring to ~3 px by re-drawing at small offsets.
        const thick = document.createElement('canvas');
        thick.width = w; thick.height = h;
        const tctx = thick.getContext('2d');
        for (const [dx, dy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
            tctx.drawImage(out, dx, dy);
        }
        return thick;
    }

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

    // ── HRnet MIP cache + client-side mask builder ───────────────────────
    // When handMaskSource === 'hrnet' we threshold each frame's max-over-
    // joints heatmap (loaded once per trial) and place it into the canvas
    // at the per-frame bbox.  Slider changes (threshold / smooth) just
    // re-run this builder — no server round-trip.
    let _hrnetMaskData = null;       // {mip_L: Float32Array, mip_R, bbox_L, bbox_R, n_frames, start_frame, mip_size}
    let _hrnetMaskTrialIdx = -1;     // which trial _hrnetMaskData was loaded for
    let _hrnetMaskFetchInFlight = null;

    function _decodeFloat16ToFloat32(b64) {
        // Decode a base64-encoded float16 buffer into a Float32Array via
        // a one-pass conversion through Uint16Array.  IEEE 754 half →
        // float reconstruction; only handles finite values which is what
        // HRnet MIPs contain (∈ [0, 1]).
        if (!b64) return null;
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        const f32 = new Float32Array(u16.length);
        for (let i = 0; i < u16.length; i++) {
            const h = u16[i];
            const sign = (h & 0x8000) >> 15;
            const exp = (h & 0x7c00) >> 10;
            const frac = h & 0x03ff;
            let v;
            if (exp === 0) v = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
            else if (exp === 0x1f) v = frac ? NaN : (sign ? -Infinity : Infinity);
            else v = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
            f32[i] = v;
        }
        return f32;
    }

    async function _ensureHrnetMaskData() {
        if (currentTrialIdx < 0 || !subjectId) return null;
        if (_hrnetMaskTrialIdx === currentTrialIdx && _hrnetMaskData) return _hrnetMaskData;
        if (_hrnetMaskFetchInFlight) return _hrnetMaskFetchInFlight;
        _hrnetMaskFetchInFlight = (async () => {
            try {
                const r = await API.get(`/api/deidentify/${subjectId}/hrnet-mask-data?trial_idx=${currentTrialIdx}`);
                if (!r.available) {
                    _hrnetMaskData = null;
                    return null;
                }
                _hrnetMaskData = {
                    mip_L: _decodeFloat16ToFloat32(r.mip_L_b64),
                    mip_L_shape: r.mip_L_shape,
                    mip_R: _decodeFloat16ToFloat32(r.mip_R_b64),
                    mip_R_shape: r.mip_R_shape,
                    bbox_L: r.bbox_L || [],
                    bbox_R: r.bbox_R || [],
                    n_frames: r.n_frames,
                    start_frame: r.start_frame,
                    mip_size: r.mip_size || 64,
                };
                _hrnetMaskTrialIdx = currentTrialIdx;
                return _hrnetMaskData;
            } catch (e) {
                console.warn('hrnet-mask-data fetch failed:', e);
                return null;
            } finally {
                _hrnetMaskFetchInFlight = null;
            }
        })();
        return _hrnetMaskFetchInFlight;
    }

    function _buildHandMaskHrnet(frameInTrial, side, smoothPx, w, h) {
        // Returns an offscreen canvas with the binary HRnet mask painted
        // white where MIP > threshold, then Gaussian-blurred + thresholded
        // for smooth edges.  Trial-relative frame index, since the MIP is
        // sized to the trial.
        const data = _hrnetMaskData;
        if (!data) return null;
        const f = Math.min(Math.max(0, frameInTrial), data.n_frames - 1);
        const mip = side === 'OS' ? data.mip_L : data.mip_R;
        const bboxes = side === 'OS' ? data.bbox_L : data.bbox_R;
        if (!mip || !bboxes || !bboxes[f]) return null;
        const S = data.mip_size;                         // 64
        const off = f * S * S;
        const bbox = bboxes[f];                           // [x1, y1, x2, y2] in image px
        const bx = bbox[0], by = bbox[1];
        const bw = bbox[2] - bbox[0], bh = bbox[3] - bbox[1];
        if (bw <= 0 || bh <= 0) return null;

        // Step 1: build a 64×64 binary mask at the threshold.
        //
        // The saved MIP comes from sigmoid(logits) in the inference
        // pipeline → bimodal: most pixels at the ~0.7 background floor,
        // a thin tail at joint peaks near 1.0.  Anchoring the lower
        // end at the per-frame MEDIAN (which sits inside that
        // background spike) and the upper end at the per-frame max
        // makes the threshold slider span the actually-interesting
        // shoulder + peak portion instead of cliff-dropping at
        // threshold ≈ 0.  Matches the server-side normalisation.
        const samples = [];
        let hi = -Infinity;
        for (let i = 0; i < S * S; i++) {
            const v = mip[off + i];
            samples.push(v);
            if (v > hi) hi = v;
        }
        samples.sort((a, b) => a - b);
        const baseline = samples[(S * S) >> 1]; // median
        const span = hi - baseline;
        const small = document.createElement('canvas');
        small.width = S; small.height = S;
        const sCtx = small.getContext('2d');
        const img = sCtx.createImageData(S, S);
        const thresh = hrnetMaskThresh;
        for (let i = 0; i < S * S; i++) {
            let v = span > 1e-6 ? (mip[off + i] - baseline) / span : 0;
            if (v < 0) v = 0;
            else if (v > 1) v = 1;
            const on = v > thresh ? 255 : 0;
            const j = i * 4;
            img.data[j] = on; img.data[j+1] = on; img.data[j+2] = on; img.data[j+3] = on;
        }
        sCtx.putImageData(img, 0, 0);

        // Step 2: upscale the 64×64 binary mask onto a canvas-sized buffer
        // at the bbox location.  Use nearest-neighbor (smoothing disabled)
        // so the intermediate stays binary — the morphological close in
        // step 3 owns all the smoothing + dilation.
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const cx = c.getContext('2d');
        cx.imageSmoothingEnabled = false;
        cx.drawImage(small,
            offsetX + bx * scale, offsetY + by * scale,
            bw * scale,           bh * scale);

        // Step 3: morphological close — identical to the MP path's
        // _buildHandMask finalization.  Blurs the binary mask, stacks
        // 8 draws to saturate alpha, then thresholds at >30.  This
        // DILATES the foreground (the hand) as smoothPx grows.  The
        // earlier inline `blur + threshold@127` was approximately
        // identity at the hand interior but eroded the hand at the
        // bbox edges (where the binary white neighbors empty canvas),
        // which is what made the slider look like it was acting on
        // the background.
        return _morphClose(c, smoothPx);
    }

    // ── Build smoothed hand protection mask ──
    // Threshold (image-pixel units) for the DLC-vs-MP tip-disagreement
    // check.  When DLC's thumb or index tip is more than this far from
    // the matching MP tip, the chain "MCP/CMC -> DLC tip" gets seeded
    // with extra circles in the mask so coverage extends to wherever
    // DLC says the fingertip is -- WITHOUT removing the MP-derived
    // circles (a more generous mask is preferable to a wrong one).
    const _DLC_TIP_DISAGREE_PX = 20;

    function _appendDlcCorrectedFingerPoints(allPoints, byJoint, rawLandmarks) {
        // Find DLC-derived joints in the unfiltered landmark list.
        const dlcByJoint = {};
        for (const lm of rawLandmarks) {
            if (lm.type === 'dlc' && lm.joint != null) {
                dlcByJoint[lm.joint] = lm;
            }
        }
        // ── Index finger: MCP=5, MP tip=8, DLC tip=8 ──
        const ixMcp    = byJoint[5];
        const ixTipMp  = byJoint[8];
        const ixTipDlc = dlcByJoint[8];
        if (ixMcp && ixTipMp && ixTipDlc) {
            const d = Math.hypot(ixTipMp.x - ixTipDlc.x, ixTipMp.y - ixTipDlc.y);
            if (d > _DLC_TIP_DISAGREE_PX) {
                const dx = ixTipDlc.x - ixMcp.x;
                const dy = ixTipDlc.y - ixMcp.y;
                // Seed along MCP -> DLC tip every 1/6 of the way --
                // produces overlapping circles even with small radii.
                for (const f of [1/6, 1/3, 1/2, 2/3, 5/6, 1.0]) {
                    allPoints.push({ x: ixMcp.x + f * dx, y: ixMcp.y + f * dy, type: 'dlc_interp' });
                }
            }
        }
        // ── Thumb: CMC=1, MP tip=4, DLC tip=4 ──
        const thCmc    = byJoint[1];
        const thTipMp  = byJoint[4];
        const thTipDlc = dlcByJoint[4];
        if (thCmc && thTipMp && thTipDlc) {
            const d = Math.hypot(thTipMp.x - thTipDlc.x, thTipMp.y - thTipDlc.y);
            if (d > _DLC_TIP_DISAGREE_PX) {
                const dx = thTipDlc.x - thCmc.x;
                const dy = thTipDlc.y - thCmc.y;
                for (const f of [1/6, 1/3, 1/2, 2/3, 5/6, 1.0]) {
                    allPoints.push({ x: thCmc.x + f * dx, y: thCmc.y + f * dy, type: 'dlc_interp' });
                }
            }
        }
    }

    function _buildHandMask(landmarks, radiusPx, forearmPx, smoothPx, smooth2Px, w, h,
                              armDorsalPx, armVentralPx) {
        // Hand landmarks (MediaPipe) + DLC fallback ONLY for joints MP
        // missed on this frame.  Two earlier strategies both regressed:
        //   - "replace MP with DLC" (original): when DLC's tip
        //     prediction was wrong, the mask drifted off the MP
        //     fingertip the user could still see on screen.
        //   - "add DLC alongside MP" (previous fix): when DLC was bad
        //     on one camera, the mask grew an extra circle far from
        //     the hand, making the slider feel frozen on that side.
        // Treating DLC strictly as a fill-in keeps MP authoritative
        // whenever it detected the joint and only draws a DLC-derived
        // circle when MP genuinely lacked that joint.  Both
        // regressions are avoided and the mask is symmetric across
        // the two cameras.
        const dlcLms = landmarks.filter(l => l.type === 'dlc');
        let mergedLandmarks = landmarks.filter(l => l.type !== 'dlc');
        if (dlcLms.length > 0) {
            const mpJoints = new Set(
                mergedLandmarks
                    .filter(l => l.type === 'hand' && l.joint != null)
                    .map(l => l.joint)
            );
            for (const dlc of dlcLms) {
                if (!mpJoints.has(dlc.joint)) {
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
                if (!a || !b) continue;
                // wrist→MCP segment (ci=0) gets three palm-fill rows at
                // 1/4, 1/2, 3/4; finger inter-joint segments only need 1/2.
                const fracs = ci === 0 ? [0.25, 0.5, 0.75] : [0.5];
                for (const f of fracs) {
                    allPoints.push({
                        x: a.x + f * (b.x - a.x),
                        y: a.y + f * (b.y - a.y),
                        type: 'interp',
                    });
                }
            }
        }

        // Extra midpoint between thumb MCP (joint 2) and index MCP (joint 5)
        const thumbMCP = byJoint[2];
        const indexMCP = byJoint[5];
        if (thumbMCP && indexMCP) {
            allPoints.push({ x: (thumbMCP.x + indexMCP.x) / 2, y: (thumbMCP.y + indexMCP.y) / 2, type: 'interp' });
        }

        // DLC tip-disagreement: when DLC's tip is > 20 px from MP's
        // tip, seed extra circles along MCP / CMC -> DLC tip.  Keeps
        // the MP-derived circles too (more generous coverage).
        _appendDlcCorrectedFingerPoints(allPoints, byJoint, landmarks);

        // Draw circles for all hand keypoints + interpolated midpoints
        // (hand-only, no arm).  Smoothing below dilates this layer only.
        for (const lm of allPoints) {
            ctx1.beginPath();
            ctx1.arc(offsetX + lm.x * scale, offsetY + lm.y * scale, radiusPx, 0, Math.PI * 2);
            ctx1.fill();
        }
        // Smooth (= Hand-dilate) applies ONLY to the hand circles --
        // not to the arm triangle drawn on the separate canvas below.
        const handLayer = _morphClose(c1, smoothPx);

        // Step 2: Build the arm-triangle mask on its own canvas so the
        // arm doesn't pick up the Hand-dilate smoothing.  Per-edge
        // dilation uses the user's Dorsal / Ventral sliders.
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

        const c2 = document.createElement('canvas');
        c2.width = w; c2.height = h;
        const ctx2 = c2.getContext('2d');
        ctx2.fillStyle = '#fff';
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

            ctx2.lineCap = 'round';

            // Filled triangle (the bare arm region).
            ctx2.beginPath();
            ctx2.moveTo(pts[0].sx, pts[0].sy);
            ctx2.lineTo(pts[1].sx, pts[1].sy);
            ctx2.lineTo(pts[2].sx, pts[2].sy);
            ctx2.closePath();
            ctx2.fill();

            // Ventral edge (elbow → thumb CMC).
            const ventral = (armVentralPx != null ? armVentralPx : 0);
            if (ventral > 0) {
                ctx2.strokeStyle = '#fff';
                ctx2.lineWidth = ventral * 2;
                ctx2.beginPath();
                ctx2.moveTo(pts[1].sx, pts[1].sy);
                ctx2.lineTo(pts[2].sx, pts[2].sy);
                ctx2.stroke();
            }
            // Dorsal edge (elbow → pinky MCP).
            const dorsal = (armDorsalPx != null ? armDorsalPx : forearmPx);
            if (dorsal > 0) {
                ctx2.strokeStyle = '#fff';
                ctx2.lineWidth = dorsal * 2;
                ctx2.beginPath();
                ctx2.moveTo(pts[1].sx, pts[1].sy);
                ctx2.lineTo(pts[0].sx, pts[0].sy);
                ctx2.stroke();
            }
        }

        // Union the (smoothed) hand layer and the (un-smoothed) arm
        // layer into one mask canvas.
        const out = document.createElement('canvas');
        out.width = w; out.height = h;
        const octx = out.getContext('2d');
        octx.drawImage(handLayer, 0, 0);
        octx.drawImage(c2, 0, 0);
        return out;
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

        // Build and subtract hand protection mask.  Two source paths:
        //  - MediaPipe: stamp circles at landmark positions (requires
        //    landmarks for the current frame).
        //  - HRnet: threshold the MIP at the per-frame bbox (works
        //    without any MediaPipe labels).
        const landmarks = _getVisibleLandmarks();
        const activeSegment = _getActiveHandSegment();
        if (activeSegment) {
            let handMask = null;
            if (handMaskSource === 'hrnet') {
                if (_hrnetMaskData && _hrnetMaskTrialIdx === currentTrialIdx) {
                    const frameInTrial = currentFrame - (_hrnetMaskData.start_frame || 0);
                    const _curSide = _sideLabel();
                    handMask = _buildHandMaskHrnet(
                        frameInTrial, _curSide === 'left' ? 'OS' : 'OD',
                        hrnetMaskSmooth * scale, cw, ch,
                    );
                } else {
                    _ensureHrnetMaskData().then(d => { if (d) render(); });
                }
            } else if (landmarks.length > 0) {
                const radiusPx = (activeSegment.radius || handMaskRadius) * scale;
                const faPx = forearmRadius * scale;
                const smPx = (activeSegment.smooth != null ? activeSegment.smooth : handSmooth) * scale;
                const sm2Px = handSmooth2 * scale;
                handMask = _buildHandMask(landmarks, radiusPx, faPx, smPx, sm2Px, cw, ch,
                                            armDorsalDilate * scale, armVentralDilate * scale);
            }
            if (handMask) {
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
        {
            const faces = _facesForCurrentSide({ faces: faceDetByLocalFrame.get(localFrame) || [] });
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
            currentFrame >= s.start && currentFrame <= s.end &&
            (!s.side || s.side === curSideLabel || s.side === 'full')
        );
        // HRnet doesn't need MP landmarks — it builds the mask from the
        // MIP heatmap at the per-frame bbox.  Only require MP labels for
        // the MediaPipe source.
        const handProtectActive = activeSeg
            && (handMaskSource === 'hrnet' || visibleLandmarks.length > 0);
        const activeProtectRadius = activeSeg ? (activeSeg.radius || handMaskRadius) : handMaskRadius;
        const activeSmooth = activeSeg ? (activeSeg.smooth != null ? activeSeg.smooth : handSmooth) : handSmooth;

        // Build smoothed hand mask (morphological close via blur+threshold).
        // For HRnet source we use the per-frame MIP threshold instead of
        // the MP circle stamping; both produce a same-size canvas mask
        // that the rest of the compositing path consumes unchanged.
        let handMaskCanvas = null;
        if (handProtectActive) {
            if (handMaskSource === 'hrnet') {
                if (_hrnetMaskData && _hrnetMaskTrialIdx === currentTrialIdx) {
                    const frameInTrial = currentFrame - (_hrnetMaskData.start_frame || 0);
                    handMaskCanvas = _buildHandMaskHrnet(
                        frameInTrial, curSideLabel === 'left' ? 'OS' : 'OD',
                        hrnetMaskSmooth * scale, cw, ch,
                    );
                } else {
                    // Kick off the fetch; once it lands, _ensureHrnetMaskData
                    // will return data and the next render() call (via the
                    // .then) picks it up.
                    _ensureHrnetMaskData().then(d => { if (d) render(); });
                }
            } else {
                handMaskCanvas = _buildHandMask(
                    visibleLandmarks, activeProtectRadius * scale, forearmRadius * scale,
                    activeSmooth * scale, handSmooth2 * scale, cw, ch,
                    armDorsalDilate * scale, armVentralDilate * scale
                );
            }
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

        // Draw hand protection outline from the smoothed mask.
        // Strategy: extract the BOUNDARY pixels of the same
        // ``handMaskCanvas`` that's already being used to cut hands
        // out of the blur fills (see _renderPreviewMask too).  This
        // guarantees the green outline traces the exact contour of
        // the actual protected region -- previously the outline used
        // a separately-built "unified" mask whose smoothing was
        // applied to BOTH hand and arm, so the green ring sat ~4-8 px
        // outside the actual blur-cutout edge.  Now both ride the
        // same mask and the local + remote render's
        // ``_build_hand_mask_from_landmarks`` (hand-only smoothing +
        // arm triangle with optional dorsal/ventral edge dilation)
        // is what the user sees previewed in the canvas overlay.
        if (handMaskCanvas) {
            const ring = _alphaEdgeRing(handMaskCanvas, 76, 175, 80, 0.85);
            ctx.drawImage(ring, 0, 0);
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
        // Don't handle clicks after a pan/drag
        if (panning || didDrag) { didDrag = false; return; }
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
        {
            // Filter to faces actually visible in the current camera --
            // unfiltered faces from the other side can overlap in
            // per-half pixel coords and steal the click.
            const _faceList = _facesForCurrentSide({
                faces: faceDetByLocalFrame.get(localFrame) || []
            });
            if (_faceList.length > 0) {
                const curSide = _sideLabel();
                for (const f of _faceList) {
                    if (ix >= f.x1 && ix <= f.x2 && iy >= f.y1 && iy <= f.y2) {
                        const fcx = (f.x1 + f.x2) / 2;
                        const fcy = (f.y1 + f.y2) / 2;
                        // Use the face's own side when available, falling
                        // back to the current camera (so clicks always
                        // create a spot tied to the camera the user is
                        // looking at).
                        const fSide = f.side || curSide;

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
                            _updateShapeToggle();
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
                        _updateShapeToggle();
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
            const dy = iy - pos.y;
            // Use the larger dimension for hit testing
            const hitR = Math.max(spot.width || spot.radius, spot.height || spot.radius) / 2;
            if (Math.sqrt(dx * dx + dy * dy) <= hitR) {
                selectedSpotId = spot.id;
                renderSpotList();
                _updateShapeToggle();
                updateSpotControls();
                render();
                renderTimeline();
                return;
            }
        }

        // In original view + addingCustom mode, clicking empty canvas creates a custom spot
        if (viewMode === 'original' && addingCustom) {
            const spot = {
                id: nextSpotId++,
                spot_type: 'custom',
                shape: _pendingCustomShape || 'oval',
                x: Math.round(ix),
                y: Math.round(iy),
                radius: 40,
                width: 80,
                height: 80,
                offset_x: 0,
                offset_y: 0,
                frame_start: trialMeta.start_frame,
                frame_end: trialMeta.end_frame,
                side: _sideLabel(),
            };
            blurSpots.push(spot);
            selectedSpotId = spot.id;
            addingCustom = false;
            const customBtn = document.getElementById('addCustomBtn');
            if (customBtn) { customBtn.style.background = ''; customBtn.style.color = ''; }
            renderSpotList();
            _updateShapeToggle();
            updateSpotControls();
            scheduleSave();
            render();
            renderTimeline();
            return;
        }

        // Click on nothing → deselect
        selectedSpotId = null;
        renderSpotList();
        _updateShapeToggle();
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
        didDrag = false;
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
        // Movement during mousedown beyond threshold = suppress click
        if (e.buttons > 0 && panStart) {
            const dx = Math.abs(e.clientX - panStart.x);
            const dy = Math.abs(e.clientY - panStart.y);
            if (dx > 4 || dy > 4) didDrag = true;
        }

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
            didDrag = true;  // suppress the click event that follows
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
        if (playing) {
            if (videoPlaying && videoEl) {
                // Update video rate live — no restart needed
                videoEl.playbackRate = val;
            } else {
                // Fallback timer interval depends on speed — restart
                togglePlay();
                togglePlay();
            }
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
            // Restart from beginning if at the end
            if (currentFrame >= trialMeta.end_frame) {
                currentFrame = trialMeta.start_frame;
                document.getElementById('frameDisplay').textContent = currentFrame;
            }
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
        videoEl.playbackRate = playbackSpeed;
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
        }, Math.round(1000 / (fps * playbackSpeed)));
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
            _rebuildFaceDetMap();
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
            // Discard results if the user navigated to a different trial while detection ran
            if (currentTrialIdx !== trialIdx) return;
            faceDetections = result.faces || [];
            _rebuildFaceDetMap();
            const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
            document.getElementById('faceDetStatus').textContent =
                `${nFaces} frames with faces`;
            document.getElementById('detectFacesBtn').style.display = 'block';
            document.getElementById('detectFacesBtn').textContent = 'Re-detect Faces';

            _autoCreateFaceSpots();
            renderSpotList();
            // Only save if new spots were created (autoCreate skips if spots exist)
            if (blurSpots.some(s => s.spot_type === 'face')) scheduleSave();
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

        // If the trial has any saved blur spots the user has been here before — don't auto-create
        if (blurSpots.length > 0) return;

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
                    radius: Math.round(Math.max(c.w, c.h) / 4),
                    width: Math.round(c.w / 2),
                    height: Math.round(c.h / 2),
                    offset_x: 0,
                    offset_y: 0,
                    frame_start: trialMeta.start_frame,
                    frame_end: trialMeta.end_frame,
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
            list.innerHTML = '';
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
        // Show the toggle whenever we're actively building / editing a
        // custom spot: either staging mode (+Custom Spot pressed) or an
        // existing custom spot is selected.  Hide otherwise (no spot
        // selected, or a face spot selected).
        if (addingCustom) {
            toggle.style.display = 'block';
            btn.textContent = `Shape: ${_pendingCustomShape === 'rect' ? 'Rectangle' : 'Oval'}`;
        } else if (spot && spot.spot_type === 'custom') {
            toggle.style.display = 'block';
            const shape = spot.shape || 'oval';
            btn.textContent = `Shape: ${shape === 'rect' ? 'Rectangle' : 'Oval'}`;
        } else {
            toggle.style.display = 'none';
        }
    }

    function toggleSpotShape() {
        // Staging mode -- flip the shape used for the next-created spot.
        if (addingCustom) {
            _pendingCustomShape = _pendingCustomShape === 'oval' ? 'rect' : 'oval';
            _updateShapeToggle();
            render();
            return;
        }
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
        _updateShapeToggle();
        render();
    }

    // ── Copy custom + face spots from other camera ──
    function copyFromOtherCamera() {
        if (cameraMode === 'single' || cameraNames.length < 2) return;
        _pushUndo();

        const curSide = _sideLabel();
        const otherSide = curSide === 'left' ? 'right' : 'left';

        // ── Custom spots (copy absolute coords) ──
        const otherCustom = blurSpots.filter(s =>
            s.spot_type === 'custom' && s.side === otherSide
        );
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

        // ── Face spots (relative to the per-camera face detection) ──
        // Only meaningful when there's exactly ONE face detected per
        // camera -- otherwise we can't unambiguously pair them up.
        const otherFace = blurSpots.filter(s =>
            s.spot_type === 'face' && s.side === otherSide
        );
        let copiedFaces = 0;
        if (otherFace.length > 0) {
            // Find a representative face-detection centroid for each
            // camera (use the first frame that has a face in each).
            let curCentroid = null, otherCentroid = null;
            let curW = 0, curH = 0, otherW = 0, otherH = 0;
            for (const entry of faceDetections) {
                for (const f of (entry.faces || [])) {
                    const sideOf = f.side || 'full';
                    if (sideOf === curSide && !curCentroid) {
                        curCentroid = { x: (f.x1 + f.x2) / 2, y: (f.y1 + f.y2) / 2 };
                        curW = f.x2 - f.x1; curH = f.y2 - f.y1;
                    } else if (sideOf === otherSide && !otherCentroid) {
                        otherCentroid = { x: (f.x1 + f.x2) / 2, y: (f.y1 + f.y2) / 2 };
                        otherW = f.x2 - f.x1; otherH = f.y2 - f.y1;
                    }
                    if (curCentroid && otherCentroid) break;
                }
                if (curCentroid && otherCentroid) break;
            }
            if (curCentroid && otherCentroid && curW > 0 && otherW > 0) {
                const sx = curW / otherW;
                const sy = curH / otherH;
                for (const s of otherFace) {
                    // Skip if a face spot already exists on this side
                    // near the current-camera face centroid.
                    const exists = blurSpots.find(b =>
                        b.spot_type === 'face' && b.side === curSide &&
                        Math.abs(b.x - curCentroid.x) < Math.max(20, curW * 0.25) &&
                        Math.abs(b.y - curCentroid.y) < Math.max(20, curH * 0.25)
                    );
                    if (exists) continue;
                    // Convert other-camera spot to relative coords on its
                    // face detection, then re-apply on current camera's.
                    const relDx = s.x - otherCentroid.x;
                    const relDy = s.y - otherCentroid.y;
                    const newX = curCentroid.x + relDx * sx;
                    const newY = curCentroid.y + relDy * sy;
                    blurSpots.push({
                        id: nextSpotId++,
                        spot_type: 'face',
                        shape: s.shape || 'oval',
                        x: Math.round(newX),
                        y: Math.round(newY),
                        radius: Math.round((s.radius || 0) * Math.max(sx, sy)),
                        width:  Math.round((s.width  || s.radius * 2) * sx),
                        height: Math.round((s.height || s.radius * 2) * sy),
                        offset_x: Math.round((s.offset_x || 0) * sx),
                        offset_y: Math.round((s.offset_y || 0) * sy),
                        frame_start: s.frame_start,
                        frame_end: s.frame_end,
                        side: curSide,
                    });
                    copiedFaces++;
                }
            }
        }

        const totalCopied = copied + copiedFaces;
        if (totalCopied > 0) {
            renderSpotList();
            scheduleSave();
            render();
            renderTimeline();
            const parts = [];
            if (copied > 0)      parts.push(`${copied} custom`);
            if (copiedFaces > 0) parts.push(`${copiedFaces} face`);
            document.getElementById('renderStatus').textContent =
                `Copied ${parts.join(' + ')} spot(s) from ${otherSide} camera.`;
        } else {
            document.getElementById('renderStatus').textContent =
                'Nothing new to copy from the other camera.';
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
        await saveHandSettings();
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
            // Disable wrist extent slider if no pose data (no elbow available)
            const hasPose = res.has_pose || false;
            const extentSlider = document.getElementById('handExtentSlider');
            const extentLabel = document.getElementById('handExtentVal');
            if (extentSlider) {
                extentSlider.disabled = !hasPose;
                extentSlider.style.opacity = hasPose ? '' : '0.4';
            }
            if (extentLabel && !hasPose) {
                extentLabel.textContent = 'n/a';
            }
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
        _scheduleHandSaveRender();
        renderTimeline();
    }

    function updateForearmRadius(val) {
        forearmRadius = parseInt(val);
        document.getElementById('handForearmVal').textContent = forearmRadius;
        _scheduleHandSaveRender();
    }

    function updateForearmExtent(val) {
        forearmExtent = parseFloat(val);
        document.getElementById('handExtentVal').textContent = forearmExtent.toFixed(1);
        _scheduleHandSaveRender();
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
        _scheduleHandSaveRender();
    }

    function updateHandSmooth2(val) {
        // Deprecated -- kept as a no-op for back-compat with any code
        // path that still references it.
        handSmooth2 = parseInt(val) || 0;
    }

    function updateDorsalDilate(val) {
        armDorsalDilate = parseInt(val) || 0;
        const lbl = document.getElementById('handDorsalVal');
        if (lbl) lbl.textContent = armDorsalDilate;
        _scheduleHandSaveRender();
    }

    function updateVentralDilate(val) {
        armVentralDilate = parseInt(val) || 0;
        const lbl = document.getElementById('handVentralVal');
        if (lbl) lbl.textContent = armVentralDilate;
        _scheduleHandSaveRender();
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
                hand_smooth2: 0,
                hand_temporal: 0,
                show_landmarks: handOverlayEnabled,
                mask_source: 'mediapipe',
                arm_dorsal_dilate: armDorsalDilate,
                arm_ventral_dilate: armVentralDilate,
                segments: handProtectSegments.map(s => ({
                    start: s.start, end: s.end, radius: s.radius,
                    smooth: s.smooth != null ? s.smooth : handSmooth,
                    side: s.side || null,
                })),
            });
        } catch (e) {}
        renderTimeline();
    }

    async function deleteSelectedHandSeg() {
        if (!selectedHandSegId) return;
        _pushUndo();
        handProtectSegments = handProtectSegments.filter(s => s.id !== selectedHandSegId);
        selectedHandSegId = null;
        await saveHandSettings();
        render();
        renderTimeline();
    }

    // ── Hand settings save + render (debounced) ──
    let _handSettingsTimer = null;
    function _scheduleHandSaveRender() {
        if (_handSettingsTimer) clearTimeout(_handSettingsTimer);
        _handSettingsTimer = setTimeout(async () => {
            await saveHandSettings();
            render();
            // When in preview mode, reload the frame so the preview reflects
            // the updated hand settings (server re-renders with new params).
            if (viewMode === 'preview') loadFrame(currentFrame);
        }, 150);
    }

    // ── Save blur specs (immediate) ──
    function scheduleSave() {
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(saveSpecs, 0);
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
        const statusEl = document.getElementById('statusMsg');
        try {
            await API.put(`/api/deidentify/${subjectId}/blur-specs`, {
                trial_idx: currentTrialIdx,
                specs: specs,
            });
            if (statusEl) {
                statusEl.textContent = 'Saved';
                statusEl.style.color = 'var(--green, #4caf50)';
                setTimeout(() => { if (statusEl.textContent === 'Saved') statusEl.textContent = ''; }, 1500);
            }
        } catch (e) {
            console.error('Save blur specs failed:', e);
            if (statusEl) {
                statusEl.textContent = 'Save failed: ' + e.message;
                statusEl.style.color = 'var(--danger, #f44336)';
            }
        }
    }

    // ── Render current trial ──
    async function renderTrial() {
        if (!subjectId || currentTrialIdx < 0) return;

        // Prompt Local/Remote
        const btn = document.getElementById('renderBtn');
        const target = btn ? await _promptTarget(btn) : 'local-cpu';
        if (!target) return;

        // Save current specs first
        await saveSpecs();

        const trialName = trialMeta ? trialMeta.trial_name : `trial ${currentTrialIdx}`;
        const status = document.getElementById('renderStatus');

        // Always submit to the backend queue — it handles concurrency
        _startRender(subjectId, currentTrialIdx, trialName, target);
    }

    async function _startRender(sid, trialIdx, trialName, target = 'local-cpu') {
        const btn = document.getElementById('renderBtn');
        const status = document.getElementById('renderStatus');
        status.textContent = `Submitting ${trialName}...`;

        try {
            const result = await API.post(`/api/deidentify/${sid}/render`, {
                trial_idx: trialIdx,
                execution_target: target,
            });
            let jobId = result.job_id;

            // If job_id not yet available (queue manager hasn't started it), poll
            if (!jobId) {
                status.textContent = `Queued ${trialName}... waiting to start`;
                for (let attempt = 0; attempt < 30 && !jobId; attempt++) {
                    await new Promise(r => setTimeout(r, 1000));
                    try {
                        const check = await API.get(`/api/jobs?status=running&status=pending&subject_id=${sid}&job_type=deidentify`);
                        if (check && check.length > 0) {
                            jobId = check[0].id;
                        }
                    } catch (e) {}
                }
                if (!jobId) {
                    status.textContent = 'Job not started — check Processing page';
                    return;
                }
            }

            status.textContent = `Rendering ${trialName}...`;
            API.streamJob(jobId,
                (data) => {
                    if (data.status === 'running') {
                        const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                        status.textContent = `Rendering ${trialName}... ${pct}%`;
                    }
                },
                (data) => {
                    if (data.status === 'completed') {
                        status.textContent = `Render complete! ${trialName} saved.`;
                        // Update trial button color to green
                        const trialBtns = document.querySelectorAll('.trial-btn');
                        // Find the trial by matching trialIdx across all trials
                        const matchIdx = trials.findIndex(t => t.trial_idx === trialIdx);
                        if (matchIdx >= 0 && trialBtns[matchIdx]) {
                            trialBtns[matchIdx].style.borderColor = 'var(--green)';
                            trialBtns[matchIdx].style.color = 'var(--green)';
                        }
                        if (trialIdx === currentTrialIdx && trialMeta) {
                            trialMeta.has_blurred = true;
                            _updateSidebarState();
                        }
                    } else if (data.status === 'failed') {
                        status.textContent = 'Render failed: ' + (data.error_msg || 'unknown');
                    } else {
                        status.textContent = 'Render ' + data.status;
                    }
                    // Process next queued render
                    _processNextInQueue();
                },
            );
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
        }
    }

    function _processNextInQueue() {
        if (_renderQueue.length === 0) return;
        const next = _renderQueue.shift();
        const status = document.getElementById('renderStatus');
        status.textContent = `Starting queued render: ${next.trialName}...`;
        _startRender(next.subjectId, next.trialIdx, next.trialName, next.target || 'local-cpu');
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
        tlCanvas.addEventListener('dblclick', onTlDblClick);
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
        if (hasHands) {
            const nHandRows = cameraMode === 'single' ? 1 : 2;
            neededH += nHandRows * (handRowH + gap);
        }
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

        // ── Hand rows: one per camera with coverage + draggable protection segments ──
        // Only show the row for the active camera side so the timeline isn't cluttered.
        if (handCoverage.length > 0 || hasMediapipe) {
            const allHandSides = cameraMode === 'single' ? [{ side: 'full', label: 'Hands' }]
                : [{ side: 'left', label: cameraNames[0] || 'OS' },
                   { side: 'right', label: cameraNames[1] || 'OD' }];
            // Map currentSide (camera name) to row side ('left'/'right'/'full')
            const activeSide = cameraMode === 'single' ? 'full'
                : (currentSide === cameraNames[0] ? 'left' : 'right');
            const handSides = allHandSides.filter(s => s.side === activeSide);

            for (const { side: rowSide, label: rowLabel } of handSides) {
                tlCtx.fillStyle = 'rgba(150,150,150,0.5)';
                tlCtx.fillText(rowLabel, 2, y + 12);

                // Background coverage heatmap
                if (handCoverage.length > 0) {
                    tlCtx.fillStyle = 'rgba(76,175,80,0.15)';
                    const pw = Math.max(1, L.barW / L.range);
                    for (const f of handCoverage) {
                        const x = _frameToTlX(f, L);
                        tlCtx.fillRect(x, y, pw, handRowH);
                    }
                }

                // Protection segment bars for this camera side
                const sideSegs = handProtectSegments.filter(s =>
                    !s.side || s.side === rowSide || s.side === 'full');
                for (const seg of sideSegs) {
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

        // Hit test hand protection segments — only the active camera row (matches render)
        if (handCoverage.length > 0 || hasMediapipe) {
            const activeSide = cameraMode === 'single' ? 'full'
                : (currentSide === cameraNames[0] ? 'left' : 'right');
            const handSides = [activeSide];
            for (const rowSide of handSides) {
                const rowSegs = handProtectSegments.filter(s =>
                    !s.side || s.side === rowSide || s.side === 'full');
                for (const seg of rowSegs) {
                    const hx1 = _frameToTlX(seg.start, L);
                    const hx2 = _frameToTlX(seg.end, L);

                    if (my >= y + 2 && my <= y + handRowH - 2) {
                        if (Math.abs(mx - hx1) < 6) return { type: 'hand', seg, edge: 'start' };
                        if (Math.abs(mx - hx2) < 6) return { type: 'hand', seg, edge: 'end' };
                        if (mx > hx1 + 4 && mx < hx2 - 4) return { type: 'hand', seg, edge: 'move' };
                    }
                }
                // If in this camera's hand row but not on a segment → create new for this side
                if (my >= y + 2 && my <= y + handRowH - 2) {
                    return { type: 'hand_empty', edge: 'create', side: rowSide };
                }
                y += handRowH + gap;
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
            // Store which camera row was clicked for the new segment
            tlDragNewHandSide = hit.side || _sideLabel();
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
                    handProtectSegments.push({ id: -1, start: s, end: en, radius: handMaskRadius, smooth: handSmooth, side: tlDragNewHandSide || _sideLabel() });
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

    async function onTlMouseUp(e) {
        if (tlDragSpot) {
            if (tlDragSpot === 'newhand') {
                // Finalize the new segment (replace the preview id=-1)
                const preview = handProtectSegments.find(s => s.id === -1);
                if (preview && preview.end > preview.start) {
                    preview.id = nextHandSegId++;
                    selectedHandSegId = preview.id;
                } else {
                    // Too small — treat as click
                    // If no segments exist for this camera, create a full-trial segment
                    const side = tlDragNewHandSide || _sideLabel();
                    const hasSegsForSide = handProtectSegments.some(s =>
                        s.id !== -1 && (!s.side || s.side === side || s.side === 'full'));
                    handProtectSegments = handProtectSegments.filter(s => s.id !== -1);
                    if (!hasSegsForSide && trialMeta) {
                        const newSeg = {
                            id: nextHandSegId++,
                            start: trialMeta.start_frame,
                            end: trialMeta.end_frame,
                            radius: handMaskRadius,
                            smooth: handSmooth,
                            side: side,
                        };
                        handProtectSegments.push(newSeg);
                        selectedHandSegId = newSeg.id;
                    }
                }
                await saveHandSettings();
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

    function onTlDblClick(e) {
        // Double-click on the current-camera's hand row when no segment
        // exists there → create a full-trial hand-mask segment.  Lets
        // the user "protect the whole trial" with a single gesture.
        const hit = _tlHitTest(e);
        if (!hit || hit.type !== 'hand_empty') return;
        if (!trialMeta) return;
        const sf = trialMeta.start_frame || 0;
        const ef = trialMeta.end_frame != null
            ? trialMeta.end_frame
            : (sf + (trialMeta.frame_count || 1) - 1);
        _pushUndo();
        const side = hit.side || _sideLabel();
        const newSeg = {
            id: nextHandSegId++,
            start: sf, end: ef,
            radius: handMaskRadius, smooth: handSmooth, side,
        };
        handProtectSegments.push(newSeg);
        selectedHandSegId = newSeg.id;
        saveHandSettings();
        renderTimeline();
        render();
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
        updateDorsalDilate,
        updateVentralDilate,
        goToAnalyze,
        renderTrial,
        toggleHasFaces,
        undo,
        redo,
    };

    async function toggleHasFaces(hasFaces) {
        if (!subjectId || currentTrialIdx < 0 || !trialMeta) return;
        const stem = trialMeta.trial_name;
        try {
            // Get current subject data to read existing no_face_videos
            const subjects = await API.get('/api/subjects');
            const subj = subjects.find(s => s.id === subjectId);
            if (!subj) return;
            let noFace = [];
            try { noFace = JSON.parse(subj.no_face_videos || '[]'); } catch {}
            if (!Array.isArray(noFace)) noFace = [];

            if (hasFaces) {
                // Remove from no_face list
                noFace = noFace.filter(s => s !== stem);
            } else {
                // Add to no_face list
                if (!noFace.includes(stem)) noFace.push(stem);
            }
            await API.patch(`/api/subjects/${subjectId}`, { no_face_videos: noFace });
            trialMeta.has_faces = hasFaces;
            // Update trial button color
            const btns = document.querySelectorAll('.trial-btn');
            if (btns[currentTrialIdx]) {
                const color = (!hasFaces || trialMeta.has_blurred) ? 'var(--green)' : '';
                btns[currentTrialIdx].style.borderColor = color;
                btns[currentTrialIdx].style.color = color;
            }
        } catch (e) {
            console.error('Failed to update has_faces:', e);
            // Revert checkbox
            const hfToggle = document.getElementById('hasFacesToggle');
            if (hfToggle) hfToggle.checked = !hasFaces;
        }
    }
})();
