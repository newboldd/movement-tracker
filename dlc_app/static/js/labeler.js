/* Canvas labeling engine for DLC keypoint annotation
   with MediaPipe ghost markers, click-to-accept, and 3D distance trace */

const labeler = (() => {
    // ── State ──────────────────────────────────────────
    let sessionId = null;
    let sessionInfo = null;
    let trials = [];
    let totalFrames = 0;

    // Dynamic from settings
    let bodyparts = ['thumb', 'index'];
    let cameraNames = ['OS', 'OD'];

    let currentFrame = 0;
    let currentSide = 'OS';
    let playing = false;
    let playTimer = null;
    let playbackRate = 1;

    // Labels: Map<frameKey, {[bodypart]: [x, y]}>
    // frameKey = `${frame}_${side}`
    const labels = new Map();

    // MediaPipe prelabels: {OS: {thumb: [...], index: [...]}, OD: {...}, distances: [...]}
    let mpLabels = null;

    // DLC analysis predictions: same shape as mpLabels
    let dlcLabels = null;

    // Committed manual labels from prior sessions: Map<frameKey, {[bodypart]: [x, y]}>
    // These take priority over DLC predictions for frames the user already hand-labeled
    let committedLabels = new Map();

    // 3D distance trace
    let distances = null;

    // True for refine sessions: ghost priority = committed > DLC > MP, distances from DLC
    let isRefine = false;

    // Committed frame count from DLC labeled-data/
    let committedFrameCount = 0;

    // Corrections mode state
    let isCorrections = false;
    // Final (read-only) mode state
    let isFinal = false;
    let availableStages = [];
    const stageData = {};             // cache: {stage: {camera: {bodypart: [...]}}}
    let stageFiles = {};              // {stage: [csv_filename, ...]}
    let selectedStage = 'auto';       // 'auto' or specific stage name

    const STAGE_CHAIN = ['corrections', 'refine', 'dlc', 'labels', 'mp'];

    // Subject navigation
    let allSubjects = [];
    let currentSubjectId = null;

    // Color palette for bodyparts
    const COLORS = [
        '#ff4444', '#222222', '#4a9eff', '#4caf50',
        '#ff9800', '#9c27b0', '#00bcd4', '#e91e63', '#795548',
    ];

    function bpColor(idx) { return COLORS[idx % COLORS.length]; }
    function bpLetter(name) { return name[0].toUpperCase(); }

    // Final mode: per-camera crop boxes {cam: {x1, y1, x2, y2}}
    let finalCropBoxes = null;

    // Canvas state
    let canvas, ctx;
    let timeline, tlCtx;
    let distCanvas, distCtx;
    // User-defined Y max for distance plots (null = auto)
    let userYMax = null;
    // Distance trace viewport (zoomed ~10s window)
    let distViewStart = 0;      // first visible frame
    let distViewFrames = 0;     // number of frames in visible window (set from fps)
    let distDragging = false;
    let distDragStartX = 0;
    let distDragStartView = 0;
    let distAutoScroll = true; // false while user is manually panning the trace
    let containerEl;
    let currentImage = null;
    let imgW = 0, imgH = 0;

    // Zoom/pan
    let scale = 1;
    let offsetX = 0, offsetY = 0;

    // Drag state
    let dragging = null; // bodypart name | 'pan' | 'pending'
    let dragStartX = 0, dragStartY = 0;
    let dragOrigX = 0, dragOrigY = 0;
    let didDrag = false; // true once mouse moves past threshold during 'pending'
    const DRAG_THRESHOLD = 4; // pixels before a click becomes a pan drag

    // Camera shift computed from paired OS/OD labels (image pixels)
    let computedCameraShiftX = null; // horizontal, or null = use default
    let computedCameraShiftY = null; // vertical, or null = no shift

    // Undo stack: each entry = { key, bp, prev (coords or null) }
    const undoStack = [];
    const MAX_UNDO = 50;

    // Review mode: null = all bodyparts, or bodypart name for focused review
    let reviewBp = null;

    // Video element for smooth playback
    let videoEl = null;
    let videoPlaying = false;
    let currentTrialIdx = -1; // which trial the video element is loaded with

    // Deleted frame/side keys — sent to server on save so DB stays in sync
    const deletedKeys = new Set();
    // Dirty (modified since last save) keys — only these are sent to server
    const dirtyKeys = new Set();
    // Rejected stage labels: Set of "frame_side_bp" — suppresses stage label display
    // so next-priority ghost can appear after user deletes a correction
    const rejectedStageLabels = new Set();

    // V2 training exclusions (refine mode): Set of `${frame}_${side}` keys.
    // All labeled frames are included by default; adding a key here excludes it
    // from DLC training while still saving it to the corrections CSV on commit.
    const v2Excludes = new Set();

    // Frames that differ between corrections and DLC stage data (refine mode).
    // These are pre-existing manual corrections and are shown as green dots.
    // Populated by computeCorrectionFrames() after stage data loads.
    let correctionFrames = new Set();

    // Prefetch cache
    const imageCache = new Map();
    const PREFETCH_AHEAD = 3;

    // Point detection radius
    const HIT_RADIUS = 12;
    const POINT_RADIUS = 6;

    // ── Init ──────────────────────────────────────────
    function init() {
        const params = new URLSearchParams(window.location.search);
        sessionId = parseInt(params.get('session'));
        if (!sessionId) {
            alert('No session ID in URL. Go to Dashboard to start labeling.');
            return;
        }

        canvas = document.getElementById('labelCanvas');
        ctx = canvas.getContext('2d');
        timeline = document.getElementById('timelineCanvas');
        tlCtx = timeline.getContext('2d');
        distCanvas = document.getElementById('distanceTraceCanvas');
        distCtx = distCanvas ? distCanvas.getContext('2d') : null;
        containerEl = document.getElementById('canvasContainer');
        videoEl = document.getElementById('videoPlayer');

        setupCanvasEvents();
        setupTimeline();
        if (distCanvas) setupDistanceTrace();

        // Y-max input
        const ymaxInput = document.getElementById('ymaxInput');
        if (ymaxInput) {
            ymaxInput.addEventListener('change', () => {
                const val = parseFloat(ymaxInput.value);
                userYMax = (isFinite(val) && val > 0) ? val : null;
                renderDistanceTrace();
                renderTrialPlots();
            });
            ymaxInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') ymaxInput.blur();
            });
        }

        loadSession();
    }

    async function loadSession() {
        try {
            sessionInfo = await API.get(`/api/labeling/sessions/${sessionId}/info`);
            trials = sessionInfo.trials;
            totalFrames = sessionInfo.total_frames;

            // Get dynamic bodyparts and camera names from session info
            if (sessionInfo.bodyparts) bodyparts = sessionInfo.bodyparts;
            if (sessionInfo.camera_names) cameraNames = sessionInfo.camera_names;
            if (sessionInfo.committed_frame_count) committedFrameCount = sessionInfo.committed_frame_count;
            currentSide = cameraNames[0] || 'OS';

            isRefine = sessionInfo.session && sessionInfo.session.session_type === 'refine';
            isCorrections = sessionInfo.session && sessionInfo.session.session_type === 'corrections';
            isFinal = sessionInfo.session && sessionInfo.session.session_type === 'final';

            // Populate subject navigation dropdown
            currentSubjectId = sessionInfo.subject.id;
            const typeLabel = document.getElementById('sessionTypeLabel');
            if (isFinal) typeLabel.textContent = 'Final:';
            else if (isCorrections) typeLabel.textContent = 'Corrections:';
            else if (isRefine) typeLabel.textContent = 'Refine:';
            else typeLabel.textContent = 'Labeling:';

            try {
                allSubjects = await API.get('/api/subjects');
                const sel = document.getElementById('subjectSelect');
                sel.innerHTML = '';
                allSubjects.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s.id;
                    opt.textContent = s.name;
                    if (s.id === currentSubjectId) opt.selected = true;
                    sel.appendChild(opt);
                });
                sel.addEventListener('change', () => switchSubject(parseInt(sel.value)));
                updateSubjectNavButtons();
            } catch (e) {
                console.log('Could not load subjects list for navigation');
            }

            // Update commit buttons for mode
            const mainCommitBtn = document.getElementById('mainCommitBtn');
            const saveCorrectionsBtn = document.getElementById('saveCorrectionsBtn');
            const commitDlcBtn = document.getElementById('commitDlcBtn');
            if (isFinal) {
                if (mainCommitBtn) mainCommitBtn.style.display = 'none';
            } else if (isRefine) {
                if (mainCommitBtn) mainCommitBtn.style.display = 'none';
                if (saveCorrectionsBtn) saveCorrectionsBtn.style.display = '';
                if (commitDlcBtn) commitDlcBtn.style.display = '';
            } else if (isCorrections) {
                if (mainCommitBtn) mainCommitBtn.textContent = 'Save Corrections';
            }

            // Update sidebar with dynamic shortcuts
            updateShortcutsSidebar();
            document.getElementById('sideToggle').textContent = currentSide;

            // Setup keyboard after bodyparts are known
            setupKeyboard();

            // Load existing labels (user edits in this session) — skip for final mode
            if (!isFinal) {
                const saved = await API.get(`/api/labeling/sessions/${sessionId}/labels`);
                saved.forEach(l => {
                    const key = `${l.frame_num}_${l.side}`;
                    labels.set(key, l.keypoints || {});
                });
            }

            if (isCorrections || isFinal || isRefine) {
                // Corrections / Final / Refine mode: load stage data
                try {
                    const stagesResp = await API.get(`/api/labeling/sessions/${sessionId}/available_stages`);
                    availableStages = stagesResp.stages || [];
                    stageFiles = stagesResp.stage_files || {};
                } catch (e) {
                    console.log('Could not load available stages');
                    availableStages = [];
                    stageFiles = {};
                }

                // Default to corrections stage if available
                if (availableStages.includes('corrections')) {
                    selectedStage = 'corrections';
                }

                // Load all stage data and merge distances
                await loadAllStages();
                if (isRefine) computeCorrectionFrames();
                populateStageSelector();

                // Final mode: hide timeline/distance trace, build trial plots at bottom
                if (isFinal) {
                    const timelineContainer = document.querySelector('.timeline-container');
                    if (timelineContainer) timelineContainer.style.display = 'none';
                    buildTrialPlots();
                }
            } else {
                // Initial mode: load MP + DLC + committed labels as ghosts
                try {
                    const mpData = await API.get(`/api/labeling/sessions/${sessionId}/mediapipe`);
                    if (mpData && Object.keys(mpData).length > 0) {
                        mpLabels = mpData;
                        distances = mpData.distances || null;
                    }
                } catch (e) {
                    console.log('No MediaPipe prelabels available');
                }

                try {
                    const dlcData = await API.get(`/api/labeling/sessions/${sessionId}/dlc_predictions`);
                    if (dlcData && Object.keys(dlcData).length > 0) {
                        dlcLabels = dlcData;
                        if (isRefine && dlcData.distances) {
                            distances = dlcData.distances;
                        }
                    }
                } catch (e) {
                    console.log('No DLC predictions available');
                }

                if (isRefine) {
                    try {
                        const committed = await API.get(`/api/labeling/sessions/${sessionId}/committed_labels`);
                        if (committed && committed.length > 0) {
                            committed.forEach(l => {
                                const key = `${l.frame_num}_${l.side}`;
                                if (!committedLabels.has(key)) {
                                    committedLabels.set(key, l.keypoints || {});
                                }
                            });
                            console.log(`Loaded ${committedLabels.size} committed manual labels`);
                        }
                    } catch (e) {
                        console.log('No committed labels available');
                    }
                }
            }

            // Always initialize the distance trace window size from fps
            initDistanceTraceWindow();

            // Show distance trace if we have data; hide timeline to save space
            // (in final mode: trial plots handle this instead)
            if (distances && distances.some(d => d !== null) && !isFinal) {
                const traceContainer = document.getElementById('distanceTraceContainer');
                if (traceContainer) traceContainer.style.display = 'block';
                const timelineContainer = document.querySelector('.timeline-container');
                if (timelineContainer) timelineContainer.style.display = 'none';
                const ymaxContainer = document.getElementById('ymaxContainer');
                if (ymaxContainer) ymaxContainer.style.display = 'flex';
            }

            recomputeCameraShift();
            updateLabelCount();
            updateLabelNavButtons();
            goToFrame(0);

        } catch (e) {
            alert('Error loading session: ' + e.message);
        }
    }

    // ── Frame loading ─────────────────────────────────
    function frameUrl(frame, side) {
        return `/api/labeling/sessions/${sessionId}/frame?n=${frame}&side=${side}`;
    }

    function loadImage(frame, side) {
        return new Promise((resolve, reject) => {
            const key = `${frame}_${side}`;
            if (imageCache.has(key)) {
                resolve(imageCache.get(key));
                return;
            }
            const img = new Image();
            img.onload = () => {
                imageCache.set(key, img);
                // Evict old entries
                if (imageCache.size > 30) {
                    const first = imageCache.keys().next().value;
                    imageCache.delete(first);
                }
                resolve(img);
            };
            img.onerror = reject;
            img.src = frameUrl(frame, side);
        });
    }

    function prefetchFrames(frame) {
        for (let i = 1; i <= PREFETCH_AHEAD; i++) {
            const f = frame + i;
            if (f < totalFrames) {
                loadImage(f, currentSide);
            }
        }
    }

    let hasUserZoom = false; // true once user has zoomed/panned

    async function goToFrame(frame) {
        if (frame < 0 || frame >= totalFrames) return;
        currentFrame = frame;
        distAutoScroll = true; // frame navigation re-enables auto-scroll

        try {
            currentImage = await loadImage(frame, currentSide);
            imgW = currentImage.width;
            imgH = currentImage.height;
            if (!hasUserZoom) {
                if (isFinal && finalCropBoxes) {
                    // Final mode: zoom to crop box, then lock
                    zoomToCropBox();
                    hasUserZoom = true;
                } else if (isFinal) {
                    fitImage();
                    hasUserZoom = true;
                } else {
                    if (!autoZoomForFrame(frame, currentSide)) fitImage();
                }
            }
            render();
            prefetchFrames(frame);
        } catch (e) {
            console.error('Failed to load frame', frame, e);
        }

        updateFrameDisplay();
        renderTimeline();
        renderDistanceTrace();
        if (isRefine) updateV2TrainingBtn();
    }

    function fitImage() {
        if (!imgW || !imgH) return;
        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        scale = Math.min(cw / imgW, ch / imgH);
        offsetX = (cw - imgW * scale) / 2;
        offsetY = (ch - imgH * scale) / 2;
    }

    // ── MediaPipe ghost helpers ───────────────────────
    function getMpLabel(frame, side, bodypart) {
        if (!mpLabels) return null;
        const camData = mpLabels[side];
        if (!camData) return null;
        const arr = camData[bodypart];
        if (!arr || frame >= arr.length) return null;
        return arr[frame]; // [x, y] or null
    }

    function getDlcLabel(frame, side, bodypart) {
        if (!dlcLabels) return null;
        const camData = dlcLabels[side];
        if (!camData) return null;
        const arr = camData[bodypart];
        if (!arr || frame >= arr.length) return null;
        return arr[frame]; // [x, y] or null
    }

    function getCommittedLabel(frame, side, bodypart) {
        const key = `${frame}_${side}`;
        const lbl = committedLabels.get(key);
        if (!lbl) return null;
        const coords = lbl[bodypart];
        if (!coords || coords[0] == null) return null;
        return coords; // [x, y]
    }

    function hasManualLabel(frame, side, bodypart) {
        const key = `${frame}_${side}`;
        const lbl = labels.get(key);
        if (!lbl) return false;
        const coords = lbl[bodypart];
        return coords && coords[0] != null;
    }

    // ── Corrections mode: stage selector + fallback ───
    async function ensureStageLoaded(stage) {
        if (stageData[stage] !== undefined) return;
        try {
            const data = await API.get(`/api/labeling/sessions/${sessionId}/stage_data?stage=${stage}`);
            stageData[stage] = (data && Object.keys(data).length > 0) ? data : null;
        } catch (e) {
            console.log(`Failed to load stage ${stage}:`, e);
            stageData[stage] = null;
        }
    }

    async function loadAllStages() {
        const loadPromises = availableStages.map(s => ensureStageLoaded(s));
        await Promise.all(loadPromises);
        computeMergedDistances();
        if (isFinal) computeFinalCropBoxes();
    }

    function computeFinalCropBoxes() {
        /** Compute a stable crop box per camera from stage data label positions. */
        const MARGIN = 40;
        const boxes = {};
        for (const cam of cameraNames) {
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            // Scan the selected stage (or all via priority chain)
            const stagesToUse = (selectedStage !== 'auto')
                ? [selectedStage]
                : STAGE_CHAIN.filter(s => availableStages.includes(s));

            for (const stage of stagesToUse) {
                const sd = stageData[stage];
                if (!sd || !sd[cam]) continue;
                for (const bp of bodyparts) {
                    const arr = sd[cam][bp];
                    if (!arr) continue;
                    for (let f = 0; f < arr.length; f++) {
                        const pt = arr[f];
                        if (!pt) continue;
                        if (pt[0] < minX) minX = pt[0];
                        if (pt[1] < minY) minY = pt[1];
                        if (pt[0] > maxX) maxX = pt[0];
                        if (pt[1] > maxY) maxY = pt[1];
                    }
                }
                if (minX < Infinity) break; // got data from this stage, stop
            }

            if (minX < Infinity) {
                boxes[cam] = {
                    x1: Math.max(0, Math.floor(minX - MARGIN)),
                    y1: Math.max(0, Math.floor(minY - MARGIN)),
                    x2: Math.ceil(maxX + MARGIN),
                    y2: Math.ceil(maxY + MARGIN),
                };
            }
        }
        finalCropBoxes = Object.keys(boxes).length > 0 ? boxes : null;
    }

    function populateStageSelector() {
        const container = document.getElementById('stageSelectorContainer');
        const select = document.getElementById('stageSelector');
        const csvList = document.getElementById('stageCsvList');
        if (!container || !select) return;
        if ((!isFinal && !isRefine) || availableStages.length === 0) return;

        container.style.display = 'block';
        select.innerHTML = '<option value="auto">Auto (priority merge)</option>';

        // Add available CSV-based stages
        for (const stage of STAGE_CHAIN) {
            if (!availableStages.includes(stage)) continue;
            const files = stageFiles[stage];
            const label = files
                ? `${stage} (${files.length} csv${files.length > 1 ? 's' : ''})`
                : stage;
            const opt = document.createElement('option');
            opt.value = stage;
            opt.textContent = label;
            select.appendChild(opt);
        }

        function updateCsvList() {
            if (!csvList) return;
            const stage = select.value;
            const files = stageFiles[stage];
            if (files && files.length > 0) {
                csvList.textContent = files.join(', ');
            } else {
                csvList.textContent = '';
            }
        }

        select.addEventListener('change', () => {
            selectedStage = select.value;
            select.blur();
            updateCsvList();
            updateLabelNavButtons();
            computeMergedDistances();
            if (isFinal) computeFinalCropBoxes();
            render();
            if (isFinal) renderTrialPlots();
            else renderDistanceTrace();
        });

        // Sync dropdown to selectedStage (set before loadAllStages)
        if (selectedStage !== 'auto') {
            select.value = selectedStage;
        }

        updateCsvList();
    }

    function computeCorrectionFrames() {
        /** In refine mode: find all frame+side combos where corrections stage
         *  coords differ from DLC stage coords by >= CORR_THRESHOLD pixels.
         *  These represent pre-existing manual corrections and are shown as
         *  green dots on the distance trace.
         */
        correctionFrames = new Set();
        if (!isRefine) return;

        const corrData = stageData['corrections'];
        const dlcData = stageData['dlc'];
        if (!corrData || !dlcData) return;

        const CORR_THRESHOLD = 3.0; // pixels — below this = rounding / noise

        for (const cam of cameraNames) {
            if (!corrData[cam] || !dlcData[cam]) continue;
            for (const bp of bodyparts) {
                const corrArr = corrData[cam][bp];
                const dlcArr = dlcData[cam][bp];
                if (!corrArr || !dlcArr) continue;
                const n = Math.min(corrArr.length, dlcArr.length);
                for (let f = 0; f < n; f++) {
                    const c = corrArr[f];
                    const d = dlcArr[f];
                    if (!c || !d) continue;
                    if (Math.hypot(c[0] - d[0], c[1] - d[1]) >= CORR_THRESHOLD) {
                        correctionFrames.add(`${f}_${cam}`);
                    }
                }
            }
        }
        console.log(`[refine] ${correctionFrames.size} correction frames found`);
    }

    function computeMergedDistances() {
        /** Build a single merged distance array from the selected stage or
         *  priority chain.  Gaps in the selected stage stay as gaps. */
        distances = null;

        const stagesToUse = (selectedStage !== 'auto')
            ? [selectedStage]
            : STAGE_CHAIN.filter(s => availableStages.includes(s));

        // Find total frames from any stage that has distances
        let nFrames = 0;
        for (const s of stagesToUse) {
            const sd = stageData[s];
            if (sd && sd.distances) {
                nFrames = Math.max(nFrames, sd.distances.length);
            }
        }
        if (nFrames === 0) return;

        const merged = new Array(nFrames).fill(null);
        for (let f = 0; f < nFrames; f++) {
            for (const s of stagesToUse) {
                const sd = stageData[s];
                if (!sd || !sd.distances) continue;
                const d = sd.distances[f];
                if (d !== null && d !== undefined) {
                    merged[f] = d;
                    break;
                }
            }
        }

        if (merged.some(d => d !== null)) {
            distances = merged;
            const ymaxContainer = document.getElementById('ymaxContainer');
            if (ymaxContainer) ymaxContainer.style.display = 'flex';
            if (!isFinal) {
                const traceContainer = document.getElementById('distanceTraceContainer');
                if (traceContainer) traceContainer.style.display = 'block';
                const timelineContainer = document.querySelector('.timeline-container');
                if (timelineContainer) timelineContainer.style.display = 'none';
            } else {
                renderTrialPlots();
            }
        }
    }

    function getMergedLabel(frame, side, bodypart) {
        /** Look up label from auto priority merge (all stages, highest priority first).
         *  Respects rejections so deleted labels fall through to next priority. */
        const stagesToUse = STAGE_CHAIN.filter(s => availableStages.includes(s));
        const rejKey = `${frame}_${side}_${bodypart}`;
        for (const stage of stagesToUse) {
            if (rejectedStageLabels.has(`${rejKey}_${stage}`)) continue;
            const sd = stageData[stage];
            if (!sd || !sd[side]) continue;
            const arr = sd[side][bodypart];
            if (arr && frame < arr.length && arr[frame] != null) {
                return arr[frame];
            }
        }
        return null;
    }

    function isGapFrame(frame, side) {
        /** True if any bodypart is missing a label on this frame — either
         *  no stage data, or the stage label was rejected and a ghost is showing. */

        // If session has manual labels for all bodyparts, not a gap
        const key = `${frame}_${side}`;
        const lbl = labels.get(key);

        for (const bp of bodyparts) {
            const hasManual = lbl && lbl[bp] && lbl[bp][0] != null;
            if (hasManual) continue;

            const stageCoords = getStageLabel(frame, side, bp);
            if (!stageCoords) return true; // missing or rejected — it's a gap
        }
        return false;
    }

    function getStageLabel(frame, side, bodypart) {
        /** Look up label from selected stage or all stages (highest priority first).
         *  Skips stages whose label was rejected by the user for this frame/side/bp. */
        if (!isCorrections && !isFinal && !isRefine) return null;

        const stagesToUse = (selectedStage !== 'auto')
            ? [selectedStage]
            : STAGE_CHAIN.filter(s => availableStages.includes(s));

        const rejKey = `${frame}_${side}_${bodypart}`;
        for (const stage of stagesToUse) {
            // Skip this stage if user rejected its label
            if (rejectedStageLabels.has(`${rejKey}_${stage}`)) continue;
            const sd = stageData[stage];
            if (!sd || !sd[side]) continue;
            const arr = sd[side][bodypart];
            if (arr && frame < arr.length && arr[frame] != null) {
                return arr[frame];
            }
        }
        return null;
    }

    function getStageLabelSource(frame, side, bodypart) {
        /** Like getStageLabel but returns {coords, stage} or null. */
        if (!isCorrections && !isFinal && !isRefine) return null;

        const stagesToUse = (selectedStage !== 'auto')
            ? [selectedStage]
            : STAGE_CHAIN.filter(s => availableStages.includes(s));

        const rejKey = `${frame}_${side}_${bodypart}`;
        for (const stage of stagesToUse) {
            if (rejectedStageLabels.has(`${rejKey}_${stage}`)) continue;
            const sd = stageData[stage];
            if (!sd || !sd[side]) continue;
            const arr = sd[side][bodypart];
            if (arr && frame < arr.length && arr[frame] != null) {
                return { coords: arr[frame], stage };
            }
        }
        return null;
    }

    // ── Rendering ─────────────────────────────────────
    function render() {
        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        canvas.width = cw;
        canvas.height = ch;

        ctx.clearRect(0, 0, cw, ch);

        if (currentImage) {
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            ctx.drawImage(currentImage, 0, 0);
            ctx.restore();

        }

        drawLabelsOverlay();
    }

    function zoomToCropBox() {
        /** Zoom to the crop box for the current camera. */
        const box = finalCropBoxes ? finalCropBoxes[currentSide] : null;
        if (!box) { fitImage(); return; }

        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        const bw = box.x2 - box.x1;
        const bh = box.y2 - box.y1;
        scale = Math.min(cw / bw, ch / bh);
        offsetX = (cw - bw * scale) / 2 - box.x1 * scale;
        offsetY = (ch - bh * scale) / 2 - box.y1 * scale;
    }

    /** Draw labels for currentFrame/currentSide — used by both render() and videoDrawLoop(). */
    function drawLabelsOverlay() {
        const key = `${currentFrame}_${currentSide}`;
        const lbl = labels.get(key);
        const placedBps = [];

        bodyparts.forEach((bp, idx) => {
            const manualCoords = lbl ? lbl[bp] : null;
            const hasManual = manualCoords && manualCoords[0] != null && manualCoords[1] != null;

            if (hasManual) {
                drawPoint(manualCoords[0], manualCoords[1], bpColor(idx), bpLetter(bp));
                placedBps.push({ bp, x: manualCoords[0], y: manualCoords[1] });
            } else if (isFinal) {
                // Final mode: only show stage-sourced labels (read-only)
                const stageCoords = getStageLabel(currentFrame, currentSide, bp);
                if (stageCoords) {
                    drawPoint(stageCoords[0], stageCoords[1], bpColor(idx), bpLetter(bp));
                    placedBps.push({ bp, x: stageCoords[0], y: stageCoords[1] });
                }
            } else if (isCorrections || isRefine) {
                const stageCoords = getStageLabel(currentFrame, currentSide, bp);
                if (stageCoords) {
                    if (isRefine) {
                        // Compare corrections label to DLC label for this bodypart
                        const dlcBpArr = stageData['dlc'] && stageData['dlc'][currentSide]
                            ? stageData['dlc'][currentSide][bp] : null;
                        const dlcPt = dlcBpArr && currentFrame < dlcBpArr.length
                            ? dlcBpArr[currentFrame] : null;
                        if (dlcPt) {
                            const diff = Math.hypot(stageCoords[0] - dlcPt[0], stageCoords[1] - dlcPt[1]);
                            if (diff >= 3.0) {
                                // Genuine correction: show DLC as ghost underneath, corrections as full label
                                drawGhostPoint(dlcPt[0], dlcPt[1], bpColor(idx), 'D');
                                drawPoint(stageCoords[0], stageCoords[1], bpColor(idx), bpLetter(bp));
                                placedBps.push({ bp, x: stageCoords[0], y: stageCoords[1], stageSource: true });
                            } else {
                                // Matches DLC: show as ghost (no real correction here)
                                drawGhostPoint(stageCoords[0], stageCoords[1], bpColor(idx), bpLetter(bp));
                                placedBps.push({ bp, x: stageCoords[0], y: stageCoords[1], stageSource: true, ghost: true });
                            }
                        } else {
                            // No DLC reference: show as full label
                            drawPoint(stageCoords[0], stageCoords[1], bpColor(idx), bpLetter(bp));
                            placedBps.push({ bp, x: stageCoords[0], y: stageCoords[1], stageSource: true });
                        }
                    } else {
                        drawPoint(stageCoords[0], stageCoords[1], bpColor(idx), bpLetter(bp));
                        placedBps.push({ bp, x: stageCoords[0], y: stageCoords[1], stageSource: true });
                    }
                } else {
                    // Gap frame: show auto-merge as ghost
                    const mergedCoords = getMergedLabel(currentFrame, currentSide, bp);
                    if (mergedCoords) {
                        drawGhostPoint(mergedCoords[0], mergedCoords[1], bpColor(idx), 'A');
                        placedBps.push({ bp, x: mergedCoords[0], y: mergedCoords[1], ghost: true });
                    }
                }
            } else {
                // Ghost priority:
                //   Refine: committed manual > DLC > MP
                //   Initial: MP > DLC
                const mpCoords = getMpLabel(currentFrame, currentSide, bp);
                const dlcCoords = getDlcLabel(currentFrame, currentSide, bp);
                const comCoords = isRefine ? getCommittedLabel(currentFrame, currentSide, bp) : null;

                let ghostCoords = null;
                let ghostTag = '';
                if (isRefine) {
                    if (comCoords) { ghostCoords = comCoords; ghostTag = 'M'; }
                    else if (dlcCoords) { ghostCoords = dlcCoords; ghostTag = 'D'; }
                    else if (mpCoords) { ghostCoords = mpCoords; ghostTag = 'MP'; }
                } else {
                    if (mpCoords) { ghostCoords = mpCoords; ghostTag = 'MP'; }
                    else if (dlcCoords) { ghostCoords = dlcCoords; ghostTag = 'D'; }
                }

                if (ghostCoords) {
                    drawGhostPoint(ghostCoords[0], ghostCoords[1], bpColor(idx), ghostTag);
                    placedBps.push({ bp, x: ghostCoords[0], y: ghostCoords[1], ghost: true });
                }
            }
        });

        // Draw lines between consecutive placed bodyparts
        for (let i = 1; i < placedBps.length; i++) {
            const a = placedBps[i - 1];
            const b = placedBps[i];
            const ax = a.x * scale + offsetX;
            const ay = a.y * scale + offsetY;
            const bx = b.x * scale + offsetX;
            const by = b.y * scale + offsetY;
            const isGhost = a.ghost || b.ghost;
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.strokeStyle = isGhost ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            if (isGhost) ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    function drawPoint(imgX, imgY, color, letter) {
        const sx = imgX * scale + offsetX;
        const sy = imgY * scale + offsetY;
        const r = POINT_RADIUS;

        // Outer ring
        ctx.beginPath();
        ctx.arc(sx, sy, r + 2, 0, Math.PI * 2);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Filled circle
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Letter label
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(letter, sx, sy);

        // Crosshair
        ctx.beginPath();
        ctx.moveTo(sx - r - 4, sy);
        ctx.lineTo(sx - r - 1, sy);
        ctx.moveTo(sx + r + 1, sy);
        ctx.lineTo(sx + r + 4, sy);
        ctx.moveTo(sx, sy - r - 4);
        ctx.lineTo(sx, sy - r - 1);
        ctx.moveTo(sx, sy + r + 1);
        ctx.lineTo(sx, sy + r + 4);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    function drawGhostPoint(imgX, imgY, color, letter) {
        const sx = imgX * scale + offsetX;
        const sy = imgY * scale + offsetY;
        const r = POINT_RADIUS;

        // Dashed outer ring (ghost style)
        ctx.beginPath();
        ctx.arc(sx, sy, r + 2, 0, Math.PI * 2);
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);

        // Semi-transparent filled circle
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.globalAlpha = 0.4;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1.0;

        // Letter label
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(letter, sx, sy);
    }

    // ── Screen <-> Image coordinate conversion ────────
    function screenToImage(sx, sy) {
        return {
            x: (sx - offsetX) / scale,
            y: (sy - offsetY) / scale,
        };
    }

    function imageToScreen(ix, iy) {
        return {
            x: ix * scale + offsetX,
            y: iy * scale + offsetY,
        };
    }

    // ── Hit testing (manual labels + ghost markers) ───
    function hitTest(sx, sy) {
        const key = `${currentFrame}_${currentSide}`;
        const lbl = labels.get(key);

        // Check manual labels first
        if (lbl) {
            for (const bp of bodyparts) {
                const coords = lbl[bp];
                if (coords && coords[0] != null) {
                    const p = imageToScreen(coords[0], coords[1]);
                    if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) return { bp, ghost: false };
                }
            }
        }

        // Corrections / Refine mode: check stage-sourced labels and ghosts
        if (isCorrections || isRefine) {
            for (const bp of bodyparts) {
                if (hasManualLabel(currentFrame, currentSide, bp)) continue;
                const stageCoords = getStageLabel(currentFrame, currentSide, bp);
                if (stageCoords) {
                    const p = imageToScreen(stageCoords[0], stageCoords[1]);
                    if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) {
                        return { bp, ghost: false, stageSource: true };
                    }
                } else {
                    // Check ghost (auto-merge) markers on gap frames
                    const mergedCoords = getMergedLabel(currentFrame, currentSide, bp);
                    if (mergedCoords) {
                        const p = imageToScreen(mergedCoords[0], mergedCoords[1]);
                        if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) {
                            return { bp, ghost: true, source: 'merge' };
                        }
                    }
                }
            }
            return null;
        }

        // Check ghost markers (refine: committed > DLC > MP; initial: MP > DLC)
        for (const bp of bodyparts) {
            if (hasManualLabel(currentFrame, currentSide, bp)) continue;
            const mpCoords = getMpLabel(currentFrame, currentSide, bp);
            const dlcCoords = getDlcLabel(currentFrame, currentSide, bp);
            const comCoords = isRefine ? getCommittedLabel(currentFrame, currentSide, bp) : null;

            // Build priority list
            const candidates = [];
            if (isRefine) {
                if (comCoords) candidates.push({ coords: comCoords, source: 'committed' });
                if (dlcCoords) candidates.push({ coords: dlcCoords, source: 'dlc' });
                if (mpCoords) candidates.push({ coords: mpCoords, source: 'mp' });
            } else {
                if (mpCoords) candidates.push({ coords: mpCoords, source: 'mp' });
                if (dlcCoords) candidates.push({ coords: dlcCoords, source: 'dlc' });
            }

            for (const c of candidates) {
                const p = imageToScreen(c.coords[0], c.coords[1]);
                if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) return { bp, ghost: true, source: c.source };
            }
        }

        return null;
    }

    // ── Canvas events ─────────────────────────────────
    function setupCanvasEvents() {
        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('contextmenu', onRightClick);
        canvas.addEventListener('wheel', onWheel, { passive: false });

        // Resize handler
        const ro = new ResizeObserver(() => {
            if (currentImage) {
                if (!hasUserZoom) fitImage();
                render();
                renderTimeline();
                renderDistanceTrace();
            }
        });
        ro.observe(containerEl);
    }

    function onMouseDown(e) {
        if (e.button === 2) return; // right-click handled separately
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        // Final mode: only allow pan, no label interaction
        if (isFinal) {
            dragging = 'pending';
            didDrag = false;
            dragStartX = sx;
            dragStartY = sy;
            dragOrigX = offsetX;
            dragOrigY = offsetY;
            return;
        }

        const hit = hitTest(sx, sy);
        if (hit) {
            if (hit.ghost) {
                // Click on ghost marker: accept position as manual label
                let ghostCoords;
                if (isCorrections) {
                    ghostCoords = getMergedLabel(currentFrame, currentSide, hit.bp);
                } else {
                    const mpCoords = getMpLabel(currentFrame, currentSide, hit.bp);
                    const dlcCoords = getDlcLabel(currentFrame, currentSide, hit.bp);
                    const comCoords = isRefine ? getCommittedLabel(currentFrame, currentSide, hit.bp) : null;
                    ghostCoords = isRefine ? (comCoords || dlcCoords || mpCoords) : (mpCoords || dlcCoords);
                }
                if (ghostCoords) {
                    const key = `${currentFrame}_${currentSide}`;
                    let lbl = labels.get(key);
                    if (!lbl) { lbl = {}; labels.set(key, lbl); }
                    pushUndo(key, hit.bp, null);
                    lbl[hit.bp] = [ghostCoords[0], ghostCoords[1]];
                    dirtyKeys.add(key);

                    // Enter drag mode immediately
                    dragging = hit.bp;
                    didDrag = false;
                    dragOrigX = ghostCoords[0];
                    dragOrigY = ghostCoords[1];
                    dragStartX = sx;
                    dragStartY = sy;
                    canvas.style.cursor = 'grabbing';

                    render();
                    updateLabelCount();
                }
            } else if (hit.stageSource) {
                // Corrections mode: click on stage-sourced label — promote to manual
                const stageCoords = getStageLabel(currentFrame, currentSide, hit.bp);
                if (stageCoords) {
                    const key = `${currentFrame}_${currentSide}`;
                    let lbl = labels.get(key);
                    if (!lbl) { lbl = {}; labels.set(key, lbl); }
                    pushUndo(key, hit.bp, null);
                    lbl[hit.bp] = [stageCoords[0], stageCoords[1]];
                    dirtyKeys.add(key);

                    dragging = hit.bp;
                    didDrag = false;
                    dragOrigX = stageCoords[0];
                    dragOrigY = stageCoords[1];
                    dragStartX = sx;
                    dragStartY = sy;
                    canvas.style.cursor = 'grabbing';

                    render();
                    updateLabelCount();
                }
            } else {
                // Start dragging existing manual point
                dragging = hit.bp;
                didDrag = false;
                const key = `${currentFrame}_${currentSide}`;
                const lbl = labels.get(key);
                const coords = lbl[hit.bp];
                dragOrigX = coords[0];
                dragOrigY = coords[1];
                dragStartX = sx;
                dragStartY = sy;
                canvas.style.cursor = 'grabbing';
            }
        } else {
            // Pending: could become a pan (drag) or a click (place label)
            dragging = 'pending';
            didDrag = false;
            dragStartX = sx;
            dragStartY = sy;
            dragOrigX = offsetX;
            dragOrigY = offsetY;
        }
    }

    function onMouseMove(e) {
        if (!dragging) return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        if (dragging === 'pending') {
            // Check if mouse has moved enough to become a pan drag
            if (Math.hypot(sx - dragStartX, sy - dragStartY) > DRAG_THRESHOLD) {
                dragging = 'pan';
                didDrag = true;
                canvas.style.cursor = 'move';
            }
            return;
        }

        if (dragging === 'pan') {
            offsetX = dragOrigX + (sx - dragStartX);
            offsetY = dragOrigY + (sy - dragStartY);
            render();
        } else {
            // Dragging a bodypart point
            const dx = (sx - dragStartX) / scale;
            const dy = (sy - dragStartY) / scale;
            const key = `${currentFrame}_${currentSide}`;
            const lbl = labels.get(key);
            lbl[dragging] = [dragOrigX + dx, dragOrigY + dy];
            render();
        }
    }

    function onMouseUp(e) {
        if (dragging === 'pending') {
            // Mouse didn't move much — this is a click
            if (!isFinal) {
                const img = screenToImage(dragStartX, dragStartY);
                if (img.x >= 0 && img.x < imgW && img.y >= 0 && img.y < imgH) {
                    placeLabel(img.x, img.y);
                }
            }
        } else if (dragging === 'pan') {
            hasUserZoom = true;
        } else if (dragging) {
            // Finished dragging a bodypart point — record undo with pre-drag position
            const key = `${currentFrame}_${currentSide}`;
            pushUndo(key, dragging, [dragOrigX, dragOrigY]);
            dirtyKeys.add(key);
            scheduleSave();
            recomputeCameraShift();
        }
        dragging = null;
        didDrag = false;
        canvas.style.cursor = 'crosshair';
    }

    function onRightClick(e) {
        e.preventDefault();
        if (isFinal) return; // Read-only mode
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        const hit = hitTest(sx, sy);
        if (hit && !hit.ghost) {
            removeLabel(hit.bp);
        }
    }

    function onWheel(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.05 : 1 / 1.05;
        const newScale = scale * zoomFactor;

        // Zoom toward cursor
        offsetX = mx - (mx - offsetX) * zoomFactor;
        offsetY = my - (my - offsetY) * zoomFactor;
        scale = newScale;
        hasUserZoom = true;

        render();
    }

    // ── Undo ──────────────────────────────────────────
    function pushUndo(key, bp, prevCoords) {
        undoStack.push({ key, bp, prev: prevCoords });
        if (undoStack.length > MAX_UNDO) undoStack.shift();
    }

    function undo() {
        if (undoStack.length === 0) return;
        const action = undoStack.pop();
        const { key, bp, prev } = action;

        let lbl = labels.get(key);
        if (prev) {
            // Restore previous coordinates
            if (!lbl) { lbl = {}; labels.set(key, lbl); }
            lbl[bp] = prev;
            // If restoring a label, it's no longer deleted
            deletedKeys.delete(key);
            dirtyKeys.add(key);
        } else {
            // Was a new placement — remove it
            if (lbl) {
                delete lbl[bp];
                const hasAny = bodyparts.some(b => lbl[b] && lbl[b][0] != null);
                if (!hasAny) {
                    labels.delete(key);
                    deletedKeys.add(key);
                } else {
                    dirtyKeys.add(key);
                }
            }
        }

        render();
        updateLabelCount();
        scheduleSave();
        recomputeCameraShift();
        if (isRefine) updateV2TrainingBtn();
    }

    // ── Zoom helpers ─────────────────────────────────
    function autoZoomForFrame(frame, side) {
        // Collect points from manual labels, falling back to MP detections
        // In review mode, only zoom to the reviewed bodypart (tight zoom)
        const key = `${frame}_${side}`;
        const lbl = labels.get(key);
        const pts = [];
        const bpsToShow = reviewBp ? [reviewBp] : bodyparts;

        for (const bp of bpsToShow) {
            const manual = lbl ? lbl[bp] : null;
            if (manual && manual[0] != null) {
                pts.push(manual);
            } else if (isCorrections || isRefine) {
                const sc = getStageLabel(frame, side, bp);
                if (sc) pts.push(sc);
            } else {
                const com = isRefine ? getCommittedLabel(frame, side, bp) : null;
                if (com) {
                    pts.push(com);
                } else {
                    const mp = getMpLabel(frame, side, bp);
                    if (mp) {
                        pts.push(mp);
                    } else {
                        const dlc = getDlcLabel(frame, side, bp);
                        if (dlc) pts.push(dlc);
                    }
                }
            }
        }
        if (pts.length === 0) return false;

        zoomToPoints(pts, !!reviewBp);
        return true;
    }

    function zoomToPoints(pts, tight) {
        // Bounding box in image coords
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const [x, y] of pts) {
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        const span = Math.max(maxX - minX, maxY - minY, 20);
        const pad = tight ? span * 2.5 + 30 : span * 0.8 + 40;
        minX -= pad; minY -= pad;
        maxX += pad; maxY += pad;

        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        const bboxW = maxX - minX;
        const bboxH = maxY - minY;

        scale = Math.min(cw / bboxW, ch / bboxH);
        offsetX = (cw - bboxW * scale) / 2 - minX * scale;
        offsetY = (ch - bboxH * scale) / 2 - minY * scale;
        hasUserZoom = true;
    }

    function zoomToLabels(frame, side) {
        const key = `${frame}_${side}`;
        const lbl = labels.get(key);
        if (!lbl) return;

        // In review mode, zoom to just the reviewed bodypart
        const bpsToShow = reviewBp ? [reviewBp] : bodyparts;

        // Collect placed coordinates
        const pts = [];
        for (const bp of bpsToShow) {
            const c = lbl[bp];
            if (c && c[0] != null) pts.push(c);
        }
        if (pts.length === 0) return;

        zoomToPoints(pts, !!reviewBp);
    }

    // ── Label placement ───────────────────────────────
    function placeLabel(imgX, imgY) {
        const key = `${currentFrame}_${currentSide}`;
        let lbl = labels.get(key);
        if (!lbl) {
            lbl = {};
            labels.set(key, lbl);
        }

        // Find first unplaced bodypart
        let placed = false;
        for (const bp of bodyparts) {
            const coords = lbl[bp];
            if (!coords || coords[0] == null) {
                pushUndo(key, bp, null);
                lbl[bp] = [imgX, imgY];
                placed = true;
                const remaining = bodyparts.filter(b => !lbl[b] || lbl[b][0] == null);
                if (remaining.length > 0) {
                    updateLabelInfo(`${bp} placed. Click to place ${remaining[0]}.`);
                } else {
                    updateLabelInfo('All keypoints placed.');
                }
                break;
            }
        }

        if (!placed) {
            // All bodyparts placed — move the closest one
            let closest = null;
            let minDist = Infinity;
            for (const bp of bodyparts) {
                const coords = lbl[bp];
                if (coords && coords[0] != null) {
                    const d = Math.hypot(imgX - coords[0], imgY - coords[1]);
                    if (d < minDist) { minDist = d; closest = bp; }
                }
            }
            if (closest) {
                pushUndo(key, closest, [...lbl[closest]]);
                lbl[closest] = [imgX, imgY];
            }
        }

        dirtyKeys.add(key);
        render();
        updateLabelCount();
        scheduleSave();
        recomputeCameraShift();
        if (isRefine) updateV2TrainingBtn();
    }

    function removeLabel(which) {
        const key = `${currentFrame}_${currentSide}`;
        const lbl = labels.get(key);

        if (lbl && lbl[which] && lbl[which][0] != null) {
            // Remove manual label
            const prev = lbl[which];
            pushUndo(key, which, [...prev]);
            delete lbl[which];

            const hasAny = bodyparts.some(bp => lbl[bp] && lbl[bp][0] != null);
            if (!hasAny) {
                labels.delete(key);
                deletedKeys.add(key);
            } else {
                dirtyKeys.add(key);
            }

            render();
            updateLabelCount();
            scheduleSave();
            if (isRefine) updateV2TrainingBtn();
            return;
        }

        // Corrections/Refine mode: reject the stage-sourced label so next-priority appears
        if (isCorrections || isRefine) {
            rejectStageLabel(which);
        }
    }

    function rejectStageLabel(bp) {
        /** Reject the current stage label for a bodypart, exposing the next-priority one. */
        const src = getStageLabelSource(currentFrame, currentSide, bp);
        if (!src) return;
        rejectedStageLabels.add(`${currentFrame}_${currentSide}_${bp}_${src.stage}`);
        render();
    }

    // ── Auto-save ─────────────────────────────────────
    let saveTimeout = null;
    let saveInFlight = false;
    let saveQueued = false;
    let savePromise = null;  // tracks the current in-flight save

    function scheduleSave() {
        // Save immediately (next tick) with in-flight guard to prevent
        // concurrent requests while still avoiding delays.
        if (saveInFlight) {
            saveQueued = true;
            return;
        }
        if (saveTimeout) clearTimeout(saveTimeout);
        saveTimeout = setTimeout(() => { savePromise = saveLabels(); }, 0);
    }

    async function saveLabels() {
        if (saveTimeout) clearTimeout(saveTimeout);
        saveInFlight = true;

        // Send DELETE requests for removed labels
        const deletePromises = [];
        for (const key of deletedKeys) {
            const [frameStr, side] = key.split('_');
            deletePromises.push(
                API.del(`/api/labeling/sessions/${sessionId}/labels/${frameStr}?side=${encodeURIComponent(side)}`)
                    .catch(e => console.error('Delete failed for', key, e))
            );
        }
        if (deletePromises.length > 0) {
            await Promise.all(deletePromises);
            deletedKeys.clear();
        }

        const batch = [];

        // Only send labels that have been modified since last save
        labels.forEach((lbl, key) => {
            if (!dirtyKeys.has(key)) return;
            const [frameStr, side] = key.split('_');
            const frame = parseInt(frameStr);
            // Determine trial index
            let trialIdx = 0;
            for (let i = 0; i < trials.length; i++) {
                if (frame >= trials[i].start_frame && frame <= trials[i].end_frame) {
                    trialIdx = i;
                    break;
                }
            }
            batch.push({
                frame_num: frame,
                trial_idx: trialIdx,
                side: side,
                keypoints: lbl,
            });
        });
        dirtyKeys.clear();

        if (batch.length === 0) {
            saveInFlight = false;
            return;
        }

        try {
            const result = await API.put(`/api/labeling/sessions/${sessionId}/labels`, { labels: batch });
            // Update distances from server response
            if (result.updated_distances && Object.keys(result.updated_distances).length > 0) {
                // Create distances array if it didn't exist yet
                if (!distances) {
                    distances = new Array(totalFrames).fill(null);
                    const traceContainer = document.getElementById('distanceTraceContainer');
                    if (traceContainer && !isFinal) traceContainer.style.display = 'block';
                }
                for (const [frameStr, dist] of Object.entries(result.updated_distances)) {
                    const frame = parseInt(frameStr);
                    if (frame >= 0 && frame < distances.length) {
                        distances[frame] = dist;
                    }
                }
                renderDistanceTrace();
                if (isFinal) renderTrialPlots();
            }
        } catch (e) {
            console.error('Save failed:', e);
        } finally {
            saveInFlight = false;
            if (saveQueued) {
                saveQueued = false;
                scheduleSave();
            }
        }
    }

    // ── Commit ────────────────────────────────────────
    async function commitSession() {
        // Wait for any in-flight auto-save to complete first
        if (savePromise) await savePromise;
        // Then flush any remaining dirty keys
        await saveLabels();

        if (!isRefine && labels.size === 0) {
            alert('No labels to commit.');
            return;
        }

        try {
            const commitBody = {};
            if (isRefine) {
                // v2_train_frames = correction frames the user has NOT excluded
                commitBody.v2_train_frames = [...correctionFrames]
                    .filter(key => !v2Excludes.has(key))
                    .map(key => {
                        const [frameStr, side] = key.split('_');
                        return { frame_num: parseInt(frameStr), side };
                    });
                if (commitBody.v2_train_frames.length === 0) {
                    updateLabelInfo('No training frames selected — toggle some on first.');
                    return;
                }
            }
            const result = await API.post(`/api/labeling/sessions/${sessionId}/commit`, commitBody);
            updateLabelInfo(`Committed ${result.frame_count} frames to DLC labeled-data.`);
        } catch (e) {
            alert('Commit error: ' + e.message);
        }
    }

    async function saveCorrectionsOnly() {
        if (!isRefine) return;
        if (savePromise) await savePromise;
        await saveLabels();
        if (labels.size === 0) {
            updateLabelInfo('No manual corrections to save yet.');
            return;
        }
        try {
            const result = await API.post(`/api/labeling/sessions/${sessionId}/save_corrections`);
            updateLabelInfo(`Saved ${result.frame_count} corrected frames to corrections CSV.`);
        } catch (e) {
            alert('Save error: ' + e.message);
        }
    }

    // ── Navigation ────────────────────────────────────
    function nextFrame() { goToFrame(currentFrame + 1); }
    function prevFrame() { goToFrame(currentFrame - 1); }

    async function nextLabel() {
        if (isRefine) {
            const sorted = getCorrectionFramesSorted();
            const next = sorted.find(f => f > currentFrame);
            if (next !== undefined) await goToFrame(next);
            return;
        }
        const sorted = getLabeledFrames();
        const next = sorted.find(f => f > currentFrame);
        if (next !== undefined) {
            zoomToLabels(next, currentSide);
            await goToFrame(next);
        }
    }

    async function prevLabel() {
        if (isRefine) {
            const sorted = getCorrectionFramesSorted();
            const prev = [...sorted].reverse().find(f => f < currentFrame);
            if (prev !== undefined) await goToFrame(prev);
            return;
        }
        const sorted = getLabeledFrames();
        const prev = [...sorted].reverse().find(f => f < currentFrame);
        if (prev !== undefined) {
            zoomToLabels(prev, currentSide);
            await goToFrame(prev);
        }
    }

    function getCorrectionFramesSorted() {
        const frames = new Set();
        correctionFrames.forEach(key => {
            const [f] = key.split('_');
            frames.add(parseInt(f));
        });
        return [...frames].sort((a, b) => a - b);
    }

    async function nextGap() {
        if (!isCorrections && !isRefine) return;
        // Check both cameras at each frame, current camera first
        const sides = [currentSide, ...cameraNames.filter(c => c !== currentSide)];
        for (let f = currentFrame + 1; f < totalFrames; f++) {
            for (const side of sides) {
                if (isGapFrame(f, side)) {
                    if (side !== currentSide) toggleSide();
                    zoomToMergeLabels(f, side);
                    await goToFrame(f);
                    updateLabelInfo(`Gap frame ${f} (${side}) — Enter to accept merge`);
                    return;
                }
            }
        }
        // Wrap: check frames before current
        for (let f = 0; f <= currentFrame; f++) {
            for (const side of sides) {
                if (isGapFrame(f, side)) {
                    if (side !== currentSide) toggleSide();
                    zoomToMergeLabels(f, side);
                    await goToFrame(f);
                    updateLabelInfo(`Gap frame ${f} (${side}) — Enter to accept merge`);
                    return;
                }
            }
        }
        updateLabelInfo('No more gaps');
    }

    async function prevGap() {
        if (!isCorrections && !isRefine) return;
        const sides = [currentSide, ...cameraNames.filter(c => c !== currentSide)];
        for (let f = currentFrame - 1; f >= 0; f--) {
            for (const side of sides) {
                if (isGapFrame(f, side)) {
                    if (side !== currentSide) toggleSide();
                    zoomToMergeLabels(f, side);
                    await goToFrame(f);
                    updateLabelInfo(`Gap frame ${f} (${side}) — Enter to accept merge`);
                    return;
                }
            }
        }
        // Wrap: check frames after current
        for (let f = totalFrames - 1; f >= currentFrame; f--) {
            for (const side of sides) {
                if (isGapFrame(f, side)) {
                    if (side !== currentSide) toggleSide();
                    zoomToMergeLabels(f, side);
                    await goToFrame(f);
                    updateLabelInfo(`Gap frame ${f} (${side}) — Enter to accept merge`);
                    return;
                }
            }
        }
        updateLabelInfo('No more gaps');
    }

    function zoomToMergeLabels(frame, side) {
        /** Zoom to the candidate merge label positions for a gap frame. */
        const pts = [];
        for (const bp of bodyparts) {
            if (hasManualLabel(frame, side, bp)) continue;
            const merged = getMergedLabel(frame, side, bp);
            if (merged) pts.push(merged);
        }
        if (pts.length > 0) {
            zoomToPoints(pts, true);
        }
    }

    function acceptMergedLabels() {
        /** Accept auto-merge labels for current frame, promoting them to manual corrections.
         *  Only operates on gap bodyparts (where getStageLabel returns null). */
        if (!isCorrections && !isRefine) return;
        const key = `${currentFrame}_${currentSide}`;
        let lbl = labels.get(key);
        if (!lbl) { lbl = {}; labels.set(key, lbl); }

        let accepted = 0;
        for (const bp of bodyparts) {
            // Skip if already has manual label
            if (lbl[bp] && lbl[bp][0] != null) continue;
            // Only accept if this bodypart is a gap (no stage label)
            if (getStageLabel(currentFrame, currentSide, bp)) continue;
            const merged = getMergedLabel(currentFrame, currentSide, bp);
            if (merged) {
                pushUndo(key, bp, null);
                lbl[bp] = [merged[0], merged[1]];
                accepted++;
            }
        }

        if (accepted > 0) {
            dirtyKeys.add(key);
            render();
            updateLabelCount();
            scheduleSave();
            updateLabelInfo(`Accepted ${accepted} labels — W for next gap`);
        } else {
            updateLabelInfo('Nothing to accept');
        }
    }

    function getLabeledFrames() {
        // In final mode, navigate through frames that have data in the 'labels' stage
        if (isFinal) {
            const sd = stageData['labels'];
            if (!sd || !sd[currentSide]) return [];
            const camData = sd[currentSide];
            const frames = new Set();
            for (const bp of bodyparts) {
                const arr = camData[bp];
                if (!arr) continue;
                for (let f = 0; f < arr.length; f++) {
                    if (arr[f] != null) frames.add(f);
                }
            }
            return [...frames].sort((a, b) => a - b);
        }

        const frames = new Set();
        labels.forEach((lbl, key) => {
            const [frameStr, side] = key.split('_');
            if (side !== currentSide) return;
            // In review mode, only include frames that have the reviewed bodypart
            if (reviewBp) {
                const c = lbl[reviewBp];
                if (!c || c[0] == null) return;
            }
            frames.add(parseInt(frameStr));
        });
        return [...frames].sort((a, b) => a - b);
    }

    function updateLabelNavButtons() {
        const prev = document.getElementById('prevLabelBtn');
        const next = document.getElementById('nextLabelBtn');
        if (!prev || !next) return;
        const show = !isFinal || selectedStage === 'labels';
        prev.style.display = show ? '' : 'none';
        next.style.display = show ? '' : 'none';
        if (isRefine) {
            prev.textContent = '\u2190 Corr';
            next.textContent = 'Corr \u2192';
        }
    }

    function recomputeCameraShift() {
        // Compute the average horizontal and vertical offset between cam0/cam1.
        // Uses manual labels first; falls back to MP detections for more samples.
        if (cameraNames.length < 2) return;
        const cam0 = cameraNames[0];
        const cam1 = cameraNames[1];
        const dxValues = [];
        const dyValues = [];

        // 1. Manual labels: find frames with labels in both cameras
        const frameNums = new Set();
        labels.forEach((_, key) => {
            const [f] = key.split('_');
            frameNums.add(f);
        });

        for (const f of frameNums) {
            const lbl0 = labels.get(`${f}_${cam0}`);
            const lbl1 = labels.get(`${f}_${cam1}`);
            if (!lbl0 || !lbl1) continue;

            for (const bp of bodyparts) {
                const c0 = lbl0[bp];
                const c1 = lbl1[bp];
                if (c0 && c0[0] != null && c1 && c1[0] != null) {
                    dxValues.push(c1[0] - c0[0]);
                    dyValues.push(c1[1] - c0[1]);
                }
            }
        }

        // 2a. Corrections/Final/Refine mode: use stage labels for camera shift estimation
        if (dxValues.length < 4 && (isCorrections || isFinal || isRefine) && availableStages.length > 0) {
            for (let f = 0; f < totalFrames; f += 10) {
                for (const bp of bodyparts) {
                    const c0 = getStageLabel(f, cam0, bp);
                    const c1 = getStageLabel(f, cam1, bp);
                    if (c0 && c1) {
                        dxValues.push(c1[0] - c0[0]);
                        dyValues.push(c1[1] - c0[1]);
                    }
                }
            }
        }

        // 2. MP labels: sample every 10th frame for efficiency
        if (dxValues.length < 4 && mpLabels && mpLabels[cam0] && mpLabels[cam1]) {
            const mpDx = [];
            const mpDy = [];
            for (let f = 0; f < totalFrames; f += 10) {
                for (const bp of bodyparts) {
                    const c0 = getMpLabel(f, cam0, bp);
                    const c1 = getMpLabel(f, cam1, bp);
                    if (c0 && c1) {
                        mpDx.push(c1[0] - c0[0]);
                        mpDy.push(c1[1] - c0[1]);
                    }
                }
            }
            if (mpDx.length > 0) {
                // Use median to be robust to MP outliers
                mpDx.sort((a, b) => a - b);
                mpDy.sort((a, b) => a - b);
                const mid = Math.floor(mpDx.length / 2);
                const mpShiftX = mpDx.length % 2 ? mpDx[mid] : (mpDx[mid - 1] + mpDx[mid]) / 2;
                const mpShiftY = mpDy.length % 2 ? mpDy[mid] : (mpDy[mid - 1] + mpDy[mid]) / 2;

                if (dxValues.length === 0) {
                    // No manual data at all — use MP directly
                    computedCameraShiftX = mpShiftX;
                    computedCameraShiftY = mpShiftY;
                    return;
                }
                // Few manual samples — blend: manual mean weighted 2x over MP median
                dxValues.push(mpShiftX, mpShiftX);
                dyValues.push(mpShiftY, mpShiftY);
            }
        }

        if (dxValues.length > 0) {
            computedCameraShiftX = dxValues.reduce((a, b) => a + b, 0) / dxValues.length;
            computedCameraShiftY = dyValues.reduce((a, b) => a + b, 0) / dyValues.length;
        }
    }

    function cycleReviewMode() {
        if (!reviewBp) {
            reviewBp = bodyparts[0];
        } else {
            const idx = bodyparts.indexOf(reviewBp);
            if (idx < bodyparts.length - 1) {
                reviewBp = bodyparts[idx + 1];
            } else {
                reviewBp = null;
            }
        }
        updateReviewIndicator();
        // Re-zoom to current frame's labels with new mode
        if (reviewBp) {
            zoomToLabels(currentFrame, currentSide);
            render();
        }
    }

    function updateReviewIndicator() {
        const el = document.getElementById('labelInfo');
        if (reviewBp) {
            const idx = bodyparts.indexOf(reviewBp);
            el.textContent = `Review mode: ${reviewBp}`;
            el.style.color = bpColor(idx);
        } else {
            el.textContent = 'Click to place keypoints';
            el.style.color = '';
        }
    }

    function getTrialForFrame(frame) {
        for (let i = 0; i < trials.length; i++) {
            if (frame >= trials[i].start_frame && frame <= trials[i].end_frame) {
                return i;
            }
        }
        return 0;
    }

    function toggleSide() {
        const idx = cameraNames.indexOf(currentSide);
        const newIdx = (idx + 1) % cameraNames.length;

        // Shift viewport to keep targets roughly centered when switching cameras.
        // Uses computed shift from paired labels, falls back to 7% horizontal default.
        if (hasUserZoom && imgW) {
            let shiftX, shiftY;
            if (computedCameraShiftX != null) {
                shiftX = computedCameraShiftX;
                shiftY = computedCameraShiftY || 0;
            } else {
                shiftX = imgW * 0.07;
                shiftY = 0;
            }
            // cam0→cam1: targets move by shift, compensate viewport in opposite direction
            const direction = (newIdx > idx) ? -1 : 1;
            offsetX += direction * shiftX * scale;
            offsetY += direction * shiftY * scale;
        }

        currentSide = cameraNames[newIdx];
        document.getElementById('sideToggle').textContent = currentSide;
        goToFrame(currentFrame);
    }

    function togglePlay() {
        playing = !playing;
        const btn = document.getElementById('playBtn');
        if (playing) {
            btn.innerHTML = '&#9646;&#9646;';
            playbackRate = parseFloat(document.getElementById('playbackRate').value);
            startVideoPlayback();
        } else {
            btn.innerHTML = '&#9654;';
            stopVideoPlayback();
        }
    }

    async function startVideoPlayback() {
        if (!videoEl) {
            fallbackPlay();
            return;
        }

        const trialIdx = getTrialForFrame(currentFrame);
        const trial = trials[trialIdx];
        if (!trial) return;

        const localFrame = currentFrame - trial.start_frame;
        const startTime = localFrame / trial.fps;

        // Use streaming URL (not blob) — browser handles buffering natively
        const videoUrl = `/api/labeling/sessions/${sessionId}/video?trial=${trialIdx}&_=${Date.now()}`;
        if (currentTrialIdx !== trialIdx) {
            videoEl.src = videoUrl;
            currentTrialIdx = trialIdx;
            console.log(`[video] Loading trial ${trialIdx}, readyState=${videoEl.readyState}`);

            // Wait for enough data to play, with error/timeout fallback
            const ready = await new Promise(resolve => {
                if (videoEl.readyState >= 3) { resolve(true); return; }
                const timer = setTimeout(() => { resolve(false); }, 8000);
                const onReady = () => {
                    clearTimeout(timer);
                    videoEl.removeEventListener('error', onError);
                    resolve(true);
                };
                const onError = () => {
                    clearTimeout(timer);
                    videoEl.removeEventListener('canplay', onReady);
                    console.error('[video] Error loading:', videoEl.error);
                    resolve(false);
                };
                videoEl.addEventListener('canplay', onReady, { once: true });
                videoEl.addEventListener('error', onError, { once: true });
            });

            console.log(`[video] Ready=${ready}, readyState=${videoEl.readyState}`);
            if (!ready || !playing) {
                if (!ready) {
                    console.warn('[video] Timed out or error, falling back to frame-by-frame');
                    fallbackPlay();
                }
                return;
            }
        }

        videoEl.playbackRate = playbackRate;
        videoEl.currentTime = startTime;
        videoPlaying = true;

        try {
            await videoEl.play();
            console.log(`[video] Playing trial ${trialIdx} at ${startTime.toFixed(2)}s`);
            requestAnimationFrame(videoDrawLoop);
        } catch (e) {
            console.error('[video] play() rejected:', e);
            videoPlaying = false;
            fallbackPlay();
        }

        // Handle trial end — stop or advance to next trial
        videoEl.onended = () => {
            const nextTrialIdx = trialIdx + 1;
            if (nextTrialIdx < trials.length && playing) {
                currentFrame = trials[nextTrialIdx].start_frame;
                startVideoPlayback();
            } else {
                playing = false;
                videoPlaying = false;
                document.getElementById('playBtn').innerHTML = '&#9654;';
                goToFrame(currentFrame);
                canvas.focus();
            }
        };
    }

    function videoDrawLoop() {
        if (!videoPlaying || !playing) return;

        const trial = trials[currentTrialIdx];
        if (!trial) return;

        // Calculate current global frame from video time
        const localFrame = Math.floor(videoEl.currentTime * trial.fps);
        currentFrame = trial.start_frame + Math.min(localFrame, trial.frame_count - 1);

        // Draw video frame to canvas (cropped to correct camera half)
        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        canvas.width = cw;
        canvas.height = ch;
        ctx.clearRect(0, 0, cw, ch);

        const vw = videoEl.videoWidth;
        const vh = videoEl.videoHeight;
        if (vw > 0 && vh > 0) {
            const midline = Math.floor(vw / 2);
            // Source crop: left half or right half of stereo video
            let sx, sw;
            if (cameraNames.length >= 2 && currentSide === cameraNames[1]) {
                sx = midline; sw = vw - midline;
            } else {
                sx = 0; sw = midline;
            }

            imgW = sw;
            imgH = vh;
            if (!hasUserZoom) fitImage();

            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            ctx.drawImage(videoEl, sx, 0, sw, vh, 0, 0, sw, vh);
            ctx.restore();
        }

        drawLabelsOverlay();
        updateFrameDisplay();

        // Only update timeline/trace every ~10 frames to reduce work during playback
        if (currentFrame % 10 === 0) {
            renderTimeline();
            renderDistanceTrace();
        }

        requestAnimationFrame(videoDrawLoop);
    }

    function stopVideoPlayback() {
        const wasVideoPlaying = videoPlaying;
        videoPlaying = false;
        if (videoEl) {
            videoEl.pause();
            // Only recalculate from video time if the video element was actually
            // rendering frames (not in fallback frame-by-frame mode)
            if (wasVideoPlaying) {
                const trial = trials[currentTrialIdx];
                if (trial) {
                    const localFrame = Math.round(videoEl.currentTime * trial.fps);
                    currentFrame = trial.start_frame + Math.min(Math.max(0, localFrame), trial.frame_count - 1);
                }
            }
        }
        goToFrame(currentFrame);
        canvas.focus();
    }

    function fallbackPlay() {
        // Fallback: frame-by-frame if video streaming fails
        playbackRate = parseFloat(document.getElementById('playbackRate').value);
        const fps = trials.length > 0 ? trials[0].fps : 30;
        const interval = 1000 / (fps * playbackRate);
        (async function playLoop() {
            while (playing && currentFrame < totalFrames - 1) {
                const start = performance.now();
                await goToFrame(currentFrame + 1);
                const elapsed = performance.now() - start;
                const wait = Math.max(0, interval - elapsed);
                await new Promise(r => setTimeout(r, wait));
            }
            if (playing) {
                playing = false;
                document.getElementById('playBtn').innerHTML = '&#9654;';
            }
        })();
    }

    function resetZoom() {
        hasUserZoom = false;
        fitImage();
        render();
    }

    // ── Keyboard shortcuts ────────────────────────────
    // Letter keys for deleting bodyparts (first two get D/F, rest use number keys)
    const DELETE_KEYS = { 'd': 0, 'f': 1 };

    function setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            // Ctrl+Z: undo (disabled in final mode)
            if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                if (!isFinal) undo();
                return;
            }

            switch (e.key) {
                case 'a': case 'ArrowLeft':
                    e.preventDefault();
                    prevFrame();
                    break;
                case 's': case 'ArrowRight':
                    e.preventDefault();
                    nextFrame();
                    break;
                case 'q':
                    e.preventDefault();
                    if (isCorrections || isRefine) prevGap();
                    else prevLabel();
                    break;
                case 'w':
                    e.preventDefault();
                    if (isCorrections || isRefine) nextGap();
                    else nextLabel();
                    break;
                case 'r':
                    e.preventDefault();
                    cycleReviewMode();
                    break;
                case 'e':
                    e.preventDefault();
                    toggleSide();
                    break;
                case 'z':
                    e.preventDefault();
                    resetZoom();
                    break;
                case 't':
                    e.preventDefault();
                    if (isRefine) toggleV2Training();
                    break;
                case 'Enter':
                    if (isCorrections || isRefine) {
                        e.preventDefault();
                        acceptMergedLabels();
                    }
                    break;
                case ' ':
                    e.preventDefault();
                    togglePlay();
                    break;
                default:
                    if (isFinal) break; // Read-only: no delete shortcuts
                    // D/F: delete bodypart by letter
                    if (e.key in DELETE_KEYS) {
                        const idx = DELETE_KEYS[e.key];
                        if (idx < bodyparts.length) {
                            e.preventDefault();
                            removeLabel(bodyparts[idx]);
                        }
                        break;
                    }
                    // Number keys 1-9: delete bodypart by index
                    if (e.key >= '1' && e.key <= '9') {
                        const idx = parseInt(e.key) - 1;
                        if (idx < bodyparts.length) {
                            e.preventDefault();
                            removeLabel(bodyparts[idx]);
                        }
                    }
                    break;
            }
        });
    }

    // ── UI updates ────────────────────────────────────
    function updateFrameDisplay() {
        // Find current trial and compute local frame
        let trialName = '--';
        let localFrame = currentFrame;
        let trialFrameCount = totalFrames;
        for (const t of trials) {
            if (currentFrame >= t.start_frame && currentFrame <= t.end_frame) {
                trialName = t.trial_name;
                localFrame = currentFrame - t.start_frame;
                trialFrameCount = t.frame_count;
                break;
            }
        }
        document.getElementById('frameDisplay').textContent =
            `Frame: ${localFrame} / ${trialFrameCount - 1}`;
        document.getElementById('trialDisplay').textContent = `Trial: ${trialName}`;

        // Show distance for current frame (unless in review mode)
        if (!reviewBp) {
            const distInfo = document.getElementById('labelInfo');
            if (distances && distances[currentFrame] !== null && distances[currentFrame] !== undefined) {
                distInfo.textContent = `Distance: ${distances[currentFrame].toFixed(1)} mm`;
            } else {
                distInfo.textContent = 'Click to place keypoints';
            }
        }
    }

    function updateLabelCount() {
        const count = labels.size;
        const committedStr = committedFrameCount > 0 ? ` (${committedFrameCount} committed)` : '';
        document.getElementById('labelCount').innerHTML =
            `Labels: <strong>${count}</strong>${committedStr}`;
    }

    function updateLabelInfo(msg) {
        document.getElementById('labelInfo').textContent = msg;
    }

    function updateShortcutsSidebar() {
        const el = document.getElementById('shortcutList');
        if (!el) return;

        if (isFinal) {
            el.innerHTML = `
                <div><kbd>A</kbd> / <kbd>&larr;</kbd> Prev frame</div>
                <div><kbd>S</kbd> / <kbd>&rarr;</kbd> Next frame</div>
                <div><kbd>E</kbd> Toggle ${cameraNames.join('/')}</div>
                <div><kbd>Z</kbd> Reset zoom</div>
                <div><kbd>Space</kbd> Play/pause</div>
                <div><kbd>Scroll</kbd> Zoom at cursor</div>
                <div><kbd>Drag</kbd> Pan</div>
            `;
            return;
        }

        const deleteLetters = ['D', 'F'];
        let html = `
            <div><kbd>A</kbd> / <kbd>&larr;</kbd> Prev frame</div>
            <div><kbd>S</kbd> / <kbd>&rarr;</kbd> Next frame</div>
        `;
        if (isCorrections || isRefine) {
            html += `
                <div><kbd>Q</kbd> Prev gap</div>
                <div><kbd>W</kbd> Next gap</div>
                <div><kbd>Enter</kbd> Accept merge</div>
            `;
            if (isRefine) {
                html += `<div><kbd>T</kbd> Toggle v2 training</div>`;
            }
        } else {
            html += `
                <div><kbd>Q</kbd> Prev label</div>
                <div><kbd>W</kbd> Next label</div>
            `;
        }
        bodyparts.forEach((bp, idx) => {
            const letter = deleteLetters[idx] ? `<kbd>${deleteLetters[idx]}</kbd> / ` : '';
            html += `<div>${letter}<kbd>${idx + 1}</kbd> Delete ${bp}</div>`;
        });
        html += `
            <div><kbd>Ctrl+Z</kbd> Undo</div>
            <div><kbd>R</kbd> Review mode (cycle)</div>
            <div><kbd>E</kbd> Toggle ${cameraNames.join('/')}</div>
            <div><kbd>Z</kbd> Reset zoom</div>
            <div><kbd>Space</kbd> Play/pause</div>
            <div><kbd>Scroll</kbd> Zoom at cursor</div>
            <div><kbd>Click</kbd> Place / accept MP</div>
            <div><kbd>Drag image</kbd> Pan</div>
            <div><kbd>Drag label</kbd> Move label</div>
            <div><kbd>Right-click</kbd> Remove label</div>
        `;
        el.innerHTML = html;
    }

    // ── Timeline ──────────────────────────────────────
    function setupTimeline() {
        timeline.addEventListener('click', onTimelineClick);

        const ro = new ResizeObserver(() => renderTimeline());
        ro.observe(timeline.parentElement);
    }

    function onTimelineClick(e) {
        const rect = timeline.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const frame = Math.max(0, Math.min(Math.floor((x / rect.width) * totalFrames), totalFrames - 1));
        // Auto-zoom to labels/MP at the target frame
        autoZoomForFrame(frame, currentSide);
        goToFrame(frame);
    }

    function renderTimeline() {
        if (isFinal) {
            renderTrialPlots();
            return;
        }

        const container = timeline.parentElement;
        const w = container.clientWidth;
        const h = container.clientHeight;
        timeline.width = w;
        timeline.height = h;

        if (totalFrames === 0) return;

        const nCams = cameraNames.length;
        const rowH = (h - 20) / Math.max(nCams, 1);
        const labelY = {};
        cameraNames.forEach((cam, i) => { labelY[cam] = 10 + i * rowH; });

        // Background
        tlCtx.fillStyle = '#1a1a2e';
        tlCtx.fillRect(0, 0, w, h);

        // Row labels
        tlCtx.fillStyle = '#8892a0';
        tlCtx.font = '10px sans-serif';
        cameraNames.forEach(cam => {
            tlCtx.fillText(cam, 2, labelY[cam] + rowH / 2 + 3);
        });

        const barX = 24;
        const barW = w - barX - 4;

        // Trial boundaries
        for (const t of trials) {
            const x = barX + (t.start_frame / totalFrames) * barW;
            tlCtx.beginPath();
            tlCtx.moveTo(x, 8);
            tlCtx.lineTo(x, h - 2);
            tlCtx.strokeStyle = '#2a3a5c';
            tlCtx.lineWidth = 1;
            tlCtx.stroke();
        }

        // MP coverage bars (thin dim blue lines where MP detected hand)
        if (mpLabels) {
            cameraNames.forEach(cam => {
                const camData = mpLabels[cam];
                if (!camData) return;
                const thumbArr = camData.thumb;
                if (!thumbArr) return;

                const yBase = labelY[cam];
                const barY = yBase + rowH / 2;

                tlCtx.beginPath();
                let inSegment = false;
                for (let f = 0; f < totalFrames && f < thumbArr.length; f++) {
                    const x = barX + (f / totalFrames) * barW;
                    if (thumbArr[f] !== null) {
                        if (!inSegment) {
                            tlCtx.moveTo(x, barY);
                            inSegment = true;
                        } else {
                            tlCtx.lineTo(x, barY);
                        }
                    } else {
                        inSegment = false;
                    }
                }
                tlCtx.strokeStyle = 'rgba(74, 158, 255, 0.2)';
                tlCtx.lineWidth = Math.max(rowH * 0.4, 3);
                tlCtx.stroke();
            });
        }

        // Manual label dots (bright, on top of MP bars)
        labels.forEach((lbl, key) => {
            const [frameStr, side] = key.split('_');
            const frame = parseInt(frameStr);
            const yBase = labelY[side];
            if (yBase === undefined) return;

            const x = barX + (frame / totalFrames) * barW;
            bodyparts.forEach((bp, idx) => {
                const coords = lbl[bp];
                if (coords && coords[0] != null) {
                    const dotY = yBase + rowH / 2 + (idx - bodyparts.length / 2) * 6;
                    tlCtx.beginPath();
                    tlCtx.arc(x, dotY, 2.5, 0, Math.PI * 2);
                    tlCtx.fillStyle = bpColor(idx);
                    tlCtx.fill();
                }
            });
        });

        // Current frame indicator
        const cx = barX + (currentFrame / totalFrames) * barW;
        tlCtx.beginPath();
        tlCtx.moveTo(cx, 4);
        tlCtx.lineTo(cx, h - 2);
        tlCtx.strokeStyle = '#ff4444';
        tlCtx.lineWidth = 2;
        tlCtx.stroke();

    }

    // ── Trial Plots (final mode) ────────────────────────
    const trialCanvases = [];  // [{canvas, ctx, trialIdx}]

    function buildTrialPlots() {
        const container = document.getElementById('trialPlotsContainer');
        if (!container || trials.length === 0) return;
        container.style.display = 'flex';
        container.innerHTML = '';
        trialCanvases.length = 0;

        for (let ti = 0; ti < trials.length; ti++) {
            const t = trials[ti];
            const row = document.createElement('div');
            row.className = 'trial-plot-row';
            row.style.height = '120px';

            const label = document.createElement('span');
            label.className = 'trial-plot-label';
            // Strip subject prefix for compact label
            let trialLabel = t.trial_name;
            const subj = sessionInfo.subject.name;
            if (trialLabel.startsWith(subj + '_')) trialLabel = trialLabel.slice(subj.length + 1);
            label.textContent = trialLabel;
            row.appendChild(label);

            const scrollWrap = document.createElement('div');
            scrollWrap.className = 'trial-plot-scroll';
            const cvs = document.createElement('canvas');
            scrollWrap.appendChild(cvs);
            row.appendChild(scrollWrap);

            const entry = { canvas: cvs, ctx: cvs.getContext('2d'), trialIdx: ti };
            trialCanvases.push(entry);

            // Click → navigate to frame (fixed 30s scale)
            cvs.addEventListener('click', (e) => {
                const rect = cvs.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const padL = 40;
                const fps = t.fps || 30;
                const pxPerFrame = (scrollWrap.clientWidth - padL - 8) / (fps * 15);
                const localFrame = Math.floor((x - padL) / pxPerFrame);
                const frame = t.start_frame + Math.max(0, Math.min(localFrame, t.frame_count - 1));
                goToFrame(frame);
            });

            container.appendChild(row);
        }

        // Resize observer on container
        const ro = new ResizeObserver(() => renderTrialPlots());
        ro.observe(container);

        renderTrialPlots();
    }

    function renderTrialPlots() {
        if (trialCanvases.length === 0) return;
        if (!distances) {
            // Clear all canvases when no distance data for selected stage
            for (const entry of trialCanvases) {
                const { canvas: cvs, ctx: c } = entry;
                const sw = cvs.parentElement;
                cvs.width = sw.clientWidth;
                cvs.height = sw.clientHeight;
                cvs.style.width = cvs.width + 'px';
                cvs.style.height = cvs.height + 'px';
                c.fillStyle = '#16213e';
                c.fillRect(0, 0, cvs.width, cvs.height);
                c.fillStyle = '#8892a0';
                c.font = '11px sans-serif';
                c.textAlign = 'center';
                c.fillText('No distance data', cvs.width / 2, cvs.height / 2);
            }
            return;
        }

        // Compute global Y range across all trials
        let globalMin = Infinity, globalMax = -Infinity;
        for (const d of distances) {
            if (d !== null && d !== undefined) {
                globalMin = Math.min(globalMin, d);
                globalMax = Math.max(globalMax, d);
            }
        }
        if (globalMin === Infinity) return;
        const range = globalMax - globalMin || 10;
        globalMin = Math.max(0, globalMin - range * 0.05);
        globalMax = userYMax !== null ? userYMax : globalMax + range * 0.05;

        // Fixed scale: 30 seconds = visible plot width
        const padL = 40, padR = 8, padT = 16, padB = 22;

        for (const entry of trialCanvases) {
            const { canvas: cvs, ctx: c, trialIdx: ti } = entry;
            const t = trials[ti];
            const scrollWrap = cvs.parentElement;
            const visibleW = scrollWrap.clientWidth;
            const h = scrollWrap.clientHeight;

            const fps = t.fps || 30;
            const frames30s = fps * 15;
            const visiblePlotW = visibleW - padL - padR;
            const pxPerFrame = visiblePlotW / frames30s;

            const trialPlotW = pxPerFrame * t.frame_count;
            const canvasW = Math.max(visibleW, Math.ceil(padL + trialPlotW + padR));

            cvs.width = canvasW;
            cvs.height = h;
            cvs.style.width = canvasW + 'px';
            cvs.style.height = h + 'px';

            const plotH = h - padT - padB;

            const fToX = (f) => padL + (f - t.start_frame) * pxPerFrame;
            const dToY = (d) => padT + ((globalMax - d) / (globalMax - globalMin)) * plotH;

            // Background
            c.fillStyle = '#16213e';
            c.fillRect(0, 0, canvasW, h);

            // Y-axis labels
            c.fillStyle = '#8892a0';
            c.font = '9px sans-serif';
            c.textAlign = 'right';
            for (let i = 0; i <= 2; i++) {
                const val = globalMin + (globalMax - globalMin) * (1 - i / 2);
                const y = padT + (i / 2) * plotH;
                c.fillText(val.toFixed(0), padL - 4, y + 3);
                c.beginPath();
                c.moveTo(padL, y);
                c.lineTo(padL + trialPlotW, y);
                c.strokeStyle = 'rgba(42, 58, 92, 0.5)';
                c.lineWidth = 0.5;
                c.stroke();
            }

            // X-axis time labels (every 5 seconds)
            c.fillStyle = '#8892a0';
            c.font = '9px sans-serif';
            c.textAlign = 'center';
            const trialDurationSec = t.frame_count / fps;
            for (let sec = 0; sec <= trialDurationSec; sec += 5) {
                const x = padL + sec * fps * pxPerFrame;
                if (x > padL + trialPlotW + 1) break;
                c.fillText(sec + 's', x, h - 2);
                c.beginPath();
                c.moveTo(x, h - padB);
                c.lineTo(x, h - padB + 3);
                c.strokeStyle = '#8892a0';
                c.lineWidth = 0.5;
                c.stroke();
            }

            // Distance trace line
            c.beginPath();
            let started = false;
            for (let f = t.start_frame; f <= t.end_frame && f < distances.length; f++) {
                const d = distances[f];
                if (d === null || d === undefined) { started = false; continue; }
                const x = fToX(f);
                const y = dToY(d);
                if (!started) { c.moveTo(x, y); started = true; }
                else { c.lineTo(x, y); }
            }
            c.strokeStyle = 'rgba(74, 158, 255, 0.8)';
            c.lineWidth = 1.5;
            c.stroke();

            // Current frame cursor (if in this trial)
            if (currentFrame >= t.start_frame && currentFrame <= t.end_frame) {
                const cx = fToX(currentFrame);
                c.beginPath();
                c.moveTo(cx, padT);
                c.lineTo(cx, h - padB);
                c.strokeStyle = '#ff4444';
                c.lineWidth = 2;
                c.stroke();

                // Dot on the line
                const curD = distances[currentFrame];
                if (curD !== null && curD !== undefined) {
                    c.beginPath();
                    c.arc(cx, dToY(curD), 4, 0, Math.PI * 2);
                    c.fillStyle = '#ff4444';
                    c.fill();
                }

                // Auto-scroll to keep cursor visible
                const scrollLeft = scrollWrap.scrollLeft;
                if (cx < scrollLeft + padL + 20 || cx > scrollLeft + visibleW - 20) {
                    scrollWrap.scrollLeft = Math.max(0, cx - visibleW / 2);
                }
            }

            // Dim border on right edge of trial data area
            c.beginPath();
            c.moveTo(padL + trialPlotW, padT);
            c.lineTo(padL + trialPlotW, h - padB);
            c.strokeStyle = 'rgba(42, 58, 92, 0.5)';
            c.lineWidth = 1;
            c.stroke();
        }
    }

    // ── Distance Trace ────────────────────────────────
    function setupDistanceTrace() {
        distCanvas.addEventListener('mousedown', onDistTraceMouseDown);
        distCanvas.addEventListener('wheel', onDistTraceWheel, { passive: false });

        const container = distCanvas.parentElement;
        const ro = new ResizeObserver(() => renderDistanceTrace());
        ro.observe(container);
    }

    /** Called after trials are loaded so we know the real fps. */
    function initDistanceTraceWindow() {
        const fps = trials.length > 0 ? trials[0].fps : 30;
        distViewFrames = Math.round(fps * 10);
        console.log(`Distance trace: ${distViewFrames} frame window (${fps} fps × 10s), ${totalFrames} total frames`);
    }

    function clampDistView() {
        const maxStart = Math.max(0, totalFrames - distViewFrames);
        distViewStart = Math.max(0, Math.min(distViewStart, maxStart));
    }

    /** Ensure currentFrame is inside the visible window, scrolling if needed. */
    function ensureFrameVisible() {
        if (distViewFrames <= 0 || totalFrames === 0) return;
        const margin = Math.round(distViewFrames * 0.15);
        if (currentFrame < distViewStart + margin) {
            distViewStart = currentFrame - margin;
        } else if (currentFrame > distViewStart + distViewFrames - margin) {
            distViewStart = currentFrame - distViewFrames + margin;
        }
        clampDistView();
    }

    function distXToFrame(clientX) {
        const rect = distCanvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const padL = 40, padR = 8;
        const plotW = rect.width - padL - padR;
        const frac = (x - padL) / plotW;
        return Math.max(0, Math.min(
            Math.floor(distViewStart + frac * distViewFrames),
            totalFrames - 1));
    }

    function onDistTraceMouseDown(e) {
        if (!distances || totalFrames === 0) return;
        e.preventDefault();
        distDragging = true;
        distDragStartX = e.clientX;
        distDragStartView = distViewStart;
        distCanvas.style.cursor = 'grabbing';

        const onMove = (ev) => {
            if (!distDragging) return;
            distAutoScroll = false;
            const rect = distCanvas.getBoundingClientRect();
            const padL = 40, padR = 8;
            const plotW = rect.width - padL - padR;
            const dx = ev.clientX - distDragStartX;
            const dFrames = Math.round((-dx / plotW) * distViewFrames);
            distViewStart = distDragStartView + dFrames;
            clampDistView();
            renderDistanceTrace();
        };

        const onUp = (ev) => {
            const moved = Math.abs(ev.clientX - distDragStartX) > 4;
            distDragging = false;
            distCanvas.style.cursor = 'pointer';
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
            if (!moved) {
                // Click — navigate to frame
                const frame = distXToFrame(ev.clientX);
                autoZoomForFrame(frame, currentSide);
                goToFrame(frame);
            }
        };

        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    }

    function onDistTraceWheel(e) {
        if (!distances || totalFrames === 0) return;
        e.preventDefault();
        distAutoScroll = false;
        // Use horizontal scroll (deltaX) if available, fall back to vertical (deltaY)
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        // Scale by actual delta magnitude for smooth trackpad scrolling
        const pxPerFrame = (distCanvas.getBoundingClientRect().width - 48) / distViewFrames;
        const frameDelta = Math.round(delta / Math.max(1, pxPerFrame));
        distViewStart += Math.sign(frameDelta) * Math.max(1, Math.abs(frameDelta));
        clampDistView();
        renderDistanceTrace();
    }

    function renderDistanceTrace() {
        if (!distCanvas || !distCtx || !distances) return;

        const container = distCanvas.parentElement;
        const w = container.clientWidth;
        const h = container.clientHeight;
        distCanvas.width = w;
        distCanvas.height = h;

        if (totalFrames === 0) return;
        const effectiveViewFrames = (distViewFrames > 0 && distViewFrames < totalFrames)
            ? distViewFrames : totalFrames;

        if (distAutoScroll) ensureFrameVisible();

        const vStart = distViewStart;
        const vEnd = Math.min(vStart + effectiveViewFrames, totalFrames);

        const padL = 40, padR = 8, padT = 16, padB = 14;
        const plotW = w - padL - padR;
        const plotH = h - padT - padB;

        const fToX = (f) => padL + ((f - vStart) / effectiveViewFrames) * plotW;

        // Use global data range so Y-axis stays constant while scrolling
        let minD = Infinity, maxD = -Infinity;
        for (const d of distances) {
            if (d !== null && d !== undefined) {
                minD = Math.min(minD, d);
                maxD = Math.max(maxD, d);
            }
        }
        if (minD === Infinity) return;

        const range = maxD - minD || 10;
        minD = Math.max(0, minD - range * 0.05);
        maxD = userYMax !== null ? userYMax : maxD + range * 0.05;

        const dToY = (d) => padT + ((maxD - d) / (maxD - minD)) * plotH;

        // Background
        distCtx.fillStyle = '#16213e';
        distCtx.fillRect(0, 0, w, h);

        // Y-axis labels
        distCtx.fillStyle = '#8892a0';
        distCtx.font = '9px sans-serif';
        distCtx.textAlign = 'right';
        const nTicks = 3;
        for (let i = 0; i <= nTicks; i++) {
            const val = minD + (maxD - minD) * (1 - i / nTicks);
            const y = padT + (i / nTicks) * plotH;
            distCtx.fillText(val.toFixed(0), padL - 4, y + 3);
            distCtx.beginPath();
            distCtx.moveTo(padL, y);
            distCtx.lineTo(w - padR, y);
            distCtx.strokeStyle = 'rgba(42, 58, 92, 0.5)';
            distCtx.lineWidth = 0.5;
            distCtx.stroke();
        }

        // Trial boundaries
        for (const t of trials) {
            if (t.start_frame >= vStart && t.start_frame < vEnd) {
                const x = fToX(t.start_frame);
                distCtx.beginPath();
                distCtx.moveTo(x, padT);
                distCtx.lineTo(x, h - padB);
                distCtx.strokeStyle = 'rgba(42, 58, 92, 0.8)';
                distCtx.lineWidth = 1;
                distCtx.stroke();
            }
        }

        // Draw single distance line
        distCtx.beginPath();
        let started = false;
        for (let f = Math.max(0, vStart - 1); f < vEnd + 1 && f < distances.length; f++) {
            const d = distances[f];
            if (d === null || d === undefined) {
                started = false;
                continue;
            }
            const x = fToX(f);
            const y = dToY(d);
            if (!started) {
                distCtx.moveTo(x, y);
                started = true;
            } else {
                distCtx.lineTo(x, y);
            }
        }
        distCtx.strokeStyle = 'rgba(74, 158, 255, 0.7)';
        distCtx.lineWidth = 1.5;
        distCtx.stroke();

        // Green dots/circles for correction frames (refine mode only)
        if (isRefine) {
            correctionFrames.forEach(key => {
                const [frameStr, side] = key.split('_');
                const frame = parseInt(frameStr);
                if (frame < vStart || frame >= vEnd) return;

                const x = fToX(frame);
                let y;
                if (distances && frame < distances.length && distances[frame] !== null) {
                    y = Math.max(padT + 4, Math.min(padT + plotH - 4, dToY(distances[frame])));
                } else {
                    y = padT + plotH * 0.5;
                }

                distCtx.beginPath();
                distCtx.arc(x, y, 4, 0, Math.PI * 2);
                distCtx.strokeStyle = '#00cc44';
                distCtx.lineWidth = 1.5;
                if (v2Excludes.has(key)) {
                    distCtx.stroke(); // empty circle = excluded from v2 training
                } else {
                    distCtx.fillStyle = '#00cc44';
                    distCtx.fill(); // filled dot = included in v2 training
                    distCtx.stroke();
                }
            });
        }

        // Camera ticks for frames with manual corrections (visible only)
        labels.forEach((lbl, key) => {
            const [frameStr, side] = key.split('_');
            const frame = parseInt(frameStr);
            if (frame < vStart || frame >= vEnd) return;
            const x = fToX(frame);

            // Camera tick at bottom edge
            const camIdx = cameraNames.indexOf(side);
            const tickY = h - padB - (camIdx === 0 ? 6 : 1);
            distCtx.fillStyle = camIdx === 0 ? '#ff4444' : '#4a9eff';
            distCtx.fillRect(x - 0.5, tickY, 1.5, 4);
        });

        // Current frame cursor
        const cx = fToX(currentFrame);
        distCtx.beginPath();
        distCtx.moveTo(cx, padT);
        distCtx.lineTo(cx, h - padB);
        distCtx.strokeStyle = '#ff4444';
        distCtx.lineWidth = 1.5;
        distCtx.stroke();

        // Show value at current frame on primary trace
        if (distances) {
            const curD = distances[currentFrame];
            if (curD !== null && curD !== undefined) {
                const y = dToY(curD);
                distCtx.beginPath();
                distCtx.arc(cx, y, 4, 0, Math.PI * 2);
                distCtx.fillStyle = '#ff4444';
                distCtx.fill();
            }
        }

        // Scrollbar
        const sbY = h - 3;
        const sbH = 3;
        distCtx.fillStyle = 'rgba(42, 58, 92, 0.5)';
        distCtx.fillRect(padL, sbY, plotW, sbH);
        const thumbL = padL + (vStart / totalFrames) * plotW;
        const thumbW = Math.max(6, (effectiveViewFrames / totalFrames) * plotW);
        distCtx.fillStyle = 'rgba(74, 158, 255, 0.5)';
        distCtx.fillRect(thumbL, sbY, thumbW, sbH);
    }

    // ── Subject navigation ─────────────────────────────
    function currentSessionType() {
        if (isFinal) return 'final';
        if (isCorrections) return 'corrections';
        if (isRefine) return 'refine';
        return 'initial';
    }

    async function switchSubject(subjectId) {
        if (subjectId === currentSubjectId) return;
        try {
            const session = await API.post(`/api/labeling/${subjectId}/sessions`, {
                session_type: currentSessionType(),
            });
            window.location.href = `/labeling?session=${session.id}`;
        } catch (e) {
            alert('Could not switch subject: ' + e.message);
            // Reset dropdown to current
            document.getElementById('subjectSelect').value = currentSubjectId;
        }
    }

    function subjectIndex() {
        return allSubjects.findIndex(s => s.id === currentSubjectId);
    }

    function prevSubject() {
        const idx = subjectIndex();
        if (idx > 0) switchSubject(allSubjects[idx - 1].id);
    }

    function nextSubject() {
        const idx = subjectIndex();
        if (idx >= 0 && idx < allSubjects.length - 1) switchSubject(allSubjects[idx + 1].id);
    }

    function updateSubjectNavButtons() {
        const idx = subjectIndex();
        const prevBtn = document.getElementById('prevSubjectBtn');
        const nextBtn = document.getElementById('nextSubjectBtn');
        if (prevBtn) prevBtn.disabled = idx <= 0;
        if (nextBtn) nextBtn.disabled = idx >= allSubjects.length - 1;
    }

    // ── V2 training toggle (refine mode) ─────────────
    function toggleV2Training() {
        if (!isRefine) return;
        const key = `${currentFrame}_${currentSide}`;
        if (!correctionFrames.has(key)) return; // not a correction frame — nothing to toggle
        if (v2Excludes.has(key)) {
            v2Excludes.delete(key);
        } else {
            v2Excludes.add(key);
        }
        updateV2TrainingBtn();
        renderDistanceTrace();
    }

    function updateV2TrainingBtn() {
        const btn = document.getElementById('v2ToggleBtn');
        if (!btn || !isRefine) return;
        const key = `${currentFrame}_${currentSide}`;
        if (!correctionFrames.has(key)) {
            btn.style.display = 'none';
            return;
        }
        btn.style.display = '';
        if (v2Excludes.has(key)) {
            btn.textContent = 'Training: Off';
            btn.style.background = '';
            btn.style.color = 'var(--text-muted)';
        } else {
            btn.textContent = 'Training: On \u2713';
            btn.style.background = 'rgba(0,204,68,0.15)';
            btn.style.color = '#00cc44';
        }
    }

    // ── Public API ────────────────────────────────────
    return {
        init,
        nextFrame, prevFrame, nextLabel, prevLabel,
        nextGap, prevGap, acceptMergedLabels,
        toggleSide, togglePlay, resetZoom, cycleReviewMode,
        saveLabels, commitSession, saveCorrectionsOnly,
        toggleV2Training,
        prevSubject, nextSubject,
    };
})();

// Init on page load
document.addEventListener('DOMContentLoaded', () => labeler.init());
