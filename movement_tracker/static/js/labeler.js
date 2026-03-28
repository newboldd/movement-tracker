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
    let cameraMode = 'stereo';  // 'single', 'stereo', or 'multicam'

    let currentFrame = 0;
    let currentSide = 'OS';
    let playing = false;
    let playTimer = null;
    let playbackRate = 1;

    // Playback speed presets for the slider
    const SPEED_PRESETS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 16, 18, 60, 120, 240];

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
    // Events tab: whole-file priority (no frame-level merge), mediapipe before labels
    const EVENTS_STAGE_CHAIN = ['corrections', 'refine', 'dlc', 'mp', 'labels'];

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

    // Undo / Redo stacks.
    // Label entries:  { type: 'label',  key, bp, prev (coords or null), frame }
    // Events entries: { type: 'events', prev: snapshot, frame }
    const undoStack = [];
    const redoStack = [];
    const MAX_UNDO = 50;

    // Review mode: null = all bodyparts, or bodypart name for focused review
    let reviewBp = null;

    // Video element for smooth playback
    let videoEl = null;
    let videoPlaying = false;
    let currentTrialIdx = -1; // which trial the video element is loaded with
    let currentVideoSide = ''; // which camera side the video element is loaded with
    let videoFrameMode = false; // true when video element is the frame source (vs JPEG)

    // MediaPipe bounding-box crop state
    let mpCropMode = false;            // bounding box editing active
    let mpCropBoxes = {};              // {trialIdx: {cam: {x1, y1, x2, y2}}} loaded from DB
    let mpCropEditBox = null;          // current editing box {x1, y1, x2, y2} (unsaved)
    let mpCropDragHandle = null;       // which handle is being dragged
    let mpCropDragStart = null;        // {mx, my, box: {...}} at drag start
    let mpCropAdjusted = {};           // {cam: bool} whether user manually dragged each camera's box
    let mpHasMediapipe = {};           // {trialIdx: bool} whether mediapipe labels exist per trial
    let mpRunVisible = { current: true };  // {current: bool, 1: bool, 2: bool, ...} toggle visibility

    // Frame display mode: 'frame' | 'time' | 'both'
    let frameDisplayMode = 'frame';

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

    // Stable Y-axis range computed once from cleanest available data (corrections > others)
    // Null means not yet computed; recomputed on subject change.
    let stableDistRange = null; // { min, max }

    // Events mode state
    let isEvents = false;
    // Dynamic event types (loaded from settings via session info)
    let EVENT_TYPES = ['open', 'peak', 'close', 'pause'];
    let EVENT_COLORS = { open: '#00cc44', peak: '#ffcc00', close: '#ff4444', pause: '#cc66ff' };
    let EVENT_SHORTCUTS = { open: '1', peak: '2', close: '3', pause: '4' };
    const AUTO_DETECT_TYPES = ['open', 'peak', 'close']; // always fixed
    let eventMarkers = {};
    let savedEventFrames = {};
    let eventVisibility = {};

    function _initEventState() {
        eventMarkers = {};
        savedEventFrames = {};
        eventVisibility = {};
        EVENT_TYPES.forEach(t => {
            eventMarkers[t] = [];
            savedEventFrames[t] = new Set();
            eventVisibility[t] = true;
        });
    }
    _initEventState();

    // Per-trial event filtering
    let currentEventTrialIdx = 0;
    // Per-trial metrics cache: { trialIdx: {distance, reversal, motion_ssd, per_cam_ssd} }
    let metricsCache = {};
    let metricsLoading = new Set();  // trial indices currently being computed
    // Current detect focus mode: 'all', 'open', 'peak', 'close'
    let detectFocus = 'all';

    // Prefetch cache
    const imageCache = new Map();
    const PREFETCH_AHEAD = 3;

    // Point detection radius
    const HIT_RADIUS = 12;
    const POINT_RADIUS = 6;

    // ── Preferences persistence ───────────────────────
    function savePreferences() {
        const prefs = {
            playbackRate: playbackRate,
            selectedStage: selectedStage,
            eventVisibility: eventVisibility,
        };
        localStorage.setItem('movement_tracker_prefs', JSON.stringify(prefs));
    }

    function restorePreferences() {
        // Only restore speed/stage when switching subjects within the labeling screen.
        // Fresh navigation (from dashboard etc.) uses defaults: 1x speed, 'auto' label source.
        const isSubjectSwitch = sessionStorage.getItem('dlc_subjectSwitch');
        if (isSubjectSwitch) {
            sessionStorage.removeItem('dlc_subjectSwitch');
        }

        try {
            const saved = localStorage.getItem('movement_tracker_prefs')
                || localStorage.getItem('hand_tracker_prefs')
                || localStorage.getItem('dlc_labeler_prefs');
            if (saved) {
                const prefs = JSON.parse(saved);
                if (isSubjectSwitch && prefs.playbackRate) {
                    playbackRate = prefs.playbackRate;
                    const playbackSlider = document.getElementById('playbackRate');
                    if (playbackSlider) {
                        const index = SPEED_PRESETS.indexOf(playbackRate);
                        if (index !== -1) {
                            playbackSlider.value = index;
                        }
                        const speedDisplay = document.getElementById('playbackRateDisplay');
                        if (speedDisplay) {
                            speedDisplay.textContent = `${playbackRate}x`;
                        }
                    }
                }
                if (isSubjectSwitch && prefs.selectedStage) {
                    selectedStage = prefs.selectedStage;
                }
                if (prefs.eventVisibility) {
                    eventVisibility = Object.assign({}, eventVisibility, prefs.eventVisibility);
                    // Sync dynamic checkboxes
                    EVENT_TYPES.forEach(t => {
                        const cb = document.getElementById(`showEvent_${t}`);
                        if (cb) cb.checked = eventVisibility[t] !== false;
                    });
                }
            }
        } catch (e) {
            console.log('Could not restore preferences:', e);
        }
    }

    // ── Init ──────────────────────────────────────────
    // Detect whether this is the MediaPipe page or the DLC page
    const isMediaPipePage = window.location.pathname === '/mediapipe';

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

        // Format speed: show fractions for slow speeds (1/100x, 1/50x, etc.)
        function _formatSpeed(rate) {
            if (rate < 1 && rate > 0) {
                const denom = Math.round(1 / rate);
                return `1/${denom}x`;
            }
            return `${rate}x`;
        }

        // Playback speed slider
        const speedSlider = document.getElementById('playbackRate');
        const speedDisplay = document.getElementById('playbackRateDisplay');
        if (speedSlider && speedDisplay) {
            // Update display and playbackRate when slider moves
            speedSlider.addEventListener('input', () => {
                const index = parseInt(speedSlider.value);
                playbackRate = SPEED_PRESETS[index];
                speedDisplay.textContent = _formatSpeed(playbackRate);
            });
            // Initialize slider to match current playbackRate
            const currentIndex = SPEED_PRESETS.indexOf(playbackRate);
            if (currentIndex !== -1) {
                speedSlider.value = currentIndex;
            }
            // Initialize display with current value
            speedDisplay.textContent = _formatSpeed(playbackRate);
        }

        // Threshold param inputs → re-render distance metric canvas on change
        ['paramMinPeakHeight', 'paramValleyThresh'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', () => {
                    const cached = metricsCache[currentEventTrialIdx];
                    if (cached) renderMetricCanvas('distPlotCanvas', cached.distance, '#4a9eff', 'Distance');
                });
            }
        });

        loadSession();
    }

    async function loadSession() {
        try {
            currentTrialIdx = -1;  // force video reload (e.g. after blur setting change)
            currentVideoSide = '';
            sessionInfo = await API.get(`/api/labeling/sessions/${sessionId}/info`);
            trials = sessionInfo.trials;
            totalFrames = sessionInfo.total_frames;

            // Get dynamic bodyparts, camera names, and event types from session info
            if (sessionInfo.bodyparts) bodyparts = sessionInfo.bodyparts;
            if (sessionInfo.camera_names) cameraNames = sessionInfo.camera_names;
            if (sessionInfo.event_types && sessionInfo.event_types.length > 0) {
                EVENT_TYPES = sessionInfo.event_types.map(et => et.name);
                EVENT_COLORS = {};
                EVENT_SHORTCUTS = {};
                sessionInfo.event_types.forEach(et => {
                    EVENT_COLORS[et.name] = et.color;
                    EVENT_SHORTCUTS[et.name] = et.shortcut;
                });
                _initEventState();
            }
            if (sessionInfo.committed_frame_count) committedFrameCount = sessionInfo.committed_frame_count;
            if (sessionInfo.camera_mode) cameraMode = sessionInfo.camera_mode;

            // Adjust cameraNames and currentSide based on camera_mode
            if (cameraMode === 'single') {
                // Single camera: collapse to one entry
                cameraNames = [cameraNames[0] || 'OS'];
                currentSide = cameraNames[0];
            } else if (cameraMode === 'multicam' && trials.length > 0
                && trials[0].cameras && trials[0].cameras.length > 0) {
                // Multicam: use actual camera names from trial data
                cameraNames = trials[0].cameras.map(c => c.name);
                currentSide = cameraNames[0];
            } else {
                currentSide = cameraNames[0] || 'OS';
            }

            isRefine = sessionInfo.session && sessionInfo.session.session_type === 'refine';
            isCorrections = sessionInfo.session && sessionInfo.session.session_type === 'corrections';
            isEvents = sessionInfo.session && sessionInfo.session.session_type === 'events';
            isFinal = sessionInfo.session && (sessionInfo.session.session_type === 'final' || isEvents);

            // Populate subject navigation dropdown
            currentSubjectId = sessionInfo.subject.id;
            // Persist subject, mode, and session ID for cross-page navigation
            sessionStorage.setItem('dlc_lastSubjectId', String(currentSubjectId));
            if (typeof setLastSubject === 'function') setLastSubject(currentSubjectId);
            if (typeof setNavState === 'function') setNavState({ subjectId: currentSubjectId });
            sessionStorage.setItem(`dlc_labelTab_${currentSubjectId}`, currentSessionType());
            sessionStorage.setItem('dlc_lastSessionId', String(sessionId));
            // Update nav links with current subject/session context
            const resultsLink = document.getElementById('resultsLink');
            if (resultsLink) resultsLink.href = `/results?subject=${currentSubjectId}&from=labeling`;
            // Keep "Labeling" nav link pointing at current session so navigating
            // away and back doesn't switch subjects
            const labelingLink = document.querySelector('nav a[href*="labeling"]');
            if (labelingLink) labelingLink.href = `/labeling?session=${sessionId}`;
            const typeLabel = document.getElementById('sessionTypeLabel');
            if (isEvents) typeLabel.textContent = 'Events:';
            else if (isFinal) typeLabel.textContent = 'Final:';
            else if (isCorrections) typeLabel.textContent = 'Corrections:';
            else if (isRefine) typeLabel.textContent = 'Refine:';
            else typeLabel.textContent = 'DLC:';

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
            const refineBtn = document.getElementById('refineBtn');
            if (isEvents || isFinal) {
                if (mainCommitBtn) mainCommitBtn.style.display = 'none';
            } else if (isRefine) {
                // Legacy refine sessions: show same UI as corrections with refine
                if (mainCommitBtn) mainCommitBtn.style.display = 'none';
                if (saveCorrectionsBtn) saveCorrectionsBtn.style.display = '';
                if (refineBtn) refineBtn.style.display = '';
            } else if (isCorrections) {
                if (mainCommitBtn) mainCommitBtn.style.display = 'none';
                if (saveCorrectionsBtn) saveCorrectionsBtn.style.display = '';
                if (refineBtn) refineBtn.style.display = '';
            }

            // Update sidebar with dynamic shortcuts
            updateShortcutsSidebar();

            // ── Camera: hide switch button for single, update label ──
            const sideBtn = document.getElementById('sideToggle');
            const camLabel = document.getElementById('cameraLabel');
            if (cameraMode === 'single') {
                if (sideBtn) sideBtn.style.display = 'none';
                if (camLabel) camLabel.style.display = 'none';
            } else {
                if (sideBtn) sideBtn.style.display = '';
                if (camLabel) { camLabel.style.display = ''; camLabel.textContent = currentSide; }
            }

            // ── Load saved crop boxes from session info ──
            if (sessionInfo.crop_boxes) {
                mpCropBoxes = {};
                for (const [ti, cams] of Object.entries(sessionInfo.crop_boxes)) {
                    mpCropBoxes[parseInt(ti)] = cams;
                }
            }

            // ── Page-specific UI: MediaPipe page vs DLC page ──
            const mpCropSection = document.getElementById('mpCropSection');
            const modeSwitcher = document.getElementById('modeSwitcher');
            const actionsSection = mainCommitBtn ? mainCommitBtn.parentElement : null;

            if (isMediaPipePage) {
                // MediaPipe page: show MP controls, hide DLC-specific UI
                if (mpCropSection) mpCropSection.style.display = 'block';
                if (modeSwitcher) modeSwitcher.style.display = 'none';
                if (actionsSection) actionsSection.style.display = 'none';
                // Update session type label
                const typeLabel = document.getElementById('sessionTypeLabel');
                if (typeLabel) typeLabel.textContent = 'MediaPipe:';
                // Set active nav link
                document.querySelectorAll('nav a').forEach(a => {
                    a.classList.toggle('active', a.getAttribute('href') === '/mediapipe-select');
                });
                // Slightly enlarge distance trace for comparison
                const distContainer = document.getElementById('distanceTraceContainer');
                if (distContainer) distContainer.style.height = '180px';
            } else {
                // DLC page: hide MP controls, show DLC UI
                if (mpCropSection) mpCropSection.style.display = 'none';
            }

            // All tabs are always visible regardless of subject stage

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

                // selectedStage defaults to 'auto'; restorePreferences() may override
                // it for intra-labeling subject switches.

                // Load all stage data and merge distances
                await loadAllStages();
                computeStableDistRange(); // compute once from cleanest source
                if (isRefine) { computeCorrectionFrames(); updateLabelCount(); }
                populateStageSelector();

                // Events mode: show distance trace, load events, show events panel
                if (isEvents) {
                    const timelineContainer = document.querySelector('.timeline-container');
                    if (timelineContainer) timelineContainer.style.display = 'none';
                    buildEventsPanel(); // populate dynamic buttons/toggles
                    const eventsPanel = document.getElementById('eventsPanel');
                    if (eventsPanel) eventsPanel.style.display = 'block';
                    // Load saved events
                    loadEvents();
                // Final mode: hide timeline/distance trace, build trial plots at bottom
                } else if (isFinal) {
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

                // Compute per-trial mediapipe availability for Run vs Re-run button text
                _computeMpHasMediapipe();

                // Show "Clear history" button if run history exists (MediaPipe page only)
                const clearHistBtn = document.getElementById('mpClearHistoryBtn');
                if (clearHistBtn) {
                    clearHistBtn.style.display =
                        (isMediaPipePage && mpLabels && mpLabels.run_history && mpLabels.run_history.length > 0)
                        ? 'block' : 'none';
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

                // Load committed labels for timeline display (all session types)
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

            // Always initialize the distance trace window size from fps
            initDistanceTraceWindow();

            // For initial/refine modes: compute stable range from available distances now
            if (!stableDistRange && distances && distances.some(d => d !== null)) {
                computeStableDistRange();
            }

            // Show distance trace if we have data; hide timeline to save space
            // (in final mode: trial plots handle this instead; events mode always shows trace)
            if (distances && distances.some(d => d !== null) && (!isFinal || isEvents)) {
                const traceContainer = document.getElementById('distanceTraceContainer');
                if (traceContainer) traceContainer.style.display = 'block';
                const timelineContainer = document.querySelector('.timeline-container');
                if (timelineContainer) timelineContainer.style.display = 'none';
                const ymaxContainer = document.getElementById('ymaxContainer');
                if (ymaxContainer) ymaxContainer.style.display = 'flex';
            }

            // Restore user preferences (playback speed, label source)
            restorePreferences();

            // Update stage selector with restored selectedStage
            const stageSelect = document.getElementById('stageSelector');
            if (stageSelect && availableStages.length > 0) {
                if (selectedStage && availableStages.includes(selectedStage)) {
                    stageSelect.value = selectedStage;
                    updateLabelNavButtons();
                    computeMergedDistances();
                    if (isFinal && !isEvents) computeFinalCropBoxes();
                }
            }

            recomputeCameraShift();
            updateLabelCount();
            updateLabelNavButtons();
            // Restore frame/zoom only for tab switches (same subject);
            // new subjects always start at frame 0.
            const isModeSwitch = sessionStorage.getItem('dlc_modeSwitch') === '1';
            sessionStorage.removeItem('dlc_modeSwitch');
            const navRestored = isModeSwitch && restoreNavState();
            // Cross-page nav state: restore frame/side if subject matches and not a mode switch
            let crossPageRestored = false;
            if (!navRestored && typeof getNavState === 'function') {
                const nav = getNavState();
                if (nav.subjectId === currentSubjectId) {
                    if (nav.frame != null && nav.frame >= 0 && nav.frame < totalFrames) {
                        currentFrame = nav.frame;
                        crossPageRestored = true;
                    }
                    if (nav.side && cameraNames.includes(nav.side)) {
                        currentSide = nav.side;
                        const _camLabel = document.getElementById('cameraLabel');
                        if (_camLabel) _camLabel.textContent = currentSide;
                    }
                }
            }
            await goToFrame((navRestored || crossPageRestored) ? currentFrame : 0);

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

    // ── Video-element frame rendering ─────────────────
    // Uses the browser's native video decoder instead of OpenCV JPEG extraction,
    // which avoids the 1-frame seek offset that OpenCV has for many H.264 videos.
    async function tryRenderVideoFrame(frame) {
        // Don't interfere with active playback (videoDrawLoop or fallbackPlay)
        if (!videoEl || playing) return false;

        const trialIdx = getTrialForFrame(frame);
        const trial = trials[trialIdx];
        if (!trial) return false;

        // Load correct trial's video into the element if needed.
        // Also reload when the camera side changes (multicam subjects have separate video files per camera).
        let justLoaded = false;
        if (currentTrialIdx !== trialIdx || currentVideoSide !== currentSide) {
            const videoUrl = `/api/labeling/sessions/${sessionId}/video?trial=${trialIdx}&side=${encodeURIComponent(currentSide)}&_=${Date.now()}`;
            videoEl.src = videoUrl;
            currentTrialIdx = trialIdx;
            currentVideoSide = currentSide;
            // Wait for video metadata to load so we can seek accurately
            const loaded = await new Promise(resolve => {
                if (videoEl.readyState >= 1) { resolve(true); return; }
                const timer = setTimeout(() => { resolve(false); }, 5000); // 5 second timeout
                const onLoaded = () => { clearTimeout(timer); videoEl.removeEventListener('error', onError); resolve(true); };
                const onError = () => { clearTimeout(timer); videoEl.removeEventListener('loadedmetadata', onLoaded); resolve(false); };
                videoEl.addEventListener('loadedmetadata', onLoaded, { once: true });
                videoEl.addEventListener('error', onError, { once: true });
            });
            if (!loaded) return false; // Failed to load, fall back to JPEG
            justLoaded = true;
        }
        if (videoEl.readyState < 1) return false; // still loading metadata

        // Seek to the target frame's timestamp.
        // Use the midpoint of the frame duration (frame + 0.5) / fps to ensure
        // the video element reliably shows the correct frame despite keyframe seeking
        // or floating-point precision issues.
        // frame_offset compensates for videos where OpenCV decodes extra pre-roll
        // frames (negative-PTS disposable packets) that the browser skips.
        const localFrame = frame - trial.start_frame;
        const frameOffset = trial.frame_offset || 0;
        const halfFrame = 0.5 / trial.fps;
        const targetTime = Math.max(0, (localFrame - frameOffset + 0.5) / trial.fps);

        // Always seek after freshly loading a video — loadedmetadata doesn't
        // decode any frame, so drawImage would paint blank without an explicit seek.
        if (justLoaded || Math.abs(videoEl.currentTime - targetTime) > halfFrame) {
            videoEl.currentTime = targetTime;
            await new Promise(resolve => {
                videoEl.addEventListener('seeked', resolve, { once: true });
                setTimeout(resolve, 2000); // timeout fallback — don't block forever
            });
        }

        // Draw the video element to canvas, same logic as videoDrawLoop
        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        canvas.width = cw;
        canvas.height = ch;
        ctx.clearRect(0, 0, cw, ch);

        const vw = videoEl.videoWidth;
        const vh = videoEl.videoHeight;
        if (vw > 0 && vh > 0) {
            let sx, sw;
            if (cameraMode === 'multicam' || cameraMode === 'single') {
                // Full frame — no stereo cropping
                sx = 0; sw = vw;
            } else {
                // Stereo: crop to left or right half
                const midline = Math.floor(vw / 2);
                if (cameraNames.length >= 2 && currentSide === cameraNames[1]) {
                    sx = midline; sw = vw - midline;
                } else {
                    sx = 0; sw = midline;
                }
            }
            imgW = sw;
            imgH = vh;
            if (!hasUserZoom && !mpCropMode) {
                if (isMediaPipePage) {
                    fitImage();
                    hasUserZoom = true;
                } else if (isFinal && !isEvents && finalCropBoxes) {
                    zoomToCropBox();
                    hasUserZoom = true;
                } else if (isFinal && !isEvents) {
                    fitImage();
                    hasUserZoom = true;
                } else {
                    if (!autoZoomForFrame(frame, currentSide)) fitImage();
                }
            }
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            ctx.drawImage(videoEl, sx, 0, sw, vh, 0, 0, sw, vh);
            ctx.restore();
        }

        drawLabelsOverlay();
        if (mpCropMode && mpCropEditBox) drawMpCropOverlay();
        videoFrameMode = true;
        return true;
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
        if (typeof setNavState === 'function') setNavState({ frame: currentFrame, trialIdx: getTrialForFrame(currentFrame) });
        distAutoScroll = true; // frame navigation re-enables auto-scroll
        _updateMpButtonText();

        // Prefer video element for accurate frame display.
        // OpenCV's cap.set(POS_FRAMES, N) is unreliable for many H.264 videos —
        // it can overshoot by 1 frame or accumulate drift at dropped frames.
        // The browser's native video decoder (used in play mode) is the ground truth.
        const renderedFromVideo = await tryRenderVideoFrame(frame);

        if (!renderedFromVideo) {
            // Fallback: load JPEG from backend (used before the video element is
            // ready, or when play mode is active)
            videoFrameMode = false;
            try {
                currentImage = await loadImage(frame, currentSide);
                imgW = currentImage.width;
                imgH = currentImage.height;
                if (!hasUserZoom && !mpCropMode) {
                    if (isMediaPipePage) {
                        fitImage();
                        hasUserZoom = true;
                    } else if (isFinal && !isEvents && finalCropBoxes) {
                        // Final mode: zoom to crop box, then lock
                        zoomToCropBox();
                        hasUserZoom = true;
                    } else if (isFinal && !isEvents) {
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
        }

        updateFrameDisplay();
        renderTimeline();
        renderDistanceTrace();
        if (isRefine) updateV2TrainingBtn();

        // Trial-follows-frame: auto-sync currentEventTrialIdx when frame crosses trial boundary
        if (isEvents) {
            const newTrial = getTrialForFrame(currentFrame);
            if (newTrial !== currentEventTrialIdx) {
                currentEventTrialIdx = newTrial;
                const trialLabel = document.getElementById('trialSelectorLabel');
                if (trialLabel && trials[currentEventTrialIdx]) {
                    trialLabel.textContent = trials[currentEventTrialIdx].trial_name;
                }
                updateEventCounts();
                // Re-render distance trace for the new trial
                renderDistanceTrace();
                // If detect modal is open, refresh metric plots for new trial
                const overlay = document.getElementById('detectModalOverlay');
                if (overlay && overlay.classList.contains('active')) {
                    const modalTrial = document.getElementById('detectModalTrial');
                    if (modalTrial && trials[currentEventTrialIdx]) {
                        modalTrial.textContent = `(${trials[currentEventTrialIdx].trial_name})`;
                    }
                    showMetricPlotsForCurrentTrial();
                }
            }
        }
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
        if (isFinal && !isEvents) computeFinalCropBoxes();
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
        if ((!isFinal && !isRefine && !isEvents) || availableStages.length === 0) return;

        container.style.display = 'block';
        select.innerHTML = '';

        // Only show "Auto (priority merge)" option in refine mode, not final/events
        if (!isFinal || isEvents) {
            // Events mode: allow "auto" so user can see best available distance data
            // Refine mode: auto = merged priority chain
            // Other modes: auto available
            select.innerHTML = isEvents
                ? '<option value="auto">Auto (best distance)</option>'
                : '<option value="auto">Auto (priority merge)</option>';
        } else {
            select.innerHTML = '';
        }

        // Add available stages (events mode uses its own priority order)
        for (const stage of (isEvents ? EVENTS_STAGE_CHAIN : STAGE_CHAIN)) {
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
            if (isFinal && !isEvents) computeFinalCropBoxes();
            render();
            if (isFinal && !isEvents) renderTrialPlots();
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

        // Restore persisted excludes, keeping only keys still in correctionFrames
        const saved = localStorage.getItem(`v2excludes_${sessionId}`);
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                parsed.forEach(k => { if (correctionFrames.has(k)) v2Excludes.add(k); });
            } catch (_) {}
        }
    }

    function saveV2Excludes() {
        localStorage.setItem(`v2excludes_${sessionId}`, JSON.stringify([...v2Excludes]));
    }

    function computeMergedDistances() {
        /** Build a single merged distance array from the selected stage or
         *  priority chain.
         *
         *  Events mode (auto): pick the first whole stage that has distances
         *  in priority order — no frame-level gap-filling across files.
         *
         *  Other modes / explicit stage: frame-level priority merge so labels
         *  from different stages can complement each other. */
        distances = null;

        if (selectedStage !== 'auto') {
            // Explicit stage — use only that stage's distances
            const sd = stageData[selectedStage];
            if (sd && sd.distances && sd.distances.some(d => d !== null)) {
                distances = sd.distances;
            }
        } else if (isEvents) {
            // Events auto: whole-file priority — first stage with distances wins
            const chain = EVENTS_STAGE_CHAIN.filter(s => availableStages.includes(s));
            for (const s of chain) {
                const sd = stageData[s];
                if (sd && sd.distances && sd.distances.some(d => d !== null)) {
                    distances = sd.distances;
                    break;
                }
            }
        } else {
            // Non-events auto: frame-level merge across priority chain
            const stagesToUse = STAGE_CHAIN.filter(s => availableStages.includes(s));

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
            }
        }

        if (distances) {
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

        if (videoFrameMode && videoEl && videoEl.readyState >= 2 && !videoPlaying) {
            // Video element is the frame source — redraw from the paused video.
            // This keeps labels correctly aligned during zoom/pan after frame-by-frame nav.
            const vw = videoEl.videoWidth;
            const vh = videoEl.videoHeight;
            if (vw > 0 && vh > 0) {
                let sx, sw;
                if (cameraMode === 'multicam' || cameraMode === 'single') {
                    sx = 0; sw = vw;
                } else {
                    const midline = Math.floor(vw / 2);
                    if (cameraNames.length >= 2 && currentSide === cameraNames[1]) {
                        sx = midline; sw = vw - midline;
                    } else {
                        sx = 0; sw = midline;
                    }
                }
                ctx.save();
                ctx.translate(offsetX, offsetY);
                ctx.scale(scale, scale);
                ctx.drawImage(videoEl, sx, 0, sw, vh, 0, 0, sw, vh);
                ctx.restore();
            }
        } else if (currentImage) {
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            ctx.drawImage(currentImage, 0, 0);
            ctx.restore();
        }

        drawLabelsOverlay();
        if (mpCropMode && mpCropEditBox) drawMpCropOverlay();
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

    // ── MediaPipe crop box overlay ───────────────────
    function drawMpCropOverlay() {
        const box = mpCropEditBox;
        if (!box) return;
        const cw = canvas.width;
        const ch = canvas.height;

        // Convert box corners to screen coords
        const s1 = imageToScreen(box.x1, box.y1);
        const s2 = imageToScreen(box.x2, box.y2);
        const sx1 = s1.x, sy1 = s1.y, sx2 = s2.x, sy2 = s2.y;

        // Dim outside the crop box
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        // Top
        ctx.fillRect(0, 0, cw, sy1);
        // Bottom
        ctx.fillRect(0, sy2, cw, ch - sy2);
        // Left
        ctx.fillRect(0, sy1, sx1, sy2 - sy1);
        // Right
        ctx.fillRect(sx2, sy1, cw - sx2, sy2 - sy1);

        // Dashed border
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
        ctx.setLineDash([]);

        // Draw 8 handles: 4 corners + 4 edge midpoints
        const handles = _getMpCropHandles(sx1, sy1, sx2, sy2);
        const handleSize = 6;
        for (const h of handles) {
            ctx.fillStyle = '#00ff88';
            ctx.fillRect(h.x - handleSize, h.y - handleSize, handleSize * 2, handleSize * 2);
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.strokeRect(h.x - handleSize, h.y - handleSize, handleSize * 2, handleSize * 2);
        }
    }

    function _getMpCropHandles(sx1, sy1, sx2, sy2) {
        const mx = (sx1 + sx2) / 2;
        const my = (sy1 + sy2) / 2;
        return [
            { name: 'nw', x: sx1, y: sy1 },
            { name: 'n',  x: mx,  y: sy1 },
            { name: 'ne', x: sx2, y: sy1 },
            { name: 'e',  x: sx2, y: my  },
            { name: 'se', x: sx2, y: sy2 },
            { name: 's',  x: mx,  y: sy2 },
            { name: 'sw', x: sx1, y: sy2 },
            { name: 'w',  x: sx1, y: my  },
        ];
    }

    function _mpCropHitTest(sx, sy) {
        /** Hit-test the crop box handles / interior. Returns handle name or 'move' or null. */
        const box = mpCropEditBox;
        if (!box) return null;
        const s1 = imageToScreen(box.x1, box.y1);
        const s2 = imageToScreen(box.x2, box.y2);
        const handles = _getMpCropHandles(s1.x, s1.y, s2.x, s2.y);
        const thresh = 10;

        for (const h of handles) {
            if (Math.abs(sx - h.x) < thresh && Math.abs(sy - h.y) < thresh) return h.name;
        }
        // Interior: move
        if (sx >= s1.x && sx <= s2.x && sy >= s1.y && sy <= s2.y) return 'move';
        return null;
    }

    function _mpCropCursor(handle) {
        const cursors = {
            'nw': 'nwse-resize', 'se': 'nwse-resize',
            'ne': 'nesw-resize', 'sw': 'nesw-resize',
            'n': 'ns-resize', 's': 'ns-resize',
            'e': 'ew-resize', 'w': 'ew-resize',
            'move': 'move',
        };
        return cursors[handle] || 'default';
    }

    function _applyMpCropDrag(sx, sy) {
        /** Update mpCropEditBox based on current drag handle and mouse position. */
        if (!mpCropDragStart || !mpCropDragHandle) return;
        const { mx: startMx, my: startMy, box: origBox } = mpCropDragStart;
        const dxImg = (sx - startMx) / scale;
        const dyImg = (sy - startMy) / scale;

        let { x1, y1, x2, y2 } = origBox;

        if (mpCropDragHandle === 'move') {
            x1 += dxImg; x2 += dxImg;
            y1 += dyImg; y2 += dyImg;
        } else {
            if (mpCropDragHandle.includes('w')) x1 += dxImg;
            if (mpCropDragHandle.includes('e')) x2 += dxImg;
            if (mpCropDragHandle.includes('n')) y1 += dyImg;
            if (mpCropDragHandle.includes('s')) y2 += dyImg;
        }

        // Ensure min size and correct ordering
        if (x2 - x1 < 20) x2 = x1 + 20;
        if (y2 - y1 < 20) y2 = y1 + 20;

        // Clamp to image bounds
        x1 = Math.max(0, x1); y1 = Math.max(0, y1);
        x2 = Math.min(imgW, x2); y2 = Math.min(imgH, y2);

        mpCropEditBox = {
            x1: Math.round(x1), y1: Math.round(y1),
            x2: Math.round(x2), y2: Math.round(y2),
        };
        // Mark this camera as manually adjusted
        mpCropAdjusted[currentSide] = true;
        _updateMpCamStatus();
    }

    // ── Helper: get current trial index from currentFrame ──
    function _getCurrentTrialIdx() {
        for (let i = 0; i < trials.length; i++) {
            if (currentFrame >= trials[i].start_frame && currentFrame <= trials[i].end_frame) {
                return i;
            }
        }
        return 0;
    }

    // ── Compute per-trial mediapipe availability ──
    function _computeMpHasMediapipe() {
        mpHasMediapipe = {};
        if (!mpLabels) return;
        // Check if any camera has landmark data for each trial's frame range
        for (let ti = 0; ti < trials.length; ti++) {
            const start = trials[ti].start_frame;
            const end = trials[ti].end_frame;
            let hasData = false;
            for (const cam of cameraNames) {
                const key = cam + '_landmarks';
                if (mpLabels[key]) {
                    // Check if any frame in this trial has non-null data
                    for (let f = start; f <= end && !hasData; f++) {
                        if (mpLabels[key][f]) hasData = true;
                    }
                }
                if (hasData) break;
            }
            mpHasMediapipe[ti] = hasData;
        }
    }

    // ── Update Run/Re-run button text based on current trial ──
    function _updateMpButtonText() {
        const ti = _getCurrentTrialIdx();
        const hasLabels = mpHasMediapipe[ti];
        const runBtn = document.getElementById('mpRunBtn');
        const rerunBtn = document.getElementById('mpRerunBtn');
        if (runBtn) runBtn.textContent = hasLabels ? 'Re-detect Hands' : 'Detect Hands';
        if (rerunBtn) rerunBtn.textContent = hasLabels
            ? 'Re-detect Hands (this trial)' : 'Detect Hands (this trial)';
    }

    // ── Update per-camera adjustment status display ──
    function _updateMpCamStatus() {
        const el = document.getElementById('mpCamStatus');
        if (!el) return;
        const parts = cameraNames.map(cam => {
            const adj = mpCropAdjusted[cam];
            return `${cam}: ${adj ? 'adjusted \u2713' : 'default'}`;
        });
        el.textContent = parts.join(' | ');
    }

    // ── Default crop box: 10% inset from edges (visible on screen) ──
    function _defaultCropBox() {
        return {
            x1: Math.round(imgW * 0.1),
            y1: Math.round(imgH * 0.1),
            x2: Math.round(imgW * 0.9),
            y2: Math.round(imgH * 0.9),
        };
    }

    function toggleMpCrop(enabled) {
        mpCropMode = enabled;
        const actionsEl = document.getElementById('mpCropActions');
        if (enabled) {
            const ti = _getCurrentTrialIdx();
            // Initialize adjustment tracking for both cameras
            mpCropAdjusted = {};
            for (const cam of cameraNames) {
                const saved = mpCropBoxes[ti] && mpCropBoxes[ti][cam];
                mpCropAdjusted[cam] = !!saved;  // DB-saved counts as adjusted
            }
            // Initialize edit box from saved (DB) or default (10% inset)
            const saved = mpCropBoxes[ti] && mpCropBoxes[ti][currentSide];
            mpCropEditBox = saved ? { ...saved } : _defaultCropBox();
            if (actionsEl) actionsEl.style.display = 'flex';
            // Lock zoom so frame navigation doesn't shift the view
            hasUserZoom = true;
            _updateMpCamStatus();
        } else {
            mpCropEditBox = null;
            mpCropDragHandle = null;
            mpCropDragStart = null;
            if (actionsEl) actionsEl.style.display = 'none';
        }
        render();
    }

    async function saveMpCrop() {
        const ti = _getCurrentTrialIdx();
        const statusEl = document.getElementById('mpRerunStatus');

        // Save current edit box for current camera before processing
        if (mpCropEditBox) {
            mpCropAdjusted[currentSide] = true;
            if (!mpCropBoxes[ti]) mpCropBoxes[ti] = {};
            mpCropBoxes[ti][currentSide] = { ...mpCropEditBox };
        }

        // Check adjustment status
        const adjustedCams = cameraNames.filter(c => mpCropAdjusted[c]);
        const unadjustedCams = cameraNames.filter(c => !mpCropAdjusted[c]);

        // If neither camera adjusted, just uncheck
        if (adjustedCams.length === 0) {
            cancelMpCrop();
            return;
        }

        // If only some cameras adjusted, set unadjusted ones to full frame
        if (unadjustedCams.length > 0) {
            if (!mpCropBoxes[ti]) mpCropBoxes[ti] = {};
            for (const cam of unadjustedCams) {
                mpCropBoxes[ti][cam] = { x1: 0, y1: 0, x2: imgW, y2: imgH };
            }
            if (statusEl) {
                statusEl.textContent = `${unadjustedCams.join(', ')} not adjusted \u2014 set to full frame.`;
            }
        }

        // Save to database
        const boxes = mpCropBoxes[ti] || {};
        try {
            await API.post(`/api/labeling/sessions/${sessionId}/crop-boxes`, {
                trial_idx: ti,
                boxes: boxes,
                apply_to_all: true,
            });
        } catch (err) {
            if (statusEl) statusEl.textContent = 'Error saving crop boxes: ' + (err.message || err);
        }

        // Exit edit mode
        cancelMpCrop();
        _updateMpButtonText();
    }

    function cancelMpCrop() {
        const toggle = document.getElementById('mpCropToggle');
        if (toggle) toggle.checked = false;
        toggleMpCrop(false);
    }

    async function rerunMediapipe() {
        const ti = _getCurrentTrialIdx();

        // Gather crops for ALL cameras
        const crops = {};
        for (const cam of cameraNames) {
            const saved = mpCropBoxes[ti] && mpCropBoxes[ti][cam];
            crops[cam] = saved || { x1: 0, y1: 0, x2: imgW, y2: imgH };
        }

        const statusEl = document.getElementById('mpRerunStatus');
        const rerunBtn = document.getElementById('mpRerunBtn');
        const runLabel = mpHasMediapipe[ti] ? 'Re-detecting' : 'Detecting';
        if (statusEl) statusEl.textContent = `${runLabel} hands…`;
        if (rerunBtn) rerunBtn.disabled = true;

        try {
            const result = await API.post(`/api/labeling/sessions/${sessionId}/rerun-mediapipe`, {
                trial_idx: ti,
                crops: crops,
            });

            const jobId = result.job_id;
            if (!jobId) throw new Error('No job_id returned');

            // Track progress via SSE stream (also visible on dashboard/processing page)
            const evtSource = new EventSource(`/api/jobs/${jobId}/stream`);
            evtSource.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.status === 'running') {
                    const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                    if (statusEl) statusEl.textContent = `${runLabel} hands… ${pct}%`;
                } else if (data.status === 'completed') {
                    evtSource.close();
                    if (statusEl) statusEl.textContent = 'Done! Reloading page…';
                    // Full page reload so distance trace renders cleanly
                    window.location.reload();
                } else if (data.status === 'failed') {
                    evtSource.close();
                    if (statusEl) statusEl.textContent = 'Error: ' + (data.error_msg || 'unknown');
                    if (rerunBtn) rerunBtn.disabled = false;
                } else if (data.status === 'cancelled') {
                    evtSource.close();
                    if (statusEl) statusEl.textContent = 'Cancelled.';
                    if (rerunBtn) rerunBtn.disabled = false;
                }
            };
            evtSource.onerror = () => {
                evtSource.close();
                if (statusEl) statusEl.textContent = 'Connection lost — check job status on dashboard.';
                if (rerunBtn) rerunBtn.disabled = false;
            };
        } catch (err) {
            if (statusEl) statusEl.textContent = 'Error: ' + (err.message || err);
            if (rerunBtn) rerunBtn.disabled = false;
        }
    }

    async function clearMpHistory() {
        try {
            await API.post(`/api/labeling/sessions/${sessionId}/clear-mediapipe-history`);
            if (mpLabels) delete mpLabels.run_history;
            const btn = document.getElementById('mpClearHistoryBtn');
            if (btn) btn.style.display = 'none';
            renderDistanceTrace();
        } catch (err) {
            const statusEl = document.getElementById('mpRerunStatus');
            if (statusEl) statusEl.textContent = 'Error clearing history: ' + (err.message || err);
        }
    }

    async function runPose() {
        const statusEl = document.getElementById('mpPoseStatus');
        const btn = document.getElementById('mpRunPoseBtn');
        if (statusEl) statusEl.textContent = 'Running Pose Detection...';
        if (btn) btn.disabled = true;

        try {
            const result = await API.post(`/api/labeling/sessions/${sessionId}/run-pose`);
            const jobId = result.job_id;
            if (!jobId) throw new Error('No job_id returned');

            const evtSource = new EventSource(`/api/jobs/${jobId}/stream`);
            evtSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.status === 'running') {
                    const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                    if (statusEl) statusEl.textContent = `Running Pose Detection... ${pct}%`;
                } else if (data.status === 'completed') {
                    evtSource.close();
                    if (statusEl) statusEl.textContent = 'Pose detection complete.';
                    if (btn) btn.disabled = false;
                } else if (data.status === 'failed') {
                    evtSource.close();
                    if (statusEl) statusEl.textContent = 'Error: ' + (data.error_msg || 'unknown');
                    if (btn) btn.disabled = false;
                } else if (data.status === 'cancelled') {
                    evtSource.close();
                    if (statusEl) statusEl.textContent = 'Cancelled.';
                    if (btn) btn.disabled = false;
                }
            };
            evtSource.onerror = () => {
                evtSource.close();
                if (statusEl) statusEl.textContent = 'Connection lost.';
                if (btn) btn.disabled = false;
            };
        } catch (err) {
            if (statusEl) statusEl.textContent = 'Error: ' + (err.message || err);
            if (btn) btn.disabled = false;
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

        // MediaPipe crop mode: intercept for box manipulation
        if (mpCropMode && mpCropEditBox) {
            const handle = _mpCropHitTest(sx, sy);
            if (handle) {
                mpCropDragHandle = handle;
                mpCropDragStart = { mx: sx, my: sy, box: { ...mpCropEditBox } };
                canvas.style.cursor = _mpCropCursor(handle);
                return;
            }
            // Click outside box — fall through to normal pan
        }

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
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        // MediaPipe crop mode drag
        if (mpCropDragHandle && mpCropDragStart) {
            _applyMpCropDrag(sx, sy);
            render();
            return;
        }
        // MediaPipe crop mode hover cursor
        if (mpCropMode && mpCropEditBox && !dragging) {
            const handle = _mpCropHitTest(sx, sy);
            canvas.style.cursor = handle ? _mpCropCursor(handle) : 'crosshair';
        }

        if (!dragging) return;

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
            // Round to 1 decimal place to avoid floating point accumulation errors
            const newX = Math.round((dragOrigX + dx) * 10) / 10;
            const newY = Math.round((dragOrigY + dy) * 10) / 10;
            lbl[dragging] = [newX, newY];
            render();
        }
    }

    function onMouseUp(e) {
        // MediaPipe crop mode: finish drag
        if (mpCropDragHandle) {
            mpCropDragHandle = null;
            mpCropDragStart = null;
            canvas.style.cursor = 'crosshair';
            return;
        }

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

    // ── Undo / Redo ──────────────────────────────────
    function snapshotEventMarkers() {
        const snap = {};
        EVENT_TYPES.forEach(t => snap[t] = [...(eventMarkers[t] || [])]);
        return snap;
    }

    function pushUndo(key, bp, prevCoords) {
        undoStack.push({ type: 'label', key, bp, prev: prevCoords, frame: currentFrame });
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        redoStack.length = 0;
    }

    function pushEventUndo(prevSnapshot) {
        undoStack.push({ type: 'events', prev: prevSnapshot, frame: currentFrame });
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        redoStack.length = 0;
    }

    // Core undo/redo logic: pop from `fromStack`, apply, push reverse to `toStack`.
    function _applyUndoEntry(action, toStack) {
        if (action.type === 'events') {
            const currentSnapshot = snapshotEventMarkers();
            toStack.push({ type: 'events', prev: currentSnapshot, frame: action.frame });
            for (const t of EVENT_TYPES) eventMarkers[t] = [...(action.prev[t] || [])];
            updateEventCounts();
            renderDistanceTrace();
            goToFrame(action.frame);
            return;
        }

        // Label action
        const { key, bp, prev } = action;
        const lbl = labels.get(key);
        const currentCoords = (lbl && lbl[bp]) ? lbl[bp] : null;
        toStack.push({ type: 'label', key, bp, prev: currentCoords, frame: action.frame });

        let lbl2 = labels.get(key);
        if (prev) {
            if (!lbl2) { lbl2 = {}; labels.set(key, lbl2); }
            lbl2[bp] = prev;
            deletedKeys.delete(key);
            dirtyKeys.add(key);
        } else {
            if (lbl2) {
                delete lbl2[bp];
                const hasAny = bodyparts.some(b => lbl2[b] && lbl2[b][0] != null);
                if (!hasAny) {
                    labels.delete(key);
                    deletedKeys.add(key);
                } else {
                    dirtyKeys.add(key);
                }
            }
        }

        updateLabelCount();
        // Clear distance for current frame so it gets recalculated by server
        const frameNum = parseInt(action.frame || currentFrame);
        if (distances && frameNum < distances.length) {
            distances[frameNum] = null;
        }
        scheduleSave();
        recomputeCameraShift();
        if (isRefine) updateV2TrainingBtn();
        goToFrame(action.frame);
    }

    function undo() {
        if (undoStack.length === 0) return;
        _applyUndoEntry(undoStack.pop(), redoStack);
    }

    function redo() {
        if (redoStack.length === 0) return;
        _applyUndoEntry(redoStack.pop(), undoStack);
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

        // Round to 1 decimal place to avoid floating point precision issues
        const roundedX = Math.round(imgX * 10) / 10;
        const roundedY = Math.round(imgY * 10) / 10;

        // Find first unplaced bodypart
        let placed = false;
        for (const bp of bodyparts) {
            const coords = lbl[bp];
            if (!coords || coords[0] == null) {
                pushUndo(key, bp, null);
                lbl[bp] = [roundedX, roundedY];
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
                lbl[closest] = [roundedX, roundedY];
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
                if (isFinal && !isEvents) renderTrialPlots();
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

        const isRefineCommit = (isRefine || isCorrections) && correctionFrames && correctionFrames.size > 0;

        if (!isRefineCommit && labels.size === 0) {
            alert('No labels to commit.');
            return;
        }

        try {
            const commitBody = {};
            if (isRefineCommit) {
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
        if (!isRefine && !isCorrections) return;
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

    // ── Refine Flow (within corrections mode) ──────────
    async function startRefineFlow() {
        if (!isCorrections && !isRefine) return;

        // Save any pending corrections first
        if (savePromise) await savePromise;
        await saveLabels();

        // Ensure all stages are loaded
        if (availableStages.length === 0) {
            try {
                const resp = await API.get(`/api/labeling/sessions/${sessionId}/available_stages`);
                availableStages = resp.stages || [];
                stageFiles = resp.files || {};
            } catch (e) { /* ignore */ }
        }
        await loadAllStages();

        // Compute correction frames (where corrections differ from DLC by ≥3px)
        computeCorrectionFrames();

        const refineInfo = document.getElementById('refineInfo');
        const v2Btn = document.getElementById('v2ToggleBtn');
        const commitBtn = document.getElementById('commitDlcBtn');

        if (correctionFrames.size === 0) {
            if (refineInfo) {
                refineInfo.style.display = '';
                refineInfo.textContent = 'No correction frames found (corrections must differ from DLC by ≥3px).';
            }
            return;
        }

        if (refineInfo) {
            refineInfo.style.display = '';
            refineInfo.textContent = `${correctionFrames.size} correction frame(s) found for V2 training.`;
        }
        if (v2Btn) v2Btn.style.display = '';
        if (commitBtn) commitBtn.style.display = '';

        // Update V2 toggle label
        updateV2TrainingBtn();
        render();
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
        // Only relevant for stereo mode (paired halves of a single image).
        if (cameraMode === 'multicam' || cameraMode === 'single') return;
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

    function _getActiveCameraNames() {
        // In multicam mode, use per-trial camera names if available
        if (cameraMode === 'multicam' && trials.length > 0) {
            const trial = _currentTrial();
            if (trial && trial.cameras && trial.cameras.length > 1) {
                return trial.cameras.map(c => c.name);
            }
        }
        return cameraNames;
    }

    function _currentTrial() {
        if (!trials.length) return null;
        for (const t of trials) {
            if (currentFrame >= t.start_frame && currentFrame <= t.end_frame) return t;
        }
        return trials[0];
    }

    function toggleSide() {
        if (cameraMode === 'single') return;
        const activeCams = _getActiveCameraNames();
        const idx = activeCams.indexOf(currentSide);
        const newIdx = (idx + 1) % activeCams.length;

        // Shift viewport to keep targets roughly centered when switching cameras.
        // Uses computed shift from paired labels, falls back to 7% horizontal default.
        // (Only for stereo — multicam files are separate so no cropping shift needed)
        if (cameraMode !== 'multicam' && hasUserZoom && imgW) {
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

        // --- Per-camera bounding box handling on camera switch ---
        const ti = _getCurrentTrialIdx();
        if (mpCropMode) {
            // Save current edit box for the old camera before switching
            if (mpCropEditBox) {
                if (!mpCropBoxes[ti]) mpCropBoxes[ti] = {};
                mpCropBoxes[ti][currentSide] = { ...mpCropEditBox };
                mpCropAdjusted[currentSide] = true;
            }
        }

        currentSide = activeCams[newIdx];
        if (typeof setNavState === 'function') setNavState({ side: currentSide });
        const _camLabel = document.getElementById('cameraLabel');
        if (_camLabel) _camLabel.textContent = currentSide;

        if (mpCropMode) {
            // Load saved box for new camera, or use 10% inset default
            const saved = mpCropBoxes[ti] && mpCropBoxes[ti][currentSide];
            mpCropEditBox = saved ? { ...saved } : _defaultCropBox();
            mpCropDragHandle = null;
            mpCropDragStart = null;
            _updateMpCamStatus();
        }

        goToFrame(currentFrame);
    }

    function togglePlay() {
        playing = !playing;
        const btn = document.getElementById('playBtn');
        if (playing) {
            btn.innerHTML = '\u23F8';
            // Slider value is an index into SPEED_PRESETS, not a direct rate value
            const sliderIndex = parseInt(document.getElementById('playbackRate').value);
            playbackRate = SPEED_PRESETS[sliderIndex] || 1;
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
        const frameOffset = trial.frame_offset || 0;
        const startTime = Math.max(0, (localFrame - frameOffset + 0.5) / trial.fps);

        // Use streaming URL (not blob) — browser handles buffering natively
        const videoUrl = `/api/labeling/sessions/${sessionId}/video?trial=${trialIdx}&side=${encodeURIComponent(currentSide)}&_=${Date.now()}`;
        if (currentTrialIdx !== trialIdx || currentVideoSide !== currentSide) {
            videoEl.src = videoUrl;
            currentTrialIdx = trialIdx;
            currentVideoSide = currentSide;
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
                    fallbackPlay(trial.fps);
                }
                return;
            }
        }

        // Try to set playback rate. If unsupported (e.g., 0.02x, 0.05x), fall back to manual frame-by-frame.
        try {
            videoEl.playbackRate = playbackRate;
        } catch (e) {
            console.warn(`[video] Playback rate ${playbackRate}x unsupported, using frame-by-frame fallback:`, e.message);
            videoPlaying = false;
            fallbackPlay(trial.fps);
            return;
        }
        videoEl.currentTime = startTime;
        videoPlaying = true;

        try {
            await videoEl.play();
            console.log(`[video] Playing trial ${trialIdx} at ${startTime.toFixed(2)}s (rate: ${playbackRate}x)`);
            // Prefer requestVideoFrameCallback — fires only when a new frame
            // is actually painted, keeping labels in sync with the displayed frame.
            // Falls back to requestAnimationFrame for older browsers.
            if ('requestVideoFrameCallback' in videoEl) {
                videoEl.requestVideoFrameCallback(videoDrawLoop);
            } else {
                requestAnimationFrame(videoDrawLoop);
            }
        } catch (e) {
            console.error('[video] play() rejected:', e);
            videoPlaying = false;
            fallbackPlay(trial.fps);
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

    function videoDrawLoop(now, metadata) {
        if (!videoPlaying || !playing) return;

        const trial = trials[currentTrialIdx];
        if (!trial) return;

        // Use the actual presented media time when available (from
        // requestVideoFrameCallback metadata), otherwise fall back to
        // videoEl.currentTime.  This ensures labels never draw ahead of
        // the video frame that is actually on screen.
        const mediaTime = (metadata && metadata.mediaTime != null)
            ? metadata.mediaTime
            : videoEl.currentTime;
        const frameOffset = trial.frame_offset || 0;
        const localFrame = Math.floor(mediaTime * trial.fps) + frameOffset;
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
            let sx, sw;
            if (cameraMode === 'multicam' || cameraMode === 'single') {
                sx = 0; sw = vw;
            } else {
                const midline = Math.floor(vw / 2);
                if (cameraNames.length >= 2 && currentSide === cameraNames[1]) {
                    sx = midline; sw = vw - midline;
                } else {
                    sx = 0; sw = midline;
                }
            }

            imgW = sw;
            imgH = vh;
            if (!hasUserZoom && !mpCropMode) fitImage();

            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            ctx.drawImage(videoEl, sx, 0, sw, vh, 0, 0, sw, vh);
            ctx.restore();
        }

        drawLabelsOverlay();
        if (mpCropMode && mpCropEditBox) drawMpCropOverlay();
        updateFrameDisplay();

        // Only update timeline/trace every ~10 frames to reduce work during playback
        if (currentFrame % 10 === 0) {
            renderTimeline();
            renderDistanceTrace();
        }

        // Schedule next frame using the same mechanism we started with
        if ('requestVideoFrameCallback' in videoEl) {
            videoEl.requestVideoFrameCallback(videoDrawLoop);
        } else {
            requestAnimationFrame(videoDrawLoop);
        }
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
                    const frameOffset = trial.frame_offset || 0;
                    const localFrame = Math.round(videoEl.currentTime * trial.fps) + frameOffset;
                    currentFrame = trial.start_frame + Math.min(Math.max(0, localFrame), trial.frame_count - 1);
                }
            }
        }
        goToFrame(currentFrame);
        canvas.focus();
    }

    function fallbackPlay(fpsOverride = null) {
        // Fallback: frame-by-frame if video streaming fails
        // Slider value is an index into SPEED_PRESETS, not a direct rate value
        const sliderIndex = parseInt(document.getElementById('playbackRate').value);
        playbackRate = SPEED_PRESETS[sliderIndex] || 1;
        let fps = fpsOverride;
        // If no fps provided, determine from current trial
        if (fps === null) {
            const trialIdx = getTrialForFrame(currentFrame);
            fps = trials.length > 0 && trialIdx >= 0 && trialIdx < trials.length ? trials[trialIdx].fps : 30;
        }
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

    function cycleFrameDisplay() {
        const modes = ['frame', 'time', 'both'];
        const idx = modes.indexOf(frameDisplayMode);
        frameDisplayMode = modes[(idx + 1) % modes.length];
        render();
    }

    // ── Keyboard shortcuts ────────────────────────────
    // Letter keys for deleting bodyparts (first two get D/F, rest use number keys)
    const DELETE_KEYS = { 'd': 0, 'f': 1 };

    function setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in a text input (but not slider or other input types)
            if (e.target.tagName === 'INPUT' &&
                (e.target.type === 'text' || e.target.type === 'number' || e.target.type === 'password')) {
                return;
            }
            // Ignore if in a textarea or select (unless it's the speed slider)
            if (e.target.id !== 'playbackRate' && (e.target.tagName === 'TEXTAREA' ||
                (e.target.tagName === 'SELECT' && e.target.id !== 'playbackRate'))) {
                return;
            }

            // Ctrl+Z: undo (disabled in final mode, but allowed in events mode)
            if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
                e.preventDefault();
                if (!isFinal || isEvents) undo();
                return;
            }

            // Ctrl+Y or Ctrl+Shift+Z: redo (disabled in final mode, but allowed in events mode)
            if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Z')) {
                e.preventDefault();
                if (!isFinal || isEvents) redo();
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
                    if (isEvents) prevEvent();
                    else if (isRefine) prevLabel();
                    else if (isCorrections) prevGap();
                    else prevLabel();
                    break;
                case 'w':
                    e.preventDefault();
                    if (isEvents) nextEvent();
                    else if (isRefine) nextLabel();
                    else if (isCorrections) nextGap();
                    else nextLabel();
                    break;
                case 'r':
                    e.preventDefault();
                    cycleReviewMode();
                    break;
                case 'e':
                    if (cameraMode !== 'single') {
                        e.preventDefault();
                        toggleSide();
                    }
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
                // Events mode shortcuts
                case 'x':
                    if (isEvents) { e.preventDefault(); deleteEvent(); }
                    break;
                case '[':
                    if (isEvents) { e.preventDefault(); shiftEvent(-1); }
                    break;
                case ']':
                    if (isEvents) { e.preventDefault(); shiftEvent(1); }
                    break;
                default:
                    // Events mode: dynamic shortcut keys for placing events
                    if (isEvents) {
                        const evtType = EVENT_TYPES.find(t => EVENT_SHORTCUTS[t] === e.key);
                        if (evtType) {
                            e.preventDefault();
                            placeEventType(evtType);
                            break;
                        }
                    }
                    if (isFinal && !isEvents) break; // Read-only: no delete shortcuts
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
        // Update frame/time display based on display mode
        const fps = trials.length > 0 ? (trials[0].fps || 30) : 30;
        const timeSec = (localFrame / fps).toFixed(2);
        const totalSec = ((trialFrameCount - 1) / fps).toFixed(2);
        let displayText;
        if (frameDisplayMode === 'frame') {
            displayText = `Frame: ${localFrame} / ${trialFrameCount - 1}`;
        } else if (frameDisplayMode === 'time') {
            displayText = `${timeSec}s / ${totalSec}s`;
        } else {
            displayText = `${localFrame} / ${trialFrameCount - 1}  (${timeSec}s)`;
        }
        document.getElementById('frameDisplay').textContent = displayText;
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
        if (isEvents) {
            const total = EVENT_TYPES.reduce((s, t) => s + (eventMarkers[t] || []).length, 0);
            document.getElementById('labelCount').innerHTML =
                `Events: <strong>${total}</strong>`;
            return;
        }
        if (isRefine) {
            const total = correctionFrames.size;
            const training = total - v2Excludes.size;
            document.getElementById('labelCount').innerHTML =
                `Corrections: <strong>${training}</strong> / ${total} training on`;
            return;
        }
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

        if (isEvents) {
            el.innerHTML = `
                <div><kbd>A</kbd> / <kbd>&larr;</kbd> Prev frame</div>
                <div><kbd>S</kbd> / <kbd>&rarr;</kbd> Next frame</div>
                <div><kbd>Q</kbd> Prev event</div>
                <div><kbd>W</kbd> Next event</div>
                <div><kbd>E</kbd> Toggle ${cameraNames.join('/')}</div>
                <div><kbd>Z</kbd> Reset zoom</div>
                <div><kbd>Space</kbd> Play/pause</div>
                <div><kbd>1</kbd> Place open</div>
                <div><kbd>2</kbd> Place peak</div>
                <div><kbd>3</kbd> Place close</div>
                <div><kbd>X</kbd> Delete event</div>
                <div><kbd>[</kbd> Shift event ←1</div>
                <div><kbd>]</kbd> Shift event →1</div>
            `;
            return;
        }

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
        if (isRefine) {
            html += `
                <div><kbd>Q</kbd> Prev corr</div>
                <div><kbd>W</kbd> Next corr</div>
                <div><kbd>T</kbd> Toggle training</div>
            `;
        } else if (isCorrections) {
            html += `
                <div><kbd>Q</kbd> Prev gap</div>
                <div><kbd>W</kbd> Next gap</div>
                <div><kbd>Enter</kbd> Accept merge</div>
            `;
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
        // Return focus to main canvas so spacebar play/pause keeps working
        canvas.focus();
    }

    function renderTimeline() {
        if (isFinal && !isEvents) {
            renderTrialPlots();
            return;
        }
        if (isEvents) {
            renderDistanceTrace();
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

        // Committed label dots (filled circles, dimmer)
        committedLabels.forEach((lbl, key) => {
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
                    tlCtx.globalAlpha = 0.5;
                    tlCtx.fill();
                    tlCtx.globalAlpha = 1;
                }
            });
        });

        // Unsaved label dots (open circles with stroke, on top)
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
                    tlCtx.arc(x, dotY, 3, 0, Math.PI * 2);
                    tlCtx.strokeStyle = bpColor(idx);
                    tlCtx.lineWidth = 1.5;
                    tlCtx.stroke();
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
        distCanvas.addEventListener('mousemove', (e) => {
            if (distDragging) return;  // already in a drag
            const marker = _findNearestEventMarker(e.clientX);
            distCanvas.style.cursor = marker ? 'ew-resize' : 'pointer';
        });

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

    /** Find the closest visible event marker near a click position. Returns {type, frame, dist} or null. */
    function _findNearestEventMarker(clientX) {
        if (!isEvents) return null;
        const rect = distCanvas.getBoundingClientRect();
        const clickX = clientX - rect.left;
        const padL = 40, padR = 8;
        const plotW = rect.width - padL - padR;
        const effectiveViewFrames = distViewFrames > 0 ? distViewFrames : totalFrames;
        const fToX = (f) => padL + ((f - distViewStart) / effectiveViewFrames) * plotW;
        const LENIENCY_PX = 10;
        let best = null;

        EVENT_TYPES.forEach(etype => {
            if (!eventVisibility[etype]) return;
            eventMarkers[etype].forEach(f => {
                if (f < distViewStart || f >= distViewStart + effectiveViewFrames) return;
                const markerX = fToX(f);
                const dx = Math.abs(clickX - markerX);
                if (dx < LENIENCY_PX && (!best || dx < best.dist)) {
                    best = { type: etype, frame: f, dist: dx };
                }
            });
        });
        return best;
    }

    let _distLegendAreas = [];  // populated by renderDistanceTrace legend drawing

    function onDistTraceMouseDown(e) {
        if (!distances || totalFrames === 0) return;
        e.preventDefault();

        // Check if click is on a legend toggle entry
        if (_distLegendAreas.length > 0) {
            const rect = distCanvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            for (const area of _distLegendAreas) {
                if (mx >= area.x && mx <= area.x + area.w && my >= area.y && my <= area.y + area.h) {
                    // Toggle visibility
                    mpRunVisible[area.key] = mpRunVisible[area.key] === false ? true : false;
                    renderDistanceTrace();
                    return;
                }
            }
        }

        // In events mode, check if mousedown is on an event marker → start drag
        const hitMarker = _findNearestEventMarker(e.clientX);
        if (hitMarker) {
            // ── Event marker drag mode ──
            const dragType = hitMarker.type;
            const dragOrigFrame = hitMarker.frame;
            let dragCurrentFrame = dragOrigFrame;
            const undoSnapshot = JSON.parse(JSON.stringify(eventMarkers));
            distCanvas.style.cursor = 'ew-resize';

            // Navigate to the event immediately
            goToFrame(dragOrigFrame);

            const onDragMove = (ev) => {
                const newFrame = distXToFrame(ev.clientX);
                if (newFrame === dragCurrentFrame) return;

                // Move the marker: remove old position, insert new
                const frames = eventMarkers[dragType];
                const idx = frames.indexOf(dragCurrentFrame);
                if (idx !== -1) frames.splice(idx, 1);
                if (!frames.includes(newFrame)) {
                    frames.push(newFrame);
                    frames.sort((a, b) => a - b);
                }
                dragCurrentFrame = newFrame;
                currentFrame = newFrame;
                updateFrameDisplay();
                renderDistanceTrace();
                render();
            };

            const onDragUp = () => {
                distCanvas.style.cursor = 'pointer';
                window.removeEventListener('mousemove', onDragMove);
                window.removeEventListener('mouseup', onDragUp);

                if (dragCurrentFrame !== dragOrigFrame) {
                    // Push undo with the pre-drag snapshot
                    pushEventUndo(undoSnapshot);
                    updateEventCounts();
                }
                goToFrame(dragCurrentFrame);
            };

            window.addEventListener('mousemove', onDragMove);
            window.addEventListener('mouseup', onDragUp);
            return;
        }

        // ── Normal view-panning drag mode ──
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
                // Click — snap to nearest event marker if close
                const marker = _findNearestEventMarker(ev.clientX);
                const targetFrame = marker ? marker.frame : distXToFrame(ev.clientX);
                autoZoomForFrame(targetFrame, currentSide);
                goToFrame(targetFrame);
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

        // Use stable range (computed from cleanest source, outlier-filtered) for consistent Y-axis.
        // Fall back to percentile range of current distances if stable range not yet set.
        let minD, maxD;
        const rangeSource = stableDistRange || percentileRange(distances);
        if (!rangeSource) return;
        minD = rangeSource.min;
        maxD = userYMax !== null ? userYMax : rangeSource.max;

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

        // Draw run history lines (toggleable, behind current)
        const historyColors = [
            'rgba(255, 140, 0, 0.6)',    // orange
            'rgba(0, 200, 100, 0.6)',    // green
            'rgba(200, 80, 200, 0.6)',   // purple
            'rgba(255, 80, 80, 0.6)',    // red
            'rgba(100, 200, 255, 0.6)',  // cyan
        ];
        if (mpLabels && mpLabels.run_history) {
            mpLabels.run_history.forEach((run, ri) => {
                if (mpRunVisible[run.run] === false) return;
                const runDist = run.distances;
                if (!runDist) return;
                distCtx.beginPath();
                let s = false;
                for (let f = Math.max(0, vStart - 1); f < vEnd + 1 && f < runDist.length; f++) {
                    const d = runDist[f];
                    if (d === null || d === undefined) { s = false; continue; }
                    const x = fToX(f);
                    const y = dToY(d);
                    if (!s) { distCtx.moveTo(x, y); s = true; }
                    else distCtx.lineTo(x, y);
                }
                distCtx.strokeStyle = historyColors[ri % historyColors.length];
                distCtx.lineWidth = 1.5;
                distCtx.stroke();
            });
        }

        // Draw current distance line (toggleable)
        if (mpRunVisible.current !== false) {
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
            distCtx.strokeStyle = 'rgba(74, 158, 255, 0.9)';
            distCtx.lineWidth = 1.5;
            distCtx.stroke();
        }

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

        // Event markers (events mode only)
        if (isEvents) {
            const trialRange = getTrialFrameRange(currentEventTrialIdx);
            EVENT_TYPES.forEach(etype => {
                if (!eventVisibility[etype]) return;
                const color = EVENT_COLORS[etype];
                eventMarkers[etype].forEach(f => {
                    if (f < vStart || f >= vEnd) return;
                    const x = fToX(f);
                    let y;
                    if (distances && f < distances.length && distances[f] !== null) {
                        y = Math.max(padT + 5, Math.min(padT + plotH - 5, dToY(distances[f])));
                    } else {
                        y = padT + plotH * 0.5;
                    }
                    // Dim events outside current trial
                    const inCurrentTrial = f >= trialRange.start && f <= trialRange.end;
                    distCtx.globalAlpha = inCurrentTrial ? 1.0 : 0.25;

                    // Draw diamond: filled=saved to CSV, outline=pending
                    const r = 5;
                    const isSaved = savedEventFrames[etype].has(f);
                    distCtx.beginPath();
                    distCtx.moveTo(x, y - r);
                    distCtx.lineTo(x + r, y);
                    distCtx.lineTo(x, y + r);
                    distCtx.lineTo(x - r, y);
                    distCtx.closePath();
                    if (isSaved) {
                        distCtx.fillStyle = color;
                        distCtx.fill();
                        distCtx.strokeStyle = 'rgba(255,255,255,0.6)';
                        distCtx.lineWidth = 1;
                    } else {
                        distCtx.fillStyle = 'transparent';
                        distCtx.strokeStyle = color;
                        distCtx.lineWidth = 2;
                    }
                    distCtx.stroke();

                    // Highlight if this is the active frame
                    if (f === currentFrame) {
                        distCtx.beginPath();
                        distCtx.moveTo(x, y - r - 3);
                        distCtx.lineTo(x + r + 3, y);
                        distCtx.lineTo(x, y + r + 3);
                        distCtx.lineTo(x - r - 3, y);
                        distCtx.closePath();
                        distCtx.strokeStyle = 'white';
                        distCtx.lineWidth = 1.5;
                        distCtx.stroke();
                    }
                    distCtx.globalAlpha = 1.0;
                });
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
        if (distances && mpRunVisible.current !== false) {
            const curD = distances[currentFrame];
            if (curD !== null && curD !== undefined) {
                const y = dToY(curD);
                distCtx.beginPath();
                distCtx.arc(cx, y, 4, 0, Math.PI * 2);
                distCtx.fillStyle = '#ff4444';
                distCtx.fill();
            }
        }

        // Run history legend (top-right, clickable toggles)
        if (mpLabels && mpLabels.run_history && mpLabels.run_history.length > 0) {
            const legendX = w - padR - 90;
            let legendY = padT + 4;
            distCtx.font = '10px sans-serif';
            distCtx.textAlign = 'left';
            // Store legend hit areas for click handling
            _distLegendAreas = [];

            // Current run entry
            const curVisible = mpRunVisible.current !== false;
            distCtx.fillStyle = curVisible ? 'rgba(74, 158, 255, 0.9)' : 'rgba(74, 158, 255, 0.2)';
            distCtx.fillRect(legendX, legendY - 4, 12, 2);
            distCtx.fillStyle = curVisible ? '#ccc' : '#555';
            distCtx.fillText('Current', legendX + 16, legendY);
            _distLegendAreas.push({ key: 'current', x: legendX - 4, y: legendY - 10, w: 90, h: 14 });
            legendY += 14;

            // History run entries
            mpLabels.run_history.forEach((run, ri) => {
                const vis = mpRunVisible[run.run] !== false;
                const color = historyColors[ri % historyColors.length];
                distCtx.fillStyle = vis ? color : color.replace(/[\d.]+\)$/, '0.15)');
                distCtx.fillRect(legendX, legendY - 4, 12, 2);
                distCtx.fillStyle = vis ? '#aaa' : '#555';
                distCtx.fillText(`Run ${run.run}`, legendX + 16, legendY);
                _distLegendAreas.push({ key: run.run, x: legendX - 4, y: legendY - 10, w: 90, h: 14 });
                legendY += 14;
            });
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
        if (isEvents) return 'events';
        if (isFinal) return 'final';
        if (isCorrections) return 'corrections';
        if (isRefine) return 'refine';
        return 'initial';
    }

    async function switchSubject(subjectId) {
        if (subjectId === currentSubjectId) return;
        try {
            // Save current frame/zoom and preferences before switching
            saveNavState();
            savePreferences();
            // Mark this as an intra-labeling subject switch so preferences are restored
            sessionStorage.setItem('dlc_subjectSwitch', '1');
            // Clear mode switch flag — new subject should start at frame 0
            sessionStorage.removeItem('dlc_modeSwitch');
            // Update results nav link
            const resultsLink = document.getElementById('resultsLink');
            if (resultsLink) resultsLink.href = `/results?subject=${subjectId}&from=labeling`;
            sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
            const session = await API.post(`/api/labeling/${subjectId}/sessions`, {
                session_type: isMediaPipePage ? 'initial' : currentSessionType(),
            });
            const basePath = isMediaPipePage ? '/mediapipe' : '/labeling';
            window.location.href = `${basePath}?session=${session.id}`;
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
        if (!isRefine && !isCorrections) return;
        const key = `${currentFrame}_${currentSide}`;
        if (!correctionFrames.has(key)) return; // not a correction frame — nothing to toggle
        if (v2Excludes.has(key)) {
            v2Excludes.delete(key);
        } else {
            v2Excludes.add(key);
        }
        updateV2TrainingBtn();
        updateLabelCount();
        saveV2Excludes();
        renderDistanceTrace();
    }

    function updateV2TrainingBtn() {
        const btn = document.getElementById('v2ToggleBtn');
        if (!btn || (!isRefine && !isCorrections)) return;
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

    // ── Stable Y-range computation ─────────────────────
    /** Compute a percentile-based Y range, ignoring outliers.
     *  Uses P2 – P98 of valid values, then adds 5% padding. */
    function percentileRange(arr, pLow = 2, pHigh = 98) {
        const valid = arr.filter(d => d !== null && d !== undefined && isFinite(d)).sort((a, b) => a - b);
        if (valid.length === 0) return null;
        const lo = valid[Math.max(0, Math.floor(valid.length * pLow / 100))];
        const hi = valid[Math.min(valid.length - 1, Math.floor(valid.length * pHigh / 100))];
        if (lo === hi) return { min: Math.max(0, lo - 5), max: hi + 5 };
        const pad = (hi - lo) * 0.05;
        return { min: Math.max(0, lo - pad), max: hi + pad };
    }

    function computeStableDistRange() {
        /** Compute once from the cleanest available data source.
         *  Priority: corrections > refine > dlc > mp > current distances.
         *  Uses percentile filtering to exclude outliers. */
        const priority = ['corrections', 'refine', 'dlc', 'mp'];
        for (const stage of priority) {
            const sd = stageData[stage];
            if (sd && sd.distances && sd.distances.some(d => d !== null)) {
                const r = percentileRange(sd.distances);
                if (r) { stableDistRange = r; return; }
            }
        }
        // Fallback: current distances
        if (distances && distances.some(d => d !== null)) {
            const r = percentileRange(distances);
            if (r) stableDistRange = r;
        }
    }

    // ── Frame / zoom navigation persistence ───────────
    function saveNavState() {
        if (!currentSubjectId) return;
        try {
            localStorage.setItem(`dlc_nav_${currentSubjectId}`, JSON.stringify({
                frame: currentFrame,
                scale,
                offsetX,
                offsetY,
                hasUserZoom,
            }));
        } catch (_) {}
    }

    function restoreNavState() {
        if (!currentSubjectId) return false;
        try {
            const saved = localStorage.getItem(`dlc_nav_${currentSubjectId}`);
            if (!saved) return false;
            const nav = JSON.parse(saved);
            if (typeof nav.frame === 'number' && nav.frame > 0 && nav.frame < totalFrames) {
                currentFrame = nav.frame;
                if (nav.hasUserZoom && nav.scale) {
                    scale = nav.scale;
                    offsetX = nav.offsetX || 0;
                    offsetY = nav.offsetY || 0;
                    hasUserZoom = true;
                }
                return true;
            }
        } catch (_) {}
        return false;
    }

    // ── Events panel builder (dynamic from EVENT_TYPES) ────────

    function buildEventsPanel() {
        // Populate event type buttons
        const btnContainer = document.getElementById('eventTypeButtons');
        if (btnContainer) {
            btnContainer.innerHTML = '';
            // 2 columns for sidebar width; 1 column if only 1 type
            const cols = EVENT_TYPES.length === 1 ? 1 : 2;
            btnContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
            EVENT_TYPES.forEach(t => {
                const displayName = t[0].toUpperCase() + t.slice(1);
                const shortcut = EVENT_SHORTCUTS[t] || '';
                const color = EVENT_COLORS[t] || '#888';
                const btn = document.createElement('button');
                btn.className = 'btn btn-sm';
                btn.style.color = color;
                btn.style.borderColor = color;
                btn.title = `Place ${displayName} at current frame (${shortcut})`;
                btn.textContent = `${shortcut} ${displayName}`;
                btn.onclick = () => placeEventType(t);
                btnContainer.appendChild(btn);
            });
        }

        // Populate visibility toggles
        const toggleContainer = document.getElementById('eventVisToggles');
        if (toggleContainer) {
            toggleContainer.innerHTML = '';
            EVENT_TYPES.forEach(t => {
                const displayName = t[0].toUpperCase() + t.slice(1);
                const color = EVENT_COLORS[t] || '#888';
                const label = document.createElement('label');
                label.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.id = `showEvent_${t}`;
                cb.checked = eventVisibility[t] !== false;
                cb.onchange = () => setEventVisibility(t, cb.checked);
                label.appendChild(cb);
                const dot = document.createElement('span');
                dot.style.color = color;
                dot.textContent = '\u25cf';
                label.appendChild(dot);
                label.appendChild(document.createTextNode(` ${displayName}`));
                toggleContainer.appendChild(label);
            });
        }

        // Populate shortcuts list (events mode additions)
        const shortcutList = document.getElementById('shortcutList');
        if (shortcutList) {
            // Remove any previously added event shortcuts
            shortcutList.querySelectorAll('.event-shortcut').forEach(el => el.remove());
            EVENT_TYPES.forEach(t => {
                const displayName = t[0].toUpperCase() + t.slice(1);
                const shortcut = EVENT_SHORTCUTS[t] || '';
                if (shortcut) {
                    const div = document.createElement('div');
                    div.className = 'event-shortcut';
                    div.innerHTML = `<kbd>${shortcut}</kbd> Place ${displayName}`;
                    shortcutList.appendChild(div);
                }
            });
            // Add standard event shortcuts
            ['X Delete event', '[ Shift left', '] Shift right'].forEach(text => {
                const div = document.createElement('div');
                div.className = 'event-shortcut';
                const [key, ...desc] = text.split(' ');
                div.innerHTML = `<kbd>${key}</kbd> ${desc.join(' ')}`;
                shortcutList.appendChild(div);
            });
        }
    }

    // ── Events mode functions ──────────────────────────

    async function loadEvents() {
        try {
            const result = await API.get(`/api/labeling/sessions/${sessionId}/events`);
            EVENT_TYPES.forEach(t => {
                eventMarkers[t] = result[t] || [];
                savedEventFrames[t] = new Set(eventMarkers[t]);
            });
            // Initialize trial switcher
            currentEventTrialIdx = getTrialForFrame(currentFrame);
            const trialLabel = document.getElementById('trialSelectorLabel');
            if (trialLabel && trials[currentEventTrialIdx]) {
                trialLabel.textContent = trials[currentEventTrialIdx].trial_name;
            }
            updateEventCounts();
            renderDistanceTrace();

            // Reset metrics cache — metrics are computed on demand when
            // the auto-detect modal opens (and cached to disk on the server,
            // so subsequent loads are instant).
            metricsCache = {};
            metricsLoading.clear();
        } catch (e) {
            console.log('Could not load events:', e);
        }
    }

    async function computeMetricsForTrial(trialIdx) {
        if (metricsCache[trialIdx] || metricsLoading.has(trialIdx)) return;
        metricsLoading.add(trialIdx);
        try {
            const result = await API.post(
                `/api/labeling/sessions/${sessionId}/compute_metrics`,
                { trial_index: trialIdx }
            );
            metricsCache[trialIdx] = result;
            // If detect modal is open and showing this trial, refresh plots
            const overlay = document.getElementById('detectModalOverlay');
            if (overlay && overlay.classList.contains('active') && currentEventTrialIdx === trialIdx) {
                showMetricPlotsForCurrentTrial();
            }
        } catch (e) {
            console.log(`Metrics computation failed for trial ${trialIdx}:`, e);
        } finally {
            metricsLoading.delete(trialIdx);
        }
    }

    async function saveEvents() {
        try {
            // Only send visible types — backend merges with existing CSV for hidden types
            const body = {};
            EVENT_TYPES.forEach(t => { if (eventVisibility[t]) body[t] = eventMarkers[t]; });
            await API.put(`/api/labeling/sessions/${sessionId}/events`, body);
            // Mark saved types as persisted
            EVENT_TYPES.forEach(t => {
                if (eventVisibility[t]) savedEventFrames[t] = new Set(eventMarkers[t]);
            });
            savePreferences();
            renderDistanceTrace();
            const counts = EVENT_TYPES.filter(t => eventVisibility[t])
                .map(t => `${t}: ${eventMarkers[t].length}`).join(', ');
            updateLabelInfo(`Saved — ${counts}`);
        } catch (e) {
            alert('Error saving events: ' + e.message);
        }
    }

    function setEventVisibility(type, visible) {
        eventVisibility[type] = visible;
        savePreferences();
        renderDistanceTrace();
    }

    function placeEventType(type) {
        if (!isEvents || !EVENT_TYPES.includes(type)) return;
        // Don't allow placing events of types that are not currently plotted/visible
        if (!eventVisibility[type]) return;
        const frames = eventMarkers[type];
        if (!frames.includes(currentFrame)) {
            const snapshot = snapshotEventMarkers();
            frames.push(currentFrame);
            frames.sort((a, b) => a - b);
            pushEventUndo(snapshot);
        }
        updateEventCounts();
        renderDistanceTrace();
    }

    // Returns the first visible event type that has a marker at currentFrame, or null.
    function _plottedTypeAtFrame() {
        for (const t of EVENT_TYPES)
            if (eventVisibility[t] && eventMarkers[t].includes(currentFrame)) return t;
        return null;
    }

    function deleteEvent() {
        if (!isEvents) return;
        const type = _plottedTypeAtFrame();
        if (!type) return;
        const frames = eventMarkers[type];
        const idx = frames.indexOf(currentFrame);
        if (idx !== -1) {
            const snapshot = snapshotEventMarkers();
            frames.splice(idx, 1);
            pushEventUndo(snapshot);
        }
        updateEventCounts();
        renderDistanceTrace();
    }

    function shiftEvent(delta) {
        if (!isEvents) return;
        const type = _plottedTypeAtFrame();
        if (!type) return;
        const frames = eventMarkers[type];
        const idx = frames.indexOf(currentFrame);
        if (idx === -1) return;

        const snapshot = snapshotEventMarkers();
        const newFrame = Math.max(0, Math.min(totalFrames - 1, currentFrame + delta));
        frames.splice(idx, 1);
        if (!frames.includes(newFrame)) {
            frames.push(newFrame);
            frames.sort((a, b) => a - b);
        }
        pushEventUndo(snapshot);
        updateEventCounts();
        // Navigate to the shifted frame
        goToFrame(newFrame);
    }

    // ── Trial switching (auto-follows frame position now) ──

    function setEventTrial(trialIdx) {
        if (trialIdx < 0 || trialIdx >= trials.length) return;
        currentEventTrialIdx = trialIdx;
        const trial = trials[trialIdx];
        goToFrame(trial.start_frame);
    }

    function prevTrial() { setEventTrial(currentEventTrialIdx - 1); }
    function nextTrial() { setEventTrial(currentEventTrialIdx + 1); }

    function getTrialFrameRange(trialIdx) {
        const t = trials[trialIdx];
        return t ? { start: t.start_frame, end: t.end_frame } : { start: 0, end: totalFrames - 1 };
    }

    function eventsInTrial(trialIdx) {
        const { start, end } = getTrialFrameRange(trialIdx);
        const result = {};
        EVENT_TYPES.forEach(t => {
            result[t] = eventMarkers[t].filter(f => f >= start && f <= end);
        });
        return result;
    }

    // ── Auto-detect modal ─────────────────────────────────

    function openDetectModal() {
        const overlay = document.getElementById('detectModalOverlay');
        if (overlay) overlay.classList.add('active');
        const trialLabel = document.getElementById('detectModalTrial');
        if (trialLabel && trials[currentEventTrialIdx]) {
            trialLabel.textContent = `(${trials[currentEventTrialIdx].trial_name})`;
        }

        // Auto-default "Peaks only" based on saved events for current trial
        const peaksOnlyCb = document.getElementById('stepPeaksOnly');
        if (peaksOnlyCb) {
            const trialEvts = eventsInTrial(currentEventTrialIdx);
            const hasSavedPeaks = (savedEventFrames.peak || new Set()).size > 0 &&
                trialEvts.peak && trialEvts.peak.some(f => savedEventFrames.peak.has(f));
            // Peaks saved → uncheck peaks_only so open/close can be detected
            // No peaks saved → check peaks_only for first-pass peak detection
            peaksOnlyCb.checked = !hasSavedPeaks;
        }

        showMetricPlotsForCurrentTrial();
        // Apply current focus mode
        applyDetectFocus();
    }

    function closeDetectModal() {
        const overlay = document.getElementById('detectModalOverlay');
        if (overlay) overlay.classList.remove('active');
    }

    function detectEvents() {
        if (!isEvents) return;
        openDetectModal();
    }

    // Show metric plots from cache, or show loading/unavailable state
    function showMetricPlotsForCurrentTrial() {
        const loading = document.getElementById('metricPlotsLoading');
        const container = document.getElementById('metricPlotsContainer');
        const unavailable = document.getElementById('metricPlotsUnavailable');

        const cached = metricsCache[currentEventTrialIdx];
        if (cached) {
            if (loading) loading.style.display = 'none';
            if (unavailable) unavailable.style.display = 'none';
            if (container) container.style.display = 'block';
            renderMetricCanvas('distPlotCanvas', cached.distance, '#4a9eff', 'Distance');
            renderMetricCanvas('reversalPlotCanvas', cached.reversal, '#ff9800', 'Reversal');
            renderMetricCanvas('ssdPlotCanvas', cached.motion_ssd, '#4caf50', 'SSD Motion');
        } else if (metricsLoading.has(currentEventTrialIdx)) {
            if (container) container.style.display = 'none';
            if (unavailable) unavailable.style.display = 'none';
            if (loading) loading.style.display = 'block';
            // Poll until done
            const trialToWatch = currentEventTrialIdx;
            const poll = setInterval(() => {
                if (metricsCache[trialToWatch]) {
                    clearInterval(poll);
                    if (currentEventTrialIdx === trialToWatch) showMetricPlotsForCurrentTrial();
                } else if (!metricsLoading.has(trialToWatch)) {
                    clearInterval(poll);
                    // Failed
                    if (currentEventTrialIdx === trialToWatch) {
                        if (loading) loading.style.display = 'none';
                        if (unavailable) unavailable.style.display = 'block';
                    }
                }
            }, 300);
        } else {
            // Not cached and not loading — trigger computation now
            if (container) container.style.display = 'none';
            if (unavailable) unavailable.style.display = 'none';
            if (loading) loading.style.display = 'block';
            computeMetricsForTrial(currentEventTrialIdx);
            // Poll until done
            const trialToWatch2 = currentEventTrialIdx;
            const poll2 = setInterval(() => {
                if (metricsCache[trialToWatch2]) {
                    clearInterval(poll2);
                    if (currentEventTrialIdx === trialToWatch2) showMetricPlotsForCurrentTrial();
                } else if (!metricsLoading.has(trialToWatch2)) {
                    clearInterval(poll2);
                    if (currentEventTrialIdx === trialToWatch2) {
                        if (loading) loading.style.display = 'none';
                        if (unavailable) unavailable.style.display = 'block';
                    }
                }
            }, 300);
        }
    }

    function renderMetricCanvas(canvasId, data, color, label) {
        const cvs = document.getElementById(canvasId);
        if (!cvs || !data) return;
        const c = cvs.getContext('2d');
        const w = cvs.parentElement.clientWidth;
        const h = 90;
        cvs.width = w;
        cvs.height = h;

        const padL = 50, padR = 8, padT = 14, padB = 12;
        const plotW = w - padL - padR;
        const plotH = h - padT - padB;

        c.fillStyle = '#16213e';
        c.fillRect(0, 0, w, h);

        const sorted = [...data].filter(v => v != null && isFinite(v)).sort((a, b) => a - b);
        if (sorted.length === 0) return;
        const minD = sorted[Math.floor(sorted.length * 0.01)];
        const maxD = sorted[Math.floor(sorted.length * 0.99)] || sorted[sorted.length - 1];
        const range = maxD - minD || 1;

        const fToX = (f) => padL + (f / data.length) * plotW;
        const dToY = (d) => padT + ((maxD - d) / range) * plotH;

        // Y-axis ticks and labels
        const nTicks = 4;
        c.fillStyle = '#8892a0';
        c.font = '9px sans-serif';
        c.textAlign = 'right';
        for (let i = 0; i <= nTicks; i++) {
            const val = minD + (maxD - minD) * (1 - i / nTicks);
            const y = padT + (i / nTicks) * plotH;
            c.fillText(val.toFixed(1), padL - 4, y + 3);
            // Grid line
            c.beginPath();
            c.moveTo(padL, y);
            c.lineTo(w - padR, y);
            c.strokeStyle = 'rgba(42, 58, 92, 0.4)';
            c.lineWidth = 0.5;
            c.stroke();
        }
        c.textAlign = 'left';

        // Label
        c.fillStyle = '#8892a0';
        c.font = '10px sans-serif';
        c.fillText(label, 4, padT + 4);

        // Threshold lines for distance plot
        if (canvasId === 'distPlotCanvas') {
            const mph = parseFloat(document.getElementById('paramMinPeakHeight')?.value);
            if (isFinite(mph) && mph >= minD && mph <= maxD) {
                const ty = dToY(mph);
                c.beginPath();
                c.setLineDash([4, 3]);
                c.moveTo(padL, ty);
                c.lineTo(w - padR, ty);
                c.strokeStyle = '#f44336';
                c.lineWidth = 1;
                c.stroke();
                c.setLineDash([]);
                c.fillStyle = '#f44336';
                c.font = '8px sans-serif';
                c.fillText('min_peak', w - padR - 44, ty - 3);
            }
            const vt = parseFloat(document.getElementById('paramValleyThresh')?.value);
            if (isFinite(vt) && vt >= minD && vt <= maxD) {
                const ty = dToY(vt);
                c.beginPath();
                c.setLineDash([4, 3]);
                c.moveTo(padL, ty);
                c.lineTo(w - padR, ty);
                c.strokeStyle = '#ff9800';
                c.lineWidth = 1;
                c.stroke();
                c.setLineDash([]);
                c.fillStyle = '#ff9800';
                c.font = '8px sans-serif';
                c.fillText('valley', w - padR - 30, ty - 3);
            }
        }

        // Data line
        c.beginPath();
        let started = false;
        for (let f = 0; f < data.length; f++) {
            const d = data[f];
            if (d == null || !isFinite(d)) { started = false; continue; }
            const x = fToX(f);
            const y = Math.max(padT, Math.min(padT + plotH, dToY(d)));
            if (!started) { c.moveTo(x, y); started = true; }
            else c.lineTo(x, y);
        }
        c.strokeStyle = color;
        c.lineWidth = 1;
        c.stroke();

        // Overlay event markers from current trial
        const trial = trials[currentEventTrialIdx];
        if (trial) {
            const sf = trial.start_frame;
            EVENT_TYPES.forEach(etype => {
                if (!eventVisibility[etype]) return;
                eventMarkers[etype].forEach(gf => {
                    if (gf < trial.start_frame || gf > trial.end_frame) return;
                    const localF = gf - sf;
                    const x = fToX(localF);
                    c.beginPath();
                    c.moveTo(x, padT);
                    c.lineTo(x, padT + plotH);
                    c.strokeStyle = EVENT_COLORS[etype] + '80';
                    c.lineWidth = 1;
                    c.stroke();
                });
            });
        }
    }

    // ── Event-type focus mode ─────────────────────────────

    // Which params are relevant for each focus mode
    const FOCUS_RELEVANCE = {
        open: {
            params: ['paramOpenThresh', 'paramNback', 'paramSsdRadius', 'paramOpenBias', 'paramDistGuard', 'paramGaussianSigma', 'paramMaxValidDist', 'paramMinEventGap'],
            cards: ['cardSsd'],
            canvases: ['distPlotCanvas', 'ssdPlotCanvas'],
        },
        peak: {
            params: ['paramMinPeakHeight', 'paramValleyThresh', 'paramMinEventGap', 'paramReversalRadius', 'paramPeakGuard', 'paramEdgeMinPeak', 'paramMaxValidDist'],
            cards: ['cardReversal'],
            canvases: ['distPlotCanvas', 'reversalPlotCanvas'],
        },
        close: {
            params: ['paramSsdRadius', 'paramCloseBias', 'paramDistGuard', 'paramGaussianSigma', 'paramMaxValidDist', 'paramMinEventGap'],
            cards: ['cardSsd'],
            canvases: ['distPlotCanvas', 'ssdPlotCanvas'],
        },
    };

    function setDetectFocus(focus) {
        detectFocus = focus;
        // Update button active states
        const btns = document.querySelectorAll('#detectFocusSelector button');
        btns.forEach(b => {
            b.classList.toggle('active', b.getAttribute('data-focus') === focus);
        });
        applyDetectFocus();
    }

    function applyDetectFocus() {
        if (detectFocus === 'all') {
            // Enable everything
            document.querySelectorAll('.detect-core-params label, .detect-param label').forEach(el => {
                el.classList.remove('detect-param-dimmed');
            });
            document.querySelectorAll('.detect-step-card').forEach(el => {
                el.classList.remove('detect-dimmed');
            });
            document.querySelectorAll('#metricPlotsContainer canvas').forEach(el => {
                el.classList.remove('detect-dimmed');
            });
            return;
        }

        const rel = FOCUS_RELEVANCE[detectFocus];
        if (!rel) return;

        // Dim/enable param labels based on their input ID
        document.querySelectorAll('.detect-core-params label, .detect-param label').forEach(el => {
            const input = el.querySelector('input[type="number"]');
            if (!input) return;
            if (rel.params.includes(input.id)) {
                el.classList.remove('detect-param-dimmed');
            } else {
                el.classList.add('detect-param-dimmed');
            }
        });

        // Dim/enable step cards
        document.querySelectorAll('.detect-step-card').forEach(el => {
            if (rel.cards.includes(el.id)) {
                el.classList.remove('detect-dimmed');
            } else {
                el.classList.add('detect-dimmed');
            }
        });

        // Dim/enable metric canvases
        document.querySelectorAll('#metricPlotsContainer canvas').forEach(el => {
            if (rel.canvases.includes(el.id)) {
                el.classList.remove('detect-dimmed');
            } else {
                el.classList.add('detect-dimmed');
            }
        });
    }

    async function runDetection() {
        const btn = document.getElementById('runDetectBtn');
        if (btn) { btn.textContent = 'Running…'; btn.disabled = true; }

        const snapshot = snapshotEventMarkers();

        const peaksOnly = document.getElementById('stepPeaksOnly').checked;
        const enforceSequence = document.getElementById('stepEnforceSequence').checked;

        const params = {
            min_peak_height: parseFloat(document.getElementById('paramMinPeakHeight').value),
            min_event_gap: parseInt(document.getElementById('paramMinEventGap').value),
            open_start_thresh: parseFloat(document.getElementById('paramOpenThresh').value),
            valley_thresh: parseFloat(document.getElementById('paramValleyThresh').value),
            nback: parseInt(document.getElementById('paramNback').value),
            reversal_search_radius: parseInt(document.getElementById('paramReversalRadius').value),
            ssd_search_radius: parseInt(document.getElementById('paramSsdRadius').value),
            open_bias: parseInt(document.getElementById('paramOpenBias').value),
            close_bias: parseInt(document.getElementById('paramCloseBias').value),
            dist_guard_factor: parseFloat(document.getElementById('paramDistGuard').value),
            peak_guard_factor: parseFloat(document.getElementById('paramPeakGuard').value),
            gaussian_sigma: parseFloat(document.getElementById('paramGaussianSigma').value),
            max_valid_dist: parseFloat(document.getElementById('paramMaxValidDist').value),
            edge_min_peak: parseFloat(document.getElementById('paramEdgeMinPeak').value),
        };

        const steps = {
            use_reversal: document.getElementById('stepReversal').checked,
            use_ssd: document.getElementById('stepSsd').checked,
            use_dist_guard: document.getElementById('stepSsd').checked,
            use_peak_guard: document.getElementById('stepReversal').checked,
        };

        const cached = metricsCache[currentEventTrialIdx];
        const metrics = cached
            ? {
                reversal: cached.reversal,
                motion_ssd: cached.motion_ssd,
                per_cam_ssd: cached.per_cam_ssd,
              }
            : null;

        try {
            const result = await API.post(`/api/labeling/sessions/${sessionId}/detect_events_v2`, {
                trial_index: currentEventTrialIdx,
                peaks_only: peaksOnly,
                enforce_sequence: enforceSequence,
                params,
                steps,
                metrics,
            });

            // Replace events for current trial only; keep other trials' events
            const trial = trials[currentEventTrialIdx];
            // In peaks_only mode, only replace peak markers (keep existing open/close)
            const typesToReplace = peaksOnly ? ['peak'] : EVENT_TYPES;
            typesToReplace.forEach(t => {
                const otherTrialEvents = eventMarkers[t].filter(
                    f => f < trial.start_frame || f > trial.end_frame
                );
                eventMarkers[t] = [...otherTrialEvents, ...(result[t] || [])].sort((a, b) => a - b);
            });

            // Safety net: ensure all saved events preserved
            for (const t of EVENT_TYPES) {
                for (const f of savedEventFrames[t]) {
                    if (!eventMarkers[t].includes(f)) eventMarkers[t].push(f);
                }
                eventMarkers[t].sort((a, b) => a - b);
            }

            pushEventUndo(snapshot);
            updateEventCounts();
            renderDistanceTrace();
            if (cached) showMetricPlotsForCurrentTrial();

            const trialEvents = eventsInTrial(currentEventTrialIdx);
            const total = EVENT_TYPES.reduce((s, t) => s + (trialEvents[t] || []).length, 0);
            const modeLabel = peaksOnly ? 'peaks' : 'events';
            updateLabelInfo(`Detected ${total} ${modeLabel} in ${trial.trial_name} — Save to keep`);
        } catch (e) {
            alert('Detection failed: ' + e.message);
        } finally {
            if (btn) { btn.textContent = 'Run Detection'; btn.disabled = false; }
        }
    }

    function prevEvent() {
        const { start, end } = getTrialFrameRange(currentEventTrialIdx);
        const allFrames = [...new Set(EVENT_TYPES.filter(t => eventVisibility[t]).flatMap(t => eventMarkers[t]))]
            .filter(f => f >= start && f <= end)
            .sort((a, b) => a - b);
        const prev = [...allFrames].reverse().find(f => f < currentFrame);
        if (prev !== undefined) goToFrame(prev);
    }

    function nextEvent() {
        const { start, end } = getTrialFrameRange(currentEventTrialIdx);
        const allFrames = [...new Set(EVENT_TYPES.filter(t => eventVisibility[t]).flatMap(t => eventMarkers[t]))]
            .filter(f => f >= start && f <= end)
            .sort((a, b) => a - b);
        const next = allFrames.find(f => f > currentFrame);
        if (next !== undefined) goToFrame(next);
    }

    function updateEventCounts() {
        const el = document.getElementById('eventCounts');
        if (!el) return;
        const trialEvents = eventsInTrial(currentEventTrialIdx);
        el.innerHTML = EVENT_TYPES.map(t => {
            const displayName = t[0].toUpperCase() + t.slice(1);
            const count = (trialEvents[t] || []).length;
            return `<span style="color:${EVENT_COLORS[t]};">●</span> ${displayName}: <strong>${count}</strong>`;
        }).join('<br>');
    }

    // ── Video export context ─────────────────────────
    function getExportContext() {
        const fps = trials[currentTrialIdx]?.fps || 30;
        return {
            videoEl,
            fps,
            playbackRate,
            nFrames: totalFrames,
            get currentFrame() { return currentFrame; },
            canvasLayers: [canvas],
            distanceCanvas: distCanvas,
            getCompositeSize() {
                return { width: canvas.width, height: canvas.height };
            },
            async seekAndRender(n) {
                currentFrame = Math.max(0, Math.min(n, totalFrames - 1));
                // tryRenderVideoFrame handles trial switching, seeking, and overlay drawing
                const rendered = await tryRenderVideoFrame(currentFrame);
                if (!rendered) {
                    // Fallback: load JPEG from backend
                    try {
                        currentImage = await loadImage(currentFrame, currentSide);
                        imgW = currentImage.width;
                        imgH = currentImage.height;
                        render();
                    } catch (e) {
                        console.error('Export: failed to render frame', n, e);
                    }
                }
                renderDistanceTrace();
                const fd = document.getElementById('frameDisplay');
                if (fd) fd.textContent = `Frame: ${currentFrame} / ${totalFrames - 1}`;
            },
            renderThreeJS() { return null; }, // no WebGL in label viewer
        };
    }

    // ── Public API ────────────────────────────────────
    return {
        init,
        nextFrame, prevFrame, nextLabel, prevLabel,
        nextGap, prevGap, acceptMergedLabels,
        toggleSide, togglePlay, resetZoom, cycleReviewMode, cycleFrameDisplay,
        saveLabels, commitSession, saveCorrectionsOnly,
        toggleV2Training,
        prevSubject, nextSubject,
        // Events mode
        loadEvents, saveEvents, detectEvents,
        setEventVisibility,
        placeEventType, deleteEvent, shiftEvent,
        prevEvent, nextEvent,
        // Trial switching
        prevTrial, nextTrial, setEventTrial,
        // Detection modal
        openDetectModal, closeDetectModal,
        runDetection, setDetectFocus,
        // Navigation persistence
        saveNavState,
        // Video export
        getExportContext,
        // MediaPipe crop box
        toggleMpCrop, saveMpCrop, cancelMpCrop, rerunMediapipe, clearMpHistory, runPose,
        // Refine flow (within corrections mode)
        startRefineFlow,
    };
})();

// Init on page load
document.addEventListener('DOMContentLoaded', () => labeler.init());
