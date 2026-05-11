/**
 * Events Page — standalone event annotation viewer/editor.
 *
 * Canvas-based stereo video viewer with distance trace, event markers,
 * auto-detect modal, and event editing tools.
 * Uses the labeling session API for frame serving, events, and detection.
 */
const eventsPage = (() => {
    // ── State ─────────��──────────────────────────────────
    let sessionId = null;
    let sessionInfo = null;
    let allSubjects = [];
    let currentSubjectId = null;
    let trials = [];
    let totalFrames = 0;

    let currentFrame = 0;
    let currentSide = 'OS';
    let cameraNames = ['OS', 'OD'];
    let cameraMode = 'stereo';
    let playing = false;
    let playTimer = null;
    let playbackRate = 1;
    const SPEED_PRESETS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 16, 60];

    // Video element for native playback
    let videoEl = null;
    let currentTrialIdx = -1;
    let currentVideoSide = '';
    let videoPlaying = false;
    let videoFrameMode = false;

    // Image fallback
    let currentImage = null;
    const imageCache = new Map();
    const IMAGE_CACHE_MAX = 30;
    const PREFETCH_AHEAD = 3;

    // Canvas
    let canvas, ctx;
    let containerEl;
    let distCanvas, distCtx;

    // Zoom/pan
    let scale = 1, offsetX = 0, offsetY = 0;
    let hasUserZoom = false;
    let imgW = 0, imgH = 0;
    let dragging = false;
    let dragStartX = 0, dragStartY = 0;
    let panStartOX = 0, panStartOY = 0;

    // Distance data
    let distances = null;
    let distViewStart = 0;
    let distViewFrames = 0;
    let distAutoScroll = true;
    let stableDistRange = null;

    // Event state
    let EVENT_TYPES = ['open', 'peak', 'close', 'pause'];
    let EVENT_COLORS = { open: '#00cc44', peak: '#ffcc00', close: '#ff4444', pause: '#cc66ff' };
    let EVENT_SHORTCUTS = { open: '1', peak: '2', close: '3', pause: '4' };
    const AUTO_DETECT_TYPES = ['open', 'peak', 'close'];
    let eventMarkers = {};
    let savedEventFrames = {};
    let eventVisibility = {};

    // Per-trial event filtering
    let currentEventTrialIdx = 0;
    // Metrics cache for auto-detect
    let metricsCache = {};
    let metricsLoading = new Set();
    let detectFocus = 'all';

    // Undo
    const undoStack = [];
    const redoStack = [];
    const MAX_UNDO = 50;

    // ── Helpers ───────��───────────────────────────────────
    const $ = id => document.getElementById(id);

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

    // ── Initialisation ───────────────────────────────────
    async function init() {
        // Parse session ID from URL
        const params = new URLSearchParams(window.location.search);
        sessionId = parseInt(params.get('session'));
        if (!sessionId) {
            alert('No session ID provided. Use ?session=N in the URL.');
            return;
        }

        // Setup canvases
        canvas = $('videoCanvas');
        ctx = canvas.getContext('2d');
        containerEl = $('canvasContainer');
        distCanvas = $('distanceCanvas');
        distCtx = distCanvas.getContext('2d');

        // Create hidden video element
        videoEl = document.createElement('video');
        videoEl.muted = true;
        videoEl.crossOrigin = 'anonymous';

        // Setup controls
        setupControls();
        setupCanvasEvents();

        // Load session
        await loadSession();
    }

    async function loadSession() {
        try {
            currentTrialIdx = -1;
            currentVideoSide = '';

            sessionInfo = await API.get(`/api/labeling/sessions/${sessionId}/info`);
            trials = sessionInfo.trials;
            totalFrames = sessionInfo.total_frames;

            // Dynamic event types from settings
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

            // Camera setup
            if (sessionInfo.camera_names) cameraNames = sessionInfo.camera_names;
            if (sessionInfo.camera_mode) cameraMode = sessionInfo.camera_mode;
            if (cameraMode === 'single') {
                cameraNames = [cameraNames[0] || 'OS'];
                currentSide = cameraNames[0];
            } else if (cameraMode === 'multicam' && trials.length > 0
                && trials[0].cameras && trials[0].cameras.length > 0) {
                cameraNames = trials[0].cameras.map(c => c.name);
                currentSide = cameraNames[0];
            } else {
                currentSide = cameraNames[0] || 'OS';
            }

            // Disable side toggle for non-stereo
            if (cameraMode !== 'stereo') {
                const btn = $('sideToggle');
                if (btn) { btn.disabled = true; btn.title = 'Single camera'; }
            }

            // Subject info
            currentSubjectId = sessionInfo.subject.id;
            sessionStorage.setItem('dlc_lastSubjectId', String(currentSubjectId));
            sessionStorage.setItem('dlc_lastSessionId', String(sessionId));
            if (typeof setLastSubject === 'function') setLastSubject(currentSubjectId);
            if (typeof setNavState === 'function') setNavState({ subjectId: currentSubjectId });

            // Populate subject dropdown
            try {
                allSubjects = await API.get('/api/subjects');
                const sel = $('subjectSelect');
                sel.innerHTML = '';
                allSubjects.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s.id;
                    opt.textContent = s.name;
                    if (s.id === currentSubjectId) opt.selected = true;
                    sel.appendChild(opt);
                });
                sel.addEventListener('change', () => switchSubject(parseInt(sel.value)));
            } catch (e) {
                console.log('Could not load subjects list');
            }

            // Update UI
            $('totalFrames').textContent = totalFrames;
            updateSideToggle();
            buildTrialButtons();
            buildEventsPanel();

            // Load distances from available stages
            await loadDistances();

            // Init distance trace window
            initDistanceTraceWindow();
            if (distances) computeStableDistRange();

            // Load saved events
            await loadEvents();

            // Go to frame 0 (or restored)
            let startFrame = 0;
            if (typeof getNavState === 'function') {
                const nav = getNavState();
                if (nav.subjectId === currentSubjectId && nav.frame != null
                    && nav.frame >= 0 && nav.frame < totalFrames) {
                    startFrame = nav.frame;
                }
                if (nav.side && cameraNames.includes(nav.side)) {
                    currentSide = nav.side;
                    updateSideToggle();
                }
            }
            await goToFrame(startFrame);

        } catch (e) {
            alert('Error loading session: ' + e.message);
            console.error(e);
        }
    }

    async function switchSubject(newSubjectId) {
        if (!newSubjectId) return;
        try {
            const session = await API.post(`/api/labeling/${newSubjectId}/sessions`, {
                session_type: 'events'
            });
            window.location.href = `/events?session=${session.id}`;
        } catch (e) {
            alert('Error switching subject: ' + e.message);
        }
    }

    // ── Distance Loading ─────���───────────────────────────
    const STAGE_CHAIN = ['corrections', 'refine', 'dlc', 'mp', 'labels'];

    async function loadDistances() {
        try {
            const stagesResult = await API.get(`/api/labeling/sessions/${sessionId}/available_stages`);
            const available = stagesResult.stages || [];
            const chain = STAGE_CHAIN.filter(s => available.includes(s));
            for (const stage of chain) {
                try {
                    const data = await API.get(`/api/labeling/sessions/${sessionId}/stage_data?stage=${stage}`);
                    if (data && data.distances && data.distances.some(d => d !== null)) {
                        distances = data.distances;
                        return;
                    }
                } catch (e) {
                    console.log(`Stage ${stage} data not available`);
                }
            }
        } catch (e) {
            console.log('Could not load stages:', e);
        }
    }

    function initDistanceTraceWindow() {
        if (!trials.length) return;
        const fps = trials[0].fps || 30;
        distViewFrames = Math.min(totalFrames, Math.round(fps * 15));
    }

    function computeStableDistRange() {
        if (!distances) return;
        const valid = distances.filter(d => d !== null && d !== undefined && isFinite(d));
        if (valid.length === 0) return;
        const sorted = [...valid].sort((a, b) => a - b);
        const p02 = sorted[Math.floor(sorted.length * 0.02)];
        const p98 = sorted[Math.floor(sorted.length * 0.98)] || sorted[sorted.length - 1];
        stableDistRange = { min: Math.max(0, p02 - 5), max: p98 + 10 };
    }

    // ── Frame Loading ────────────────────────────────────
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
                if (imageCache.size > IMAGE_CACHE_MAX) {
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
                const key = `${f}_${currentSide}`;
                if (!imageCache.has(key)) {
                    const img = new Image();
                    img.src = frameUrl(f, currentSide);
                    img.onload = () => { imageCache.set(key, img); };
                }
            }
        }
    }

    // ── Video-element frame rendering ────────────────────
    async function tryRenderVideoFrame(frame) {
        if (!videoEl || playing) return false;

        const trialIdx = getTrialForFrame(frame);
        const trial = trials[trialIdx];
        if (!trial) return false;

        let justLoaded = false;
        if (currentTrialIdx !== trialIdx || currentVideoSide !== currentSide) {
            const videoUrl = `/api/labeling/sessions/${sessionId}/video?trial=${trialIdx}&side=${encodeURIComponent(currentSide)}&_=${Date.now()}`;
            videoEl.src = videoUrl;
            currentTrialIdx = trialIdx;
            currentVideoSide = currentSide;
            const loaded = await new Promise(resolve => {
                if (videoEl.readyState >= 1) { resolve(true); return; }
                const timer = setTimeout(() => { resolve(false); }, 5000);
                const onLoaded = () => { clearTimeout(timer); videoEl.removeEventListener('error', onError); resolve(true); };
                const onError = () => { clearTimeout(timer); videoEl.removeEventListener('loadedmetadata', onLoaded); resolve(false); };
                videoEl.addEventListener('loadedmetadata', onLoaded, { once: true });
                videoEl.addEventListener('error', onError, { once: true });
            });
            if (!loaded) return false;
            justLoaded = true;
        }
        if (videoEl.readyState < 1) return false;

        const localFrame = frame - trial.start_frame;
        const frameOffset = trial.frame_offset || 0;
        const halfFrame = 0.5 / trial.fps;
        const targetTime = Math.max(0, (localFrame - frameOffset + 0.5) / trial.fps);

        if (justLoaded || Math.abs(videoEl.currentTime - targetTime) > halfFrame) {
            videoEl.currentTime = targetTime;
            await new Promise(resolve => {
                videoEl.addEventListener('seeked', resolve, { once: true });
                setTimeout(resolve, 2000);
            });
        }

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
            if (!hasUserZoom) fitImage();
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            ctx.drawImage(videoEl, sx, 0, sw, vh, 0, 0, sw, vh);
            ctx.restore();
        }

        videoFrameMode = true;
        return true;
    }

    // ── Frame Navigation ─────────────────────────────────
    async function goToFrame(frame) {
        if (frame < 0 || frame >= totalFrames) return;
        currentFrame = frame;
        if (typeof setNavState === 'function') {
            setNavState({ frame: currentFrame, trialIdx: getTrialForFrame(currentFrame) });
        }
        distAutoScroll = true;

        const renderedFromVideo = await tryRenderVideoFrame(frame);

        if (!renderedFromVideo) {
            videoFrameMode = false;
            try {
                currentImage = await loadImage(frame, currentSide);
                imgW = currentImage.width;
                imgH = currentImage.height;
                if (!hasUserZoom) fitImage();
                render();
                prefetchFrames(frame);
            } catch (e) {
                console.error('Failed to load frame', frame, e);
            }
        }

        updateFrameDisplay();
        renderDistanceTrace();

        // Trial-follows-frame
        const newTrial = getTrialForFrame(currentFrame);
        if (newTrial !== currentEventTrialIdx) {
            currentEventTrialIdx = newTrial;
            updateTrialLabel();
            updateEventCounts();
            updateTrialButtons();
            // Refresh detect modal if open
            const overlay = $('detectModalOverlay');
            if (overlay && overlay.classList.contains('active')) {
                const modalTrial = $('detectModalTrial');
                if (modalTrial && trials[currentEventTrialIdx]) {
                    const tn = trials[currentEventTrialIdx].trial_name;
                    modalTrial.textContent = `(${_shortTrialName(tn)})`;
                }
                showMetricPlotsForCurrentTrial();
            }
        }
    }

    function render() {
        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        canvas.width = cw;
        canvas.height = ch;
        ctx.clearRect(0, 0, cw, ch);

        if (videoFrameMode && videoEl && videoEl.readyState >= 2 && !videoPlaying) {
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
    }

    function fitImage() {
        if (!imgW || !imgH) return;
        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        scale = Math.min(cw / imgW, ch / imgH);
        offsetX = (cw - imgW * scale) / 2;
        offsetY = (ch - imgH * scale) / 2;
    }

    function updateFrameDisplay() {
        $('frameDisplay').textContent = currentFrame;
        const trial = trials[getTrialForFrame(currentFrame)];
        $('trialName').textContent = trial ? _shortTrialName(trial.trial_name) : '-';
    }

    function _shortTrialName(tn) {
        return tn.includes('_') ? tn.split('_').slice(1).join('_') : tn;
    }

    function getTrialForFrame(frame) {
        for (let i = 0; i < trials.length; i++) {
            if (frame >= trials[i].start_frame && frame <= trials[i].end_frame) return i;
        }
        return 0;
    }

    function getTrialFrameRange(trialIdx) {
        const t = trials[trialIdx];
        return t ? { start: t.start_frame, end: t.end_frame } : { start: 0, end: totalFrames - 1 };
    }

    // ── Play/Pause ───────────────────────────────────────
    function togglePlay() {
        playing = !playing;
        const btn = $('playBtn');
        if (playing) {
            btn.innerHTML = '\u23F8';
            const sliderIndex = parseInt($('speedSlider').value);
            playbackRate = SPEED_PRESETS[sliderIndex] || 1;
            startVideoPlayback();
        } else {
            btn.innerHTML = '&#9654;';
            stopVideoPlayback();
        }
    }

    async function startVideoPlayback() {
        if (!videoEl) { fallbackPlay(); return; }

        const trialIdx = getTrialForFrame(currentFrame);
        const trial = trials[trialIdx];
        if (!trial) return;

        const localFrame = currentFrame - trial.start_frame;
        const frameOffset = trial.frame_offset || 0;
        const startTime = Math.max(0, (localFrame - frameOffset + 0.5) / trial.fps);

        const videoUrl = `/api/labeling/sessions/${sessionId}/video?trial=${trialIdx}&side=${encodeURIComponent(currentSide)}&_=${Date.now()}`;
        if (currentTrialIdx !== trialIdx || currentVideoSide !== currentSide) {
            videoEl.src = videoUrl;
            currentTrialIdx = trialIdx;
            currentVideoSide = currentSide;
            const ready = await new Promise(resolve => {
                if (videoEl.readyState >= 3) { resolve(true); return; }
                const timer = setTimeout(() => { resolve(false); }, 8000);
                const onReady = () => { clearTimeout(timer); videoEl.removeEventListener('error', onError); resolve(true); };
                const onError = () => { clearTimeout(timer); videoEl.removeEventListener('canplay', onReady); resolve(false); };
                videoEl.addEventListener('canplay', onReady, { once: true });
                videoEl.addEventListener('error', onError, { once: true });
            });
            if (!ready || !playing) {
                if (!ready) fallbackPlay(trial.fps);
                return;
            }
        }

        try { videoEl.playbackRate = playbackRate; }
        catch (e) { fallbackPlay(trial.fps); return; }

        videoEl.currentTime = startTime;
        videoPlaying = true;

        try { await videoEl.play(); }
        catch (e) { videoPlaying = false; fallbackPlay(trial.fps); return; }

        videoDrawLoop(trial);
    }

    function videoDrawLoop(trial) {
        if (!playing || !videoPlaying) return;
        const fps = trial.fps || 30;
        const frameOffset = trial.frame_offset || 0;
        const elapsed = videoEl.currentTime;
        const globalFrame = trial.start_frame + Math.round(elapsed * fps + frameOffset - 0.5);

        if (globalFrame >= 0 && globalFrame < totalFrames && globalFrame !== currentFrame) {
            currentFrame = Math.min(globalFrame, totalFrames - 1);
            // Draw video to canvas
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
                if (!hasUserZoom) fitImage();
                ctx.save();
                ctx.translate(offsetX, offsetY);
                ctx.scale(scale, scale);
                ctx.drawImage(videoEl, sx, 0, sw, vh, 0, 0, sw, vh);
                ctx.restore();
            }
            updateFrameDisplay();
            renderDistanceTrace();

            const newTrial = getTrialForFrame(currentFrame);
            if (newTrial !== currentEventTrialIdx) {
                currentEventTrialIdx = newTrial;
                updateTrialLabel();
                updateEventCounts();
                updateTrialButtons();
            }
        }

        // Check if video ended (beyond trial end)
        if (globalFrame > trial.end_frame) {
            // Move to next trial or stop
            const nextTrialIdx = getTrialForFrame(trial.end_frame) + 1;
            if (nextTrialIdx < trials.length) {
                currentFrame = trials[nextTrialIdx].start_frame;
                videoPlaying = false;
                videoEl.pause();
                startVideoPlayback();
            } else {
                stopVideoPlayback();
                playing = false;
                $('playBtn').innerHTML = '&#9654;';
            }
            return;
        }

        requestAnimationFrame(() => videoDrawLoop(trial));
    }

    function fallbackPlay(fps) {
        fps = fps || (trials.length > 0 ? trials[0].fps : 30);
        const interval = 1000 / (fps * playbackRate);
        playTimer = setInterval(async () => {
            if (!playing) { clearInterval(playTimer); return; }
            const next = currentFrame + 1;
            if (next >= totalFrames) {
                clearInterval(playTimer);
                playing = false;
                $('playBtn').innerHTML = '&#9654;';
                return;
            }
            await goToFrame(next);
        }, interval);
    }

    function stopVideoPlayback() {
        if (videoPlaying) {
            videoEl.pause();
            videoPlaying = false;
        }
        if (playTimer) {
            clearInterval(playTimer);
            playTimer = null;
        }
    }

    // ── Trial Buttons ────────────────────────────────────
    function buildTrialButtons() {
        const container = $('trialBtns');
        container.innerHTML = '';
        trials.forEach((t, i) => {
            const btn = document.createElement('button');
            btn.className = 'trial-btn';
            btn.textContent = _shortTrialName(t.trial_name);
            btn.addEventListener('click', () => {
                setEventTrial(i);
            });
            container.appendChild(btn);
        });
        updateTrialButtons();
    }

    function updateTrialButtons() {
        const btns = $('trialBtns').querySelectorAll('.trial-btn');
        btns.forEach((btn, i) => {
            btn.classList.toggle('active', i === currentEventTrialIdx);
        });
    }

    function updateTrialLabel() {
        const label = $('trialSelectorLabel');
        if (label && trials[currentEventTrialIdx]) {
            label.textContent = _shortTrialName(trials[currentEventTrialIdx].trial_name);
        }
    }

    // ── Events Panel ─────────────────────────────────────
    function buildEventsPanel() {
        // Event type buttons
        const btnContainer = $('eventTypeButtons');
        btnContainer.innerHTML = '';
        EVENT_TYPES.forEach(t => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-sm';
            btn.style.color = EVENT_COLORS[t];
            btn.style.borderColor = EVENT_COLORS[t] + '60';
            const shortcut = EVENT_SHORTCUTS[t] || '';
            btn.textContent = `${t[0].toUpperCase() + t.slice(1)} (${shortcut})`;
            btn.addEventListener('click', () => placeEventType(t));
            btnContainer.appendChild(btn);
        });

        // Visibility toggles
        const toggleContainer = $('eventVisToggles');
        toggleContainer.innerHTML = '';
        EVENT_TYPES.forEach(t => {
            const label = document.createElement('label');
            label.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = eventVisibility[t] !== false;
            cb.addEventListener('change', () => {
                eventVisibility[t] = cb.checked;
                renderDistanceTrace();
            });
            const swatch = document.createElement('span');
            swatch.style.cssText = `display:inline-block;width:10px;height:10px;border-radius:2px;background:${EVENT_COLORS[t]};`;
            label.appendChild(cb);
            label.appendChild(swatch);
            label.appendChild(document.createTextNode(t[0].toUpperCase() + t.slice(1)));
            toggleContainer.appendChild(label);
        });
    }

    // ── Event Operations ─────────────────────────────────
    async function loadEvents() {
        try {
            const result = await API.get(`/api/labeling/sessions/${sessionId}/events`);
            EVENT_TYPES.forEach(t => {
                eventMarkers[t] = result[t] || [];
                savedEventFrames[t] = new Set(eventMarkers[t]);
            });
            currentEventTrialIdx = getTrialForFrame(currentFrame);
            updateTrialLabel();
            updateEventCounts();
            renderDistanceTrace();
            metricsCache = {};
            metricsLoading.clear();
        } catch (e) {
            console.log('Could not load events:', e);
        }
    }

    async function saveEvents() {
        try {
            const body = {};
            EVENT_TYPES.forEach(t => { if (eventVisibility[t]) body[t] = eventMarkers[t]; });
            await API.put(`/api/labeling/sessions/${sessionId}/events`, body);
            EVENT_TYPES.forEach(t => {
                if (eventVisibility[t]) savedEventFrames[t] = new Set(eventMarkers[t]);
            });
            renderDistanceTrace();
            const counts = EVENT_TYPES.filter(t => eventVisibility[t])
                .map(t => `${t}: ${eventMarkers[t].length}`).join(', ');
            setStatus(`Saved -- ${counts}`);
        } catch (e) {
            alert('Error saving events: ' + e.message);
        }
    }

    function placeEventType(type) {
        if (!EVENT_TYPES.includes(type)) return;
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

    function _plottedTypeAtFrame() {
        for (const t of EVENT_TYPES)
            if (eventVisibility[t] && eventMarkers[t].includes(currentFrame)) return t;
        return null;
    }

    function deleteEvent() {
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
        goToFrame(newFrame);
    }

    function setEventTrial(trialIdx) {
        if (trialIdx < 0 || trialIdx >= trials.length) return;
        currentEventTrialIdx = trialIdx;
        goToFrame(trials[trialIdx].start_frame);
        updateTrialLabel();
        updateTrialButtons();
    }

    function eventsInTrial(trialIdx) {
        const { start, end } = getTrialFrameRange(trialIdx);
        const result = {};
        EVENT_TYPES.forEach(t => {
            result[t] = eventMarkers[t].filter(f => f >= start && f <= end);
        });
        return result;
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
        const el = $('eventCounts');
        if (!el) return;
        const trialEvents = eventsInTrial(currentEventTrialIdx);
        el.innerHTML = EVENT_TYPES.map(t => {
            const displayName = t[0].toUpperCase() + t.slice(1);
            const count = (trialEvents[t] || []).length;
            return `<span style="color:${EVENT_COLORS[t]};">\u25CF</span> ${displayName}: <strong>${count}</strong>`;
        }).join('<br>');
    }

    // ��─ Undo/Redo ────────────────────────────────────────
    function snapshotEventMarkers() {
        const snap = {};
        EVENT_TYPES.forEach(t => snap[t] = [...(eventMarkers[t] || [])]);
        return snap;
    }

    function pushEventUndo(prevSnapshot) {
        undoStack.push({ prev: prevSnapshot, frame: currentFrame });
        if (undoStack.length > MAX_UNDO) undoStack.shift();
        redoStack.length = 0;
    }

    function undo() {
        if (undoStack.length === 0) return;
        const action = undoStack.pop();
        const currentSnapshot = snapshotEventMarkers();
        redoStack.push({ prev: currentSnapshot, frame: action.frame });
        for (const t of EVENT_TYPES) eventMarkers[t] = [...(action.prev[t] || [])];
        updateEventCounts();
        renderDistanceTrace();
        goToFrame(action.frame);
    }

    function redo() {
        if (redoStack.length === 0) return;
        const action = redoStack.pop();
        const currentSnapshot = snapshotEventMarkers();
        undoStack.push({ prev: currentSnapshot, frame: action.frame });
        for (const t of EVENT_TYPES) eventMarkers[t] = [...(action.prev[t] || [])];
        updateEventCounts();
        renderDistanceTrace();
        goToFrame(action.frame);
    }

    // ── Distance Trace Rendering ─────────────────────────
    function ensureFrameVisible() {
        if (distViewFrames <= 0 || distViewFrames >= totalFrames) { distViewStart = 0; return; }
        if (currentFrame < distViewStart) distViewStart = currentFrame;
        else if (currentFrame >= distViewStart + distViewFrames) distViewStart = currentFrame - distViewFrames + 1;
        clampDistView();
    }

    function clampDistView() {
        distViewStart = Math.max(0, Math.min(totalFrames - distViewFrames, distViewStart));
    }

    function renderDistanceTrace() {
        if (!distCanvas || !distCtx || !distances) return;

        const w = distCanvas.parentElement ? distCanvas.parentElement.clientWidth : distCanvas.clientWidth;
        const h = 140;
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

        let minD, maxD;
        if (stableDistRange) {
            minD = stableDistRange.min;
            maxD = stableDistRange.max;
        } else {
            const valid = distances.filter(d => d !== null && d !== undefined && isFinite(d));
            if (valid.length === 0) return;
            const sorted = [...valid].sort((a, b) => a - b);
            minD = Math.max(0, sorted[Math.floor(sorted.length * 0.02)] - 5);
            maxD = (sorted[Math.floor(sorted.length * 0.98)] || sorted[sorted.length - 1]) + 10;
        }

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

        // Distance line
        distCtx.beginPath();
        let started = false;
        for (let f = Math.max(0, vStart - 1); f < vEnd + 1 && f < distances.length; f++) {
            const d = distances[f];
            if (d === null || d === undefined) { started = false; continue; }
            const x = fToX(f);
            const y = dToY(d);
            if (!started) { distCtx.moveTo(x, y); started = true; }
            else distCtx.lineTo(x, y);
        }
        distCtx.strokeStyle = 'rgba(74, 158, 255, 0.9)';
        distCtx.lineWidth = 1.5;
        distCtx.stroke();

        // Event markers
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
                const inCurrentTrial = f >= trialRange.start && f <= trialRange.end;
                distCtx.globalAlpha = inCurrentTrial ? 1.0 : 0.25;

                // Diamond marker: filled = saved, outline = pending
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

                // Highlight active frame
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

        // Current frame cursor
        const cx = fToX(currentFrame);
        distCtx.beginPath();
        distCtx.moveTo(cx, padT);
        distCtx.lineTo(cx, h - padB);
        distCtx.strokeStyle = 'rgba(255,255,255,0.5)';
        distCtx.lineWidth = 1;
        distCtx.stroke();
    }

    function distXToFrame(clientX) {
        const rect = distCanvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const padL = 40, padR = 8;
        const plotW = distCanvas.width - padL - padR;
        const effectiveViewFrames = (distViewFrames > 0 && distViewFrames < totalFrames)
            ? distViewFrames : totalFrames;
        const f = distViewStart + ((x - padL) / plotW) * effectiveViewFrames;
        return Math.max(0, Math.min(totalFrames - 1, Math.round(f)));
    }

    function _findNearestEventMarker(clientX) {
        const rect = distCanvas.getBoundingClientRect();
        const x = clientX - rect.left;
        let best = null;
        let bestDist = 12; // pixel threshold
        const effectiveViewFrames = (distViewFrames > 0 && distViewFrames < totalFrames)
            ? distViewFrames : totalFrames;
        const padL = 40, padR = 8;
        const plotW = distCanvas.width - padL - padR;
        const fToX = (f) => padL + ((f - distViewStart) / effectiveViewFrames) * plotW;
        EVENT_TYPES.forEach(etype => {
            if (!eventVisibility[etype]) return;
            eventMarkers[etype].forEach(f => {
                const mx = fToX(f);
                const d = Math.abs(mx - x);
                if (d < bestDist) { bestDist = d; best = { frame: f, type: etype }; }
            });
        });
        return best;
    }

    // ── Auto-detect Modal ────────────────────────────────
    function openDetectModal() {
        const overlay = $('detectModalOverlay');
        if (overlay) overlay.classList.add('active');
        const trialLabel = $('detectModalTrial');
        if (trialLabel && trials[currentEventTrialIdx]) {
            trialLabel.textContent = `(${_shortTrialName(trials[currentEventTrialIdx].trial_name)})`;
        }

        // Auto-default "Peaks only" based on saved events
        const peaksOnlyCb = $('stepPeaksOnly');
        if (peaksOnlyCb) {
            const trialEvts = eventsInTrial(currentEventTrialIdx);
            const hasSavedPeaks = (savedEventFrames.peak || new Set()).size > 0 &&
                trialEvts.peak && trialEvts.peak.some(f => savedEventFrames.peak.has(f));
            peaksOnlyCb.checked = !hasSavedPeaks;
        }

        showMetricPlotsForCurrentTrial();
        applyDetectFocus();
    }

    function closeDetectModal() {
        const overlay = $('detectModalOverlay');
        if (overlay) overlay.classList.remove('active');
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
            const overlay = $('detectModalOverlay');
            if (overlay && overlay.classList.contains('active') && currentEventTrialIdx === trialIdx) {
                showMetricPlotsForCurrentTrial();
            }
        } catch (e) {
            console.log(`Metrics computation failed for trial ${trialIdx}:`, e);
        } finally {
            metricsLoading.delete(trialIdx);
        }
    }

    function showMetricPlotsForCurrentTrial() {
        const loading = $('metricPlotsLoading');
        const container = $('metricPlotsContainer');
        const unavailable = $('metricPlotsUnavailable');

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
            const trialToWatch = currentEventTrialIdx;
            const poll = setInterval(() => {
                if (metricsCache[trialToWatch]) {
                    clearInterval(poll);
                    if (currentEventTrialIdx === trialToWatch) showMetricPlotsForCurrentTrial();
                } else if (!metricsLoading.has(trialToWatch)) {
                    clearInterval(poll);
                    if (currentEventTrialIdx === trialToWatch) {
                        if (loading) loading.style.display = 'none';
                        if (unavailable) unavailable.style.display = 'block';
                    }
                }
            }, 300);
        } else {
            if (container) container.style.display = 'none';
            if (unavailable) unavailable.style.display = 'none';
            if (loading) loading.style.display = 'block';
            computeMetricsForTrial(currentEventTrialIdx);
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
        const cvs = $(canvasId);
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

        // Y-axis ticks
        const nTicks = 4;
        c.fillStyle = '#8892a0';
        c.font = '9px sans-serif';
        c.textAlign = 'right';
        for (let i = 0; i <= nTicks; i++) {
            const val = minD + (maxD - minD) * (1 - i / nTicks);
            const y = padT + (i / nTicks) * plotH;
            c.fillText(val.toFixed(1), padL - 4, y + 3);
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
            const mph = parseFloat($('paramMinPeakHeight')?.value);
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
            const vt = parseFloat($('paramValleyThresh')?.value);
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
        let lineStarted = false;
        for (let f = 0; f < data.length; f++) {
            const d = data[f];
            if (d == null || !isFinite(d)) { lineStarted = false; continue; }
            const x = fToX(f);
            const y = Math.max(padT, Math.min(padT + plotH, dToY(d)));
            if (!lineStarted) { c.moveTo(x, y); lineStarted = true; }
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

    // ── Focus mode for detect modal ──────────────────────
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
        const btns = document.querySelectorAll('#detectFocusSelector button');
        btns.forEach(b => {
            b.classList.toggle('active', b.getAttribute('data-focus') === focus);
        });
        applyDetectFocus();
    }

    function applyDetectFocus() {
        if (detectFocus === 'all') {
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

        document.querySelectorAll('.detect-core-params label, .detect-param label').forEach(el => {
            const input = el.querySelector('input[type="number"]');
            if (!input) return;
            if (rel.params.includes(input.id)) el.classList.remove('detect-param-dimmed');
            else el.classList.add('detect-param-dimmed');
        });

        document.querySelectorAll('.detect-step-card').forEach(el => {
            if (rel.cards.includes(el.id)) el.classList.remove('detect-dimmed');
            else el.classList.add('detect-dimmed');
        });

        document.querySelectorAll('#metricPlotsContainer canvas').forEach(el => {
            if (rel.canvases.includes(el.id)) el.classList.remove('detect-dimmed');
            else el.classList.add('detect-dimmed');
        });
    }

    async function runDetection() {
        const btn = $('runDetectBtn');
        if (btn) { btn.textContent = 'Running...'; btn.disabled = true; }

        const snapshot = snapshotEventMarkers();

        const peaksOnly = $('stepPeaksOnly').checked;
        const enforceSequence = $('stepEnforceSequence').checked;

        const params = {
            min_peak_height: parseFloat($('paramMinPeakHeight').value),
            min_event_gap: parseInt($('paramMinEventGap').value),
            open_start_thresh: parseFloat($('paramOpenThresh').value),
            valley_thresh: parseFloat($('paramValleyThresh').value),
            nback: parseInt($('paramNback').value),
            reversal_search_radius: parseInt($('paramReversalRadius').value),
            ssd_search_radius: parseInt($('paramSsdRadius').value),
            open_bias: parseInt($('paramOpenBias').value),
            close_bias: parseInt($('paramCloseBias').value),
            dist_guard_factor: parseFloat($('paramDistGuard').value),
            peak_guard_factor: parseFloat($('paramPeakGuard').value),
            gaussian_sigma: parseFloat($('paramGaussianSigma').value),
            max_valid_dist: parseFloat($('paramMaxValidDist').value),
            edge_min_peak: parseFloat($('paramEdgeMinPeak').value),
        };

        const steps = {
            use_reversal: $('stepReversal').checked,
            use_ssd: $('stepSsd').checked,
            use_dist_guard: $('stepSsd').checked,
            use_peak_guard: $('stepReversal').checked,
        };

        const cached = metricsCache[currentEventTrialIdx];
        const metrics = cached
            ? { reversal: cached.reversal, motion_ssd: cached.motion_ssd, per_cam_ssd: cached.per_cam_ssd }
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

            const trial = trials[currentEventTrialIdx];
            const typesToReplace = peaksOnly ? ['peak'] : EVENT_TYPES;
            typesToReplace.forEach(t => {
                const otherTrialEvents = eventMarkers[t].filter(
                    f => f < trial.start_frame || f > trial.end_frame
                );
                eventMarkers[t] = [...otherTrialEvents, ...(result[t] || [])].sort((a, b) => a - b);
            });

            // Safety net: ensure saved events preserved
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
            const tn = _shortTrialName(trial.trial_name);
            setStatus(`Detected ${total} ${modeLabel} in ${tn} -- Save to keep`);
        } catch (e) {
            alert('Detection failed: ' + e.message);
        } finally {
            if (btn) { btn.textContent = 'Run Detection'; btn.disabled = false; }
        }
    }

    // ── UI Helpers ─────────���──────────────────────────────
    function setStatus(msg) {
        const el = $('statusInfo');
        if (el) el.textContent = msg;
    }

    function updateSideToggle() {
        const btn = $('sideToggle');
        if (btn) btn.textContent = currentSide;
    }

    function toggleSide() {
        if (cameraMode === 'single') return;
        const idx = cameraNames.indexOf(currentSide);
        currentSide = cameraNames[(idx + 1) % cameraNames.length];
        updateSideToggle();
        currentTrialIdx = -1; // force video reload
        currentVideoSide = '';
        hasUserZoom = false;
        goToFrame(currentFrame);
    }

    // ── Controls Setup ──────���────────────────────────────
    function setupControls() {
        $('prevFrameBtn').addEventListener('click', () => goToFrame(currentFrame - 1));
        $('nextFrameBtn').addEventListener('click', () => goToFrame(currentFrame + 1));
        $('playBtn').addEventListener('click', togglePlay);
        $('sideToggle').addEventListener('click', toggleSide);

        // Speed slider
        const speedSlider = $('speedSlider');
        const speedDisplay = $('speedDisplay');
        speedSlider.value = SPEED_PRESETS.indexOf(1);
        if (speedSlider.value < 0) speedSlider.value = 6;
        speedSlider.max = SPEED_PRESETS.length - 1;
        speedSlider.addEventListener('input', () => {
            const idx = parseInt(speedSlider.value);
            playbackRate = SPEED_PRESETS[idx] || 1;
            speedDisplay.textContent = playbackRate >= 1 ? `${playbackRate}x` : `${playbackRate}x`;
            if (playing && videoPlaying) {
                try { videoEl.playbackRate = playbackRate; } catch (e) {}
            }
        });
        speedDisplay.textContent = '1x';

        // Sidebar buttons
        $('deleteEventBtn').addEventListener('click', deleteEvent);
        $('shiftLeftBtn').addEventListener('click', () => shiftEvent(-1));
        $('shiftRightBtn').addEventListener('click', () => shiftEvent(1));
        $('prevEventBtn').addEventListener('click', prevEvent);
        $('nextEventBtn').addEventListener('click', nextEvent);
        $('saveEventsBtn').addEventListener('click', saveEvents);
        $('detectEventsBtn').addEventListener('click', openDetectModal);

        // Detect modal buttons
        $('detectCancelBtn').addEventListener('click', closeDetectModal);
        $('runDetectBtn').addEventListener('click', runDetection);
        document.querySelectorAll('#detectFocusSelector button').forEach(btn => {
            btn.addEventListener('click', () => setDetectFocus(btn.getAttribute('data-focus')));
        });

        // Keyboard
        document.addEventListener('keydown', onKeyDown);
    }

    function onKeyDown(e) {
        // Ignore when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

        // Ctrl+Z / Ctrl+Y / Ctrl+S
        if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo(); return; }
        if ((e.ctrlKey || e.metaKey) && e.key === 'y') { e.preventDefault(); redo(); return; }
        if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveEvents(); return; }

        switch (e.key) {
            case 'ArrowLeft': case 'a': case 'A':
                e.preventDefault(); goToFrame(currentFrame - 1); break;
            case 'ArrowRight': case 's': case 'S':
                e.preventDefault(); goToFrame(currentFrame + 1); break;
            case ' ':
                e.preventDefault(); togglePlay(); break;
            case 'e': case 'E':
                toggleSide(); break;
            case 'z': case 'Z':
                if (!e.ctrlKey && !e.metaKey) { hasUserZoom = false; fitImage(); render(); renderDistanceTrace(); }
                break;
            case 'q': case 'Q':
                prevEvent(); break;
            case 'w': case 'W':
                nextEvent(); break;
            case 'x': case 'X':
                deleteEvent(); break;
            case '[':
                shiftEvent(-1); break;
            case ']':
                shiftEvent(1); break;
            default:
                // Event type shortcuts (1, 2, 3, 4, ...)
                for (const [type, shortcut] of Object.entries(EVENT_SHORTCUTS)) {
                    if (e.key === shortcut) {
                        placeEventType(type);
                        break;
                    }
                }
                break;
        }
    }

    function setupCanvasEvents() {
        // Canvas zoom/pan
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const newScale = scale * zoomFactor;

            offsetX = mx - (mx - offsetX) * (newScale / scale);
            offsetY = my - (my - offsetY) * (newScale / scale);
            scale = newScale;
            hasUserZoom = true;
            render();
        });

        canvas.addEventListener('mousedown', (e) => {
            if (e.button === 0) {
                dragging = true;
                dragStartX = e.clientX;
                dragStartY = e.clientY;
                panStartOX = offsetX;
                panStartOY = offsetY;
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            offsetX = panStartOX + (e.clientX - dragStartX);
            offsetY = panStartOY + (e.clientY - dragStartY);
            hasUserZoom = true;
            render();
        });

        window.addEventListener('mouseup', () => { dragging = false; });

        // Distance trace click and scroll
        distCanvas.addEventListener('mousedown', onDistTraceDown);
        distCanvas.addEventListener('wheel', onDistTraceWheel);

        // Resize
        window.addEventListener('resize', () => {
            if (!hasUserZoom) fitImage();
            render();
            renderDistanceTrace();
        });
    }

    function onDistTraceDown(ev) {
        let lastX = ev.clientX;
        let moved = false;

        const onMove = (e) => {
            const dx = e.clientX - lastX;
            if (Math.abs(e.clientX - ev.clientX) > 3) moved = true;
            if (!moved) return;
            // Drag to pan
            distAutoScroll = false;
            const effectiveViewFrames = (distViewFrames > 0 && distViewFrames < totalFrames)
                ? distViewFrames : totalFrames;
            const rect = distCanvas.getBoundingClientRect();
            const padL = 40, padR = 8;
            const plotW = rect.width - padL - padR;
            const pxPerFrame = plotW / effectiveViewFrames;
            distViewStart -= Math.round(dx / pxPerFrame);
            clampDistView();
            renderDistanceTrace();
            lastX = e.clientX;
        };

        const onUp = (e) => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
            if (!moved) {
                const marker = _findNearestEventMarker(ev.clientX);
                const targetFrame = marker ? marker.frame : distXToFrame(ev.clientX);
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
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        const padL = 40, padR = 8;
        const plotW = distCanvas.getBoundingClientRect().width - padL - padR;
        const effectiveViewFrames = (distViewFrames > 0 && distViewFrames < totalFrames)
            ? distViewFrames : totalFrames;
        const pxPerFrame = plotW / effectiveViewFrames;
        const frameDelta = Math.round(delta / Math.max(1, pxPerFrame));
        distViewStart += Math.sign(frameDelta) * Math.max(1, Math.abs(frameDelta));
        clampDistView();
        renderDistanceTrace();
    }

    // ── Public API ───────────────────────────────────────
    // Init on load
    document.addEventListener('DOMContentLoaded', init);

    return {
        goToFrame,
        togglePlay,
        prevEvent,
        nextEvent,
        deleteEvent,
        shiftEvent,
        placeEventType,
        saveEvents,
        loadEvents,
        openDetectModal,
        closeDetectModal,
        runDetection,
        setDetectFocus,
        undo,
        redo,
    };
})();
