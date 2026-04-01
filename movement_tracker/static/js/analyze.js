/**
 * Analyze Page — 2D keypoint viewer with distance trace
 *
 * Canvas-based stereo video viewer with MediaPipe, Vision, and DLC overlays,
 * distance trace chart, per-finger visibility toggles, and zoom/pan.
 */

const analyzeViewer = (() => {
    // ── State ────────────────────────────────────────────────
    let allSubjects = [];
    let subjectId = null;
    let subjectName = '';
    let trials = [];
    let currentTrialIdx = -1;
    let trialData = null;

    let currentFrame = 0;
    let totalFrames = 0;
    let fps = 30;
    let currentSide = 'OS';
    let cameraNames = ['OS', 'OD'];
    let cameraMode = 'stereo';
    let playing = false;
    let playTimer = null;
    let playbackRate = 1;
    const SPEED_PRESETS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120];

    // Layer visibility
    let showVideo = true;
    let showMediapipe = true;
    let showVision = false;
    let showDLC = false;
    let showSkeleton = true;

    // Finger visibility
    const fingerVisibility = {
        wrist: true, thumb: true, index: true,
        middle: true, ring: true, pinky: true,
    };
    let visibleJoints = new Set([...Array(21).keys()]);

    // Distance metric
    let selectedMetric = '';

    // Canvas
    let canvas, ctx;
    let distCanvas, distCtx;
    let videoEl;

    // Zoom/pan (in video-pixel space)
    let scale = 1, offsetX = 0, offsetY = 0;
    let defaultScale = 1, defaultOX = 0, defaultOY = 0;
    let hasUserZoom = false;
    let dragging = null;
    let dragStartX = 0, dragStartY = 0;
    let panStartOX = 0, panStartOY = 0;

    // Video dimensions
    let imgW = 0, imgH = 0, midline = 0;

    // Hand skeleton chains (MediaPipe 21-joint model)
    const HAND_SKELETON = [
        [0, 1, 2, 3, 4],       // thumb
        [0, 5, 6, 7, 8],       // index
        [0, 9, 10, 11, 12],    // middle
        [0, 13, 14, 15, 16],   // ring
        [0, 17, 18, 19, 20],   // pinky
    ];

    // ── Helpers ───────────────────────────────────────────────
    const $ = id => document.getElementById(id);

    async function api(url, opts) {
        const resp = await fetch(url, opts);
        if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
        return resp.json();
    }

    function isJointVisible(j) { return visibleJoints.has(j); }
    function isBoneVisible(i, j) { return visibleJoints.has(i) && visibleJoints.has(j); }

    function updateVisibleJoints() {
        visibleJoints.clear();
        if (!trialData) return;
        const groups = trialData.finger_groups;
        if (!groups) {
            // Fallback: all joints visible
            for (let i = 0; i < 21; i++) visibleJoints.add(i);
            return;
        }
        for (const [finger, joints] of Object.entries(groups)) {
            if (fingerVisibility[finger]) {
                joints.forEach(j => visibleJoints.add(j));
            }
        }
    }

    /** Convert skeleton chains to bone pairs for rendering. */
    function skeletonBones() {
        if (trialData && trialData.skeleton) return trialData.skeleton;
        // Derive from chains
        const bones = [];
        for (const chain of HAND_SKELETON) {
            for (let i = 0; i < chain.length - 1; i++) {
                bones.push([chain[i], chain[i + 1]]);
            }
        }
        return bones;
    }

    // ── Initialisation ───────────────────────────────────────
    async function init() {
        const params = new URLSearchParams(window.location.search);
        const subjectParam = params.get('subject');

        // Load camera names from settings
        try {
            const cfg = await api('/api/settings');
            if (cfg.default_camera_mode) cameraMode = cfg.default_camera_mode;
            if (Array.isArray(cfg.camera_names) && cfg.camera_names.length >= 1) {
                cameraNames = cfg.camera_names;
                currentSide = cameraNames[0];
            }
        } catch { /* defaults */ }

        // Load subjects
        allSubjects = await api('/api/subjects');
        const sel = $('subjectSelect');
        sel.innerHTML = '';
        allSubjects.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            sel.appendChild(opt);
        });

        // Restore subject from URL, nav state, or sessionStorage
        const nav = typeof getNavState === 'function' ? getNavState() : {};
        const savedSubject = subjectParam
            || (nav.subjectId ? String(nav.subjectId) : null)
            || sessionStorage.getItem('analyze_lastSubjectId');
        if (savedSubject && allSubjects.some(s => String(s.id) === savedSubject)) {
            sel.value = savedSubject;
        } else if (allSubjects.length) {
            sel.value = allSubjects[0].id;
        }

        sel.addEventListener('change', () => loadSubject(parseInt(sel.value)));

        // Setup canvases
        canvas = $('videoCanvas');
        ctx = canvas.getContext('2d');
        distCanvas = $('distanceCanvas');
        distCtx = distCanvas.getContext('2d');
        videoEl = $('videoEl');

        // Disable side toggle for non-stereo modes
        if (cameraMode !== 'stereo') {
            const btn = $('sideToggle');
            if (btn) { btn.disabled = true; btn.title = 'Single camera'; }
        }

        // Setup controls and events
        setupControls();
        setupCanvasEvents();

        // Load initial subject and restore trial/frame from nav state
        if (sel.value) {
            const navState = typeof getNavState === 'function' ? getNavState() : {};
            await loadSubject(parseInt(sel.value));
            if (navState.subjectId === parseInt(sel.value)) {
                if (navState.trialIdx != null && navState.trialIdx >= 0 && navState.trialIdx < trials.length) {
                    await loadTrial(navState.trialIdx);
                    if (navState.frame != null && trialData && navState.frame >= 0 && navState.frame < trialData.n_frames) {
                        goToFrame(navState.frame);
                    }
                }
                if (navState.side && cameraNames.includes(navState.side) && cameraMode === 'stereo') {
                    currentSide = navState.side;
                    $('sideToggle').textContent = currentSide;
                    computeAutoCrop();
                    render();
                }
            }
        }
    }

    // ── Subject / Trial loading ──────────────────────────────
    async function loadSubject(sid) {
        subjectId = sid;
        sessionStorage.setItem('analyze_lastSubjectId', String(sid));
        if (typeof setNavState === 'function') setNavState({ subjectId: sid });
        const subj = allSubjects.find(s => s.id === sid);
        subjectName = subj ? subj.name : '';
        if (subj && subj.camera_mode) cameraMode = subj.camera_mode;

        try {
            trials = await api(`/api/analyze/${sid}/trials`);
        } catch {
            trials = [];
        }

        const trialBtns = $('trialBtns');
        trialBtns.innerHTML = '';
        if (!trials.length) {
            trialBtns.innerHTML = '<span style="font-size:12px;color:var(--text-muted);">No data</span>';
            trialData = null;
            render();
            return;
        }

        trials.forEach((t, i) => {
            const btn = document.createElement('button');
            btn.className = 'trial-btn';
            btn.textContent = t.trial_stem || t.trial_name || `Trial ${i}`;
            btn.title = t.trial_name || '';
            btn.addEventListener('click', () => loadTrial(i));
            trialBtns.appendChild(btn);
        });

        await loadTrial(0);
    }

    function highlightTrialButton(idx) {
        const btns = $('trialBtns').querySelectorAll('.trial-btn');
        btns.forEach((b, i) => b.classList.toggle('active', i === idx));
    }

    async function loadTrial(idx) {
        if (idx < 0 || idx >= trials.length) return;
        const trial = trials[idx];
        currentTrialIdx = idx;
        highlightTrialButton(idx);
        if (typeof setNavState === 'function') setNavState({ trialIdx: idx, frame: currentFrame });

        $('trialName').textContent = trial.trial_stem || trial.trial_name || '';
        $('totalFrames').textContent = trial.n_frames || '';

        // Load bulk data
        try {
            trialData = await api(`/api/analyze/${subjectId}/trial/${trial.trial_idx != null ? trial.trial_idx : idx}/data`);
        } catch (e) {
            console.error('Failed to load trial data:', e);
            trialData = null;
            return;
        }

        totalFrames = trialData.n_frames || 0;
        fps = trialData.fps || 30;

        // Populate distance selector
        const distSel = $('distanceSelect');
        distSel.innerHTML = '';
        if (trialData.distance_options) {
            // distance_options can be an array of {name,...} or an object keyed by name
            if (Array.isArray(trialData.distance_options)) {
                trialData.distance_options.forEach(opt => {
                    const el = document.createElement('option');
                    el.value = opt.name || opt;
                    el.textContent = opt.name || opt;
                    distSel.appendChild(el);
                });
            } else {
                for (const name of Object.keys(trialData.distance_options)) {
                    const el = document.createElement('option');
                    el.value = name;
                    el.textContent = name;
                    distSel.appendChild(el);
                }
            }
        }
        if (trialData.distances) {
            for (const name of Object.keys(trialData.distances)) {
                if (!distSel.querySelector(`option[value="${CSS.escape(name)}"]`)) {
                    const el = document.createElement('option');
                    el.value = name;
                    el.textContent = name;
                    distSel.appendChild(el);
                }
            }
        }
        if (selectedMetric && distSel.querySelector(`option[value="${CSS.escape(selectedMetric)}"]`)) {
            distSel.value = selectedMetric;
        } else if (distSel.options.length) {
            selectedMetric = distSel.options[0].value;
            distSel.value = selectedMetric;
        }

        // Load video
        const trialIdx = trial.trial_idx != null ? trial.trial_idx : idx;
        videoEl.src = `/api/analyze/${subjectId}/trial/${trialIdx}/video`;
        videoEl.addEventListener('loadedmetadata', () => {
            imgW = videoEl.videoWidth;
            imgH = videoEl.videoHeight;
            midline = cameraMode === 'stereo' ? imgW / 2 : imgW;
            sizeCanvases();
            if (!hasUserZoom) computeAutoCrop();
            // Seek to midpoint of frame 0
            videoEl.currentTime = 0.5 / fps;
            videoEl.addEventListener('seeked', () => {
                render();
                renderDistanceTrace();
            }, { once: true });
        }, { once: true });

        // Update state
        currentFrame = 0;
        $('frameDisplay').textContent = '0';
        updateVisibleJoints();
    }

    // ── Controls setup ───────────────────────────────────────
    function setupControls() {
        $('prevFrameBtn').addEventListener('click', () => goToFrame(currentFrame - 1));
        $('nextFrameBtn').addEventListener('click', () => goToFrame(currentFrame + 1));
        $('playBtn').addEventListener('click', togglePlay);
        $('sideToggle').addEventListener('click', toggleSide);
        $('resetZoomBtn').addEventListener('click', resetZoom);

        const prevSubBtn = $('prevSubjectBtn');
        const nextSubBtn = $('nextSubjectBtn');
        if (prevSubBtn) prevSubBtn.addEventListener('click', prevSubject);
        if (nextSubBtn) nextSubBtn.addEventListener('click', nextSubject);

        // Speed slider
        const speedSlider = $('speedSlider');
        speedSlider.addEventListener('input', () => {
            playbackRate = SPEED_PRESETS[parseInt(speedSlider.value)];
            $('speedDisplay').textContent = playbackRate + 'x';
        });

        // Layer toggles
        $('showVideo').addEventListener('change', e => {
            showVideo = e.target.checked;
            render();
        });
        $('showMP').addEventListener('change', e => {
            showMediapipe = e.target.checked;
            render();
            renderDistanceTrace();
        });
        $('showVision').addEventListener('change', e => {
            showVision = e.target.checked;
            render();
            renderDistanceTrace();
        });
        $('showDLC').addEventListener('change', e => {
            showDLC = e.target.checked;
            render();
            renderDistanceTrace();
        });
        $('showSkeleton').addEventListener('change', e => {
            showSkeleton = e.target.checked;
            render();
        });

        // Distance selector
        $('distanceSelect').addEventListener('change', () => {
            selectedMetric = $('distanceSelect').value;
            renderDistanceTrace();
        });

        // Finger toggles
        document.querySelectorAll('#fingerToggles input').forEach(cb => {
            cb.addEventListener('change', () => {
                fingerVisibility[cb.dataset.finger] = cb.checked;
                updateFingerVisibility();
            });
        });

        // Keyboard
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            switch (e.key) {
                case 'a': case 'ArrowLeft':
                    goToFrame(currentFrame - 1); e.preventDefault(); break;
                case 's': case 'ArrowRight':
                    goToFrame(currentFrame + 1); e.preventDefault(); break;
                case ' ':
                    togglePlay(); e.preventDefault(); break;
                case 'e':
                    toggleSide(); break;
                case 'z':
                    resetZoom(); break;
            }
        });
    }

    function toggleSide() {
        if (cameraMode !== 'stereo') return;
        currentSide = currentSide === cameraNames[0] ? cameraNames[1] : cameraNames[0];
        $('sideToggle').textContent = currentSide;
        if (typeof setNavState === 'function') setNavState({ side: currentSide });
        hasUserZoom = false;
        computeAutoCrop();
        render();
        renderDistanceTrace();
    }

    function resetZoom() {
        hasUserZoom = false;
        computeAutoCrop();
        render();
    }

    function updateFingerVisibility() {
        updateVisibleJoints();
        render();
        renderDistanceTrace();
    }

    function prevSubject() {
        const sel = $('subjectSelect');
        const idx = allSubjects.findIndex(s => s.id === subjectId);
        if (idx > 0) {
            sel.value = allSubjects[idx - 1].id;
            loadSubject(allSubjects[idx - 1].id);
        }
    }

    function nextSubject() {
        const sel = $('subjectSelect');
        const idx = allSubjects.findIndex(s => s.id === subjectId);
        if (idx < allSubjects.length - 1) {
            sel.value = allSubjects[idx + 1].id;
            loadSubject(allSubjects[idx + 1].id);
        }
    }

    // ── Frame navigation ─────────────────────────────────────
    function goToFrame(n) {
        if (!trialData) return;
        currentFrame = Math.max(0, Math.min(n, trialData.n_frames - 1));
        if (typeof setNavState === 'function') setNavState({ frame: currentFrame });
        sessionStorage.setItem('analyze_lastFrame', String(currentFrame));

        $('frameDisplay').textContent = currentFrame;

        // Update distance trace
        renderDistanceTrace();

        // Seek video then render
        if (videoEl.readyState >= 2 && fps) {
            const t = (currentFrame + 0.5) / fps;
            videoEl.currentTime = t;
            videoEl.addEventListener('seeked', render, { once: true });
        } else {
            render();
        }
    }

    function seekFrame(f) {
        goToFrame(f);
    }

    function togglePlay() {
        if (playing) {
            playing = false;
            videoEl.pause();
            if (playTimer) { cancelAnimationFrame(playTimer); playTimer = null; }
            $('playBtn').innerHTML = '&#9654;';
        } else {
            playing = true;
            $('playBtn').innerHTML = '&#9646;&#9646;';
            videoEl.playbackRate = Math.min(playbackRate, 16);
            videoEl.play().catch(() => {});
            playLoop();
        }
    }

    function playLoop() {
        if (!playing) return;
        if (videoEl.readyState >= 2 && trialData) {
            const f = Math.floor(videoEl.currentTime * fps);
            if (f !== currentFrame && f >= 0 && f < trialData.n_frames) {
                currentFrame = f;
                $('frameDisplay').textContent = currentFrame;
                render();
                // Update distance trace less frequently during playback
                if (currentFrame % 5 === 0) renderDistanceTrace();
            }
            if (f >= trialData.n_frames - 1) {
                togglePlay();
                return;
            }
        }
        playTimer = requestAnimationFrame(playLoop);
    }

    // ── Canvas events (zoom/pan) ─────────────────────────────
    function setupCanvasEvents() {
        canvas.addEventListener('wheel', handleZoom);
        canvas.addEventListener('mousedown', e => {
            if (e.button === 0 || e.button === 1) handlePanStart(e);
        });
        canvas.addEventListener('mousemove', handlePanMove);
        canvas.addEventListener('mouseup', () => { dragging = null; });
        canvas.addEventListener('mouseleave', () => { dragging = null; });

        // Distance trace click to seek
        distCanvas.addEventListener('click', e => {
            if (!trialData) return;
            const rect = distCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const padLeft = 40; // match renderDistanceTrace padding
            const plotW = rect.width - padLeft - 8;
            const fx = x - padLeft;
            if (fx < 0) return;
            const f = Math.round((fx / plotW) * (trialData.n_frames - 1));
            goToFrame(Math.max(0, Math.min(f, trialData.n_frames - 1)));
        });

        // Resize observer
        const viewport = canvas.parentElement.parentElement;
        const ro = new ResizeObserver(() => {
            sizeCanvases();
            if (!hasUserZoom) computeAutoCrop();
            render();
            renderDistanceTrace();
        });
        ro.observe(viewport);
    }

    function handleZoom(e) {
        e.preventDefault();
        hasUserZoom = true;
        const viewport = canvas.parentElement.parentElement;
        const rect = viewport.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const newScale = Math.max(0.1, Math.min(scale * factor, 50));
        offsetX = mx - (mx - offsetX) * (newScale / scale);
        offsetY = my - (my - offsetY) * (newScale / scale);
        scale = newScale;
        render();
    }

    function handlePanStart(e) {
        dragging = 'pan';
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        panStartOX = offsetX;
        panStartOY = offsetY;
        hasUserZoom = true;
        e.preventDefault();
    }

    function handlePanMove(e) {
        if (dragging === 'pan') {
            offsetX = panStartOX + (e.clientX - dragStartX);
            offsetY = panStartOY + (e.clientY - dragStartY);
            render();
        }
    }

    function sizeCanvases() {
        const viewport = canvas.parentElement.parentElement; // .analyze-viewports
        const vw = viewport.clientWidth;
        const vh = viewport.clientHeight;
        canvas.width = vw;
        canvas.height = vh;

        distCanvas.width = distCanvas.clientWidth;
        distCanvas.height = distCanvas.clientHeight || 120;
    }

    // ── Auto-crop ────────────────────────────────────────────
    /**
     * Scan keypoints across frames to find hand bounding box,
     * then set default scale/offset to zoom-to-fit with 15% padding.
     */
    function computeAutoCrop() {
        if (!trialData || !canvas || imgW === 0) {
            defaultScale = 1; defaultOX = 0; defaultOY = 0;
            scale = 1; offsetX = 0; offsetY = 0;
            return;
        }

        const isStereo = cameraMode === 'stereo';
        const isLeft = currentSide === cameraNames[0];
        const side = currentSide;

        // Gather all keypoint sources for current side
        const mpLandmarks = trialData.mediapipe?.[side]?.landmarks;
        const viLandmarks = trialData.vision?.[side]?.landmarks;

        // For stereo right camera, offset coordinates by -midline
        const xOff = isStereo ? (isLeft ? 0 : -midline) : 0;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

        // Sample every 5th frame for speed
        const nFrames = trialData.n_frames || 0;
        for (let f = 0; f < nFrames; f += 5) {
            const sources = [];
            if (mpLandmarks && mpLandmarks[f]) sources.push(mpLandmarks[f]);
            if (viLandmarks && viLandmarks[f]) sources.push(viLandmarks[f]);

            for (const pts of sources) {
                if (!pts) continue;
                for (let j = 0; j < pts.length; j++) {
                    if (!pts[j]) continue;
                    const px = pts[j][0] + xOff;
                    const py = pts[j][1];
                    if (px < minX) minX = px;
                    if (py < minY) minY = py;
                    if (px > maxX) maxX = px;
                    if (py > maxY) maxY = py;
                }
            }
        }

        // Also scan DLC thumb/index
        const dlcData = trialData.dlc?.[side];
        if (dlcData) {
            for (let f = 0; f < nFrames; f += 5) {
                const pts = [dlcData.thumb?.[f], dlcData.index?.[f]];
                for (const pt of pts) {
                    if (!pt) continue;
                    const px = pt[0] + xOff;
                    const py = pt[1];
                    if (px < minX) minX = px;
                    if (py < minY) minY = py;
                    if (px > maxX) maxX = px;
                    if (py > maxY) maxY = py;
                }
            }
        }

        if (!isFinite(minX) || !isFinite(minY)) {
            defaultScale = 1; defaultOX = 0; defaultOY = 0;
            scale = 1; offsetX = 0; offsetY = 0;
            return;
        }

        // Add 15% padding
        const bw = maxX - minX;
        const bh = maxY - minY;
        const pad = 0.15;
        minX -= bw * pad; maxX += bw * pad;
        minY -= bh * pad; maxY += bh * pad;
        const cropW = maxX - minX;
        const cropH = maxY - minY;

        const cw = canvas.width;
        const ch = canvas.height;
        const sw = isStereo ? (isLeft ? midline : imgW - midline) : imgW;
        const bps = cw / sw; // base pixel scale

        const scaleX = cw / (cropW * bps);
        const scaleY = ch / (cropH * bps);
        defaultScale = Math.min(scaleX, scaleY);

        const cropCX = (minX + maxX) / 2;
        const cropCY = (minY + maxY) / 2;
        defaultOX = cw / 2 - defaultScale * cropCX * bps;
        defaultOY = ch / 2 - defaultScale * cropCY * bps;

        scale = defaultScale;
        offsetX = defaultOX;
        offsetY = defaultOY;
    }

    // ── 2D Rendering ─────────────────────────────────────────
    function render() {
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        if (videoEl.readyState >= 2 && imgW > 0) {
            const isStereo = cameraMode === 'stereo';
            const isLeft = currentSide === cameraNames[0];
            const sx = isStereo ? (isLeft ? 0 : midline) : 0;
            const sw = isStereo ? (isLeft ? midline : imgW - midline) : imgW;

            // Base pixel scale: maps 1 video pixel to 1 canvas pixel at scale=1
            const bps = w / sw;

            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);

            // Draw video frame
            if (showVideo) {
                ctx.drawImage(videoEl, sx, 0, sw, imgH, 0, 0, sw * bps, imgH * bps);
            }

            // Draw overlays
            if (trialData) {
                drawOverlays(bps);
            }

            ctx.restore();
        } else if (!showVideo && trialData) {
            // No video but draw overlays on black background
            const isStereo = cameraMode === 'stereo';
            const isLeft = currentSide === cameraNames[0];
            const sw = isStereo ? (isLeft ? midline : imgW - midline) : (imgW || w);
            const bps = w / (sw || w);

            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            drawOverlays(bps);
            ctx.restore();
        }
    }

    function drawOverlays(pixelScale) {
        const fn = currentFrame;
        if (!trialData || fn >= trialData.n_frames) return;

        const isStereo = cameraMode === 'stereo';
        const isLeft = currentSide === cameraNames[0];
        const side = currentSide;

        // For stereo right camera, offset by -midline
        const xOff = isStereo ? (isLeft ? 0 : -midline) : 0;

        // Get keypoints for current frame and side
        const mpPts = trialData.mediapipe?.[side]?.landmarks?.[fn];
        const viPts = trialData.vision?.[side]?.landmarks?.[fn];
        const dlcData = trialData.dlc?.[side];
        const bones = skeletonBones();

        // Draw skeleton lines
        if (showSkeleton && bones) {
            bones.forEach(([i, j]) => {
                if (!isBoneVisible(i, j)) return;

                // MediaPipe skeleton (cyan)
                if (showMediapipe && mpPts && mpPts[i] && mpPts[j]) {
                    drawLine(
                        (mpPts[i][0] + xOff) * pixelScale,
                        mpPts[i][1] * pixelScale,
                        (mpPts[j][0] + xOff) * pixelScale,
                        mpPts[j][1] * pixelScale,
                        '#4a9eff', 1.5, 0.6
                    );
                }

                // Vision skeleton (orange)
                if (showVision && viPts && viPts[i] && viPts[j]) {
                    drawLine(
                        (viPts[i][0] + xOff) * pixelScale,
                        viPts[i][1] * pixelScale,
                        (viPts[j][0] + xOff) * pixelScale,
                        viPts[j][1] * pixelScale,
                        '#ff9800', 1.5, 0.6
                    );
                }
            });
        }

        // MediaPipe joints (cyan circles)
        if (showMediapipe && mpPts) {
            for (let j = 0; j < mpPts.length; j++) {
                if (!isJointVisible(j) || !mpPts[j]) continue;
                const x = (mpPts[j][0] + xOff) * pixelScale;
                const y = mpPts[j][1] * pixelScale;
                drawJoint(x, y, '#4a9eff', 3);
            }
        }

        // Vision joints (orange circles)
        if (showVision && viPts) {
            for (let j = 0; j < viPts.length; j++) {
                if (!isJointVisible(j) || !viPts[j]) continue;
                const x = (viPts[j][0] + xOff) * pixelScale;
                const y = viPts[j][1] * pixelScale;
                drawJoint(x, y, '#ff9800', 3);
            }
        }

        // DLC thumb + index (green circles, larger)
        if (showDLC && dlcData) {
            const thumbPt = dlcData.thumb?.[fn];
            const indexPt = dlcData.index?.[fn];
            if (thumbPt && isJointVisible(4)) {
                drawJoint(
                    (thumbPt[0] + xOff) * pixelScale,
                    thumbPt[1] * pixelScale,
                    '#4caf50', 5
                );
            }
            if (indexPt && isJointVisible(8)) {
                drawJoint(
                    (indexPt[0] + xOff) * pixelScale,
                    indexPt[1] * pixelScale,
                    '#4caf50', 5
                );
            }
        }
    }

    function drawLine(x1, y1, x2, y2, color, width, alpha) {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = width / scale;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.restore();
    }

    function drawJoint(x, y, color, radius) {
        ctx.save();
        ctx.fillStyle = color;
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 0.5 / scale;
        ctx.beginPath();
        ctx.arc(x, y, radius / scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }

    // ── Distance trace ───────────────────────────────────────
    function renderDistanceTrace() {
        if (!distCtx || !trialData) return;
        const w = distCanvas.width, h = distCanvas.height;
        distCtx.clearRect(0, 0, w, h);

        // Background
        distCtx.fillStyle = '#1a1a2e';
        distCtx.fillRect(0, 0, w, h);

        const N = trialData.n_frames;
        if (N < 2) return;

        const metric = selectedMetric;
        const distances = trialData.distances?.[metric];
        if (!distances) return;

        // Gather visible series
        const series = [];
        const mpColor = '#4a9eff';
        const viColor = '#ff9800';
        const dlcColor = '#4caf50';

        // MediaPipe series
        if (showMediapipe) {
            const keyOS = `mediapipe_${cameraNames[0]}`;
            const keyOD = `mediapipe_${cameraNames[1]}`;
            const key = `mediapipe_${currentSide}`;
            // Try side-specific key first, then generic
            const data = distances[key] || distances.mediapipe;
            if (data) series.push({ data, color: mpColor, label: 'MediaPipe' });
        }
        if (showVision) {
            const key = `vision_${currentSide}`;
            const data = distances[key] || distances.vision;
            if (data) series.push({ data, color: viColor, label: 'Vision' });
        }
        if (showDLC) {
            const key = `dlc_${currentSide}`;
            const data = distances[key] || distances.dlc;
            if (data) series.push({ data, color: dlcColor, label: 'DLC' });
        }

        // If no side-specific keys found, try without side suffix
        if (series.length === 0) {
            // Fallback: try all keys that exist and match visibility
            for (const [key, data] of Object.entries(distances)) {
                if (!Array.isArray(data)) continue;
                if (showMediapipe && key.startsWith('mediapipe')) {
                    series.push({ data, color: mpColor, label: key });
                } else if (showVision && key.startsWith('vision')) {
                    series.push({ data, color: viColor, label: key });
                } else if (showDLC && key.startsWith('dlc')) {
                    series.push({ data, color: dlcColor, label: key });
                }
            }
        }

        // Compute Y range
        let yMin = Infinity, yMax = -Infinity;
        for (const s of series) {
            for (const v of s.data) {
                if (v != null && isFinite(v)) {
                    yMin = Math.min(yMin, v);
                    yMax = Math.max(yMax, v);
                }
            }
        }
        if (!isFinite(yMin)) { yMin = 0; yMax = 100; }
        const yPad = (yMax - yMin) * 0.1 || 10;
        yMin = Math.max(0, yMin - yPad);
        yMax += yPad;

        // Plot area with left padding for labels
        const padLeft = 40;
        const padRight = 8;
        const plotW = w - padLeft - padRight;
        const plotH = h;

        const toX = f => padLeft + (f / (N - 1)) * plotW;
        const toY = v => plotH - ((v - yMin) / (yMax - yMin)) * plotH;

        // Draw series
        for (const s of series) {
            distCtx.strokeStyle = s.color;
            distCtx.lineWidth = 1.5;
            distCtx.beginPath();
            let started = false;
            for (let i = 0; i < N; i++) {
                if (s.data[i] == null || !isFinite(s.data[i])) { started = false; continue; }
                if (!started) { distCtx.moveTo(toX(i), toY(s.data[i])); started = true; }
                else distCtx.lineTo(toX(i), toY(s.data[i]));
            }
            distCtx.stroke();
        }

        // Current frame indicator (vertical dashed line)
        distCtx.strokeStyle = '#fff';
        distCtx.lineWidth = 1;
        distCtx.setLineDash([3, 3]);
        distCtx.beginPath();
        distCtx.moveTo(toX(currentFrame), 0);
        distCtx.lineTo(toX(currentFrame), h);
        distCtx.stroke();
        distCtx.setLineDash([]);

        // Y-axis labels
        distCtx.fillStyle = '#888';
        distCtx.font = '10px monospace';
        distCtx.textAlign = 'left';
        distCtx.fillText(yMax.toFixed(0) + 'mm', 2, 12);
        distCtx.fillText(yMin.toFixed(0) + 'mm', 2, h - 4);

        // Mid-point label
        const yMid = (yMin + yMax) / 2;
        distCtx.fillText(yMid.toFixed(0), 2, h / 2 + 4);

        // Legend
        distCtx.textAlign = 'right';
        let lx = w - 8, ly = 14;
        for (const s of series) {
            distCtx.fillStyle = s.color;
            distCtx.fillText(s.label, lx, ly);
            ly += 14;
        }

        // Metric name at bottom center
        distCtx.fillStyle = '#666';
        distCtx.textAlign = 'center';
        distCtx.fillText(metric, w / 2, h - 4);
    }

    // ── Layer setter (for external use) ──────────────────────
    function setLayer(layer, visible) {
        switch (layer) {
            case 'video':
                showVideo = visible;
                $('showVideo').checked = visible;
                break;
            case 'mediapipe':
                showMediapipe = visible;
                $('showMP').checked = visible;
                break;
            case 'vision':
                showVision = visible;
                $('showVision').checked = visible;
                break;
            case 'dlc':
                showDLC = visible;
                $('showDLC').checked = visible;
                break;
            case 'skeleton':
                showSkeleton = visible;
                $('showSkeleton').checked = visible;
                break;
        }
        render();
        renderDistanceTrace();
    }

    // ── Run detection models ────────────────────────────────────
    async function _runDetection(endpoint, btnId, label) {
        const btn = $(btnId);
        const status = $('detectionStatus');
        if (btn) btn.disabled = true;
        if (status) status.textContent = `Running ${label}...`;

        try {
            const result = await api(`/api/analyze/${subjectId}/${endpoint}`, { method: 'POST' });
            const jobId = result.job_id;
            if (!jobId) throw new Error('No job_id returned');

            // Stream progress via SSE
            const evtSource = new EventSource(`/api/jobs/${jobId}/stream`);
            evtSource.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.status === 'running') {
                    const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                    if (status) status.textContent = `${label}... ${pct}%`;
                } else if (data.status === 'completed') {
                    evtSource.close();
                    if (status) status.textContent = `${label} complete. Reloading...`;
                    if (btn) btn.disabled = false;
                    // Reload trial data to pick up new detections
                    await loadTrial(currentTrialIdx);
                    if (status) status.textContent = `${label} complete.`;
                } else if (data.status === 'failed') {
                    evtSource.close();
                    if (status) status.textContent = `${label} failed: ${data.error_msg || 'unknown'}`;
                    if (btn) btn.disabled = false;
                } else if (data.status === 'cancelled') {
                    evtSource.close();
                    if (status) status.textContent = `${label} cancelled.`;
                    if (btn) btn.disabled = false;
                }
            };
            evtSource.onerror = () => {
                evtSource.close();
                if (status) status.textContent = 'Connection lost — check Processing page.';
                if (btn) btn.disabled = false;
            };
        } catch (err) {
            if (status) status.textContent = `Error: ${err.message || err}`;
            if (btn) btn.disabled = false;
        }
    }

    function runMediapipe() { _runDetection('run-mediapipe', 'runMPBtn', 'MediaPipe Hands'); }
    function runVision() { _runDetection('run-vision', 'runVisionBtn', 'Apple Vision Hands'); }
    function runPose() { _runDetection('run-pose', 'runPoseBtn', 'Pose Detection'); }

    // ── Public API ───────────────────────────────────────────
    return {
        init,
        togglePlay,
        toggleSide,
        resetZoom,
        setLayer,
        updateFingerVisibility,
        seekFrame,
        runMediapipe,
        runVision,
        runPose,
    };
})();

// ── Bootstrap ────────────────────────────────────────────────
(async function bootstrap() {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', async () => {
            // Small delay for nav.js to finish its DOM manipulation
            await new Promise(r => setTimeout(r, 50));
            await analyzeViewer.init();
        });
    } else {
        await new Promise(r => setTimeout(r, 50));
        await analyzeViewer.init();
    }
})();
