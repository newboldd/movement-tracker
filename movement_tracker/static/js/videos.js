/**
 * Videos Viewer
 *
 * Full-viewport stereo video browser. Assumes side-by-side stereo (left =
 * camera[0], right = camera[1]) unless trial metadata reports is_stereo=false.
 * Controls live in a bottom toolbar; trials are shown as buttons in the topbar.
 */
(function () {
    'use strict';

    const SPEED_PRESETS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120];

    // ── State ────────────────────────────────────────────────
    let allSubjects = [];
    let subjectId = null;
    let subjectName = '';
    let trials = [];        // [{trial_idx, trial_name, n_frames, fps, width, height, is_stereo}]
    let trialMeta = null;
    let currentTrialIdx = -1;

    let currentFrame = 0;
    let playing = false;
    let playTimer = null;
    let playbackRate = 1;

    let cameraNames = ['OS', 'OD'];
    let currentSide = 'OS';
    let currentCameraIdx = 0;   // index into multicamCameras or cameraNames
    let multicamCameras = [];   // [{name, idx}] from trial data (multicam only)
    let mpHints = [];  // per-trial mediapipe hints

    let cameraMode = 'stereo'; // 'single', 'stereo', or 'multicam'

    let videoEl = null;
    let vidW = 0, vidH = 0, midline = 0;
    let isStereo = true;

    let canvas, ctx;

    // Zoom/pan (same system as mano.js)
    let scale = 1, offsetX = 0, offsetY = 0;
    let dragging = false;
    let dragStartX = 0, dragStartY = 0;
    let panStartOX = 0, panStartOY = 0;

    // ── Helpers ──────────────────────────────────────────────
    const $ = id => document.getElementById(id);

    async function api(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
    }

    /** Strip subject-name prefix from trial name: "Con01_R1" → "R1" */
    function trialLabel(trialName) {
        if (subjectName && trialName.startsWith(subjectName + '_')) {
            return trialName.slice(subjectName.length + 1);
        }
        return trialName;
    }

    // ── Init ────────────────────────────────────────────────
    async function init() {
        canvas = $('canvas');
        ctx = canvas.getContext('2d');

        videoEl = document.createElement('video');
        videoEl.muted = true;
        videoEl.crossOrigin = 'anonymous';

        // Load camera names from settings (camera mode is per-subject)
        try {
            const cfg = await api('/api/settings');
            if (cfg.default_camera_mode) cameraMode = cfg.default_camera_mode;
            if (Array.isArray(cfg.camera_names) && cfg.camera_names.length >= 1) {
                cameraNames = cfg.camera_names;
                currentSide = cameraNames[0];
            }
        } catch { /* defaults */ }

        updateCameraButton();

        // Load subjects
        allSubjects = await api('/api/subjects');
        const sel = $('subjectSelect');
        sel.innerHTML = '<option value="">Select subject…</option>';
        allSubjects.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            sel.appendChild(opt);
        });

        // Restore from URL, nav state, or default to first subject with videos
        const params = new URLSearchParams(window.location.search);
        const subParam = params.get('subject');
        const nav = typeof getNavState === 'function' ? getNavState() : {};
        const savedSubject = subParam || (nav.subjectId ? String(nav.subjectId) : null)
            || sessionStorage.getItem('dlc_lastSubjectId');
        if (savedSubject && allSubjects.some(s => String(s.id) === savedSubject)) {
            sel.value = savedSubject;
        } else {
            const first = allSubjects.find(s => s.video_count > 0);
            if (first) sel.value = first.id;
        }

        setupControls();
        setupCanvasEvents();

        if (sel.value) {
            await loadSubject(parseInt(sel.value));
            // Restore trial/frame if nav state subject matches
            if (nav.subjectId === parseInt(sel.value)) {
                if (nav.trialIdx != null && nav.trialIdx >= 0 && nav.trialIdx < trials.length) {
                    await loadTrial(nav.trialIdx);
                    if (nav.frame != null && nav.frame >= 0 && trialMeta && nav.frame < trialMeta.n_frames) {
                        goToFrame(nav.frame);
                    }
                }
            }
        }
    }

    // ── Subject / Trial loading ──────────────────────────────
    async function loadSubject(sid) {
        subjectId = sid;
        sessionStorage.setItem('dlc_lastSubjectId', String(sid));
        if (typeof setNavState === 'function') setNavState({ subjectId: sid });
        const subj = allSubjects.find(s => s.id === sid);
        subjectName = subj ? subj.name : '';
        // Per-subject camera mode
        if (subj && subj.camera_mode) cameraMode = subj.camera_mode;

        const u = new URL(window.location);
        u.searchParams.set('subject', sid);
        history.replaceState(null, '', u);

        trials = [];
        mpHints = [];
        try { trials = await api(`/api/mano/${sid}/video_list`); } catch { /* no videos */ }
        try { mpHints = await api(`/api/mano/${sid}/mediapipe_hints`); } catch { /* no hints */ }

        buildTrialButtons();

        if (trials.length) loadTrial(0);
    }

    /** Load a video file selected via the Browse button (outside the dataset). */
    function loadBrowsedFile(file) {
        // Clear subject context
        subjectId = null;
        subjectName = '';
        trials = [];
        mpHints = [];
        $('trialBtns').innerHTML = '';

        // Store filename so onboarding can prefill it
        sessionStorage.setItem('onboard_prefill_video', file.name);

        // Show "Add Subject" link next to browse button
        let addLink = $('addSubjectLink');
        if (!addLink) {
            addLink = document.createElement('a');
            addLink.id = 'addSubjectLink';
            addLink.href = '/onboarding';
            addLink.className = 'btn btn-sm btn-primary';
            addLink.textContent = '+ Add Subject';
            addLink.style.textDecoration = 'none';
            $('browseBtn').insertAdjacentElement('afterend', addLink);
        }

        trialMeta = { n_frames: 0, fps: 30, trial_idx: -1, trial_name: file.name, is_stereo: true };
        isStereo = true;
        currentFrame = 0;
        scale = 1; offsetX = 0; offsetY = 0;

        $('frameDisplay').textContent = 0;
        $('totalFramesDisplay').textContent = '?';
        $('timelineSlider').value = 0;

        const url = URL.createObjectURL(file);
        videoEl.src = url;
        videoEl.addEventListener('loadedmetadata', () => {
            vidW = videoEl.videoWidth;
            vidH = videoEl.videoHeight;
            // Use camera_mode setting, fall back to heuristic for browsed files
            isStereo = cameraMode === 'stereo';
            midline = isStereo ? Math.round(vidW / 2) : vidW;
            trialMeta.n_frames = Math.round(videoEl.duration * trialMeta.fps);
            trialMeta.is_stereo = isStereo;
            $('totalFramesDisplay').textContent = trialMeta.n_frames;
            $('timelineSlider').max = trialMeta.n_frames - 1;
            sizeCanvas();
            videoEl.currentTime = 0.01;
            videoEl.addEventListener('seeked', render, { once: true });
        }, { once: true });
    }

    function buildTrialButtons() {
        const container = $('trialBtns');
        container.innerHTML = '';
        trials.forEach((t, i) => {
            const btn = document.createElement('button');
            btn.className = 'trial-btn';
            btn.textContent = trialLabel(t.trial_name);
            btn.title = t.trial_name;
            btn.addEventListener('click', () => loadTrial(i));
            container.appendChild(btn);
        });
    }

    function highlightTrialButton(idx) {
        const btns = $('trialBtns').querySelectorAll('.trial-btn');
        btns.forEach((b, i) => b.classList.toggle('active', i === idx));
    }

    async function loadTrial(idx) {
        if (idx < 0 || idx >= trials.length) return;
        currentTrialIdx = idx;
        if (typeof setNavState === 'function') setNavState({ trialIdx: idx, frame: currentFrame });
        trialMeta = trials[idx];
        // Use camera_mode setting; trialMeta.is_stereo is authoritative from server
        isStereo = cameraMode === 'stereo' && trialMeta.is_stereo !== false;

        // Multicam: populate camera list from trial data
        multicamCameras = trialMeta.cameras || [];
        if (cameraMode === 'multicam' && multicamCameras.length > 0) {
            currentCameraIdx = 0;
            currentSide = multicamCameras[0].name;
        } else if (isStereo) {
            // Apply best camera from mediapipe hints
            const hint = mpHints.find(h => h.trial_idx === trialMeta.trial_idx);
            if (hint) {
                currentCameraIdx = hint.best_camera === 1 ? 1 : 0;
                currentSide = cameraNames[currentCameraIdx];
            } else {
                currentCameraIdx = 0;
                currentSide = cameraNames[0];
            }
        } else {
            currentCameraIdx = 0;
            currentSide = cameraNames[0];
        }
        updateCameraButton();

        highlightTrialButton(idx);

        currentFrame = 0;
        scale = 1; offsetX = 0; offsetY = 0;

        $('totalFramesDisplay').textContent = trialMeta.n_frames;
        $('frameDisplay').textContent = 0;
        $('timelineSlider').max = trialMeta.n_frames - 1;
        $('timelineSlider').value = 0;

        loadCurrentCameraVideo();
    }

    /** Build the video URL for the current trial + camera and load it. */
    function loadCurrentCameraVideo() {
        let src = `/api/mano/${subjectId}/trial/${trialMeta.trial_idx}/video`;
        if (cameraMode === 'multicam' && multicamCameras.length > 0) {
            src += `?camera=${multicamCameras[currentCameraIdx].idx}`;
        }
        videoEl.src = src;
        videoEl.addEventListener('loadedmetadata', () => {
            vidW = videoEl.videoWidth;
            vidH = videoEl.videoHeight;
            midline = isStereo ? Math.round(vidW / 2) : vidW;
            sizeCanvas();
            const hint = mpHints.find(h => h.trial_idx === trialMeta.trial_idx);
            if (hint && isStereo) applyMpCrop(hint);
            // Seek to midpoint of frame 0 — t=0 cannot be decoded by many codecs
            const seekTarget = trialMeta.fps ? 0.5 / trialMeta.fps : 0.01;
            videoEl.currentTime = seekTarget;
            videoEl.addEventListener('seeked', render, { once: true });
        }, { once: true });
    }

    /** Compute base pixel scale that fits the full frame in the canvas. */
    function getBaseMetrics() {
        const w = canvas.width, h = canvas.height;
        const sw = isStereo ? midline : vidW;
        const bps = Math.min(w / sw, h / vidH);
        const baseOX = (w - sw * bps) / 2;
        const baseOY = (h - vidH * bps) / 2;
        return { bps, baseOX, baseOY, sw };
    }

    /** Apply mediapipe-derived crop for the current camera side. */
    function applyMpCrop(hint) {
        const bbox = currentSide === cameraNames[0] ? hint.bbox_OS : hint.bbox_OD;
        if (!bbox) return;
        const [minX, minY, maxX, maxY] = bbox;
        const cropW = maxX - minX;
        const cropH = maxY - minY;
        const cropCX = minX + cropW / 2;
        const cropCY = minY + cropH / 2;
        const cw = canvas.width, ch = canvas.height;
        const { bps, baseOX, baseOY } = getBaseMetrics();
        scale = Math.min(cw / (cropW * bps), ch / (cropH * bps)) * 0.85;
        // Offset is relative to the base-centered origin (render adds baseOX/baseOY)
        offsetX = cw / 2 - baseOX - scale * cropCX * bps;
        offsetY = ch / 2 - baseOY - scale * cropCY * bps;
    }

    // ── Controls setup ───────────────────────────────────────
    function setupControls() {
        $('prevFrameBtn').addEventListener('click', () => goToFrame(currentFrame - 1));
        $('nextFrameBtn').addEventListener('click', () => goToFrame(currentFrame + 1));
        $('playBtn').addEventListener('click', togglePlay);
        $('sideToggle').addEventListener('click', switchCamera);
        $('resetZoomBtn').addEventListener('click', resetZoom);
        $('subjectSelect').addEventListener('change', e => {
            const v = parseInt(e.target.value);
            if (v) loadSubject(v);
            e.target.blur();  // return focus so keyboard shortcuts work
        });

        // Browse button — load a video file from outside the dataset
        $('browseBtn').addEventListener('click', () => $('browseInput').click());
        $('browseInput').addEventListener('change', e => {
            const file = e.target.files[0];
            if (file) loadBrowsedFile(file);
            e.target.value = '';  // allow re-selecting same file
        });

        // Speed slider — start at index 3 = 1x
        const speedSlider = $('speedSlider');
        speedSlider.value = 3;
        speedSlider.addEventListener('input', () => {
            playbackRate = SPEED_PRESETS[parseInt(speedSlider.value)];
            $('speedDisplay').textContent = playbackRate + 'x';
            if (playing) {
                videoEl.playbackRate = Math.min(playbackRate, 16);
            }
        });
        speedSlider.addEventListener('change', () => {
            speedSlider.blur();  // return focus so keyboard shortcuts work
        });

        // Timeline seek
        const timeline = $('timelineSlider');
        timeline.addEventListener('input', () => {
            if (!trialMeta) return;
            goToFrame(parseInt(timeline.value));
        });

        // Keyboard — blur focused controls so shortcuts always work
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
            switch (e.key) {
                case 'a': case 'ArrowLeft':  goToFrame(currentFrame - 1); e.preventDefault(); break;
                case 's': case 'ArrowRight': goToFrame(currentFrame + 1); e.preventDefault(); break;
                case ' ': togglePlay(); e.preventDefault(); break;
                case 'e': case 'E': switchCamera(); break;
                case 'z': case 'Z': resetZoom(); break;
            }
        });
    }

    // ── Camera switching ─────────────────────────────────────

    /** Update the camera toggle button label and visibility. */
    function updateCameraButton() {
        const btn = $('sideToggle');
        if (cameraMode === 'single') {
            btn.style.display = 'none';
        } else {
            btn.style.display = '';
            btn.textContent = currentSide;
        }
    }

    /** Cycle to the next camera view. */
    function switchCamera() {
        if (cameraMode === 'single') return;

        if (cameraMode === 'multicam' && multicamCameras.length > 1) {
            // Multicam: cycle through camera files (blink-free preload)
            currentCameraIdx = (currentCameraIdx + 1) % multicamCameras.length;
            currentSide = multicamCameras[currentCameraIdx].name;
            updateCameraButton();

            const frameToRestore = currentFrame;
            let src = `/api/mano/${subjectId}/trial/${trialMeta.trial_idx}/video`;
            src += `?camera=${multicamCameras[currentCameraIdx].idx}`;

            // Preload in a temp video element so the canvas keeps showing the old frame
            const tmp = document.createElement('video');
            tmp.preload = 'auto';
            tmp.muted = true;
            tmp.src = src;
            tmp.addEventListener('loadedmetadata', () => {
                vidW = tmp.videoWidth;
                vidH = tmp.videoHeight;
                midline = vidW;
                sizeCanvas();
                const seekTarget = trialMeta.fps
                    ? (frameToRestore + 0.5) / trialMeta.fps
                    : 0.01;
                tmp.currentTime = seekTarget;
                tmp.addEventListener('seeked', () => {
                    // Swap: replace the old video element
                    videoEl.pause();
                    videoEl.removeAttribute('src');
                    videoEl.load(); // release old resources
                    videoEl = tmp;
                    render();
                }, { once: true });
            }, { once: true });
        } else if (isStereo && cameraNames.length > 1) {
            // Stereo: toggle left/right crop
            currentCameraIdx = (currentCameraIdx + 1) % cameraNames.length;
            currentSide = cameraNames[currentCameraIdx];
            updateCameraButton();
            // Re-apply mediapipe crop for the new camera
            const hint = mpHints.find(h => h.trial_idx === (trialMeta ? trialMeta.trial_idx : -1));
            if (hint) { applyMpCrop(hint); }
            render();
        }
    }

    // ── Playback ─────────────────────────────────────────────
    function goToFrame(n) {
        if (!trialMeta) return;
        currentFrame = Math.max(0, Math.min(n, trialMeta.n_frames - 1));
        if (typeof setNavState === 'function') setNavState({ frame: currentFrame });
        $('frameDisplay').textContent = currentFrame;
        $('timelineSlider').value = currentFrame;
        // Seek then render — never render before seeked or the previous decoded
        // frame (or a blank) flashes before the new one arrives.
        if (videoEl.readyState >= 2 && trialMeta.fps) {
            const t = (currentFrame + 0.5) / trialMeta.fps;
            videoEl.currentTime = t;
            videoEl.addEventListener('seeked', render, { once: true });
        } else {
            // Video not ready — wait for it to become ready then seek+render.
            // Calling render() here would flash the stale decoded frame.
            videoEl.addEventListener('loadeddata', () => {
                const t = (currentFrame + 0.5) / (trialMeta.fps || 30);
                videoEl.currentTime = t;
                videoEl.addEventListener('seeked', render, { once: true });
            }, { once: true });
        }
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
        if (videoEl.readyState >= 2 && trialMeta) {
            const f = Math.floor(videoEl.currentTime * trialMeta.fps);
            if (f !== currentFrame && f >= 0 && f < trialMeta.n_frames) {
                currentFrame = f;
                $('frameDisplay').textContent = currentFrame;
                $('timelineSlider').value = currentFrame;
                render();
            }
            if (f >= trialMeta.n_frames - 1) { togglePlay(); return; }
        }
        playTimer = requestAnimationFrame(playLoop);
    }

    // ── Zoom / pan ───────────────────────────────────────────
    function resetZoom() {
        scale = 1; offsetX = 0; offsetY = 0;
        const hint = mpHints.find(h => h.trial_idx === (trialMeta ? trialMeta.trial_idx : -1));
        if (hint) applyMpCrop(hint);
        render();
    }

    function setupCanvasEvents() {
        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const { baseOX, baseOY } = getBaseMetrics();
            // Zoom pivot relative to the base-centered origin
            const lx = mx - baseOX;
            const ly = my - baseOY;
            const factor = e.deltaY < 0 ? 1.05 : 1 / 1.05;
            const ns = Math.max(0.1, Math.min(scale * factor, 50));
            offsetX = lx - (lx - offsetX) * (ns / scale);
            offsetY = ly - (ly - offsetY) * (ns / scale);
            scale = ns;
            render();
        }, { passive: false });

        canvas.addEventListener('mousedown', e => {
            if (e.button === 0 || e.button === 1) {
                dragging = true;
                dragStartX = e.clientX; dragStartY = e.clientY;
                panStartOX = offsetX;   panStartOY = offsetY;
                e.preventDefault();
            }
        });
        document.addEventListener('mousemove', e => {
            if (!dragging) return;
            offsetX = panStartOX + (e.clientX - dragStartX);
            offsetY = panStartOY + (e.clientY - dragStartY);
            render();
        });
        document.addEventListener('mouseup', () => { dragging = false; });

        // Resize
        const ro = new ResizeObserver(() => { sizeCanvas(); render(); });
        ro.observe(canvas.parentElement);
    }

    // ── Canvas sizing ────────────────────────────────────────
    function sizeCanvas() {
        const vp = canvas.parentElement;
        canvas.width  = vp.clientWidth;
        canvas.height = vp.clientHeight;
    }

    // ── Rendering ────────────────────────────────────────────
    function render() {
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, w, h);

        if (videoEl.readyState < 2 || vidW === 0) return;

        const isFirst = currentSide === cameraNames[0];
        const sx = isStereo ? (isFirst ? 0 : midline) : 0;
        const { bps, baseOX, baseOY, sw } = getBaseMetrics();

        ctx.save();
        ctx.translate(baseOX + offsetX, baseOY + offsetY);
        ctx.scale(scale, scale);
        ctx.drawImage(videoEl, sx, 0, sw, vidH, 0, 0, sw * bps, vidH * bps);
        ctx.restore();
    }

    // ── Export ───────────────────────────────────────────────
    function openExportModal() {
        if (!trialMeta) { alert('No video loaded'); return; }
        if (window.VideoExport) window.VideoExport.open(getExportContext());
        else alert('Export module not loaded');
    }

    function getExportContext() {
        return {
            videoEl,
            fps: trialMeta ? trialMeta.fps : 30,
            playbackRate,
            nFrames: trialMeta ? trialMeta.n_frames : 0,
            get currentFrame() { return currentFrame; },
            canvasLayers: [canvas],
            distanceCanvas: null,
            getCompositeSize() { return { width: canvas.width, height: canvas.height }; },
            renderThreeJS() { return null; },
            seekAndRender(frameNum) {
                return new Promise(resolve => {
                    currentFrame = Math.max(0, Math.min(frameNum, (trialMeta ? trialMeta.n_frames - 1 : 0)));
                    $('frameDisplay').textContent = currentFrame;
                    $('timelineSlider').value = currentFrame;
                    if (videoEl.readyState >= 2 && trialMeta && trialMeta.fps) {
                        const t = (currentFrame + 0.5) / trialMeta.fps;
                        videoEl.currentTime = t;
                        videoEl.addEventListener('seeked', () => { render(); resolve(); }, { once: true });
                    } else {
                        resolve();
                    }
                });
            },
        };
    }

    window.videosViewer = { getExportContext, openExportModal };
    window.openExportModal = openExportModal;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
