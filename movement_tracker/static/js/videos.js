/**
 * Videos Viewer
 *
 * Full-viewport stereo video browser. Assumes side-by-side stereo (left =
 * camera[0], right = camera[1]) unless trial metadata reports is_stereo=false.
 * Controls live in a bottom toolbar; trials are shown as buttons in the topbar.
 */
(function () {
    'use strict';

    const SPEED_PRESETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120];
    const SPEED_DEFAULT_IDX = SPEED_PRESETS.indexOf(1);    // == 6
    let exportMode = false;
    let exportRunning = false;

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

    // Zoom/pan (same system as labels.js)
    let scale = 1, offsetX = 0, offsetY = 0;
    let dragging = false;
    let dragStartX = 0, dragStartY = 0;
    let panStartOX = 0, panStartOY = 0;

    // Export-mode crop box (in canvas-buffer pixel coords).
    let cropX = 0, cropY = 0, cropW = 0, cropH = 0;
    let cropDragMode = null;        // 'move' | 'n','s','e','w','nw','ne','sw','se'
    let cropDragStart = null;       // {mx,my,x,y,w,h}
    const CROP_COLOR = '#FF9800';
    const CROP_HANDLE = 10;         // px (canvas-buffer) on each side of a handle hit-box
    const CROP_HANDLE_DRAW = 8;     // drawn handle size

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
        _wireTrimHandles();

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
        try { trials = await api(`/api/skeleton/${sid}/video_list`); } catch { /* no videos */ }
        try { mpHints = await api(`/api/skeleton/${sid}/mediapipe_hints`); } catch { /* no hints */ }

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
        let src = `/api/skeleton/${subjectId}/trial/${trialMeta.trial_idx}/video`;
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

    /** Compute base pixel scale that fits the full frame in the canvas.
     *  Returns zeros (not NaN/Infinity) when the video hasn't loaded yet so
     *  callers can detect "not ready" without poisoning the math. */
    function getBaseMetrics() {
        const w = canvas.width, h = canvas.height;
        const sw = isStereo ? midline : vidW;
        if (!(sw > 0) || !(vidH > 0) || !(w > 0) || !(h > 0)) {
            return { bps: 0, baseOX: 0, baseOY: 0, sw: 0 };
        }
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
        // Bail out early if the bbox is degenerate or the video hasn't sized
        // the canvas yet — otherwise bps below becomes 0 / NaN / Infinity and
        // the resulting offset is NaN, which silently kills every later
        // ctx.translate() (so zoom and pan appear to do nothing).
        if (!(cropW > 0 && cropH > 0)) return;
        const cw = canvas.width, ch = canvas.height;
        const { bps, baseOX, baseOY } = getBaseMetrics();
        if (!isFinite(bps) || bps <= 0) return;
        const cropCX = minX + cropW / 2;
        const cropCY = minY + cropH / 2;
        const ns = Math.min(cw / (cropW * bps), ch / (cropH * bps)) * 0.85;
        if (!isFinite(ns) || ns <= 0) return;
        scale = ns;
        offsetX = cw / 2 - baseOX - scale * cropCX * bps;
        offsetY = ch / 2 - baseOY - scale * cropCY * bps;
        // Final NaN/Inf guard.
        if (!isFinite(offsetX) || !isFinite(offsetY)) {
            scale = 1; offsetX = 0; offsetY = 0;
        }
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

        // Speed slider — range covers SPEED_PRESETS; default to 1x.
        const speedSlider = $('speedSlider');
        speedSlider.min = 0;
        speedSlider.max = SPEED_PRESETS.length - 1;
        speedSlider.value = SPEED_DEFAULT_IDX;
        playbackRate = SPEED_PRESETS[SPEED_DEFAULT_IDX];
        $('speedDisplay').textContent = playbackRate + 'x';
        speedSlider.addEventListener('input', () => {
            playbackRate = SPEED_PRESETS[parseInt(speedSlider.value)];
            $('speedDisplay').textContent = playbackRate + 'x';
            if (playing) {
                // <video> requires playbackRate in [0.0625, 16]; below that we
                // step manually in playLoop.
                videoEl.playbackRate = Math.max(0.0625, Math.min(playbackRate, 16));
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

    /** Update the camera toggle button label and visibility.  Uses the
     *  "OS | OD" dual-toggle style (active half bold, inactive muted) from
     *  the labels/analyze pages. */
    function updateCameraButton() {
        const btn = $('sideToggle');
        if (cameraMode === 'single') {
            btn.style.display = 'none';
            return;
        }
        btn.style.display = '';
        // Multicam: list every camera; stereo: just the two named eyes.
        const opts = (cameraMode === 'multicam' && multicamCameras.length > 0)
            ? multicamCameras.map(c => c.name)
            : [cameraNames[0], cameraNames[1] || cameraNames[0]];
        const act = 'color:var(--text);font-weight:bold;';
        const ina = 'color:var(--text-muted);';
        btn.innerHTML = opts.map((name, i) => {
            const span = `<span style="${name === currentSide ? act : ina}">${name}</span>`;
            const sep = (i < opts.length - 1)
                ? `<span style="opacity:0.35;margin:0 3px;">|</span>` : '';
            return span + sep;
        }).join('');
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
            let src = `/api/skeleton/${subjectId}/trial/${trialMeta.trial_idx}/video`;
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
            if (playTimer) {
                if (typeof playTimer === 'number') clearTimeout(playTimer);
                else cancelAnimationFrame(playTimer);
                playTimer = null;
            }
            $('playBtn').innerHTML = '&#9654;';
        } else {
            playing = true;
            $('playBtn').innerHTML = '&#9646;&#9646;';
            // <video> only supports playbackRate >= ~0.0625x.  Below that
            // we pause the native player and step frame-by-frame on a
            // timer so the user gets true ultra-slow playback.
            if (playbackRate >= 0.0625) {
                videoEl.playbackRate = Math.min(playbackRate, 16);
                videoEl.play().catch(() => {});
                playLoop();
            } else {
                videoEl.pause();
                playStepManual();
            }
        }
    }

    function playStepManual() {
        if (!playing || !trialMeta) return;
        if (currentFrame >= trialMeta.n_frames - 1) { togglePlay(); return; }
        goToFrame(currentFrame + 1);
        const fps = trialMeta.fps || 30;
        const ms = 1000 / Math.max(fps * playbackRate, 0.1);
        playTimer = setTimeout(playStepManual, ms);
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

    /** Ensure the canvas pixel buffer matches the displayed CSS size.
     *  Without this guard the first wheel/click can fire while the
     *  buffer is still 300×150 (the <canvas> default), which makes
     *  baseOX/Y and the zoom pivot use wildly wrong units → the image
     *  appears to "jump" in size on the first scroll event. */
    function ensureCanvasSized() {
        const vp = canvas.parentElement;
        if (!vp) return;
        if (canvas.width !== vp.clientWidth || canvas.height !== vp.clientHeight) {
            sizeCanvas();
        }
    }

    // No-op (the on-page debug overlay was removed once pan/zoom worked).
    function dbg() {}

    // ── Export-mode crop box ─────────────────────────────────

    /** Visible video rectangle in canvas-buffer coords, clipped to canvas. */
    function _visibleVideoRect() {
        const { bps, baseOX, baseOY, sw } = getBaseMetrics();
        if (!(bps > 0) || !canvas) {
            return { left: 0, top: 0,
                     right: canvas ? canvas.width  : 0,
                     bot:   canvas ? canvas.height : 0,
                     bps, sw };
        }
        const imgX = baseOX + offsetX;
        const imgY = baseOY + offsetY;
        const imgW = sw * bps * scale;
        const imgH = vidH * bps * scale;
        return {
            left:  Math.max(0, imgX),
            top:   Math.max(0, imgY),
            right: Math.min(canvas.width,  imgX + imgW),
            bot:   Math.min(canvas.height, imgY + imgH),
            bps, sw,
        };
    }

    /** Reset crop to the visible video rect, then trim one edge so its
     *  aspect matches the source video's (sw / vidH). */
    function _resetCropToView() {
        const r = _visibleVideoRect();
        let w = r.right - r.left;
        let h = r.bot - r.top;
        if (!(w > 0) || !(h > 0) || !(vidH > 0) || !(r.sw > 0)) {
            cropX = r.left; cropY = r.top;
            cropW = Math.max(20, w);
            cropH = Math.max(20, h);
            return;
        }
        const targetAspect = r.sw / vidH;  // video's own aspect ratio
        // Trim one edge: shrink whichever dimension is too generous.
        if (w / h > targetAspect) {
            w = h * targetAspect;          // too wide → trim width
        } else {
            h = w / targetAspect;          // too tall → trim height
        }
        cropX = r.left;
        cropY = r.top;
        cropW = Math.max(20, w);
        cropH = Math.max(20, h);
    }

    function _cropHandles() {
        const x = cropX, y = cropY, w = cropW, h = cropH;
        return [
            { name: 'nw', x: x,         y: y         },
            { name: 'n',  x: x + w / 2, y: y         },
            { name: 'ne', x: x + w,     y: y         },
            { name: 'w',  x: x,         y: y + h / 2 },
            { name: 'e',  x: x + w,     y: y + h / 2 },
            { name: 'sw', x: x,         y: y + h     },
            { name: 's',  x: x + w / 2, y: y + h     },
            { name: 'se', x: x + w,     y: y + h     },
        ];
    }

    /** Mouse → 'move' | one of the 8 handle names | null. */
    function _cropHitTest(mx, my) {
        if (!exportMode) return null;
        for (const h of _cropHandles()) {
            if (Math.abs(mx - h.x) <= CROP_HANDLE &&
                Math.abs(my - h.y) <= CROP_HANDLE) {
                return h.name;
            }
        }
        if (mx >= cropX && mx <= cropX + cropW &&
            my >= cropY && my <= cropY + cropH) {
            return 'move';
        }
        return null;
    }

    function _cursorForCrop(hit) {
        switch (hit) {
            case 'move': return 'move';
            case 'n':    case 's':  return 'ns-resize';
            case 'e':    case 'w':  return 'ew-resize';
            case 'nw':   case 'se': return 'nwse-resize';
            case 'ne':   case 'sw': return 'nesw-resize';
            default:                return '';
        }
    }

    /** Draw the orange crop overlay (rectangle + 8 handles + dim outside). */
    function _drawCropOverlay() {
        if (!exportMode || exportRunning) return;
        if (!(cropW > 0) || !(cropH > 0)) return;
        // Dim everything outside the crop box.
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        ctx.fillRect(0, 0, canvas.width, cropY);
        ctx.fillRect(0, cropY + cropH, canvas.width, canvas.height - (cropY + cropH));
        ctx.fillRect(0, cropY, cropX, cropH);
        ctx.fillRect(cropX + cropW, cropY, canvas.width - (cropX + cropW), cropH);
        // Frame + handles.
        ctx.strokeStyle = CROP_COLOR;
        ctx.lineWidth = 2;
        ctx.strokeRect(cropX + 1, cropY + 1, cropW - 2, cropH - 2);
        const s = CROP_HANDLE_DRAW;
        for (const h of _cropHandles()) {
            // If a handle sits on the canvas edge it would be half-clipped
            // and hard to see — nudge it inward so the full square shows.
            const hx = Math.max(s / 2, Math.min(canvas.width  - s / 2, h.x));
            const hy = Math.max(s / 2, Math.min(canvas.height - s / 2, h.y));
            ctx.fillStyle = CROP_COLOR;
            ctx.fillRect(hx - s / 2, hy - s / 2, s, s);
            // White outline gives contrast against any video background.
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.strokeRect(hx - s / 2 + 0.5, hy - s / 2 + 0.5, s - 1, s - 1);
        }
        ctx.restore();
    }
    function setupCanvasEvents() {
        dbg('init');

        // Document-level mousedown catches clicks even if something is layered
        // over the canvas — we record the target so we can see what's eating them.
        document.addEventListener('mousedown', e => {
            dbg('docDown:' + (e.target === canvas ? 'CANVAS' : (e.target.tagName + (e.target.id ? '#'+e.target.id : ''))));
        }, true);

        // Wheel = zoom around the cursor.
        canvas.addEventListener('wheel', e => {
            dbg('wheel:' + (e.deltaY < 0 ? 'in' : 'out'));
            // Self-heal: if a prior bad state poisoned the offsets to NaN, reset.
            if (!isFinite(offsetX) || !isFinite(offsetY) || !isFinite(scale) || scale <= 0) {
                scale = 1; offsetX = 0; offsetY = 0;
            }
            e.preventDefault();
            ensureCanvasSized();
            const rect = canvas.getBoundingClientRect();
            // Convert CSS-px mouse coords into canvas-pixel-buffer coords
            // (defends against any CSS-vs-buffer-size mismatch).
            const csx = canvas.width / rect.width;
            const csy = canvas.height / rect.height;
            const mx = (e.clientX - rect.left) * csx;
            const my = (e.clientY - rect.top) * csy;
            const { baseOX, baseOY } = getBaseMetrics();
            const lx = mx - baseOX;
            const ly = my - baseOY;
            const factor = e.deltaY < 0 ? 1.025 : 1 / 1.025;
            const ns = Math.max(0.1, Math.min(scale * factor, 50));
            offsetX = lx - (lx - offsetX) * (ns / scale);
            offsetY = ly - (ly - offsetY) * (ns / scale);
            scale = ns;
            render();
        }, { passive: false });

        // Pan = left-mouse drag.  Down on canvas, move/up on window so the
        // drag continues even if the cursor leaves the canvas.
        // Mouse coords helper: CSS px → canvas-buffer px.
        const _bufXY = (e) => {
            const rect = canvas.getBoundingClientRect();
            const csx = canvas.width / rect.width;
            const csy = canvas.height / rect.height;
            return { mx: (e.clientX - rect.left) * csx,
                     my: (e.clientY - rect.top)  * csy };
        };

        canvas.addEventListener('mousedown', e => {
            dbg('canvasDown b=' + e.button);
            if (e.button !== 0 && e.button !== 1) return;
            ensureCanvasSized();
            if (!isFinite(offsetX) || !isFinite(offsetY) || !isFinite(scale) || scale <= 0) {
                scale = 1; offsetX = 0; offsetY = 0;
            }
            // Export-mode crop box takes priority over pan.
            if (exportMode) {
                const { mx, my } = _bufXY(e);
                const hit = _cropHitTest(mx, my);
                if (hit) {
                    cropDragMode = hit;
                    cropDragStart = { mx, my, x: cropX, y: cropY, w: cropW, h: cropH };
                    e.preventDefault();
                    return;
                }
            }
            // Default: pan.
            dragging = true;
            dragStartX = e.clientX; dragStartY = e.clientY;
            panStartOX = offsetX;   panStartOY = offsetY;
            e.preventDefault();
        });

        // Cursor hint when hovering the crop box / handles.
        canvas.addEventListener('mousemove', e => {
            if (!exportMode || cropDragMode || dragging) return;
            const { mx, my } = _bufXY(e);
            canvas.style.cursor = _cursorForCrop(_cropHitTest(mx, my));
        });
        canvas.addEventListener('mouseleave', () => {
            if (!cropDragMode && !dragging) canvas.style.cursor = '';
        });

        // Double-click inside the crop box → reset to default (full visible
        // video, aspect-trimmed to match the source).
        canvas.addEventListener('dblclick', e => {
            if (!exportMode) return;
            const { mx, my } = _bufXY(e);
            if (mx >= cropX && mx <= cropX + cropW &&
                my >= cropY && my <= cropY + cropH) {
                _resetCropToView();
                render();
                e.preventDefault();
            }
        });

        window.addEventListener('mousemove', e => {
            if (cropDragMode) {
                const { mx, my } = _bufXY(e);
                const s = cropDragStart;
                const dx = mx - s.mx, dy = my - s.my;
                const minSize = 20;
                // Constrain to the *currently visible video rectangle*, not
                // the whole canvas — the crop must stay over actual content.
                const r = _visibleVideoRect();
                if (cropDragMode === 'move') {
                    const maxX = (r.right - r.left) - s.w;
                    const maxY = (r.bot   - r.top)  - s.h;
                    cropX = r.left + Math.max(0, Math.min(maxX, (s.x - r.left) + dx));
                    cropY = r.top  + Math.max(0, Math.min(maxY, (s.y - r.top)  + dy));
                } else {
                    let nx = s.x, ny = s.y, nw = s.w, nh = s.h;
                    if (cropDragMode.includes('w')) {
                        nx = Math.max(r.left, Math.min(s.x + s.w - minSize, s.x + dx));
                        nw = s.x + s.w - nx;
                    }
                    if (cropDragMode.includes('e')) {
                        nw = Math.max(minSize, Math.min(r.right - s.x, s.w + dx));
                    }
                    if (cropDragMode.includes('n')) {
                        ny = Math.max(r.top, Math.min(s.y + s.h - minSize, s.y + dy));
                        nh = s.y + s.h - ny;
                    }
                    if (cropDragMode.includes('s')) {
                        nh = Math.max(minSize, Math.min(r.bot - s.y, s.h + dy));
                    }
                    cropX = nx; cropY = ny; cropW = nw; cropH = nh;
                }
                render();
                return;
            }
            if (!dragging) return;
            offsetX = panStartOX + (e.clientX - dragStartX);
            offsetY = panStartOY + (e.clientY - dragStartY);
            render();
            dbg('pan');
        });
        window.addEventListener('mouseup', () => {
            if (cropDragMode) { cropDragMode = null; cropDragStart = null; return; }
            if (dragging) { dragging = false; }
        });

        // Resize
        const ro = new ResizeObserver(() => { sizeCanvas(); render(); });
        ro.observe(canvas.parentElement);
        // Defensive: ensure pixel buffer matches displayed size at init time
        // (ResizeObserver fires async; the first user interaction may beat it).
        sizeCanvas();
    }

    // ── Canvas sizing ────────────────────────────────────────
    function sizeCanvas() {
        const vp = canvas.parentElement;
        canvas.width  = vp.clientWidth;
        canvas.height = vp.clientHeight;
    }

    // ── Rendering ────────────────────────────────────────────
    function render() {
        dbg();
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
        // Orange crop overlay sits on TOP of the video, only in export mode
        // and only when we're not currently capturing frames for export.
        _drawCropOverlay();
    }

    // ── Export ───────────────────────────────────────────────

    function _updateTrimTrack() {
        const tStart = $('trimStart'), tEnd = $('trimEnd'), track = $('trimTrack');
        if (!tStart || !tEnd || !track) return;
        const max = parseInt(tStart.max) || 1;
        const a = Math.min(parseInt(tStart.value), parseInt(tEnd.value));
        const b = Math.max(parseInt(tStart.value), parseInt(tEnd.value));
        const aP = (a / max) * 100;
        const bP = (b / max) * 100;
        // Orange outside the trim range, blue inside.
        track.style.background =
            `linear-gradient(to right,
                #FF9800 0%, #FF9800 ${aP}%,
                #2196f3 ${aP}%, #2196f3 ${bP}%,
                #FF9800 ${bP}%, #FF9800 100%)`;
    }

    function enterExportMode() {
        if (!trialMeta) { alert('No video loaded'); return; }
        if (playing) togglePlay();
        exportMode = true;
        document.body.classList.add('export-mode');
        const N = trialMeta.n_frames;
        const tStart = $('trimStart'), tEnd = $('trimEnd');
        tStart.min = tEnd.min = 0;
        tStart.max = tEnd.max = Math.max(0, N - 1);
        tStart.value = 0;
        tEnd.value = Math.max(0, N - 1);
        $('timelineSlider').style.display = 'none';
        $('trimUI').style.display = '';
        _updateTrimTrack();
        const btn = $('exportBtn');
        btn.textContent = 'Export';
        btn.classList.add('btn-primary');
        $('exportCancelBtn').style.display = '';
        $('exportStatus').textContent = '';
        // Default crop box = currently-visible video rectangle.
        _resetCropToView();
        render();
    }

    function exitExportMode() {
        if (exportRunning) return;          // don't bail out mid-export
        exportMode = false;
        document.body.classList.remove('export-mode');
        $('timelineSlider').style.display = '';
        $('trimUI').style.display = 'none';
        const btn = $('exportBtn');
        btn.textContent = 'Export Video';
        btn.classList.remove('btn-primary');
        btn.disabled = false;
        $('exportCancelBtn').style.display = 'none';
        $('exportStatus').textContent = '';
        cropDragMode = null;
        cropDragStart = null;
        canvas.style.cursor = '';
        render();    // redraw to clear the overlay
    }

    function toggleExportMode() {
        if (!exportMode) enterExportMode();
        else runExport();
    }

    // Run the export in-page (no modal), using the trim handles + speed.
    async function runExport() {
        if (exportRunning) return;
        const tA = parseInt($('trimStart').value);
        const tB = parseInt($('trimEnd').value);
        const startFrame = Math.min(tA, tB);
        const endFrame   = Math.max(tA, tB);
        if (endFrame <= startFrame) { alert('Trim range is empty'); return; }
        const totalFrames = endFrame - startFrame + 1;

        const status = $('exportStatus');
        const btn = $('exportBtn');
        const cancelBtn = $('exportCancelBtn');
        exportRunning = true;
        btn.disabled = true;
        cancelBtn.disabled = true;
        status.textContent = 'Starting…';

        const savedFrame = currentFrame;
        let exportId = null;
        // Snapshot the crop in canvas-buffer pixels (round to integers — the
        // server's ffmpeg encoder needs even dimensions; clamp to canvas).
        const cx = Math.max(0, Math.round(cropX));
        const cy = Math.max(0, Math.round(cropY));
        let cw = Math.min(canvas.width  - cx, Math.round(cropW));
        let ch = Math.min(canvas.height - cy, Math.round(cropH));
        // Force even dimensions (libx264 yuv420p requires even W×H).
        if (cw % 2) cw -= 1;
        if (ch % 2) ch -= 1;
        try {
            const outW = cw;
            const outH = ch;
            const outFps = (trialMeta.fps || 30) * (playbackRate || 1);

            const startResp = await fetch('/api/export-video/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fps: outFps,
                    width: outW,
                    height: outH,
                    total_frames: totalFrames,
                }),
            });
            if (!startResp.ok) throw new Error('start session failed');
            exportId = (await startResp.json()).export_id;

            const offscreen = document.createElement('canvas');
            offscreen.width = outW;
            offscreen.height = outH;
            const offCtx = offscreen.getContext('2d');
            const BATCH = 100;

            for (let batchStart = startFrame; batchStart <= endFrame; batchStart += BATCH) {
                const batchEnd = Math.min(batchStart + BATCH - 1, endFrame);
                const fd = new FormData();
                fd.append('start_index', batchStart - startFrame);
                for (let f = batchStart; f <= batchEnd; f++) {
                    await seekAndRenderFrame(f);
                    offCtx.fillStyle = '#000';
                    offCtx.fillRect(0, 0, outW, outH);
                    // Capture only the crop region (overlay is suppressed
                    // because exportRunning is true).
                    offCtx.drawImage(canvas, cx, cy, cw, ch, 0, 0, outW, outH);
                    const blob = await new Promise(r => offscreen.toBlob(r, 'image/jpeg', 0.92));
                    const globalIdx = f - startFrame;
                    fd.append(`frame_${f - batchStart}`, blob,
                              `frame_${String(globalIdx).padStart(6, '0')}.jpg`);
                    status.textContent = `Capturing ${f - startFrame + 1} / ${totalFrames}`;
                }
                status.textContent = `Uploading batch…`;
                const upResp = await fetch(`/api/export-video/${exportId}/frames`, {
                    method: 'POST', body: fd,
                });
                if (!upResp.ok) throw new Error('frame upload failed');
            }

            status.textContent = 'Encoding…';
            const encResp = await fetch(`/api/export-video/${exportId}/encode`, { method: 'POST' });
            if (!encResp.ok) throw new Error('encoding failed');
            const mp4Blob = await encResp.blob();
            const url = URL.createObjectURL(mp4Blob);
            const a = document.createElement('a');
            const stem = (trialMeta.trial_name || 'export').replace(/\.\w+$/, '');
            a.href = url;
            a.download = `${stem}_trim.mp4`;
            a.textContent = 'Download MP4';
            status.textContent = 'Done.';
            status.appendChild(a);
            exportId = null;
        } catch (err) {
            console.error(err);
            status.textContent = 'Error: ' + err.message;
        } finally {
            exportRunning = false;
            btn.disabled = false;
            cancelBtn.disabled = false;
            // Restore playback position
            goToFrame(savedFrame);
        }
    }

    function seekAndRenderFrame(f) {
        return new Promise(resolve => {
            currentFrame = Math.max(0, Math.min(f, trialMeta.n_frames - 1));
            $('frameDisplay').textContent = currentFrame;
            if (videoEl.readyState >= 2 && trialMeta && trialMeta.fps) {
                const t = (currentFrame + 0.5) / trialMeta.fps;
                videoEl.currentTime = t;
                videoEl.addEventListener('seeked', () => { render(); resolve(); }, { once: true });
            } else {
                resolve();
            }
        });
    }

    // Wire trim slider events once the DOM is ready.
    function _wireTrimHandles() {
        const tStart = $('trimStart'), tEnd = $('trimEnd');
        if (!tStart || !tEnd) return;
        const onInput = (e) => {
            // Keep start ≤ end by pushing the other handle out of the way.
            let a = parseInt(tStart.value), b = parseInt(tEnd.value);
            if (a > b) {
                if (e.target === tStart) tEnd.value = a;
                else                     tStart.value = b;
            }
            // Live-preview the frame at the handle being dragged.
            goToFrame(parseInt(e.target.value));
            _updateTrimTrack();
        };
        tStart.addEventListener('input', onInput);
        tEnd.addEventListener('input', onInput);
    }

    // Legacy alias (other code may still reference openExportModal).
    function openExportModal() { toggleExportMode(); }

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

    window.videosViewer = { getExportContext, openExportModal, toggleExportMode, exitExportMode };
    window.openExportModal = openExportModal;
    window.toggleExportMode = toggleExportMode;
    window.exitExportMode = exitExportMode;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
