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

    // On-page debug overlay (no DevTools needed).
    let _dbgLastEvent = '';
    function dbg(label) {
        if (label) _dbgLastEvent = label;
        const o = document.getElementById('debugOverlay');
        if (!o) return;
        o.textContent =
            `canvas buf: ${canvas ? canvas.width : '?'}×${canvas ? canvas.height : '?'}\n` +
            `viewport:   ${canvas && canvas.parentElement ? canvas.parentElement.clientWidth : '?'}×${canvas && canvas.parentElement ? canvas.parentElement.clientHeight : '?'}\n` +
            `vid: ${vidW}×${vidH}  rs=${videoEl ? videoEl.readyState : '?'}\n` +
            `scale=${scale.toFixed(3)}  off=(${offsetX.toFixed(1)},${offsetY.toFixed(1)})\n` +
            `drag=${dragging}  last=${_dbgLastEvent}`;
    }
    function setupCanvasEvents() {
        canvas.style.cursor = 'grab';
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
            const factor = e.deltaY < 0 ? 1.05 : 1 / 1.05;
            const ns = Math.max(0.1, Math.min(scale * factor, 50));
            offsetX = lx - (lx - offsetX) * (ns / scale);
            offsetY = ly - (ly - offsetY) * (ns / scale);
            scale = ns;
            render();
        }, { passive: false });

        // Pan = left-mouse drag.  Down on canvas, move/up on window so the
        // drag continues even if the cursor leaves the canvas.
        canvas.addEventListener('mousedown', e => {
            dbg('canvasDown b=' + e.button);
            if (e.button !== 0 && e.button !== 1) return;
            ensureCanvasSized();
            // Self-heal NaN/Inf so the very first drag works even if some
            // earlier code (e.g. a degenerate MP-crop) poisoned the offsets.
            if (!isFinite(offsetX) || !isFinite(offsetY) || !isFinite(scale) || scale <= 0) {
                scale = 1; offsetX = 0; offsetY = 0;
            }
            dragging = true;
            canvas.style.cursor = 'grabbing';
            dragStartX = e.clientX; dragStartY = e.clientY;
            panStartOX = offsetX;   panStartOY = offsetY;
            e.preventDefault();
        });
        window.addEventListener('mousemove', e => {
            if (!dragging) return;
            offsetX = panStartOX + (e.clientX - dragStartX);
            offsetY = panStartOY + (e.clientY - dragStartY);
            render();
            dbg('pan');
        });
        window.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false;
            canvas.style.cursor = 'grab';
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
        // DEBUG: red outline + diagonal so we can see if the transform actually
        // applies.  Should move with pan and grow with zoom.
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 4 / scale;
        ctx.strokeRect(0, 0, sw * bps, vidH * bps);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(sw * bps, vidH * bps);
        ctx.stroke();
        ctx.restore();
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
        try {
            // Always use the on-canvas size so the output matches what the
            // user sees (including any zoom/pan crop).
            const outW = canvas.width, outH = canvas.height;
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
                    offCtx.drawImage(canvas, 0, 0, outW, outH);
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
