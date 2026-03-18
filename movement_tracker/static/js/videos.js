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
    let playbackRate = 0.5;

    let cameraNames = ['OS', 'OD'];
    let currentSide = 'OS';

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

        // Load camera names from settings
        try {
            const cfg = await api('/api/settings');
            if (Array.isArray(cfg.camera_names) && cfg.camera_names.length >= 1) {
                cameraNames = cfg.camera_names;
                currentSide = cameraNames[0];
            }
        } catch { /* defaults */ }

        $('sideToggle').textContent = currentSide;

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

        // Restore from URL
        const params = new URLSearchParams(window.location.search);
        const subParam = params.get('subject');
        if (subParam) sel.value = subParam;

        setupControls();
        setupCanvasEvents();

        if (sel.value) loadSubject(parseInt(sel.value));
    }

    // ── Subject / Trial loading ──────────────────────────────
    async function loadSubject(sid) {
        subjectId = sid;
        const subj = allSubjects.find(s => s.id === sid);
        subjectName = subj ? subj.name : '';

        const u = new URL(window.location);
        u.searchParams.set('subject', sid);
        history.replaceState(null, '', u);

        trials = [];
        try { trials = await api(`/api/mano/${sid}/video_list`); } catch { /* no videos */ }

        buildTrialButtons();

        if (trials.length) loadTrial(0);
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
        trialMeta = trials[idx];
        isStereo = trialMeta.is_stereo !== false;

        highlightTrialButton(idx);

        currentFrame = 0;
        scale = 1; offsetX = 0; offsetY = 0;

        $('totalFramesDisplay').textContent = trialMeta.n_frames;
        $('frameDisplay').textContent = 0;
        $('timelineSlider').max = trialMeta.n_frames - 1;
        $('timelineSlider').value = 0;

        videoEl.src = `/api/mano/${subjectId}/trial/${trialMeta.trial_idx}/video`;
        videoEl.addEventListener('loadedmetadata', () => {
            vidW = videoEl.videoWidth;
            vidH = videoEl.videoHeight;
            midline = isStereo ? Math.round(vidW / 2) : vidW;
            sizeCanvas();
            // Seek to midpoint of frame 0 — t=0 cannot be decoded by many codecs
            videoEl.currentTime = 0.5 / trialMeta.fps;
            videoEl.addEventListener('seeked', render, { once: true });
        }, { once: true });
    }

    // ── Controls setup ───────────────────────────────────────
    function setupControls() {
        $('prevFrameBtn').addEventListener('click', () => goToFrame(currentFrame - 1));
        $('nextFrameBtn').addEventListener('click', () => goToFrame(currentFrame + 1));
        $('playBtn').addEventListener('click', togglePlay);
        $('sideToggle').addEventListener('click', toggleSide);
        $('resetZoomBtn').addEventListener('click', resetZoom);
        $('subjectSelect').addEventListener('change', e => {
            const v = parseInt(e.target.value);
            if (v) loadSubject(v);
        });

        // Speed slider — start at index 2 = 0.5x
        const speedSlider = $('speedSlider');
        speedSlider.value = 2;
        speedSlider.addEventListener('input', () => {
            playbackRate = SPEED_PRESETS[parseInt(speedSlider.value)];
            $('speedDisplay').textContent = playbackRate + 'x';
            if (playing) {
                videoEl.playbackRate = Math.min(playbackRate, 16);
            }
        });

        // Timeline seek
        const timeline = $('timelineSlider');
        timeline.addEventListener('input', () => {
            if (!trialMeta) return;
            goToFrame(parseInt(timeline.value));
        });

        // Keyboard
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            switch (e.key) {
                case 'a': case 'ArrowLeft':  goToFrame(currentFrame - 1); e.preventDefault(); break;
                case 's': case 'ArrowRight': goToFrame(currentFrame + 1); e.preventDefault(); break;
                case ' ': togglePlay(); e.preventDefault(); break;
                case 'e': case 'E': toggleSide(); break;
                case 'z': case 'Z': resetZoom(); break;
            }
        });
    }

    // ── Camera toggle ────────────────────────────────────────
    function toggleSide() {
        if (!isStereo) return;
        currentSide = currentSide === cameraNames[0]
            ? (cameraNames[1] || cameraNames[0])
            : cameraNames[0];
        $('sideToggle').textContent = currentSide;
        render();
    }

    // ── Playback ─────────────────────────────────────────────
    function goToFrame(n) {
        if (!trialMeta) return;
        currentFrame = Math.max(0, Math.min(n, trialMeta.n_frames - 1));
        $('frameDisplay').textContent = currentFrame;
        $('timelineSlider').value = currentFrame;
        // Seek then render — never render before seeked or the previous decoded
        // frame (or a blank) flashes before the new one arrives.
        if (videoEl.readyState >= 2 && trialMeta.fps) {
            const t = (currentFrame + 0.5) / trialMeta.fps;
            videoEl.currentTime = t;
            videoEl.addEventListener('seeked', render, { once: true });
        } else {
            render();
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
    function resetZoom() { scale = 1; offsetX = 0; offsetY = 0; render(); }

    function setupCanvasEvents() {
        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const ns = Math.max(0.1, Math.min(scale * factor, 50));
            offsetX = mx - (mx - offsetX) * (ns / scale);
            offsetY = my - (my - offsetY) * (ns / scale);
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
        const sw = isStereo ? midline : vidW;
        const bps = w / sw;  // base pixel scale at scale=1

        ctx.save();
        ctx.translate(offsetX, offsetY);
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
            canvasLayers: [canvas],
            distanceCanvas: null,
            seekAndRender(frameNum) { goToFrame(frameNum); },
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
