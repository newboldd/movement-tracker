/**
 * MANO 3D Hand Model Viewer
 *
 * Canvas-based stereo video viewer with 2D overlay (MANO, MediaPipe, DLC, heatmaps),
 * Three.js 3D hand viewport, distance trace, and per-finger visibility toggles.
 */
import * as THREE from 'three';

const manoViewer = (() => {
    // ── State ────────────────────────────────────────────────
    let allSubjects = [];
    let subjectId = null;
    let subjectName = '';
    let trials = [];
    let currentTrialIdx = -1;
    let trialData = null;

    let currentFrame = 0;
    let currentSide = 'OS';
    let cameraNames = ['OS', 'OD'];
    let playing = false;
    let playTimer = null;
    let playbackRate = 1;
    const SPEED_PRESETS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120];

    // Layer visibility
    let showVideo = true;
    let show3D = true;
    let showMano = true;
    let showMP = false;
    let showDLC = false;
    let showSkeleton = true;
    let showHeatmap = false;

    // Finger visibility
    const fingerVisibility = {
        wrist: true, thumb: true, index: true,
        middle: true, ring: true, pinky: true,
    };
    let visibleJoints = new Set([...Array(21).keys()]);

    // Heatmap
    let heatmapJoint = 4; // thumb tip default
    const heatmapCache = {};

    // Distance metric
    let selectedMetric = 'Thumb-Index Aperture';

    // Canvas
    let canvas, ctx;
    let distCanvas, distCtx;
    let videoEl;

    // Zoom/pan (in video-pixel space; render maps to canvas)
    let scale = 1, offsetX = 0, offsetY = 0;
    let defaultScale = 1, defaultOX = 0, defaultOY = 0; // auto-crop defaults
    let dragging = null;
    let dragStartX = 0, dragStartY = 0;
    let panStartOX = 0, panStartOY = 0;

    // Video dimensions
    let vidW = 0, vidH = 0, midline = 0;

    // Three.js
    let scene, camera3d, renderer;
    let manoGroup, mpGroup, dlcGroup;
    let camera3dInit = false;

    // Scene-space orbit: rotate content around hand center while camera stays fixed
    let orbitQuat = new THREE.Quaternion();
    let orbitPivot = new THREE.Vector3();
    let orbitDragging = false;
    let orbitLastX = 0, orbitLastY = 0;

    // ── Helpers ───────────────────────────────────────────────
    const $ = id => document.getElementById(id);

    async function api(url) {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
        return resp.json();
    }

    function isJointVisible(j) { return visibleJoints.has(j); }
    function isBoneVisible(i, j) { return visibleJoints.has(i) && visibleJoints.has(j); }

    function updateVisibleJoints() {
        visibleJoints.clear();
        if (!trialData) return;
        const groups = trialData.finger_groups;
        for (const [finger, joints] of Object.entries(groups)) {
            if (fingerVisibility[finger]) {
                joints.forEach(j => visibleJoints.add(j));
            }
        }
    }

    // ── Initialisation ───────────────────────────────────────
    async function init() {
        // Parse subject from URL
        const params = new URLSearchParams(window.location.search);
        const subjectParam = params.get('subject');

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

        if (subjectParam) {
            sel.value = subjectParam;
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

        // Setup Three.js
        setup3D();

        // Setup controls
        setupControls();
        setupCanvasEvents();

        // Populate joint selector
        const jointSel = $('jointSelect');
        const jointNames = [
            'Wrist', 'T_CMC', 'T_MCP', 'T_IP', 'T_Tip',
            'I_MCP', 'I_PIP', 'I_DIP', 'I_Tip',
            'M_MCP', 'M_PIP', 'M_DIP', 'M_Tip',
            'R_MCP', 'R_PIP', 'R_DIP', 'R_Tip',
            'P_MCP', 'P_PIP', 'P_DIP', 'P_Tip',
        ];
        jointNames.forEach((name, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `${i}: ${name}`;
            jointSel.appendChild(opt);
        });
        jointSel.value = heatmapJoint;
        jointSel.addEventListener('change', () => {
            heatmapJoint = parseInt(jointSel.value);
            render();
        });

        // Load initial subject
        if (sel.value) loadSubject(parseInt(sel.value));
    }

    async function loadSubject(sid) {
        subjectId = sid;
        const subj = allSubjects.find(s => s.id === sid);
        subjectName = subj ? subj.name : '';

        try {
            trials = await api(`/api/mano/${sid}/trials`);
        } catch {
            trials = [];
        }

        const trialSel = $('trialSelect');
        trialSel.innerHTML = '';
        if (!trials.length) {
            trialSel.innerHTML = '<option>No MANO data</option>';
            trialData = null;
            render();
            return;
        }

        trials.forEach((t, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = t.trial_stem;
            trialSel.appendChild(opt);
        });

        trialSel.addEventListener('change', () => loadTrial(parseInt(trialSel.value)));
        await loadTrial(0);
    }

    async function loadTrial(idx) {
        if (idx < 0 || idx >= trials.length) return;
        const trial = trials[idx];
        currentTrialIdx = idx;

        $('trialName').textContent = trial.trial_stem;
        $('totalFrames').textContent = trial.n_frames;

        // Load bulk data
        try {
            trialData = await api(`/api/mano/${subjectId}/trial/${trial.trial_idx}/data`);
        } catch (e) {
            console.error('Failed to load trial data:', e);
            trialData = null;
            return;
        }

        // Populate distance selector
        const distSel = $('distanceSelect');
        distSel.innerHTML = '';
        if (trialData.distance_options) {
            for (const name of Object.keys(trialData.distance_options)) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                distSel.appendChild(opt);
            }
        }
        distSel.value = selectedMetric;
        distSel.addEventListener('change', () => {
            selectedMetric = distSel.value;
            renderDistanceTrace();
        });

        // Load video
        videoEl.src = `/api/mano/${subjectId}/trial/${trial.trial_idx}/video`;
        videoEl.addEventListener('loadedmetadata', () => {
            vidW = videoEl.videoWidth;
            vidH = videoEl.videoHeight;
            midline = vidW / 2;
            sizeCanvases();
            computeAutoCrop();
            snapToCamera(); // default view = snapped to calibration camera
            // Seek to midpoint of frame 0 — t=0 cannot be decoded by many codecs
            videoEl.currentTime = 0.5 / trialData.fps;
            videoEl.addEventListener('seeked', () => {
                render();
                renderDistanceTrace();
                update3D();
            }, { once: true });
        }, { once: true });

        // Update state
        currentFrame = 0;
        updateVisibleJoints();
        camera3dInit = false;

        // Clear heatmap cache
        for (const k in heatmapCache) delete heatmapCache[k];
    }

    // ── Controls setup ───────────────────────────────────────
    function setupControls() {
        $('prevFrameBtn').addEventListener('click', () => goToFrame(currentFrame - 1));
        $('nextFrameBtn').addEventListener('click', () => goToFrame(currentFrame + 1));
        $('playBtn').addEventListener('click', togglePlay);
        $('sideToggle').addEventListener('click', toggleSide);
        $('resetZoomBtn').addEventListener('click', resetZoom);
        $('snapCamBtn').addEventListener('click', snapToCamera);
        $('prevSubjectBtn').addEventListener('click', prevSubject);
        $('nextSubjectBtn').addEventListener('click', nextSubject);

        // Speed slider
        const speedSlider = $('speedSlider');
        speedSlider.addEventListener('input', () => {
            playbackRate = SPEED_PRESETS[parseInt(speedSlider.value)];
            $('speedDisplay').textContent = playbackRate + 'x';
        });

        // Video / 3D visibility toggles
        $('showVideo').addEventListener('change', e => {
            showVideo = e.target.checked;
            canvas.parentElement.style.visibility = showVideo ? 'visible' : 'hidden';
            render();
        });
        $('show3D').addEventListener('change', e => {
            show3D = e.target.checked;
            const wrap = $('threejsContainer');
            wrap.style.visibility = show3D ? 'visible' : 'hidden';
            wrap.classList.toggle('interactive', show3D);
            if (show3D && renderer) renderer.render(scene, camera3d);
        });

        // Layer toggles
        $('showMano').addEventListener('change', e => { showMano = e.target.checked; render(); update3D(); });
        $('showMP').addEventListener('change', e => { showMP = e.target.checked; render(); update3D(); });
        $('showDLC').addEventListener('change', e => { showDLC = e.target.checked; render(); update3D(); });
        $('showSkeleton').addEventListener('change', e => { showSkeleton = e.target.checked; render(); update3D(); });
        $('showHeatmap').addEventListener('change', e => {
            showHeatmap = e.target.checked;
            $('heatmapControls').style.display = showHeatmap ? 'block' : 'none';
            render();
        });

        // Finger toggles
        document.querySelectorAll('#fingerToggles input').forEach(cb => {
            cb.addEventListener('change', () => {
                fingerVisibility[cb.dataset.finger] = cb.checked;
                updateVisibleJoints();
                render();
                update3D();
            });
        });

        // Keyboard
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            switch (e.key) {
                case 'a': case 'ArrowLeft':  goToFrame(currentFrame - 1); e.preventDefault(); break;
                case 's': case 'ArrowRight': goToFrame(currentFrame + 1); e.preventDefault(); break;
                case ' ': togglePlay(); e.preventDefault(); break;
                case 'e': toggleSide(); break;
                case 'z': resetZoom(); break;
                case 'c': snapToCamera(); break;
            }
        });
    }

    function toggleSide() {
        currentSide = currentSide === cameraNames[0] ? cameraNames[1] : cameraNames[0];
        $('sideToggle').textContent = currentSide;
        computeAutoCrop();
        render();
        snapToCamera(); // re-snap with new camera params
    }

    function resetZoom() {
        scale = defaultScale;
        offsetX = defaultOX;
        offsetY = defaultOY;
        render();
    }

    /**
     * Compute auto-crop from the bounding box of all projected points across
     * the entire trial, then set default scale/offset so the hand region fills
     * the canvas with ~15% padding.
     *
     * Render mapping (after the fix):
     *   bps = canvasWidth / sw          (base pixel scale at scale=1)
     *   ctx.translate(offsetX, offsetY)
     *   ctx.scale(scale, scale)
     *   → video drawn at (0,0)..(sw*bps, vidH*bps)
     *   → overlay point at videoPixel (vx,vy) drawn at (vx*bps, vy*bps)
     *   Final canvas coord = offsetX + scale * vx * bps
     */
    function computeAutoCrop() {
        if (!trialData || !canvas || vidW === 0) {
            defaultScale = 1; defaultOX = 0; defaultOY = 0;
            return;
        }

        const isLeft = currentSide === cameraNames[0];
        const proj = isLeft ? trialData.mano_proj_L : trialData.mano_proj_R;
        const mp   = isLeft ? trialData.mp_tracked_L : trialData.mp_tracked_R;
        const xOff = isLeft ? 0 : -midline;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

        // Sample every 5th frame for speed
        for (let f = 0; f < trialData.n_frames; f += 5) {
            const sources = [proj[f], mp[f]];
            for (const pts of sources) {
                if (!pts) continue;
                for (let j = 0; j < 21; j++) {
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

        if (!isFinite(minX) || !isFinite(minY)) {
            defaultScale = 1; defaultOX = 0; defaultOY = 0;
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
        const sw = isLeft ? midline : vidW - midline;
        const bps = cw / sw; // base pixel scale

        // We want the crop box to fill the canvas:
        //   cw = scale * cropW * bps   →  scaleX = cw / (cropW * bps)
        //   ch = scale * cropH * bps   →  scaleY = ch / (cropH * bps)
        const scaleX = cw / (cropW * bps);
        const scaleY = ch / (cropH * bps);
        defaultScale = Math.min(scaleX, scaleY);

        // Offset so crop center maps to canvas center:
        //   canvas_center = offsetX + scale * cropCenterX * bps
        //   offsetX = canvas_center - scale * cropCenterX * bps
        const cropCX = (minX + maxX) / 2;
        const cropCY = (minY + maxY) / 2;
        defaultOX = cw / 2 - defaultScale * cropCX * bps;
        defaultOY = ch / 2 - defaultScale * cropCY * bps;

        scale = defaultScale;
        offsetX = defaultOX;
        offsetY = defaultOY;
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

        $('frameDisplay').textContent = currentFrame;

        // Update non-video things immediately
        renderDistanceTrace();
        update3D();

        // Update fit error
        const side = currentSide === cameraNames[0] ? 'fit_error_L' : 'fit_error_R';
        const err = trialData[side]?.[currentFrame];
        $('fitError').textContent = err != null ? err.toFixed(1) + 'px' : '-';

        // Seek then render — never call render() before seeked or the previous
        // decoded frame (or a blank) flashes before the new one arrives.
        if (videoEl.readyState >= 2 && trialData.fps) {
            const t = (currentFrame + 0.5) / trialData.fps;
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
            videoEl.playbackRate = Math.min(playbackRate, 16); // browser limit
            videoEl.play().catch(() => {});
            playLoop();
        }
    }

    function playLoop() {
        if (!playing) return;
        if (videoEl.readyState >= 2 && trialData) {
            const f = Math.floor(videoEl.currentTime * trialData.fps);
            if (f !== currentFrame && f >= 0 && f < trialData.n_frames) {
                currentFrame = f;
                $('frameDisplay').textContent = currentFrame;
                render();
                update3D();
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
    function handleVideoZoom(e) {
        e.preventDefault();
        // Use the viewport container rect (both canvases fill the same area)
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
        applySnapProjection();
    }

    function handleVideoPanStart(e) {
        dragging = 'pan';
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        panStartOX = offsetX;
        panStartOY = offsetY;
        e.preventDefault();
    }

    function handleVideoPanMove(e) {
        if (dragging === 'pan') {
            offsetX = panStartOX + (e.clientX - dragStartX);
            offsetY = panStartOY + (e.clientY - dragStartY);
            render();
            applySnapProjection();
        }
    }

    function setupCanvasEvents() {
        // Video canvas: direct events (active when 3D layer is hidden / pointer-events: none)
        canvas.addEventListener('wheel', handleVideoZoom);
        canvas.addEventListener('mousedown', e => {
            if (e.button === 0 || e.button === 1) handleVideoPanStart(e);
        });
        canvas.addEventListener('mousemove', handleVideoPanMove);
        canvas.addEventListener('mouseup', () => { dragging = null; });
        canvas.addEventListener('mouseleave', () => { dragging = null; });

        // Three.js container event routing:
        // - Scroll → always zoom video+3D together (custom projection tracks zoom)
        // - Ctrl+drag → video pan
        // - Plain drag → handled by setup3D() for scene orbit
        const threeContainer = $('threejsContainer');
        threeContainer.addEventListener('wheel', e => {
            e.preventDefault();
            handleVideoZoom(e);
        }, { capture: true });

        threeContainer.addEventListener('mousedown', e => {
            if ((e.ctrlKey || e.metaKey) && (e.button === 0 || e.button === 1)) {
                e.stopPropagation();
                handleVideoPanStart(e);
            }
        }, { capture: true });

        threeContainer.addEventListener('mousemove', e => {
            if (dragging === 'pan') {
                handleVideoPanMove(e);
            }
        }, { capture: true });

        threeContainer.addEventListener('mouseup', () => { dragging = null; });
        threeContainer.addEventListener('mouseleave', () => { dragging = null; });

        // Distance trace click
        distCanvas.addEventListener('click', e => {
            if (!trialData) return;
            const rect = distCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const f = Math.floor((x / rect.width) * trialData.n_frames);
            goToFrame(f);
        });

        // Resize observer — observe the viewport container (.mano-viewports)
        const viewport = canvas.parentElement.parentElement;
        const ro = new ResizeObserver(() => {
            sizeCanvases();
            render();
            renderDistanceTrace();
            applySnapProjection();
        });
        ro.observe(viewport);
    }

    function sizeCanvases() {
        // Both canvases fill the same viewport container
        const viewport = canvas.parentElement.parentElement; // .mano-viewports
        const vw = viewport.clientWidth;
        const vh = viewport.clientHeight;
        canvas.width = vw;
        canvas.height = vh;

        distCanvas.width = distCanvas.clientWidth;
        distCanvas.height = distCanvas.clientHeight || 120;

        // Resize Three.js (overlaid, same size)
        if (renderer) {
            renderer.setSize(vw, vh);
            camera3d.aspect = vw / vh;
            // Don't reset — custom projection is re-applied by applySnapProjection().
            // camera3d.updateProjectionMatrix() would overwrite it.
        }
    }

    // ── 2D Rendering ─────────────────────────────────────────
    function render() {
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // Draw video frame + overlays with consistent zoom/pan transform
        if (videoEl.readyState >= 2 && vidW > 0) {
            const isLeft = currentSide === cameraNames[0];
            const sx = isLeft ? 0 : midline;
            const sw = isLeft ? midline : vidW - midline;

            // Base pixel scale: maps 1 video pixel to 1 canvas pixel at scale=1
            // We use the canvas width as reference so scale=1 means the half-frame
            // exactly fills the canvas width.
            const bps = w / sw;

            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);

            // Draw video: sw video pixels → sw * bps canvas pixels (at scale=1)
            ctx.drawImage(videoEl, sx, 0, sw, vidH, 0, 0, sw * bps, vidH * bps);

            // Draw overlays — pixelScale = bps (video-pixel → pre-transform canvas)
            if (trialData) {
                drawOverlays(bps);
            }

            ctx.restore();
        }
    }

    function drawOverlays(pixelScale) {
        const fn = currentFrame;
        if (!trialData || fn >= trialData.n_frames) return;

        const isLeft = currentSide === cameraNames[0];
        const manoProj = isLeft ? trialData.mano_proj_L : trialData.mano_proj_R;
        const mpKp = isLeft ? trialData.mp_tracked_L : trialData.mp_tracked_R;

        // For right camera, offset coordinates by -midline
        const xOff = isLeft ? 0 : -midline;

        // Draw skeleton lines
        if (showSkeleton && trialData.skeleton) {
            trialData.skeleton.forEach(([i, j]) => {
                if (!isBoneVisible(i, j)) return;

                // MANO skeleton (lime)
                if (showMano && manoProj[fn] && manoProj[fn][i] && manoProj[fn][j]) {
                    drawLine(
                        (manoProj[fn][i][0] + xOff) * pixelScale,
                        manoProj[fn][i][1] * pixelScale,
                        (manoProj[fn][j][0] + xOff) * pixelScale,
                        manoProj[fn][j][1] * pixelScale,
                        'lime', 2, 0.7
                    );
                }

                // MediaPipe skeleton (cyan)
                if (showMP && mpKp[fn] && mpKp[fn][i] && mpKp[fn][j]) {
                    drawLine(
                        (mpKp[fn][i][0] + xOff) * pixelScale,
                        mpKp[fn][i][1] * pixelScale,
                        (mpKp[fn][j][0] + xOff) * pixelScale,
                        mpKp[fn][j][1] * pixelScale,
                        'cyan', 1.5, 0.5
                    );
                }
            });
        }

        // MANO joints (lime circles)
        if (showMano && manoProj[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !manoProj[fn][j]) continue;
                const x = (manoProj[fn][j][0] + xOff) * pixelScale;
                const y = manoProj[fn][j][1] * pixelScale;
                drawJoint(x, y, 'lime', 4);
            }
        }

        // MediaPipe joints (cyan X)
        if (showMP && mpKp[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !mpKp[fn][j]) continue;
                const x = (mpKp[fn][j][0] + xOff) * pixelScale;
                const y = mpKp[fn][j][1] * pixelScale;
                drawCross(x, y, 'cyan', 4);
            }
        }

        // DLC thumb/index
        if (showDLC) {
            const thumbKey = isLeft ? 'dlc_thumb_OS' : 'dlc_thumb_OD';
            const indexKey = isLeft ? 'dlc_index_OS' : 'dlc_index_OD';
            if (isJointVisible(4) && trialData[thumbKey][fn]) {
                const pt = trialData[thumbKey][fn];
                drawJoint((pt[0] + xOff) * pixelScale, pt[1] * pixelScale, '#ff4444', 5);
            }
            if (isJointVisible(8) && trialData[indexKey][fn]) {
                const pt = trialData[indexKey][fn];
                drawJoint((pt[0] + xOff) * pixelScale, pt[1] * pixelScale, '#222', 5);
            }
        }

        // Heatmap overlay
        if (showHeatmap) {
            drawHeatmapOverlay(fn, heatmapJoint, pixelScale, xOff);
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

    function drawCross(x, y, color, size) {
        const s = size / scale;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5 / scale;
        ctx.beginPath();
        ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
        ctx.stroke();
        ctx.restore();
    }

    // ── Heatmap overlay ──────────────────────────────────────
    function hotColormap(val) {
        // black → red → yellow → white
        if (val <= 0) return [0, 0, 0];
        if (val <= 0.33) {
            const t = val / 0.33;
            return [Math.round(255 * t), 0, 0];
        }
        if (val <= 0.66) {
            const t = (val - 0.33) / 0.33;
            return [255, Math.round(255 * t), 0];
        }
        const t = (val - 0.66) / 0.34;
        return [255, 255, Math.round(255 * t)];
    }

    async function drawHeatmapOverlay(frame, joint, pixelScale, xOff) {
        if (!trialData) return;
        const trial = trials[currentTrialIdx];
        if (!trial || !trial.has_heatmaps) return;

        const key = `${frame}_${joint}_${currentSide}`;
        if (!heatmapCache[key]) {
            try {
                heatmapCache[key] = await api(
                    `/api/mano/${subjectId}/trial/${trial.trial_idx}/heatmap?frame=${frame}&joint=${joint}&side=${currentSide}`
                );
            } catch {
                return;
            }
        }

        const data = heatmapCache[key];
        if (!data || !data.heatmap) return;

        const hm = data.heatmap;
        const [bx1, by1, bx2, by2] = data.bbox;
        const hmH = hm.length, hmW = hm[0].length;

        // Create ImageData
        const offCanvas = document.createElement('canvas');
        offCanvas.width = hmW;
        offCanvas.height = hmH;
        const offCtx = offCanvas.getContext('2d');
        const imgData = offCtx.createImageData(hmW, hmH);

        for (let r = 0; r < hmH; r++) {
            for (let c = 0; c < hmW; c++) {
                const [cr, cg, cb] = hotColormap(hm[r][c]);
                const idx = (r * hmW + c) * 4;
                imgData.data[idx] = cr;
                imgData.data[idx + 1] = cg;
                imgData.data[idx + 2] = cb;
                imgData.data[idx + 3] = hm[r][c] > 0.05 ? 140 : 0;
            }
        }
        offCtx.putImageData(imgData, 0, 0);

        // Draw scaled to bbox position
        const dx = (bx1 + xOff) * pixelScale;
        const dy = by1 * pixelScale;
        const dw = (bx2 - bx1) * pixelScale;
        const dh = (by2 - by1) * pixelScale;

        ctx.drawImage(offCanvas, dx, dy, dw, dh);
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

        const manoDist = trialData.distances_mano?.[selectedMetric];
        const mpDist = trialData.distances_mp?.[selectedMetric];

        // Compute Y range
        let yMin = Infinity, yMax = -Infinity;
        const checkRange = arr => {
            if (!arr) return;
            for (const v of arr) {
                if (v != null) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
            }
        };
        if (showMano) checkRange(manoDist);
        if (showMP) checkRange(mpDist);

        if (!isFinite(yMin)) { yMin = 0; yMax = 100; }
        const pad = (yMax - yMin) * 0.1 || 10;
        yMin = Math.max(0, yMin - pad);
        yMax += pad;

        const toX = f => (f / (N - 1)) * w;
        const toY = v => h - ((v - yMin) / (yMax - yMin)) * h;

        // Draw series
        function drawSeries(data, color) {
            if (!data) return;
            distCtx.strokeStyle = color;
            distCtx.lineWidth = 1;
            distCtx.beginPath();
            let started = false;
            for (let i = 0; i < N; i++) {
                if (data[i] == null) { started = false; continue; }
                if (!started) { distCtx.moveTo(toX(i), toY(data[i])); started = true; }
                else distCtx.lineTo(toX(i), toY(data[i]));
            }
            distCtx.stroke();
        }

        if (showMano) drawSeries(manoDist, 'lime');
        if (showMP) drawSeries(mpDist, 'cyan');

        // Frame marker
        distCtx.strokeStyle = '#4a9eff';
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
        distCtx.fillText(yMax.toFixed(0) + 'mm', 4, 12);
        distCtx.fillText(yMin.toFixed(0) + 'mm', 4, h - 4);

        // Legend
        distCtx.textAlign = 'right';
        let lx = w - 8, ly = 14;
        if (showMano) {
            distCtx.fillStyle = 'lime';
            distCtx.fillText('MANO', lx, ly); ly += 14;
        }
        if (showMP) {
            distCtx.fillStyle = 'cyan';
            distCtx.fillText('MediaPipe', lx, ly); ly += 14;
        }

        // Metric name
        distCtx.fillStyle = '#666';
        distCtx.textAlign = 'center';
        distCtx.fillText(selectedMetric, w / 2, h - 4);
    }

    // ── Three.js 3D viewport ─────────────────────────────────
    function setup3D() {
        const container = $('threejsContainer');
        if (!container) return;

        scene = new THREE.Scene();
        scene.background = null; // transparent so video shows through

        camera3d = new THREE.PerspectiveCamera(
            50, container.clientWidth / container.clientHeight, 1, 50000
        );
        camera3d.position.set(0, 0, 500);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearColor(0x000000, 0);
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        container.classList.add('interactive');

        // Lights
        scene.add(new THREE.AmbientLight(0x606060));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(100, 100, 100);
        scene.add(dirLight);

        // Groups
        manoGroup = new THREE.Group();
        mpGroup = new THREE.Group();
        dlcGroup = new THREE.Group();
        scene.add(manoGroup, mpGroup, dlcGroup);

        renderer.render(scene, camera3d);

        // ── Manual orbit: rotate scene content, camera stays fixed ──
        // This keeps the custom projection valid at all times.
        container.addEventListener('mousedown', e => {
            // Ctrl+click → video pan (handled by capture listener)
            if (e.ctrlKey || e.metaKey) return;
            if (e.button === 0) {
                orbitDragging = true;
                orbitLastX = e.clientX;
                orbitLastY = e.clientY;
            }
        });
        container.addEventListener('mousemove', e => {
            if (!orbitDragging) return;
            const dx = e.clientX - orbitLastX;
            const dy = e.clientY - orbitLastY;
            orbitLastX = e.clientX;
            orbitLastY = e.clientY;
            // Horizontal drag → rotate around Y, vertical → around X
            const q = new THREE.Quaternion();
            q.setFromEuler(new THREE.Euler(dy * 0.005, dx * 0.005, 0, 'YXZ'));
            orbitQuat.premultiply(q);
            update3D();
        });
        container.addEventListener('mouseup', () => { orbitDragging = false; });
        container.addEventListener('mouseleave', () => { orbitDragging = false; });
    }

    /** Create a cylinder mesh between two Three.js Vector3 points. */
    function makeBone(a, b, radius, material) {
        const dir = new THREE.Vector3().subVectors(b, a);
        const len = dir.length();
        if (len < 0.01) return null;
        const geom = new THREE.CylinderGeometry(radius, radius, len, 6, 1);
        const mesh = new THREE.Mesh(geom, material);
        mesh.position.copy(a).add(b).multiplyScalar(0.5);
        const axis = new THREE.Vector3(0, 1, 0);
        mesh.quaternion.setFromUnitVectors(axis, dir.normalize());
        return mesh;
    }

    function update3D() {
        if (!scene || !trialData) return;
        const fn = currentFrame;

        // Clear groups
        [manoGroup, mpGroup, dlcGroup].forEach(g => {
            while (g.children.length) {
                const child = g.children[0];
                g.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            }
        });

        const mano3d = trialData.mano_joints_3d?.[fn];
        const mp3d = trialData.mp_joints_3d?.[fn];

        const sphereGeom = new THREE.SphereGeometry(2.5, 12, 8);

        // Helper to convert OpenCV (x,y,z) to Three.js scene (x,-y,-z)
        const toScene = p => new THREE.Vector3(p[0], -p[1], -p[2]);

        // Compute orbit pivot from current frame hand center
        const pivotPts = mano3d || mp3d;
        if (pivotPts) {
            let px = 0, py = 0, pz = 0, pn = 0;
            for (let j = 0; j < 21; j++) {
                if (!pivotPts[j]) continue;
                px += pivotPts[j][0]; py += -pivotPts[j][1]; pz += -pivotPts[j][2];
                pn++;
            }
            if (pn > 0) orbitPivot.set(px/pn, py/pn, pz/pn);
        }

        // Apply orbit rotation around pivot: p' = pivot + Q * (p - pivot)
        const orbitPt = p => {
            if (orbitQuat.w === 1) return p; // identity, skip math
            return p.clone().sub(orbitPivot).applyQuaternion(orbitQuat).add(orbitPivot);
        };

        // MANO joints (green)
        if (showMano && mano3d) {
            const manoMat = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
            const boneMat = new THREE.MeshPhongMaterial({ color: 0x00dd00 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !mano3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, manoMat);
                sphere.position.copy(orbitPt(toScene(mano3d[j])));
                manoGroup.add(sphere);
            }
            if (showSkeleton && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !mano3d[i] || !mano3d[j]) return;
                    const bone = makeBone(
                        orbitPt(toScene(mano3d[i])),
                        orbitPt(toScene(mano3d[j])),
                        1.2, boneMat
                    );
                    if (bone) manoGroup.add(bone);
                });
            }
        }

        // MediaPipe joints (cyan)
        if (showMP && mp3d) {
            const mpMat = new THREE.MeshPhongMaterial({ color: 0x5bc0de });
            const mpBoneMat = new THREE.MeshPhongMaterial({ color: 0x4aa0bb });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !mp3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, mpMat);
                sphere.position.copy(orbitPt(toScene(mp3d[j])));
                mpGroup.add(sphere);
            }
            if (showSkeleton && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !mp3d[i] || !mp3d[j]) return;
                    const bone = makeBone(
                        orbitPt(toScene(mp3d[i])),
                        orbitPt(toScene(mp3d[j])),
                        1.0, mpBoneMat
                    );
                    if (bone) mpGroup.add(bone);
                });
            }
        }

        // DLC thumb/index
        if (showDLC) {
            // Need 3D positions — triangulate if available
            // For now skip 3D DLC (only 2D data available)
        }

        manoGroup.visible = showMano;
        mpGroup.visible = showMP;
        dlcGroup.visible = showDLC;

        renderer.render(scene, camera3d);
    }

    // ── Snap-to-camera ──────────────────────────────────────
    // Positions the camera from calibration, sets the custom projection,
    // and resets any orbit rotation.  The camera never moves after this —
    // orbit drags rotate the scene content instead (see update3D / orbitPt).

    function snapToCamera() {
        if (!trialData?.calib || !canvas || vidW === 0) return;

        // Reset orbit rotation
        orbitQuat.identity();

        // Camera position & orientation from calibration
        const isLeft = currentSide === cameraNames[0];
        if (isLeft) {
            camera3d.position.set(0, 0, 0);
            camera3d.quaternion.identity();
        } else {
            const R = trialData.calib.R;
            const T = trialData.calib.T;
            const Rt = [
                [R[0][0], R[1][0], R[2][0]],
                [R[0][1], R[1][1], R[2][1]],
                [R[0][2], R[1][2], R[2][2]],
            ];
            const camPos = [
                -(Rt[0][0]*T[0] + Rt[0][1]*T[1] + Rt[0][2]*T[2]),
                -(Rt[1][0]*T[0] + Rt[1][1]*T[1] + Rt[1][2]*T[2]),
                -(Rt[2][0]*T[0] + Rt[2][1]*T[1] + Rt[2][2]*T[2]),
            ];
            camera3d.position.set(camPos[0], -camPos[1], -camPos[2]);
            const m = new THREE.Matrix4();
            m.set(
                R[0][0], -R[0][1], -R[0][2], 0,
                -R[1][0], R[1][1], R[1][2], 0,
                -R[2][0], R[2][1], R[2][2], 0,
                0, 0, 0, 1,
            );
            camera3d.quaternion.setFromRotationMatrix(m.clone().invert());
        }
        camera3d.updateMatrixWorld(true);

        // Apply custom projection
        applySnapProjection();
        update3D();
    }

    /** Recompute and apply the custom projection matrix for the current
     *  canvas size & zoom/pan state.  Called after snap, resize, and zoom/pan. */
    function applySnapProjection() {
        if (!trialData?.calib || !canvas || vidW === 0) return;

        const isLeft = currentSide === cameraNames[0];
        const K = isLeft ? trialData.calib.K_L : trialData.calib.K_R;
        const sw = isLeft ? midline : vidW - midline;

        const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
        const w = canvas.width, h = canvas.height;
        const bps = w / sw;
        const near = 0.1, far = 50000;

        const m00 = 2 * scale * fx * bps / w;
        const m02 = 1 - 2 * (offsetX + scale * cx * bps) / w;
        const m11 = 2 * scale * fy * bps / h;
        const m12 = 2 * (offsetY + scale * cy * bps) / h - 1;
        const m22 = -(far + near) / (far - near);
        const m23 = -2 * far * near / (far - near);

        camera3d.projectionMatrix.set(
            m00, 0,   m02, 0,
            0,   m11, m12, 0,
            0,   0,   m22, m23,
            0,   0,   -1,  0,
        );
        camera3d.projectionMatrixInverse.copy(camera3d.projectionMatrix).invert();

        renderer.render(scene, camera3d);
    }

    // ── Video export context ────────────────────────────────
    function getExportContext() {
        return {
            videoEl,
            fps: trialData?.fps || 30,
            playbackRate,
            nFrames: trialData?.n_frames || 0,
            get currentFrame() { return currentFrame; },
            canvasLayers: [canvas],
            distanceCanvas: distCanvas,
            getCompositeSize() {
                return { width: canvas.width, height: canvas.height };
            },
            async seekAndRender(n) {
                currentFrame = Math.max(0, Math.min(n, (trialData?.n_frames || 1) - 1));
                if (videoEl.readyState >= 2 && trialData?.fps) {
                    videoEl.currentTime = currentFrame / trialData.fps + 0.001;
                    await new Promise(resolve => {
                        videoEl.addEventListener('seeked', resolve, { once: true });
                        setTimeout(resolve, 2000); // timeout fallback
                    });
                }
                render();
                update3D();
                renderDistanceTrace();
                $('frameDisplay').textContent = currentFrame;
            },
            renderThreeJS() {
                if (renderer && scene && camera3d) {
                    applySnapProjection(); // calls renderer.render(scene, camera3d)
                }
                return renderer?.domElement || null;
            },
        };
    }

    // ── Boot ─────────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            init().catch(e => console.error('MANO viewer init failed:', e));
        });
    } else {
        init().catch(e => console.error('MANO viewer init failed:', e));
    }

    return { goToFrame, togglePlay, toggleSide, resetZoom, prevSubject, nextSubject, getExportContext };
})();

// Expose on window for cross-module access (mano.js is an ES module)
window.manoViewer = manoViewer;
