/* Canvas labeling engine for DLC finger-tapping keypoint annotation */

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

    // Color palette for bodyparts
    const COLORS = [
        '#ff4444', '#222222', '#4a9eff', '#4caf50',
        '#ff9800', '#9c27b0', '#00bcd4', '#e91e63', '#795548',
    ];

    function bpColor(idx) { return COLORS[idx % COLORS.length]; }
    function bpLetter(name) { return name[0].toUpperCase(); }

    // Canvas state
    let canvas, ctx;
    let timeline, tlCtx;
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

    // Undo stack: each entry = { key, bp, prev (coords or null), cur (coords or null) }
    const undoStack = [];
    const MAX_UNDO = 50;

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
        containerEl = document.getElementById('canvasContainer');

        setupCanvasEvents();
        setupTimeline();

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
            currentSide = cameraNames[0] || 'OS';

            document.getElementById('headerTitle').textContent =
                `Labeling: ${sessionInfo.subject.name}`;

            // Update sidebar with dynamic shortcuts
            updateShortcutsSidebar();
            document.getElementById('sideToggle').textContent = currentSide;

            // Setup keyboard after bodyparts are known
            setupKeyboard();

            // Load existing labels
            const saved = await API.get(`/api/labeling/sessions/${sessionId}/labels`);
            saved.forEach(l => {
                const key = `${l.frame_num}_${l.side}`;
                labels.set(key, l.keypoints || {});
            });

            recomputeCameraShift();
            updateLabelCount();
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

        updateFrameDisplay();
        renderTimeline();
    }

    function fitImage() {
        if (!imgW || !imgH) return;
        const cw = containerEl.clientWidth;
        const ch = containerEl.clientHeight;
        scale = Math.min(cw / imgW, ch / imgH);
        offsetX = (cw - imgW * scale) / 2;
        offsetY = (ch - imgH * scale) / 2;
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

        // Draw labels for current frame + side
        const key = `${currentFrame}_${currentSide}`;
        const lbl = labels.get(key);
        if (lbl) {
            const placedBps = [];
            bodyparts.forEach((bp, idx) => {
                const coords = lbl[bp];
                if (coords && coords[0] != null && coords[1] != null) {
                    drawPoint(coords[0], coords[1], bpColor(idx), bpLetter(bp));
                    placedBps.push({ bp, x: coords[0], y: coords[1] });
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
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
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

    // ── Hit testing ───────────────────────────────────
    function hitTest(sx, sy) {
        const key = `${currentFrame}_${currentSide}`;
        const lbl = labels.get(key);
        if (!lbl) return null;

        for (const bp of bodyparts) {
            const coords = lbl[bp];
            if (coords && coords[0] != null) {
                const p = imageToScreen(coords[0], coords[1]);
                if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) return bp;
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
            }
        });
        ro.observe(containerEl);
    }

    function onMouseDown(e) {
        if (e.button === 2) return; // right-click handled separately
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        const hit = hitTest(sx, sy);
        if (hit) {
            // Start dragging existing point
            dragging = hit;
            didDrag = false;
            const key = `${currentFrame}_${currentSide}`;
            const lbl = labels.get(key);
            const coords = lbl[hit];
            dragOrigX = coords[0];
            dragOrigY = coords[1];
            dragStartX = sx;
            dragStartY = sy;
            canvas.style.cursor = 'grabbing';
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
            // Mouse didn't move much — this is a click, place a label
            const img = screenToImage(dragStartX, dragStartY);
            if (img.x >= 0 && img.x < imgW && img.y >= 0 && img.y < imgH) {
                placeLabel(img.x, img.y);
            }
        } else if (dragging === 'pan') {
            hasUserZoom = true;
        } else if (dragging) {
            // Finished dragging a bodypart point — record undo with pre-drag position
            const key = `${currentFrame}_${currentSide}`;
            pushUndo(key, dragging, [dragOrigX, dragOrigY]);
            scheduleSave();
            recomputeCameraShift();
        }
        dragging = null;
        didDrag = false;
        canvas.style.cursor = 'crosshair';
    }

    function onRightClick(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        const hit = hitTest(sx, sy);
        if (hit) {
            removeLabel(hit);
        }
    }

    function onWheel(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
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
        } else {
            // Was a new placement — remove it
            if (lbl) {
                delete lbl[bp];
                const hasAny = bodyparts.some(b => lbl[b] && lbl[b][0] != null);
                if (!hasAny) labels.delete(key);
            }
        }

        render();
        updateLabelCount();
        scheduleSave();
        recomputeCameraShift();
    }

    // ── Zoom to labels ───────────────────────────────
    function zoomToLabels(frame, side) {
        const key = `${frame}_${side}`;
        const lbl = labels.get(key);
        if (!lbl) return;

        // Collect all placed coordinates
        const pts = [];
        for (const bp of bodyparts) {
            const c = lbl[bp];
            if (c && c[0] != null) pts.push(c);
        }
        if (pts.length === 0) return;

        // Bounding box in image coords
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const [x, y] of pts) {
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        // Add padding around the bounding box (in image pixels)
        const pad = Math.max(maxX - minX, maxY - minY, 80) * 0.8;
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

        render();
        updateLabelCount();
        scheduleSave();
        recomputeCameraShift();
    }

    function removeLabel(which) {
        const key = `${currentFrame}_${currentSide}`;
        const lbl = labels.get(key);
        if (!lbl) return;

        const prev = lbl[which];
        if (prev && prev[0] != null) pushUndo(key, which, [...prev]);
        delete lbl[which];

        // Remove entry if all bodyparts are empty
        const hasAny = bodyparts.some(bp => lbl[bp] && lbl[bp][0] != null);
        if (!hasAny) {
            labels.delete(key);
        }

        render();
        updateLabelCount();
        scheduleSave();
    }

    // ── Auto-save ─────────────────────────────────────
    let saveTimeout = null;

    function scheduleSave() {
        if (saveTimeout) clearTimeout(saveTimeout);
        saveTimeout = setTimeout(saveLabels, 2000);
    }

    async function saveLabels() {
        if (saveTimeout) clearTimeout(saveTimeout);
        const batch = [];

        labels.forEach((lbl, key) => {
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

        if (batch.length === 0) return;

        try {
            await API.put(`/api/labeling/sessions/${sessionId}/labels`, { labels: batch });
        } catch (e) {
            console.error('Save failed:', e);
        }
    }

    // ── Commit ────────────────────────────────────────
    async function commitSession() {
        // Save first
        await saveLabels();

        const count = labels.size;
        if (count === 0) {
            alert('No labels to commit.');
            return;
        }

        if (!confirm(`Commit ${count} labeled frames to DLC? This will extract PNGs and write CollectedData CSV.`)) {
            return;
        }

        try {
            const result = await API.post(`/api/labeling/sessions/${sessionId}/commit`);
            alert(`Committed ${result.frame_count} frames to ${result.labeled_data_dir}`);
        } catch (e) {
            alert('Commit error: ' + e.message);
        }
    }

    // ── Navigation ────────────────────────────────────
    function nextFrame() { goToFrame(currentFrame + 1); }
    function prevFrame() { goToFrame(currentFrame - 1); }

    async function nextLabel() {
        const sorted = getLabeledFrames();
        const next = sorted.find(f => f > currentFrame);
        if (next !== undefined) {
            zoomToLabels(next, currentSide);
            await goToFrame(next);
        }
    }

    async function prevLabel() {
        const sorted = getLabeledFrames();
        const prev = [...sorted].reverse().find(f => f < currentFrame);
        if (prev !== undefined) {
            zoomToLabels(prev, currentSide);
            await goToFrame(prev);
        }
    }

    function getLabeledFrames() {
        const frames = new Set();
        labels.forEach((_, key) => {
            const [frameStr, side] = key.split('_');
            if (side === currentSide) frames.add(parseInt(frameStr));
        });
        return [...frames].sort((a, b) => a - b);
    }

    function recomputeCameraShift() {
        // Compute the average horizontal and vertical offset between OS/OD labels
        // for frames that have labels in both cameras.
        if (cameraNames.length < 2) return;
        const cam0 = cameraNames[0];
        const cam1 = cameraNames[1];
        const dxValues = [];
        const dyValues = [];

        // Find frames that have labels in both cameras
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

        if (dxValues.length > 0) {
            computedCameraShiftX = dxValues.reduce((a, b) => a + b, 0) / dxValues.length;
            computedCameraShiftY = dyValues.reduce((a, b) => a + b, 0) / dyValues.length;
        }
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
            const fps = trials.length > 0 ? trials[0].fps : 30;
            const interval = 1000 / (fps * playbackRate);

            // Sequential async loop — waits for each frame to load
            // before scheduling the next, preventing stacking
            (async function playLoop() {
                while (playing && currentFrame < totalFrames - 1) {
                    const start = performance.now();
                    await goToFrame(currentFrame + 1);
                    const elapsed = performance.now() - start;
                    const wait = Math.max(0, interval - elapsed);
                    await new Promise(r => setTimeout(r, wait));
                }
                if (playing) {
                    // Reached end — stop
                    playing = false;
                    btn.innerHTML = '&#9654;';
                }
            })();
        } else {
            btn.innerHTML = '&#9654;';
        }
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

            // Ctrl+Z: undo
            if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                undo();
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
                    prevLabel();
                    break;
                case 'w':
                    e.preventDefault();
                    nextLabel();
                    break;
                case 'e':
                    e.preventDefault();
                    toggleSide();
                    break;
                case 'z':
                    e.preventDefault();
                    resetZoom();
                    break;
                case ' ':
                    e.preventDefault();
                    togglePlay();
                    break;
                default:
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
        document.getElementById('frameDisplay').textContent =
            `Frame: ${currentFrame} / ${totalFrames - 1}`;

        // Find current trial
        let trialName = '--';
        for (const t of trials) {
            if (currentFrame >= t.start_frame && currentFrame <= t.end_frame) {
                trialName = t.trial_name;
                break;
            }
        }
        document.getElementById('trialDisplay').textContent = `Trial: ${trialName}`;
    }

    function updateLabelCount() {
        const count = labels.size;
        document.getElementById('labelCount').innerHTML =
            `Labels: <strong>${count}</strong>`;
    }

    function updateLabelInfo(msg) {
        document.getElementById('labelInfo').textContent = msg;
    }

    function updateShortcutsSidebar() {
        const el = document.getElementById('shortcutList');
        if (!el) return;

        const deleteLetters = ['D', 'F'];
        let html = `
            <div><kbd>A</kbd> / <kbd>&larr;</kbd> Prev frame</div>
            <div><kbd>S</kbd> / <kbd>&rarr;</kbd> Next frame</div>
            <div><kbd>Q</kbd> Prev label</div>
            <div><kbd>W</kbd> Next label</div>
        `;
        bodyparts.forEach((bp, idx) => {
            const letter = deleteLetters[idx] ? `<kbd>${deleteLetters[idx]}</kbd> / ` : '';
            html += `<div>${letter}<kbd>${idx + 1}</kbd> Delete ${bp}</div>`;
        });
        html += `
            <div><kbd>Ctrl+Z</kbd> Undo</div>
            <div><kbd>E</kbd> Toggle ${cameraNames.join('/')}</div>
            <div><kbd>Z</kbd> Reset zoom</div>
            <div><kbd>Space</kbd> Play/pause</div>
            <div><kbd>Scroll</kbd> Zoom at cursor</div>
            <div><kbd>Click</kbd> Place label</div>
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
        const frame = Math.floor((x / rect.width) * totalFrames);
        goToFrame(Math.max(0, Math.min(frame, totalFrames - 1)));
    }

    function renderTimeline() {
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

        // Label dots
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
                    tlCtx.arc(x, dotY, 2, 0, Math.PI * 2);
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

    // ── Public API ────────────────────────────────────
    return {
        init,
        nextFrame, prevFrame, nextLabel, prevLabel,
        toggleSide, togglePlay, resetZoom,
        saveLabels, commitSession,
    };
})();

// Init on page load
document.addEventListener('DOMContentLoaded', () => labeler.init());
