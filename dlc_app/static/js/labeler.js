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
    let dragging = null; // bodypart name | 'pan'
    let dragStartX = 0, dragStartY = 0;
    let dragOrigX = 0, dragOrigY = 0;

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

    async function goToFrame(frame) {
        if (frame < 0 || frame >= totalFrames) return;
        currentFrame = frame;

        try {
            currentImage = await loadImage(frame, currentSide);
            imgW = currentImage.width;
            imgH = currentImage.height;
            fitImage();
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
                fitImage();
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
            const key = `${currentFrame}_${currentSide}`;
            const lbl = labels.get(key);
            const coords = lbl[hit];
            dragOrigX = coords[0];
            dragOrigY = coords[1];
            dragStartX = sx;
            dragStartY = sy;
            canvas.style.cursor = 'grabbing';
        } else {
            // Check if click is within image bounds
            const img = screenToImage(sx, sy);
            if (img.x >= 0 && img.x < imgW && img.y >= 0 && img.y < imgH) {
                // Place new label
                placeLabel(img.x, img.y);
            } else {
                // Pan
                dragging = 'pan';
                dragStartX = sx;
                dragStartY = sy;
                dragOrigX = offsetX;
                dragOrigY = offsetY;
                canvas.style.cursor = 'move';
            }
        }
    }

    function onMouseMove(e) {
        if (!dragging) return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

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
        if (dragging && dragging !== 'pan') {
            // Auto-save after drag
            scheduleSave();
        }
        dragging = null;
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

        render();
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
                lbl[closest] = [imgX, imgY];
            }
        }

        render();
        updateLabelCount();
        scheduleSave();
    }

    function removeLabel(which) {
        const key = `${currentFrame}_${currentSide}`;
        const lbl = labels.get(key);
        if (!lbl) return;

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

    function nextLabel() {
        const sorted = getLabeledFrames();
        const next = sorted.find(f => f > currentFrame);
        if (next !== undefined) goToFrame(next);
    }

    function prevLabel() {
        const sorted = getLabeledFrames();
        const prev = [...sorted].reverse().find(f => f < currentFrame);
        if (prev !== undefined) goToFrame(prev);
    }

    function getLabeledFrames() {
        const frames = new Set();
        labels.forEach((_, key) => {
            const [frameStr, side] = key.split('_');
            if (side === currentSide) frames.add(parseInt(frameStr));
        });
        return [...frames].sort((a, b) => a - b);
    }

    function toggleSide() {
        const idx = cameraNames.indexOf(currentSide);
        currentSide = cameraNames[(idx + 1) % cameraNames.length];
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
            playTimer = setInterval(() => {
                if (currentFrame < totalFrames - 1) {
                    goToFrame(currentFrame + 1);
                } else {
                    togglePlay();
                }
            }, interval);
        } else {
            btn.innerHTML = '&#9654;';
            clearInterval(playTimer);
            playTimer = null;
        }
    }

    function resetZoom() {
        fitImage();
        render();
    }

    // ── Keyboard shortcuts ────────────────────────────
    function setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            switch (e.key) {
                case 'a': case 'ArrowLeft':
                    e.preventDefault();
                    prevFrame();
                    break;
                case 's': case 'ArrowRight':
                    e.preventDefault();
                    nextFrame();
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

        let html = `
            <div><kbd>A</kbd> / <kbd>&larr;</kbd> Prev frame</div>
            <div><kbd>S</kbd> / <kbd>&rarr;</kbd> Next frame</div>
        `;
        bodyparts.forEach((bp, idx) => {
            html += `<div><kbd>${idx + 1}</kbd> Delete ${bp}</div>`;
        });
        html += `
            <div><kbd>E</kbd> Toggle ${cameraNames.join('/')}</div>
            <div><kbd>Z</kbd> Reset zoom</div>
            <div><kbd>Space</kbd> Play/pause</div>
            <div><kbd>Scroll</kbd> Zoom at cursor</div>
            <div><kbd>Click</kbd> Place label</div>
            <div><kbd>Drag</kbd> Move label</div>
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
