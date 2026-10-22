/* The labeling page.
 *
 * One page, three session modes against the same canvas:
 *   Label    place keypoints on raw video to build the first training set
 *   Correct  edit the model's predictions and save them as corrections
 *   Refine   pick corrected frames to add to the training set
 *
 * Everything the page draws is a "layer": the manual labels of this
 * session, labels committed earlier, DLC predictions, saved corrections,
 * and each MediaPipe pass.  Layers share one shape, so the canvas, the
 * timeline and the distance trace all read them the same way.
 */


const labeler = (() => {

// ── Constants ────────────────────────────────────────────────────────────

const HIT_RADIUS = 12;          // px — how close a click counts as "on a point"
const DRAG_THRESHOLD = 4;       // px — below this a mousedown+up is a click, not a pan
const PREFETCH_AHEAD = 4;       // frames to warm the JPEG cache with
const SPEED_PRESETS = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8];

const HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
];

const MANUAL_COLOR = '#ffffff';
const MP_COLOR = '#00cccc';

// ── State ────────────────────────────────────────────────────────────────

let subjectId = null, sessionId = null;
let subject = null, trials = [], totalFrames = 0;
let bodyparts = [], cameraNames = ['OS', 'OD'], cameraMode = 'stereo';
let hasCalibration = false;

let currentFrame = 0, currentSide = 'OS', currentTrialIdx = -1;
let videoTrialLoaded = -1, videoSideLoaded = null;

// Manual labels of this session: "frame_side" -> {bodypart: [x, y]}
let manual = new Map();
const dirtyKeys = new Set();
const deletedKeys = new Set();

// The MediaPipe forward pass for the loaded trial.  One pass, because
// one is what the app runs.
let mpTrial = null;          // {trial_idx, start_frame, os, od, distances}
let showMp = false;          // draw the landmarks; the zoom uses them regardless

// Canvas / view
let canvas, ctx, containerEl, vp2d, videoEl;
let imgW = 0, imgH = 0, scale = 1, offsetX = 0, offsetY = 0;
let hasUserZoom = false;
let currentImage = null, videoFrameMode = false;
const imageCache = new Map();

// Interaction
let dragging = null, dragStartX = 0, dragStartY = 0, dragOrigX = 0, dragOrigY = 0;
const undoStack = [], redoStack = [];

// Playback
let playing = false, videoPlaying = false, playbackRate = 1;

// Crop box editing

// Distances for this session's labels, returned by the save endpoint.
let manualDistances = {};

// Guided labeling: the k-means selection, and where we are in it.
let selection = null;        // {frames:[{global_frame, trial_idx, crops}], ...}
let bodypartJoints = {};     // bodypart -> MediaPipe landmark index
let guidedTarget = null;     // {frameIdx, side, bp} — the point being asked for
let guidedBusy = false;      // true while the view is moving to a point
let offFrame = false;        // MediaPipe's guess for this point is out of frame
let autoAdvance = false;     // move to the next point as soon as one is placed

let traceCanvas, traceCtx;
let frameDisplayMode = 0;    // 0 frame, 1 time, 2 both

// ── Small helpers ────────────────────────────────────────────────────────

const el = (id) => document.getElementById(id);

async function api(url, options) {
    const r = await fetch(url, options);
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* not JSON */ }
        throw new Error(detail);
    }
    return r.status === 204 ? null : r.json();
}

let toastTimer = null;
function toast(msg, kind) {
    let box = el('toastBox');
    if (!box) {
        box = document.createElement('div');
        box.id = 'toastBox';
        document.body.appendChild(box);
    }
    box.className = 'toast' + (kind ? ' ' + kind : '');
    box.textContent = msg;
    box.style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { box.style.display = 'none'; },
                            kind === 'err' ? 7000 : 3000);
}

/* Decode the server's {shape, b64} float32 payload into a typed array. */
function decodeF32(packed) {
    if (!packed || !packed.b64) return null;
    const bin = atob(packed.b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return { data: new Float32Array(bytes.buffer), shape: packed.shape };
}

function trialForFrame(frame) {
    for (let i = 0; i < trials.length; i++) {
        if (frame >= trials[i].start_frame && frame <= trials[i].end_frame) return i;
    }
    return trials.length ? trials.length - 1 : 0;
}

function sideIndex(side) {
    const i = cameraNames.indexOf(side);
    return i < 0 ? 0 : i;
}

function otherSide() {
    return cameraNames.length > 1
        ? cameraNames[1 - sideIndex(currentSide)] : currentSide;
}

// ── Layer access ─────────────────────────────────────────────────────────

/* Manual label for a frame/side/bodypart in this session, or null. */
function manualAt(frame, side, bp) {
    const lbl = manual.get(`${frame}_${side}`);
    const c = lbl && lbl[bp];
    return (c && c[0] != null) ? c : null;
}

/* 21x2 landmarks for one frame and camera, as a flat Float32Array view. */
function mpFrame2d(frame, side) {
    if (!mpTrial) return null;
    const arr = side === cameraNames[0] ? mpTrial.os : mpTrial.od;
    if (!arr) return null;
    const local = frame - mpTrial.start_frame;
    const [n, j] = arr.shape;
    if (local < 0 || local >= n) return null;
    return arr.data.subarray(local * j * 2, (local + 1) * j * 2);
}

/* Every finite MediaPipe landmark for a frame and camera.
 *
 * Read whether or not the landmarks are drawn: the guided zoom aims at
 * them, and hiding them is about what you look at, not what the page
 * knows. */
function mpPointsFor(frame, side) {
    const lm = mpFrame2d(frame, side);
    if (!lm) return [];
    const pts = [];
    for (let j = 0; j < lm.length / 2; j++) {
        if (Number.isFinite(lm[j * 2])) pts.push([lm[j * 2], lm[j * 2 + 1]]);
    }
    return pts;
}

function centroid(pts) {
    if (!pts.length) return null;
    let sx = 0, sy = 0;
    for (const [x, y] of pts) { sx += x; sy += y; }
    return [sx / pts.length, sy / pts.length];
}

/* How far the hand moves between the two cameras on this frame.
 *
 * Both ends have to be measured the same way or the "disparity" is really
 * the difference between two definitions of where the hand is.  That is
 * what breaks the moment you label one camera: a single thumb label on
 * OS against a 21-landmark MediaPipe centroid on OD is a comparison of a
 * fingertip with the middle of a palm, and the view jumps.
 *
 * So: use labels only for the bodyparts that exist on *both* cameras,
 * and otherwise use MediaPipe on both.  Returns [dx, dy] or null. */
function cameraShift(frame, from, to) {
    const paired = [];
    for (const bp of bodyparts) {
        const a = manualAt(frame, from, bp);
        const b = manualAt(frame, to, bp);
        if (a && b) paired.push([a, b]);
    }
    if (paired.length) {
        const a = centroid(paired.map(p => p[0]));
        const b = centroid(paired.map(p => p[1]));
        return [b[0] - a[0], b[1] - a[1]];
    }
    const a = centroid(mpPointsFor(frame, from));
    const b = centroid(mpPointsFor(frame, to));
    return (a && b) ? [b[0] - a[0], b[1] - a[1]] : null;
}

// ── Session loading ──────────────────────────────────────────────────────

async function openSubject(id) {
    subjectId = id;
    playing = false;
    stopVideoPlayback();
    await openSession();
}

async function openSession() {
    let session;
    try {
        session = await api(`/api/labeling/${subjectId}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_type: 'initial' }),
        });
    } catch (e) {
        toast(`Could not open a session: ${e.message}`, 'err');
        return;
    }
    sessionId = session.id;

    manual.clear();
    dirtyKeys.clear();
    deletedKeys.clear();
    undoStack.length = 0;
    redoStack.length = 0;
    imageCache.clear();
    currentTrialIdx = -1;
    videoTrialLoaded = -1;
    videoSideLoaded = null;
    mpTrial = null;

    const info = await api(`/api/labeling/sessions/${sessionId}/info`);
    subject = info.subject;
    trials = info.trials || [];
    totalFrames = info.total_frames || 0;
    bodyparts = info.bodyparts || [];
    cameraNames = info.camera_names || ['OS', 'OD'];
    cameraMode = info.camera_mode || 'stereo';
    hasCalibration = !!info.has_calibration;
    bodypartJoints = info.bodypart_joints || {};
    selection = null;
    guidedTarget = null;

    if (!cameraNames.includes(currentSide)) currentSide = cameraNames[0];
    updateSideToggle();

    el('committedCount').textContent =
        `${info.committed_frame_count || 0} frames committed to DLC`;
    el('calibNote').textContent = hasCalibration
        ? 'Stereo calibration loaded — distances in mm'
        : 'No stereo calibration for this subject — 2D only, distances in pixels';
    el('calibNote').style.color = hasCalibration ? '' : 'var(--orange)';

    const labels = await api(`/api/labeling/sessions/${sessionId}/labels`);
    for (const l of labels) {
        manual.set(`${l.frame_num}_${l.side}`, { ...l.keypoints });
    }

    updateLabelCount();
    updateShortcuts();

    // The selection is the page: without it there is nothing to label,
    // so it is loaded (or offered) before anything is drawn.
    await enterGuided();
    renderTrialButtons(trials, trialForFrame(currentFrame), pickTrial);
}

/* Fetch the MediaPipe forward pass for one trial. */
async function loadMpTrial(trialIdx) {
    if (mpTrial && mpTrial.trial_idx === trialIdx) return;
    try {
        const d = await api(`/api/labeling/sessions/${sessionId}`
            + `/trial/${trialIdx}/mp`);
        mpTrial = {
            trial_idx: trialIdx,
            start_frame: d.start_frame || 0,
            frame_count: d.frame_count || 0,
            os: decodeF32(d.OS),
            od: decodeF32(d.OD),
            distances: d.distances || null,
        };
    } catch (e) {
        console.warn('MediaPipe trial load failed', e);
        mpTrial = { trial_idx: trialIdx, start_frame: 0, os: null, od: null };
    }
}

// ── Frame navigation ─────────────────────────────────────────────────────

function frameUrl(frame, side) {
    return `/api/labeling/sessions/${sessionId}/frame?n=${frame}`
        + `&side=${encodeURIComponent(side)}`;
}

function loadImage(frame, side) {
    return new Promise((resolve, reject) => {
        const key = `${frame}_${side}`;
        if (imageCache.has(key)) { resolve(imageCache.get(key)); return; }
        const img = new Image();
        img.onload = () => {
            imageCache.set(key, img);
            if (imageCache.size > 40) imageCache.delete(imageCache.keys().next().value);
            resolve(img);
        };
        img.onerror = reject;
        img.src = frameUrl(frame, side);
    });
}

function prefetch(frame) {
    for (let i = 1; i <= PREFETCH_AHEAD; i++) {
        const f = frame + i;
        if (f < totalFrames) loadImage(f, currentSide).catch(() => {});
    }
}

/* Paint one frame straight from the <video> element.
 *
 * This is the path that makes stepping and playback feel immediate, and
 * it is also the more correct one: OpenCV's frame seeking is off by one
 * on many H.264 files, while the browser's decoder is the same decoder
 * that plays the video, so the drawn frame and the overlay always agree.
 * Returns false if the video isn't usable, and the JPEG path takes over. */
async function renderVideoFrame(frame) {
    if (!videoEl || playing) return false;

    const trialIdx = trialForFrame(frame);
    const trial = trials[trialIdx];
    if (!trial) return false;
    // A labeling package is a folder of images; there is no video to
    // decode, so don't ask for one and wait for the 404.
    if (trial.kind === 'frames') return false;

    // A stereo file holds both cameras, so switching camera is a different
    // crop of the same decode — reloading the element for it would throw
    // away the buffered video and re-seek, which guided labeling does on
    // every other point.  Only multicam has a different file per camera.
    const needsNewFile = videoTrialLoaded !== trialIdx
        || (cameraMode === 'multicam' && videoSideLoaded !== currentSide);

    let justLoaded = false;
    if (needsNewFile) {
        videoEl.src = `/api/labeling/sessions/${sessionId}/video?trial=${trialIdx}`
            + `&side=${encodeURIComponent(currentSide)}&_=${Date.now()}`;
        videoTrialLoaded = trialIdx;
        videoSideLoaded = currentSide;
        const ok = await new Promise(resolve => {
            if (videoEl.readyState >= 1) { resolve(true); return; }
            const timer = setTimeout(() => resolve(false), 5000);
            const onLoaded = () => { clearTimeout(timer); resolve(true); };
            const onError = () => { clearTimeout(timer); resolve(false); };
            videoEl.addEventListener('loadedmetadata', onLoaded, { once: true });
            videoEl.addEventListener('error', onError, { once: true });
        });
        if (!ok) return false;
        justLoaded = true;
    }
    if (videoEl.readyState < 1) return false;

    // Seek to the middle of the target frame's duration.  Landing exactly on
    // a boundary lets keyframe rounding and float error pick the neighbour.
    // frame_offset covers videos whose container exposes pre-roll frames to
    // OpenCV that the browser skips, so the two agree on frame numbering.
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

    drawVideoToCanvas(frame);
    videoFrameMode = true;
    return true;
}

/* Source rectangle for the current camera. A stereo file holds both
 * cameras side by side, so each view is one half of the decoded frame. */
function cameraSourceRect(vw) {
    if (cameraMode !== 'stereo') return { sx: 0, sw: vw };
    const midline = Math.floor(vw / 2);
    if (cameraNames.length >= 2 && currentSide === cameraNames[1]) {
        return { sx: midline, sw: vw - midline };
    }
    return { sx: 0, sw: midline };
}

function drawVideoToCanvas(frame, { rezoom = true } = {}) {
    sizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
    if (vw > 0 && vh > 0) {
        const { sx, sw } = cameraSourceRect(vw);
        const firstSize = !imgW || !imgH;
        imgW = sw; imgH = vh;
        if (rezoom) applyAutoZoom(frame);
        else if (firstSize && !hasUserZoom) fitImage();
        ctx.save();
        ctx.translate(offsetX, offsetY);
        ctx.scale(scale, scale);
        ctx.drawImage(videoEl, sx, 0, sw, vh, 0, 0, sw, vh);
        ctx.restore();
    }
    drawOverlays();
}

async function gotoFrame(frame, opts = {}) {
    if (frame < 0 || frame >= totalFrames) return;
    currentFrame = frame;
    const trialIdx = trialForFrame(frame);
    const trialChanged = trialIdx !== currentTrialIdx;
    currentTrialIdx = trialIdx;
    setNavState({ frame, trialIdx, side: currentSide, subjectId });

    if (trialChanged || opts.force || !mpTrial || mpTrial.trial_idx !== trialIdx) {
        await loadMpTrial(trialIdx);
        renderTrialButtons(trials, trialIdx, pickTrial);
    }

    const fromVideo = await renderVideoFrame(frame);
    if (!fromVideo) {
        videoFrameMode = false;
        try {
            currentImage = await loadImage(frame, currentSide);
            imgW = currentImage.width;
            imgH = currentImage.height;
            applyAutoZoom(frame);
            render();
            prefetch(frame);
        } catch (e) {
            console.error('Could not load frame', frame, e);
        }
    }

    updateFrameDisplay();
    renderPointGrid();
    updateTargetBanner();
    renderTrace();
}

function pickTrial(trialIdx) {
    const t = trials[trialIdx];
    if (t) gotoFrame(t.start_frame);
}

// ── Zoom ─────────────────────────────────────────────────────────────────

function sizeCanvas() {
    const cw = vp2d.clientWidth, ch = vp2d.clientHeight;
    if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
    }
}

function fitImage() {
    if (!imgW || !imgH) return;
    const cw = vp2d.clientWidth, ch = vp2d.clientHeight;
    scale = Math.min(cw / imgW, ch / imgH);
    offsetX = (cw - imgW * scale) / 2;
    offsetY = (ch - imgH * scale) / 2;
}

function zoomToBox(minX, minY, maxX, maxY, pad) {
    minX -= pad; minY -= pad; maxX += pad; maxY += pad;
    const cw = vp2d.clientWidth, ch = vp2d.clientHeight;
    const w = Math.max(maxX - minX, 1), h = Math.max(maxY - minY, 1);
    scale = Math.min(cw / w, ch / h);
    offsetX = (cw - w * scale) / 2 - minX * scale;
    offsetY = (ch - h * scale) / 2 - minY * scale;
}

/* Frame the view on the hand, for the odd frame the guided zoom has not
 * aimed at — the first load, and anywhere you arrow to with no zoom of
 * your own.  Labels first, then the whole MediaPipe hand. */
function applyAutoZoom(frame) {
    if (hasUserZoom) return;

    let pts = [];
    for (const bp of bodyparts) {
        const c = manualAt(frame, currentSide, bp);
        if (c) pts.push(c);
    }
    if (!pts.length) pts = mpPointsFor(frame, currentSide);
    if (!pts.length) { fitImage(); return; }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of pts) {
        minX = Math.min(minX, x); minY = Math.min(minY, y);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    const span = Math.max(maxX - minX, maxY - minY, 20);
    zoomToBox(minX, minY, maxX, maxY, span * 0.8 + 40);
}

function resetZoom() {
    hasUserZoom = false;
    if (videoFrameMode) drawVideoToCanvas(currentFrame);
    else { applyAutoZoom(currentFrame); render(); }
}

function screenToImage(sx, sy) {
    return { x: (sx - offsetX) / scale, y: (sy - offsetY) / scale };
}

function imageToScreen(ix, iy) {
    return { x: ix * scale + offsetX, y: iy * scale + offsetY };
}

// ── 2D rendering ─────────────────────────────────────────────────────────

function render() {
    if (videoFrameMode) { drawVideoToCanvas(currentFrame); return; }
    sizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (currentImage) {
        ctx.save();
        ctx.translate(offsetX, offsetY);
        ctx.scale(scale, scale);
        ctx.drawImage(currentImage, 0, 0);
        ctx.restore();
    }
    drawOverlays();
}

function drawOverlays() {
    drawMpOverlays();
    drawManualOverlays();
}

/* This session's labels, always drawn — they are the work. */
function drawManualOverlays() {
    for (const bp of bodyparts) {
        const c = manualAt(currentFrame, currentSide, bp);
        if (c) drawPoint(c[0], c[1], MANUAL_COLOR, bp[0].toUpperCase(), true);
    }
}

function drawMpOverlays() {
    if (!showMp) return;
    const lm = mpFrame2d(currentFrame, currentSide);
    if (!lm) return;

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = MP_COLOR;
    ctx.globalAlpha = 0.65;
    for (const [a, b] of HAND_BONES) {
        if (!Number.isFinite(lm[a * 2]) || !Number.isFinite(lm[b * 2])) continue;
        const p = imageToScreen(lm[a * 2], lm[a * 2 + 1]);
        const q = imageToScreen(lm[b * 2], lm[b * 2 + 1]);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
        ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = MP_COLOR;
    for (let j = 0; j < lm.length / 2; j++) {
        if (!Number.isFinite(lm[j * 2])) continue;
        const p = imageToScreen(lm[j * 2], lm[j * 2 + 1]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
    }
}

function drawPoint(ix, iy, color, letter, solid) {
    const p = imageToScreen(ix, iy);
    const r = 7;
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(p.x - r, p.y); ctx.lineTo(p.x + r, p.y);
    ctx.moveTo(p.x, p.y - r); ctx.lineTo(p.x, p.y + r);
    ctx.stroke();
    if (solid) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.fillStyle = color;
    ctx.font = '600 10px -apple-system, sans-serif';
    ctx.fillText(letter, p.x + r + 2, p.y - 3);
}

// ── Crop box editing ─────────────────────────────────────────────────────

// ── Label editing ────────────────────────────────────────────────────────

function hitTest(sx, sy) {
    for (const bp of bodyparts) {
        const c = manualAt(currentFrame, currentSide, bp);
        if (!c) continue;
        const p = imageToScreen(c[0], c[1]);
        if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) return { bp };
    }
    return null;
}

function setManual(bp, x, y) {
    const key = `${currentFrame}_${currentSide}`;
    let lbl = manual.get(key);
    if (!lbl) { lbl = {}; manual.set(key, lbl); }
    lbl[bp] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
    dirtyKeys.add(key);
    deletedKeys.delete(key);
}

function pushUndo(key, bp, prev) {
    undoStack.push({ key, bp, prev: prev ? [...prev] : null });
    if (undoStack.length > 200) undoStack.shift();
    redoStack.length = 0;
}

/* Place the point the mode is asking for.
 *
 * Only that one: every labeled frame here is a k-means pick, and a label
 * dropped on a neighbouring frame would never be saved but would still
 * look placed. Refusing is clearer than silently keeping it. */
function placeLabel(ix, iy) {
    if (!guidedTarget || !onTargetPoint()) {
        setLabelInfo(onTargetFrame()
            ? 'Switch camera or press N — this is not the point being asked for.'
            : 'Not the frame being labeled — press B to go back to it.');
        return;
    }

    const key = `${currentFrame}_${currentSide}`;
    const lbl = manual.get(key) || {};
    const bp = guidedTarget.bp;

    pushUndo(key, bp, lbl[bp] || null);
    setManual(bp, ix, iy);

    render();
    updateLabelCount();
    scheduleSave();
    renderPointGrid();

    if (autoAdvance) {
        goToNextPoint();
    } else {
        // The next point is usually the same bodypart on the other
        // camera, so naming it beats "place the next one".
        const next = nextUnlabeledPoint(currentPointIndex());
        setLabelInfo(`${bp} (${currentSide}) placed`
            + (next ? ` — N for ${next.bp} (${next.side}), `
                      + `frame ${next.frame.global_frame}`
                    : ' — that was the last one'));
    }
}

/* True when the view is on the exact point the mode is asking for. */
function onTargetPoint() {
    if (!guidedTarget || !selection) return false;
    const f = (selection.frames || [])[guidedTarget.frameIdx];
    return !!f && currentFrame === f.global_frame
        && currentSide === guidedTarget.side;
}

function removeLabel(bp) {
    const key = `${currentFrame}_${currentSide}`;
    const lbl = manual.get(key);
    if (!lbl || !lbl[bp] || lbl[bp][0] == null) return;

    pushUndo(key, bp, lbl[bp]);
    delete lbl[bp];
    if (!bodyparts.some(b => lbl[b] && lbl[b][0] != null)) {
        manual.delete(key);
        deletedKeys.add(key);
        dirtyKeys.delete(key);
    } else {
        dirtyKeys.add(key);
    }
    render();
    updateLabelCount();
    scheduleSave();
    renderPointGrid();
}

function applyUndoEntry(entry, toStack) {
    const { key, bp, prev } = entry;
    const lbl = manual.get(key) || {};
    const current = (lbl[bp] && lbl[bp][0] != null) ? [...lbl[bp]] : null;
    toStack.push({ key, bp, prev: current });

    if (prev) {
        if (!manual.has(key)) manual.set(key, {});
        manual.get(key)[bp] = [...prev];
        dirtyKeys.add(key);
        deletedKeys.delete(key);
    } else if (manual.has(key)) {
        delete manual.get(key)[bp];
        const rest = manual.get(key);
        if (!bodyparts.some(b => rest[b] && rest[b][0] != null)) {
            manual.delete(key);
            deletedKeys.add(key);
            dirtyKeys.delete(key);
        } else {
            dirtyKeys.add(key);
        }
    }

    // Undo is only useful if it also takes you to the frame it changed.
    const [f, ...sideParts] = key.split('_');
    const frame = parseInt(f, 10);
    const side = sideParts.join('_');
    if (frame !== currentFrame || side !== currentSide) {
        currentSide = side;
        gotoFrame(frame);
    } else {
        render();
    }
    updateLabelCount();
    scheduleSave();
}

function afterUndoRedo() {
    renderPointGrid();
    updateTargetBanner();
}

function undo() {
    const e = undoStack.pop();
    if (e) applyUndoEntry(e, redoStack);
    afterUndoRedo();
}

function redo() {
    const e = redoStack.pop();
    if (e) applyUndoEntry(e, undoStack);
    afterUndoRedo();
}

// ── Saving ───────────────────────────────────────────────────────────────

let saveTimer = null, saveInFlight = false, saveQueued = false;

function setSaved(state) {
    const pill = el('savePill');
    pill.className = 'status-pill ' + state;
    pill.textContent = state === 'dirty' ? 'unsaved'
        : state === 'saving' ? 'saving…' : 'saved';
}

function scheduleSave() {
    setSaved('dirty');
    clearTimeout(saveTimer);
    saveTimer = setTimeout(flushSave, 400);
}

async function flushSave() {
    if (saveInFlight) { saveQueued = true; return; }
    if (!dirtyKeys.size && !deletedKeys.size) { setSaved('saved'); return; }

    saveInFlight = true;
    setSaved('saving');

    const keys = [...dirtyKeys];
    const dels = [...deletedKeys];
    dirtyKeys.clear();
    deletedKeys.clear();

    try {
        if (keys.length) {
            const payload = keys.map(key => {
                const [f, ...sp] = key.split('_');
                const frame = parseInt(f, 10);
                return {
                    frame_num: frame,
                    trial_idx: trialForFrame(frame),
                    side: sp.join('_'),
                    keypoints: manual.get(key) || {},
                };
            });
            const r = await api(`/api/labeling/sessions/${sessionId}/labels`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ labels: payload }),
            });
            // The server returns the recomputed distance for each frame it
            // touched; folding it into the manual layer keeps the trace
            // honest without refetching the whole layer.
            manualDistances = { ...manualDistances, ...(r.updated_distances || {}) };
            renderTrace();
        }
        for (const key of dels) {
            const [f, ...sp] = key.split('_');
            await api(`/api/labeling/sessions/${sessionId}/labels/${parseInt(f, 10)}`
                + `?side=${encodeURIComponent(sp.join('_'))}`, { method: 'DELETE' });
        }
        setSaved('saved');
    } catch (e) {
        // Put the keys back so the next attempt retries them rather than
        // dropping the user's work on a transient failure.
        keys.forEach(k => dirtyKeys.add(k));
        dels.forEach(k => deletedKeys.add(k));
        setSaved('dirty');
        toast(`Save failed: ${e.message}`, 'err');
    } finally {
        saveInFlight = false;
        if (saveQueued) { saveQueued = false; flushSave(); }
    }
}

async function commit() {
    await flushSave();

    const n = manual.size;
    if (!n) { toast('Nothing to save — place some labels first', 'err'); return; }
    const frames = (selection && selection.frames.length) || 0;
    if (!confirm(`Save ${frames} labeled frame(s) as DeepLabCut training data?`))
        return;
    try {
        const r = await api(`/api/labeling/sessions/${sessionId}/commit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ train_frames: [] }),
        });
        toast(`Saved ${r.frame_count} frame(s) to `
            + `${(r.labeled_data_dir || '').split('/').slice(-2).join('/')}.`, 'ok');
        await openSession();
    } catch (e) {
        toast(`Save failed: ${e.message}`, 'err');
    }
}

// ── Playback ─────────────────────────────────────────────────────────────

function togglePlay() {
    playing = !playing;
    el('playBtn').innerHTML = playing ? '&#10073;&#10073;' : '&#9654;';
    if (playing) startPlayback(); else stopVideoPlayback();
}

async function startPlayback() {
    playbackRate = SPEED_PRESETS[parseInt(el('speedSlider').value, 10)] || 1;
    if (!videoEl) { fallbackPlay(); return; }

    const trialIdx = trialForFrame(currentFrame);
    const trial = trials[trialIdx];
    if (!trial) return;
    currentTrialIdx = trialIdx;
    // No video behind a package's frames — step them instead.
    if (trial.kind === 'frames') { fallbackPlay(trial.fps); return; }

    const frameOffset = trial.frame_offset || 0;
    const startTime = Math.max(0,
        (currentFrame - trial.start_frame - frameOffset + 0.5) / trial.fps);

    if (videoTrialLoaded !== trialIdx
        || (cameraMode === 'multicam' && videoSideLoaded !== currentSide)) {
        videoEl.src = `/api/labeling/sessions/${sessionId}/video?trial=${trialIdx}`
            + `&side=${encodeURIComponent(currentSide)}&_=${Date.now()}`;
        videoTrialLoaded = trialIdx;
        videoSideLoaded = currentSide;
        const ready = await new Promise(resolve => {
            if (videoEl.readyState >= 3) { resolve(true); return; }
            const timer = setTimeout(() => resolve(false), 8000);
            const onReady = () => { clearTimeout(timer); resolve(true); };
            const onError = () => { clearTimeout(timer); resolve(false); };
            videoEl.addEventListener('canplay', onReady, { once: true });
            videoEl.addEventListener('error', onError, { once: true });
        });
        if (!ready) { if (playing) fallbackPlay(trial.fps); return; }
        if (!playing) return;
    }

    // Very slow rates aren't supported by every decoder; stepping frames
    // by hand covers those instead of silently playing at 1x.
    try {
        videoEl.playbackRate = playbackRate;
    } catch (e) {
        videoPlaying = false;
        fallbackPlay(trial.fps);
        return;
    }

    videoEl.currentTime = startTime;
    videoPlaying = true;
    try {
        await videoEl.play();
        // requestVideoFrameCallback fires once per *painted* frame, so the
        // overlay can never be drawn for a frame the viewer isn't seeing.
        if ('requestVideoFrameCallback' in videoEl) {
            videoEl.requestVideoFrameCallback(videoDrawLoop);
        } else {
            requestAnimationFrame(videoDrawLoop);
        }
    } catch (e) {
        videoPlaying = false;
        fallbackPlay(trial.fps);
    }

    videoEl.onended = () => {
        const next = trialIdx + 1;
        if (next < trials.length && playing) {
            currentFrame = trials[next].start_frame;
            startPlayback();
        } else {
            playing = false;
            videoPlaying = false;
            el('playBtn').innerHTML = '&#9654;';
            gotoFrame(currentFrame);
        }
    };
}

function videoDrawLoop(now, metadata) {
    if (!videoPlaying || !playing) return;
    const trial = trials[currentTrialIdx];
    if (!trial) return;

    // mediaTime is the timestamp of the frame actually on screen; currentTime
    // can already have advanced past it.
    const mediaTime = (metadata && metadata.mediaTime != null)
        ? metadata.mediaTime : videoEl.currentTime;
    const frameOffset = trial.frame_offset || 0;
    const local = Math.floor(mediaTime * trial.fps) + frameOffset;
    currentFrame = trial.start_frame + Math.min(local, trial.frame_count - 1);

    drawVideoToCanvas(currentFrame, { rezoom: false });
    updateFrameDisplay();

    // The trace is context, not the thing being watched; refreshing it
    // every frame at 60 fps costs more than it adds.
    if (currentFrame % 10 === 0) renderTrace();

    if ('requestVideoFrameCallback' in videoEl) {
        videoEl.requestVideoFrameCallback(videoDrawLoop);
    } else {
        requestAnimationFrame(videoDrawLoop);
    }
}

function stopVideoPlayback() {
    const wasVideo = videoPlaying;
    videoPlaying = false;
    if (videoEl) {
        videoEl.pause();
        const trial = trials[currentTrialIdx];
        if (wasVideo && trial) {
            const frameOffset = trial.frame_offset || 0;
            const local = Math.round(videoEl.currentTime * trial.fps) + frameOffset;
            currentFrame = trial.start_frame
                + Math.min(Math.max(0, local), trial.frame_count - 1);
        }
    }
    if (sessionId) gotoFrame(currentFrame);
}

function fallbackPlay(fpsOverride) {
    const trial = trials[trialForFrame(currentFrame)];
    const fps = fpsOverride || (trial ? trial.fps : 30);
    const interval = 1000 / (fps * playbackRate);
    (async function loop() {
        while (playing && currentFrame < totalFrames - 1) {
            const t0 = performance.now();
            await gotoFrame(currentFrame + 1);
            const wait = Math.max(0, interval - (performance.now() - t0));
            await new Promise(r => setTimeout(r, wait));
        }
        if (playing) {
            playing = false;
            el('playBtn').innerHTML = '&#9654;';
        }
    })();
}

// ── Guided labeling ──────────────────────────────────────────────────────
//
// A labeling set is 30 frames x 2 cameras x 2 bodyparts = 120 points, and
// the slow part is not placing them, it is finding them: scrolling to the
// right frame, switching camera, zooming in. So the page does that part.
// One key takes you to the next point nobody has placed, framed on
// MediaPipe's guess at it, and the grid shows what is left.

/* Every point to place, in the order the mode walks them.
 *
 * Frame-major, then camera, then bodypart: it finishes a frame before
 * moving on, so each frame is decoded twice rather than once per point,
 * and the two camera hops are the cheap ones now that the view carries
 * across the switch. */
function guidedPoints() {
    if (!selection) return [];
    const out = [];
    selection.frames.forEach((f, frameIdx) => {
        for (const row of guidedRows()) {
            out.push({ frameIdx, side: row.side, bp: row.bp, frame: f });
        }
    });
    return out;
}

function pointDone(pt) {
    const c = manualAt(pt.frame.global_frame, pt.side, pt.bp);
    return !!(c && c[0] != null);
}

/* The points of one frame, in the order the mode asks for them.
 *
 * Bodypart first, then camera: placing the same landmark on both cameras
 * back to back is the pairing you can actually check by eye, and it is
 * the one the camera-switch alignment is built for. */
function guidedRows() {
    const cams = cameraNames.slice(0, singleCamera() ? 1 : 2);
    const rows = [];
    for (const bp of bodyparts) for (const side of cams) rows.push({ side, bp });
    return rows;
}

/* MediaPipe's guess at one bodypart on one frame and camera. */
function mpTargetFor(frame, side, bp) {
    const joint = bodypartJoints[bp];
    if (joint == null) return null;
    const lm = mpFrame2d(frame, side);
    if (!lm || lm.length < (joint + 1) * 2) return null;
    const x = lm[joint * 2], y = lm[joint * 2 + 1];
    return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

/* Frame the view on the point being asked for.
 *
 * Tight enough that placing it is a single click rather than a zoom and
 * a click, wide enough to see which finger you are looking at. Falls
 * back to the stored crop, then to the usual auto-zoom, so a frame
 * MediaPipe missed is still labelable — just not pre-aimed. */
function zoomToPoint(pt) {
    const raw = mpTargetFor(pt.frame.global_frame, pt.side, pt.bp);
    if (raw) {
        // MediaPipe extrapolates tips it cannot see, so its guess can sit
        // outside the frame.  Centring there would put the whole image
        // off-screen and silently reject every click, since a label
        // outside the frame is not a label.  Clamp into the image and say
        // so, because "the hand is leaving frame" is worth knowing.
        const inset = 4;
        const target = [
            Math.min(Math.max(raw[0], inset), Math.max(imgW - inset, inset)),
            Math.min(Math.max(raw[1], inset), Math.max(imgH - inset, inset)),
        ];
        offFrame = Math.hypot(target[0] - raw[0], target[1] - raw[1]) > 1;
        const box = (pt.frame.crops || {})[pt.side];
        const span = box ? Math.max(box.side * 0.07, 15) : 22;
        zoomToBox(target[0] - span, target[1] - span,
                  target[0] + span, target[1] + span, 0);
        hasUserZoom = true;
        return true;
    }
    offFrame = false;
    const box = (pt.frame.crops || {})[pt.side];
    if (box) {
        zoomToBox(box.x0, box.y0, box.x0 + box.side, box.y0 + box.side, 10);
        hasUserZoom = true;
        return true;
    }
    return false;
}

async function goToPoint(pt) {
    if (!pt) return;
    // Clicks are ignored until the new frame is up and aimed.  Without
    // this, a click that lands mid-seek is placed on the point we were
    // just on — which at labeling speed means quietly overwriting the
    // point you only just finished.
    guidedBusy = true;
    try {
        guidedTarget = { frameIdx: pt.frameIdx, side: pt.side, bp: pt.bp };
        if (pt.side !== currentSide) {
            currentSide = pt.side;
            updateSideToggle();
            if (cameraMode === 'multicam') videoSideLoaded = null;
        }
        hasUserZoom = false;          // let gotoFrame load, then aim
        await gotoFrame(pt.frame.global_frame, { force: true });
        if (zoomToPoint(pt)) render();
        renderPointGrid();
        setLabelInfo(`Place ${pt.bp} (${pt.side}) — frame ${pt.frame.global_frame}`
            + (offFrame ? ' · MediaPipe puts it outside the frame; place it '
                        + 'where you see it, or press N to skip' : ''));
        updateTargetBanner();
    } catch (e) {
        // Never leave the guard latched: a stuck guard silently swallows
        // every later click, which looks exactly like the page freezing.
        console.error('Could not move to the next point', e);
        setLabelInfo('Could not load that frame — N tries the next one.');
    } finally {
        guidedBusy = false;
    }
}

/* The frame the mode is currently asking about, or null. */
function targetFrame() {
    if (!guidedTarget || !selection) return null;
    const f = (selection.frames || [])[guidedTarget.frameIdx];
    return f ? f.global_frame : null;
}

function onTargetFrame() {
    const t = targetFrame();
    return t == null || currentFrame === t;
}

/* Show whether what is on screen is the frame being asked about.
 *
 * Stepping away to watch how the point moves either side of the target is
 * a real part of judging where it goes, so it has to be allowed — but
 * labeling the neighbour by mistake is then one click away, and the only
 * thing standing between the two is knowing which frame you are on. */
function updateTargetBanner() {
    const away = !onTargetFrame();
    el('backToTargetBtn').style.display = away ? '' : 'none';
    el('vp2d').classList.toggle('off-target', away);
    const t = targetFrame();
    el('backToTargetBtn').textContent = `Back to frame ${t} (B)`;
    // The "you are not on the target" note has to go when you are, or it
    // reads as a complaint about the frame you just came back to.
    if (!away && /not the (frame|point)/i.test(el('labelInfo').textContent)) {
        setLabelInfo(guidedTarget
            ? `Place ${guidedTarget.bp} (${guidedTarget.side}) — frame ${t}`
            : '');
    }
}

/* Step to the next or previous frame in the selection. */
function stepSelectedFrame(dir) {
    const frames = (selection && selection.frames) || [];
    if (!frames.length) return;
    const here = currentFrame;
    const sorted = frames.map(f => f.global_frame).sort((a, b) => a - b);
    const next = dir > 0 ? sorted.find(f => f > here)
                         : [...sorted].reverse().find(f => f < here);
    if (next != null) gotoFrame(next);
}

/* Return to the target frame without disturbing the view.
 *
 * Deliberately keeps zoom and pan: you went looking at a neighbouring
 * frame to see the same patch of image, and re-aiming on the way back
 * would undo the comparison you just made. */
async function backToTarget() {
    const t = targetFrame();
    if (t == null || currentFrame === t) return;
    const pt = guidedPoints()[currentPointIndex()];
    if (pt && pt.side !== currentSide) {
        const shift = cameraShift(currentFrame, currentSide, pt.side);
        currentSide = pt.side;
        updateSideToggle();
        if (cameraMode === 'multicam') videoSideLoaded = null;
        if (shift) { offsetX -= shift[0] * scale; offsetY -= shift[1] * scale; }
    }
    hasUserZoom = true;               // hold the view exactly as it is
    await gotoFrame(t, { force: true });
}

function nextUnlabeledPoint(fromIdx) {
    const pts = guidedPoints();
    if (!pts.length) return null;
    const start = (fromIdx == null ? -1 : fromIdx) + 1;
    for (let i = start; i < pts.length; i++) if (!pointDone(pts[i])) return pts[i];
    // Wrap, so finishing the last point lands on whatever was skipped.
    for (let i = 0; i < Math.max(start, 0); i++) if (!pointDone(pts[i])) return pts[i];
    return null;
}

function currentPointIndex() {
    if (!guidedTarget) return null;
    const pts = guidedPoints();
    const i = pts.findIndex(p => p.frameIdx === guidedTarget.frameIdx
        && p.side === guidedTarget.side && p.bp === guidedTarget.bp);
    return i < 0 ? null : i;
}

async function goToNextPoint() {
    const pt = nextUnlabeledPoint(currentPointIndex());
    if (!pt) {
        setLabelInfo('Every point in the selection is placed.');
        toast('Every point in the selection is placed — Save when ready.', 'ok');
        return;
    }
    await goToPoint(pt);
}

function renderPointGrid() {
    const host = el('pointGrid');
    if (!host) return;
    const rows = guidedRows();
    const frames = (selection && selection.frames) || [];
    host.style.gridTemplateColumns = `auto repeat(${frames.length}, 15px)`;

    const cells = ['<span></span>'];
    frames.forEach((f, i) => cells.push(
        `<span class="chdr" title="frame ${f.global_frame}">${i + 1}</span>`));

    const curIdx = currentPointIndex();
    const pts = guidedPoints();
    for (const row of rows) {
        cells.push(`<span class="rlabel">${row.side} ${row.bp}</span>`);
        frames.forEach((f, frameIdx) => {
            const pt = { frameIdx, side: row.side, bp: row.bp, frame: f };
            const i = pts.findIndex(p => p.frameIdx === frameIdx
                && p.side === row.side && p.bp === row.bp);
            const done = pointDone(pt);
            cells.push(`<span class="cell${done ? ' done' : ''}`
                + `${i === curIdx ? ' current' : ''}" data-pt="${i}"`
                + ` title="frame ${f.global_frame} · ${row.side} ${row.bp}`
                + `${done ? ' · placed' : ''}"></span>`);
        });
    }
    host.innerHTML = cells.join('');
    host.querySelectorAll('[data-pt]').forEach(c =>
        c.addEventListener('click', () => {
            const pt = guidedPoints()[parseInt(c.dataset.pt, 10)];
            if (pt) goToPoint(pt);
        }));

    const total = pts.length;
    const done = pts.filter(pointDone).length;
    el('guidedStatus').textContent =
        `${done} / ${total} points placed · ${frames.length} frames selected`;
}

/* Open the subject for labeling.
 *
 * There is nothing to label without a selection, so this is where the
 * page either loads one or offers to make one. */
async function enterGuided() {
    await loadSelection();
    if (!selection) {
        el('guidedStatus').textContent = 'No frames selected for this subject yet.';
        setLabelInfo('');
        if (!confirm('No frames have been chosen for this subject yet.\n\n'
                   + 'Pick them now? k-means reads every trial once, so it '
                   + 'takes a minute or two.')) {
            // Show something rather than a blank canvas.
            const start = trials[0] ? trials[0].start_frame : 0;
            await gotoFrame(start, { force: true });
            return;
        }
        await runSelection();
        if (!selection) return;
    }
    renderPointGrid();
    await goToNextPoint();
}

async function loadSelection() {
    try {
        const d = await api(`/api/labeling/${subjectId}/selection`);
        selection = d.exists ? d : null;
    } catch (e) {
        selection = null;
    }
    return selection;
}

async function runSelection() {
    const n = parseInt(prompt(
        'How many frames should k-means pick for this subject?', '30') || '', 10);
    if (!Number.isFinite(n) || n < 1) return;
    el('guidedStatus').textContent = 'Selecting frames…';
    try {
        await api(`/api/labeling/${subjectId}/selection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_pick: n }),
        });
    } catch (e) {
        el('guidedStatus').textContent = '';
        toast(`Frame selection failed: ${e.message}`, 'err');
        return;
    }

    const done = await watchSelection();
    if (done.status === 'failed') {
        el('guidedStatus').textContent = '';
        toast(`Frame selection failed: ${done.error || 'unknown error'}`, 'err');
        return;
    }

    await loadSelection();
    if (!selection) {
        el('guidedStatus').textContent = '';
        toast('Frame selection produced nothing — are there MediaPipe files '
            + 'for this subject?', 'err');
        return;
    }
    toast(`Selected ${selection.frames.length} frames`, 'ok');
}

/* Poll until the selection finishes.
 *
 * The phase matters more than the number: clustering is seconds with
 * nothing to tick, and a percentage sitting still reads as a hang. */
async function watchSelection() {
    for (;;) {
        await new Promise(r => setTimeout(r, 700));
        let st;
        try {
            st = await api(`/api/labeling/${subjectId}/selection/progress`);
        } catch (e) { continue; }
        el('guidedStatus').textContent =
            `${st.phase || 'Selecting frames'} · ${Math.round(st.pct || 0)}%`;
        if (st.status === 'done' || st.status === 'failed') return st;
    }
}

/* Send these frames to somebody else to label.
 *
 * Writes the selection as a self-contained folder — images, the crop
 * geometry that puts the labels back, and instructions — so the person
 * labeling needs this app and nothing else. */
async function exportForLabeling() {
    if (!selection) { toast('Pick frames for this subject first', 'err'); return; }
    const name = prompt('Folder name for the package (blank for the default):',
                        '');
    if (name === null) return;

    const btn = el('exportBtn');
    btn.disabled = true;
    try {
        await api('/api/packages/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject: subject.name,
                                   package_name: name.trim() || null }),
        });
    } catch (e) {
        btn.disabled = false;
        toast(`Export failed: ${e.message}`, 'err');
        return;
    }

    for (;;) {
        await new Promise(r => setTimeout(r, 700));
        let st;
        try {
            st = await api('/api/packages/export/progress'
                + `?subject=${encodeURIComponent(subject.name)}`);
        } catch (e) { continue; }
        btn.textContent = `Exporting ${Math.round(st.pct || 0)}%`;
        if (st.status === 'done') {
            toast(`Wrote ${st.image_count} images to ${st.package_dir}`, 'ok');
            break;
        }
        if (st.status === 'failed') {
            toast(`Export failed: ${st.error || 'unknown error'}`, 'err');
            break;
        }
    }
    btn.disabled = false;
    btn.textContent = 'Export for labeling';
}

// ── Distance trace ───────────────────────────────────────────────────────

/* What the trace plots.
 *
 * Always the MediaPipe aperture for the loaded trial, whether or not the
 * landmarks are drawn: it is how you tell a frame worth labeling from
 * one where the hand is mid-blur, and that is useful before you have
 * decided to look at the landmarks themselves. */
function traceSeries() {
    const out = [];
    if (mpTrial && mpTrial.distances) {
        out.push({ label: 'MediaPipe', color: MP_COLOR,
                   values: mpTrial.distances, offset: mpTrial.start_frame });
    }
    const mine = Object.keys(manualDistances);
    if (mine.length) {
        const values = [];
        for (const [f, v] of Object.entries(manualDistances)) values[+f] = v;
        out.push({ label: 'This session', color: MANUAL_COLOR,
                   values, offset: 0 });
    }
    return out;
}

function renderTrace() {
    if (!traceCanvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = traceCanvas.clientWidth, h = traceCanvas.clientHeight;
    if (!w || !h) return;
    traceCanvas.width = w * dpr;
    traceCanvas.height = h * dpr;
    const c = traceCtx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);

    const trial = trials[currentTrialIdx];
    if (!trial) return;

    // One trial at a time: the aperture signal is only readable at that
    // scale, and it is the scale you label at.
    const f0 = trial.start_frame, f1 = trial.end_frame;
    const series = traceSeries();
    const label = el('traceLabel');

    if (!series.length) {
        label.textContent = 'Thumb-index distance — turn on a layer to plot it';
        return;
    }

    let lo = Infinity, hi = -Infinity;
    for (const s of series) {
        for (let f = f0; f <= f1; f++) {
            const v = s.values[f - s.offset];
            if (v == null) continue;
            lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
    }
    if (!Number.isFinite(lo)) {
        label.textContent = 'Thumb-index distance — no values in this trial';
        return;
    }
    const pad = Math.max((hi - lo) * 0.1, 1);
    lo -= pad; hi += pad;

    const x = (f) => ((f - f0) / Math.max(f1 - f0, 1)) * w;
    const y = (v) => h - ((v - lo) / Math.max(hi - lo, 1e-6)) * (h - 14) - 7;

    c.strokeStyle = 'rgba(255,255,255,0.07)';
    c.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const yy = 7 + (i / 4) * (h - 14);
        c.beginPath(); c.moveTo(0, yy); c.lineTo(w, yy); c.stroke();
    }

    for (const s of series) {
        c.strokeStyle = s.color;
        c.lineWidth = 1.3;
        c.beginPath();
        let open = false;
        for (let f = f0; f <= f1; f++) {
            const v = s.values[f - s.offset];
            if (v == null) { open = false; continue; }
            const px = x(f), py = y(v);
            if (open) c.lineTo(px, py); else { c.moveTo(px, py); open = true; }
        }
        c.stroke();
    }

    // Labeled frames in this trial, as ticks along the bottom.
    c.fillStyle = 'rgba(255,255,255,0.85)';
    for (const key of manual.keys()) {
        const [fs, ...sp] = key.split('_');
        if (sp.join('_') !== currentSide) continue;
        const f = parseInt(fs, 10);
        if (f < f0 || f > f1) continue;
        c.fillRect(x(f) - 0.5, h - 4, 1.5, 4);
    }

    c.strokeStyle = '#a195c0';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(x(currentFrame), 0);
    c.lineTo(x(currentFrame), h);
    c.stroke();

    const units = hasCalibration ? 'mm' : 'px';
    label.innerHTML = `Thumb-index distance (${units}) &nbsp;`
        + series.map(s => `<span style="color:${s.color};">■</span> ${s.label}`)
            .join(' &nbsp;');
}

function traceSeek(clientX) {
    const trial = trials[currentTrialIdx];
    if (!trial) return;
    const rect = traceCanvas.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    gotoFrame(trial.start_frame
        + Math.round(frac * (trial.end_frame - trial.start_frame)));
}

// ── Status line ──────────────────────────────────────────────────────────

function updateFrameDisplay() {
    const trial = trials[currentTrialIdx];
    const local = trial ? currentFrame - trial.start_frame : currentFrame;
    const fps = trial ? trial.fps : 60;
    const t = (local / fps).toFixed(2);
    const name = trial ? trial.trial_name : '';
    const text = frameDisplayMode === 0
        ? `Frame ${currentFrame} / ${totalFrames - 1}`
        : frameDisplayMode === 1
            ? `${t}s in ${name}`
            : `Frame ${currentFrame} · ${t}s · ${name}`;
    el('frameDisplay').textContent = text;
}

function updateLabelCount() {
    let complete = 0;
    for (const lbl of manual.values()) {
        if (bodyparts.every(bp => lbl[bp] && lbl[bp][0] != null)) complete++;
    }
    el('labelCount').textContent = `${complete}`;
}

function setLabelInfo(msg) {
    el('labelInfo').textContent = msg || '';
}

function updateSideToggle() {
    const t = el('sideToggle');
    t.textContent = currentSide;
    t.className = 'side-toggle ' + (sideIndex(currentSide) === 0 ? 'os' : 'od');
    // One camera means one image per frame; a switch that showed the
    // same picture under a different name would only invite labeling
    // the same frame twice.
    t.style.display = singleCamera() ? 'none' : '';
}

function singleCamera() {
    return cameraMode === 'single' || cameraNames.length < 2;
}

/* Switch camera without losing your place.
 *
 * The same hand sits at different pixels in OS and OD — that offset is
 * the stereo disparity, and at labeling zoom it is most of the screen.
 * Keeping the pixel transform would put the hand off-view and make the
 * E key useless for the thing it is for: placing the same point on both
 * cameras. So shift the view by the disparity, measured from the labels
 * when this frame has them and from MediaPipe when it does not.
 *
 * Only matters once the user has zoomed; before that applyAutoZoom
 * reframes from scratch and already lands on the hand. */
function toggleSide() {
    if (singleCamera()) return;
    const from = currentSide, to = otherSide();
    let shift = null;
    if (hasUserZoom) shift = cameraShift(currentFrame, from, to);

    currentSide = to;
    updateSideToggle();
    setNavState({ side: currentSide });
    if (cameraMode === 'multicam') videoSideLoaded = null;

    if (shift) {
        offsetX -= shift[0] * scale;
        offsetY -= shift[1] * scale;
    }
    gotoFrame(currentFrame, { force: true });
}

function updateShortcuts() {
    el('shortcutList').innerHTML = [
        ['&larr; &rarr;', 'previous / next frame'],
        ['Shift + &larr;&rarr;', 'jump 10 frames'],
        ['&uarr; &darr;', 'previous / next selected frame'],
        ['Space', 'play / pause'],
        ['E', 'switch camera'],
        ['N', 'next unlabeled point'],
        ['B', 'back to the target frame'],
        ['R', 'reset zoom'],
        ['1 … 9', 'jump to trial'],
        ['Ctrl/&#8984; Z', 'undo'],
        ['Ctrl/&#8984; &#8679; Z', 'redo'],
        ['Right-click', 'remove a point'],
    ].map(([k, v]) => `<div><kbd>${k}</kbd> ${v}</div>`).join('');
}

// ── Events ───────────────────────────────────────────────────────────────

function onMouseDown(e) {
    if (e.button === 2) return;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

    const hit = hitTest(sx, sy);
    if (hit) {
        const key = `${currentFrame}_${currentSide}`;
        const coords = manual.get(key)[hit.bp];
        dragging = hit.bp;
        dragOrigX = coords[0];
        dragOrigY = coords[1];
        dragStartX = sx;
        dragStartY = sy;
        canvas.style.cursor = 'grabbing';
        render();
        return;
    }

    dragging = 'pending';
    dragStartX = sx;
    dragStartY = sy;
    dragOrigX = offsetX;
    dragOrigY = offsetY;
}

function onMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

    if (!dragging) return;

    if (dragging === 'pending') {
        if (Math.hypot(sx - dragStartX, sy - dragStartY) > DRAG_THRESHOLD) {
            dragging = 'pan';
            canvas.style.cursor = 'move';
        }
        return;
    }
    if (dragging === 'pan') {
        offsetX = dragOrigX + (sx - dragStartX);
        offsetY = dragOrigY + (sy - dragStartY);
        hasUserZoom = true;
        render();
        return;
    }

    const lbl = manual.get(`${currentFrame}_${currentSide}`);
    if (!lbl) return;
    lbl[dragging] = [
        Math.round((dragOrigX + (sx - dragStartX) / scale) * 10) / 10,
        Math.round((dragOrigY + (sy - dragStartY) / scale) * 10) / 10,
    ];
    render();
}

function onMouseUp() {
    if (dragging === 'pending') {
        const img = screenToImage(dragStartX, dragStartY);
        if (!guidedBusy
            && img.x >= 0 && img.x < imgW && img.y >= 0 && img.y < imgH) {
            placeLabel(img.x, img.y);
        }
    } else if (dragging === 'pan') {
        hasUserZoom = true;
    } else if (dragging) {
        pushUndo(`${currentFrame}_${currentSide}`, dragging, [dragOrigX, dragOrigY]);
        dirtyKeys.add(`${currentFrame}_${currentSide}`);
        scheduleSave();
    }
    dragging = null;
    canvas.style.cursor = 'crosshair';
}

function onRightClick(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    if (hit) removeLabel(hit.bp);
}

function onWheel(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
    scale *= factor;
    offsetX = mx - (mx - offsetX) * factor;
    offsetY = my - (my - offsetY) * factor;
    hasUserZoom = true;
    render();
}

function onKeyDown(e) {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;

    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
        return;
    }
    if (mod && e.key.toLowerCase() === 's') {
        e.preventDefault();
        flushSave();
        return;
    }
    if (mod) return;

    const step = e.shiftKey ? 10 : 1;
    switch (e.key) {
        case 'ArrowLeft':  e.preventDefault(); gotoFrame(currentFrame - step); break;
        case 'ArrowRight': e.preventDefault(); gotoFrame(currentFrame + step); break;
        case 'ArrowUp':    e.preventDefault(); stepSelectedFrame(-1); break;
        case 'ArrowDown':  e.preventDefault(); stepSelectedFrame(1); break;
        case ' ': e.preventDefault(); togglePlay(); break;
        case 'e': case 'E': toggleSide(); break;
        case 'n': case 'N': goToNextPoint(); break;
        case 'b': case 'B': backToTarget(); break;
        case 'r': case 'R': resetZoom(); break;
        default:
            if (/^[1-9]$/.test(e.key)) {
                const idx = parseInt(e.key, 10) - 1;
                if (trials[idx]) pickTrial(idx);
            }
    }
}

// ── Boot ─────────────────────────────────────────────────────────────────

async function init() {
    canvas = el('labelCanvas');
    ctx = canvas.getContext('2d');
    containerEl = el('canvasContainer');
    vp2d = el('vp2d');
    videoEl = el('videoPlayer');
    traceCanvas = el('traceCanvas');
    traceCtx = traceCanvas.getContext('2d');

    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('contextmenu', onRightClick);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('keydown', onKeyDown);

    traceCanvas.addEventListener('mousedown', (e) => traceSeek(e.clientX));

    el('playBtn').addEventListener('click', togglePlay);
    el('resetZoomBtn').addEventListener('click', resetZoom);
    el('sideToggle').addEventListener('click', toggleSide);
    el('frameDisplay').addEventListener('click', () => {
        frameDisplayMode = (frameDisplayMode + 1) % 3;
        updateFrameDisplay();
    });
    el('speedSlider').addEventListener('input', () => {
        playbackRate = SPEED_PRESETS[parseInt(el('speedSlider').value, 10)] || 1;
        el('speedLabel').textContent = `${playbackRate}x`;
        if (videoPlaying) {
            try { videoEl.playbackRate = playbackRate; } catch (e) { /* unsupported */ }
        }
    });
    el('commitBtn').addEventListener('click', commit);
    el('nextPointBtn').addEventListener('click', goToNextPoint);
    el('exportBtn').addEventListener('click', exportForLabeling);
    const mpCb = el('showMpCb');
    mpCb.checked = showMp;
    mpCb.addEventListener('change', () => { showMp = mpCb.checked; render(); });
    el('backToTargetBtn').addEventListener('click', backToTarget);
    const advCb = el('autoAdvanceCb');
    // Remembered per browser: it is a working habit, not a project setting.
    try { autoAdvance = localStorage.getItem('dlcAutoAdvance') === '1'; }
    catch (e) { /* private window */ }
    advCb.checked = autoAdvance;
    advCb.addEventListener('change', () => {
        autoAdvance = advCb.checked;
        try { localStorage.setItem('dlcAutoAdvance', autoAdvance ? '1' : '0'); }
        catch (e) { /* private window */ }
    });
    new ResizeObserver(() => {
        if (!hasUserZoom) applyAutoZoom(currentFrame);
        render();
        renderTrace();
    }).observe(vp2d);

    // Don't let a close lose unsaved edits silently.
    window.addEventListener('beforeunload', (e) => {
        if (dirtyKeys.size || deletedKeys.size) {
            e.preventDefault();
            e.returnValue = '';
        }
    });

    updateSideToggle();
    el('speedLabel').textContent =
        `${SPEED_PRESETS[parseInt(el('speedSlider').value, 10)]}x`;

    await initSubjectNav(openSubject);
    if (!subjectId) {
        setLabelInfo('No subjects yet — add videos and press Sync on the '
                   + 'Subjects page.');
    }
    canvas.focus();
}

document.addEventListener('DOMContentLoaded', init);

return { gotoFrame, openSession };

})();

export default labeler;
