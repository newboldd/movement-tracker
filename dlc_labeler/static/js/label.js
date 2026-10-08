/* The labeling page.
 *
 * One page, three modes against the same canvas:
 *   Label    place keypoints on raw video to build the first training set
 *   Correct  edit the model's predictions and save them as corrections
 *   Refine   pick corrected frames to add to the training set
 *
 * Everything the page draws is a "layer": the manual labels of this
 * session, labels committed earlier, DLC predictions, saved corrections,
 * and each MediaPipe pass.  Layers share one shape, so the 2D canvas, the
 * 3D view, the timeline and the distance trace all read them the same way.
 */

import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const labeler = (() => {

// ── Constants ────────────────────────────────────────────────────────────

const HIT_RADIUS = 12;          // px — how close a click counts as "on a point"
const DRAG_THRESHOLD = 4;       // px — below this a mousedown+up is a click, not a pan
const PREFETCH_AHEAD = 4;       // frames to warm the JPEG cache with
const SPEED_PRESETS = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8];

// MediaPipe's 21 hand landmarks, grouped so a finger can be hidden as a unit.
const FINGERS = {
    wrist:  { label: 'Wrist',  joints: [0] },
    thumb:  { label: 'Thumb',  joints: [1, 2, 3, 4] },
    index:  { label: 'Index',  joints: [5, 6, 7, 8] },
    middle: { label: 'Middle', joints: [9, 10, 11, 12] },
    ring:   { label: 'Ring',   joints: [13, 14, 15, 16] },
    pinky:  { label: 'Pinky',  joints: [17, 18, 19, 20] },
};
const HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
];

// Bodypart layers: one coordinate per labeled bodypart per frame.
// Ordered as drawn, most authoritative last.
const BP_LAYERS = [
    { key: 'dlc',         label: 'Predictions (v1)', color: '#f44336', stage: 'dlc' },
    { key: 'refine',      label: 'Predictions (v2)', color: '#ff8a65', stage: 'refine' },
    { key: 'corrections', label: 'Corrections',      color: '#ffd54f', stage: 'corrections' },
    { key: 'committed',   label: 'Committed labels', color: '#80deea', stage: 'labels' },
];
const MANUAL_COLOR = '#ffffff';

const MODE_HELP = {
    initial: 'Click to place each keypoint, drag to adjust, right-click to remove. '
           + 'Label a spread of frames across every trial, then commit.',
    corrections: 'Predictions are shown as ghosts. Click one to take it over, then '
               + 'drag it into place. Save when the trial reads correctly.',
    refine: 'Tick the corrected frames worth training on — the ones the model got '
          + 'wrong. Committing adds just those as a new training round.',
};

// ── State ────────────────────────────────────────────────────────────────

let subjectId = null, sessionId = null, mode = 'initial';
let subject = null, trials = [], totalFrames = 0;
let bodyparts = [], cameraNames = ['OS', 'OD'], cameraMode = 'stereo';
let hasCalibration = false;

let currentFrame = 0, currentSide = 'OS', currentTrialIdx = -1;
let videoTrialLoaded = -1, videoSideLoaded = null;

// Manual labels of this session: "frame_side" -> {bodypart: [x, y]}
let manual = new Map();
const dirtyKeys = new Set();
const deletedKeys = new Set();

// Layer data. Bodypart layers hold subject-wide arrays; MediaPipe layers
// hold one trial's typed arrays and are refetched when the trial changes.
const bpData = {};          // key -> {cam: {bp: [...]}, joints_3d: {...}, distances: [...]}
let mpTrial = null;         // {trial_idx, start_frame, passes: {key: {...}}}
let mpPassSpecs = [];       // from the server: key, label, colour
let availableMp = {};       // trial_idx -> [pass keys]
let availableStages = [];   // which bodypart stages have data
let hasCommittedLabels = false;

// Layer visibility: key -> {d2: bool, d3: bool}
const layerVis = new Map();
const visibleJoints = new Set([...Array(21).keys()]);

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
let boxMode = false, editBox = null, boxDragHandle = null, boxDragStart = null;
let cropBoxes = {};         // trial_idx -> {cam: {x1,y1,x2,y2}}

// Refine selection: "frame_side" of frames to add to the training set
const refineSelected = new Set();

// Distances the server recomputed for frames edited in this session.
let manualDistances = {};

// 3D
let three = null;            // {scene, camera, renderer, controls, root}
let show3d = false;

// Timeline / trace
let timelineCanvas, timelineCtx, traceCanvas, traceCtx;
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

/* Bodypart-layer coordinate, or null. */
function bpAt(layerKey, frame, side, bp) {
    const d = bpData[layerKey];
    if (!d) return null;
    const cam = d[side];
    const seq = cam && cam[bp];
    const c = seq && seq[frame];
    return (c && c[0] != null) ? c : null;
}

/* The best available coordinate to show as a ghost under the cursor.
 *
 * Priority runs from most to least authoritative, which is also what a
 * click adopts: a human correction beats a model prediction, and a
 * prediction beats a raw MediaPipe tip. */
function ghostAt(frame, side, bp) {
    const order = mode === 'initial'
        ? ['committed', 'dlc', 'refine', 'corrections']
        : ['corrections', 'refine', 'dlc', 'committed'];
    for (const key of order) {
        const c = bpAt(key, frame, side, bp);
        if (c) return { coords: c, source: key };
    }
    return null;
}

/* The MediaPipe pass arrays for the loaded trial, or null. */
function mpPass(key) {
    return (mpTrial && mpTrial.passes && mpTrial.passes[key]) || null;
}

/* 21x2 landmarks for one frame of a pass, as a flat Float32Array view. */
function mpFrame2d(passKey, frame, side) {
    const p = mpPass(passKey);
    if (!p) return null;
    const arr = side === cameraNames[0] ? p.os : p.od;
    if (!arr) return null;
    const local = frame - mpTrial.start_frame;
    const [n, j] = arr.shape;
    if (local < 0 || local >= n) return null;
    return arr.data.subarray(local * j * 2, (local + 1) * j * 2);
}

/* 21x3 triangulated joints for one frame of a pass. */
function mpFrame3d(passKey, frame) {
    const p = mpPass(passKey);
    if (!p || !p.j3d) return null;
    const local = frame - mpTrial.start_frame;
    const [n, j] = p.j3d.shape;
    if (local < 0 || local >= n) return null;
    return p.j3d.data.subarray(local * j * 3, (local + 1) * j * 3);
}

function vis(key) {
    return layerVis.get(key) || { d2: false, d3: false };
}

// ── Session loading ──────────────────────────────────────────────────────

async function openSubject(id) {
    subjectId = id;
    playing = false;
    stopVideoPlayback();
    await openSession(mode);
}

async function openSession(newMode) {
    mode = newMode;
    document.querySelectorAll('.mode-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode));
    el('modeHelp').textContent = MODE_HELP[mode] || '';

    // Refine edits the same corrections a Correct session does; it adds the
    // choice of which frames graduate into training.  Sharing the session
    // type means edits made in one mode are visible in the other.
    const sessionType = mode === 'refine' ? 'corrections' : mode;

    let session;
    try {
        session = await api(`/api/labeling/${subjectId}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_type: sessionType }),
        });
    } catch (e) {
        toast(`Could not open a session: ${e.message}`, 'err');
        return;
    }
    sessionId = session.id;

    manual.clear();
    dirtyKeys.clear();
    deletedKeys.clear();
    refineSelected.clear();
    undoStack.length = 0;
    redoStack.length = 0;
    imageCache.clear();
    currentTrialIdx = -1;
    videoTrialLoaded = -1;
    videoSideLoaded = null;
    mpTrial = null;
    manual3dCache = { key: null, points: {}, distance: null };
    if (three) three.framed = false;
    for (const k of Object.keys(bpData)) delete bpData[k];

    const info = await api(`/api/labeling/sessions/${sessionId}/info`);
    subject = info.subject;
    trials = info.trials || [];
    totalFrames = info.total_frames || 0;
    bodyparts = info.bodyparts || [];
    cameraNames = info.camera_names || ['OS', 'OD'];
    cameraMode = info.camera_mode || 'stereo';
    hasCalibration = !!info.has_calibration;
    cropBoxes = info.crop_boxes || {};
    mpPassSpecs = info.mp_passes || [];

    if (!cameraNames.includes(currentSide)) currentSide = cameraNames[0];
    updateSideToggle();

    el('committedCount').textContent =
        `${info.committed_frame_count || 0} frames committed to DLC`;
    el('calibNote').textContent = hasCalibration
        ? 'Stereo calibration loaded — 3D and mm distances available'
        : 'No stereo calibration for this subject — 2D only, distances in pixels';
    el('calibNote').style.color = hasCalibration ? '' : 'var(--orange)';

    // Load the per-session labels and the layer inventory together.
    const [labels, layers] = await Promise.all([
        api(`/api/labeling/sessions/${sessionId}/labels`),
        api(`/api/labeling/sessions/${sessionId}/layers`),
    ]);

    for (const l of labels) {
        manual.set(`${l.frame_num}_${l.side}`, { ...l.keypoints });
    }

    availableMp = layers.mp_by_trial || {};
    availableStages = layers.stages || [];
    hasCommittedLabels = !!layers.has_committed_labels;
    if (layers.mp_passes) mpPassSpecs = layers.mp_passes;

    buildLayerGrid();
    buildFingerGrid();
    updateModeUi();
    updateLabelCount();
    updateShortcuts();

    // Jump back to where this subject was left, else the first trial.
    const nav = getNavState();
    const startFrame = (nav.frame != null && nav.frame < totalFrames)
        ? nav.frame : (trials[0] ? trials[0].start_frame : 0);

    await loadBodypartLayers();
    await gotoFrame(startFrame, { force: true });
    renderTrialButtons(trials, trialForFrame(currentFrame), pickTrial);
}

/* Fetch every bodypart layer that has data.  Each is subject-wide, so this
 * runs once per session rather than per trial. */
async function loadBodypartLayers() {
    const jobs = [];
    for (const spec of BP_LAYERS) {
        const wanted = spec.stage === 'labels'
            ? hasCommittedLabels
            : availableStages.includes(spec.stage);
        if (!wanted) continue;
        jobs.push(
            api(`/api/labeling/sessions/${sessionId}/stage?stage=${spec.stage}`)
                .then(d => { if (d && Object.keys(d).length) bpData[spec.key] = d; })
                .catch(e => console.warn(`layer ${spec.key} failed`, e))
        );
    }
    await Promise.all(jobs);
    buildLayerGrid();
    if (mode === 'refine') buildRefineList();
}

/* Fetch the MediaPipe passes for one trial. */
async function loadMpTrial(trialIdx) {
    if (mpTrial && mpTrial.trial_idx === trialIdx) return;
    const present = availableMp[trialIdx] || availableMp[String(trialIdx)] || [];
    if (!present.length) { mpTrial = { trial_idx: trialIdx, start_frame: 0, passes: {} }; return; }

    try {
        const d = await api(`/api/labeling/sessions/${sessionId}`
            + `/trial/${trialIdx}/mp?passes=${present.join(',')}`);
        const passes = {};
        for (const [key, p] of Object.entries(d.passes || {})) {
            passes[key] = {
                label: p.label, color: p.color,
                os: decodeF32(p.OS), od: decodeF32(p.OD), j3d: decodeF32(p.joints_3d),
                distances: p.distances || null,
                distancesClean: p.distances_clean || null,
                sourceOS: p.source_OS || null, sourceOD: p.source_OD || null,
            };
        }
        mpTrial = { trial_idx: trialIdx, start_frame: d.start_frame || 0,
                    frame_count: d.frame_count || 0, passes };
    } catch (e) {
        console.warn('MediaPipe trial load failed', e);
        mpTrial = { trial_idx: trialIdx, start_frame: 0, passes: {} };
    }
    buildLayerGrid();
}

// ── Sidebar: layers and fingers ──────────────────────────────────────────

function layerCatalog() {
    const out = [];
    out.push({ key: 'manual', label: 'Manual (this session)', color: MANUAL_COLOR,
               available: true, has3d: hasCalibration });
    for (const spec of BP_LAYERS) {
        out.push({ key: spec.key, label: spec.label, color: spec.color,
                   available: !!bpData[spec.key],
                   has3d: hasCalibration && !!(bpData[spec.key] || {}).joints_3d });
    }
    const presentHere = availableMp[currentTrialIdx]
        || availableMp[String(currentTrialIdx)] || [];
    for (const spec of mpPassSpecs) {
        out.push({ key: `mp_${spec.key}`, label: spec.label, color: spec.color,
                   indent: true, mpPass: spec.key,
                   available: presentHere.includes(spec.key),
                   has3d: hasCalibration });
    }
    return out;
}

function buildLayerGrid() {
    const grid = el('layerGrid');
    if (!grid) return;
    const cat = layerCatalog();

    // Sensible first view: what you are working on, plus the hand.
    if (!layerVis.size) {
        const defaults = {
            manual: { d2: true, d3: true },
            corrections: { d2: true, d3: false },
            dlc: { d2: true, d3: false },
            refine: { d2: true, d3: false },
            committed: { d2: true, d3: false },
            mp_best: { d2: true, d3: true },
            mp_forward: { d2: true, d3: true },
        };
        for (const l of cat) layerVis.set(l.key, { ...(defaults[l.key] || { d2: false, d3: false }) });
        // Only one MediaPipe pass on by default: the fused one if it exists.
        if (cat.some(l => l.key === 'mp_best' && l.available)) {
            layerVis.set('mp_forward', { d2: false, d3: false });
        }
    }
    for (const l of cat) if (!layerVis.has(l.key)) layerVis.set(l.key, { d2: false, d3: false });

    grid.innerHTML = '<span></span><span class="hdr">2D</span><span class="hdr">3D</span>';
    for (const l of cat) {
        const v = vis(l.key);
        const name = document.createElement('span');
        name.className = 'lname' + (l.indent ? ' indent' : '')
            + (l.available ? '' : ' unavailable');
        name.innerHTML = `<span class="swatch" style="background:${l.color};"></span>`
            + `<span>${l.label}</span>`;
        if (!l.available) name.title = 'No data for this trial yet';
        grid.appendChild(name);

        for (const dim of ['d2', 'd3']) {
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = !!v[dim] && l.available && (dim === 'd2' || l.has3d);
            cb.disabled = !l.available || (dim === 'd3' && !l.has3d);
            if (dim === 'd3' && !l.has3d) {
                cb.title = hasCalibration
                    ? 'No triangulated points for this layer'
                    : 'Needs a stereo calibration for this subject';
            }
            cb.addEventListener('change', () => {
                layerVis.get(l.key)[dim] = cb.checked;
                if (dim === 'd3' && cb.checked && !show3d) toggle3d(true);
                render();
                update3d();
                renderTrace();
            });
            grid.appendChild(cb);
        }
    }

    const note = el('layerNote');
    const bestOn = vis('mp_best').d2 || vis('mp_best').d3;
    note.textContent = bestOn
        ? 'Best per frame picks, for each frame and camera, whichever pass '
        + 'triangulates to the most plausible thumb-index aperture.'
        : '';
}

function buildFingerGrid() {
    const grid = el('fingerGrid');
    if (!grid) return;
    grid.innerHTML = '';
    for (const [key, f] of Object.entries(FINGERS)) {
        const lab = document.createElement('label');
        lab.className = 'finger-toggle';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = f.joints.every(j => visibleJoints.has(j));
        cb.addEventListener('change', () => {
            for (const j of f.joints) {
                if (cb.checked) visibleJoints.add(j); else visibleJoints.delete(j);
            }
            render();
            update3d();
        });
        lab.appendChild(cb);
        lab.appendChild(document.createTextNode(f.label));
        grid.appendChild(lab);
    }
}

function updateModeUi() {
    const isRefine = mode === 'refine';
    const isCorr = mode === 'corrections';
    el('saveCorrBtn').style.display = (isCorr || isRefine) ? '' : 'none';
    el('refinePanel').style.display = isRefine ? '' : 'none';
    el('commitBtn').innerHTML = isRefine ? 'Commit selected → training'
        : isCorr ? 'Save &amp; finish corrections' : 'Save &amp; commit';
    el('actionHelp').textContent = isRefine
        ? 'Commits the ticked frames as a new labeled-data round. Then run '
        + 'Refine on the Jobs page.'
        : isCorr
            ? 'Writes every trial\'s reviewed labels to corrections/ as DLC CSVs.'
            : 'Extracts your labeled frames as training images and writes the '
            + 'DLC CollectedData files. Then run Train on the Jobs page.';
    if (isRefine) buildRefineList();
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

    let justLoaded = false;
    if (videoTrialLoaded !== trialIdx || videoSideLoaded !== currentSide) {
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
    renderTimeline();
    renderTrace();
    update3d();
}

function pickTrial(trialIdx) {
    const t = trials[trialIdx];
    if (t) gotoFrame(t.start_frame);
}

function nextLabeledFrame(dir) {
    const keys = [...manual.keys()]
        .map(k => ({ f: parseInt(k.split('_')[0], 10), side: k.split('_').slice(1).join('_') }))
        .filter(x => x.side === currentSide)
        .map(x => x.f)
        .sort((a, b) => a - b);
    if (!keys.length) return null;
    return dir > 0 ? keys.find(f => f > currentFrame)
                   : [...keys].reverse().find(f => f < currentFrame);
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

/* Frame the view on the hand without anyone having to pan.
 *
 * Preference order: the labels on this frame, then whatever layer is
 * showing, then the whole MediaPipe hand, then the trial's crop box.
 * Falling back through those means the zoom is useful on a frame with no
 * labels yet — which is every frame you are about to label. */
function applyAutoZoom(frame) {
    if (hasUserZoom || boxMode) return;
    if (!el('autoZoomCb').checked) { fitImage(); return; }

    if (el('boxZoomCb').checked) {
        const box = (cropBoxes[currentTrialIdx]
            || cropBoxes[String(currentTrialIdx)] || {})[currentSide];
        if (box) {
            zoomToBox(box.x1, box.y1, box.x2, box.y2, 10);
            return;
        }
    }

    const pts = [];
    for (const bp of bodyparts) {
        const c = manualAt(frame, currentSide, bp) || (ghostAt(frame, currentSide, bp) || {}).coords;
        if (c) pts.push(c);
    }
    // No bodypart coordinates: use the whole hand from whichever
    // MediaPipe pass is on, which is the common case on a fresh subject.
    if (!pts.length) {
        for (const spec of mpPassSpecs) {
            if (!vis(`mp_${spec.key}`).d2) continue;
            const lm = mpFrame2d(spec.key, frame, currentSide);
            if (!lm) continue;
            for (let j = 0; j < lm.length / 2; j++) {
                if (Number.isFinite(lm[j * 2])) pts.push([lm[j * 2], lm[j * 2 + 1]]);
            }
            if (pts.length) break;
        }
    }

    if (!pts.length) {
        const box = (cropBoxes[currentTrialIdx]
            || cropBoxes[String(currentTrialIdx)] || {})[currentSide];
        if (box) zoomToBox(box.x1, box.y1, box.x2, box.y2, 10);
        else fitImage();
        return;
    }

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
    drawBodypartOverlays();
    if (boxMode && editBox) drawBoxOverlay();
}

function drawMpOverlays() {
    for (const spec of mpPassSpecs) {
        if (!vis(`mp_${spec.key}`).d2) continue;
        const lm = mpFrame2d(spec.key, currentFrame, currentSide);
        if (!lm) continue;
        const color = spec.color;

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.65;
        for (const [a, b] of HAND_BONES) {
            if (!visibleJoints.has(a) || !visibleJoints.has(b)) continue;
            if (!Number.isFinite(lm[a * 2]) || !Number.isFinite(lm[b * 2])) continue;
            const p = imageToScreen(lm[a * 2], lm[a * 2 + 1]);
            const q = imageToScreen(lm[b * 2], lm[b * 2 + 1]);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        for (let j = 0; j < lm.length / 2; j++) {
            if (!visibleJoints.has(j) || !Number.isFinite(lm[j * 2])) continue;
            const p = imageToScreen(lm[j * 2], lm[j * 2 + 1]);
            ctx.beginPath();
            ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

function drawBodypartOverlays() {
    // Model and committed layers first, so the manual label a user is
    // actually moving is never hidden under a ghost.
    for (const spec of BP_LAYERS) {
        if (!vis(spec.key).d2) continue;
        for (const bp of bodyparts) {
            const c = bpAt(spec.key, currentFrame, currentSide, bp);
            if (!c) continue;
            if (manualAt(currentFrame, currentSide, bp)) {
                drawGhost(c[0], c[1], spec.color, bp[0].toUpperCase());
            } else {
                drawPoint(c[0], c[1], spec.color, bp[0].toUpperCase(), false);
            }
        }
    }

    if (vis('manual').d2) {
        for (const bp of bodyparts) {
            const c = manualAt(currentFrame, currentSide, bp);
            if (c) drawPoint(c[0], c[1], MANUAL_COLOR, bp[0].toUpperCase(), true);
        }
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

function drawGhost(ix, iy, color, letter) {
    const p = imageToScreen(ix, iy);
    ctx.save();
    ctx.globalAlpha = 0.4;
    ctx.setLineDash([2, 2]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
}

// ── Crop box editing ─────────────────────────────────────────────────────

function defaultBox() {
    return { x1: imgW * 0.25, y1: imgH * 0.25, x2: imgW * 0.75, y2: imgH * 0.75 };
}

async function startBoxEdit() {
    boxMode = true;
    el('boxActions').style.display = '';
    el('editBoxBtn').style.display = 'none';

    const saved = (cropBoxes[currentTrialIdx]
        || cropBoxes[String(currentTrialIdx)] || {})[currentSide];
    if (saved) {
        editBox = { ...saved };
        el('boxStatus').textContent = 'Editing the saved box for this trial.';
    } else {
        // Nothing saved — ask the server for one derived from the landmarks,
        // which puts the box roughly right before anyone drags a handle.
        try {
            const r = await api(`/api/labeling/sessions/${sessionId}`
                + `/bbox?trial_idx=${currentTrialIdx}`);
            const b = (r.boxes || {})[currentSide];
            editBox = b ? { x1: b[0], y1: b[1], x2: b[2], y2: b[3] } : defaultBox();
            el('boxStatus').textContent = b
                ? 'Starting from the MediaPipe landmark extent.'
                : 'No landmarks yet — drag the box onto the hand.';
        } catch (e) {
            editBox = defaultBox();
            el('boxStatus').textContent = 'Drag the box onto the hand.';
        }
    }
    hasUserZoom = false;
    render();
}

function endBoxEdit() {
    boxMode = false;
    editBox = null;
    boxDragHandle = null;
    el('boxActions').style.display = 'none';
    el('editBoxBtn').style.display = '';
    render();
}

function drawBoxOverlay() {
    const a = imageToScreen(editBox.x1, editBox.y1);
    const b = imageToScreen(editBox.x2, editBox.y2);
    ctx.save();
    ctx.strokeStyle = '#ffd54f';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    ctx.setLineDash([]);
    ctx.fillStyle = '#ffd54f';
    for (const [hx, hy] of [[a.x, a.y], [b.x, a.y], [a.x, b.y], [b.x, b.y]]) {
        ctx.fillRect(hx - 4, hy - 4, 8, 8);
    }
    ctx.restore();
}

function boxHitTest(sx, sy) {
    const a = imageToScreen(editBox.x1, editBox.y1);
    const b = imageToScreen(editBox.x2, editBox.y2);
    const near = (px, py) => Math.hypot(sx - px, sy - py) < 9;
    if (near(a.x, a.y)) return 'nw';
    if (near(b.x, a.y)) return 'ne';
    if (near(a.x, b.y)) return 'sw';
    if (near(b.x, b.y)) return 'se';
    if (sx > a.x && sx < b.x && sy > a.y && sy < b.y) return 'move';
    return null;
}

function applyBoxDrag(sx, sy) {
    const d = screenToImage(sx, sy);
    const start = screenToImage(boxDragStart.mx, boxDragStart.my);
    const o = boxDragStart.box;
    const dx = d.x - start.x, dy = d.y - start.y;
    const clampX = (v) => Math.max(0, Math.min(imgW, v));
    const clampY = (v) => Math.max(0, Math.min(imgH, v));

    if (boxDragHandle === 'move') {
        editBox = { x1: clampX(o.x1 + dx), y1: clampY(o.y1 + dy),
                    x2: clampX(o.x2 + dx), y2: clampY(o.y2 + dy) };
        return;
    }
    const b = { ...o };
    if (boxDragHandle.includes('n')) b.y1 = clampY(o.y1 + dy);
    if (boxDragHandle.includes('s')) b.y2 = clampY(o.y2 + dy);
    if (boxDragHandle.includes('w')) b.x1 = clampX(o.x1 + dx);
    if (boxDragHandle.includes('e')) b.x2 = clampX(o.x2 + dx);
    editBox = {
        x1: Math.min(b.x1, b.x2), y1: Math.min(b.y1, b.y2),
        x2: Math.max(b.x1, b.x2), y2: Math.max(b.y1, b.y2),
    };
}

async function saveBox() {
    if (!editBox) return;
    const boxes = {};
    boxes[currentSide] = {
        x1: Math.round(editBox.x1), y1: Math.round(editBox.y1),
        x2: Math.round(editBox.x2), y2: Math.round(editBox.y2),
    };
    try {
        const r = await api(`/api/labeling/sessions/${sessionId}/bbox`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trial_idx: currentTrialIdx, boxes,
                apply_to_all: el('boxAllTrials').checked,
            }),
        });
        for (const ti of r.applied_trials) {
            cropBoxes[ti] = { ...(cropBoxes[ti] || {}), ...boxes };
        }
        toast(r.applied_trials.length > 1
            ? `Box saved for ${r.applied_trials.length} trials`
            : 'Box saved', 'ok');
        el('boxStatus').textContent = 'Saved. Re-detect to apply it.';
    } catch (e) {
        toast(`Could not save the box: ${e.message}`, 'err');
    }
}

async function redetectTrial() {
    if (!editBox) return;
    const crops = {};
    crops[currentSide] = {
        x1: Math.round(editBox.x1), y1: Math.round(editBox.y1),
        x2: Math.round(editBox.x2), y2: Math.round(editBox.y2),
    };
    try {
        const r = await api(`/api/labeling/sessions/${sessionId}/rerun-mediapipe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ trial_idx: currentTrialIdx, crops }),
        });
        el('boxStatus').textContent = 'Detecting…';
        watchJob(r.job_id, async (job) => {
            if (job.status === 'completed') {
                el('boxStatus').textContent = 'Detection finished — reloading.';
                mpTrial = null;
                const layers = await api(
                    `/api/labeling/sessions/${sessionId}/layers`);
                availableMp = layers.mp_by_trial || {};
                await loadMpTrial(currentTrialIdx);
                render();
                update3d();
                renderTrace();
            } else if (job.status === 'failed') {
                el('boxStatus').textContent = `Detection failed: ${job.error_msg || ''}`;
            } else {
                el('boxStatus').textContent =
                    `Detecting… ${Math.round(job.progress_pct || 0)}%`;
            }
        });
    } catch (e) {
        toast(`Could not start detection: ${e.message}`, 'err');
    }
}

/* Poll one job until it reaches a terminal state. */
function watchJob(jobId, onUpdate) {
    const tick = async () => {
        let job;
        try {
            job = await api(`/api/jobs/${jobId}`);
        } catch (e) { return; }
        onUpdate(job);
        if (!['completed', 'failed', 'cancelled'].includes(job.status)) {
            setTimeout(tick, 1000);
        }
    };
    tick();
}

// ── Label editing ────────────────────────────────────────────────────────

function hitTest(sx, sy) {
    // Manual points first: they are what a drag is most likely aimed at.
    for (const bp of bodyparts) {
        const c = manualAt(currentFrame, currentSide, bp);
        if (!c) continue;
        const p = imageToScreen(c[0], c[1]);
        if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) return { bp, ghost: false };
    }
    for (const bp of bodyparts) {
        if (manualAt(currentFrame, currentSide, bp)) continue;
        const g = ghostAt(currentFrame, currentSide, bp);
        if (!g) continue;
        const p = imageToScreen(g.coords[0], g.coords[1]);
        if (Math.hypot(sx - p.x, sy - p.y) < HIT_RADIUS) {
            return { bp, ghost: true, coords: g.coords, source: g.source };
        }
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

function placeLabel(ix, iy) {
    const key = `${currentFrame}_${currentSide}`;
    const lbl = manual.get(key) || {};

    // Fill the next unplaced bodypart; once all are placed, move the
    // nearest one instead, so a stray click corrects rather than adds.
    const nextBp = bodyparts.find(bp => !(lbl[bp] && lbl[bp][0] != null));
    if (nextBp) {
        pushUndo(key, nextBp, null);
        setManual(nextBp, ix, iy);
        const left = bodyparts.filter(bp => {
            const c = manual.get(key)[bp];
            return !(c && c[0] != null);
        });
        setLabelInfo(left.length
            ? `${nextBp} placed — click to place ${left[0]}.`
            : 'All keypoints placed on this frame.');
    } else {
        let closest = null, best = Infinity;
        for (const bp of bodyparts) {
            const c = lbl[bp];
            if (!c || c[0] == null) continue;
            const d = Math.hypot(ix - c[0], iy - c[1]);
            if (d < best) { best = d; closest = bp; }
        }
        if (!closest) return;
        pushUndo(key, closest, lbl[closest]);
        setManual(closest, ix, iy);
    }

    render();
    update3d();
    updateLabelCount();
    scheduleSave();
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
    update3d();
    updateLabelCount();
    scheduleSave();
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
        update3d();
    }
    updateLabelCount();
    scheduleSave();
}

function undo() {
    const e = undoStack.pop();
    if (e) applyUndoEntry(e, redoStack);
}

function redo() {
    const e = redoStack.pop();
    if (e) applyUndoEntry(e, undoStack);
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

async function saveCorrections() {
    await flushSave();
    try {
        const r = await api(`/api/labeling/sessions/${sessionId}/save_corrections`,
                            { method: 'POST' });
        toast(`Corrections written: ${r.csv_count} CSV file(s), `
            + `${r.frame_count || r.corrected_frame_count || 0} frames`, 'ok');
        await loadBodypartLayers();
        render();
    } catch (e) {
        toast(`Could not save corrections: ${e.message}`, 'err');
    }
}

async function commit() {
    await flushSave();

    if (mode === 'refine') {
        if (!refineSelected.size) {
            toast('Tick at least one frame to add to the training set', 'err');
            return;
        }
        const frames = [...refineSelected].map(k => {
            const [f, ...sp] = k.split('_');
            return { frame_num: parseInt(f, 10), side: sp.join('_') };
        });
        if (!confirm(`Add ${frames.length} corrected frame(s) to the training set `
                   + `as a new round?`)) return;
        try {
            const r = await api(`/api/labeling/sessions/${sessionId}/commit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ train_frames: frames }),
            });
            toast(`Committed ${r.frame_count} training frame(s) to `
                + `${r.labeled_data_dir.split('/').slice(-2).join('/')}. `
                + `Now run Refine on the Jobs page.`, 'ok');
            await openSession('refine');
        } catch (e) {
            toast(`Commit failed: ${e.message}`, 'err');
        }
        return;
    }

    if (mode === 'corrections') {
        if (!confirm('Write this subject\'s reviewed labels to corrections/ '
                   + 'and mark the subject corrected?')) return;
        try {
            const r = await api(`/api/labeling/sessions/${sessionId}/commit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ train_frames: [] }),
            });
            toast(`Corrections written: ${r.csv_count} CSV file(s). Switch to `
                + `Refine to choose frames to retrain on.`, 'ok');
            await openSession('corrections');
        } catch (e) {
            toast(`Could not finish corrections: ${e.message}`, 'err');
        }
        return;
    }

    const n = manual.size;
    if (!n) { toast('Nothing to commit — place some labels first', 'err'); return; }
    if (!confirm(`Commit ${n} labeled frame(s) as DLC training data?`)) return;
    try {
        const r = await api(`/api/labeling/sessions/${sessionId}/commit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ train_frames: [] }),
        });
        toast(`Committed ${r.frame_count} frame(s). Now run Train on the `
            + `Jobs page.`, 'ok');
        await openSession('initial');
    } catch (e) {
        toast(`Commit failed: ${e.message}`, 'err');
    }
}

// ── Refine selection ─────────────────────────────────────────────────────

/* Frames where the saved correction moved a point away from the model's
 * prediction.  Those are the frames the model got wrong, and the only
 * ones worth adding to the training set. */
function correctedFrames() {
    const corr = bpData.corrections;
    if (!corr) return [];
    const pred = bpData.refine || bpData.dlc;
    const out = [];
    for (const side of cameraNames) {
        const cside = corr[side];
        if (!cside) continue;
        const len = Math.max(...bodyparts.map(bp => (cside[bp] || []).length), 0);
        for (let f = 0; f < len; f++) {
            let worst = 0;
            let any = false;
            for (const bp of bodyparts) {
                const c = (cside[bp] || [])[f];
                if (!c || c[0] == null) continue;
                any = true;
                const p = pred && pred[side] && (pred[side][bp] || [])[f];
                if (p && p[0] != null) {
                    worst = Math.max(worst, Math.hypot(c[0] - p[0], c[1] - p[1]));
                } else {
                    worst = Infinity;   // no prediction at all: definitely worth training
                }
            }
            if (any && worst > 1.0) out.push({ frame: f, side, delta: worst });
        }
    }
    out.sort((a, b) => b.delta - a.delta);
    return out;
}

function buildRefineList() {
    const list = el('refineList');
    const summary = el('refineSummary');
    if (!list) return;

    const rows = correctedFrames();
    if (!rows.length) {
        list.innerHTML = '';
        summary.textContent = bpData.corrections
            ? 'No corrections differ from the predictions yet. Switch to '
            + 'Correct, fix the frames the model gets wrong, and save.'
            : 'No corrections saved for this subject yet.';
        return;
    }

    summary.innerHTML = `${rows.length} corrected frame(s), largest change first. `
        + `<span id="refineCount">${refineSelected.size}</span> selected.`;
    list.innerHTML = '<table><thead><tr><th></th><th>Frame</th><th>Cam</th>'
        + '<th>Moved</th></tr></thead><tbody>'
        + rows.map(r => {
            const key = `${r.frame}_${r.side}`;
            const moved = Number.isFinite(r.delta) ? `${r.delta.toFixed(1)} px` : 'new';
            return `<tr><td><input type="checkbox" data-key="${key}"
                        ${refineSelected.has(key) ? 'checked' : ''}></td>`
                + `<td><a href="#" data-goto="${r.frame}" data-side="${r.side}"
                        style="color:var(--blue);">${r.frame}</a></td>`
                + `<td>${r.side}</td><td>${moved}</td></tr>`;
        }).join('') + '</tbody></table>';

    list.querySelectorAll('input[type=checkbox]').forEach(cb => {
        cb.addEventListener('change', () => {
            if (cb.checked) refineSelected.add(cb.dataset.key);
            else refineSelected.delete(cb.dataset.key);
            const c = el('refineCount');
            if (c) c.textContent = String(refineSelected.size);
            renderTimeline();
        });
    });
    list.querySelectorAll('a[data-goto]').forEach(a => {
        a.addEventListener('click', (e) => {
            e.preventDefault();
            currentSide = a.dataset.side;
            updateSideToggle();
            gotoFrame(parseInt(a.dataset.goto, 10));
        });
    });
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

    if (videoTrialLoaded !== trialIdx || videoSideLoaded !== currentSide) {
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

    // The timeline, trace and 3D view are context, not the thing being
    // watched; refreshing them every frame at 60 fps costs more than it adds.
    if (currentFrame % 10 === 0) {
        renderTimeline();
        renderTrace();
        update3d();
    }

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

// ── 3D view ──────────────────────────────────────────────────────────────

function toggle3d(on) {
    show3d = (on == null) ? !show3d : on;
    el('vp3d').classList.toggle('active', show3d);
    el('view3dBtn').classList.toggle('active', show3d);
    if (show3d) {
        if (!three) setup3d();
        resize3d();
        update3d();
    }
    // The 2D viewport just changed width.
    requestAnimationFrame(() => {
        if (videoFrameMode) drawVideoToCanvas(currentFrame);
        else { if (!hasUserZoom) applyAutoZoom(currentFrame); render(); }
    });
}

function setup3d() {
    const host = el('vp3d');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x120822);

    const camera = new THREE.PerspectiveCamera(
        40, host.clientWidth / Math.max(host.clientHeight, 1), 1, 20000);
    camera.position.set(0, 0, -400);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dir = new THREE.DirectionalLight(0xffffff, 0.5);
    dir.position.set(1, 1, -1);
    scene.add(dir);

    // Everything we draw hangs off one group, so a frame change can clear
    // it without touching the lights, camera or controls.
    const root = new THREE.Group();
    scene.add(root);

    three = { scene, camera, renderer, controls, root, framed: false };

    (function animate() {
        if (!three) return;
        requestAnimationFrame(animate);
        if (!show3d) return;
        three.controls.update();
        three.renderer.render(three.scene, three.camera);
    })();

    new ResizeObserver(resize3d).observe(host);
}

function resize3d() {
    if (!three) return;
    const host = el('vp3d');
    const w = host.clientWidth, h = Math.max(host.clientHeight, 1);
    three.camera.aspect = w / h;
    three.camera.updateProjectionMatrix();
    three.renderer.setSize(w, h);
}

function clear3d() {
    const root = three.root;
    while (root.children.length) {
        const c = root.children.pop();
        c.geometry?.dispose();
        if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
        else c.material?.dispose();
    }
}

function add3dSphere(x, y, z, color, radius) {
    const m = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 12, 10),
        new THREE.MeshLambertMaterial({ color }));
    m.position.set(x, y, z);
    three.root.add(m);
}

function add3dLine(pts, color, opacity) {
    const g = new THREE.BufferGeometry().setFromPoints(
        pts.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    three.root.add(new THREE.Line(g, new THREE.LineBasicMaterial({
        color, transparent: opacity < 1, opacity,
    })));
}

function update3d() {
    if (!three || !show3d) return;
    clear3d();

    const centroid = [0, 0, 0];
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    let nPts = 0;
    const note = (x, y, z) => {
        centroid[0] += x; centroid[1] += y; centroid[2] += z; nPts++;
        lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x);
        lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y);
        lo[2] = Math.min(lo[2], z); hi[2] = Math.max(hi[2], z);
    };

    // MediaPipe hands: the full 21-joint skeleton, which is what makes the
    // 3D view legible as a hand rather than a pair of floating dots.
    for (const spec of mpPassSpecs) {
        if (!vis(`mp_${spec.key}`).d3) continue;
        const j3d = mpFrame3d(spec.key, currentFrame);
        if (!j3d) continue;
        const color = new THREE.Color(spec.color);
        for (let j = 0; j < j3d.length / 3; j++) {
            if (!visibleJoints.has(j)) continue;
            const x = j3d[j * 3], y = j3d[j * 3 + 1], z = j3d[j * 3 + 2];
            if (!Number.isFinite(x)) continue;
            add3dSphere(x, y, z, color, 3);
            note(x, y, z);
        }
        for (const [a, b] of HAND_BONES) {
            if (!visibleJoints.has(a) || !visibleJoints.has(b)) continue;
            const ax = j3d[a * 3], bx = j3d[b * 3];
            if (!Number.isFinite(ax) || !Number.isFinite(bx)) continue;
            add3dLine([[ax, j3d[a * 3 + 1], j3d[a * 3 + 2]],
                       [bx, j3d[b * 3 + 1], j3d[b * 3 + 2]]], color, 0.85);
        }
    }

    // Bodypart layers: a marker per labeled point, plus the segment between
    // them — the thumb-index aperture is the measurement, so show it.
    const drawBpLayer = (key, color) => {
        const d = bpData[key];
        const j3 = d && d.joints_3d;
        if (!j3) return null;
        const pts = [];
        for (const bp of bodyparts) {
            const p = (j3[bp] || [])[currentFrame];
            if (!p || p[0] == null) continue;
            add3dSphere(p[0], p[1], p[2], new THREE.Color(color), 4.5);
            note(p[0], p[1], p[2]);
            pts.push(p);
        }
        if (pts.length === 2) add3dLine(pts, new THREE.Color(color), 0.9);
        return pts;
    };

    for (const spec of BP_LAYERS) {
        if (vis(spec.key).d3) drawBpLayer(spec.key, spec.color);
    }

    // Manual labels triangulate client-side only when both cameras have a
    // point; without that there is no depth to show, so nothing is drawn
    // rather than something misleading.
    if (vis('manual').d3) {
        const pts = manual3dForFrame();
        for (const p of pts) {
            add3dSphere(p[0], p[1], p[2], new THREE.Color(MANUAL_COLOR), 5);
            note(p[0], p[1], p[2]);
        }
        if (pts.length === 2) add3dLine(pts, new THREE.Color(MANUAL_COLOR), 1);
    }

    if (nPts) {
        const t = new THREE.Vector3(centroid[0] / nPts, centroid[1] / nPts,
                                    centroid[2] / nPts);
        if (!three.framed) {
            // Frame the content once, then leave the camera to the user.
            // Triangulated points sit a few metres from the camera origin in
            // millimetres, so a fixed starting pose shows a speck.
            const radius = Math.max(
                Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, 30);
            three.camera.near = Math.max(radius / 100, 0.1);
            three.camera.far = radius * 200;
            three.camera.updateProjectionMatrix();
            // Look along -Z, the direction the cameras face, so the first
            // view matches what the 2D canvas shows.
            three.camera.position.set(t.x, t.y, t.z - radius * 3.2);
            three.controls.target.copy(t);
            three.framed = true;
        } else {
            // Re-centre gently afterwards: snapping the orbit target on every
            // frame during playback makes the view lurch.
            three.controls.target.lerp(t, 0.2);
        }
        three.controls.update();
    }
}

/* 3D for this session's manual labels.
 *
 * The saved layers arrive already triangulated, but manual points move as
 * you drag, so these are triangulated on demand by the server — the same
 * undistort-and-triangulate the stored coordinates came from, so the live
 * marker and the saved layer can't disagree. Cached per frame, and only
 * re-requested when a point actually moves. */
let manual3dCache = { key: null, points: {}, distance: null };
let manual3dPending = false;

function manualPointsForFrame() {
    if (cameraNames.length < 2) return null;
    const base = bpData.corrections || bpData.refine || bpData.dlc;
    const points = {};
    let any = false;
    for (const bp of bodyparts) {
        const cams = {};
        for (const cam of cameraNames.slice(0, 2)) {
            const m = manualAt(currentFrame, cam, bp);
            const b = base && base[cam] && (base[cam][bp] || [])[currentFrame];
            const c = m || b;
            if (c && c[0] != null) cams[cam] = [c[0], c[1]];
        }
        if (Object.keys(cams).length === 2) { points[bp] = cams; any = true; }
    }
    return any ? points : null;
}

function manual3dKey(points) {
    return currentFrame + '|' + JSON.stringify(points);
}

/* Returns whatever 3D we have for the manual layer now, and refreshes it in
 * the background when the points have changed. Drawing the previous answer
 * for one frame beats blocking the render on a round trip. */
function manual3dForFrame() {
    if (!hasCalibration) return [];
    const points = manualPointsForFrame();
    if (!points) { manual3dCache = { key: null, points: {}, distance: null }; return []; }

    const key = manual3dKey(points);
    if (key !== manual3dCache.key && !manual3dPending) {
        manual3dPending = true;
        api(`/api/labeling/sessions/${sessionId}/triangulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points }),
        }).then(r => {
            manual3dCache = { key, points: r.points || {},
                              distance: r.distance ?? null };
            if (show3d) update3d();
        }).catch(() => {
            // Leave the previous answer in place; a failed triangulation is
            // not worth clearing a correct marker over.
        }).finally(() => { manual3dPending = false; });
    }

    return bodyparts
        .map(bp => manual3dCache.points[bp])
        .filter(p => p && p[0] != null);
}

// ── Timeline ─────────────────────────────────────────────────────────────

function renderTimeline() {
    if (!timelineCanvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = timelineCanvas.clientWidth, h = timelineCanvas.clientHeight;
    if (!w || !h) return;
    timelineCanvas.width = w * dpr;
    timelineCanvas.height = h * dpr;
    const c = timelineCtx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);

    if (!totalFrames) return;
    const x = (f) => (f / Math.max(totalFrames - 1, 1)) * w;

    // Three stacked rows: trial names on top, then one tick row per camera,
    // then the refine picks.  Laid out from the top so the rows stay put
    // whatever height the container ends up.
    const nameY = 9;
    const rowH = 8;
    const rowTop = 16;
    const rowY = [rowTop, rowTop + rowH + 2];
    const refineY = rowTop + 2 * (rowH + 2);

    // Trial bands, so the whole subject reads as one strip.
    trials.forEach((t, i) => {
        c.fillStyle = (i % 2 === 0) ? 'rgba(255,255,255,0.035)'
                                    : 'rgba(255,255,255,0.015)';
        c.fillRect(x(t.start_frame), 0, x(t.end_frame) - x(t.start_frame), h);
        c.fillStyle = (i === currentTrialIdx) ? '#e0e0e0' : 'rgba(170,159,189,0.8)';
        c.font = '9px -apple-system, sans-serif';
        const name = t.trial_name.includes('_')
            ? t.trial_name.split('_').slice(1).join('_') : t.trial_name;
        c.fillText(name, x(t.start_frame) + 3, nameY);
    });

    // Per-camera label ticks: at a glance, which frames are labeled, and
    // whether both cameras have them — a frame labeled on one camera only
    // triangulates to nothing.
    c.font = '8px -apple-system, sans-serif';
    // With one camera there is nothing to compare against, so the row
    // labels would just be a column of names for a single row.
    if (!singleCamera()) {
        cameraNames.slice(0, 2).forEach((cam, i) => {
            c.fillStyle = 'rgba(170,159,189,0.55)';
            c.fillText(cam, 2, rowY[i] + rowH - 1);
        });
    }
    for (const key of manual.keys()) {
        const [f, ...sp] = key.split('_');
        const side = sp.join('_');
        const idx = sideIndex(side);
        if (idx > 1) continue;
        const lbl = manual.get(key);
        const complete = bodyparts.every(bp => lbl[bp] && lbl[bp][0] != null);
        c.fillStyle = complete ? '#ffffff' : 'rgba(255,255,255,0.4)';
        c.fillRect(x(parseInt(f, 10)) - 0.5, rowY[idx], 1.5, rowH);
    }

    if (mode === 'refine') {
        c.fillStyle = 'rgba(170,159,189,0.55)';
        c.fillText('train', 2, refineY + rowH - 1);
        c.fillStyle = '#ffd54f';
        for (const key of refineSelected) {
            const f = parseInt(key.split('_')[0], 10);
            c.fillRect(x(f) - 0.5, refineY, 1.5, rowH);
        }
    }

    // Playhead.
    c.strokeStyle = '#a195c0';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(x(currentFrame), 0);
    c.lineTo(x(currentFrame), h);
    c.stroke();
}

function timelineSeek(clientX) {
    const rect = timelineCanvas.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    gotoFrame(Math.round(frac * (totalFrames - 1)));
}

// ── Distance trace ───────────────────────────────────────────────────────

/* Which layers contribute a distance trace, and in what colour. */
function traceSeries() {
    const out = [];
    if (!mpTrial) return out;
    for (const spec of mpPassSpecs) {
        if (!vis(`mp_${spec.key}`).d2 && !vis(`mp_${spec.key}`).d3) continue;
        const p = mpPass(spec.key);
        if (!p) continue;
        const seq = p.distancesClean || p.distances;
        if (!seq) continue;
        out.push({ label: spec.label, color: spec.color, values: seq,
                   offset: mpTrial.start_frame });
    }
    for (const spec of BP_LAYERS) {
        if (!vis(spec.key).d2 && !vis(spec.key).d3) continue;
        const d = bpData[spec.key];
        if (!d || !d.distances) continue;
        out.push({ label: spec.label, color: spec.color, values: d.distances,
                   offset: 0 });
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

function toggleSide() {
    if (singleCamera()) return;
    currentSide = otherSide();
    updateSideToggle();
    setNavState({ side: currentSide });
    videoSideLoaded = null;   // force the video element onto the other half
    gotoFrame(currentFrame, { force: true });
}

function updateShortcuts() {
    el('shortcutList').innerHTML = [
        ['&larr; &rarr;', 'previous / next frame'],
        ['Shift + &larr;&rarr;', 'jump 10 frames'],
        ['&uarr; &darr;', 'previous / next labeled frame'],
        ['Space', 'play / pause'],
        ['E', 'switch camera'],
        ['D', 'show / hide 3D'],
        ['R', 'reset zoom'],
        ['1 … 9', 'jump to trial'],
        ['Ctrl/&#8984; Z', 'undo'],
        ['Ctrl/&#8984; &#8679; Z', 'redo'],
        ['Right-click', 'remove a point'],
        ...(mode === 'refine' ? [['T', 'tick this frame for training']] : []),
    ].map(([k, v]) => `<div><kbd>${k}</kbd> ${v}</div>`).join('');
}

// ── Events ───────────────────────────────────────────────────────────────

function onMouseDown(e) {
    if (e.button === 2) return;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

    if (boxMode && editBox) {
        const handle = boxHitTest(sx, sy);
        if (handle) {
            boxDragHandle = handle;
            boxDragStart = { mx: sx, my: sy, box: { ...editBox } };
            return;
        }
    }

    const hit = hitTest(sx, sy);
    if (hit) {
        const key = `${currentFrame}_${currentSide}`;
        if (hit.ghost) {
            // Taking over a prediction: adopt its position as a manual label
            // and go straight into dragging, so one gesture both accepts and
            // corrects it.
            pushUndo(key, hit.bp, null);
            setManual(hit.bp, hit.coords[0], hit.coords[1]);
            updateLabelCount();
            scheduleSave();
        }
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

    if (boxDragHandle) {
        applyBoxDrag(sx, sy);
        render();
        return;
    }
    if (boxMode && editBox && !dragging) {
        canvas.style.cursor = boxHitTest(sx, sy) ? 'move' : 'crosshair';
    }
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
    if (boxDragHandle) {
        boxDragHandle = null;
        boxDragStart = null;
        canvas.style.cursor = 'crosshair';
        return;
    }
    if (dragging === 'pending') {
        const img = screenToImage(dragStartX, dragStartY);
        if (!boxMode && img.x >= 0 && img.x < imgW && img.y >= 0 && img.y < imgH) {
            placeLabel(img.x, img.y);
        }
    } else if (dragging === 'pan') {
        hasUserZoom = true;
    } else if (dragging) {
        pushUndo(`${currentFrame}_${currentSide}`, dragging, [dragOrigX, dragOrigY]);
        dirtyKeys.add(`${currentFrame}_${currentSide}`);
        scheduleSave();
        update3d();
    }
    dragging = null;
    canvas.style.cursor = 'crosshair';
}

function onRightClick(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    if (hit && !hit.ghost) removeLabel(hit.bp);
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
        case 'ArrowUp': {
            e.preventDefault();
            const f = nextLabeledFrame(-1);
            if (f != null) gotoFrame(f);
            break;
        }
        case 'ArrowDown': {
            e.preventDefault();
            const f = nextLabeledFrame(1);
            if (f != null) gotoFrame(f);
            break;
        }
        case ' ': e.preventDefault(); togglePlay(); break;
        case 'e': case 'E': toggleSide(); break;
        case 'd': case 'D': toggle3d(); break;
        case 'r': case 'R': resetZoom(); break;
        case 't': case 'T':
            if (mode === 'refine') {
                const key = `${currentFrame}_${currentSide}`;
                if (refineSelected.has(key)) refineSelected.delete(key);
                else refineSelected.add(key);
                buildRefineList();
                renderTimeline();
            }
            break;
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
    timelineCanvas = el('timelineCanvas');
    timelineCtx = timelineCanvas.getContext('2d');
    traceCanvas = el('traceCanvas');
    traceCtx = traceCanvas.getContext('2d');

    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('contextmenu', onRightClick);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('keydown', onKeyDown);

    timelineCanvas.addEventListener('mousedown', (e) => timelineSeek(e.clientX));
    traceCanvas.addEventListener('mousedown', (e) => traceSeek(e.clientX));

    el('playBtn').addEventListener('click', togglePlay);
    el('resetZoomBtn').addEventListener('click', resetZoom);
    el('view3dBtn').addEventListener('click', () => toggle3d());
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
    el('autoZoomCb').addEventListener('change', resetZoom);
    el('boxZoomCb').addEventListener('change', resetZoom);
    el('editBoxBtn').addEventListener('click', startBoxEdit);
    el('cancelBoxBtn').addEventListener('click', endBoxEdit);
    el('saveBoxBtn').addEventListener('click', saveBox);
    el('redetectBtn').addEventListener('click', redetectTrial);
    el('commitBtn').addEventListener('click', commit);
    el('saveCorrBtn').addEventListener('click', saveCorrections);
    el('refineAllBtn').addEventListener('click', () => {
        correctedFrames().forEach(r => refineSelected.add(`${r.frame}_${r.side}`));
        buildRefineList();
        renderTimeline();
    });
    el('refineNoneBtn').addEventListener('click', () => {
        refineSelected.clear();
        buildRefineList();
        renderTimeline();
    });

    document.querySelectorAll('.mode-btn').forEach(b => {
        b.addEventListener('click', async () => {
            if (b.dataset.mode === mode) return;
            await flushSave();
            await openSession(b.dataset.mode);
            await gotoFrame(currentFrame, { force: true });
        });
    });

    new ResizeObserver(() => {
        if (!hasUserZoom) applyAutoZoom(currentFrame);
        render();
        renderTimeline();
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

return { gotoFrame, toggle3d, openSession };

})();

export default labeler;
