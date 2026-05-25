/**
 * Skeleton 3D Hand Model Viewer
 *
 * Canvas-based stereo video viewer with 2D overlay (Skeleton, MediaPipe, DLC, heatmaps),
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
    let cameraMode = 'stereo'; // 'single', 'stereo', or 'multicam'
    let playing = false;
    let playTimer = null;
    let playbackRate = 1;
    let _seekGeneration = 0; // incremented on each seek/play to invalidate stale seeked callbacks
    const SPEED_PRESETS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120];

    // Layer visibility
    let showVideo = true;
    let showSkeleton = true; // legacy, derived from per-model flags
    let showMPSkel = true;
    let showManoSkel = true;
    let showMP2D = false;
    let showMP3D = false;
    // Reverse-pass MediaPipe layer (mediapipe_reverse_prelabels.npz).
    // Same schema and skeleton as MP; drawn in magenta so the user
    // can compare forward vs reverse runs frame-by-frame.
    let showReverse2D = false;
    let showReverse3D = false;
    let showCropped2D = false;
    let showCropped3D = false;
    let showStatic2D = false;
    let showStatic3D = false;
    let showCombined2D = false;
    let showCombined3D = false;
    let showReverseSkel = true;
    let showCroppedSkel = true;
    let showStaticSkel = true;
    let showCombinedSkel = true;
    // Stereo (cross-camera image-alignment) overlay.  Only 2D —
    // there's no 3D representation; the partner-camera MP label is
    // translated into the current view by the per-frame per-joint
    // phase-correlation shift, drawn as pink crosses next to the cyan
    // MP markers.
    let showStereo2D = false;
    let stereoSelectedJoint = null;   // joint whose per-joint bbox to draw
    let stereoConfThreshold = 0;      // hide markers with response < this
    let showStereoOutline2D = false;  // outline-driven stereo variant
    let stereoOutlineConfThreshold = 0;  // same, for outline variant
    let showStereoHybrid2D = false;   // outline-vote Pass 1 + image Pass 2
    let stereoHybridConfThreshold = 0;
    let showOutline2D = false;        // per-frame hand boundary polygon
    let showVision2D = false;
    let showVision3D = false;
    let showVisionSkel = true;
    let showMano2D = false;
    let showMano3D = false;
    let showSkelV2_2D = false;
    let showSkelV2_3D = false;
    let showSkelV2Skel = true;
    let showLegacyV2_2D = false;
    let showLegacyV2_3D = false;
    let showLegacyV2Skel = true;
    // Error-display stage buttons under Skeleton row.  Each stage can be
    // toggled independently; when active, that stage is rendered on the
    // Skeleton layer with its own color + factor-specific errors.
    let _errorStages = new Set();   // union of _stages2D + _stages3D (auto-synced)
    let _stages2D = new Set();      // stages whose 2D overlay is shown
    let _stages3D = new Set();      // stages whose 3D model is shown
    let _stagesErr = new Set();     // stages whose error markers are shown
    const ALL_STAGES = ['mediapipe', 'stereo_correct', 'z_correct', 'z_smooth', 'hrnet_snap', 'bone_correct', 'bone_smooth'];
    function _syncErrorStages() {
        _errorStages = new Set([..._stages2D, ..._stages3D, ..._stagesErr]);
    }

    // ── Checkbox-state persistence ──────────────────────────────────
    // Snapshots every checkbox (by id), every stage row, and every
    // heatmap-row into localStorage on change.  On page load, the
    // saved state is restored so the user's last selection persists
    // across reloads.  First-ever load has no saved state and falls
    // through to the as-defined HTML defaults (all unchecked).
    const _CHECKBOX_STATE_KEY = 'manoCheckboxState_v1';
    let _checkboxRestoring = false;     // suppresses save during restore

    function _saveCheckboxes() {
        if (_checkboxRestoring) return;
        try {
            const ids = {};
            document.querySelectorAll('input[type="checkbox"][id]').forEach(cb => {
                ids[cb.id] = !!cb.checked;
            });
            const stages = {};
            document.querySelectorAll('input.stage-row[type="checkbox"][data-stage]').forEach(cb => {
                const col = cb.classList.contains('stage-2d') ? '2d'
                          : cb.classList.contains('stage-3d') ? '3d'
                          : cb.classList.contains('stage-err') ? 'err' : null;
                if (!col) return;
                stages[`${col}|${cb.dataset.stage}`] = !!cb.checked;
            });
            const hm = {};
            document.querySelectorAll('input.hm-row[type="checkbox"][data-hm]').forEach(cb => {
                const col = cb.classList.contains('hm-2d') ? '2d'
                          : cb.classList.contains('hm-3d') ? '3d' : null;
                if (!col) return;
                hm[`${col}|${cb.dataset.hm}`] = !!cb.checked;
            });
            localStorage.setItem(_CHECKBOX_STATE_KEY, JSON.stringify({
                ids, stages, hm,
                v3Expanded: !!_v3Expanded,
                heatmapExpanded: !!_heatmapExpanded,
            }));
        } catch {}
    }

    function _restoreCheckboxes() {
        let s = null;
        try { s = JSON.parse(localStorage.getItem(_CHECKBOX_STATE_KEY) || 'null'); }
        catch { s = null; }
        if (!s) return false;
        _checkboxRestoring = true;
        try {
            const touched = [];
            Object.entries(s.ids || {}).forEach(([id, val]) => {
                const el = document.getElementById(id);
                if (el && el.type === 'checkbox') {
                    el.checked = !!val;
                    touched.push(el);
                }
            });
            Object.entries(s.stages || {}).forEach(([key, val]) => {
                const [col, stage] = key.split('|');
                if (!col || !stage) return;
                const cb = document.querySelector(
                    `input.stage-row.stage-${col}[data-stage="${stage}"]`);
                if (cb) { cb.checked = !!val; touched.push(cb); }
            });
            Object.entries(s.hm || {}).forEach(([key, val]) => {
                const [col, stage] = key.split('|');
                if (!col || !stage) return;
                const cb = document.querySelector(
                    `input.hm-row.hm-${col}[data-hm="${stage}"]`);
                if (cb) { cb.checked = !!val; touched.push(cb); }
            });
            if (typeof s.v3Expanded === 'boolean') _v3Expanded = s.v3Expanded;
            if (typeof s.heatmapExpanded === 'boolean') _heatmapExpanded = s.heatmapExpanded;
            // Stage Sets are reseeded from the restored DOM (the
            // change handlers below will keep them in sync going
            // forward).
            _stages2D.clear(); _stages3D.clear(); _stagesErr.clear();
            document.querySelectorAll(
                'input.stage-row.stage-2d[type="checkbox"][data-stage]').forEach(cb => {
                if (cb.checked) _stages2D.add(cb.dataset.stage);
            });
            document.querySelectorAll(
                'input.stage-row.stage-3d[type="checkbox"][data-stage]').forEach(cb => {
                if (cb.checked) _stages3D.add(cb.dataset.stage);
            });
            document.querySelectorAll(
                'input.stage-row.stage-err[type="checkbox"][data-stage]').forEach(cb => {
                if (cb.checked) _stagesErr.add(cb.dataset.stage);
            });
            _syncErrorStages();
            // Dispatch change events so the existing per-checkbox
            // handlers sync the JS state variables (showXxx) for us.
            // Wrapped in try/catch because some handlers touch
            // trialData / canvas state that may not exist yet at
            // page-init time.
            touched.forEach(el => {
                try { el.dispatchEvent(new Event('change', { bubbles: true })); }
                catch (e) { /* handler not ready -- state will sync on next user click */ }
            });
        } finally {
            _checkboxRestoring = false;
        }
        return true;
    }
    let showMPErrors = false;      // kept as compatibility flag (always false now)
    let showSkelErrors = false;    // true if any stage is active
    let mpErrorMatrix = null;      // unused (kept for compatibility)
    let skelErrorMatrices = {};    // { stage_name: [N][21][2] errors }
    const STAGE_CONFIGS = {
        // Stage 0: raw MP, pre-corrections.  Shows the union of
        // Y-disparity and Z-outlier errors (the things stage 1 fixes).
        // Distance plots use MP data.
        mediapipe: {
            // MP-stage errors now come from the stereo-correction
            // distance (joints where MP and Stereo differ by more than
            // the user's Stereo distance slider, gated by confidence).
            // The y_disp / z_outlier factors move to the stereo_correct
            // stage below.
            factor: 'stereo_dist',
            color: '#00cccc',
            color3d: 0x00cccc,
            emissive3d: 0x008888,
            metricSrc: 'mp',
            proj2D: (isLeft, td) => isLeft ? td.mp_tracked_L : td.mp_tracked_R,
            proj3D: (td, fn) => td.mp_joints_3d?.[fn],
        },
        // Stage 0b: after stereo-correction (MP labels replaced with
        // the stereo label on the camera attributed as bad).  Shows
        // the union of Y-disparity and Z-outlier errors -- the things
        // the next pass (Y/Z-correct) fixes.
        stereo_correct: {
            factor: ['y_disp', 'z_outlier'],
            color: '#f48fb1',
            color3d: 0xf48fb1,
            emissive3d: 0x7a4858,
            metricSrc: 'skel_v2_sc',
            proj2D: (isLeft, td) => isLeft ? (td.skel_v2_proj_sc_L || td.mp_tracked_L)
                                            : (td.skel_v2_proj_sc_R || td.mp_tracked_R),
            proj3D: (td, fn) => (td.skel_v2_joints_sc_3d || td.mp_joints_3d)?.[fn],
        },
        // Stage 1: after combined Y-disparity + Z-outlier correction.
        // Shows z_jump errors (frames still jumpy after Y+Z lateral fix —
        // cleaned up by the next correction pass).
        z_correct: {
            factor: 'z_jump',
            color: '#ffa726',   // lighter orange — distinct from final
            color3d: 0xffa726,
            emissive3d: 0x80501c,
            metricSrc: 'skel_v2_z',
            proj2D: (isLeft, td) => isLeft ? (td.skel_v2_proj_z_L || td.skel_v2_proj_L)
                                            : (td.skel_v2_proj_z_R || td.skel_v2_proj_R),
            proj3D: (td, fn) => (td.skel_v2_joints_z_3d || td.skel_v2_joints_3d)?.[fn],
        },
        // Stage 3: after Y + Z-outlier + Z-jump (pre-HRnet).  Red markers
        // flag joints whose MP label disagrees with the HRnet peak
        // (after systematic-offset correction) by more than the HRnet-
        // peak-distance slider threshold.
        z_smooth: {
            factor: 'hrnet_mismatch',
            color: '#ff9800',   // orange
            color3d: 0xff9800,
            emissive3d: 0x804400,
            metricSrc: 'skel_v2_zs',
            proj2D: (isLeft, td) => isLeft ? (td.skel_v2_proj_zs_L || td.skel_v2_proj_L)
                                            : (td.skel_v2_proj_zs_R || td.skel_v2_proj_R),
            proj3D: (td, fn) => (td.skel_v2_joints_zs_3d || td.skel_v2_joints_3d)?.[fn],
        },
        // Stage 4: after HRnet snap + re-clean (pre-BL).  Red markers
        // come from the regular detection factors on this snapshot.
        hrnet_snap: {
            factor: 'bone_length',
            color: '#ab47bc',   // purple — distinct mid-pipeline stage
            color3d: 0xab47bc,
            emissive3d: 0x4a148c,
            metricSrc: 'skel_v2_hr',
            proj2D: (isLeft, td) => isLeft ? (td.skel_v2_proj_hr_L || td.skel_v2_proj_L)
                                            : (td.skel_v2_proj_hr_R || td.skel_v2_proj_R),
            proj3D: (td, fn) => (td.skel_v2_joints_hr_3d || td.skel_v2_joints_3d)?.[fn],
        },
        // Stage 5: after bone-length correction.  Shows bone-length-jump
        // errors (sudden frame-to-frame length changes).
        bone_correct: {
            factor: 'bone_agreement',
            color: '#66bb6a',
            color3d: 0x66bb6a,
            emissive3d: 0x1b5e20,
            metricSrc: 'skel_v2_bc',
            proj2D: (isLeft, td) => isLeft ? (td.skel_v2_proj_bc_L || td.skel_v2_proj_L)
                                            : (td.skel_v2_proj_bc_R || td.skel_v2_proj_R),
            proj3D: (td, fn) => (td.skel_v2_joints_bc_3d || td.skel_v2_joints_3d)?.[fn],
        },
        // Stage 6: after bone-length-jump smoothing (final).  Shows
        // joint-angle constraint violations.
        bone_smooth: {
            factor: 'angle',
            color: '#26c6da',   // teal
            color3d: 0x26c6da,
            emissive3d: 0x0d5963,
            metricSrc: 'skel_v2',
            proj2D: (isLeft, td) => isLeft ? td.skel_v2_proj_L : td.skel_v2_proj_R,
            proj3D: (td, fn) => td.skel_v2_joints_3d?.[fn],
        },
    };
    let _mpErrorWeights = {
        detection: { z_jump: 0, z_outlier: 0, y_disp: 0, bone_length: 0, bone_agreement: 0, angle: 0, reproj: 0, confidence: 0, hrnet_mismatch: 0, stereo_dist: 0, stereo_conf: 0, stereo_occlusion: 0 },
        attribution: { jump_2d: 0, confidence: 0 },
        corrections: { y_disp: 0 },
        // Stereo-correction (v3 step 0) bake-only settings.
        stereo: { mode: 'image', mask_dilate_px: 10, gauss_center_weight: 0.0 },
    };
    let mpCorrectedL = null;  // (N, 21, 2) or null — used in place of mp_tracked_L when present
    let mpCorrectedR = null;
    let _mpErrorRecomputeTimer = null;
    let _mpErrorPending = false;
    let _xScale = 1.0;  // distance plot X scale (1 = fit full trial, >1 scrolls)
    // Per-bone per-frame flags for the z_smooth stage, keyed by "a-b"
    // (a<b).  Rebuilt by _computeBoneLengthFlags() whenever inputs change;
    // consumed by 2D/3D bone rendering + distance plot.
    let _boneFlagsByPair = null;
    let _constraintFocusMetric = null; // metric name whose constraints are visible (e.g., "Flex: Index MCP"), null = hidden
    let _lastAutoFitKey = ''; // tracks which metrics were plotted last auto-fit
    // Temporary constraint overrides from dragging on the plot (not saved until editor opened)
    const _constraintOverrides = {}; // { "Thumb CMC": {flex_min, flex_max, abd_min, abd_max}, ... }
    let _constraintHitZones = []; // rebuilt each renderDistanceTrace()
    let _plotMetricData = {}; // { metricName: { data, toY } } — rebuilt each render for click hit-testing
    let showDLC = false;
    let showDLC3D = false;
    let showPose2D = false;
    let showPose3D = false;
    let showPrev2D = false;
    let showPrev3D = false;
    let showPrevSkel = true;
    let prevFitData = null; // loaded historical fit data
    let showHeatmap = false;
    // Peak Select removed — these flags are kept as inert false constants
    // so any remaining display references short-circuit cleanly.
    let showHeatmap2D = false;
    let showHeatmap3D = false;
    let showHRnet2D = false;       // HRnet "Peaks" sub-stage 2D (cluster centroid)
    let showHRnet3D = false;       // HRnet "Peaks" sub-stage 3D
    let showStereoHun2D = false;   // HRnet "Stereo-Hungarian" sub-stage 2D
    let showStereoHun3D = false;   // HRnet "Stereo-Hungarian" sub-stage 3D
    let showHRnetYZC2D = false;    // HRnet "Y/Z-correct" sub-stage 2D
    let showHRnetYZC3D = false;    // HRnet "Y/Z-correct" sub-stage 3D
    let showHRnetZSM2D = false;    // HRnet "Z-smooth" sub-stage 2D
    let showHRnetZSM3D = false;    // HRnet "Z-smooth" sub-stage 3D
    let showHRnetCorrections = false; // Live overlay of Y/Z-correct flags on Peaks
    // Lookup wrapper around the sparse preview response so per-frame
    // rendering paths can ask `hasYZ(f, j, c)` / `pred(f, j, cam)` in
    // O(1) without ever materialising dense (N, 21, 2) arrays in JS.
    let hrnetPreview = null;
    function _buildHRnetPreviewLookup(data) {
        if (!data || !data.available) return null;
        const yErr = new Set();
        for (const t of (data.y_disp_errors || [])) yErr.add(t[0] * 100 + t[1] * 4 + t[2]);
        const zErr = new Set();
        for (const t of (data.z_outlier_errors || [])) zErr.add(t[0] * 100 + t[1] * 4 + t[2]);
        // Per-frame (frame → set of joints with any error on either cam).
        // Built once at receipt so the plot's red-line pass is O(errors).
        const flaggedJointsByFrame = new Map();
        const _addFJ = (f, j) => {
            let s = flaggedJointsByFrame.get(f);
            if (!s) { s = new Set(); flaggedJointsByFrame.set(f, s); }
            s.add(j);
        };
        for (const t of (data.y_disp_errors || [])) _addFJ(t[0], t[1]);
        for (const t of (data.z_outlier_errors || [])) _addFJ(t[0], t[1]);
        const predL = new Map();
        for (const r of (data.pred_L || [])) predL.set(r[0] * 32 + r[1], [r[2], r[3]]);
        const predR = new Map();
        for (const r of (data.pred_R || [])) predR.set(r[0] * 32 + r[1], [r[2], r[3]]);
        return {
            available: true,
            z_outlier_bounds: data.z_outlier_bounds || {},
            n_frames: data.n_frames | 0,
            flaggedJointsByFrame,
            hasYZ: (f, j, c) => yErr.has(f * 100 + j * 4 + c),
            hasZO: (f, j, c) => zErr.has(f * 100 + j * 4 + c),
            pred:  (f, j, cam) => (cam === 0 ? predL : predR).get(f * 32 + j) || null,
        };
    }
    let _hrnetPreviewToken = 0;    // race-guard for in-flight preview fetches
    let _hrnetPreviewPending = null;
    let _heatmapExpanded = false;  // sub-stage row visibility
    let _v3Expanded = false;       // v3 stage-row visibility
    // Snapshots of stage selections taken when a model collapses,
    // restored when it expands.  Collapsing clears all of the model's
    // stage checkboxes; expanding pulls them back from the snapshot.
    let _v3SavedStages = null;
    let _heatmapSavedState = null;
    let showHeatmapSkel = true;
    // Legacy aliases used in rendering
    let show3D = true; // always true — 3D viewport visibility
    let showMano = false; // derived from showMano2D || showMano3D
    let showMP = true; // derived from showMP2D || showMP3D

    // Finger visibility
    const fingerVisibility = {
        wrist: true, thumb: true, index: true,
        middle: true, ring: true, pinky: true,
    };
    let visibleJoints = new Set([...Array(21).keys()]);

    // Heatmap
    // Default selection: index fingertip (joint 8) — first joint shown when
    // the heat map is first turned on.  Persisted across on/off toggles.
    // Default: no joints selected.  Clicking a joint on the sidebar hand
    // activates the heat map for that joint (and dims the threshold slider
    // back when the user clicks it off again).
    let heatmapActiveJoints = new Set();
    // Anatomical-offset arrow overlay (button below MIP).  When on, each
    // displayed Skeleton v3 stage gets arrows pointing from each joint to
    // its expected HRnet peak (the per-joint median offset).
    let showOffsetArrows = false;
    let heatmapMipMode = false; // when true, show MIP of all joints regardless of selection
    let _savedHeatmapJoints = null; // saved selection before MIP mode
    let heatmapThreshold = 0.1;
    const heatmapCache = {};
    let _heatmapImageData = null; // pre-rendered ImageData for current frame

    // Selected metrics: Map of metric_name → color
    const METRIC_COLORS = ['#00e5ff','#69f0ae','#ffab40','#ea80fc','#ff6e40','#40c4ff','#f9a825','#ff80ab'];
    let selectedMetrics = new Map([['Thumb-Index Aperture', METRIC_COLORS[0]]]);
    // Map of joint_idx → { mode:'flex'|'abd'|'both', colorFlex, colorAbd }
    const plotJointStates = new Map();
    let wristPanelOpen = false;
    // Finger spread names and their required joints (wrist + two MCPs)
    const SPREAD_NAMES = ['Spread 1', 'Spread 2', 'Spread 3', 'Spread 4'];
    const SPREAD_JOINT_PAIRS = [[0, 2, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17]];
    const COORD_NAMES = ['Wrist X', 'Wrist Y', 'Wrist Z'];
    let plotMode = 'angle'; // 'angle' or 'position'
    const posJointStates = new Map(); // joint_idx → {colorX, colorY, colorZ}
    // Maps joint index to its finger's metacarpal anchor [proximal, distal]
    const JOINT_FINGER_ANCHOR_JS = {1:[0,1],2:[0,1],3:[0,1],5:[0,5],6:[0,5],7:[0,5],9:[0,9],10:[0,9],11:[0,9],13:[0,13],14:[0,13],15:[0,13],17:[0,17],18:[0,17],19:[0,17]};

    let _updatePlotHighlight = () => {}; // assigned inside setupControls
    let _applyHRnetFitParams = () => {}; // assigned inside setupControls
    let _scheduleHRnetPreview = () => {}; // debounced refresh
    let _refreshHRnetPreview = () => {};  // immediate refresh
    let _hasRemoteServer = false;
    const _activeEventSources = new Set(); // track open SSE connections
    // Check once at init whether a remote server is configured
    (async () => {
        try {
            const s = await api('/api/settings');
            _hasRemoteServer = !!(s.remote_host && s.remote_host.trim());
        } catch {}
    })();

    /** Submit a job through the queue manager. */
    async function _submitViaQueue(jobType, extraParams = {}, target = 'local-cpu') {
        // Always include trial_name for display in queue/history badges
        const trial = trials[currentTrialIdx];
        if (trial && !extraParams.trial_name) {
            const stem = trial.trial_stem || '';
            extraParams.trial_name = stem.includes('_') ? stem.split('_').slice(1).join('_') : stem;
        }
        return api('/api/remote/launch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_type: jobType,
                subject_ids: [subjectId],
                execution_target: target,
                extra_params: extraParams,
            }),
        });
    }

    /**
     * Show inline "Local | Remote" buttons inside a Run button's space.
     * If no remote server, skip and resolve with 'local-cpu' immediately.
     * Returns a promise that resolves with the chosen target.
     */
    function _promptExecTarget(runBtn) {
        return new Promise(resolve => {
            if (!_hasRemoteServer) {
                resolve('local-cpu');
                return;
            }
            const origHTML = runBtn.innerHTML;
            const origOnclick = runBtn.onclick;
            runBtn.onclick = null;
            runBtn.style.padding = '0';
            runBtn.innerHTML = `
                <span style="display:flex;gap:2px;width:100%;height:100%;">
                    <button class="btn btn-sm btn-success" style="flex:1;font-size:10px;border-radius:3px 0 0 3px;">Local</button>
                    <button class="btn btn-sm btn-success" style="flex:1;font-size:10px;border-radius:0 3px 3px 0;">Remote</button>
                </span>
            `;
            const [localBtn, remoteBtn] = runBtn.querySelectorAll('button');
            const cleanup = () => { runBtn.innerHTML = origHTML; runBtn.onclick = origOnclick; runBtn.style.padding = ''; };
            localBtn.addEventListener('click', e => { e.stopPropagation(); cleanup(); resolve('local-cpu'); });
            remoteBtn.addEventListener('click', e => { e.stopPropagation(); cleanup(); resolve('remote'); });
        });
    }

    function _nextMetricColor() {
        const used = new Set(selectedMetrics.values());
        return METRIC_COLORS.find(c => !used.has(c)) || METRIC_COLORS[selectedMetrics.size % METRIC_COLORS.length];
    }

    const _TIA = 'Thumb-Index Aperture';
    let _tiaIsDefault = true; // true = aperture was auto-set as default, not user-selected
    /** Auto-manage Thumb-Index Aperture: remove the default aperture when
     *  other metrics are added; restore it when everything is removed.
     *  User-selected aperture (via clicking the SVG line) is never auto-removed. */
    function _autoManageAperture() {
        const hasOther = [...selectedMetrics.keys()].some(k => k !== _TIA);
        if (hasOther && _tiaIsDefault) {
            selectedMetrics.delete(_TIA);
        } else if (selectedMetrics.size === 0) {
            selectedMetrics.set(_TIA, METRIC_COLORS[0]);
            _tiaIsDefault = true;
        }
    }

    // Visual approximation of cv2.dilate on a closed outline polygon:
    // push each vertex along its bisector by ``dilatePx`` so the result
    // is roughly the offset polygon.  Used to mirror the backend's
    // hybrid-mode mask dilation in the canvas overlay.  Winding-order
    // independent (sign-adjusted via the signed area).
    function _dilatePolygon(poly, dilatePx) {
        if (!poly || poly.length < 3 || dilatePx <= 0) return poly;
        const n = poly.length;
        // Signed area in canvas coords (y-down): positive = CCW visually.
        // Outward normal of edge (e.x, e.y) is then (+e.y, -e.x); flip
        // for CW.
        let area2 = 0;
        for (let i = 0; i < n; i++) {
            const a = poly[i], b = poly[(i + 1) % n];
            area2 += a[0] * b[1] - b[0] * a[1];
        }
        const sign = area2 > 0 ? 1 : -1;
        const out = new Array(n);
        for (let i = 0; i < n; i++) {
            const prev = poly[(i - 1 + n) % n];
            const cur  = poly[i];
            const next = poly[(i + 1) % n];
            const e1x = cur[0] - prev[0], e1y = cur[1] - prev[1];
            const e2x = next[0] - cur[0], e2y = next[1] - cur[1];
            const l1 = Math.hypot(e1x, e1y) || 1;
            const l2 = Math.hypot(e2x, e2y) || 1;
            // Outward unit normals.
            const n1x =  sign * e1y / l1, n1y = -sign * e1x / l1;
            const n2x =  sign * e2y / l2, n2y = -sign * e2x / l2;
            const sx = n1x + n2x, sy = n1y + n2y;
            const sLen = Math.hypot(sx, sy);
            if (sLen < 1e-6) {
                out[i] = [cur[0] + dilatePx * n1x, cur[1] + dilatePx * n1y];
                continue;
            }
            // Bisector offset = dilatePx / cos(theta/2) where
            // cos(theta/2) = |n1 + n2| / 2.  Cap at 4*dilatePx to avoid
            // huge spikes at very sharp convex corners.
            const cosHalf = sLen / 2;
            const offDist = dilatePx / Math.max(cosHalf, 0.25);
            out[i] = [cur[0] + offDist * sx / sLen,
                      cur[1] + offDist * sy / sLen];
        }
        return out;
    }

    // When neither 2D nor 3D is checked for a model, force Skel checked + inactive
    function _syncSkelCheckbox(id2D, id3D, idSkel, skelVar) {
        const e2D = $(id2D), e3D = $(id3D), eSkel = $(idSkel);
        if (!eSkel) return skelVar;
        const hasView = (e2D && e2D.checked) || (e3D && e3D.checked);
        if (!hasView) {
            eSkel.checked = true;
            eSkel.disabled = true;
            eSkel.style.opacity = '0.35';
            return true;
        } else {
            eSkel.disabled = false;
            eSkel.style.opacity = '';
            return eSkel.checked;
        }
    }

    function _syncAllSkelCheckboxes() {
        showMPSkel       = _syncSkelCheckbox('showMP2D',        'showMP3D',        'showMPSkel',       showMPSkel);
        showManoSkel     = _syncSkelCheckbox('showMano2D',      'showMano3D',      'showManoSkel',     showManoSkel);
        showSkelV2Skel   = _syncSkelCheckbox('showSkelV2_2D',   'showSkelV2_3D',   'showSkelV2Skel',   showSkelV2Skel);
        showLegacyV2Skel = _syncSkelCheckbox('showLegacyV2_2D', 'showLegacyV2_3D', 'showLegacyV2Skel', showLegacyV2Skel);
        showVisionSkel   = _syncSkelCheckbox('showVision2D',    'showVision3D',    'showVisionSkel',   showVisionSkel);
    }

    // Canvas
    let canvas, ctx;
    let distCanvas, distCtx;
    let videoEl;

    // Zoom/pan (in video-pixel space; render maps to canvas)
    let scale = 1, offsetX = 0, offsetY = 0;
    let defaultScale = 1, defaultOX = 0, defaultOY = 0; // auto-crop defaults
    let _userZoom = false; // true when user has manually zoomed/panned
    let trackingZoom = true; // on by default
    let _trackSegments = []; // pre-planned zoom path: [{startFrame, endFrame, scale, offsetX, offsetY}]
    let _trackCurrentSeg = -1;
    let dragging = null;
    let dragStartX = 0, dragStartY = 0;
    let panStartOX = 0, panStartOY = 0;

    // Video dimensions
    let vidW = 0, vidH = 0, midline = 0;

    // Three.js
    let scene, camera3d, renderer;
    let manoGroup, skelV2Group, legacyGroup, mpGroup, croppedGroup, reverseGroup, staticGroup, combinedGroup, visionGroup, dlcGroup, poseGroup, heatmapGroup, angleArcGroup;
    let camera3dInit = false;

    // Scene-space orbit: rotate content around hand center while camera stays fixed
    let orbitQuat = new THREE.Quaternion();
    let orbitPivot = new THREE.Vector3();
    let orbitDragging = false;
    let orbitLastX = 0, orbitLastY = 0;

    // Per-model 3D translation offsets (persist across frames)
    const modelTranslations = {
        skeleton: new THREE.Vector3(),
        skel_v2: new THREE.Vector3(),
        mp: new THREE.Vector3(),
        vision: new THREE.Vector3(),
        dlc: new THREE.Vector3(),
    };
    let _translateDragging = false;
    let _translateTarget = null; // 'skeleton', 'mp', 'vision', 'dlc'
    let _translateLastX = 0, _translateLastY = 0;

    // Video ready state — prevents drawing labels before video frame is loaded
    let _videoReady = false;
    let _pendingFrame = null;

    // Projection auto-correction (measured offset between 3D projection and 2D overlay)
    let _projCorrNdcX = 0, _projCorrNdcY = 0;
    let _projCorrComputed = false;

    // ── Helpers ───────────────────────────────────────────────
    const $ = id => document.getElementById(id);

    function _trialHasHeatmaps() {
        return !!(trialData && trialData.has_heatmaps);
    }

    function _refreshHeatmapState() {
        // Single source of truth for the heat-map UI.  Called whenever
        // the joint selection, MIP mode, or trial changes.
        //   - showHeatmap = data available AND (≥1 joint active OR MIP).
        //   - Threshold row hidden entirely when no data, dimmed when
        //     data exists but no joints are active.
        const hasData = _trialHasHeatmaps();
        const anySelected = heatmapMipMode || heatmapActiveJoints.size > 0;
        showHeatmap = hasData && anySelected;
        const row = $('heatmapThreshRow');
        if (row) row.style.display = hasData ? '' : 'none';
        const slider = $('heatmapThreshSlider');
        const valLbl = $('heatmapThreshVal');
        const lblTxt = $('heatmapThreshLabel');
        const live = hasData && anySelected;
        if (slider) {
            slider.disabled = !live;
            slider.style.opacity = live ? '1' : '0.4';
        }
        if (valLbl) valLbl.style.opacity = live ? '1' : '0.4';
        if (lblTxt) lblTxt.style.opacity = live ? '1' : '0.4';
        _updateHeatmapHighlight();
        render(); update3D(); renderDistanceTrace();
    }

    function _updateHeatmapHighlight() {
        // Joint highlights only show when the heat map is active.
        // Selection state is preserved internally so it returns when
        // the user reactivates a joint.  MIP button is shown whenever
        // the trial has HRnet heatmaps, independent of selection.
        document.querySelectorAll('#handDiagramLabels .joint').forEach(j => {
            const jIdx = parseInt(j.dataset.joint);
            const active = showHeatmap && (heatmapMipMode || heatmapActiveJoints.has(jIdx));
            j.classList.toggle('heatmap-active', active);
        });
        const mipBtn = $('heatmapMipBtn');
        if (mipBtn) {
            mipBtn.classList.toggle('active', heatmapMipMode);
            mipBtn.style.display = _trialHasHeatmaps() ? '' : 'none';
        }
    }

    async function api(url, options) {
        const resp = await fetch(url, options);
        if (!resp.ok) {
            let msg = `${resp.status} ${resp.statusText}`;
            try {
                const body = await resp.json();
                if (body.detail) msg = body.detail;
            } catch {}
            throw new Error(msg);
        }
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
    function _wirePlotResizer() {
        const handle = document.getElementById('plotResizer');
        const row    = document.getElementById('plotFlexRow');
        if (!handle || !row) return;
        let startY = 0, startH = 0, pointerId = null;
        const onMove = (e) => {
            if (pointerId === null) return;
            // Drag up → more plot height (handle above row, so y decreases = row grows)
            const delta = startY - e.clientY;
            const next = Math.max(40, Math.min(window.innerHeight - 120, startH + delta));
            row.style.setProperty('--plot-h', next + 'px');
            row.style.height = next + 'px';
            if (typeof sizeCanvases === 'function') sizeCanvases();
            if (typeof renderDistanceTrace === 'function') renderDistanceTrace();
        };
        const onUp = (e) => {
            if (pointerId === null) return;
            try { handle.releasePointerCapture(pointerId); } catch {}
            pointerId = null;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
        handle.addEventListener('pointerdown', (e) => {
            pointerId = e.pointerId;
            startY = e.clientY;
            startH = row.getBoundingClientRect().height;
            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';
            try { handle.setPointerCapture(e.pointerId); } catch {}
            e.preventDefault();
        });
        handle.addEventListener('pointermove', onMove);
        handle.addEventListener('pointerup', onUp);
        handle.addEventListener('pointercancel', onUp);
    }

    async function init() {
        _wirePlotResizer();
        // Relocate the Plotting section into its pinned container so it
        // stays at the bottom of the sidebar regardless of scroll.
        const pinned = document.getElementById('plottingPinned');
        if (pinned) {
            const all = document.querySelectorAll('#manoSidebar *');
            let inSection = false;
            for (const node of all) {
                if (node.nodeType !== 1) continue;
            }
            // Move everything from the start marker to the end marker
            const root = document.getElementById('manoSidebar');
            if (root) {
                const walker = document.createTreeWalker(root, NodeFilter.SHOW_COMMENT);
                let start = null, end = null;
                while (walker.nextNode()) {
                    const t = walker.currentNode.nodeValue.trim();
                    if (t === 'PLOTTING_SECTION_START') start = walker.currentNode;
                    else if (t === 'PLOTTING_SECTION_END') end = walker.currentNode;
                }
                if (start && end) {
                    let node = start.nextSibling;
                    while (node && node !== end) {
                        const next = node.nextSibling;
                        pinned.appendChild(node);
                        node = next;
                    }
                }
            }
        }

        // Parse subject from URL
        const params = new URLSearchParams(window.location.search);
        const subjectParam = params.get('subject');

        // Load camera names from settings (camera mode is per-subject)
        try {
            const cfg = await api('/api/settings');
            if (cfg.default_camera_mode) cameraMode = cfg.default_camera_mode;
            if (Array.isArray(cfg.camera_names) && cfg.camera_names.length >= 1) {
                cameraNames = cfg.camera_names;
                currentSide = cameraNames[0];
            }
            // Initialise the camera toggle with the actual names + active highlighting
            _renderDualToggle('sideToggle', cameraNames[0], cameraNames[1] || cameraNames[0], currentSide);
        } catch { /* defaults */ }

        // Load subjects
        allSubjects = await api('/api/subjects');
        const sel = $('subjectSelect');
        sel.innerHTML = '';
        allSubjects.forEach(s => {
            const opt = document.createElement('option');
            opt.value = String(s.id);
            opt.textContent = s.name;
            sel.appendChild(opt);
        });

        const nav = typeof getNavState === 'function' ? getNavState() : {};
        const savedSubject = subjectParam || (nav.subjectId ? String(nav.subjectId) : null)
            || sessionStorage.getItem('dlc_lastSubjectId');
        if (savedSubject && allSubjects.some(s => String(s.id) === savedSubject)) {
            sel.value = savedSubject;
        } else if (allSubjects.length) {
            sel.value = String(allSubjects[0].id);
        }

        sel.addEventListener('change', () => { loadSubject(parseInt(sel.value)); sel.blur(); });

        // Setup canvases
        canvas = $('videoCanvas');
        ctx = canvas.getContext('2d');
        distCanvas = $('distanceCanvas');
        distCtx = distCanvas.getContext('2d');
        videoEl = null; // created per-trial in loadTrial

        // Disable side toggle for non-stereo modes
        if (cameraMode !== 'stereo') {
            const btn = $('sideToggle');
            if (btn) { btn.disabled = true; btn.title = 'Single camera'; }
        }

        // Setup Three.js
        setup3D();

        // Setup controls
        setupControls();
        setupCanvasEvents();
        setupFitSliders();
        setupDetectionButtons();
        _updateTrackBtn(); // reflect default tracking state

        // Fit Skeleton v1 expand/collapse — toggle panel, highlight button
        // When opening, restore sliders from the saved fit params (if a fit exists)
        $('fitSkeletonBtn').addEventListener('click', () => {
            const open = $('fitOptionsPanel').style.display !== 'block';
            $('fitOptionsPanel').style.display = open ? 'block' : 'none';
            $('fitSkeletonBtn').classList.toggle('active', open);
            if (open) {
                // Close v2 panel
                $('fitV2Panel').style.display = 'none';
                $('fitSkeletonV2Btn').classList.remove('active');
                _restoreV1Params();
            }
        });
        $('fitCancelBtn').addEventListener('click', () => {
            $('fitOptionsPanel').style.display = 'none';
            $('fitSkeletonBtn').classList.remove('active');
        });

        // Fit Skeleton v2 expand/collapse — toggle panel, highlight button
        // When opening, restore sliders from the saved fit params (if a fit exists)
        $('fitSkeletonV2Btn').addEventListener('click', () => {
            const open = $('fitV2Panel').style.display !== 'block';
            $('fitV2Panel').style.display = open ? 'block' : 'none';
            $('fitSkeletonV2Btn').classList.toggle('active', open);
            if (open) {
                // Close v1 panel
                $('fitOptionsPanel').style.display = 'none';
                $('fitSkeletonBtn').classList.remove('active');
                _restoreV2Params();
            }
            render();  // refresh the Occlusion-radius circle overlay
        });
        $('fitV2CancelBtn').addEventListener('click', () => {
            $('fitV2Panel').style.display = 'none';
            $('fitSkeletonV2Btn').classList.remove('active');
            render();  // hide the Occlusion-radius circles
        });

        // HRnet Fit panel — same toggle pattern as Skeleton Fit (v3).
        const HRNET_FIT_DEFAULTS = {
            hfSliderCluster: 1,
            hfSliderSpike:   0,
            hfSliderEdge:    0,
            hfSliderHM:      1,
            hfSliderOverlap: 8,
            hfSliderYZ_y:    0,
            hfSliderYZ_zo:   0,
            hfSliderYZ_aJ:   0,
            hfSliderYZ_aC:   0,
            hfSliderYZ_zm:   0,
            hfSliderZS:      0,
            hfSliderZSWin:   15,
            hfSliderBoneThr: 0,
            hfSliderBoneK:   8,
            hfSliderBoneW:   1,
            hfSliderWristSm: 5,
        };
        const _hfPairs = [
            ['hfSliderCluster', 'hfWCluster', 0],
            ['hfSliderSpike',   'hfWSpike',   2],
            ['hfSliderEdge',    'hfWEdge',    0],
            ['hfSliderHM',      'hfWHM',      2],
            ['hfSliderOverlap', 'hfWOverlap', 0],
            ['hfSliderYZ_y',    'hfWYZ_y',    2],
            ['hfSliderYZ_zo',   'hfWYZ_zo',   2],
            ['hfSliderYZ_aJ',   'hfWYZ_aJ',   2],
            ['hfSliderYZ_aC',   'hfWYZ_aC',   2],
            ['hfSliderYZ_zm',   'hfWYZ_zm',   0],
            ['hfSliderZS',      'hfWZS',      2],
            ['hfSliderZSWin',   'hfWZSWin',   0],
            ['hfSliderBoneThr', 'hfWBoneThr', 1],
            ['hfSliderBoneK',   'hfWBoneK',   0],
            ['hfSliderBoneW',   'hfWBoneW',   2],
            ['hfSliderWristSm', 'hfWWristSm', 0],
        ];
        // ── Live Y/Z-correct preview (Peaks "Corrections" overlay) ─────
        _refreshHRnetPreview = async function() {
            if (!subjectId || currentTrialIdx < 0 || !trials?.[currentTrialIdx]) return;
            const myToken = ++_hrnetPreviewToken;
            const body = {
                trial_idx:     trials[currentTrialIdx].trial_idx,
                cluster_size:  parseInt($('hfSliderCluster')?.value ?? 1),
                spike_support: parseFloat($('hfSliderSpike')?.value ?? 0),
                edge_margin:   parseInt($('hfSliderEdge')?.value ?? 0),
                yzc_y_disp:    parseFloat($('hfSliderYZ_y')?.value ?? 0),
                yzc_z_outlier: parseFloat($('hfSliderYZ_zo')?.value ?? 0),
                yzc_attr_jump: parseFloat($('hfSliderYZ_aJ')?.value ?? 0),
                yzc_attr_auc:  parseFloat($('hfSliderYZ_aC')?.value ?? 0),
                yzc_z_median_mm: parseFloat($('hfSliderYZ_zm')?.value ?? 0),
            };
            try {
                const res = await fetch(`/api/skeleton/${subjectId}/hrnet_correct_preview`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                if (myToken !== _hrnetPreviewToken) return; // a newer call superseded us
                if (!res.ok) { hrnetPreview = null; render(); return; }
                const data = await res.json();
                if (myToken !== _hrnetPreviewToken) return;
                hrnetPreview = data?.available ? _buildHRnetPreviewLookup(data) : null;
                render();
                renderDistanceTrace();
            } catch (e) {
                if (myToken === _hrnetPreviewToken) {
                    hrnetPreview = null;
                    render();
                }
            }
        };
        _scheduleHRnetPreview = function() {
            if (_hrnetPreviewPending) clearTimeout(_hrnetPreviewPending);
            _hrnetPreviewPending = setTimeout(() => {
                _hrnetPreviewPending = null;
                _refreshHRnetPreview();
            }, 350);
        };

        // Slider IDs that should refresh the live Y/Z-correct overlay
        // when the Corrections checkbox is on.
        const _HF_PREVIEW_SLIDERS = new Set([
            'hfSliderCluster', 'hfSliderSpike', 'hfSliderEdge',
            'hfSliderYZ_y',    'hfSliderYZ_zo', 'hfSliderYZ_zm',
            'hfSliderYZ_aJ',   'hfSliderYZ_aC',
        ]);
        for (const [sid, lid, dec] of _hfPairs) {
            const s = $(sid); const lbl = $(lid);
            if (!s) continue;
            const upd = () => { if (lbl) lbl.textContent = parseFloat(s.value).toFixed(dec); };
            s.addEventListener('input', upd);
            if (_HF_PREVIEW_SLIDERS.has(sid)) {
                s.addEventListener('input', () => {
                    if (showHRnetCorrections) _scheduleHRnetPreview();
                });
            }
            // Z-smooth window controls a live polynomial-reference line
            // on Pos: <Joint> Z plots — redraw the trace on every tick.
            if (sid === 'hfSliderZSWin') {
                s.addEventListener('input', () => {
                    if (showHRnetCorrections) renderDistanceTrace();
                });
            }
            upd();
        }
        $('fitHRnetBtn')?.addEventListener('click', () => {
            const open = $('fitHRnetPanel').style.display !== 'block';
            $('fitHRnetPanel').style.display = open ? 'block' : 'none';
            $('fitHRnetBtn').classList.toggle('active', open);
            if (open) {
                // Close other fit panels for room.
                $('fitOptionsPanel').style.display = 'none';
                $('fitSkeletonBtn').classList.remove('active');
                $('fitV2Panel').style.display = 'none';
                $('fitSkeletonV2Btn').classList.remove('active');
                $('fitLegacyPanel').style.display = 'none';
                $('fitSkeletonLegacyBtn').classList.remove('active');
            }
        });
        $('hrnetFitCloseBtn')?.addEventListener('click', () => {
            $('fitHRnetPanel').style.display = 'none';
            $('fitHRnetBtn').classList.remove('active');
        });
        $('hrnetFitResetBtn')?.addEventListener('click', () => {
            for (const [sid, lid, dec] of _hfPairs) {
                const s = $(sid); if (!s) continue;
                s.value = HRNET_FIT_DEFAULTS[sid];
                s.dispatchEvent(new Event('input'));
            }
        });
        // Maps backend hrnet_fit_params keys → slider DOM ids.  Used to
        // restore the last-run parameters when a trial is loaded.
        const _HF_PARAM_MAP = {
            cluster_size:  'hfSliderCluster',
            spike_support: 'hfSliderSpike',
            edge_margin:   'hfSliderEdge',
            w_hm:          'hfSliderHM',
            overlap_px:    'hfSliderOverlap',
            yzc_y_disp:    'hfSliderYZ_y',
            yzc_z_outlier: 'hfSliderYZ_zo',
            yzc_attr_jump: 'hfSliderYZ_aJ',
            yzc_attr_auc:  'hfSliderYZ_aC',
            yzc_z_median_mm: 'hfSliderYZ_zm',
            zsm_z_jump:    'hfSliderZS',
            zsm_smooth_window: 'hfSliderZSWin',
            bone_thresh_mm: 'hfSliderBoneThr',
            bone_K:         'hfSliderBoneK',
            w_bone:         'hfSliderBoneW',
            wrist_smooth_window: 'hfSliderWristSm',
        };
        _applyHRnetFitParams = function(params) {
            // Always start from defaults so missing keys revert (e.g. older
            // runs that pre-date w_anchor).
            for (const [sid] of _hfPairs) {
                const s = $(sid); if (!s) continue;
                s.value = HRNET_FIT_DEFAULTS[sid];
            }
            if (params && typeof params === 'object') {
                for (const [k, sid] of Object.entries(_HF_PARAM_MAP)) {
                    if (params[k] === undefined || params[k] === null) continue;
                    const s = $(sid); if (!s) continue;
                    s.value = params[k];
                }
            }
            for (const [sid] of _hfPairs) {
                $(sid)?.dispatchEvent(new Event('input'));
            }
        };
        $('runHRnetFitBtn')?.addEventListener('click', async () => {
            const st = $('fitHRnetStatus');
            if (!subjectId || currentTrialIdx < 0 || !trials?.[currentTrialIdx]) {
                if (st) st.textContent = 'Select a trial first.';
                return;
            }
            const body = {
                trial_idx:     trials[currentTrialIdx].trial_idx,
                cluster_size:  parseInt($('hfSliderCluster').value),
                w_hm:          parseFloat($('hfSliderHM').value),
                overlap_px:    parseFloat($('hfSliderOverlap').value),
                spike_support: parseFloat($('hfSliderSpike').value),
                edge_margin:   parseInt($('hfSliderEdge').value),
                bone_thresh_mm: parseFloat($('hfSliderBoneThr').value),
                bone_K:         parseInt($('hfSliderBoneK').value),
                w_bone:         parseFloat($('hfSliderBoneW').value),
                wrist_smooth_window: parseInt($('hfSliderWristSm').value),
                yzc_y_disp:    parseFloat($('hfSliderYZ_y').value),
                yzc_z_outlier: parseFloat($('hfSliderYZ_zo').value),
                yzc_attr_jump: parseFloat($('hfSliderYZ_aJ').value),
                yzc_attr_auc:  parseFloat($('hfSliderYZ_aC').value),
                yzc_z_median_mm: parseFloat($('hfSliderYZ_zm').value),
                zsm_z_jump:    parseFloat($('hfSliderZS').value),
                zsm_smooth_window: parseInt($('hfSliderZSWin').value),
            };
            try {
                if (st) st.textContent = 'Submitting…';
                const res = await api(`/api/skeleton/${subjectId}/hrnet_fit`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                const jobId = res.job_id;
                if (st) st.textContent = `Job ${jobId} running…`;
                // Stream job progress over SSE (matches the v1/v2/v3 fit pattern).
                const savedSubjectId = subjectId;
                const source = new EventSource(`/api/jobs/${jobId}/stream`);
                _activeEventSources.add(source);
                source.onmessage = async (event) => {
                    if (subjectId !== savedSubjectId) {
                        source.close(); _activeEventSources.delete(source); return;
                    }
                    let data;
                    try { data = JSON.parse(event.data); } catch { return; }
                    const pct = Math.round(data.progress_pct || 0);
                    if (data.status === 'running' || data.status === 'pending') {
                        if (st) st.textContent = `HRnet Fit ${pct}%`;
                    } else if (data.status === 'completed') {
                        source.close(); _activeEventSources.delete(source);
                        if (st) st.textContent = 'Done — refreshing trial…';
                        await loadTrial(currentTrialIdx);
                        if (st) st.textContent = `HRnet Fit completed.`;
                        // Auto-show the canonical fit result on the plot:
                        // expand the HRnet sub-stage section and enable the
                        // Stereo-Hungarian 2D + 3D toggles.  Without this,
                        // the HRnet Fit traces are computed but invisible
                        // until the user manually expands and clicks the
                        // (often-overlooked) sub-stage checkboxes.
                        try {
                            _setHeatmapExpanded(true);
                            const cb2 = document.querySelector('.hm-row.hm-2d[data-hm="stereohun"]');
                            const cb3 = document.querySelector('.hm-row.hm-3d[data-hm="stereohun"]');
                            if (cb2 && !cb2.checked) { cb2.checked = true; showStereoHun2D = true; }
                            if (cb3 && !cb3.checked) { cb3.checked = true; showStereoHun3D = true; }
                            render(); update3D(); renderDistanceTrace();
                        } catch (_e) { /* best-effort UI sync */ }
                    } else if (data.status === 'cancelled') {
                        source.close(); _activeEventSources.delete(source);
                        if (st) st.textContent = 'Cancelled.';
                    } else if (data.status === 'failed') {
                        source.close(); _activeEventSources.delete(source);
                        if (st) st.textContent = `Failed: ${data.error_msg || 'unknown error'}`;
                    }
                };
                source.onerror = () => {
                    source.close(); _activeEventSources.delete(source);
                };
            } catch (err) {
                if (st) st.textContent = `Error: ${err.message || err}`;
            }
        });

        // Skeleton v2 (legacy smoothing) expand/collapse
        $('fitSkeletonLegacyBtn').addEventListener('click', () => {
            const open = $('fitLegacyPanel').style.display !== 'block';
            $('fitLegacyPanel').style.display = open ? 'block' : 'none';
            $('fitSkeletonLegacyBtn').classList.toggle('active', open);
            if (open) {
                $('fitOptionsPanel').style.display = 'none';
                $('fitSkeletonBtn').classList.remove('active');
                $('fitV2Panel').style.display = 'none';
                $('fitSkeletonV2Btn').classList.remove('active');
            }
        });
        $('fitLegacyCancelBtn').addEventListener('click', () => {
            $('fitLegacyPanel').style.display = 'none';
            $('fitSkeletonLegacyBtn').classList.remove('active');
        });

        // Live value updates for legacy sliders
        const _lgSliders = [
            ['lgSliderMediapipe',    'lgWMediapipe',    1],
            ['lgSliderDLC',          'lgWDLC',          1],
            ['lgSliderSmoothWrist',  'lgWSmoothWrist',  1],
            ['lgSliderSmoothXY',     'lgWSmoothXY',     1],
            ['lgSliderSmoothZ',      'lgWSmoothZ',      1],
            ['lgSliderSmoothAngles', 'lgWSmoothAngles', 1],
            ['lgSliderConstraints',  'lgWConstraints',  1],
        ];
        for (const [sid, lid, dec] of _lgSliders) {
            const s = $(sid), l = $(lid);
            if (s && l) {
                const u = () => { l.textContent = parseFloat(s.value).toFixed(dec); };
                s.addEventListener('input', u); u();
            }
        }



        // Restore the user's previous checkbox selections (if any)
        // before loading the trial, so updateFitStatus's availability
        // gating runs against the restored state.  Then attach a
        // capture-phase change listener that re-saves on every
        // checkbox toggle going forward.
        _restoreCheckboxes();
        _updateStageRowVisibility();
        _updateHeatmapRowVisibility?.();
        document.addEventListener('change', (ev) => {
            const t = ev.target;
            if (t && t.tagName === 'INPUT' && t.type === 'checkbox') {
                _saveCheckboxes();
            }
        }, true);

        // ``Run in reverse`` flips which sidecar (forward vs reverse
        // pass) drives the restored defaults.  Re-load when toggled
        // while MP detect mode is active so the bbox + flags reflect
        // whichever pass the user is about to run.
        const _mpRev = document.getElementById('mpReverse');
        if (_mpRev) {
            _mpRev.addEventListener('change', () => {
                if (_detectModel === 'run-mediapipe') {
                    _applyMpSidecarDefaults()
                        .catch(() => {})
                        .finally(() => render());
                }
            });
        }
        // "Use bounding box" off -> hide green bbox + dim shading;
        // back on -> reveal.  Just re-render; _drawBboxOverlay handles
        // the state.
        const _mpUb = document.getElementById('mpUseBbox');
        if (_mpUb) {
            _mpUb.addEventListener('change', () => {
                if (_detectModel === 'run-mediapipe') render();
            });
        }

        // Load initial subject and restore trial/frame from nav state.
        // ``?trial=N`` URL param (used by Jobs-page deep links) takes
        // precedence over the sessionStorage nav-state restoration.
        const trialParam = params.get('trial');
        const trialIdxFromUrl = trialParam != null && trialParam !== ''
            ? parseInt(trialParam, 10) : null;
        if (sel.value) {
            const nav = typeof getNavState === 'function' ? getNavState() : {};
            await loadSubject(parseInt(sel.value));
            if (trialIdxFromUrl != null && Number.isFinite(trialIdxFromUrl)
                && trialIdxFromUrl >= 0 && trialIdxFromUrl < trials.length) {
                await loadTrial(trialIdxFromUrl);
            } else if (nav.subjectId === parseInt(sel.value)) {
                if (nav.trialIdx != null && nav.trialIdx >= 0 && nav.trialIdx < trials.length) {
                    await loadTrial(nav.trialIdx);
                }
                if (nav.side && cameraNames.includes(nav.side) && cameraMode === 'stereo') {
                    currentSide = nav.side;
                    _renderDualToggle('sideToggle', cameraNames[0], cameraNames[1], currentSide);
                    computeAutoCrop();
                    render();
                }
            }
        }
    }

    async function loadSubject(sid) {
        // Close any SSE connections from the previous subject's jobs
        for (const es of _activeEventSources) {
            try { es.close(); } catch {}
        }
        _activeEventSources.clear();
        subjectId = sid;
        sessionStorage.setItem('dlc_lastSubjectId', String(sid));
        if (typeof setNavState === 'function') setNavState({ subjectId: sid });
        // Keep URL in sync so refresh loads the same subject
        const url = new URL(window.location);
        url.searchParams.set('subject', sid);
        history.replaceState(null, '', url);
        const subj = allSubjects.find(s => s.id === sid);
        subjectName = subj ? subj.name : '';
        // Per-subject camera mode
        if (subj && subj.camera_mode) cameraMode = subj.camera_mode;

        try {
            trials = await api(`/api/skeleton/${sid}/trials`);
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
            const shortName = t.trial_stem.includes('_') ? t.trial_stem.split('_').slice(1).join('_') : t.trial_stem;
            btn.textContent = shortName;
            btn.title = t.trial_stem;
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
        _prewarmStarted = false;  // allow prewarm for the new trial
        _userZoom = false; // allow auto-crop for new trial
        highlightTrialButton(idx);
        if (typeof setNavState === 'function') setNavState({ trialIdx: idx, frame: currentFrame });
        // If the user was actively editing a bbox when they switched
        // trials, refresh the bbox to the saved value for the NEW trial.
        // Without this, ``bboxOS`` / ``bboxOD`` kept the previous trial's
        // values, the canvas overlay drew them on top of the new trial,
        // and a Save Box click would silently copy that old bbox onto
        // the new trial -- making every trial look like it shared one
        // bbox.  ``mp_crop_boxes`` has always been keyed per-trial; this
        // just keeps the UI honest.
        if (bboxEditMode && typeof _loadDefaultBbox === 'function') {
            _loadDefaultBbox().then(() => render());
        }
        // Clear stale save-status from the previous trial so the next
        // "Saved bbox for ..." message is unambiguously this trial's.
        const _ds = $('detectStatus');
        if (_ds) _ds.textContent = '';

        const trialNameEl = $('trialName');
        if (trialNameEl) trialNameEl.textContent = trial.trial_stem.includes('_') ? trial.trial_stem.split('_').slice(1).join('_') : trial.trial_stem;
        $('totalFrames').textContent = trial.n_frames;

        // Clear the stale trial data + error overlays so the previous
        // subject's labels don't flash over the new subject's video while
        // we wait for the fresh API response.
        trialData = null;
        skelErrorMatrices = {};
        mpCorrectedL = null; mpCorrectedR = null;
        _boneFlagsByPair = null;
        render(); update3D(); renderDistanceTrace();

        // Load bulk data (may fail if no calibration — still load video)
        try {
            trialData = await api(`/api/skeleton/${subjectId}/trial/${trial.trial_idx}/data`);
        } catch (e) {
            console.error('Failed to load trial data:', e);
            trialData = null;
            const statusEl = $('manoFitStatus');
            if (statusEl) statusEl.innerHTML = `<span style="color:var(--red);">${e.message}</span>`;
        }

        // Restore HRnet Fit slider values from the last run for this trial
        // (falling back to global defaults when nothing is saved).
        _applyHRnetFitParams(trialData?.hrnet_peaks?.hrnet_fit_params || null);

        // Populate distance selector
        const distSel = $('distanceSelect');
        distSel.innerHTML = '';
        if (trialData && trialData.distance_options) {
            for (const name of Object.keys(trialData.distance_options)) {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                distSel.appendChild(opt);
            }
        }
        // distSel is hidden; no event listener needed
        // Y-range sliders — one pair for distances, one for angles
        function _setupYSliders(minId, maxId, fillId) {
            const mn = $(minId), mx = $(maxId), fill = $(fillId);
            if (!mn || !mx) return;
            function updateFill() {
                if (!fill) return;
                const mnV = parseInt(mn.value), mxV = parseInt(mx.value);
                const range = parseInt(mx.max) - parseInt(mx.min);
                const h = mn.offsetHeight || 110;
                const pctMin = (mnV - parseInt(mn.min)) / range;
                const pctMax = (mxV - parseInt(mx.min)) / range;
                fill.style.bottom = (pctMin * h) + 'px';
                fill.style.height = ((pctMax - pctMin) * h) + 'px';
            }
            function onInput() {
                if (parseInt(mn.value) > parseInt(mx.value)) mn.value = mx.value;
                updateFill();
                renderDistanceTrace();
            }
            mn.addEventListener('input', onInput);
            mx.addEventListener('input', onInput);
            updateFill();
        }
        _setupYSliders('distYMinSlider', 'distYMaxSlider', 'yRangeFill');
        _setupYSliders('angleYMinSlider', 'angleYMaxSlider', 'yRangeFillAngle');

        // Update Skeleton fit status and checkbox state
        updateFitStatus();

        // Restore fitting sliders + constraints from last fit
        _restoreV1Params();
        _restoreV2Params();

        // Populate previous fit dropdown
        prevFitData = null;
        showPrev2D = false; showPrev3D = false;
        _renderPrevFitControls(null);
        const prevSel = $('prevFitSelect');
        if (prevSel) {
            prevSel.innerHTML = '';
            const history = trialData?.v2_fit_history || [];
            if (history.length) {
                prevSel.disabled = false;
                const _psk1 = $('showPrevSkel'); if (_psk1) _psk1.disabled = false;
                prevSel.innerHTML = '<option value="">— select —</option>';
                for (const h of history) {
                    const opt = document.createElement('option');
                    opt.value = h.slot;
                    opt.textContent = h.label;
                    opt.dataset.version = h.version || 'v2';
                    prevSel.appendChild(opt);
                }
            } else {
                prevSel.innerHTML = '<option value="">No history</option>';
                prevSel.disabled = true;
                const _psk2 = $('showPrevSkel'); if (_psk2) _psk2.disabled = true;
            }
        }

        // Reset frame before loading video (loadedmetadata may fire sync for cached vids)
        currentFrame = 0;
        _videoReady = false;

        // Load video — create fresh element per trial (avoids stale listeners)
        const trialFps = (trialData && trialData.fps) || trial.fps || 30;
        const oldVideo = videoEl;
        if (oldVideo) { oldVideo.pause(); oldVideo.removeAttribute('src'); oldVideo.load(); }

        const newVideo = document.createElement('video');
        newVideo.muted = true;
        newVideo.preload = 'auto';
        videoEl = newVideo;

        videoEl.src = `/api/skeleton/${subjectId}/trial/${trial.trial_idx}/video`;

        videoEl.addEventListener('loadedmetadata', () => {
            vidW = videoEl.videoWidth;
            vidH = videoEl.videoHeight;
            if (!vidW) return;
            midline = cameraMode === 'stereo' ? vidW / 2 : vidW;
            sizeCanvases();
            computeAutoCrop();
            _planTrackingPath();
            if (trackingZoom && _trackSegments.length) {
                const seg = _trackSegments[0];
                scale = seg.scale; offsetX = seg.offsetX; offsetY = seg.offsetY;
            }
            if (trialData && trialData.calib) {
                snapToCamera(_pendingFrame != null);
            } else if (!trackingZoom) {
                resetZoom();
            }
            // Seek to target frame then render once decoded
            const seekFrame = _pendingFrame != null ? _pendingFrame : 0;
            const seekTime = (seekFrame + 0.5) / trialFps;
            videoEl.currentTime = seekTime;
        }, { once: true });

        // loadeddata fires when first frame is decoded (readyState >= 2)
        videoEl.addEventListener('loadeddata', () => {
            _videoReady = true;
            if (_pendingFrame != null) {
                currentFrame = _pendingFrame;
                _pendingFrame = null;
            }
            render();
            renderDistanceTrace();
            update3D();
        }, { once: true });

        // Pause when video reaches its natural end
        videoEl.addEventListener('ended', () => {
            if (playing) togglePlay();
        });

        // If the video element rejects the source (codec mismatch — e.g.
        // mpeg4 ASP de-identified renders, some yuvj420p variants), it
        // never fires loadedmetadata/loadeddata, so render() is never
        // triggered from the normal path.  Force a render here so the
        // page still shows 2D landmarks + 3D skeleton on a black canvas.
        videoEl.addEventListener('error', () => {
            _videoReady = false;
            // Use trial dimensions if available so overlays scale correctly.
            if (!vidW && trialData) {
                vidW = trialData.video_width  || 3840;
                vidH = trialData.video_height || 1080;
                midline = cameraMode === 'stereo' ? vidW / 2 : vidW;
                sizeCanvases();
                computeAutoCrop();
            }
            render();
            renderDistanceTrace();
            update3D();
        }, { once: true });

        // Update state
        currentFrame = 0;
        updateVisibleJoints();
        camera3dInit = false;

        // Render whatever we have immediately (before video loads)
        render();
        renderDistanceTrace();
        update3D();

        // Clear heatmap cache
        for (const k in heatmapCache) delete heatmapCache[k];

        // Background-prewarm per-stage scores so the first error-slider
        // adjustment doesn't trigger a long "Computing..." wait.
        _prewarmStages();
    }

    // ── Controls setup ───────────────────────────────────────
    function setupControls() {
        $('prevFrameBtn').addEventListener('click', () => goToFrame(currentFrame - 1));
        $('nextFrameBtn').addEventListener('click', () => goToFrame(currentFrame + 1));
        // Timeline X-scale
        const xScaleEl = $('xScaleSlider');
        if (xScaleEl) {
            xScaleEl.addEventListener('input', e => {
                _xScale = parseFloat(e.target.value) || 1.0;
                sizeCanvases();
                renderDistanceTrace();
                _scrollDistToFrame(currentFrame, true);
            });
        }
        // Sidebar toggle for iPad / narrow screens
        const sidebarToggle = $('sidebarToggle');
        const manoSidebar = $('manoSidebar');
        if (sidebarToggle && manoSidebar) {
            sidebarToggle.addEventListener('click', () => {
                manoSidebar.classList.toggle('open');
            });
            // Close sidebar when tapping outside on narrow screens
            document.addEventListener('click', (e) => {
                if (!manoSidebar.classList.contains('open')) return;
                if (e.target === sidebarToggle || sidebarToggle.contains(e.target)) return;
                if (manoSidebar.contains(e.target)) return;
                manoSidebar.classList.remove('open');
            }, true);
        }
        $('playBtn').addEventListener('click', togglePlay);
        $('sideToggle').addEventListener('click', toggleSide);
        $('trackZoomBtn').addEventListener('click', toggleTrackingZoom);
        $('snapCamBtn').addEventListener('click', () => snapToCamera());
        $('prevSubjectBtn').addEventListener('click', prevSubject);
        $('nextSubjectBtn').addEventListener('click', nextSubject);

        // Speed slider
        const speedSlider = $('speedSlider');
        speedSlider.value = 3;
        speedSlider.addEventListener('input', () => {
            playbackRate = SPEED_PRESETS[parseInt(speedSlider.value)];
            $('speedDisplay').textContent = playbackRate + 'x';
            // Always update the video element rate (works during playback)
            videoEl.playbackRate = Math.min(playbackRate, 16);
        });
        speedSlider.addEventListener('change', () => {
            speedSlider.blur();
        });

        // Video visibility toggle button
        $('toggleVideoBtn').addEventListener('click', () => toggleVideo());
        function toggleVideo() {
            showVideo = !showVideo;
            $('toggleVideoBtn').textContent = showVideo ? 'Hide Video' : 'Show Video';
            render();
        }
        $('showMPSkel')?.addEventListener('change', e => { showMPSkel = e.target.checked; render(); update3D(); });
        $('showVision2D').addEventListener('change', e => { showVision2D = e.target.checked; updateLayerFlags(); });
        $('showVision3D').addEventListener('change', e => { showVision3D = e.target.checked; updateLayerFlags(); });
        $('showVisionSkel')?.addEventListener('change', e => { showVisionSkel = e.target.checked; render(); update3D(); });
        $('showManoSkel')?.addEventListener('change', e => { showManoSkel = e.target.checked; render(); update3D(); });

        // Layer toggles — update derived flags
        function updateLayerFlags() {
            showMP = showMP2D || showMP3D;
            showMano = showMano2D || showMano3D;
            _syncAllSkelCheckboxes();
            _updateHandDiagramColor();
            render();
            update3D();
            renderDistanceTrace();
        }
        $('showMP2D').addEventListener('change', e => { showMP2D = e.target.checked; updateLayerFlags(); });
        $('showReverse2D').addEventListener('change', e => { showReverse2D = e.target.checked; updateLayerFlags(); });
        $('showCropped2D')?.addEventListener('change', e => { showCropped2D = e.target.checked; updateLayerFlags(); });
        $('showStatic2D')?.addEventListener('change', e => { showStatic2D = e.target.checked; updateLayerFlags(); });
        $('showCombined2D')?.addEventListener('change', e => { showCombined2D = e.target.checked; updateLayerFlags(); });
        // Helper: clear the selected-joint marker if none of the three
        // Stereo variants is visible anymore.
        const _maybeClearStereoSelected = () => {
            if (!showStereo2D && !showStereoOutline2D && !showStereoHybrid2D) {
                stereoSelectedJoint = null;
            }
        };
        $('showStereo2D')?.addEventListener('change', e => {
            showStereo2D = e.target.checked;
            _maybeClearStereoSelected();
            const wrap = $('stereoConfWrap');
            if (wrap) wrap.style.display = showStereo2D ? 'flex' : 'none';
            updateLayerFlags();
        });
        $('stereoConfSlider')?.addEventListener('input', e => {
            stereoConfThreshold = parseFloat(e.target.value);
            const lbl = $('stereoConfVal');
            if (lbl) lbl.textContent = stereoConfThreshold.toFixed(2);
            render();
        });
        $('showStereoOutline2D')?.addEventListener('change', e => {
            showStereoOutline2D = e.target.checked;
            _maybeClearStereoSelected();
            const wrap = $('stereoOutlineConfWrap');
            if (wrap) wrap.style.display = showStereoOutline2D ? 'flex' : 'none';
            updateLayerFlags();
        });
        $('stereoOutlineConfSlider')?.addEventListener('input', e => {
            stereoOutlineConfThreshold = parseFloat(e.target.value);
            const lbl = $('stereoOutlineConfVal');
            if (lbl) lbl.textContent = stereoOutlineConfThreshold.toFixed(2);
            render();
        });
        $('showStereoHybrid2D')?.addEventListener('change', e => {
            showStereoHybrid2D = e.target.checked;
            _maybeClearStereoSelected();
            const wrap = $('stereoHybridConfWrap');
            if (wrap) wrap.style.display = showStereoHybrid2D ? 'flex' : 'none';
            updateLayerFlags();
        });
        $('stereoHybridConfSlider')?.addEventListener('input', e => {
            stereoHybridConfThreshold = parseFloat(e.target.value);
            const lbl = $('stereoHybridConfVal');
            if (lbl) lbl.textContent = stereoHybridConfThreshold.toFixed(2);
            render();
        });
        $('showOutline2D')?.addEventListener('change', e => {
            showOutline2D = e.target.checked;
            updateLayerFlags();
        });
        // Helper: outline is shown on the canvas when the Stereo panel
        // is OPEN and the Hybrid radio is selected (the user is staging
        // a hybrid bake and wants to see what's being masked).  Also
        // gates the Dilate slider visibility.
        const _stereoIsHybridStaging = () => {
            if ($('stereoPanel')?.style.display !== 'block') return false;
            const m = document.querySelector('input[name="stereoMode"]:checked');
            return !!(m && m.value === 'hybrid');
        };
        // The Gauss (centre-weight) slider applies to BOTH image and
        // hybrid modes -- it shows whenever the Stereo panel is open
        // and the selected mode is one of those two.
        const _stereoSupportsGauss = () => {
            if ($('stereoPanel')?.style.display !== 'block') return false;
            const m = document.querySelector('input[name="stereoMode"]:checked');
            return !!(m && (m.value === 'image' || m.value === 'hybrid'));
        };
        const _refreshStereoHybridUI = () => {
            const dilateOn = _stereoIsHybridStaging();
            const dWrap = $('stereoDilateWrap');
            if (dWrap) dWrap.style.display = dilateOn ? 'flex' : 'none';
            const gWrap = $('stereoGaussWrap');
            if (gWrap) gWrap.style.display = _stereoSupportsGauss() ? 'flex' : 'none';
            render();
        };
        // Stereo button: toggle panel (Run + Close).  Mirrors HRnet Correct.
        $('runStereoBtn')?.addEventListener('click', () => {
            const open = $('stereoPanel').style.display !== 'block';
            $('stereoPanel').style.display = open ? 'block' : 'none';
            $('runStereoBtn').classList.toggle('active', open);
            _refreshStereoHybridUI();
        });
        $('stereoCloseBtn')?.addEventListener('click', () => {
            $('stereoPanel').style.display = 'none';
            $('runStereoBtn').classList.remove('active');
            _refreshStereoHybridUI();
        });
        // Radio + dilate slider events.
        document.querySelectorAll('input[name="stereoMode"]').forEach(el => {
            el.addEventListener('change', _refreshStereoHybridUI);
        });
        $('stereoDilateSlider')?.addEventListener('input', e => {
            const v = parseInt(e.target.value, 10) || 0;
            const lbl = $('stereoDilateVal');
            if (lbl) lbl.textContent = `${v} px`;
            render();
        });
        $('stereoGaussSlider')?.addEventListener('input', e => {
            const v = (parseInt(e.target.value, 10) || 0) / 100;
            const lbl = $('stereoGaussVal');
            if (lbl) lbl.textContent = v.toFixed(2);
        });
        $('runStereoGoBtn')?.addEventListener('click', async () => {
            const st = $('stereoStatus');
            const goBtn = $('runStereoGoBtn');
            if (!subjectId || currentTrialIdx < 0 || !trials?.[currentTrialIdx]) {
                if (st) st.textContent = 'Select a trial first.';
                return;
            }
            const target = await _promptExecTarget(goBtn);
            if (target === 'remote' && st) {
                st.textContent = 'Remote not wired for Stereo yet — running locally.';
            } else if (st) {
                st.textContent = 'Submitting…';
            }
            // Read selected mode from the Image/Outline/Hybrid radios.
            const _modeEl = document.querySelector('input[name="stereoMode"]:checked');
            const stereoMode = _modeEl ? _modeEl.value : 'image';
            const maskDilatePx = parseInt(($('stereoDilateSlider')?.value ?? '10'), 10) || 0;
            const gaussWeight = (parseInt(($('stereoGaussSlider')?.value ?? '0'), 10) || 0) / 100;
            try {
                const res = await api(`/api/skeleton/${subjectId}/run_stereo`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        trial_idx: trials[currentTrialIdx].trial_idx,
                        mode: stereoMode,
                        mask_dilate_px: maskDilatePx,
                        gauss_center_weight: gaussWeight,
                    }),
                });
                const jobId = res.job_id;
                if (st) st.textContent = `Job ${jobId} running…`;
                const savedSid = subjectId;
                const evt = new EventSource(`/api/jobs/${jobId}/stream`);
                _activeEventSources.add(evt);
                evt.onmessage = async (e) => {
                    if (subjectId !== savedSid) {
                        evt.close(); _activeEventSources.delete(evt); return;
                    }
                    const data = JSON.parse(e.data);
                    if (data.status === 'running') {
                        const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                        if (st) st.textContent = `Running… ${pct}%`;
                    } else if (data.status === 'completed') {
                        evt.close(); _activeEventSources.delete(evt);
                        if (st) st.textContent = 'Complete. Reloading…';
                        await loadTrial(currentTrialIdx);
                        // Auto-enable the model row that matches the
                        // variant we just baked.
                        // Auto-enable the model row that matches the
                        // variant we just baked.
                        const _autoOn = (cbId, varName, wrapId) => {
                            const cb = $(cbId);
                            if (cb && !cb.checked) {
                                cb.checked = true;
                                if (varName === 'showStereo2D') showStereo2D = true;
                                else if (varName === 'showStereoOutline2D') showStereoOutline2D = true;
                                else if (varName === 'showStereoHybrid2D') showStereoHybrid2D = true;
                                const wrap = $(wrapId);
                                if (wrap) wrap.style.display = 'flex';
                                updateLayerFlags();
                            }
                        };
                        if (stereoMode === 'outline') {
                            _autoOn('showStereoOutline2D', 'showStereoOutline2D', 'stereoOutlineConfWrap');
                        } else if (stereoMode === 'hybrid') {
                            _autoOn('showStereoHybrid2D', 'showStereoHybrid2D', 'stereoHybridConfWrap');
                        } else {
                            _autoOn('showStereo2D', 'showStereo2D', 'stereoConfWrap');
                        }
                        if (st) st.textContent = 'Done.';
                    } else if (data.status === 'failed') {
                        evt.close(); _activeEventSources.delete(evt);
                        if (st) st.textContent = `Failed: ${data.error_msg || ''}`;
                    } else if (data.status === 'cancelled') {
                        evt.close(); _activeEventSources.delete(evt);
                        if (st) st.textContent = 'Cancelled.';
                    }
                };
            } catch (e) {
                if (st) st.textContent = `Error: ${e.message || e}`;
            }
        });
        $('showMP3D').addEventListener('change', e => { showMP3D = e.target.checked; updateLayerFlags(); });
        $('showReverse3D').addEventListener('change', e => { showReverse3D = e.target.checked; updateLayerFlags(); });
        $('showCropped3D')?.addEventListener('change', e => { showCropped3D = e.target.checked; updateLayerFlags(); });
        $('showStatic3D')?.addEventListener('change', e => { showStatic3D = e.target.checked; updateLayerFlags(); });
        $('showCombined3D')?.addEventListener('change', e => { showCombined3D = e.target.checked; updateLayerFlags(); });
        $('showMano2D').addEventListener('change', e => { showMano2D = e.target.checked; updateLayerFlags(); _updateHandDiagramColor(); });
        $('showMano3D').addEventListener('change', e => { showMano3D = e.target.checked; updateLayerFlags(); _updateHandDiagramColor(); });
        $('showSkelV2_2D').addEventListener('change', e => {
            showSkelV2_2D = e.target.checked;
            _onSkelMasterToggle('2D', showSkelV2_2D);
            _syncAllSkelCheckboxes(); _updateHandDiagramColor();
            render(); update3D(); renderDistanceTrace();
        });
        $('showSkelV2_3D').addEventListener('change', e => {
            showSkelV2_3D = e.target.checked;
            _onSkelMasterToggle('3D', showSkelV2_3D);
            _syncAllSkelCheckboxes(); _updateHandDiagramColor();
            render(); update3D(); renderDistanceTrace();
        });
        $('showSkelV2Skel')?.addEventListener('change', e => { showSkelV2Skel = e.target.checked; render(); update3D(); });
        $('showLegacyV2_2D').addEventListener('change', e => { showLegacyV2_2D = e.target.checked; _syncAllSkelCheckboxes(); _updateHandDiagramColor(); render(); update3D(); renderDistanceTrace(); });
        $('showLegacyV2_3D').addEventListener('change', e => { showLegacyV2_3D = e.target.checked; _syncAllSkelCheckboxes(); _updateHandDiagramColor(); render(); update3D(); renderDistanceTrace(); });
        $('showLegacyV2Skel')?.addEventListener('change', e => { showLegacyV2Skel = e.target.checked; render(); update3D(); });
        // Stage-row checkbox handlers: individual 2D/3D/Err toggles per stage
        document.querySelectorAll('.stage-row.stage-2d').forEach(cb => {
            cb.addEventListener('change', () => _onStageToggle(cb.dataset.stage, '2D', cb.checked));
        });
        document.querySelectorAll('.stage-row.stage-3d').forEach(cb => {
            cb.addEventListener('change', () => _onStageToggle(cb.dataset.stage, '3D', cb.checked));
        });
        document.querySelectorAll('.stage-row.stage-err').forEach(cb => {
            cb.addEventListener('change', () => _onStageToggle(cb.dataset.stage, 'Err', cb.checked));
        });
        // Clickable model labels: Skeleton v3 + Heatmap expand/collapse.
        document.querySelectorAll('.model-toggle').forEach(el => {
            el.addEventListener('click', () => {
                const grp = el.dataset.group;
                if (grp === 'v3') _setV3Expanded(!_v3Expanded);
                else if (grp === 'heatmap') _setHeatmapExpanded(!_heatmapExpanded);
            });
        });
        // HRnet (raw argmax) sub-stage 2D/3D + Peak-select 2D/3D bindings.
        const hmDispatch = (e) => {
            const which = e.target;
            const hmKey = which.dataset.hm;
            if (hmKey === 'hrnet') {
                if (which.classList.contains('hm-2d')) showHRnet2D = which.checked;
                else if (which.classList.contains('hm-3d')) showHRnet3D = which.checked;
                else if (which.classList.contains('hm-err')) {
                    showHRnetCorrections = which.checked;
                    if (showHRnetCorrections) _refreshHRnetPreview();
                    else { hrnetPreview = null; }
                }
                // hm-img is showHeatmap which has its own handler
            } else if (hmKey === 'stereohun') {
                if (which.classList.contains('hm-2d')) showStereoHun2D = which.checked;
                else if (which.classList.contains('hm-3d')) showStereoHun3D = which.checked;
            } else if (hmKey === 'yzc') {
                if (which.classList.contains('hm-2d')) showHRnetYZC2D = which.checked;
                else if (which.classList.contains('hm-3d')) showHRnetYZC3D = which.checked;
            } else if (hmKey === 'zsm') {
                if (which.classList.contains('hm-2d')) showHRnetZSM2D = which.checked;
                else if (which.classList.contains('hm-3d')) showHRnetZSM3D = which.checked;
            }
            render(); update3D(); renderDistanceTrace();
        };
        document.querySelectorAll('.hm-row.hm-2d, .hm-row.hm-3d, .hm-row.hm-err').forEach(el => {
            el.addEventListener('change', hmDispatch);
        });

        _updateStageRowVisibility();
        _updateHeatmapRowVisibility();

        // Error-detection & camera-attribution sliders
        const _edSliders = [
            ['edSliderZ',      'edWZ',      'detection',   'z_jump'],
            ['edSliderZOut',   'edWZOut',   'detection',   'z_outlier'],
            ['edSliderY',      'edWY',      'detection',   'y_disp'],
            ['edSliderBL',     'edWBL',     'detection',   'bone_length'],
            ['edSliderBA',     'edWBA',     'detection',   'bone_agreement'],
            ['edSliderAng',    'edWAng',    'detection',   'angle'],
            ['edSliderHR',     'edWHR',     'detection',   'hrnet_mismatch', 0],
            ['edSliderStereoConf', 'edWStereoConf', 'detection', 'stereo_conf', 2],
            ['edSliderStereoDist', 'edWStereoDist', 'detection', 'stereo_dist', 0],
            ['edSliderStereoOcc',  'edWStereoOcc',  'detection', 'stereo_occlusion', 1],
            ['eaSliderJump',   'eaWJump',   'attribution', 'jump_2d'],
            ['eaSliderConf',   'eaWConf',   'attribution', 'confidence'],
            ['eaSliderHRnet',  'eaWHRnet',  'attribution', 'hrnet'],
            ['ecSliderYDisp',  'ecWYDisp',  'corrections', 'y_disp'],
        ];
        for (const row of _edSliders) {
            const [sid, lid, group, key, decOverride] = row;
            const s = $(sid), lbl = $(lid);
            if (!s) continue;
            // Attribution sliders use finer step (0.001) → 3 decimals; others 2.
            const decimals = (decOverride != null) ? decOverride
                          : (group === 'attribution') ? 3 : 2;
            const update = () => {
                const v = parseFloat(s.value);
                if (!_mpErrorWeights[group]) _mpErrorWeights[group] = {};
                _mpErrorWeights[group][key] = v;
                if (lbl) lbl.textContent = v.toFixed(decimals);
                _scheduleMPErrorRecompute();
            };
            s.addEventListener('input', update);
            update();
        }
        // The Occlusion-radius slider drives a live canvas overlay
        // (dashed circles around every MP joint).  The generic update
        // above only schedules an error recompute; we also need an
        // immediate render() so the circles follow the slider as the
        // user drags.
        $('edSliderStereoOcc')?.addEventListener('input', () => render());
        // Stereo mode radios + dilate/center sliders -- visibility wired
        // to the radio choice (dilate only for Hybrid; center for image+
        // hybrid).
        const _v3StereoMode = () => {
            const m = document.querySelector('input[name="v3StereoMode"]:checked');
            return m ? m.value : 'image';
        };
        const _refreshV3StereoUI = () => {
            const mode = _v3StereoMode();
            _mpErrorWeights.stereo.mode = mode;
            const dWrap = $('v3StereoDilateWrap');
            if (dWrap) dWrap.style.display = (mode === 'hybrid') ? 'flex' : 'none';
            const gWrap = $('v3StereoGaussWrap');
            if (gWrap) gWrap.style.display = (mode === 'image' || mode === 'hybrid') ? 'flex' : 'none';
        };
        document.querySelectorAll('input[name="v3StereoMode"]').forEach(el => {
            el.addEventListener('change', () => { _refreshV3StereoUI(); _scheduleMPErrorRecompute(); });
        });
        _refreshV3StereoUI();
        $('v3StereoDilateSlider')?.addEventListener('input', e => {
            const v = parseInt(e.target.value, 10) || 0;
            _mpErrorWeights.stereo.mask_dilate_px = v;
            const lbl = $('v3StereoDilateVal');
            if (lbl) lbl.textContent = `${v} px`;
        });
        $('v3StereoGaussSlider')?.addEventListener('input', e => {
            const v = (parseInt(e.target.value, 10) || 0) / 100;
            _mpErrorWeights.stereo.gauss_center_weight = v;
            const lbl = $('v3StereoGaussVal');
            if (lbl) lbl.textContent = v.toFixed(2);
        });
        // Constraint lines are shown when a plotted angle line is clicked
        $('showDLC').addEventListener('change', e => { showDLC = e.target.checked; render(); update3D(); renderDistanceTrace(); });
        $('showDLC3D').addEventListener('change', e => { showDLC3D = e.target.checked; update3D(); renderDistanceTrace(); });
        $('showPose2D').addEventListener('change', e => { showPose2D = e.target.checked; render(); update3D(); renderDistanceTrace(); });
        $('showPose3D').addEventListener('change', e => { showPose3D = e.target.checked; render(); update3D(); renderDistanceTrace(); });
        $('showPrevSkel')?.addEventListener('change', e => { showPrevSkel = e.target.checked; render(); update3D(); });
        $('prevExpand')?.addEventListener('click', () => _setPrevExpanded(!_prevExpanded));
        $('prevFitSelect').addEventListener('change', async e => {
            const slot = parseInt(e.target.value);
            prevFitData = null;
            showPrev2D = false; showPrev3D = false;
            if (!slot || !trialData) {
                // Deselected — restore current fit's sliders/constraints
                _renderPrevFitControls(null);
                _restoreV2Params();
                render(); update3D(); renderDistanceTrace(); return;
            }
            try {
                const trial = trials[currentTrialIdx];
                prevFitData = await api(`/api/skeleton/${subjectId}/trial/${trial.trial_idx}/fit_history/${slot}`);
            } catch (err) {
                console.error('Failed to load previous fit:', err);
            }
            // Determine the version of the selected fit and render the
            // appropriate inline controls (2D/3D checkboxes for v1/v2,
            // expandable stage rows for v3).
            const opt = e.target.selectedOptions?.[0];
            const version = opt?.dataset.version || 'v2';
            _renderPrevFitControls(version);
            // Auto-enable 3D display when data loads.
            if (prevFitData) showPrev3D = true;
            // Restore sliders + constraints from the historical fit
            if (prevFitData?.fit_params) {
                const p = prevFitData.fit_params;
                _setSlider('v2SliderMediapipe',    p.w_mediapipe);
                _setSlider('v2SliderVision',       p.w_vision);
                _setSlider('v2SliderDLC',          p.w_dlc);
                _setSlider('v2SliderHRNet',        p.w_hrnet);
                if ($('v2HRNetFingertipsOnly')) $('v2HRNetFingertipsOnly').checked = !!p.hrnet_fingertips_only;
                _setSlider('v2SliderSmoothWrist',  p.w_smooth_wrist);
                _setSlider('v2SliderSmoothXY',     p.w_smooth_xy);
                _setSlider('v2SliderSmoothZ',      p.w_smooth_z);
                _setSlider('v2SliderSmoothAngles', p.w_smooth_angles);
                _setSlider('v2SliderConstraints',  p.w_constraints);
            }
            if (prevFitData?.fit_constraints) {
                api('/api/skeleton/joint-constraints', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(prevFitData.fit_constraints),
                }).then(() => _refreshConstraints()).catch(() => {});
            }
            render(); update3D(); renderDistanceTrace();
        });
        // Heat-map state is now driven by joint selection alone.
        //   - Heatmap data unavailable for the trial → entire row hidden,
        //     slider disabled, joint clicks no-op.
        //   - Heatmap data available + ≥1 joint active → showHeatmap=true,
        //     slider live.
        //   - Heatmap data available + 0 joints active → slider dimmed.
        const _setHeatmapEnabled = (on) => {
            // Compatibility shim for the heatmap-section collapse/expand
            // handlers.  Off → drop active joints; on → no auto-selection
            // (the user picks joints directly on the sidebar hand model).
            if (!on) heatmapActiveJoints.clear();
            _refreshHeatmapState();
        };
        // Show the default joint selection on the sidebar hand from the start.
        _refreshHeatmapState();
        // (Peak Select sub-stage removed — its 2D/3D checkboxes no longer
        // exist in the DOM, so no handlers are bound here.)
        $('showHeatmapSkel')?.addEventListener('change', e => {
            showHeatmapSkel = e.target.checked;
            render(); update3D(); renderDistanceTrace();
        });
        $('heatmapMipBtn').addEventListener('click', () => {
            if (!_trialHasHeatmaps()) return;
            heatmapMipMode = !heatmapMipMode;
            if (heatmapMipMode) {
                // Entering MIP: save current selection AND activate every
                // joint, so the slider lights up regardless of whether the
                // user had any joints selected previously.
                _savedHeatmapJoints = new Set(heatmapActiveJoints);
                heatmapActiveJoints = new Set();
                for (let j = 0; j < 21; j++) heatmapActiveJoints.add(j);
            } else if (_savedHeatmapJoints) {
                heatmapActiveJoints = new Set(_savedHeatmapJoints);
                _savedHeatmapJoints = null;
            }
            _heatmapImageData = null;
            _refreshHeatmapState();
            if (showHeatmap) _prefetchHeatmap().then(() => render());
        });
        $('offsetsToggleBtn')?.addEventListener('click', () => {
            showOffsetArrows = !showOffsetArrows;
            $('offsetsToggleBtn').classList.toggle('active', showOffsetArrows);
            render(); update3D();
        });
        const hmThreshSlider = $('heatmapThreshSlider');
        if (hmThreshSlider) {
            hmThreshSlider.addEventListener('input', e => {
                heatmapThreshold = parseFloat(e.target.value);
                $('heatmapThreshVal').textContent = heatmapThreshold.toFixed(2);
                _heatmapImageData = null;
                _prefetchHeatmap().then(() => render());
            });
        }

        // ── Labels Hand: joint → heatmap, text → finger visibility ──
        document.querySelectorAll('#handDiagramLabels .joint').forEach(el => {
            el.addEventListener('mouseenter', () => {
                if (_trialHasHeatmaps()) {
                    // Heatmap data available → joint clicks drive selection;
                    // hover highlights just this joint.
                    el.classList.add('joint-hover');
                }
                // No heatmap data → no hover effect; joint click is a no-op.
            });
            el.addEventListener('mouseleave', () => {
                el.classList.remove('joint-hover');
            });
            el.addEventListener('click', e => {
                e.stopPropagation();
                if (!_trialHasHeatmaps()) return;  // no data → no-op
                const j = parseInt(el.dataset.joint);
                if (isNaN(j)) return;
                if (heatmapMipMode) {
                    // Exit MIP mode and select just this joint
                    heatmapMipMode = false;
                    heatmapActiveJoints.clear();
                    heatmapActiveJoints.add(j);
                    _savedHeatmapJoints = null;
                } else if (e.shiftKey) {
                    // Shift+click: additive toggle — preserves prior selection
                    if (heatmapActiveJoints.has(j)) heatmapActiveJoints.delete(j);
                    else heatmapActiveJoints.add(j);
                } else if (heatmapActiveJoints.has(j)) {
                    // Plain click on an active joint → turn just it off.
                    heatmapActiveJoints.delete(j);
                } else {
                    // Plain click on an inactive joint → single-select.
                    heatmapActiveJoints.clear();
                    heatmapActiveJoints.add(j);
                }
                _heatmapImageData = null;
                _refreshHeatmapState();
                if (showHeatmap) _prefetchHeatmap().then(() => render());
            });
        });
        document.querySelectorAll('#handDiagramLabels text').forEach(el => {
            el.addEventListener('mouseenter', () => { const g = el.closest('g[data-finger]'); if (g) g.classList.add('finger-hover'); });
            el.addEventListener('mouseleave', () => { const g = el.closest('g[data-finger]'); if (g) g.classList.remove('finger-hover'); });
            el.addEventListener('click', e => {
                e.stopPropagation();
                const g = el.closest('g[data-finger]');
                if (!g) return;
                const finger = g.dataset.finger;
                fingerVisibility[finger] = !fingerVisibility[finger];
                g.classList.toggle('dimmed', !fingerVisibility[finger]);
                updateVisibleJoints();
                render();
                update3D();
            });
        });


        // ── Plotting Hand: bones → distance, joints → angle, aperture → aperture ──
        document.addEventListener('plotReset', () => {
            plotJointStates.clear();
            wristPanelOpen = false;
            _updatePlotHighlight();
            update3D();
        });

        // Aperture hover
        const apertureLineEl = $('apertureLine');
        const apertureHitEl = $('apertureHit');
        if (apertureLineEl && apertureHitEl) {
            [apertureLineEl, apertureHitEl].forEach(el => {
                el.addEventListener('mouseenter', () => apertureLineEl.classList.add('aperture-hover'));
                el.addEventListener('mouseleave', () => apertureLineEl.classList.remove('aperture-hover'));
            });
        }

        // Bone and aperture clicks → toggle distance in/out of selectedMetrics
        document.querySelectorAll('#handDiagramPlot .bone, #handDiagramPlot .aperture, #handDiagramPlot .aperture-hit').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                const target = el.classList.contains('aperture-hit') ? $('apertureLine') : el;
                const j1 = parseInt(target?.dataset?.j1), j2 = parseInt(target?.dataset?.j2);
                if (isNaN(j1) || isNaN(j2)) return;
                const distOpts = trialData?.distance_options || {};
                for (const [name, pair] of Object.entries(distOpts)) {
                    if ((pair[0] === j1 && pair[1] === j2) || (pair[0] === j2 && pair[1] === j1)) {
                        if (selectedMetrics.has(name)) {
                            selectedMetrics.delete(name);
                        } else {
                            selectedMetrics.set(name, _nextMetricColor());
                            if (name === _TIA) _tiaIsDefault = false; // user explicitly added
                            _enforceSourceConstraint();
                        }
                        renderDistanceTrace();
                        _updatePlotHighlight();
                        update3D();
                        return;
                    }
                }
            });
        });

        // MCP distance line clicks → toggle MCP distance metric
        document.querySelectorAll('#handDiagramPlot .mcp-dist, #handDiagramPlot .mcp-dist-hit').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                const name = el.dataset.mcp || el.previousElementSibling?.dataset?.mcp;
                if (!name) return;
                if (selectedMetrics.has(name)) {
                    selectedMetrics.delete(name);
                } else {
                    selectedMetrics.set(name, _nextMetricColor());
                    _enforceSourceConstraint();
                }
                renderDistanceTrace();
                _updatePlotHighlight();
                update3D();
            });
        });

        // Spread label clicks → toggle individual spread
        document.querySelectorAll('#handDiagramPlot .spread-label').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                const name = el.dataset.spread;
                if (!name) return;
                if (selectedMetrics.has(name)) {
                    selectedMetrics.delete(name);
                } else {
                    selectedMetrics.set(name, _nextMetricColor());
                    _enforceSourceConstraint();
                }
                renderDistanceTrace();
                _updatePlotHighlight();
            });
        });

        // Coord label clicks → toggle individual wrist coordinate
        document.querySelectorAll('#handDiagramPlot .coord-label').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                const name = el.dataset.coord;
                if (!name) return;
                if (selectedMetrics.has(name)) {
                    selectedMetrics.delete(name);
                } else {
                    selectedMetrics.set(name, _nextMetricColor());
                    _enforceSourceConstraint();
                }
                renderDistanceTrace();
                _updatePlotHighlight();
            });
        });

        // Joint clicks — behavior depends on plotMode.
        // Click semantics (matches the labeling hand's heat-map joints):
        //   Plain click on INACTIVE joint  → clear all selections in this mode,
        //                                    then activate just the clicked joint.
        //   Plain click on ACTIVE joint    → turn just that joint off.
        //   Shift+click (any state)        → toggle clicked joint without
        //                                    affecting other selections.
        document.querySelectorAll('#handDiagramPlot .joint').forEach(el => {
            el.addEventListener('click', e => {
                e.stopPropagation();
                const j = parseInt(el.dataset.joint);
                if (isNaN(j)) return;

                if (plotMode === 'position') {
                    const jname = (oj) => trialData?.joint_names?.[oj] || `J${oj}`;
                    const _addJoint = (oj) => {
                        const cx = _nextMetricColor();
                        const cy = _nextMetricColor();
                        const cz = _nextMetricColor();
                        posJointStates.set(oj, { colorX: cx, colorY: cy, colorZ: cz });
                        selectedMetrics.set(`Pos: ${jname(oj)} Z`, cz);
                    };
                    const _removeJoint = (oj) => {
                        const n = jname(oj);
                        selectedMetrics.delete(`Pos: ${n} X`);
                        selectedMetrics.delete(`Pos: ${n} Y`);
                        selectedMetrics.delete(`Pos: ${n} Z`);
                        posJointStates.delete(oj);
                    };
                    const wasActive = posJointStates.has(j);
                    if (e.shiftKey) {
                        // Shift+click: toggle this joint without touching others.
                        if (wasActive) _removeJoint(j);
                        else { _addJoint(j); _enforceSourceConstraint(); }
                    } else if (wasActive) {
                        // Plain click on an active joint → turn just it off.
                        _removeJoint(j);
                    } else {
                        // Plain click on an inactive joint → clear others, single-select.
                        for (const oj of [...posJointStates.keys()]) _removeJoint(oj);
                        _addJoint(j);
                        _enforceSourceConstraint();
                    }
                    renderDistanceTrace();
                    _updatePlotHighlight();
                    update3D();
                    return;
                }

                // Angle mode (default)
                // Wrist: toggle panel open/closed; closing clears all wrist metrics
                if (j === 0) {
                    wristPanelOpen = !wristPanelOpen;
                    if (wristPanelOpen) {
                        // If pose-derived elbow is available, also add wrist
                        // flex/abd to the plot/3D selection so F/A labels and
                        // arcs appear at the wrist (same UX as finger joints).
                        if (trialData?.has_wrist_angles && !plotJointStates.has(0)) {
                            const c1 = _nextMetricColor();
                            const c2 = _nextMetricColor();
                            selectedMetrics.set('Flex: Wrist', c1);
                            selectedMetrics.set('Abd: Wrist', c2);
                            plotJointStates.set(0, { mode: 'both', colorFlex: c1, colorAbd: c2 });
                            _enforceSourceConstraint();
                        }
                    } else {
                        SPREAD_NAMES.forEach(n => selectedMetrics.delete(n));
                        COORD_NAMES.forEach(n => selectedMetrics.delete(n));
                        selectedMetrics.delete('Flex: Wrist');
                        selectedMetrics.delete('Abd: Wrist');
                        plotJointStates.delete(0);
                    }
                    renderDistanceTrace();
                    _updatePlotHighlight();
                    update3D();
                    return;
                }

                const flexOpts = trialData?.flex_angle_options || [];
                const _flexMatchFor = (oj) => flexOpts.find(f => f.joint === oj);
                const flexMatch = _flexMatchFor(j);
                if (!flexMatch) return;
                const _addJoint = (oj) => {
                    const fm = _flexMatchFor(oj);
                    if (!fm) return;
                    const flexName = fm.name;
                    const abdName = flexName.replace('Flex:', 'Abd:');
                    const c1 = _nextMetricColor();
                    selectedMetrics.set(flexName, c1);
                    const c2 = _nextMetricColor();
                    selectedMetrics.set(abdName, c2);
                    plotJointStates.set(oj, { mode: 'both', colorFlex: c1, colorAbd: c2 });
                };
                const _removeJoint = (oj) => {
                    const fm = _flexMatchFor(oj);
                    if (fm) {
                        selectedMetrics.delete(fm.name);
                        selectedMetrics.delete(fm.name.replace('Flex:', 'Abd:'));
                    }
                    const _km = {9: 'Knuckle: I-M-R', 13: 'Knuckle: M-R-P'};
                    if (_km[oj]) selectedMetrics.delete(_km[oj]);
                    plotJointStates.delete(oj);
                };
                const wasActive = plotJointStates.has(j);
                if (e.shiftKey) {
                    // Shift+click: toggle this joint without touching others.
                    if (wasActive) _removeJoint(j);
                    else { _addJoint(j); _enforceSourceConstraint(); }
                } else if (wasActive) {
                    // Plain click on an active joint → turn just it off.
                    _removeJoint(j);
                } else {
                    // Plain click on an inactive joint → clear others, single-select.
                    for (const oj of [...plotJointStates.keys()]) _removeJoint(oj);
                    _addJoint(j);
                    _enforceSourceConstraint();
                }
                renderDistanceTrace();
                _updatePlotHighlight();
                update3D();
            });
        });

        _updatePlotHighlight = function() {
            const svg = $('handDiagramPlot');
            if (!svg) return;

            // Reset
            svg.querySelectorAll('.bone').forEach(el => { el.style.stroke = ''; el.style.strokeWidth = ''; el.style.filter = ''; el.style.opacity = ''; });
            svg.querySelectorAll('.aperture').forEach(el => { el.style.stroke = ''; el.style.filter = ''; });
            svg.querySelectorAll('.mcp-dist').forEach(el => { el.classList.remove('selected-dist'); });
            svg.querySelectorAll('.joint').forEach(el => { el.style.stroke = ''; el.style.strokeWidth = ''; el.style.filter = ''; el.style.fill = ''; });
            svg.querySelectorAll('.angle-label, .pos-label').forEach(el => el.remove());
            // Show tip joints only in position mode
            svg.classList.toggle('pos-mode', plotMode === 'position');

            const distOpts = trialData?.distance_options || {};
            const flexOpts = trialData?.flex_angle_options || [];
            const ns = 'http://www.w3.org/2000/svg';

            // Distance / aperture / MCP dist highlights
            for (const [metric, color] of selectedMetrics) {
                if (metric.startsWith('Flex:') || metric.startsWith('Abd:')) continue;
                // MCP distance lines
                if (metric.startsWith('MCP:')) {
                    svg.querySelectorAll('.mcp-dist').forEach(el => {
                        if (el.dataset.mcp === metric) el.classList.add('selected-dist');
                    });
                    continue;
                }
                const pair = distOpts[metric];
                if (!pair) continue;
                svg.querySelectorAll('.bone, .aperture').forEach(el => {
                    const a = parseInt(el.dataset.j1), b = parseInt(el.dataset.j2);
                    if ((a === pair[0] && b === pair[1]) || (a === pair[1] && b === pair[0])) {
                        el.style.stroke = color;
                        el.style.strokeWidth = '3';
                        el.style.filter = `drop-shadow(0 0 3px ${color})`;
                        el.style.opacity = '1';
                    }
                });
            }

            // Joint highlights and F/A labels (with click handlers)
            for (const [j, state] of plotJointStates) {
                const fm = flexOpts.find(f => f.joint === j);
                if (!fm) continue;
                const jointEl = svg.querySelector(`.joint[data-joint="${j}"]`);
                if (!jointEl) continue;

                const cx = parseFloat(jointEl.getAttribute('cx'));
                const cy = parseFloat(jointEl.getAttribute('cy'));
                const flexName = fm.name;
                const abdName  = fm.name.replace('Flex:', 'Abd:');
                const flexActive = state.mode === 'flex' || state.mode === 'both';
                const abdActive  = state.mode === 'abd'  || state.mode === 'both';

                // Joint ring: use whichever active color is available
                const ringColor = flexActive ? state.colorFlex : state.colorAbd;
                jointEl.style.stroke = ringColor;
                jointEl.style.strokeWidth = '2';
                jointEl.style.filter = `drop-shadow(0 0 3px ${ringColor})`;

                // F label — colored when active, dimmed when not
                const fLbl = document.createElementNS(ns, 'text');
                fLbl.classList.add('angle-label');
                fLbl.setAttribute('x', cx + 6);
                fLbl.setAttribute('y', cy - 4);
                fLbl.style.fill = flexActive ? state.colorFlex : '#555';
                fLbl.style.cursor = 'pointer';
                fLbl.textContent = 'F';
                fLbl.addEventListener('click', e => {
                    e.stopPropagation();
                    const s = plotJointStates.get(j);
                    if (!s) return;
                    if (flexActive) {
                        selectedMetrics.delete(flexName);
                        if (s.mode === 'both') { s.mode = 'abd'; s.colorFlex = null; }
                        else { plotJointStates.delete(j); }
                    } else {
                        const c = _nextMetricColor();
                        s.colorFlex = c;
                        s.mode = 'both';
                        selectedMetrics.set(flexName, c);
                        _enforceSourceConstraint();
                    }
                    renderDistanceTrace();
                    _updatePlotHighlight();
                    update3D();
                });
                svg.appendChild(fLbl);

                // A label — colored when active, dimmed when not
                const aLbl = document.createElementNS(ns, 'text');
                aLbl.classList.add('angle-label');
                aLbl.setAttribute('x', cx + 6);
                aLbl.setAttribute('y', cy + 9);
                aLbl.style.fill = abdActive ? state.colorAbd : '#555';
                aLbl.style.cursor = 'pointer';
                aLbl.textContent = 'A';
                aLbl.addEventListener('click', e => {
                    e.stopPropagation();
                    const s = plotJointStates.get(j);
                    if (!s) return;
                    if (abdActive) {
                        selectedMetrics.delete(abdName);
                        if (s.mode === 'both') { s.mode = 'flex'; s.colorAbd = null; }
                        else { plotJointStates.delete(j); }
                    } else {
                        const c = _nextMetricColor();
                        s.colorAbd = c;
                        s.mode = 'both';
                        selectedMetrics.set(abdName, c);
                        _enforceSourceConstraint();
                    }
                    renderDistanceTrace();
                    _updatePlotHighlight();
                    update3D();
                });
                svg.appendChild(aLbl);

                // K label (knuckle angle) for M_MCP (j=9) and R_MCP (j=13)
                const KNUCKLE_MAP = { 9: 'Knuckle: I-M-R', 13: 'Knuckle: M-R-P' };
                const knuckleName = KNUCKLE_MAP[j];
                if (knuckleName) {
                    const kActive = selectedMetrics.has(knuckleName);
                    const kLbl = document.createElementNS(ns, 'text');
                    kLbl.classList.add('angle-label');
                    kLbl.setAttribute('x', cx - 10);
                    kLbl.setAttribute('y', cy + 3);
                    kLbl.style.fill = kActive ? (state.colorKnuckle || '#ff8040') : '#555';
                    kLbl.style.cursor = 'pointer';
                    kLbl.style.textAnchor = 'end';
                    kLbl.textContent = 'K';
                    kLbl.addEventListener('click', e => {
                        e.stopPropagation();
                        if (kActive) {
                            selectedMetrics.delete(knuckleName);
                            const s = plotJointStates.get(j);
                            if (s) delete s.colorKnuckle;
                        } else {
                            const c = _nextMetricColor();
                            selectedMetrics.set(knuckleName, c);
                            _enforceSourceConstraint();
                            const s = plotJointStates.get(j);
                            if (s) s.colorKnuckle = c;
                        }
                        renderDistanceTrace();
                        _updatePlotHighlight();
                    });
                    svg.appendChild(kLbl);
                }
            }

            // Wrist circle — highlight when panel is open
            const wristEl = svg.querySelector('.joint[data-joint="0"]');
            if (wristEl) {
                if (wristPanelOpen) {
                    wristEl.style.fill = '#ff4080';
                    wristEl.style.filter = 'drop-shadow(0 0 3px #ff4080)';
                } else {
                    wristEl.style.fill = '';
                    wristEl.style.filter = '';
                }
            }

            // Spread/coord labels — visible when panel open, colored only when selected
            svg.querySelectorAll('.spread-label').forEach(lbl => {
                const name = lbl.dataset.spread;
                if (!wristPanelOpen) { lbl.style.display = 'none'; return; }
                lbl.style.display = 'inline';
                lbl.style.fill = (name && selectedMetrics.has(name)) ? selectedMetrics.get(name) : '#555';
            });
            // Position mode: show X/Y/Z labels next to selected joints
            if (plotMode === 'position') {
                for (const [j, state] of posJointStates) {
                    const jointEl = svg.querySelector(`.joint[data-joint="${j}"]`);
                    if (!jointEl) continue;
                    const cx = parseFloat(jointEl.getAttribute('cx'));
                    const cy = parseFloat(jointEl.getAttribute('cy'));
                    jointEl.style.fill = '#ff4080';
                    jointEl.style.filter = 'drop-shadow(0 0 3px #ff4080)';
                    jointEl.style.strokeWidth = '2';

                    const jname = trialData?.joint_names?.[j] || `J${j}`;
                    const axes = [['X', state.colorX, -8], ['Y', state.colorY, 2], ['Z', state.colorZ, 12]];
                    for (const [axLabel, color, dx] of axes) {
                        const metricName = `Pos: ${jname} ${axLabel}`;
                        const isActive = selectedMetrics.has(metricName);
                        const lbl = document.createElementNS(ns, 'text');
                        lbl.classList.add('pos-label');
                        lbl.setAttribute('x', cx + dx);
                        lbl.setAttribute('y', cy - 6);
                        lbl.style.fontSize = '8px';
                        lbl.style.fontWeight = 'bold';
                        // Inactive axes render dimmed in their assigned colour
                        // so the user can still see which colour each axis will
                        // plot in once activated.
                        lbl.style.fill = color;
                        lbl.style.opacity = isActive ? '1' : '0.35';
                        lbl.style.cursor = 'pointer';
                        lbl.style.pointerEvents = 'all';
                        lbl.style.userSelect = 'none';
                        lbl.style.transition = 'opacity 0.1s, filter 0.1s';
                        lbl.textContent = axLabel;
                        lbl.addEventListener('mouseenter', () => {
                            lbl.style.opacity = '1';
                            lbl.style.filter = 'drop-shadow(0 0 2px #fff) brightness(1.4)';
                        });
                        lbl.addEventListener('mouseleave', () => {
                            lbl.style.opacity = selectedMetrics.has(metricName) ? '1' : '0.35';
                            lbl.style.filter = '';
                        });
                        lbl.addEventListener('click', ev => {
                            ev.stopPropagation();
                            const allAxes = axes.map(([a]) => `Pos: ${jname} ${a}`);
                            if (ev.shiftKey) {
                                // Shift-click: toggle this axis only, leave
                                // the others' state intact.
                                if (selectedMetrics.has(metricName)) {
                                    selectedMetrics.delete(metricName);
                                } else {
                                    selectedMetrics.set(metricName, color);
                                }
                            } else {
                                // Plain click on an axis label.
                                // Special case: clicking the ONLY active axis
                                // deactivates the joint entirely (collapses
                                // its labels and removes it from selection).
                                const activeCount = allAxes.filter(n => selectedMetrics.has(n)).length;
                                const isOnlyActive = activeCount === 1 && selectedMetrics.has(metricName);
                                if (isOnlyActive) {
                                    for (const n of allAxes) selectedMetrics.delete(n);
                                    posJointStates.delete(j);
                                } else {
                                    // Otherwise: single-select this axis for the
                                    // joint — deactivate the other two and
                                    // activate this one.
                                    for (const n of allAxes) selectedMetrics.delete(n);
                                    selectedMetrics.set(metricName, color);
                                }
                            }
                            renderDistanceTrace();
                            _updatePlotHighlight();
                        });
                        svg.appendChild(lbl);
                    }
                }
            }
        };

        // Blur range/checkbox inputs as soon as the user releases them so the
        // focused element doesn't swallow subsequent arrow keys (which should
        // always navigate frames after a slider adjustment is complete).
        ['pointerup', 'mouseup', 'touchend'].forEach(evt => {
            document.addEventListener(evt, e => {
                const t = e.target;
                if (t instanceof HTMLInputElement && (t.type === 'range' || t.type === 'checkbox')) {
                    t.blur();
                }
            }, true);
        });
        // Also preventDefault arrow/page keys if a range slider is still
        // focused (e.g., clicked but mouse not yet released) so arrows are
        // never consumed by the slider's native handler.
        document.addEventListener('keydown', e => {
            const t = e.target;
            if (t instanceof HTMLInputElement && t.type === 'range' &&
                ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown',
                 'PageUp','PageDown','Home','End'].includes(e.key)) {
                e.preventDefault();
                t.blur();
            }
        }, true);

        // Keyboard shortcuts — skip text inputs; blur checkboxes/ranges after handling
        document.addEventListener('keydown', e => {
            const t = e.target;
            if (t.tagName === 'INPUT' && t.type !== 'checkbox' && t.type !== 'range') return;
            if (t.tagName === 'SELECT') return;
            let handled = false;
            switch (e.key) {
                case 'a': case 'ArrowLeft':  goToFrame(currentFrame - 1); handled = true; break;
                case 's': case 'ArrowRight': goToFrame(currentFrame + 1); handled = true; break;
                case ' ': togglePlay(); handled = true; break;
                case 'e': toggleSide(); handled = true; break;
                case 'z': toggleTrackingZoom(); handled = true; break;
                case 'c': snapToCamera(); handled = true; break;
                case 'v': toggleVideo(); handled = true; break;
            }
            if (handled) {
                e.preventDefault();
                // Blur focused checkbox/range so Space doesn't also toggle it
                if (t.tagName === 'INPUT') t.blur();
            }
        });
    }

    function toggleSide() {
        if (cameraMode !== 'stereo') return; // no side toggle in single/multicam

        // ── Collect landmark canvas positions BEFORE switching ──
        const fn = currentFrame;
        const oldIsLeft = currentSide === cameraNames[0];
        const oldBps = canvas.width / (oldIsLeft ? midline : vidW - midline);

        // If a Stereo-model joint is selected, align the NEW camera's
        // MP label of that joint with the OLD camera's stereo label
        // (priority: Hybrid > Outline > Image).  This lets the user
        // visually compare partner-camera MP vs. stereo prediction at
        // the same screen position.
        const stereoTarget = _stereoTargetForSwitch(fn, oldIsLeft);
        let oldAnchorCX, oldAnchorCY;
        let oldPts = null;
        if (stereoTarget) {
            oldAnchorCX = offsetX + scale * stereoTarget.pt[0] * oldBps;
            oldAnchorCY = offsetY + scale * stereoTarget.pt[1] * oldBps;
        } else {
            oldPts = _gatherLandmark2D(fn, oldIsLeft);
            if (oldPts.length) {
                let mx = 0, my = 0;
                for (const [px, py] of oldPts) {
                    mx += offsetX + scale * px * oldBps;
                    my += offsetY + scale * py * oldBps;
                }
                oldAnchorCX = mx / oldPts.length;
                oldAnchorCY = my / oldPts.length;
            }
        }

        // ── Switch camera ──
        currentSide = currentSide === cameraNames[0] ? cameraNames[1] : cameraNames[0];
        _renderDualToggle('sideToggle', cameraNames[0], cameraNames[1], currentSide);

        const newIsLeft = currentSide === cameraNames[0];
        const newBps = canvas.width / (newIsLeft ? midline : vidW - midline);

        // ── Match zoom and compute offset to land the anchor ──
        let anchored = false;
        if (stereoTarget) {
            const newMP = (newIsLeft ? trialData.mp_tracked_L
                                     : trialData.mp_tracked_R)?.[fn]?.[stereoTarget.joint];
            if (newMP) {
                offsetX = oldAnchorCX - scale * newMP[0] * newBps;
                offsetY = oldAnchorCY - scale * newMP[1] * newBps;
                anchored = true;
            }
        }
        if (!anchored) {
            const newPts = _gatherLandmark2D(fn, newIsLeft);
            if ((oldPts && oldPts.length) && newPts.length && oldAnchorCX != null) {
                let newPX = 0, newPY = 0;
                for (const [px, py] of newPts) { newPX += px; newPY += py; }
                newPX /= newPts.length; newPY /= newPts.length;
                offsetX = oldAnchorCX - scale * newPX * newBps;
                offsetY = oldAnchorCY - scale * newPY * newBps;
            } else {
                computeAutoCrop();
            }
        }

        _planTrackingPath(); // replan for new camera
        if (trackingZoom) _applyTrackingZoom(currentFrame, true);
        // Invalidate heatmap (bbox differs per camera)
        _heatmapImageData = null;
        if (showHeatmap) _prefetchHeatmap().then(() => render());
        render();
        update3D();          // per-camera error coloring depends on active side
        snapToCamera();      // re-snap with new camera params
    }

    /** When a Stereo-model joint is selected and at least one of the
     *  three Stereo variants is visible, return the OLD-camera stereo
     *  label position for that joint and the joint index.  Priority:
     *  Hybrid > Outline > Image.  Returns null if no selection / no
     *  visible variant / no data this frame. */
    function _stereoTargetForSwitch(fn, oldIsLeft) {
        if (stereoSelectedJoint == null || !trialData) return null;
        const j = stereoSelectedJoint;
        const trySrc = (vis, hasFlag, arrL, arrR) => {
            if (!vis || !trialData[hasFlag]) return null;
            const arr = oldIsLeft ? trialData[arrL] : trialData[arrR];
            const pt = arr?.[fn]?.[j];
            return pt ? { joint: j, pt } : null;
        };
        return (
            trySrc(showStereoHybrid2D,  'has_stereo_hybrid',
                   'stereo_hybrid_tracked_L',  'stereo_hybrid_tracked_R')
            || trySrc(showStereoOutline2D, 'has_stereo_outline',
                      'stereo_outline_tracked_L', 'stereo_outline_tracked_R')
            || trySrc(showStereo2D,        'has_stereo',
                      'stereo_tracked_L',         'stereo_tracked_R')
        );
    }

    /** Gather 2D landmark pixel positions for the current frame on the given camera side. */
    function _gatherLandmark2D(frameIdx, isLeft) {
        if (!trialData) return [];
        const pts = [];
        const sources = [
            isLeft ? trialData.skel_v2_proj_L : trialData.skel_v2_proj_R,
            isLeft ? trialData.skeleton_proj_L : trialData.skeleton_proj_R,
            isLeft ? trialData.mp_tracked_L : trialData.mp_tracked_R,
            isLeft ? trialData.vision_tracked_L : trialData.vision_tracked_R,
        ];
        // Use the first source that has data for this frame
        for (const src of sources) {
            const frame = src?.[frameIdx];
            if (!frame) continue;
            let count = 0;
            for (let j = 0; j < 21; j++) {
                if (frame[j]) { pts.push(frame[j]); count++; }
            }
            if (count > 5) break; // good enough
        }
        return pts;
    }

    function resetZoom() {
        trackingZoom = false;
        _userZoom = false;
        _updateTrackBtn();
        scale = defaultScale;
        offsetX = defaultOX;
        offsetY = defaultOY;
        render();
        applySnapProjection();
    }

    function toggleTrackingZoom() {
        trackingZoom = !trackingZoom;
        if (trackingZoom) _userZoom = false;
        _updateTrackBtn();
        if (trackingZoom) {
            // Plan path if needed and jump to current frame's segment
            if (!_trackSegments.length) _planTrackingPath();
            _applyTrackingZoom(currentFrame, true);
            // Snap immediately (no lerp for initial jump)
            if (_trackSegments.length) {
                let seg = _trackSegments[0];
                for (const s of _trackSegments) if (currentFrame >= s.startFrame) seg = s;
                scale = seg.scale; offsetX = seg.offsetX; offsetY = seg.offsetY;
            }
            render();
            applySnapProjection();
        } else {
            // Snap back to static auto-crop
            scale = defaultScale;
            offsetX = defaultOX;
            offsetY = defaultOY;
            render();
            applySnapProjection();
        }
    }

    function _updateTrackBtn() {
        const btn = $('trackZoomBtn');
        if (btn) {
            btn.classList.toggle('active', trackingZoom);
        }
    }

    /** Get hand bounding box for a single frame. Returns {minX,minY,maxX,maxY} or null. */
    function _getFrameBbox(frameIdx) {
        if (!trialData) return null;
        const isLeft = currentSide === cameraNames[0];
        const proj = isLeft ? trialData.skeleton_proj_L : trialData.skeleton_proj_R;
        const mp = isLeft ? trialData.mp_tracked_L : trialData.mp_tracked_R;
        const vis = isLeft ? trialData.vision_tracked_L : trialData.vision_tracked_R;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const pts of [proj?.[frameIdx], mp?.[frameIdx], vis?.[frameIdx]]) {
            if (!pts) continue;
            for (let j = 0; j < 21; j++) {
                if (!pts[j]) continue;
                if (pts[j][0] < minX) minX = pts[j][0];
                if (pts[j][1] < minY) minY = pts[j][1];
                if (pts[j][0] > maxX) maxX = pts[j][0];
                if (pts[j][1] > maxY) maxY = pts[j][1];
            }
        }
        return isFinite(minX) ? { minX, minY, maxX, maxY } : null;
    }

    /** Compute zoom/offset for a bounding box with given padding fraction. */
    function _cropFromBbox(minX, minY, maxX, maxY, pad) {
        if (!canvas || vidW === 0) return null;
        const bw = maxX - minX, bh = maxY - minY;
        minX -= bw * pad; maxX += bw * pad;
        minY -= bh * pad; maxY += bh * pad;
        const cropW = maxX - minX, cropH = maxY - minY;
        const isStereo = cameraMode === 'stereo';
        const isLeft = currentSide === cameraNames[0];
        const cw = canvas.width, ch = canvas.height;
        const sw = isStereo ? (isLeft ? midline : vidW - midline) : vidW;
        const bps = cw / sw;
        const tScale = Math.min(cw / (cropW * bps), ch / (cropH * bps));
        const cropCX = (minX + maxX) / 2, cropCY = (minY + maxY) / 2;
        return { scale: tScale, offsetX: cw / 2 - tScale * cropCX * bps, offsetY: ch / 2 - tScale * cropCY * bps };
    }

    /** Pre-plan the tracking zoom path for the current trial/camera.
     *
     * Algorithm: For each segment, anchor the crop on the starting frame's hand
     * position with generous padding. Hold that crop as long as the hand stays
     * within margin. When the hand leaves, start a new segment. This keeps the
     * camera zoomed in tight and only moves when the hand moves significantly.
     */
    function _planTrackingPath() {
        _trackSegments = [];
        _trackCurrentSeg = -1;
        if (!trialData || !canvas || vidW === 0) return;
        const N = trialData.n_frames;

        // Compute all per-frame bboxes
        const bboxes = [];
        for (let f = 0; f < N; f++) bboxes[f] = _getFrameBbox(f);

        const PAD = 0.40;
        const MIN_MARGIN = 0.05;

        let f = 0;
        while (f < N) {
            // Find next valid frame for anchor
            while (f < N && !bboxes[f]) f++;
            if (f >= N) break;

            // Anchor on this single frame's bbox
            const anchor = bboxes[f];
            const aw = anchor.maxX - anchor.minX, ah = anchor.maxY - anchor.minY;
            const cropMinX = anchor.minX - aw * PAD, cropMaxX = anchor.maxX + aw * PAD;
            const cropMinY = anchor.minY - ah * PAD, cropMaxY = anchor.maxY + ah * PAD;
            const cropW = cropMaxX - cropMinX, cropH = cropMaxY - cropMinY;

            // Extend segment as far as hand stays within this crop
            const segStart = f;
            let segEnd = f;
            for (let g = f; g < N; g++) {
                const bb = bboxes[g];
                if (!bb) { segEnd = g; continue; }
                const mL = (bb.minX - cropMinX) / cropW;
                const mR = (cropMaxX - bb.maxX) / cropW;
                const mT = (bb.minY - cropMinY) / cropH;
                const mB = (cropMaxY - bb.maxY) / cropH;
                if (Math.min(mL, mR, mT, mB) < MIN_MARGIN) break;
                segEnd = g;
            }

            const crop = _cropFromBbox(anchor.minX, anchor.minY, anchor.maxX, anchor.maxY, PAD);
            if (crop) _trackSegments.push({ startFrame: segStart, endFrame: segEnd, ...crop });
            f = segEnd + 1;
        }

        console.log(`Tracking path: ${_trackSegments.length} segment(s)`, _trackSegments.map(s => `${s.startFrame}-${s.endFrame} scale=${s.scale.toFixed(2)}`));
    }

    /** Apply pre-planned tracking zoom for the given frame. Smooth lerp between segments.
     *  After lerping, clamp offsets so the hand bbox stays in frame. */
    function _applyTrackingZoom(frameIdx, snap) {
        if (!_trackSegments.length) return;

        // Find target segment
        let targetSeg = _trackSegments[0];
        for (let i = 0; i < _trackSegments.length; i++) {
            if (frameIdx >= _trackSegments[i].startFrame) targetSeg = _trackSegments[i];
        }

        if (snap) {
            scale = targetSeg.scale;
            offsetX = targetSeg.offsetX;
            offsetY = targetSeg.offsetY;
        } else {
            const alpha = 0.12;
            scale += alpha * (targetSeg.scale - scale);
            offsetX += alpha * (targetSeg.offsetX - offsetX);
            offsetY += alpha * (targetSeg.offsetY - offsetY);

            // Clamp: ensure hand bbox stays visible in canvas during transitions
            const bb = _getFrameBbox(frameIdx);
            if (bb && canvas && vidW > 0) {
                const isStereo = cameraMode === 'stereo';
                const isLeft = currentSide === cameraNames[0];
                const sw = isStereo ? (isLeft ? midline : vidW - midline) : vidW;
                const bps = canvas.width / sw;
                const margin = 20; // px margin from canvas edge

                // Hand bbox in canvas coords: canvasX = offsetX + scale * px * bps
                const handLeft = offsetX + scale * bb.minX * bps;
                const handRight = offsetX + scale * bb.maxX * bps;
                const handTop = offsetY + scale * bb.minY * bps;
                const handBottom = offsetY + scale * bb.maxY * bps;

                // Push offsets so hand stays within canvas
                if (handLeft < margin) offsetX += (margin - handLeft);
                if (handRight > canvas.width - margin) offsetX -= (handRight - (canvas.width - margin));
                if (handTop < margin) offsetY += (margin - handTop);
                if (handBottom > canvas.height - margin) offsetY -= (handBottom - (canvas.height - margin));
            }
        }
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

        const isStereo = cameraMode === 'stereo';
        const isLeft = currentSide === cameraNames[0];
        const proj = isLeft ? trialData.skeleton_proj_L : trialData.skeleton_proj_R;
        const mp   = isLeft ? trialData.mp_tracked_L : trialData.mp_tracked_R;
        const vis  = isLeft ? trialData.vision_tracked_L : trialData.vision_tracked_R;
        // All 2D data is in camera-half coords — no offset needed
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

        // Sample every 5th frame for speed
        for (let f = 0; f < trialData.n_frames; f += 5) {
            const sources = [
                { pts: proj[f], off: 0 },
                { pts: mp[f], off: 0 },
                { pts: vis?.[f], off: 0 },
            ];
            for (const { pts, off } of sources) {
                if (!pts) continue;
                for (let j = 0; j < 21; j++) {
                    if (!pts[j]) continue;
                    const px = pts[j][0] + off;
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
        const sw = isStereo ? (isLeft ? midline : vidW - midline) : vidW;
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

        if (!trackingZoom && !playing && !_userZoom) {
            scale = defaultScale;
            offsetX = defaultOX;
            offsetY = defaultOY;
        }
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
        const nFrames = trialData?.n_frames || trials[currentTrialIdx]?.n_frames || 1;
        currentFrame = Math.max(0, Math.min(n, nFrames - 1));
        if (typeof setNavState === 'function') setNavState({ frame: currentFrame });

        $('frameDisplay').textContent = currentFrame;

        // Apply tracking zoom for manual frame stepping (snap, no lerp)
        if (trackingZoom && _trackSegments.length) {
            let seg = _trackSegments[0];
            for (const s of _trackSegments) if (currentFrame >= s.startFrame) seg = s;
            scale = seg.scale; offsetX = seg.offsetX; offsetY = seg.offsetY;
            applySnapProjection();
        }

        // Update non-video things immediately
        renderDistanceTrace();
        _scrollDistToFrame(currentFrame, false);

        // Update fit error
        const side = currentSide === cameraNames[0] ? 'fit_error_L' : 'fit_error_R';
        const err = trialData[side]?.[currentFrame];
        $('fitError').textContent = err != null ? err.toFixed(1) + 'px' : '-';

        // Seek then render — never call render() before seeked or the previous
        // decoded frame (or a blank) flashes before the new one arrives.
        // Invalidate heatmap for new frame
        _heatmapImageData = null;

        if (playing) {
            // Seek the video to the new frame; playback continues from there.
            const fps = trialData?.fps || trials[currentTrialIdx]?.fps || 30;
            if (videoEl && fps) videoEl.currentTime = (currentFrame + 0.5) / fps;
            if (showHeatmap) _prefetchHeatmap().then(() => render());
            return;
        }
        _seekGeneration++;
        const gen = _seekGeneration;
        if (showHeatmap) _prefetchHeatmap(); // fire-and-forget; will re-render when done
        const fps = trialData?.fps || trials[currentTrialIdx]?.fps || 30;
        if (videoEl && videoEl.readyState >= 2 && fps) {
            const t = (currentFrame + 0.5) / fps;
            videoEl.currentTime = t;
            videoEl.addEventListener('seeked', async () => {
                if (gen !== _seekGeneration || playing) return;
                if (showHeatmap && !_heatmapImageData) await _prefetchHeatmap();
                render(); update3D();
            }, { once: true });
        } else {
            (async () => {
                if (showHeatmap) await _prefetchHeatmap();
                render(); update3D();
            })();
        }
    }

    // Use requestVideoFrameCallback when available — it fires only when
    // a new frame is actually painted and reports the exact media time,
    // eliminating the off-by-one between overlays and displayed frame.
    const _hasRVFC = typeof HTMLVideoElement !== 'undefined'
                     && 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

    function _cancelPlayTimer() {
        if (!playTimer) return;
        if (_hasRVFC && videoEl) {
            try { videoEl.cancelVideoFrameCallback(playTimer); } catch {}
        } else {
            cancelAnimationFrame(playTimer);
        }
        playTimer = null;
    }

    function togglePlay() {
        if (playing) {
            playing = false;
            if (videoEl) videoEl.pause();
            _cancelPlayTimer();
            $('playBtn').innerHTML = '&#9654;';

            // Re-sync: derive frame from final video position, then seek to
            // the exact frame center so the paused image matches the overlay.
            if (videoEl && trialData) {
                const fps = trialData.fps || 30;
                const f = Math.round(videoEl.currentTime * fps);
                currentFrame = Math.max(0, Math.min(f, trialData.n_frames - 1));
            }
            goToFrame(currentFrame);
        } else {
            if (!videoEl || videoEl.readyState < 2) return;
            // Restart from beginning if at the last frame
            const nFrames = trialData?.n_frames || 1;
            if (currentFrame >= nFrames - 1) {
                currentFrame = 0;
                $('frameDisplay').textContent = 0;
            }
            playing = true;
            _seekGeneration++;
            $('playBtn').innerHTML = '&#9646;&#9646;';
            // Ensure video position matches currentFrame before playing
            const fps = trialData?.fps || 30;
            videoEl.currentTime = (currentFrame + 0.5) / fps;
            videoEl.playbackRate = Math.min(playbackRate, 16);
            videoEl.play().catch(() => { playing = false; $('playBtn').innerHTML = '&#9654;'; });
            _schedulePlayLoop();
        }
    }

    function _schedulePlayLoop() {
        if (_hasRVFC && videoEl) {
            playTimer = videoEl.requestVideoFrameCallback(_playLoopRVFC);
        } else {
            playTimer = requestAnimationFrame(_playLoopRAF);
        }
    }

    // rVFC path — metadata.mediaTime is the exact presentation timestamp
    function _playLoopRVFC(now, metadata) {
        if (!playing) return;
        _playUpdate(metadata.mediaTime);
        _schedulePlayLoop();
    }

    // rAF fallback
    function _playLoopRAF() {
        if (!playing) return;
        _playUpdate(videoEl ? videoEl.currentTime : 0);
        _schedulePlayLoop();
    }

    function _playUpdate(mediaTime) {
        try {
            if (!videoEl || !trialData) return;
            const f = Math.round(mediaTime * trialData.fps);
            if (f > trialData.n_frames - 1) {
                togglePlay();
                return;
            }
            if (f !== currentFrame && f >= 0 && f < trialData.n_frames) {
                currentFrame = f;
                $('frameDisplay').textContent = currentFrame;
                if (trackingZoom) _applyTrackingZoom(currentFrame);
                render();
                update3D();
                if (trackingZoom) applySnapProjection();
                renderDistanceTrace();
            }
        } catch (e) {
            console.error('playLoop error:', e);
        }
    }

    // ── Canvas events (zoom/pan) ─────────────────────────────
    function handleVideoZoom(e) {
        e.preventDefault();
        if (trackingZoom) { trackingZoom = false; _updateTrackBtn(); }
        _userZoom = true;
        // Use the viewport container rect (both canvases fill the same area)
        const viewport = canvas.parentElement.parentElement;
        const rect = viewport.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const factor = e.deltaY < 0 ? 1.07 : 1 / 1.07;
        const newScale = Math.max(0.1, Math.min(scale * factor, 50));
        offsetX = mx - (mx - offsetX) * (newScale / scale);
        offsetY = my - (my - offsetY) * (newScale / scale);
        scale = newScale;
        render();
        applySnapProjection();
    }

    function handleVideoPanStart(e) {
        if (trackingZoom) { trackingZoom = false; _updateTrackBtn(); }
        _userZoom = true;
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
            // Bbox editing intercepts left-click -- but only when an
            // editable green bbox is actually on screen.  With MP +
            // "Use bounding box" unchecked, the bbox is hidden and
            // any drag handles would be invisible; let the click fall
            // through to the normal pan/zoom path.
            const _ubCb = document.getElementById('mpUseBbox');
            const _mpNoBbox = _detectModel === 'run-mediapipe'
                && _ubCb && !_ubCb.checked;
            if (bboxEditMode && !_mpNoBbox && e.button === 0) {
                const isStereo = cameraMode === 'stereo';
                const isLeft = currentSide === cameraNames[0];
                const sw = isStereo ? (isLeft ? midline : vidW - midline) : vidW;
                const bps = canvas.width / sw;
                const rect = canvas.getBoundingClientRect();
                const cmx = (e.clientX - rect.left - offsetX) / scale;
                const cmy = (e.clientY - rect.top - offsetY) / scale;
                const handle = _bboxHandleHitTest(cmx, cmy, bps);
                if (handle) {
                    e.preventDefault();
                    e.stopPropagation();
                    const box = isLeft ? bboxOS : bboxOD;
                    bboxDrag = { handle, startMx: e.clientX, startMy: e.clientY, origBox: [...box] };
                    const onMove = ev => {
                        _applyBboxDrag(ev.clientX - bboxDrag.startMx, ev.clientY - bboxDrag.startMy, bps);
                    };
                    const onUp = () => {
                        bboxDrag = null;
                        document.removeEventListener('mousemove', onMove);
                        document.removeEventListener('mouseup', onUp);
                    };
                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup', onUp);
                    return;
                }
            }
            if (e.button === 0 || e.button === 1) handleVideoPanStart(e);
        });
        canvas.addEventListener('mousemove', handleVideoPanMove);
        canvas.addEventListener('mouseup', () => { dragging = null; });
        canvas.addEventListener('mouseleave', () => { dragging = null; });

        // Click on a joint dot → toggle its angle plot (same as Plotting hand SVG)
        canvas.addEventListener('click', e => {
            if (bboxEditMode) return;
            // Only fire if no significant drag occurred
            const dx = e.clientX - dragStartX, dy = e.clientY - dragStartY;
            if (dx * dx + dy * dy > 25) return;  // 5px drag threshold

            if (!trialData) return;
            const fn = currentFrame;
            const rect = canvas.getBoundingClientRect();
            // Mouse position in image-pixel coords
            const mx = (e.clientX - rect.left - offsetX) / scale;
            const my = (e.clientY - rect.top  - offsetY) / scale;
            const pixelScale = canvas.width / (cameraMode === 'stereo'
                ? (currentSide === cameraNames[0] ? midline : vidW - midline) : vidW);

            const isLeft = currentSide === cameraNames[0];

            // Build candidate list: {joint, source, dist²}
            const HIT_R = 12 / pixelScale;  // 12 canvas-px hit radius
            const HIT_R2 = HIT_R * HIT_R;
            let best = null;
            const _check = (projArr, source) => {
                if (!projArr?.[fn]) return;
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !projArr[fn][j]) continue;
                    const jx = projArr[fn][j][0], jy = projArr[fn][j][1];
                    const d2 = (jx - mx/pixelScale) ** 2 + (jy - my/pixelScale) ** 2;
                    if (d2 < HIT_R2 && (!best || d2 < best.d2)) {
                        best = { j, source, d2 };
                    }
                }
            };
            if (showMano2D)      _check(isLeft ? trialData.skeleton_proj_L : trialData.skeleton_proj_R, 'skeleton');
            // Skeleton 2D: check whatever stage projections are actually
            // being drawn (may be hr/bc/final), not always the master.
            if (showSkelV2_2D) {
                if (_stages2D.size === 0) {
                    _check(isLeft ? trialData.skel_v2_proj_L : trialData.skel_v2_proj_R, 'skel_v2');
                } else {
                    for (const s of _skelStages2D(isLeft)) _check(s.proj, 'skel_v2');
                }
            }
            if (showLegacyV2_2D) _check(isLeft ? trialData.skel_legacy_proj_L : trialData.skel_legacy_proj_R, 'skel_legacy');
            if (showMP2D)        _check(isLeft ? trialData.mp_tracked_L : trialData.mp_tracked_R, 'mp');
            if (showVision2D)    _check(isLeft ? trialData.vision_tracked_L : trialData.vision_tracked_R, 'vision');
            if (showHeatmap2D && trialData.hrnet_peaks) {
                const pd = isLeft ? trialData.hrnet_peaks.peaks_L : trialData.hrnet_peaks.peaks_R;
                const jnames = trialData.joint_names || [];
                if (pd && jnames.length) {
                    // Build a per-frame sparse array mirroring other projections.
                    const hmArr = (() => {
                        if (!trialData._hmProjCache) trialData._hmProjCache = { L: null, R: null };
                        const cache = trialData._hmProjCache;
                        const key = isLeft ? 'L' : 'R';
                        if (cache[key]) return cache[key];
                        const N = trialData.n_frames || 0;
                        const arr = new Array(N);
                        for (let f = 0; f < N; f++) arr[f] = new Array(21).fill(null);
                        for (let j = 0; j < 21; j++) {
                            const frames = pd[jnames[j]];
                            if (!frames) continue;
                            for (let f = 0; f < N; f++) {
                                if (frames[f]) arr[f][j] = frames[f];
                            }
                        }
                        cache[key] = arr;
                        return arr;
                    })();
                    _check(hmArr, 'heatmap');
                }
            }

            if (!best) return;

            // Toggle angle metric for this joint (same logic as Plotting hand click)
            const flexOpts = trialData?.flex_angle_options || [];
            const flexMatch = flexOpts.find(f => f.joint === best.j);
            if (!flexMatch) return;

            const flexName = flexMatch.name;
            const abdName  = flexName.replace('Flex:', 'Abd:');

            if (plotJointStates.has(best.j)) {
                const s = plotJointStates.get(best.j);
                selectedMetrics.delete(flexName);
                selectedMetrics.delete(abdName);
                plotJointStates.delete(best.j);
            } else {
                const c1 = _nextMetricColor();
                selectedMetrics.set(flexName, c1);
                const c2 = _nextMetricColor();
                selectedMetrics.set(abdName, c2);
                plotJointStates.set(best.j, { mode: 'both', colorFlex: c1, colorAbd: c2 });

                // Enable the clicked model's source if not already visible
                const srcMap = {
                    skeleton:        ['showMano2D',     'showMano3D'],
                    skel_v2:     ['showSkelV2_2D',  'showSkelV2_3D'],
                    skel_legacy: ['showLegacyV2_2D','showLegacyV2_3D'],
                    mp:          ['showMP2D',       'showMP3D'],
                    vision:      ['showVision2D',   'showVision3D'],
                };
                const ids = srcMap[best.source] || [];
                for (const id of ids) {
                    const el = $(id);
                    if (el && !el.checked) { el.checked = true; el.dispatchEvent(new Event('change')); }
                }
                _enforceSourceConstraint();
            }
            renderDistanceTrace();
            _updatePlotHighlight();
            update3D();
        });

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
            // Plain drag → pan; Cmd/Shift+drag → orbit (handled by non-capture listener)
            if (!(e.metaKey || e.shiftKey || e.ctrlKey) && (e.button === 0 || e.button === 1)) {
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

        // Distance trace click / constraint drag
        let _cstDrag = null; // {jointName, key, startY, startVal, toYAngle, angYMin, angYMax, plotH}
        distCanvas.addEventListener('mousedown', e => {
            if (!_constraintHitZones.length) return;
            const rect = distCanvas.getBoundingClientRect();
            const my = e.clientY - rect.top;
            // Hit-test constraint lines (8px tolerance)
            for (const hz of _constraintHitZones) {
                if (Math.abs(my - hz.y) < 8) {
                    const ovr = _constraintOverrides[hz.jointName] || {};
                    const cst = (trialData.angle_constraints || []).find(c => c.name === hz.jointName);
                    const curVal = ovr[hz.key] ?? (cst ? cst[hz.key] : 0);
                    _cstDrag = { jointName: hz.jointName, key: hz.key, startY: my, startVal: curVal,
                                 angYMin: hz.angYMin, angYMax: hz.angYMax, plotH: hz.plotH };
                    e.preventDefault();
                    return;
                }
            }
        });
        distCanvas.addEventListener('mousemove', e => {
            if (!_cstDrag) {
                // Change cursor when hovering over a constraint line
                if (_constraintHitZones.length) {
                    const rect = distCanvas.getBoundingClientRect();
                    const my = e.clientY - rect.top;
                    const near = _constraintHitZones.some(hz => Math.abs(my - hz.y) < 8);
                    distCanvas.style.cursor = near ? 'ns-resize' : 'crosshair';
                }
                return;
            }
            const rect = distCanvas.getBoundingClientRect();
            const my = e.clientY - rect.top;
            // Convert pixel Y back to angle value
            const { angYMin, angYMax, plotH } = _cstDrag;
            const angleVal = angYMin + (1 - my / plotH) * (angYMax - angYMin);
            if (!_constraintOverrides[_cstDrag.jointName]) _constraintOverrides[_cstDrag.jointName] = {};
            _constraintOverrides[_cstDrag.jointName][_cstDrag.key] = Math.round(angleVal / 5) * 5;
            renderDistanceTrace();
        });
        distCanvas.addEventListener('mouseup', () => { _cstDrag = null; });
        distCanvas.addEventListener('mouseleave', () => { _cstDrag = null; });
        distCanvas.addEventListener('click', e => {
            if (_cstDrag) return;
            if (!trialData) return;
            const rect = distCanvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const f = Math.floor((mx / rect.width) * trialData.n_frames);

            // Check if click is near a plotted angle line → toggle constraint focus
            const HIT_PX = 8;
            let hitMetric = null;
            let hitDist = HIT_PX;
            for (const [metric, info] of Object.entries(_plotMetricData)) {
                if (!metric.startsWith('Flex:') && !metric.startsWith('Abd:') && !metric.startsWith('MCP:')) continue;
                const val = info.data[f];
                if (val == null) continue;
                const lineY = info.toY(val);
                const d = Math.abs(my - lineY);
                if (d < hitDist) { hitDist = d; hitMetric = metric; }
            }

            if (hitMetric) {
                // Toggle: if already focused on this metric, unfocus
                _constraintFocusMetric = (_constraintFocusMetric === hitMetric) ? null : hitMetric;
                renderDistanceTrace();
                return;
            }

            // No line hit — clear focus if active, and always seek frame
            if (_constraintFocusMetric) {
                _constraintFocusMetric = null;
                renderDistanceTrace();
            }
            goToFrame(f);
        });

        // Resize observer — observe the viewport container (.skeleton-viewports)
        const viewport = canvas.parentElement.parentElement;
        const ro = new ResizeObserver(() => {
            sizeCanvases();
            computeAutoCrop();
            _planTrackingPath();
            if (trackingZoom) _applyTrackingZoom(currentFrame, true);
            render();
            renderDistanceTrace();
            applySnapProjection();
        });
        ro.observe(viewport);

        // ── Touch/iPad support ────────────────────────────────────────────
        // Single-touch gestures → synthetic mouse events (existing handlers work).
        // Pinch → synthetic wheel events (handleVideoZoom works unchanged).
        // The 3D container is above the video canvas; enable touch on both so
        // pan/tap work regardless of which layer is visible.
        if (typeof window.pinchAsWheel === 'function') {
            window.pinchAsWheel(canvas);
            window.pinchAsWheel(threeContainer);
        }
        // Distance plot: single-touch tap/drag for seek and constraint editing
        if (typeof window.enableTouch === 'function') {
            window.enableTouch(distCanvas);
        }
    }

    /** Scroll the distance canvas so `frame` is visible.
     *  `force=true` centres it; otherwise only scrolls when it's offscreen. */
    function _scrollDistToFrame(frame, force) {
        const wrap = distCanvas && distCanvas.parentElement;
        if (!wrap || !trialData) return;
        const N = trialData.n_frames || 1;
        if (N < 2 || distCanvas.width <= wrap.clientWidth + 1) return;
        const x = (frame / (N - 1)) * distCanvas.width;
        const left = wrap.scrollLeft;
        const right = left + wrap.clientWidth;
        const MARGIN = 32;
        if (force) {
            wrap.scrollLeft = Math.max(0, x - wrap.clientWidth / 2);
        } else if (x < left + MARGIN) {
            wrap.scrollLeft = Math.max(0, x - MARGIN);
        } else if (x > right - MARGIN) {
            wrap.scrollLeft = Math.min(distCanvas.width - wrap.clientWidth,
                                        x - wrap.clientWidth + MARGIN);
        }
    }

    function sizeCanvases() {
        // Both canvases fill the same viewport container
        const viewport = canvas.parentElement.parentElement; // .skeleton-viewports
        const vw = viewport.clientWidth;
        const vh = viewport.clientHeight;
        canvas.width = vw;
        canvas.height = vh;

        // Distance canvas lives inside a horizontal-scroll container.
        // Its drawable width = container width × user-controlled x-scale.
        // Height is fixed to the wrap's own pixel height so growing the
        // canvas width can't feed back into a taller parent.
        const distWrap = distCanvas.parentElement;
        const containerW = (distWrap && distWrap.clientWidth) || 600;
        const containerH = (distWrap && distWrap.clientHeight) || 115;
        const scaledW = Math.max(10, Math.round(containerW * _xScale));
        distCanvas.style.width = scaledW + 'px';
        distCanvas.style.height = containerH + 'px';
        distCanvas.width = scaledW;
        distCanvas.height = containerH;

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
        _computeBoneLengthFlags();
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // When the <video> element fails to decode (unsupported codec —
        // notably yuvj420p in some browsers and the legacy mpeg4 ASP used
        // by older de-identified renders), readyState stays at 0 and vidW
        // never populates.  We still want to render the 2D labels on a
        // black canvas so the page remains useful — and to make the
        // failure visible, log the underlying media error once.
        const videoFailed = !!(videoEl && videoEl.error);
        if (videoFailed && !videoEl._loggedError) {
            videoEl._loggedError = true;
            const e = videoEl.error;
            console.error('[skeleton] video failed to decode:', {
                code: e?.code, message: e?.message, src: videoEl.src,
            });
        }
        // Fallback dimensions when video is broken: derive from trialData
        // (or use a sensible 1920×1080 stereo half-default) so overlays
        // and zoom math have something to scale against.
        if ((!videoEl || videoEl.readyState < 2 || vidW === 0)) {
            if (trialData) {
                if (!vidW) vidW = trialData.video_width  || 3840;
                if (!vidH) vidH = trialData.video_height || 1080;
                midline = cameraMode === 'stereo' ? vidW / 2 : vidW;
                if (canvas.width === 0) sizeCanvases();
            }
        }

        // Draw video frame + overlays with consistent zoom/pan transform.
        // When the video is unavailable, skip the drawImage call but still
        // run the overlay pass so labels remain visible.
        if (videoEl && videoEl.readyState >= 2 && vidW > 0) {
            const isStereo = cameraMode === 'stereo';
            const isLeft = currentSide === cameraNames[0];
            const sx = isStereo ? (isLeft ? 0 : midline) : 0;
            const sw = isStereo ? (isLeft ? midline : vidW - midline) : vidW;

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

            // Bounding box overlay for detection
            _drawBboxOverlay(bps);

            ctx.restore();
        } else if (trialData && vidW > 0) {
            // Video failed to decode — render labels on a black canvas so
            // the page is still usable for landmark / fit review.  Same
            // zoom/pan transform as the video path so the overlays line
            // up correctly with where the video would be.
            const isStereo = cameraMode === 'stereo';
            const isLeft = currentSide === cameraNames[0];
            const sw = isStereo ? (isLeft ? midline : vidW - midline) : vidW;
            const bps = w / sw;
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, w, h);
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            drawOverlays(bps);
            _drawBboxOverlay(bps);
            ctx.restore();
            // Visible hint to the user — once per render call.
            ctx.save();
            ctx.fillStyle = videoFailed ? 'rgba(255,80,80,0.85)' : 'rgba(180,180,180,0.7)';
            ctx.font = '12px sans-serif';
            ctx.fillText(videoFailed
                ? 'Video codec unsupported by browser — labels only'
                : 'Video loading…', 8, 16);
            ctx.restore();
        }
    }

    // Return list of { proj, color, errors } for the Skeleton layer.
    // If any stage buttons are active, one entry per stage; otherwise the
    // default (corrected Skeleton data, orange, no errors).
    // Compute per-bone flag matrix using the same percentile threshold as
    // the bone-length detector: pool |length - median| across all bones &
    // frames, flag frames whose dev exceeds the (1-W) quantile.  Result:
    // { "a-b": boolArrayN } keyed with a<b, plus medians + threshold.
    function _computeBoneLengthFlags() {
        if (!trialData || !_stagesErr.has('z_smooth')) {
            _boneFlagsByPair = null;
            return { flagsByPair: null, medians: null, threshold: null };
        }
        const N = trialData.n_frames || 0;
        const skel = trialData.skeleton || [];
        // z_smooth shows the pre-bone-length model → use that model's
        // bone lengths when computing the flags.
        const distsV2 = trialData.distances_skel_v2_zs || trialData.distances_skel_v2 || {};
        const distOpts = trialData.distance_options || {};
        const medians = {};        // bone metric name → median
        const pairKeyOf = {};      // bone metric name → "a-b" (a<b)
        const pooledDevs = [];
        const bonesFound = [];
        for (const [a, b] of skel) {
            let boneName = null;
            for (const [nm, pr] of Object.entries(distOpts)) {
                if ((pr[0] === a && pr[1] === b) || (pr[0] === b && pr[1] === a)) { boneName = nm; break; }
            }
            if (!boneName) continue;
            const series = distsV2[boneName];
            if (!series) continue;
            const vals = series.filter(v => v != null).slice().sort((x, y) => x - y);
            if (vals.length < 3) continue;
            const med = vals[Math.floor(vals.length / 2)];
            medians[boneName] = med;
            const lo = Math.min(a, b), hi = Math.max(a, b);
            pairKeyOf[boneName] = `${lo}-${hi}`;
            bonesFound.push({ boneName, series, med });
            for (const v of series) if (v != null) pooledDevs.push(Math.abs(v - med));
        }
        const W = _mpErrorWeights.detection.bone_length || 0;
        let threshold = null;
        if (pooledDevs.length > 0 && W > 0) {
            pooledDevs.sort((a, b) => a - b);
            const idx = Math.min(pooledDevs.length - 1,
                                 Math.max(0, Math.floor(pooledDevs.length * (1 - W))));
            threshold = pooledDevs[idx];
        }
        const flagsByPair = {};
        if (threshold != null && threshold > 0) {
            for (const { boneName, series, med } of bonesFound) {
                const flags = new Array(N).fill(false);
                for (let f = 0; f < N; f++) {
                    const v = series[f];
                    if (v != null && Math.abs(v - med) > threshold) flags[f] = true;
                }
                flagsByPair[pairKeyOf[boneName]] = flags;
            }
        }
        _boneFlagsByPair = flagsByPair;
        return { flagsByPair, medians, threshold, pairKeyOf };
    }

    // Is joint `j` adjacent to a bone whose length is out of range at frame
    // `fn`?  Used to pair the joint-red rendering with the bone-red
    // rendering on the z_smooth / bone_length stage, so we never get a red
    // joint without at least one adjacent red bone.
    function _isJointFlaggedByBoneLength(j, fn) {
        if (!_boneFlagsByPair) return false;
        const skel = trialData?.skeleton || [];
        for (const [a, b] of skel) {
            if (a !== j && b !== j) continue;
            const lo = Math.min(a, b), hi = Math.max(a, b);
            if (_boneFlagsByPair[`${lo}-${hi}`]?.[fn]) return true;
        }
        return false;
    }

    function _skelStages2D(isLeft) {
        if (_stages2D.size === 0) {
            return [{
                proj: isLeft ? trialData.skel_v2_proj_L : trialData.skel_v2_proj_R,
                color: '#ff9800',
                errors: null,
            }];
        }
        const out = [];
        for (const name of _stages2D) {
            const cfg = STAGE_CONFIGS[name];
            if (!cfg) continue;
            out.push({
                proj: cfg.proj2D(isLeft, trialData),
                color: cfg.color,
                errors: _stagesErr.has(name) ? (skelErrorMatrices[name] || null) : null,
                factor: _stagesErr.has(name) ? cfg.factor : null,
            });
        }
        return out;
    }

    function _skelStages3D(isLeftSide) {
        if (_stages3D.size === 0) {
            return [{ pts: trialData.skel_v2_joints_3d, color: 0xff9800, emissive: 0x804400, errors: null,
                      proj2D: isLeftSide ? trialData.skel_v2_proj_L : trialData.skel_v2_proj_R }];
        }
        const out = [];
        const STAGE_PTS = {
            mediapipe: () => trialData.mp_joints_3d,
            // Fall back to the final Skeleton 3D if intermediates weren't saved
            stereo_correct: () => trialData.skel_v2_joints_sc_3d || trialData.mp_joints_3d,
            z_correct: () => trialData.skel_v2_joints_z_3d || trialData.skel_v2_joints_3d,
            z_smooth:  () => trialData.skel_v2_joints_zs_3d || trialData.skel_v2_joints_3d,
            hrnet_snap: () => trialData.skel_v2_joints_hr_3d || trialData.skel_v2_joints_3d,
            bone_correct: () => trialData.skel_v2_joints_bc_3d || trialData.skel_v2_joints_3d,
            bone_smooth: () => trialData.skel_v2_joints_3d,
        };
        for (const name of _stages3D) {
            const cfg = STAGE_CONFIGS[name];
            if (!cfg) continue;
            const getPts = STAGE_PTS[name];
            if (!getPts) continue;
            // Stage's per-frame 2D — used to pixel-anchor the 3D rendering
            // so the sphere lands on the stage's own 2D label even when
            // the stereo calibration has reprojection error.
            let proj2D = null;
            try { proj2D = cfg.proj2D ? cfg.proj2D(isLeftSide, trialData) : null; } catch {}
            out.push({
                pts: getPts(),
                color: cfg.color3d,
                emissive: cfg.emissive3d,
                errors: _stagesErr.has(name) ? (skelErrorMatrices[name] || null) : null,
                factor: _stagesErr.has(name) ? cfg.factor : null,
                proj2D,
            });
        }
        return out;
    }

    function drawOverlays(pixelScale) {
        const fn = currentFrame;
        if (!trialData || fn >= trialData.n_frames) return;

        const isStereo = cameraMode === 'stereo';
        const isLeft = currentSide === cameraNames[0];
        // Heatmap drawn first so 2D labels, skeletons, and everything else
        // render on top of it.
        if (showHeatmap && _heatmapImageData) {
            drawHeatmapOverlay(pixelScale, 0);
        }
        // Gather current-camera heatmap peaks for this frame.
        //   hmPeaks       = centroid-refined ("Peak select" stage)
        //   hmPeaksRaw    = raw argmax ("HRnet" stage)
        const _gatherPeaks = (rawKey, refinedKey, want) => {
            if (!want) return null;
            const pd = trialData?.hrnet_peaks;
            if (!pd) return null;
            const camPeaks = isLeft ? pd[isLeft ? rawKey : refinedKey] : pd[isLeft ? refinedKey : rawKey];
            // Note: structure is `pd.peaks_L`, `pd.peaks_R`, etc.  We pick
            // peaks_L_raw / peaks_R_raw vs peaks_L / peaks_R based on which
            // stage is being rendered.
            return null;
        };
        // Lookup helper for HRnet peak fields stored in trialData.hrnet_peaks.
        // Supported ``kind`` values:
        //   'centroid'  → HRnet Fit Stage-1 cluster centroid (canonical "Peaks")
        //   'hungarian' → HRnet Fit Stage-2 joint stereo Hungarian
        //   'raw'       → legacy raw argmax (pre-HRnet-Fit)
        //   'refined'   → legacy MP-Hungarian Peak-Select (deprecated)
        const _peakArrFor = (cam, kind) => {
            const pd = trialData?.hrnet_peaks;
            if (!pd) return null;
            const FIELD = {
                centroid:  cam === 'L' ? 'peaks_centroid_L'  : 'peaks_centroid_R',
                yzc:       cam === 'L' ? 'peaks_yzc_L'       : 'peaks_yzc_R',
                zsmooth:   cam === 'L' ? 'peaks_zsmooth_L'   : 'peaks_zsmooth_R',
                hungarian: cam === 'L' ? 'peaks_hungarian_L' : 'peaks_hungarian_R',
                raw:       cam === 'L' ? 'peaks_L_raw'       : 'peaks_R_raw',
                refined:   cam === 'L' ? 'peaks_L'           : 'peaks_R',
            };
            // For the canonical "Peaks" sub-stage, prefer the HRnet Fit
            // centroid output if present, fall back to legacy raw argmax.
            let camPeaks = pd[FIELD[kind]];
            if ((!camPeaks || !Object.keys(camPeaks).length) && kind === 'centroid') {
                camPeaks = pd[FIELD.raw];
            }
            const jnames = trialData?.joint_names || [];
            if (!camPeaks || !Object.keys(camPeaks).length || !jnames.length) return null;
            const out = new Array(21).fill(null);
            for (let j = 0; j < 21; j++) {
                const name = jnames[j];
                const frames = camPeaks[name];
                if (frames && frames[fn]) out[j] = frames[fn];
            }
            return out;
        };
        const hmCam = isLeft ? 'L' : 'R';
        const hmPeaks    = showHeatmap2D ? _peakArrFor(hmCam, 'refined') : null;
        const hmPeaksRaw = showHRnet2D ? _peakArrFor(hmCam, 'centroid') : null;
        const hmPeaksYZC = showHRnetYZC2D ? _peakArrFor(hmCam, 'yzc') : null;
        const hmPeaksZSM = showHRnetZSM2D ? _peakArrFor(hmCam, 'zsmooth') : null;
        const hmPeaksHun = showStereoHun2D ? _peakArrFor(hmCam, 'hungarian') : null;
        const manoProj = isLeft ? trialData.skeleton_proj_L : trialData.skeleton_proj_R;
        const v2Proj = isLeft ? trialData.skel_v2_proj_L : trialData.skel_v2_proj_R;
        const legacyProj = isLeft ? trialData.skel_legacy_proj_L : trialData.skel_legacy_proj_R;
        // Historical-fit 2D projection -- pick the active stage's arrays.
        // 'final' is the bone-smooth output (proj_L / proj_R); other tags
        // map to proj_<tag>_L / proj_<tag>_R written by the fit-history
        // loader.  Falls back to 'final' if the selected stage is missing.
        const _prevProj2DFor = (tag) => {
            if (!prevFitData) return null;
            if (tag === 'final') return isLeft ? prevFitData.proj_L : prevFitData.proj_R;
            return isLeft ? prevFitData[`proj_${tag}_L`] : prevFitData[`proj_${tag}_R`];
        };
        let prevProj = _prevProj2DFor(_prevStage2D);
        if (prevProj == null && prevFitData) {
            prevProj = isLeft ? prevFitData.proj_L : prevFitData.proj_R;
        }
        // Use corrected MP positions when available (after a correction pass
        // has been applied server-side).  Falls back to original otherwise.
        const mpKp = isLeft
            ? (mpCorrectedL || trialData.mp_tracked_L)
            : (mpCorrectedR || trialData.mp_tracked_R);
        const visionKp = isLeft ? trialData.vision_tracked_L : trialData.vision_tracked_R;
        // Reverse-pass MediaPipe layer — same schema as MP but loaded
        // from mediapipe_reverse_prelabels.npz.  Empty arrays when no
        // reverse run exists for this subject yet.
        const reverseKp = isLeft
            ? trialData.reverse_tracked_L
            : trialData.reverse_tracked_R;
        // Cropped-forward MediaPipe layer (forward with bbox crop).
        const croppedKp = isLeft
            ? trialData.cropped_tracked_L
            : trialData.cropped_tracked_R;
        // Static-mode MediaPipe layer (no temporal tracker).
        const staticKp = isLeft
            ? trialData.static_tracked_L
            : trialData.static_tracked_R;
        // Combined MediaPipe layer (per-camera-frame forward/reverse fusion).
        const combinedKp = isLeft
            ? trialData.combined_tracked_L
            : trialData.combined_tracked_R;

        // All 2D data is in camera-half coords — no offset needed
        const manoXOff = 0;
        const mpXOff = 0;
        const visionXOff = 0;

        // Draw skeleton lines (per-model skeleton toggle)
        if (trialData.skeleton) {
            trialData.skeleton.forEach(([i, j]) => {
                if (!isBoneVisible(i, j)) return;

                // Skeleton skeleton v1 (lime)
                if (showManoSkel && showMano2D && manoProj[fn] && manoProj[fn][i] && manoProj[fn][j]) {
                    drawLine(
                        (manoProj[fn][i][0] + manoXOff) * pixelScale,
                        manoProj[fn][i][1] * pixelScale,
                        (manoProj[fn][j][0] + manoXOff) * pixelScale,
                        manoProj[fn][j][1] * pixelScale,
                        'lime', 2, 0.7
                    );
                }

                // Skeleton (v3) — one bone per active stage.  Bone goes red
                // when BOTH endpoints are flagged (natural match for
                // bone_length errors, which pair their endpoints).
                if (showSkelV2Skel && showSkelV2_2D) {
                    for (const s of _skelStages2D(isLeft)) {
                        if (s.proj?.[fn]?.[i] && s.proj[fn][j]) {
                            let bc = s.color;
                            if (s.factor === 'bone_length' && _boneFlagsByPair) {
                                const lo = Math.min(i, j), hi = Math.max(i, j);
                                if (_boneFlagsByPair[`${lo}-${hi}`]?.[fn]) bc = '#ff2222';
                            } else if (s.factor === 'bone_agreement' && s.errors?.[fn]) {
                                const eI = s.errors[fn][i];
                                const eJ = s.errors[fn][j];
                                const eiOn = eI && (eI[0] || eI[1]);
                                const ejOn = eJ && (eJ[0] || eJ[1]);
                                if (eiOn && ejOn) bc = '#ff2222';
                            }
                            drawLine(
                                s.proj[fn][i][0] * pixelScale,
                                s.proj[fn][i][1] * pixelScale,
                                s.proj[fn][j][0] * pixelScale,
                                s.proj[fn][j][1] * pixelScale,
                                bc, 2, 0.7
                            );
                        }
                    }
                }

                // Skeleton v2 legacy (magenta)
                if (showLegacyV2Skel && showLegacyV2_2D && legacyProj?.[fn]?.[i] && legacyProj[fn][j]) {
                    drawLine(
                        legacyProj[fn][i][0] * pixelScale,
                        legacyProj[fn][i][1] * pixelScale,
                        legacyProj[fn][j][0] * pixelScale,
                        legacyProj[fn][j][1] * pixelScale,
                        '#e040fb', 2, 0.7
                    );
                }

                // MediaPipe skeleton (cyan)
                if (showMPSkel && showMP2D && mpKp[fn] && mpKp[fn][i] && mpKp[fn][j]) {
                    drawLine(
                        (mpKp[fn][i][0] + mpXOff) * pixelScale,
                        mpKp[fn][i][1] * pixelScale,
                        (mpKp[fn][j][0] + mpXOff) * pixelScale,
                        mpKp[fn][j][1] * pixelScale,
                        '#00cccc', 1.5, 0.5
                    );
                }

                // Reverse (magenta) — same bone connections as MP,
                // gated independently so the user can compare the two
                // skeletons side-by-side.
                if (showReverseSkel && showReverse2D
                    && reverseKp?.[fn]?.[i] && reverseKp[fn][j]) {
                    drawLine(
                        reverseKp[fn][i][0] * pixelScale,
                        reverseKp[fn][i][1] * pixelScale,
                        reverseKp[fn][j][0] * pixelScale,
                        reverseKp[fn][j][1] * pixelScale,
                        '#e040fb', 1.5, 0.5
                    );
                }
                // Cropped (green) — forward MP with bbox crop applied.
                if (showCroppedSkel && showCropped2D
                    && croppedKp?.[fn]?.[i] && croppedKp[fn][j]) {
                    drawLine(
                        croppedKp[fn][i][0] * pixelScale,
                        croppedKp[fn][i][1] * pixelScale,
                        croppedKp[fn][j][0] * pixelScale,
                        croppedKp[fn][j][1] * pixelScale,
                        '#7cb342', 1.5, 0.5
                    );
                }
                // Static (cyan) — per-frame palm detector, no tracker.
                if (showStaticSkel && showStatic2D
                    && staticKp?.[fn]?.[i] && staticKp[fn][j]) {
                    drawLine(
                        staticKp[fn][i][0] * pixelScale,
                        staticKp[fn][i][1] * pixelScale,
                        staticKp[fn][j][0] * pixelScale,
                        staticKp[fn][j][1] * pixelScale,
                        '#26c6da', 1.5, 0.5
                    );
                }
                // Combined (orange) — fused forward+reverse layer.
                if (showCombinedSkel && showCombined2D
                    && combinedKp?.[fn]?.[i] && combinedKp[fn][j]) {
                    drawLine(
                        combinedKp[fn][i][0] * pixelScale,
                        combinedKp[fn][i][1] * pixelScale,
                        combinedKp[fn][j][0] * pixelScale,
                        combinedKp[fn][j][1] * pixelScale,
                        '#ffa726', 1.5, 0.5
                    );
                }

                // Vision skeleton (blue)
                if (showVisionSkel && showVision2D && visionKp?.[fn]?.[i] && visionKp[fn][j]) {
                    drawLine(
                        (visionKp[fn][i][0] + visionXOff) * pixelScale,
                        visionKp[fn][i][1] * pixelScale,
                        (visionKp[fn][j][0] + visionXOff) * pixelScale,
                        visionKp[fn][j][1] * pixelScale,
                        '#2196f3', 1.5, 0.5
                    );
                }

                // Previous fit skeleton (dark orange)
                if (showPrevSkel && showPrev2D && prevProj?.[fn]?.[i] && prevProj[fn][j]) {
                    drawLine(
                        prevProj[fn][i][0] * pixelScale, prevProj[fn][i][1] * pixelScale,
                        prevProj[fn][j][0] * pixelScale, prevProj[fn][j][1] * pixelScale,
                        '#b35b00', 2, 0.6
                    );
                }
            });
        }

        // Skeleton v1 joints (lime circles)
        if (showMano2D && manoProj[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !manoProj[fn][j]) continue;
                const x = (manoProj[fn][j][0] + manoXOff) * pixelScale;
                const y = manoProj[fn][j][1] * pixelScale;
                drawJoint(x, y, 'lime', 4);
            }
        }

        // Skeleton (v3) joints — per active stage.  Each stage uses its own
        // data source and base color; flagged joints turn red for this-camera
        // errors, and get a red circle when flagged on either camera.
        if (showSkelV2_2D) {
            const camIdx = isLeft ? 0 : 1;
            const otherIdx = isLeft ? 1 : 0;
            for (const s of _skelStages2D(isLeft)) {
                if (!s.proj?.[fn]) continue;
                const errFrame = s.errors ? s.errors[fn] : null;
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !s.proj[fn][j]) continue;
                    const x = s.proj[fn][j][0] * pixelScale;
                    const y = s.proj[fn][j][1] * pixelScale;
                    let color = s.color;
                    let circleRed = false;
                    if (s.factor === 'bone_length') {
                        const flagged = _isJointFlaggedByBoneLength(j, fn);
                        if (flagged) { color = '#ff2222'; circleRed = true; }
                    } else if (errFrame && errFrame[j]) {
                        const thisCamErr  = !!errFrame[j][camIdx];
                        const otherCamErr = !!errFrame[j][otherIdx];
                        if (thisCamErr) color = '#ff2222';
                        circleRed = thisCamErr || otherCamErr;
                    }
                    drawJoint(x, y, color, 4);
                    if (circleRed) {
                        ctx.save();
                        ctx.strokeStyle = '#ff2222';
                        ctx.lineWidth = 1.5 / scale;
                        ctx.beginPath();
                        ctx.arc(x, y, 7, 0, Math.PI * 2);
                        ctx.stroke();
                        ctx.restore();
                    }
                }
            }
        }

        // Skeleton v2 legacy joints (magenta circles)
        if (showLegacyV2_2D && legacyProj?.[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !legacyProj[fn][j]) continue;
                const x = legacyProj[fn][j][0] * pixelScale;
                const y = legacyProj[fn][j][1] * pixelScale;
                drawJoint(x, y, '#e040fb', 4);
            }
        }

        // Stereo-correct OCCLUSION radius preview: whenever the v3 fit
        // panel is open and the Occlusion-radius slider is > 0, draw a
        // thin dashed circle of that radius (image-pixel units) around
        // every MP joint so the user can preview the overlap region.
        // Shown regardless of the MP-2D checkbox.
        {
            const _occPx = parseFloat(
                _mpErrorWeights?.detection?.stereo_occlusion || 0);
            const _v3PanelOpen = $('fitV2Panel')?.style.display === 'block';
            if (_occPx > 0 && _v3PanelOpen && mpKp && mpKp[fn]) {
                ctx.save();
                ctx.strokeStyle = 'rgba(0, 204, 204, 0.55)';
                ctx.lineWidth = 0.8;
                ctx.setLineDash([3, 3]);
                const rPx = _occPx * pixelScale;
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !mpKp[fn][j]) continue;
                    const x = (mpKp[fn][j][0] + mpXOff) * pixelScale;
                    const y = mpKp[fn][j][1] * pixelScale;
                    ctx.beginPath();
                    ctx.arc(x, y, rPx, 0, Math.PI * 2);
                    ctx.stroke();
                }
                ctx.setLineDash([]);
                ctx.restore();
            }
        }

        // MediaPipe joints (always cyan; errors are displayed on the
        // Skeleton layer via the stage-picker buttons instead)
        if (showMP2D && mpKp[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !mpKp[fn][j]) continue;
                const x = (mpKp[fn][j][0] + mpXOff) * pixelScale;
                const y = mpKp[fn][j][1] * pixelScale;
                drawCross(x, y, '#00cccc', 4);
            }
            // When the MP-stage Err column is on, overlay pink X's at
            // the per-joint stereo-correct position for joints whose
            // BLAMED camera is the current view.  Joint set comes
            // from the live MP-stage error matrix (responds to the
            // Stereo conf / Stereo distance sliders); positions come
            // from the baked stereo_L_pts / stereo_R_pts.
            if (_stagesErr.has('mediapipe')) {
                const camIdx = isLeft ? 0 : 1;
                const stereoPts = isLeft ? trialData.stereo_L_pts
                                         : trialData.stereo_R_pts;
                const liveErr = skelErrorMatrices?.mediapipe;
                if (stereoPts && liveErr && liveErr[fn]) {
                    for (let j = 0; j < 21; j++) {
                        if (!isJointVisible(j)) continue;
                        const eRow = liveErr[fn][j];
                        if (!eRow || !eRow[camIdx]) continue;
                        const sp = stereoPts[fn]?.[j];
                        if (!sp) continue;
                        drawCross(sp[0] * pixelScale, sp[1] * pixelScale,
                                   '#f48fb1', 5);
                    }
                }
            }
        }

        // Reverse-pass MediaPipe joint markers (magenta).  Skeleton
        // lines are drawn alongside MP's in the bone loop above so the
        // user can compare the two skeletons; this block handles the
        // per-joint dots.  Values come from
        // mediapipe_reverse_prelabels.npz.
        if (showReverse2D && reverseKp && reverseKp[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !reverseKp[fn][j]) continue;
                const x = reverseKp[fn][j][0] * pixelScale;
                const y = reverseKp[fn][j][1] * pixelScale;
                drawCross(x, y, '#e040fb', 4);
            }
        }

        // Cropped-forward joint markers (green).
        if (showCropped2D && croppedKp && croppedKp[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !croppedKp[fn][j]) continue;
                const x = croppedKp[fn][j][0] * pixelScale;
                const y = croppedKp[fn][j][1] * pixelScale;
                drawCross(x, y, '#7cb342', 4);
            }
        }

        // Static-mode MediaPipe joint markers (cyan).
        if (showStatic2D && staticKp && staticKp[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !staticKp[fn][j]) continue;
                const x = staticKp[fn][j][0] * pixelScale;
                const y = staticKp[fn][j][1] * pixelScale;
                drawCross(x, y, '#26c6da', 4);
            }
        }

        // Combined-pass MediaPipe joint markers (orange).  Same pattern
        // as Reverse: bones in the loop above, per-joint dots here.
        if (showCombined2D && combinedKp && combinedKp[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !combinedKp[fn][j]) continue;
                const x = combinedKp[fn][j][0] * pixelScale;
                const y = combinedKp[fn][j][1] * pixelScale;
                drawCross(x, y, '#ffa726', 4);
            }
        }

        // Stereo: partner-camera MP label translated into this camera
        // by the phase-correlation shift discovered by ``run_stereo``.
        // Pink (#e91e63) crosses; size scaled by per-joint confidence.
        // Also overlays the crop windows: solid pink rect = hand-wide
        // (pass-1) crop, dashed pink rect = per-joint (pass-2) crop
        // for the currently-selected joint.
        if (showStereo2D && trialData?.has_stereo) {
            const stereoKp = isLeft ? trialData.stereo_tracked_L
                                    : trialData.stereo_tracked_R;
            const mpHere = isLeft ? trialData.mp_tracked_L
                                  : trialData.mp_tracked_R;
            // Hand-wide bbox (solid).
            if (mpHere && mpHere[fn]) {
                let sx = 0, sy = 0, n = 0;
                for (let j = 0; j < 21; j++) {
                    const pt = mpHere[fn][j];
                    if (!pt) continue;
                    sx += pt[0]; sy += pt[1]; n += 1;
                }
                const hch = (trialData.stereo_hand_crop_half != null
                             ? trialData.stereo_hand_crop_half : 80);
                if (n > 0) {
                    const cx = (sx / n) * pixelScale;
                    const cy = (sy / n) * pixelScale;
                    const w = (2 * hch + 1) * pixelScale;
                    ctx.save();
                    ctx.strokeStyle = '#e91e63';
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(cx - w / 2, cy - w / 2, w, w);
                    ctx.restore();
                }
            }
            // Per-joint bbox for selected joint (dashed).
            if (stereoSelectedJoint != null && mpHere && mpHere[fn]) {
                const pt = mpHere[fn][stereoSelectedJoint];
                if (pt) {
                    const perJoint = trialData.stereo_crop_halves_per_joint;
                    const ch = (Array.isArray(perJoint) && perJoint[stereoSelectedJoint] != null)
                        ? perJoint[stereoSelectedJoint]
                        : (trialData.stereo_crop_half != null ? trialData.stereo_crop_half : 40);
                    const cx = pt[0] * pixelScale;
                    const cy = pt[1] * pixelScale;
                    const w = (2 * ch + 1) * pixelScale;
                    ctx.save();
                    ctx.strokeStyle = '#e91e63';
                    ctx.lineWidth = 1.2;
                    ctx.setLineDash([4, 3]);
                    ctx.strokeRect(cx - w / 2, cy - w / 2, w, w);
                    ctx.setLineDash([]);
                    ctx.restore();
                }
            }
            // Pink crosses, sized by per-joint confidence; conf below
            // the slider threshold are hidden entirely.
            if (stereoKp && stereoKp[fn]) {
                const respFrame = trialData.stereo_response?.[fn];
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !stereoKp[fn][j]) continue;
                    const rawConf = (respFrame && respFrame[j] != null) ? respFrame[j] : 0;
                    if (rawConf < stereoConfThreshold) continue;
                    const x = stereoKp[fn][j][0] * pixelScale;
                    const y = stereoKp[fn][j][1] * pixelScale;
                    let conf = rawConf;
                    if (conf < 0) conf = 0;
                    if (conf > 1) conf = 1;
                    const size = 4 * (0.3 + 0.7 * conf);
                    drawCross(x, y, '#e91e63', size);
                }
            }
        }

        // Stereo (outline) -- same display as Stereo but driven by
        // the outline-based alignment + light-purple crosses
        // (#ce93d8).  No hand-wide bbox -- the whole-hand alignment
        // uses the ENTIRE outline, not a cropped patch.  Per-joint
        // dashed bbox on click, same as Stereo.
        if (showStereoOutline2D && trialData?.has_stereo_outline) {
            const stereoKp = isLeft ? trialData.stereo_outline_tracked_L
                                    : trialData.stereo_outline_tracked_R;
            const mpHere = isLeft ? trialData.mp_tracked_L
                                  : trialData.mp_tracked_R;
            const mpOther = isLeft ? trialData.mp_tracked_R
                                   : trialData.mp_tracked_L;
            // Per-joint bbox for selected joint (dashed) + the outlines
            // INSIDE it that were consumed by the per-joint alignment
            // (yellow = current camera, green = opposite camera shifted
            // so its MP joint sits on the current joint -- the same
            // MP-centering the phase-corr aligns on).
            if (stereoSelectedJoint != null && mpHere && mpHere[fn]) {
                const pt = mpHere[fn][stereoSelectedJoint];
                if (pt) {
                    const perJoint = trialData.stereo_crop_halves_per_joint;
                    const ch = (Array.isArray(perJoint) && perJoint[stereoSelectedJoint] != null)
                        ? perJoint[stereoSelectedJoint]
                        : (trialData.stereo_crop_half != null ? trialData.stereo_crop_half : 40);
                    const cx = pt[0] * pixelScale;
                    const cy = pt[1] * pixelScale;
                    const w = (2 * ch + 1) * pixelScale;
                    const bx = cx - w / 2, by = cy - w / 2;
                    // Outlines clipped to the bbox.
                    if (trialData.has_outlines) {
                        const curOut = isLeft ? trialData.outlines_L?.[fn]
                                              : trialData.outlines_R?.[fn];
                        const oppOut = isLeft ? trialData.outlines_R?.[fn]
                                              : trialData.outlines_L?.[fn];
                        const mpOpp = (mpOther && mpOther[fn])
                            ? mpOther[fn][stereoSelectedJoint] : null;
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(bx, by, w, w);
                        ctx.clip();
                        // Yellow current outline at native position.
                        if (curOut && curOut.length >= 3) {
                            ctx.strokeStyle = '#ffd54f';
                            ctx.lineWidth = 0.6;
                            ctx.lineJoin = 'round';
                            ctx.lineCap = 'round';
                            ctx.beginPath();
                            ctx.moveTo(curOut[0][0] * pixelScale, curOut[0][1] * pixelScale);
                            for (let i = 1; i < curOut.length; i++) {
                                ctx.lineTo(curOut[i][0] * pixelScale, curOut[i][1] * pixelScale);
                            }
                            ctx.closePath();
                            ctx.stroke();
                        }
                        // Green opposite outline translated by
                        // (mp_current - mp_opposite) so joint centres
                        // line up.
                        if (oppOut && oppOut.length >= 3 && mpOpp) {
                            const tx = (pt[0] - mpOpp[0]) * pixelScale;
                            const ty = (pt[1] - mpOpp[1]) * pixelScale;
                            ctx.strokeStyle = '#66bb6a';
                            ctx.lineWidth = 0.6;
                            ctx.lineJoin = 'round';
                            ctx.lineCap = 'round';
                            ctx.beginPath();
                            ctx.moveTo(oppOut[0][0] * pixelScale + tx,
                                       oppOut[0][1] * pixelScale + ty);
                            for (let i = 1; i < oppOut.length; i++) {
                                ctx.lineTo(oppOut[i][0] * pixelScale + tx,
                                           oppOut[i][1] * pixelScale + ty);
                            }
                            ctx.closePath();
                            ctx.stroke();
                        }
                        ctx.restore();
                    }
                    // Dashed bbox over the clipped outlines.
                    ctx.save();
                    ctx.strokeStyle = '#ce93d8';
                    ctx.lineWidth = 1.2;
                    ctx.setLineDash([4, 3]);
                    ctx.strokeRect(bx, by, w, w);
                    ctx.setLineDash([]);
                    ctx.restore();
                }
            }
            if (stereoKp && stereoKp[fn]) {
                const respFrame = trialData.stereo_outline_response?.[fn];
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !stereoKp[fn][j]) continue;
                    const rawConf = (respFrame && respFrame[j] != null) ? respFrame[j] : 0;
                    if (rawConf < stereoOutlineConfThreshold) continue;
                    const x = stereoKp[fn][j][0] * pixelScale;
                    const y = stereoKp[fn][j][1] * pixelScale;
                    let conf = rawConf;
                    if (conf < 0) conf = 0;
                    if (conf > 1) conf = 1;
                    const size = 4 * (0.3 + 0.7 * conf);
                    drawCross(x, y, '#ce93d8', size);
                }
            }
        }

        // Stereo (hybrid) -- outline-vote Pass 1 + image-phase-corr
        // Pass 2.  Light-pink crosses + dashed per-joint bbox (no
        // outlines drawn inside, since the per-joint alignment uses
        // the raw image, not the outline).
        if (showStereoHybrid2D && trialData?.has_stereo_hybrid) {
            const stereoKp = isLeft ? trialData.stereo_hybrid_tracked_L
                                    : trialData.stereo_hybrid_tracked_R;
            const mpHere = isLeft ? trialData.mp_tracked_L
                                  : trialData.mp_tracked_R;
            if (stereoSelectedJoint != null && mpHere && mpHere[fn]) {
                const pt = mpHere[fn][stereoSelectedJoint];
                if (pt) {
                    const perJoint = trialData.stereo_crop_halves_per_joint;
                    const ch = (Array.isArray(perJoint) && perJoint[stereoSelectedJoint] != null)
                        ? perJoint[stereoSelectedJoint]
                        : (trialData.stereo_crop_half != null ? trialData.stereo_crop_half : 40);
                    const cx = pt[0] * pixelScale;
                    const cy = pt[1] * pixelScale;
                    const w = (2 * ch + 1) * pixelScale;
                    ctx.save();
                    ctx.strokeStyle = '#f48fb1';
                    ctx.lineWidth = 1.2;
                    ctx.setLineDash([4, 3]);
                    ctx.strokeRect(cx - w / 2, cy - w / 2, w, w);
                    ctx.setLineDash([]);
                    ctx.restore();
                }
            }
            if (stereoKp && stereoKp[fn]) {
                const respFrame = trialData.stereo_hybrid_response?.[fn];
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !stereoKp[fn][j]) continue;
                    const rawConf = (respFrame && respFrame[j] != null) ? respFrame[j] : 0;
                    if (rawConf < stereoHybridConfThreshold) continue;
                    const x = stereoKp[fn][j][0] * pixelScale;
                    const y = stereoKp[fn][j][1] * pixelScale;
                    let conf = rawConf;
                    if (conf < 0) conf = 0;
                    if (conf > 1) conf = 1;
                    const size = 4 * (0.3 + 0.7 * conf);
                    drawCross(x, y, '#f48fb1', size);
                }
            }
        }

        // Outline -- current camera's hand boundary polygon for the
        // current frame, from the preproc bake (inverse-warped server-
        // side into this camera's original-frame coords).  Drawn when
        // the Outline model is on OR when the Stereo panel is open
        // with Hybrid selected (so the user can see what's being
        // masked).
        const _showOutlineForHybridStaging = (() => {
            if ($('stereoPanel')?.style.display !== 'block') return false;
            const m = document.querySelector('input[name="stereoMode"]:checked');
            return !!(m && m.value === 'hybrid');
        })();
        if ((showOutline2D || _showOutlineForHybridStaging) && trialData?.has_outlines) {
            let poly = isLeft ? trialData.outlines_L?.[fn]
                              : trialData.outlines_R?.[fn];
            // Hybrid staging: draw the DILATED outline (mirrors the
            // backend mask the bake will use).
            if (_showOutlineForHybridStaging && poly && poly.length >= 3) {
                const dPx = parseInt($('stereoDilateSlider')?.value ?? '0', 10) || 0;
                if (dPx > 0) poly = _dilatePolygon(poly, dPx);
            }
            if (poly && poly.length >= 3) {
                ctx.save();
                ctx.strokeStyle = '#ffd54f';
                ctx.lineWidth = 1.5;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(poly[0][0] * pixelScale, poly[0][1] * pixelScale);
                for (let i = 1; i < poly.length; i++) {
                    ctx.lineTo(poly[i][0] * pixelScale, poly[i][1] * pixelScale);
                }
                ctx.closePath();
                ctx.stroke();
                ctx.restore();
            }
        }

        // Vision joints (blue)
        if (showVision2D && visionKp?.[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !visionKp[fn][j]) continue;
                const x = (visionKp[fn][j][0] + visionXOff) * pixelScale;
                const y = visionKp[fn][j][1] * pixelScale;
                drawJoint(x, y, '#2196f3', 3.5);
            }
        }

        // Previous fit joints (dark orange)
        if (showPrev2D && prevProj?.[fn]) {
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !prevProj[fn][j]) continue;
                drawJoint(prevProj[fn][j][0] * pixelScale, prevProj[fn][j][1] * pixelScale, '#b35b00', 4);
            }
        }

        // DLC thumb/index
        if (showDLC) {
            const thumbKey = isLeft ? 'dlc_thumb_OS' : 'dlc_thumb_OD';
            const indexKey = isLeft ? 'dlc_index_OS' : 'dlc_index_OD';
            if (isJointVisible(4) && trialData[thumbKey][fn]) {
                const pt = trialData[thumbKey][fn];
                drawJoint((pt[0] + mpXOff) * pixelScale, pt[1] * pixelScale, '#ff4444', 5);
            }
            if (isJointVisible(8) && trialData[indexKey][fn]) {
                const pt = trialData[indexKey][fn];
                drawJoint((pt[0] + mpXOff) * pixelScale, pt[1] * pixelScale, '#222', 5);
            }
        }

        // Pose body landmarks (arm chain, yellow — matching body side only)
        if (showPose2D) {
            const poseKp = isLeft ? trialData.pose_tracked_L : trialData.pose_tracked_R;
            if (poseKp?.[fn] && trialData.pose_side) {
                // Indices in sliced array (POSE_ARM_INDICES = [11..22]):
                // Left body: shoulder=0, elbow=2, wrist=4, pinky=6, index=8, thumb=10
                // Right body: shoulder=1, elbow=3, wrist=5, pinky=7, index=9, thumb=11
                const sIdx = trialData.pose_side === 'left' ? [0,2,4,6,8,10] : [1,3,5,7,9,11];
                // Bones: shoulder→elbow→wrist→pinky/index/thumb
                const POSE_BONE_PAIRS = [[0,1],[1,2],[2,3],[2,4],[2,5]];
                for (const [a, b] of POSE_BONE_PAIRS) {
                    const pa = poseKp[fn][sIdx[a]], pb = poseKp[fn][sIdx[b]];
                    if (pa && pb) {
                        drawLine(
                            pa[0] * pixelScale, pa[1] * pixelScale,
                            pb[0] * pixelScale, pb[1] * pixelScale,
                            '#ffeb3b', 2, 0.5
                        );
                    }
                }
                for (const si of sIdx) {
                    if (!poseKp[fn][si]) continue;
                    drawJoint(poseKp[fn][si][0] * pixelScale, poseKp[fn][si][1] * pixelScale, '#ffeb3b', 3);
                }
            }
        }

        // Heatmap-peaks model (orange) — 2D peaks + skeleton bones.
        // Skel lines only render when 2D is also on; otherwise the
        // skeleton lives in the 3D view only (or not at all).
        if (hmPeaks && showHeatmap2D) {
            const HM_COLOR = '#ff6600';
            if (showHeatmapSkel && trialData.skeleton) {
                for (const [i, j] of trialData.skeleton) {
                    const a = hmPeaks[i], b = hmPeaks[j];
                    if (!a || !b) continue;
                    if (!isBoneVisible(i, j)) continue;
                    drawLine(a[0] * pixelScale, a[1] * pixelScale,
                             b[0] * pixelScale, b[1] * pixelScale,
                             HM_COLOR, 2, 0.7);
                }
            }
            if (showHeatmap2D) {
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !hmPeaks[j]) continue;
                    drawJoint(hmPeaks[j][0] * pixelScale, hmPeaks[j][1] * pixelScale, HM_COLOR, 3);
                }
            }
        }

        // HRnet "Peaks" sub-stage (cluster centroid, lighter orange).
        if (hmPeaksRaw && showHRnet2D) {
            const HM_RAW_COLOR = '#ff9966';
            if (showHeatmapSkel && trialData.skeleton) {
                for (const [i, j] of trialData.skeleton) {
                    const a = hmPeaksRaw[i], b = hmPeaksRaw[j];
                    if (!a || !b) continue;
                    if (!isBoneVisible(i, j)) continue;
                    drawLine(a[0] * pixelScale, a[1] * pixelScale,
                             b[0] * pixelScale, b[1] * pixelScale,
                             HM_RAW_COLOR, 2, 0.6);
                }
            }
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !hmPeaksRaw[j]) continue;
                drawJoint(hmPeaksRaw[j][0] * pixelScale, hmPeaksRaw[j][1] * pixelScale, HM_RAW_COLOR, 3);
            }
        }

        // ── Corrections overlay on the Peaks sub-stage ──
        // Red ring on EITHER camera when any error (y_disp or z_outlier)
        // is flagged on any camera; the centroid joint dot also turns
        // red on the BLAMED camera's side (this matches the MP error
        // visualisation pattern).  When the Cluster-AUC attribution
        // slider > 0, also draws an empty black circle at the predicted
        // corrected position (per camera) so you can see the magnitude
        // of the AUC delta visually.
        if (showHRnet2D && showHRnetCorrections && hrnetPreview && hmPeaksRaw) {
            const aucSliderOn = (parseFloat($('hfSliderYZ_aC')?.value ?? 0) > 0);
            const camIdx = isLeft ? 0 : 1;
            const otherIdx = isLeft ? 1 : 0;
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !hmPeaksRaw[j]) continue;
                const yThis = hrnetPreview.hasYZ(fn, j, camIdx);
                const zThis = hrnetPreview.hasZO(fn, j, camIdx);
                const yOther = hrnetPreview.hasYZ(fn, j, otherIdx);
                const zOther = hrnetPreview.hasZO(fn, j, otherIdx);
                const anyErr = yThis || zThis || yOther || zOther;
                if (!anyErr) continue;
                const x = hmPeaksRaw[j][0] * pixelScale;
                const y = hmPeaksRaw[j][1] * pixelScale;
                if (yThis || zThis) {
                    drawJoint(x, y, '#ff2222', 3);
                }
                ctx.save();
                ctx.strokeStyle = '#ff2222';
                ctx.lineWidth = 1.5 / scale;
                ctx.beginPath();
                ctx.arc(x, y, 7, 0, 2 * Math.PI);
                ctx.stroke();
                ctx.restore();
                if (aucSliderOn) {
                    const p = hrnetPreview.pred(fn, j, camIdx);
                    if (p) {
                        ctx.save();
                        ctx.strokeStyle = '#000';
                        ctx.lineWidth = 1.5 / scale;
                        ctx.beginPath();
                        ctx.arc(p[0] * pixelScale, p[1] * pixelScale, 6, 0, 2 * Math.PI);
                        ctx.stroke();
                        ctx.restore();
                    }
                }
            }
        }

        // HRnet "Y/Z-correct" sub-stage (warmer orange).
        if (hmPeaksYZC && showHRnetYZC2D) {
            const HM_YZC_COLOR = '#ffb074';
            if (showHeatmapSkel && trialData.skeleton) {
                for (const [i, j] of trialData.skeleton) {
                    const a = hmPeaksYZC[i], b = hmPeaksYZC[j];
                    if (!a || !b) continue;
                    if (!isBoneVisible(i, j)) continue;
                    drawLine(a[0] * pixelScale, a[1] * pixelScale,
                             b[0] * pixelScale, b[1] * pixelScale,
                             HM_YZC_COLOR, 2, 0.6);
                }
            }
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !hmPeaksYZC[j]) continue;
                drawJoint(hmPeaksYZC[j][0] * pixelScale, hmPeaksYZC[j][1] * pixelScale, HM_YZC_COLOR, 3);
            }
        }

        // HRnet "Z-smooth" sub-stage (saturated orange).
        if (hmPeaksZSM && showHRnetZSM2D) {
            const HM_ZSM_COLOR = '#ffa040';
            if (showHeatmapSkel && trialData.skeleton) {
                for (const [i, j] of trialData.skeleton) {
                    const a = hmPeaksZSM[i], b = hmPeaksZSM[j];
                    if (!a || !b) continue;
                    if (!isBoneVisible(i, j)) continue;
                    drawLine(a[0] * pixelScale, a[1] * pixelScale,
                             b[0] * pixelScale, b[1] * pixelScale,
                             HM_ZSM_COLOR, 2, 0.65);
                }
            }
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !hmPeaksZSM[j]) continue;
                drawJoint(hmPeaksZSM[j][0] * pixelScale, hmPeaksZSM[j][1] * pixelScale, HM_ZSM_COLOR, 3);
            }
        }

        // HRnet "Stereo-Hungarian" sub-stage (coral, post HRnet-Fit).
        if (hmPeaksHun && showStereoHun2D) {
            const HM_HUN_COLOR = '#ff8a65';
            if (showHeatmapSkel && trialData.skeleton) {
                for (const [i, j] of trialData.skeleton) {
                    const a = hmPeaksHun[i], b = hmPeaksHun[j];
                    if (!a || !b) continue;
                    if (!isBoneVisible(i, j)) continue;
                    drawLine(a[0] * pixelScale, a[1] * pixelScale,
                             b[0] * pixelScale, b[1] * pixelScale,
                             HM_HUN_COLOR, 2, 0.7);
                }
            }
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !hmPeaksHun[j]) continue;
                drawJoint(hmPeaksHun[j][0] * pixelScale, hmPeaksHun[j][1] * pixelScale, HM_HUN_COLOR, 3);
            }
        }

        // Anatomical-offset arrows in 2D.  Drawn on every displayed
        // Skeleton-v3 stage when the "Offsets" toggle is on.  Each arrow
        // points from the joint's stage position toward where the HRnet
        // peak is expected (joint + per-joint median offset).
        if (showOffsetArrows) {
            if (!window._offsetArrowDbg2) {
                window._offsetArrowDbg2 = true;
                console.log('[offset-arrows-2d entry]', {
                    isLeft, stages2D: [..._stages2D], stages3D: [..._stages3D],
                    showSkelV2_2D, showMP2D, showLegacyV2_2D, showMano2D,
                    hasPeaks: !!trialData?.hrnet_peaks,
                });
            }
        }
        if (showOffsetArrows && _stages2D.size > 0) {
            // Use the same 3D offset model the 3D arrows use, then project
            // (joint_3d + offset_3d) into the current camera.  This makes
            // the 2D arrow exactly the camera-projection of the 3D arrow,
            // so they line up across views (and within a stage between
            // its 2D and 3D model).  Falls back to the legacy 2D-only
            // bone-direction model only when 3D offset data isn't loaded.
            let off3d = {
                along: trialData.hrnet_along_3d, flex: trialData.hrnet_flex_3d,
                abd:   trialData.hrnet_abd_3d,   child: trialData.hrnet_child_3d,
            };
            const _allZero = a => a && Array.from(a).every(v => Math.abs(v) < 1e-6);
            const _allMinus = a => a && Array.from(a).every(v => v < 0);
            const degenerate3d = !off3d.along || !off3d.flex || !off3d.abd
                              || _allZero(off3d.along) || _allZero(off3d.flex)
                              || (off3d.child && _allMinus(off3d.child));
            if (degenerate3d) off3d = _computeHRnetOffsets3DClient();

            const STAGE_PROJ = {
                mediapipe:    () => isLeft ? trialData.mp_tracked_L : trialData.mp_tracked_R,
                z_correct:    () => (isLeft ? trialData.skel_v2_proj_z_L  : trialData.skel_v2_proj_z_R)
                                || (isLeft ? trialData.skel_v2_proj_L    : trialData.skel_v2_proj_R),
                z_smooth:     () => (isLeft ? trialData.skel_v2_proj_zs_L : trialData.skel_v2_proj_zs_R)
                                || (isLeft ? trialData.skel_v2_proj_L    : trialData.skel_v2_proj_R),
                hrnet_snap:   () => (isLeft ? trialData.skel_v2_proj_hr_L : trialData.skel_v2_proj_hr_R)
                                || (isLeft ? trialData.skel_v2_proj_zs_L : trialData.skel_v2_proj_zs_R)
                                || (isLeft ? trialData.skel_v2_proj_L    : trialData.skel_v2_proj_R),
                bone_correct: () => (isLeft ? trialData.skel_v2_proj_bc_L : trialData.skel_v2_proj_bc_R)
                                || (isLeft ? trialData.skel_v2_proj_L    : trialData.skel_v2_proj_R),
                bone_smooth:  () => isLeft ? trialData.skel_v2_proj_L    : trialData.skel_v2_proj_R,
            };
            const STAGE_3D = {
                mediapipe:    trialData.mp_joints_3d,
                z_correct:    trialData.skel_v2_joints_z_3d  || trialData.skel_v2_joints_3d,
                z_smooth:     trialData.skel_v2_joints_zs_3d || trialData.skel_v2_joints_3d,
                hrnet_snap:   trialData.skel_v2_joints_hr_3d || trialData.skel_v2_joints_zs_3d
                                || trialData.skel_v2_joints_3d,
                bone_correct: trialData.skel_v2_joints_bc_3d || trialData.skel_v2_joints_3d,
                bone_smooth:  trialData.skel_v2_joints_3d,
            };

            if (off3d && off3d.along) {
                for (const stage of _stages2D) {
                    const proj = STAGE_PROJ[stage]?.();
                    const pts2dFrame = proj?.[fn];
                    const pts3dFrame = STAGE_3D[stage]?.[fn];
                    if (!pts3dFrame) continue;
                    _drawHRnet3DArrows2D(pts2dFrame, pts3dFrame, off3d, isLeft, pixelScale);
                }
            }
        }
    }


    // Client-side 3D systematic-offset computation matching the backend's
    // `_hrnet_snap_compute_offsets_3d`.  Uses z_smooth 3D labels + HRnet
    // peaks_3d, builds the joint-angle-style basis (b_in + flex + abd),
    // and medians per joint.  Cached on trialData.
    const _HR3D_CHILD = { 1:2, 2:3, 3:4, 5:6, 6:7, 7:8, 9:10, 10:11, 11:12,
                          13:14, 14:15, 15:16, 17:18, 18:19, 19:20 };
    const _HR3D_TIPS = { 4:3, 8:7, 12:11, 16:15, 20:19 };
    const _HR3D_PARENT = { 1:0,2:1,3:2,4:3, 5:0,6:5,7:6,8:7, 9:0,10:9,11:10,12:11,
                           13:0,14:13,15:14,16:15, 17:0,18:17,19:18,20:19 };

    function _jointBasis3D(joints_f, j) {
        const parent = _HR3D_PARENT[j];
        if (parent == null) return null;
        const p = joints_f[parent], q = joints_f[j];
        if (!p || !q) return null;
        const bin = [q[0]-p[0], q[1]-p[1], q[2]-p[2]];
        const L = Math.hypot(bin[0], bin[1], bin[2]);
        if (L < 1e-6) return null;
        const e_along = [bin[0]/L, bin[1]/L, bin[2]/L];
        // Palm normal from wrist + index MCP + ring MCP.
        const w = joints_f[0], m5 = joints_f[5], m13 = joints_f[13];
        if (!w || !m5 || !m13) return null;
        const v1 = [m5[0]-w[0], m5[1]-w[1], m5[2]-w[2]];
        const v2 = [m13[0]-w[0], m13[1]-w[1], m13[2]-w[2]];
        let palm = [v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]];
        const pL = Math.hypot(palm[0], palm[1], palm[2]);
        if (pL < 1e-6) return null;
        palm = [palm[0]/pL, palm[1]/pL, palm[2]/pL];
        // Gram-Schmidt: e_flex = palm ⊥ e_along
        const dot_pa = palm[0]*e_along[0] + palm[1]*e_along[1] + palm[2]*e_along[2];
        let e_flex = [palm[0]-dot_pa*e_along[0], palm[1]-dot_pa*e_along[1], palm[2]-dot_pa*e_along[2]];
        const fl = Math.hypot(e_flex[0], e_flex[1], e_flex[2]);
        if (fl < 1e-6) return null;
        e_flex = [e_flex[0]/fl, e_flex[1]/fl, e_flex[2]/fl];
        // e_abd = e_flex × e_along
        const e_abd = [
            e_flex[1]*e_along[2] - e_flex[2]*e_along[1],
            e_flex[2]*e_along[0] - e_flex[0]*e_along[2],
            e_flex[0]*e_along[1] - e_flex[1]*e_along[0],
        ];
        return { along: e_along, flex: e_flex, abd: e_abd };
    }

    function _computeHRnetOffsets3DClient() {
        if (!trialData) return null;
        if (trialData._hrClient3D) return trialData._hrClient3D;
        const mp3d = trialData.skel_v2_joints_zs_3d || trialData.skel_v2_joints_3d
                   || trialData.mp_joints_3d;
        const hr3d = trialData.hrnet_peaks_3d;
        if (!mp3d || !hr3d) return null;
        const N = trialData.n_frames || 0;
        const along = new Float32Array(21);
        const flex  = new Float32Array(21);
        const abd   = new Float32Array(21);
        const child = new Int32Array(21).fill(-1);
        for (let j = 0; j < 21; j++) {
            if (_HR3D_CHILD[j] != null) child[j] = _HR3D_CHILD[j];
            else if (_HR3D_TIPS[j] != null) child[j] = j;
            else continue;
            const av = [], fv = [], bv = [];
            for (let f = 0; f < N; f++) {
                const mp_f = mp3d[f]; const hr_f = hr3d[f];
                if (!mp_f || !hr_f) continue;
                const mp = mp_f[j]; const hr = hr_f[j];
                if (!mp || !hr) continue;
                const basis = _jointBasis3D(mp_f, j);
                if (!basis) continue;
                const d = [hr[0]-mp[0], hr[1]-mp[1], hr[2]-mp[2]];
                av.push(d[0]*basis.along[0] + d[1]*basis.along[1] + d[2]*basis.along[2]);
                fv.push(d[0]*basis.flex[0]  + d[1]*basis.flex[1]  + d[2]*basis.flex[2]);
                bv.push(d[0]*basis.abd[0]   + d[1]*basis.abd[1]   + d[2]*basis.abd[2]);
            }
            if (av.length >= 5) {
                av.sort((a,b)=>a-b); fv.sort((a,b)=>a-b); bv.sort((a,b)=>a-b);
                along[j] = av[Math.floor(av.length/2)];
                flex[j]  = fv[Math.floor(fv.length/2)];
                abd[j]   = bv[Math.floor(bv.length/2)];
            }
        }
        const out = { along: Array.from(along), flex: Array.from(flex),
                      abd: Array.from(abd), child: Array.from(child) };
        trialData._hrClient3D = out;
        return out;
    }

    // 2D arrows pointing from each HRnet-snap joint to the expected HRnet
    // peak location.  Bone direction is computed from the current HRnet-snap
    // geometry (not the raw MP).  For fingertips, the "parent → tip" bone
    // direction is used (extrapolated past the tip).
    // Pinhole projection of a 3D world point into the current camera.
    // Distortion is ignored (it's a sub-pixel correction over the short
    // arrow span, not visible to the eye).
    function _project3D(X, isLeftCam) {
        const cal = trialData?.calib;
        if (!cal || !X) return null;
        const K = isLeftCam ? cal.K_L : cal.K_R;
        if (!K) return null;
        let cx, cy, cz;
        if (isLeftCam) {
            cx = X[0]; cy = X[1]; cz = X[2];
        } else {
            const R = cal.R, T = cal.T;
            if (!R || !T) return null;
            cx = R[0][0]*X[0] + R[0][1]*X[1] + R[0][2]*X[2] + T[0];
            cy = R[1][0]*X[0] + R[1][1]*X[1] + R[1][2]*X[2] + T[1];
            cz = R[2][0]*X[0] + R[2][1]*X[1] + R[2][2]*X[2] + T[2];
        }
        if (!(cz > 0)) return null;
        return [K[0][0] * cx / cz + K[0][2],
                K[1][1] * cy / cz + K[1][2]];
    }

    // 2D arrows derived from the 3D offset model.  Each arrow is the 2D
    // delta from projecting (tail_3d → head_3d) attached to the saved
    // empirical 2D label.  Anchoring on the empirical pixel (rather than
    // the calibration-projected pixel) cancels the small calibration
    // Ty bias that otherwise pulls arrow heads up in OD and down in OS.
    // Skips joints hidden via finger-label toggle.
    function _drawHRnet3DArrows2D(pts2d, pts3d, off3d, isLeftCam, pixelScale) {
        if (!off3d?.along || !pts3d) return;
        for (let j = 0; j < 21; j++) {
            if (!isJointVisible(j)) continue;
            if ((off3d.child?.[j] ?? -1) < 0) continue;
            const tail3d = pts3d[j];
            if (!tail3d) continue;
            const basis = _jointBasis3D(pts3d, j);
            if (!basis) continue;
            const a = off3d.along[j] || 0, fl = off3d.flex[j] || 0, b = off3d.abd[j] || 0;
            const head3d = [
                tail3d[0] + a*basis.along[0] + fl*basis.flex[0] + b*basis.abd[0],
                tail3d[1] + a*basis.along[1] + fl*basis.flex[1] + b*basis.abd[1],
                tail3d[2] + a*basis.along[2] + fl*basis.flex[2] + b*basis.abd[2],
            ];
            const tail2dEmp = pts2d && pts2d[j];
            const tail2dCal = _project3D(tail3d, isLeftCam);
            const head2dCal = _project3D(head3d, isLeftCam);
            if (!tail2dCal || !head2dCal) continue;
            // Calibration-derived arrow VECTOR (cancels the Ty bias).
            const dx = head2dCal[0] - tail2dCal[0];
            const dy = head2dCal[1] - tail2dCal[1];
            // Anchor on the saved empirical pixel (matches the visible
            // 2D label).  Fall back to the calibration projection when
            // the saved pixel is missing.
            const anchor = (tail2dEmp && Number.isFinite(tail2dEmp[0])) ? tail2dEmp : tail2dCal;
            const rootX = anchor[0] * pixelScale, rootY = anchor[1] * pixelScale;
            const tipX  = (anchor[0] + dx) * pixelScale;
            const tipY  = (anchor[1] + dy) * pixelScale;
            _drawHRnetArrowSegment(rootX, rootY, tipX, tipY);
        }
    }

    function _drawHRnetArrowSegment(rootX, rootY, tipX, tipY) {
        drawLine(rootX, rootY, tipX, tipY, '#ff6600', 1.5, 0.9);
        const dx = tipX - rootX, dy = tipY - rootY;
        const len = Math.hypot(dx, dy);
        if (len < 1) return;
        const ux = dx / len, uy = dy / len;
        const HEAD = 4 / scale;
        const pxh = -uy * HEAD * 0.6, pyh = ux * HEAD * 0.6;
        const baseX = tipX - ux * HEAD, baseY = tipY - uy * HEAD;
        ctx.save();
        ctx.fillStyle = '#ff6600';
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(baseX + pxh, baseY + pyh);
        ctx.lineTo(baseX - pxh, baseY - pyh);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }


    // Global bump factor for 2D overlay weight (joint radius, cross size,
    // line thickness).  Makes labels easier to see on high-res video.
    const LABEL_SCALE = 1.6;

    function drawLine(x1, y1, x2, y2, color, width, alpha) {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = (width * LABEL_SCALE) / scale;
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
        ctx.lineWidth = 1 / scale;
        ctx.beginPath();
        ctx.arc(x, y, (radius * LABEL_SCALE) / scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }

    function drawCross(x, y, color, size) {
        const s = (size * LABEL_SCALE) / scale;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = (2.5 * LABEL_SCALE) / scale;
        ctx.beginPath();
        ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
        ctx.stroke();
        ctx.restore();
    }

    // ── Heatmap overlay ──────────────────────────────────────
    function hotColormap(val) {
        // black → red → orange → yellow → white.  Most of the range lives
        // in red/orange so the saturated (bright/white) end only kicks in
        // at the very top of the heat map.
        if (val <= 0) return [0, 0, 0];
        const v = Math.sqrt(val);
        if (v <= 0.5) {
            const t = v / 0.5;
            return [Math.round(255 * t), 0, 0];
        }
        if (v <= 0.85) {
            const t = (v - 0.5) / 0.35;
            return [255, Math.round(180 * t), 0];
        }
        if (v <= 0.95) {
            const t = (v - 0.85) / 0.1;
            return [255, 180 + Math.round(75 * t), Math.round(50 * t)];
        }
        const t = (v - 0.95) / 0.05;
        return [255, 255, 50 + Math.round(205 * t)];
    }

    /** Pre-fetch heatmap data for the current frame and build a MIP canvas.
     *  Call this before render() — the result is drawn synchronously. */
    async function _prefetchHeatmap() {
        _heatmapImageData = null;
        if (!showHeatmap || !trialData) return;
        const trial = trials[currentTrialIdx];
        if (!trial || !trial.has_heatmaps) return;

        const frame = currentFrame;
        const side = currentSide;
        const displayJoints = heatmapMipMode
            ? [...Array(21).keys()]
            : heatmapActiveJoints.size > 0
                ? [...heatmapActiveJoints]
                : [];
        if (!displayJoints.length) return;

        // Fast-path for MIP mode: fetch one precomputed MIP slice instead
        // of 21 per-joint heatmaps.  Skips softmax (which wants per-joint).
        if (heatmapMipMode) {
            const key = `${frame}_MIP_${side}`;
            if (!heatmapCache[key]) {
                try {
                    heatmapCache[key] = await api(
                        `/api/skeleton/${subjectId}/trial/${trial.trial_idx}/heatmap?frame=${frame}&joint=-1&side=${side}`
                    );
                } catch { return; }
            }
            const resp = heatmapCache[key];
            if (!resp?.heatmap) return;
            const hmH = resp.heatmap.length, hmW = resp.heatmap[0].length;
            const [bx1, by1, bx2, by2] = resp.bbox;
            const nPix = hmH * hmW;
            const mip = new Float32Array(nPix);
            for (let r = 0; r < hmH; r++) {
                for (let c = 0; c < hmW; c++) mip[r * hmW + c] = resp.heatmap[r][c];
            }
            const offCanvas = document.createElement('canvas');
            offCanvas.width = hmW; offCanvas.height = hmH;
            const offCtx = offCanvas.getContext('2d');
            const imgData = offCtx.createImageData(hmW, hmH);
            for (let i = 0; i < nPix; i++) {
                const val = mip[i];
                const [r, g, b] = hotColormap(val);
                const idx = i * 4;
                imgData.data[idx] = r; imgData.data[idx + 1] = g; imgData.data[idx + 2] = b;
                imgData.data[idx + 3] = val > heatmapThreshold ? 140 : 0;
            }
            offCtx.putImageData(imgData, 0, 0);
            // Peaks for MIP mode come from the prefetched trialData.hrnet_peaks
            const assignedPeaks = {};
            const peakData = trialData?.hrnet_peaks;
            if (peakData) {
                const isLeft = currentSide === cameraNames[0];
                const camPeaks = isLeft ? peakData.peaks_L : peakData.peaks_R;
                const jnames = trialData?.joint_names || [];
                if (camPeaks) {
                    for (let j = 0; j < 21; j++) {
                        const frames = camPeaks[jnames[j]];
                        if (frames && frames[currentFrame]) {
                            assignedPeaks[j] = { x: frames[currentFrame][0], y: frames[currentFrame][1] };
                        }
                    }
                }
            }
            _heatmapImageData = { canvas: offCanvas, bbox: [bx1, by1, bx2, by2], peaks: assignedPeaks };
            return;
        }

        const fetchJoints = displayJoints;

        // Fetch all needed joints (cached)
        const allMaps = {};  // j → heatmap data
        for (const j of fetchJoints) {
            const key = `${frame}_${j}_${side}`;
            if (!heatmapCache[key]) {
                try {
                    heatmapCache[key] = await api(
                        `/api/skeleton/${subjectId}/trial/${trial.trial_idx}/heatmap?frame=${frame}&joint=${j}&side=${side}`
                    );
                } catch { continue; }
            }
            if (heatmapCache[key]?.heatmap) allMaps[j] = heatmapCache[key];
        }
        const maps = displayJoints.filter(j => allMaps[j]).map(j => allMaps[j]);
        if (!maps.length) return;

        const first = maps[0];
        const hmH = first.heatmap.length, hmW = first.heatmap[0].length;
        const [bx1, by1, bx2, by2] = first.bbox;
        const nPix = hmH * hmW;

        // Standard MIP — max over the selected joints' heatmap values.
        const mip = new Float32Array(nPix);
        for (const m of maps) {
            const hm = m.heatmap;
            for (let r = 0; r < hmH; r++) {
                for (let c = 0; c < hmW; c++) {
                    const idx = r * hmW + c;
                    if (hm[r][c] > mip[idx]) mip[idx] = hm[r][c];
                }
            }
        }

        // Render to offscreen canvas
        const offCanvas = document.createElement('canvas');
        offCanvas.width = hmW; offCanvas.height = hmH;
        const offCtx = offCanvas.getContext('2d');
        const imgData = offCtx.createImageData(hmW, hmH);

        for (let i = 0; i < mip.length; i++) {
            const val = mip[i];
            const [cr, cg, cb] = hotColormap(val);
            const idx = i * 4;
            imgData.data[idx] = cr;
            imgData.data[idx + 1] = cg;
            imgData.data[idx + 2] = cb;
            imgData.data[idx + 3] = val > heatmapThreshold ? 140 : 0;
        }
        offCtx.putImageData(imgData, 0, 0);

        // Load pre-computed peak assignments from trialData
        const assignedPeaks = {};
        const peakData = trialData?.hrnet_peaks;
        if (peakData) {
            const isLeft = currentSide === cameraNames[0];
            const camPeaks = isLeft ? peakData.peaks_L : peakData.peaks_R;
            const jnames = trialData?.joint_names || [];
            if (camPeaks) {
                for (let j = 0; j < 21; j++) {
                    const name = jnames[j];
                    const frames = camPeaks[name];
                    if (frames && frames[currentFrame]) {
                        assignedPeaks[j] = { x: frames[currentFrame][0], y: frames[currentFrame][1] };
                    }
                }
            }
        }

        _heatmapImageData = { canvas: offCanvas, bbox: [bx1, by1, bx2, by2], peaks: assignedPeaks };
    }

    /** Draw pre-fetched heatmap synchronously (called from drawOverlays). */
    function drawHeatmapOverlay(pixelScale, xOff) {
        if (!_heatmapImageData) return;
        const { canvas: offCanvas, bbox: [bx1, by1, bx2, by2] } = _heatmapImageData;
        const dx = (bx1 + xOff) * pixelScale;
        const dy = by1 * pixelScale;
        const dw = (bx2 - bx1) * pixelScale;
        const dh = (by2 - by1) * pixelScale;
        ctx.drawImage(offCanvas, dx, dy, dw, dh);
        // Note: legacy black-X peak overlay removed.  Peak-select stage's
        // 2D model now renders peak markers in its own color.
    }

    // ── Distance trace ───────────────────────────────────────
    function renderDistanceTrace() {
        if (!distCtx || !trialData) return;
        _autoManageAperture();
        const w = distCanvas.width, h = distCanvas.height;
        distCtx.clearRect(0, 0, w, h);

        const N = trialData.n_frames;
        if (N < 2) { distCtx.fillStyle = '#1a1a2e'; distCtx.fillRect(0, 0, w, h); return; }

        // Build per-source data for each selected metric, then draw all
        const camW = (cameraMode === 'stereo') ? midline : vidW;
        const isLeft = currentSide === cameraNames[0];

        const MCP_JOINT_MAP = { 'MCP: Thumb-Index': [0,1,5], 'MCP: Index-Middle': [5,9], 'MCP: Middle-Ring': [9,13], 'MCP: Ring-Pinky': [13,17] };

        function _getRequiredJoints(metric) {
            if (metric === 'Knuckle: I-M-R') return [5, 9, 13];
            if (metric === 'Knuckle: M-R-P') return [9, 13, 17];
            if (metric.startsWith('MCP:')) return MCP_JOINT_MAP[metric] || [];
            if (metric.startsWith('Pos:')) {
                // "Pos: Wrist X" → joint 0, "Pos: I_MCP Y" → joint 5, etc.
                const jnames = trialData?.joint_names || [];
                for (let j = 0; j < jnames.length; j++) {
                    if (metric.includes(jnames[j])) return [j];
                }
                return [];
            }
            if (metric.startsWith('Wrist ')) return [0];
            if (metric.startsWith('Spread ')) {
                const idx = SPREAD_NAMES.indexOf(metric);
                return idx >= 0 ? SPREAD_JOINT_PAIRS[idx].slice() : [];
            }
            const isAng = metric.startsWith('Flex:') || metric.startsWith('Abd:');
            if (!isAng) {
                const pair = trialData.distance_options?.[metric];
                return pair ? pair.slice() : [];
            } else if (metric.startsWith('Flex:')) {
                if (metric === 'Flex: Wrist') return [0];
                const opt = (trialData.flex_angle_options || []).find(o => o.name === metric);
                return opt ? [opt.parent, opt.joint, opt.child].filter(j => j >= 0) : [];
            } else if (metric.startsWith('Abd:')) {
                if (metric === 'Abd: Wrist') return [0];
                const opt = (trialData.abd_angle_options || []).find(o => o.name === metric);
                return opt ? [opt.parent, opt.joint, opt.child].filter(j => j >= 0) : [];
            } else {
                return [];
            }
        }

        function _getMetricData(src, metric) {
            if (metric.startsWith('Spread ')) return trialData[`spreads_${src}`]?.[metric];
            if (metric.startsWith('Flex:') || metric.startsWith('Abd:') || metric.startsWith('Knuckle:')) return trialData[`angles_${src}`]?.[metric];
            if (metric.startsWith('Pos:')) return trialData[`positions_${src}`]?.[metric];
            if (metric.startsWith('Wrist ')) return trialData[`wrist_coords_${src}`]?.[metric];
            return trialData[`distances_${src}`]?.[metric];
        }

        function _inFrame(framePts, joints) {
            if (!framePts) return false;
            for (const j of joints) {
                const pt = framePts[j];
                if (!pt || pt[0] < 0 || pt[0] > camW || pt[1] < 0 || pt[1] > vidH) return false;
            }
            return true;
        }
        // A frame is valid only when all required joints are in-frame on BOTH cameras
        function _makeInFrameMask(tracked2dL, tracked2dR, joints) {
            if (joints.length === 0) return null;
            if (!tracked2dL && !tracked2dR) return null;
            const mask = [];
            for (let i = 0; i < N; i++) {
                const okL = tracked2dL ? _inFrame(tracked2dL[i], joints) : true;
                const okR = tracked2dR ? _inFrame(tracked2dR[i], joints) : true;
                mask.push(okL && okR);
            }
            return mask;
        }

        function _applyMask(data, mask) {
            if (!data || !mask) return data;
            return data.map((v, i) => mask[i] ? v : null);
        }

        // Compute Y range across all selected metrics and enabled sources
        let distDataMin = Infinity, distDataMax = -Infinity;
        let angDataMin = Infinity, angDataMax = -Infinity;
        for (const [metric] of selectedMetrics) {
            const isAng = metric.startsWith('Flex:') || metric.startsWith('Abd:') || metric.startsWith('Spread ') || metric.startsWith('Knuckle:');
            const skeleton = _getMetricData('skeleton', metric);
            const v2d  = _getMetricData('skel_v2', metric);
            const legacyd = _getMetricData('skel_legacy', metric);
            const mp   = _getMetricData('mp',   metric);
            const crp  = _getMetricData('cropped', metric);
            const rev  = _getMetricData('reverse', metric);
            const stt  = _getMetricData('static', metric);
            const cmb  = _getMetricData('combined', metric);
            const vis  = _getMetricData('vision', metric);
            const dlc  = isAng ? null : trialData.distances_dlc?.[metric];
            const chk = (arr, isA) => {
                if (!arr) return;
                for (const v of arr) {
                    if (v == null) continue;
                    if (isA) { angDataMin = Math.min(angDataMin, v); angDataMax = Math.max(angDataMax, v); }
                    else     { distDataMin = Math.min(distDataMin, v); distDataMax = Math.max(distDataMax, v); }
                }
            };
            if (showMano2D || showMano3D) chk(skeleton, isAng);
            if (showSkelV2_2D || showSkelV2_3D) chk(v2d, isAng);
            if (showLegacyV2_2D || showLegacyV2_3D) chk(legacyd, isAng);
            if (showMP2D || showMP3D) chk(mp, isAng);
            if (showCropped2D || showCropped3D) chk(crp, isAng);
            if (showReverse2D || showReverse3D) chk(rev, isAng);
            if (showStatic2D || showStatic3D) chk(stt, isAng);
            if (showCombined2D || showCombined3D) chk(cmb, isAng);
            if (showVision2D || showVision3D) chk(vis, isAng);
            if (showDLC || showDLC3D) chk(dlc, isAng);
            if (showHeatmap2D || showHeatmap3D) chk(_getMetricData('heatmap', metric), isAng);
            if (showHRnet2D || showHRnet3D)         chk(_getMetricData('hrnet_centroid',  metric), isAng);
            if (showHRnetYZC2D || showHRnetYZC3D)   chk(_getMetricData('hrnet_yzc',       metric), isAng);
            if (showHRnetZSM2D || showHRnetZSM3D)   chk(_getMetricData('hrnet_zsmooth',   metric), isAng);
            if (showStereoHun2D || showStereoHun3D) chk(_getMetricData('hrnet_hungarian', metric), isAng);
        }

        // Auto-fit Y ranges only when the plotted metrics change
        const autoFitKey = [...selectedMetrics.keys()].sort().join('|');
        const dMinSl = $('distYMinSlider'), dMaxSl = $('distYMaxSlider');
        const aMinSl = $('angleYMinSlider'), aMaxSl = $('angleYMaxSlider');

        if (autoFitKey !== _lastAutoFitKey) {
            _lastAutoFitKey = autoFitKey;
            function _autoRange(dMin, dMax, fallbackLo, fallbackHi) {
                if (!isFinite(dMin) || !isFinite(dMax)) return [fallbackLo, fallbackHi];
                const pad = Math.max((dMax - dMin) * 0.1, 5);
                return [Math.floor(dMin - pad), Math.ceil(dMax + pad)];
            }
            const [adMin, adMax] = _autoRange(distDataMin, distDataMax, 0, 200);
            const [aaMin, aaMax] = _autoRange(angDataMin, angDataMax, -150, 50);
            if (dMinSl) dMinSl.value = adMin;
            if (dMaxSl) dMaxSl.value = adMax;
            if (aMinSl) aMinSl.value = aaMin;
            if (aMaxSl) aMaxSl.value = aaMax;
        }

        const distYMin = parseInt(dMinSl?.value ?? '-200');
        const distYMax = parseInt(dMaxSl?.value ?? '200');
        const angYMin  = parseInt(aMinSl?.value ?? '-200');
        const angYMax  = parseInt(aMaxSl?.value ?? '200');

        // Background
        distCtx.fillStyle = '#1a1a2e';
        distCtx.fillRect(0, 0, w, h);

        const xLabelHeight = 14;
        const plotH = h - xLabelHeight;
        const toX = f => (f / (N - 1)) * w;
        // Each metric uses its own Y transform based on type
        const toYDist  = v => plotH - ((v - distYMin) / (distYMax - distYMin)) * plotH;
        const toYAngle = v => plotH - ((v - angYMin)  / (angYMax  - angYMin)  ) * plotH;

        // ── Error-frame markers (vertical red lines) ─────────────────────
        // Drawn behind data series.  A frame is flagged when any joint
        // required by any selected metric has an error on either camera.
        // Uses the union of all active stages' error matrices, plus the
        // HRnet Y/Z-correct preview flags when the Peaks Corrections
        // overlay is on.
        const _orMergeErr = (acc, m) => {
            if (!m || !m.length) return acc;
            if (acc === null) return m;
            const N_ = Math.min(acc.length, m.length);
            const merged = new Array(N_);
            for (let f = 0; f < N_; f++) {
                const a = acc[f], b = m[f];
                if (!a && !b) { merged[f] = null; continue; }
                const row = new Array(21);
                for (let j = 0; j < 21; j++) {
                    const ra = a?.[j], rb = b?.[j];
                    row[j] = [
                        (ra?.[0] ? 1 : 0) | (rb?.[0] ? 1 : 0),
                        (ra?.[1] ? 1 : 0) | (rb?.[1] ? 1 : 0),
                    ];
                }
                merged[f] = row;
            }
            return merged;
        };
        let _activeErrMat = null;
        if (showSkelErrors && skelErrorMatrices) {
            for (const name of _stagesErr) {
                _activeErrMat = _orMergeErr(_activeErrMat, skelErrorMatrices[name]);
            }
        }
        // HRnet "Peaks Corrections" flags are kept sparse (Map<frame,
        // Set<joint>>) and unioned with metricJoints below — we don't
        // expand them into a dense (N, J, 2) matrix.
        const _hrnetFlaggedByFrame = (showHRnetCorrections && hrnetPreview)
            ? hrnetPreview.flaggedJointsByFrame
            : null;
        if ((_activeErrMat && _activeErrMat.length) || _hrnetFlaggedByFrame) {
            const metricJoints = new Set();
            const zSmoothActive = _stagesErr.has('z_smooth');
            const skelPairs = trialData.skeleton || [];
            const isBoneMetric = (m) => {
                const pr = trialData.distance_options?.[m];
                return !!(pr && skelPairs.some(([a, b]) =>
                    (a === pr[0] && b === pr[1]) || (a === pr[1] && b === pr[0])));
            };
            for (const [metric] of selectedMetrics) {
                if (zSmoothActive && isBoneMetric(metric)) continue;
                for (const j of _getRequiredJoints(metric)) metricJoints.add(j);
            }
            if (metricJoints.size) {
                distCtx.save();
                distCtx.strokeStyle = 'rgba(255, 34, 34, 0.55)';
                distCtx.lineWidth = 1;
                distCtx.setLineDash([]);
                distCtx.beginPath();
                const Nerr = _activeErrMat ? Math.min(N, _activeErrMat.length) : N;
                for (let f = 0; f < Nerr; f++) {
                    let flagged = false;
                    // v3 stages: check the dense per-stage merge.
                    if (_activeErrMat) {
                        const frame = _activeErrMat[f];
                        if (frame) {
                            for (const j of metricJoints) {
                                const e = frame[j];
                                if (e && (e[0] || e[1])) { flagged = true; break; }
                            }
                        }
                    }
                    // HRnet preview: sparse Map<frame, Set<joint>>.
                    if (!flagged && _hrnetFlaggedByFrame) {
                        const set = _hrnetFlaggedByFrame.get(f);
                        if (set) {
                            for (const j of metricJoints) {
                                if (set.has(j)) { flagged = true; break; }
                            }
                        }
                    }
                    if (flagged) {
                        const x = toX(f);
                        distCtx.moveTo(x + 0.5, 0);
                        distCtx.lineTo(x + 0.5, plotH);
                    }
                }
                distCtx.stroke();
                distCtx.restore();
            }
        }

        // Source line styles: [lineWidth, dash]
        const SOURCE_STYLES = {
            skel_v2:[2,   []],
            skeleton:   [1.5, []],
            mp:     [1,   []],
            vision: [1,   []],
            dlc:    [1,   []],
            heatmap:[1,   []],
            hrnet_centroid:  [1, []],
            hrnet_yzc:       [1, []],
            hrnet_zsmooth:   [1, []],
            hrnet_hungarian: [1.5, []],
            solid:  [1.2, []],
        };

        function drawSeries(data, color, srcKey, toY, dashOverride) {
            if (!data) return;
            const [lw, dash] = SOURCE_STYLES[srcKey] || [1, []];
            distCtx.strokeStyle = color;
            distCtx.lineWidth = lw;
            distCtx.setLineDash(dashOverride || dash);
            distCtx.beginPath();
            let started = false;
            for (let i = 0; i < N; i++) {
                if (data[i] == null) { started = false; continue; }
                if (!started) { distCtx.moveTo(toX(i), toY(data[i])); started = true; }
                else distCtx.lineTo(toX(i), toY(data[i]));
            }
            distCtx.stroke();
            distCtx.setLineDash([]);
        }

        // Source colors used for Thumb-Index Aperture (always) and single-metric mode
        const SOURCE_COLORS = {
            skeleton: 'lime', skel_v2: '#ff9800', skel_legacy: '#e040fb',
            mp: '#00cccc', cropped: '#7cb342', reverse: '#e040fb',
            static: '#26c6da', combined: '#ffa726',
            vision: '#2196f3', dlc: '#ff4444',
            prev: '#b35b00', heatmap: '#ff6600',
            hrnet_centroid:  '#ff9966',
            hrnet_yzc:       '#ffb074',
            hrnet_zsmooth:   '#ffa040',
            hrnet_hungarian: '#ff8a65',
        };

        function _getPrevMetricData(metric) {
            if (!prevFitData) return null;
            if (metric.startsWith('Spread ')) return prevFitData.spreads?.[metric];
            if (metric.startsWith('Flex:') || metric.startsWith('Abd:') || metric.startsWith('Knuckle:')) return prevFitData.angles?.[metric];
            if (metric.startsWith('Pos:')) return prevFitData.positions?.[metric];
            return prevFitData.distances?.[metric];
        }

        // Draw all selected metrics
        // When exactly one joint's angles are plotted, use model colors and
        // solid for flex/dotted for abd instead of per-metric colors.
        const singleJointAngle = plotJointStates.size === 1 && selectedMetrics.size === 2
            && [...selectedMetrics.keys()].every(m => m.startsWith('Flex:') || m.startsWith('Abd:'));

        _plotMetricData = {};
        let hasAngle = false, hasDist = false;

        // Pooled bone-length threshold: the W-fraction of (bone, frame)
        // samples with the largest |length - median| across ALL bones.  Used
        // for the reference lines on z_smooth bone-length plots so the
        // dashed lines sit at the same percentile cutoff used by the
        // detector.
        const _bl = (_stagesErr.has('z_smooth') && (showSkelV2_2D || showSkelV2_3D))
            ? _computeBoneLengthFlags() : { medians: null, threshold: null };
        const _boneMedians = _bl.medians;
        const _boneDevThreshold = _bl.threshold;

        for (const [metric, metricColor] of selectedMetrics) {
            const isAng = metric.startsWith('Flex:') || metric.startsWith('Abd:') || metric.startsWith('Spread ') || metric.startsWith('Knuckle:');
            if (isAng) hasAngle = true; else hasDist = true;
            const toY = isAng ? toYAngle : toYDist;
            const joints = _getRequiredJoints(metric);
            const mpMask   = _makeInFrameMask(trialData.mp_tracked_L,     trialData.mp_tracked_R,     joints);
            // Reverse-pass MP is a SEPARATE source from forward MP --
            // each has its own valid-frame coverage.  Using mpMask
            // here suppressed reverse on every frame forward had
            // dropped, which on subjects like Con03_R1 (forward MP
            // bad for most of the trial, reverse MP good for nearly
            // all of it) hid the majority of valid reverse distances.
            const revMask  = _makeInFrameMask(trialData.reverse_tracked_L, trialData.reverse_tracked_R, joints);
            const crpMask  = _makeInFrameMask(trialData.cropped_tracked_L, trialData.cropped_tracked_R, joints);
            const sttMask  = _makeInFrameMask(trialData.static_tracked_L, trialData.static_tracked_R, joints);
            const cmbMask  = _makeInFrameMask(trialData.combined_tracked_L, trialData.combined_tracked_R, joints);
            const manoMask = _makeInFrameMask(trialData.skeleton_proj_L,      trialData.skeleton_proj_R,      joints);
            const visMask  = _makeInFrameMask(trialData.vision_tracked_L, trialData.vision_tracked_R, joints);

            const rawMano    = _getMetricData('skeleton',        metric);
            const rawV2      = _getMetricData('skel_v2',     metric);
            const rawLegacy  = _getMetricData('skel_legacy', metric);
            const rawMp      = _getMetricData('mp',          metric);
            const rawCropped  = _getMetricData('cropped',   metric);
            const rawReverse = _getMetricData('reverse',     metric);
            const rawStatic   = _getMetricData('static',    metric);
            const rawCombined = _getMetricData('combined',  metric);
            const rawVis     = _getMetricData('vision',      metric);
            const rawDlc   = isAng ? null : trialData.distances_dlc?.[metric];

            const v2Mask   = _makeInFrameMask(trialData.skel_v2_proj_L, trialData.skel_v2_proj_R, joints);

            // Color/dash logic:
            // - Single joint angle: use model color, flex=solid, abd=dotted
            // - Thumb-Index Aperture: always model colors
            // - Otherwise: metric color
            const useSourceColor = singleJointAngle || (metric === 'Thumb-Index Aperture');
            const isAbd = metric.startsWith('Abd:');
            const abdDash = singleJointAngle && isAbd ? [5, 3] : null;

            if (showMano2D || showMano3D)             drawSeries(_applyMask(rawMano,  manoMask), useSourceColor ? SOURCE_COLORS.skeleton        : metricColor, 'skeleton',        toY, abdDash);
            // Note: the Skeleton (v3) distance line is NOT drawn here on its
            // own anymore.  It's drawn per-active-stage below, so the plot
            // shows exactly the stages the user has selected (and nothing
            // when no stage is active).
            if (showLegacyV2_2D || showLegacyV2_3D)   drawSeries(rawLegacy,                       useSourceColor ? SOURCE_COLORS.skel_legacy : metricColor, 'skel_legacy', toY, abdDash);
            if (showMP2D || showMP3D)                 drawSeries(_applyMask(rawMp,    mpMask),   useSourceColor ? SOURCE_COLORS.mp          : metricColor, 'mp',          toY, abdDash);
            if (showCropped2D || showCropped3D)       drawSeries(_applyMask(rawCropped, crpMask),  useSourceColor ? SOURCE_COLORS.cropped   : metricColor, 'cropped',     toY, abdDash);
            if (showReverse2D || showReverse3D)       drawSeries(_applyMask(rawReverse, revMask), useSourceColor ? SOURCE_COLORS.reverse     : metricColor, 'reverse',     toY, abdDash);
            if (showStatic2D || showStatic3D)         drawSeries(_applyMask(rawStatic, sttMask),   useSourceColor ? SOURCE_COLORS.static    : metricColor, 'static',      toY, abdDash);
            if (showCombined2D || showCombined3D)     drawSeries(_applyMask(rawCombined, cmbMask), useSourceColor ? SOURCE_COLORS.combined  : metricColor, 'combined',    toY, abdDash);
            if (showVision2D || showVision3D)         drawSeries(_applyMask(rawVis,   visMask),  useSourceColor ? SOURCE_COLORS.vision      : metricColor, 'vision',      toY, abdDash);
            if (showDLC || showDLC3D)                 drawSeries(rawDlc,                                                     useSourceColor ? SOURCE_COLORS.dlc         : metricColor, 'dlc',         toY, abdDash);
            if (showHeatmap2D || showHeatmap3D) {
                const rawHm = _getMetricData('heatmap', metric);
                if (rawHm) drawSeries(rawHm, useSourceColor ? SOURCE_COLORS.heatmap : metricColor, 'heatmap', toY, abdDash);
            }
            // HRnet Z-outlier reference lines on Pos: <Joint> Z plots when
            // the Corrections checkbox is on.  Two horizontal red dashed
            // lines at the lo/hi cutoff used by the corrector for the
            // current Z-outlier slider.
            if (showHRnetCorrections && hrnetPreview?.z_outlier_bounds &&
                metric.startsWith('Pos: ') && metric.endsWith(' Z')) {
                const jname = metric.slice(5, -2);
                const b = hrnetPreview.z_outlier_bounds[jname];
                if (b && Number.isFinite(b.lo) && Number.isFinite(b.hi)) {
                    distCtx.save();
                    distCtx.strokeStyle = 'rgba(255, 34, 34, 0.7)';
                    distCtx.lineWidth = 1;
                    distCtx.setLineDash([4, 3]);
                    const yLo = toY(b.lo);
                    const yHi = toY(b.hi);
                    distCtx.beginPath();
                    distCtx.moveTo(0, yLo); distCtx.lineTo(w, yLo);
                    distCtx.moveTo(0, yHi); distCtx.lineTo(w, yHi);
                    distCtx.stroke();
                    distCtx.setLineDash([]);
                    distCtx.restore();
                }
            }

            // HRnet "Peaks" sub-stage (cluster centroid)
            if (showHRnet2D || showHRnet3D) {
                const rawHr = _getMetricData('hrnet_centroid', metric);
                if (rawHr) drawSeries(rawHr, useSourceColor ? SOURCE_COLORS.hrnet_centroid : metricColor, 'hrnet_centroid', toY, abdDash);
            }
            // HRnet "Y/Z-correct" sub-stage
            if (showHRnetYZC2D || showHRnetYZC3D) {
                const rawYZ = _getMetricData('hrnet_yzc', metric);
                if (rawYZ) drawSeries(rawYZ, useSourceColor ? SOURCE_COLORS.hrnet_yzc : metricColor, 'hrnet_yzc', toY, abdDash);
            }
            // HRnet "Z-smooth" sub-stage
            if (showHRnetZSM2D || showHRnetZSM3D) {
                const rawZS = _getMetricData('hrnet_zsmooth', metric);
                if (rawZS) drawSeries(rawZS, useSourceColor ? SOURCE_COLORS.hrnet_zsmooth : metricColor, 'hrnet_zsmooth', toY, abdDash);
            }
            // HRnet "Stereo-Hungarian" sub-stage
            if (showStereoHun2D || showStereoHun3D) {
                const rawSh = _getMetricData('hrnet_hungarian', metric);
                if (rawSh) drawSeries(rawSh, useSourceColor ? SOURCE_COLORS.hrnet_hungarian : metricColor, 'hrnet_hungarian', toY, abdDash);
            }
            if (showPrev2D || showPrev3D) { const rawPrev = _getPrevMetricData(metric); if (rawPrev) drawSeries(rawPrev,     useSourceColor ? SOURCE_COLORS.prev        : metricColor, 'prev',        toY, abdDash); }

            // Per active stage: draw that stage's metric data in its stage
            // color.  If the stage's dedicated metrics aren't present in the
            // trial (e.g., older fit without this intermediate snapshot),
            // fall back to the final Skeleton metrics so the line still
            // renders — the user at least sees *something* at the stage's color.
            // Only plot stages the user has actively displayed (2D or 3D
            // checked).  Stages toggled on via Err alone don't get a
            // dedicated line — the Err overlay rides on whichever stage is
            // displayed.
            const displayStages = new Set([..._stages2D, ..._stages3D]);
            if (displayStages.size > 0 && (showSkelV2_2D || showSkelV2_3D)) {
                const STAGE_FALLBACK = { mediapipe: 'mp',
                                          stereo_correct: 'mp',
                                          z_correct: 'skel_v2', z_smooth: 'skel_v2',
                                          hrnet_snap: 'skel_v2', bone_correct: 'skel_v2',
                                          bone_smooth: 'skel_v2' };
                for (const name of displayStages) {
                    const cfg = STAGE_CONFIGS[name];
                    if (!cfg) continue;
                    let raw = _getMetricData(cfg.metricSrc, metric);
                    if (!raw) raw = _getMetricData(STAGE_FALLBACK[name] || 'skel_v2', metric);
                    if (raw) {
                        // The Mediapipe v3 stage is the same data as the
                        // standalone MediaPipe model — render it with the
                        // thinner 'mp' line style so it visually matches.
                        const lineStyle = (name === 'mediapipe') ? 'mp' : 'skel_v2';
                        drawSeries(raw, cfg.color, lineStyle, toY, abdDash);
                    }
                }
            }

            // Reference lines for bone_length detection: when z_smooth is
            // active and this metric is a bone, show the robust median and
            // the ± threshold (pooled across all bones at the percentile
            // corresponding to the current bone_length slider).
            if (!isAng && _boneMedians && _boneMedians[metric] != null) {
                const med = _boneMedians[metric];
                const stageColor = STAGE_CONFIGS.z_smooth.color;
                const hLine = (yVal, dashed) => {
                    const y = toY(yVal);
                    distCtx.strokeStyle = stageColor;
                    distCtx.globalAlpha = dashed ? 0.7 : 0.9;
                    distCtx.lineWidth = 1;
                    distCtx.setLineDash(dashed ? [4, 4] : []);
                    distCtx.beginPath();
                    distCtx.moveTo(0, y);
                    distCtx.lineTo(w, y);
                    distCtx.stroke();
                    distCtx.setLineDash([]);
                    distCtx.globalAlpha = 1;
                };
                hLine(med, false);
                if (_boneDevThreshold != null && _boneDevThreshold > 0) {
                    hLine(med + _boneDevThreshold, true);
                    hLine(med - _boneDevThreshold, true);
                    // Vertical error markers for every frame this bone's
                    // length sits outside the ± threshold band.  Scoped
                    // to the bone's own out-of-range region so overlapping
                    // plots don't cross-pollute with unrelated bones.
                    const rawBone = _getMetricData('skel_v2_zs', metric)
                                    || _getMetricData('skel_v2', metric);
                    if (rawBone) {
                        const yHi = toY(med + _boneDevThreshold);
                        const yLo = toY(med - _boneDevThreshold);
                        const yTop = Math.min(yHi, yLo);
                        const yBot = Math.max(yHi, yLo);
                        distCtx.strokeStyle = '#ff2222';
                        distCtx.globalAlpha = 0.4;
                        distCtx.lineWidth = 1;
                        distCtx.setLineDash([]);
                        distCtx.beginPath();
                        for (let f = 0; f < N; f++) {
                            const v = rawBone[f];
                            if (v == null) continue;
                            const d = v - med;
                            if (Math.abs(d) <= _boneDevThreshold) continue;
                            const x = toX(f);
                            if (d > 0) {
                                // value above upper band — draw from upper
                                // dashed line up to plot top
                                distCtx.moveTo(x, 0);
                                distCtx.lineTo(x, yTop);
                            } else {
                                // value below lower band — draw from lower
                                // dashed line down to plot bottom
                                distCtx.moveTo(x, yBot);
                                distCtx.lineTo(x, h);
                            }
                        }
                        distCtx.stroke();
                        distCtx.globalAlpha = 1;
                    }
                }
            }

            // Z-outlier reference lines: when the mediapipe stage is active
            // and the plotted metric is a "Pos: <joint> Z", overlay the
            // joint's robust median Z and the ± current Z-outlier
            // threshold (per-joint percentile cutoff matching the
            // current z_outlier slider value).
            if (metric.startsWith('Pos:') && metric.endsWith(' Z') &&
                (_stages2D.has('mediapipe') || _stages3D.has('mediapipe'))) {
                const reqJ = _getRequiredJoints(metric);
                const j = reqJ[0];
                if (j != null) {
                    const yc = _getMetricData('mp', metric);
                    if (yc) {
                        const vals = yc.filter(v => v != null).slice().sort((a, b) => a - b);
                        if (vals.length >= 5) {
                            const med = vals[Math.floor(vals.length / 2)];
                            const W = _mpErrorWeights.detection.z_outlier || 0;
                            const stageColor = STAGE_CONFIGS.mediapipe.color;
                            const hLine = (yVal, dashed) => {
                                const yp = toY(yVal);
                                distCtx.strokeStyle = stageColor;
                                distCtx.globalAlpha = dashed ? 0.7 : 0.9;
                                distCtx.lineWidth = 1;
                                distCtx.setLineDash(dashed ? [4, 4] : []);
                                distCtx.beginPath();
                                distCtx.moveTo(0, yp);
                                distCtx.lineTo(w, yp);
                                distCtx.stroke();
                                distCtx.setLineDash([]);
                                distCtx.globalAlpha = 1;
                            };
                            hLine(med, false);
                            // Per-joint percentile threshold: rank |Z − median|
                            // for this joint and take the (1 − W) quantile.
                            let thr = null;
                            if (W > 0) {
                                const devs = yc.filter(v => v != null)
                                    .map(v => Math.abs(v - med))
                                    .sort((a, b) => a - b);
                                if (devs.length > 0) {
                                    const idx = Math.min(devs.length - 1,
                                        Math.max(0, Math.floor(devs.length * (1 - W))));
                                    thr = devs[idx];
                                }
                            }
                            if (thr != null && thr > 0) {
                                hLine(med + thr, true);
                                hLine(med - thr, true);
                                // Vertical red marks span the full plot
                                // height for frames outside the band.
                                distCtx.strokeStyle = '#ff2222';
                                distCtx.globalAlpha = 0.4;
                                distCtx.lineWidth = 1;
                                distCtx.setLineDash([]);
                                distCtx.beginPath();
                                for (let f = 0; f < N; f++) {
                                    const v = yc[f];
                                    if (v == null) continue;
                                    if (Math.abs(v - med) <= thr) continue;
                                    const x = toX(f);
                                    distCtx.moveTo(x, 0);
                                    distCtx.lineTo(x, h);
                                }
                                distCtx.stroke();
                                distCtx.globalAlpha = 1;
                            }
                        }
                    }
                }
            }

            // Polynomial-fit overlay for Z position when the z_correct
            // stage is checked (2D or 3D) — draws the per-frame deg-2 fit
            // to ±15 surrounding frames in a contrasting color so the
            // residual the corrector sees is visible directly.
            // Helper: per-frame deg-2 poly fit residual on a 1D series,
            // using ±WIN neighbours (excluding f).  Returns N-length array
            // of fit predictions at each frame (or null where impossible).
            const _polyFitOverlay = (series, WIN = 15) => {
                const out = new Array(N).fill(null);
                for (let f = 0; f < N; f++) {
                    if (series[f] == null) continue;
                    const lo = Math.max(0, f - WIN), hi = Math.min(N, f + WIN + 1);
                    const xs = [], ys = [];
                    for (let k = lo; k < hi; k++) {
                        if (k === f || series[k] == null) continue;
                        xs.push(k); ys.push(series[k]);
                    }
                    if (xs.length < 3) continue;
                    const deg = xs.length >= 5 ? 2 : 1;
                    const m = xs.length, p = deg + 1;
                    const ATA = Array.from({length: p}, () => new Array(p).fill(0));
                    const ATy = new Array(p).fill(0);
                    for (let i = 0; i < m; i++) {
                        const xi = xs[i];
                        const xpows = [1, xi, xi*xi].slice(0, p);
                        for (let r = 0; r < p; r++) {
                            ATy[r] += xpows[r] * ys[i];
                            for (let c = 0; c < p; c++) ATA[r][c] += xpows[r] * xpows[c];
                        }
                    }
                    const M = ATA.map((row, i) => [...row, ATy[i]]);
                    let bad = false;
                    for (let i = 0; i < p; i++) {
                        let pi = i;
                        for (let k = i + 1; k < p; k++)
                            if (Math.abs(M[k][i]) > Math.abs(M[pi][i])) pi = k;
                        if (Math.abs(M[pi][i]) < 1e-12) { bad = true; break; }
                        [M[i], M[pi]] = [M[pi], M[i]];
                        for (let k = i + 1; k < p; k++) {
                            const r = M[k][i] / M[i][i];
                            for (let c = i; c <= p; c++) M[k][c] -= r * M[i][c];
                        }
                    }
                    if (bad) continue;
                    const coef = new Array(p).fill(0);
                    for (let i = p - 1; i >= 0; i--) {
                        let s = M[i][p];
                        for (let c = i + 1; c < p; c++) s -= M[i][c] * coef[c];
                        coef[i] = s / M[i][i];
                    }
                    let v = coef[0];
                    if (p > 1) v += coef[1] * f;
                    if (p > 2) v += coef[2] * f * f;
                    out[f] = v;
                }
                return out;
            };

            // Z-axis polynomial overlay on z_correct stage (existing behavior).
            if (metric.startsWith('Pos:') && metric.endsWith(' Z') &&
                (_stages2D.has('z_correct') || _stages3D.has('z_correct'))) {
                const zc = _getMetricData('skel_v2_z', metric)
                        || _getMetricData('skel_v2', metric);
                if (zc) drawSeries(_polyFitOverlay(zc), '#ff66cc', 'solid', toY, [4, 3]);
            }
            // HRnet Z-smooth polynomial reference — fits the corrector's
            // current smooth-window setting onto the cluster-centroid
            // (Peaks) Z series so the user can see the baseline that
            // would be used to predict each frame's "correct" Z.
            if (metric.startsWith('Pos:') && metric.endsWith(' Z') &&
                showHRnetCorrections && hrnetPreview) {
                const zc = _getMetricData('hrnet_centroid', metric);
                if (zc) {
                    const win = parseInt($('hfSliderZSWin')?.value ?? 15);
                    drawSeries(_polyFitOverlay(zc, win), '#ff66cc', 'solid', toY, [4, 3]);
                }
            }

            // 2D-jump-style polynomial overlay on the MP model — fires when
            // (MP 2D or 3D) is on AND mediapipe stage Err is checked, for
            // any X/Y/Z position metric.  Uses the same ±15-frame deg-2
            // poly that drives camera-attribution 2D-jump scoring.
            if (metric.startsWith('Pos:') &&
                (showMP2D || showMP3D) && _stagesErr.has('mediapipe')) {
                const mpSeries = _getMetricData('mp', metric);
                if (mpSeries) drawSeries(_polyFitOverlay(mpSeries),
                                          '#ff66cc', 'solid', toY, [4, 3]);
            }

            // Store primary data for click hit-testing
            const primary = (showSkelV2_2D || showSkelV2_3D) ? rawV2
                          : (showLegacyV2_2D || showLegacyV2_3D) ? rawLegacy
                          : (showMano2D || showMano3D) ? rawMano
                          : (showMP2D || showMP3D) ? rawMp : rawVis;
            if (primary) _plotMetricData[metric] = { data: primary, toY };
        }

        // Show/hide angle slider pair (use visibility to preserve layout width)
        const angleWrap = $('angleYRangeWrap'), angleAxisWrap = $('angleYAxisWrap');
        if (angleWrap) angleWrap.style.visibility = hasAngle ? 'visible' : 'hidden';
        if (angleAxisWrap) angleAxisWrap.style.visibility = hasAngle ? 'visible' : 'hidden';

        // Highlight the focused metric line
        if (_constraintFocusMetric && _plotMetricData[_constraintFocusMetric]) {
            const info = _plotMetricData[_constraintFocusMetric];
            distCtx.save();
            distCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            distCtx.lineWidth = 6;
            distCtx.beginPath();
            let started = false;
            for (let i = 0; i < N; i++) {
                const v = info.data[i];
                if (v == null) { started = false; continue; }
                if (!started) { distCtx.moveTo(toX(i), info.toY(v)); started = true; }
                else distCtx.lineTo(toX(i), info.toY(v));
            }
            distCtx.stroke();
            distCtx.restore();
        }

        // Constraint boundary lines — only for the focused metric
        _constraintHitZones = [];
        if (_constraintFocusMetric && (hasAngle || _constraintFocusMetric.startsWith('MCP:')) && trialData.angle_constraints) {
            const metric = _constraintFocusMetric;
            const isFlx = metric.startsWith('Flex:');
            const isAbd = metric.startsWith('Abd:');
            if (isFlx || isAbd) {
                const constraints = trialData.angle_constraints;
                const jointName = metric.replace('Flex: ', '').replace('Abd: ', '');
                const cst = constraints.find(c => c.name === jointName);
                if (cst) {
                    const ovr = _constraintOverrides[jointName] || {};
                    const loKey = isFlx ? 'flex_min' : 'abd_min';
                    const hiKey = isFlx ? 'flex_max' : 'abd_max';
                    const lo = ovr[loKey] ?? cst[loKey];
                    const hi = ovr[hiKey] ?? cst[hiKey];
                    distCtx.lineWidth = 1;
                    distCtx.setLineDash([4, 4]);
                    distCtx.strokeStyle = 'rgba(255, 80, 80, 0.5)';
                    const yLo = toYAngle(lo);
                    if (yLo >= 0 && yLo <= plotH) {
                        distCtx.beginPath(); distCtx.moveTo(toX(0), yLo); distCtx.lineTo(toX(N - 1), yLo); distCtx.stroke();
                        _constraintHitZones.push({ y: yLo, jointName, key: loKey, toYAngle, angYMin, angYMax, plotH });
                        // Draw value label on left
                        distCtx.setLineDash([]);
                        distCtx.fillStyle = 'rgba(255, 80, 80, 0.8)';
                        distCtx.font = '10px sans-serif';
                        distCtx.textAlign = 'left';
                        distCtx.fillText(`${lo}°`, 2, yLo - 3);
                        distCtx.setLineDash([4, 4]);
                    }
                    const yHi = toYAngle(hi);
                    if (yHi >= 0 && yHi <= plotH) {
                        distCtx.beginPath(); distCtx.moveTo(toX(0), yHi); distCtx.lineTo(toX(N - 1), yHi); distCtx.stroke();
                        _constraintHitZones.push({ y: yHi, jointName, key: hiKey, toYAngle, angYMin, angYMax, plotH });
                        distCtx.setLineDash([]);
                        distCtx.fillStyle = 'rgba(255, 80, 80, 0.8)';
                        distCtx.font = '10px sans-serif';
                        distCtx.textAlign = 'left';
                        distCtx.fillText(`${hi}°`, 2, yHi - 3);
                        distCtx.setLineDash([4, 4]);
                    }
                    distCtx.setLineDash([]);
                }
            }
        }

        // Frame marker — bold bright-blue line with a triangle cap at the
        // top so it's distinguishable from the red error-frame markers.
        const cfx = Math.round(toX(currentFrame)) + 0.5;
        distCtx.save();
        // Soft glow first (wider, translucent) so the line stands out against
        // clusters of red error bars.
        distCtx.strokeStyle = 'rgba(74, 158, 255, 0.35)';
        distCtx.lineWidth = 5;
        distCtx.setLineDash([]);
        distCtx.beginPath();
        distCtx.moveTo(cfx, 0);
        distCtx.lineTo(cfx, plotH);
        distCtx.stroke();
        // Solid core line
        distCtx.strokeStyle = '#4a9eff';
        distCtx.lineWidth = 2;
        distCtx.beginPath();
        distCtx.moveTo(cfx, 0);
        distCtx.lineTo(cfx, plotH);
        distCtx.stroke();
        // Top triangle marker for unmistakable locator
        distCtx.fillStyle = '#4a9eff';
        distCtx.beginPath();
        distCtx.moveTo(cfx - 5, 0);
        distCtx.lineTo(cfx + 5, 0);
        distCtx.lineTo(cfx, 6);
        distCtx.closePath();
        distCtx.fill();
        distCtx.restore();

        // 0° reference line for angle plots
        if (hasAngle && angYMin < 0 && angYMax > 0) {
            const y0 = toYAngle(0);
            distCtx.strokeStyle = 'rgba(255,255,255,0.25)';
            distCtx.lineWidth = 1;
            distCtx.setLineDash([4, 4]);
            distCtx.beginPath();
            distCtx.moveTo(0, y0);
            distCtx.lineTo(w, y0);
            distCtx.stroke();
            distCtx.setLineDash([]);
        }

        // X-axis baseline
        distCtx.strokeStyle = '#444';
        distCtx.lineWidth = 1;
        distCtx.beginPath();
        distCtx.moveTo(0, plotH);
        distCtx.lineTo(w, plotH);
        distCtx.stroke();

        // Y-axis labels — left side: distance only; right side: angle only
        distCtx.fillStyle = '#888';
        distCtx.font = '10px monospace';
        distCtx.textAlign = 'left';
        if (hasDist) {
            distCtx.fillText(distYMax.toFixed(0), 4, 12);
            distCtx.fillText(distYMin.toFixed(0), 4, plotH - 4);
        }
        // Angle values always on the right
        if (hasAngle) {
            distCtx.textAlign = 'right';
            distCtx.fillText(angYMax.toFixed(0) + '°', w - 2, 12);
            distCtx.fillText(angYMin.toFixed(0) + '°', w - 2, plotH - 4);
        }

        // Update Y-axis sidebar labels — distance label left, angle label right
        const yLabel = $('yAxisLabel');
        if (yLabel) yLabel.textContent = 'Distance (mm)';
        const distYAxisWrap = $('distYAxisWrap');
        if (distYAxisWrap) distYAxisWrap.style.visibility = hasDist ? 'visible' : 'hidden';

        // X-axis tick labels (seconds)
        const fps = trialData.fps || 30;
        const totalSec = N / fps;
        // Choose tick interval: 1s, 2s, 5s, 10s, 30s, 60s...
        const intervals = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
        let tickStep = 1;
        for (const iv of intervals) {
            if (totalSec / iv <= 10) { tickStep = iv; break; }
        }
        distCtx.fillStyle = '#666';
        distCtx.font = '9px monospace';
        distCtx.textAlign = 'center';
        for (let t = 0; t <= totalSec; t += tickStep) {
            const x = toX(Math.round(t * fps));
            distCtx.fillText(t % 1 === 0 ? t.toFixed(0) + 's' : t.toFixed(1) + 's', x, plotH + 12);
            // Small tick mark
            distCtx.strokeStyle = '#444';
            distCtx.lineWidth = 1;
            distCtx.beginPath();
            distCtx.moveTo(x, plotH);
            distCtx.lineTo(x, plotH + 3);
            distCtx.stroke();
        }

        _updatePlotValues();
    }

    function _updatePlotValues() {
        const el = $('plotValues');
        if (!el || !trialData) { if (el) el.textContent = ''; return; }
        const fn = currentFrame;
        const parts = [];
        for (const [metric, color] of selectedMetrics) {
            const info = _plotMetricData[metric];
            if (!info || !info.data) continue;
            const val = info.data[fn];
            if (val == null) continue;
            const isAng = metric.startsWith('Flex:') || metric.startsWith('Abd:') || metric.startsWith('Spread ') || metric.startsWith('Knuckle:');
            const unit = isAng ? '°' : metric.startsWith('Pos:') ? 'mm' : 'mm';
            // Short label: strip prefix
            const label = metric.replace('Flex: ', 'F:').replace('Abd: ', 'A:').replace('MCP: ', '').replace('Pos: ', '');
            parts.push(`<span style="color:${color};">${label} ${val.toFixed(1)}${unit}</span>`);
        }
        el.innerHTML = parts.join(' &nbsp; ') || '';
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
        // Disable auto projection matrix updates — we set a custom matrix
        // from calibration intrinsics. Without this, Three.js overwrites
        // our matrix before every render, making all projection fixes invisible.
        camera3d.projectionMatrixAutoUpdate = false;
        camera3d.position.set(0, 0, 500);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearColor(0x000000, 0);
        renderer.setSize(container.clientWidth, container.clientHeight);
        // Don't scale for devicePixelRatio — the custom projection matrix
        // uses CSS-pixel dimensions (same as the 2D canvas). Setting DPR
        // makes the renderer's internal resolution 2x while the projection
        // stays at 1x, causing a vertical offset on Retina displays.
        renderer.setPixelRatio(1);
        container.appendChild(renderer.domElement);

        container.classList.add('interactive');

        // Lights
        scene.add(new THREE.AmbientLight(0x606060));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(100, 100, 100);
        scene.add(dirLight);

        // Groups
        manoGroup = new THREE.Group();
        skelV2Group = new THREE.Group();
        legacyGroup = new THREE.Group();
        mpGroup = new THREE.Group();
        croppedGroup = new THREE.Group();
        reverseGroup = new THREE.Group();
        staticGroup = new THREE.Group();
        combinedGroup = new THREE.Group();
        visionGroup = new THREE.Group();
        dlcGroup = new THREE.Group();
        poseGroup = new THREE.Group();
        heatmapGroup = new THREE.Group();
        angleArcGroup = new THREE.Group();
        scene.add(manoGroup, skelV2Group, legacyGroup, mpGroup, croppedGroup, reverseGroup, staticGroup, combinedGroup, visionGroup, dlcGroup, poseGroup, heatmapGroup, angleArcGroup);

        renderer.render(scene, camera3d);

        // ── Manual orbit: rotate scene content, camera stays fixed ──
        // This keeps the custom projection valid at all times.
        container.addEventListener('mousedown', e => {
            // Plain drag → pan (handled by capture listener on video canvas)
            // Shift+drag → 3D orbit rotation (all models)
            // Cmd/Ctrl+drag → 3D translate (clicked model only)
            if (!(e.metaKey || e.shiftKey || e.ctrlKey)) return;
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();

            if (e.metaKey || e.ctrlKey) {
                // Translate mode — find which model was clicked
                const target = _hitTestModelGroup(e);
                if (target) {
                    _translateDragging = true;
                    _translateTarget = target;
                    _translateLastX = e.clientX;
                    _translateLastY = e.clientY;
                }
            } else {
                // Rotate mode
                orbitDragging = true;
                orbitLastX = e.clientX;
                orbitLastY = e.clientY;
            }
        });
        container.addEventListener('mousemove', e => {
            if (orbitDragging) {
                const dx = e.clientX - orbitLastX;
                const dy = e.clientY - orbitLastY;
                orbitLastX = e.clientX;
                orbitLastY = e.clientY;
                const camRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera3d.quaternion);
                const camUp    = new THREE.Vector3(0, 1, 0).applyQuaternion(camera3d.quaternion);
                const qH = new THREE.Quaternion().setFromAxisAngle(camUp, dx * 0.005);
                const qV = new THREE.Quaternion().setFromAxisAngle(camRight, dy * 0.005);
                orbitQuat.premultiply(qH).premultiply(qV);
                update3D();
            }
            if (_translateDragging && _translateTarget) {
                const dx = e.clientX - _translateLastX;
                const dy = e.clientY - _translateLastY;
                _translateLastX = e.clientX;
                _translateLastY = e.clientY;
                // Convert screen pixels to 3D world units using camera
                const camRight = new THREE.Vector3(1, 0, 0).applyQuaternion(camera3d.quaternion);
                const camUp    = new THREE.Vector3(0, 1, 0).applyQuaternion(camera3d.quaternion);
                const speed = 0.5; // world units per pixel
                const offset = modelTranslations[_translateTarget];
                offset.addScaledVector(camRight, dx * speed);
                offset.addScaledVector(camUp, -dy * speed);
                update3D();
            }
        });
        container.addEventListener('mouseup', () => {
            orbitDragging = false;
            _translateDragging = false;
            _translateTarget = null;
        });
        container.addEventListener('mouseleave', () => {
            orbitDragging = false;
            _translateDragging = false;
            _translateTarget = null;
        });

        // Click on a 3D joint sphere → toggle angle plot
        let _3dMouseDownPos = null;
        container.addEventListener('mousedown', e => { _3dMouseDownPos = { x: e.clientX, y: e.clientY }; }, true);
        container.addEventListener('click', e => {
            if (!_3dMouseDownPos) return;
            const dx = e.clientX - _3dMouseDownPos.x, dy = e.clientY - _3dMouseDownPos.y;
            _3dMouseDownPos = null;
            if (dx * dx + dy * dy > 25) return; // was a drag, not a click

            if (!renderer || !camera3d || !trialData) return;
            const rect = renderer.domElement.getBoundingClientRect();
            const mouse = new THREE.Vector2(
                ((e.clientX - rect.left) / rect.width) * 2 - 1,
                -((e.clientY - rect.top) / rect.height) * 2 + 1,
            );
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(mouse, camera3d);

            // Stereo 2D hit-test (no 3D sphere to raycast against).
            // Pink crosses sit on the video canvas below this overlay,
            // so we have to do the hit-test HERE rather than on the
            // videoCanvas (whose click handler the overlay blocks).
            if ((showStereo2D && trialData?.has_stereo)
                || (showStereoOutline2D && trialData?.has_stereo_outline)
                || (showStereoHybrid2D && trialData?.has_stereo_hybrid)) {
                const cRect = canvas.getBoundingClientRect();
                const cmx = (e.clientX - cRect.left - offsetX) / scale;
                const cmy = (e.clientY - cRect.top  - offsetY) / scale;
                const _isLeft = currentSide === cameraNames[0];
                const _midline = vidW / 2;
                const _pixelScale = canvas.width / (cameraMode === 'stereo'
                    ? (_isLeft ? _midline : vidW - _midline)
                    : vidW);
                const HIT_R = 12 / _pixelScale;
                const HIT_R2 = HIT_R * HIT_R;
                const fn = currentFrame;
                // Search both variants' crosses; nearer one wins.
                const candidates = [];
                if (showStereo2D && trialData?.has_stereo) {
                    candidates.push(_isLeft ? trialData.stereo_tracked_L
                                            : trialData.stereo_tracked_R);
                }
                if (showStereoOutline2D && trialData?.has_stereo_outline) {
                    candidates.push(_isLeft ? trialData.stereo_outline_tracked_L
                                            : trialData.stereo_outline_tracked_R);
                }
                if (showStereoHybrid2D && trialData?.has_stereo_hybrid) {
                    candidates.push(_isLeft ? trialData.stereo_hybrid_tracked_L
                                            : trialData.stereo_hybrid_tracked_R);
                }
                let sBest = null;
                for (const sArr of candidates) {
                    if (!sArr?.[fn]) continue;
                    for (let j = 0; j < 21; j++) {
                        if (!isJointVisible(j) || !sArr[fn][j]) continue;
                        const jx = sArr[fn][j][0], jy = sArr[fn][j][1];
                        const d2 = (jx - cmx / _pixelScale) ** 2
                                 + (jy - cmy / _pixelScale) ** 2;
                        if (d2 < HIT_R2 && (!sBest || d2 < sBest.d2)) {
                            sBest = { j, d2 };
                        }
                    }
                }
                if (sBest) {
                    stereoSelectedJoint = (stereoSelectedJoint === sBest.j) ? null : sBest.j;
                    render();
                    return;
                }
            }

            // Raycast against all model groups, find closest sphere
            const groups = [
                ['skel_v2', skelV2Group],
                ['skeleton', manoGroup],
                ['mp', mpGroup],
                ['vision', visionGroup],
            ];
            let bestHit = null;
            for (const [src, group] of groups) {
                if (!group || group.children.length === 0) continue;
                const hits = raycaster.intersectObjects(group.children, false);
                for (const h of hits) {
                    if (!h.object.geometry?.type?.includes('Sphere')) continue;
                    if (!bestHit || h.distance < bestHit.distance) {
                        // Find which joint index this sphere is — it's the nth sphere in the group
                        const spheres = group.children.filter(c => c.geometry?.type?.includes('Sphere'));
                        const idx = spheres.indexOf(h.object);
                        if (idx >= 0) bestHit = { joint: idx, source: src, distance: h.distance };
                    }
                }
            }
            if (!bestHit) return;

            // Map sphere index to actual joint index (spheres are added in order 0-20, skipping invisible ones)
            // We need to find which joint the nth visible sphere corresponds to
            const j = bestHit.joint; // This is already the index in the visible-joint list
            // For now, the spheres correspond to joints in order of visibility — find the actual joint
            // Actually spheres are added for j=0..20, skipping invisible/null, so the index matches
            // the position in the filtered list. We need to reconstruct.
            const fn = currentFrame;
            const src = bestHit.source;
            const pts = src === 'skel_v2' ? trialData.skel_v2_joints_3d?.[fn]
                      : src === 'skeleton' ? trialData.skeleton_joints_3d?.[fn]
                      : src === 'mp' ? trialData.mp_joints_3d?.[fn]
                      : trialData.vision_joints_3d?.[fn];
            if (!pts) return;
            let visibleIdx = 0;
            let actualJoint = -1;
            for (let jj = 0; jj < 21; jj++) {
                if (!isJointVisible(jj) || !pts[jj]) continue;
                if (visibleIdx === bestHit.joint) { actualJoint = jj; break; }
                visibleIdx++;
            }
            if (actualJoint < 0) return;

            // Toggle angle plot (same as Plotting hand click and canvas click)
            const flexOpts = trialData?.flex_angle_options || [];
            const flexMatch = flexOpts.find(f => f.joint === actualJoint);
            if (!flexMatch) return;
            const flexName = flexMatch.name;
            const abdName = flexName.replace('Flex:', 'Abd:');

            if (plotJointStates.has(actualJoint)) {
                selectedMetrics.delete(flexName);
                selectedMetrics.delete(abdName);
                plotJointStates.delete(actualJoint);
            } else {
                const c1 = _nextMetricColor();
                selectedMetrics.set(flexName, c1);
                const c2 = _nextMetricColor();
                selectedMetrics.set(abdName, c2);
                plotJointStates.set(actualJoint, { mode: 'both', colorFlex: c1, colorAbd: c2 });
                _enforceSourceConstraint();
            }
            renderDistanceTrace();
            _updatePlotHighlight();
            render();
            update3D();
        });
    }

    /** Hit-test which model group the mouse is over using Three.js raycasting. */
    function _hitTestModelGroup(mouseEvent) {
        if (!renderer || !camera3d) return null;
        const rect = renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((mouseEvent.clientX - rect.left) / rect.width) * 2 - 1,
            -((mouseEvent.clientY - rect.top) / rect.height) * 2 + 1,
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera3d);

        // Test each group — return the name of the closest hit
        const groups = [
            ['skeleton', manoGroup],
            ['skel_v2', skelV2Group],
            ['mp', mpGroup],
            ['vision', visionGroup],
            ['dlc', dlcGroup],
        ];
        let closest = null;
        let closestDist = Infinity;
        for (const [name, group] of groups) {
            if (!group || !group.visible || group.children.length === 0) continue;
            const hits = raycaster.intersectObjects(group.children, true);
            if (hits.length > 0 && hits[0].distance < closestDist) {
                closestDist = hits[0].distance;
                closest = name;
            }
        }
        return closest;
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

    /** Build a partial-circle arc as a solid tube mesh (bold, like skeleton bones).
     *  center:    THREE.Vector3 — arc center (the joint)
     *  zeroDir:   normalized THREE.Vector3 — direction at 0° (along incoming bone)
     *  sweepAxis: normalized THREE.Vector3 — perpendicular in the arc plane; positive = positive angle
     *  angleDeg:  signed arc angle in degrees
     *  radius:    arc radius in scene units
     *  hexColor:  CSS hex color string
     */
    function makeAngleArc(center, zeroDir, sweepAxis, angleDeg, radius, hexColor, showTick = true) {
        if (Math.abs(angleDeg) < 0.5) return null;
        const segments = 40;
        const tubeRadius = 0.9;  // slightly narrower than Skeleton bones (1.2)
        const angleRad = THREE.MathUtils.degToRad(angleDeg);

        const color = new THREE.Color(hexColor);
        const mat = new THREE.MeshPhongMaterial({ color });

        // Build arc path as a CatmullRomCurve3 for TubeGeometry
        const points = [];
        for (let i = 0; i <= segments; i++) {
            const t = (i / segments) * angleRad;
            points.push(new THREE.Vector3(
                center.x + radius * (Math.cos(t) * zeroDir.x + Math.sin(t) * sweepAxis.x),
                center.y + radius * (Math.cos(t) * zeroDir.y + Math.sin(t) * sweepAxis.y),
                center.z + radius * (Math.cos(t) * zeroDir.z + Math.sin(t) * sweepAxis.z),
            ));
        }
        const curve = new THREE.CatmullRomCurve3(points);
        const arcMesh = new THREE.Mesh(new THREE.TubeGeometry(curve, segments, tubeRadius, 6, false), mat);

        const group = new THREE.Group();
        group.add(arcMesh);

        if (showTick) {
            // Zero-reference radial tick (thinner spoke to arc start)
            const tickEnd = center.clone().addScaledVector(zeroDir, radius);
            const tick = makeBone(center, tickEnd, tubeRadius * 0.55, mat);
            if (tick) group.add(tick);
        }
        return group;
    }

    function update3D() {
        if (!scene || !trialData) return;
        _computeBoneLengthFlags();
        const fn = currentFrame;
        let arcSrcKey = null;

        // Clear groups
        [manoGroup, skelV2Group, legacyGroup, mpGroup, croppedGroup, reverseGroup, staticGroup, combinedGroup, visionGroup, dlcGroup, poseGroup, heatmapGroup, angleArcGroup].forEach(g => {
            while (g.children.length) {
                const child = g.children[0];
                g.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            }
        });

        const skel3d = trialData.skeleton_joints_3d?.[fn];
        const v2_3d = trialData.skel_v2_joints_3d?.[fn];
        const legacy_3d = trialData.skel_legacy_joints_3d?.[fn];
        const mp3d = trialData.mp_joints_3d?.[fn];
        const cropped3d = trialData.cropped_joints_3d?.[fn];
        const reverse3d = trialData.reverse_joints_3d?.[fn];
        const static3d = trialData.static_joints_3d?.[fn];
        const combined3d = trialData.combined_joints_3d?.[fn];
        const vision3d = trialData.vision_joints_3d?.[fn];

        const sphereGeom = new THREE.SphereGeometry(2.5, 12, 8);

        // Helper to convert OpenCV (x,y,z) to Three.js scene (x,-y,-z)
        const toScene = p => new THREE.Vector3(p[0], -p[1], -p[2]);

        // Pivot computed after getScenePos is defined (below)

        // Position 3D spheres so they project to the correct 2D positions.
        // Uses the 2D overlay coordinates + triangulated depth to place each
        // joint exactly where it should appear on screen.
        const isLeftCam = currentSide === cameraNames[0];
        const K_cam = isLeftCam ? trialData?.calib?.K_L : trialData?.calib?.K_R;
        const mp2d_cam = isLeftCam ? trialData?.mp_tracked_L : trialData?.mp_tracked_R;
        const R_cam = trialData?.calib?.R;
        const T_cam = trialData?.calib?.T;

        const getScenePos = (pts3d, j, isMano, override2dFrame) => {
            // For Skeleton joints, use raw 3D positions (they have their own fitting),
            // UNLESS an explicit per-stage 2D anchor is provided — then snap the
            // sphere to that 2D pixel using the same depth-from-3D math we use
            // for MP joints, so the 3D sphere lands at the stage's own 2D label
            // even when the calibration has reprojection error.
            const anchor = override2dFrame?.[j];
            if (isMano && !anchor) return toScene(pts3d[j]);

            // 2D pixel + triangulated depth → 3D scene position
            const u_v = anchor || mp2d_cam?.[fn]?.[j];
            if (K_cam && u_v && pts3d[j]) {
                const cfx = K_cam[0][0], cfy = K_cam[1][1];
                const ccx = K_cam[0][2], ccy = K_cam[1][2];
                const u = u_v[0];
                const v = u_v[1];

                // Get depth in current camera frame
                let X = pts3d[j][0], Y = pts3d[j][1], Z = pts3d[j][2];
                let camZ;
                if (!isLeftCam && R_cam && T_cam) {
                    camZ = R_cam[2][0]*X + R_cam[2][1]*Y + R_cam[2][2]*Z + T_cam[2];
                } else {
                    camZ = Z;
                }
                if (camZ <= 0) return toScene(pts3d[j]);

                // Unproject: 2D pixel + depth → camera 3D
                const camX = (u - ccx) * camZ / cfx;
                const camY = (v - ccy) * camZ / cfy;

                // Convert to world coords
                let wX, wY, wZ;
                if (!isLeftCam && R_cam && T_cam) {
                    const dx = camX - T_cam[0], dy = camY - T_cam[1], dz = camZ - T_cam[2];
                    wX = R_cam[0][0]*dx + R_cam[1][0]*dy + R_cam[2][0]*dz;
                    wY = R_cam[0][1]*dx + R_cam[1][1]*dy + R_cam[2][1]*dz;
                    wZ = R_cam[0][2]*dx + R_cam[1][2]*dy + R_cam[2][2]*dz;
                } else {
                    wX = camX; wY = camY; wZ = camZ;
                }
                return new THREE.Vector3(wX, -wY, -wZ);
            }
            return toScene(pts3d[j]);
        };

        // Compute orbit pivot = centroid of highest-priority visible model
        if (!orbitDragging) {
            const v2Valid = v2_3d && v2_3d.some(j => j != null);
            const manoValid = skel3d && skel3d.some(j => j != null);
            const mpValid = mp3d && mp3d.some(j => j != null);
            const visionValid = vision3d && vision3d.some(j => j != null);
            const pivotPts = v2Valid ? v2_3d : (manoValid ? skel3d : (mpValid ? mp3d : (visionValid ? vision3d : null)));
            const pivotIsMano = !!(v2Valid || manoValid);
            if (pivotPts) {
                let px = 0, py = 0, pz = 0, pn = 0;
                for (let j = 0; j < 21; j++) {
                    if (!pivotPts[j]) continue;
                    const p = getScenePos(pivotPts, j, pivotIsMano);
                    px += p.x; py += p.y; pz += p.z;
                    pn++;
                }
                if (pn > 0) orbitPivot.set(px/pn, py/pn, pz/pn);
            }
        }

        // Apply orbit rotation
        const orbitPt = (p) => {
            if (orbitQuat.w === 1) return p;
            return p.clone().sub(orbitPivot).applyQuaternion(orbitQuat).add(orbitPivot);
        };

        // Skeleton joints (green)
        if (showMano3D && skel3d) {
            const manoMat = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
            const boneMat = new THREE.MeshPhongMaterial({ color: 0x00dd00 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !skel3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, manoMat);
                sphere.position.copy(orbitPt(getScenePos(skel3d, j, true)));
                manoGroup.add(sphere);
            }
            if (showManoSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !skel3d[i] || !skel3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(skel3d, i, true)),
                        orbitPt(getScenePos(skel3d, j, true)),
                        1.2, boneMat
                    );
                    if (bone) manoGroup.add(bone);
                });
            }
        }

        // Skeleton (v3) joints — per active stage.  Each stage uses its own
        // 3D point set and base color; errors colored per current camera.
        if (showSkelV2_3D) {
            const errHalo = new THREE.MeshBasicMaterial({
                color: 0xff2222, transparent: true, opacity: 0.35, depthWrite: false,
            });
            const haloGeom = new THREE.SphereGeometry(4.2, 12, 8);
            const errMat = new THREE.MeshPhongMaterial({ color: 0xff2222, emissive: 0x881111 });
            const camIdx3d = (currentSide === cameraNames[0]) ? 0 : 1;
            for (const s of _skelStages3D(isLeftCam)) {
                const pts = s.pts?.[fn];
                if (!pts) continue;
                const stageProj2D = s.proj2D?.[fn] || null;
                const base = new THREE.MeshPhongMaterial({ color: s.color, emissive: s.emissive });
                const boneMat = new THREE.MeshPhongMaterial({ color: s.color, emissive: s.emissive });
                const errBoneMat = new THREE.MeshPhongMaterial({ color: 0xff2222, emissive: 0x881111 });
                const errFrame3d = s.errors ? s.errors[fn] : null;
                // Helper: treat [NaN,NaN,NaN] same as null so NaN positions
                // don't end up in THREE.js geometry.
                const validPt = (p) => p && !Number.isNaN(p[0]) && !Number.isNaN(p[1]) && !Number.isNaN(p[2]);
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !validPt(pts[j])) continue;
                    let thisCamErr, anyErr;
                    if (s.factor === 'bone_length') {
                        const flagged = _isJointFlaggedByBoneLength(j, fn);
                        thisCamErr = flagged;
                        anyErr     = flagged;
                    } else {
                        const e = errFrame3d && errFrame3d[j] ? errFrame3d[j] : null;
                        thisCamErr = !!(e && e[camIdx3d]);
                        anyErr     = !!(e && (e[0] || e[1]));
                    }
                    // Use the larger halo geometry for the red error
                    // sphere too so its size matches the translucent halo
                    // shown when the other camera is blamed.
                    const geom = thisCamErr ? haloGeom : sphereGeom;
                    const sphere = new THREE.Mesh(geom, thisCamErr ? errMat : base);
                    sphere.position.copy(orbitPt(getScenePos(pts, j, true, stageProj2D)));
                    skelV2Group.add(sphere);
                    // Halo only when the OTHER camera is blamed and this
                    // one isn't — so the red sphere stays clearly visible
                    // when this camera is the bad side.
                    if (anyErr && !thisCamErr) {
                        const halo = new THREE.Mesh(haloGeom, errHalo);
                        halo.position.copy(sphere.position);
                        skelV2Group.add(halo);
                    }
                }
                if (showSkelV2Skel && trialData.skeleton) {
                    trialData.skeleton.forEach(([i, j]) => {
                        if (!isBoneVisible(i, j) || !validPt(pts[i]) || !validPt(pts[j])) return;
                        let bm = boneMat;
                        if (s.factor === 'bone_length' && _boneFlagsByPair) {
                            const lo = Math.min(i, j), hi = Math.max(i, j);
                            if (_boneFlagsByPair[`${lo}-${hi}`]?.[fn]) bm = errBoneMat;
                        } else if (s.factor === 'bone_agreement' && errFrame3d) {
                            const eI = errFrame3d[i];
                            const eJ = errFrame3d[j];
                            const eiOn = eI && (eI[0] || eI[1]);
                            const ejOn = eJ && (eJ[0] || eJ[1]);
                            if (eiOn && ejOn) bm = errBoneMat;
                        }
                        const bone = makeBone(
                            orbitPt(getScenePos(pts, i, true, stageProj2D)),
                            orbitPt(getScenePos(pts, j, true, stageProj2D)),
                            1.2, bm
                        );
                        if (bone) skelV2Group.add(bone);
                    });
                }
            }
        }

        // Skeleton v2 (legacy) joints (magenta)
        if (showLegacyV2_3D && legacy_3d) {
            const legMat = new THREE.MeshPhongMaterial({ color: 0xe040fb, emissive: 0x601080 });
            const legBoneMat = new THREE.MeshPhongMaterial({ color: 0xe040fb, emissive: 0x500870 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !legacy_3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, legMat);
                sphere.position.copy(orbitPt(getScenePos(legacy_3d, j, true)));
                legacyGroup.add(sphere);
            }
            if (showLegacyV2Skel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !legacy_3d[i] || !legacy_3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(legacy_3d, i, true)),
                        orbitPt(getScenePos(legacy_3d, j, true)),
                        1.2, legBoneMat
                    );
                    if (bone) legacyGroup.add(bone);
                });
            }
        }

        // Per-frame (21, 2) 2D peak arrays for the current camera, used as
        // 2D-pixel anchors when rendering the HRnet 3D models so they sit
        // on top of the camera's actual peak (matches calibration y-bias
        // compensation applied to the v3 stage spheres).
        const _hrnetPeakAnchorFrame = (kind /* 'refined'|'raw'|'centroid'|'hungarian' */) => {
            const pd = trialData?.hrnet_peaks;
            const jnames = trialData?.joint_names;
            if (!pd || !jnames) return null;
            const FIELD = {
                centroid:  isLeftCam ? 'peaks_centroid_L'  : 'peaks_centroid_R',
                yzc:       isLeftCam ? 'peaks_yzc_L'       : 'peaks_yzc_R',
                zsmooth:   isLeftCam ? 'peaks_zsmooth_L'   : 'peaks_zsmooth_R',
                hungarian: isLeftCam ? 'peaks_hungarian_L' : 'peaks_hungarian_R',
                raw:       isLeftCam ? 'peaks_L_raw'       : 'peaks_R_raw',
                refined:   isLeftCam ? 'peaks_L'           : 'peaks_R',
            };
            let dict = pd[FIELD[kind]];
            // Fall back to legacy raw argmax if HRnet Fit hasn't run yet.
            if ((!dict || !Object.keys(dict).length) && kind === 'centroid') {
                dict = pd[FIELD.raw];
            }
            if (!dict) return null;
            const out = new Array(21).fill(null);
            for (let j = 0; j < 21; j++) {
                const arr = dict[jnames[j]];
                if (arr && arr[fn]) out[j] = arr[fn];
            }
            return out;
        };

        // Heatmap-peaks 3D (orange triangulated HRNet peaks).  Only show
        // 3D spheres/tubes when the 3D box is on; otherwise 2D-only Skel
        // renders as lines in the 2D canvas without 3D tubes.
        if (showHeatmap3D && trialData.hrnet_peaks_3d) {
            const hm3d = trialData.hrnet_peaks_3d[fn];
            if (hm3d) {
                const hmMat = new THREE.MeshPhongMaterial({ color: 0xff6600, emissive: 0x803000 });
                const hmBoneMat = new THREE.MeshPhongMaterial({ color: 0xff6600, emissive: 0x602200 });
                const validPt = (p) => p && !Number.isNaN(p[0]) && !Number.isNaN(p[1]) && !Number.isNaN(p[2]);
                const peakAnchor = _hrnetPeakAnchorFrame('refined');
                if (showHeatmap3D) {
                    for (let j = 0; j < 21; j++) {
                        if (!isJointVisible(j) || !validPt(hm3d[j])) continue;
                        const sphere = new THREE.Mesh(sphereGeom, hmMat);
                        sphere.position.copy(orbitPt(getScenePos(hm3d, j, true, peakAnchor)));
                        heatmapGroup.add(sphere);
                    }
                }
                if (showHeatmapSkel && trialData.skeleton) {
                    trialData.skeleton.forEach(([i, j]) => {
                        if (!isBoneVisible(i, j) || !validPt(hm3d[i]) || !validPt(hm3d[j])) return;
                        const bone = makeBone(
                            orbitPt(getScenePos(hm3d, i, true, peakAnchor)),
                            orbitPt(getScenePos(hm3d, j, true, peakAnchor)),
                            1.2, hmBoneMat
                        );
                        if (bone) heatmapGroup.add(bone);
                    });
                }
            }
        }

        // HRnet "Peaks" sub-stage 3D (cluster centroid; legacy raw argmax fallback).
        if (showHRnet3D) {
            const pts3d = (trialData.hrnet_centroid_3d
                          || trialData.hrnet_peaks_raw_3d || []);
            const hm3dr = pts3d[fn];
            if (hm3dr) {
                const hmMat = new THREE.MeshPhongMaterial({ color: 0xff9966, emissive: 0x603020 });
                const errMat = new THREE.MeshPhongMaterial({ color: 0xff2222, emissive: 0x881111 });
                const hmBoneMat = new THREE.MeshPhongMaterial({ color: 0xff9966, emissive: 0x502818 });
                // Translucent red halo for the "other camera is blamed"
                // case — matches the Skeleton-v3 error visualisation.
                const errHalo = new THREE.MeshBasicMaterial({
                    color: 0xff2222, transparent: true, opacity: 0.35, depthWrite: false,
                });
                const haloGeom = new THREE.SphereGeometry(4.2, 12, 8);
                const validPt = (p) => p && !Number.isNaN(p[0]) && !Number.isNaN(p[1]) && !Number.isNaN(p[2]);
                const peakAnchor = trialData.hrnet_centroid_3d
                    ? _hrnetPeakAnchorFrame('centroid')
                    : _hrnetPeakAnchorFrame('raw');
                const correctionsOn = (showHRnetCorrections && hrnetPreview);
                const camIdx = isLeftCam ? 0 : 1;
                const otherIdx = isLeftCam ? 1 : 0;
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !validPt(hm3dr[j])) continue;
                    const thisCamErr = correctionsOn && (
                        hrnetPreview.hasYZ(fn, j, camIdx)
                        || hrnetPreview.hasZO(fn, j, camIdx));
                    const otherCamErr = correctionsOn && (
                        hrnetPreview.hasYZ(fn, j, otherIdx)
                        || hrnetPreview.hasZO(fn, j, otherIdx));
                    const pos = orbitPt(getScenePos(hm3dr, j, true, peakAnchor));
                    // Use the larger halo geometry for the red error
                    // sphere too so its size matches the translucent halo.
                    const geom = thisCamErr ? haloGeom : sphereGeom;
                    const sphere = new THREE.Mesh(geom, thisCamErr ? errMat : hmMat);
                    sphere.position.copy(pos);
                    heatmapGroup.add(sphere);
                    // Halo only when the OTHER camera is blamed and this
                    // one isn't — keeps the red joint clearly visible
                    // when THIS camera is the blamed side.
                    if (otherCamErr && !thisCamErr) {
                        const halo = new THREE.Mesh(haloGeom, errHalo);
                        halo.position.copy(pos);
                        heatmapGroup.add(halo);
                    }
                }
                if (showHeatmapSkel && trialData.skeleton) {
                    trialData.skeleton.forEach(([i, j]) => {
                        if (!isBoneVisible(i, j) || !validPt(hm3dr[i]) || !validPt(hm3dr[j])) return;
                        const bone = makeBone(
                            orbitPt(getScenePos(hm3dr, i, true, peakAnchor)),
                            orbitPt(getScenePos(hm3dr, j, true, peakAnchor)),
                            1.2, hmBoneMat
                        );
                        if (bone) heatmapGroup.add(bone);
                    });
                }
            }
        }

        // HRnet "Y/Z-correct" sub-stage 3D (warmer orange).
        if (showHRnetYZC3D && trialData.hrnet_yzc_3d) {
            const hm3dy = trialData.hrnet_yzc_3d[fn];
            if (hm3dy) {
                const hmMat = new THREE.MeshPhongMaterial({ color: 0xffb074, emissive: 0x804020 });
                const hmBoneMat = new THREE.MeshPhongMaterial({ color: 0xffb074, emissive: 0x603018 });
                const validPt = (p) => p && !Number.isNaN(p[0]) && !Number.isNaN(p[1]) && !Number.isNaN(p[2]);
                const peakAnchor = _hrnetPeakAnchorFrame('yzc');
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !validPt(hm3dy[j])) continue;
                    const sphere = new THREE.Mesh(sphereGeom, hmMat);
                    sphere.position.copy(orbitPt(getScenePos(hm3dy, j, true, peakAnchor)));
                    heatmapGroup.add(sphere);
                }
                if (showHeatmapSkel && trialData.skeleton) {
                    trialData.skeleton.forEach(([i, j]) => {
                        if (!isBoneVisible(i, j) || !validPt(hm3dy[i]) || !validPt(hm3dy[j])) return;
                        const bone = makeBone(
                            orbitPt(getScenePos(hm3dy, i, true, peakAnchor)),
                            orbitPt(getScenePos(hm3dy, j, true, peakAnchor)),
                            1.2, hmBoneMat
                        );
                        if (bone) heatmapGroup.add(bone);
                    });
                }
            }
        }

        // HRnet "Z-smooth" sub-stage 3D (saturated orange).
        if (showHRnetZSM3D && trialData.hrnet_zsmooth_3d) {
            const hm3dz = trialData.hrnet_zsmooth_3d[fn];
            if (hm3dz) {
                const hmMat = new THREE.MeshPhongMaterial({ color: 0xffa040, emissive: 0x803010 });
                const hmBoneMat = new THREE.MeshPhongMaterial({ color: 0xffa040, emissive: 0x602008 });
                const validPt = (p) => p && !Number.isNaN(p[0]) && !Number.isNaN(p[1]) && !Number.isNaN(p[2]);
                const peakAnchor = _hrnetPeakAnchorFrame('zsmooth');
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !validPt(hm3dz[j])) continue;
                    const sphere = new THREE.Mesh(sphereGeom, hmMat);
                    sphere.position.copy(orbitPt(getScenePos(hm3dz, j, true, peakAnchor)));
                    heatmapGroup.add(sphere);
                }
                if (showHeatmapSkel && trialData.skeleton) {
                    trialData.skeleton.forEach(([i, j]) => {
                        if (!isBoneVisible(i, j) || !validPt(hm3dz[i]) || !validPt(hm3dz[j])) return;
                        const bone = makeBone(
                            orbitPt(getScenePos(hm3dz, i, true, peakAnchor)),
                            orbitPt(getScenePos(hm3dz, j, true, peakAnchor)),
                            1.2, hmBoneMat
                        );
                        if (bone) heatmapGroup.add(bone);
                    });
                }
            }
        }

        // HRnet "Stereo-Hungarian" sub-stage 3D (coral).
        if (showStereoHun3D && trialData.hrnet_hungarian_3d) {
            const hm3dh = trialData.hrnet_hungarian_3d[fn];
            if (hm3dh) {
                const hmMat = new THREE.MeshPhongMaterial({ color: 0xff8a65, emissive: 0x803020 });
                const hmBoneMat = new THREE.MeshPhongMaterial({ color: 0xff8a65, emissive: 0x602218 });
                const validPt = (p) => p && !Number.isNaN(p[0]) && !Number.isNaN(p[1]) && !Number.isNaN(p[2]);
                const peakAnchor = _hrnetPeakAnchorFrame('hungarian');
                for (let j = 0; j < 21; j++) {
                    if (!isJointVisible(j) || !validPt(hm3dh[j])) continue;
                    const sphere = new THREE.Mesh(sphereGeom, hmMat);
                    sphere.position.copy(orbitPt(getScenePos(hm3dh, j, true, peakAnchor)));
                    heatmapGroup.add(sphere);
                }
                if (showHeatmapSkel && trialData.skeleton) {
                    trialData.skeleton.forEach(([i, j]) => {
                        if (!isBoneVisible(i, j) || !validPt(hm3dh[i]) || !validPt(hm3dh[j])) return;
                        const bone = makeBone(
                            orbitPt(getScenePos(hm3dh, i, true, peakAnchor)),
                            orbitPt(getScenePos(hm3dh, j, true, peakAnchor)),
                            1.2, hmBoneMat
                        );
                        if (bone) heatmapGroup.add(bone);
                    });
                }
            }
        }

        // Anatomical-offset arrows in 3D — drawn for every displayed
        // Skeleton-v3 stage when the "Offsets" toggle is on.  Tail is
        // anchored to the same 2D-pixel-projected sphere position as
        // the stage's joint sphere (matches calibration y-bias compensation
        // applied in 3D rendering); head adds the offset in scene space.
        if (showOffsetArrows && _stages3D.size > 0) {
            // Arrows show the PREDICTED HRnet peak position from the
            // median (along, flex, abd) systematic-offset model.  Any
            // misalignment between an arrow tip and the actual Peak Select
            // 3D sphere is meaningful — it surfaces snap-correction errors.
            let off = {
                along: trialData.hrnet_along_3d, flex: trialData.hrnet_flex_3d,
                abd:   trialData.hrnet_abd_3d,   child: trialData.hrnet_child_3d,
            };
            const _allZero = a => a && Array.from(a).every(v => Math.abs(v) < 1e-6);
            const _allMinus = a => a && Array.from(a).every(v => v < 0);
            const degenerate3d = !off.along || !off.flex || !off.abd
                              || _allZero(off.along) || _allZero(off.flex)
                              || (off.child && _allMinus(off.child));
            if (degenerate3d) off = _computeHRnetOffsets3DClient();
            if (off && off.along) {
                const arrMat  = new THREE.MeshPhongMaterial({ color: 0xff6600, emissive: 0x803000 });
                const headMat = new THREE.MeshPhongMaterial({ color: 0xff6600, emissive: 0x803000 });
                for (const s of _skelStages3D(isLeftCam)) {
                    const pts3d = s.pts?.[fn];
                    if (!pts3d) continue;
                    const stageProj2D = s.proj2D?.[fn] || null;
                    for (let j = 0; j < 21; j++) {
                        if (!isJointVisible(j)) continue;
                        if ((off.child?.[j] ?? -1) < 0) continue;
                        const basis = _jointBasis3D(pts3d, j);
                        if (!basis) continue;
                        const a = off.along[j] || 0, fl = off.flex[j] || 0, b = off.abd[j] || 0;
                        const tail = pts3d[j];
                        if (!tail) continue;
                        const head = [
                            tail[0] + a*basis.along[0] + fl*basis.flex[0] + b*basis.abd[0],
                            tail[1] + a*basis.along[1] + fl*basis.flex[1] + b*basis.abd[1],
                            tail[2] + a*basis.along[2] + fl*basis.flex[2] + b*basis.abd[2],
                        ];
                        const offsetScene = new THREE.Vector3().subVectors(toScene(head), toScene(tail));
                        const pA = orbitPt(getScenePos(pts3d, j, true, stageProj2D));
                        const pB = pA.clone().add(offsetScene);
                        const shaft = makeBone(pA, pB, 0.6, arrMat);
                        if (shaft) heatmapGroup.add(shaft);
                        const dir = new THREE.Vector3().subVectors(pB, pA);
                        const len = dir.length();
                        if (len > 0.5) {
                            const coneH = Math.min(3.0, len * 0.3);
                            const coneGeom = new THREE.ConeGeometry(1.4, coneH, 10);
                            const cone = new THREE.Mesh(coneGeom, headMat);
                            const midTip = new THREE.Vector3().copy(pB)
                                .sub(dir.clone().normalize().multiplyScalar(coneH * 0.5));
                            cone.position.copy(midTip);
                            cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
                            heatmapGroup.add(cone);
                        }
                    }
                }
            }
        }

        // MediaPipe joints (always cyan; errors are shown on the Skeleton
        // layer via the stage-picker buttons instead)
        if (showMP3D && mp3d) {
            const mpMat = new THREE.MeshPhongMaterial({ color: 0x00cccc, emissive: 0x008888 });
            const mpBoneMat = new THREE.MeshPhongMaterial({ color: 0x00cccc, emissive: 0x007777 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !mp3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, mpMat);
                sphere.position.copy(orbitPt(getScenePos(mp3d, j)));
                mpGroup.add(sphere);
            }
            if (showMPSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !mp3d[i] || !mp3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(mp3d, i)),
                        orbitPt(getScenePos(mp3d, j)),
                        1.0, mpBoneMat
                    );
                    if (bone) mpGroup.add(bone);
                });
            }
        }

        // Reverse-pass MediaPipe joints (magenta).
        if (showReverse3D && reverse3d) {
            const revMat = new THREE.MeshPhongMaterial({ color: 0xe040fb, emissive: 0x6a1b9a });
            const revBoneMat = new THREE.MeshPhongMaterial({ color: 0xe040fb, emissive: 0x4a148c });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !reverse3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, revMat);
                sphere.position.copy(orbitPt(getScenePos(reverse3d, j)));
                reverseGroup.add(sphere);
            }
            if (showReverseSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !reverse3d[i] || !reverse3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(reverse3d, i)),
                        orbitPt(getScenePos(reverse3d, j)),
                        1.0, revBoneMat
                    );
                    if (bone) reverseGroup.add(bone);
                });
            }
        }

        // Cropped-forward MediaPipe joints (green) — forward pass
        // with bbox crop, gated by its own checkbox.
        if (showCropped3D && cropped3d) {
            const crpMat = new THREE.MeshPhongMaterial({ color: 0x7cb342, emissive: 0x33691e });
            const crpBoneMat = new THREE.MeshPhongMaterial({ color: 0x7cb342, emissive: 0x1b5e20 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !cropped3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, crpMat);
                sphere.position.copy(orbitPt(getScenePos(cropped3d, j)));
                croppedGroup.add(sphere);
            }
            if (showCroppedSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !cropped3d[i] || !cropped3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(cropped3d, i)),
                        orbitPt(getScenePos(cropped3d, j)),
                        1.0, crpBoneMat
                    );
                    if (bone) croppedGroup.add(bone);
                });
            }
        }

        // Static-mode MediaPipe joints (cyan) — per-frame palm
        // detector, gated by its own checkbox.
        if (showStatic3D && static3d) {
            const sttMat = new THREE.MeshPhongMaterial({ color: 0x26c6da, emissive: 0x006978 });
            const sttBoneMat = new THREE.MeshPhongMaterial({ color: 0x26c6da, emissive: 0x004d5a });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !static3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, sttMat);
                sphere.position.copy(orbitPt(getScenePos(static3d, j)));
                staticGroup.add(sphere);
            }
            if (showStaticSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !static3d[i] || !static3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(static3d, i)),
                        orbitPt(getScenePos(static3d, j)),
                        1.0, sttBoneMat
                    );
                    if (bone) staticGroup.add(bone);
                });
            }
        }

        // Combined-pass MediaPipe joints (orange) — per-camera fusion of
        // forward + reverse, gated by its own checkbox.
        if (showCombined3D && combined3d) {
            const cmbMat = new THREE.MeshPhongMaterial({ color: 0xffa726, emissive: 0xb26500 });
            const cmbBoneMat = new THREE.MeshPhongMaterial({ color: 0xffa726, emissive: 0x8a4a00 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !combined3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, cmbMat);
                sphere.position.copy(orbitPt(getScenePos(combined3d, j)));
                combinedGroup.add(sphere);
            }
            if (showCombinedSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !combined3d[i] || !combined3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(combined3d, i)),
                        orbitPt(getScenePos(combined3d, j)),
                        1.0, cmbBoneMat
                    );
                    if (bone) combinedGroup.add(bone);
                });
            }
        }

        // Vision joints (light blue)
        if (showVision3D && vision3d) {
            const vMat = new THREE.MeshPhongMaterial({ color: 0x2196f3, emissive: 0x0d47a1 });
            const vBoneMat = new THREE.MeshPhongMaterial({ color: 0x2196f3, emissive: 0x0a3680 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !vision3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, vMat);
                sphere.position.copy(orbitPt(getScenePos(vision3d, j)));
                visionGroup.add(sphere);
            }
            if (showVisionSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !vision3d[i] || !vision3d[j]) return;
                    const bone = makeBone(
                        orbitPt(getScenePos(vision3d, i)),
                        orbitPt(getScenePos(vision3d, j)),
                        1.0, vBoneMat
                    );
                    if (bone) visionGroup.add(bone);
                });
            }
        }

        // Previous fit 3D (purple) -- pick the active stage's array.
        const _prev3DArr = (() => {
            if (!prevFitData) return null;
            if (_prevStage3D === 'final') return prevFitData.joints_3d;
            return prevFitData[`joints_3d_${_prevStage3D}`]
                   || prevFitData.joints_3d;
        })();
        const prev3d = _prev3DArr?.[fn];
        if (showPrev3D && prev3d) {
            const prevMat = new THREE.MeshPhongMaterial({ color: 0xb35b00, emissive: 0x502800 });
            const prevBoneMat = new THREE.MeshPhongMaterial({ color: 0xb35b00, emissive: 0x4a2500 });
            for (let j = 0; j < 21; j++) {
                if (!isJointVisible(j) || !prev3d[j]) continue;
                const sphere = new THREE.Mesh(sphereGeom, prevMat);
                sphere.position.copy(orbitPt(toScene(prev3d[j])));
                poseGroup.add(sphere);  // reuse poseGroup for prev fit
            }
            if (showPrevSkel && trialData.skeleton) {
                trialData.skeleton.forEach(([i, j]) => {
                    if (!isBoneVisible(i, j) || !prev3d[i] || !prev3d[j]) return;
                    const bone = makeBone(orbitPt(toScene(prev3d[i])), orbitPt(toScene(prev3d[j])), 1.0, prevBoneMat);
                    if (bone) poseGroup.add(bone);
                });
            }
        }

        // DLC thumb/index 3D — use 2D pixel + depth to align with 2D dots
        const dlc3dThumb = trialData.dlc_3d_thumb?.[fn];
        const dlc3dIndex = trialData.dlc_3d_index?.[fn];
        if (showDLC3D && (dlc3dThumb || dlc3dIndex)) {
            const dlcThumbMat = new THREE.MeshPhongMaterial({ color: 0xff4444, emissive: 0xaa2222 });
            const dlcIndexMat = new THREE.MeshPhongMaterial({ color: 0x444444, emissive: 0x222222 });
            const thumbKey = isLeftCam ? 'dlc_thumb_OS' : 'dlc_thumb_OD';
            const indexKey = isLeftCam ? 'dlc_index_OS' : 'dlc_index_OD';
            // Helper: unproject 2D DLC pixel + triangulated depth → scene pos
            const dlcScenePos = (pt2d, pt3d) => {
                if (!pt2d || !pt3d || !K_cam) return toScene(pt3d);
                const cfx = K_cam[0][0], cfy = K_cam[1][1];
                const ccx = K_cam[0][2], ccy = K_cam[1][2];
                const u = pt2d[0], v = pt2d[1];
                let X = pt3d[0], Y = pt3d[1], Z = pt3d[2];
                let camZ;
                if (!isLeftCam && R_cam && T_cam) {
                    camZ = R_cam[2][0]*X + R_cam[2][1]*Y + R_cam[2][2]*Z + T_cam[2];
                } else {
                    camZ = Z;
                }
                if (camZ <= 0) return toScene(pt3d);
                const camX = (u - ccx) * camZ / cfx;
                const camY = (v - ccy) * camZ / cfy;
                let wX, wY, wZ;
                if (!isLeftCam && R_cam && T_cam) {
                    const dx = camX - T_cam[0], dy = camY - T_cam[1], dz = camZ - T_cam[2];
                    wX = R_cam[0][0]*dx + R_cam[1][0]*dy + R_cam[2][0]*dz;
                    wY = R_cam[0][1]*dx + R_cam[1][1]*dy + R_cam[2][1]*dz;
                    wZ = R_cam[0][2]*dx + R_cam[1][2]*dy + R_cam[2][2]*dz;
                } else {
                    wX = camX; wY = camY; wZ = camZ;
                }
                return new THREE.Vector3(wX, -wY, -wZ);
            };
            if (dlc3dThumb && isJointVisible(4)) {
                const sphere = new THREE.Mesh(sphereGeom, dlcThumbMat);
                sphere.position.copy(orbitPt(dlcScenePos(trialData[thumbKey]?.[fn], dlc3dThumb)));
                dlcGroup.add(sphere);
            }
            if (dlc3dIndex && isJointVisible(8)) {
                const sphere = new THREE.Mesh(sphereGeom, dlcIndexMat);
                sphere.position.copy(orbitPt(dlcScenePos(trialData[indexKey]?.[fn], dlc3dIndex)));
                dlcGroup.add(sphere);
            }
        }

        // ── Pose 3D arm chain (yellow) ──
        const pose3d = trialData.pose_arm_3d?.[fn];
        if (showPose3D && pose3d) {
            const poseMat = new THREE.MeshPhongMaterial({ color: 0xffeb3b, emissive: 0xaa9900 });
            const poseBoneMat = new THREE.MeshPhongMaterial({ color: 0xffeb3b, emissive: 0x887700 });
            // Pose arm indices in sliced array:
            // Left body: shoulder=0, elbow=2, wrist=4, pinky=6, index=8, thumb=10
            // Right body: shoulder=1, elbow=3, wrist=5, pinky=7, index=9, thumb=11
            const side = trialData.pose_side;
            const sIdx = side === 'left' ? [0,2,4,6,8,10] : [1,3,5,7,9,11];
            const POSE_BONES = [[0,1],[1,2],[2,3],[2,4],[2,5]]; // shoulder→elbow→wrist→pinky/index/thumb
            const posePts = sIdx.map(i => pose3d[i] ? orbitPt(toScene(pose3d[i])) : null);
            // Spheres
            for (const pt of posePts) {
                if (!pt) continue;
                const s = new THREE.Mesh(sphereGeom, poseMat);
                s.position.copy(pt);
                poseGroup.add(s);
            }
            // Bones
            for (const [a, b] of POSE_BONES) {
                if (posePts[a] && posePts[b]) {
                    const bone = makeBone(posePts[a], posePts[b], 1.0, poseBoneMat);
                    if (bone) poseGroup.add(bone);
                }
            }
        }

        // ── Angle arcs ───────────────────────────────────────────────────────
        const hasAnyArc = plotJointStates.size > 0 || SPREAD_NAMES.some(n => selectedMetrics.has(n));
        if (hasAnyArc) {
            // Use the highest-priority visible model for joint positions & angles
            let pts3d = null, anglesData = null, isManoSrc = false;
            if (showSkelV2_3D && v2_3d)        { pts3d = v2_3d;     anglesData = trialData.angles_skel_v2; isManoSrc = true; arcSrcKey = 'skel_v2'; }
            else if (showMano3D && skel3d)     { pts3d = skel3d;    anglesData = trialData.angles_mano;    isManoSrc = true; arcSrcKey = 'skeleton'; }
            else if (showMP3D && mp3d)         { pts3d = mp3d;      anglesData = trialData.angles_mp;      arcSrcKey = 'mp'; }
            else if (showVision3D && vision3d) { pts3d = vision3d;  anglesData = trialData.angles_vision;  arcSrcKey = 'vision'; }

            if (pts3d) {
                const gsp = j => pts3d[j] ? orbitPt(getScenePos(pts3d, j, isManoSrc)) : null;

                // ── Joint flex/abd arcs ───────────────────────────────────────
                if (anglesData && plotJointStates.size > 0) {
                    const flexOpts = trialData?.flex_angle_options || [];

                    // Palm normal: average of 4 cross-product estimates (robust to single-joint depth error)
                    const palmW  = gsp(0), palmI = gsp(5), palmM = gsp(9), palmR = gsp(13), palmP = gsp(17);
                    if (!palmW || !palmI || !palmM || !palmR || !palmP) {
                        // Palm joints missing on this frame — skip all arcs
                    } else {
                    const eIdx = palmI.clone().sub(palmW), eMid = palmM.clone().sub(palmW);
                    const eRng = palmR.clone().sub(palmW), ePnk = palmP.clone().sub(palmW);
                    const nIp = new THREE.Vector3().crossVectors(eIdx, ePnk).normalize();
                    const nIm = new THREE.Vector3().crossVectors(eIdx, eMid).normalize();
                    const nMr = new THREE.Vector3().crossVectors(eMid, eRng).normalize();
                    const nRp = new THREE.Vector3().crossVectors(eRng, ePnk).normalize();
                    if (nIm.dot(nIp) < 0) nIm.negate();
                    if (nMr.dot(nIp) < 0) nMr.negate();
                    if (nRp.dot(nIp) < 0) nRp.negate();
                    const palmN = nIp.clone().add(nIm).add(nMr).add(nRp).normalize();
                    const pinkyDir = ePnk.clone().normalize();
                    const thumbRef = pinkyDir;

                    // Flex-propagated dorsal axis: for PIP/DIP, rotate the parent's dorsal
                    // axis by the parent flex angle, using the dorsal component of b_in_child
                    // as sin(θ) — invariant to abduction.
                    // sin(θ) = -dot(bIn_child, da_parent), cos(θ) = sqrt(1-sin²)
                    // ref_child = sin(θ)*bIn_parent + cos(θ)*da_parent  (Rodrigues)
                    const jointDorsal = {};
                    const jointState3D = {};  // j -> {da, fa, bIn}
                    const THUMB_JOINTS_3D = new Set([1, 2, 3]);
                    const FLEX_CHAINS_3D = [
                        [true,  [0, 1, 2, 3]],
                        [false, [0, 5, 6, 7]],
                        [false, [0, 9, 10, 11]],
                        [false, [0, 13, 14, 15]],
                        [false, [0, 17, 18, 19]],
                    ];
                    // MCP local dorsal refs: cross product of adjacent inter-MCP segments,
                    // sign-aligned opposite to palmN (palmar side), matching _compute_angles.
                    const MCP_NBR = {5: [1,9], 9: [5,13], 13: [9,17]};
                    const mcpLocalRef = {};
                    for (const [mj, [na, nb]] of Object.entries(MCP_NBR)) {
                        const mji = parseInt(mj);
                        if (!pts3d[mji] || !pts3d[na] || !pts3d[nb]) continue;
                        const va = gsp(na).clone().sub(gsp(mji)).normalize();
                        const vb = gsp(nb).clone().sub(gsp(mji)).normalize();
                        const ln = new THREE.Vector3().crossVectors(va, vb).normalize();
                        if (ln.dot(palmN) > 0) ln.negate();  // align palmar (opposite to palmN)
                        mcpLocalRef[mji] = ln;
                    }
                    // Pinky MCP: one neighbor + metacarpal
                    if (pts3d[17] && pts3d[13] && pts3d[0]) {
                        const vaP = gsp(13).clone().sub(gsp(17)).normalize();
                        const vbP = gsp(0).clone().sub(gsp(17)).normalize();
                        const lnP = new THREE.Vector3().crossVectors(vaP, vbP).normalize();
                        if (lnP.dot(palmN) > 0) lnP.negate();
                        mcpLocalRef[17] = lnP;
                    }

                    for (const [isThumb, chain] of FLEX_CHAINS_3D) {
                        const rootRef = isThumb ? thumbRef : palmN;
                        for (let ci = 0; ci < chain.length - 1; ci++) {
                            const p = chain[ci], j = chain[ci + 1];
                            if (!pts3d[p] || !pts3d[j]) continue;
                            const bIn = gsp(j).clone().sub(gsp(p)).normalize();
                            let ref;
                            if (p === 0 && mcpLocalRef[j]) {
                                ref = mcpLocalRef[j].clone();
                            } else if (p === 0) {
                                ref = rootRef.clone();
                            } else if (jointState3D[p]) {
                                const ps = jointState3D[p];
                                const sinT = Math.max(-1, Math.min(1, -bIn.dot(ps.da)));
                                const cosT = Math.sqrt(Math.max(0, 1 - sinT * sinT));
                                // ref = cos(θ)*da_parent + sin(θ)*bIn_parent
                                ref = ps.da.clone().multiplyScalar(cosT).addScaledVector(ps.bIn, sinT);
                            } else {
                                ref = rootRef.clone();
                            }
                            const da = ref.clone().addScaledVector(bIn, -ref.dot(bIn)).normalize();
                            if (da.lengthSq() < 1e-8) continue;
                            const fa = new THREE.Vector3().crossVectors(da, bIn).normalize();
                            jointState3D[j] = { da: da.clone(), fa, bIn: bIn.clone() };
                            jointDorsal[j] = da;
                        }
                    }

                    for (const [j, state] of plotJointStates) {
                        const fm = flexOpts.find(f => f.joint === j);
                        if (!fm) continue;

                        // Wrist (parent === -1): use elbow_3d as parent.
                        let pPos, jPos, cPos, dorsalAxis;
                        if (fm.parent === -1) {
                            const elb = trialData?.elbow_3d?.[fn];
                            if (!elb || elb.some(v => v === null || isNaN(v))) continue;
                            if (!pts3d[fm.joint] || !pts3d[fm.child]) continue;
                            pPos = orbitPt(toScene(elb));
                            jPos = gsp(fm.joint);
                            cPos = gsp(fm.child);
                            // Dorsal: palm normal projected ⊥ (jPos - pPos)
                            const bInTmp = jPos.clone().sub(pPos).normalize();
                            const proj = palmN.clone().addScaledVector(bInTmp, -palmN.dot(bInTmp));
                            if (proj.lengthSq() < 1e-8) continue;
                            dorsalAxis = proj.normalize();
                        } else {
                            if (!pts3d[fm.parent] || !pts3d[fm.joint] || !pts3d[fm.child]) continue;
                            dorsalAxis = jointDorsal[fm.joint];
                            if (!dorsalAxis) continue;
                            pPos = gsp(fm.parent);
                            jPos = gsp(fm.joint);
                            cPos = gsp(fm.child);
                        }

                        const bIn = jPos.clone().sub(pPos).normalize();
                        const flexAxis = new THREE.Vector3().crossVectors(dorsalAxis, bIn).normalize();

                        // Single radius for both arcs
                        const boneLen = Math.min(jPos.distanceTo(pPos), cPos.distanceTo(jPos));
                        const r = boneLen * 0.55;

                        const flexName = fm.name;

                        // Flex angle (always needed — positions the abd arc even in 'abd'-only mode)
                        const flexDeg = anglesData[flexName]?.[fn] ?? 0;
                        const flexRad = THREE.MathUtils.degToRad(flexDeg);

                        // bFlexEnd: direction at the end of the flex arc (abd arc starts here)
                        // Flex arc sweeps from -bIn toward +dorsalAxis by (180+f)°.
                        // Endpoint = cos(180+f)*(-bIn) + sin(180+f)*dorsalAxis
                        //          = cos(f)*bIn - sin(f)*dorsalAxis
                        const bFlexEnd = new THREE.Vector3()
                            .addScaledVector(bIn, Math.cos(flexRad))
                            .addScaledVector(dorsalAxis, -Math.sin(flexRad))
                            .normalize();

                        // Flex arc: sweeps toward dorsalAxis (now palmar) from -bIn
                        if ((state.mode === 'flex' || state.mode === 'both') && state.colorFlex) {
                            const arcDeg = 180 + flexDeg;
                            if (arcDeg >= 1 && arcDeg <= 359) {
                                const arc = makeAngleArc(jPos, bIn.clone().negate(), dorsalAxis.clone(), arcDeg, r, state.colorFlex);
                                if (arc) angleArcGroup.add(arc);
                            }
                        }

                        // Abd arc: perpendicular to flex plane
                        if ((state.mode === 'abd' || state.mode === 'both') && state.colorAbd) {
                            const abdName = flexName.replace('Flex:', 'Abd:');
                            const abdDeg = anglesData[abdName]?.[fn] ?? 0;
                            if (Math.abs(abdDeg) >= 0.5) {
                                const arc = makeAngleArc(jPos, bFlexEnd, flexAxis.clone(), abdDeg, r, state.colorAbd, false);
                                if (arc) angleArcGroup.add(arc);
                            }
                        }
                    }
                    } // end else (palm joints present)
                }

                // ── Spread arcs at wrist ──────────────────────────────────────
                const spreadsData = arcSrcKey ? trialData[`spreads_${arcSrcKey}`] : null;
                const spreadMcpPairs = [[2, 5], [5, 9], [9, 13], [13, 17]];
                if (spreadsData && pts3d[0]) {
                    const wristPos = gsp(0);
                    for (let i = 0; i < SPREAD_NAMES.length; i++) {
                        const name = SPREAD_NAMES[i];
                        if (!selectedMetrics.has(name)) continue;
                        const color = selectedMetrics.get(name);
                        const [j1, j2] = spreadMcpPairs[i];
                        if (!pts3d[j1] || !pts3d[j2]) continue;

                        const mcp1Pos = gsp(j1);
                        const mcp2Pos = gsp(j2);
                        const dir1 = mcp1Pos.clone().sub(wristPos).normalize();
                        const dir2 = mcp2Pos.clone().sub(wristPos).normalize();

                        // Sweep axis: component of dir2 perpendicular to dir1
                        const dot12 = dir2.dot(dir1);
                        const sweepDir = dir2.clone().addScaledVector(dir1, -dot12);
                        if (sweepDir.lengthSq() < 1e-8) continue;
                        sweepDir.normalize();

                        const spreadDeg = spreadsData[name]?.[fn] ?? 0;
                        if (spreadDeg < 0.5) continue;

                        const r = wristPos.distanceTo(mcp1Pos) * 0.5;
                        const arc = makeAngleArc(wristPos, dir1, sweepDir, spreadDeg, r, color, false);
                        if (arc) angleArcGroup.add(arc);
                    }
                }
            }
        }

        manoGroup.visible = showMano3D;
        mpGroup.visible = showMP3D;
        croppedGroup.visible = showCropped3D;
        reverseGroup.visible = showReverse3D;
        staticGroup.visible = showStatic3D;
        combinedGroup.visible = showCombined3D;
        visionGroup.visible = showVision3D;
        dlcGroup.visible = showDLC3D;

        // Apply per-model translations
        manoGroup.position.copy(modelTranslations.skeleton);
        skelV2Group.position.copy(modelTranslations.skel_v2);
        mpGroup.position.copy(modelTranslations.mp);
        reverseGroup.position.copy(modelTranslations.reverse || modelTranslations.mp);
        visionGroup.position.copy(modelTranslations.vision);
        dlcGroup.position.copy(modelTranslations.dlc);
        // Arcs follow the source model's translation
        if (arcSrcKey) angleArcGroup.position.copy(modelTranslations[arcSrcKey]);
        else angleArcGroup.position.set(0, 0, 0);

        // Render 3D scene — applySnapProjection builds a custom projection
        // from calibration; if unavailable, fall back to a simple render.
        if (trialData?.calib && canvas && vidW > 0) {
            applySnapProjection();
        } else if (renderer) {
            renderer.render(scene, camera3d);
        }
    }

    // ── Snap-to-camera ──────────────────────────────────────
    // Positions the camera from calibration, sets the custom projection,
    // and resets any orbit rotation.  The camera never moves after this —
    // orbit drags rotate the scene content instead (see update3D / orbitPt).

    function snapToCamera(skipRender) {
        if (!trialData?.calib || !canvas || vidW === 0) return;

        // Reset orbit rotation and per-model translations
        orbitQuat.identity();
        modelTranslations.skeleton.set(0, 0, 0);
        modelTranslations.skel_v2.set(0, 0, 0);
        modelTranslations.mp.set(0, 0, 0);
        modelTranslations.vision.set(0, 0, 0);
        modelTranslations.dlc.set(0, 0, 0);
        _projCorrComputed = false; // recompute correction for new camera/trial

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
        if (!skipRender) update3D();
    }

    /** Recompute and apply the custom projection matrix for the current
     *  canvas size & zoom/pan state.  Called after snap, resize, and zoom/pan. */
    function applySnapProjection() {
        if (!trialData?.calib || !canvas || vidW === 0) return;

        const isStereo = cameraMode === 'stereo';
        const isLeft = currentSide === cameraNames[0];
        const K = isLeft ? trialData.calib.K_L : trialData.calib.K_R;
        const sw = isStereo ? (isLeft ? midline : vidW - midline) : vidW;

        const fx = K[0][0], fy = K[1][1];
        let cx = K[0][2], cy = K[1][2];
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

        // Measure projection error ONCE (at identity orbit) and cache.
        // Only recompute after snap resets _projCorrComputed.
        // Clear any leftover CSS transform
        if (renderer?.domElement && renderer.domElement.style.transform) {
            renderer.domElement.style.transform = '';
        }
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
            init().catch(e => console.error('Skeleton viewer init failed:', e));
        });
    } else {
        init().catch(e => console.error('Skeleton viewer init failed:', e));
    }

    function _updateHandDiagramColor() {
        const color = (showSkelV2_2D || showSkelV2_3D) ? '#ff9800'
                    : (showLegacyV2_2D || showLegacyV2_3D) ? '#e040fb'
                    : (showMano2D || showMano3D) ? 'lime'
                    : '#60c0e0';
        document.querySelectorAll('#handDiagramLabels .bone').forEach(el => el.setAttribute('stroke', color));
        document.querySelectorAll('#handDiagramLabels .joint:not([data-joint="0"])').forEach(el => el.setAttribute('fill', color));
    }

    function updateFitStatus() {
        const statusEl = $('manoFitStatus');
        const btn = $('runStage1Btn');
        if (!statusEl || !btn) return;

        const hasFit = trialData && trialData.has_skeleton_v1;
        const hasV2  = trialData && trialData.has_skel_v2;
        const hasAnyFit = hasFit || hasV2;
        const hasMP = trialData && trialData.has_mp;
        const hasMP3D = trialData && trialData.has_mp_3d;
        const hasVision = trialData && trialData.has_vision;
        const hasVision3D = trialData && trialData.has_vision_3d;
        const hasDLC = trialData && trialData.has_dlc;

        // Status label only -- model selection is NOT auto-picked.
        // Initial-load defaults are "all unchecked" and any user-set
        // state persists across reloads (see _saveCheckboxes /
        // _restoreCheckboxes below).
        const hasLegacy = trialData && trialData.has_skel_legacy;
        const hasHRnet3D = !!(trialData && trialData.hrnet_peaks_3d);
        statusEl.innerHTML = hasV2 ? ''
                          : hasFit ? '<span style="color:var(--green);">v1 fit available</span>'
                          : hasLegacy || hasMP3D || hasHRnet3D || hasVision3D ? ''
                          : 'No fit available';

        _setLayerAvail('showSkelV2_2D', hasV2);
        _setLayerAvail('showSkelV2_3D', hasV2);
        _setLayerAvail('showSkelV2Skel', hasV2);

        _setLayerAvail('showLegacyV2_2D', hasLegacy);
        _setLayerAvail('showLegacyV2_3D', hasLegacy);
        _setLayerAvail('showLegacyV2Skel', hasLegacy);

        // MP errors: initialise from saved file if present
        if (trialData?.mp_errors) {
            mpErrorMatrix = trialData.mp_errors;
        } else if (!_anyMPWeightActive()) {
            mpErrorMatrix = null;
        }
        skelErrorMatrices = {};
        // Clear any corrections from the previous trial (they're trial-specific)
        mpCorrectedL = null;
        mpCorrectedR = null;
        _updateMPErrorAvailability();

        // Stage checkboxes are NOT auto-picked here -- they restore
        // from localStorage (or stay unchecked on first visit) and
        // persist across page reloads.  Just re-sync the derived
        // _errorStages + run the MP-error recompute if needed.
        _syncErrorStages();
        showSkelErrors = _errorStages.size > 0;
        if (_errorStages.size > 0) _scheduleMPErrorRecompute();

        // MediaPipe
        _setLayerAvail('showMP2D', hasMP);
        _setLayerAvail('showMP3D', hasMP3D);
        if (!hasMP) {
            if ($('showMP2D')) { $('showMP2D').checked = false; showMP2D = false; }
        }
        if (!hasMP3D) {
            if ($('showMP3D')) { $('showMP3D').checked = false; showMP3D = false; }
        }
        showMP = showMP2D || showMP3D;

        // Stereo — dim + reset when no saved alignment exists.  Trial
        // load also clears the per-joint bbox selection so a stale
        // joint index from the previous trial doesn't paint a phantom
        // box at the same index here.
        const hasStereo = !!(trialData && trialData.has_stereo);
        _setLayerAvail('showStereo2D', hasStereo);
        if (!hasStereo && $('showStereo2D')) {
            $('showStereo2D').checked = false;
            showStereo2D = false;
        }
        stereoSelectedJoint = null;
        const _scw = $('stereoConfWrap');
        if (_scw) _scw.style.display = (showStereo2D && hasStereo) ? 'flex' : 'none';
        // Stereo (outline) + Outline layer availability.
        const hasStereoOutline = !!(trialData && trialData.has_stereo_outline);
        _setLayerAvail('showStereoOutline2D', hasStereoOutline);
        if (!hasStereoOutline && $('showStereoOutline2D')) {
            $('showStereoOutline2D').checked = false;
            showStereoOutline2D = false;
        }
        const _socw = $('stereoOutlineConfWrap');
        if (_socw) _socw.style.display = (showStereoOutline2D && hasStereoOutline) ? 'flex' : 'none';
        const hasStereoHybrid = !!(trialData && trialData.has_stereo_hybrid);
        _setLayerAvail('showStereoHybrid2D', hasStereoHybrid);
        if (!hasStereoHybrid && $('showStereoHybrid2D')) {
            $('showStereoHybrid2D').checked = false;
            showStereoHybrid2D = false;
        }
        const _shcw = $('stereoHybridConfWrap');
        if (_shcw) _shcw.style.display = (showStereoHybrid2D && hasStereoHybrid) ? 'flex' : 'none';
        const hasOutlines = !!(trialData && trialData.has_outlines);
        _setLayerAvail('showOutline2D', hasOutlines);
        if (!hasOutlines && $('showOutline2D')) {
            $('showOutline2D').checked = false;
            showOutline2D = false;
        }

        // Vision
        _setLayerAvail('showVision2D', hasVision);
        _setLayerAvail('showVision3D', hasVision3D);
        _setLayerAvail('showVisionSkel', hasVision);
        if (!hasVision) {
            if ($('showVision2D')) { $('showVision2D').checked = false; showVision2D = false; }
            if ($('showVision3D')) { $('showVision3D').checked = false; showVision3D = false; }
        }

        // DLC
        _setLayerAvail('showDLC', hasDLC);
        if (!hasDLC) {
            if ($('showDLC')) { $('showDLC').checked = false; showDLC = false; }
        }

        _syncAllSkelCheckboxes();
        // Sync heatmap UI to whatever the new trial supports (hides the
        // threshold row entirely when this trial has no HRnet heatmaps,
        // dims it when no joint is active).
        _refreshHeatmapState();

        // Dim + collapse the parent model toggles when the underlying
        // model hasn't been run.  Without this, clicking "Skeleton v3"
        // expands an empty stage list on subjects that have never had
        // v3 run, and the HRnet section pretends to be interactive
        // even when no heatmaps exist.
        _setModelToggleAvail('v3', hasV2);
        _setModelToggleAvail('heatmap', hasHRnet3D);
    }

    function _setLayerAvail(id, available) {
        const el = $(id);
        if (!el) return;
        el.disabled = !available;
        el.style.opacity = available ? '' : '0.15';
        // Also dim the model name span (previous sibling in the grid that's a span with text)
        let prev = el.previousElementSibling;
        while (prev && prev.tagName === 'INPUT') prev = prev.previousElementSibling;
        if (prev && prev.tagName === 'SPAN' && prev.textContent.trim()) {
            prev.style.opacity = available ? '' : '0.15';
        }
    }

    // Dim / collapse a multi-stage model's parent toggle (e.g. "Skeleton v3",
    // "HRnet") when the underlying model hasn't been run on this trial.
    // When ``available`` is false:
    //   - the parent label is dimmed
    //   - clicking it doesn't expand the sub-stage rows (pointer-events: none)
    //   - if it was expanded, force it collapsed
    function _setModelToggleAvail(group, available) {
        const el = document.querySelector(`.model-toggle[data-group="${group}"]`);
        if (!el) return;
        el.style.opacity = available ? '' : '0.3';
        el.style.pointerEvents = available ? '' : 'none';
        if (!available) {
            if (group === 'v3' && _v3Expanded) _setV3Expanded(false);
            if (group === 'heatmap' && _heatmapExpanded) _setHeatmapExpanded(false);
        }
    }

    // ── 2D Pose Detection with bounding box ───────────────────
    let _detectModel = null; // current model endpoint (e.g. 'run-mediapipe')
    let _detectLabel = '';
    let bboxEditMode = false;
    let bboxOS = null; // [x1, y1, x2, y2] in camera-half pixel coords
    let bboxOD = null;
    // Sidecar-recorded "previous" bbox -- the bbox that produced the
    // current mediapipe.npz / mediapipe_reverse.npz output for this
    // trial.  Drawn as a non-interactive grey reference outline while
    // MP detect mode is active so the user can see how a freshly-edited
    // bbox differs from the one baked into the existing output.
    let prevBboxOS = null;
    let prevBboxOD = null;
    let bboxDrag = null; // {handle, startMx, startMy, origBox}

    const MODEL_LABELS = {
        'run-mediapipe': 'MediaPipe', 'run-vision': 'Vision',
        'run-pose': 'Pose', 'run-hrnet': 'HRNet',
    };

    async function _loadDefaultBbox() {
        if (!subjectId || currentTrialIdx < 0) return;
        const trial = trials[currentTrialIdx];
        const modelParam = _detectModel ? `&model=${encodeURIComponent(_detectModel)}` : '';
        try {
            const data = await api(`/api/analyze/${subjectId}/hrnet/bbox?trial_idx=${trial.trial_idx}${modelParam}`);
            bboxOS = data.bbox_os;
            bboxOD = data.bbox_od;
        } catch {
            // Default to full frame
            const sw = cameraMode === 'stereo' ? midline : vidW;
            bboxOS = [0, 0, sw, vidH];
            bboxOD = [0, 0, sw, vidH];
        }
    }

    function _enterDetectMode(endpoint) {
        // Cancel previous if any
        if (_detectModel) _exitDetectMode();

        _detectModel = endpoint;
        _detectLabel = MODEL_LABELS[endpoint] || endpoint;
        bboxEditMode = true;

        // Reveal MP-specific options (Static image mode) only when the
        // active model is MediaPipe (hand joints).  All other models
        // hide it.
        const _simRow = document.getElementById('mpStaticImageRow');
        if (_simRow) _simRow.style.display = (endpoint === 'run-mediapipe') ? 'flex' : 'none';
        const _revRow = document.getElementById('mpReverseRow');
        if (_revRow) _revRow.style.display = (endpoint === 'run-mediapipe') ? 'flex' : 'none';
        const _ubRow = document.getElementById('mpUseBboxRow');
        if (_ubRow) _ubRow.style.display = (endpoint === 'run-mediapipe') ? 'flex' : 'none';

        // Disable pointer-events on the Three.js overlay so the video canvas
        // receives mouse events directly (needed for bbox drag handles).
        const threeEl = $('threejsContainer');
        if (threeEl) threeEl.style.pointerEvents = 'none';

        // Highlight the selected model button and insert action buttons below it
        document.querySelectorAll('.detect-model-row').forEach(row => {
            const btn = row.querySelector('.detect-model-btn');
            if (row.dataset.model === endpoint) {
                btn.classList.add('active');
                // Insert action buttons if not already there
                if (!row.querySelector('.detect-actions')) {
                    const actions = document.createElement('div');
                    actions.className = 'detect-actions';
                    actions.style.cssText = 'display:flex;gap:4px;margin-top:3px;';
                    actions.innerHTML = `
                        <button class="btn btn-sm btn-success detect-run-btn" style="flex:1;font-size:10px;">Run</button>
                        <button class="btn btn-sm detect-save-btn" style="font-size:10px;">Save Box</button>
                        <button class="btn btn-sm detect-cancel-btn" style="font-size:10px;">Cancel</button>
                    `;
                    row.appendChild(actions);
                    // MediaPipe-only: a second row with the Re-combine
                    // button so the user can manually rebuild
                    // mediapipe_combined.npz from the current source
                    // npzs without re-running detection.
                    if (endpoint === 'run-mediapipe') {
                        const recombineRow = document.createElement('div');
                        recombineRow.className = 'detect-recombine-row';
                        recombineRow.style.cssText = 'display:flex;gap:4px;margin-top:3px;';
                        recombineRow.innerHTML = `
                            <button class="btn btn-sm detect-recombine-btn"
                                    title="Rebuild mediapipe_combined.npz from the existing forward / cropped / reverse / static npzs for the current trial."
                                    style="flex:1;font-size:10px;">Re-combine</button>
                        `;
                        row.appendChild(recombineRow);
                        recombineRow.querySelector('.detect-recombine-btn')
                            .addEventListener('click', _recombineMpCurrentTrial);
                    }
                    actions.querySelector('.detect-run-btn').addEventListener('click', _runActiveDetection);
                    actions.querySelector('.detect-save-btn').addEventListener('click', _saveBbox);
                    actions.querySelector('.detect-cancel-btn').addEventListener('click', _exitDetectMode);
                    // Save Box only matters when MediaPipe will use the
                    // bbox — hide it otherwise, and keep it in sync as
                    // Use Bounding Box is toggled.
                    if (endpoint === 'run-mediapipe') {
                        const useBboxCb = document.getElementById('mpUseBbox');
                        const saveBtn = actions.querySelector('.detect-save-btn');
                        const _syncSaveVisible = () => {
                            saveBtn.style.display = (useBboxCb && useBboxCb.checked) ? '' : 'none';
                        };
                        _syncSaveVisible();
                        if (useBboxCb && !useBboxCb._saveVisListener) {
                            useBboxCb._saveVisListener = true;
                            useBboxCb.addEventListener('change', () => {
                                document.querySelectorAll('.detect-actions .detect-save-btn').forEach(b => {
                                    b.style.display = useBboxCb.checked ? '' : 'none';
                                });
                            });
                        }
                    }
                }
            } else {
                btn.classList.remove('active');
            }
        });
        // Sequence the two bbox sources: DB-saved bbox (mp_crop_boxes)
        // is what the user is editing -- it's the bbox the next run
        // will consume.  Sidecar bbox is informational only ("what
        // produced the CURRENT npz").  Awaiting _loadDefaultBbox
        // before consulting the sidecar avoids the race where the
        // older sidecar bbox silently masked a newly-saved Save-Box
        // value.
        (async () => {
            await _loadDefaultBbox();
            if (endpoint === 'run-mediapipe') {
                await _applyMpSidecarDefaults();
            }
            render();
        })().catch(e => console.warn('[labels] detect-mode init failed:', e));
    }

    async function _applyMpSidecarDefaults() {
        // Pulls the per-trial mediapipe params sidecar and applies the
        // non-bbox parameters (static_image_mode, use_bbox) as defaults.
        // The bbox is intentionally NOT overridden when the user already
        // has a saved per-trial bbox -- that bbox is what the next run
        // will consume, and silently swapping it back to the sidecar's
        // value would mask a deliberate Save Box.  Instead we compare
        // and surface the divergence in #detectStatus so the user
        // knows clicking Run will change the bbox the output was
        // produced from.
        if (!subjectId || currentTrialIdx < 0) return;
        const trial = trials[currentTrialIdx];
        const revCb = document.getElementById('mpReverse');
        const reverse = !!(revCb && revCb.checked);
        let data;
        try {
            data = await api(
                `/api/analyze/${subjectId}/mp-params`
                + `?trial_idx=${trial.trial_idx}`
                + `&reverse=${reverse ? 1 : 0}`,
            );
        } catch {
            return;
        }
        if (!data || data.status !== 'ok' || !data.params) return;
        const p = data.params;
        const simCb = document.getElementById('mpStaticImageMode');
        if (simCb && typeof p.static_image_mode === 'boolean') {
            simCb.checked = p.static_image_mode;
        }
        const ubCb = document.getElementById('mpUseBbox');
        if (ubCb && typeof p.use_bbox === 'boolean') {
            ubCb.checked = p.use_bbox;
        }
        // Sidecar bbox = the bbox that produced the current output.
        // Stash on prevBboxOS / prevBboxOD so render() can paint a
        // grey, non-interactive reference outline while MP detect
        // mode is active.  We do NOT overwrite the editable bbox
        // -- that's what the next Run will consume and the user has
        // already saved their preferred value via Save Box.
        prevBboxOS = (Array.isArray(p.bbox_os) && p.bbox_os.length === 4)
            ? p.bbox_os.map(Number) : null;
        prevBboxOD = (Array.isArray(p.bbox_od) && p.bbox_od.length === 4)
            ? p.bbox_od.map(Number) : null;
    }

    function _exitDetectMode() {
        _detectModel = null;
        bboxEditMode = false;
        bboxDrag = null;
        // Clear the sidecar reference outline -- it's MP-only and
        // shouldn't persist once detect mode is closed.
        prevBboxOS = null;
        prevBboxOD = null;
        // Restore pointer-events on Three.js overlay
        const threeEl = $('threejsContainer');
        if (threeEl) threeEl.style.pointerEvents = '';
        // Remove all action buttons and unhighlight (detection + fit rows)
        document.querySelectorAll('.detect-model-row').forEach(row => {
            row.querySelector('.detect-model-btn')?.classList.remove('active');
            row.querySelector('.detect-actions')?.remove();
            row.querySelector('.detect-recombine-row')?.remove();
        });
        // Hide MP-specific Static-image-mode row whenever no model is active.
        const _simRow = document.getElementById('mpStaticImageRow');
        if (_simRow) _simRow.style.display = 'none';
        const _revRow = document.getElementById('mpReverseRow');
        if (_revRow) _revRow.style.display = 'none';
        const _ubRow = document.getElementById('mpUseBboxRow');
        if (_ubRow) _ubRow.style.display = 'none';
        render();
    }

    async function _recombineMpCurrentTrial() {
        const btn = document.querySelector('.detect-recombine-btn');
        if (!btn || !subjectId || currentTrialIdx < 0) return;
        const trial = trials[currentTrialIdx];
        if (!trial) return;
        const orig = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Recombining…';
        try {
            const res = await api(`/api/analyze/${subjectId}/recombine-mp`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ trial_stem: trial.trial_stem }),
            });
            btn.textContent = res && res.ok ? 'Done' : 'Error';
        } catch (e) {
            console.warn('[labels] re-combine failed:', e);
            btn.textContent = 'Error';
        }
        setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 1200);
    }

    async function _saveBbox() {
        if (!subjectId || currentTrialIdx < 0) return;
        const trial = trials[currentTrialIdx];
        const statusEl = $('detectStatus');
        // Clear first so the user sees the message re-appear even when the
        // previous save's text is identical -- otherwise the static label
        // gives no feedback on subsequent clicks.
        if (statusEl) statusEl.textContent = 'Saving…';
        try {
            await api(`/api/analyze/${subjectId}/hrnet/bbox`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trial_idx: trial.trial_idx,
                    bbox_os: bboxOS, bbox_od: bboxOD,
                    model: _detectModel || 'default',
                }),
            });
            if (statusEl) {
                const stem = (trial.trial_stem || '').includes('_')
                    ? trial.trial_stem.split('_').slice(1).join('_')
                    : (trial.trial_stem || `trial ${trial.trial_idx}`);
                const ts = new Date().toLocaleTimeString();
                statusEl.textContent = `Saved bbox for ${stem} at ${ts}.`;
            }
        } catch (e) {
            if (statusEl) statusEl.textContent = 'Save failed: ' + e.message;
        }
    }

    async function _runActiveDetection() {
        if (!_detectModel || !subjectId) return;
        const status = $('detectStatus');
        const label = _detectLabel;
        const model = _detectModel;
        const savedBboxOS = bboxOS ? [...bboxOS] : null;
        const savedBboxOD = bboxOD ? [...bboxOD] : null;

        // Prompt for Local/Remote before running
        const runBtn = document.querySelector('.detect-run-btn');
        const target = runBtn ? await _promptExecTarget(runBtn) : 'local-cpu';
        if (!target) return;

        // Save bbox and exit detect mode before running
        await _saveBbox();
        _exitDetectMode();
        if (status) status.textContent = `Running ${label}...`;

        try {
            const trial = trials[currentTrialIdx];
            const body = { trial_idx: trial.trial_idx };
            if (savedBboxOS) body.bbox_os = savedBboxOS;
            if (savedBboxOD) body.bbox_od = savedBboxOD;
            // MediaPipe-only: opt into static_image_mode (no between-frame
            // tracker — each frame is a fresh palm-detector pass).  Slower
            // but more robust when the tracker keeps losing the hand.
            if (model === 'run-mediapipe') {
                const cb = document.getElementById('mpStaticImageMode');
                if (cb && cb.checked) body.static_image_mode = true;
                // Reverse-pass option: feed frames to MediaPipe in
                // descending temporal order so the tracker enters
                // cold-start frames already locked on.  Saved to
                // mediapipe_reverse_prelabels.npz, displayed as the
                // Reverse model row.
                const rev = document.getElementById('mpReverse');
                if (rev && rev.checked) body.reverse = true;
                // Use-bbox option: when unchecked, run on the full
                // camera-half frame instead of the saved bbox.  Default
                // true; only forward when explicitly off so older
                // backends that don't understand the flag keep working.
                const ub = document.getElementById('mpUseBbox');
                if (ub && !ub.checked) body.use_bbox = false;
            }

            const DETECT_JOB_TYPE = {
                'run-mediapipe': 'mediapipe', 'run-vision': 'vision',
                'run-pose': 'pose', 'run-hrnet': 'hrnet',
            };
            // Always submit through the queue so jobs appear in the Jobs page
            const result = await _submitViaQueue(DETECT_JOB_TYPE[model] || model, body, target);
            const jobId = result.job_id || result.queue_id;
            if (!jobId) throw new Error('No job_id returned');
            const savedSid = subjectId;
            const evtSource = new EventSource(`/api/jobs/${jobId}/stream`);
            _activeEventSources.add(evtSource);
            evtSource.onmessage = async (event) => {
                if (subjectId !== savedSid) { evtSource.close(); _activeEventSources.delete(evtSource); return; }
                const data = JSON.parse(event.data);
                if (data.status === 'running') {
                    const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                    if (status) status.textContent = `${label}... ${pct}%`;
                } else if (data.status === 'completed') {
                    evtSource.close(); _activeEventSources.delete(evtSource);
                    if (status) status.textContent = `${label} complete. Reloading...`;
                    await loadTrial(currentTrialIdx);
                    if (status) status.textContent = `${label} complete.`;
                } else if (data.status === 'failed') {
                    evtSource.close(); _activeEventSources.delete(evtSource);
                    if (status) status.textContent = `${label} failed: ${data.error_msg || 'unknown'}`;
                } else if (data.status === 'cancelled') {
                    evtSource.close(); _activeEventSources.delete(evtSource);
                    if (status) status.textContent = `${label} cancelled.`;
                }
            };
            evtSource.onerror = () => {
                evtSource.close();
                if (status) status.textContent = `${label} connection lost.`;
            };
        } catch (e) {
            if (status) status.textContent = `${label} error: ${e.message}`;
        }
    }

    // Bbox drawing overlay (called from render)
    function _drawBboxOverlay(pixelScale) {
        if (!bboxEditMode) return;
        // MediaPipe with "Use bounding box" unchecked: the next run
        // will scan the full camera-half frame, so the green editable
        // bbox + dim shading is misleading.  Hide them.  The grey
        // sidecar reference outline still draws below so the user can
        // see what bbox produced the existing output.
        const _ubCb = document.getElementById('mpUseBbox');
        const _mpNoBbox = _detectModel === 'run-mediapipe'
            && _ubCb && !_ubCb.checked;
        // ── Grey reference outline: sidecar bbox that produced the
        // current MP output.  Non-interactive; informational only.
        const prevBox = currentSide === cameraNames[0] ? prevBboxOS : prevBboxOD;
        if (prevBox && Array.isArray(prevBox) && prevBox.length === 4) {
            const [gx1, gy1, gx2, gy2] = prevBox;
            ctx.save();
            ctx.strokeStyle = 'rgba(180,180,180,0.75)';
            ctx.lineWidth = 1.5 / scale;
            ctx.setLineDash([5 / scale, 3 / scale]);
            ctx.strokeRect(gx1 * pixelScale, gy1 * pixelScale,
                           (gx2 - gx1) * pixelScale, (gy2 - gy1) * pixelScale);
            ctx.setLineDash([]);
            ctx.restore();
        }
        if (_mpNoBbox) return;        // skip the editable green bbox + shading
        const box = currentSide === cameraNames[0] ? bboxOS : bboxOD;
        if (!box) return;
        const [x1, y1, x2, y2] = box;
        const px1 = x1 * pixelScale, py1 = y1 * pixelScale;
        const px2 = x2 * pixelScale, py2 = y2 * pixelScale;
        // Dim outside
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.4)';
        const cw = canvas.width / scale, ch = canvas.height / scale;
        ctx.fillRect(0, 0, cw, py1);
        ctx.fillRect(0, py2, cw, ch - py2);
        ctx.fillRect(0, py1, px1, py2 - py1);
        ctx.fillRect(px2, py1, cw - px2, py2 - py1);
        // Border
        ctx.strokeStyle = '#0f0';
        ctx.lineWidth = 2 / scale;
        ctx.setLineDash([6 / scale, 4 / scale]);
        ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
        ctx.setLineDash([]);
        // Handles
        const hs = 6 / scale;
        ctx.fillStyle = '#0f0';
        const mx = (px1 + px2) / 2, my = (py1 + py2) / 2;
        for (const [hx, hy] of [
            [px1, py1], [px2, py1], [px1, py2], [px2, py2],
            [mx, py1], [mx, py2], [px1, my], [px2, my],
        ]) {
            ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
        }
        ctx.restore();
    }

    function _bboxHandleHitTest(canvasMx, canvasMy, pixelScale) {
        const box = currentSide === cameraNames[0] ? bboxOS : bboxOD;
        if (!box) return null;
        const [x1, y1, x2, y2] = box;
        const px1 = x1 * pixelScale, py1 = y1 * pixelScale;
        const px2 = x2 * pixelScale, py2 = y2 * pixelScale;
        const r = 10 / scale;
        const midx = (px1 + px2) / 2, midy = (py1 + py2) / 2;
        const handles = [
            ['nw', px1, py1], ['ne', px2, py1], ['sw', px1, py2], ['se', px2, py2],
            ['n', midx, py1], ['s', midx, py2], ['w', px1, midy], ['e', px2, midy],
        ];
        for (const [name, hx, hy] of handles) {
            if (Math.abs(canvasMx - hx) < r && Math.abs(canvasMy - hy) < r) return name;
        }
        if (canvasMx >= px1 && canvasMx <= px2 && canvasMy >= py1 && canvasMy <= py2) return 'move';
        return null;
    }

    function _applyBboxDrag(dx, dy, pixelScale) {
        if (!bboxDrag) return;
        const box = [...bboxDrag.origBox];
        const dpx = dx / (pixelScale * scale);
        const dpy = dy / (pixelScale * scale);
        const h = bboxDrag.handle;
        if (h === 'move') {
            box[0] += dpx; box[1] += dpy; box[2] += dpx; box[3] += dpy;
        } else {
            if (h.includes('w')) box[0] += dpx;
            if (h.includes('e')) box[2] += dpx;
            if (h.includes('n')) box[1] += dpy;
            if (h.includes('s')) box[3] += dpy;
        }
        if (box[2] - box[0] < 20) box[2] = box[0] + 20;
        if (box[3] - box[1] < 20) box[3] = box[1] + 20;
        const sideBox = box.map(v => Math.round(v));
        if (currentSide === cameraNames[0]) bboxOS = sideBox;
        else bboxOD = sideBox;
        render();
    }

    function setupDetectionButtons() {
        document.querySelectorAll('.detect-model-row').forEach(row => {
            row.querySelector('.detect-model-btn').addEventListener('click', () => {
                if (_detectModel === row.dataset.model) {
                    _exitDetectMode(); // toggle off if already selected
                } else {
                    _enterDetectMode(row.dataset.model);
                }
            });
        });
    }

    function _setSlider(id, val) {
        const el = $(id);
        if (el && val != null) { el.value = val; el.dispatchEvent(new Event('input')); }
    }

    // Restore sliders from saved fit params (called when panel opens)
    function _restoreV1Params() {
        const p = trialData?.v1_fit_params;
        if (!p) return;
        _setSlider('fitSliderReproj', p.w_reproj);
        _setSlider('fitSliderBone',   p.w_bone);
        _setSlider('fitSliderSmooth',  p.w_smooth);
        _setSlider('fitSliderAngle',   p.w_angle);
        const snap = $('fitSnapBones');
        if (snap && p.snap_bones != null) snap.checked = p.snap_bones;
    }

    function _restoreV2Params() {
        const p = trialData?.v2_fit_params;
        if (!p) return;
        _setSlider('v2SliderMediapipe',    p.w_mediapipe);
        _setSlider('v2SliderVision',       p.w_vision);
        _setSlider('v2SliderDLC',          p.w_dlc);
        _setSlider('v2SliderHRNet',        p.w_hrnet);
        if ($('v2HRNetFingertipsOnly')) $('v2HRNetFingertipsOnly').checked = !!p.hrnet_fingertips_only;
        _setSlider('v2SliderSmoothWrist',  p.w_smooth_wrist);
        _setSlider('v2SliderSmoothXY',     p.w_smooth_xy);
        _setSlider('v2SliderSmoothZ',      p.w_smooth_z);
        _setSlider('v2SliderSmoothAngles', p.w_smooth_angles);
        _setSlider('v2SliderConstraints',  p.w_constraints);

        // Restore joint constraints from the last fit (saves them as the
        // active custom constraints so the next fit uses the same values)
        const cst = trialData?.v2_fit_constraints;
        if (cst) {
            api('/api/skeleton/joint-constraints', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cst),
            }).then(() => _refreshConstraints()).catch(() => {});
        }

        // Restore error-detection + camera-attribution sliders from the
        // last Run (the correction pipeline saves these into the same
        // skeleton_v3_params.json under 'detection' / 'attribution' keys).
        const detSaved  = p.detection  || {};
        const attrSaved = p.attribution || {};
        const detMap = {
            z_jump:         'edSliderZ',
            z_outlier:      'edSliderZOut',
            y_disp:         'edSliderY',
            bone_length:    'edSliderBL',
            bone_agreement: 'edSliderBA',
            angle:          'edSliderAng',
            hrnet_mismatch: 'edSliderHR',
            reproj:         'edSliderReproj',
            confidence:     'edSliderConf',
        };
        const attrMap = {
            jump_2d:    'eaSliderJump',
            confidence: 'eaSliderConf',
            hrnet:      'eaSliderHRnet',
        };
        for (const [key, id] of Object.entries(detMap)) {
            if (key in detSaved) _setSlider(id, detSaved[key]);
        }
        for (const [key, id] of Object.entries(attrMap)) {
            if (key in attrSaved) _setSlider(id, attrSaved[key]);
        }
        // Restore Step 0 (stereo-correction) settings.
        const stSaved = p.stereo || {};
        if (typeof stSaved.mode === 'string') {
            const r = document.querySelector(`input[name="v3StereoMode"][value="${stSaved.mode}"]`);
            if (r) { r.checked = true; r.dispatchEvent(new Event('change')); }
        }
        if (typeof stSaved.mask_dilate_px === 'number') {
            const el = $('v3StereoDilateSlider');
            if (el) { el.value = stSaved.mask_dilate_px; el.dispatchEvent(new Event('input')); }
        }
        if (typeof stSaved.gauss_center_weight === 'number') {
            const el = $('v3StereoGaussSlider');
            if (el) { el.value = Math.round(100 * stSaved.gauss_center_weight); el.dispatchEvent(new Event('input')); }
        }
        if (typeof stSaved.conf === 'number') _setSlider('edSliderStereoConf', stSaved.conf);
        if (typeof stSaved.dist_px === 'number') _setSlider('edSliderStereoDist', stSaved.dist_px);
        if (typeof stSaved.occlusion_px === 'number') _setSlider('edSliderStereoOcc', stSaved.occlusion_px);
        // Newer param files also carry these under ``detection`` -- prefer
        // those if present (the slider is the source of truth for the
        // live error-recompute pipeline).
        if (typeof detSaved.stereo_conf === 'number') _setSlider('edSliderStereoConf', detSaved.stereo_conf);
        if (typeof detSaved.stereo_dist === 'number') _setSlider('edSliderStereoDist', detSaved.stereo_dist);
        if (typeof detSaved.stereo_occlusion === 'number') _setSlider('edSliderStereoOcc', detSaved.stereo_occlusion);
    }

    function resetFitDefaults() {
        const defaults = { fitSliderReproj: 1, fitSliderBone: 5, fitSliderSmooth: 1, fitSliderAngle: 2 };
        for (const [id, val] of Object.entries(defaults)) {
            _setSlider(id, val);
        }
        const snap = $('fitSnapBones');
        if (snap) snap.checked = false;
    }



    // Fitting parameter slider display updates
    function setupFitSliders() {
        const sliders = [
            // v1
            { id: 'fitSliderReproj',      display: 'fitWReproj' },
            { id: 'fitSliderBone',         display: 'fitWBone' },
            { id: 'fitSliderSmooth',       display: 'fitWSmooth' },
            { id: 'fitSliderAngle',        display: 'fitWAngle' },
            // v2
            { id: 'v2SliderMediapipe',     display: 'v2WMediapipe' },
            { id: 'v2SliderVision',        display: 'v2WVision' },
            { id: 'v2SliderDLC',           display: 'v2WDLC' },
            { id: 'v2SliderHRNet',         display: 'v2WHRNet' },
            { id: 'v2SliderSmoothWrist',   display: 'v2WSmoothWrist' },
            { id: 'v2SliderSmoothXY',      display: 'v2WSmoothXY' },
            { id: 'v2SliderSmoothZ',       display: 'v2WSmoothZ' },
            { id: 'v2SliderSmoothAngles',  display: 'v2WSmoothAngles' },
            { id: 'v2SliderConstraints',   display: 'v2WConstraints' },
        ];
        sliders.forEach(({ id, display }) => {
            const el = $(id);
            if (el) el.addEventListener('input', () => { $(display).textContent = parseFloat(el.value).toFixed(1); });
        });
    }

    async function runStage1() {
        if (!subjectId || currentTrialIdx < 0) {
            alert('Select a subject and trial first.');
            return;
        }
        const btn = $('runStage1Btn');
        const statusEl = $('fitV1Status');
        const target = 'local-cpu';

        // Collapse panel while fitting
        $('fitOptionsPanel').style.display = 'none';
        $('fitSkeletonBtn').classList.remove('active');
        btn.disabled = true;

        const trial = trials[currentTrialIdx];
        const fitParams = {
            trial_idx: trial.trial_idx,
            w_reproj: parseFloat($('fitSliderReproj')?.value ?? 1),
            w_bone: parseFloat($('fitSliderBone')?.value ?? 5),
            w_smooth: parseFloat($('fitSliderSmooth')?.value ?? 1),
            snap_bones: $('fitSnapBones')?.checked ?? false,
            w_angle: parseFloat($('fitSliderAngle')?.value ?? 2),
        };

        try {
            let result;
            if (target === 'local-cpu') {
                result = await api(`/api/skeleton/${subjectId}/fit`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...fitParams, stage: 1 }),
                });
            } else {
                result = await _submitViaQueue('skeleton_v1', fitParams, target);
            }
            statusEl.innerHTML = '<span style="color:var(--blue);">Fitting...</span>';
            const jobId = result.job_id || result.queue_id;
            if (jobId) pollFitJob(jobId);
        } catch (e) {
            statusEl.innerHTML = `<span style="color:var(--red);">Error: ${e.message}</span>`;
            btn.disabled = false;
        }
    }

    function pollFitJob(jobId) {
        const statusEl = $('fitV1Status');
        const btn = $('runStage1Btn');
        const savedSubjectId = subjectId;
        const source = new EventSource(`/api/jobs/${jobId}/stream`);
        _activeEventSources.add(source);
        source.onmessage = (event) => {
            if (subjectId !== savedSubjectId) { source.close(); _activeEventSources.delete(source); return; }
            const data = JSON.parse(event.data);
            const pct = Math.round(data.progress_pct || 0);
            statusEl.innerHTML = `<span style="color:var(--blue);">Fitting (v1)... ${pct}%</span>`;

            if (data.status === 'completed') {
                source.close(); _activeEventSources.delete(source);
                statusEl.innerHTML = '';
                btn.disabled = false;
                loadTrial(currentTrialIdx);
            } else if (data.status === 'failed') {
                source.close(); _activeEventSources.delete(source);
                statusEl.innerHTML = `<span style="color:var(--red);">Failed: ${data.error_msg || 'unknown error'}</span>`;
                btn.disabled = false;
            }
        };
        source.onerror = () => {
            source.close();
            btn.disabled = false;
        };
    }

    // ── Fit Skeleton v2 ────────────────────────────────────────────────────────

    async function runFitV2() {
        // Runs the MP correction pipeline (currently: Y-disparity winner-take-all
        // for every detected error), saves the output as skeleton_v3.npz, and
        // reloads the trial so it shows up as the Skeleton model.  Also saves
        // the current error matrix (mp_errors.npz) alongside.
        if (!subjectId || currentTrialIdx < 0) {
            alert('Select a subject and trial first.');
            return;
        }
        const btn      = $('runFitV2Btn');
        const statusEl = $('fitV2Status');
        const errStEl  = $('errorsStatus');

        btn.disabled = true;
        if (statusEl) statusEl.innerHTML = '<span style="color:var(--blue);">Running corrections…</span>';
        if (errStEl)  errStEl.textContent = 'Running corrections…';

        try {
            const result = await api(`/api/skeleton/${subjectId}/run_corrections`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trial_idx: trials[currentTrialIdx].trial_idx,
                    detection: _mpErrorWeights.detection,
                    attribution: _mpErrorWeights.attribution,
                    corrections: _mpErrorWeights.corrections || {},
                    hrnet_source: $('v2HRnetSource')?.value || 'auto',
                    // Step 0: stereo-correction config.
                    stereo_mode:         _mpErrorWeights.stereo?.mode || 'image',
                    mask_dilate_px:      _mpErrorWeights.stereo?.mask_dilate_px ?? 10,
                    gauss_center_weight: _mpErrorWeights.stereo?.gauss_center_weight ?? 0.0,
                    stereo_conf:         _mpErrorWeights.detection?.stereo_conf ?? 0.0,
                    stereo_dist_px:      _mpErrorWeights.detection?.stereo_dist ?? 0.0,
                    stereo_occlusion_px: _mpErrorWeights.detection?.stereo_occlusion ?? 0.0,
                }),
            });
            mpCorrectedL = null;
            mpCorrectedR = null;
            if (result?.job_id) {
                _pollFitV2Job(result.job_id);
            } else {
                if (statusEl) statusEl.innerHTML = '';
                await loadTrial(currentTrialIdx);
                btn.disabled = false;
            }
        } catch (e) {
            if (statusEl) statusEl.innerHTML = `<span style="color:var(--red);">Error: ${e.message}</span>`;
            if (errStEl)  errStEl.textContent = `Error: ${e.message}`;
            btn.disabled = false;
        }
    }

    function _pollFitV2Job(jobId) {
        const statusEl = $('fitV2Status');
        const btn      = $('runFitV2Btn');
        const savedSubjectId = subjectId;
        const source   = new EventSource(`/api/jobs/${jobId}/stream`);
        _activeEventSources.add(source);
        source.onmessage = (event) => {
            if (subjectId !== savedSubjectId) { source.close(); _activeEventSources.delete(source); return; }
            const data = JSON.parse(event.data);
            const pct  = Math.round(data.progress_pct || 0);
            statusEl.innerHTML = `<span style="color:var(--blue);">Fitting (v3)... ${pct}%</span>`;

            if (data.status === 'completed') {
                source.close(); _activeEventSources.delete(source);
                statusEl.innerHTML = '';
                btn.disabled = false;
                loadTrial(currentTrialIdx);
            } else if (data.status === 'failed') {
                source.close(); _activeEventSources.delete(source);
                statusEl.innerHTML = `<span style="color:var(--red);">Failed: ${data.error_msg || 'unknown error'}</span>`;
                btn.disabled = false;
            }
        };
        source.onerror = () => {
            source.close();
            btn.disabled = false;
        };
    }

    function resetFitV2Defaults() {
        const defaults = {
            v2SliderMediapipe:    10,
            v2SliderVision:       0,
            v2SliderDLC:          0,
            v2SliderHRNet:        0,
            v2SliderSmoothWrist:  1,
            v2SliderSmoothXY:     10,
            v2SliderSmoothZ:      10,
            v2SliderSmoothAngles: 10,
            v2SliderConstraints:  10,
        };
        for (const [id, val] of Object.entries(defaults)) {
            const el = $(id);
            if (el) { el.value = val; el.dispatchEvent(new Event('input')); }
        }
        // Also refresh constraints from server (clears drag overrides)
        _refreshConstraints();
    }

    // ── Fit Skeleton (legacy smoothing) ────────────────────────────────────────
    async function runFitLegacy() {
        if (!subjectId || currentTrialIdx < 0) {
            alert('Select a subject and trial first.');
            return;
        }
        const btn      = $('runFitLegacyBtn');
        const statusEl = $('fitLegacyStatus');

        $('fitLegacyPanel').style.display = 'none';
        $('fitSkeletonLegacyBtn').classList.remove('active');
        btn.disabled = true;

        const params = {
            trial_idx:            trials[currentTrialIdx].trial_idx,
            w_mediapipe:          parseFloat($('lgSliderMediapipe')?.value  ?? 10),
            w_dlc:                parseFloat($('lgSliderDLC')?.value        ?? 1),
            w_bone:               0,
            w_smooth_wrist:       parseFloat($('lgSliderSmoothWrist')?.value  ?? 1),
            w_smooth_xy:          parseFloat($('lgSliderSmoothXY')?.value     ?? 10),
            w_smooth_z:           parseFloat($('lgSliderSmoothZ')?.value      ?? 10),
            w_smooth_angles:      parseFloat($('lgSliderSmoothAngles')?.value ?? 10),
            use_angle_constraints: parseFloat($('lgSliderConstraints')?.value ?? 10) > 0,
            w_constraints:        parseFloat($('lgSliderConstraints')?.value  ?? 10),
        };

        try {
            const result = await api(`/api/skeleton/${subjectId}/fit_v2_legacy`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });
            statusEl.innerHTML = '<span style="color:var(--blue);">Fitting...</span>';
            const jobId = result.job_id || result.queue_id;
            if (jobId) _pollFitLegacyJob(jobId);
        } catch (e) {
            statusEl.innerHTML = `<span style="color:var(--red);">Error: ${e.message}</span>`;
            btn.disabled = false;
        }
    }

    function _pollFitLegacyJob(jobId) {
        const statusEl = $('fitLegacyStatus');
        const btn      = $('runFitLegacyBtn');
        const savedSubjectId = subjectId;
        const source   = new EventSource(`/api/jobs/${jobId}/stream`);
        _activeEventSources.add(source);
        source.onmessage = (event) => {
            if (subjectId !== savedSubjectId) { source.close(); _activeEventSources.delete(source); return; }
            const data = JSON.parse(event.data);
            const pct  = Math.round(data.progress_pct || 0);
            statusEl.innerHTML = `<span style="color:var(--blue);">Fitting (Skeleton v2)... ${pct}%</span>`;
            if (data.status === 'completed') {
                source.close(); _activeEventSources.delete(source);
                statusEl.innerHTML = '';
                btn.disabled = false;
                loadTrial(currentTrialIdx);
            } else if (data.status === 'failed') {
                source.close(); _activeEventSources.delete(source);
                statusEl.innerHTML = `<span style="color:var(--red);">Failed: ${data.error_msg || 'unknown error'}</span>`;
                btn.disabled = false;
            }
        };
        source.onerror = () => { source.close(); btn.disabled = false; };
    }

    function resetFitLegacyDefaults() {
        const defaults = {
            lgSliderMediapipe:    10,
            lgSliderDLC:          1,
            lgSliderSmoothWrist:  1,
            lgSliderSmoothXY:     10,
            lgSliderSmoothZ:      10,
            lgSliderSmoothAngles: 10,
            lgSliderConstraints:  10,
        };
        for (const [id, val] of Object.entries(defaults)) {
            const el = $(id);
            if (el) { el.value = val; el.dispatchEvent(new Event('input')); }
        }
        _refreshConstraints();
    }

    // ── MediaPipe error detection: live recompute + save ─────────────────────
    function _anyMPWeightActive() {
        for (const v of Object.values(_mpErrorWeights.detection)) if (v > 0) return true;
        for (const v of Object.values(_mpErrorWeights.corrections || {})) if (v > 0) return true;
        return false;
    }

    function _scheduleMPErrorRecompute() {
        if (_mpErrorRecomputeTimer) clearTimeout(_mpErrorRecomputeTimer);
        // Near-instant debounce (just enough to collapse bursts from a
        // slider drag).  The server caches normalised scores per trial so
        // every request after the first is essentially free.
        _mpErrorRecomputeTimer = setTimeout(_recomputeMPErrors, 30);
    }

    // Remember the last-active 2D / 3D stage selections so the master
    // Skeleton 2D / 3D checkboxes can restore them when re-toggled.
    let _saved2D = null;
    let _saved3D = null;

    function _mostRecentAvailableStage() {
        // Priority: latest pipeline stage whose intermediate is available.
        if (trialData?.skel_v2_joints_bc_3d)   return 'bone_smooth';
        if (trialData?.skel_v2_joints_hr_3d)   return 'bone_correct';
        if (trialData?.skel_v2_joints_zs_3d)   return 'hrnet_snap';
        if (trialData?.skel_v2_joints_3d)      return 'z_smooth';
        if (trialData?.skel_v2_joints_z_3d)    return 'z_correct';
        if (trialData?.mp_joints_3d)           return 'mediapipe';
        return null;
    }

    function _applyStageCheckboxes() {
        for (const s of ALL_STAGES) {
            const cb2 = document.querySelector(`.stage-row.stage-2d[data-stage="${s}"]`);
            const cb3 = document.querySelector(`.stage-row.stage-3d[data-stage="${s}"]`);
            const cbe = document.querySelector(`.stage-row.stage-err[data-stage="${s}"]`);
            if (cb2) cb2.checked = _stages2D.has(s);
            if (cb3) cb3.checked = _stages3D.has(s);
            if (cbe) cbe.checked = _stagesErr.has(s);
        }
    }

    function _updateStageRowVisibility() {
        // v3 stages — visible only when the user has clicked-expanded the
        // Skeleton v3 model row.  Per-stage rows that the current fit
        // didn't produce are hidden even when expanded:
        //   - stereo_correct: only when the bake's stereo-correct step
        //     actually ran (trialData.has_skel_v2_sc).
        //   - hrnet_snap: only when the user set "HRnet peak dist" > 0
        //     AND a peak file existed at bake time
        //     (trialData.has_skel_v2_hr).
        const _hideStages = new Set();
        if (trialData && trialData.has_skel_v2_sc === false) _hideStages.add('stereo_correct');
        if (trialData && trialData.has_skel_v2_hr === false) _hideStages.add('hrnet_snap');
        document.querySelectorAll('.stage-row').forEach(el => {
            const stage = el.getAttribute('data-stage');
            const hide = !_v3Expanded || _hideStages.has(stage);
            el.style.display = hide ? 'none' : '';
        });
        const ind = document.getElementById('v3Expand');
        if (ind) ind.textContent = _v3Expanded ? '▼' : '▶';
    }

    function _updateHeatmapRowVisibility() {
        document.querySelectorAll('.hm-row').forEach(el => {
            el.style.display = _heatmapExpanded ? '' : 'none';
        });
        const ind = document.getElementById('heatmapExpand');
        if (ind) ind.textContent = _heatmapExpanded ? '▼' : '▶';
    }

    function _setV3Expanded(open) {
        if (!!open === !!_v3Expanded) return;
        if (!open) {
            // Collapse: snapshot which v3 stages are active, then clear them.
            _v3SavedStages = {
                s2: new Set(_stages2D),
                s3: new Set(_stages3D),
                se: new Set(_stagesErr),
            };
            _stages2D.clear(); _stages3D.clear(); _stagesErr.clear();
            _applyStageCheckboxes();
        } else if (_v3SavedStages) {
            // Expand: restore the saved checkbox state.
            _stages2D = new Set(_v3SavedStages.s2);
            _stages3D = new Set(_v3SavedStages.s3);
            _stagesErr = new Set(_v3SavedStages.se);
            _v3SavedStages = null;
            _applyStageCheckboxes();
        }
        _v3Expanded = !!open;
        _updateStageRowVisibility();
        _syncErrorStages();
        showSkelErrors = _errorStages.size > 0;
        if (_errorStages.size > 0) _scheduleMPErrorRecompute();
        else { skelErrorMatrices = {}; }
        _saveCheckboxes();
        render(); update3D(); renderDistanceTrace();
    }

    // Render the historical-fit's inline controls based on its version.
    // v1/v2 → 2D + 3D checkboxes inline (current behavior).
    // v3   → expandable chevron + stage rows mirroring the live Skeleton v3 UI.
    let _prevExpanded = false;
    function _renderPrevFitControls(version) {
        const slot2D = $('prevCtrlSlot2D');
        const slot3D = $('prevCtrlSlot3D');
        const slotErr = $('prevCtrlSlotErr');
        const expand  = $('prevExpand');
        if (!slot2D || !slot3D || !slotErr) return;
        slot2D.innerHTML = ''; slot3D.innerHTML = ''; slotErr.innerHTML = '';
        // Drop any prev-stage rows from a previous selection.
        document.querySelectorAll('.prev-stage-row').forEach(el => el.remove());
        if (!version) {
            if (expand) { expand.style.display = 'none'; expand.textContent = '▶'; }
            _prevExpanded = false;
            return;
        }
        const isV3 = (version === 'v3' || (typeof version === 'string' && version.startsWith('corrections')));
        if (isV3) {
            // Hide the inline 2D/3D checkboxes; expose the chevron so the
            // user can expand stage rows beneath the dropdown row.
            if (expand) { expand.style.display = 'inline-block'; expand.textContent = _prevExpanded ? '▼' : '▶'; }
            _renderPrevStageRows();
        } else {
            // v1/v2 — show the standard 2D + 3D checkboxes inline.
            if (expand) { expand.style.display = 'none'; expand.textContent = '▶'; }
            const cb2 = document.createElement('input');
            cb2.type = 'checkbox'; cb2.id = 'showPrev2D'; cb2.style.margin = '0';
            cb2.checked = !!showPrev2D;
            cb2.addEventListener('change', ev => {
                showPrev2D = ev.target.checked;
                render(); renderDistanceTrace();
            });
            slot2D.appendChild(cb2);
            const cb3 = document.createElement('input');
            cb3.type = 'checkbox'; cb3.id = 'showPrev3D'; cb3.style.margin = '0';
            cb3.checked = !!showPrev3D;
            cb3.addEventListener('change', ev => {
                showPrev3D = ev.target.checked;
                update3D(); renderDistanceTrace();
            });
            slot3D.appendChild(cb3);
        }
    }

    // Per-stage display state for the historical fit's stage rows.
    // ``_prevStage2D`` / ``_prevStage3D`` are tag strings naming which
    // intermediate snapshot to show; ``showPrev2D`` / ``showPrev3D``
    // remain the master visibility flags (a stage is only drawn when
    // its dimension's flag is true AND its tag matches).  Single-stage-
    // active model (one stage per dimension) keeps the rendering path
    // simple -- the existing prevProj / prev3d lookups just read the
    // selected stage's arrays from prevFitData.
    let _prevStage2D = 'final';
    let _prevStage3D = 'final';
    // Stages exposed by the historical-fit endpoint (the live fit's v3
    // stages, minus 'mediapipe' which is the raw MP source and doesn't
    // belong to any specific saved fit).
    const _PREV_STAGE_DEFS = [
        { tag: 'sc',     label: 'Stereo-correct', color: '#f48fb1' },
        { tag: 'z',      label: 'Y/Z-correct',    color: '#ffa726' },
        { tag: 'zs',     label: 'Z-smooth',       color: '#ff9800' },
        { tag: 'hr',     label: 'HRnet-snap',     color: '#ab47bc' },
        { tag: 'bc',     label: 'Bone-correct',   color: '#66bb6a' },
        { tag: 'final',  label: 'Bone-smooth',    color: '#26c6da' },
    ];

    function _renderPrevStageRows() {
        const grid = document.getElementById('prevExpand')?.closest('div')?.parentElement;
        if (!grid) return;
        const insertAfter = $('prevCtrlSlotErr');
        if (!insertAfter) return;
        const _mk = (tag, attrs = {}, txt = '') => {
            const el = document.createElement(tag);
            for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
            if (txt) el.textContent = txt;
            return el;
        };
        // Drop the joints_3d_<tag> for stages that aren't actually
        // populated in this fit (e.g. older v3 fits saved before the
        // stereo-correct step existed).  Always include 'final'.
        const has = (t) => t === 'final'
            ? (prevFitData?.joints_3d != null)
            : (prevFitData?.[`joints_3d_${t}`] != null);
        const rows = _PREV_STAGE_DEFS.filter(d => has(d.tag));
        const fragRefs = [];
        let ref = insertAfter.nextSibling;
        for (const def of rows) {
            const swatch = `<span class="layer-swatch" style="background:${def.color};"></span>`;
            const lbl = _mk('span', {
                class: 'prev-stage-row',
                style: 'color:var(--text-muted);padding-left:18px;display:'
                       + (_prevExpanded ? '' : 'none') + ';',
            });
            lbl.innerHTML = `${swatch} ${def.label}`;
            const cb2 = _mk('input', {
                type: 'checkbox', class: 'prev-stage-row',
                style: 'margin:0;justify-self:center;display:'
                       + (_prevExpanded ? '' : 'none') + ';',
            });
            cb2.checked = !!showPrev2D && _prevStage2D === def.tag;
            cb2.addEventListener('change', (ev) => {
                if (ev.target.checked) {
                    _prevStage2D = def.tag;
                    showPrev2D = true;
                } else if (_prevStage2D === def.tag) {
                    showPrev2D = false;
                }
                // Single-active: re-render rows to update siblings.
                document.querySelectorAll('.prev-stage-row').forEach(el => el.remove());
                _renderPrevStageRows();
                render(); renderDistanceTrace();
            });
            const cb3 = _mk('input', {
                type: 'checkbox', class: 'prev-stage-row',
                style: 'margin:0;justify-self:center;display:'
                       + (_prevExpanded ? '' : 'none') + ';',
            });
            cb3.checked = !!showPrev3D && _prevStage3D === def.tag;
            cb3.addEventListener('change', (ev) => {
                if (ev.target.checked) {
                    _prevStage3D = def.tag;
                    showPrev3D = true;
                } else if (_prevStage3D === def.tag) {
                    showPrev3D = false;
                }
                document.querySelectorAll('.prev-stage-row').forEach(el => el.remove());
                _renderPrevStageRows();
                update3D(); renderDistanceTrace();
            });
            const filler = _mk('span', {
                class: 'prev-stage-row',
                style: 'display:' + (_prevExpanded ? '' : 'none') + ';',
            });
            for (const el of [lbl, cb2, cb3, filler]) {
                grid.insertBefore(el, ref);
            }
            fragRefs.push(lbl, cb2, cb3, filler);
        }
    }

    function _setPrevExpanded(open) {
        _prevExpanded = !!open;
        const expand = $('prevExpand');
        if (expand) expand.textContent = _prevExpanded ? '▼' : '▶';
        document.querySelectorAll('.prev-stage-row').forEach(el => {
            el.style.display = _prevExpanded ? '' : 'none';
        });
    }

    function _setHeatmapExpanded(open) {
        if (!!open === !!_heatmapExpanded) return;
        if (!open) {
            _heatmapSavedState = {
                hr2d: showHRnet2D, hr3d: showHRnet3D,
                hm2d: showHeatmap2D, hm3d: showHeatmap3D,
                img: showHeatmap,
            };
            showHRnet2D = false; showHRnet3D = false;
            showHeatmap2D = false; showHeatmap3D = false;
            if (showHeatmap && typeof _setHeatmapEnabled === 'function') {
                _setHeatmapEnabled(false);
            }
            const ids = ['showHeatmap2D', 'showHeatmap3D'];
            ids.forEach(id => { const el = $(id); if (el) el.checked = false; });
            document.querySelectorAll('.hm-row.hm-2d, .hm-row.hm-3d').forEach(el => {
                if (el.id !== 'showHeatmap2D' && el.id !== 'showHeatmap3D') el.checked = false;
            });
        } else if (_heatmapSavedState) {
            const s = _heatmapSavedState;
            showHRnet2D = !!s.hr2d; showHRnet3D = !!s.hr3d;
            showHeatmap2D = !!s.hm2d; showHeatmap3D = !!s.hm3d;
            const e2 = $('showHeatmap2D'); if (e2) e2.checked = showHeatmap2D;
            const e3 = $('showHeatmap3D'); if (e3) e3.checked = showHeatmap3D;
            document.querySelectorAll('.hm-row.hm-2d, .hm-row.hm-3d').forEach(el => {
                if (el.id === 'showHeatmap2D' || el.id === 'showHeatmap3D') return;
                if (el.classList.contains('hm-2d')) el.checked = showHRnet2D;
                if (el.classList.contains('hm-3d')) el.checked = showHRnet3D;
            });
            if (s.img && typeof _setHeatmapEnabled === 'function') _setHeatmapEnabled(true);
            _heatmapSavedState = null;
        }
        _heatmapExpanded = !!open;
        _updateHeatmapRowVisibility();
        _saveCheckboxes();
        render(); update3D(); renderDistanceTrace();
    }

    function _refreshSkelErrors(scheduleRecompute) {
        _syncErrorStages();
        showMPErrors = false;
        showSkelErrors = _stagesErr.size > 0;
        if (_stagesErr.size === 0) {
            skelErrorMatrices = {};
            const stEl = $('errorsStatus');
            if (stEl) stEl.textContent = '';
        }
        // Always update canvases immediately so toggles feel instant.
        render(); update3D(); renderDistanceTrace();
        if (scheduleRecompute && _stagesErr.size > 0) _scheduleMPErrorRecompute();
    }

    function _onSkelMasterToggle(dim, checked) {
        const tgt = dim === '2D' ? _stages2D : _stages3D;
        const savedRef = dim === '2D' ? () => _saved2D : () => _saved3D;
        const setSaved = v => { if (dim === '2D') _saved2D = v; else _saved3D = v; };
        if (checked) {
            // Restore previous selection if any, else pick the most-recent stage
            if (tgt.size === 0) {
                const saved = savedRef();
                if (saved && saved.size > 0) {
                    for (const s of saved) tgt.add(s);
                    setSaved(null);
                } else {
                    const best = _mostRecentAvailableStage();
                    if (best) tgt.add(best);
                }
            }
        } else {
            // Remember and clear
            if (tgt.size > 0) setSaved(new Set(tgt));
            tgt.clear();
        }
        _applyStageCheckboxes();
        _updateStageRowVisibility();
        _refreshSkelErrors(true);
    }

    function _onStageToggle(stage, dim, checked) {
        const tgt = dim === '2D' ? _stages2D : (dim === '3D' ? _stages3D : _stagesErr);
        if (checked) tgt.add(stage);
        else         tgt.delete(stage);
        // Err needs a displayed model to overlay red markers on — if
        // neither this stage's 2D nor 3D is on, auto-check 2D so the user
        // actually sees something.
        if (dim === 'Err' && checked && !_stages2D.has(stage) && !_stages3D.has(stage)) {
            _stages2D.add(stage);
            if (!showSkelV2_2D) {
                showSkelV2_2D = true;
                const m = $('showSkelV2_2D');
                if (m) m.checked = true;
            }
            _applyStageCheckboxes();
        }
        // For 2D / 3D dimensions also sync the master Skeleton checkbox.
        if (dim === '2D' || dim === '3D') {
            const masterId = dim === '2D' ? 'showSkelV2_2D' : 'showSkelV2_3D';
            const master = $(masterId);
            if (master) {
                if (tgt.size > 0 && !master.checked) {
                    master.checked = true;
                    if (dim === '2D') showSkelV2_2D = true; else showSkelV2_3D = true;
                } else if (tgt.size === 0 && master.checked) {
                    master.checked = false;
                    if (dim === '2D') showSkelV2_2D = false; else showSkelV2_3D = false;
                }
            }
        }
        _syncAllSkelCheckboxes?.();
        _updateStageRowVisibility();
        _updateHandDiagramColor?.();
        _refreshSkelErrors(true);
    }

    // Background score-cache prewarm — fires one /mp_errors request per
    // stage shortly after trial load.  The backend's in-process score
    // cache and on-disk per-stage error cache are both keyed by
    // (subject, trial, stage_tag, sliders), so a successful prewarm
    // turns the user's first slider tweak into a fast threshold-only
    // recompute instead of a full score build.
    //
    // Skip entirely when neither Skeleton v3 nor HRnet (the only
    // consumers of the mp_errors score) has been run on this trial —
    // computing scores nobody will ever look at burns ~5 server calls
    // per trial load and pulls the (potentially large) MP error data
    // into memory for no reason.
    let _prewarmStarted = false;
    async function _prewarmStages() {
        if (_prewarmStarted) return;
        if (!subjectId || currentTrialIdx < 0 || !trials?.[currentTrialIdx]) return;
        const hasV3   = !!(trialData && trialData.has_skel_v2);
        const hasHR   = !!(trialData && trialData.hrnet_peaks_3d);
        if (!hasV3 && !hasHR) {
            return;
        }
        _prewarmStarted = true;
        // Wait for browser idle so the trial's first paint, video decode, and
        // 3D scene assembly all finish before we start hitting the backend
        // with N parallel score-build requests.  Falls back to a 2 s delay
        // when requestIdleCallback isn't available.
        await new Promise(resolve => {
            if (typeof requestIdleCallback === 'function') {
                requestIdleCallback(resolve, { timeout: 4000 });
            } else {
                setTimeout(resolve, 2000);
            }
        });
        if (!_prewarmStarted) return;  // trial may have changed
        const trialIdx = trials[currentTrialIdx]?.trial_idx;
        if (trialIdx == null) return;
        for (const stage of ALL_STAGES) {
            const cfg = STAGE_CONFIGS[stage];
            if (!cfg?.factor) continue;
            try {
                await api(`/api/skeleton/${subjectId}/mp_errors`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        trial_idx: trialIdx,
                        detection: _mpErrorWeights.detection,
                        attribution: _mpErrorWeights.attribution,
                        corrections: {},
                        stage,
                    }),
                });
            } catch { /* best-effort prewarm */ }
        }
    }

    async function _recomputeMPErrors() {
        if (!subjectId || currentTrialIdx < 0 || !trials?.[currentTrialIdx]) return;
        if (_mpErrorPending) { _mpErrorPending = 'retry'; return; }
        const activeStages = [..._stagesErr];
        if (activeStages.length === 0) {
            skelErrorMatrices = {};
            render(); update3D(); renderDistanceTrace();
            return;
        }
        _mpErrorPending = true;
        const stEl = $('errorsStatus');
        // Defer the "Computing…" label so fast cache-hit responses don't
        // flash a status flicker.  Only shown if the request actually
        // takes more than 300 ms.
        let _computingTimer = null;
        if (stEl) {
            _computingTimer = setTimeout(() => { stEl.textContent = 'Computing…'; }, 300);
        }
        try {
            // Fetch per-stage errors in parallel — each filtered to its own factor.
            // Stages with factor=null (e.g. z_correct, the final output) display
            // no error markers — skip the request entirely for those.
            const stagesNeedingErrors = activeStages.filter(s => STAGE_CONFIGS[s]?.factor);
            const responses = await Promise.all(stagesNeedingErrors.map(stage => {
                const factor = STAGE_CONFIGS[stage].factor;
                const active = Array.isArray(factor) ? new Set(factor) : new Set([factor]);
                const detFiltered = {};
                for (const k of Object.keys(_mpErrorWeights.detection)) {
                    detFiltered[k] = active.has(k) ? _mpErrorWeights.detection[k] : 0;
                }
                return api(`/api/skeleton/${subjectId}/mp_errors`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        trial_idx: trials[currentTrialIdx].trial_idx,
                        detection: detFiltered,
                        attribution: _mpErrorWeights.attribution,
                        corrections: {},
                        stage,
                    }),
                }).then(resp => ({ stage, errors: resp?.errors || null,
                                   total: resp?.total_flagged ?? 0 }));
            }));
            skelErrorMatrices = {};
            let totalFlagged = 0;
            const parts = [];
            for (const r of responses) {
                skelErrorMatrices[r.stage] = r.errors;
                totalFlagged += r.total;
                parts.push(`${r.stage}: ${r.total}`);
            }
            if (_computingTimer) { clearTimeout(_computingTimer); _computingTimer = null; }
            if (stEl) stEl.textContent = parts.join(' • ');
            render(); update3D(); renderDistanceTrace();
        } catch (e) {
            if (_computingTimer) { clearTimeout(_computingTimer); _computingTimer = null; }
            if (stEl) stEl.textContent = `Error: ${e.message}`;
            console.error('MP error recompute failed', e);
        } finally {
            const pending = _mpErrorPending === 'retry';
            _mpErrorPending = false;
            if (pending) _scheduleMPErrorRecompute();
        }
    }

    function _updateMPErrorAvailability() {
        // No-op; stage button availability is handled separately.
    }

    async function saveErrors() {
        if (!subjectId || currentTrialIdx < 0) return;
        const stEl = $('errorsStatus');
        try {
            await api(`/api/skeleton/${subjectId}/save_mp_errors`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trial_idx: trials[currentTrialIdx].trial_idx,
                    detection: _mpErrorWeights.detection,
                    attribution: _mpErrorWeights.attribution,
                }),
            });
            if (stEl) stEl.textContent = 'Saved mp_errors.npz ✓';
            if (trialData) trialData.has_mp_errors = true;
            _updateMPErrorAvailability();
        } catch (e) {
            if (stEl) stEl.textContent = `Save failed: ${e.message}`;
        }
    }

    function _enforceSourceConstraint() {
        // When multiple metrics are selected, keep only the highest-priority active source.
        // Priority: Skeleton (Skeleton) > MediaPipe > Vision
        if (selectedMetrics.size <= 1) return;
        const manoOn = showMano2D || showMano3D;
        const mpOn   = showMP2D   || showMP3D;
        const visOn  = showVision2D || showVision3D;
        const keep = manoOn ? 'skeleton' : mpOn ? 'mp' : visOn ? 'vision' : null;
        if (!keep) return;
        const groups = {
            skeleton:   ['showMano2D',   'showMano3D',   'showManoSkel'],
            mp:     ['showMP2D',     'showMP3D',     'showMPSkel'],
            vision: ['showVision2D', 'showVision3D', 'showVisionSkel'],
        };
        for (const [src, ids] of Object.entries(groups)) {
            if (src === keep) continue;
            for (const id of ids) {
                const el = $(id);
                if (el && el.checked) { el.checked = false; el.dispatchEvent(new Event('change')); }
            }
        }
    }

    function resetPlotSelection() {
        selectedMetrics.clear();
        posJointStates.clear();
        plotJointStates.clear();
        _tiaIsDefault = true;
        if (plotMode === 'angle') {
            selectedMetrics.set(_TIA, METRIC_COLORS[0]);
        }
        wristPanelOpen = false;
        document.dispatchEvent(new CustomEvent('plotReset'));
        renderDistanceTrace();
        _updatePlotHighlight();
    }

    function _renderDualToggle(btnId, optA, optB, activeVal) {
        const btn = $(btnId);
        if (!btn) return;
        const actStyle = 'color:var(--text);font-weight:bold;';
        const inaStyle = 'color:var(--text-muted);';
        btn.innerHTML =
            `<span style="${optA === activeVal ? actStyle : inaStyle}">${optA}</span>` +
            `<span style="opacity:0.35;margin:0 3px;">|</span>` +
            `<span style="${optB === activeVal ? actStyle : inaStyle}">${optB}</span>`;
    }

    function togglePlotMode() {
        plotMode = plotMode === 'angle' ? 'position' : 'angle';
        _renderDualToggle('plotModeToggle', 'Angle', 'Position',
                          plotMode === 'angle' ? 'Angle' : 'Position');
        selectedMetrics.clear();
        plotJointStates.clear();
        posJointStates.clear();
        wristPanelOpen = false;
        _tiaIsDefault = true;
        if (plotMode === 'angle') {
            selectedMetrics.set(_TIA, METRIC_COLORS[0]);
        }
        renderDistanceTrace();
        _updatePlotHighlight();
        update3D();
    }

    // ── Joint angle constraints editor ───────────────────────────────────

    async function openConstraintsEditor() {
        const modal = $('constraintsModal');
        const body  = $('constraintsBody');
        const badge = $('constraintsCustomBadge');
        if (!modal || !body) return;

        body.innerHTML = '<tr><td colspan="3" style="padding:8px;color:var(--text-muted);">Loading...</td></tr>';
        modal.style.display = 'flex';

        try {
            const data = await api('/api/skeleton/joint-constraints');
            const c = data.constraints;
            badge.style.display = data.is_custom ? '' : 'none';
            // Apply any drag overrides to the loaded values before rendering
            if (c.joints) {
                for (const jt of c.joints) {
                    const ovr = _constraintOverrides[jt.name];
                    if (ovr) {
                        if (ovr.flex_min != null) jt.flex_min = ovr.flex_min;
                        if (ovr.flex_max != null) jt.flex_max = ovr.flex_max;
                        if (ovr.abd_min  != null) jt.abd_min  = ovr.abd_min;
                        if (ovr.abd_max  != null) jt.abd_max  = ovr.abd_max;
                    }
                }
            }
            _renderConstraintsTable(body, c);
        } catch (e) {
            body.innerHTML = `<tr><td colspan="3" style="color:var(--red);padding:8px;">Error: ${e.message}</td></tr>`;
        }
    }

    const _inputStyle = 'width:48px;padding:2px 3px;font-size:11px;background:var(--bg);border:1px solid var(--border);border-radius:3px;color:var(--text);text-align:right;';

    const _CONSTRAINT_GROUPS = [
        { key: 'thumb_cmc_mcp',  label: 'Thumb CMC/MCP',    joints: ['Thumb CMC', 'Thumb MCP'] },
        { key: 'thumb_ip',       label: 'Thumb IP',          joints: ['Thumb IP'] },
        { key: 'finger_mcp',     label: 'Finger MCP',        joints: ['Index MCP', 'Middle MCP', 'Ring MCP', 'Pinky MCP'] },
        { key: 'finger_pip_dip', label: 'Finger PIP/DIP',    joints: ['Index PIP', 'Index DIP', 'Middle PIP', 'Middle DIP', 'Ring PIP', 'Ring DIP', 'Pinky PIP', 'Pinky DIP'] },
    ];

    function _renderConstraintsTable(body, constraints) {
        let html = '';
        const joints = constraints.joints || [];
        const groups = constraints.constraint_groups || {};

        // Group enable/disable checkboxes
        html += '<tr><td colspan="5" style="padding:4px 0 6px;font-weight:600;font-size:11px;color:var(--text-muted);">Enable/Disable Groups</td></tr>';
        html += '<tr style="font-size:10px;color:var(--text-muted);"><td></td><td colspan="2" style="text-align:center;">Flex</td><td colspan="2" style="text-align:center;">Abd</td></tr>';
        for (const g of _CONSTRAINT_GROUPS) {
            const grp = groups[g.key] || { flex: true, abd: true };
            html += `<tr data-group="${g.key}">
                <td style="padding:2px 6px;font-size:11px;">${g.label}</td>
                <td colspan="2" style="text-align:center;"><input type="checkbox" class="cst-grp-flex" ${grp.flex ? 'checked' : ''} style="margin:0;"></td>
                <td colspan="2" style="text-align:center;"><input type="checkbox" class="cst-grp-abd" ${grp.abd ? 'checked' : ''} style="margin:0;"></td>
            </tr>`;
        }
        html += '<tr><td colspan="5" style="padding:6px 0 2px;border-top:1px solid var(--border);"></td></tr>';

        for (const jt of joints) {
            html += `<tr data-name="${jt.name}">
                <td style="padding:3px 6px;white-space:nowrap;cursor:pointer;color:var(--blue);" class="cst-joint-link" data-joint-name="${jt.name}">${jt.name}</td>
                <td style="padding:3px 2px;"><input type="number" class="cst-flex-min" value="${jt.flex_min}" step="1" style="${_inputStyle}"></td>
                <td style="padding:3px 2px;"><input type="number" class="cst-flex-max" value="${jt.flex_max}" step="1" style="${_inputStyle}"></td>
                <td style="padding:3px 2px;"><input type="number" class="cst-abd-min"  value="${jt.abd_min}"  step="1" style="${_inputStyle}"></td>
                <td style="padding:3px 2px;"><input type="number" class="cst-abd-max"  value="${jt.abd_max}"  step="1" style="${_inputStyle}"></td>
            </tr>`;
        }
        // Flex coupling weight slider
        const fcw = constraints.flex_coupling ?? 1.0;
        html += '<tr><td colspan="5" style="padding:8px 0 2px;font-weight:600;font-size:11px;color:var(--text-muted);border-top:1px solid var(--border);">Flex Coupling</td></tr>';
        html += `<tr><td style="padding:3px 6px;white-space:nowrap;">Within-finger</td>
            <td colspan="4" style="padding:3px 2px;">
                <div style="display:flex;align-items:center;gap:6px;">
                    <input type="range" id="cstFlexCoupling" min="0" max="20" step="0.1" value="${fcw}" style="flex:1;height:4px;">
                    <span id="cstFlexCouplingVal" style="font-size:11px;color:var(--accent);font-weight:bold;min-width:24px;">${fcw.toFixed(1)}</span>
                </div>
            </td></tr>`;

        body.innerHTML = html;

        // Wire up flex coupling slider display
        const fcSlider = body.querySelector('#cstFlexCoupling');
        if (fcSlider) fcSlider.addEventListener('input', () => {
            const valEl = body.querySelector('#cstFlexCouplingVal');
            if (valEl) valEl.textContent = parseFloat(fcSlider.value).toFixed(1);
        });

        // Clicking a joint name: save constraints, close modal, plot that joint's angles
        body.querySelectorAll('.cst-joint-link').forEach(td => {
            td.addEventListener('click', async () => {
                const jointName = td.dataset.jointName;
                // Save current edits first
                await saveConstraints();
                // Close modal
                closeConstraintsEditor();

                // Find the joint index from flex_angle_options
                const flexOpts = trialData?.flex_angle_options || [];
                const match = flexOpts.find(f => f.name === `Flex: ${jointName}`);
                if (!match) return;
                const j = match.joint;
                const flexName = `Flex: ${jointName}`;
                const abdName = `Abd: ${jointName}`;

                // Clear all plots and set this joint
                selectedMetrics.clear();
                plotJointStates.clear();
                posJointStates.clear();
                _constraintFocusMetric = null;

                const c1 = _nextMetricColor();
                selectedMetrics.set(flexName, c1);
                const c2 = _nextMetricColor();
                selectedMetrics.set(abdName, c2);
                plotJointStates.set(j, { mode: 'both', colorFlex: c1, colorAbd: c2 });

                // Focus on flex to show constraint lines
                _constraintFocusMetric = flexName;

                renderDistanceTrace();
                _updatePlotHighlight();
                render();
                update3D();
            });
        });
    }

    /** Refresh constraint data in trialData and clear drag overrides. */
    async function _refreshConstraints() {
        try {
            const data = await api('/api/skeleton/joint-constraints');
            const c = data.constraints;
            if (trialData) {
                trialData.angle_constraints = c.joints || [];
            }
            // Clear drag overrides — saved values are now canonical
            for (const k in _constraintOverrides) delete _constraintOverrides[k];
            renderDistanceTrace();
        } catch {}
    }

    async function saveConstraints() {
        const body = $('constraintsBody');
        if (!body) return;

        // Rebuild full JSON from current inputs + original structural data
        let orig;
        try { orig = (await api('/api/skeleton/joint-constraints')).constraints; } catch { return; }

        const rows = body.querySelectorAll('tr[data-name]');
        const edits = {};
        rows.forEach(row => {
            edits[row.dataset.name] = {
                flex_min: parseFloat(row.querySelector('.cst-flex-min').value),
                flex_max: parseFloat(row.querySelector('.cst-flex-max').value),
                abd_min:  parseFloat(row.querySelector('.cst-abd-min').value),
                abd_max:  parseFloat(row.querySelector('.cst-abd-max').value),
            };
        });

        const updated = JSON.parse(JSON.stringify(orig));
        for (const jt of (updated.joints || [])) {
            const e = edits[jt.name];
            if (e) { jt.flex_min = e.flex_min; jt.flex_max = e.flex_max; jt.abd_min = e.abd_min; jt.abd_max = e.abd_max; }
        }
        // Flex coupling weight
        const fcSlider = body.querySelector('#cstFlexCoupling');
        if (fcSlider) updated.flex_coupling = parseFloat(fcSlider.value);

        // Constraint group checkboxes
        const groupRows = body.querySelectorAll('tr[data-group]');
        if (groupRows.length) {
            if (!updated.constraint_groups) updated.constraint_groups = {};
            groupRows.forEach(row => {
                const key = row.dataset.group;
                updated.constraint_groups[key] = {
                    flex: row.querySelector('.cst-grp-flex').checked,
                    abd:  row.querySelector('.cst-grp-abd').checked,
                };
            });
        }

        try {
            await api('/api/skeleton/joint-constraints', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updated),
            });
            $('constraintsCustomBadge').style.display = '';
            await _refreshConstraints();
            closeConstraintsEditor();
        } catch (e) {
            alert('Save failed: ' + e.message);
        }
    }

    async function resetConstraints() {
        try {
            await api('/api/skeleton/joint-constraints', { method: 'DELETE' });
            // Reload table with defaults
            const data = await api('/api/skeleton/joint-constraints');
            $('constraintsCustomBadge').style.display = 'none';
            _renderConstraintsTable($('constraintsBody'), data.constraints);
            await _refreshConstraints();
        } catch (e) {
            alert('Reset failed: ' + e.message);
        }
    }

    function closeConstraintsEditor() {
        const modal = $('constraintsModal');
        if (modal) modal.style.display = 'none';
    }

    async function cancelConstraintsEditor() {
        // Discard unsaved edits by re-loading current server state into the table
        try {
            const data = await api('/api/skeleton/joint-constraints');
            _renderConstraintsTable($('constraintsBody'), data.constraints);
        } catch {}
        closeConstraintsEditor();
    }

    return { goToFrame, togglePlay, toggleSide, resetZoom, toggleTrackingZoom, prevSubject, nextSubject, getExportContext, runStage1, renderDistanceTrace, resetFitDefaults, resetPlotSelection, togglePlotMode, runFitV2, resetFitV2Defaults, runFitLegacy, resetFitLegacyDefaults, saveErrors, openConstraintsEditor, saveConstraints, resetConstraints, closeConstraintsEditor, cancelConstraintsEditor };
})();

// Expose on window for cross-module access (labels.js is an ES module)
window.manoViewer = manoViewer;
