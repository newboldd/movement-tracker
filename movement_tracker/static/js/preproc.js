/* Pre-proc page (Phase A): camera-trajectory extraction + visualisation. */
(() => {
    const $ = id => document.getElementById(id);
    const api = (url, opts) => fetch(url, opts).then(async r => {
        if (!r.ok) throw new Error(`${url}: HTTP ${r.status} — ${await r.text()}`);
        return r.json();
    });

    let subjects = [];
    let subjectId = null;
    let subjectName = null;
    let trials = [];
    let currentTrialIdx = -1;
    let trialMeta = null;
    let trajectory = null;
    let currentFrame = 0;
    let nFrames = 0;
    // The "Stabilized view" and "Mark jerks" toggles were removed from
    // the View panel -- use the Stabilized overlay radio for the
    // warped view.  These stay as fixed-false flags so the render /
    // plot code that still reads them is a no-op.
    const isStabilized = false;
    const showJerks = false;
    let playing = false;
    let playTimer = null;
    // Hidden <video> element with the trial's source video.  All
    // playback and scrubbing seeks against this element — same pattern
    // as mano.js / videos.js, which is what makes smooth playback work.
    let liveVideoEl = null;
    let liveVideoReady = false;
    let vidW = 0, vidH = 0;     // video native dimensions (stereo: full 3840×1080)
    let isStereo = false;
    const _hasRVFC = typeof HTMLVideoElement !== 'undefined'
                     && 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
    let activeJobId = null;
    let activeEvtSource = null;
    // Background+bake state.  The compute job is independent of the
    // trajectory job — it uses a second SSE source.  The "overlay" is
    // one of three modes: 'off' (live frame), 'stable' (read from
    // baked stable.mp4 by seeking a hidden video element), or 'fg'
    // (same but reading fg.mp4).
    let backgroundData = null;
    let bgJobId = null;
    let bgEvtSource = null;
    let overlayMode = 'off';    // 'off' | 'stable' | 'bg'
    let stableVideoEl = null;   // hidden <video> tied to stable.mp4
    let showOutline = true;     // checkbox: show live-fetched hand boundary
    let showFgFill = false;     // checkbox: also fetch + paint JET heatmap
    let fgFillOpacity = 0.5;    // 0..1, controlled by slider next to checkbox
    let showBgZones = true;     // checkbox: preview palm zone + forearm cone
    let showSkinMask = false;   // checkbox: live YCrCb skin-color overlay
    // Offscreen canvas + cache key for the live skin-color mask.  The
    // classification mirrors the server's _is_skin_ycc (BT.601 YCrCb,
    // leniency-scaled window) so the preview matches what Compute
    // Background will actually do.
    let _skinMaskCanvas = null;
    let _skinMaskKey = null;
    // Live outline state -- replaces the old fg.mp4 / outline.mp4 bake.
    // Server returns a per-frame closed-polygon contour for the
    // current dilation; we fetch it whenever the frame or slider
    // changes (debounced ~150 ms) and draw it on top of the canvas.
    let outlineData = null;          // {frame, dilation_px, is_stereo, OS, OD, fg_OS, fg_OD}
    let outlineFetchTimer = null;    // debounce handle
    let outlineFetchSeq = 0;         // request-id so late responses don't clobber newer state
    // Foreground heatmap <img> elements (one per side) -- decoded
    // out-of-band so the canvas drawImage path doesn't block on
    // base64 -> bitmap on every render tick.
    const fgImageEls = { OS: null, OD: null };
    const fgImageBboxes = { OS: null, OD: null };
    // MP keypoints for the current trial — used to draw the dilated-
    // skeleton preview on the canvas while the Compute Stable + Mask
    // panel is open.  Two coordinate spaces:
    //   raw_OS / raw_OD : frame pixel coords (1920×1080 per half).
    //                     Used when the canvas shows the live video.
    //   ref_OS / ref_OD : same keypoints warped into stable.mp4 ref
    //                     coords via the saved trajectory.  Used for
    //                     stable / fg / bg overlays.
    let mpKeypoints = null;     // { raw_OS, raw_OD, ref_OS, ref_OD, n_frames }
    // MediaPipe finger chain joint indices (same convention as the
    // service module).
    const _MP_FINGER_CHAINS = [
        [0, 1, 2, 3, 4],     // thumb
        [0, 5, 6, 7, 8],     // index
        [0, 9, 10, 11, 12],  // middle
        [0, 13, 14, 15, 16], // ring
        [0, 17, 18, 19, 20], // pinky
    ];
    // Palm-arc (MCP chain) for closing the silhouette.
    const _MP_PALM_CHAIN = [1, 5, 9, 13, 17];

    /**
     * Synthetic ulnar-heel point: thumb CMC (joint 1) reflected across
     * the wrist (0) -> middle-MCP (9) axis.  Mirrors
     * _reflect_thumb_cmc in background.py so the on-canvas previews
     * show the same ulnar-palm coverage the server bakes in.
     * ``f`` is a [21][2] keypoint array; returns [x, y] or null.
     */
    function _reflectThumbCmc(f) {
        const has = j => f[j] && f[j][0] != null && f[j][1] != null;
        if (!has(0) || !has(1) || !has(9)) return null;
        const wx = f[0][0], wy = f[0][1];
        let ax = f[9][0] - wx, ay = f[9][1] - wy;
        const n = Math.hypot(ax, ay);
        if (n < 1e-6) return null;
        ax /= n; ay /= n;
        const vx = f[1][0] - wx, vy = f[1][1] - wy;
        const dot = vx * ax + vy * ay;
        const perpx = vx - dot * ax, perpy = vy - dot * ay;
        // reflected = cmc - 2 * perp
        return [f[1][0] - 2 * perpx, f[1][1] - 2 * perpy];
    }
    // Stereo controls: side toggle alternates between 'OS' and 'OD'.
    // The /api/deidentify/{id}/frame endpoint does the half-crop
    // server-side via the ``side=`` query param.
    let currentSide = 'OS';
    const CAMERA_NAMES = ['OS', 'OD'];
    // Playback speed presets — slider index maps to a multiplier.
    // Default index = 3 (1x).  Frame fetches are async, so speeds above
    // 4x will skip frames rather than queueing up loads.
    const SPEED_PRESETS = [0.25, 0.5, 1, 2, 4, 8, 16, 32];
    let playbackRate = 1;

    // Zoom/pan — same conventions as videos.js/mano.js.  Offsets are in
    // canvas pixels relative to the base-centered origin (computed in
    // getBaseMetrics()).  Scale stacks on top of the base fit-scale.
    let scale = 1, offsetX = 0, offsetY = 0;
    let dragging = false;
    let dragStartX = 0, dragStartY = 0;
    let panStartOX = 0, panStartOY = 0;

    const canvas = $('canvas');
    const ctx    = canvas.getContext('2d');
    const plot   = $('trajPlot');
    const pctx   = plot.getContext('2d');

    async function loadSubjects() {
        try { subjects = await api('/api/subjects'); }
        catch (e) { subjects = []; }
        const sel = $('subjectSelect');
        sel.innerHTML = subjects.map(s =>
            `<option value="${s.id}">${s.name}</option>`).join('');
        // Subject pre-selection priority:
        //   1. ``?subject=N`` URL param  (deep-link from the Subjects page)
        //   2. ``sessionStorage.lastSubjectId`` (cross-page persistence)
        //   3. first subject in the list
        const fromUrl = new URLSearchParams(window.location.search).get('subject');
        const saved   = sessionStorage.getItem('lastSubjectId');
        const initialId = (fromUrl && subjects.some(s => String(s.id) === fromUrl))
            ? fromUrl
            : (saved && subjects.some(s => String(s.id) === saved) ? saved : null);
        if (initialId) sel.value = initialId;
        sel.addEventListener('change', () => onSubjectChange(parseInt(sel.value)));
        if (sel.value) await onSubjectChange(parseInt(sel.value));
    }

    async function onSubjectChange(id) {
        subjectId = id;
        const subj = subjects.find(s => s.id === id);
        subjectName = subj ? subj.name : '';
        sessionStorage.setItem('lastSubjectId', String(id));
        try { trials = await api(`/api/mano/${id}/trials`); }
        catch (e) { trials = []; }
        const wrap = $('trialBtns');
        // /api/mano/{id}/trials returns `trial_stem` (e.g. "Con01_R1").
        // Strip the "{subject}_" prefix for a compact label like "R1" —
        // same convention as the video browser.  Fall back to the stem.
        const _trialLabel = (t) => {
            const stem = t.trial_stem || t.trial_name || '';
            if (subjectName && stem.startsWith(subjectName + '_')) {
                return stem.slice(subjectName.length + 1);
            }
            return stem || '?';
        };
        wrap.innerHTML = trials.map((t, i) =>
            `<button class="trial-btn" data-i="${i}" title="${t.trial_stem || ''}">${_trialLabel(t)}</button>`).join('');
        wrap.querySelectorAll('.trial-btn').forEach(b => {
            b.addEventListener('click', () => selectTrial(parseInt(b.dataset.i)));
        });
        if (trials.length > 0) {
            // Restore the previously-selected trial across page reloads.
            // Look up by trial_stem so we survive trial-order changes.
            const savedStem = sessionStorage.getItem('preprocLastTrialStem');
            let idx = 0;
            if (savedStem) {
                const i = trials.findIndex(t => t.trial_stem === savedStem);
                if (i >= 0) idx = i;
            }
            await selectTrial(idx);
        } else {
            $('statusMsg').textContent = 'no trials for this subject';
        }
    }

    async function selectTrial(idx) {
        currentTrialIdx = idx;
        trialMeta = trials[idx];
        if (trialMeta?.trial_stem) {
            sessionStorage.setItem('preprocLastTrialStem', trialMeta.trial_stem);
        }
        nFrames = trialMeta.n_frames || trialMeta.frame_count || 0;
        currentFrame = 0;
        // New trial → reset zoom/pan so the image fits the viewport.
        scale = 1; offsetX = 0; offsetY = 0;
        document.querySelectorAll('.trial-btn').forEach((b, i) =>
            b.classList.toggle('active', i === idx));
        $('frameDisplay').textContent = `Frame: 0 / ${nFrames}`;
        $('trajectoryStatus').textContent = '';
        $('trajectoryStats').textContent = '';
        $('osodAgree').textContent = '';
        trajectory = null;
        // Default to OS on every trial open.  ``updateCameraControls``
        // collapses to 'full' for non-stereo trials (single-camera).
        currentSide = 'OS';
        updateCameraControls();
        await _loadTrialVideo();
        await refreshTrajectoryFromServer();
        await refreshBackgroundFromServer();
        // MP keypoints for the dilated-skeleton preview.  Best-effort —
        // returns ``{available: false}`` if MP hasn't been run yet, in
        // which case the preview just won't draw.
        mpKeypoints = null;
        try {
            const k = await api(`/api/preproc/${subjectId}/trial/${trialMeta.trial_idx}/mp_keypoints`);
            if (k && k.available) mpKeypoints = k;
        } catch (_e) {}
        drawPlot();
    }

    function _loadTrialVideo() {
        // Point the hidden <video> element at the trial's source mp4.
        // We seek into it for scrubbing and let the browser decode at
        // native speed for playback — same pattern as videos.js / mano.js.
        liveVideoReady = false;
        if (!liveVideoEl) {
            liveVideoEl = document.createElement('video');
            liveVideoEl.muted = true;
            liveVideoEl.playsInline = true;
            liveVideoEl.preload = 'auto';
            liveVideoEl.crossOrigin = 'anonymous';
            liveVideoEl.style.display = 'none';
            document.body.appendChild(liveVideoEl);
        }
        const src = `/api/mano/${subjectId}/trial/${trialMeta.trial_idx}/video`;
        liveVideoEl.src = src;
        return new Promise(resolve => {
            const onMd = () => {
                vidW = liveVideoEl.videoWidth;
                vidH = liveVideoEl.videoHeight;
                // Stereo heuristic — same as videos.js: aspect > 1.7 ⇒ side-by-side.
                isStereo = (vidW / vidH) > 1.7;
                liveVideoReady = true;
                _refreshActiveVideo();
                // Seek to mid-frame-0 so the first frame is decodable (t=0 isn't
                // always decodable on H.264 streams that start at a non-zero PTS).
                const fps = trialMeta?.fps || 30;
                liveVideoEl.currentTime = 0.5 / fps;
                liveVideoEl.addEventListener('seeked', () => {
                    render();
                    resolve();
                }, { once: true });
            };
            liveVideoEl.addEventListener('loadedmetadata', onMd, { once: true });
        });
    }

    function updateCameraControls() {
        // Show the OS/OD toggle only for stereo trials.  Single-camera
        // trials force ``side=full`` (the only sensible request) and
        // hide the toggle button.
        const isStereo = trajectory ? trajectory.is_stereo
                                      : (trialMeta?.is_stereo !== false);
        const wrap = $('cameraControls');
        if (!wrap) return;
        if (isStereo) {
            wrap.style.display = 'inline-flex';
            // Re-normalize away from any legacy 'full' state.
            if (currentSide !== 'OS' && currentSide !== 'OD') currentSide = 'OS';
            $('cameraLabel').textContent = currentSide;
        } else {
            wrap.style.display = 'none';
            currentSide = 'full';
        }
    }

    function switchCamera() {
        // Alternate OS ↔ OD.  No 'full' state in the rotation — single-
        // camera trials hide the toggle entirely.  All sources are
        // <video> elements now, so switching sides is just a re-render
        // with a different sub-rect crop.
        currentSide = (currentSide === 'OS') ? 'OD' : 'OS';
        updateCameraControls();
        render();
    }

    async function refreshTrajectoryFromServer() {
        if (subjectId == null || currentTrialIdx < 0) return;
        try {
            const t = await api(`/api/preproc/${subjectId}/trial/${trialMeta.trial_idx}/trajectory`);
            if (t.available) {
                trajectory = t;
                $('dot-trajectory').classList.add('done');
                $('dot-trajectory').classList.remove('running', 'failed');
                showTrajectoryStats();
            } else {
                trajectory = null;
                $('dot-trajectory').classList.remove('done', 'running', 'failed');
                $('trajectoryStatus').textContent = 'Not computed yet.';
            }
        } catch (e) {
            $('trajectoryStatus').textContent = `Load error: ${e.message}`;
        }
        updateCameraControls();
        // Trajectory readiness gates Stabilize.  Foreground is gated
        // separately on stable.mp4 existing -- see refreshBackgroundFromServer.
        const stableBtn = $('runStableBtn');
        if (stableBtn) stableBtn.disabled = !trajectory;
        drawPlot();
        render();
    }

    async function refreshBackgroundFromServer() {
        if (subjectId == null || currentTrialIdx < 0) return;
        // Reset cached overlay videos so any pending overlay redraw won't use stale data.
        if (stableVideoEl)  { stableVideoEl.removeAttribute('src');  stableVideoEl.load(); }
        outlineData = null;
        $('bgThumbOS').src = '';
        $('bgThumbOD').src = '';
        $('bgThumbOD').style.display = 'none';
        let b = null;
        try {
            b = await api(`/api/preproc/${subjectId}/trial/${trialMeta.trial_idx}/background`);
        } catch (e) {
            $('stableStatus').textContent = `Load error: ${e.message}`;
            return;
        }
        // Always store the response -- it carries stable_mp4_exists and
        // available independently, so the UI can show three states:
        // nothing / stabilized-only / background-done.
        backgroundData = b;
        const stableReady = !!b.stable_mp4_exists;
        const bgReady = !!b.available;   // background.npz exists

        if (bgReady) {
            $('dot-background').classList.add('done');
            $('dot-background').classList.remove('running', 'failed');
            showBackgroundStats();
            _loadBackgroundArtifacts();          // bg thumbs + stable video
            $('backgroundStatus').textContent = 'Done.';
        } else {
            $('dot-background').classList.remove('done', 'running', 'failed');
            $('backgroundStats').textContent = '';
            $('backgroundPreview').style.display = 'none';
            if (stableReady) {
                // Stabilize done, Background not yet -- still load the
                // stable video so the overlay works.
                _loadStableVideo();
                $('stableStatus').textContent = 'Done.';
                $('backgroundStatus').textContent = 'Not computed yet.';
            } else {
                $('stableStatus').textContent = trajectory
                    ? 'Not computed yet.'
                    : 'Waiting for trajectory (run step 1 first).';
                $('backgroundStatus').textContent =
                    'Waiting for stable.mp4 (run Stabilize first).';
            }
            $('outlineStatus').textContent = '';
            overlayMode = 'off';
            if ($('ovOff'))  $('ovOff').checked = true;
        }
        // Gating:
        //   Stabilize  -> needs trajectory
        //   Background -> needs stable.mp4
        //   overlays   -> ovStable needs stable.mp4, ovBg needs background.npz
        //   hand-boundary checkboxes -> need background.npz (compute_outline_frame
        //     reads it)
        $('runStableBtn').disabled = !trajectory;
        $('runBackgroundBtn').disabled = !stableReady;
        if ($('ovStable')) $('ovStable').disabled = !stableReady;
        if ($('ovBg'))     $('ovBg').disabled = !bgReady;
        $('cbShowOutline').disabled = !bgReady;
        $('cbShowFgFill').disabled = !bgReady;
        if (bgReady && (showOutline || showFgFill)) scheduleOutlineFetch();
    }

    function showBackgroundStats() {
        if (!backgroundData) return;
        const b = backgroundData;
        const lines = [
            `Bake: <strong>${b.frames_written}</strong> / ${b.n_frames} frames`,
            `BG sampled from: <strong>${b.n_samples_used}</strong> frames`,
        ];
        if (b.stable_mp4_exists) {
            lines.push(`stable.mp4: <strong>${b.stable_mp4_size_mb.toFixed(1)} MB</strong>`);
        } else {
            lines.push(`<span style="color:var(--red);">stable.mp4 missing</span>`);
        }
        lines.push(`OS MAD: p95=<strong>${b.mad_OS_p95.toFixed(1)}</strong>  mean=${b.mad_OS_mean.toFixed(1)}`);
        if (b.is_stereo) {
            lines.push(`OD MAD: p95=<strong>${b.mad_OD_p95.toFixed(1)}</strong>  mean=${b.mad_OD_mean.toFixed(1)}`);
        }
        $('backgroundStats').innerHTML = lines.join('<br>');
    }

    /** (Re)load the hidden stable.mp4 <video> element.  Safe to call
     *  whenever stable.mp4 exists -- doesn't depend on background.npz. */
    function _loadStableVideo() {
        if (!backgroundData || !backgroundData.stable_mp4_exists) return;
        const base = `/api/preproc/${subjectId}/trial/${trialMeta.trial_idx}`;
        const t = Date.now();
        const _makeVideo = () => {
            const v = document.createElement('video');
            v.muted = true; v.playsInline = true;
            v.preload = 'auto'; v.crossOrigin = 'anonymous';
            v.style.display = 'none';
            document.body.appendChild(v);
            return v;
        };
        if (!stableVideoEl) stableVideoEl = _makeVideo();
        stableVideoEl.src = `${base}/stable_video?_=${t}`;
        stableVideoEl.addEventListener('loadedmetadata', () => {
            if ($('ovStable')) $('ovStable').disabled = false;
        }, { once: true });
    }

    /** Load the background.npz-derived artifacts: BG thumbnails for the
     *  sidebar preview + the stable video.  Only call when
     *  ``backgroundData.available`` is true. */
    function _loadBackgroundArtifacts() {
        if (!backgroundData || !backgroundData.available) return;
        const base = `/api/preproc/${subjectId}/trial/${trialMeta.trial_idx}`;
        const t = Date.now();   // bust HTTP cache after recompute
        $('ovBg').disabled = true;
        $('bgThumbOS').addEventListener('load', () => {
            $('ovBg').disabled = false;
        }, { once: true });
        $('bgThumbOS').src = `${base}/background_image?side=OS&kind=bg&_=${t}`;
        if (backgroundData.is_stereo) {
            $('bgThumbOD').src = `${base}/background_image?side=OD&kind=bg&_=${t}`;
            $('bgThumbOD').style.display = '';
        } else {
            $('bgThumbOD').style.display = 'none';
        }
        $('backgroundPreview').style.display = '';
        _loadStableVideo();
    }

    /**
     * Spawn a Stabilize or Background job and stream its progress into
     * ``statusEl``.  Shared by computeStable + computeBackground.
     * ``extraBody`` merges into the POST body (e.g. palm_grow_px).
     */
    async function _runPreprocJob(endpoint, statusElId, extraBody = {}) {
        const statusEl = $(statusElId);
        try {
            const res = await api(`/api/preproc/${subjectId}/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ trial_idx: trialMeta.trial_idx, ...extraBody }),
            });
            bgJobId = res.job_id;
            $('dot-background').classList.add('running');
            $('dot-background').classList.remove('done', 'failed');
            statusEl.textContent = `Job ${bgJobId} starting…`;
            if (bgEvtSource) bgEvtSource.close();
            bgEvtSource = new EventSource(`/api/jobs/${bgJobId}/stream`);
            bgEvtSource.onmessage = async ev => {
                const d = JSON.parse(ev.data);
                if (d.status === 'running') {
                    const pct = d.progress_pct ? Math.round(d.progress_pct) : 0;
                    statusEl.textContent = `Running… ${pct}%`;
                } else if (d.status === 'completed') {
                    bgEvtSource.close(); bgEvtSource = null;
                    statusEl.textContent = 'Done.';
                    await refreshBackgroundFromServer();
                } else if (d.status === 'failed') {
                    bgEvtSource.close(); bgEvtSource = null;
                    $('dot-background').classList.remove('running');
                    $('dot-background').classList.add('failed');
                    statusEl.textContent = `Failed: ${d.error_msg || ''}`;
                } else if (d.status === 'cancelled') {
                    bgEvtSource.close(); bgEvtSource = null;
                    $('dot-background').classList.remove('running');
                    statusEl.textContent = 'Cancelled.';
                }
            };
        } catch (e) {
            statusEl.textContent = `Error: ${e.message}`;
        }
    }

    async function computeStable() {
        if (subjectId == null || currentTrialIdx < 0) return;
        if (!trajectory) {
            $('stableStatus').textContent = 'Run Compute Trajectory first.';
            return;
        }
        await _runPreprocJob('compute_stable', 'stableStatus');
    }

    async function computeBackground() {
        if (subjectId == null || currentTrialIdx < 0) return;
        if (!backgroundData || !backgroundData.stable_mp4_exists) {
            $('backgroundStatus').textContent = 'Run Stabilize first.';
            return;
        }
        const palmSlider = $('palmGrowSlider');
        const palmGrowPx = palmSlider ? parseInt(palmSlider.value, 10) : 15;
        const lenSlider = $('skinLeniencySlider');
        const skinLeniency = lenSlider ? parseFloat(lenSlider.value) : 1.0;
        await _runPreprocJob('compute_background', 'backgroundStatus',
                              { palm_grow_px: palmGrowPx,
                                skin_leniency: skinLeniency });
    }

    /**
     * Fetch the hand-boundary contour for the current frame at the
     * current dilation.  Live -- called whenever the frame or the
     * dilation slider changes (debounced via scheduleOutlineFetch).
     * Uses a monotonic sequence number so a stale late response
     * doesn't overwrite newer state.
     */
    async function fetchOutline() {
        if (subjectId == null || currentTrialIdx < 0) return;
        if (!backgroundData || !backgroundData.stable_mp4_exists) return;
        if (!showOutline && !showFgFill) return;
        const dilSlider = $('fgDilateSlider');
        const dilationPx = dilSlider ? parseInt(dilSlider.value, 10) : 14;
        const openSlider = $('fgOpenSlider');
        const openRadiusPx = openSlider ? parseInt(openSlider.value, 10) : 0;
        const frame = currentFrame;
        const seq = ++outlineFetchSeq;
        const includeFg = showFgFill ? 1 : 0;
        const url = `/api/preproc/${subjectId}/trial/${trialMeta.trial_idx}/outline_frame`
                  + `?frame=${frame}&dilation_px=${dilationPx}`
                  + `&open_radius_px=${openRadiusPx}&include_fg=${includeFg}`;
        $('outlineStatus').textContent = 'Updating…';
        try {
            const resp = await fetch(url);
            if (seq !== outlineFetchSeq) return;
            if (!resp.ok) {
                outlineData = null;
                $('outlineStatus').textContent = `HTTP ${resp.status}`;
            } else {
                outlineData = await resp.json();
                _updateFgImages(outlineData);
                const nOS = outlineData.OS ? outlineData.OS.length : 0;
                const nOD = outlineData.OD ? outlineData.OD.length : 0;
                $('outlineStatus').textContent =
                    `f${frame} d${dilationPx}px · OS ${nOS}pt` +
                    (outlineData.is_stereo ? ` · OD ${nOD}pt` : '') +
                    (includeFg ? ' · +fg' : '');
            }
        } catch (e) {
            if (seq === outlineFetchSeq) {
                outlineData = null;
                $('outlineStatus').textContent = `Error: ${e.message}`;
            }
        } finally {
            if (seq === outlineFetchSeq) render();
        }
    }

    /**
     * Decode the base64 JET-heatmap PNGs into HTMLImageElement objects
     * so the canvas drawImage() path stays synchronous.  Called when a
     * fresh outlineData response lands; clears the previous image if
     * include_fg was off.
     */
    function _updateFgImages(data) {
        for (const side of ['OS', 'OD']) {
            const pack = data['fg_' + side];
            if (pack && pack.b64) {
                const img = new Image();
                img.onload = () => render();
                img.src = 'data:image/png;base64,' + pack.b64;
                fgImageEls[side] = img;
                fgImageBboxes[side] = pack.bbox;
            } else {
                fgImageEls[side] = null;
                fgImageBboxes[side] = null;
            }
        }
    }

    /** Debounced fetch: coalesce slider drags + frame steps. */
    function scheduleOutlineFetch(delay = 150) {
        clearTimeout(outlineFetchTimer);
        outlineFetchTimer = setTimeout(fetchOutline, delay);
    }

    /**
     * Build (and cache) the live skin-color mask for the current
     * frame at the current Skin-colour-leniency setting.  Classifies
     * the displayed sub-rect with the same BT.601 YCrCb test the
     * server's _is_skin_ycc uses, scaled by ``leniency``.
     *
     * Returns an offscreen <canvas> with magenta where skin, fully
     * transparent elsewhere -- or null if the source can't be read
     * yet (cold video, tainted canvas).  Cached by frame + leniency +
     * source so playback / re-renders don't reclassify needlessly.
     */
    function _updateSkinMask(srcImage, drawSx, drawSy, drawSw, drawSh) {
        const lenSlider = $('skinLeniencySlider');
        const leniency = lenSlider ? parseFloat(lenSlider.value) : 1.0;
        const key = `${currentFrame}|${leniency}|${overlayMode}|`
                  + `${currentSide}|${drawSw}x${drawSh}`;
        if (key === _skinMaskKey && _skinMaskCanvas) return _skinMaskCanvas;
        // Classify at half-res -- plenty for a preview, ~4x faster.
        const cw = Math.max(1, Math.round(drawSw / 2));
        const ch = Math.max(1, Math.round(drawSh / 2));
        if (!_skinMaskCanvas) _skinMaskCanvas = document.createElement('canvas');
        _skinMaskCanvas.width = cw;
        _skinMaskCanvas.height = ch;
        const sctx = _skinMaskCanvas.getContext('2d', { willReadFrequently: true });
        let img;
        try {
            sctx.clearRect(0, 0, cw, ch);
            sctx.drawImage(srcImage, drawSx, drawSy, drawSw, drawSh, 0, 0, cw, ch);
            img = sctx.getImageData(0, 0, cw, ch);
        } catch (e) {
            _skinMaskKey = null;
            return null;          // video not ready / tainted -- skip
        }
        const d = img.data;
        // Universal YCrCb skin window (mirrors _SKIN_* in background.py),
        // half-width scaled by leniency around the centre.
        const CR_LO = 130, CR_HI = 175, CB_LO = 80, CB_HI = 130;
        const crC = (CR_LO + CR_HI) / 2, crH = (CR_HI - CR_LO) / 2 * leniency;
        const cbC = (CB_LO + CB_HI) / 2, cbH = (CB_HI - CB_LO) / 2 * leniency;
        for (let i = 0; i < d.length; i += 4) {
            const r = d[i], g = d[i + 1], b = d[i + 2];
            const y  = 0.299 * r + 0.587 * g + 0.114 * b;
            const cr = (r - y) * 0.713 + 128;
            const cb = (b - y) * 0.564 + 128;
            if (cr >= crC - crH && cr <= crC + crH &&
                cb >= cbC - cbH && cb <= cbC + cbH) {
                d[i] = 255; d[i + 1] = 0; d[i + 2] = 255; d[i + 3] = 130;
            } else {
                d[i + 3] = 0;
            }
        }
        sctx.putImageData(img, 0, 0);
        _skinMaskKey = key;
        return _skinMaskCanvas;
    }

    function showTrajectoryStats() {
        if (!trajectory) return;
        const njerk = trajectory.jerk_flag.filter(Boolean).length;
        const inlMedL = median(trajectory.n_inliers_L.filter(x => x > 0));
        const inlMedR = median(trajectory.n_inliers_R.filter(x => x > 0));
        const rotRange = (() => { const r = trajectory.OS.rot_deg;
            return [Math.min(...r), Math.max(...r)]; })();
        const txRange = (() => { const x = trajectory.OS.tx;
            return [Math.min(...x), Math.max(...x)]; })();
        const tyRange = (() => { const y = trajectory.OS.ty;
            return [Math.min(...y), Math.max(...y)]; })();
        $('trajectoryStats').innerHTML = [
            `Reference frame: <strong>${trajectory.reference_frame}</strong>`,
            `Jerk-flagged frames: <strong>${njerk}/${trajectory.n_frames}</strong>`,
            `Median inliers (OS / OD): <strong>${inlMedL} / ${inlMedR}</strong>`,
            `OS tx range: ${txRange[0].toFixed(1)} … ${txRange[1].toFixed(1)} px`,
            `OS ty range: ${tyRange[0].toFixed(1)} … ${tyRange[1].toFixed(1)} px`,
            `OS rotation range: ${rotRange[0].toFixed(2)} … ${rotRange[1].toFixed(2)}°`,
        ].join('<br>');
        if (typeof trajectory.os_od_rms_disagreement_px === 'number') {
            const r = trajectory.os_od_rms_disagreement_px;
            $('osodAgree').textContent = `OS/OD disagreement: ${r.toFixed(2)} px RMS`;
            $('osodAgree').style.color = r > 1.5 ? 'var(--red)' : (r > 0.5 ? 'var(--yellow)' : 'var(--green)');
        }
    }

    function median(arr) {
        if (!arr.length) return 0;
        const s = [...arr].sort((a, b) => a - b);
        const m = Math.floor(s.length / 2);
        return s.length % 2 ? s[m] : Math.round((s[m - 1] + s[m]) / 2);
    }

    async function computeTrajectory() {
        if (subjectId == null || currentTrialIdx < 0) return;
        try {
            const res = await api(`/api/preproc/${subjectId}/compute_trajectory`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ trial_idx: trialMeta.trial_idx }),
            });
            activeJobId = res.job_id;
            $('dot-trajectory').classList.add('running');
            $('dot-trajectory').classList.remove('done', 'failed');
            $('trajectoryStatus').textContent = `Job ${activeJobId} starting…`;
            if (activeEvtSource) activeEvtSource.close();
            activeEvtSource = new EventSource(`/api/jobs/${activeJobId}/stream`);
            activeEvtSource.onmessage = async ev => {
                const d = JSON.parse(ev.data);
                if (d.status === 'running') {
                    const pct = d.progress_pct ? Math.round(d.progress_pct) : 0;
                    $('trajectoryStatus').textContent = `Running… ${pct}%`;
                } else if (d.status === 'completed') {
                    activeEvtSource.close(); activeEvtSource = null;
                    $('trajectoryStatus').textContent = 'Done.';
                    await refreshTrajectoryFromServer();
                } else if (d.status === 'failed') {
                    activeEvtSource.close(); activeEvtSource = null;
                    $('dot-trajectory').classList.remove('running');
                    $('dot-trajectory').classList.add('failed');
                    $('trajectoryStatus').textContent = `Failed: ${d.error_msg || ''}`;
                } else if (d.status === 'cancelled') {
                    activeEvtSource.close(); activeEvtSource = null;
                    $('dot-trajectory').classList.remove('running');
                    $('trajectoryStatus').textContent = 'Cancelled.';
                }
            };
        } catch (e) {
            $('trajectoryStatus').textContent = `Error: ${e.message}`;
        }
    }

    function loadFrame(frameNum) {
        // Seek the active video to the requested frame and trigger a
        // redraw once the new frame is presented.  Returns a Promise that
        // resolves after the seek (or immediately if no video).
        currentFrame = frameNum;
        if (!trialMeta) return Promise.resolve();
        $('frameDisplay').textContent = `Frame: ${frameNum} / ${nFrames}`;
        // Hand-boundary / fg-heatmap refetch -- debounced so scrubbing isn't laggy.
        if (showOutline || showFgFill) scheduleOutlineFetch();
        _refreshActiveVideo();
        const companions = _companionVideos();
        if (!videoEl || videoEl.readyState < 1) return Promise.resolve();
        const fps = trialMeta?.fps || 30;
        const t = Math.max(0, Math.min((frameNum + 0.5) / fps,
                                        (videoEl.duration || 1) - 1e-3));
        // Keep companion tracks (outline when its checkbox is on) seeked
        // alongside.  No need to await their 'seeked' events — render()
        // reads from them lazily and any catch-up paint happens within
        // a frame or two.
        for (const c of companions) c.currentTime = t;
        return new Promise(resolve => {
            videoEl.currentTime = t;
            videoEl.addEventListener('seeked', () => {
                render();
                resolve();
            }, { once: true });
        });
    }

    // Step to an absolute frame (clamped).  Pauses playback so the user
    // takes over from the new position.
    function goToFrame(f) {
        if (!nFrames) return;
        f = Math.max(0, Math.min(nFrames - 1, f | 0));
        if (f === currentFrame) return;
        // If we're playing and the user manually steps, pause first —
        // otherwise the rAF loop fights with the seek for currentTime.
        if (playing) togglePlay();
        loadFrame(f);
        drawPlot();
    }

    // Base placement metrics: fit-scale + centered offsets for an image
    // of size (w, h) inside the canvas.  Zoom/pan transforms layer on top.
    function _getBaseMetrics(w, h) {
        const cw = canvas.width, ch = canvas.height;
        const bps = Math.min(cw / w, ch / h);
        return {
            bps,
            baseOX: (cw - w * bps) / 2,
            baseOY: (ch - h * bps) / 2,
        };
    }

    function render() {
        const rect = canvas.getBoundingClientRect();
        canvas.width  = rect.width  || 1;
        canvas.height = rect.height || 1;
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Pick the source image + dimensions based on overlay mode.
        // The stabilized + fg videos are already in reference coords so
        // no warp matrix is applied at draw time.  Either side of the
        // stereo pair is cropped from the source via a sub-rect draw.
        let srcImage = null, srcW = 0, srcH = 0, applyWarpH = false, labelText = null;
        let drawSx = 0, drawSy = 0, drawSw = 0, drawSh = 0;   // sub-rect of srcImage
        if (overlayMode === 'bg' && backgroundData) {
            // Static background PNG — the existing sidebar thumbnail
            // <img> is the same per-side PNG served by the API, already
            // loaded at full natural resolution.  drawImage on it uses
            // the natural dimensions regardless of CSS display size.
            const isOD = (currentSide === 'OD');
            const img = (isOD && backgroundData.is_stereo)
                ? $('bgThumbOD') : $('bgThumbOS');
            if (img && img.complete && img.naturalWidth > 0) {
                srcImage = img;
                drawSx = 0; drawSy = 0;
                drawSw = img.naturalWidth; drawSh = img.naturalHeight;
                srcW = drawSw; srcH = drawSh;
                labelText = 'Background  (temporal median)';
            }
        } else if (overlayMode !== 'off' && backgroundData) {
            // Stabilized overlay only -- fg.mp4 is gone (hand boundary
            // is computed on demand and drawn as a contour, not a
            // raster).  Skip the paint if the video isn't ready;
            // 'seeked'/'loadedmetadata' will re-render shortly.
            const vid = stableVideoEl;
            if (vid && vid.readyState >= 1 && vid.videoWidth > 0) {
                // Use the module-level isStereo (set from the live
                // video's aspect ratio).  backgroundData.is_stereo is
                // absent until Background runs, so relying on it left
                // stable.mp4 drawn full-width with no OS/OD crop.
                const fullW = vid.videoWidth, fullH = vid.videoHeight;
                const half = isStereo ? Math.floor(fullW / 2) : fullW;
                const isOD = (currentSide === 'OD');
                drawSx = (isStereo && isOD) ? half : 0;
                drawSy = 0;
                drawSw = isStereo ? half : fullW;
                drawSh = fullH;
                srcImage = vid;
                srcW = drawSw; srcH = drawSh;
                labelText = 'Stabilized  (warped to reference frame)';
            } else {
                return;
            }
        } else if (overlayMode === 'off'
                    && liveVideoEl && liveVideoReady
                    && liveVideoEl.readyState >= 2) {
            // Live source.  Stereo: crop the OS / OD half via sub-rect.
            const fullW = liveVideoEl.videoWidth;
            const fullH = liveVideoEl.videoHeight;
            const half = isStereo ? Math.floor(fullW / 2) : fullW;
            const isOD = (currentSide === 'OD');
            drawSx = (isStereo && isOD) ? half : 0;
            drawSy = 0;
            drawSw = isStereo ? half : fullW;
            drawSh = fullH;
            srcImage = liveVideoEl;
            srcW = drawSw; srcH = drawSh;
            applyWarpH = isStabilized && !!trajectory;
        }
        if (!srcImage) return;

        const { bps, baseOX, baseOY } = _getBaseMetrics(srcW, srcH);

        ctx.save();
        // Zoom/pan transform — pivot is the centered base placement.
        ctx.translate(baseOX + offsetX, baseOY + offsetY);
        ctx.scale(scale, scale);
        // After this transform, (0,0) is the top-left of the centered
        // base image (in unscaled image coords × bps).
        ctx.scale(bps, bps);

        if (applyWarpH) {
            const H = trajectoryWarpForCurrentFrame();
            if (H) {
                ctx.transform(H[0][0], H[1][0], H[0][1], H[1][1], H[0][2], H[1][2]);
            }
        }
        // Sub-rect draw — overlay-video sources need the OS/OD half
        // cropped out; live-frame source draws the whole image.
        ctx.drawImage(srcImage, drawSx, drawSy, drawSw, drawSh,
                                0,      0,      drawSw, drawSh);

        // Live skin-color mask -- magenta tint over every pixel the
        // current Skin colour leniency would classify as hand-coloured.
        // Client-side YCrCb classification (mirrors _is_skin_ycc), so
        // it tracks the leniency slider with zero latency.
        if (showSkinMask && srcImage) {
            const mask = _updateSkinMask(srcImage, drawSx, drawSy, drawSw, drawSh);
            if (mask) {
                ctx.save();
                ctx.imageSmoothingEnabled = false;
                ctx.drawImage(mask, 0, 0, drawSw, drawSh);
                ctx.restore();
            }
        }

        // Live foreground heatmap fill -- JET colormap of |frame - BG|
        // inside the gate, painted UNDER the outline polygon.  Image
        // is cropped to the gate bbox and positioned via outlineData.
        if (showFgFill && outlineData
            && outlineData.frame === currentFrame) {
            const side = (currentSide === 'OD' && outlineData.is_stereo)
                ? 'OD' : 'OS';
            const img = fgImageEls[side];
            const bbox = fgImageBboxes[side];
            if (img && img.complete && img.naturalWidth > 0 && bbox) {
                const [x0, y0, x1, y1] = bbox;
                ctx.save();
                ctx.globalAlpha = Math.max(0, Math.min(1, fgFillOpacity));
                ctx.drawImage(img, x0, y0, x1 - x0, y1 - y0);
                ctx.restore();
            }
        }

        // Live hand-boundary overlay -- draws the server-computed
        // closed polygon as a yellow stroked path, tracking the
        // current side's contour from outlineData.  No mp4 reads,
        // no compositing -- just a vector path that scales cleanly
        // with the zoom transform.
        if (showOutline && outlineData
            && outlineData.frame === currentFrame) {
            const pts = (currentSide === 'OD' && outlineData.is_stereo)
                ? outlineData.OD
                : outlineData.OS;
            if (pts && pts.length >= 3) {
                ctx.save();
                ctx.strokeStyle = 'rgba(255, 235, 59, 0.95)';   // MP-yellow
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                // Stay 2px wide on screen regardless of zoom.  We're
                // inside the (scale * bps) transform, so divide.
                ctx.lineWidth = 2 / Math.max(0.01, scale * bps);
                ctx.beginPath();
                ctx.moveTo(pts[0][0], pts[0][1]);
                for (let i = 1; i < pts.length; i++) {
                    ctx.lineTo(pts[i][0], pts[i][1]);
                }
                ctx.closePath();
                ctx.stroke();
                ctx.restore();
            }
        }

        // Dilated MP-skeleton preview, shown while the Foreground
        // section's dilation slider is in focus / being dragged so the
        // user can see what gate they're about to use.  Draws thick
        // strokes along every adjacent joint pair in each finger chain
        // plus the palm-MCP arc, line thickness = 2 x dilation_px.
        const _fgSlider = document.getElementById('fgDilateSlider');
        const _showDilPreview = _fgSlider
            && (document.activeElement === _fgSlider) && mpKeypoints;
        if (_showDilPreview) {
            const dilation = parseInt(_fgSlider.value, 10) || 14;
            // Choose coord system: live frame uses raw MP, every other
            // overlay (stable / fg / bg) draws on ref-space
            // pixels so we need the warped keypoints.
            const useRef = overlayMode !== 'off';
            const isOD = (currentSide === 'OD');
            const arr = useRef
                ? (isOD ? mpKeypoints.ref_OD : mpKeypoints.ref_OS)
                : (isOD ? mpKeypoints.raw_OD : mpKeypoints.raw_OS);
            const f = arr && currentFrame < arr.length ? arr[currentFrame] : null;
            if (f) {
                ctx.save();
                ctx.strokeStyle = 'rgba(255, 235, 59, 0.85)';   // MP-yellow
                ctx.fillStyle   = 'rgba(255, 235, 59, 0.85)';
                ctx.lineCap     = 'round';
                ctx.lineJoin    = 'round';
                ctx.lineWidth   = Math.max(1, dilation * 2);
                const _hasPt = j => f[j] && f[j][0] != null && f[j][1] != null;
                const _drawSeg = (a, b) => {
                    if (!_hasPt(a) || !_hasPt(b)) return;
                    ctx.beginPath();
                    ctx.moveTo(f[a][0], f[a][1]);
                    ctx.lineTo(f[b][0], f[b][1]);
                    ctx.stroke();
                };
                for (const chain of _MP_FINGER_CHAINS) {
                    for (let ci = 0; ci < chain.length - 1; ci++) {
                        _drawSeg(chain[ci], chain[ci + 1]);
                    }
                }
                for (let i = 0; i < _MP_PALM_CHAIN.length - 1; i++) {
                    _drawSeg(_MP_PALM_CHAIN[i], _MP_PALM_CHAIN[i + 1]);
                }
                // Dots at each joint (radius = dilation so the dot
                // matches the line-end semicircle visually).
                for (let j = 0; j < 21; j++) {
                    if (!_hasPt(j)) continue;
                    ctx.beginPath();
                    ctx.arc(f[j][0], f[j][1], dilation, 0, Math.PI * 2);
                    ctx.fill();
                }
                // Synthetic ulnar-heel point: reflected thumb CMC,
                // closing the pinky-side palm edge (matches the
                // server's _build_kpt_hand_region).
                const _refl = _reflectThumbCmc(f);
                if (_refl) {
                    const _segTo = j => {
                        if (!_hasPt(j)) return;
                        ctx.beginPath();
                        ctx.moveTo(f[j][0], f[j][1]);
                        ctx.lineTo(_refl[0], _refl[1]);
                        ctx.stroke();
                    };
                    _segTo(17); _segTo(1);
                    ctx.beginPath();
                    ctx.arc(_refl[0], _refl[1], dilation, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            }
        }

        // Background-zone preview: palm zone (grown MP region) +
        // forearm cone, drawn from the current frame's MP keypoints so
        // the user can see what region Compute Background will operate
        // on BEFORE running it.  Approximate -- the bake unions over
        // all sample frames -- but good enough to dial the sliders.
        if (showBgZones && mpKeypoints) {
            const useRef = overlayMode !== 'off';
            const isOD = (currentSide === 'OD');
            const arr = useRef
                ? (isOD ? mpKeypoints.ref_OD : mpKeypoints.ref_OS)
                : (isOD ? mpKeypoints.raw_OD : mpKeypoints.raw_OS);
            const f = arr && currentFrame < arr.length ? arr[currentFrame] : null;
            if (f) {
                const _hasPt = j => f[j] && f[j][0] != null && f[j][1] != null;
                const palmSlider = $('palmGrowSlider');
                const palmGrow = palmSlider ? parseInt(palmSlider.value, 10) : 15;
                // Server builds the palm zone from the BG-mask gate
                // (~dilation 14 full-res) then dilates by palm_grow_px.
                const palmThick = Math.max(2, (14 + palmGrow) * 2);

                ctx.save();
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                // 1. Palm zone -- translucent green strokes along the
                //    MP skeleton + palm arc.
                ctx.strokeStyle = 'rgba(0, 230, 0, 0.28)';
                ctx.lineWidth = palmThick;
                const _seg = (a, b) => {
                    if (!_hasPt(a) || !_hasPt(b)) return;
                    ctx.beginPath();
                    ctx.moveTo(f[a][0], f[a][1]);
                    ctx.lineTo(f[b][0], f[b][1]);
                    ctx.stroke();
                };
                for (const chain of _MP_FINGER_CHAINS)
                    for (let ci = 0; ci < chain.length - 1; ci++)
                        _seg(chain[ci], chain[ci + 1]);
                for (let i = 0; i < _MP_PALM_CHAIN.length - 1; i++)
                    _seg(_MP_PALM_CHAIN[i], _MP_PALM_CHAIN[i + 1]);
                // Synthetic ulnar-heel point closing the pinky-side
                // palm edge (matches _build_kpt_hand_region).
                const _zrefl = _reflectThumbCmc(f);
                if (_zrefl) {
                    for (const j of [17, 1]) {
                        if (!_hasPt(j)) continue;
                        ctx.beginPath();
                        ctx.moveTo(f[j][0], f[j][1]);
                        ctx.lineTo(_zrefl[0], _zrefl[1]);
                        ctx.stroke();
                    }
                }

                // 2. Forearm cone -- trapezoid positioned ON the
                //    palm-heel line (midpoint of thumb CMC <->
                //    reflected ulnar heel) but kept WIDE: base width
                //    = MCP knuckle spread x 1.2, widening toward the
                //    elbow.  Mirrors _build_forearm_cone.
                const mcpIdx = [5, 9, 13, 17].filter(_hasPt);
                const _crefl = _reflectThumbCmc(f);
                if (_hasPt(0) && mcpIdx.length && _hasPt(1) && _crefl) {
                    const wrist = f[0];
                    let cx = 0, cy = 0;
                    for (const j of mcpIdx) { cx += f[j][0]; cy += f[j][1]; }
                    cx /= mcpIdx.length; cy /= mcpIdx.length;
                    let dx = wrist[0] - cx, dy = wrist[1] - cy;
                    const dn = Math.hypot(dx, dy);
                    if (dn > 1e-3) {
                        dx /= dn; dy /= dn;
                        const px = -dy, py = dx;          // perpendicular
                        // Base sits at the heel-line midpoint.
                        const baseMid = [(f[1][0] + _crefl[0]) / 2,
                                          (f[1][1] + _crefl[1]) / 2];
                        // Width = MCP spread x 1.2 (wide, like before).
                        let baseW = 30;
                        if (_hasPt(5) && _hasPt(17)) {
                            baseW = Math.hypot(f[5][0] - f[17][0],
                                                f[5][1] - f[17][1]) * 1.2;
                        }
                        const tipW = baseW * 1.8;
                        const len = 220;                  // full-res px
                        const ex = baseMid[0] + dx * len;
                        const ey = baseMid[1] + dy * len;
                        const baseA = [baseMid[0] + px * baseW / 2, baseMid[1] + py * baseW / 2];
                        const baseB = [baseMid[0] - px * baseW / 2, baseMid[1] - py * baseW / 2];
                        const tipA  = [ex + px * tipW / 2, ey + py * tipW / 2];
                        const tipB  = [ex - px * tipW / 2, ey - py * tipW / 2];
                        ctx.fillStyle = 'rgba(0, 200, 255, 0.18)';
                        ctx.strokeStyle = 'rgba(0, 200, 255, 0.7)';
                        ctx.lineWidth = 2 / Math.max(0.01, scale * bps);
                        ctx.beginPath();
                        ctx.moveTo(baseA[0], baseA[1]);
                        ctx.lineTo(baseB[0], baseB[1]);
                        ctx.lineTo(tipB[0], tipB[1]);
                        ctx.lineTo(tipA[0], tipA[1]);
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                    }
                }
                ctx.restore();
            }
        }
        ctx.restore();

        if (labelText) {
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 11px sans-serif';
            ctx.fillText(labelText, 10, 16);
        }
        if (showJerks && trajectory && trajectory.jerk_flag[currentFrame]) {
            ctx.save();
            ctx.strokeStyle = '#e53935';
            ctx.lineWidth = 4;
            ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
            ctx.fillStyle = '#e53935';
            ctx.font = '12px sans-serif';
            ctx.fillText('jerk frame', 10, 18);
            ctx.restore();
        }
        if (trajectory && currentFrame === trajectory.reference_frame) {
            ctx.save();
            ctx.fillStyle = 'rgba(76,175,80,0.85)';
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText('REFERENCE FRAME', 10, canvas.height - 10);
            ctx.restore();
        }
    }

    function _resetZoom() {
        scale = 1; offsetX = 0; offsetY = 0;
        render();
    }

    function trajectoryWarpForCurrentFrame() {
        if (!trajectory) return null;
        // Pick the per-camera homography matching the displayed side.
        // OD camera uses H_to_ref_R; OS or full uses H_to_ref_L.
        const useOD = (currentSide === 'OD' && trajectory.OD);
        const series = useOD ? trajectory.OD : trajectory.OS;
        if (!series) return null;
        const i = currentFrame;
        const tx = series.tx[i] || 0;
        const ty = series.ty[i] || 0;
        const rotRad = (series.rot_deg[i] || 0) * Math.PI / 180;
        const c = Math.cos(rotRad), s = Math.sin(rotRad);
        return [[c, -s, tx], [s, c, ty], [0, 0, 1]];
    }

    function drawPlot() {
        const rect = plot.getBoundingClientRect();
        plot.width  = rect.width  || 1;
        plot.height = rect.height || 1;
        pctx.fillStyle = getCss('--bg-card', '#1a1a1a');
        pctx.fillRect(0, 0, plot.width, plot.height);
        if (!trajectory) {
            pctx.fillStyle = getCss('--text-muted', '#888');
            pctx.font = '12px sans-serif';
            pctx.fillText('No trajectory yet — click "Compute Trajectory" to extract.',
                          10, plot.height / 2);
            return;
        }
        const N = trajectory.n_frames;
        if (N < 2) return;
        const w = plot.width, h = plot.height;
        const pad = { l: 30, r: 5, t: 8, b: 18 };
        const innerW = w - pad.l - pad.r;
        const innerH = h - pad.t - pad.b;

        const series = [];
        series.push({ vals: trajectory.OS.tx, color: '#ff6b6b' });
        series.push({ vals: trajectory.OS.ty, color: '#ffa94d' });
        if (trajectory.OD) {
            series.push({ vals: trajectory.OD.tx, color: '#4dabf7' });
            series.push({ vals: trajectory.OD.ty, color: '#74c0fc' });
        }
        const rotSeries = trajectory.OS.rot_deg;

        let ymin = Infinity, ymax = -Infinity;
        for (const s of series) {
            for (const v of s.vals) {
                if (v < ymin) ymin = v;
                if (v > ymax) ymax = v;
            }
        }
        if (!isFinite(ymin) || !isFinite(ymax) || ymin === ymax) { ymin -= 1; ymax += 1; }
        const yPad = (ymax - ymin) * 0.05;
        ymin -= yPad; ymax += yPad;

        const xToPx = i => pad.l + (i / (N - 1)) * innerW;
        const yToPx = v => pad.t + ((ymax - v) / (ymax - ymin)) * innerH;

        pctx.strokeStyle = getCss('--border', '#333');
        pctx.lineWidth = 1;
        pctx.beginPath();
        pctx.moveTo(pad.l, pad.t); pctx.lineTo(pad.l, h - pad.b);
        pctx.lineTo(w - pad.r, h - pad.b);
        pctx.stroke();

        // ── X-axis tick marks + labels (seconds) ────────────────────
        // Pick a tick step that gives ~6–10 labelled ticks across the
        // trial duration.  Falls back to frame indices if fps is
        // missing or non-positive.
        const fps = trialMeta?.fps || 0;
        const totalSec = fps > 0 ? (N - 1) / fps : 0;
        let tickStepSec = 1;
        if (totalSec > 0) {
            const candidates = [0.5, 1, 2, 5, 10, 15, 30, 60];
            for (const c of candidates) {
                if (totalSec / c <= 10) { tickStepSec = c; break; }
                tickStepSec = c;  // fall through, keep widening
            }
        }
        pctx.fillStyle = getCss('--text-muted', '#888');
        pctx.font = '10px sans-serif';
        pctx.strokeStyle = getCss('--border', '#333');
        pctx.lineWidth = 1;
        pctx.textAlign = 'center';
        pctx.textBaseline = 'top';
        if (fps > 0) {
            for (let sec = 0; sec <= totalSec + 1e-6; sec += tickStepSec) {
                const f = sec * fps;
                if (f < 0 || f > N - 1) continue;
                const px = pad.l + (f / (N - 1)) * innerW;
                pctx.beginPath();
                pctx.moveTo(px, h - pad.b);
                pctx.lineTo(px, h - pad.b + 4);
                pctx.stroke();
                const label = tickStepSec >= 1 ? `${sec.toFixed(0)}s` : `${sec.toFixed(1)}s`;
                pctx.fillText(label, px, h - pad.b + 5);
            }
        } else {
            // Frame-index fallback: ~8 evenly-spaced labelled ticks.
            const steps = 8;
            for (let i = 0; i <= steps; i++) {
                const f = Math.round((i / steps) * (N - 1));
                const px = pad.l + (f / (N - 1)) * innerW;
                pctx.beginPath();
                pctx.moveTo(px, h - pad.b);
                pctx.lineTo(px, h - pad.b + 4);
                pctx.stroke();
                pctx.fillText(`f${f}`, px, h - pad.b + 5);
            }
        }
        pctx.textAlign = 'left';
        pctx.textBaseline = 'alphabetic';

        if (ymin < 0 && ymax > 0) {
            const zy = yToPx(0);
            pctx.strokeStyle = 'rgba(255,255,255,0.15)';
            pctx.beginPath(); pctx.moveTo(pad.l, zy); pctx.lineTo(w - pad.r, zy); pctx.stroke();
        }
        pctx.fillStyle = getCss('--text-muted', '#888');
        pctx.font = '10px sans-serif';
        pctx.fillText(ymax.toFixed(1), 2, pad.t + 8);
        pctx.fillText(ymin.toFixed(1), 2, h - pad.b);

        if (showJerks) {
            pctx.strokeStyle = 'rgba(229,57,53,0.45)';
            pctx.lineWidth = 1;
            for (let i = 0; i < N; i++) {
                if (trajectory.jerk_flag[i]) {
                    const x = xToPx(i);
                    pctx.beginPath();
                    pctx.moveTo(x, pad.t); pctx.lineTo(x, h - pad.b);
                    pctx.stroke();
                }
            }
        }

        const refX = xToPx(trajectory.reference_frame);
        pctx.strokeStyle = 'rgba(76,175,80,0.9)';
        pctx.lineWidth = 1.5;
        pctx.setLineDash([4, 3]);
        pctx.beginPath(); pctx.moveTo(refX, pad.t); pctx.lineTo(refX, h - pad.b); pctx.stroke();
        pctx.setLineDash([]);

        for (const s of series) {
            pctx.strokeStyle = s.color;
            pctx.lineWidth = 1.2;
            pctx.beginPath();
            let started = false;
            for (let i = 0; i < N; i++) {
                const v = s.vals[i];
                if (!isFinite(v)) { started = false; continue; }
                const px = xToPx(i), py = yToPx(v);
                if (!started) { pctx.moveTo(px, py); started = true; }
                else           pctx.lineTo(px, py);
            }
            pctx.stroke();
        }

        let rmin = Infinity, rmax = -Infinity;
        for (const v of rotSeries) { if (v < rmin) rmin = v; if (v > rmax) rmax = v; }
        if (rmax - rmin < 0.001) { rmin -= 0.5; rmax += 0.5; }
        const rotPad = 0.05 * (rmax - rmin);
        const rotMin = rmin - rotPad, rotMax = rmax + rotPad;
        const rotY = v => pad.t + ((rotMax - v) / (rotMax - rotMin)) * innerH;
        pctx.strokeStyle = '#fbc02d';
        pctx.lineWidth = 1.2;
        pctx.setLineDash([2, 2]);
        pctx.beginPath();
        let started = false;
        for (let i = 0; i < N; i++) {
            const v = rotSeries[i];
            if (!isFinite(v)) { started = false; continue; }
            const px = xToPx(i), py = rotY(v);
            if (!started) { pctx.moveTo(px, py); started = true; }
            else           pctx.lineTo(px, py);
        }
        pctx.stroke();
        pctx.setLineDash([]);
        pctx.fillStyle = '#fbc02d';
        pctx.fillText(`rot ${rmin.toFixed(2)}° … ${rmax.toFixed(2)}°`, w - 130, pad.t + 10);

        const cx = xToPx(currentFrame);
        pctx.strokeStyle = '#fff';
        pctx.lineWidth = 1;
        pctx.beginPath(); pctx.moveTo(cx, pad.t); pctx.lineTo(cx, h - pad.b); pctx.stroke();
    }

    function getCss(name, fallback) {
        try {
            const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
            return v || fallback;
        } catch { return fallback; }
    }

    // The one video element we're currently displaying & controlling.
    // Set by _refreshActiveVideo() whenever overlayMode changes.  All
    // playback ops (play, pause, seek, rate) target this and only this.
    let videoEl = null;

    function _refreshActiveVideo() {
        const prev = videoEl;
        if (overlayMode === 'stable' && stableVideoEl?.readyState >= 1) videoEl = stableVideoEl;
        else                                                             videoEl = liveVideoEl;
        return prev !== videoEl;
    }

    /** Videos that must be seeked / played in lockstep with ``videoEl``.
     *  Hand boundary is now a vector overlay -- no companion video.
     */
    function _companionVideos() { return []; }

    // Mirrors mano.js exactly — one videoEl, rVFC when available,
    // playbackRate clamped to 16 (browsers get unreliable above that).
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
        _refreshActiveVideo();
        const companions = _companionVideos();
        if (playing) {
            playing = false;
            if (videoEl) videoEl.pause();
            for (const c of companions) c.pause();
            _cancelPlayTimer();
            $('playBtn').innerHTML = '&#9654;';
            // Re-sync currentFrame from the video's final position.
            if (videoEl && trialMeta) {
                const fps = trialMeta.fps || 30;
                const f = Math.round(videoEl.currentTime * fps);
                currentFrame = Math.max(0, Math.min(f, nFrames - 1));
                $('frameDisplay').textContent = `Frame: ${currentFrame} / ${nFrames}`;
                drawPlot();
                render();
            }
            return;
        }
        if (!videoEl || videoEl.readyState < 2 || !nFrames) return;
        if (currentFrame >= nFrames - 1) currentFrame = 0;
        playing = true;
        $('playBtn').innerHTML = '&#9646;&#9646;';
        const fps = trialMeta?.fps || 30;
        const rate = Math.min(playbackRate, 16);
        const t = (currentFrame + 0.5) / fps;
        videoEl.currentTime = t;
        videoEl.playbackRate = rate;
        for (const c of companions) {
            c.currentTime = t;
            c.playbackRate = rate;
        }
        // play() returns a Promise that resolves once playback actually
        // starts.  Re-apply playbackRate then — setting it pre-play()
        // doesn't always stick across the seek→play state transition.
        videoEl.play().then(() => {
            if (videoEl) videoEl.playbackRate = rate;
        }).catch(() => {
            playing = false; $('playBtn').innerHTML = '&#9654;';
        });
        for (const c of companions) c.play().catch(() => {});
        _schedulePlayLoop();
    }

    function _schedulePlayLoop() {
        if (_hasRVFC && videoEl) {
            playTimer = videoEl.requestVideoFrameCallback(_playLoopRVFC);
        } else {
            playTimer = requestAnimationFrame(_playLoopRAF);
        }
    }

    function _playLoopRVFC(now, metadata) {
        if (!playing) return;
        _playUpdate(metadata.mediaTime);
        _schedulePlayLoop();
    }

    function _playLoopRAF() {
        if (!playing) return;
        _playUpdate(videoEl ? videoEl.currentTime : 0);
        _schedulePlayLoop();
    }

    function _playUpdate(mediaTime) {
        if (!trialMeta) return;
        const fps = trialMeta.fps || 30;
        const f = Math.round(mediaTime * fps);
        if (f > nFrames - 1) {
            togglePlay();
            return;
        }
        if (f !== currentFrame && f >= 0 && f < nFrames) {
            currentFrame = f;
            $('frameDisplay').textContent = `Frame: ${currentFrame} / ${nFrames}`;
            drawPlot();
        }
        render();
    }

    function prevSubject() {
        const sel = $('subjectSelect');
        const idx = subjects.findIndex(s => s.id === subjectId);
        if (idx > 0) {
            sel.value = subjects[idx - 1].id;
            onSubjectChange(subjects[idx - 1].id);
        }
    }

    function nextSubject() {
        const sel = $('subjectSelect');
        const idx = subjects.findIndex(s => s.id === subjectId);
        if (idx < subjects.length - 1) {
            sel.value = subjects[idx + 1].id;
            onSubjectChange(subjects[idx + 1].id);
        }
    }

    document.addEventListener('DOMContentLoaded', async () => {
        await loadSubjects();
        $('prevSubjectBtn').addEventListener('click', prevSubject);
        $('nextSubjectBtn').addEventListener('click', nextSubject);
        $('runTrajectoryBtn').addEventListener('click', computeTrajectory);
        // Three baked-or-derived stages: Stabilize (stable.mp4),
        // Background (background.npz), Hand Boundary (live, on demand).
        $('runStableBtn').addEventListener('click', computeStable);
        $('runBackgroundBtn').addEventListener('click', computeBackground);
        // Palm-zone grow slider: updates its label AND re-renders so
        // the on-canvas zone preview tracks it.  The value is read
        // when Compute Background is clicked (it's a bake param).
        const _palmGrowSlider = $('palmGrowSlider');
        const _palmGrowVal    = $('palmGrowVal');
        if (_palmGrowSlider && _palmGrowVal) {
            _palmGrowSlider.addEventListener('input', () => {
                _palmGrowVal.textContent = _palmGrowSlider.value;
                try { render(); } catch (_e) {}
            });
        }
        // Skin-leniency slider: it's a Background bake param (read when
        // Compute Background is clicked), but it also drives the live
        // skin-color mask preview -- so re-render on every input so the
        // magenta overlay tracks the slider with zero latency.
        const _skinLenSlider = $('skinLeniencySlider');
        const _skinLenVal    = $('skinLeniencyVal');
        if (_skinLenSlider && _skinLenVal) {
            _skinLenSlider.addEventListener('input', () => {
                _skinLenVal.textContent =
                    parseFloat(_skinLenSlider.value).toFixed(2);
                if (showSkinMask) try { render(); } catch (_e) {}
            });
        }
        // Preview-zones checkbox: pure render toggle.
        $('cbPreviewZones').addEventListener('change', e => {
            showBgZones = e.target.checked;
            try { render(); } catch (_e) {}
        });
        // Skin-mask checkbox: pure render toggle (classification is
        // cached, so toggling on is instant after the first compute).
        $('cbShowSkinMask').addEventListener('change', e => {
            showSkinMask = e.target.checked;
            try { render(); } catch (_e) {}
        });
        const _fgDilateSlider = $('fgDilateSlider');
        const _fgDilateVal    = $('fgDilateVal');
        if (_fgDilateSlider && _fgDilateVal) {
            _fgDilateSlider.addEventListener('input', () => {
                _fgDilateVal.textContent = _fgDilateSlider.value;
                // Instant feedback: re-render so the on-canvas
                // dilated-skeleton preview tracks the slider.
                try { render(); } catch (_e) {}
                // Debounced backend fetch for the new outline / heatmap.
                if (showOutline || showFgFill) scheduleOutlineFetch();
            });
        }
        // Strand-clip slider: live -- refetch the outline so the
        // morphological-open radius takes effect immediately.
        const _fgOpenSlider = $('fgOpenSlider');
        const _fgOpenVal    = $('fgOpenVal');
        if (_fgOpenSlider && _fgOpenVal) {
            _fgOpenSlider.addEventListener('input', () => {
                _fgOpenVal.textContent = _fgOpenSlider.value;
                if (showOutline || showFgFill) scheduleOutlineFetch();
            });
        }
        // Overlay radio group → mutually exclusive: live, stable.mp4.
        // Switching swaps which video element is the active source.  If
        // playback was running, transfer it to the new source seamlessly.
        const _setOverlay = (mode) => {
            const wasPlaying = playing;
            const fps = trialMeta?.fps || 30;
            const t = (currentFrame + 0.5) / fps;
            // Pause the prior source so it doesn't keep running invisibly.
            if (wasPlaying && videoEl) {
                videoEl.pause();
                _cancelPlayTimer();
                playing = false;
            }
            overlayMode = mode;
            _refreshActiveVideo();
            // Seek the new active video to the current frame and render
            // once the seek completes — otherwise we paint while the
            // video is still in seeking state, leaving the canvas blank
            // until something else triggers a render.
            const _onReady = () => {
                if (wasPlaying) togglePlay();
                else render();
            };
            if (videoEl && videoEl.readyState >= 1) {
                const targetT = Math.max(0, Math.min(t, (videoEl.duration || t) - 1e-3));
                // Already there? render synchronously.  Otherwise wait for seeked.
                if (Math.abs(videoEl.currentTime - targetT) < 1 / (fps * 2)) {
                    _onReady();
                } else {
                    videoEl.addEventListener('seeked', _onReady, { once: true });
                    videoEl.currentTime = targetT;
                }
            } else {
                _onReady();
            }
        };
        $('ovOff').addEventListener('change',      e => e.target.checked && _setOverlay('off'));
        $('ovStable').addEventListener('change',   e => e.target.checked && _setOverlay('stable'));
        $('ovBg').addEventListener('change',       e => e.target.checked && _setOverlay('bg'));
        // Hand-boundary checkbox: trigger a fresh fetch when turned on
        // so the user sees the contour for the current frame
        // immediately; clear the cached data when turned off.
        $('cbShowOutline').addEventListener('change', e => {
            showOutline = e.target.checked;
            if (showOutline) {
                fetchOutline();
            } else {
                if (!showFgFill) outlineData = null;
                render();
            }
        });
        // Foreground-fill checkbox: independent of the outline checkbox
        // -- you can show either one alone or both together.  Turning
        // it on triggers a fresh fetch with include_fg=1 (the heatmap
        // PNGs aren't included in plain outline fetches).
        $('cbShowFgFill').addEventListener('change', e => {
            showFgFill = e.target.checked;
            if (showFgFill) {
                fetchOutline();
            } else {
                fgImageEls.OS = fgImageEls.OD = null;
                fgImageBboxes.OS = fgImageBboxes.OD = null;
                render();
            }
        });
        // Opacity slider: no refetch, just a re-render with the new
        // globalAlpha.  The image bytes don't change.
        const _opacitySlider = $('fgOpacitySlider');
        const _opacityVal    = $('fgOpacityVal');
        if (_opacitySlider && _opacityVal) {
            _opacitySlider.addEventListener('input', () => {
                fgFillOpacity = parseInt(_opacitySlider.value, 10) / 100;
                _opacityVal.textContent = fgFillOpacity.toFixed(2);
                try { render(); } catch (_e) {}
            });
        }
        // Plot click + drag scrubs frames.  Pointer-x is mapped from
        // the canvas's pixel-padding to a frame index using the same
        // ``pad.l`` margin the plot uses.
        let scrubbing = false;
        const _frameAtClientX = (clientX) => {
            const N = trajectory ? trajectory.n_frames : nFrames;
            if (!N || N < 2) return 0;
            const rect = plot.getBoundingClientRect();
            const padL = 30, padR = 5;
            const x = clientX - rect.left;
            const innerW = (rect.width - padL - padR) || 1;
            const frac = (x - padL) / innerW;
            const f = Math.round(Math.max(0, Math.min(1, frac)) * (N - 1));
            return f;
        };
        plot.addEventListener('mousedown', e => {
            scrubbing = true;
            const f = _frameAtClientX(e.clientX);
            loadFrame(f); drawPlot();
        });
        plot.addEventListener('mousemove', e => {
            if (!scrubbing) return;
            const f = _frameAtClientX(e.clientX);
            if (f !== currentFrame) { loadFrame(f); drawPlot(); }
        });
        window.addEventListener('mouseup', () => { scrubbing = false; });
        plot.addEventListener('mouseleave', () => { scrubbing = false; });
        $('playBtn').addEventListener('click', togglePlay);
        $('prevFrameBtn').addEventListener('click', () => goToFrame(currentFrame - 1));
        $('nextFrameBtn').addEventListener('click', () => goToFrame(currentFrame + 1));
        $('sideToggle').addEventListener('click', switchCamera);

        // Speed slider (matches videos.js / mano.js pattern).
        const speedSlider = $('speedSlider');
        const speedDisplay = $('speedDisplay');
        if (speedSlider) {
            speedSlider.value = 2;  // default → 1x (SPEED_PRESETS[2])
            playbackRate = SPEED_PRESETS[2];
            speedDisplay.textContent = playbackRate + 'x';
            speedSlider.max = String(SPEED_PRESETS.length - 1);
            speedSlider.addEventListener('input', () => {
                const idx = parseInt(speedSlider.value);
                playbackRate = SPEED_PRESETS[idx] ?? 1;
                speedDisplay.textContent = playbackRate + 'x';
                // Browsers cap reliable playbackRate around 16×; clamp to match.
                _refreshActiveVideo();
                const rate = Math.min(playbackRate, 16);
                if (videoEl) videoEl.playbackRate = rate;
                for (const c of _companionVideos()) c.playbackRate = rate;
            });
            speedSlider.addEventListener('change', () => speedSlider.blur());
        }

        // Sidebar buttons / radios / checkboxes shouldn't trap keyboard
        // focus — otherwise Space re-clicks the button instead of
        // toggling play.  Mirror mano.js: blur after pointerup so the
        // page-level shortcuts keep working.
        ['pointerup', 'mouseup', 'touchend'].forEach(evt => {
            document.addEventListener(evt, e => {
                const t = e.target;
                if (t instanceof HTMLButtonElement) { t.blur(); return; }
                if (t instanceof HTMLInputElement &&
                    (t.type === 'range' || t.type === 'checkbox' || t.type === 'radio')) {
                    t.blur();
                }
            }, true);
        });
        // If a range slider still has focus when arrow keys are pressed
        // (clicked but mouse not yet released), prevent the slider's
        // native arrow-key handler from consuming them.
        document.addEventListener('keydown', e => {
            const t = e.target;
            if (t instanceof HTMLInputElement && t.type === 'range' &&
                ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown',
                 'PageUp','PageDown','Home','End'].includes(e.key)) {
                e.preventDefault();
                t.blur();
            }
        }, true);

        // Keyboard shortcuts:
        //   Space          → play/pause
        //   ← / a          → prev frame   ( Shift = jump 10 )
        //   → / s          → next frame   ( Shift = jump 10 )
        //   Home / End     → jump to first / last frame
        //   E              → toggle camera side
        //   R              → reset zoom
        // Skip text inputs and selects, but keep shortcuts working when a
        // checkbox / radio / range is focused (and blur it after).
        document.addEventListener('keydown', e => {
            const t = e.target;
            if (t.tagName === 'INPUT' &&
                !['checkbox', 'radio', 'range'].includes(t.type)) return;
            if (t.tagName === 'SELECT' || t.tagName === 'TEXTAREA') return;
            const step = e.shiftKey ? 10 : 1;
            let handled = false;
            switch (e.key) {
                case ' ':
                    togglePlay(); handled = true; break;
                case 'ArrowLeft': case 'a': case 'A':
                    goToFrame(currentFrame - step); handled = true; break;
                case 'ArrowRight': case 's': case 'S':
                    goToFrame(currentFrame + step); handled = true; break;
                case 'Home':
                    goToFrame(0); handled = true; break;
                case 'End':
                    goToFrame(nFrames - 1); handled = true; break;
                case 'e': case 'E':
                    switchCamera(); handled = true; break;
                case 'r': case 'R':
                    _resetZoom(); handled = true; break;
            }
            if (handled) {
                e.preventDefault();
                if (t.tagName === 'INPUT' || t.tagName === 'BUTTON') t.blur();
            }
        });
        // Zoom (mouse wheel) + pan (drag) on the video canvas — same
        // conventions as the video browser and mano page.
        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            // Pick dimensions matching whatever the renderer is showing.
            let srcW = 0, srcH = 0;
            _refreshActiveVideo();
            if (videoEl && videoEl.videoWidth > 0) {
                srcW = isStereo ? Math.floor(videoEl.videoWidth / 2) : videoEl.videoWidth;
                srcH = videoEl.videoHeight;
            }
            if (!srcW || !srcH) return;
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const { baseOX, baseOY } = _getBaseMetrics(srcW, srcH);
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
                canvas.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });
        document.addEventListener('mousemove', e => {
            if (!dragging) return;
            offsetX = panStartOX + (e.clientX - dragStartX);
            offsetY = panStartOY + (e.clientY - dragStartY);
            render();
        });
        document.addEventListener('mouseup', () => {
            if (dragging) { dragging = false; canvas.style.cursor = ''; }
        });
        canvas.style.cursor = 'grab';
        window.addEventListener('resize', () => { render(); drawPlot(); });
    });
})();
