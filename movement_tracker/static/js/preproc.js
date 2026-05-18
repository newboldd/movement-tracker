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
    // as labels.js / videos.js, which is what makes smooth playback work.
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
    let overlayMode = 'off';    // 'off' | 'stable' | 'bg' | 'refined'
    let stableVideoEl = null;   // hidden <video> tied to stable.mp4
    let refinedImgOS = null;    // <img> for the stump-removed background
    let refinedImgOD = null;
    // Background PNGs are saved at the downscaled bake resolution.  The
    // canvas overlay coords (MP keypoints etc.) live in full per-camera
    // pixel space, so each background image is upscaled into a
    // full-resolution offscreen <canvas> for the overlay to draw on --
    // otherwise the picture and the overlays would be misaligned by the
    // downscale factor.  Built on image load, keyed OS/OD x raw/refined.
    let bgFullOS = null, bgFullOD = null;
    let refinedFullOS = null, refinedFullOD = null;
    // Upscale a loaded background <img> to full per-camera resolution
    // (naturalSize x downscale) into a reusable offscreen canvas.
    function _toFullRes(img) {
        if (!img || !img.complete || !img.naturalWidth) return null;
        const ds = (backgroundData && backgroundData.downscale) || 1;
        const w = img.naturalWidth * ds, h = img.naturalHeight * ds;
        const cv = document.createElement('canvas');
        cv.width = w; cv.height = h;
        const cx = cv.getContext('2d');
        cx.imageSmoothingEnabled = true;
        cx.drawImage(img, 0, 0, w, h);
        return cv;
    }
    let showOutline = true;     // checkbox: show live-fetched hand boundary
    let showOtherBoundary = false;  // checkbox: the other camera's boundary,
                                    // translated to align with this one
    let showRefinedBoundary = false; // checkbox: this camera's boundary with
                                     // its red runs patched from the other
    let showFgFill = false;     // checkbox: also fetch + paint JET heatmap
    const fgFillOpacity = 1.0;  // JET foreground fill always fully opaque
    // Transient previews: shown only while the relevant slider is being
    // dragged, then auto-hidden a short moment after the last input.
    // Each flash also switches the viewport to the view that preview
    // belongs on: "MP dilate (color sample)" -> Stabilized; "Skin
    // color leniency" + "MP dilate (mask)" -> Background (raw median).
    let showBgZones = false;    // palm zone preview
    let showSkinMask = false;   // live YCrCb skin-color mask + sample region
    let showBgEdge = false;     // background-edge band preview (Hand Boundary)
    let _bgZonesTimer = null;
    let _skinMaskTimer = null;
    let _bgEdgeTimer = null;
    const _PREVIEW_HOLD_MS = 1100;
    // Switch the viewport overlay by clicking the radio (so the change
    // handler / video-seek logic runs).  No-op if already there or the
    // target view isn't available yet.
    function _switchOverlay(mode) {
        const id = { stable: 'ovStable', bg: 'ovBg',
                     refined: 'ovRefined', off: 'ovOff' }[mode];
        const el = id && document.getElementById(id);
        if (el && !el.disabled && !el.checked) el.click();
    }
    function _flashBgZones() {
        _switchOverlay('bg');
        showBgZones = true;
        try { render(); } catch (_e) {}
        if (_bgZonesTimer) clearTimeout(_bgZonesTimer);
        _bgZonesTimer = setTimeout(() => {
            showBgZones = false;
            try { render(); } catch (_e) {}
        }, _PREVIEW_HOLD_MS);
    }
    function _flashSkinMask(mode) {
        if (mode) _switchOverlay(mode);
        showSkinMask = true;
        try { render(); } catch (_e) {}
        if (_skinMaskTimer) clearTimeout(_skinMaskTimer);
        _skinMaskTimer = setTimeout(() => {
            showSkinMask = false;
            try { render(); } catch (_e) {}
        }, _PREVIEW_HOLD_MS);
    }
    // Background-edge band preview -- flashed while the Hand Boundary
    // "BG-edge ..." sliders are dragged.  Drawn over whatever overlay
    // is active (the band is in reference coords like the background).
    function _flashBgEdge() {
        showBgEdge = true;
        try { render(); } catch (_e) {}
        if (_bgEdgeTimer) clearTimeout(_bgEdgeTimer);
        _bgEdgeTimer = setTimeout(() => {
            showBgEdge = false;
            try { render(); } catch (_e) {}
        }, _PREVIEW_HOLD_MS);
    }
    // Offscreen canvas + cache key for the live skin-color mask.  The
    // classification mirrors the server's _is_skin_ycc (BT.601 YCrCb,
    // leniency-scaled window) so the preview matches what Compute
    // Background will actually do.
    let _skinMaskCanvas = null;
    let _skinMaskKey = null;
    // Per-side background-edge band cache (client-side echo of
    // _bg_edge_map + threshold + dilate on the server).  Each entry is
    // {key, canvas, bin, w, h, scale, gradX, gradY, gradT2} -- the band
    // canvas for the flash preview, plus the binary band + per-pixel
    // Sobel gradient so the outline drawing can flag polygon sides
    // running ALONG a BG edge.  Keyed OS/OD so both cameras' bands can
    // be queried at once (needed for the other-camera overlay).
    // Invalidated whenever the background images reload.
    const _bgEdge = { OS: null, OD: null };
    // Cached optimal translation aligning the other camera's boundary
    // onto the current one.  Recomputed when outlineData / side change.
    let _otherAlign = null;       // {dx, dy} or null
    let _otherAlignData = null;   // the outlineData it was computed for
    let _otherAlignSide = null;
    // Cached refined outline: this camera's boundary with its red runs
    // patched from the other camera's original.  {pts, segColors}.
    let _refinedOutline = null;
    let _refinedOutlineData = null;
    let _refinedOutlineSide = null;
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

    // Zoom/pan — same conventions as videos.js/labels.js.  Offsets are in
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
        try { trials = await api(`/api/skeleton/${id}/trials`); }
        catch (e) { trials = []; }
        const wrap = $('trialBtns');
        // /api/skeleton/{id}/trials returns `trial_stem` (e.g. "Con01_R1").
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
        // native speed for playback — same pattern as videos.js / labels.js.
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
        const src = `/api/skeleton/${subjectId}/trial/${trialMeta.trial_idx}/video`;
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
        const bgReady = !!b.available;            // background.npz exists
        const refinedReady = !!b.refined_available;  // background_refined.npz

        if (bgReady) {
            $('dot-background').classList.add('done');
            $('dot-background').classList.remove('running', 'failed');
            showBackgroundStats();
            _loadBackgroundArtifacts();          // bg thumbs + stable video
            $('medianStatus').textContent = 'Done.';
            $('backgroundStatus').textContent = refinedReady
                ? 'Done.' : 'Not computed yet.';
            // The refined overlay may have just been invalidated (a
            // fresh median drops the stale refined artifacts).
            if (overlayMode === 'refined' && !refinedReady) {
                overlayMode = 'bg';
                if ($('ovBg')) $('ovBg').checked = true;
            }
        } else {
            $('dot-background').classList.remove('done', 'running', 'failed');
            $('backgroundStats').textContent = '';
            $('backgroundPreview').style.display = 'none';
            if (stableReady) {
                // Stabilize done, Background not yet -- still load the
                // stable video so the overlay works.
                _loadStableVideo();
                $('stableStatus').textContent = 'Done.';
                $('medianStatus').textContent = 'Not computed yet.';
                $('backgroundStatus').textContent = 'Run Compute Background first.';
            } else {
                $('stableStatus').textContent = trajectory
                    ? 'Not computed yet.'
                    : 'Waiting for trajectory (run step 1 first).';
                $('medianStatus').textContent =
                    'Waiting for stable.mp4 (run Stabilize first).';
                $('backgroundStatus').textContent = '';
            }
            $('outlineStatus').textContent = '';
            overlayMode = 'off';
            if ($('ovOff'))  $('ovOff').checked = true;
        }
        // Gating:
        //   Stabilize          -> needs trajectory
        //   Compute Background -> needs stable.mp4
        //   Remove stump       -> needs background.npz (the median)
        //   overlays           -> ovStable needs stable.mp4, ovBg needs
        //     background.npz, ovRefined needs background_refined.npz
        //   hand-boundary checkboxes -> need background.npz
        $('runStableBtn').disabled = !trajectory;
        $('runMedianBtn').disabled = !stableReady;
        $('runRefineBtn').disabled = !bgReady;
        if ($('ovStable'))  $('ovStable').disabled = !stableReady;
        if ($('ovBg'))      $('ovBg').disabled = !bgReady;
        if ($('ovRefined')) $('ovRefined').disabled = !refinedReady;
        $('cbShowOutline').disabled = !bgReady;
        $('cbShowFgFill').disabled = !bgReady;
        if ($('cbShowOtherBoundary'))
            $('cbShowOtherBoundary').disabled = !bgReady || !b.is_stereo;
        if ($('cbShowRefinedBoundary'))
            $('cbShowRefinedBoundary').disabled = !bgReady || !b.is_stereo;
        if ($('runOutlinesBtn')) $('runOutlinesBtn').disabled = !bgReady;
        if (bgReady && (showOutline || showFgFill || showOtherBoundary
                        || showRefinedBoundary))
            scheduleOutlineFetch();
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
        bgFullOS = bgFullOD = refinedFullOS = refinedFullOD = null;
        // background changed -> drop cached edge bands + alignment +
        // refined outline
        _bgEdge.OS = _bgEdge.OD = null;
        _otherAlign = null; _otherAlignData = null;
        _refinedOutline = null; _refinedOutlineData = null;
        // On load: upscale the (downscaled) PNG to full-res for the
        // overlay coord space, then re-render -- the canvas 'bg'
        // overlay draws from the full-res canvas, so if the user (or a
        // slider flash) switches to the Background view before it
        // decodes, the canvas would otherwise stay blank / misaligned.
        $('bgThumbOS').addEventListener('load', () => {
            $('ovBg').disabled = false;
            bgFullOS = _toFullRes($('bgThumbOS'));
            try { render(); } catch (_e) {}
        }, { once: true });
        $('bgThumbOS').src = `${base}/background_image?side=OS&kind=bg&_=${t}`;
        if (backgroundData.is_stereo) {
            $('bgThumbOD').addEventListener('load', () => {
                bgFullOD = _toFullRes($('bgThumbOD'));
                try { render(); } catch (_e) {}
            }, { once: true });
            $('bgThumbOD').src = `${base}/background_image?side=OD&kind=bg&_=${t}`;
            $('bgThumbOD').style.display = '';
        } else {
            $('bgThumbOD').style.display = 'none';
        }
        $('backgroundPreview').style.display = '';
        // Refined (stump-removed) background -- off-DOM <img>s, upscaled
        // to full-res canvases the same way.
        if (backgroundData.refined_available) {
            refinedImgOS = new Image();
            refinedImgOD = new Image();
            refinedImgOS.onload = () => {
                if ($('ovRefined')) $('ovRefined').disabled = false;
                refinedFullOS = _toFullRes(refinedImgOS);
                render();
            };
            refinedImgOS.src = `${base}/background_image?side=OS&kind=refined&_=${t}`;
            if (backgroundData.is_stereo) {
                refinedImgOD.onload = () => {
                    refinedFullOD = _toFullRes(refinedImgOD);
                    render();
                };
                refinedImgOD.src = `${base}/background_image?side=OD&kind=refined&_=${t}`;
            }
        } else {
            refinedImgOS = refinedImgOD = null;
        }
        _loadStableVideo();
    }

    /**
     * Spawn a preproc job (Stabilize / Compute Background / Remove
     * stump) and stream its progress into ``statusEl``.  Shared by
     * computeStable + computeMedian + computeRefine.  ``extraBody``
     * merges into the POST body (e.g. palm_grow_px).
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

    /** Stage 2a -- "Compute Background": the raw masked-median only. */
    async function computeMedian() {
        if (subjectId == null || currentTrialIdx < 0) return;
        if (!backgroundData || !backgroundData.stable_mp4_exists) {
            $('medianStatus').textContent = 'Run Stabilize first.';
            return;
        }
        await _runPreprocJob('compute_background', 'medianStatus');
    }

    /** Stage 2b -- "Remove stump": apply the sliders to the median. */
    async function computeRefine() {
        if (subjectId == null || currentTrialIdx < 0) return;
        if (!backgroundData || !backgroundData.available) {
            $('backgroundStatus').textContent = 'Run Compute Background first.';
            return;
        }
        const palmSlider = $('palmGrowSlider');
        const palmGrowPx = palmSlider ? parseInt(palmSlider.value, 10) : 15;
        const colorDilateSlider = $('colorDilateSlider');
        const colorDilatePx = colorDilateSlider
            ? parseInt(colorDilateSlider.value, 10) : 0;
        const lenSlider = $('skinLeniencySlider');
        const skinLeniency = lenSlider ? parseFloat(lenSlider.value) : 1.0;
        const darkBoostSlider = $('darkBoostSlider');
        const darkBoost = darkBoostSlider
            ? parseFloat(darkBoostSlider.value) : 1.0;
        await _runPreprocJob('refine_background', 'backgroundStatus',
                              { palm_grow_px: palmGrowPx,
                                color_dilate_px: colorDilatePx,
                                skin_leniency: skinLeniency,
                                dark_boost: darkBoost });
    }

    /** Bake the hand boundary for every frame at the current
     *  Hand-Boundary slider settings -> outlines.json. */
    async function computeOutlines() {
        if (subjectId == null || currentTrialIdx < 0) return;
        if (!backgroundData || !backgroundData.available) {
            $('outlinesBakeStatus').textContent = 'Run Compute Background first.';
            return;
        }
        const _i = (id, dflt) => {
            const s = $(id); return s ? parseInt(s.value, 10) : dflt;
        };
        const _f = (id, dflt) => {
            const s = $(id); return s ? parseFloat(s.value) : dflt;
        };
        await _runPreprocJob('compute_outlines', 'outlinesBakeStatus', {
            dilation_px:          _i('fgDilateSlider', 14),
            open_radius_px:       _i('fgOpenSlider', 0),
            bg_edge_band_thresh:  _f('bgEdgeThreshSlider', 0.35),
            bg_edge_band_dilate:  _i('bgEdgeBandSlider', 3),
            bg_edge_open_radius:  _i('bgEdgeClipSlider', 5),
            threshold_scale:      _f('outlineThreshSlider', 1.0),
        });
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
        if (!showOutline && !showFgFill && !showOtherBoundary
            && !showRefinedBoundary) return;
        const dilSlider = $('fgDilateSlider');
        const dilationPx = dilSlider ? parseInt(dilSlider.value, 10) : 14;
        const openSlider = $('fgOpenSlider');
        const openRadiusPx = openSlider ? parseInt(openSlider.value, 10) : 0;
        const beThreshSlider = $('bgEdgeThreshSlider');
        const beThresh = beThreshSlider ? parseFloat(beThreshSlider.value) : 0.35;
        const beBandSlider = $('bgEdgeBandSlider');
        const beBand = beBandSlider ? parseInt(beBandSlider.value, 10) : 3;
        const beClipSlider = $('bgEdgeClipSlider');
        const beClip = beClipSlider ? parseInt(beClipSlider.value, 10) : 5;
        const threshSlider = $('outlineThreshSlider');
        const threshScale = threshSlider ? parseFloat(threshSlider.value) : 1.0;
        const frame = currentFrame;
        const seq = ++outlineFetchSeq;
        const includeFg = showFgFill ? 1 : 0;
        const url = `/api/preproc/${subjectId}/trial/${trialMeta.trial_idx}/outline_frame`
                  + `?frame=${frame}&dilation_px=${dilationPx}`
                  + `&open_radius_px=${openRadiusPx}&include_fg=${includeFg}`
                  + `&bg_edge_band_thresh=${beThresh}`
                  + `&bg_edge_band_dilate=${beBand}`
                  + `&bg_edge_open_radius=${beClip}`
                  + `&threshold_scale=${threshScale}`;
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

    /** Universal YCrCb skin box (mirrors _SKIN_* in background.py) --
     *  the fallback when no trial-specific range can be fit. */
    const _SKIN_UNIVERSAL = { crLo: 130, crHi: 175, cbLo: 80, cbHi: 130 };

    /** BT.601 RGB -> (Cr, Cb).  Matches cv2.cvtColor(BGR2YCrCb). */
    function _rgbToCrCb(r, g, b) {
        const y = 0.299 * r + 0.587 * g + 0.114 * b;
        return [(r - y) * 0.713 + 128, (b - y) * 0.564 + 128];
    }

    // The dilated-MP-skeleton region the last skin-range fit sampled
    // from -- recorded as line segments (sub-rect coords) + a stroke
    // width so render() can draw the region back, so the user sees
    // exactly which pixels fed the trial skin tone.
    let _skinSampleSegs   = [];  // [[x0,y0,x1,y1],...] sub-rect coords
    let _skinSampleStrokeW = 0;  // stroke width, sub-rect px
    let _skinSampleMaskCv = null;  // reusable offscreen mask canvas

    // Tight-MP-mask half-thickness, sub-rect (~full-res) px -- echoes
    // the server's _TIGHT_STAMP_DOWN (half the BG-mask dilation stamp).
    const _SKIN_TIGHT_RADIUS = 7;

    /**
     * Trial-specific skin window from the current frame's MP keypoints
     * -- the client-side echo of _fit_skin_range_cbcr.  Builds a
     * dilated MP-skeleton region (same construction as the hand mask:
     * thick strokes along the finger/palm chains + the ulnar-heel
     * segments), then collects Cr/Cb from every offscreen pixel inside
     * it and returns {crLo,crHi,cbLo,cbHi} from the [2,98] percentiles.
     * Null if too few samples (caller falls back to the universal
     * box).  Side effect: populates _skinSampleSegs / _skinSampleStrokeW
     * with the region it sampled (or [] on a null return) so the
     * preview can draw it.
     */
    function _skinRangeFromKpts(imgData, cw, ch, drawSw, drawSh) {
        _skinSampleSegs = [];
        if (!mpKeypoints) return null;
        const isOD = (currentSide === 'OD');
        const useRef = overlayMode !== 'off';
        const arr = useRef
            ? (isOD ? mpKeypoints.ref_OD : mpKeypoints.ref_OS)
            : (isOD ? mpKeypoints.raw_OD : mpKeypoints.raw_OS);
        const f = arr && currentFrame < arr.length ? arr[currentFrame] : null;
        if (!f) return null;
        const sx = cw / drawSw, sy = ch / drawSh;
        const d = imgData.data;
        const hasPt = j => f[j] && f[j][0] != null && f[j][1] != null;

        // Stroke half-width: tight MP radius +/- the "MP dilate (color
        // sample)" knob (full-res px) -- the client echo of the
        // server's erode/dilate of the tight MP mask.
        const colorDilateSlider = $('colorDilateSlider');
        const colorDilatePx = colorDilateSlider
            ? parseInt(colorDilateSlider.value, 10) : 0;
        const sampleRadius = Math.max(1, _SKIN_TIGHT_RADIUS + colorDilatePx);
        const strokeWSub = sampleRadius * 2;          // sub-rect px

        // Collect the skeleton segments (sub-rect coords): finger
        // chains, palm arc, and the two ulnar-heel closers -- exactly
        // the construction _build_kpt_hand_region uses on the server.
        const segs = [];
        const addSeg = (a, b) => {
            if (hasPt(a) && hasPt(b))
                segs.push([f[a][0], f[a][1], f[b][0], f[b][1]]);
        };
        for (const chain of _MP_FINGER_CHAINS)
            for (let ci = 0; ci < chain.length - 1; ci++)
                addSeg(chain[ci], chain[ci + 1]);
        for (let i = 0; i < _MP_PALM_CHAIN.length - 1; i++)
            addSeg(_MP_PALM_CHAIN[i], _MP_PALM_CHAIN[i + 1]);
        const refl = _reflectThumbCmc(f);
        if (refl) {
            for (const j of [17, 1]) {
                if (hasPt(j))
                    segs.push([f[j][0], f[j][1], refl[0], refl[1]]);
            }
        }
        if (!segs.length) return null;

        // Rasterise the region into an offscreen mask at the same
        // resolution as imgData, stroking the skeleton with round
        // caps/joins so joints stamp as discs (like the server).
        if (!_skinSampleMaskCv)
            _skinSampleMaskCv = document.createElement('canvas');
        _skinSampleMaskCv.width = cw;
        _skinSampleMaskCv.height = ch;
        const mctx = _skinSampleMaskCv.getContext(
            '2d', { willReadFrequently: true });
        mctx.clearRect(0, 0, cw, ch);
        mctx.lineCap = 'round';
        mctx.lineJoin = 'round';
        mctx.strokeStyle = '#fff';
        mctx.lineWidth = Math.max(1, strokeWSub * sx);
        mctx.beginPath();
        for (const s of segs) {
            mctx.moveTo(s[0] * sx, s[1] * sy);
            mctx.lineTo(s[2] * sx, s[3] * sy);
        }
        mctx.stroke();
        let maskData;
        try {
            maskData = mctx.getImageData(0, 0, cw, ch).data;
        } catch (e) {
            return null;
        }

        // Collect Cr/Cb from every source pixel inside the region.
        const crs = [], cbs = [];
        for (let p = 0, n = cw * ch; p < n; p++) {
            if (maskData[p * 4 + 3] === 0) continue;
            const i = p * 4;
            const [cr, cb] = _rgbToCrCb(d[i], d[i + 1], d[i + 2]);
            crs.push(cr); cbs.push(cb);
        }
        if (crs.length < 50) return null;
        // Fit succeeded -- expose the sampled region for the overlay.
        _skinSampleSegs = segs;
        _skinSampleStrokeW = strokeWSub;
        const pct = (a, p) => {
            const s = a.slice().sort((x, y) => x - y);
            return s[Math.min(s.length - 1,
                              Math.max(0, Math.round(p / 100 * (s.length - 1))))];
        };
        let crLo = pct(crs, 2), crHi = pct(crs, 98);
        let cbLo = pct(cbs, 2), cbHi = pct(cbs, 98);
        if (crHi - crLo < 2) { crLo -= 1; crHi += 1; }
        if (cbHi - cbLo < 2) { cbLo -= 1; cbHi += 1; }
        return { crLo, crHi, cbLo, cbHi };
    }

    /**
     * Build (and cache) the live skin-color mask for the current
     * frame at the current Skin-colour-leniency setting.  Uses a
     * trial-specific Cr/Cb window fit from this frame's MP keypoints
     * (echoing the server's _fit_skin_range_cbcr); falls back to the
     * universal box when MP isn't available.  ``leniency`` scales the
     * window's half-width around its center.
     *
     * Returns an offscreen <canvas> with magenta where skin, fully
     * transparent elsewhere -- or null if the source can't be read
     * yet (cold video, tainted canvas).  Cached by frame + leniency +
     * source so playback / re-renders don't reclassify needlessly.
     */
    function _updateSkinMask(srcImage, drawSx, drawSy, drawSw, drawSh) {
        const lenSlider = $('skinLeniencySlider');
        const leniency = lenSlider ? parseFloat(lenSlider.value) : 1.0;
        const cdSlider = $('colorDilateSlider');
        const colorDilate = cdSlider ? parseInt(cdSlider.value, 10) : 0;
        const dbSlider = $('darkBoostSlider');
        const darkBoost = dbSlider ? parseFloat(dbSlider.value) : 1.0;
        const key = `${currentFrame}|${leniency}|${colorDilate}|${darkBoost}|`
                  + `${overlayMode}|${currentSide}|${drawSw}x${drawSh}`;
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
        // Trial-specific window from this frame's keypoints; universal
        // box if MP unavailable.  Scale half-width by leniency.
        const rng = _skinRangeFromKpts(img, cw, ch, drawSw, drawSh)
                 || _SKIN_UNIVERSAL;
        const crC = (rng.crLo + rng.crHi) / 2;
        const crHalf = (rng.crHi - rng.crLo) / 2;
        const cbC = (rng.cbLo + rng.cbHi) / 2;
        const cbHalf = (rng.cbHi - rng.cbLo) / 2;
        // "Dark-color boost": effective leniency is scaled per-pixel by
        // 1 + (darkBoost-1)*(1 - Y/255) -- bright pixels untouched, dark
        // pixels get the full boost.  Mirrors _is_skin_ycc on the server.
        const dbExtra = darkBoost - 1;
        for (let i = 0; i < d.length; i += 4) {
            const r = d[i], g = d[i + 1], b = d[i + 2];
            const y = 0.299 * r + 0.587 * g + 0.114 * b;
            const eff = dbExtra > 0
                ? leniency * (1 + dbExtra * (1 - y / 255)) : leniency;
            const crH = crHalf * eff, cbH = cbHalf * eff;
            const [cr, cb] = _rgbToCrCb(r, g, b);
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

    /** Binary dilation by radius ``r`` via two separable max passes. */
    function _dilateBinary(src, w, h, r) {
        if (r <= 0) return src;
        const tmp = new Uint8Array(w * h);
        for (let y = 0; y < h; y++) {
            const row = y * w;
            for (let x = 0; x < w; x++) {
                let m = 0;
                const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
                for (let xx = x0; xx <= x1; xx++) {
                    if (src[row + xx]) { m = 1; break; }
                }
                tmp[row + x] = m;
            }
        }
        const out = new Uint8Array(w * h);
        for (let x = 0; x < w; x++) {
            for (let y = 0; y < h; y++) {
                let m = 0;
                const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
                for (let yy = y0; yy <= y1; yy++) {
                    if (tmp[yy * w + x]) { m = 1; break; }
                }
                out[y * w + x] = m;
            }
        }
        return out;
    }

    /**
     * Background-edge band for ``side`` ('OS' / 'OD') -- the client-side
     * echo of the server's _bg_edge_map + threshold + dilate (the
     * ``edge_band`` of the outline's step-5b retraction).  Sobel
     * magnitude of that camera's background, 99th-percentile
     * normalized, thresholded by "BG-edge threshold" and dilated by
     * "BG-edge band width".  Caches into ``_bgEdge[side]`` (band canvas
     * + binary band + per-pixel gradient); returns the cache entry, or
     * null if that camera's background isn't loaded yet.
     */
    function _bgEdgeBand(side) {
        const isOD = (side === 'OD');
        const bcv = (isOD && backgroundData && backgroundData.is_stereo)
            ? (refinedFullOD || bgFullOD)
            : (refinedFullOS || bgFullOS);
        if (!bcv || !bcv.width) return null;
        const thr = parseFloat(($('bgEdgeThreshSlider') || {}).value || 0.35);
        const band = parseInt(($('bgEdgeBandSlider') || {}).value || 3, 10);
        const key = `${side}|${bcv.width}x${bcv.height}|${thr}|${band}`;
        const cached = _bgEdge[side];
        if (cached && cached.key === key) return cached;
        // Work at half-res -- plenty for a preview.
        const cw = Math.max(3, Math.round(bcv.width / 2));
        const ch = Math.max(3, Math.round(bcv.height / 2));
        const tmp = document.createElement('canvas');
        tmp.width = cw; tmp.height = ch;
        const tctx = tmp.getContext('2d', { willReadFrequently: true });
        let img;
        try {
            tctx.drawImage(bcv, 0, 0, cw, ch);
            img = tctx.getImageData(0, 0, cw, ch);
        } catch (e) { return null; }
        const d = img.data;
        // Grayscale.
        const gray = new Float32Array(cw * ch);
        for (let i = 0, p = 0; p < gray.length; i += 4, p++) {
            gray[p] = 0.114 * d[i] + 0.587 * d[i + 1] + 0.299 * d[i + 2];
        }
        // Sobel magnitude + gradient (edge-normal direction).
        const mag = new Float32Array(cw * ch);
        const gradX = new Float32Array(cw * ch);
        const gradY = new Float32Array(cw * ch);
        for (let y = 1; y < ch - 1; y++) {
            for (let x = 1; x < cw - 1; x++) {
                const o = y * cw + x;
                const tl = gray[o - cw - 1], tc = gray[o - cw], tr = gray[o - cw + 1];
                const ml = gray[o - 1],                         mr = gray[o + 1];
                const bl = gray[o + cw - 1], bc = gray[o + cw], br = gray[o + cw + 1];
                const sx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
                const sy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
                gradX[o] = sx; gradY[o] = sy;
                mag[o] = Math.sqrt(sx * sx + sy * sy);
            }
        }
        // 99th-percentile normalization.
        const sorted = Float32Array.from(mag).sort();
        const p99 = sorted[Math.min(sorted.length - 1,
                                    Math.floor(0.99 * sorted.length))] || 1;
        // Threshold -> binary, then dilate (half-res radius).
        let bin = new Uint8Array(cw * ch);
        for (let p = 0; p < bin.length; p++) {
            bin[p] = (mag[p] / p99) > thr ? 1 : 0;
        }
        bin = _dilateBinary(bin, cw, ch, Math.max(0, Math.round(band / 2)));
        // Paint orange band into an RGBA canvas.
        for (let p = 0, i = 0; p < bin.length; p++, i += 4) {
            if (bin[p]) {
                d[i] = 255; d[i + 1] = 140; d[i + 2] = 0; d[i + 3] = 120;
            } else {
                d[i + 3] = 0;
            }
        }
        tctx.putImageData(img, 0, 0);
        const entry = {
            key, canvas: tmp, bin, w: cw, h: ch,
            scale: cw / bcv.width, gradX, gradY,
            gradT2: (0.35 * thr * p99) * (0.35 * thr * p99),
        };
        _bgEdge[side] = entry;
        return entry;
    }

    /** True if a polygon side (full-res reference coords of ``side``'s
     *  camera) runs ALONG that camera's background edge -- mostly on
     *  the BG-edge band AND, where the band has a clear gradient,
     *  mostly parallel to that edge (not crossing it).  Samples ~every
     *  2 px along the side. */
    function _segOnBgEdge(x0, y0, x1, y1, side) {
        const be = _bgEdge[side];
        if (!be || !be.bin) return false;
        const s = be.scale, W = be.w, H = be.h;
        const dx = x1 - x0, dy = y1 - y0;
        const len = Math.hypot(dx, dy);
        if (len < 1e-3) return false;
        const sdx = dx / len, sdy = dy / len;        // unit side direction
        const n = Math.max(1, Math.round(len / 2));
        let inBounds = 0, onBand = 0, dirN = 0, par = 0;
        for (let i = 0; i <= n; i++) {
            const t = i / n;
            const bx = Math.round((x0 + dx * t) * s);
            const by = Math.round((y0 + dy * t) * s);
            if (bx < 0 || by < 0 || bx >= W || by >= H) continue;
            inBounds++;
            const idx = by * W + bx;
            if (!be.bin[idx]) continue;
            onBand++;
            // Where the band has a clear gradient, the side runs ALONG
            // the edge when its direction is ~perpendicular to the
            // gradient (the edge-normal): |sideDir . gradUnit| small.
            const gx = be.gradX[idx], gy = be.gradY[idx];
            const g2 = gx * gx + gy * gy;
            if (g2 < be.gradT2) continue;
            dirN++;
            const dot = Math.abs((sdx * gx + sdy * gy) / Math.sqrt(g2));
            if (dot < 0.5) par++;                    // within ~30 deg of parallel
        }
        if (inBounds === 0 || onBand < 0.5 * inBounds) return false;
        // On the band: red only if it mostly runs along the edge (not
        // crossing it).  If the band stretch had no clear gradient at
        // all, stay conservative and don't mark it red.
        return dirN > 0 && par >= 0.5 * dirN;
    }

    /** Per-segment red/yellow classification for a closed polygon,
     *  optionally dilated by ``spread`` along the ring -- so any yellow
     *  segment within ``spread`` segments of a red one also becomes
     *  red (a generous interpretation of "near a BG edge"). */
    function _classifyPolyRed(poly, side, spread) {
        const N = poly.length;
        let red = new Array(N);
        for (let i = 0; i < N; i++) {
            const a = poly[i], b = poly[(i + 1) % N];
            red[i] = _segOnBgEdge(a[0], a[1], b[0], b[1], side);
        }
        for (let s = 0; s < (spread | 0); s++) {
            const next = new Array(N);
            for (let i = 0; i < N; i++) {
                next[i] = red[i] || red[(i - 1 + N) % N] || red[(i + 1) % N];
            }
            red = next;
        }
        return red;
    }

    /**
     * Optimal translation to align the OTHER camera's boundary onto
     * the current one.  Dense-samples both polygons (~every 3 px),
     * drops points on red (BG-edge) segments -- those are unreliable
     * -- then votes every pairwise offset into a 2D accumulator
     * centred on the centroid difference.  The peak (of the 3x3-summed
     * accumulator) is the translation that makes the MOST sampled
     * points coincide, i.e. it prefers a large portion exactly
     * aligned even if that pulls other edges apart.  Returns {dx,dy}.
     */
    function _computeOtherAlign(curPoly, otherPoly, curSide, otherSide) {
        const SP = 3;            // sample spacing, px
        const R = 130;           // accumulator half-extent around the
                                 // centroid-difference seed, px
        // Dense, red-filtered samples along a closed polygon.  The
        // "Red spread" slider is deliberately NOT applied here -- it
        // only changes which CUR-camera segments get refined; the
        // alignment uses each camera's raw red classification so it
        // doesn't lose voters along the way.
        const _sample = (poly, side) => {
            const red = _classifyPolyRed(poly, side, 0);
            const pts = [];
            const n = poly.length;
            for (let i = 0; i < n; i++) {
                if (red[i]) continue;
                const a = poly[i], b = poly[(i + 1) % n];
                const dx = b[0] - a[0], dy = b[1] - a[1];
                const L = Math.hypot(dx, dy);
                const steps = Math.max(1, Math.round(L / SP));
                for (let k = 0; k < steps; k++) {
                    const t = k / steps;
                    pts.push([a[0] + dx * t, a[1] + dy * t]);
                }
            }
            return pts;
        };
        const cur = _sample(curPoly, curSide);
        const other = _sample(otherPoly, otherSide);
        if (cur.length < 8 || other.length < 8) {
            // Not enough reliable edge -- fall back to centroid match.
            const c = _centroid(curPoly), o = _centroid(otherPoly);
            return { dx: c[0] - o[0], dy: c[1] - o[1] };
        }
        const cc = _centroid(cur), oc = _centroid(other);
        const t0x = Math.round(cc[0] - oc[0]);
        const t0y = Math.round(cc[1] - oc[1]);
        const W = 2 * R + 1;
        const acc = new Uint16Array(W * W);
        for (let i = 0; i < other.length; i++) {
            const px = other[i][0], py = other[i][1];
            for (let j = 0; j < cur.length; j++) {
                const ix = Math.round(cur[j][0] - px - t0x) + R;
                if (ix < 0 || ix >= W) continue;
                const iy = Math.round(cur[j][1] - py - t0y) + R;
                if (iy < 0 || iy >= W) continue;
                acc[iy * W + ix]++;
            }
        }
        // Peak of the 3x3-summed accumulator -- "exactly aligned" with
        // a +/-1 px tolerance for the polygon's integer coords.
        let best = -1, bx = R, by = R;
        for (let y = 1; y < W - 1; y++) {
            for (let x = 1; x < W - 1; x++) {
                let s = 0;
                for (let dy = -1; dy <= 1; dy++)
                    for (let dx = -1; dx <= 1; dx++)
                        s += acc[(y + dy) * W + (x + dx)];
                if (s > best) { best = s; bx = x; by = y; }
            }
        }
        return { dx: (bx - R) + t0x, dy: (by - R) + t0y };
    }

    function _centroid(pts) {
        let sx = 0, sy = 0;
        for (let i = 0; i < pts.length; i++) { sx += pts[i][0]; sy += pts[i][1]; }
        const n = Math.max(1, pts.length);
        return [sx / n, sy / n];
    }

    /** (Re)compute the cached other-camera alignment translation if the
     *  outlineData / side it was computed for has changed.  The
     *  alignment is independent of the "Red spread" slider. */
    function _ensureOtherAlign(curPts, otherPts, curSide, otherSide) {
        if (_otherAlign === null
            || _otherAlignData !== outlineData
            || _otherAlignSide !== curSide) {
            _otherAlign = _computeOtherAlign(
                curPts, otherPts, curSide, otherSide);
            _otherAlignData = outlineData;
            _otherAlignSide = curSide;
        }
        return _otherAlign;
    }

    /**
     * Refined boundary for the current camera: its own outline with
     * every red run (a side running along a background edge -- locally
     * unreliable) patched from the OTHER camera's original outline,
     * which is reliable exactly there because background edges are
     * camera-specific.
     *
     * Per red run:
     *   - find an anchor on each side -- the junction yellow vertex, or
     *     hop outward up to 3 vertices to one that lands within
     *     _REFINE_ANCHOR_THRESH px of the aligned other outline (any
     *     yellow vertices skipped to reach it are absorbed/rewritten);
     *     stop hopping at another red run.
     *   - splice in the corresponding arc of the (translated) other
     *     outline between the two anchors, rubber-banded so its ends
     *     land exactly on the anchors.
     *   - bail (keep the original red) if no anchor is found within 3
     *     hops, or if that other-camera arc is itself mostly red.
     *
     * Returns {pts: [[x,y]...], segColors: ['own'|'borrowed'|
     * 'unresolved'...]} -- a closed polygon + per-segment provenance.
     */
    function _refineOutline(curPoly, curSide, otherPoly, otherSide, align,
                            thresh, maxHop, spread) {
        const THRESH = thresh;     // "very close to the green line", px
        const MAXHOP = maxHop | 0; // max yellow vertices to hop for an anchor
        const SP = 3;              // other-outline densification, px
        const GREEN_RED_FRAC = 0.5; // bail if the borrowed arc is this red
        const N = curPoly.length;
        const own = () => ({ pts: curPoly, segColors: curPoly.map(() => 'own') });
        if (N < 3) return own();
        _bgEdgeBand(curSide); _bgEdgeBand(otherSide);
        // Classify this camera's segments (with the user's red spread).
        const curRed = _classifyPolyRed(curPoly, curSide, spread);
        if (!curRed.some(Boolean)) return own();
        const M = otherPoly.length;
        if (M < 3 || curRed.every(Boolean)) {
            return { pts: curPoly,
                     segColors: curRed.map(r => r ? 'unresolved' : 'own') };
        }
        // Densify the translated other outline + per-point redness.
        // Red spread is intentionally NOT applied to the other camera
        // -- it only decides which CUR segments to refine; when
        // considering the other camera as a SOURCE for replacement,
        // its raw red classification is what we want (no dilation
        // shrinking the borrowable arc).
        const otherRed = _classifyPolyRed(otherPoly, otherSide, 0);
        const dense = [], denseRed = [];
        for (let j = 0; j < M; j++) {
            const a = otherPoly[j], b = otherPoly[(j + 1) % M];
            const red = otherRed[j];
            const dx = b[0] - a[0], dy = b[1] - a[1];
            const steps = Math.max(1, Math.round(Math.hypot(dx, dy) / SP));
            for (let k = 0; k < steps; k++) {
                const t = k / steps;
                dense.push([a[0] + dx * t + align.dx, a[1] + dy * t + align.dy]);
                denseRed.push(red);
            }
        }
        const D = dense.length;
        const nearest = (p) => {
            let bi = 0, bd = Infinity;
            for (let k = 0; k < D; k++) {
                const ex = dense[k][0] - p[0], ey = dense[k][1] - p[1];
                const d2 = ex * ex + ey * ey;
                if (d2 < bd) { bd = d2; bi = k; }
            }
            return { idx: bi, dist: Math.sqrt(bd) };
        };
        const closeIdx = curPoly.map(p => {
            const nr = nearest(p);
            return nr.dist <= THRESH ? nr.idx : -1;
        });
        // Red runs: maximal contiguous true spans of curRed, scanned
        // from a yellow segment so none wraps.
        let s0 = -1;
        for (let i = 0; i < N; i++) if (!curRed[i]) { s0 = i; break; }
        const runs = [];
        let runStart = -1;
        for (let c = 0; c < N; c++) {
            const i = (s0 + c) % N;
            if (curRed[i]) {
                if (runStart < 0) runStart = i;
            } else if (runStart >= 0) {
                runs.push([runStart, (i - 1 + N) % N]);  // seg span
                runStart = -1;
            }
        }
        // A run that reaches the end of the scan closes at the segment
        // just before s0 (which is yellow).
        if (runStart >= 0) runs.push([runStart, (s0 - 1 + N) % N]);
        // Anchor search on one side of a run.  ``dir`` is -1 (left of
        // the run) or +1 (right).  ``v0`` is the junction vertex.
        const findAnchor = (v0, dir) => {
            let v = v0;
            for (let hop = 0; hop <= MAXHOP; hop++) {
                if (closeIdx[v] >= 0) return v;
                // segment we'd cross to hop one more vertex outward
                const seg = dir < 0 ? (v - 1 + N) % N : v;
                if (curRed[seg]) return -1;          // hit another red run
                v = (v + dir + N) % N;
            }
            return -1;
        };
        // Build a patch per run; skip on conflict / bail.
        const consumed = new Array(N).fill(false);
        const jumpTo = {};                 // aLeft vertex -> patch
        for (const [segS, segE] of runs) {
            const lj = segS;               // left junction vertex
            const rj = (segE + 1) % N;     // right junction vertex
            const aL = findAnchor(lj, -1);
            const aR = findAnchor(rj, +1);
            if (aL < 0 || aR < 0) continue;          // unanchorable -> bail
            // Vertices strictly between aL and aR (through the run).
            const interior = [];
            for (let v = (aL + 1) % N; v !== aR; v = (v + 1) % N)
                interior.push(v);
            if (interior.some(v => consumed[v]) || jumpTo[aL]) continue; // overlap
            // Corresponding arc of the dense other outline.
            const gL = closeIdx[aL], gR = closeIdx[aR];
            const fwd = [], bwd = [];
            for (let k = gL; ; k = (k + 1) % D) { fwd.push(k); if (k === gR) break; }
            for (let k = gL; ; k = (k - 1 + D) % D) { bwd.push(k); if (k === gR) break; }
            // Pick the arc whose midpoint is nearer the run midpoint.
            const runMid = curPoly[(segS + Math.floor((((segE + 1) % N) - segS + N) % N
                                                       / 2)) % N];
            const arcMidD = (arc) => {
                const m = dense[arc[Math.floor(arc.length / 2)]];
                return (m[0] - runMid[0]) ** 2 + (m[1] - runMid[1]) ** 2;
            };
            let arc = arcMidD(fwd) <= arcMidD(bwd) ? fwd : bwd;
            if (arc.length < 2) continue;
            // Bail if the borrowed arc is itself mostly red.
            let redN = 0;
            for (const k of arc) if (denseRed[k]) redN++;
            if (redN / arc.length > GREEN_RED_FRAC) continue;
            // Rubber-band: snap arc ends onto the anchors, distributing
            // the correction linearly along the arc.
            const A = curPoly[aL], B = curPoly[aR];
            const e0x = A[0] - dense[arc[0]][0], e0y = A[1] - dense[arc[0]][1];
            const e1x = B[0] - dense[arc[arc.length - 1]][0];
            const e1y = B[1] - dense[arc[arc.length - 1]][1];
            const patchPts = [], patchColors = [];
            for (let k = 0; k < arc.length; k++) {
                const f = k / (arc.length - 1);
                const d = dense[arc[k]];
                patchPts.push([d[0] + e0x * (1 - f) + e1x * f,
                               d[1] + e0y * (1 - f) + e1y * f]);
                if (k > 0) {
                    patchColors.push((denseRed[arc[k - 1]] || denseRed[arc[k]])
                                     ? 'unresolved' : 'borrowed');
                }
            }
            jumpTo[aL] = { aRight: aR, patchPts, patchColors };
            for (const v of interior) consumed[v] = true;
        }
        if (Object.keys(jumpTo).length === 0) {
            return { pts: curPoly,
                     segColors: curRed.map(r => r ? 'unresolved' : 'own') };
        }
        // Assemble the refined closed polygon by walking the ring.
        let v0 = -1;
        for (let v = 0; v < N; v++) if (!consumed[v]) { v0 = v; break; }
        if (v0 < 0) return own();
        const outPts = [], segColors = [];
        let i = v0;
        do {
            outPts.push(curPoly[i]);
            const J = jumpTo[i];
            if (J) {
                for (let k = 1; k < J.patchPts.length - 1; k++) {
                    segColors.push(J.patchColors[k - 1]);
                    outPts.push(J.patchPts[k]);
                }
                segColors.push(J.patchColors[J.patchColors.length - 1]);
                i = J.aRight;                // curPoly[aRight] emitted next loop
            } else {
                segColors.push(curRed[i] ? 'unresolved' : 'own');
                i = (i + 1) % N;
            }
        } while (i !== v0);
        return { pts: outPts, segColors };
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
        if (showOutline || showFgFill || showOtherBoundary || showRefinedBoundary) scheduleOutlineFetch();
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
        if ((overlayMode === 'bg' || overlayMode === 'refined')
                && backgroundData) {
            // Static background, upscaled to full per-camera resolution
            // so the overlay coords (MP keypoints etc.) line up with the
            // picture.  'bg' = raw masked-median; 'refined' = stump
            // removed.  Each is a full-res offscreen <canvas>.
            const isOD = (currentSide === 'OD');
            let cv;
            if (overlayMode === 'refined') {
                cv = (isOD && backgroundData.is_stereo)
                    ? refinedFullOD : refinedFullOS;
            } else {
                cv = (isOD && backgroundData.is_stereo)
                    ? bgFullOD : bgFullOS;
            }
            if (cv && cv.width > 0) {
                srcImage = cv;
                drawSx = 0; drawSy = 0;
                drawSw = cv.width; drawSh = cv.height;
                srcW = drawSw; srcH = drawSh;
                labelText = overlayMode === 'refined'
                    ? 'Refined background  (stump removed)'
                    : 'Background  (raw temporal median)';
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
            // Show the dilated-MP-skeleton region that fed the trial
            // skin-tone range, so the user sees exactly which pixels
            // were sampled.  Same construction as the hand mask: thick
            // translucent strokes along the skeleton, with a thin
            // white centreline so the region reads clearly.
            if (_skinSampleSegs.length) {
                ctx.save();
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.30)';
                ctx.lineWidth = _skinSampleStrokeW;
                ctx.beginPath();
                for (const s of _skinSampleSegs) {
                    ctx.moveTo(s[0], s[1]);
                    ctx.lineTo(s[2], s[3]);
                }
                ctx.stroke();
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
                ctx.lineWidth = 1.5 / Math.max(0.01, scale * bps);
                ctx.beginPath();
                for (const s of _skinSampleSegs) {
                    ctx.moveTo(s[0], s[1]);
                    ctx.lineTo(s[2], s[3]);
                }
                ctx.stroke();
                ctx.restore();
            }
        }

        // Background-edge band -- orange tint over the static
        // background edges the outline's step-5b retraction acts on.
        // Flashed while the Hand Boundary "BG-edge ..." sliders move.
        if (showBgEdge && srcImage) {
            const be = _bgEdgeBand(currentSide);
            if (be && be.canvas) {
                ctx.save();
                ctx.imageSmoothingEnabled = false;
                ctx.drawImage(be.canvas, 0, 0, drawSw, drawSh);
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

        // Stroke a closed polygon, batching consecutive same-colour
        // sides into one path.  ``segColor(a, b, i)`` -> CSS colour for
        // the side a->b (index i); ``ox``/``oy`` translate the drawn
        // path (used for the other camera's aligned boundary).
        function _strokePoly(pts, segColor, ox, oy) {
            ox = ox || 0; oy = oy || 0;
            const N = pts.length;
            let curColor = null;
            for (let i = 0; i < N; i++) {
                const a = pts[i], b = pts[(i + 1) % N];
                const col = segColor(a, b, i);
                if (col !== curColor) {
                    if (curColor !== null) ctx.stroke();
                    ctx.strokeStyle = col;
                    ctx.beginPath();
                    ctx.moveTo(a[0] + ox, a[1] + oy);
                    curColor = col;
                }
                ctx.lineTo(b[0] + ox, b[1] + oy);
            }
            if (curColor !== null) ctx.stroke();
        }

        // Refined-boundary tunables read from the Hand Boundary sliders.
        const _redSpread = parseInt(
            ($('redSpreadSlider') || {}).value || 0, 10);
        const _anchorHops = parseInt(
            ($('anchorHopsSlider') || {}).value || 3, 10);
        const _anchorDist = parseInt(
            ($('anchorDistSlider') || {}).value || 6, 10);

        // Live hand-boundary overlay -- draws the server-computed
        // closed polygon as a yellow stroked path (red where a side
        // runs along a background edge), tracking the current side's
        // contour from outlineData.
        const YELLOW = 'rgba(255, 235, 59, 0.95)';
        const RED    = 'rgba(255, 64, 48, 0.95)';
        const GREEN  = 'rgba(70, 220, 90, 0.95)';
        if (showOutline && outlineData
            && outlineData.frame === currentFrame) {
            const pts = (currentSide === 'OD' && outlineData.is_stereo)
                ? outlineData.OD
                : outlineData.OS;
            if (pts && pts.length >= 3) {
                ctx.save();
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                // Stay 2px wide on screen regardless of zoom.  We're
                // inside the (scale * bps) transform, so divide.
                ctx.lineWidth = 2 / Math.max(0.01, scale * bps);
                // Sides on a background edge are drawn red (with the
                // "Red spread" dilation applied along the ring).
                _bgEdgeBand(currentSide);
                const red = _classifyPolyRed(pts, currentSide, _redSpread);
                _strokePoly(pts, (a, b, i) => red[i] ? RED : YELLOW);
                ctx.restore();
            }
        }

        // Cross-camera overlays: the other camera's aligned boundary,
        // and/or this camera's red runs patched from it.  Both need the
        // two cameras' polygons + BG-edge bands + the alignment.
        if ((showOtherBoundary || showRefinedBoundary) && outlineData
            && outlineData.is_stereo
            && outlineData.frame === currentFrame) {
            const curSide = (currentSide === 'OD') ? 'OD' : 'OS';
            const otherSide = (curSide === 'OD') ? 'OS' : 'OD';
            const curPts = (curSide === 'OD') ? outlineData.OD : outlineData.OS;
            const otherPts = (otherSide === 'OD') ? outlineData.OD : outlineData.OS;
            if (curPts && curPts.length >= 3
                && otherPts && otherPts.length >= 3) {
                // Both cameras' BG-edge bands feed the red test + the
                // red-aware alignment.
                _bgEdgeBand(curSide);
                _bgEdgeBand(otherSide);
                const align = _ensureOtherAlign(
                    curPts, otherPts, curSide, otherSide);
                ctx.save();
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.lineWidth = 2 / Math.max(0.01, scale * bps);
                // Other camera's boundary -- green, red where it runs
                // along the OTHER camera's BG edges (its own coords),
                // the whole path shifted by the alignment.  Red spread
                // is NOT applied here -- the other camera is treated
                // as a source of replacement segments, so its raw red
                // classification is what's relevant.
                if (showOtherBoundary) {
                    const oRed = _classifyPolyRed(otherPts, otherSide, 0);
                    _strokePoly(otherPts, (a, b, i) => oRed[i] ? RED : GREEN,
                                align.dx, align.dy);
                }
                // Refined boundary -- this camera's outline with red
                // runs patched from the other.  Provenance colours:
                // yellow = own, green = borrowed, red = unresolved.
                if (showRefinedBoundary) {
                    if (_refinedOutline === null
                        || _refinedOutlineData !== outlineData
                        || _refinedOutlineSide !== curSide) {
                        _refinedOutline = _refineOutline(
                            curPts, curSide, otherPts, otherSide, align,
                            _anchorDist, _anchorHops, _redSpread);
                        _refinedOutlineData = outlineData;
                        _refinedOutlineSide = curSide;
                    }
                    const ref = _refinedOutline;
                    const COL = { own: YELLOW, borrowed: GREEN,
                                  unresolved: RED };
                    _strokePoly(ref.pts,
                        (a, b, i) => COL[ref.segColors[i]] || YELLOW);
                }
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

                // 2. Forearm cone -- trapezoid anchored to the
                //    palm-heel line (thumb CMC <-> reflected ulnar
                //    heel) in BOTH position and orientation: base edge
                //    runs ALONG the heel line, cone axis runs
                //    PERPENDICULAR to it, so the top is parallel to
                //    the heel line.  Width = MCP spread x 1.2 (wide).
                //    Mirrors _build_forearm_cone.
                const mcpIdx = [5, 9, 13, 17].filter(_hasPt);
                const _crefl = _reflectThumbCmc(f);
                if (_hasPt(0) && mcpIdx.length && _hasPt(1) && _crefl) {
                    const wrist = f[0];
                    let cx = 0, cy = 0;
                    for (const j of mcpIdx) { cx += f[j][0]; cy += f[j][1]; }
                    cx /= mcpIdx.length; cy /= mcpIdx.length;
                    // Heel line: thumb CMC -> reflected ulnar heel.
                    let hx = _crefl[0] - f[1][0], hy = _crefl[1] - f[1][1];
                    const hn = Math.hypot(hx, hy);
                    if (hn > 1e-3) {
                        hx /= hn; hy /= hn;                 // heel-line dir
                        const px = hx, py = hy;             // base edge runs along it
                        // Cone axis = perpendicular to the heel line,
                        // pointing toward the elbow (positive dot with
                        // the rough wrist - centroid vector).
                        let ax = -hy, ay = hx;
                        if (ax * (wrist[0] - cx) + ay * (wrist[1] - cy) < 0) {
                            ax = -ax; ay = -ay;
                        }
                        const baseMid = [(f[1][0] + _crefl[0]) / 2,
                                          (f[1][1] + _crefl[1]) / 2];
                        let baseW = 30;
                        if (_hasPt(5) && _hasPt(17)) {
                            baseW = Math.hypot(f[5][0] - f[17][0],
                                                f[5][1] - f[17][1]) * 1.2;
                        }
                        const tipW = baseW * 1.8;
                        const len = 220;                  // full-res px
                        const ex = baseMid[0] + ax * len;
                        const ey = baseMid[1] + ay * len;
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

    // Mirrors labels.js exactly — one videoEl, rVFC when available,
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
        // Stages: Stabilize (stable.mp4), Compute Background (raw
        // median), Remove stump (refined), Hand Boundary (live).
        $('runStableBtn').addEventListener('click', computeStable);
        $('runMedianBtn').addEventListener('click', computeMedian);
        $('runRefineBtn').addEventListener('click', computeRefine);
        if ($('runOutlinesBtn'))
            $('runOutlinesBtn').addEventListener('click', computeOutlines);
        // "MP dilate (mask)" slider: a Remove-stump param.  While the
        // user drags it, the palm-zone preview flashes on over the
        // Background (raw median) view so they can see the hard
        // boundary they're setting; it auto-hides after the last input.
        const _palmGrowSlider = $('palmGrowSlider');
        const _palmGrowVal    = $('palmGrowVal');
        if (_palmGrowSlider && _palmGrowVal) {
            _palmGrowSlider.addEventListener('input', () => {
                _palmGrowVal.textContent = _palmGrowSlider.value;
                _flashBgZones();
            });
        }
        // "Skin color leniency" slider: a Remove-stump param.  While
        // dragged, the live skin-color mask + sample-region preview
        // flashes on over the Background (raw median) view so the user
        // sees what the leniency classifies as hand-coloured.
        // Leniency 0 skips the color-based refinement entirely, so both
        // MP-dilate knobs ("MP dilate (color sample)" and "MP dilate
        // (mask)") -- which feed only that refinement -- are dimmed +
        // disabled in that case.
        const _skinLenSlider = $('skinLeniencySlider');
        const _skinLenVal    = $('skinLeniencyVal');
        const _colorDilateSlider = $('colorDilateSlider');
        const _colorDilateVal    = $('colorDilateVal');
        function _syncMpDilateEnabled() {
            if (!_skinLenSlider) return;
            const off = parseFloat(_skinLenSlider.value) <= 0;
            for (const id of ['darkBoostSlider', 'colorDilateSlider',
                              'palmGrowSlider']) {
                const sl = $(id);
                if (!sl) continue;
                sl.disabled = off;
                const wrap = sl.parentElement;
                if (wrap) wrap.style.opacity = off ? '0.4' : '1';
            }
        }
        if (_skinLenSlider && _skinLenVal) {
            _skinLenSlider.addEventListener('input', () => {
                _skinLenVal.textContent =
                    parseFloat(_skinLenSlider.value).toFixed(2);
                _syncMpDilateEnabled();
                _flashSkinMask('bg');
            });
        }
        // "Dark-color boost" slider: a Remove-stump param.  Widens the
        // skin window for darker pixels; flashes the skin-color preview
        // over the Background (raw median) view, like skin leniency.
        const _darkBoostSlider = $('darkBoostSlider');
        const _darkBoostVal    = $('darkBoostVal');
        if (_darkBoostSlider && _darkBoostVal) {
            _darkBoostSlider.addEventListener('input', () => {
                _darkBoostVal.textContent =
                    parseFloat(_darkBoostSlider.value).toFixed(1);
                _flashSkinMask('bg');
            });
        }
        // "MP dilate (color sample)" slider: a Remove-stump param.
        // While dragged, the same skin-color mask + sample-region
        // preview flashes on over the Stabilized view (the sample
        // region tracks this slider); auto-hides after the last input.
        if (_colorDilateSlider && _colorDilateVal) {
            _colorDilateSlider.addEventListener('input', () => {
                _colorDilateVal.textContent = _colorDilateSlider.value;
                _flashSkinMask('stable');
            });
        }
        _syncMpDilateEnabled();
        const _fgDilateSlider = $('fgDilateSlider');
        const _fgDilateVal    = $('fgDilateVal');
        if (_fgDilateSlider && _fgDilateVal) {
            _fgDilateSlider.addEventListener('input', () => {
                _fgDilateVal.textContent = _fgDilateSlider.value;
                // Instant feedback: re-render so the on-canvas
                // dilated-skeleton preview tracks the slider.
                try { render(); } catch (_e) {}
                // Debounced backend fetch for the new outline / heatmap.
                if (showOutline || showFgFill || showOtherBoundary || showRefinedBoundary) scheduleOutlineFetch();
            });
        }
        // Background-edge retraction sliders + strand-clip slider: all
        // live -- refetch the outline so the change takes effect
        // immediately.  Each updates its label and reschedules; the
        // three "BG-edge ..." sliders also flash the background-edge
        // band preview on the canvas while dragged.
        const _outlineSliders = [
            ['outlineThreshSlider', 'outlineThreshVal',
             v => parseFloat(v).toFixed(2), false],
            ['bgEdgeThreshSlider', 'bgEdgeThreshVal',
             v => parseFloat(v).toFixed(2), true],
            ['bgEdgeBandSlider',   'bgEdgeBandVal',   v => v, true],
            ['bgEdgeClipSlider',   'bgEdgeClipVal',   v => v, true],
            ['fgOpenSlider',       'fgOpenVal',       v => v, false],
        ];
        for (const [sId, vId, fmt, flashEdge] of _outlineSliders) {
            const sl = $(sId), vl = $(vId);
            if (!sl || !vl) continue;
            sl.addEventListener('input', () => {
                vl.textContent = fmt(sl.value);
                if (flashEdge) _flashBgEdge();
                if (showOutline || showFgFill || showOtherBoundary || showRefinedBoundary) scheduleOutlineFetch();
            });
        }
        // Refined-boundary sliders -- pure client-side post-processing,
        // so no server refetch.  Update the label, invalidate the
        // cached alignment / refined polygon (anything they affect),
        // and re-render.
        const _refineSliders = [
            // Red spread is now CUR-only -- the alignment + green
            // replacement source use the raw classification -- so it
            // doesn't invalidate _otherAlign, only _refinedOutline.
            ['redSpreadSlider',  'redSpreadVal',  v => v, false],
            ['anchorHopsSlider', 'anchorHopsVal', v => v, false],
            ['anchorDistSlider', 'anchorDistVal', v => v, false],
        ];
        for (const [sId, vId, fmt, invalidAlign] of _refineSliders) {
            const sl = $(sId), vl = $(vId);
            if (!sl || !vl) continue;
            sl.addEventListener('input', () => {
                vl.textContent = fmt(sl.value);
                if (invalidAlign) _otherAlign = null;
                _refinedOutline = null;
                try { render(); } catch (_e) {}
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
        if ($('ovRefined'))
            $('ovRefined').addEventListener('change', e => e.target.checked && _setOverlay('refined'));
        // Hand-boundary checkbox: trigger a fresh fetch when turned on
        // so the user sees the contour for the current frame
        // immediately; clear the cached data when turned off.
        $('cbShowOutline').addEventListener('change', e => {
            showOutline = e.target.checked;
            if (showOutline) {
                fetchOutline();
            } else {
                if (!showFgFill && !showOtherBoundary && !showRefinedBoundary) outlineData = null;
                render();
            }
        });
        // Other-camera boundary checkbox: needs outlineData (which
        // carries both cameras' polygons), so fetch it if not present;
        // turning off just re-renders.
        if ($('cbShowOtherBoundary')) {
            $('cbShowOtherBoundary').addEventListener('change', e => {
                showOtherBoundary = e.target.checked;
                if (showOtherBoundary) {
                    _otherAlign = null;
                    if (outlineData && outlineData.frame === currentFrame) render();
                    else fetchOutline();
                } else {
                    if (!showFgFill && !showOutline && !showRefinedBoundary) outlineData = null;
                    render();
                }
            });
        }
        // Refined-boundary checkbox: same data needs as the other-camera
        // overlay (both cameras' polygons).
        if ($('cbShowRefinedBoundary')) {
            $('cbShowRefinedBoundary').addEventListener('change', e => {
                showRefinedBoundary = e.target.checked;
                if (showRefinedBoundary) {
                    _otherAlign = null; _refinedOutline = null;
                    if (outlineData && outlineData.frame === currentFrame) render();
                    else fetchOutline();
                } else {
                    if (!showFgFill && !showOutline && !showOtherBoundary) outlineData = null;
                    render();
                }
            });
        }
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

        // Speed slider (matches videos.js / labels.js pattern).
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
        // toggling play.  Mirror labels.js: blur after pointerup so the
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
        // conventions as the video browser and skeleton page.
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
