/* Interactive Deidentify/Blur Page */
const deid = (() => {
    // ── State ──
    let subjects = [];
    let subjectId = null;
    let subjectName = '';
    let trials = [];
    let currentTrialIdx = -1;
    let trialMeta = null;     // current trial metadata

    let currentFrame = 0;
    let totalFrames = 0;
    let fps = 30;
    let playing = false;
    let playTimer = null;

    // Camera
    let cameraMode = 'stereo';
    let cameraNames = ['OS', 'OD'];
    let currentSide = 'OS';
    let hasMediapipe = false;

    // Face detection results: [{frame, faces: [{x1,y1,x2,y2,side}]}]
    let faceDetections = [];
    // Active blur spots: [{id, spot_type, x, y, radius, frame_start, frame_end, side}]
    let blurSpots = [];
    let nextSpotId = 1;
    let selectedSpotId = null;
    let addingCustom = false;

    // Hand overlay
    let handOverlayEnabled = false;
    let handLandmarks = [];   // [{x, y, side}] for current frame (after smoothing)
    let handLandmarksBulk = {}; // {frameNum: [{x, y, side}]} all frames
    let handTemporalSmooth = 0; // temporal smoothing window (frames each direction)
    let handMaskRadius = 10;
    let handSmooth = 10;  // morphological close: dilate then erode
    let handMaskEnabled = true;

    // Canvas
    let canvas, ctx;
    let currentImage = null;
    let imgW = 0, imgH = 0;
    let scale = 1, offsetX = 0, offsetY = 0;
    let hasUserZoom = false;

    // Pan state
    let panning = false;
    let panStart = null;

    // Timeline
    let tlCanvas, tlCtx;
    let tlDragSpot = null;    // spot being dragged (blur spot object, segment object, or 'newhand')
    let tlDragEdge = null;    // 'start', 'end', 'move', or 'create'
    let tlDragStartX = null;  // mouse X at drag start
    let tlDragOrigRange = null; // {start, end} at drag start
    let tlDragCreateFrame = null; // frame at start of hand segment creation drag
    let handCoverage = [];    // array of frame numbers with MP hand data
    // Multiple hand protection segments: [{id, start, end, radius}]
    let handProtectSegments = [];
    let nextHandSegId = 1;
    let selectedHandSegId = null;

    // Debounce save timer
    let saveTimer = null;

    // ── Init ──
    async function init() {
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');

        // Load subjects
        try {
            subjects = await API.get('/api/subjects');
        } catch (e) {
            subjects = [];
        }

        const sel = document.getElementById('subjectSelect');
        sel.innerHTML = '<option value="">Select subject...</option>';
        subjects.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = s.name;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', () => {
            if (sel.value) loadSubject(parseInt(sel.value));
        });

        // Canvas events
        canvas.addEventListener('click', onCanvasClick);
        canvas.addEventListener('wheel', onWheel, { passive: false });
        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('mouseleave', onMouseUp);

        // Timeline setup
        tlCanvas = document.getElementById('timelineCanvas');
        tlCtx = tlCanvas ? tlCanvas.getContext('2d') : null;
        if (tlCanvas) {
            setupTimeline();
        }

        // Keyboard
        document.addEventListener('keydown', onKeyDown);

        // Resize
        window.addEventListener('resize', () => {
            if (!hasUserZoom) fitImage();
            render();
            renderTimeline();
        });

        // Auto-select subject from URL param or if only one
        const params = new URLSearchParams(window.location.search);
        const urlSubject = params.get('subject');
        if (urlSubject) {
            sel.value = urlSubject;
            loadSubject(parseInt(urlSubject));
        } else if (subjects.length === 1) {
            sel.value = subjects[0].id;
            loadSubject(subjects[0].id);
        }
    }

    // ── Load subject ──
    async function loadSubject(sid) {
        subjectId = sid;
        faceDetections = [];
        blurSpots = [];
        selectedSpotId = null;
        handLandmarks = [];

        try {
            const data = await API.get(`/api/deidentify/${sid}/trials`);
            subjectName = data.subject.name;
            trials = data.trials;
            cameraMode = data.subject.camera_mode || 'stereo';
            hasMediapipe = data.has_mediapipe || false;

            // Set camera names
            if (cameraMode === 'single') {
                cameraNames = [data.camera_names ? data.camera_names[0] : 'OS'];
                currentSide = cameraNames[0];
            } else if (cameraMode === 'multicam' && trials.length > 0
                       && trials[0].cameras && trials[0].cameras.length > 0) {
                cameraNames = trials[0].cameras.map(c => c.name);
                currentSide = cameraNames[0];
            } else {
                cameraNames = data.camera_names || ['OS', 'OD'];
                currentSide = cameraNames[0];
            }
        } catch (e) {
            document.getElementById('statusMsg').textContent = 'Error loading subject: ' + e.message;
            return;
        }

        // Update camera label
        const camLabel = document.getElementById('cameraLabel');
        if (camLabel) {
            camLabel.style.display = (cameraMode === 'single') ? 'none' : '';
            camLabel.textContent = currentSide;
        }

        // Update hand section availability
        _updateHandSection();

        // Build trial buttons
        const btns = document.getElementById('trialBtns');
        btns.innerHTML = '';
        trials.forEach((t, i) => {
            const btn = document.createElement('button');
            btn.className = 'trial-btn';
            btn.textContent = t.trial_name;
            btn.onclick = () => selectTrial(i);
            btns.appendChild(btn);
        });

        if (trials.length > 0) selectTrial(0);
    }

    // ── Update hand overlay section availability ──
    function _updateHandSection() {
        const handSection = document.getElementById('handSection');
        if (!handSection) return;

        // Grey out controls if no MP labels, but keep MP button always active
        const inputs = handSection.querySelectorAll('input[type=checkbox], input[type=range]');
        inputs.forEach(el => { el.disabled = !hasMediapipe; });
        if (!hasMediapipe) {
            handSection.classList.add('hand-disabled');
        } else {
            handSection.classList.remove('hand-disabled');
        }

        // Update button text — always visible
        const mpBtn = document.getElementById('goMediapipeBtn');
        if (mpBtn) {
            mpBtn.textContent = hasMediapipe ? 'Re-run MediaPipe' : 'Run MediaPipe';
        }
    }

    // ── Select trial ──
    async function selectTrial(idx) {
        if (playing) togglePlay();
        currentTrialIdx = idx;
        trialMeta = trials[idx];
        currentFrame = trialMeta.start_frame;
        totalFrames = trialMeta.frame_count;
        fps = trialMeta.fps || 30;
        handLandmarks = [];
        handLandmarksBulk = {}; // clear bulk cache for new trial

        // Load saved face detections from DB
        try {
            const fdResp = await API.get(`/api/deidentify/${subjectId}/face-detections?trial_idx=${idx}`);
            faceDetections = fdResp.faces || [];
            if (faceDetections.length > 0) {
                const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
                document.getElementById('faceDetStatus').textContent =
                    `Loaded ${nFaces} frames with faces (saved)`;
            }
        } catch (e) {
            faceDetections = [];
        }

        // Update UI
        document.querySelectorAll('.trial-btn').forEach((b, i) => {
            b.classList.toggle('active', i === idx);
        });

        // (frame slider removed — timeline handles navigation)

        // Load saved blur specs
        try {
            const resp = await API.get(`/api/deidentify/${subjectId}/blur-specs?trial_idx=${idx}`);
            blurSpots = (resp.specs || []).map(s => ({ ...s, id: nextSpotId++ }));
        } catch (e) {
            blurSpots = [];
        }

        // Load hand protection segments
        handProtectSegments = [];
        selectedHandSegId = null;
        try {
            const hs = await API.get(`/api/deidentify/${subjectId}/hand-settings?trial_idx=${idx}`);
            handMaskRadius = hs.mask_radius || 30;
            document.getElementById('handRadiusSlider').value = handMaskRadius;
            document.getElementById('handRadiusVal').textContent = handMaskRadius;
            if (hs.segments && hs.segments.length > 0) {
                handProtectSegments = hs.segments.map(s => ({
                    id: nextHandSegId++,
                    start: s.start,
                    end: s.end,
                    radius: s.radius || handMaskRadius,
                    smooth: s.smooth != null ? s.smooth : handSmooth,
                }));
            }
        } catch (e) {}

        // Reset zoom for new trial
        hasUserZoom = false;
        scale = 1; offsetX = 0; offsetY = 0;

        // Load hand coverage for timeline
        handCoverage = [];
        if (hasMediapipe) {
            try {
                const hc = await API.get(`/api/deidentify/${subjectId}/hand-coverage?trial_idx=${idx}`);
                handCoverage = hc.frames || [];
            } catch (e) {}
        }

        await loadFrame(currentFrame);
        renderSpotList();
        renderTimeline();
    }

    // ── Frame loading ──
    async function loadFrame(frameNum) {
        currentFrame = frameNum;
        const url = `/api/deidentify/${subjectId}/frame?trial_idx=${currentTrialIdx}&frame_num=${frameNum}&side=${encodeURIComponent(currentSide)}`;

        try {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = url;
            });
            currentImage = img;
            imgW = img.width;
            imgH = img.height;
        } catch (e) {
            return;
        }

        if (!hasUserZoom) fitImage();

        // Update display
        const localFrame = frameNum - (trialMeta ? trialMeta.start_frame : 0);
        document.getElementById('frameDisplay').textContent =
            `Frame: ${localFrame} / ${totalFrames - 1}`;

        // Load hand landmarks from bulk cache with temporal smoothing
        const inProtectSeg = handProtectSegments.some(s => frameNum >= s.start && frameNum <= s.end);
        const needHands = hasMediapipe && (handOverlayEnabled || inProtectSeg);
        if (needHands) {
            await _loadBulkLandmarks();
            _applyTemporalSmoothing();
        } else {
            handLandmarks = [];
        }

        render();
        renderTimeline();
    }

    function fitImage() {
        if (!currentImage || !canvas) return;
        const cw = canvas.clientWidth;
        const ch = canvas.clientHeight;
        canvas.width = cw;
        canvas.height = ch;
        scale = Math.min(cw / imgW, ch / imgH);
        offsetX = (cw - imgW * scale) / 2;
        offsetY = (ch - imgH * scale) / 2;
    }

    function resetZoom() {
        hasUserZoom = false;
        fitImage();
        render();
    }

    // ── Camera toggle ──
    function toggleSide() {
        if (cameraMode === 'single') return;
        const idx = cameraNames.indexOf(currentSide);
        const newIdx = (idx + 1) % cameraNames.length;
        currentSide = cameraNames[newIdx];

        const camLabel = document.getElementById('cameraLabel');
        if (camLabel) camLabel.textContent = currentSide;

        loadFrame(currentFrame);
    }

    // ── Go to MediaPipe page for this subject ──
    function goToMediapipe() {
        if (!subjectId) return;
        // Preserve current frame and zoom state via session storage
        sessionStorage.setItem('deid_returnFrame', currentFrame);
        sessionStorage.setItem('deid_returnSubject', subjectId);
        window.location.href = `/mediapipe-select?subject=${subjectId}`;
    }

    // ── Side label mapping (stereo: left/right ↔ camera names) ──
    function _sideLabel() {
        // Map currentSide camera name to the detection side label
        if (cameraMode !== 'stereo') return 'full';
        if (cameraNames.length >= 2 && currentSide === cameraNames[1]) return 'right';
        return 'left';
    }

    function _facesForCurrentSide(entry) {
        if (!entry || !entry.faces) return [];
        if (cameraMode === 'single') return entry.faces;
        const label = _sideLabel();
        return entry.faces.filter(f => (f.side || 'full') === label || (f.side || 'full') === 'full');
    }

    // ── Get face spot position for current frame (tracks detection centroid) ──
    function _getSpotPosition(spot) {
        // For face spots: track the nearest face detection centroid + offset
        if (spot.spot_type === 'face' && faceDetections.length > 0) {
            const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
            if (localFrame >= 0 && localFrame < faceDetections.length) {
                const faces = _facesForCurrentSide(faceDetections[localFrame]);
                if (faces.length > 0) {
                    // Find closest face to spot's reference position
                    let bestFace = faces[0];
                    let bestDist = Infinity;
                    for (const f of faces) {
                        const cx = (f.x1 + f.x2) / 2;
                        const cy = (f.y1 + f.y2) / 2;
                        const d = Math.sqrt((cx - spot.x) ** 2 + (cy - spot.y) ** 2);
                        if (d < bestDist) {
                            bestDist = d;
                            bestFace = f;
                        }
                    }
                    return {
                        x: (bestFace.x1 + bestFace.x2) / 2 + (spot.offset_x || 0),
                        y: (bestFace.y1 + bestFace.y2) / 2 + (spot.offset_y || 0),
                    };
                }
            }
        }
        // Fallback: static position + offset
        return { x: spot.x + (spot.offset_x || 0), y: spot.y + (spot.offset_y || 0) };
    }

    // ── Build smoothed hand protection mask (morphological close approx) ──
    function _buildHandMask(landmarks, radiusPx, smoothPx, w, h) {
        // Step 1: Draw circles at base radius + forearm triangle
        const c1 = document.createElement('canvas');
        c1.width = w; c1.height = h;
        const ctx1 = c1.getContext('2d');
        ctx1.fillStyle = '#fff';

        // Draw circles for all hand landmarks
        for (const lm of landmarks) {
            ctx1.beginPath();
            ctx1.arc(offsetX + lm.x * scale, offsetY + lm.y * scale, radiusPx, 0, Math.PI * 2);
            ctx1.fill();
        }

        // Draw forearm triangle: pinky MCP (joint 17) → elbow → thumb CMC (joint 1)
        // Dilated by drawing with thick rounded lines + filled
        const pinkyMCP = landmarks.find(l => l.type === 'hand' && l.joint === 17);
        const thumbCMC = landmarks.find(l => l.type === 'hand' && l.joint === 1);
        // Use pose wrists (15=left, 16=right) to determine which arm side,
        // then pick the matching elbow (13=left, 14=right)
        const poseWrists = landmarks.filter(l => l.type === 'pose' && (l.joint === 15 || l.joint === 16));
        const handWrist = landmarks.find(l => l.type === 'hand' && l.joint === 0);
        let elbow = null;
        if (handWrist && poseWrists.length > 0) {
            // Find which pose wrist is closest to the hand wrist
            let closestPoseWrist = poseWrists[0];
            if (poseWrists.length === 2) {
                const d0 = Math.hypot(poseWrists[0].x - handWrist.x, poseWrists[0].y - handWrist.y);
                const d1 = Math.hypot(poseWrists[1].x - handWrist.x, poseWrists[1].y - handWrist.y);
                closestPoseWrist = d0 < d1 ? poseWrists[0] : poseWrists[1];
            }
            // Pose wrist 15 → left elbow 13, pose wrist 16 → right elbow 14
            const elbowJoint = closestPoseWrist.joint === 15 ? 13 : 14;
            elbow = landmarks.find(l => l.type === 'pose' && l.joint === elbowJoint);
        }
        // Fallback: just grab any elbow
        if (!elbow) {
            elbow = landmarks.find(l => l.type === 'pose' && (l.joint === 13 || l.joint === 14));
        }

        if (pinkyMCP && thumbCMC && elbow) {
            const pts = [pinkyMCP, elbow, thumbCMC].map(p => ({
                sx: offsetX + p.x * scale,
                sy: offsetY + p.y * scale,
            }));

            // Filled triangle
            ctx1.beginPath();
            ctx1.moveTo(pts[0].sx, pts[0].sy);
            ctx1.lineTo(pts[1].sx, pts[1].sy);
            ctx1.lineTo(pts[2].sx, pts[2].sy);
            ctx1.closePath();
            ctx1.fill();

            // Thick rounded stroke to dilate the triangle by radiusPx
            ctx1.lineWidth = radiusPx * 2;
            ctx1.lineJoin = 'round';
            ctx1.lineCap = 'round';
            ctx1.strokeStyle = '#fff';
            ctx1.beginPath();
            ctx1.moveTo(pts[0].sx, pts[0].sy);
            ctx1.lineTo(pts[1].sx, pts[1].sy);
            ctx1.lineTo(pts[2].sx, pts[2].sy);
            ctx1.closePath();
            ctx1.stroke();
        }

        if (smoothPx <= 0) return c1;

        // Step 2: Blur the mask (dilate approximation)
        const c2 = document.createElement('canvas');
        c2.width = w; c2.height = h;
        const ctx2 = c2.getContext('2d');
        ctx2.filter = `blur(${smoothPx}px)`;
        ctx2.drawImage(c1, 0, 0);
        ctx2.filter = 'none';

        // Step 3: Threshold — draw the blurred result multiple times with
        // 'lighter' compositing to push alpha toward 1, simulating a threshold
        const c3 = document.createElement('canvas');
        c3.width = w; c3.height = h;
        const ctx3 = c3.getContext('2d');
        ctx3.globalCompositeOperation = 'source-over';
        // Each pass roughly doubles the alpha; 8 passes: 0.5^8 ≈ 0 becomes visible
        for (let i = 0; i < 8; i++) {
            ctx3.drawImage(c2, 0, 0);
        }

        // Step 4: Make it a clean binary mask by thresholding via getImageData
        const imgData = ctx3.getImageData(0, 0, w, h);
        const d = imgData.data;
        for (let i = 3; i < d.length; i += 4) {
            d[i] = d[i] > 30 ? 255 : 0;  // threshold alpha
        }
        ctx3.putImageData(imgData, 0, 0);

        return c3;
    }

    // ── Render ──
    function render() {
        if (!ctx || !canvas) return;
        const cw = canvas.width;
        const ch = canvas.height;
        ctx.clearRect(0, 0, cw, ch);

        if (!currentImage) return;

        // Draw video frame
        ctx.drawImage(currentImage, offsetX, offsetY, imgW * scale, imgH * scale);

        // Draw face detections (blue dashed rectangles) — filtered by current side
        const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
        if (faceDetections.length > localFrame) {
            const faces = _facesForCurrentSide(faceDetections[localFrame]);
            if (faces.length > 0) {
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = 'rgba(33,150,243,0.8)';
                ctx.lineWidth = 2;
                for (const f of faces) {
                    const sx = offsetX + f.x1 * scale;
                    const sy = offsetY + f.y1 * scale;
                    const sw = (f.x2 - f.x1) * scale;
                    const sh = (f.y2 - f.y1) * scale;
                    ctx.strokeRect(sx, sy, sw, sh);
                }
                ctx.setLineDash([]);
            }
        }

        // Find active hand protection segment for current frame
        const curSideLabel = _sideLabel();
        const visibleLandmarks = handLandmarks.filter(lm =>
            (lm.side || 'full') === curSideLabel || (lm.side || 'full') === 'full'
        );
        const activeSeg = handProtectSegments.find(s =>
            currentFrame >= s.start && currentFrame <= s.end
        );
        const handProtectActive = activeSeg && visibleLandmarks.length > 0;
        const activeProtectRadius = activeSeg ? (activeSeg.radius || handMaskRadius) : handMaskRadius;
        const activeSmooth = activeSeg ? (activeSeg.smooth != null ? activeSeg.smooth : handSmooth) : handSmooth;

        // Build smoothed hand mask (morphological close via blur+threshold)
        let handMaskCanvas = null;
        if (handProtectActive) {
            handMaskCanvas = _buildHandMask(
                visibleLandmarks, activeProtectRadius * scale, activeSmooth * scale, cw, ch
            );
        }

        // Draw ALL blur spots on one offscreen canvas, then subtract hand protection
        const hasVisibleSpots = blurSpots.some(s => {
            if (currentFrame < s.frame_start || currentFrame > s.frame_end) return false;
            const ss = s.side || 'full';
            return ss === 'full' || ss === curSideLabel;
        });

        if (hasVisibleSpots) {
            // Offscreen canvas for compositing blur fills with hand subtraction
            const offCanvas = document.createElement('canvas');
            offCanvas.width = cw;
            offCanvas.height = ch;
            const off = offCanvas.getContext('2d');

            for (const spot of blurSpots) {
                if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
                const spotSide = spot.side || 'full';
                if (spotSide !== 'full' && spotSide !== curSideLabel) continue;

                const pos = _getSpotPosition(spot);
                const sx = offsetX + pos.x * scale;
                const sy = offsetY + pos.y * scale;
                const isFace = spot.spot_type === 'face';
                const isSelected = spot.id === selectedSpotId;
                const sw = (spot.width || spot.radius * 2) * scale / 2;
                const sh = (spot.height || spot.radius * 2) * scale / 2;

                const cr = isFace ? 33 : 244;
                const cg = isFace ? 150 : 67;
                const cb = isFace ? 243 : 54;
                const fillAlpha = isSelected ? 0.35 : 0.2;

                off.fillStyle = `rgba(${cr},${cg},${cb},${fillAlpha})`;
                off.beginPath();
                off.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
                off.fill();
            }

            // Subtract hand protection mask from ALL blur fills
            if (handMaskCanvas) {
                off.globalCompositeOperation = 'destination-out';
                off.drawImage(handMaskCanvas, 0, 0);
                off.globalCompositeOperation = 'source-over';
            }

            // Composite result onto main canvas
            ctx.drawImage(offCanvas, 0, 0);

            // Draw outlines on main canvas (not affected by subtraction)
            for (const spot of blurSpots) {
                if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
                const spotSide = spot.side || 'full';
                if (spotSide !== 'full' && spotSide !== curSideLabel) continue;

                const pos = _getSpotPosition(spot);
                const sx = offsetX + pos.x * scale;
                const sy = offsetY + pos.y * scale;
                const isFace = spot.spot_type === 'face';
                const isSelected = spot.id === selectedSpotId;
                const sw = (spot.width || spot.radius * 2) * scale / 2;
                const sh = (spot.height || spot.radius * 2) * scale / 2;

                const cr = isFace ? 33 : 244;
                const cg = isFace ? 150 : 67;
                const cb = isFace ? 243 : 54;

                ctx.beginPath();
                ctx.ellipse(sx, sy, sw, sh, 0, 0, Math.PI * 2);
                ctx.strokeStyle = isSelected
                    ? `rgba(${cr},${cg},${cb},0.9)`
                    : `rgba(${cr},${cg},${cb},0.5)`;
                ctx.lineWidth = isSelected ? 2 : 1;
                ctx.stroke();
            }
        }

        // Draw landmarks — green for hand, orange for pose
        if (handOverlayEnabled && visibleLandmarks.length > 0) {
            for (const lm of visibleLandmarks) {
                const sx = offsetX + lm.x * scale;
                const sy = offsetY + lm.y * scale;
                if (lm.type === 'pose') {
                    // Pose: orange diamond shape to distinguish from hand circles
                    ctx.fillStyle = 'rgba(255,152,0,0.8)';
                    ctx.beginPath();
                    ctx.moveTo(sx, sy - 5);
                    ctx.lineTo(sx + 5, sy);
                    ctx.lineTo(sx, sy + 5);
                    ctx.lineTo(sx - 5, sy);
                    ctx.closePath();
                    ctx.fill();
                } else {
                    // Hand: green circle
                    ctx.fillStyle = 'rgba(76,175,80,0.7)';
                    ctx.beginPath();
                    ctx.arc(sx, sy, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }

        // Draw hand protection outline from the smoothed mask
        if (handMaskCanvas) {
            // Extract outline: erode the mask inward by 2px
            const hOff = document.createElement('canvas');
            hOff.width = cw;
            hOff.height = ch;
            const hCtx = hOff.getContext('2d');

            // Draw the smoothed mask
            hCtx.drawImage(handMaskCanvas, 0, 0);

            // Erode inward by 2px: shrink the mask slightly and erase it
            const innerCanvas = document.createElement('canvas');
            innerCanvas.width = cw;
            innerCanvas.height = ch;
            const ic = innerCanvas.getContext('2d');
            // Shrink by drawing at reduced radius
            const shrinkR = activeProtectRadius * scale - 2;
            if (shrinkR > 0) {
                if (activeSmooth > 0) {
                    // Rebuild mask at slightly smaller radius
                    ic.drawImage(_buildHandMask(
                        visibleLandmarks, shrinkR, activeSmooth * scale, cw, ch
                    ), 0, 0);
                } else {
                    ic.fillStyle = '#fff';
                    for (const lm of visibleLandmarks) {
                        ic.beginPath();
                        ic.arc(offsetX + lm.x * scale, offsetY + lm.y * scale, shrinkR, 0, Math.PI * 2);
                        ic.fill();
                    }
                }
                hCtx.globalCompositeOperation = 'destination-out';
                hCtx.drawImage(innerCanvas, 0, 0);
                hCtx.globalCompositeOperation = 'source-over';
            }

            // Tint green
            hCtx.globalCompositeOperation = 'source-in';
            hCtx.fillStyle = 'rgba(76,175,80,0.8)';
            hCtx.fillRect(0, 0, cw, ch);

            ctx.drawImage(hOff, 0, 0);
        }

        // "Adding custom" cursor indicator
        if (addingCustom) {
            ctx.fillStyle = 'rgba(255,152,0,0.15)';
            ctx.strokeStyle = 'rgba(255,152,0,0.6)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.arc(cw / 2, ch / 2, 30, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    // ── Canvas click ──
    function onCanvasClick(e) {
        // Don't handle clicks after a pan drag
        if (panning) return;

        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        // Convert to image coordinates
        const ix = (cx - offsetX) / scale;
        const iy = (cy - offsetY) / scale;

        if (ix < 0 || iy < 0 || ix > imgW || iy > imgH) return;

        // Check if clicking on a face detection → create face spot
        const localFrame = currentFrame - (trialMeta ? trialMeta.start_frame : 0);
        if (faceDetections.length > localFrame) {
            const entry = faceDetections[localFrame];
            if (entry && entry.faces) {
                for (const f of entry.faces) {
                    if (ix >= f.x1 && ix <= f.x2 && iy >= f.y1 && iy <= f.y2) {
                        const fw = f.x2 - f.x1;
                        const fh = f.y2 - f.y1;
                        const spot = {
                            id: nextSpotId++,
                            spot_type: 'face',
                            x: Math.round((f.x1 + f.x2) / 2),
                            y: Math.round((f.y1 + f.y2) / 2),
                            radius: Math.round(Math.max(fw, fh) / 2 * 1.2),
                            width: Math.round(fw * 1.2),
                            height: Math.round(fh * 1.2),
                            offset_x: 0,
                            offset_y: 0,
                            frame_start: trialMeta.start_frame,
                            frame_end: trialMeta.end_frame,
                            side: f.side || 'full',
                        };
                        blurSpots.push(spot);
                        selectedSpotId = spot.id;
                        renderSpotList();
                        updateSpotControls();
                        scheduleSave();
                        render();
                        return;
                    }
                }
            }
        }

        // Check if clicking on an existing blur spot → select it
        for (const spot of blurSpots) {
            if (currentFrame < spot.frame_start || currentFrame > spot.frame_end) continue;
            const dx = ix - spot.x;
            const dy = iy - spot.y;
            if (Math.sqrt(dx * dx + dy * dy) <= spot.radius) {
                selectedSpotId = spot.id;
                renderSpotList();
                updateSpotControls();
                render();
                return;
            }
        }

        // Adding custom spot mode
        if (addingCustom) {
            const spot = {
                id: nextSpotId++,
                spot_type: 'custom',
                x: Math.round(ix),
                y: Math.round(iy),
                radius: 40,
                width: 80,
                height: 80,
                offset_x: 0,
                offset_y: 0,
                frame_start: Math.max(trialMeta.start_frame, currentFrame - 30),
                frame_end: Math.min(trialMeta.end_frame, currentFrame + 30),
                side: 'full',
            };
            blurSpots.push(spot);
            selectedSpotId = spot.id;
            addingCustom = false;
            document.getElementById('addCustomBtn').style.background = '';
            renderSpotList();
            updateSpotControls();
            scheduleSave();
            render();
            return;
        }

        // Click on nothing → deselect
        selectedSpotId = null;
        renderSpotList();
        updateSpotControls();
        render();
    }

    // ── Zoom ──
    function onWheel(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
        const newScale = scale * zoomFactor;

        offsetX = mx - (mx - offsetX) * (newScale / scale);
        offsetY = my - (my - offsetY) * (newScale / scale);
        scale = newScale;
        hasUserZoom = true;
        render();
    }

    // ── Pan (right-click drag or middle-click drag) ──
    function onMouseDown(e) {
        // Right-click or middle-click for pan
        if (e.button === 1 || e.button === 2) {
            e.preventDefault();
            panning = true;
            panStart = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
            canvas.style.cursor = 'grabbing';
            return;
        }
        // Left-click pan when zoomed in (hold and drag)
        if (e.button === 0 && hasUserZoom && !addingCustom) {
            panning = true;
            panStart = { x: e.clientX, y: e.clientY, ox: offsetX, oy: offsetY };
            canvas.style.cursor = 'grabbing';
        }
    }

    function onMouseMove(e) {
        if (!panning || !panStart) return;
        offsetX = panStart.ox + (e.clientX - panStart.x);
        offsetY = panStart.oy + (e.clientY - panStart.y);
        render();
    }

    function onMouseUp(e) {
        if (panning) {
            canvas.style.cursor = '';
            // If mouse barely moved, don't suppress click
            if (panStart) {
                const dx = Math.abs(e.clientX - panStart.x);
                const dy = Math.abs(e.clientY - panStart.y);
                if (dx < 3 && dy < 3) {
                    panning = false;
                    panStart = null;
                    return; // allow click to fire
                }
            }
            // Suppress click after real pan drag
            setTimeout(() => { panning = false; }, 50);
            panStart = null;
        }
    }

    // ── Keyboard ──
    function onKeyDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        if (e.key === 'ArrowRight' || e.key === 'd') {
            e.preventDefault();
            seekFrame(Math.min(currentFrame + 1, trialMeta ? trialMeta.end_frame : 0));
        } else if (e.key === 'ArrowLeft' || e.key === 'a') {
            e.preventDefault();
            seekFrame(Math.max(currentFrame - 1, trialMeta ? trialMeta.start_frame : 0));
        } else if (e.key === ' ') {
            e.preventDefault();
            togglePlay();
        } else if (e.key === 'e') {
            toggleSide();
        } else if (e.key === 'r') {
            resetZoom();
        } else if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedHandSegId !== null) {
                deleteSelectedHandSeg();
            } else if (selectedSpotId !== null) {
                deleteSpot(selectedSpotId);
            }
        } else if (e.key === 'Escape') {
            if (addingCustom) {
                addingCustom = false;
                document.getElementById('addCustomBtn').style.background = '';
                render();
            }
        }
    }

    // ── Frame navigation ──
    function seekFrame(n) {
        if (!trialMeta) return;
        n = Math.max(trialMeta.start_frame, Math.min(trialMeta.end_frame, n));
        loadFrame(n);
    }

    function togglePlay() {
        const btn = document.getElementById('playBtn');
        if (playing) {
            // Stop
            playing = false;
            if (playTimer) { clearInterval(playTimer); playTimer = null; }
            btn.innerHTML = '&#9654;';
        } else {
            if (!trialMeta) return;
            // Start
            playing = true;
            btn.innerHTML = '&#9616;&#9616;';
            let loading = false; // prevent overlapping loads
            playTimer = setInterval(() => {
                if (!playing) return;
                if (loading) return;
                if (currentFrame >= trialMeta.end_frame) {
                    togglePlay();
                    return;
                }
                loading = true;
                currentFrame++;
                // Simple sync render: just update frame and draw
                const url = `/api/deidentify/${subjectId}/frame?trial_idx=${currentTrialIdx}&frame_num=${currentFrame}&side=${encodeURIComponent(currentSide)}`;
                const img = new Image();
                img.onload = () => {
                    currentImage = img;
                    imgW = img.width;
                    imgH = img.height;
                    const localFrame = currentFrame - trialMeta.start_frame;
                    document.getElementById('frameDisplay').textContent =
                        `Frame: ${localFrame} / ${totalFrames - 1}`;
                    render();
                    loading = false;
                };
                img.onerror = () => { loading = false; };
                img.src = url;
            }, 1000 / fps);
        }
    }

    // ── Face detection ──
    async function detectFaces() {
        if (!subjectId || currentTrialIdx < 0) return;

        const btn = document.getElementById('detectFacesBtn');
        const status = document.getElementById('faceDetStatus');
        btn.disabled = true;
        status.textContent = 'Detecting faces...';

        try {
            const result = await API.post(`/api/deidentify/${subjectId}/detect-faces`, {
                trial_idx: currentTrialIdx,
            });
            faceDetections = result.faces || [];
            const nFaces = faceDetections.filter(f => f.faces.length > 0).length;
            status.textContent = `Found faces in ${nFaces}/${faceDetections.length} frames`;

            // Auto-create blur spots from unique face clusters
            _autoCreateFaceSpots();
            renderSpotList();
            scheduleSave();
            render();
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
        }
        btn.disabled = false;
    }

    // ── Auto-create blur spots from face detections ──
    function _autoCreateFaceSpots() {
        if (!faceDetections.length || !trialMeta) return;

        // Find the most common face position per side (cluster by center)
        const clusters = {}; // key: side → [{cx, cy, w, h, count, firstFrame, lastFrame}]

        for (const entry of faceDetections) {
            for (const f of entry.faces) {
                const side = f.side || 'full';
                if (!clusters[side]) clusters[side] = [];

                const cx = (f.x1 + f.x2) / 2;
                const cy = (f.y1 + f.y2) / 2;
                const w = f.x2 - f.x1;
                const h = f.y2 - f.y1;

                // Find nearest existing cluster
                let matched = false;
                for (const c of clusters[side]) {
                    const dist = Math.sqrt((cx - c.cx) ** 2 + (cy - c.cy) ** 2);
                    if (dist < Math.max(w, h) * 1.5) {
                        // Update running average
                        c.cx = (c.cx * c.count + cx) / (c.count + 1);
                        c.cy = (c.cy * c.count + cy) / (c.count + 1);
                        c.w = Math.max(c.w, w);
                        c.h = Math.max(c.h, h);
                        c.count++;
                        c.firstFrame = Math.min(c.firstFrame, entry.frame);
                        c.lastFrame = Math.max(c.lastFrame, entry.frame);
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    clusters[side].push({
                        cx, cy, w, h, count: 1,
                        firstFrame: entry.frame, lastFrame: entry.frame,
                    });
                }
            }
        }

        // Remove existing auto-generated face spots for this trial
        blurSpots = blurSpots.filter(s => s.spot_type !== 'face');

        // Create a blur spot for each significant cluster
        for (const [side, clusterList] of Object.entries(clusters)) {
            for (const c of clusterList) {
                if (c.count < 3) continue; // Skip noise (< 3 frames)
                blurSpots.push({
                    id: nextSpotId++,
                    spot_type: 'face',
                    x: Math.round(c.cx),
                    y: Math.round(c.cy),
                    radius: Math.round(Math.max(c.w, c.h) / 2 * 1.2),
                    width: Math.round(c.w * 1.2),
                    height: Math.round(c.h * 1.2),
                    offset_x: 0,
                    offset_y: 0,
                    frame_start: c.firstFrame,
                    frame_end: c.lastFrame,
                    side: side,
                });
            }
        }
    }

    // ── Spot management ──
    function renderSpotList() {
        const list = document.getElementById('spotList');
        if (!list) return;

        if (blurSpots.length === 0) {
            list.innerHTML = '<div style="font-size:11px;color:var(--text-muted);">No blur spots yet. Click a detected face or use + Custom Spot.</div>';
            return;
        }

        list.innerHTML = blurSpots.map(s => `
            <div class="spot-item ${s.id === selectedSpotId ? 'selected' : ''}"
                 onclick="deid.selectSpot(${s.id})">
                <span class="spot-label">${s.spot_type === 'face' ? '\u{1F464}' : '\u2295'}
                    r=${s.radius} [${s.frame_start}-${s.frame_end}]</span>
                <span class="spot-delete" onclick="event.stopPropagation(); deid.deleteSpot(${s.id})">×</span>
            </div>
        `).join('');
    }

    function selectSpot(id) {
        selectedSpotId = id;
        renderSpotList();
        updateSpotControls();
        render();
    }

    function deleteSpot(id) {
        blurSpots = blurSpots.filter(s => s.id !== id);
        if (selectedSpotId === id) {
            selectedSpotId = null;
            updateSpotControls();
        }
        renderSpotList();
        scheduleSave();
        render();
        renderTimeline();
    }

    function updateSpotControls() {
        const controls = document.getElementById('spotControls');
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (!spot) {
            if (controls) controls.style.display = 'none';
            return;
        }
        if (controls) controls.style.display = 'block';
        const w = spot.width || spot.radius * 2;
        const h = spot.height || spot.radius * 2;
        document.getElementById('widthSlider').value = w;
        document.getElementById('widthVal').textContent = Math.round(w);
        document.getElementById('heightSlider').value = h;
        document.getElementById('heightVal').textContent = Math.round(h);
        document.getElementById('offsetXSlider').value = spot.offset_x || 0;
        document.getElementById('offsetXVal').textContent = Math.round(spot.offset_x || 0);
        document.getElementById('offsetYSlider').value = spot.offset_y || 0;
        document.getElementById('offsetYVal').textContent = Math.round(spot.offset_y || 0);
        document.getElementById('frameStartInput').value = spot.frame_start;
        document.getElementById('frameEndInput').value = spot.frame_end;
    }

    function updateSpotDim(dim, val) {
        val = parseInt(val);
        const valEl = document.getElementById(
            dim === 'width' ? 'widthVal' :
            dim === 'height' ? 'heightVal' :
            dim === 'offset_x' ? 'offsetXVal' : 'offsetYVal'
        );
        if (valEl) valEl.textContent = val;
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (spot) {
            spot[dim] = val;
            // Keep radius in sync as max(width, height) / 2 for backward compat
            if (dim === 'width' || dim === 'height') {
                spot.radius = Math.round(Math.max(spot.width || 0, spot.height || 0) / 2);
            }
            scheduleSave();
            render();
        }
    }

    function updateSpotFrameRange() {
        const spot = blurSpots.find(s => s.id === selectedSpotId);
        if (!spot) return;
        spot.frame_start = parseInt(document.getElementById('frameStartInput').value) || 0;
        spot.frame_end = parseInt(document.getElementById('frameEndInput').value) || 0;
        scheduleSave();
        render();
        renderTimeline();
    }

    // ── Custom spot toggle ──
    function toggleAddCustom() {
        addingCustom = !addingCustom;
        const btn = document.getElementById('addCustomBtn');
        btn.style.background = addingCustom ? 'var(--blue)' : '';
        btn.style.color = addingCustom ? '#fff' : '';
        render();
    }

    // ── Hand overlay ──
    async function toggleHandOverlay(enabled) {
        handOverlayEnabled = enabled;
        if (enabled && subjectId && hasMediapipe) {
            await _loadBulkLandmarks();
            _applyTemporalSmoothing();
        } else {
            handLandmarks = [];
        }
        render();
    }

    async function _loadBulkLandmarks() {
        if (Object.keys(handLandmarksBulk).length > 0) return; // already loaded
        try {
            const res = await API.get(
                `/api/deidentify/${subjectId}/hand-landmarks-bulk?trial_idx=${currentTrialIdx}`
            );
            handLandmarksBulk = res.landmarks || {};
        } catch (e) {
            handLandmarksBulk = {};
        }
    }

    function _applyTemporalSmoothing() {
        // Average landmark positions across a window of frames
        const win = handTemporalSmooth;
        const cf = currentFrame;
        const sideLabel = _sideLabel();

        if (Object.keys(handLandmarksBulk).length === 0) {
            handLandmarks = [];
            return;
        }

        if (win === 0) {
            // No smoothing — use raw data for current frame (copy to avoid mutation)
            const raw = handLandmarksBulk[String(cf)];
            handLandmarks = raw ? raw.map(lm => ({ ...lm })) : [];
            return;
        }

        // Collect landmarks from nearby frames, grouped by side
        const frameKeys = [];
        for (let f = cf - win; f <= cf + win; f++) {
            const key = String(f);
            if (handLandmarksBulk[key]) frameKeys.push(key);
        }

        if (frameKeys.length === 0) {
            handLandmarks = [];
            return;
        }

        // Group by side, then average positions per keypoint index
        const sides = {};
        for (const key of frameKeys) {
            for (const lm of handLandmarksBulk[key]) {
                const s = lm.side || 'full';
                if (!sides[s]) sides[s] = [];
                sides[s].push(lm);
            }
        }

        // For each side, average the keypoints across frames
        // Landmarks come in groups of 21 (per hand), so we average by index
        const result = [];
        for (const [side, allPts] of Object.entries(sides)) {
            // Count landmarks per frame to know how many hands/keypoints
            const ptsPerFrame = {};
            for (const key of frameKeys) {
                const framePts = (handLandmarksBulk[key] || []).filter(
                    lm => (lm.side || 'full') === side
                );
                ptsPerFrame[key] = framePts;
            }

            // Find the frame with the most points as reference
            let refKey = frameKeys[0];
            let maxPts = 0;
            for (const key of frameKeys) {
                const n = (ptsPerFrame[key] || []).length;
                if (n > maxPts) { maxPts = n; refKey = key; }
            }
            if (maxPts === 0) continue;

            // Average each keypoint position
            const refPts = ptsPerFrame[refKey];
            for (let i = 0; i < refPts.length; i++) {
                let sumX = 0, sumY = 0, count = 0;
                for (const key of frameKeys) {
                    const pts = ptsPerFrame[key] || [];
                    if (i < pts.length) {
                        sumX += pts[i].x;
                        sumY += pts[i].y;
                        count++;
                    }
                }
                if (count > 0) {
                    result.push({
                        x: Math.round((sumX / count) * 10) / 10,
                        y: Math.round((sumY / count) * 10) / 10,
                        side: side,
                        type: refPts[i].type || 'hand',
                        joint: refPts[i].joint,
                    });
                }
            }
        }

        handLandmarks = result;
    }

    function updateHandTemporalSmooth(val) {
        handTemporalSmooth = parseInt(val);
        document.getElementById('handTemporalVal').textContent = handTemporalSmooth;
        _applyTemporalSmoothing();
        render();
    }

    function updateHandRadius(val) {
        handMaskRadius = parseInt(val);
        document.getElementById('handRadiusVal').textContent = handMaskRadius;
        // Update ALL segments if none selected, or just the selected one
        if (selectedHandSegId) {
            const seg = handProtectSegments.find(s => s.id === selectedHandSegId);
            if (seg) seg.radius = handMaskRadius;
        } else {
            // Apply to all segments
            for (const seg of handProtectSegments) {
                seg.radius = handMaskRadius;
            }
        }
        saveHandSettings();
        render();
        renderTimeline();
    }

    function updateHandSmooth(val) {
        handSmooth = parseInt(val);
        document.getElementById('handSmoothVal').textContent = handSmooth;
        // Update selected segment's smooth, or all if none selected
        if (selectedHandSegId) {
            const seg = handProtectSegments.find(s => s.id === selectedHandSegId);
            if (seg) seg.smooth = handSmooth;
        } else {
            for (const seg of handProtectSegments) {
                seg.smooth = handSmooth;
            }
        }
        saveHandSettings();
        render();
    }

    async function saveHandSettings() {
        if (!subjectId || currentTrialIdx < 0) return;
        try {
            await API.put(`/api/deidentify/${subjectId}/hand-settings`, {
                trial_idx: currentTrialIdx,
                enabled: true,
                mask_radius: handMaskRadius,
                segments: handProtectSegments.map(s => ({
                    start: s.start, end: s.end, radius: s.radius,
                    smooth: s.smooth != null ? s.smooth : handSmooth,
                })),
            });
        } catch (e) {}
        renderTimeline();
    }

    function deleteSelectedHandSeg() {
        if (!selectedHandSegId) return;
        handProtectSegments = handProtectSegments.filter(s => s.id !== selectedHandSegId);
        selectedHandSegId = null;
        saveHandSettings();
        render();
        renderTimeline();
    }

    // ── Save blur specs (debounced) ──
    function scheduleSave() {
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(saveSpecs, 500);
    }

    async function saveSpecs() {
        if (!subjectId || currentTrialIdx < 0) return;
        const specs = blurSpots.map(s => ({
            spot_type: s.spot_type,
            x: s.x,
            y: s.y,
            radius: s.radius,
            width: s.width || null,
            height: s.height || null,
            offset_x: s.offset_x || 0,
            offset_y: s.offset_y || 0,
            frame_start: s.frame_start,
            frame_end: s.frame_end,
        }));
        try {
            await API.put(`/api/deidentify/${subjectId}/blur-specs`, {
                trial_idx: currentTrialIdx,
                specs: specs,
            });
        } catch (e) {
            console.error('Save blur specs failed:', e);
        }
    }

    // ── Render all trials ──
    async function renderAll() {
        if (!subjectId) return;

        // Save current specs first
        await saveSpecs();

        const btn = document.getElementById('renderBtn');
        const status = document.getElementById('renderStatus');
        btn.disabled = true;
        status.textContent = 'Starting render...';

        try {
            const result = await API.post(`/api/deidentify/${subjectId}/render`);
            const jobId = result.job_id;

            API.streamJob(jobId,
                (data) => {
                    if (data.status === 'running') {
                        const pct = data.progress_pct ? Math.round(data.progress_pct) : 0;
                        status.textContent = `Rendering... ${pct}%`;
                    }
                },
                (data) => {
                    btn.disabled = false;
                    if (data.status === 'completed') {
                        status.textContent = 'Render complete! Deidentified videos saved.';
                    } else if (data.status === 'failed') {
                        status.textContent = 'Render failed: ' + (data.error_msg || 'unknown');
                    } else {
                        status.textContent = 'Render ' + data.status;
                    }
                },
            );
        } catch (e) {
            status.textContent = 'Error: ' + e.message;
            btn.disabled = false;
        }
    }

    // ── Timeline ──────────────────────────────────────────────────────────

    function setupTimeline() {
        if (!tlCanvas) return;
        const ro = new ResizeObserver(() => renderTimeline());
        ro.observe(tlCanvas.parentElement);

        tlCanvas.addEventListener('mousedown', onTlMouseDown);
        tlCanvas.addEventListener('mousemove', onTlMouseMove);
        tlCanvas.addEventListener('mouseup', onTlMouseUp);
        tlCanvas.addEventListener('mouseleave', onTlMouseUp);
        tlCanvas.addEventListener('click', onTlClick);
    }

    function _tlLayout() {
        // Returns layout metrics for timeline rendering
        if (!trialMeta || !tlCanvas) return null;
        const cw = tlCanvas.width;
        const ch = tlCanvas.height;
        const margin = 4;
        const labelW = 50; // left label area
        const barX = labelW;
        const barW = cw - labelW - margin;
        const start = trialMeta.start_frame;
        const end = trialMeta.end_frame;
        const range = end - start || 1;
        return { cw, ch, margin, labelW, barX, barW, start, end, range };
    }

    function _frameToTlX(frame, layout) {
        return layout.barX + ((frame - layout.start) / layout.range) * layout.barW;
    }

    function _tlXToFrame(x, layout) {
        return Math.round(layout.start + ((x - layout.barX) / layout.barW) * layout.range);
    }

    function renderTimeline() {
        if (!tlCtx || !tlCanvas || !trialMeta) return;

        const container = tlCanvas.parentElement;
        tlCanvas.width = container.clientWidth;
        tlCanvas.height = container.clientHeight;
        const L = _tlLayout();
        if (!L) return;

        tlCtx.clearRect(0, 0, L.cw, L.ch);

        const sideLabel = _sideLabel();
        const visibleSpots = blurSpots.filter(s => {
            const ss = s.side || 'full';
            return ss === 'full' || ss === sideLabel;
        });

        const faceRowH = 35;
        const handRowH = 25;
        const gap = 3;
        let y = 2;

        // ── Face row: dark detection coverage + bright blue blur bars ──
        tlCtx.fillStyle = 'rgba(150,150,150,0.5)';
        tlCtx.font = '10px sans-serif';
        tlCtx.fillText('Faces', 2, y + 12);

        // Dark background showing where faces were detected
        if (faceDetections.length > 0) {
            tlCtx.fillStyle = 'rgba(33,150,243,0.15)';
            const pw = Math.max(1, L.barW / L.range);
            for (let i = 0; i < faceDetections.length; i++) {
                const entry = faceDetections[i];
                if (!entry || !entry.faces || entry.faces.length === 0) continue;
                const frame = entry.frame != null ? entry.frame : (trialMeta.start_frame + i);
                const x = _frameToTlX(frame, L);
                tlCtx.fillRect(x, y, pw, faceRowH);
            }
        }

        // Bright blue blur spot bars on top (matching face detection box color)
        if (visibleSpots.length > 0) {
            const spotBarH = Math.min(14, (faceRowH - 4) / visibleSpots.length);
            let spotY = y + 2;
            for (const spot of visibleSpots) {
                const x1 = _frameToTlX(spot.frame_start, L);
                const x2 = _frameToTlX(spot.frame_end, L);
                const w = Math.max(3, x2 - x1);
                const isSelected = spot.id === selectedSpotId;

                tlCtx.fillStyle = isSelected
                    ? 'rgba(33,150,243,0.6)'
                    : 'rgba(33,150,243,0.35)';
                tlCtx.fillRect(x1, spotY, w, spotBarH - 1);

                tlCtx.strokeStyle = isSelected
                    ? 'rgba(33,150,243,1.0)'
                    : 'rgba(33,150,243,0.7)';
                tlCtx.lineWidth = isSelected ? 1.5 : 0.5;
                tlCtx.strokeRect(x1, spotY, w, spotBarH - 1);

                // Edge handles for selected
                if (isSelected) {
                    tlCtx.fillStyle = 'rgba(33,150,243,1.0)';
                    tlCtx.fillRect(x1 - 1, spotY, 3, spotBarH - 1);
                    tlCtx.fillRect(x2 - 2, spotY, 3, spotBarH - 1);
                }

                spotY += spotBarH;
            }
        }

        y += faceRowH + gap;

        // ── Hand row: green coverage + multiple draggable protection segments ──
        if (handCoverage.length > 0 || hasMediapipe) {
            tlCtx.fillStyle = 'rgba(150,150,150,0.5)';
            tlCtx.fillText('Hands', 2, y + 12);

            // Background coverage heatmap
            if (handCoverage.length > 0) {
                tlCtx.fillStyle = 'rgba(76,175,80,0.15)';
                const pw = Math.max(1, L.barW / L.range);
                for (const f of handCoverage) {
                    const x = _frameToTlX(f, L);
                    tlCtx.fillRect(x, y, pw, handRowH);
                }
            }

            // Protection segment bars (multiple, draggable)
            for (const seg of handProtectSegments) {
                const hx1 = _frameToTlX(seg.start, L);
                const hx2 = _frameToTlX(seg.end, L);
                const hw = Math.max(3, hx2 - hx1);
                const isSel = seg.id === selectedHandSegId;

                tlCtx.fillStyle = isSel ? 'rgba(76,175,80,0.55)' : 'rgba(76,175,80,0.35)';
                tlCtx.fillRect(hx1, y + 2, hw, handRowH - 4);
                tlCtx.strokeStyle = isSel ? 'rgba(76,175,80,1.0)' : 'rgba(76,175,80,0.8)';
                tlCtx.lineWidth = isSel ? 2 : 1;
                tlCtx.strokeRect(hx1, y + 2, hw, handRowH - 4);

                if (isSel) {
                    tlCtx.fillStyle = 'rgba(76,175,80,1.0)';
                    tlCtx.fillRect(hx1 - 1, y + 2, 3, handRowH - 4);
                    tlCtx.fillRect(hx2 - 2, y + 2, 3, handRowH - 4);
                }
            }

            y += handRowH + gap;
        }

        // ── Current frame indicator ──
        const cfx = _frameToTlX(currentFrame, L);
        tlCtx.strokeStyle = '#fff';
        tlCtx.lineWidth = 1;
        tlCtx.beginPath();
        tlCtx.moveTo(cfx, 0);
        tlCtx.lineTo(cfx, L.ch);
        tlCtx.stroke();
    }

    // ── Timeline mouse handlers ──

    // Convert mouse event to canvas-space coordinates (handles CSS vs buffer size mismatch)
    function _tlMouseToCanvas(e) {
        const rect = tlCanvas.getBoundingClientRect();
        const scaleX = tlCanvas.width / rect.width;
        const scaleY = tlCanvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function _tlHitTest(e) {
        // Returns { type: 'spot'|'hand'|'hand_empty', spot?, seg?, edge } or null
        const { x: mx, y: my } = _tlMouseToCanvas(e);
        const L = _tlLayout();
        if (!L) return null;

        const sideLabel = _sideLabel();
        const visibleSpots = blurSpots.filter(s => {
            const ss = s.side || 'full';
            return ss === 'full' || ss === sideLabel;
        });

        const faceRowH = 35, handRowH = 25, gap = 3;
        let y = 2;

        // Hit test blur spot bars within face row
        if (visibleSpots.length > 0) {
            const spotBarH = Math.min(12, (faceRowH - 4) / visibleSpots.length);
            let spotY = y + 2;
            for (const spot of visibleSpots) {
                const x1 = _frameToTlX(spot.frame_start, L);
                const x2 = _frameToTlX(spot.frame_end, L);

                if (my >= spotY && my <= spotY + spotBarH) {
                    if (Math.abs(mx - x1) < 6) return { type: 'spot', spot, edge: 'start' };
                    if (Math.abs(mx - x2) < 6) return { type: 'spot', spot, edge: 'end' };
                    if (mx > x1 + 4 && mx < x2 - 4) return { type: 'spot', spot, edge: 'move' };
                }
                spotY += spotBarH;
            }
        }

        y += faceRowH + gap;

        // Hit test hand protection segments
        if (handCoverage.length > 0 || hasMediapipe) {
            for (const seg of handProtectSegments) {
                const hx1 = _frameToTlX(seg.start, L);
                const hx2 = _frameToTlX(seg.end, L);

                if (my >= y + 2 && my <= y + handRowH - 2) {
                    if (Math.abs(mx - hx1) < 6) return { type: 'hand', seg, edge: 'start' };
                    if (Math.abs(mx - hx2) < 6) return { type: 'hand', seg, edge: 'end' };
                    if (mx > hx1 + 4 && mx < hx2 - 4) return { type: 'hand', seg, edge: 'move' };
                }
            }
            // If in hand row but not on a segment → create new
            if (my >= y + 2 && my <= y + handRowH - 2) {
                return { type: 'hand_empty', edge: 'create' };
            }
        }

        return null;
    }

    function onTlMouseDown(e) {
        const hit = _tlHitTest(e);
        if (!hit) return;
        e.preventDefault();

        if (hit.type === 'spot') {
            tlDragSpot = hit.spot;
            tlDragEdge = hit.edge;
            tlDragStartX = e.clientX;
            tlDragOrigRange = { start: hit.spot.frame_start, end: hit.spot.frame_end };
            selectedSpotId = hit.spot.id;
            renderSpotList();
            updateSpotControls();
        } else if (hit.type === 'hand') {
            tlDragSpot = hit.seg;
            tlDragEdge = hit.edge;
            tlDragStartX = e.clientX;
            tlDragOrigRange = { start: hit.seg.start, end: hit.seg.end };
            selectedHandSegId = hit.seg.id;
            // Update sliders to show this segment's values
            handMaskRadius = hit.seg.radius || handMaskRadius;
            document.getElementById('handRadiusSlider').value = handMaskRadius;
            document.getElementById('handRadiusVal').textContent = handMaskRadius;
            handSmooth = hit.seg.smooth != null ? hit.seg.smooth : handSmooth;
            document.getElementById('handSmoothSlider').value = handSmooth;
            document.getElementById('handSmoothVal').textContent = handSmooth;
        } else if (hit.type === 'hand_empty') {
            // Start creating a new segment by dragging
            const L = _tlLayout();
            if (!L) return;
            const { x: mx } = _tlMouseToCanvas(e);
            tlDragCreateFrame = _tlXToFrame(mx, L);
            tlDragSpot = 'newhand';
            tlDragEdge = 'create';
            tlDragStartX = e.clientX;
        }
        renderTimeline();
    }

    function onTlMouseMove(e) {
        if (tlDragSpot && tlDragEdge) {
            const L = _tlLayout();
            if (!L) return;
            const dx = e.clientX - tlDragStartX;
            const dFrames = Math.round((dx / L.barW) * L.range);

            // Creating new hand segment by dragging
            if (tlDragSpot === 'newhand') {
                const L2 = _tlLayout();
                if (!L2) return;
                const { x: mx } = _tlMouseToCanvas(e);
                const endFrame = _tlXToFrame(mx, L2);
                // Show preview by temporarily adding a segment
                const existing = handProtectSegments.find(s => s.id === -1);
                const s = Math.min(tlDragCreateFrame, endFrame);
                const en = Math.max(tlDragCreateFrame, endFrame);
                if (existing) {
                    existing.start = s; existing.end = en;
                } else {
                    handProtectSegments.push({ id: -1, start: s, end: en, radius: handMaskRadius, smooth: handSmooth });
                }
                renderTimeline();
                return;
            }

            // Drag existing hand segment
            if (tlDragSpot && typeof tlDragSpot === 'object' && tlDragSpot.start !== undefined) {
                if (tlDragEdge === 'start') {
                    tlDragSpot.start = Math.max(
                        trialMeta.start_frame,
                        Math.min(tlDragSpot.end - 1, tlDragOrigRange.start + dFrames)
                    );
                } else if (tlDragEdge === 'end') {
                    tlDragSpot.end = Math.min(
                        trialMeta.end_frame,
                        Math.max(tlDragSpot.start + 1, tlDragOrigRange.end + dFrames)
                    );
                } else if (tlDragEdge === 'move') {
                    const len = tlDragOrigRange.end - tlDragOrigRange.start;
                    let newStart = tlDragOrigRange.start + dFrames;
                    newStart = Math.max(trialMeta.start_frame, Math.min(trialMeta.end_frame - len, newStart));
                    tlDragSpot.start = newStart;
                    tlDragSpot.end = newStart + len;
                }
                renderTimeline();
                return;
            }

            if (tlDragEdge === 'start') {
                tlDragSpot.frame_start = Math.max(
                    trialMeta.start_frame,
                    Math.min(tlDragSpot.frame_end - 1, tlDragOrigRange.start + dFrames)
                );
            } else if (tlDragEdge === 'end') {
                tlDragSpot.frame_end = Math.min(
                    trialMeta.end_frame,
                    Math.max(tlDragSpot.frame_start + 1, tlDragOrigRange.end + dFrames)
                );
            } else if (tlDragEdge === 'move') {
                const len = tlDragOrigRange.end - tlDragOrigRange.start;
                let newStart = tlDragOrigRange.start + dFrames;
                newStart = Math.max(trialMeta.start_frame, Math.min(trialMeta.end_frame - len, newStart));
                tlDragSpot.frame_start = newStart;
                tlDragSpot.frame_end = newStart + len;
            }

            updateSpotControls();
            render();
            renderTimeline();
            return;
        }

        // Hover cursor
        const hit = _tlHitTest(e);
        if (hit) {
            tlCanvas.style.cursor = hit.edge === 'move' ? 'grab' : 'ew-resize';
        } else {
            tlCanvas.style.cursor = 'pointer';
        }
    }

    function onTlMouseUp(e) {
        if (tlDragSpot) {
            if (tlDragSpot === 'newhand') {
                // Finalize the new segment (replace the preview id=-1)
                const preview = handProtectSegments.find(s => s.id === -1);
                if (preview && preview.end > preview.start) {
                    preview.id = nextHandSegId++;
                    selectedHandSegId = preview.id;
                } else {
                    // Too small, remove it
                    handProtectSegments = handProtectSegments.filter(s => s.id !== -1);
                }
                saveHandSettings();
                render();
            } else if (tlDragSpot && typeof tlDragSpot === 'object' && tlDragSpot.start !== undefined) {
                saveHandSettings();
            } else {
                scheduleSave();
            }
            tlDragSpot = null;
            tlDragEdge = null;
            tlDragStartX = null;
            tlDragOrigRange = null;
            tlDragCreateFrame = null;
        }
    }

    function onTlClick(e) {
        if (tlDragSpot) return; // was a drag, not click
        const { x: mx } = _tlMouseToCanvas(e);
        const L = _tlLayout();
        if (!L) return;

        // Always seek to clicked frame
        const frame = Math.max(L.start, Math.min(L.end, _tlXToFrame(mx, L)));
        seekFrame(frame);

        // Also select spot if clicking on one
        const hit = _tlHitTest(e);
        if (hit && hit.type === 'spot') {
            selectedSpotId = hit.spot.id;
            renderSpotList();
            updateSpotControls();
            renderTimeline();
        }
    }

    // ── Public API ──
    document.addEventListener('DOMContentLoaded', init);

    return {
        detectFaces,
        togglePlay,
        toggleSide,
        resetZoom,
        seekFrame,
        toggleAddCustom,
        selectSpot,
        deleteSpot,
        updateSpotDim,
        updateSpotFrameRange,
        toggleHandOverlay,
        updateHandRadius,
        deleteSelectedHandSeg,
        updateHandSmooth,
        updateHandTemporalSmooth,
        goToMediapipe,
        renderAll,
    };
})();
