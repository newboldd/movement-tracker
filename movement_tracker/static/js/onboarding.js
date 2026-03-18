/* Subject onboarding: file browser, video trimmer, segment management */

const onboard = (() => {
    let subjectName = '';
    let cameraMode = 'stereo';
    let selectedCameraSetup = null;
    let cameraNames = [];         // populated from setup in multicam mode
    let currentCameraIdx = 0;     // which camera we're selecting a video for
    let currentPath = '';
    let selectedVideoPath = null;
    let videoMeta = null;
    let inPoint = null;
    let outPoint = null;
    const segments = []; // {source_path, start_time, end_time, trial_label, source_name}

    // ── Step 1: Subject name ──────────────────────────────

    async function confirmName() {
        const name = document.getElementById('subjectName').value.trim();
        if (!name) return alert('Enter a subject name');
        if (!/^[A-Za-z0-9_]+$/.test(name)) return alert('Name must be alphanumeric (letters, numbers, underscores)');

        subjectName = name;
        document.getElementById('step1Num').classList.add('done');
        document.getElementById('subjectName').disabled = true;

        // Check camera mode to decide whether to show camera setup step
        try {
            const settings = await API.get('/api/settings');
            cameraMode = settings.camera_mode || 'stereo';
        } catch (e) { /* default to stereo */ }

        if (cameraMode === 'single') {
            // Skip camera setup step — go straight to file browser
            _updateStepNumbers(false);
            document.getElementById('step2').style.display = 'block';
            await loadDirectory('');
            await prefillFromSession();
        } else {
            // Show camera setup selection
            _updateStepNumbers(true);
            await _loadCameraSetups();
            document.getElementById('step1b').style.display = 'block';
        }
    }

    // ── Step 1b: Camera setup selection ───────────────────

    async function _loadCameraSetups() {
        const select = document.getElementById('cameraSetupSelect');
        try {
            const setups = await API.get('/api/camera-setups');
            select.innerHTML = '<option value="">-- Select --</option>';
            for (const s of setups) {
                const calib = s.has_calibration ? ' (calibrated)' : '';
                select.innerHTML += `<option value="${s.id}" data-name="${s.name}">${s.name} — ${s.mode}, ${s.camera_count} cams${calib}</option>`;
            }
            if (setups.length === 0) {
                select.innerHTML = '<option value="">No setups — create one first</option>';
            }
        } catch (e) {
            select.innerHTML = '<option value="">Error loading setups</option>';
        }
    }

    async function confirmCameraSetup() {
        const select = document.getElementById('cameraSetupSelect');
        const setupId = select.value;
        if (!setupId) return alert('Select a camera setup or create a new one');

        // Fetch full setup details to get camera names
        try {
            const setup = await API.get(`/api/camera-setups/${setupId}`);
            selectedCameraSetup = setup;
            cameraNames = setup.camera_names || [];
        } catch (e) {
            const opt = select.options[select.selectedIndex];
            selectedCameraSetup = { id: parseInt(setupId), name: opt.dataset.name || opt.textContent };
            cameraNames = [];
        }

        currentCameraIdx = 0;
        document.getElementById('step1bNum').classList.add('done');
        document.getElementById('step2').style.display = 'block';

        // In multicam mode, show which camera we're selecting for
        _updateCameraLabel();

        await loadDirectory('');
        await prefillFromSession();
    }

    function _updateCameraLabel() {
        const label = document.getElementById('cameraLabel');
        if (!label) return;
        if (cameraMode === 'multicam' && cameraNames.length > 1) {
            label.textContent = `Selecting video for camera: ${cameraNames[currentCameraIdx]} (${currentCameraIdx + 1}/${cameraNames.length})`;
            label.style.display = 'block';
        } else {
            label.style.display = 'none';
        }
    }

    function _updateStepNumbers(showCameraStep) {
        // Adjust step numbers based on whether camera setup step is shown
        if (showCameraStep) {
            // Steps: 1=Name, 2=Camera, 3=Video, 4=Trim, 5=Process
            document.getElementById('step1bNum').textContent = '2';
            document.getElementById('step2Num').textContent = '3';
            document.getElementById('step3Num').textContent = '4';
            document.getElementById('step4Num').textContent = '5';
        } else {
            // Steps: 1=Name, 2=Video, 3=Trim, 4=Process
            document.getElementById('step2Num').textContent = '2';
            document.getElementById('step3Num').textContent = '3';
            document.getElementById('step4Num').textContent = '4';
        }
    }

    // ── Step 2: File browser ──────────────────────────────

    async function loadDirectory(path) {
        const browser = document.getElementById('fileBrowser');
        browser.innerHTML = '<div style="padding:12px;color:var(--text-muted)">Loading...</div>';

        try {
            const data = await API.get(`/api/files?path=${encodeURIComponent(path)}`);
            currentPath = data.path || '';
            selectedVideoPath = null;
            document.getElementById('selectVideoBtn').disabled = true;

            let html = '';

            // Show locations if at root
            if (!currentPath && data.locations) {
                html += '<div class="fb-locations">';
                for (const loc of data.locations) {
                    html += `<div class="fb-loc" onclick="onboard.browse('${escPath(loc.path)}')">
                        <span class="icon">&#128193;</span>
                        <span>${loc.name}</span>
                        <span style="color:var(--text-muted);font-size:11px;margin-left:auto;">${loc.path}</span>
                    </div>`;
                }
                html += '</div>';
            }

            // Breadcrumbs
            if (data.breadcrumbs && data.breadcrumbs.length > 0) {
                html += '<div class="fb-breadcrumbs">';
                html += `<a onclick="onboard.browse('')">Home</a> /`;
                for (const crumb of data.breadcrumbs) {
                    html += ` <a onclick="onboard.browse('${escPath(crumb.path)}')">${crumb.name}</a> /`;
                }
                html += '</div>';
            }

            // Items
            if (data.parent) {
                html += `<div class="fb-item" onclick="onboard.browse('${escPath(data.parent)}')">
                    <span class="icon">&#8592;</span> <span>..</span>
                </div>`;
            }

            for (const item of data.items) {
                if (item.type === 'dir') {
                    html += `<div class="fb-item" onclick="onboard.browse('${escPath(item.path)}')">
                        <span class="icon">&#128193;</span>
                        <span>${item.name}</span>
                    </div>`;
                } else {
                    const dateStr = item.created ? new Date(item.created).toLocaleDateString(undefined, {year:'numeric',month:'short',day:'numeric'}) : '';
                    html += `<div class="fb-item" data-path="${escPath(item.path)}" data-name="${item.name}" onclick="onboard.selectFile('${escPath(item.path)}', this)" ondblclick="onboard.dblClickFile('${escPath(item.path)}')">
                        <span class="icon">&#127909;</span>
                        <span>${item.name}</span>
                        <span style="color:var(--text-muted);font-size:11px;margin-left:auto;">${dateStr}</span>
                        <span class="size" style="min-width:60px;text-align:right;">${item.size_mb} MB</span>
                    </div>`;
                }
            }

            if (!data.items || data.items.length === 0) {
                if (currentPath) {
                    html += '<div style="padding:12px;color:var(--text-muted)">No video files found</div>';
                }
            }

            browser.innerHTML = html;
        } catch (e) {
            browser.innerHTML = `<div style="padding:12px;color:var(--red)">${e.message}</div>`;
        }
    }

    function browse(path) {
        loadDirectory(path);
    }

    function selectFile(path, el) {
        // Deselect all
        document.querySelectorAll('.fb-item.selected').forEach(e => e.classList.remove('selected'));
        el.classList.add('selected');
        selectedVideoPath = path;
        document.getElementById('selectVideoBtn').disabled = false;
    }

    async function selectVideo() {
        if (!selectedVideoPath) return;

        // Probe video
        try {
            videoMeta = await API.get(`/api/video-tools/probe?path=${encodeURIComponent(selectedVideoPath)}`);
        } catch (e) {
            alert('Error probing video: ' + e.message);
            return;
        }

        document.getElementById('step2Num').classList.add('done');
        document.getElementById('step3').style.display = 'block';

        // Show video name and metadata
        const name = selectedVideoPath.split(/[/\\]/).pop();
        document.getElementById('selectedVideoName').textContent = name;

        const meta = document.getElementById('videoMeta');
        meta.innerHTML = `
            <span class="meta-badge">${videoMeta.width}x${videoMeta.height}</span>
            <span class="meta-badge">${videoMeta.fps} fps</span>
            <span class="meta-badge">${videoMeta.duration.toFixed(1)}s</span>
            <span class="meta-badge">${videoMeta.is_stereo ? 'Stereo' : 'Single camera'}</span>
            <span class="meta-badge">${videoMeta.codec}</span>
        `;

        // Load video for trimming
        const video = document.getElementById('trimVideo');
        video.src = `/api/video-tools/stream?path=${encodeURIComponent(selectedVideoPath)}`;

        // Reset trim points
        inPoint = null;
        outPoint = null;
        updateTrimDisplay();

        // Auto-advance trial label
        autoAdvanceTrialLabel();
    }

    // ── Step 3: Trimmer ───────────────────────────────────

    function setInPoint() {
        const video = document.getElementById('trimVideo');
        inPoint = video.currentTime;
        updateTrimDisplay();
    }

    function setOutPoint() {
        const video = document.getElementById('trimVideo');
        outPoint = video.currentTime;
        updateTrimDisplay();
    }

    function updateTrimDisplay() {
        document.getElementById('inTime').textContent =
            inPoint !== null ? `In: ${formatTime(inPoint)}` : 'In: --';
        document.getElementById('outTime').textContent =
            outPoint !== null ? `Out: ${formatTime(outPoint)}` : 'Out: --';
    }

    function addSegment() {
        if (inPoint === null || outPoint === null) {
            return alert('Set both in-point and out-point first');
        }
        if (outPoint <= inPoint) {
            return alert('Out-point must be after in-point');
        }

        const trial = document.getElementById('trialLabel').value.trim();
        if (!trial) return alert('Enter a trial name');
        const sourceName = selectedVideoPath.split(/[/\\]/).pop();

        const seg = {
            source_path: selectedVideoPath,
            start_time: inPoint,
            end_time: outPoint,
            trial_label: trial,
            source_name: sourceName,
        };

        // In multicam mode, tag with camera name
        if (cameraMode === 'multicam' && cameraNames.length > 1) {
            seg.camera_name = cameraNames[currentCameraIdx];
        }

        segments.push(seg);

        inPoint = null;
        outPoint = null;
        updateTrimDisplay();
        renderSegments();
        autoAdvanceTrialLabel();
        updateStep4Visibility();
    }

    function removeSegment(idx) {
        segments.splice(idx, 1);
        renderSegments();
        updateStep4Visibility();
    }

    function renderSegments() {
        const list = document.getElementById('segmentList');
        if (segments.length === 0) {
            list.innerHTML = '';
            return;
        }

        list.innerHTML = segments.map((s, i) => {
            const camTag = s.camera_name
                ? `<span style="color:var(--blue);font-size:11px;font-weight:600;">[${s.camera_name}]</span>`
                : '';
            return `
                <div class="segment-item">
                    <span class="trial">${s.trial_label} ${camTag}</span>
                    <span class="times">${formatTime(s.start_time)} - ${formatTime(s.end_time)}
                        (${(s.end_time - s.start_time).toFixed(1)}s)</span>
                    <span class="source">${s.source_name}</span>
                    <span class="remove" onclick="onboard.removeSegment(${i})">&times;</span>
                </div>
            `;
        }).join('');
    }

    function pickAnotherVideo() {
        // Go back to file browser to pick a different source video
        document.getElementById('step3').style.display = 'none';
        document.getElementById('step2Num').classList.remove('done');
        selectedVideoPath = null;
        videoMeta = null;
    }

    function autoAdvanceTrialLabel() {
        const input = document.getElementById('trialLabel');
        const used = new Set(segments.map(s => s.trial_label));
        // Suggest the next unused datalist option, or clear for free typing
        const suggestions = Array.from(
            document.querySelectorAll('#trialLabelSuggestions option')
        ).map(o => o.value);
        const next = suggestions.find(v => !used.has(v));
        input.value = next || '';
    }

    function updateStep4Visibility() {
        const step4 = document.getElementById('step4');
        if (segments.length > 0) {
            step4.style.display = 'block';
            document.getElementById('step3Num').classList.add('done');

            // Render review
            const review = document.getElementById('reviewSegments');
            review.innerHTML = `
                <p style="margin-bottom:8px;">Subject: <strong>${subjectName}</strong></p>
                <p style="margin-bottom:8px;">${segments.length} segment(s) will be trimmed and saved.</p>
            ` + segments.map(s => {
                const outName = s.camera_name
                    ? `${subjectName}_${s.trial_label}_${s.camera_name}.mp4`
                    : `${subjectName}_${s.trial_label}.mp4`;
                const camTag = s.camera_name
                    ? ` <span style="color:var(--blue);font-size:11px;">[${s.camera_name}]</span>`
                    : '';
                return `
                    <div class="segment-item">
                        <span class="trial">${s.trial_label}${camTag}</span>
                        <span class="times">${formatTime(s.start_time)} - ${formatTime(s.end_time)}</span>
                        <span class="source">${s.source_name}</span>
                        <span style="color:var(--text-muted);font-size:11px;">&#8594; ${outName}</span>
                    </div>
                `;
            }).join('');

            updateProcessButton();
        } else {
            step4.style.display = 'none';
            document.getElementById('step3Num').classList.remove('done');
        }
    }

    function updateProcessButton() {
        const blur = document.getElementById('blurFaces');
        const btn = document.getElementById('processBtn');
        if (blur && btn) {
            btn.textContent = blur.checked
                ? 'Trim, Blur Faces & Create Subject'
                : 'Trim & Create Subject';
        }
    }

    // ── Step 4: Process ───────────────────────────────────

    async function startProcessing() {
        if (segments.length === 0) return;

        const statusEl = document.getElementById('processingStatus');
        const fillEl = document.getElementById('processingFill');
        const msgEl = document.getElementById('processingMsg');
        statusEl.style.display = 'block';
        msgEl.textContent = 'Starting...';

        try {
            const blurFaces = document.getElementById('blurFaces').checked;
            const result = await API.post('/api/video-tools/process-subject', {
                subject_name: subjectName,
                blur_faces: blurFaces,
                segments: segments.map(s => {
                    const seg = {
                        source_path: s.source_path,
                        start_time: s.start_time,
                        end_time: s.end_time,
                        trial_label: s.trial_label,
                    };
                    if (s.camera_name) seg.camera_name = s.camera_name;
                    return seg;
                }),
            });

            if (result.job_id) {
                // Track job progress
                API.streamJob(result.job_id,
                    (data) => {
                        const pct = data.progress_pct || 0;
                        fillEl.style.width = `${pct}%`;
                        msgEl.textContent = pct < 2 ? 'Initializing...' : `Processing: ${pct.toFixed(0)}%`;
                    },
                    (data) => {
                        if (data.status === 'completed') {
                            msgEl.textContent = 'Complete! Redirecting to dashboard...';
                            fillEl.style.width = '100%';
                            setTimeout(() => { window.location.href = '/'; }, 1500);
                        } else if (data.status === 'failed') {
                            msgEl.textContent = `Failed: ${data.error_msg || 'Unknown error'}`;
                            msgEl.style.color = 'var(--red)';
                        }
                    }
                );
            }
        } catch (e) {
            msgEl.textContent = `Error: ${e.message}`;
            msgEl.style.color = 'var(--red)';
        }
    }

    // ── Checkbox listener ─────────────────────────────────

    document.addEventListener('DOMContentLoaded', () => {
        const blur = document.getElementById('blurFaces');
        if (blur) blur.addEventListener('change', updateProcessButton);
    });

    // ── Double-click to select + advance ─────────────────
    function dblClickFile(path) {
        // Single-click already fired (selectFile called), so selectedVideoPath is set.
        selectVideo();
    }

    // ── Prefill from video viewer session ─────────────────
    async function prefillFromSession() {
        const videoName = sessionStorage.getItem('onboard_prefill_video');
        if (!videoName) return;
        sessionStorage.removeItem('onboard_prefill_video');

        try {
            const settings = await API.get('/api/settings');
            if (!settings.video_path) return;

            // Navigate file browser to the configured video directory
            await loadDirectory(settings.video_path);

            // Find and auto-select the matching file by name
            const items = document.querySelectorAll('#fileBrowser .fb-item[data-name]');
            for (const el of items) {
                if (el.dataset.name === videoName) {
                    selectFile(el.dataset.path, el);
                    el.scrollIntoView({ block: 'nearest' });
                    break;
                }
            }
        } catch (e) { /* ignore — prefill is best-effort */ }
    }

    // ── Keyboard shortcuts ────────────────────────────────

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        if (e.key === 'i' || e.key === 'I') {
            e.preventDefault();
            setInPoint();
        } else if (e.key === 'o' || e.key === 'O') {
            e.preventDefault();
            setOutPoint();
        }
    });

    // ── Helpers ───────────────────────────────────────────

    function formatTime(seconds) {
        if (seconds == null) return '--';
        const m = Math.floor(seconds / 60);
        const s = (seconds % 60).toFixed(1);
        return `${m}:${s.padStart(4, '0')}`;
    }

    function escPath(path) {
        // Escape single quotes and backslashes for inline onclick
        return path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    }

    // ── Public API ────────────────────────────────────────

    return {
        confirmName,
        confirmCameraSetup,
        browse,
        selectFile,
        dblClickFile,
        selectVideo,
        setInPoint,
        setOutPoint,
        addSegment,
        removeSegment,
        pickAnotherVideo,
        startProcessing,
    };
})();
