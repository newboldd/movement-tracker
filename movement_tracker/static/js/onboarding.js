/* Subject onboarding: file browser, video trimmer, segment management
   New flow: 1) Select Video → 2) Preview + Name → 3) Camera Mode → 3b) Sync → 4) Trim → 5) Review */

const onboard = (() => {
    let subjectName = '';
    let subjectGroup = 'Control';   // diagnosis group
    let diagnosisGroups = ['Control', 'MSA', 'PD', 'PSP'];
    let cameraMode = 'stereo';      // single | stereo | multicam
    let selectedCameraSetup = null;  // {id, name, camera_names, ...}
    let cameraNames = [];            // from setup
    let currentPath = '';
    let selectedVideoPath = null;
    let videoMeta = null;
    let inPoint = null;
    let outPoint = null;
    const segments = [];

    // Multicam state
    const multicamVideos = {};     // cameraName → path
    const syncOffsets = {};        // cameraName → integer frame offset
    let syncFps = 30;
    let syncFrame = 0;
    let syncTotalFrames = 0;       // max frames across all cameras
    let syncPlaying = false;       // play/pause state
    let syncPlayTimer = null;      // requestAnimationFrame id
    let syncPlayRate = 1;          // playback speed multiplier
    let syncZoom = 1;              // zoom level for sync video panels
    let syncPanX = 0;              // horizontal pan offset (fraction)
    let syncPanY = 0;              // vertical pan offset (fraction)
    let syncDragging = false;      // true during pan drag
    let syncDragStart = null;      // {x, y} of mouse at drag start
    let syncPanStart = null;       // {x, y} of pan at drag start

    // Per-segment "has faces" state — true by default
    const segmentHasFaces = [];    // parallel to segments[]

    // ── Group dropdown helpers ────────────────────────────────

    function _populateGroupDropdown(sel, selectedValue) {
        sel.innerHTML = diagnosisGroups
            .map(g => `<option value="${g}">${g}</option>`)
            .join('') +
            '<option value="__new__">+ New Group…</option>';
        if (selectedValue && diagnosisGroups.includes(selectedValue)) {
            sel.value = selectedValue;
        }
    }

    async function _onGroupSelectChange(sel) {
        if (sel.value !== '__new__') {
            subjectGroup = sel.value;
            return;
        }
        const name = prompt('Enter new group name:');
        if (!name || !name.trim()) {
            sel.value = subjectGroup; // revert
            return;
        }
        const trimmed = name.trim();
        if (!diagnosisGroups.includes(trimmed)) {
            diagnosisGroups.push(trimmed);
            // Save to settings so it persists
            try {
                const cfg = await API.get('/api/settings');
                cfg.diagnosis_groups = diagnosisGroups;
                await API.put('/api/settings', cfg);
            } catch { /* best-effort */ }
        }
        _populateGroupDropdown(sel, trimmed);
        sel.value = trimmed;
        subjectGroup = trimmed;
    }

    // ── Step 1: File browser (opens immediately) ─────────────

    async function loadDirectory(path) {
        const browser = document.getElementById('fileBrowser');
        browser.innerHTML = '<div style="padding:12px;color:var(--text-muted)">Loading...</div>';

        try {
            const data = await API.get(`/api/files?path=${encodeURIComponent(path)}`);
            currentPath = data.path || '';
            selectedVideoPath = null;
            document.getElementById('selectVideoBtn').disabled = true;

            let html = '';

            if (!currentPath && data.locations) {
                html += '<div class="fb-locations">';
                for (const loc of data.locations) {
                    html += `<div class="fb-loc" onclick="onboard.browse('${escPath(loc.path)}')">
                        <span class="icon">&#128193;</span>
                        <span>${loc.name}</span>
                        <span style="color:var(--text-muted);font-size:11px;margin-left:auto;">${loc.path}</span>
                    </div>`;
                }
                // "Browse other…" option → derive home dir from existing locations
                let homePath = '/';
                if (data.locations.length > 0) {
                    const parts = data.locations[0].path.replace(/\\\\/g, '/').split('/');
                    // e.g. /Users/john/Desktop → /Users/john
                    homePath = parts.length >= 3 ? parts.slice(0, 3).join('/') : parts.join('/');
                }
                html += `<div class="fb-loc" onclick="onboard.browse('${escPath(homePath)}')" style="border-top:1px solid var(--border);margin-top:4px;padding-top:10px;">
                    <span class="icon">&#128269;</span>
                    <span>Browse other location&hellip;</span>
                    <span style="color:var(--text-muted);font-size:11px;margin-left:auto;">${homePath}</span>
                </div>`;
                html += '</div>';
            }

            if (data.breadcrumbs && data.breadcrumbs.length > 0) {
                html += '<div class="fb-breadcrumbs">';
                html += `<a onclick="onboard.browse('')">Home</a> /`;
                for (const crumb of data.breadcrumbs) {
                    html += ` <a onclick="onboard.browse('${escPath(crumb.path)}')">${crumb.name}</a> /`;
                }
                html += '</div>';
            }

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

    function browse(path) { loadDirectory(path); }

    function selectFile(path, el) {
        document.querySelectorAll('.fb-item.selected').forEach(e => e.classList.remove('selected'));
        el.classList.add('selected');
        selectedVideoPath = path;
        document.getElementById('selectVideoBtn').disabled = false;
    }

    function dblClickFile(path) { selectVideo(); }

    // ── Step 1 → 2: Select video and show preview ───────────

    async function selectVideo() {
        if (!selectedVideoPath) return;

        try {
            videoMeta = await API.get(`/api/video-tools/probe?path=${encodeURIComponent(selectedVideoPath)}`);
        } catch (e) {
            alert('Error probing video: ' + e.message);
            return;
        }

        document.getElementById('step1Num').classList.add('done');
        document.getElementById('step2').style.display = 'block';

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

        // Load preview video
        const video = document.getElementById('previewVideo');
        video.src = `/api/video-tools/stream?path=${encodeURIComponent(selectedVideoPath)}`;

        // Auto-detect camera mode from aspect ratio
        if (videoMeta.is_stereo) {
            _preselectCameraMode('stereo');
        }

        // Auto-infer group from subject name input as user types
        const nameInput = document.getElementById('subjectName');
        const groupSel = document.getElementById('subjectGroup');
        if (nameInput && groupSel) {
            nameInput.addEventListener('input', () => {
                const val = nameInput.value.trim().toLowerCase();
                for (const g of diagnosisGroups) {
                    if (val.startsWith(g.toLowerCase().slice(0, 3))) {
                        groupSel.value = g;
                        break;
                    }
                }
            });
        }

        // Scroll to step 2
        document.getElementById('step2').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Step 2: Confirm subject name ─────────────────────────

    async function confirmName() {
        const name = document.getElementById('subjectName').value.trim();
        if (!name) return alert('Enter a subject name');
        if (!/^[A-Za-z0-9_]+$/.test(name)) return alert('Name must be alphanumeric (letters, numbers, underscores)');

        subjectName = name;
        subjectGroup = document.getElementById('subjectGroup').value || 'Control';
        document.getElementById('step2Num').classList.add('done');
        document.getElementById('subjectName').disabled = true;
        document.getElementById('subjectGroup').disabled = true;

        // Show camera mode step
        document.getElementById('step3').style.display = 'block';
        await _loadCameraSetups();

        document.getElementById('step3').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Step 3: Camera mode & setup ──────────────────────────

    function _preselectCameraMode(mode) {
        cameraMode = mode;
        document.querySelectorAll('.camera-mode-option').forEach(el => {
            el.classList.toggle('selected', el.querySelector('input').value === mode);
            if (el.querySelector('input').value === mode) el.querySelector('input').checked = true;
        });
    }

    function selectCameraMode(mode) {
        cameraMode = mode;

        // Update UI selection
        document.querySelectorAll('.camera-mode-option').forEach(el => {
            el.classList.toggle('selected', el.querySelector('input').value === mode);
        });

        // Show/hide sub-panels
        document.getElementById('stereoPanel').style.display = (mode === 'stereo') ? 'block' : 'none';
        document.getElementById('multicamPanel').style.display = (mode === 'multicam') ? 'block' : 'none';
    }

    async function _loadCameraSetups() {
        const stereoSelect = document.getElementById('cameraSetupSelect');
        const multicamSelect = document.getElementById('multicamSetupSelect');

        try {
            const setups = await API.get('/api/camera-setups');

            // Stereo dropdown: existing calibrations + DB setups
            stereoSelect.innerHTML = '<option value="">-- Select --</option>';

            // Add preset calibrations from DEFAULT_CALIBRATIONS
            const settings = await API.get('/api/settings');
            const calibrations = settings.calibrations || {};
            for (const [camName, calibPath] of Object.entries(calibrations)) {
                // Check if already covered by a DB setup
                const covered = setups.some(s => s.name === camName);
                if (!covered) {
                    stereoSelect.innerHTML += `<option value="preset:${camName}" data-name="${camName}">
                        ${camName} (existing calibration)</option>`;
                }
            }

            for (const s of setups) {
                if (s.mode === 'stereo') {
                    const calib = s.has_calibration ? ' (calibrated)' : '';
                    stereoSelect.innerHTML += `<option value="db:${s.id}" data-name="${s.name}">
                        ${s.name} — ${s.camera_count} cams${calib}</option>`;
                }
            }

            // Multicam dropdown
            multicamSelect.innerHTML = '<option value="">-- Select --</option>';
            for (const s of setups) {
                if (s.mode === 'multicam') {
                    multicamSelect.innerHTML += `<option value="db:${s.id}" data-name="${s.name}">
                        ${s.name} — ${s.camera_count} cams</option>`;
                }
            }
        } catch (e) {
            stereoSelect.innerHTML = '<option value="">Error loading setups</option>';
            multicamSelect.innerHTML = '<option value="">Error loading setups</option>';
        }
    }

    function toggleNewSetupForm(mode) {
        if (mode === 'stereo') {
            const form = document.getElementById('newSetupForm');
            form.style.display = form.style.display === 'none' ? 'block' : 'none';
        } else {
            const form = document.getElementById('newMulticamSetupForm');
            form.style.display = form.style.display === 'none' ? 'block' : 'none';
        }
    }

    async function createCameraSetup(mode) {
        let nameInput, camNamesInput;
        if (mode === 'stereo') {
            nameInput = document.getElementById('newSetupName');
            camNamesInput = document.getElementById('newSetupCamNames');
        } else {
            nameInput = document.getElementById('newMulticamSetupName');
            camNamesInput = document.getElementById('newMulticamCamNames');
        }

        const name = nameInput.value.trim();
        if (!name) return alert('Enter a setup name');

        const camNames = camNamesInput.value.split(',').map(s => s.trim()).filter(Boolean);
        if (camNames.length < 2) return alert('Enter at least 2 camera names');

        try {
            const result = await API.post('/api/camera-setups', {
                name,
                mode: mode === 'multicam' ? 'multicam' : 'stereo',
                camera_count: camNames.length,
                camera_names: camNames,
            });

            // Reload dropdowns and select the new one
            await _loadCameraSetups();

            const select = mode === 'stereo'
                ? document.getElementById('cameraSetupSelect')
                : document.getElementById('multicamSetupSelect');

            // Select the newly created option
            for (const opt of select.options) {
                if (opt.value === `db:${result.id}`) {
                    opt.selected = true;
                    break;
                }
            }

            // Hide the form
            if (mode === 'stereo') {
                document.getElementById('newSetupForm').style.display = 'none';
            } else {
                document.getElementById('newMulticamSetupForm').style.display = 'none';
            }

            nameInput.value = '';
        } catch (e) {
            alert('Failed to create setup: ' + e.message);
        }
    }

    function _updateMulticamVideoList() {
        if (!selectedCameraSetup || cameraMode !== 'multicam') return;

        const container = document.getElementById('multicamVideoList');
        const items = document.getElementById('multicamCameraItems');
        container.style.display = 'block';

        // First camera uses the already-selected video
        const firstName = selectedVideoPath ? selectedVideoPath.split(/[/\\]/).pop() : 'Not selected';

        items.innerHTML = cameraNames.map((cam, i) => {
            const file = i === 0 ? firstName : (multicamVideos[cam] ? multicamVideos[cam].split(/[/\\]/).pop() : '<em>Not selected</em>');
            const btnHtml = i === 0
                ? ''
                : `<button class="btn btn-sm" onclick="onboard.pickMulticamVideo('${cam}')">Browse</button>`;
            return `
                <div class="multicam-video-item">
                    <span class="cam-name">${cam}</span>
                    <span class="cam-file">${file}</span>
                    ${btnHtml}
                </div>
            `;
        }).join('');
    }

    // Multicam: pick video for a specific camera
    let _pickingForCamera = null;
    let _pickingFileBrowserCB = null;

    function pickMulticamVideo(cameraName) {
        _pickingForCamera = cameraName;
        // Show a modal file browser — reuse the main file browser but in a picking state
        const step1 = document.getElementById('step1');
        step1.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Change the select button behavior temporarily
        const btn = document.getElementById('selectVideoBtn');
        btn.textContent = `Select for ${cameraName}`;
        btn.onclick = () => _confirmMulticamPick();
    }

    function _confirmMulticamPick() {
        if (!selectedVideoPath || !_pickingForCamera) return;

        multicamVideos[_pickingForCamera] = selectedVideoPath;
        _pickingForCamera = null;

        // Restore selectedVideoPath to the first camera's video (the original selection)
        if (cameraNames.length > 0 && multicamVideos[cameraNames[0]]) {
            selectedVideoPath = multicamVideos[cameraNames[0]];
        }

        // Restore button
        const btn = document.getElementById('selectVideoBtn');
        btn.textContent = 'Select Video';
        btn.onclick = () => selectVideo();

        // Update the multicam video list
        _updateMulticamVideoList();

        // Scroll back to step 3
        document.getElementById('step3').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    async function confirmCameraMode() {
        if (cameraMode === 'stereo') {
            const select = document.getElementById('cameraSetupSelect');
            const val = select.value;
            if (!val) return alert('Select a camera setup');

            if (val.startsWith('preset:')) {
                const presetName = val.replace('preset:', '');
                selectedCameraSetup = { name: presetName, preset: true };
                cameraNames = ['OS', 'OD']; // default for presets
            } else {
                const setupId = parseInt(val.replace('db:', ''));
                try {
                    const setup = await API.get(`/api/camera-setups/${setupId}`);
                    selectedCameraSetup = setup;
                    cameraNames = setup.camera_names || ['OS', 'OD'];
                } catch (e) {
                    return alert('Error loading setup: ' + e.message);
                }
            }

            document.getElementById('step3Num').classList.add('done');
            _showTrimStep();

        } else if (cameraMode === 'multicam') {
            const select = document.getElementById('multicamSetupSelect');
            const val = select.value;
            if (!val) return alert('Select a camera setup');

            const setupId = parseInt(val.replace('db:', ''));
            try {
                const setup = await API.get(`/api/camera-setups/${setupId}`);
                selectedCameraSetup = setup;
                cameraNames = setup.camera_names || [];
            } catch (e) {
                return alert('Error loading setup: ' + e.message);
            }

            // Assign first camera to already-selected video (only if not yet assigned)
            if (cameraNames.length > 0 && !multicamVideos[cameraNames[0]]) {
                multicamVideos[cameraNames[0]] = selectedVideoPath;
            }

            // Check if all videos are selected
            _updateMulticamVideoList();

            const allSelected = cameraNames.every(cam => multicamVideos[cam]);
            if (!allSelected) {
                return alert('Please select a video for each camera before continuing.');
            }

            document.getElementById('step3Num').classList.add('done');

            // Show sync step
            _initSyncModule();

        } else {
            // Single camera — skip setup, go to trim
            document.getElementById('step3Num').classList.add('done');
            _showTrimStep();
        }
    }

    function _showTrimStep() {
        document.getElementById('step4').style.display = 'block';

        // Load video for trimming (same video or first camera video)
        const video = document.getElementById('trimVideo');
        video.src = `/api/video-tools/stream?path=${encodeURIComponent(selectedVideoPath)}`;

        inPoint = null;
        outPoint = null;
        updateTrimDisplay();
        autoAdvanceTrialLabel();

        document.getElementById('step4').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Step 3b: Multicam Sync ───────────────────────────────

    function _initSyncModule() {
        document.getElementById('step3b').style.display = 'block';

        syncFps = videoMeta ? videoMeta.fps : 30;
        syncFrame = 0;
        syncTotalFrames = 0;
        syncPlaying = false;
        syncPlayRate = 1;
        syncZoom = 1;
        syncPanX = 0;
        syncPanY = 0;

        const container = document.getElementById('syncContainer');
        container.innerHTML = '';

        // Initialize offsets
        for (const cam of cameraNames) {
            syncOffsets[cam] = 0;
        }

        // Create video panels
        for (let i = 0; i < cameraNames.length; i++) {
            const cam = cameraNames[i];
            const videoPath = multicamVideos[cam];
            if (!videoPath) continue;

            const panel = document.createElement('div');
            panel.className = 'sync-video-panel';
            panel.innerHTML = `
                <div class="cam-label">${cam}</div>
                <div class="sync-video-wrap" id="syncWrap_${i}">
                    <video id="syncVideo_${i}" preload="auto" muted
                        src="/api/video-tools/stream?path=${encodeURIComponent(videoPath)}"></video>
                </div>
                <div class="sync-offset-controls">
                    <button onclick="onboard.syncStepSingle(${i}, -1)">-1</button>
                    <span class="sync-offset-display" id="syncOffset_${i}">offset: 0</span>
                    <button onclick="onboard.syncStepSingle(${i}, 1)">+1</button>
                </div>
            `;
            container.appendChild(panel);

            // Setup pan drag on the wrapper
            const wrap = panel.querySelector('.sync-video-wrap');
            wrap.addEventListener('mousedown', _syncPanStart);
            wrap.addEventListener('wheel', _syncOnWheel, { passive: false });
        }

        // Determine total frames once videos load metadata
        let loaded = 0;
        const total = cameraNames.length;
        for (let i = 0; i < cameraNames.length; i++) {
            const video = document.getElementById(`syncVideo_${i}`);
            if (!video) continue;
            video.addEventListener('loadedmetadata', () => {
                const frames = Math.round(video.duration * syncFps);
                if (frames > syncTotalFrames) syncTotalFrames = frames;
                loaded++;
                if (loaded >= total) _updateSyncTimeline();
            }, { once: true });
        }

        _updateSyncFrameDisplay();
        _updateSyncTimeline();
        _updateSyncZoomDisplay();
        document.getElementById('step3b').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function syncStepAll(delta) {
        syncFrame += delta;
        if (syncFrame < 0) syncFrame = 0;
        if (syncTotalFrames > 0 && syncFrame >= syncTotalFrames) syncFrame = syncTotalFrames - 1;

        _syncSeekAll();
        _updateSyncFrameDisplay();
        _updateSyncTimeline();
    }

    function _syncSeekAll() {
        for (let i = 0; i < cameraNames.length; i++) {
            const cam = cameraNames[i];
            const video = document.getElementById(`syncVideo_${i}`);
            if (video) {
                const targetFrame = syncFrame + syncOffsets[cam];
                video.currentTime = Math.max(0, targetFrame / syncFps);
            }
        }
    }

    function syncStepSingle(idx, delta) {
        const cam = cameraNames[idx];
        syncOffsets[cam] = (syncOffsets[cam] || 0) + delta;

        const video = document.getElementById(`syncVideo_${idx}`);
        if (video) {
            const targetFrame = syncFrame + syncOffsets[cam];
            video.currentTime = Math.max(0, targetFrame / syncFps);
        }

        document.getElementById(`syncOffset_${idx}`).textContent = `offset: ${syncOffsets[cam]}`;
    }

    // ── Sync play/pause ──────────────────────────────────────

    function syncTogglePlay() {
        syncPlaying = !syncPlaying;
        const btn = document.getElementById('syncPlayBtn');
        if (syncPlaying) {
            btn.textContent = '⏸';
            btn.title = 'Pause (Space)';
            _syncPlayTick();
        } else {
            btn.textContent = '▶';
            btn.title = 'Play (Space)';
            if (syncPlayTimer) { cancelAnimationFrame(syncPlayTimer); syncPlayTimer = null; }
        }
    }

    function _syncPlayTick() {
        if (!syncPlaying) return;

        syncFrame += 1;
        if (syncTotalFrames > 0 && syncFrame >= syncTotalFrames) {
            syncFrame = 0; // loop
        }
        _syncSeekAll();
        _updateSyncFrameDisplay();
        _updateSyncTimeline();

        // Schedule next tick based on playback rate
        const delay = 1000 / (syncFps * syncPlayRate);
        syncPlayTimer = setTimeout(() => {
            if (syncPlaying) requestAnimationFrame(_syncPlayTick);
        }, delay);
    }

    function syncSetRate(rate) {
        syncPlayRate = rate;
        const label = document.getElementById('syncRateLabel');
        if (label) label.textContent = rate + 'x';
    }

    // ── Sync timeline scrubber ───────────────────────────────

    function _updateSyncTimeline() {
        const bar = document.getElementById('syncTimelineBar');
        const thumb = document.getElementById('syncTimelineThumb');
        if (!bar || !thumb) return;

        if (syncTotalFrames <= 0) {
            thumb.style.left = '0%';
            return;
        }
        const pct = (syncFrame / (syncTotalFrames - 1)) * 100;
        thumb.style.left = `${Math.min(100, Math.max(0, pct))}%`;
    }

    function syncTimelineSeek(e) {
        const bar = document.getElementById('syncTimelineBar');
        if (!bar || syncTotalFrames <= 0) return;

        const rect = bar.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        syncFrame = Math.round(pct * (syncTotalFrames - 1));

        _syncSeekAll();
        _updateSyncFrameDisplay();
        _updateSyncTimeline();
    }

    function _syncTimelineDragStart(e) {
        e.preventDefault();
        syncTimelineSeek(e);

        const onMove = (ev) => syncTimelineSeek(ev);
        const onUp = () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    }

    function _updateSyncFrameDisplay() {
        const el = document.getElementById('syncFrameDisplay');
        if (el) {
            const time = (syncFrame / syncFps).toFixed(1);
            el.textContent = `Frame ${syncFrame}` + (syncTotalFrames ? ` / ${syncTotalFrames}` : '') + ` (${time}s)`;
        }
    }

    // ── Sync zoom & pan ──────────────────────────────────────

    function syncZoomIn() {
        syncZoom = Math.min(8, syncZoom * 1.3);
        _applySyncZoom();
    }

    function syncZoomOut() {
        syncZoom = Math.max(1, syncZoom / 1.3);
        if (syncZoom < 1.05) { syncZoom = 1; syncPanX = 0; syncPanY = 0; }
        _applySyncZoom();
    }

    function syncZoomReset() {
        syncZoom = 1;
        syncPanX = 0;
        syncPanY = 0;
        _applySyncZoom();
    }

    function _applySyncZoom() {
        for (let i = 0; i < cameraNames.length; i++) {
            const video = document.getElementById(`syncVideo_${i}`);
            if (video) {
                video.style.transform = `scale(${syncZoom}) translate(${syncPanX * 100}%, ${syncPanY * 100}%)`;
                video.style.transformOrigin = 'center center';
            }
        }
        _updateSyncZoomDisplay();
    }

    function _updateSyncZoomDisplay() {
        const el = document.getElementById('syncZoomLabel');
        if (el) el.textContent = `${Math.round(syncZoom * 100)}%`;
    }

    function _syncPanStart(e) {
        if (syncZoom <= 1 || e.button !== 0) return;
        e.preventDefault();
        syncDragging = true;
        syncDragStart = { x: e.clientX, y: e.clientY };
        syncPanStart = { x: syncPanX, y: syncPanY };

        const onMove = (ev) => {
            if (!syncDragging) return;
            // Translate mouse pixel delta into pan fraction (relative to video size)
            const wrap = document.querySelector('.sync-video-wrap');
            if (!wrap) return;
            const w = wrap.clientWidth;
            const h = wrap.clientHeight;
            const dx = (ev.clientX - syncDragStart.x) / (w * syncZoom);
            const dy = (ev.clientY - syncDragStart.y) / (h * syncZoom);
            syncPanX = syncPanStart.x + dx;
            syncPanY = syncPanStart.y + dy;
            // Clamp pan so we don't pan beyond the video edge
            const maxPan = (syncZoom - 1) / (2 * syncZoom);
            syncPanX = Math.max(-maxPan, Math.min(maxPan, syncPanX));
            syncPanY = Math.max(-maxPan, Math.min(maxPan, syncPanY));
            _applySyncZoom();
        };
        const onUp = () => {
            syncDragging = false;
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    }

    function _syncOnWheel(e) {
        e.preventDefault();
        if (e.deltaY < 0) {
            syncZoomIn();
        } else {
            syncZoomOut();
        }
    }

    function confirmSync() {
        // Stop playback if running
        if (syncPlaying) syncTogglePlay();
        document.getElementById('step3bNum').textContent = '✓';

        // Proceed to trim step using first camera's video
        _showTrimStep();
    }

    // ── Step 4: Trimmer ──────────────────────────────────────

    function setInPoint(time) {
        const video = document.getElementById('trimVideo');
        if (time === 0) {
            inPoint = 0;
        } else if (time !== undefined && time !== null) {
            inPoint = time;
        } else {
            inPoint = video.currentTime;
        }
        updateTrimDisplay();
    }

    function setOutPoint(time) {
        const video = document.getElementById('trimVideo');
        if (time === -1) {
            outPoint = video.duration || 0;
        } else if (time !== undefined && time !== null) {
            outPoint = time;
        } else {
            outPoint = video.currentTime;
        }
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

        // In edit mode, warn if trial label matches an existing trial (unless actively editing it)
        if (editMode && trial !== editingTrialLabel && existingTrials.some(t => {
            const trialPart = t.replace(subjectName + '_', '');
            return trialPart === trial;
        })) {
            if (!confirm(`Trial "${trial}" already exists for ${subjectName}. Adding a new segment will overwrite it. Continue?`)) return;
        }
        editingTrialLabel = null;

        const sourceName = selectedVideoPath.split(/[/\\]/).pop();

        if (cameraMode === 'multicam' && cameraNames.length > 1) {
            // Add a segment for EACH camera with adjusted times based on sync offsets
            for (const cam of cameraNames) {
                const offset = syncOffsets[cam] || 0;
                const offsetSec = offset / syncFps;
                segments.push({
                    source_path: multicamVideos[cam] || selectedVideoPath,
                    start_time: inPoint + offsetSec,
                    end_time: outPoint + offsetSec,
                    trial_label: trial,
                    camera_name: cam,
                    source_name: (multicamVideos[cam] || selectedVideoPath).split(/[/\\]/).pop(),
                });
                segmentHasFaces.push(true);
            }
        } else {
            segments.push({
                source_path: selectedVideoPath,
                start_time: inPoint,
                end_time: outPoint,
                trial_label: trial,
                source_name: sourceName,
            });
            segmentHasFaces.push(true);
        }

        inPoint = null;
        outPoint = null;
        updateTrimDisplay();
        renderSegments();
        autoAdvanceTrialLabel();
        updateStep5Visibility();
    }

    function removeSegment(idx) {
        // For multicam, remove all segments with same trial_label
        if (cameraMode === 'multicam' && cameraNames.length > 1) {
            const trial = segments[idx].trial_label;
            for (let i = segments.length - 1; i >= 0; i--) {
                if (segments[i].trial_label === trial) {
                    segments.splice(i, 1);
                    segmentHasFaces.splice(i, 1);
                }
            }
        } else {
            segments.splice(idx, 1);
            segmentHasFaces.splice(idx, 1);
        }
        renderSegments();
        updateStep5Visibility();
    }

    function renderSegments() {
        const list = document.getElementById('segmentList');
        if (segments.length === 0) {
            list.innerHTML = '';
            return;
        }

        // For multicam, group by trial label to show condensed view
        if (cameraMode === 'multicam' && cameraNames.length > 1) {
            const trials = {};
            segments.forEach((s, i) => {
                if (!trials[s.trial_label]) trials[s.trial_label] = { seg: s, indices: [] };
                trials[s.trial_label].indices.push(i);
            });

            list.innerHTML = Object.entries(trials).map(([trial, data]) => {
                const s = data.seg;
                return `
                    <div class="segment-item">
                        <span class="trial">${trial}</span>
                        <span class="times">${formatTime(s.start_time)} - ${formatTime(s.end_time)}
                            (${(s.end_time - s.start_time).toFixed(1)}s)</span>
                        <span class="source">${cameraNames.length} cameras</span>
                        <span class="remove" onclick="onboard.removeSegment(${data.indices[0]})">&times;</span>
                    </div>
                `;
            }).join('');
        } else {
            list.innerHTML = segments.map((s, i) => `
                <div class="segment-item">
                    <span class="trial">${s.trial_label}</span>
                    <span class="times">${formatTime(s.start_time)} - ${formatTime(s.end_time)}
                        (${(s.end_time - s.start_time).toFixed(1)}s)</span>
                    <span class="source">${s.source_name}</span>
                    <span class="remove" onclick="onboard.removeSegment(${i})">&times;</span>
                </div>
            `).join('');
        }
    }

    function pickAnotherVideo() {
        document.getElementById('step4').style.display = 'none';
        document.getElementById('step1Num').classList.remove('done');
        document.getElementById('step1').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function autoAdvanceTrialLabel() {
        const input = document.getElementById('trialLabel');
        const used = new Set(segments.map(s => s.trial_label));
        if (editMode) {
            for (const t of existingTrials) {
                used.add(t.replace(subjectName + '_', ''));
            }
        }
        const suggestions = Array.from(
            document.querySelectorAll('#trialLabelSuggestions option')
        ).map(o => o.value);
        const next = suggestions.find(v => !used.has(v));
        input.value = next || '';
    }

    // ── Step 5: Review & Process ─────────────────────────────

    function updateStep5Visibility() {
        const step5 = document.getElementById('step5');
        if (segments.length > 0) {
            step5.style.display = 'block';
            document.getElementById('step4Num').classList.add('done');

            if (editMode) {
                const btn = document.getElementById('processBtn');
                if (btn) btn.textContent = 'Trim & Save';
            }

            // Render review with per-segment has-faces checkboxes
            const review = document.getElementById('reviewSegments');
            const editNote = editMode
                ? `<p style="margin-bottom:8px;color:var(--blue);">Adding to existing subject <strong>${subjectName}</strong> (${existingTrials.length} existing trial${existingTrials.length !== 1 ? 's' : ''}).</p>`
                : '';

            // Group segments by trial for display
            const trialMap = {};
            segments.forEach((s, i) => {
                if (!trialMap[s.trial_label]) trialMap[s.trial_label] = [];
                trialMap[s.trial_label].push({ seg: s, idx: i });
            });

            let segHtml = '';
            for (const [trial, items] of Object.entries(trialMap)) {
                const s = items[0].seg;
                const firstIdx = items[0].idx;
                const outName = s.camera_name
                    ? `${subjectName}_${trial}_*.mp4 (${items.length} cameras)`
                    : `${subjectName}_${trial}.mp4`;
                const hasFacesChecked = segmentHasFaces[firstIdx] !== false ? 'checked' : '';

                segHtml += `
                    <div class="segment-item">
                        <span class="trial">${trial}</span>
                        <span class="times">${formatTime(s.start_time)} - ${formatTime(s.end_time)}</span>
                        <span class="source" style="flex:0;">&#8594; ${outName}</span>
                        <label style="display:flex;align-items:center;gap:4px;font-size:11px;cursor:pointer;margin-left:auto;">
                            <input type="checkbox" ${hasFacesChecked}
                                onchange="onboard.toggleSegmentFaces('${trial}', this.checked)">
                            Has faces
                        </label>
                    </div>
                `;
            }

            review.innerHTML = `
                <p style="margin-bottom:8px;">Subject: <strong>${subjectName}</strong>
                    <span class="meta-badge" style="margin-left:8px;">${subjectGroup}</span>
                    <span class="meta-badge" style="margin-left:4px;">${cameraMode}</span>
                    ${selectedCameraSetup ? `<span class="meta-badge">${selectedCameraSetup.name}</span>` : ''}
                </p>
                ${editNote}
                <p style="margin-bottom:8px;">${Object.keys(trialMap).length} trial(s) will be trimmed and saved.</p>
                ${segHtml}
            `;

            updateProcessButton();
        } else {
            step5.style.display = 'none';
            document.getElementById('step4Num').classList.remove('done');
        }
    }

    function toggleSegmentFaces(trialLabel, hasFaces) {
        segments.forEach((s, i) => {
            if (s.trial_label === trialLabel) {
                segmentHasFaces[i] = hasFaces;
            }
        });
    }

    function updateProcessButton() {
        const btn = document.getElementById('processBtn');
        if (btn) {
            btn.textContent = editMode ? 'Trim & Save' : 'Trim & Create Subject';
        }
    }

    async function startProcessing() {
        if (segments.length === 0) return;

        const statusEl = document.getElementById('processingStatus');
        const fillEl = document.getElementById('processingFill');
        const msgEl = document.getElementById('processingMsg');
        statusEl.style.display = 'block';
        msgEl.textContent = 'Starting...';

        try {
            // Build no_face_trials from segmentHasFaces
            const noFaceTrials = [];
            const seen = new Set();
            segments.forEach((s, i) => {
                if (!seen.has(s.trial_label)) {
                    seen.add(s.trial_label);
                    if (segmentHasFaces[i] === false) {
                        noFaceTrials.push(s.trial_label);
                    }
                }
            });

            const result = await API.post('/api/video-tools/process-subject', {
                subject_name: subjectName,
                blur_faces: false,
                camera_mode: cameraMode,
                camera_name: selectedCameraSetup ? selectedCameraSetup.name : null,
                diagnosis: subjectGroup,
                no_face_trials: noFaceTrials,
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
                API.streamJob(result.job_id,
                    (data) => {
                        const pct = data.progress_pct || 0;
                        fillEl.style.width = `${pct}%`;
                        msgEl.textContent = pct < 2 ? 'Initializing...' : `Processing: ${pct.toFixed(0)}%`;
                    },
                    (data) => {
                        if (data.status === 'completed') {
                            msgEl.textContent = editMode
                                ? 'Segments added! Redirecting to dashboard...'
                                : 'Complete! Redirecting to dashboard...';
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

    // ── Edit mode ────────────────────────────────────────────

    let editMode = false;
    let editSubjectId = null;
    let existingTrials = [];
    let existingSegments = [];
    let editDetail = null;
    let editingTrialLabel = null;

    async function initEditMode(subjectId) {
        editMode = true;
        editSubjectId = subjectId;

        try {
            const detail = await API.get(`/api/subjects/${subjectId}`);
            editDetail = detail;
            subjectName = detail.name;
            cameraMode = detail.camera_mode || 'stereo';
            existingTrials = detail.trials || [];
            existingSegments = detail.segments || [];

            // Step 1 becomes a file browser (already visible)
            // Show existing trials card
            _renderExistingTrials();

            // Pre-populate diagnosis group
            subjectGroup = detail.diagnosis || 'Control';

            // Also show the name field pre-filled in step 2
            document.getElementById('step2').style.display = 'block';
            document.getElementById('step2Num').classList.add('done');

            // Replace step 2 header and contents for edit mode
            const step2 = document.getElementById('step2');
            const nameInput = step2.querySelector('#subjectName');
            if (nameInput) {
                nameInput.value = subjectName;
            }
            // Pre-populate group dropdown and wire up live-save
            const groupSelect = step2.querySelector('#subjectGroup');
            if (groupSelect) {
                groupSelect.value = subjectGroup;
                groupSelect.onchange = async () => {
                    subjectGroup = groupSelect.value;
                    try {
                        await API.patch(`/api/subjects/${editSubjectId}`, { diagnosis: subjectGroup });
                    } catch (e) {
                        alert('Error updating group: ' + e.message);
                    }
                };
            }
            // Replace Confirm button with Rename
            const confirmBtn = step2.querySelector('.btn-primary');
            if (confirmBtn) {
                confirmBtn.textContent = 'Rename';
                confirmBtn.onclick = () => renameSubject();
            }
            const step2Header = step2.querySelector('h3');
            if (step2Header) step2Header.textContent = 'Edit Subject';

            // Skip camera mode step in edit mode (already set)
            document.getElementById('step3Num').classList.add('done');

            // Open the file browser
            await loadDirectory('');
            await prefillFromSession();

        } catch (e) {
            alert('Error loading subject: ' + e.message);
        }
    }

    async function renameSubject() {
        const nameInput = document.getElementById('subjectName');
        const newName = nameInput.value.trim();
        if (!newName) return alert('Enter a subject name');
        if (!/^[A-Za-z0-9_]+$/.test(newName)) return alert('Name must be alphanumeric (letters, numbers, underscores)');
        if (newName === subjectName) return;

        try {
            await API.patch(`/api/subjects/${editSubjectId}`, { name: newName });
            const oldName = subjectName;
            subjectName = newName;

            existingTrials = existingTrials.map(t => t.replace(oldName, newName));
            _renderExistingTrials();
            alert(`Subject renamed to "${newName}"`);
        } catch (e) {
            alert('Rename failed: ' + e.message);
        }
    }

    async function deleteExistingTrial(trialLabel) {
        if (!confirm(`Delete trial "${trialLabel}" and its video file(s)? This cannot be undone.`)) return;
        try {
            await API.del(`/api/subjects/${editSubjectId}/trials/${encodeURIComponent(trialLabel)}`);
            existingTrials = existingTrials.filter(t => {
                const label = t.replace(subjectName + '_', '');
                return label !== trialLabel;
            });
            existingSegments = existingSegments.filter(s => s.trial_label !== trialLabel);
            _renderExistingTrials();
        } catch (e) {
            alert('Delete failed: ' + e.message);
        }
    }

    async function editExistingTrial(trialLabel) {
        editingTrialLabel = trialLabel;

        const seg = existingSegments.find(s => s.trial_label === trialLabel);
        if (!seg) {
            alert('No source metadata found for this trial. You can delete it and re-add from a source video.');
            return;
        }

        selectedVideoPath = seg.source_path;

        try {
            videoMeta = await API.get(`/api/video-tools/probe?path=${encodeURIComponent(selectedVideoPath)}`);
        } catch (e) {
            alert('Cannot load source video. The file may have been moved.\n\n' + e.message);
            return;
        }

        // Show trimmer
        document.getElementById('step4').style.display = 'block';

        const video = document.getElementById('trimVideo');
        video.src = `/api/video-tools/stream?path=${encodeURIComponent(selectedVideoPath)}`;

        inPoint = seg.start_time;
        outPoint = seg.end_time;
        updateTrimDisplay();

        document.getElementById('trialLabel').value = trialLabel;

        video.addEventListener('loadedmetadata', function _seek() {
            video.currentTime = seg.start_time;
            video.removeEventListener('loadedmetadata', _seek);
        });

        document.getElementById('step4').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function _renderExistingTrials() {
        const segMap = {};
        for (const s of existingSegments) {
            segMap[s.trial_label] = s;
        }

        let existingCard = document.getElementById('existingTrialsCard');
        if (!existingCard) {
            existingCard = document.createElement('div');
            existingCard.className = 'card';
            existingCard.id = 'existingTrialsCard';
            const step1 = document.getElementById('step1');
            step1.parentNode.insertBefore(existingCard, step1);
        }

        if (existingTrials.length === 0) {
            existingCard.innerHTML = `
                <div class="step-header">
                    <div class="step-num" style="background:var(--border);color:var(--text-muted);">0</div>
                    <h3>No Existing Trials</h3>
                </div>
                <p style="font-size:13px;color:var(--text-muted);">
                    Add segments below to create trials for this subject.
                </p>
            `;
            return;
        }

        existingCard.innerHTML = `
            <div class="step-header">
                <div class="step-num done" style="background:var(--green);">&#10003;</div>
                <h3>Existing Trials (${existingTrials.length})</h3>
            </div>
            <p style="font-size:13px;color:var(--text-muted);margin-bottom:8px;">
                Edit or delete existing trials, or add new segments below.
            </p>
            <div class="segment-list">
                ${existingTrials.map(t => {
                    const trialLabel = t.replace(subjectName + '_', '');
                    const seg = segMap[trialLabel];
                    const sourceName = seg ? seg.source_path.split(/[/\\]/).pop() : '';
                    const timeStr = seg
                        ? `${formatTime(seg.start_time)} - ${formatTime(seg.end_time)} (${(seg.end_time - seg.start_time).toFixed(1)}s)`
                        : '';
                    return `
                        <div class="segment-item">
                            <span class="trial" style="min-width:50px;">${trialLabel}</span>
                            ${seg ? `
                                <span class="times">${timeStr}</span>
                                <span class="source">${sourceName}</span>
                                <span class="edit-trial" onclick="onboard.editExistingTrial('${trialLabel}')"
                                    style="cursor:pointer;color:var(--blue);font-size:12px;font-weight:600;"
                                    title="Re-trim this trial from source video">edit</span>
                            ` : `
                                <span class="source" style="flex:1;">${t}.mp4</span>
                                <span style="color:var(--text-muted);font-size:11px;" title="Source metadata not available">no source info</span>
                            `}
                            <span class="remove" onclick="onboard.deleteExistingTrial('${trialLabel}')"
                                title="Delete this trial">&times;</span>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    // ── Checkbox listener & init ─────────────────────────────

    document.addEventListener('DOMContentLoaded', async () => {
        // Populate subject group dropdown from settings
        try {
            const cfg = await API.get('/api/settings');
            if (Array.isArray(cfg.diagnosis_groups) && cfg.diagnosis_groups.length) {
                diagnosisGroups = cfg.diagnosis_groups;
            }
        } catch { /* use defaults */ }
        const groupSel = document.getElementById('subjectGroup');
        if (groupSel) {
            _populateGroupDropdown(groupSel);
            groupSel.addEventListener('change', () => _onGroupSelectChange(groupSel));
        }

        const params = new URLSearchParams(window.location.search);
        const subjectId = params.get('subject');
        if (subjectId) {
            initEditMode(parseInt(subjectId));
        } else {
            // Normal flow: open file browser immediately
            await loadDirectory('');
            await prefillFromSession();
        }
    });

    // ── Prefill from video viewer session ────────────────────

    async function prefillFromSession() {
        const videoName = sessionStorage.getItem('onboard_prefill_video');
        if (!videoName) return;
        sessionStorage.removeItem('onboard_prefill_video');

        try {
            const settings = await API.get('/api/settings');
            if (!settings.video_path) return;

            await loadDirectory(settings.video_path);

            const items = document.querySelectorAll('#fileBrowser .fb-item[data-name]');
            for (const el of items) {
                if (el.dataset.name === videoName) {
                    selectFile(el.dataset.path, el);
                    el.scrollIntoView({ block: 'nearest' });
                    break;
                }
            }
        } catch (e) { /* best-effort */ }
    }

    // ── Keyboard shortcuts ───────────────────────────────────

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        // Trim shortcuts
        if (e.key === 'i' || e.key === 'I') {
            e.preventDefault();
            setInPoint();
        } else if (e.key === 'o' || e.key === 'O') {
            e.preventDefault();
            setOutPoint();
        }

        // Sync shortcuts (arrow keys, space, zoom)
        if (document.getElementById('step3b').style.display !== 'none') {
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                syncStepAll(e.shiftKey ? -10 : -1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                syncStepAll(e.shiftKey ? 10 : 1);
            } else if (e.key === ' ') {
                e.preventDefault();
                syncTogglePlay();
            } else if (e.key === '=' || e.key === '+') {
                e.preventDefault();
                syncZoomIn();
            } else if (e.key === '-' || e.key === '_') {
                e.preventDefault();
                syncZoomOut();
            } else if (e.key === '0') {
                e.preventDefault();
                syncZoomReset();
            }
        }
    });

    // ── Helpers ──────────────────────────────────────────────

    function formatTime(seconds) {
        if (seconds == null) return '--';
        const m = Math.floor(seconds / 60);
        const s = (seconds % 60).toFixed(1);
        return `${m}:${s.padStart(4, '0')}`;
    }

    function escPath(path) {
        return path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    }

    // ── Public API ───────────────────────────────────────────

    return {
        browse,
        selectFile,
        dblClickFile,
        selectVideo,
        confirmName,
        selectCameraMode,
        toggleNewSetupForm,
        createCameraSetup,
        confirmCameraMode,
        pickMulticamVideo,
        syncStepAll,
        syncStepSingle,
        syncTogglePlay,
        syncSetRate,
        syncTimelineDrag: _syncTimelineDragStart,
        syncZoomIn,
        syncZoomOut,
        syncZoomReset,
        confirmSync,
        setInPoint,
        setOutPoint,
        addSegment,
        removeSegment,
        pickAnotherVideo,
        startProcessing,
        toggleSegmentFaces,
        renameSubject,
        deleteExistingTrial,
        editExistingTrial,
    };
})();
