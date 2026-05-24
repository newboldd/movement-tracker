/* Camera setup & calibration page logic */

const calibUI = (() => {
    let setups = [];
    let selectedSetup = null;
    let calibSource = 'video'; // 'video' or 'images'
    let currentPath = '';
    let selectedFilePath = null;
    let selectedImageDir = null;
    let inPoint = null;
    let outPoint = null;

    // ── Init ──────────────────────────────────────────────

    async function init() {
        await loadSetups();
    }

    // ── Setup list ────────────────────────────────────────

    async function loadSetups() {
        const list = document.getElementById('setupList');
        try {
            setups = await API.get('/api/camera-setups');
            if (setups.length === 0) {
                list.innerHTML = '<div style="color:var(--text-muted);font-size:13px;">No camera setups yet. Create one to get started.</div>';
                return;
            }
            list.innerHTML = setups.map(s => `
                <div class="setup-card${selectedSetup && selectedSetup.id === s.id ? ' selected' : ''}"
                     onclick="calibUI.selectSetup(${s.id})">
                    <div class="status-dot ${s.has_calibration ? 'calibrated' : 'uncalibrated'}"></div>
                    <div style="flex:1;">
                        <div class="name">${esc(s.name)}</div>
                        <div class="meta">${s.mode} &middot; ${s.camera_count} cameras &middot; ${s.camera_names.join(', ')}
                            ${s.has_calibration ? ' &middot; calibrated' : ''}</div>
                    </div>
                    <button class="btn btn-sm" style="color:var(--red);border-color:var(--red);"
                            onclick="event.stopPropagation(); calibUI.deleteSetup(${s.id}, '${escAttr(s.name)}')">Delete</button>
                </div>
            `).join('');
        } catch (e) {
            list.innerHTML = `<div style="color:var(--red);font-size:13px;">${esc(e.message)}</div>`;
        }
    }

    function selectSetup(id) {
        selectedSetup = setups.find(s => s.id === id) || null;
        if (!selectedSetup) return;

        // Update list selection
        loadSetups();

        // Show calibration section
        const section = document.getElementById('calibrateSection');
        section.style.display = 'block';
        document.getElementById('calibSetupName').textContent = selectedSetup.name;
        document.getElementById('calibStatus').textContent = selectedSetup.has_calibration
            ? 'Calibration file exists. Re-running will overwrite it.'
            : 'No calibration yet.';

        // Pre-fill checkerboard from setup if available
        if (selectedSetup.checkerboard_rows) {
            document.getElementById('cbRows').value = selectedSetup.checkerboard_rows;
        }
        if (selectedSetup.checkerboard_cols) {
            document.getElementById('cbCols').value = selectedSetup.checkerboard_cols;
        }

        // Reset state
        selectedFilePath = null;
        selectedImageDir = null;
        inPoint = null;
        outPoint = null;
        document.getElementById('calibProgress').style.display = 'none';

        // Init file browser for default source
        setSource(calibSource);
    }

    async function deleteSetup(id, name) {
        if (!confirm(`Delete camera setup "${name}"?`)) return;
        try {
            await API.del(`/api/camera-setups/${id}`);
            if (selectedSetup && selectedSetup.id === id) {
                selectedSetup = null;
                document.getElementById('calibrateSection').style.display = 'none';
            }
            await loadSetups();
        } catch (e) {
            alert(e.message);
        }
    }

    // ── Create form ───────────────────────────────────────

    function showCreateForm() {
        document.getElementById('createSection').style.display = 'block';
        document.getElementById('setupName').value = '';
        document.getElementById('setupMode').value = 'stereo';
        document.getElementById('camCount').value = 2;
        document.getElementById('camNames').value = 'cam1, cam2';
        onModeChange();
    }

    function hideCreateForm() {
        document.getElementById('createSection').style.display = 'none';
    }

    function onModeChange() {
        const mode = document.getElementById('setupMode').value;
        document.getElementById('camCountRow').style.display = mode === 'multicam' ? '' : 'none';
    }

    function updateCamNames() {
        const count = parseInt(document.getElementById('camCount').value) || 2;
        const names = [];
        for (let i = 1; i <= count; i++) names.push(`cam${i}`);
        document.getElementById('camNames').value = names.join(', ');
    }

    async function createSetup() {
        const name = document.getElementById('setupName').value.trim();
        if (!name) return alert('Enter a setup name');

        const mode = document.getElementById('setupMode').value;
        const camNamesRaw = document.getElementById('camNames').value;
        const camNames = camNamesRaw.split(',').map(s => s.trim()).filter(Boolean);
        const camCount = mode === 'multicam'
            ? parseInt(document.getElementById('camCount').value) || camNames.length
            : 2;

        try {
            await API.post('/api/camera-setups', {
                name,
                mode,
                camera_count: camCount,
                camera_names: camNames,
            });
            hideCreateForm();
            await loadSetups();
        } catch (e) {
            alert(e.message);
        }
    }

    // ── Calibration source toggle ─────────────────────────

    function setSource(source) {
        calibSource = source;
        const videoBtn = document.getElementById('srcVideoBtn');
        const imagesBtn = document.getElementById('srcImagesBtn');
        const videoSection = document.getElementById('videoSourceSection');
        const imageSection = document.getElementById('imageSourceSection');

        if (source === 'video') {
            videoBtn.style.borderColor = 'var(--blue)';
            imagesBtn.style.borderColor = '';
            videoSection.style.display = '';
            imageSection.style.display = 'none';
            loadFileBrowser('calibFileBrowser', '', 'video');
        } else {
            videoBtn.style.borderColor = '';
            imagesBtn.style.borderColor = 'var(--blue)';
            videoSection.style.display = 'none';
            imageSection.style.display = '';
            loadFileBrowser('calibImageBrowser', '', 'images');
        }
    }

    // ── File browser (shared for video and image dir) ─────

    async function loadFileBrowser(browserId, path, mode) {
        const browser = document.getElementById(browserId);
        browser.innerHTML = '<div style="padding:12px;color:var(--text-muted)">Loading...</div>';

        try {
            const data = await API.get(`/api/files?path=${encodeURIComponent(path)}`);
            currentPath = data.path || '';
            let html = '';

            // Breadcrumbs
            if (data.breadcrumbs && data.breadcrumbs.length > 0) {
                html += '<div class="fb-breadcrumbs">';
                html += `<a onclick="calibUI.browseTo('${browserId}', '', '${mode}')">Home</a> /`;
                for (const crumb of data.breadcrumbs) {
                    html += ` <a onclick="calibUI.browseTo('${browserId}', '${escPath(crumb.path)}', '${mode}')">${esc(crumb.name)}</a> /`;
                }
                html += '</div>';
            }

            // Locations at root
            if (!currentPath && data.locations) {
                for (const loc of data.locations) {
                    html += `<div class="fb-item" onclick="calibUI.browseTo('${browserId}', '${escPath(loc.path)}', '${mode}')">
                        <span class="icon">&#128193;</span>
                        <span>${esc(loc.name)}</span>
                        <span style="color:var(--text-muted);font-size:11px;margin-left:auto;">${esc(loc.path)}</span>
                    </div>`;
                }
            }

            // Parent
            if (data.parent) {
                html += `<div class="fb-item" onclick="calibUI.browseTo('${browserId}', '${escPath(data.parent)}', '${mode}')">
                    <span class="icon">&#8592;</span> <span>..</span>
                </div>`;
            }

            // Items
            for (const item of data.items) {
                if (item.type === 'dir') {
                    if (mode === 'images') {
                        // Directories are selectable as image pair source
                        html += `<div class="fb-item" ondblclick="calibUI.browseTo('${browserId}', '${escPath(item.path)}', '${mode}')"
                                      onclick="calibUI.selectImageDir('${escPath(item.path)}', this)">
                            <span class="icon">&#128193;</span>
                            <span>${esc(item.name)}</span>
                        </div>`;
                    } else {
                        html += `<div class="fb-item" onclick="calibUI.browseTo('${browserId}', '${escPath(item.path)}', '${mode}')">
                            <span class="icon">&#128193;</span>
                            <span>${esc(item.name)}</span>
                        </div>`;
                    }
                } else if (mode === 'video') {
                    html += `<div class="fb-item" onclick="calibUI.selectCalibVideo('${escPath(item.path)}', this)"
                                  ondblclick="calibUI.selectCalibVideo('${escPath(item.path)}', this)">
                        <span class="icon">&#127909;</span>
                        <span>${esc(item.name)}</span>
                        <span style="color:var(--text-muted);font-size:11px;margin-left:auto;">${item.size_mb} MB</span>
                    </div>`;
                }
            }

            if ((!data.items || data.items.length === 0) && currentPath) {
                html += '<div style="padding:12px;color:var(--text-muted)">No items found</div>';
            }

            browser.innerHTML = html;
        } catch (e) {
            browser.innerHTML = `<div style="padding:12px;color:var(--red)">${esc(e.message)}</div>`;
        }
    }

    function browseTo(browserId, path, mode) {
        loadFileBrowser(browserId, path, mode);
    }

    // ── Video file selection + trimmer ────────────────────

    function selectCalibVideo(path, el) {
        // Deselect siblings
        el.parentElement.querySelectorAll('.fb-item.selected').forEach(e => e.classList.remove('selected'));
        el.classList.add('selected');
        selectedFilePath = path;

        // Show trimmer
        const trimmer = document.getElementById('calibVideoTrimmer');
        trimmer.style.display = '';
        document.getElementById('calibVideoName').textContent = path.split(/[/\\]/).pop();

        const video = document.getElementById('calibTrimVideo');
        video.src = `/api/video-tools/stream?path=${encodeURIComponent(path)}`;

        inPoint = null;
        outPoint = null;
        updateTrimDisplay();
    }

    function pickAnotherCalibVideo() {
        selectedFilePath = null;
        inPoint = null;
        outPoint = null;
        document.getElementById('calibVideoTrimmer').style.display = 'none';
    }

    function setIn() {
        const video = document.getElementById('calibTrimVideo');
        inPoint = video.currentTime;
        updateTrimDisplay();
    }

    function setOut() {
        const video = document.getElementById('calibTrimVideo');
        outPoint = video.currentTime;
        updateTrimDisplay();
    }

    function updateTrimDisplay() {
        document.getElementById('calibInTime').textContent =
            inPoint !== null ? `In: ${formatTime(inPoint)}` : 'In: --';
        document.getElementById('calibOutTime').textContent =
            outPoint !== null ? `Out: ${formatTime(outPoint)}` : 'Out: --';
    }

    // ── Image directory selection ─────────────────────────

    function selectImageDir(path, el) {
        el.parentElement.querySelectorAll('.fb-item.selected').forEach(e => e.classList.remove('selected'));
        el.classList.add('selected');
        selectedImageDir = path;

        const display = document.getElementById('selectedImageDir');
        display.textContent = `Selected: ${path}`;
        display.style.display = '';
    }

    // ── Run calibration ───────────────────────────────────

    async function runCalibration() {
        if (!selectedSetup) return alert('Select a camera setup first');

        const rows = parseInt(document.getElementById('cbRows').value);
        const cols = parseInt(document.getElementById('cbCols').value);
        const squareSize = parseFloat(document.getElementById('cbSquare').value);

        let endpoint, body;

        if (calibSource === 'video') {
            if (!selectedFilePath) return alert('Select a calibration video');
            endpoint = '/api/camera-setups/calibrate-from-video';
            body = {
                setup_id: selectedSetup.id,
                video_path: selectedFilePath,
                start_time: inPoint || 0,
                end_time: outPoint || 0,
                checkerboard_rows: rows,
                checkerboard_cols: cols,
                square_size_mm: squareSize,
            };
        } else {
            if (!selectedImageDir) return alert('Select a directory with calibration images');
            endpoint = '/api/camera-setups/calibrate-from-images';
            body = {
                setup_id: selectedSetup.id,
                image_dir: selectedImageDir,
                checkerboard_rows: rows,
                checkerboard_cols: cols,
                square_size_mm: squareSize,
            };
        }

        const progressEl = document.getElementById('calibProgress');
        const fillEl = document.getElementById('calibFill');
        const msgEl = document.getElementById('calibMsg');
        const runBtn = document.getElementById('runCalibBtn');

        progressEl.style.display = '';
        fillEl.style.width = '0%';
        msgEl.textContent = 'Starting calibration...';
        msgEl.style.color = '';
        runBtn.disabled = true;

        try {
            const result = await API.post(endpoint, body);

            if (result.job_id) {
                API.streamJob(result.job_id,
                    (data) => {
                        const pct = data.progress_pct || 0;
                        fillEl.style.width = `${pct}%`;
                        if (pct < 5) {
                            msgEl.textContent = 'Initializing...';
                        } else if (pct < 85) {
                            msgEl.textContent = `Detecting checkerboard corners: ${pct.toFixed(0)}%`;
                        } else if (pct < 100) {
                            msgEl.textContent = `Running stereo calibration: ${pct.toFixed(0)}%`;
                        }
                    },
                    (data) => {
                        runBtn.disabled = false;
                        if (data.status === 'completed') {
                            fillEl.style.width = '100%';
                            msgEl.textContent = 'Calibration complete!';
                            msgEl.style.color = 'var(--green)';
                            loadSetups(); // refresh status dots
                        } else if (data.status === 'failed') {
                            msgEl.textContent = `Failed: ${data.error_msg || 'Unknown error'}`;
                            msgEl.style.color = 'var(--red)';
                        } else if (data.status === 'cancelled') {
                            msgEl.textContent = 'Calibration cancelled.';
                            msgEl.style.color = 'var(--orange)';
                        }
                    }
                );
            }
        } catch (e) {
            msgEl.textContent = `Error: ${e.message}`;
            msgEl.style.color = 'var(--red)';
            runBtn.disabled = false;
        }
    }

    // ── Keyboard shortcuts ────────────────────────────────

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'i' || e.key === 'I') { e.preventDefault(); setIn(); }
        else if (e.key === 'o' || e.key === 'O') { e.preventDefault(); setOut(); }
    });

    // ── Helpers ───────────────────────────────────────────

    function formatTime(seconds) {
        if (seconds == null) return '--';
        const m = Math.floor(seconds / 60);
        const s = (seconds % 60).toFixed(1);
        return `${m}:${s.padStart(4, '0')}`;
    }

    function escPath(path) {
        return path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
    }

    function esc(str) {
        const el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }

    function escAttr(str) {
        return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
    }

    // ── Boot ──────────────────────────────────────────────

    document.addEventListener('DOMContentLoaded', init);

    // ── Public API ────────────────────────────────────────

    return {
        showCreateForm,
        hideCreateForm,
        onModeChange,
        updateCamNames,
        createSetup,
        selectSetup,
        deleteSetup,
        setSource,
        browseTo,
        selectCalibVideo,
        pickAnotherCalibVideo,
        setIn,
        setOut,
        selectImageDir,
        runCalibration,
    };
})();
