/* Settings page: load/save settings, tag inputs for lists */

// ── Tag input helper ──────────────────────────────────────────
function setupTagInput(containerId, inputId) {
    const container = document.getElementById(containerId);
    const input = document.getElementById(inputId);

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            const val = input.value.trim().replace(/,/g, '');
            if (val) {
                addTag(containerId, val);
                input.value = '';
            }
        }
        if (e.key === 'Backspace' && !input.value) {
            // Remove last tag
            const tags = container.querySelectorAll('.tag');
            if (tags.length > 0) {
                tags[tags.length - 1].remove();
            }
        }
    });
}

function addTag(containerId, value) {
    const container = document.getElementById(containerId);
    const input = container.querySelector('input');
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.innerHTML = `${value}<span class="remove" onclick="this.parentElement.remove()">&times;</span>`;
    container.insertBefore(tag, input);
}

function getTagValues(containerId) {
    const container = document.getElementById(containerId);
    return Array.from(container.querySelectorAll('.tag'))
        .map(t => t.textContent.replace('\u00d7', '').trim());
}

function setTagValues(containerId, values) {
    const container = document.getElementById(containerId);
    // Remove existing tags
    container.querySelectorAll('.tag').forEach(t => t.remove());
    // Add new tags
    values.forEach(v => addTag(containerId, v));
}

// ── Init ──────────────────────────────────────────────────────
setupTagInput('camera_names_tags', 'camera_names_input');
setupTagInput('bodyparts_tags', 'bodyparts_input');

// ── Load settings ─────────────────────────────────────────────
async function loadSettings() {
    try {
        const [settings, status] = await Promise.all([
            API.get('/api/settings'),
            API.get('/api/settings/status'),
        ]);

        // Populate simple fields
        const fields = [
            'video_dir', 'dlc_dir', 'calibration_3d_config',
            'python_executable', 'dlc_scorer', 'dlc_date', 'host',
        ];
        fields.forEach(f => {
            const el = document.getElementById(f);
            if (el) el.value = settings[f] || '';
        });

        // Port
        document.getElementById('port').value = settings.port || 8080;

        // Select
        const netType = document.getElementById('dlc_net_type');
        netType.value = settings.dlc_net_type || 'resnet_50';

        // Tag inputs
        setTagValues('camera_names_tags', settings.camera_names || []);
        setTagValues('bodyparts_tags', settings.bodyparts || []);

        // Status banner
        renderStatus(status);
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
}

function renderStatus(status) {
    const banner = document.getElementById('statusBanner');
    if (status.configured) {
        banner.innerHTML = `<div class="status-indicator status-ok">Configured</div>`;
    } else {
        const issues = status.issues.map(i => `<div style="font-size:12px;">&bull; ${i}</div>`).join('');
        banner.innerHTML = `
            <div class="card" style="border-color:var(--orange);">
                <div class="status-indicator status-warn" style="margin-bottom:8px;">Not Configured</div>
                ${issues}
                <div style="font-size:12px;color:var(--text-muted);margin-top:8px;">
                    Set <strong>Video Directory</strong> and <strong>DLC Project Directory</strong> to get started.
                </div>
            </div>
        `;
    }
}

// ── Save settings ─────────────────────────────────────────────
async function saveSettings() {
    const data = {
        video_dir: document.getElementById('video_dir').value.trim(),
        dlc_dir: document.getElementById('dlc_dir').value.trim(),
        calibration_3d_config: document.getElementById('calibration_3d_config').value.trim(),
        python_executable: document.getElementById('python_executable').value.trim(),
        camera_names: getTagValues('camera_names_tags'),
        bodyparts: getTagValues('bodyparts_tags'),
        dlc_scorer: document.getElementById('dlc_scorer').value.trim(),
        dlc_date: document.getElementById('dlc_date').value.trim(),
        dlc_net_type: document.getElementById('dlc_net_type').value,
        host: document.getElementById('host').value.trim(),
        port: parseInt(document.getElementById('port').value) || 8080,
    };

    try {
        await API.put('/api/settings', data);
        const status = await API.get('/api/settings/status');
        renderStatus(status);
        alert('Settings saved.');
    } catch (e) {
        alert('Error saving settings: ' + e.message);
    }
}

// ── Directory browser ─────────────────────────────────────────
let browseTargetField = null;
let browseCurrentPath = '';

function openBrowse(fieldId) {
    browseTargetField = fieldId;
    const current = document.getElementById(fieldId).value.trim();
    document.getElementById('browseModal').classList.add('active');
    loadBrowse(current || null);
}

function closeBrowse() {
    document.getElementById('browseModal').classList.remove('active');
    browseTargetField = null;
}

function selectBrowsePath() {
    if (browseTargetField && browseCurrentPath) {
        document.getElementById(browseTargetField).value = browseCurrentPath;
    }
    closeBrowse();
}

async function loadBrowse(path) {
    const url = path
        ? '/api/settings/browse?path=' + encodeURIComponent(path)
        : '/api/settings/browse';

    try {
        const data = await API.get(url);
        browseCurrentPath = data.path;
        document.getElementById('browsePath').textContent = data.path;

        const list = document.getElementById('browseList');
        let html = '';

        if (data.parent) {
            html += `<div class="browse-item browse-up" onclick="loadBrowse('${data.parent.replace(/'/g, "\\'")}')">Parent directory</div>`;
        }

        if (data.error) {
            html += `<div style="padding:12px;color:var(--red);font-size:13px;">${data.error}</div>`;
        }

        data.dirs.forEach(d => {
            const full = data.path + (data.path.endsWith('/') ? '' : '/') + d;
            html += `<div class="browse-item" onclick="loadBrowse('${full.replace(/'/g, "\\'")}')">${d}</div>`;
        });

        if (!data.dirs.length && !data.error) {
            html += '<div style="padding:12px;color:var(--text-muted);font-size:13px;">Empty directory</div>';
        }

        list.innerHTML = html;
    } catch (e) {
        console.error('Browse failed:', e);
    }
}

// Escape closes browse modal
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeBrowse();
});

// ── DLC install ──────────────────────────────────────────────
let dlcPollTimer = null;

async function checkDlcStatus() {
    try {
        const data = await API.get('/api/settings/dlc-status');
        const statusEl = document.getElementById('dlcStatus');
        const btnEl = document.getElementById('dlcInstallBtn');
        const logEl = document.getElementById('dlcInstallLog');

        if (data.installed) {
            statusEl.innerHTML = `<span class="status-indicator status-ok">DeepLabCut ${data.version} installed</span>`;
            btnEl.innerHTML = '';
            logEl.style.display = 'none';
            stopDlcPoll();
        } else if (data.install_running) {
            statusEl.innerHTML = `<span class="status-indicator status-warn">Installing DeepLabCut...</span>`;
            btnEl.innerHTML = '';
            logEl.style.display = 'block';
            logEl.textContent = (data.install_log || []).join('\n');
            logEl.scrollTop = logEl.scrollHeight;
            startDlcPoll();
        } else if (data.install_status === 'completed') {
            statusEl.innerHTML = `<span class="status-indicator status-ok">DeepLabCut installed successfully</span>`;
            btnEl.innerHTML = '';
            logEl.style.display = 'none';
            stopDlcPoll();
        } else if (data.install_status === 'failed') {
            statusEl.innerHTML = `<span class="status-indicator status-warn">Install failed: ${data.install_error || 'unknown error'}</span>`;
            btnEl.innerHTML = `<button class="btn" onclick="installDlc()">Retry Install</button>`;
            logEl.style.display = 'block';
            logEl.textContent = (data.install_log || []).join('\n');
            logEl.scrollTop = logEl.scrollHeight;
            stopDlcPoll();
        } else {
            statusEl.innerHTML = 'DeepLabCut is not installed.';
            btnEl.innerHTML = `<button class="btn btn-primary" onclick="installDlc()">Install DeepLabCut</button>`;
            logEl.style.display = 'none';
        }
    } catch (e) {
        console.error('DLC status check failed:', e);
    }
}

async function installDlc() {
    try {
        await API.post('/api/settings/install-dlc');
        startDlcPoll();
        checkDlcStatus();
    } catch (e) {
        alert('Failed to start DLC install: ' + e.message);
    }
}

function startDlcPoll() {
    if (!dlcPollTimer) {
        dlcPollTimer = setInterval(checkDlcStatus, 3000);
    }
}

function stopDlcPoll() {
    if (dlcPollTimer) {
        clearInterval(dlcPollTimer);
        dlcPollTimer = null;
    }
}

// Load on page ready
loadSettings();
checkDlcStatus();
