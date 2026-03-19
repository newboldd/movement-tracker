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

// ── Calibration rows ──────────────────────────────────────────
function addCalibrationRow(name, path) {
    const container = document.getElementById('calibrationRows');
    const row = document.createElement('div');
    row.className = 'calib-row';
    row.innerHTML = `
        <input type="text" class="calib-name" placeholder="Camera name" value="${name}">
        <input type="text" class="calib-path" placeholder="/path/to/calibration.yaml" value="${path}">
        <span class="calib-status" title="Not checked"></span>
        <button class="btn btn-sm" onclick="validateCalibRow(this)">Check</button>
        <button class="btn btn-sm" onclick="this.closest('.calib-row').remove()" style="color:var(--red);">&times;</button>
    `;
    container.appendChild(row);
}

function getCalibrations() {
    const rows = document.querySelectorAll('.calib-row');
    const calibs = {};
    rows.forEach(row => {
        const name = row.querySelector('.calib-name').value.trim();
        const path = row.querySelector('.calib-path').value.trim();
        if (name && path) calibs[name] = path;
    });
    return calibs;
}

function setCalibrations(calibrations) {
    const container = document.getElementById('calibrationRows');
    container.innerHTML = '';
    if (calibrations) {
        Object.entries(calibrations).forEach(([name, path]) => addCalibrationRow(name, path));
    }
}

async function validateCalibRow(btn) {
    const row = btn.closest('.calib-row');
    const path = row.querySelector('.calib-path').value.trim();
    const dot = row.querySelector('.calib-status');
    if (!path) { dot.className = 'calib-status invalid'; dot.title = 'No path'; return; }
    btn.disabled = true;
    btn.textContent = '...';
    try {
        const resp = await API.post('/api/settings/validate-calibration', { path });
        dot.className = resp.valid ? 'calib-status valid' : 'calib-status invalid';
        dot.title = resp.valid ? 'Valid' : (resp.error || 'Invalid');
    } catch (e) {
        dot.className = 'calib-status invalid';
        dot.title = e.message;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Check';
    }
}

// ── Event type rows ──────────────────────────────────────────
const AUTO_DETECT_TYPES = ['open', 'peak', 'close'];

function addEventTypeRow(name, color, shortcut) {
    const container = document.getElementById('eventTypeRows');
    const row = document.createElement('div');
    row.className = 'event-type-row';
    const isSpecial = AUTO_DETECT_TYPES.includes(name.toLowerCase());
    row.innerHTML = `
        <input type="text" class="et-name" placeholder="Event name" value="${name}">
        <span class="et-label">Color</span>
        <input type="color" class="et-color" value="${color}">
        <span class="et-label">Key</span>
        <input type="text" class="et-shortcut" placeholder="1" value="${shortcut}" maxlength="1">
        ${isSpecial ? '<span class="et-special" title="Used by auto-detection">auto</span>' : ''}
        <button class="btn btn-sm" onclick="this.closest('.event-type-row').remove()" style="color:var(--red);">&times;</button>
    `;
    container.appendChild(row);
}

function getEventTypes() {
    const rows = document.querySelectorAll('.event-type-row');
    const types = [];
    rows.forEach(row => {
        const name = row.querySelector('.et-name').value.trim().toLowerCase();
        const color = row.querySelector('.et-color').value;
        const shortcut = row.querySelector('.et-shortcut').value.trim();
        if (name) types.push({ name, color, shortcut });
    });
    return types;
}

function setEventTypes(eventTypes) {
    const container = document.getElementById('eventTypeRows');
    container.innerHTML = '';
    if (eventTypes && eventTypes.length > 0) {
        eventTypes.forEach(et => addEventTypeRow(et.name || '', et.color || '#ffffff', et.shortcut || ''));
    }
}

// ── Init ──────────────────────────────────────────────────────
setupTagInput('camera_names_tags', 'camera_names_input');
setupTagInput('bodyparts_tags', 'bodyparts_input');

// Setup diagnosis groups tag input with initial input field
const diagnosisGroupsContainer = document.getElementById('diagnosisGroupsInput');
if (diagnosisGroupsContainer) {
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'e.g. Control, MSA, PD, PSP';
    diagnosisGroupsContainer.appendChild(input);
    setupTagInput('diagnosisGroupsInput', input.id || 'diagnosisGroupsInput');
    input.id = 'diagnosisGroupsInput_input';
}

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
            'remote_host', 'remote_python', 'remote_work_dir', 'remote_ssh_key',
        ];
        fields.forEach(f => {
            const el = document.getElementById(f);
            if (el) el.value = settings[f] || '';
        });

        // Ports
        document.getElementById('port').value = settings.port || 8080;
        document.getElementById('remote_ssh_port').value = settings.remote_ssh_port || 22;

        // Select
        document.getElementById('camera_mode').value = settings.camera_mode || 'stereo';
        const netType = document.getElementById('dlc_net_type');
        netType.value = settings.dlc_net_type || 'resnet_50';

        // Checkboxes
        document.getElementById('prefer_deidentified').checked = !!settings.prefer_deidentified;
        document.getElementById('show_tutorials').checked = settings.show_tutorials !== false;

        // Tag inputs
        setTagValues('camera_names_tags', settings.camera_names || []);
        setTagValues('bodyparts_tags', settings.bodyparts || []);
        setTagValues('diagnosisGroupsInput', settings.diagnosis_groups || ["Control", "MSA", "PD", "PSP"]);

        // Calibrations
        setCalibrations(settings.calibrations || {});

        // Event types
        setEventTypes(settings.event_types || [
            { name: 'open',  color: '#00cc44', shortcut: '1' },
            { name: 'peak',  color: '#ffcc00', shortcut: '2' },
            { name: 'close', color: '#ff4444', shortcut: '3' },
            { name: 'pause', color: '#cc66ff', shortcut: '4' },
        ]);

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
function _gatherSettings() {
    return {
        video_dir: document.getElementById('video_dir').value.trim(),
        dlc_dir: document.getElementById('dlc_dir').value.trim(),
        calibration_3d_config: document.getElementById('calibration_3d_config').value.trim(),
        python_executable: document.getElementById('python_executable').value.trim(),
        camera_mode: document.getElementById('camera_mode').value,
        camera_names: getTagValues('camera_names_tags'),
        bodyparts: getTagValues('bodyparts_tags'),
        diagnosis_groups: getTagValues('diagnosisGroupsInput') || ["Control", "MSA", "PD", "PSP"],
        dlc_scorer: document.getElementById('dlc_scorer').value.trim(),
        dlc_date: document.getElementById('dlc_date').value.trim(),
        dlc_net_type: document.getElementById('dlc_net_type').value,
        host: document.getElementById('host').value.trim(),
        port: parseInt(document.getElementById('port').value) || 8080,
        remote_host: document.getElementById('remote_host').value.trim(),
        remote_python: document.getElementById('remote_python').value.trim(),
        remote_work_dir: document.getElementById('remote_work_dir').value.trim(),
        remote_ssh_key: document.getElementById('remote_ssh_key').value.trim(),
        remote_ssh_port: parseInt(document.getElementById('remote_ssh_port').value) || 22,
        calibrations: getCalibrations(),
        prefer_deidentified: document.getElementById('prefer_deidentified').checked,
        show_tutorials: document.getElementById('show_tutorials').checked,
        event_types: getEventTypes(),
    };
}

async function saveSettings() {
    const data = _gatherSettings();

    try {
        await API.put('/api/settings', data);
        const status = await API.get('/api/settings/status');
        renderStatus(status);
        alert('Settings saved.');
    } catch (e) {
        alert('Error saving settings: ' + e.message);
    }
}

async function testRemoteConnection() {
    const btn = document.getElementById('testRemoteBtn');
    const result = document.getElementById('remoteTestResult');

    // Save settings first (silently) so the backend has the latest values
    try {
        await API.put('/api/settings', _gatherSettings());
    } catch (e) {
        result.innerHTML = `<span style="color:var(--red)">Failed to save settings: ${e.message}</span>`;
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Testing...';
    result.innerHTML = '<span style="color:var(--text-muted)">Connecting...</span>';

    try {
        const resp = await API.post('/api/settings/test-remote');
        if (resp.ok) {
            result.innerHTML = `<span style="color:var(--green)">${resp.message}</span>`;
        } else {
            result.innerHTML = `<span style="color:var(--red)">${resp.message}</span>`;
        }
    } catch (e) {
        result.innerHTML = `<span style="color:var(--red)">Error: ${e.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Test Connection';
    }
}

// Load on page ready
loadSettings();
