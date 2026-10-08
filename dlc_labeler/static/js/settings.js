/* Settings page: machine status, data directory, project settings, calibration. */

const settingsPage = (() => {
    const el = (id) => document.getElementById(id);
    let settings = {};

    async function api(url, options) {
        const r = await fetch(url, options);
        if (!r.ok) {
            let detail = r.statusText;
            try { detail = (await r.json()).detail || detail; } catch (e) { /* not JSON */ }
            throw new Error(detail);
        }
        return r.status === 204 ? null : r.json();
    }

    async function loadStatus() {
        const st = await api('/api/settings/status');

        el('issues').innerHTML = (st.issues || [])
            .map(i => `<div class="issue">${i}</div>`).join('');

        const rows = [
            ['Data directory', st.data_dir],
            ['Videos', st.video_dir],
            ['DLC projects', st.dlc_dir],
            ['DeepLabCut', st.dlc_installed
                ? '<span class="ok">installed</span>'
                : '<span class="bad">not installed — labeling only</span>'],
            ['DLC interpreter', st.dlc_python],
            ['GPU', st.local_gpu_available
                ? `<span class="ok">${(st.gpus || []).map(g => g.name).join(', ')}</span>`
                : '<span class="bad">none detected — training runs on CPU</span>'],
            ['CUDA', st.cuda_version || '—'],
            ['Stereo calibration', st.has_calibration
                ? '<span class="ok">configured</span>'
                : '<span class="bad">none — 2D only, distances in pixels</span>'],
        ];
        el('statusGrid').innerHTML = rows.map(([k, v]) =>
            `<span class="k">${k}</span><span class="v">${v}</span>`).join('');
    }

    async function loadDataDir() {
        const d = await api('/api/settings/data-dir');
        el('dataDir').value = d.current || '';
        const sourceText = {
            env: 'set by the DLC_DATA_DIR (or MT_DATA_DIR) environment variable, '
                 + 'which overrides anything saved here',
            bootstrap: `saved in ${d.bootstrap_file}`,
            default: 'defaulting to the folder the app was installed in',
        }[d.source];
        el('dataDirHint').textContent = sourceText || '';
        if (d.source === 'env') {
            el('dataDir').disabled = true;
            el('saveDataDir').disabled = true;
        }
    }

    async function loadSettings() {
        settings = await api('/api/settings');
        el('videoDir').value = settings.video_dir || '';
        el('cameraMode').value = settings.default_camera_mode || 'stereo';
        el('cameraNames').value = (settings.camera_names || []).join(', ');
        el('bodyparts').value = (settings.bodyparts || []).join(', ');
        el('netType').value = settings.dlc_net_type || '';
        el('pythonExe').value = settings.python_executable || '';
        el('preferDeident').checked = !!settings.prefer_deidentified;
        renderCalibrations();
    }

    function renderCalibrations() {
        const cals = settings.calibrations || {};
        const names = Object.keys(cals);
        el('calibList').innerHTML = names.length
            ? `<div class="status-grid">${names.map(n =>
                `<span class="k">${n}</span><span class="v">${cals[n]}
                 <button class="btn btn-sm" data-del-calib="${n}"
                         style="margin-left:6px;">remove</button></span>`).join('')}</div>`
            : '<div style="font-size:12px;color:var(--text-muted);">None configured.</div>';

        el('calibList').querySelectorAll('[data-del-calib]').forEach(b =>
            b.addEventListener('click', async () => {
                const next = { ...(settings.calibrations || {}) };
                delete next[b.dataset.delCalib];
                await save({ calibrations: next });
            }));
    }

    async function save(patch) {
        el('settingsStatus').textContent = 'Saving…';
        try {
            settings = await api('/api/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patch),
            });
            el('settingsStatus').textContent = 'Saved';
            setTimeout(() => { el('settingsStatus').textContent = ''; }, 2500);
            renderCalibrations();
            loadStatus();
        } catch (e) {
            el('settingsStatus').textContent = `Failed: ${e.message}`;
        }
    }

    function splitList(value) {
        return value.split(',').map(s => s.trim()).filter(Boolean);
    }

    async function saveAll() {
        const names = splitList(el('cameraNames').value);
        const bps = splitList(el('bodyparts').value);
        if (!names.length) {
            el('settingsStatus').textContent = 'At least one camera name is required';
            return;
        }
        if (!bps.length) {
            el('settingsStatus').textContent = 'At least one bodypart is required';
            return;
        }
        // Changing these after frames are committed would mean the stored
        // labels and the DLC CollectedData files disagree about what each
        // column is, so make the consequence explicit before saving.
        const changedShape =
            JSON.stringify(names) !== JSON.stringify(settings.camera_names)
            || JSON.stringify(bps) !== JSON.stringify(settings.bodyparts);
        if (changedShape && !confirm(
                'Camera names and bodyparts define the shape of every stored '
                + 'label. Changing them after frames are committed will not '
                + 'rewrite existing labels or DLC files.\n\nContinue?')) {
            return;
        }

        await save({
            video_dir: el('videoDir').value.trim(),
            default_camera_mode: el('cameraMode').value,
            camera_names: names,
            bodyparts: bps,
            dlc_net_type: el('netType').value.trim(),
            python_executable: el('pythonExe').value.trim(),
            prefer_deidentified: el('preferDeident').checked,
        });
    }

    async function checkCalib() {
        const path = el('calibPath').value.trim();
        if (!path) return;
        el('calibCheck').textContent = 'Checking…';
        try {
            const r = await api('/api/settings/validate-calibration', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path }),
            });
            el('calibCheck').innerHTML = r.valid
                ? '<span class="ok">Valid — K1 matrix found</span>'
                : `<span class="bad">${r.error}</span>`;
        } catch (e) {
            el('calibCheck').innerHTML = `<span class="bad">${e.message}</span>`;
        }
    }

    async function addCalib() {
        const name = el('calibName').value.trim();
        const path = el('calibPath').value.trim();
        if (!name || !path) {
            el('calibCheck').textContent = 'Both a name and a path are needed.';
            return;
        }
        await save({ calibrations: { ...(settings.calibrations || {}), [name]: path } });
        el('calibName').value = '';
        el('calibPath').value = '';
        el('calibCheck').textContent = '';
    }

    async function saveDataDir() {
        const path = el('dataDir').value.trim();
        if (!path) return;
        if (!confirm(`Switch the data directory to:\n${path}\n\n`
                   + 'The app will restart. Nothing in the current directory '
                   + 'is moved or deleted.')) return;
        el('dataDirStatus').textContent = 'Saving and restarting…';
        try {
            await api('/api/settings/data-dir', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path }),
            });
            // The server re-execs, so give it a moment and then reload.
            setTimeout(() => location.reload(), 3000);
        } catch (e) {
            el('dataDirStatus').textContent = `Failed: ${e.message}`;
        }
    }

    document.addEventListener('DOMContentLoaded', async () => {
        el('saveSettings').addEventListener('click', saveAll);
        el('saveDataDir').addEventListener('click', saveDataDir);
        el('checkCalib').addEventListener('click', checkCalib);
        el('addCalib').addEventListener('click', addCalib);
        try {
            await loadSettings();
            await loadDataDir();
            await loadStatus();
        } catch (e) {
            el('issues').innerHTML = `<div class="issue">Could not load settings: ${e.message}</div>`;
        }
    });

    return { save };
})();
