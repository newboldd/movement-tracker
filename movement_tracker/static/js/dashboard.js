/* Dashboard: subject table, job indicators, detail panel, diagnosis grouping */

let subjects = [];
let activeJobs = [];
let appStatus = { configured: true, has_calibration: false };
let calibrationNames = [];
let diagnosisGroups = ["Control", "MSA", "PD", "PSP"];
// ── Check settings status ────────────────────────────────
async function checkStatus() {
    try {
        appStatus = await API.get('/api/settings/status');
        const banner = document.getElementById('welcomeBanner');
        if (!appStatus.configured) {
            banner.style.display = 'block';
        } else {
            banner.style.display = 'none';
        }
    } catch (e) { /* ignore */ }
}

// ── Load diagnosis groups from settings ───────────────
async function loadDiagnosisGroups() {
    try {
        const settings = await API.get('/api/settings');
        diagnosisGroups = settings.diagnosis_groups || ["Control", "MSA", "PD", "PSP"];
    } catch (e) {
        diagnosisGroups = ["Control", "MSA", "PD", "PSP"];
    }
}

// ── Load subjects ─────────────────────────────────────
async function loadSubjects() {
    try {
        subjects = await API.get('/api/subjects');
        await loadDiagnosisGroups();
        renderDiagnosisGroups();
        populateStageFilter();
        loadJobs();
    } catch (e) {
        document.getElementById('subjectTableBody').innerHTML =
            `<tr><td colspan="9" style="text-align:center;color:var(--red)">${e.message}</td></tr>`;
    }
}

// ── Render subjects grouped by diagnosis ──────────────
function renderDiagnosisGroups() {
    const container = document.getElementById('subjectTableBody');
    const filtered = getFilteredSubjects();

    document.getElementById('subjectCount').innerHTML =
        `Showing <strong>${filtered.length}</strong> of ${subjects.length} subjects`;

    if (filtered.length === 0) {
        container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px;">No subjects found</div>';
        return;
    }

    // Group by diagnosis
    const grouped = {};
    diagnosisGroups.forEach(dg => {
        grouped[dg] = [];
    });

    filtered.forEach(s => {
        const diagnosis = s.diagnosis || "Control";
        if (!grouped[diagnosis]) {
            grouped[diagnosis] = [];
        }
        grouped[diagnosis].push(s);
    });

    let html = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; width: 100%;">';

    diagnosisGroups.forEach(diagnosis => {
        const groupSubjects = grouped[diagnosis] || [];
        html += `
            <div style="background: var(--bg-card); border-radius: var(--radius); padding: 12px; border: 1px solid var(--border);">
                <h3 style="margin: 0 0 12px 0; font-size: 14px; color: var(--text); display: flex; justify-content: space-between; align-items: center;">${diagnosis} <span style="font-weight: 400; font-size: 12px; color: var(--text-muted);">n=${groupSubjects.length}</span></h3>
                <div style="display: flex; flex-direction: column; gap: 6px; max-height: 600px; overflow-y: auto;">
        `;

        if (groupSubjects.length === 0) {
            html += '<span style="font-size: 12px; color: var(--text-muted); text-align: center;">No subjects</span>';
        } else {
            groupSubjects.forEach(s => {
                const stageColor = `badge-${s.stage}`;
                html += `
                    <div style="padding: 8px; background: var(--bg); border-radius: 4px; border: 1px solid var(--border);">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                            <span style="font-weight:600;font-size:13px;">${s.name}</span>
                            <span class="badge ${stageColor}" style="font-size:11px;">${stageLabel(s.stage)}</span>
                        </div>
                        <div style="display:flex;gap:4px;flex-wrap:wrap;">
                            ${s.video_count > 0 ? `<button class="btn btn-sm" style="white-space:nowrap;" onclick="window.location.href='/videos?subject=${s.id}'">Video</button>` : ''}
                            <button class="btn btn-sm" style="white-space:nowrap;" onclick="openLabeling(${s.id})">DLC</button>
                            <button class="btn btn-sm" style="white-space:nowrap;" onclick="sessionStorage.setItem('dlc_lastSubjectId','${s.id}');window.location.href='/results?subject=${s.id}&from=dashboard'">Results</button>
                            <button class="btn btn-sm" style="white-space:nowrap;" onclick="showDetail(${s.id})">Info</button>
                        </div>
                    </div>
                `;
            });
        }

        html += '</div></div>';
    });

    html += '</div>';
    container.innerHTML = html;
}

// ── Stage label shortening ────────────────────────────
function stageLabel(stage) {
    const overrides = { events_partial: 'events', events_complete: 'complete' };
    return (overrides[stage] || stage).replace(/_/g, ' ');
}

// Canonical stage order (matches pipeline progression)
const STAGE_ORDER = [
    "created", "videos_linked", "prelabeled", "labeling", "labeled",
    "committed", "training", "training_dataset_created", "trained",
    "cropping", "cropped", "analyzing", "analyzed",
    "triangulating", "triangulated", "refined", "corrected",
    "retraining", "retrained",
    "events_partial", "events_complete", "complete",
];

function populateStageFilter() {
    const present = new Set(subjects.map(s => s.stage));
    const sel = document.getElementById('stageFilter');
    sel.innerHTML = '<option value="">All stages</option>';
    // Add stages in pipeline order
    STAGE_ORDER.forEach(st => {
        if (present.has(st)) {
            sel.innerHTML += `<option value="${st}">${stageLabel(st)}</option>`;
        }
    });
    // Any stages not in the canonical list (e.g. error states) go at the end
    [...present].filter(st => !STAGE_ORDER.includes(st)).sort().forEach(st => {
        sel.innerHTML += `<option value="${st}">${stageLabel(st)}</option>`;
    });
}

function getFilteredSubjects() {
    const text = document.getElementById('filterInput').value.toLowerCase();
    const stage = document.getElementById('stageFilter').value;
    return subjects.filter(s => {
        if (text && !s.name.toLowerCase().includes(text)) return false;
        if (stage && s.stage !== stage) return false;
        return true;
    });
}

function renderTable() {
    const filtered = getFilteredSubjects();
    document.getElementById('subjectCount').innerHTML =
        `Showing <strong>${filtered.length}</strong> of ${subjects.length} subjects`;

    const tbody = document.getElementById('subjectTableBody');
    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--text-muted)">No subjects found</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(s => `
        <tr>
            <td class="name-col" onclick="showDetail(${s.id})">${s.name}</td>
            <td><button class="btn btn-sm" onclick="openFinal(${s.id})">View</button></td>
            <td><span class="badge badge-${s.stage}">${s.stage.replace(/_/g, ' ')}</span></td>
            <td>${getActions(s)}</td>
            <td>${s.video_count}</td>
            <td>${s.has_mediapipe ? '&check;' : '&mdash;'}</td>
            <td>${s.has_blur ? '&check;' : '&mdash;'}</td>
            <td>${s.has_labels ? '&check;' : '&mdash;'}</td>
            <td>${s.has_snapshots ? '&check;' : '&mdash;'}</td>
        </tr>
    `).join('');
}

function getActions(subject) {
    const btns = [];
    const s = subject.stage;

    // Label / Restart button (always visible)
    btns.push(`<button class="btn btn-sm" onclick="openLabeling(${subject.id})">Label</button>`);

    // Stage-specific actions
    if (s === 'committed' || s === 'labeled') {
        btns.push(`<button class="btn btn-sm btn-primary" onclick="runStep(${subject.id}, 'train')">Train</button>`);
    }
    // Refine available for analyzed and beyond
    const refineStages = ['analyzed', 'refined', 'corrected'];
    if (refineStages.includes(s)) {
        btns.push(`<button class="btn btn-sm" onclick="openRefine(${subject.id})">Refine</button>`);
        btns.push(`<button class="btn btn-sm" onclick="openCorrections(${subject.id})">Corrections</button>`);
    }

    return btns.join(' ');
}

// ── Calibration helpers ──────────────────────────────
async function loadCalibrationNames() {
    try {
        const settings = await API.get('/api/settings');
        calibrationNames = Object.keys(settings.calibrations || {});
    } catch (e) { calibrationNames = []; }
}

async function updateSubjectCamera(subjectId, cameraName) {
    try {
        await API.patch(`/api/subjects/${subjectId}`, { camera_name: cameraName });
    } catch (e) {
        alert('Error updating camera: ' + e.message);
    }
}

async function updateSubjectDiagnosis(subjectId, diagnosis) {
    try {
        await API.patch(`/api/subjects/${subjectId}`, { diagnosis: diagnosis });
        // Refresh the subjects list to update the grouped view
        await loadSubjects();
    } catch (e) {
        alert('Error updating diagnosis: ' + e.message);
    }
}

async function onDiagnosisSelectChange(subjectId, sel) {
    if (sel.value !== '__new__') {
        updateSubjectDiagnosis(subjectId, sel.value);
        return;
    }
    const name = prompt('Enter new group name:');
    if (!name || !name.trim()) {
        sel.value = sel.dataset.prev || 'Control';
        return;
    }
    const trimmed = name.trim();
    if (!diagnosisGroups.includes(trimmed)) {
        diagnosisGroups.push(trimmed);
        try {
            const cfg = await API.get('/api/settings');
            cfg.diagnosis_groups = diagnosisGroups;
            await API.put('/api/settings', cfg);
        } catch { /* best-effort */ }
    }
    updateSubjectDiagnosis(subjectId, trimmed);
}

async function updateSubjectCameraMode(subjectId, cameraMode) {
    try {
        await API.patch(`/api/subjects/${subjectId}`, { camera_mode: cameraMode });
    } catch (e) {
        alert('Error updating camera mode: ' + e.message);
    }
}

async function updateNoFaceVideos(subjectId) {
    const checkboxes = document.querySelectorAll('.no-face-cb');
    const noFaceVideos = [];
    checkboxes.forEach(cb => {
        if (!cb.checked) noFaceVideos.push(cb.dataset.stem);
    });
    try {
        await API.patch(`/api/subjects/${subjectId}`, { no_face_videos: noFaceVideos });
    } catch (e) {
        alert('Error updating face status: ' + e.message);
    }
}

// ── Detail panel ─────────────────────────────────────
async function showDetail(subjectId) {
    try {
        const detail = await API.get(`/api/subjects/${subjectId}`);
        const panel = document.getElementById('detailPanel');
        document.getElementById('detailName').textContent = detail.name;

        document.getElementById('detailContent').innerHTML = `
            <div class="detail-section">
                <h3>Info</h3>
                <div class="info-row"><span class="label">Stage</span><span class="badge badge-${detail.stage}">${detail.stage.replace(/_/g, ' ')}</span></div>
                <div class="info-row"><span class="label">Iteration</span><span>${detail.iteration}</span></div>
                <div class="info-row"><span class="label">Diagnosis</span><span>
                    <select id="diagnosisSelect" onchange="onDiagnosisSelectChange(${detail.id}, this)" style="padding:2px 6px;font-size:13px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);">
                        ${diagnosisGroups.map(dg => `<option value="${dg}"${dg === (detail.diagnosis || 'Control') ? ' selected' : ''}>${dg}</option>`).join('')}
                        <option value="__new__">+ New Group…</option>
                    </select>
                </span></div>
                <div class="info-row"><span class="label">Video Type</span><span>
                    <select onchange="updateSubjectCameraMode(${detail.id}, this.value)" style="padding:2px 6px;font-size:13px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);">
                        <option value="single"${(detail.camera_mode || 'stereo') === 'single' ? ' selected' : ''}>Single Camera</option>
                        <option value="stereo"${(detail.camera_mode || 'stereo') === 'stereo' ? ' selected' : ''}>Stereo</option>
                        <option value="multicam"${(detail.camera_mode || 'stereo') === 'multicam' ? ' selected' : ''}>Multi-Camera</option>
                    </select>
                </span></div>
                <div class="info-row"><span class="label">DLC Dir</span><span style="font-size:11px;word-break:break-all">${detail.dlc_dir || 'N/A'}</span></div>
                <div class="info-row"><span class="label">Camera</span><span>
                    ${detail.camera_name
                        ? `<select id="cameraSelect" onchange="updateSubjectCamera(${detail.id}, this.value)" style="padding:2px 6px;font-size:13px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);">
                            <option value="">None</option>
                            ${calibrationNames.map(n => `<option value="${n}"${n === detail.camera_name ? ' selected' : ''}>${n}</option>`).join('')}
                          </select>`
                        : `<span style="background:var(--red);color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">None</span>
                           <select id="cameraSelect" onchange="updateSubjectCamera(${detail.id}, this.value)" style="margin-left:6px;padding:2px 6px;font-size:13px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);">
                            <option value="">-- Assign --</option>
                            ${calibrationNames.map(n => `<option value="${n}">${n}</option>`).join('')}
                          </select>`
                    }
                </span></div>
            </div>
            <div class="detail-section">
                <h3>Trials</h3>
                ${detail.trials.length > 0
                    ? detail.trials.map(t => `<div class="info-row"><span>${t}</span></div>`).join('')
                    : '<div class="info-row"><span class="label">No videos found</span></div>'
                }
            </div>
            <div class="detail-section" style="grid-column: 1 / -1;">
                <h3>Actions</h3>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">
                    <a class="btn btn-sm" href="/onboarding?subject=${detail.id}" style="text-decoration:none;">Edit Subject</a>
                </div>
            </div>
            <div class="detail-section" style="grid-column: 1 / -1;">
                <h3>Pipeline Steps</h3>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'mediapipe')">Run MediaPipe</button>
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'deidentify')">Blur Faces</button>
                    <button class="btn btn-sm btn-primary" onclick="runStep(${detail.id}, 'train')">Train</button>
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'analyze_v1')">Analyze v1</button>
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'analyze_v2')">Analyze v2</button>
                    <button class="btn btn-sm btn-danger" onclick="removeSubject(${detail.id}, '${detail.name}')" style="margin-left:auto;">Remove</button>
                </div>
            </div>
            ${detail.jobs.length > 0 ? `
            <div class="detail-section" style="grid-column: 1 / -1;">
                <h3>Recent Jobs</h3>
                ${detail.jobs.slice(0, 5).map(j => `
                    <div class="info-row">
                        <span><span class="job-indicator job-${j.status}"></span>${j.job_type}</span>
                        <span>${j.status} ${j.progress_pct ? `(${j.progress_pct.toFixed(0)}%)` : ''}</span>
                    </div>
                `).join('')}
            </div>` : ''}
        `;

        panel.classList.add('active');
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
        alert('Error loading subject: ' + e.message);
    }
}

function closeDetail() {
    document.getElementById('detailPanel').classList.remove('active');
}

// ── Pipeline actions ─────────────────────────────────
async function runStep(subjectId, step) {
    try {
        const result = await API.post(`/api/subjects/${subjectId}/run-step`, { step });
        if (result.job_id) {
            // Start tracking job
            trackJob(result.job_id);
        }
        loadSubjects();
        showDetail(subjectId);
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

function trackJob(jobId) {
    const jobsCard = document.getElementById('jobsCard');
    jobsCard.style.display = 'block';

    API.streamJob(jobId,
        (data) => {
            updateJobDisplay(jobId, data);
        },
        (data) => {
            loadSubjects();
        }
    );
}

function updateJobDisplay(jobId, data) {
    const list = document.getElementById('jobsList');
    let el = document.getElementById(`job-${jobId}`);
    if (!el) {
        el = document.createElement('div');
        el.id = `job-${jobId}`;
        el.style.marginBottom = '8px';
        list.appendChild(el);
    }

    const remoteBadge = data.remote_host
        ? '<span class="badge" style="background:rgba(33,150,243,0.15);color:#2196f3;font-size:11px;">Remote</span>'
        : '';
    const errorMsg = (data.status === 'failed' && data.error_msg)
        ? `<div style="color:var(--red);font-size:12px;margin-top:4px;">${data.error_msg}</div>`
        : '';
    const logBtn = (data.status === 'running')
        ? `<button class="btn btn-sm" onclick="viewJobLogLive(${jobId})">View Log</button>`
        : (data.status === 'failed')
            ? `<button class="btn btn-sm" onclick="viewJobLog(${jobId})">View Log</button>`
            : '';

    el.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="job-indicator job-${data.status}"></span>
            <span style="font-weight:600;">${data.subject_name || '?'}</span>
            <span style="color:var(--text-muted);">${(data.job_type || '').replace(/_/g, ' ')}</span>
            ${remoteBadge}
            <div class="progress-bar" style="flex:1;max-width:200px;">
                <div class="fill" style="width:${data.progress_pct || 0}%"></div>
            </div>
            <span>${(data.progress_pct || 0).toFixed(0)}%</span>
            <span class="badge badge-${data.status}">${data.status}</span>
            ${data.status === 'running' ? `<button class="btn btn-sm btn-danger" onclick="cancelJob(${jobId})">Cancel</button>` : ''}
            ${logBtn}
        </div>
        ${errorMsg}
    `;
}

async function cancelJob(jobId) {
    try {
        await API.post(`/api/jobs/${jobId}/cancel`);
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

let _logStream = null;  // Active log SSE connection

async function viewJobLog(jobId) {
    const modal = document.getElementById('logModal');
    const pre = document.getElementById('logContent');
    pre.textContent = 'Loading...';
    modal.classList.add('active');
    try {
        const job = await API.get(`/api/jobs/${jobId}`);
        pre.textContent = (job.log_tail || []).join('') || 'No log output.';
        pre.scrollTop = pre.scrollHeight;
    } catch (e) {
        pre.textContent = 'Error fetching log: ' + e.message;
    }
}

function viewJobLogLive(jobId) {
    const modal = document.getElementById('logModal');
    const pre = document.getElementById('logContent');
    pre.textContent = '';
    modal.classList.add('active');

    _logStream = API.streamJobLog(jobId,
        (text) => {
            pre.textContent += text;
            pre.scrollTop = pre.scrollHeight;
        },
        (data) => {
            _logStream = null;
            if (data.status === 'failed') {
                pre.textContent += '\n\n--- Job failed ---\n';
            } else if (data.status === 'completed') {
                pre.textContent += '\n\n--- Job completed ---\n';
            }
            pre.scrollTop = pre.scrollHeight;
        }
    );
}

function closeLogModal() {
    if (_logStream) {
        _logStream.close();
        _logStream = null;
    }
    document.getElementById('logModal').classList.remove('active');
}

// ── Jobs polling ─────────────────────────────────────
async function loadJobs() {
    try {
        const jobs = await API.get('/api/jobs?status=running');
        if (jobs.length > 0) {
            document.getElementById('jobsCard').style.display = 'block';
            jobs.forEach(j => trackJob(j.id));
        }
    } catch (e) { /* ignore */ }
}

async function syncSubjects() {
    try {
        const result = await API.post('/api/subjects/sync');
        const parts = [`${result.created} new`, `${result.updated} updated`];
        if (result.removed) parts.push(`${result.removed} removed`);
        parts.push(`${result.total} total`);
        alert(`Sync complete: ${parts.join(', ')}`);
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function autoAssignDiagnosis() {
    if (!confirm('Auto-assign diagnosis groups to all subjects based on their names?\n\nThis will use name patterns (e.g., "Con" → Control, "MSA" → MSA, etc.) to assign subjects to groups.')) {
        return;
    }
    try {
        const result = await API.post('/api/subjects/auto-assign-diagnosis');
        alert(`Auto-assignment complete!\n\nUpdated: ${result.updated} subjects\nTotal subjects: ${result.total_subjects}`);
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function removeSubject(subjectId, subjectName) {
    if (!confirm(`Remove subject "${subjectName}"?\n\nThis will permanently delete:\n• All trial and deidentified videos\n• DLC directory (models, labels, config)\n• All database records\n\nThis cannot be undone.`)) {
        return;
    }
    try {
        const result = await API.del(`/api/subjects/${subjectId}`);
        alert(result.message);
        closeDetail();
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Refine navigation ────────────────────────────────
function openRefine(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    window.location.href = `/labeling-select?subject=${subjectId}&mode=refine`;
}

// ── Final (read-only) navigation ─────────────────────
function openFinal(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    window.location.href = `/labeling-select?subject=${subjectId}&mode=final`;
}

// ── Corrections navigation ───────────────────────────
function openCorrections(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    window.location.href = `/labeling-select?subject=${subjectId}&mode=corrections`;
}

// ── Labeling navigation ──────────────────────────────
function openLabeling(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    // Default to events mode; server _smart_mode handles subjects that aren't ready
    window.location.href = `/labeling-select?subject=${subjectId}`;
}

// ── Batch preprocessing ─────────────────────────────
function showBatchPreprocess() {
    document.getElementById('batchModal').classList.add('active');
}

function closeBatchModal() {
    document.getElementById('batchModal').classList.remove('active');
}

async function startBatchPreprocess() {
    const steps = [];
    if (document.getElementById('batchStepMp').checked) steps.push('mediapipe');
    if (document.getElementById('batchStepBlur').checked) steps.push('blur');

    if (steps.length === 0) {
        alert('Select at least one step');
        return;
    }

    closeBatchModal();

    try {
        const result = await API.post('/api/batch/preprocess', { steps, subjects: [] });
        if (result.job_id) {
            trackJob(result.job_id);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Event listeners ──────────────────────────────────
document.getElementById('filterInput').addEventListener('input', renderDiagnosisGroups);
document.getElementById('stageFilter').addEventListener('change', renderDiagnosisGroups);

// ── Videos panel ─────────────────────────────────────
async function loadVideosPanel() {
    const panel = document.getElementById('videosPanel');
    try {
        const data = await API.get('/api/video-tools/sources');
        const videos = data.videos || [];

        let html = '';

        if (videos.length === 0) {
            html += '<span style="color:var(--text-muted);font-size:13px;">No videos yet</span>';
        } else {
            videos.forEach(v => {
                const icon = v.source === 'sample' ? '📦' : '🎬';
                const label = v.source === 'sample' ? `${v.name} (sample)` : v.name;
                html += `<a href="/videos" class="btn btn-sm" onclick="sessionStorage.setItem('browse_video_path','${v.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}')" style="text-decoration:none;display:inline-flex;align-items:center;gap:4px;">${icon} ${label}</a>`;
            });
        }

        // Browse button always at the end
        html += `<a href="/videos" class="btn btn-sm btn-primary" style="text-decoration:none;">Browse…</a>`;

        panel.innerHTML = html;
    } catch (e) {
        panel.innerHTML = '<a href="/videos" class="btn btn-sm btn-primary" style="text-decoration:none;">Browse Videos</a>';
    }
}

// ── Init ─────────────────────────────────────────────
checkStatus().then(async () => {
    await loadCalibrationNames();
    loadSubjects();
    loadVideosPanel();
});
