/* Dashboard: subject table, job indicators, detail panel */

let subjects = [];
let activeJobs = [];
let appStatus = { configured: true, has_calibration: false };

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

// ── Load subjects ─────────────────────────────────────
async function loadSubjects() {
    try {
        subjects = await API.get('/api/subjects');
        renderTable();
        populateStageFilter();
        loadJobs();
    } catch (e) {
        document.getElementById('subjectTableBody').innerHTML =
            `<tr><td colspan="6" style="text-align:center;color:var(--red)">${e.message}</td></tr>`;
    }
}

function populateStageFilter() {
    const stages = [...new Set(subjects.map(s => s.stage))].sort();
    const sel = document.getElementById('stageFilter');
    // Keep first option
    sel.innerHTML = '<option value="">All stages</option>';
    stages.forEach(st => {
        sel.innerHTML += `<option value="${st}">${st}</option>`;
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
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">No subjects found</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(s => `
        <tr>
            <td class="name-col" onclick="showDetail(${s.id})">${s.name}</td>
            <td><span class="badge badge-${s.stage}">${s.stage.replace(/_/g, ' ')}</span></td>
            <td>${s.video_count}</td>
            <td>${s.has_labels ? '&check;' : '&mdash;'}</td>
            <td>${s.has_snapshots ? '&check;' : '&mdash;'}</td>
            <td>
                ${getActions(s)}
            </td>
        </tr>
    `).join('');
}

function getActions(subject) {
    const btns = [];
    const s = subject.stage;

    // Always show label button
    btns.push(`<button class="btn btn-sm" onclick="openLabeling(${subject.id})">Label</button>`);

    // Stage-specific actions
    if (s === 'videos_linked' || s === 'created') {
        btns.push(`<button class="btn btn-sm btn-primary" onclick="runStep(${subject.id}, 'mediapipe')">Run MediaPipe</button>`);
    }
    if (s === 'prelabeled') {
        btns.push(`<button class="btn btn-sm" onclick="runStep(${subject.id}, 'mediapipe')">Re-run MP</button>`);
    }
    if (s === 'committed' || s === 'labeled') {
        btns.push(`<button class="btn btn-sm" onclick="runStep(${subject.id}, 'create_training_dataset')">Create Dataset</button>`);
    }
    if (s === 'training_dataset_created') {
        btns.push(`<button class="btn btn-sm btn-primary" onclick="runStep(${subject.id}, 'train')">Train</button>`);
    }
    if (s === 'trained') {
        btns.push(`<button class="btn btn-sm" onclick="runStep(${subject.id}, 'crop')">Crop Videos</button>`);
    }
    if (s === 'cropped') {
        btns.push(`<button class="btn btn-sm" onclick="runStep(${subject.id}, 'analyze')">Analyze</button>`);
    }
    if (s === 'analyzed' && appStatus.has_calibration) {
        btns.push(`<button class="btn btn-sm" onclick="runStep(${subject.id}, 'triangulate')">Triangulate</button>`);
    }

    return btns.join(' ');
}

// ── Detail panel ─────────────────────────────────────
async function showDetail(subjectId) {
    try {
        const detail = await API.get(`/api/subjects/${subjectId}`);
        const panel = document.getElementById('detailPanel');
        document.getElementById('detailName').textContent = detail.name;

        const triangulateBtn = appStatus.has_calibration
            ? `<button class="btn btn-sm" onclick="runStep(${detail.id}, 'triangulate')">Triangulate</button>`
            : '';

        document.getElementById('detailContent').innerHTML = `
            <div class="detail-section">
                <h3>Info</h3>
                <div class="info-row"><span class="label">Stage</span><span class="badge badge-${detail.stage}">${detail.stage.replace(/_/g, ' ')}</span></div>
                <div class="info-row"><span class="label">Iteration</span><span>${detail.iteration}</span></div>
                <div class="info-row"><span class="label">Videos</span><span>${detail.video_count}</span></div>
                <div class="info-row"><span class="label">DLC Dir</span><span style="font-size:11px;word-break:break-all">${detail.dlc_dir || 'N/A'}</span></div>
            </div>
            <div class="detail-section">
                <h3>Trials</h3>
                ${detail.trials.length > 0
                    ? detail.trials.map(t => `<div class="info-row"><span>${t}</span></div>`).join('')
                    : '<div class="info-row"><span class="label">No videos found</span></div>'
                }
            </div>
            <div class="detail-section" style="grid-column: 1 / -1;">
                <h3>Pipeline Steps</h3>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'mediapipe')">Run MediaPipe</button>
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'create_training_dataset')">Create Dataset</button>
                    <button class="btn btn-sm btn-primary" onclick="runStep(${detail.id}, 'train')">Train</button>
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'crop')">Crop Videos</button>
                    <button class="btn btn-sm" onclick="runStep(${detail.id}, 'analyze')">Analyze</button>
                    ${triangulateBtn}
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
    const logBtn = (data.status === 'failed')
        ? `<button class="btn btn-sm" onclick="viewJobLog(${jobId})">View Log</button>`
        : '';

    el.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="job-indicator job-${data.status}"></span>
            <span>Job #${jobId}</span>
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

function closeLogModal() {
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

// ── Add subject ──────────────────────────────────────
function showAddModal() {
    document.getElementById('addModal').classList.add('active');
    document.getElementById('newSubjectName').focus();
}
function hideAddModal() {
    document.getElementById('addModal').classList.remove('active');
}

async function addSubject() {
    const name = document.getElementById('newSubjectName').value.trim();
    if (!name) return alert('Enter a subject name');

    try {
        await API.post('/api/subjects', { name });
        hideAddModal();
        document.getElementById('newSubjectName').value = '';
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function syncSubjects() {
    try {
        const result = await API.post('/api/subjects/sync');
        alert(`Sync complete: ${result.created} new, ${result.updated} updated, ${result.total} total`);
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Labeling navigation ──────────────────────────────
async function openLabeling(subjectId) {
    try {
        // Create a new session
        const session = await API.post(`/api/labeling/${subjectId}/sessions`, { session_type: 'initial' });
        window.location.href = `/labeling?session=${session.id}`;
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Event listeners ──────────────────────────────────
document.getElementById('filterInput').addEventListener('input', renderTable);
document.getElementById('stageFilter').addEventListener('change', renderTable);

// Escape closes modal
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideAddModal();
});

// ── Init ─────────────────────────────────────────────
checkStatus().then(() => loadSubjects());
