/* Processing Jobs: queue management, job launching, monitoring with local/remote/GPU support */

let subjects = [];
let steps = [];
let queueStream = null;
let _logStream = null;
let gpuAvailable = false;
let availableGpus = [];
let _lastQueueState = null;

// ── Job type → page URL mapping ──────────────────────────
function jobTypeUrl(jobType, subjectIds) {
    // Parse first subject ID from the JSON array string
    let sid = null;
    try {
        const ids = typeof subjectIds === 'string' ? JSON.parse(subjectIds) : subjectIds;
        sid = Array.isArray(ids) ? ids[0] : ids;
    } catch (e) {}
    const q = sid ? `?subject=${sid}` : '';
    const typeMap = {
        'mediapipe': '/analyze',
        'vision': '/analyze',
        'pose': '/analyze',
        'deidentify': '/deidentify',
        'blur': '/deidentify',
        'mediapipe+blur': '/deidentify',
        'train': '/labeling-select',
        'refine': '/labeling-select',
        'analyze_v1': '/results',
        'analyze_v2': '/results',
    };
    const base = typeMap[jobType] || '/subjects';
    return base + q;
}

function jobTypeLink(jobType, subjectIds) {
    const url = jobTypeUrl(jobType, subjectIds);
    return `<a href="${url}" style="text-decoration:none;color:inherit;font-weight:600;" onclick="sessionStorage.setItem('lastSubjectId',JSON.stringify(${JSON.stringify(subjectIds)}).replace(/[\\[\\]]/g,''))">${jobType}</a>`;
}

// ── Time helpers ────────────────────────────────────────
function formatDuration(ms) {
    const secs = Math.floor(ms / 1000);
    if (secs < 60) return `${secs}s`;
    const mins = Math.floor(secs / 60);
    const remSecs = secs % 60;
    if (mins < 60) return `${mins}m ${remSecs}s`;
    const hrs = Math.floor(mins / 60);
    const remMins = mins % 60;
    return `${hrs}h ${remMins}m`;
}

function formatJobTime(startedAt, pct, epochInfo) {
    if (!startedAt) return { display: '', tooltip: '' };
    // Parse UTC timestamp from server
    const started = new Date(startedAt.endsWith('Z') ? startedAt : startedAt + 'Z');
    const now = new Date();
    const elapsedMs = now - started;
    const elapsed = formatDuration(elapsedMs);

    // Epoch-based estimation for training/refine jobs
    if (epochInfo) {
        let info;
        try { info = typeof epochInfo === 'string' ? JSON.parse(epochInfo) : epochInfo; } catch { info = null; }
        if (info && info.epoch >= 1 && info.first_epoch_at && info.total > 0) {
            const firstEpoch = new Date(info.first_epoch_at.endsWith('Z') ? info.first_epoch_at : info.first_epoch_at + 'Z');
            const trainingElapsed = now - firstEpoch;
            const avgPerEpoch = trainingElapsed / info.epoch;
            const remainingEpochs = info.total - info.epoch;
            // Estimate analyze step as ~15% of total training time
            const analyzeEstimate = info.total * avgPerEpoch * 0.15;
            const remainMs = (remainingEpochs * avgPerEpoch) + analyzeEstimate;
            const eta = formatDuration(Math.max(0, remainMs));
            return {
                display: `${elapsed} · ~${eta} left`,
                tooltip: `Started: ${started.toLocaleString()}\nElapsed: ${elapsed}\nEpoch: ${info.epoch}/${info.total}\nEstimated remaining: ${eta}`,
            };
        }
        // Have epoch_info but epoch < 1: no estimate yet
        return {
            display: elapsed,
            tooltip: `Started: ${started.toLocaleString()}\nElapsed: ${elapsed}\nWaiting for first epoch...`,
        };
    }

    // Fallback: generic pct-based estimation for non-training jobs
    if (pct > 2 && pct < 100) {
        const totalEstMs = (elapsedMs / pct) * 100;
        const remainMs = totalEstMs - elapsedMs;
        const eta = formatDuration(Math.max(0, remainMs));
        return {
            display: `${elapsed} · ~${eta} left`,
            tooltip: `Started: ${started.toLocaleString()}\nElapsed: ${elapsed}\nEstimated remaining: ${eta}`,
        };
    }
    return {
        display: elapsed,
        tooltip: `Started: ${started.toLocaleString()}\nElapsed: ${elapsed}`,
    };
}

// ── Detect GPU availability ─────────────────────────────
async function loadGpuStatus() {
    try {
        const status = await API.get('/api/settings/status');
        gpuAvailable = status.local_gpu_available || false;
        availableGpus = status.gpus || [];

        // Populate GPU selector
        const gpuSel = document.getElementById('gpuSelector');
        gpuSel.innerHTML = availableGpus.map(g =>
            `<option value="${g.index}">${g.name} (${Math.round(g.memory_mb / 1024)}GB)</option>`
        ).join('');

        // Disable Local GPU option if no GPU available
        const gpuRadio = document.getElementById('targetLocalGpu');
        gpuRadio.disabled = !gpuAvailable;
        if (!gpuAvailable) {
            gpuRadio.parentElement.style.opacity = '0.5';
            gpuRadio.parentElement.style.cursor = 'not-allowed';
        }

        // Update step availability based on execution target
        updateStepAvailability();
    } catch (e) {
        console.error('Failed to load GPU status:', e);
    }
}

// ── Load steps ──────────────────────────────────────────
async function loadSteps() {
    try {
        steps = await API.get('/api/remote/steps');
        const sel = document.getElementById('stepSelect');
        sel.innerHTML = steps.map(s =>
            `<option value="${s.name}">${s.label} (${s.resource.toUpperCase()})</option>`
        ).join('');
        updateStepAvailability();
    } catch (e) {
        console.error('Failed to load steps:', e);
    }
}

// ── Update step availability based on execution target ───
function updateStepAvailability() {
    const target = getExecutionTarget();
    const sel = document.getElementById('stepSelect');
    if (!sel) return;
    // Local-only steps (no remote support yet)
    const localOnly = ['pose', 'deidentify', 'mediapipe'];
    Array.from(sel.options).forEach(opt => {
        const isLocal = localOnly.includes(opt.value);
        const isRemote = target === 'remote';
        opt.disabled = isLocal && isRemote;
    });
    updateJobWarning();
}

// ── Update warning based on step + target combination ───
function updateJobWarning() {
    const executionTarget = getExecutionTarget();
    const jobType = document.getElementById('stepSelect').value;
    const warning = document.getElementById('jobWarning');

    const trainJobs = ['train', 'analyze_v1', 'analyze_v2'];

    if (executionTarget === 'local-cpu' && trainJobs.includes(jobType)) {
        warning.style.display = 'block';
        warning.style.background = '#fff3cd';
        warning.style.color = '#856404';
        warning.innerHTML = `<strong>Warning:</strong> Training/analysis on a local CPU will be extremely slow (hours to days per subject). Consider using a GPU or the remote server instead.`;
    } else {
        warning.style.display = 'none';
    }
}

// ── Get execution target ─────────────────────────────────
function getExecutionTarget() {
    return document.querySelector('input[name="executionTarget"]:checked')?.value || 'remote';
}

// ── Get selected GPU index ───────────────────────────────
function getSelectedGpu() {
    return parseInt(document.getElementById('gpuSelector')?.value || '0');
}

// ── Load subjects ───────────────────────────────────────
async function loadSubjects() {
    try {
        subjects = await API.get('/api/subjects');
        renderSubjectGrid();
        renderRedownloadSubjects();
    } catch (e) {
        document.getElementById('subjectGrid').innerHTML =
            `<span class="empty-state" style="color:var(--red);">${e.message}</span>`;
    }
}

// Map step names to subject property that indicates completion
const STEP_DONE_MAP = {
    mediapipe: 'has_mediapipe',
    pose: 'has_pose',
    deidentify: 'has_deident',
    blur: 'has_blur',
    vision: 'has_vision',
    train: 'has_labels',
    refine: 'has_labels',
    analyze_v1: 'has_labels',
    analyze_v2: 'has_labels',
};

function renderSubjectGrid() {
    const grid = document.getElementById('subjectGrid');
    if (subjects.length === 0) {
        grid.innerHTML = '<span class="empty-state">No subjects found.</span>';
        return;
    }
    grid.innerHTML = subjects.map(s => `
        <label id="subj-${s.id}" onclick="toggleSubjectLabel(this)">
            <input type="checkbox" value="${s.id}" data-name="${s.name}" style="cursor:pointer;">
            <span>${s.name}</span>
        </label>
    `).join('');
    colorSubjectsByStep();
}

function colorSubjectsByStep() {
    const step = document.getElementById('stepSelect')?.value;
    if (!step) return;
    const prop = STEP_DONE_MAP[step];
    subjects.forEach(s => {
        const label = document.getElementById('subj-' + s.id);
        if (!label) return;
        const done = prop ? !!s[prop] : false;
        label.style.borderColor = done ? 'var(--green)' : '';
        label.style.background = done ? 'rgba(76,175,80,0.08)' : '';
    });
}

function toggleSubjectLabel(label) {
    const cb = label.querySelector('input[type=checkbox]');
    // Toggle happens naturally; just update styling
    setTimeout(() => {
        label.classList.toggle('checked', cb.checked);
    }, 0);
}

function selectAllSubjects() {
    document.querySelectorAll('#subjectGrid input[type=checkbox]').forEach(cb => {
        cb.checked = true;
        cb.parentElement.classList.add('checked');
    });
}

function clearSubjects() {
    document.querySelectorAll('#subjectGrid input[type=checkbox]').forEach(cb => {
        cb.checked = false;
        cb.parentElement.classList.remove('checked');
    });
}

function renderRedownloadSubjects() {
    const sel = document.getElementById('redownloadSubject');
    sel.innerHTML = subjects.map(s =>
        `<option value="${s.name}">${s.name}</option>`
    ).join('');
}

// ── Submit job ──────────────────────────────────────────
async function submitJob() {
    const jobType = document.getElementById('stepSelect').value;
    const checked = document.querySelectorAll('#subjectGrid input[type=checkbox]:checked');
    const subjectIds = [];
    const subjectNames = [];
    checked.forEach(cb => {
        subjectIds.push(parseInt(cb.value));
        subjectNames.push(cb.dataset.name);
    });

    if (subjectNames.length === 0) {
        alert('Select at least one subject.');
        return;
    }

    const executionTarget = getExecutionTarget();
    const gpuIndex = executionTarget === 'local-gpu' ? getSelectedGpu() : 0;

    try {
        const result = await API.post('/api/remote/launch', {
            job_type: jobType,
            subject_ids: subjectIds,
            subjects: subjectNames,
            execution_target: executionTarget,
            gpu_index: gpuIndex,
        });

        // Show inline warning if training on local CPU
        const trainJobs = ['train', 'analyze_v1', 'analyze_v2'];
        const warning = document.getElementById('jobWarning');
        if (executionTarget === 'local-cpu' && trainJobs.includes(jobType)) {
            const queueId = result?.queue_id || result?.id;
            const cancelLink = queueId
                ? ` <a href="#" onclick="cancelQueueItem(${queueId});document.getElementById('jobWarning').style.display='none';return false;" style="color:#856404;font-weight:600;">Cancel this job</a>`
                : '';
            warning.style.display = 'block';
            warning.style.background = '#fff3cd';
            warning.style.color = '#856404';
            warning.innerHTML = `<strong>Warning:</strong> ${jobType} job submitted to local CPU — this will be very slow without a GPU.${cancelLink}`;
        } else {
            warning.style.display = 'none';
        }

        clearSubjects();
        refreshQueue();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Queue display ───────────────────────────────────────
function connectQueueStream() {
    if (queueStream) queueStream.close();

    queueStream = new EventSource('/api/remote/stream');
    queueStream.onmessage = (event) => {
        const state = JSON.parse(event.data);
        renderQueue(state);
    };
    queueStream.onerror = () => {
        queueStream.close();
        queueStream = null;
        // Reconnect after a delay
        setTimeout(connectQueueStream, 5000);
    };
}

async function refreshQueue() {
    try {
        const state = await API.get('/api/remote/queue');
        renderQueue(state);
    } catch (e) {
        console.error('Failed to refresh queue:', e);
    }
}

function renderQueue(state) {
    _lastQueueState = state;
    renderLane4('localCpuLane', 'cpu', 'local-cpu', state);
    renderLane4('localGpuLane', 'gpu', 'local-gpu', state);
    renderLane4('remoteCpuLane', 'cpu', 'remote', state);
    renderLane4('remoteGpuLane', 'gpu', 'remote', state);

    // Grey out local GPU if unavailable
    const gpuPanel = document.getElementById('localGpuPanel');
    if (gpuPanel) {
        gpuPanel.style.opacity = gpuAvailable ? '' : '0.4';
    }

    renderHistory(state.history);
}

// Tick elapsed time every second for running jobs
setInterval(() => {
    if (_lastQueueState && _lastQueueState.running && _lastQueueState.running.length > 0) {
        renderLane4('localCpuLane', 'cpu', 'local-cpu', _lastQueueState);
        renderLane4('localGpuLane', 'gpu', 'local-gpu', _lastQueueState);
        renderLane4('remoteCpuLane', 'cpu', 'remote', _lastQueueState);
        renderLane4('remoteGpuLane', 'gpu', 'remote', _lastQueueState);
    }
}, 1000);

function getExecutionTargetBadge(item) {
    const target = item.execution_target || 'remote';
    const colors = {
        'local-cpu': '#4A90E2',
        'local-gpu': '#7ED321',
        'remote': '#9B9B9B',
    };
    const labels = {
        'local-cpu': 'Local CPU',
        'local-gpu': 'Local GPU',
        'remote': 'Remote',
    };
    const color = colors[target] || colors['remote'];
    const label = labels[target] || target;
    return `<span style="background:${color};color:white;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600;">${label}</span>`;
}

function renderLane4(elementId, resource, target, state) {
    const el = document.getElementById(elementId);
    if (!el) return;
    // Filter running items by resource AND execution target
    const isLocal = target.startsWith('local');
    const running = state.running.filter(r => {
        const rTarget = r.execution_target || 'remote';
        if (isLocal) return r.resource === resource && rTarget.startsWith('local');
        return r.resource === resource && rTarget === 'remote';
    });
    // Filter queued items similarly
    const allQueued = resource === 'gpu' ? state.gpu_queue : state.cpu_queue;
    const queued = allQueued.filter(q => {
        const qTarget = q.execution_target || 'remote';
        if (isLocal) return qTarget.startsWith('local');
        return qTarget === 'remote';
    });

    if (running.length === 0 && queued.length === 0) {
        el.innerHTML = '<span class="empty-state">Empty</span>';
        return;
    }

    let html = '';

    // Running items
    for (const item of running) {
        const subjects = parseSubjects(item.subject_ids);
        const pct = item.progress_pct || 0;
        const targetBadge = getExecutionTargetBadge(item);
        const timeInfo = formatJobTime(item.started_at, pct, item.epoch_info);
        let epochLabel = '';
        if (item.epoch_info) {
            try {
                const ei = typeof item.epoch_info === 'string' ? JSON.parse(item.epoch_info) : item.epoch_info;
                if (ei && ei.epoch > 0) epochLabel = `<span style="font-size:11px;color:var(--text-muted);margin-left:4px;">Epoch ${ei.epoch}/${ei.total}</span>`;
            } catch {}
        }
        html += `
            <div class="queue-item" style="border-left: 3px solid var(--orange);">
                <span class="job-indicator job-running"></span>
                <span class="type">${jobTypeLink(item.job_type, item.subject_ids)}</span>
                ${targetBadge}
                <span class="subjects" title="${subjects}">${subjects}</span>
                <div class="progress-bar" style="width:80px;">
                    <div class="fill" style="width:${pct}%"></div>
                </div>
                <span style="font-size:11px;">${pct.toFixed(0)}%</span>
                ${epochLabel}
                <span class="time-info" style="font-size:11px;color:var(--text-muted);margin-left:4px;" title="${timeInfo.tooltip}">${timeInfo.display}</span>
                ${item.job_id ? `<button class="btn btn-sm" onclick="viewJobLogLive(${item.job_id})">Log</button>` : ''}
                <button class="btn btn-sm btn-danger" onclick="cancelQueueItem(${item.id})">Cancel</button>
            </div>
        `;
    }

    // Queued items
    for (let i = 0; i < queued.length; i++) {
        const item = queued[i];
        const subjects = parseSubjects(item.subject_ids);
        const targetBadge = getExecutionTargetBadge(item);
        html += `
            <div class="queue-item">
                <span class="pos">${i + 1}.</span>
                <span class="type">${jobTypeLink(item.job_type, item.subject_ids)}</span>
                ${targetBadge}
                <span class="subjects" title="${subjects}">${subjects}</span>
                <button class="btn btn-sm btn-danger" onclick="cancelQueueItem(${item.id})">Cancel</button>
            </div>
        `;
    }

    el.innerHTML = html;
}

// ── Local jobs (mediapipe, pose, deidentify, etc.) ──────────
async function pollLocalJobs() {
    try {
        const jobs = await API.get('/api/jobs?status=running');
        const card = document.getElementById('localJobsCard');
        const lane = document.getElementById('localJobsLane');
        if (!card || !lane) return;

        if (jobs.length === 0) {
            card.style.display = 'none';
            return;
        }

        card.style.display = '';
        let html = '';
        for (const job of jobs) {
            const name = job.subject_name || ('Subject ' + job.subject_id);
            const pct = job.progress_pct || 0;
            const subjectIds = JSON.stringify([job.subject_id]);
            html += `
                <div class="queue-item" style="border-left: 3px solid var(--orange);">
                    <span class="job-indicator job-running"></span>
                    <span class="type">${jobTypeLink(job.job_type, subjectIds)}</span>
                    <span class="subjects">${name}</span>
                    <div class="progress-bar" style="width:80px;">
                        <div class="fill" style="width:${pct}%"></div>
                    </div>
                    <span style="font-size:11px;">${pct.toFixed(0)}%</span>
                    <button class="btn btn-sm" onclick="viewJobLogLive(${job.id})">Log</button>
                    <button class="btn btn-sm btn-danger" onclick="cancelLocalJob(${job.id})">Cancel</button>
                </div>
            `;
        }
        lane.innerHTML = html;
    } catch (e) {
        // ignore
    }
}

async function cancelLocalJob(jobId) {
    try {
        await API.post(`/api/jobs/${jobId}/cancel`);
        pollLocalJobs();
    } catch (e) {
        alert('Error cancelling job: ' + e.message);
    }
}

pollLocalJobs();
setInterval(pollLocalJobs, 3000);

function parseSubjects(subjectIdsJson) {
    try {
        const arr = JSON.parse(subjectIdsJson);
        if (Array.isArray(arr)) {
            if (arr.length <= 3) return arr.join(', ');
            return arr.slice(0, 3).join(', ') + ` +${arr.length - 3}`;
        }
        return subjectIdsJson;
    } catch {
        return subjectIdsJson;
    }
}

function renderHistory(history) {
    const el = document.getElementById('historyContent');
    if (!history || history.length === 0) {
        el.innerHTML = '<span class="empty-state">No completed jobs yet.</span>';
        return;
    }

    let html = `
        <table class="history-table">
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Subjects</th>
                    <th>Resource</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Finished</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const item of history) {
        const subjects = parseSubjects(item.subject_ids);
        const statusClass = item.status === 'completed' ? 'badge-complete'
            : item.status === 'failed' ? 'badge-error_detection'
            : 'badge-created';
        const errorTip = item.error_msg ? ` title="${item.error_msg}"` : '';
        const finishedAt = item.finished_at ? new Date(item.finished_at + 'Z').toLocaleString() : '-';
        const duration = (item.started_at && item.finished_at)
            ? formatDuration(new Date(item.finished_at + 'Z') - new Date(item.started_at + 'Z'))
            : '-';
        const logBtn = item.job_id
            ? `<button class="btn btn-sm" onclick="viewJobLog(${item.job_id})">Log</button>`
            : '';

        html += `
            <tr>
                <td>${jobTypeLink(item.job_type, item.subject_ids)}</td>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${subjects}">${subjects}</td>
                <td>${item.resource.toUpperCase()}</td>
                <td><span class="badge ${statusClass}"${errorTip}>${item.status}</span></td>
                <td style="font-size:12px;color:var(--text-muted);">${duration}</td>
                <td style="font-size:12px;color:var(--text-muted);">${finishedAt}</td>
                <td>${logBtn}</td>
            </tr>
        `;
    }

    html += '</tbody></table>';
    el.innerHTML = html;
}

// ── Cancel ──────────────────────────────────────────────
async function cancelQueueItem(queueId) {
    try {
        await API.post(`/api/remote/cancel/${queueId}`);
        refreshQueue();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Log viewing ─────────────────────────────────────────
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

// ── Re-download ─────────────────────────────────────────
function onRedownloadStepChange() {
    const step = document.getElementById('redownloadStep').value;
    const hint = document.getElementById('redownloadHint');
    // CPU steps support multiple subjects via the launch panel
    hint.style.display = ['mediapipe', 'blur', 'mediapipe+blur'].includes(step) ? '' : 'none';
}

async function redownload() {
    const step = document.getElementById('redownloadStep').value;
    const subject = document.getElementById('redownloadSubject').value;
    if (!subject) {
        alert('Select a subject.');
        return;
    }

    try {
        const result = await API.post('/api/remote/redownload', {
            job_type: step,
            subjects: [subject],
        });
        if (result.job_id) {
            viewJobLogLive(result.job_id);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Init ────────────────────────────────────────────────
(async function init() {
    // Add event listeners for execution target radio buttons
    document.querySelectorAll('input[name="executionTarget"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            const gpuSelector = document.getElementById('gpuSelectorDiv');
            const targetLocalGpu = e.target.value === 'local-gpu';
            gpuSelector.style.display = targetLocalGpu ? 'flex' : 'none';
            updateJobWarning();
        });
    });

    // Update warning when step changes
    document.getElementById('stepSelect')?.addEventListener('change', () => {
        updateJobWarning();
        colorSubjectsByStep();
    });

    // Load data
    await loadGpuStatus();
    await loadSteps();
    await loadSubjects();
    await refreshQueue();
    connectQueueStream();
})();
