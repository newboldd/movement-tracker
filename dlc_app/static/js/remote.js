/* Remote Jobs: queue management, job launching, monitoring */

let subjects = [];
let steps = [];
let queueStream = null;
let _logStream = null;

// ── Load steps ──────────────────────────────────────────
async function loadSteps() {
    try {
        steps = await API.get('/api/remote/steps');
        const sel = document.getElementById('stepSelect');
        sel.innerHTML = steps.map(s =>
            `<option value="${s.name}">${s.label} (${s.resource.toUpperCase()})</option>`
        ).join('');
    } catch (e) {
        console.error('Failed to load steps:', e);
    }
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

    try {
        const result = await API.post('/api/remote/launch', {
            job_type: jobType,
            subject_ids: subjectIds,
            subjects: subjectNames,
        });
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
    renderLane('gpuLane', 'gpu', state);
    renderLane('cpuLane', 'cpu', state);
    renderHistory(state.history);
}

function renderLane(elementId, resource, state) {
    const el = document.getElementById(elementId);
    const running = state.running.filter(r => r.resource === resource);
    const queued = resource === 'gpu' ? state.gpu_queue : state.cpu_queue;

    if (running.length === 0 && queued.length === 0) {
        el.innerHTML = '<span class="empty-state">Empty</span>';
        return;
    }

    let html = '';

    // Running items
    for (const item of running) {
        const subjects = parseSubjects(item.subject_ids);
        const pct = item.progress_pct || 0;
        html += `
            <div class="queue-item" style="border-left: 3px solid var(--orange);">
                <span class="job-indicator job-running"></span>
                <span class="type">${item.job_type}</span>
                <span class="subjects" title="${subjects}">${subjects}</span>
                <div class="progress-bar" style="width:80px;">
                    <div class="fill" style="width:${pct}%"></div>
                </div>
                <span style="font-size:11px;">${pct.toFixed(0)}%</span>
                ${item.job_id ? `<button class="btn btn-sm" onclick="viewJobLogLive(${item.job_id})">Log</button>` : ''}
                <button class="btn btn-sm btn-danger" onclick="cancelQueueItem(${item.id})">Cancel</button>
            </div>
        `;
    }

    // Queued items
    for (let i = 0; i < queued.length; i++) {
        const item = queued[i];
        const subjects = parseSubjects(item.subject_ids);
        html += `
            <div class="queue-item">
                <span class="pos">${i + 1}.</span>
                <span class="type">${item.job_type}</span>
                <span class="subjects" title="${subjects}">${subjects}</span>
                <button class="btn btn-sm btn-danger" onclick="cancelQueueItem(${item.id})">Cancel</button>
            </div>
        `;
    }

    el.innerHTML = html;
}

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
                    <th>Finished</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const item of history) {
        const subjects = parseSubjects(item.subject_ids);
        const statusClass = item.status === 'completed' ? 'badge-trained'
            : item.status === 'failed' ? 'badge-error_detection'
            : 'badge-created';
        const errorTip = item.error_msg ? ` title="${item.error_msg}"` : '';
        const finishedAt = item.finished_at ? new Date(item.finished_at + 'Z').toLocaleString() : '-';
        const logBtn = item.job_id
            ? `<button class="btn btn-sm" onclick="viewJobLog(${item.job_id})">Log</button>`
            : '';

        html += `
            <tr>
                <td style="font-weight:600;">${item.job_type}</td>
                <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${subjects}">${subjects}</td>
                <td>${item.resource.toUpperCase()}</td>
                <td><span class="badge ${statusClass}"${errorTip}>${item.status}</span></td>
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
    await loadSteps();
    await loadSubjects();
    await refreshQueue();
    connectQueueStream();
})();
