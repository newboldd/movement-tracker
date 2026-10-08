/* Jobs page: pick a step, pick subjects, queue it, watch it run. */

const jobsPage = (() => {
    const el = (id) => document.getElementById(id);

    let steps = [];
    let env = {};
    let subjects = [];
    let selectedStep = null;
    const selected = new Set();
    let stateSource = null;
    let logJobId = null, logTimer = null, logOffset = 0;

    async function api(url, options) {
        const r = await fetch(url, options);
        if (!r.ok) {
            let detail = r.statusText;
            try { detail = (await r.json()).detail || detail; } catch (e) { /* not JSON */ }
            throw new Error(detail);
        }
        return r.status === 204 ? null : r.json();
    }

    // ── Steps ───────────────────────────────────────────────────────────

    async function loadSteps() {
        env = await api('/api/queue/steps');
        steps = env.steps || [];

        const dlcMissing = !env.dlc_installed;
        const warn = el('envWarn');
        if (dlcMissing) {
            warn.className = 'warn bad';
            warn.innerHTML = 'DeepLabCut is not installed in this environment, so '
                + 'training and analysis are unavailable. Labeling and MediaPipe '
                + 'work without it.<br>'
                + `<button class="btn btn-sm" id="installDlcBtn" style="margin-top:6px;">`
                + 'Install DeepLabCut now</button> '
                + '<span style="font-size:11px;">several GB — equivalent to '
                + '<code>./setup.sh --with-dlc</code></span>';
            warn.style.display = '';
            el('installDlcBtn').addEventListener('click', installDlc);
        } else if (!env.gpu_available) {
            warn.className = 'warn info';
            warn.innerHTML = 'No CUDA GPU detected. Training and analysis will run '
                + 'on the CPU, which is far slower but works.';
            warn.style.display = '';
        } else {
            warn.style.display = 'none';
        }

        const gpuSel = el('gpuSelect');
        gpuSel.innerHTML = (env.gpus || []).map(g =>
            `<option value="${g.index}">GPU ${g.index} — ${g.name}`
            + `${g.memory_mb ? ` (${Math.round(g.memory_mb / 1024)} GB)` : ''}</option>`
        ).join('') || '<option value="0">CPU</option>';

        const row = el('stepRow');
        row.innerHTML = '';
        for (const s of steps) {
            const btn = document.createElement('button');
            btn.className = 'step-btn';
            btn.dataset.step = s.name;
            btn.innerHTML = `${s.label}<span class="res">${s.resource}</span>`;
            btn.title = s.help || '';
            btn.disabled = s.resource === 'gpu' && dlcMissing;
            if (btn.disabled) btn.title = 'Needs DeepLabCut installed';
            btn.addEventListener('click', () => pickStep(s.name));
            row.appendChild(btn);
        }
        pickStep(steps.find(s => !(s.resource === 'gpu' && dlcMissing))?.name
                 || steps[0]?.name);
    }

    function pickStep(name) {
        if (!name) return;
        selectedStep = name;
        const step = steps.find(s => s.name === name);
        document.querySelectorAll('.step-btn').forEach(b =>
            b.classList.toggle('active', b.dataset.step === name));
        el('stepHelp').textContent = step ? (step.help || '') : '';
        el('mpOptions').style.display = (name === 'mediapipe') ? 'flex' : 'none';
        el('gpuOptions').style.display =
            (step && step.resource === 'gpu' && (env.gpus || []).length > 1)
                ? 'flex' : 'none';
        renderSubjects();
    }

    async function installDlc() {
        const btn = el('installDlcBtn');
        btn.disabled = true;
        btn.textContent = 'Queueing…';
        try {
            await api('/api/queue/launch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_type: 'install-dlc', subjects: [] }),
            });
            btn.textContent = 'Installing — watch the CPU lane below';
        } catch (e) {
            btn.disabled = false;
            btn.textContent = 'Install DeepLabCut now';
            alert(`Could not start the install: ${e.message}`);
        }
    }

    // ── Subjects ────────────────────────────────────────────────────────

    async function loadSubjects() {
        subjects = await api('/api/queue/subjects');
        const preselect = parseInt(
            new URLSearchParams(location.search).get('subject'), 10);
        if (Number.isFinite(preselect)) {
            const s = subjects.find(x => x.id === preselect);
            if (s) selected.add(s.name);
        }
        renderSubjects();
    }

    /* Why a step can't run for a subject yet — shown rather than just
     * disabling the card, so the fix is obvious. */
    function ineligibleReason(s) {
        if (selectedStep === 'mediapipe') {
            return s.trial_count ? null : 'no trial videos found';
        }
        if (selectedStep === 'train') {
            return s.has_labeled_data ? null : 'commit some labels first';
        }
        if (selectedStep === 'refine') {
            if (!s.has_snapshots) return 'train a first model first';
            return s.has_labeled_data ? null : 'no labeled data';
        }
        if (selectedStep === 'analyze') {
            return s.has_snapshots ? null : 'no trained model yet';
        }
        return null;
    }

    function renderSubjects() {
        const grid = el('subjectGrid');
        if (!subjects.length) {
            grid.innerHTML = '<span class="empty-state">No subjects. Add videos '
                + 'and press Sync on the Subjects page.</span>';
            return;
        }
        grid.innerHTML = '';
        for (const s of subjects) {
            const reason = ineligibleReason(s);
            const card = document.createElement('div');
            card.className = 'subj-card'
                + (selected.has(s.name) ? ' selected' : '')
                + (reason ? ' ineligible' : '');
            card.title = reason || `${s.trial_count} trial(s), stage ${s.stage}`;

            const chips = selectedStep === 'mediapipe'
                ? s.trials.map(t => {
                    const n = ['has_forward', 'has_reverse', 'has_static']
                        .filter(k => t[k]).length;
                    const short = t.trial_name.includes('_')
                        ? t.trial_name.split('_').slice(1).join('_') : t.trial_name;
                    return `<span class="tchip${n ? ' done' : ''}" `
                        + `title="${t.trial_name}: ${n} pass(es)`
                        + `${t.has_best ? ', fused' : ''}">${short}${n ? ` ${n}` : ''}`
                        + `</span>`;
                }).join('')
                : s.trials.map(t => {
                    const short = t.trial_name.includes('_')
                        ? t.trial_name.split('_').slice(1).join('_') : t.trial_name;
                    return `<span class="tchip${t.has_analysis ? ' done' : ''}" `
                        + `title="${t.trial_name}`
                        + `${t.has_analysis ? ' — analyzed' : ' — not analyzed'}">`
                        + `${short}</span>`;
                }).join('');

            card.innerHTML = `<div class="nm">${s.name}</div>`
                + `<div class="meta">${reason || s.stage}</div>`
                + `<div class="tchips">${chips}</div>`;

            if (!reason) {
                card.addEventListener('click', () => {
                    if (selected.has(s.name)) selected.delete(s.name);
                    else selected.add(s.name);
                    renderSubjects();
                });
            }
            grid.appendChild(card);
        }
    }

    async function launch() {
        if (!selectedStep) return;
        const names = [...selected].filter(n => {
            const s = subjects.find(x => x.name === n);
            return s && !ineligibleReason(s);
        });
        if (!names.length) {
            el('launchStatus').textContent = 'Select at least one subject.';
            return;
        }

        const body = {
            job_type: selectedStep,
            subjects: names,
            gpu_index: parseInt(el('gpuSelect').value, 10) || 0,
        };
        if (selectedStep === 'mediapipe') {
            body.reverse = el('optReverse').checked;
            body.static_image_mode = el('optStatic').checked;
            body.use_bbox = el('optBbox').checked;
        }

        el('launchBtn').disabled = true;
        el('launchStatus').textContent = 'Queueing…';
        try {
            const r = await api('/api/queue/launch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            el('launchStatus').textContent =
                `Queued ${r.queued.length} job(s): ${names.join(', ')}`;
            selected.clear();
            renderSubjects();
        } catch (e) {
            el('launchStatus').textContent = `Failed: ${e.message}`;
        } finally {
            el('launchBtn').disabled = false;
        }
    }

    // ── Queue state ─────────────────────────────────────────────────────

    function describe(item) {
        const subs = (item.subjects || []).join(', ') || '—';
        const p = item.params || {};
        const extra = [];
        if (p.pass) extra.push(p.pass);
        if (p.trial_name) extra.push(p.trial_name);
        if (p.cropped) extra.push('cropped');
        return { subs, extra: extra.join(' · ') };
    }

    function queueItemHtml(item, running) {
        const { subs, extra } = describe(item);
        const pct = Math.round(item.progress_pct || 0);
        const epoch = (() => {
            if (!item.epoch_info) return '';
            try {
                const e = JSON.parse(item.epoch_info);
                return e.epoch ? ` epoch ${e.epoch}/${e.total}` : '';
            } catch (err) { return ''; }
        })();
        return `<div class="qitem${running ? ' running' : ''}">
            <div class="top">
                <span class="job-indicator ${running ? 'job-running' : 'job-pending'}"></span>
                <span class="nm">${subs}</span>
                <span class="ty">${item.job_type}${extra ? ' · ' + extra : ''}</span>
                <span class="acts">
                    ${item.job_id ? `<button data-log="${item.job_id}" title="Show the log">log</button>` : ''}
                    <button data-cancel="${item.id}" title="Cancel">✕</button>
                </span>
            </div>
            ${running ? `<div class="progress-bar" style="margin-top:4px;">
                <div class="fill" style="width:${pct}%"></div></div>
                <div style="font-size:10px;color:var(--text-muted);">${pct}%${epoch}</div>`
                : ''}
        </div>`;
    }

    function renderState(st) {
        const running = st.running || [];
        for (const [laneId, resource] of [['cpuLane', 'cpu'], ['gpuLane', 'gpu']]) {
            const run = running.filter(r => r.resource === resource);
            const queued = (resource === 'cpu' ? st.cpu_queue : st.gpu_queue) || [];
            const box = el(laneId);
            const html = run.map(r => queueItemHtml(r, true)).join('')
                + queued.map(q => queueItemHtml(q, false)).join('');
            box.innerHTML = html || '<span class="empty-state">Empty</span>';
        }

        const hist = st.history || [];
        el('historyBox').innerHTML = hist.length ? `<table class="hist-table">
            <thead><tr><th>Subject</th><th>Job</th><th>Status</th>
                <th>Finished</th><th></th></tr></thead><tbody>
            ${hist.map(h => {
                const { subs, extra } = describe(h);
                const cls = h.status === 'completed' ? 'job-completed'
                    : h.status === 'failed' ? 'job-failed' : 'job-pending';
                return `<tr>
                    <td>${subs}</td>
                    <td>${h.job_type}${extra ? ` <span style="color:var(--text-muted);font-size:10px;">${extra}</span>` : ''}</td>
                    <td><span class="job-indicator ${cls}"></span>${h.status}${
                        h.error_msg ? ` <span style="color:var(--red);font-size:10px;">${h.error_msg}</span>` : ''}</td>
                    <td style="color:var(--text-muted);">${(h.finished_at || '').replace('T', ' ').slice(0, 19)}</td>
                    <td><button class="btn btn-sm" data-log="${h.job_id}">log</button></td>
                </tr>`;
            }).join('')}</tbody></table>` : '<span class="empty-state">Nothing yet</span>';

        document.querySelectorAll('[data-log]').forEach(b =>
            b.addEventListener('click', () => openLog(parseInt(b.dataset.log, 10))));
        document.querySelectorAll('[data-cancel]').forEach(b =>
            b.addEventListener('click', () => cancel(parseInt(b.dataset.cancel, 10))));
    }

    async function cancel(queueId) {
        try {
            await api(`/api/queue/cancel/${queueId}`, { method: 'POST' });
        } catch (e) {
            alert(`Could not cancel: ${e.message}`);
        }
    }

    /* Live queue state over server-sent events, with polling as the
     * fallback — a reverse proxy or an old browser can break SSE, and the
     * page is useless if it stops updating. */
    function watchState() {
        try {
            stateSource = new EventSource('/api/queue/stream');
            stateSource.onmessage = (e) => {
                try { renderState(JSON.parse(e.data)); } catch (err) { /* keep-alive */ }
            };
            stateSource.onerror = () => {
                stateSource.close();
                stateSource = null;
                setInterval(pollState, 2000);
            };
        } catch (e) {
            setInterval(pollState, 2000);
        }
        pollState();
    }

    async function pollState() {
        try { renderState(await api('/api/queue/state')); } catch (e) { /* retry */ }
    }

    // ── Log viewer ──────────────────────────────────────────────────────

    async function openLog(jobId) {
        logJobId = jobId;
        logOffset = 0;
        el('logTitle').textContent = `Job ${jobId} log`;
        el('logContent').textContent = 'Loading…';
        el('logModal').classList.add('active');
        tailLog();
    }

    async function tailLog() {
        clearTimeout(logTimer);
        if (!logJobId) return;
        try {
            const job = await api(`/api/jobs/${logJobId}`);
            el('logStatus').textContent =
                `${job.status} · ${Math.round(job.progress_pct || 0)}%`;
            const text = (job.log_tail || []).join('');
            el('logContent').textContent = text || '(no output yet)';
            const pre = el('logContent');
            pre.scrollTop = pre.scrollHeight;
            if (!['completed', 'failed', 'cancelled'].includes(job.status)) {
                logTimer = setTimeout(tailLog, 1500);
            }
        } catch (e) {
            el('logContent').textContent = `Could not read the log: ${e.message}`;
        }
    }

    function closeLog() {
        logJobId = null;
        clearTimeout(logTimer);
        el('logModal').classList.remove('active');
    }

    // ── Lifetime history ────────────────────────────────────────────────

    async function loadLifetime() {
        let d;
        try { d = await api('/api/queue/history?limit=200'); } catch (e) { return; }
        const recs = d.records || [];
        const sum = d.summary || {};
        el('lifetimeSummary').textContent = sum.count
            ? `${sum.count} run(s) recorded` : '';
        if (!recs.length) return;
        el('lifetimeBox').innerHTML = `<table class="hist-table">
            <thead><tr><th>When</th><th>Job</th><th>Status</th><th>Duration</th>
                <th>Version</th></tr></thead><tbody>
            ${recs.map(r => `<tr>
                <td style="color:var(--text-muted);">${(r.ts || '').replace('T', ' ').slice(0, 19)}</td>
                <td>${r.job_type || ''}</td>
                <td>${r.status || ''}</td>
                <td>${r.duration_sec != null ? `${Math.round(r.duration_sec)}s` : ''}</td>
                <td style="color:var(--text-muted);font-family:monospace;font-size:10px;">
                    ${(r.git_version || '').slice(0, 8)}</td>
            </tr>`).join('')}</tbody></table>`;
    }

    // ── Boot ────────────────────────────────────────────────────────────

    document.addEventListener('DOMContentLoaded', async () => {
        el('launchBtn').addEventListener('click', launch);
        el('logClose').addEventListener('click', closeLog);
        el('logModal').addEventListener('click', (e) => {
            if (e.target.id === 'logModal') closeLog();
        });
        el('selectAllBtn').addEventListener('click', () => {
            subjects.filter(s => !ineligibleReason(s)).forEach(s => selected.add(s.name));
            renderSubjects();
        });
        el('selectNoneBtn').addEventListener('click', () => {
            selected.clear();
            renderSubjects();
        });

        await loadSteps();
        await loadSubjects();
        watchState();
        loadLifetime();
        setInterval(loadLifetime, 30000);
    });

    return { launch, openLog };
})();
