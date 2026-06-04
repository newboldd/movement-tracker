/* Processing Jobs: queue management, job launching, monitoring with local/remote/GPU support */

let subjects = [];
let steps = [];
let selectedStep = 'mediapipe';
let queueStream = null;
let _logStream = null;
let gpuAvailable = false;
let availableGpus = [];
let _lastQueueState = null;

// ── Job type → page URL mapping ──────────────────────────
// Job-type → page kind.  "dlc" jobs go to the DLC page (no trial granularity);
// everything else (including deidentify) goes to the Auto page where trial-
// level review is possible.  Used by both the type-column and trial badges.
function jobTypeKind(jobType) {
    const dlc = new Set(['train', 'refine', 'analyze_v1', 'analyze_v2',
                          'redownload_train', 'redownload_analyze_v1']);
    return dlc.has(jobType) ? 'dlc' : 'analyze';
}

function _subjectIdByName(name) {
    // ``subjects`` is a module-level array of {id, name, ...} populated by
    // loadSubjects().  Fall back to the name if not yet available — the
    // /analyze page also accepts a name lookup against allSubjects.
    if (!name) return null;
    if (typeof subjects !== 'undefined' && Array.isArray(subjects)) {
        const s = subjects.find(s => s.name === name);
        if (s) return s.id;
    }
    return null;
}

// Build a URL to review a specific subject (and optionally trial) on the
// page that's relevant for the given job type.
function subjectReviewUrl(jobType, subjectName, trialIdx) {
    const kind = jobTypeKind(jobType);
    const sid = _subjectIdByName(subjectName);
    const subParam = sid ? `subject=${sid}` : (subjectName ? `subject=${encodeURIComponent(subjectName)}` : '');
    if (kind === 'dlc') {
        return '/labeling-select' + (subParam ? `?${subParam}` : '');
    }
    // Auto page — accepts subject= and (newly) trial= for trial selection.
    let qs = subParam;
    if (trialIdx != null && Number.isFinite(trialIdx)) {
        qs += (qs ? '&' : '') + `trial=${trialIdx}`;
    }
    return '/analyze' + (qs ? `?${qs}` : '');
}

function trialBadge(item) {
    // Returns trial chips for a queue/history item, colored by the
    // SAME outcome-driven scheme as the trial-detail modal:
    //
    //   outcome === 'ok'           → green   (downloaded)
    //   outcome === 'remote_only'  → yellow  (completed remotely)
    //   outcome === 'failed'       → red     (failed)
    //   uploaded (no outcome)      → blue    (pending inference)
    //   no uploaded + no outcome   → dim blue (uploading / queued)
    //   first uploaded-no-outcome  → blinking yellow (running on GPU)
    //
    // For batches with many trials we cap to MAX_CHIPS and append a
    // "+N" pill so the queue row doesn't wrap forever.
    //
    // Suppressed for multi-subject jobs — the chip list would overflow
    // a narrow queue row, and the trial-detail modal is the right place
    // to show the full grid (click the row to open it).
    const raw = item && (item.params_json || item.extra_params_json || item.trial_name);
    if (!raw) return '';
    try {
        const _ids = JSON.parse(item.subject_ids);
        if (Array.isArray(_ids) && _ids.length > 1) return '';
    } catch {}
    const MAX_CHIPS = 8;
    const stripPrefix = (s) => {
        if (!s) return '';
        const i = s.indexOf('_');
        return i >= 0 ? s.slice(i + 1) : s;
    };
    const baseStyle = 'color:#fff;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:600;margin-left:4px;';
    try {
        const p = typeof raw === 'string' ? JSON.parse(raw) : raw;

        // Multi-trial path: render outcome-coloured chips for each trial.
        if (Array.isArray(p?.trials) && p.trials.length) {
            const trials = p.trials;
            // Find the active (first uploaded-no-outcome) trial — gets a
            // pulsing yellow chip via .trial-current.
            let currentRef = null;
            for (const t of trials) {
                if (t && t.uploaded && !t.outcome) { currentRef = t; break; }
            }
            const renderChip = (t) => {
                const short = stripPrefix(t.trial_name || '');
                if (!short) return '';
                let bg = 'var(--accent,#2196f3)';
                let opacity = '1';
                let suffix = '';
                let extraClass = '';
                const out = t.outcome;
                if (out === 'ok') {
                    bg = 'var(--green,#4caf50)'; suffix = ' — downloaded';
                } else if (out === 'remote_only') {
                    bg = 'var(--yellow,#fbc02d)'; suffix = ' — completed (not downloaded)';
                } else if (out === 'failed') {
                    bg = '#e53935';
                    suffix = t.outcome_error ? ` — failed (${t.outcome_error})` : ' — failed';
                } else if (t === currentRef) {
                    bg = 'var(--yellow,#fbc02d)';
                    extraClass = 'trial-current';
                    suffix = ' — running on GPU';
                } else if (t.uploaded) {
                    suffix = ' — uploaded, pending inference';
                } else {
                    opacity = '0.4';
                    suffix = ' — uploading / queued';
                }
                const label = t.subject_name ? `${t.subject_name} ${short}` : short;
                return `<span class="${extraClass}" title="${label}${suffix}" `
                     + `style="${baseStyle}background:${bg};opacity:${opacity};">${short}</span>`;
            };
            const visible = trials.slice(0, MAX_CHIPS);
            const hidden = trials.length - visible.length;
            let html = visible.map(renderChip).join('');
            if (hidden > 0) {
                html += `<span title="${hidden} more trial(s)" `
                      + `style="${baseStyle}background:var(--text-muted,#888);">+${hidden}</span>`;
            }
            return html;
        }

        // Legacy single-trial path: one chip with the trial name (e.g.
        // single deidentify or single HRnet jobs).  Color by job status
        // when available so an in-progress chip is still visually
        // distinct from a failed one.
        const name = stripPrefix(p?.trial_name || '');
        if (!name) return '';
        let bg = 'var(--accent,#2196f3)';
        const status = item?.status || '';
        if (status === 'completed') bg = 'var(--green,#4caf50)';
        else if (status === 'failed') bg = '#e53935';
        return `<span style="${baseStyle}background:${bg};">${name}</span>`;
    } catch {}
    return '';
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

// ── Detect remote-server reachability ────────────────────
let remoteReachable = null;   // null = unknown, true/false after first probe
async function loadRemoteStatus() {
    // The /test-remote endpoint runs four checks (ssh, work_dir, dlc, gpu).
    // We only need the host to be reachable for HRnet / MP / Deidentify
    // dispatch — DLC absence shouldn't disable remote.  So accept the
    // server as "reachable" when ssh + work_dir both succeed, regardless
    // of the overall ok flag.
    let ok = false;
    try {
        const res = await API.post('/api/settings/test-remote', {});
        const d = res?.details || {};
        ok = !!(d.ssh && d.work_dir);
    } catch { ok = false; }
    remoteReachable = ok;
    const radio = document.getElementById('targetRemote');
    if (!radio) return;
    radio.disabled = !ok;
    radio.parentElement.style.opacity = ok ? '' : '0.5';
    radio.parentElement.style.cursor = ok ? '' : 'not-allowed';
    radio.parentElement.title = ok ? '' : 'Remote server unreachable — using local execution';
    // If Remote was selected but the server just went down, fall back to
    // Local CPU so submissions don't quietly fail.
    if (!ok && radio.checked) {
        const fallback = document.getElementById('targetLocalCpu');
        if (fallback) {
            fallback.checked = true;
            updateStepAvailability();
        }
    }
}

// ── Detect GPU availability ─────────────────────────────
async function loadGpuStatus() {
    try {
        const status = await API.get('/api/settings/status');
        gpuAvailable = status.local_gpu_available || false;
        availableGpus = status.gpus || [];

        // Populate GPU selector
        const gpuSel = document.getElementById('gpuSelector');
        if (gpuSel) gpuSel.innerHTML = availableGpus.map(g =>
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

function selectStep(step) {
    selectedStep = step;
    document.querySelectorAll('.step-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.step === step);
    });
    // Cache Results step buttons toggle a body class.  The CSS in
    // remote.html (.cache-step ...) handles hiding the per-job
    // pickers without this JS having to enumerate everything.
    const isCacheStep = (step === 'cache_group' || step === 'cache_explore');
    document.body.classList.toggle('cache-step', isCacheStep);
    const cacheBody = document.getElementById('cacheStepBody');
    if (cacheBody) {
        cacheBody.style.display = isCacheStep ? '' : 'none';
        if (isCacheStep && typeof window.cacheResultsMount === 'function') {
            window.cacheResultsMount(step === 'cache_group' ? 'group' : 'explore');
        }
    }
    updateStepAvailability();
    updateJobWarning();
    colorSubjectsByStep();
    if (isTrialStep()) updateTrialSection();
    document.getElementById('trialSection').style.display = (isTrialStep() && !isCacheStep) ? '' : 'none';
    // MediaPipe-only "Run in reverse" + "Use bounding box" options.
    // Cleared / reset whenever the user switches away from MediaPipe
    // so a stale state doesn't accidentally tag a different step's launch.
    const isMP = (step === 'mediapipe');
    const mpReverseRow = document.getElementById('mpReverseRow');
    if (mpReverseRow) {
        mpReverseRow.style.display = isMP ? 'flex' : 'none';
        if (!isMP) {
            const cb = document.getElementById('mpReverseCb');
            if (cb) cb.checked = false;
        }
    }
    const mpUseBboxRow = document.getElementById('mpUseBboxRow');
    if (mpUseBboxRow) {
        mpUseBboxRow.style.display = isMP ? 'flex' : 'none';
        if (!isMP) {
            const cb = document.getElementById('mpUseBboxCb');
            // Default state on re-entry: bbox enabled (matches HTML
            // ``checked`` attribute and the historical default).
            if (cb) cb.checked = true;
        }
    }
    const mpStaticRow = document.getElementById('mpStaticRow');
    if (mpStaticRow) {
        mpStaticRow.style.display = isMP ? 'flex' : 'none';
        if (!isMP) {
            const cb = document.getElementById('mpStaticCb');
            if (cb) cb.checked = false;
        }
    }
    // Skeleton Fit v1 settings panel — only visible for that step.
    const skelfitBox = document.getElementById('skelfitSettings');
    if (skelfitBox) skelfitBox.style.display = (step === 'skeleton_v1') ? '' : 'none';
    // Stereo Correct settings panel — only visible for stereo_correct.
    const stereoBox = document.getElementById('stereoCorrectSettings');
    if (stereoBox) stereoBox.style.display = (step === 'stereo_correct') ? '' : 'none';
}

// ── Load steps ──────────────────────────────────────────
async function loadSteps() {
    try {
        steps = await API.get('/api/remote/steps');
        // Highlight default step button
        selectStep(selectedStep);
    } catch (e) {
        console.error('Failed to load steps:', e);
    }
}

// ── Update step availability based on execution target ───
function updateStepAvailability() {
    const target = getExecutionTarget();
    // MediaPipe has a remote handler; only pose + vision are local-only.
    const localOnly = ['pose', 'vision'];
    document.querySelectorAll('.step-btn').forEach(btn => {
        const isLocal = localOnly.includes(btn.dataset.step);
        const isRemote = target === 'remote';
        btn.disabled = isLocal && isRemote;
        btn.style.opacity = (isLocal && isRemote) ? '0.35' : '';
    });
    updateJobWarning();
}

// ── Update warning based on step + target combination ───
function updateJobWarning() {
    const executionTarget = getExecutionTarget();
    const jobType = selectedStep;
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

// Steps that support per-trial selection
const TRIAL_STEPS = new Set(['deidentify', 'hrnet', 'preproc', 'mediapipe', 'skeleton_v1', 'stereo_correct']);
// Cache: subjectId → [{trial_idx, trial_name, has_skeleton_v1}, ...]
// Populated by ensureSkeletonStatusForSubjects() the first time a
// subject is shown in the Skel Fit v1 trial picker so the trial chips
// can be coloured green (has npz) or neutral (missing).
const cachedSkeletonStatus = {};
let _skelStatusFetchInFlight = false;
async function ensureSkeletonStatusForSubjects(subjectIds) {
    const need = (subjectIds || []).filter(id => id != null && !(id in cachedSkeletonStatus));
    if (!need.length || _skelStatusFetchInFlight) return false;
    _skelStatusFetchInFlight = true;
    try {
        await Promise.all(need.map(async (sid) => {
            try {
                const data = await API.get(`/api/skeleton/${sid}/trial_skeleton_status`);
                cachedSkeletonStatus[sid] = (data && data.trials) || [];
            } catch {
                cachedSkeletonStatus[sid] = [];
            }
        }));
        return true;
    } finally {
        _skelStatusFetchInFlight = false;
    }
}
// Cache: subjectId (number) -> trial array from /api/deidentify/{id}/trials
const cachedTrialData = {};
// Cache: subjectId (number) -> trial array from /api/analyze/hrnet/job-status
const cachedHrnetStatus = {};

// Async: ensure HRnet trial status is cached for all subjects appearing in
// the given HRnet jobs.  Used by the queue + history renderers to color
// trial chips by current output existence (green = has heatmaps on disk,
// red = missing).  Fires a single batched fetch covering all uncached
// subjects, then triggers a re-render once data arrives.
let _hrnetStatusFetchInFlight = false;
async function ensureHrnetStatusForJobs(items) {
    const ids = new Set();
    for (const it of (items || [])) {
        if (!it || it.job_type !== 'hrnet') continue;
        let names = [];
        try {
            const arr = JSON.parse(it.subject_ids);
            names = Array.isArray(arr) ? arr : [arr];
        } catch { /* ignore */ }
        for (const n of names) {
            const sid = _subjectIdByName(n);
            if (sid != null && !cachedHrnetStatus[sid]) ids.add(sid);
        }
    }
    if (!ids.size || _hrnetStatusFetchInFlight) return false;
    _hrnetStatusFetchInFlight = true;
    try {
        const idList = [...ids].join(',');
        const data = await API.get(`/api/analyze/hrnet/job-status?subject_ids=${idList}`);
        for (const sid of ids) {
            cachedHrnetStatus[sid] = (data?.subjects?.[sid]?.trials) || [];
        }
        return true;
    } catch {
        for (const sid of ids) cachedHrnetStatus[sid] = [];
        return false;
    } finally {
        _hrnetStatusFetchInFlight = false;
    }
}

// Lookup a trial's HRnet output status by subject name + trial_idx.  Returns
// the trial-status object from cachedHrnetStatus, or null if not yet cached.
function _hrnetTrialStatus(subjectName, trialIdx) {
    const sid = _subjectIdByName(subjectName);
    if (sid == null) return null;
    const trials = cachedHrnetStatus[sid];
    if (!trials) return null;
    return trials.find(t => t.trial_idx === trialIdx) || null;
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
    const step = selectedStep;
    if (!step) return;
    if (step === 'hrnet') {
        // HRnet has no subject-level boolean — derive it from per-trial
        // heatmap output presence.  Kicks off the cache fill if needed
        // (it triggers another colorSubjectsByStep when results land).
        _ensureHrnetStatusForAllSubjects().then(() =>
            _colorSubjectsFromHrnetCache('has_hrnet_output'));
        _colorSubjectsFromHrnetCache('has_hrnet_output');
        return;
    }
    if (step === 'mediapipe') {
        // Per-trial MP npz presence — flip between forward / reverse
        // based on the "Run in reverse" checkbox so the green subject
        // halo always reflects "all trials have THIS pass's npz".
        const mpReverse = !!document.getElementById('mpReverseCb')?.checked;
        const flag = mpReverse ? 'has_mp_reverse_npz' : 'has_mp_npz';
        _ensureHrnetStatusForAllSubjects().then(() =>
            _colorSubjectsFromHrnetCache(flag));
        _colorSubjectsFromHrnetCache(flag);
        return;
    }
    if (step === 'skeleton_v1') {
        // Per-trial skeleton_v1.npz presence — halo a subject green
        // when EVERY trial has the npz already.
        const ids = subjects.map(s => s.id);
        ensureSkeletonStatusForSubjects(ids).then(() =>
            _colorSubjectsFromSkeletonCache());
        _colorSubjectsFromSkeletonCache();
        return;
    }
    const prop = STEP_DONE_MAP[step];
    subjects.forEach(s => {
        const label = document.getElementById('subj-' + s.id);
        if (!label) return;
        const done = prop ? !!s[prop] : false;
        label.style.borderColor = done ? 'var(--green)' : '';
        label.style.background = done ? 'rgba(76,175,80,0.08)' : '';
    });
}

function _colorSubjectsFromSkeletonCache() {
    subjects.forEach(s => {
        const label = document.getElementById('subj-' + s.id);
        if (!label) return;
        const trials = cachedSkeletonStatus[s.id];
        let done = false;
        if (Array.isArray(trials) && trials.length > 0) {
            done = trials.every(t => !!t.has_skeleton_v1);
        }
        label.style.borderColor = done ? 'var(--green)' : '';
        label.style.background = done ? 'rgba(76,175,80,0.08)' : '';
    });
}

function _colorSubjectsFromHrnetCache(trialFlag) {
    // ``trialFlag`` is the per-trial boolean key in cachedHrnetStatus
    // entries to AND across all trials -- e.g. ``has_hrnet_output`` or
    // ``has_mp_npz`` / ``has_mp_reverse_npz``.
    subjects.forEach(s => {
        const label = document.getElementById('subj-' + s.id);
        if (!label) return;
        const trials = cachedHrnetStatus[s.id];
        let done = false;
        if (Array.isArray(trials) && trials.length > 0) {
            done = trials.every(t => !!t[trialFlag]);
        }
        label.style.borderColor = done ? 'var(--green)' : '';
        label.style.background = done ? 'rgba(76,175,80,0.08)' : '';
    });
}

let _hrnetStatusAllInFlight = null;
async function _ensureHrnetStatusForAllSubjects() {
    // Backfill cachedHrnetStatus for every subject we don't already have
    // a result for.  One batched HTTP call.  Idempotent + de-duped via
    // ``_hrnetStatusAllInFlight`` so concurrent renders don't pile up
    // requests.
    if (_hrnetStatusAllInFlight) return _hrnetStatusAllInFlight;
    const ids = subjects.map(s => s.id).filter(id => !cachedHrnetStatus[id]);
    if (!ids.length) return null;
    _hrnetStatusAllInFlight = (async () => {
        try {
            const data = await API.get(
                `/api/analyze/hrnet/job-status?subject_ids=${ids.join(',')}`);
            for (const sid of ids) {
                cachedHrnetStatus[sid] = (data?.subjects?.[sid]?.trials) || [];
            }
        } catch {
            for (const sid of ids) cachedHrnetStatus[sid] = [];
        } finally {
            _hrnetStatusAllInFlight = null;
        }
    })();
    return _hrnetStatusAllInFlight;
}

// ── Trial selector ──────────────────────────────────────
function isTrialStep() {
    return TRIAL_STEPS.has(selectedStep);
}

async function updateTrialSection() {
    const section = document.getElementById('trialSection');
    if (!section) return;

    if (!isTrialStep()) {
        section.style.display = 'none';
        return;
    }

    const checkedSubjects = [];
    document.querySelectorAll('#subjectGrid input[type=checkbox]:checked').forEach(cb => {
        checkedSubjects.push({ id: parseInt(cb.value), name: cb.dataset.name });
    });

    if (checkedSubjects.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = '';

    // Toggle HRnet-only "Use bounding boxes" row + ineligibility banner.
    const useBboxRow = document.getElementById('useBboxRow');
    const banner = document.getElementById('hrnetIneligibleBanner');
    if (useBboxRow) useBboxRow.style.display = (selectedStep === 'hrnet') ? 'flex' : 'none';
    if (banner && selectedStep !== 'hrnet') banner.style.display = 'none';

    if (selectedStep === 'skeleton_v1') {
        // Per-trial skeleton_v1.npz status is fetched separately (one
        // endpoint per subject).  The deidentify /trials endpoint is
        // still the source of the trial LIST; the skeleton status is
        // merged in by trial_idx in renderTrialGrid.
        const ids = checkedSubjects.map(s => s.id);
        await ensureSkeletonStatusForSubjects(ids);
        await Promise.all(checkedSubjects.map(async (s) => {
            if (!cachedTrialData[s.id]) {
                try {
                    const data = await API.get(`/api/deidentify/${s.id}/trials`);
                    cachedTrialData[s.id] = data.trials || [];
                } catch (e) {
                    cachedTrialData[s.id] = [];
                }
            }
        }));
    } else if (selectedStep === 'hrnet' || selectedStep === 'mediapipe') {
        // Both steps consume the same /hrnet/job-status payload (which
        // includes per-trial ``has_hrnet_output`` *and* ``has_mp_npz`` /
        // ``has_mp_reverse_npz``).  Reuse ``cachedHrnetStatus`` as the
        // shared cache.
        const idsNeeded = checkedSubjects
            .filter(s => !cachedHrnetStatus[s.id])
            .map(s => s.id);
        if (idsNeeded.length) {
            try {
                const data = await API.get(`/api/analyze/hrnet/job-status?subject_ids=${idsNeeded.join(',')}`);
                for (const sid of idsNeeded) {
                    cachedHrnetStatus[sid] = (data?.subjects?.[sid]?.trials) || [];
                }
            } catch (e) {
                for (const sid of idsNeeded) cachedHrnetStatus[sid] = [];
            }
        }
    } else {
        // Per-step trial fetch — currently only deidentify uses this path.
        await Promise.all(checkedSubjects.map(async (s) => {
            if (!cachedTrialData[s.id]) {
                try {
                    const data = await API.get(`/api/deidentify/${s.id}/trials`);
                    cachedTrialData[s.id] = data.trials || [];
                } catch (e) {
                    cachedTrialData[s.id] = [];
                }
            }
        }));
    }

    renderTrialGrid(checkedSubjects);
}

function renderTrialGrid(checkedSubjects) {
    const grid = document.getElementById('trialGrid');
    if (!grid) return;

    // Preserve unchecked state from existing UI before re-render
    const prevUnchecked = new Set();
    grid.querySelectorAll('input[type=checkbox]:not(:checked)').forEach(cb => {
        prevUnchecked.add(`${cb.dataset.subjectId}-${cb.dataset.trialIdx}`);
    });

    if (checkedSubjects.length === 0) {
        grid.innerHTML = '<span class="empty-state">Select subjects above</span>';
        const banner = document.getElementById('hrnetIneligibleBanner');
        if (banner) banner.style.display = 'none';
        return;
    }

    const showHeader = checkedSubjects.length > 1;
    const useBbox = !!document.getElementById('useBboxCb')?.checked;
    let html = '';
    let ineligibleCount = 0;

    const mpReverse = !!document.getElementById('mpReverseCb')?.checked;
    for (const s of checkedSubjects) {
        const trials = (selectedStep === 'hrnet' || selectedStep === 'mediapipe')
            ? (cachedHrnetStatus[s.id] || [])
            : (cachedTrialData[s.id] || []);
        const prefix = s.name + '_';

        if (showHeader) {
            html += `<div class="trial-group"><div class="trial-group-label">${s.name}</div><div class="trial-items">`;
        } else {
            html += `<div class="trial-items">`;
        }

        for (const t of trials) {
            const shortName = t.trial_name.startsWith(prefix)
                ? t.trial_name.slice(prefix.length)
                : t.trial_name;

            // Step-specific cell state.
            let colorClass = '';
            let title = '';
            let disabled = false;
            let extraStyle = '';
            let initialChecked = true;

            if (selectedStep === 'hrnet') {
                if (t.has_hrnet_output) {
                    colorClass = 'done';   // green — output already exists
                    title = 'HRnet output already exists for this trial';
                } else if (useBbox && t.has_saved_bbox) {
                    // Saved-bbox eligible — render in red (no heatmap yet)
                    // with a blue tint to indicate the saved-bbox source.
                    extraStyle = 'border-color:#e53935;background:rgba(229,57,53,0.10);';
                    title = 'No HRnet output yet — saved bounding box will be used';
                } else if (useBbox && !t.has_saved_bbox && !t.has_mp_labels) {
                    disabled = true;
                    initialChecked = false;
                    extraStyle = 'opacity:0.4;pointer-events:none;';
                    title = 'No saved bbox and no MediaPipe labels — run MediaPipe first or save a bbox';
                    ineligibleCount++;
                } else {
                    // No HRnet output yet — colour red so the user
                    // immediately sees which trials still need a run.
                    extraStyle = 'border-color:#e53935;background:rgba(229,57,53,0.10);';
                    title = useBbox
                        ? 'No HRnet output yet — default bbox will be derived from MediaPipe labels'
                        : 'No HRnet output yet — will run on full per-camera frame';
                }
            } else if (selectedStep === 'mediapipe') {
                // Green when the per-trial MP npz exists (forward pass by
                // default, reverse pass when "Run in reverse" is checked).
                const hasNpz = mpReverse ? t.has_mp_reverse_npz : t.has_mp_npz;
                if (hasNpz) {
                    colorClass = 'done';
                    title = mpReverse
                        ? 'mediapipe_reverse.npz already exists for this trial'
                        : 'mediapipe.npz already exists for this trial';
                } else {
                    extraStyle = 'border-color:#e53935;background:rgba(229,57,53,0.10);';
                    title = mpReverse
                        ? 'No mediapipe_reverse.npz yet — will run reverse pass on this trial'
                        : 'No mediapipe.npz yet — will run on this trial';
                }
            } else if (selectedStep === 'skeleton_v1') {
                const skelTrials = cachedSkeletonStatus[s.id] || [];
                const skelEntry = skelTrials.find(x => x.trial_idx === t.trial_idx);
                const hasNpz = !!(skelEntry && skelEntry.has_skeleton_v1);
                if (hasNpz) {
                    colorClass = 'done';
                    title = 'skeleton_v1.npz already exists for this trial';
                    initialChecked = false;  // skip re-running by default
                } else {
                    extraStyle = 'border-color:#e53935;background:rgba(229,57,53,0.10);';
                    title = 'No skeleton_v1.npz yet — Submit will run the fit';
                }
            } else if (selectedStep === 'preproc') {
                // No per-trial bake-status flag yet — colour everything
                // neutral.  (We could later add ``t.has_stable_mp4`` from
                // the API and mark those green like 'done'.)
                colorClass = t.has_stable_mp4 ? 'done' : '';
                title = t.has_stable_mp4
                    ? 'stable.mp4 already exists for this trial'
                    : 'No preproc outputs yet';
            } else {
                // deidentify (existing behavior)
                colorClass = (!t.has_faces || t.has_blurred) ? 'done' : 'needs-deident';
            }

            const checkedAttr = initialChecked ? 'checked' : '';
            const labelChecked = initialChecked ? 'checked' : '';
            html += `<label class="trial-item ${colorClass} ${labelChecked}"
                           id="trial-${s.id}-${t.trial_idx}"
                           title="${title}"
                           style="${extraStyle}"
                           ${disabled ? '' : 'onclick="toggleTrialLabel(this)"'}>
                        <input type="checkbox" ${checkedAttr} ${disabled ? 'disabled' : ''}
                               data-subject-id="${s.id}"
                               data-trial-idx="${t.trial_idx}"
                               data-trial-name="${t.trial_name}"
                               style="cursor:pointer;">
                        <span>${shortName}</span>
                    </label>`;
        }

        html += showHeader ? `</div></div>` : `</div>`;
    }

    grid.innerHTML = html;

    // Restore previously unchecked trials (only for enabled cells)
    grid.querySelectorAll('input[type=checkbox]:not([disabled])').forEach(cb => {
        const key = `${cb.dataset.subjectId}-${cb.dataset.trialIdx}`;
        if (prevUnchecked.has(key)) {
            cb.checked = false;
            cb.closest('.trial-item')?.classList.remove('checked');
        }
    });

    // HRnet ineligibility banner.
    const banner = document.getElementById('hrnetIneligibleBanner');
    if (banner) {
        if (selectedStep === 'hrnet' && useBbox && ineligibleCount > 0) {
            banner.style.display = 'block';
            const noun = ineligibleCount === 1 ? 'trial' : 'trials';
            banner.textContent = `${ineligibleCount} ${noun} disabled: HRnet needs either a saved bounding box or MediaPipe labels (to derive a default bbox). Run MediaPipe first or save a bbox on the Auto page — or uncheck "Use bounding boxes" to run on the full frame.`;
        } else {
            banner.style.display = 'none';
        }
    }
}

function toggleTrialLabel(label) {
    const cb = label.querySelector('input[type=checkbox]');
    setTimeout(() => {
        label.classList.toggle('checked', cb.checked);
    }, 0);
}

function toggleSubjectLabel(label) {
    const cb = label.querySelector('input[type=checkbox]');
    // Toggle happens naturally; just update styling
    setTimeout(() => {
        label.classList.toggle('checked', cb.checked);
        updateTrialSection();
    }, 0);
}

function selectAllSubjects() {
    document.querySelectorAll('#subjectGrid input[type=checkbox]').forEach(cb => {
        cb.checked = true;
        cb.parentElement.classList.add('checked');
    });
    updateTrialSection();
}

// Select subjects (and, where the step is per-trial, individual trials)
// whose output for the current job type doesn't exist locally yet.  Uses
// the same source data as the green/red coloring:
//   - HRnet step → cachedHrnetStatus[sid].trials[i].has_hrnet_output
//   - Other steps → subject.has_<step> property (no per-trial info, so
//     entire subject is selected when the property is false)
async function selectIncompleteSubjects() {
    const step = selectedStep;
    if (!step) { alert('Pick a job type first.'); return; }

    if (step === 'hrnet' || step === 'mediapipe') {
        // Both steps consume cachedHrnetStatus.  For MP, the
        // "incomplete" flag flips between has_mp_npz and
        // has_mp_reverse_npz based on the Run-in-reverse checkbox so
        // the user can re-run the missing-pass selection without
        // changing job type.
        const trialFlag = step === 'hrnet'
            ? 'has_hrnet_output'
            : (document.getElementById('mpReverseCb')?.checked
                ? 'has_mp_reverse_npz' : 'has_mp_npz');
        await _ensureHrnetStatusForAllSubjects();
        const incompleteIds = new Set();
        subjects.forEach(s => {
            const trials = cachedHrnetStatus[s.id];
            if (!Array.isArray(trials) || trials.length === 0) {
                // No status data → assume incomplete (safer to include).
                incompleteIds.add(s.id);
                return;
            }
            if (trials.some(t => !t[trialFlag])) incompleteIds.add(s.id);
        });
        // Toggle subject checkboxes.
        document.querySelectorAll('#subjectGrid input[type=checkbox]').forEach(cb => {
            const sid = parseInt(cb.value);
            const want = incompleteIds.has(sid);
            cb.checked = want;
            cb.parentElement.classList.toggle('checked', want);
        });
        // After the trial grid renders, uncheck per-trial boxes whose
        // output already exists.  updateTrialSection is async — wait
        // for it before mutating the rendered checkboxes.
        await updateTrialSection();
        document.querySelectorAll('#trialGrid input[type=checkbox]').forEach(cb => {
            const sid = parseInt(cb.dataset.subjectId);
            const tidx = parseInt(cb.dataset.trialIdx);
            const trials = cachedHrnetStatus[sid] || [];
            const t = trials.find(tr => tr.trial_idx === tidx);
            const wanted = !(t && t[trialFlag]);
            cb.checked = wanted && !cb.disabled;
            const item = cb.closest('.trial-item');
            if (item) item.classList.toggle('checked', cb.checked);
        });
        return;
    }

    // Non-HRnet steps: subject-level boolean from /api/subjects.
    const prop = STEP_DONE_MAP[step];
    document.querySelectorAll('#subjectGrid input[type=checkbox]').forEach(cb => {
        const sid = parseInt(cb.value);
        const s = subjects.find(x => x.id === sid);
        const done = !!(s && prop && s[prop]);
        cb.checked = !done;
        cb.parentElement.classList.toggle('checked', !done);
    });
    updateTrialSection();
}

function clearSubjects() {
    document.querySelectorAll('#subjectGrid input[type=checkbox]').forEach(cb => {
        cb.checked = false;
        cb.parentElement.classList.remove('checked');
    });
    updateTrialSection();
}

function renderRedownloadSubjects() {
    // Re-download section removed — this is a no-op now
}

// ── Submit job ──────────────────────────────────────────
// Submit Batch — opt into the long-lived remote runner / per-subject fan-out.
//
//   * HRnet   → sets _useBatchRunnerForNextSubmit, then routes through the
//               normal submitJob() path so the queue manager picks up
//               extra_params._use_batch_runner=true and dispatches via
//               dispatch_remote_batch (single long-lived runner that
//               survives local-server restarts).
//   * MediaPipe → posts one /api/remote/launch per selected subject so
//                  each subject appears as its own cancellable queue row.
//                  The forward queue manager's MP dispatch only honours
//                  ``subject_names[0]``, so the legacy multi-subject
//                  submit path silently drops every subject after the
//                  first.  Per-subject fan-out side-steps that and gives
//                  the user one row per subject in the queue.  The
//                  Run-in-reverse checkbox is forwarded to every job.
let _useBatchRunnerForNextSubmit = false;
async function submitBatch() {
    // Skel Fit v1 is a local CPU job — submitJob's skeleton_v1 branch
    // already packs every selected trial into a single queue row with
    // per-trial outcome chips, which is the same UX the remote MP
    // batch provides.  Route through it regardless of the current
    // target so the user doesn't have to think about it.
    if (selectedStep === 'skeleton_v1' || selectedStep === 'stereo_correct') {
        await submitJob();
        return;
    }
    if (getExecutionTarget() !== 'remote') {
        alert('Submit Batch only runs on the remote server.  Switch the execution target to Remote.');
        return;
    }

    if (selectedStep === 'hrnet') {
        _useBatchRunnerForNextSubmit = true;
        try {
            await submitJob();
        } finally {
            _useBatchRunnerForNextSubmit = false;
        }
        return;
    }

    if (selectedStep === 'mediapipe') {
        // Single-row submission: one parent MP job for every selected
        // subject (derived from the checked trial chips).  Matches
        // HRnet's Submit-Batch UX -- one queue row, one log, one
        // progress bar.  The queue manager's MP branch then hands the
        // multi-subject set to ``remote_preprocess_batch`` which
        // processes them sequentially on the remote.
        const trialChecks = Array.from(
            document.querySelectorAll('#trialGrid input[type=checkbox]:checked:not([disabled])')
        );
        if (trialChecks.length === 0) {
            alert('Select at least one trial.');
            return;
        }
        const subjectIdSet = new Set();
        const subjectIdToName = new Map();
        for (const cb of trialChecks) {
            const sid = parseInt(cb.dataset.subjectId);
            subjectIdSet.add(sid);
            const subjName = cb.closest('.trial-item')?.parentElement
                ?.parentElement?.querySelector('.trial-group-label')?.textContent
                || (cb.dataset.subjectName);
            if (subjName) subjectIdToName.set(sid, subjName);
        }
        // Fill in missing names from the master subjects list.
        for (const sid of subjectIdSet) {
            if (!subjectIdToName.has(sid)) {
                const s = subjects.find(x => x.id === sid);
                if (s) subjectIdToName.set(sid, s.name);
            }
        }
        const subjIds = [...subjectIdSet];
        const subjNames = subjIds.map(id => subjectIdToName.get(id)).filter(Boolean);
        if (!subjIds.length) {
            alert('No subjects resolved from trial selection.');
            return;
        }
        const revCb = document.getElementById('mpReverseCb');
        const ubCb  = document.getElementById('mpUseBboxCb');
        const stCb  = document.getElementById('mpStaticCb');
        const reverse = !!(revCb && revCb.checked);
        const useBbox = !(ubCb && !ubCb.checked);
        const staticMode = !!(stCb && stCb.checked);
        const extra = { _use_batch_runner: true };
        if (reverse) extra.reverse = true;
        if (!useBbox) extra.use_bbox = false;
        if (staticMode) extra.static_image_mode = true;
        try {
            await API.post('/api/remote/launch', {
                job_type: 'mediapipe',
                subject_ids: subjIds,
                subjects: subjNames,
                execution_target: getExecutionTarget(),
                extra_params: extra,
            });
            for (const id of subjIds) delete cachedHrnetStatus[id];
            clearSubjects();
            refreshQueue();
        } catch (e) {
            alert('Error: ' + e.message);
        }
        return;
    }

    alert('Submit Batch currently supports HRnet and MediaPipe.  ' +
          'Use Submit Processing Job for ' + selectedStep + '.');
}

async function submitJob() {
    const jobType = selectedStep;
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

    // Per-trial steps: submit one job per selected trial
    if (TRIAL_STEPS.has(jobType)) {
        const trialChecks = Array.from(
            document.querySelectorAll('#trialGrid input[type=checkbox]:checked:not([disabled])')
        );
        if (trialChecks.length === 0) {
            alert('Select at least one trial.');
            return;
        }
        try {
            if (jobType === 'preproc') {
                // Per-trial preproc: same one-job-many-trials pattern as
                // deidentify.  ``extra_params.trials = [{subject_name,
                // trial_idx, trial_name}]`` is iterated by the queue
                // manager (local branch) or shipped to remote_preproc as
                // a single bundle (remote branch).
                const trialEntries = [];
                const subjectIdSet = new Set();
                const subjectIdToName = new Map();
                for (const cb of trialChecks) {
                    const subjectId = parseInt(cb.dataset.subjectId);
                    const trialIdx = parseInt(cb.dataset.trialIdx);
                    const subjName = cb.closest('.trial-item')?.parentElement
                        ?.parentElement?.querySelector('.trial-group-label')?.textContent
                        || (subjectNames.length === 1 ? subjectNames[0] : null);
                    const fullStem = cb.dataset.trialName || '';
                    const shortTrial = fullStem.includes('_')
                        ? fullStem.slice(fullStem.indexOf('_') + 1)
                        : fullStem;
                    subjectIdSet.add(subjectId);
                    if (subjName) subjectIdToName.set(subjectId, subjName);
                    trialEntries.push({
                        subject_name: subjName,
                        trial_idx: trialIdx,
                        trial_name: shortTrial,
                    });
                }
                if (!trialEntries.length) return;
                const subjIds = [...subjectIdSet];
                const subjNames = subjIds.map(id => subjectIdToName.get(id)).filter(Boolean);
                await API.post('/api/remote/launch', {
                    job_type: 'preproc',
                    subject_ids: subjIds,
                    subjects: subjNames,
                    execution_target: executionTarget,
                    extra_params: {
                        trials: trialEntries,
                        trial_name: (() => {
                            if (trialEntries.length === 1) return trialEntries[0].trial_name;
                            if (trialEntries.length >= 5) return `${trialEntries.length} trials`;
                            const multiSubj = subjIds.length > 1;
                            return trialEntries.map(t =>
                                multiSubj ? `${t.subject_name} ${t.trial_name}` : t.trial_name
                            ).join(', ');
                        })(),
                    },
                });
            } else if (jobType === 'deidentify') {
                // Single batched submit: collect every (subject, trial)
                // selection into one ``extra_params.trials`` array — same
                // pattern as HRnet so the queue manager runs them
                // sequentially under one parent job + log + progress bar
                // and the Jobs page shows one row instead of N.
                const trialEntries = [];
                const subjectIdSet = new Set();
                const subjectIdToName = new Map();
                for (const cb of trialChecks) {
                    const subjectId = parseInt(cb.dataset.subjectId);
                    const trialIdx = parseInt(cb.dataset.trialIdx);
                    const subjName = cb.closest('.trial-item')?.parentElement
                        ?.parentElement?.querySelector('.trial-group-label')?.textContent
                        || (subjectNames.length === 1 ? subjectNames[0] : null);
                    const fullStem = cb.dataset.trialName || '';
                    const shortTrial = fullStem.includes('_')
                        ? fullStem.slice(fullStem.indexOf('_') + 1)
                        : fullStem;
                    subjectIdSet.add(subjectId);
                    if (subjName) subjectIdToName.set(subjectId, subjName);
                    trialEntries.push({
                        subject_name: subjName,
                        trial_idx: trialIdx,
                        trial_name: shortTrial,
                    });
                }
                if (!trialEntries.length) return;
                const subjIds = [...subjectIdSet];
                const subjNames = subjIds.map(id => subjectIdToName.get(id)).filter(Boolean);
                await API.post('/api/remote/launch', {
                    job_type: 'deidentify',
                    subject_ids: subjIds,
                    subjects: subjNames,
                    execution_target: executionTarget,
                    extra_params: {
                        trials: trialEntries,
                        // Display label for the trial badge — same logic
                        // as HRnet: list when small, "N trials" when big.
                        trial_name: (() => {
                            if (trialEntries.length === 1) return trialEntries[0].trial_name;
                            if (trialEntries.length >= 5) return `${trialEntries.length} trials`;
                            const multiSubj = subjIds.length > 1;
                            return trialEntries.map(t =>
                                multiSubj ? `${t.subject_name} ${t.trial_name}` : t.trial_name
                            ).join(', ');
                        })(),
                    },
                });
            } else if (jobType === 'mediapipe') {
                // Per-trial MP fan-out: one job per (subject, trial) so the
                // user gets a queue row per trial that maps cleanly onto
                // the per-trial mediapipe.npz output layout.
                const revCb = document.getElementById('mpReverseCb');
                const ubCb  = document.getElementById('mpUseBboxCb');
                const stCb  = document.getElementById('mpStaticCb');
                const reverse = !!(revCb && revCb.checked);
                const useBbox = !(ubCb && !ubCb.checked);
                const staticMode = !!(stCb && stCb.checked);
                let ok = 0;
                let firstError = null;
                const submittedSubjectIds = new Set();
                for (const cb of trialChecks) {
                    const subjectId = parseInt(cb.dataset.subjectId);
                    const trialIdx = parseInt(cb.dataset.trialIdx);
                    const subjName = cb.closest('.trial-item')?.parentElement
                        ?.parentElement?.querySelector('.trial-group-label')?.textContent
                        || (subjectNames.length === 1 ? subjectNames[0] : null);
                    const fullStem = cb.dataset.trialName || '';
                    const shortTrial = fullStem.includes('_')
                        ? fullStem.slice(fullStem.indexOf('_') + 1)
                        : fullStem;
                    const extra = { trial_idx: trialIdx, trial_name: shortTrial };
                    if (reverse) extra.reverse = true;
                    if (!useBbox) extra.use_bbox = false;
                    if (staticMode) extra.static_image_mode = true;
                    try {
                        await API.post('/api/remote/launch', {
                            job_type: 'mediapipe',
                            subject_ids: [subjectId],
                            subjects: subjName ? [subjName] : [],
                            execution_target: executionTarget,
                            extra_params: extra,
                        });
                        submittedSubjectIds.add(subjectId);
                        ok++;
                    } catch (e) {
                        if (!firstError) firstError = e;
                    }
                }
                // Bust cache so the green/red coloring updates after the
                // jobs finish.
                for (const id of submittedSubjectIds) delete cachedHrnetStatus[id];
                if (firstError && ok === 0) {
                    alert('All MediaPipe submissions failed: ' + firstError.message);
                } else if (firstError) {
                    alert(`Submitted ${ok}/${trialChecks.length} MediaPipe jobs. ` +
                          `First failure: ${firstError.message}`);
                }
            } else if (jobType === 'hrnet') {
                // Single batched submit: collect every (subject, trial)
                // selection into one ``extra_params.trials`` array.  The
                // queue manager iterates them in one parent job so the
                // user sees a single progress bar + log + history row.
                const useBbox = !!document.getElementById('useBboxCb')?.checked;
                const trialEntries = [];
                const subjectIdSet = new Set();
                const subjectIdToName = new Map();
                for (const cb of trialChecks) {
                    const subjectId = parseInt(cb.dataset.subjectId);
                    const trialIdx = parseInt(cb.dataset.trialIdx);
                    const subjName = cb.closest('.trial-item')?.parentElement
                        ?.parentElement?.querySelector('.trial-group-label')?.textContent
                        || (subjectNames.length === 1 ? subjectNames[0] : null);
                    const fullStem = cb.dataset.trialName || '';
                    const shortTrial = fullStem.includes('_')
                        ? fullStem.slice(fullStem.indexOf('_') + 1)
                        : fullStem;
                    subjectIdSet.add(subjectId);
                    if (subjName) subjectIdToName.set(subjectId, subjName);

                    const entry = {
                        subject_name: subjName,
                        trial_idx: trialIdx,
                        trial_name: shortTrial,
                        use_bbox: useBbox,
                    };
                    // Don't pre-fetch a static saved bbox — when omitted, the
                    // remote script computes a tight per-frame bbox from
                    // MediaPipe landmarks (preferred for high-motion trials).
                    // A static bbox would just be broadcast across all frames,
                    // defeating per-frame computation.  Saved-bbox fallback
                    // only kicks in server-side when MP isn't available.
                    trialEntries.push(entry);
                }
                if (!trialEntries.length) return;

                const subjIds = [...subjectIdSet];
                const subjNames = subjIds.map(id => subjectIdToName.get(id)).filter(Boolean);
                await API.post('/api/remote/launch', {
                    job_type: 'hrnet',
                    subject_ids: subjIds,
                    subjects: subjNames,
                    execution_target: executionTarget,
                    gpu_index: gpuIndex,
                    extra_params: {
                        trials: trialEntries,
                        use_bbox: useBbox,
                        // Display label for the trial badge.  For batches of
                        // <5 trials, list each trial individually so the user
                        // sees exactly what's queued; collapse to "N trials"
                        // only for larger batches to keep the badge compact.
                        // Include subject prefix when multiple subjects are
                        // involved so duplicates (e.g. Con01 L1 + Con02 L1)
                        // are distinguishable.
                        trial_name: (() => {
                            if (trialEntries.length === 1) return trialEntries[0].trial_name;
                            if (trialEntries.length >= 5) return `${trialEntries.length} trials`;
                            const multiSubj = subjIds.length > 1;
                            return trialEntries.map(t =>
                                multiSubj ? `${t.subject_name} ${t.trial_name}` : t.trial_name
                            ).join(', ');
                        })(),
                        // Set by the new "Submit Batch" button — routes
                        // through dispatch_remote_batch + poll_remote_batch
                        // (long-lived remote runner that survives local
                        // server restarts).  Default false → legacy
                        // per-trial dispatch.
                        _use_batch_runner: !!_useBatchRunnerForNextSubmit,
                    },
                });
                // Bust per-subject HRnet status cache so the refresh
                // shows in-flight state.
                for (const id of subjIds) delete cachedHrnetStatus[id];
            } else if (jobType === 'skeleton_v1') {
                // Single batched submit: one queue row, one log, one
                // progress bar, per-trial outcome chips driven by the
                // queue manager updating jobs.params_json.trials[i].
                const trialEntries = [];
                const subjectIdSet = new Set();
                const subjectIdToName = new Map();
                for (const cb of trialChecks) {
                    const subjectId = parseInt(cb.dataset.subjectId);
                    const trialIdx = parseInt(cb.dataset.trialIdx);
                    const subjName = cb.closest('.trial-item')?.parentElement
                        ?.parentElement?.querySelector('.trial-group-label')?.textContent
                        || (subjectNames.length === 1 ? subjectNames[0] : null);
                    const fullStem = cb.dataset.trialName || '';
                    const shortTrial = fullStem.includes('_')
                        ? fullStem.slice(fullStem.indexOf('_') + 1)
                        : fullStem;
                    subjectIdSet.add(subjectId);
                    if (subjName) subjectIdToName.set(subjectId, subjName);
                    trialEntries.push({
                        subject_name: subjName,
                        trial_idx: trialIdx,
                        trial_name: shortTrial,
                    });
                }
                if (!trialEntries.length) return;
                const subjIds = [...subjectIdSet];
                const subjNames = subjIds.map(id => subjectIdToName.get(id)).filter(Boolean);
                const w_reproj = parseFloat(document.getElementById('skelfitSliderReproj')?.value ?? 1);
                const w_bone   = parseFloat(document.getElementById('skelfitSliderBone')?.value   ?? 5);
                const w_smooth = parseFloat(document.getElementById('skelfitSliderSmooth')?.value ?? 1);
                const snap_bones = !!document.getElementById('skelfitSnapBones')?.checked;
                const accel_k = parseFloat(document.getElementById('skelfitSliderAccelK')?.value ?? 6);
                const bone_k  = parseFloat(document.getElementById('skelfitSliderBoneK')?.value  ?? 6);
                const k_max   = parseInt(document.getElementById('skelfitSliderKmax')?.value     ?? 30);
                await API.post('/api/remote/launch', {
                    job_type: 'skeleton_v1',
                    subject_ids: subjIds,
                    subjects: subjNames,
                    execution_target: executionTarget,
                    extra_params: {
                        trials: trialEntries,
                        w_reproj, w_bone, w_smooth, snap_bones,
                        accel_k, bone_k, k_max,
                        // Joint-angle weight no longer exposed — keep
                        // the optimizer from running that term.
                        w_angle: 0,
                        trial_name: (() => {
                            if (trialEntries.length === 1) return trialEntries[0].trial_name;
                            if (trialEntries.length >= 5) return `${trialEntries.length} trials`;
                            const multiSubj = subjIds.length > 1;
                            return trialEntries.map(t =>
                                multiSubj ? `${t.subject_name} ${t.trial_name}` : t.trial_name
                            ).join(', ');
                        })(),
                    },
                });
                // Bust per-subject skeleton-status cache so the refresh
                // picks up new npz files when the batch finishes.
                for (const id of subjIds) delete cachedSkeletonStatus[id];
            } else if (jobType === 'stereo_correct') {
                // Stereo Correct: one queue row, per-trial outcome chips.
                // Same wiring shape as skeleton_v1 — read the per-trial
                // checkbox grid + the global mode/dilate/gauss settings,
                // POST one launch request batched across trials.  The
                // queue manager loops them, calling run_stereo_align
                // per trial.
                const trialEntries = [];
                const subjectIdSet = new Set();
                const subjectIdToName = new Map();
                for (const cb of trialChecks) {
                    const subjectId = parseInt(cb.dataset.subjectId);
                    const trialIdx = parseInt(cb.dataset.trialIdx);
                    const subjName = cb.closest('.trial-item')?.parentElement
                        ?.parentElement?.querySelector('.trial-group-label')?.textContent
                        || (subjectNames.length === 1 ? subjectNames[0] : null);
                    const fullStem = cb.dataset.trialName || '';
                    const shortTrial = fullStem.includes('_')
                        ? fullStem.slice(fullStem.indexOf('_') + 1)
                        : fullStem;
                    subjectIdSet.add(subjectId);
                    if (subjName) subjectIdToName.set(subjectId, subjName);
                    trialEntries.push({
                        subject_name: subjName,
                        trial_idx: trialIdx,
                        trial_name: shortTrial,
                    });
                }
                if (!trialEntries.length) return;
                const subjIds = [...subjectIdSet];
                const subjNames = subjIds.map(id => subjectIdToName.get(id)).filter(Boolean);
                const _modeEl = document.querySelector('input[name="stereoCorrectMode"]:checked');
                const mode = _modeEl ? _modeEl.value : 'image';
                const mask_dilate_px = parseInt(
                    document.getElementById('stereoCorrectDilateSlider')?.value ?? 10
                );
                const gauss_center_weight = parseFloat(
                    document.getElementById('stereoCorrectGaussSlider')?.value ?? 0
                ) / 100;
                await API.post('/api/remote/launch', {
                    job_type: 'stereo_correct',
                    subject_ids: subjIds,
                    subjects: subjNames,
                    execution_target: executionTarget,
                    extra_params: {
                        trials: trialEntries,
                        mode, mask_dilate_px, gauss_center_weight,
                        trial_name: (() => {
                            if (trialEntries.length === 1) return trialEntries[0].trial_name;
                            if (trialEntries.length >= 5) return `${trialEntries.length} trials`;
                            const multiSubj = subjIds.length > 1;
                            return trialEntries.map(t =>
                                multiSubj ? `${t.subject_name} ${t.trial_name}` : t.trial_name
                            ).join(', ');
                        })(),
                    },
                });
            }
            clearSubjects();
            refreshQueue();
        } catch (e) {
            alert('Error: ' + e.message);
        }
        return;
    }

    // MediaPipe options forwarded via extra_params:
    //   reverse  → frames fed in reverse temporal order (Reverse npz)
    //   use_bbox → honour saved per-trial crop boxes; set to false to
    //              force MP to scan the full camera-half frame
    let extraParams;
    if (jobType === 'mediapipe') {
        const rev = document.getElementById('mpReverseCb');
        const ub  = document.getElementById('mpUseBboxCb');
        const st  = document.getElementById('mpStaticCb');
        const useBbox = !!(ub && ub.checked);
        const obj = {};
        if (rev && rev.checked) obj.reverse = true;
        // Default behaviour is "use bbox" -- only forward the flag
        // when it differs from the default so older queue manager
        // builds that don't recognize the field keep working.
        if (!useBbox) obj.use_bbox = false;
        if (st && st.checked) obj.static_image_mode = true;
        if (Object.keys(obj).length) extraParams = obj;
    }

    try {
        const result = await API.post('/api/remote/launch', {
            job_type: jobType,
            subject_ids: subjectIds,
            subjects: subjectNames,
            execution_target: executionTarget,
            gpu_index: gpuIndex,
            ...(extraParams ? { extra_params: extraParams } : {}),
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
    queueStream.onmessage = async (event) => {
        const state = JSON.parse(event.data);
        // The SSE state covers queue-managed jobs only.  Merge in
        // thread-based local jobs (mano_fit_v3, etc.) the same way
        // refreshQueue() does so they keep showing in the local-CPU
        // lane between SSE ticks.
        try {
            const localJobs = await API.get('/api/jobs?status=running,pending');
            const queueJobIds = new Set();
            [...(state.running || []), ...(state.cpu_queue || []), ...(state.gpu_queue || [])].forEach(item => {
                if (item.job_id) queueJobIds.add(item.job_id);
            });
            // Only thread-based LOCAL jobs (mano_fit_v3 etc.) belong in
            // the local-CPU lane.  Remote jobs are dispatched via
            // remote_hrnet_job / remote_train_monitor and would briefly
            // appear here during the upload phase (before the job_queue
            // row is linked to the jobs row), confusing the user.  Filter
            // by remote_host: 'localhost' or empty means truly local.
            // Split jobs the queue manager doesn't own (re-download jobs,
            // thread-based local jobs like mano_fit_v3) into two buckets:
            //   _localOnly  — render in the local-CPU lane.
            //   _remoteOnly — render in the matching remote lane.
            // remote_host = 'localhost' or empty means the job runs locally.
            state._localOnly = localJobs.filter(j =>
                !queueJobIds.has(j.id) &&
                (!j.remote_host || j.remote_host === 'localhost')
            );
            state._remoteOnly = localJobs.filter(j =>
                !queueJobIds.has(j.id) &&
                j.remote_host && j.remote_host !== 'localhost'
            );
        } catch { /* best-effort */ }
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
        const [state, localJobs] = await Promise.all([
            API.get('/api/remote/queue'),
            API.get('/api/jobs?status=running,pending').catch(() => []),
        ]);
        // Merge local-only jobs (not in queue) into state for rendering
        const queueJobIds = new Set();
        [...(state.running || []), ...(state.cpu_queue || []), ...(state.gpu_queue || [])].forEach(item => {
            if (item.job_id) queueJobIds.add(item.job_id);
        });
        state._localOnly = localJobs.filter(j =>
            !queueJobIds.has(j.id) &&
            (!j.remote_host || j.remote_host === 'localhost')
        );
        renderQueue(state);
        refreshPendingDownloads();
    } catch (e) {
        console.error('Failed to refresh queue:', e);
    }
}

// ── Pending-results banner (drives the Jobs-page download flag) ──────
let _pendingDownloads = [];
async function refreshPendingDownloads(force = false) {
    try {
        const [pending, prog] = await Promise.all([
            API.get(`/api/remote/pending-downloads${force ? '?force=1' : ''}`),
            API.get('/api/remote/download-progress').catch(() => ({downloads: []})),
        ]);
        _pendingDownloads = pending?.pending || [];
        renderPendingBanner();
        // If a background download is running and we aren't already
        // polling for it, attach.  Handles the navigated-away-and-back
        // case.
        const active = (prog?.downloads || []).find(d => d.status === 'running');
        if (active && !_activeDownloadPoll) {
            _pollDownloadProgress(active.download_id);
        }
    } catch (_e) {
        _pendingDownloads = [];
        renderPendingBanner();
    }
}
function renderPendingBanner() {
    let el = document.getElementById('pendingResultsBanner');
    if (!el) {
        // Inject the banner element at the top of the Queue card if absent.
        // The first `.card h2` is the Launch-Job card on this page, so find
        // the Queue card explicitly by header text.
        let host = null;
        document.querySelectorAll('.card h2').forEach(h => {
            if (!host && h.textContent.trim() === 'Queue') host = h;
        });
        if (host) {
            el = document.createElement('div');
            el.id = 'pendingResultsBanner';
            el.style.cssText = 'display:none;margin:8px 0;padding:8px 12px;background:rgba(74,158,255,0.10);border:1px solid var(--blue);border-radius:var(--radius);font-size:13px;';
            host.parentNode.insertBefore(el, host.nextSibling);
        }
    }
    if (!el) return;
    if (!_pendingDownloads.length) {
        el.style.display = 'none'; el.innerHTML = '';
        return;
    }
    // Per-file rows with Download / Ignore radios.  Each row is keyed
    // by job_id (today MP produces one file per job; finer selection
    // will be added when vision/pose/hrnet outputs ship).
    const byJob = new Map();
    for (const p of _pendingDownloads) {
        if (!byJob.has(p.job_id)) {
            byJob.set(p.job_id, {
                subject: p.subject_name, jobType: p.job_type,
                status: p.job_status, label: p.label,
            });
        }
    }
    const rows = [...byJob.entries()].map(([jid, v]) => `
        <tr>
            <td style="padding:3px 8px;font-size:12px;">${v.subject}</td>
            <td style="padding:3px 8px;font-size:11px;color:var(--text-muted);">${v.jobType}${v.status==='failed'?' (failed)':''}</td>
            <td style="padding:3px 8px;text-align:center;">
                <label style="cursor:pointer;display:inline-flex;align-items:center;gap:3px;">
                    <input type="radio" name="dl-${jid}" value="download" checked style="margin:0;cursor:pointer;">
                    <span style="font-size:11px;">Download</span>
                </label>
            </td>
            <td style="padding:3px 8px;text-align:center;">
                <label style="cursor:pointer;display:inline-flex;align-items:center;gap:3px;">
                    <input type="radio" name="dl-${jid}" value="ignore" style="margin:0;cursor:pointer;">
                    <span style="font-size:11px;">Ignore</span>
                </label>
            </td>
        </tr>
    `).join('');
    el.style.display = 'block';
    el.innerHTML = `
        <div style="margin-bottom:6px;">
            <strong>${byJob.size} result(s) ready</strong>
            <span style="color:var(--text-muted);font-size:11px;">— jobs that were running when the app last closed (or completed since)</span>
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:12px;background:var(--bg);border-radius:var(--radius);">
            <thead>
                <tr style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">
                    <th style="padding:4px 8px;text-align:left;">Subject</th>
                    <th style="padding:4px 8px;text-align:left;">Output</th>
                    <th style="padding:4px 8px;">Download</th>
                    <th style="padding:4px 8px;">Ignore</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
        <div style="display:flex;justify-content:flex-end;gap:6px;margin-top:8px;">
            <button class="btn btn-sm btn-primary" onclick="submitPendingSelections()">Submit</button>
        </div>
    `;
}

async function submitPendingSelections() {
    const banner = document.getElementById('pendingResultsBanner');
    if (!banner) return;
    const ids = [...new Set(_pendingDownloads.map(p => p.job_id))];
    const dlIds = [], ignoreIds = [];
    for (const jid of ids) {
        const sel = banner.querySelector(`input[name="dl-${jid}"]:checked`);
        if (sel?.value === 'ignore') ignoreIds.push(jid);
        else dlIds.push(jid);
    }
    const submitBtn = banner.querySelector('button.btn-primary');
    if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Submitting…'; }
    try {
        if (ignoreIds.length) {
            await API.post('/api/remote/ignore-pending', { job_ids: ignoreIds });
        }
        if (dlIds.length) {
            const res = await API.post('/api/remote/download-pending', { job_ids: dlIds });
            if (res?.download_id) _pollDownloadProgress(res.download_id);
            else await refreshPendingDownloads(true);
        } else {
            await refreshPendingDownloads(true);
        }
    } catch (e) {
        alert('Submit failed: ' + e.message);
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Submit'; }
    }
}
async function downloadAllPending() {
    const ids = [...new Set(_pendingDownloads.map(p => p.job_id))];
    try {
        const res = await API.post('/api/remote/download-pending', { job_ids: ids });
        if (res?.download_id) {
            _pollDownloadProgress(res.download_id);
        } else {
            await refreshPendingDownloads();
        }
    } catch (e) {
        alert('Download start failed: ' + e.message);
    }
}

let _activeDownloadPoll = null;
async function _pollDownloadProgress(downloadId) {
    if (_activeDownloadPoll) return;   // already polling
    _activeDownloadPoll = downloadId;
    const tick = async () => {
        try {
            const data = await API.get('/api/remote/download-progress');
            const entry = (data?.downloads || []).find(d => d.download_id === downloadId);
            if (!entry) {
                _activeDownloadPoll = null;
                await refreshPendingDownloads();
                return;
            }
            const banner = document.getElementById('pendingResultsBanner');
            const btn = banner?.querySelector('button.btn-primary');
            if (btn) {
                btn.disabled = true;
                if (entry.status === 'running') {
                    const label = entry.current_label
                        ? ` — ${entry.current_label}`
                        : '';
                    btn.textContent = `Downloading ${entry.completed + 1} of ${entry.total}${label}`;
                } else if (entry.status === 'completed') {
                    btn.textContent = `Downloaded ${entry.downloaded}`
                        + (entry.skipped ? ` (${entry.skipped} skipped)` : '');
                } else if (entry.status === 'failed') {
                    btn.textContent = `Failed: ${entry.error || 'unknown'}`;
                }
            }
            if (entry.status !== 'running') {
                // Brief pause so the final state is visible, then refresh.
                setTimeout(async () => {
                    _activeDownloadPoll = null;
                    await refreshPendingDownloads(true);   // bypass cache
                }, 1200);
                return;
            }
            setTimeout(tick, 1000);
        } catch (_e) {
            _activeDownloadPoll = null;
        }
    };
    tick();
}
async function ignoreAllPending() {
    const ids = [...new Set(_pendingDownloads.map(p => p.job_id))];
    if (!confirm(`Ignore ${ids.length} pending result set(s)? They won't be flagged again.`)) return;
    try {
        await API.post('/api/remote/ignore-pending', { job_ids: ids });
        await refreshPendingDownloads(true);   // bypass cache
    } catch (e) {
        alert('Ignore failed: ' + e.message);
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

    // Background-fetch HRnet status for trial chips in running/queued items;
    // when fresh data lands, re-render so the chips pick up green/red.
    const all = [
        ...(state.running || []),
        ...(state.cpu_queue || []),
        ...(state.gpu_queue || []),
        ...(state._localOnly || []),
    ];
    ensureHrnetStatusForJobs(all).then(updated => {
        if (updated) {
            renderLane4('localCpuLane', 'cpu', 'local-cpu', state);
            renderLane4('localGpuLane', 'gpu', 'local-gpu', state);
            renderLane4('remoteCpuLane', 'cpu', 'remote', state);
            renderLane4('remoteGpuLane', 'gpu', 'remote', state);
        }
    });

    renderHistory(state.history);
}


// ─── Lifetime job history (append-only JSONL log) ──────────────────────

async function refreshLifetimeHistory() {
    const jt = document.getElementById('lifetimeJobType')?.value || '';
    const st = document.getElementById('lifetimeStatus')?.value || '';
    const params = new URLSearchParams({ limit: '200' });
    if (jt) params.set('job_type', jt);
    if (st) params.set('status', st);
    let data;
    try {
        data = await API.get('/api/remote/history?' + params.toString());
    } catch (e) {
        document.getElementById('lifetimeContent').innerHTML =
            `<span class="empty-state">Error loading history: ${e.message}</span>`;
        return;
    }
    const summary = data?.summary || {};
    const rows = data?.rows || [];

    const summaryEl = document.getElementById('lifetimeSummary');
    if (summaryEl) {
        if (summary.count) {
            const parts = [
                `${summary.count} total`,
                `${summary.ok} ok`,
            ];
            if (summary.failed)    parts.push(`${summary.failed} failed`);
            if (summary.cancelled) parts.push(`${summary.cancelled} cancelled`);
            if (summary.median_duration_sec != null) {
                parts.push(`median ${_fmtDuration(summary.median_duration_sec)}`);
            }
            summaryEl.textContent = '— ' + parts.join(' · ');
        } else {
            summaryEl.textContent = '';
        }
    }

    const el = document.getElementById('lifetimeContent');
    if (!rows.length) {
        el.innerHTML = '<span class="empty-state">No matching records.</span>';
        return;
    }
    // Compact table.  Hover row to see full params + per-stage timings.
    const fmt = (s) => s ? s.toString().replace('T', ' ').slice(0, 19) : '';
    el.innerHTML = '<table style="width:100%;border-collapse:collapse;font-size:11px;">'
        + '<thead><tr style="text-align:left;color:var(--text-muted);">'
            + '<th style="padding:4px 6px;">Finished</th>'
            + '<th style="padding:4px 6px;">Type</th>'
            + '<th style="padding:4px 6px;">Subj</th>'
            + '<th style="padding:4px 6px;">Status</th>'
            + '<th style="padding:4px 6px;">Total</th>'
            + '<th style="padding:4px 6px;">Stages</th>'
            + '<th style="padding:4px 6px;">git</th>'
            + '<th style="padding:4px 6px;">host</th>'
        + '</tr></thead><tbody>'
        + rows.map((r, i) => {
            const dur = r.duration_sec != null ? _fmtDuration(r.duration_sec) : '—';
            const colour = r.status === 'completed' ? '#4caf50'
                         : r.status === 'failed'   ? '#e53935'
                         : r.status === 'cancelled'? '#ffa94d' : 'var(--text-muted)';
            const gitShort = r.git_version ? r.git_version.slice(0, 8) : '—';
            const subjStr = r.subject_id != null ? `#${r.subject_id}` : '—';
            return `<tr id="lifeRow_${i}" style="border-top:1px solid var(--border);cursor:pointer;"
                          onclick="_toggleLifeDetail(${i})">
                <td style="padding:4px 6px;font-family:monospace;">${fmt(r.finished_at || r.ts)}</td>
                <td style="padding:4px 6px;">${r.job_type || '—'}</td>
                <td style="padding:4px 6px;">${subjStr}</td>
                <td style="padding:4px 6px;color:${colour};">${r.status || '—'}</td>
                <td style="padding:4px 6px;">${dur}</td>
                <td style="padding:4px 6px;">${r.n_stages || 0}</td>
                <td style="padding:4px 6px;font-family:monospace;color:var(--text-muted);">${gitShort}</td>
                <td style="padding:4px 6px;color:var(--text-muted);">${r.host || '—'}</td>
            </tr>
            <tr id="lifeDetail_${i}" style="display:none;">
                <td colspan="8" style="padding:6px 12px;background:var(--bg);">
                    <pre style="margin:0;font-size:11px;white-space:pre-wrap;word-break:break-word;">${_escape(JSON.stringify(r, null, 2))}</pre>
                </td>
            </tr>`;
        }).join('')
        + '</tbody></table>';
    // Stash rows for the toggle helper.
    window._lifetimeRows = rows;
}

function _toggleLifeDetail(i) {
    const d = document.getElementById('lifeDetail_' + i);
    if (!d) return;
    d.style.display = d.style.display === 'none' ? 'table-row' : 'none';
}

function _fmtDuration(sec) {
    if (sec == null || isNaN(sec)) return '—';
    if (sec < 60)   return sec.toFixed(1) + 's';
    if (sec < 3600) return (sec / 60).toFixed(1) + 'm';
    return (sec / 3600).toFixed(2) + 'h';
}

function _escape(s) {
    return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Refresh when filters change.
document.addEventListener('DOMContentLoaded', () => {
    const jt = document.getElementById('lifetimeJobType');
    const st = document.getElementById('lifetimeStatus');
    if (jt) jt.addEventListener('change', refreshLifetimeHistory);
    if (st) st.addEventListener('change', refreshLifetimeHistory);
    // Initial load.
    refreshLifetimeHistory();
});

// Refresh after a job completes (the queue poll picks up new completions).
setInterval(() => { if (document.visibilityState === 'visible') refreshLifetimeHistory(); }, 15000);


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
    let label = labels[target] || target;
    // If the runner stashed an actual device into tmux_session, show the
    // truth on the badge — e.g. "Local GPU (mps)" or "Local GPU (cpu)".
    // Set by services/hrnet.py via device_callback → jobs.tmux_session.
    const devTag = (item.tmux_session || '').match(/^device:(.+)$/);
    if (devTag && target.startsWith('local')) {
        label += ` (${devTag[1]})`;
    }
    return `<span style="background:${color};color:white;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600;">${label}</span>`;
}

// Resolve which lane a job_type belongs to.  Strips "redownload_" prefix so
// e.g. ``redownload_hrnet`` lands in the GPU lane next to its parent type.
function _jobResource(jobType) {
    const base = (jobType || '').replace(/^redownload_/, '');
    const gpu = ['hrnet', 'train', 'refine', 'analyze_v1', 'analyze_v2'];
    return gpu.includes(base) ? 'gpu' : 'cpu';
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

    // Include thread-based jobs not owned by the queue manager.
    //   * _localOnly  → local-CPU lane only (mano_fit_v3, etc.).
    //   * _remoteOnly → remote lane matching the job's resource (gpu/cpu).
    //     This routes redownload_* jobs into the same lane as their parent
    //     job type so the user sees them in the queue they expect.
    const localOnly = (target === 'local-cpu' && state._localOnly) ? state._localOnly : [];
    const remoteOnly = (target === 'remote' && state._remoteOnly)
        ? state._remoteOnly.filter(j => _jobResource(j.job_type) === resource)
        : [];
    const extras = [...localOnly, ...remoteOnly];

    if (running.length === 0 && queued.length === 0 && extras.length === 0) {
        el.innerHTML = '<span class="empty-state">Empty</span>';
        return;
    }

    let html = '';

    // Thread-based jobs not managed by the queue manager: redownload
    // (which spans many subjects) and one-off local jobs (single-subject).
    // Pull the subject list from params_json.subjects when present so the
    // lane row shows everything the job is actually working on, not just
    // the first subject in the foreign key.
    for (const job of extras) {
        let displayName = job.subject_name || ('Subject ' + job.subject_id);
        try {
            const p = job.params_json
                ? (typeof job.params_json === 'string' ? JSON.parse(job.params_json) : job.params_json)
                : null;
            if (p && Array.isArray(p.subjects) && p.subjects.length) {
                if (p.subjects.length <= 3) {
                    displayName = p.subjects.join(', ');
                } else {
                    displayName = p.subjects.slice(0, 3).join(', ') + ` +${p.subjects.length - 3}`;
                }
            }
        } catch { /* ignore */ }
        const pct = job.progress_pct || 0;
        html += `
            <div class="queue-item" style="border-left: 3px solid var(--orange);">
                <span class="job-indicator job-running"></span>
                <span class="type">${job.job_type}</span>
                <span style="flex:1;min-width:0;display:flex;align-items:center;gap:4px;overflow:hidden;"><span class="subjects" style="flex:none;" title="${displayName}">${displayName}</span>${trialBadge(job)}</span>
                <div class="progress-bar" style="width:80px;">
                    <div class="fill" style="width:${pct}%"></div>
                </div>
                <span style="font-size:11px;">${pct.toFixed(0)}%</span>
                <button class="btn btn-sm btn-danger" onclick="cancelLocalJob(${job.id})">Cancel</button>
            </div>
        `;
    }

    // Running items (from queue)
    for (const item of running) {
        const subjects = parseSubjects(item.subject_ids);
        const pct = item.progress_pct || 0;
        const targetBadge = getExecutionTargetBadge(item);
        // Detect remote upload/download phases — suppress progress bar and
        // time estimate.  This contract (0-10% upload, 10-85% inference,
        // 85-100% download) is HRnet-specific.  MediaPipe / Vision /
        // Pose / Deidentify report their actual work progress directly,
        // so applying the heuristic to them mislabels real progress as
        // "Uploading..." (e.g. MP at 9% looks like an upload).
        // For multi-trial batches the *global* progress also passes
        // through 85-100% multiple times during normal inference, so we
        // skip the override when params_json carries a trials list.
        const isRemote = (item.execution_target || 'remote') === 'remote';
        const isHrnet = (item.job_type || '').includes('hrnet');
        let _isBatch = false;
        try {
            const _p = item.params_json ? (typeof item.params_json === 'string'
                                            ? JSON.parse(item.params_json)
                                            : item.params_json) : null;
            _isBatch = Array.isArray(_p?.trials) && _p.trials.length > 1;
        } catch (_e) { /* ignore */ }
        // Phase labels:
        //   * Any remote job at 0% → "Uploading..." (videos/MP being SCPed
        //     before the runner reports any progress).  Switches to a
        //     normal progress bar as soon as pct > 0.
        //   * HRnet single-trial only → "Downloading..." while the local
        //     side is SCPing outputs back (85-100% slice).  Skipped for
        //     batches because their global pct sweeps through 85-100%
        //     repeatedly during normal inference.
        const showHrnetDownloadPhase = isHrnet && !_isBatch && isRemote;
        // The "Uploading..." label is for the legacy single-trial
        // remote jobs whose progress jumps from 0 → mid-range when
        // the worker actually starts running.  For the new
        // preproc-batch flow (and anything else that sets
        // params.phase explicitly) we use the phase field as the
        // source of truth instead of the pct === 0 proxy: phase
        // "uploading" → "Uploading..."; phase "running" → show the
        // progress bar at whatever pct says, even when pct is 0
        // (first subject hasn't finished yet but the bake IS
        // running).
        let _explicitPhase = null;
        try {
            const _pj = item.params_json ? JSON.parse(item.params_json) : null;
            if (_pj && typeof _pj.phase === 'string') _explicitPhase = _pj.phase;
        } catch {}
        const remotePhase = (_explicitPhase === 'uploading') ? 'Uploading...'
                          : (_explicitPhase === 'running') ? null
                          : (isRemote && pct === 0) ? 'Uploading...'
                          : (showHrnetDownloadPhase && pct >= 85 && pct < 100) ? 'Downloading...'
                          : null;
        const timeInfo = remotePhase ? { display: '', tooltip: '' }
                                     : formatJobTime(item.started_at, pct, item.epoch_info);
        let epochLabel = '';
        if (!remotePhase && item.epoch_info) {
            try {
                const ei = typeof item.epoch_info === 'string' ? JSON.parse(item.epoch_info) : item.epoch_info;
                if (ei && ei.epoch > 0) epochLabel = `<span style="font-size:11px;color:var(--text-muted);margin-left:4px;">Epoch ${ei.epoch}/${ei.total}</span>`;
            } catch {}
        }
        const progressHtml = remotePhase
            ? `<span style="font-size:11px;color:var(--text-muted);font-style:italic;">${remotePhase}</span>`
            : `<div class="progress-bar" style="width:80px;"><div class="fill" style="width:${pct}%"></div></div>
               <span style="font-size:11px;">${pct.toFixed(0)}%</span>
               ${epochLabel}
               ${timeInfo.display ? `<span class="time-info" style="font-size:11px;color:var(--text-muted);margin-left:4px;" title="${timeInfo.tooltip}">${timeInfo.display}</span>` : ''}`;
        // Click anywhere on the row body — subject text, trial badge,
        // progress bar — to open the trial-detail modal.  Log + Cancel
        // buttons stop propagation so they retain their own actions.
        //
        // Suppressed for single-subject jobs (regardless of trial count):
        // the row's own colored trial chips already convey the same
        // information the modal would show.  The modal is reserved for
        // multi-subject batches where the chip count exceeds the row's
        // horizontal space and a tabular layout helps.
        const _isMultiSubject = (() => {
            try {
                const arr = JSON.parse(item.subject_ids);
                if (Array.isArray(arr)) return arr.length > 1;
            } catch {}
            return false;
        })();
        const _clickable = item.job_id && _isMultiSubject;
        const _rowStyle = _clickable
            ? 'border-left: 3px solid var(--orange); cursor: pointer;'
            : 'border-left: 3px solid var(--orange);';
        const _rowOnclick = _clickable ? `onclick="viewBatchTrials(${item.job_id})"` : '';
        html += `
            <div class="queue-item" style="${_rowStyle}" ${_rowOnclick}>
                <span class="job-indicator job-running"></span>
                <span class="type">${item.job_type}</span>
                <span style="flex:1;min-width:0;display:flex;align-items:center;gap:4px;overflow:hidden;"><span class="subjects" style="flex:none;" title="${subjects}">${subjects}</span>${trialBadge(item)}</span>
                ${progressHtml}
                ${item.job_id ? `<button class="btn btn-sm" onclick="event.stopPropagation();viewJobLogLive(${item.job_id})">Log</button>` : ''}
                <button class="btn btn-sm btn-danger" onclick="event.stopPropagation();cancelQueueItem(${item.id})">Cancel</button>
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
                <span style="flex:1;min-width:0;display:flex;align-items:center;gap:4px;overflow:hidden;"><span class="subjects" style="flex:none;" title="${subjects}">${subjects}</span>${trialBadge(item)}</span>
                <button class="btn btn-sm btn-danger" onclick="cancelQueueItem(${item.id})">Cancel</button>
            </div>
        `;
    }

    el.innerHTML = html;
}

// ── Local jobs (mediapipe, pose, deidentify, etc.) ──────────
async function pollLocalJobs() {
    try {
        const jobs = await API.get('/api/jobs?status=running,pending');
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
            // Preproc-batch "uploading" phase: show the word instead
            // of "0%" while every subject's videos go up.  Cleared
            // automatically once the queue manager flips phase to
            // "running" at the start of the work pass.
            let _uploading = false;
            try {
                const _p = job.params_json ? JSON.parse(job.params_json)
                    : (job.extra_params_json ? JSON.parse(job.extra_params_json) : null);
                if (_p && _p.phase === 'uploading') _uploading = true;
            } catch {}
            const subjectIds = JSON.stringify([job.subject_id]);
            const isPending = job.status === 'pending';
            html += `
                <div class="queue-item" style="border-left: 3px solid ${isPending ? 'var(--accent)' : 'var(--orange)'};">
                    <span class="job-indicator ${isPending ? 'job-pending' : 'job-running'}"></span>
                    <span class="type">${job.job_type}</span>
                    <span style="flex:1;min-width:0;display:flex;align-items:center;gap:4px;overflow:hidden;"><span class="subjects" style="flex:none;" title="${name}">${name}</span>${trialBadge(job)}</span>
                    ${isPending ? '<span style="font-size:11px;color:var(--text-muted);">Queued</span>' : (_uploading ? `
                    <span style="font-size:11px;color:var(--text-muted);font-style:italic;">uploading</span>
                    <button class="btn btn-sm" onclick="viewJobLogLive(${job.id})">Log</button>` : `
                    <div class="progress-bar" style="width:80px;">
                        <div class="fill" style="width:${pct}%"></div>
                    </div>
                    <span style="font-size:11px;">${pct.toFixed(0)}%</span>
                    <button class="btn btn-sm" onclick="viewJobLogLive(${job.id})">Log</button>`)}
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
            // Compact: list the first 2 subjects then "+N more".  Anything
            // beyond 2 names overflows the narrow queue rows already; the
            // tooltip on the parent <span> shows the full list.
            if (arr.length <= 2) return arr.join(', ');
            return arr.slice(0, 2).join(', ') + ` +${arr.length - 2}`;
        }
        return subjectIdsJson;
    } catch {
        return subjectIdsJson;
    }
}

// Build the Subjects + Trials cell pair for the history table.
//
// Each subject goes on its own line.  Trials live in a separate column,
// left-aligned, with one row of clickable chips per subject — so subject
// row N in the Subjects cell aligns horizontally with that subject's
// trial chips in the Trials cell.
//
// Truncation: when a job has >4 subjects, only the first 3 are shown by
// default; a "+N more" link expands the row to show the full list.
//
// Clicking a subject opens the appropriate review page (Auto for
// inference/deidentify, DLC for train/analyze).  Clicking a trial chip
// opens the Auto page jumped to that trial.
// Persistent set of history rows the user has expanded.  Survives the
// 2-second SSE/poll re-renders that would otherwise re-collapse each row.
const _expandedHistoryRows = (typeof window._expandedHistoryRows === 'object'
    ? window._expandedHistoryRows : new Set());
window._expandedHistoryRows = _expandedHistoryRows;

function subjectsAndTrialsHtml(item, rowKey, opts) {
    const trials = (() => {
        try {
            const raw = item.params_json;
            if (!raw) return null;
            const p = typeof raw === 'string' ? JSON.parse(raw) : raw;
            // Batch jobs (HRnet, batched deidentify): explicit trials array.
            if (Array.isArray(p?.trials)) return p.trials;
            // Legacy single-trial jobs (per-trial deidentify, old HRnet,
            // mano_fit_*): synthesize a 1-element list from trial_idx +
            // trial_name so the history still shows a clickable chip.
            if (p && (p.trial_idx != null || p.trial_name)) {
                let _names = [];
                try {
                    _names = JSON.parse(item.subject_ids);
                    if (!Array.isArray(_names)) _names = [_names];
                } catch {}
                return [{
                    subject_name: p.subject_name || _names[0] || '',
                    trial_idx: p.trial_idx,
                    trial_name: p.trial_name || '',
                }];
            }
            return null;
        } catch { return null; }
    })();

    let names = [];
    try {
        names = JSON.parse(item.subject_ids);
        if (!Array.isArray(names)) names = [names];
    } catch { names = []; }

    const kind = jobTypeKind(item.job_type);
    const subjectLink = (n) => {
        const url = subjectReviewUrl(item.job_type, n, null);
        return `<a href="${url}" style="color:inherit;text-decoration:none;">${n}</a>`;
    };
    // Identify the "in-process" trial: the first one that's been uploaded
    // but doesn't have an outcome yet.  Trials are processed in array
    // order by the remote runner, so the earliest uploaded-no-outcome
    // entry is the one currently chewing on the GPU.  Only computed when
    // the caller opts in (markCurrent) — the history table doesn't want
    // a blinking chip on every row.
    let _currentTrialRef = null;
    if (opts && opts.markCurrent && Array.isArray(trials)) {
        for (const t of trials) {
            if (t && t.uploaded && !t.outcome) {
                _currentTrialRef = t;
                break;
            }
        }
    }
    // Local jobs never go through upload / remote_only / download
    // stages — there's nothing to SCP.  Collapse the chip palette to:
    //   ok          → green   "Completed"
    //   failed      → red     failed
    //   current     → pulsing yellow ("Running")
    //   else        → blue    pending
    const isLocal = (item.execution_target || 'remote').startsWith('local');
    const trialChip = (n, t) => {
        let short = t.trial_name || '';
        const idx = short.indexOf('_');
        if (idx >= 0) short = short.slice(idx + 1);
        if (!short) return '';
        let bg = 'var(--accent,#2196f3)';
        let opacity = '1';
        let titleSuffix = '';
        let extraClass = '';
        const out = t.outcome;
        if (out === 'ok') {
            bg = 'var(--green,#4caf50)';
            titleSuffix = isLocal ? ' — completed' : ' — completed + downloaded';
        } else if (out === 'remote_only' && !isLocal) {
            bg = 'var(--yellow,#fbc02d)';
            titleSuffix = ' — completed but not downloaded (re-download to fetch)';
        } else if (out === 'failed') {
            bg = '#e53935';
            const err = t.outcome_error ? ` (${t.outcome_error})` : '';
            titleSuffix = ` — failed${err}`;
        } else if (t === _currentTrialRef) {
            // The trial actively in flight — pulse-yellow it.  For
            // remote jobs this is "running on GPU"; for local jobs
            // there's no upload step so the marker simply means
            // "running now".
            extraClass = 'trial-current';
            bg = 'var(--yellow,#fbc02d)';
            titleSuffix = isLocal ? ' — running now' : ' — running on GPU now';
        } else if (!isLocal && t.uploaded) {
            titleSuffix = ' — uploaded, awaiting inference';
        } else if (!isLocal) {
            // Pre-upload — dim the chip so the user can see at a glance
            // which trials haven't been pushed to the remote yet.
            opacity = '0.4';
            titleSuffix = ' — pending upload';
        } else {
            // Local + no outcome yet — plain pending chip.
            titleSuffix = ' — pending';
        }
        const style = `display:inline-block;background:${bg};color:#fff;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;margin-right:4px;text-decoration:none;opacity:${opacity};`;
        if (kind === 'dlc') {
            return `<span class="${extraClass}" style="${style}opacity:0.7;">${short}</span>`;
        }
        const url = subjectReviewUrl(item.job_type, n, t.trial_idx);
        return `<a class="${extraClass}" href="${url}" title="Review ${n} ${short}${titleSuffix}" style="${style}">${short}</a>`;
    };

    // Build a per-subject ordered list of trials (subject → list).
    const trialsBySubject = new Map();
    if (trials && trials.length > 0) {
        for (const t of trials) {
            const sn = t.subject_name || (names[0] || '');
            if (!trialsBySubject.has(sn)) trialsBySubject.set(sn, []);
            trialsBySubject.get(sn).push(t);
        }
        // If subject_ids didn't list every subject in the trials array, add them.
        for (const sn of trialsBySubject.keys()) {
            if (!names.includes(sn)) names.push(sn);
        }
    }

    // Truncate to first 3 with "+N more" expand toggle when >4 — unless
    // the caller asks not to (e.g. trial-detail modal wants to show
    // everything).  Re-renders preserve expansion state via a Set keyed
    // by rowKey so polling (SSE every 2s) doesn't re-collapse rows.
    const noCollapse = !!(opts && opts.noCollapse);
    const expandedNow = !!rowKey && _expandedHistoryRows.has(rowKey);
    const truncated = !noCollapse && !expandedNow && names.length > 4;
    const MAX_VISIBLE = 3;
    const visibleNames = truncated ? names.slice(0, MAX_VISIBLE) : names;
    const hiddenNames = truncated ? names.slice(MAX_VISIBLE) : [];

    const subjLine = (n) => `<div class="hist-row-line">${subjectLink(n)}</div>`;
    const trialLine = (n) => {
        const ts = trialsBySubject.get(n) || [];
        // Empty line keeps vertical alignment with the subject column when
        // a subject has no trial info.  &nbsp; ensures the line has height.
        return `<div class="hist-row-line">${ts.length ? ts.map(t => trialChip(n, t)).join('') : '&nbsp;'}</div>`;
    };

    let subjectsHtml = visibleNames.map(subjLine).join('');
    let trialsHtml   = visibleNames.map(trialLine).join('');

    if (truncated) {
        const n = hiddenNames.length;
        subjectsHtml += `<div class="hist-row-line hist-more-toggle"
            data-row="${rowKey}" data-count="${n}"
            onclick="toggleHistoryRowExpand(this)"
            style="cursor:pointer;color:#bbb;text-decoration:underline;font-size:12px;">+${n} more</div>`;
    } else if (!noCollapse && expandedNow && names.length > 4) {
        // Show a "show fewer" affordance so the user can collapse again.
        const hiddenCount = names.length - 3;
        subjectsHtml += `<div class="hist-row-line hist-more-toggle"
            data-row="${rowKey}" data-count="${hiddenCount}" data-expanded="1"
            onclick="toggleHistoryRowExpand(this)"
            style="cursor:pointer;color:#bbb;text-decoration:underline;font-size:12px;">show fewer</div>`;
    }

    return { subjectsHtml, trialsHtml };
}

// Toggle expansion of a history row's subject list.  Stores the
// expanded state in ``_expandedHistoryRows`` and triggers a fresh render
// — the Set survives the next polling cycle so the row doesn't snap
// back closed automatically.
function toggleHistoryRowExpand(linkEl) {
    const rowKey = linkEl.dataset.row;
    if (!rowKey) return;
    if (_expandedHistoryRows.has(rowKey)) _expandedHistoryRows.delete(rowKey);
    else _expandedHistoryRows.add(rowKey);
    if (typeof _lastQueueState !== 'undefined' && _lastQueueState) {
        renderHistory(_lastQueueState.history || []);
    }
}
window.toggleHistoryRowExpand = toggleHistoryRowExpand;

function resourceLabel(item) {
    const et = item.execution_target || 'local-cpu';
    const res = (item.resource || 'cpu').toUpperCase();
    if (et === 'remote') return `Remote ${res}`;
    if (et === 'local-gpu') return 'Local GPU';
    return `Local ${res}`;
}

function renderHistory(history) {
    const el = document.getElementById('historyContent');
    if (!history || history.length === 0) {
        el.innerHTML = '<span class="empty-state">No completed jobs yet.</span>';
        return;
    }
    // Background-fetch HRnet status for any uncached subjects in this view,
    // then re-render once so the trial chips can be coloured by output state.
    ensureHrnetStatusForJobs(history).then(updated => {
        if (updated) renderHistory(history);
    });

    let html = `
        <style>
            .history-table td { vertical-align: top; }
            /* Each row line — subjects on the left, trial chips on the
               right — must occupy exactly the same vertical slot so chip
               row N lines up with subject row N.  Made global (not
               scoped to .history-table) so the trial-detail modal can
               reuse the same alignment without duplicating the rules. */
            .hist-row-line {
                line-height: 22px;
                min-height: 22px;
                height: 22px;
                display: flex;
                align-items: center;
                box-sizing: border-box;
                gap: 0;
            }
            .hist-row-line a,
            .hist-row-line span { line-height: 1; }
        </style>
        <table class="history-table">
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Subjects</th>
                    <th>Trials</th>
                    <th>Resource</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Finished</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (let _idx = 0; _idx < history.length; _idx++) {
        const item = history[_idx];
        const _rowKey = `hr${item.job_id || _idx}`;
        const subjects = parseSubjects(item.subject_ids);
        const { subjectsHtml: _subjectsHtml, trialsHtml: _trialsHtml } = subjectsAndTrialsHtml(item, _rowKey);
        // Status badge: derive a richer label + colour from per-trial
        // outcomes when the batch produced any.  See queue_manager batch
        // loops — it stamps params_json.trials[i].outcome and writes a
        // matching error_msg ("Ran X/Y trials" or "Download incomplete:
        // X/Y downloaded").  Three visible categories:
        //   green  completed              all trials ok
        //   yellow Download incomplete    all completed, some not pulled
        //   yellow Ran X/Y                some trials failed outright
        let _statusLabel = item.status;
        let statusClass = item.status === 'completed' ? 'badge-complete'
            : item.status === 'failed' ? 'badge-error_detection'
            : 'badge-created';
        try {
            const _p = item.params_json
                ? (typeof item.params_json === 'string'
                    ? JSON.parse(item.params_json)
                    : item.params_json)
                : null;
            const ts = Array.isArray(_p?.trials) ? _p.trials : null;
            if (ts && ts.length && ts.some(t => t.outcome)) {
                const N = ts.length;
                const nFail = ts.filter(t => t.outcome === 'failed').length;
                const nRem  = ts.filter(t => t.outcome === 'remote_only').length;
                const nOk   = ts.filter(t => t.outcome === 'ok').length;
                if (nFail > 0) {
                    _statusLabel = `Ran ${nOk + nRem}/${N}`;
                    statusClass = 'badge-warn';
                } else if (nRem > 0) {
                    _statusLabel = 'Download incomplete';
                    statusClass = 'badge-warn';
                }
            }
        } catch { /* fall through to default label */ }
        const errorTip = item.error_msg ? ` title="${item.error_msg}"` : '';
        const finishedAt = item.finished_at ? new Date(item.finished_at + 'Z').toLocaleString() : '-';
        const duration = (item.started_at && item.finished_at)
            ? formatDuration(new Date(item.finished_at + 'Z') - new Date(item.started_at + 'Z'))
            : '-';
        const logBtn = item.job_id
            ? `<button class="btn btn-sm" onclick="viewJobLog(${item.job_id})">Log</button>`
            : '';
        const isRemote = (item.execution_target || '').includes('remote');
        const et = item.execution_target || 'local-cpu';
        const subjectsEsc = subjects.replace(/'/g, "\\'");
        // Re-download is also useful on FAILED remote jobs (partial
        // per-subject results may exist before the failure).
        const dlBtn = (isRemote && (item.status === 'completed' || item.status === 'failed'))
            ? `<button class="btn btn-sm" onclick="redownloadJob('${item.job_type}', '${subjectsEsc}', ${item.job_id || 'null'})">Re-download</button>`
            : '';
        // Re-run is available on every history row regardless of final
        // status — useful for re-detecting / re-fitting with different
        // settings even after a successful run.  Requires a job_id so
        // the params (bbox, trial list, etc.) can be looked up.
        const rerunBtn = item.job_id
            ? `<button class="btn btn-sm" onclick="rerunJob('${item.job_type}', '${subjectsEsc}', '${et}', ${item.job_id})">Re-run</button>`
            : '';
        // Resume = re-submit only the trials that didn't reach outcome=ok.
        // Visible whenever a batch job has any non-ok outcomes (failed,
        // remote_only, or absent for trials that never ran due to a
        // mid-batch interruption).  Distinct from Re-run (full restart).
        let resumeBtn = '';
        try {
            const _p = item.params_json
                ? (typeof item.params_json === 'string' ? JSON.parse(item.params_json) : item.params_json)
                : null;
            const ts = Array.isArray(_p?.trials) ? _p.trials : null;
            if (ts && ts.length && ts.some(t => t.outcome !== 'ok')) {
                resumeBtn = `<button class="btn btn-sm" onclick="resumeJob('${item.job_type}', '${subjectsEsc}', '${et}', ${item.job_id || 'null'})">Resume</button>`;
            }
        } catch { /* ignore */ }

        // Subjects + trials are kept narrow (~130 px each) so the combined
        // width matches the prior single-column 260 px layout.  Trials col
        // is left-aligned and starts on the same baseline as the first
        // subject row, so subject N ↔ trials N stay horizontally aligned.
        html += `
            <tr>
                <td style="white-space:nowrap;">${item.job_type}</td>
                <td style="max-width:130px;" title="${subjects}">${_subjectsHtml}</td>
                <td style="max-width:130px;text-align:left;">${_trialsHtml}</td>
                <td style="white-space:nowrap;">${resourceLabel(item)}</td>
                <td><span class="badge ${statusClass}"${errorTip}>${_statusLabel}</span></td>
                <td style="font-size:12px;color:var(--text-muted);white-space:nowrap;">${duration}</td>
                <td style="font-size:12px;color:var(--text-muted);white-space:nowrap;">${finishedAt}</td>
                <td style="white-space:nowrap;">${logBtn} ${dlBtn} ${resumeBtn} ${rerunBtn}</td>
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

// ── Re-run failed job ────────────────────────────────────
// Resume a partial batch: re-submit only the trials whose outcome isn't
// "ok" (failed, remote_only, or never attempted).  Falls back to a full
// re-run when the original job has no per-trial outcome data (legacy or
// non-batch jobs).  The new submission strips outcome fields so it starts
// fresh.
async function resumeJob(jobType, subjects, executionTarget, jobId) {
    try {
        if (!jobId) { alert('No job ID — cannot resume'); return; }
        const orig = await API.get(`/api/jobs/${jobId}`);
        let params = null;
        if (orig?.params_json) {
            params = typeof orig.params_json === 'string'
                ? JSON.parse(orig.params_json) : orig.params_json;
        }
        const trials = Array.isArray(params?.trials) ? params.trials : null;
        if (!trials) {
            // No trials list → treat as a full re-run.
            return rerunJob(jobType, subjects, executionTarget, jobId);
        }
        const remaining = trials
            .filter(t => t.outcome !== 'ok')
            .map(t => {
                const c = { ...t };
                delete c.outcome;
                delete c.outcome_error;
                return c;
            });
        if (!remaining.length) {
            alert('All trials already completed — nothing to resume');
            return;
        }
        const subjectList = [...new Set(remaining.map(t => t.subject_name).filter(Boolean))];
        const allSubjects = await API.get('/api/subjects');
        const subjectIds = subjectList
            .map(name => allSubjects.find(s => s.name === name)?.id)
            .filter(Boolean);
        if (!subjectIds.length) { alert('Could not resolve subject IDs'); return; }
        const newParams = { ...params, trials: remaining };
        // Refresh the trial-name display label to reflect what's actually
        // being submitted now.
        if (remaining.length === 1) {
            newParams.trial_name = remaining[0].trial_name;
        } else if (remaining.length >= 5) {
            newParams.trial_name = `${remaining.length} trials`;
        } else {
            const multi = subjectIds.length > 1;
            newParams.trial_name = remaining.map(t =>
                multi ? `${t.subject_name} ${t.trial_name}` : t.trial_name
            ).join(', ');
        }
        await API.post('/api/remote/launch', {
            job_type: jobType,
            subject_ids: subjectIds,
            subjects: subjectList,
            execution_target: executionTarget || 'local-cpu',
            extra_params: newParams,
        });
        refreshQueue();
    } catch (e) {
        console.error('Resume failed:', e);
        alert('Resume failed: ' + (e?.message || e));
    }
}

async function rerunJob(jobType, subjects, executionTarget, jobId) {
    try {
        const subjectList = subjects.split(',').map(s => s.trim());
        // Resolve subject IDs from names
        const allSubjects = await API.get('/api/subjects');
        const subjectIds = subjectList.map(name => {
            const s = allSubjects.find(s => s.name === name);
            return s ? s.id : null;
        }).filter(Boolean);
        if (!subjectIds.length) { alert('Could not resolve subject IDs'); return; }

        // Preserve the original job's per-run parameters (trial_idx,
        // trial_name, bbox_os/od, use_bbox, fit weights, etc.) so a
        // per-trial re-run actually re-runs the same trial instead of
        // silently defaulting to trial 0.
        let extraParams = null;
        if (jobId) {
            try {
                const orig = await API.get(`/api/jobs/${jobId}`);
                if (orig?.params_json) {
                    const parsed = typeof orig.params_json === 'string'
                        ? JSON.parse(orig.params_json) : orig.params_json;
                    if (parsed && typeof parsed === 'object' && Object.keys(parsed).length > 0) {
                        // Strip per-trial outcome flags so the new batch
                        // starts with a clean slate (no stale yellow/red).
                        if (Array.isArray(parsed.trials)) {
                            parsed.trials = parsed.trials.map(t => {
                                const c = { ...t };
                                delete c.outcome;
                                delete c.outcome_error;
                                return c;
                            });
                        }
                        extraParams = parsed;
                    }
                }
            } catch (e) {
                console.warn('Re-run: could not load original params, falling back:', e);
            }
        }

        const payload = {
            job_type: jobType,
            subject_ids: subjectIds,
            execution_target: executionTarget || 'local-cpu',
        };
        if (extraParams) payload.extra_params = extraParams;
        await API.post('/api/remote/launch', payload);
        refreshQueue();
    } catch (e) {
        console.error('Re-run failed:', e);
    }
}

// ── Re-download results ─────────────────────────────────
async function redownloadJob(jobType, subjects, parentJobId) {
    try {
        // Strip the "+N" overflow marker parseSubjects adds for >3-subject
        // batches — the backend resolves the full subject list from the
        // parent job's params_json when ``parent_job_id`` is provided.
        const subjectList = subjects.split(',')
            .map(s => s.trim())
            .filter(s => s && !s.startsWith('+'));
        await API.post('/api/remote/redownload', {
            job_type: jobType,
            subjects: subjectList,
            parent_job_id: parentJobId || null,
        });
        refreshQueue();
    } catch (e) {
        console.error('Re-download failed:', e);
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

// ── Active-batch trial-details modal ───────────────────────────────────
// Click a running batch row → see per-trial status with live polling.
// Reuses ``subjectsAndTrialsHtml`` (used by the History table) so the
// rendering stays consistent.  Trial chips are coloured by ``outcome``:
//   ok           → green
//   remote_only  → yellow
//   failed       → red
//   undefined    → blue (pending / in-progress)
let _trialsModalJobId = null;
let _trialsModalTimer = null;

async function _refreshTrialsModal() {
    if (_trialsModalJobId == null) return;
    try {
        const j = await API.get(`/api/jobs/${_trialsModalJobId}`);
        const item = {
            job_id: j.id,
            job_type: j.job_type,
            params_json: j.params_json,
            subject_ids: j.subject_ids || j.subject_name ? JSON.stringify([j.subject_name || '']) : '[]',
            execution_target: j.remote_host && j.remote_host !== 'localhost' ? 'remote' : 'local-cpu',
        };
        // /api/jobs/{id} doesn't include subject_ids the same way the
        // queue does — fall back to params_json.subjects when present.
        try {
            const p = j.params_json
                ? (typeof j.params_json === 'string' ? JSON.parse(j.params_json) : j.params_json)
                : null;
            if (Array.isArray(p?.subjects) && p.subjects.length) {
                item.subject_ids = JSON.stringify(p.subjects);
            } else if (Array.isArray(p?.trials)) {
                const subs = [...new Set(p.trials.map(t => t.subject_name).filter(Boolean))];
                if (subs.length) item.subject_ids = JSON.stringify(subs);
            }
        } catch {}
        // Modal: show every subject — never collapse with "+N more".
        // markCurrent=true tells the chip renderer to pulse-yellow the
        // trial that's currently being processed on the GPU.
        const { subjectsHtml, trialsHtml } = subjectsAndTrialsHtml(
            item, `tm${j.id}`, { noCollapse: true, markCurrent: true });
        const titleEl = document.getElementById('trialsModalTitle');
        if (titleEl) {
            const pct = Math.round(j.progress_pct || 0);
            // Friendly job-type label: "hrnet" → "HRnet" etc.
            const TYPE_LABEL = {
                hrnet: 'HRnet',
                deidentify: 'Deidentify',
                mediapipe: 'MediaPipe',
                blur: 'Blur',
                'mediapipe+blur': 'MediaPipe + Blur',
                vision: 'Vision',
                pose: 'Pose',
                train: 'Train',
                analyze_v1: 'Analyze v1',
                analyze_v2: 'Analyze v2',
                skeleton_v1: 'Skeleton fit v1',
                skeleton_v2: 'Skeleton fit v2',
                skeleton_v3: 'Skeleton fit v3',
            };
            const typeStr = TYPE_LABEL[j.job_type] || j.job_type || '';
            const resource = (j.remote_host && j.remote_host !== 'localhost')
                ? 'Remote' : 'Local';
            // Preproc batch: while every subject's videos are still
            // uploading the progress_pct stays at 0 — show the
            // "uploading" phase from params.phase instead of "0%".
            let pctOrPhase = `${pct}%`;
            try {
                const _params = item.params_json
                    ? JSON.parse(item.params_json) : (item.extra_params_json
                        ? JSON.parse(item.extra_params_json) : null);
                if (_params && _params.phase === 'uploading') {
                    pctOrPhase = 'uploading';
                }
            } catch {}
            titleEl.textContent = `${typeStr} · ${resource} · ${j.status} (${pctOrPhase})`;
        }
        // Tailor the legend to the job's execution target.  Local
        // jobs skip the upload + remote-only-completed stages; the
        // "Downloaded" row is relabeled "Completed".
        const isLocalJob = !(j.remote_host && j.remote_host !== 'localhost');
        const _show = (id, vis) => {
            const el = document.getElementById(id);
            if (el) el.style.display = vis ? '' : 'none';
        };
        _show('legendUploading',  !isLocalJob);
        _show('legendRemoteOnly', !isLocalJob);
        const dl = document.getElementById('legendDownloaded');
        if (dl) {
            // Rebuild the label text (after the colour square) so we
            // don't wipe the inline span.
            const sq = dl.querySelector('.lg-sq');
            dl.innerHTML = '';
            if (sq) dl.appendChild(sq);
            dl.appendChild(document.createTextNode(isLocalJob ? 'Completed' : 'Downloaded'));
        }
        // Local style block ensures the row-line height + flex alignment
        // is in effect even if the History table hasn't rendered its
        // <style> yet (which is where the global ``.hist-row-line``
        // rules live).
        const html = `
            <style>
              .hist-row-line { line-height:22px;min-height:22px;height:22px;
                  display:flex;align-items:center;box-sizing:border-box; }
              .hist-row-line a, .hist-row-line span { line-height:1; }
            </style>
            <table style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr>
                        <th style="text-align:left;padding:4px 8px;color:var(--text-muted);font-weight:normal;">Subject</th>
                        <th style="text-align:left;padding:4px 8px;color:var(--text-muted);font-weight:normal;">Trials</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="vertical-align:top;padding:4px 8px;">${subjectsHtml}</td>
                        <td style="vertical-align:top;padding:4px 8px;">${trialsHtml}</td>
                    </tr>
                </tbody>
            </table>`;
        document.getElementById('trialsModalContent').innerHTML = html;
        // Stop polling once the job is in a terminal state.
        if (['completed', 'failed', 'cancelled'].includes(j.status)) {
            if (_trialsModalTimer) { clearInterval(_trialsModalTimer); _trialsModalTimer = null; }
        }
    } catch (e) {
        console.warn('Failed to refresh trials modal:', e);
    }
}

function viewBatchTrials(jobId) {
    if (!jobId) return;
    _trialsModalJobId = jobId;
    document.getElementById('trialsModalContent').innerHTML =
        '<span style="color:var(--text-muted);">Loading…</span>';
    document.getElementById('trialsModal').classList.add('active');
    _refreshTrialsModal();
    if (_trialsModalTimer) clearInterval(_trialsModalTimer);
    _trialsModalTimer = setInterval(_refreshTrialsModal, 3000);
}

function closeTrialsModal() {
    _trialsModalJobId = null;
    if (_trialsModalTimer) { clearInterval(_trialsModalTimer); _trialsModalTimer = null; }
    document.getElementById('trialsModal').classList.remove('active');
}
window.viewBatchTrials = viewBatchTrials;
window.closeTrialsModal = closeTrialsModal;

// Re-download section removed — use Re-download buttons in Job History

// ── Init ────────────────────────────────────────────────
(async function init() {
    // Add event listeners for execution target radio buttons
    document.querySelectorAll('input[name="executionTarget"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            const gpuSelector = document.getElementById('gpuSelectorDiv');
            const targetLocalGpu = e.target.value === 'local-gpu';
            gpuSelector.style.display = targetLocalGpu ? 'flex' : 'none';
            updateStepAvailability();
        });
    });

    // Step changes are handled by selectStep() via onclick on .step-btn

    // "Use bounding boxes" checkbox (HRnet only) — re-render trial cells
    // so saved-bbox highlights and ineligibility dimming update live.
    const useBboxCb = document.getElementById('useBboxCb');
    if (useBboxCb) {
        useBboxCb.addEventListener('change', () => {
            updateTrialSection();
        });
    }

    // "Run in reverse" — flips MP trial colouring between forward-pass and
    // reverse-pass npz existence.
    const mpReverseCb = document.getElementById('mpReverseCb');
    if (mpReverseCb) {
        mpReverseCb.addEventListener('change', () => {
            if (selectedStep === 'mediapipe') {
                colorSubjectsByStep();
                updateTrialSection();
            }
        });
    }

    // Load data
    await loadGpuStatus();
    // Probe the remote server once at page load (fire-and-forget so it
    // doesn't block first paint — SSH tests can take 2-5s on a cold
    // socket).  Then re-probe every 2 minutes so the radio re-enables
    // itself when the box comes back online.
    loadRemoteStatus();
    setInterval(loadRemoteStatus, 120_000);
    await loadSteps();
    await loadSubjects();
    await refreshQueue();
    connectQueueStream();
    // Live value labels for the Jobs-page Skel Fit v1 sliders.
    [
        ['skelfitSliderReproj', 'skelfitWReproj'],
        ['skelfitSliderBone',   'skelfitWBone'],
        ['skelfitSliderSmooth', 'skelfitWSmooth'],
        ['skelfitSliderAccelK', 'skelfitWAccelK'],
        ['skelfitSliderBoneK',  'skelfitWBoneK'],
        ['skelfitSliderKmax',   'skelfitWKmax'],
    ].forEach(([sId, dId]) => {
        const s = document.getElementById(sId);
        const d = document.getElementById(dId);
        if (!s || !d) return;
        const sync = () => { d.textContent = (sId === 'skelfitSliderKmax')
            ? String(parseInt(s.value || '0'))
            : Number(s.value).toFixed(1); };
        s.addEventListener('input', sync);
        sync();
    });

    // Stereo Correct settings — mirror the Labels-page Stereo panel.
    // Mode radio shows/hides the dilate slider; sliders update their
    // live value labels.  All read by submitJob's stereo_correct
    // branch and forwarded as extra_params.
    const _stereoSyncMode = () => {
        const m = document.querySelector('input[name="stereoCorrectMode"]:checked');
        const mode = m ? m.value : 'image';
        const dWrap = document.getElementById('stereoCorrectDilateWrap');
        if (dWrap) dWrap.style.display = (mode === 'outline') ? '' : 'none';
    };
    document.querySelectorAll('input[name="stereoCorrectMode"]').forEach(el => {
        el.addEventListener('change', _stereoSyncMode);
    });
    _stereoSyncMode();
    const _stereoDilSlider = document.getElementById('stereoCorrectDilateSlider');
    const _stereoDilVal    = document.getElementById('stereoCorrectDilateVal');
    if (_stereoDilSlider && _stereoDilVal) {
        const sync = () => { _stereoDilVal.textContent = String(parseInt(_stereoDilSlider.value || '0')); };
        _stereoDilSlider.addEventListener('input', sync);
        sync();
    }
    const _stereoGSlider = document.getElementById('stereoCorrectGaussSlider');
    const _stereoGVal    = document.getElementById('stereoCorrectGaussVal');
    if (_stereoGSlider && _stereoGVal) {
        const sync = () => {
            const v = parseFloat(_stereoGSlider.value || '0') / 100;
            _stereoGVal.textContent = v.toFixed(2);
        };
        _stereoGSlider.addEventListener('input', sync);
        sync();
    }
})();
