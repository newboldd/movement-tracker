/* ── Cross-page navigation state ──────────────────────────────────
   Subject-specific pages share: subject, trial index, frame number, camera side.
   Any page that loads a subject should call setNavState({...}).
   Any page that needs defaults should call getNavState(). */
function setLastSubject(id) {
    if (id) localStorage.setItem('mt_lastSubjectId', String(id));
}
function getLastSubject() {
    return parseInt(localStorage.getItem('mt_lastSubjectId')) || null;
}
function setNavState(state) {
    if (state.subjectId) localStorage.setItem('mt_lastSubjectId', String(state.subjectId));
    if (state.trialIdx != null) sessionStorage.setItem('mt_trialIdx', String(state.trialIdx));
    if (state.frame != null) sessionStorage.setItem('mt_frame', String(state.frame));
    if (state.side) sessionStorage.setItem('mt_side', state.side);
}
function getNavState() {
    return {
        subjectId: parseInt(localStorage.getItem('mt_lastSubjectId')) || null,
        trialIdx: parseInt(sessionStorage.getItem('mt_trialIdx')) ?? null,
        frame: parseInt(sessionStorage.getItem('mt_frame')) ?? null,
        side: sessionStorage.getItem('mt_side') || null,
    };
}

// ── Subject-specific pages ──
const SUBJECT_PAGES = [
    '/mediapipe', '/mediapipe-select', '/deidentify',
    '/labeling', '/labeling-select', '/labels', '/results', '/oscillations',
];

/* ── Build unified single-line header ────────────────────────────
   Target layout:
   [Movement Tracker] [Dashboard|Videos|MediaPipe|...] | [Subject ▾] [←][→] [Trial1][Trial2]... | [page-specific]
   Everything in one flex row.
*/
(function() {
    const path = window.location.pathname;
    const isSubjectPage = SUBJECT_PAGES.some(p => path.startsWith(p));

    // Restyle header as single-line flex
    const header = document.querySelector('.header');
    if (!header) return;

    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.flexWrap = 'nowrap';
    header.style.gap = '12px';
    header.style.padding = '0 16px';
    header.style.height = '40px';
    header.style.minHeight = '40px';

    // Make h1 compact
    const h1 = header.querySelector('h1');
    if (h1) {
        h1.style.margin = '0';
        h1.style.flexShrink = '0';
        h1.style.fontSize = '16px';
        h1.style.whiteSpace = 'nowrap';
    }

    // Nav keeps its CSS styling (gap:24px, font-size:14px, margin-left:auto)
    // Just ensure it aligns vertically
    const nav = header.querySelector('nav');
    if (nav) {
        nav.style.alignItems = 'center';
        nav.style.flexShrink = '0';
    }

    if (!isSubjectPage) return;

    // ── Subject selector ──
    const selContainer = document.createElement('div');
    selContainer.id = 'navSubjectContainer';
    selContainer.style.cssText = 'display:flex;align-items:center;gap:6px;margin-left:8px;flex-shrink:0;';

    // Separator
    const sep = document.createElement('span');
    sep.style.cssText = 'color:var(--border);font-size:16px;margin:0 4px;';
    sep.textContent = '|';
    selContainer.appendChild(sep);

    const sel = document.createElement('select');
    sel.id = 'navSubjectSelect';
    sel.style.cssText = 'padding:2px 6px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:12px;max-width:140px;';

    const prevBtn = document.createElement('button');
    prevBtn.id = 'navPrevSubjectBtn';
    prevBtn.className = 'btn btn-sm';
    prevBtn.innerHTML = '&larr;';
    prevBtn.title = 'Previous subject';
    prevBtn.style.cssText = 'padding:1px 6px;font-size:12px;';

    const nextBtn = document.createElement('button');
    nextBtn.id = 'navNextSubjectBtn';
    nextBtn.className = 'btn btn-sm';
    nextBtn.innerHTML = '&rarr;';
    nextBtn.title = 'Next subject';
    nextBtn.style.cssText = 'padding:1px 6px;font-size:12px;';

    selContainer.appendChild(sel);
    selContainer.appendChild(prevBtn);
    selContainer.appendChild(nextBtn);

    // Trial buttons container
    const trialDiv = document.createElement('div');
    trialDiv.id = 'navTrialBtns';
    trialDiv.style.cssText = 'display:flex;gap:3px;flex-wrap:nowrap;overflow-x:auto;margin-left:10px;margin-right:10px;';
    selContainer.appendChild(trialDiv);

    // Page-specific slot (pages can append buttons here)
    const pageSlot = document.createElement('div');
    pageSlot.id = 'navPageSlot';
    pageSlot.style.cssText = 'display:flex;gap:4px;align-items:center;margin-left:auto;flex-shrink:0;';
    selContainer.appendChild(pageSlot);

    // Insert after h1, before nav (subject selector on left, nav links on right)
    if (nav) {
        nav.insertAdjacentElement('beforebegin', selContainer);
    } else {
        header.appendChild(selContainer);
    }

    // Trial button style
    const style = document.createElement('style');
    style.textContent = `
        .nav-trial-btn {
            padding: 1px 6px;
            font-size: 11px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--card);
            color: var(--text);
            cursor: pointer;
            white-space: nowrap;
        }
        .nav-trial-btn.active {
            background: rgba(33, 150, 243, 0.15);
            color: var(--text);
            border-color: var(--blue);
            font-weight: 600;
        }
        .nav-trial-btn:hover:not(.active) {
            background: var(--hover);
        }
    `;
    document.head.appendChild(style);

    // ── Sync with local page selectors ──
    // Many pages have their own #subjectSelect. Hide it and sync.
    let allSubjects = [];

    // Wait for page JS to populate local selector, then mirror it
    function _syncFromLocal() {
        const localSel = document.getElementById('subjectSelect');
        if (!localSel || localSel === sel) return;

        // Hide local selector (nav has the visible one)
        localSel.style.display = 'none';
        // Hide any local topbar that contained subject/trial UI
        const topbar = localSel.closest('.videos-topbar, .deid-topbar');
        if (topbar) topbar.style.display = 'none';
        // Hide prev/next buttons near the local selector
        const localPrev = document.getElementById('prevSubjectBtn');
        const localNext = document.getElementById('nextSubjectBtn');
        if (localPrev) localPrev.style.display = 'none';
        if (localNext) localNext.style.display = 'none';

        // Copy options from local to nav
        sel.innerHTML = localSel.innerHTML;
        sel.value = localSel.value;

        // When nav changes, update local and trigger its change event
        sel.addEventListener('change', () => {
            localSel.value = sel.value;
            localSel.dispatchEvent(new Event('change'));
            setLastSubject(parseInt(sel.value));
        });

        // When local changes (from page JS), sync nav
        localSel.addEventListener('change', () => {
            sel.value = localSel.value;
        });

        // Extract subjects from options for prev/next
        allSubjects = Array.from(localSel.options)
            .filter(o => o.value)
            .map(o => ({ id: parseInt(o.value), name: o.textContent }));
    }

    // Also sync local trial buttons into nav
    function _syncTrialBtns() {
        const localTrials = document.getElementById('trialBtns');
        if (!localTrials) return;
        // Move local trial buttons into nav
        localTrials.style.display = 'none';
        const _rebuildNavTrials = () => {
            trialDiv.innerHTML = '';
            localTrials.querySelectorAll('.trial-btn').forEach(btn => {
                const clone = btn.cloneNode(true);
                clone.className = 'nav-trial-btn' + (btn.classList.contains('active') ? ' active' : '');
                // Preserve inline colors (e.g. red/green for deident status)
                if (btn.style.borderColor) clone.style.borderColor = btn.style.borderColor;
                if (btn.style.color) clone.style.color = btn.style.color;
                if (btn.style.background) clone.style.background = btn.style.background;
                clone.addEventListener('click', () => btn.click());
                trialDiv.appendChild(clone);
            });
        };
        const observer = new MutationObserver(_rebuildNavTrials);
        observer.observe(localTrials, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] });
    }

    // Move page-specific buttons into the nav page slot
    function _movePageButtons() {
        const slot = document.getElementById('navPageSlot');
        if (!slot) return;

        // Videos page: move Browse button
        const browseBtn = document.getElementById('browseBtn');
        if (browseBtn) slot.appendChild(browseBtn);

        // Deidentify page: move view mode buttons
        document.querySelectorAll('#viewOriginal, #viewPreview, #viewDeidentified').forEach(btn => {
            slot.appendChild(btn);
        });
    }

    // Try syncing immediately, then retry after page JS init
    _syncFromLocal();
    _syncTrialBtns();
    _movePageButtons();
    setTimeout(() => { _syncFromLocal(); _syncTrialBtns(); _movePageButtons(); }, 500);
    setTimeout(() => { _syncFromLocal(); _syncTrialBtns(); _movePageButtons(); }, 1500);

    // Prev/next
    prevBtn.addEventListener('click', () => _switchSubject(-1));
    nextBtn.addEventListener('click', () => _switchSubject(1));

    function _switchSubject(delta) {
        const current = parseInt(sel.value);
        if (!current || allSubjects.length === 0) return;
        const idx = allSubjects.findIndex(s => s.id === current);
        const newIdx = Math.max(0, Math.min(allSubjects.length - 1, idx + delta));
        if (newIdx === idx) return;
        const newId = allSubjects[newIdx].id;
        sel.value = newId;
        setLastSubject(newId);
        // Prefer a soft switch through the local selector (so in-page
        // state — sliders, checkboxes, etc. — survives).  Fall back to
        // a hard navigation only when no local selector is available.
        const localSel = document.getElementById('subjectSelect');
        if (localSel && localSel !== sel) {
            localSel.value = String(newId);
            localSel.dispatchEvent(new Event('change'));
            return;
        }
        _navigateToSubject(newId);
    }

    function _navigateToSubject(sid) {
        if (path.startsWith('/mediapipe') || path === '/mediapipe-select') {
            window.location.href = '/mediapipe-select?subject=' + sid;
        } else if (path.startsWith('/deidentify')) {
            window.location.href = '/deidentify?subject=' + sid;
        } else if (path.startsWith('/labeling') || path === '/labeling-select') {
            window.location.href = '/labeling-select?subject=' + sid;
        } else if (path.startsWith('/labels')) {
            window.location.href = '/labels?subject=' + sid;
        } else if (path.startsWith('/results')) {
            window.location.href = '/results?subject=' + sid;
        } else if (path.startsWith('/oscillations')) {
            window.location.href = '/oscillations?subject=' + sid;
        }
    }
})();

/* Update nav links to include last viewed subject where applicable. */
(function() {
    const lastSubj = getLastSubject();
    if (!lastSubj) return;

    const links = {
        'mediapipe-select': '/mediapipe-select?subject=' + lastSubj,
        'deidentify': '/deidentify?subject=' + lastSubj,
        'results': '/results?subject=' + lastSubj,
        'skeleton': '/labels?subject=' + lastSubj,
        'labeling-select': '/labeling-select?subject=' + lastSubj,
    };

    for (const [href, newUrl] of Object.entries(links)) {
        const link = document.querySelector(`nav a[href*="${href}"]`);
        if (link && !link.href.includes('subject=')) {
            link.href = newUrl;
        }
    }
})();

/* Update "DLC" nav link to return to the last active session. */
(function() {
    const sid = sessionStorage.getItem('dlc_lastSessionId');
    if (!sid) return;
    const link = document.getElementById('labelingLink')
              || document.querySelector('nav a[href*="labeling-select"]');
    if (link) link.href = '/labeling?session=' + sid;
})();

/* Conditionally show/hide Tutorials nav link based on setting. */
(function() {
    const tutLink = document.querySelector('nav a[href="/tutorials"]');
    if (!tutLink) return;
    fetch('/api/settings').then(r => r.json()).then(cfg => {
        if (cfg.show_tutorials === false) tutLink.style.display = 'none';
    }).catch(() => {});
})();

/* ── Live Job Queue in nav dropdown ──────────────────────────── */
(function() {
    const panel = document.getElementById('navJobsPanel');
    const dot = document.getElementById('navJobDot');
    if (!panel || !dot) return;

    let _lastHtml = '';

    function _timeAgo(isoStr) {
        if (!isoStr) return '';
        // SQLite CURRENT_TIMESTAMP returns UTC without a Z suffix; treat
        // bare 'YYYY-MM-DD HH:MM:SS' strings as UTC explicitly.
        let t = isoStr;
        if (typeof t === 'string' && !/[zZ]|[+-]\d\d:?\d\d$/.test(t)) {
            t = t.replace(' ', 'T') + 'Z';
        }
        const sec = Math.max(0, Math.floor((Date.now() - new Date(t).getTime()) / 1000));
        if (sec < 60) return 'just now';
        if (sec < 7200) return Math.floor(sec / 60) + 'm ago';   // up to 120 min
        return Math.floor(sec / 3600) + 'h ago';
    }

    // Cached pending-downloads payload, refreshed independently from the
    // main poll.  ``/api/remote/pending-downloads`` does SSH probes
    // against the remote and can take seconds on a cold cache; awaiting
    // it inside the main poll froze the nav dropdown for that long.
    // We now fetch it in the background and let the dropdown re-render
    // with the latest known value -- the dot might be off by a poll
    // tick but the dropdown stops feeling sluggish.
    let _navCachedPending = [];
    let _pendingInFlight = false;
    function _refreshPendingDownloads() {
        if (_pendingInFlight) return;
        _pendingInFlight = true;
        fetch('/api/remote/pending-downloads')
            .then(r => r.json())
            .then(p => { _navCachedPending = p?.pending || []; })
            .catch(() => {})
            .finally(() => { _pendingInFlight = false; });
    }

    async function _pollJobs() {
        try {
            const [data, localJobs, recentJobs, dlProg] = await Promise.all([
                fetch('/api/remote/queue').then(r => r.json()).catch(() => ({})),
                fetch('/api/jobs?status=running,pending').then(r => r.json()).catch(() => []),
                fetch('/api/jobs?status=completed,failed&limit=3').then(r => r.json()).catch(() => []),
                fetch('/api/remote/download-progress').then(r => r.json()).catch(() => ({downloads: []})),
            ]);
            // Kick off the slow SSH-probing pending-downloads fetch in
            // the background; this poll renders with the previous value.
            _refreshPendingDownloads();
            const activeDl = (dlProg?.downloads || []).filter(d => d.status === 'running');
            const running = data.running || [];
            const cpuQueue = data.cpu_queue || [];
            const gpuQueue = data.gpu_queue || [];
            const pendingResults = _navCachedPending;

            const remoteJobIds = new Set();
            [...running, ...cpuQueue, ...gpuQueue].forEach(item => {
                if (item.job_id) remoteJobIds.add(item.job_id);
            });
            const localOnly = localJobs.filter(j => !remoteJobIds.has(j.id));

            const hasActive = running.length > 0 || cpuQueue.length > 0 || gpuQueue.length > 0 || localOnly.length > 0;
            // Show the dot when active jobs exist OR remote results are
            // ready to pull (download flag).
            dot.style.display = (hasActive || pendingResults.length > 0) ? 'inline-block' : 'none';
            if (pendingResults.length > 0 && !hasActive) {
                dot.style.background = 'var(--blue, #4a9eff)';
                dot.title = `${pendingResults.length} result(s) ready to download`;
            } else {
                dot.style.background = '';
                dot.title = '';
            }

            // ── Helpers ──────────────────────────────────────────────────
            function _subj(subjectIdsJson) {
                try {
                    const arr = JSON.parse(subjectIdsJson);
                    if (Array.isArray(arr)) return arr.slice(0, 2).join(', ') + (arr.length > 2 ? '…' : '');
                    return subjectIdsJson;
                } catch { return String(subjectIdsJson); }
            }

            function _badge(item) {
                // Per-trial colored flags — mirror the outcome scheme
                // used by the Remote queue rows:
                //   ok          → green   (downloaded)
                //   remote_only → yellow  (completed remotely, not pulled)
                //   failed      → red
                //   uploaded    → blue    (pending inference)
                //   none of above → dim blue (uploading / queued)
                //   first uploaded-no-outcome → pulsing yellow (running on GPU)
                // Single-subject jobs always get flags; multi-subject
                // jobs do too — capped at MAX_CHIPS with a "+N" pill so
                // the narrow nav dropdown rows don't wrap.
                const raw = item && (item.params_json || item.trial_name);
                if (!raw) return '';
                const MAX_CHIPS = 4;
                const stripPrefix = (s) => {
                    if (!s) return '';
                    const i = s.indexOf('_');
                    return i >= 0 ? s.slice(i + 1) : s;
                };
                const baseStyle = 'color:#fff;padding:1px 4px;border-radius:3px;'
                    + 'font-size:9px;font-weight:600;margin-left:3px;flex-shrink:0;';
                try {
                    const p = typeof raw === 'string' ? JSON.parse(raw) : raw;

                    if (Array.isArray(p?.trials) && p.trials.length) {
                        const trials = p.trials;
                        let currentRef = null;
                        for (const t of trials) {
                            if (t && t.uploaded && !t.outcome) { currentRef = t; break; }
                        }
                        const renderChip = (t) => {
                            const short = stripPrefix(t.trial_name || '');
                            if (!short) return '';
                            let bg = 'var(--accent,#2196f3)';
                            let opacity = '1';
                            let extraClass = '';
                            const out = t.outcome;
                            if (out === 'ok') {
                                bg = 'var(--green,#4caf50)';
                            } else if (out === 'remote_only') {
                                bg = 'var(--yellow,#fbc02d)';
                            } else if (out === 'failed') {
                                bg = '#e53935';
                            } else if (t === currentRef) {
                                bg = 'var(--yellow,#fbc02d)';
                                extraClass = 'trial-current';
                            } else if (t.uploaded) {
                                // default blue
                            } else {
                                opacity = '0.4';
                            }
                            const label = t.subject_name
                                ? `${t.subject_name} ${short}` : short;
                            return `<span class="${extraClass}" title="${label}" `
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

                    // Legacy single-trial: one chip colored by job status.
                    if (p && p.trial_name) {
                        const short = stripPrefix(p.trial_name);
                        let bg = 'var(--accent,#2196f3)';
                        const status = item?.status || '';
                        if (status === 'completed') bg = 'var(--green,#4caf50)';
                        else if (status === 'failed') bg = '#e53935';
                        return `<span style="${baseStyle}background:${bg};">${short}</span>`;
                    }
                } catch {}
                return '';
            }

            function _bar(pct) {
                return `<div style="width:30px;height:3px;background:var(--border);border-radius:2px;flex-shrink:0;"><div style="width:${pct}%;height:100%;background:var(--orange);border-radius:2px;"></div></div><span style="font-size:9px;color:var(--text-muted);">${Math.round(pct)}%</span>`;
            }

            // ── Render one lane ──────────────────────────────────────────
            function _lane(resource, target) {
                const isLocal = target.startsWith('local');
                const laneRun = running.filter(r => {
                    const t = r.execution_target || 'remote';
                    return isLocal ? (r.resource === resource && t.startsWith('local')) : (r.resource === resource && t === 'remote');
                });
                const allQ = resource === 'gpu' ? gpuQueue : cpuQueue;
                const laneQ = allQ.filter(q => {
                    const t = q.execution_target || 'remote';
                    return isLocal ? t.startsWith('local') : t === 'remote';
                });
                const laneL = target === 'local-cpu' ? localOnly : [];
                if (!laneRun.length && !laneQ.length && !laneL.length) return '';

                let items = '';

                for (const job of laneL) {
                    const name = job.subject_name || ('Subject ' + job.subject_id);
                    const pct = job.progress_pct || 0;
                    const type = (job.job_type || '').replace(/_/g, ' ');
                    const isPending = job.status === 'pending';
                    const progress = isPending
                        ? `<span style="font-size:9px;color:var(--text-muted);">Queued</span>`
                        : _bar(pct);
                    items += `<div class="nav-qi${isPending ? '' : ' nav-qi-run'}">
                        <span class="nav-qi-dot" style="background:${isPending ? '#FF9800' : '#4CAF50'};"></span>
                        <span class="nav-qi-type">${type}</span>
                        <span class="nav-qi-name"><span>${name}</span>${_badge(job)}</span>
                        ${progress}
                        <button onclick="event.stopPropagation();fetch('/api/jobs/${job.id}/cancel',{method:'POST'}).then(()=>window._navPollJobs())">×</button>
                    </div>`;
                }

                for (const item of laneRun) {
                    const name = _subj(item.subject_ids);
                    const pct = item.progress_pct || 0;
                    const type = (item.job_type || '').replace(/_/g, ' ');
                    const isRem = (item.execution_target || 'remote') === 'remote';
                    // Any remote job at 0% → "Uploading…".  HRnet
                    // single-trial uses 85-100% as "Downloading…".
                    const isHrnet = (item.job_type || '').includes('hrnet');
                    let _isBatch = false;
                    try {
                        const _p = item.params_json
                            ? (typeof item.params_json === 'string' ? JSON.parse(item.params_json) : item.params_json)
                            : null;
                        _isBatch = Array.isArray(_p?.trials) && _p.trials.length > 1;
                    } catch {}
                    const showHrnetDl = isHrnet && !_isBatch && isRem;
                    const phase = (isRem && pct === 0) ? 'Uploading…'
                                : (showHrnetDl && pct >= 85 && pct < 100) ? 'Downloading…'
                                : null;
                    const progress = phase
                        ? `<span style="font-size:9px;color:var(--text-muted);font-style:italic;">${phase}</span>`
                        : _bar(pct);
                    items += `<div class="nav-qi nav-qi-run">
                        <span class="nav-qi-dot" style="background:#4CAF50;"></span>
                        <span class="nav-qi-type">${type}</span>
                        <span class="nav-qi-name"><span>${name}</span>${_badge(item)}</span>
                        ${progress}
                        <button onclick="event.stopPropagation();fetch('/api/remote/cancel/${item.id}',{method:'POST'}).then(()=>window._navPollJobs())">×</button>
                    </div>`;
                }

                laneQ.forEach((item, i) => {
                    const name = _subj(item.subject_ids);
                    const type = (item.job_type || '').replace(/_/g, ' ');
                    items += `<div class="nav-qi">
                        <span style="font-size:9px;color:var(--text-muted);flex-shrink:0;">${i + 1}.</span>
                        <span class="nav-qi-type">${type}</span>
                        <span class="nav-qi-name"><span>${name}</span>${_badge(item)}</span>
                        <button onclick="event.stopPropagation();fetch('/api/remote/cancel/${item.id}',{method:'POST'}).then(()=>window._navPollJobs())">×</button>
                    </div>`;
                });

                return items;
            }

            // ── Assemble lanes ───────────────────────────────────────────
            const LANES = [
                { resource: 'cpu', target: 'local-cpu', color: '#4A90E2', label: 'Local CPU' },
                { resource: 'gpu', target: 'local-gpu', color: '#7ED321', label: 'Local GPU' },
                { resource: 'cpu', target: 'remote',    color: '#9B9B9B', label: 'Remote CPU' },
                { resource: 'gpu', target: 'remote',    color: '#9B9B9B', label: 'Remote GPU' },
            ];
            const active = LANES
                .map(l => ({ ...l, items: _lane(l.resource, l.target) }))
                .filter(l => l.items);

            let html = '';
            if (!active.length) {
                html = '<div style="font-size:11px;color:var(--text-muted);padding:2px 0;">No active jobs</div>';
            } else {
                const grid = active.length > 1 ? 'display:grid;grid-template-columns:1fr 1fr;gap:10px;' : '';
                html = `<div style="${grid}">` + active.map(l => `
                    <div>
                        <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:4px;">
                            <span style="color:${l.color};">●</span> ${l.label}
                        </div>
                        ${l.items}
                    </div>`).join('') + '</div>';
            }

            // ── In-progress download(s) ───────────────────────────────
            for (const d of activeDl) {
                const lbl = d.current_label ? ` — ${d.current_label}` : '';
                html += `<div style="border-top:1px solid var(--border);margin-top:6px;padding:6px;background:rgba(74,158,255,0.10);border-radius:var(--radius);font-size:11px;">
                    ⬇ Downloading ${d.completed + 1}/${d.total}${lbl}
                </div>`;
            }

            // ── Pending downloads flag ────────────────────────────────
            if (pendingResults.length > 0) {
                const ids = [...new Set(pendingResults.map(p => p.job_id))];
                const summary = pendingResults.length === 1
                    ? `${pendingResults[0].subject_name} (${pendingResults[0].job_type})`
                    : `${pendingResults.length} results across ${ids.length} job(s)`;
                html += `<div style="border-top:1px solid var(--border);margin-top:6px;padding:6px;background:rgba(74,158,255,0.10);border-radius:var(--radius);">
                    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                        <span style="font-size:11px;flex:1;">⬇ <strong>Ready:</strong> ${summary}</span>
                        <button class="btn btn-sm" data-pending-download="${ids.join(',')}" style="font-size:9px;padding:0 4px;line-height:1.4;">Download all</button>
                        <button class="btn btn-sm" data-pending-ignore="${ids.join(',')}" style="font-size:9px;padding:0 4px;line-height:1.4;">Ignore all</button>
                    </div>
                </div>`;
            }

            // ── Recent job history (last 3 completed/failed) ──────────
            if (recentJobs.length > 0) {
                html += '<div style="border-top:1px solid var(--border);margin-top:6px;padding-top:6px;">';
                html += '<div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:4px;">Recent</div>';
                for (const j of recentJobs) {
                    const name = j.subject_name || ('Subject ' + (j.subject_id || '?'));
                    const type = (j.job_type || '').replace(/_/g, ' ');
                    const ok = j.status === 'completed';
                    const icon = ok ? '<span style="color:#4CAF50;">✓</span>' : '<span style="color:#f44;">✗</span>';
                    const ago = _timeAgo(j.finished_at);
                    const retryBtn = (!ok && j.subject_id && j.job_type)
                        ? `<button class="btn btn-sm" style="font-size:9px;padding:0 4px;line-height:1.4;flex-shrink:0;" data-retry-job="${j.id}" data-retry-type="${j.job_type}" data-retry-subj="${j.subject_id}" data-retry-params='${(j.params_json || '{}').replace(/'/g, '&#39;')}'>Retry</button>`
                        : '';
                    html += `<div style="display:flex;align-items:center;gap:6px;padding:2px 0;font-size:11px;">
                        ${icon}
                        <span style="font-weight:500;">${type}</span>
                        ${_badge(j)}
                        <span style="color:var(--text-muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${name}</span>
                        ${retryBtn}
                        <span style="font-size:9px;color:var(--text-muted);flex-shrink:0;">${ago}</span>
                    </div>`;
                }
                html += '</div>';
            }

            if (html !== _lastHtml) {
                panel.innerHTML = html;
                _lastHtml = html;
                // Wire pending-download/ignore buttons
                panel.querySelectorAll('[data-pending-download]').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        const ids = btn.dataset.pendingDownload.split(',').map(s => parseInt(s)).filter(Boolean);
                        btn.disabled = true;
                        btn.textContent = 'Starting…';
                        try {
                            await fetch('/api/remote/download-pending', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ job_ids: ids }),
                            });
                            btn.textContent = 'In progress';
                            setTimeout(() => window._navPollJobs(), 600);
                        } catch (_e) { btn.textContent = '✗'; }
                    });
                });
                panel.querySelectorAll('[data-pending-ignore]').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        const ids = btn.dataset.pendingIgnore.split(',').map(s => parseInt(s)).filter(Boolean);
                        btn.disabled = true; btn.textContent = '...';
                        try {
                            await fetch('/api/remote/ignore-pending', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ job_ids: ids }),
                            });
                            window._navPollJobs();
                        } catch (_e) { btn.textContent = '✗'; }
                    });
                });
                // Wire retry buttons
                panel.querySelectorAll('[data-retry-job]').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        const jobType = btn.dataset.retryType;
                        const subjId = parseInt(btn.dataset.retrySubj);
                        let params = {};
                        try { params = JSON.parse(btn.dataset.retryParams); } catch {}
                        btn.disabled = true;
                        btn.textContent = '...';
                        try {
                            await fetch('/api/remote/launch', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    job_type: jobType,
                                    subject_ids: [subjId],
                                    execution_target: 'remote',
                                    extra_params: params,
                                }),
                            });
                            btn.textContent = '✓';
                        } catch (err) {
                            btn.textContent = '✗';
                        }
                    });
                });
            }
        } catch (e) {
            // Silently ignore polling errors
        }
    }

    window._navPollJobs = _pollJobs;
    _pollJobs();
    setInterval(_pollJobs, 3000);
})();
