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
    '/videos', '/mediapipe', '/mediapipe-select', '/deidentify',
    '/labeling', '/labeling-select', '/mano', '/results', '/oscillations',
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
    prevBtn.className = 'btn btn-sm';
    prevBtn.innerHTML = '&larr;';
    prevBtn.title = 'Previous subject';
    prevBtn.style.cssText = 'padding:1px 6px;font-size:12px;';

    const nextBtn = document.createElement('button');
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
        sel.value = allSubjects[newIdx].id;
        setLastSubject(allSubjects[newIdx].id);
        _navigateToSubject(allSubjects[newIdx].id);
    }

    function _navigateToSubject(sid) {
        if (path.startsWith('/videos')) {
            window.location.href = '/videos?subject=' + sid;
        } else if (path.startsWith('/mediapipe') || path === '/mediapipe-select') {
            window.location.href = '/mediapipe-select?subject=' + sid;
        } else if (path.startsWith('/deidentify')) {
            window.location.href = '/deidentify?subject=' + sid;
        } else if (path.startsWith('/labeling') || path === '/labeling-select') {
            window.location.href = '/labeling-select?subject=' + sid;
        } else if (path.startsWith('/mano')) {
            window.location.href = '/mano?subject=' + sid;
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
        'videos': '/videos?subject=' + lastSubj,
        'mano': '/mano?subject=' + lastSubj,
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

    async function _pollJobs() {
        try {
            const data = await fetch('/api/queue/status').then(r => r.json());
            const running = data.running || [];
            const cpuQueue = data.cpu_queue || [];
            const gpuQueue = data.gpu_queue || [];

            const hasActive = running.length > 0 || cpuQueue.length > 0 || gpuQueue.length > 0;
            dot.style.display = hasActive ? 'inline-block' : 'none';

            function _jobRow(item) {
                const st = item.status || 'queued';
                const name = item.subject_ids ? JSON.parse(item.subject_ids).join(', ') : '?';
                const type = (item.job_type || '').replace(/_/g, ' ');
                const pct = item.progress_pct ? Math.round(item.progress_pct) + '%' : '';
                const dotCls = st === 'running' ? 'running' : 'queued';
                const qid = item.id;
                const jid = item.job_id || '';
                return `<div class="job-item">
                    <span class="job-dot ${dotCls}"></span>
                    <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:90px;">${name}</span>
                    <span style="color:var(--text-muted);font-size:10px;">${type} ${pct}</span>
                    <span class="job-actions">
                        ${jid ? `<button onclick="event.stopPropagation();fetch('/api/queue/${qid}/cancel',{method:'POST'}).then(()=>window._navPollJobs())" title="Cancel">×</button>` : ''}
                        ${jid ? `<button onclick="event.stopPropagation();window.open('/remote?job=${jid}','_blank')" title="Log">log</button>` : ''}
                    </span>
                </div>`;
            }

            const cpuItems = [...running.filter(r => r.resource === 'cpu'), ...cpuQueue];
            const gpuItems = [...running.filter(r => r.resource === 'gpu'), ...gpuQueue];

            let html = '<div class="jobs-grid">';
            html += '<div><h4>CPU</h4>';
            if (cpuItems.length === 0) html += '<div class="empty-msg">Empty</div>';
            else cpuItems.forEach(item => { html += _jobRow(item); });
            html += '</div>';
            html += '<div><h4>GPU</h4>';
            if (gpuItems.length === 0) html += '<div class="empty-msg">Empty</div>';
            else gpuItems.forEach(item => { html += _jobRow(item); });
            html += '</div></div>';
            html += `<div style="text-align:center;margin-top:6px;"><a href="/remote" style="font-size:11px;color:var(--blue);">Open Job Queue</a></div>`;

            if (html !== _lastHtml) {
                panel.innerHTML = html;
                _lastHtml = html;
            }
        } catch (e) {
            // Silently ignore polling errors
        }
    }

    window._navPollJobs = _pollJobs;
    _pollJobs();
    setInterval(_pollJobs, 3000);
})();
