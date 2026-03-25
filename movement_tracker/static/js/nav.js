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

// ── Subject-specific pages that should show the global selector ──
const SUBJECT_PAGES = [
    '/videos', '/mediapipe', '/mediapipe-select', '/deidentify',
    '/labeling', '/labeling-select', '/mano', '/results', '/oscillations',
];

/* ── Inject global subject selector into the header ────────────── */
(function() {
    const path = window.location.pathname;
    const isSubjectPage = SUBJECT_PAGES.some(p => path.startsWith(p));
    if (!isSubjectPage) return;

    const header = document.querySelector('.header');
    if (!header) return;
    const h1 = header.querySelector('h1');
    if (!h1) return;

    // Create selector container — insert right after the h1, before nav
    const selectorDiv = document.createElement('div');
    selectorDiv.id = 'globalSubjectSelector';
    selectorDiv.style.cssText = 'display:flex;align-items:center;gap:8px;margin-left:16px;flex-shrink:0;';

    // Subject dropdown
    const sel = document.createElement('select');
    sel.id = 'globalSubjectSelect';
    sel.style.cssText = 'padding:4px 8px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:13px;max-width:180px;';

    // Trial buttons container
    const trialDiv = document.createElement('div');
    trialDiv.id = 'globalTrialBtns';
    trialDiv.style.cssText = 'display:flex;gap:2px;flex-wrap:nowrap;overflow-x:auto;';

    selectorDiv.appendChild(sel);
    selectorDiv.appendChild(trialDiv);

    // Insert after h1
    h1.style.cssText = (h1.style.cssText || '') + ';flex-shrink:0;';
    h1.insertAdjacentElement('afterend', selectorDiv);

    // Make header flex
    header.style.display = 'flex';
    header.style.alignItems = 'center';

    // Fetch subjects and populate
    fetch('/api/subjects')
        .then(r => r.json())
        .then(subjects => {
            sel.innerHTML = '<option value="">Subject...</option>';
            subjects.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s.id;
                opt.textContent = s.name;
                sel.appendChild(opt);
            });

            // Pre-select from nav state or URL
            const params = new URLSearchParams(window.location.search);
            const urlSubj = params.get('subject') || params.get('session');
            const lastSubj = getLastSubject();
            if (urlSubj) {
                // Don't pre-select from URL session param — that's a session ID not subject ID
                if (params.get('subject')) sel.value = params.get('subject');
            } else if (lastSubj) {
                sel.value = lastSubj;
            }

            // Load trials for selected subject
            if (sel.value) _loadTrials(parseInt(sel.value), subjects);

            sel.addEventListener('change', () => {
                if (!sel.value) return;
                const sid = parseInt(sel.value);
                setLastSubject(sid);
                _loadTrials(sid, subjects);
                _navigateToSubject(sid);
            });
        })
        .catch(() => {});

    // Style for trial buttons
    const style = document.createElement('style');
    style.textContent = `
        .global-trial-btn {
            padding: 2px 8px;
            font-size: 11px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--card);
            color: var(--text);
            cursor: pointer;
            white-space: nowrap;
        }
        .global-trial-btn.active {
            background: var(--blue);
            color: #fff;
            border-color: var(--blue);
        }
        .global-trial-btn:hover:not(.active) {
            background: var(--hover);
        }
    `;
    document.head.appendChild(style);

    function _loadTrials(subjectId, subjects) {
        const subj = subjects.find(s => s.id === subjectId);
        if (!subj) return;

        // Use the video-tools probe to get trial list, or just clear trials
        // Trials are page-specific — the global selector just shows subject
        trialDiv.innerHTML = '';
    }

    function _navigateToSubject(sid) {
        const path = window.location.pathname;

        // Each page has its own URL pattern for subject selection
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
