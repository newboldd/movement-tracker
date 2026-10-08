/* Shared header: nav links, subject selector, trial buttons, job dot.
 *
 * Every page gets the same header markup from here rather than copying it
 * into four HTML files, so adding a page or renaming a link is one edit.
 * Pages that work on one subject at a time opt into the subject/trial
 * controls by setting `data-subject-nav` on <body>. */

const NAV_LINKS = [
    { href: '/subjects', label: 'Subjects' },
    { href: '/label', label: 'Label' },
    { href: '/jobs', label: 'Jobs' },
    { href: '/settings', label: 'Settings' },
];

/* ── Cross-page position memory ───────────────────────────────────
 * Which subject you were on belongs to the browser (localStorage, so it
 * survives a restart); which trial and frame belongs to the visit
 * (sessionStorage).  Together they mean switching pages puts you back
 * where you were instead of at subject one, frame zero. */
function setNavState(state) {
    try {
        if (state.subjectId) localStorage.setItem('dlc_lastSubjectId', String(state.subjectId));
        if (state.trialIdx != null) sessionStorage.setItem('dlc_trialIdx', String(state.trialIdx));
        if (state.frame != null) sessionStorage.setItem('dlc_frame', String(state.frame));
        if (state.side) sessionStorage.setItem('dlc_side', state.side);
    } catch (e) { /* private browsing — position memory is a nicety */ }
}

function getNavState() {
    const intOrNull = (v) => {
        const n = parseInt(v, 10);
        return Number.isFinite(n) ? n : null;
    };
    try {
        return {
            subjectId: intOrNull(localStorage.getItem('dlc_lastSubjectId')),
            trialIdx: intOrNull(sessionStorage.getItem('dlc_trialIdx')),
            frame: intOrNull(sessionStorage.getItem('dlc_frame')),
            side: sessionStorage.getItem('dlc_side') || null,
        };
    } catch (e) {
        return { subjectId: null, trialIdx: null, frame: null, side: null };
    }
}

function buildHeader() {
    const header = document.querySelector('.header');
    if (!header) return;

    const path = window.location.pathname;
    const links = NAV_LINKS.map(l => {
        const active = (l.href === path) || (path === '/' && l.href === '/subjects');
        return `<a href="${l.href}"${active ? ' class="active"' : ''}>${l.label}</a>`;
    }).join('');

    header.innerHTML = `
        <h1><a href="/subjects">DLC Labeler</a></h1>
        <div id="navSubjectBar" style="display:none;align-items:center;gap:6px;margin-left:12px;">
            <select id="navSubjectSelect" title="Subject"></select>
            <button class="btn btn-sm" id="navPrevSubject" title="Previous subject">&larr;</button>
            <button class="btn btn-sm" id="navNextSubject" title="Next subject">&rarr;</button>
            <div id="navTrialBtns" style="display:flex;gap:3px;margin-left:8px;"></div>
        </div>
        <div id="navPageSlot" style="display:flex;align-items:center;gap:8px;margin-left:auto;"></div>
        <nav>${links}</nav>
    `;

    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.gap = '10px';

    const sel = document.getElementById('navSubjectSelect');
    if (sel) {
        sel.style.cssText = 'padding:3px 6px;background:var(--bg);border:1px solid var(--border);'
            + 'border-radius:var(--radius);color:var(--text);font-size:12px;max-width:180px;';
    }

    if (document.body.hasAttribute('data-subject-nav')) {
        document.getElementById('navSubjectBar').style.display = 'flex';
    }

    const style = document.createElement('style');
    style.textContent = `
        .nav-trial-btn {
            padding: 1px 7px; font-size: 11px; border: 1px solid var(--border);
            border-radius: var(--radius); background: var(--bg);
            color: var(--text-muted); cursor: pointer; white-space: nowrap;
            font-family: inherit;
        }
        .nav-trial-btn:hover:not(.active) { background: var(--bg-hover); color: var(--text); }
        .nav-trial-btn.active {
            background: var(--blue); border-color: var(--blue); color: #1f0f3a;
            font-weight: 600;
        }
        .nav-trial-btn.has-data::after { content: ' •'; color: var(--green); }
    `;
    document.head.appendChild(style);
}

/* Fill the subject selector and wire prev/next.
 * `onChange(subjectId)` runs on every selection, including the first. */
async function initSubjectNav(onChange) {
    const sel = document.getElementById('navSubjectSelect');
    if (!sel) return [];

    let subjects = [];
    try {
        subjects = await (await fetch('/api/subjects')).json();
    } catch (e) {
        console.error('Could not load subjects', e);
        return [];
    }

    sel.innerHTML = subjects.map(s =>
        `<option value="${s.id}">${s.name}</option>`).join('');

    const urlId = parseInt(new URLSearchParams(location.search).get('subject'), 10);
    const remembered = getNavState().subjectId;
    const initial = [urlId, remembered].find(
        id => Number.isFinite(id) && subjects.some(s => s.id === id));
    if (initial != null) sel.value = String(initial);

    const fire = () => {
        const id = parseInt(sel.value, 10);
        if (!Number.isFinite(id)) return;
        setNavState({ subjectId: id });
        onChange(id);
    };

    sel.addEventListener('change', fire);
    const step = (delta) => {
        const idx = subjects.findIndex(s => s.id === parseInt(sel.value, 10));
        const next = subjects[idx + delta];
        if (next) { sel.value = String(next.id); fire(); }
    };
    document.getElementById('navPrevSubject')?.addEventListener('click', () => step(-1));
    document.getElementById('navNextSubject')?.addEventListener('click', () => step(1));

    if (subjects.length) fire();
    return subjects;
}

/* Render the trial buttons. `trials` comes from the session info payload. */
function renderTrialButtons(trials, activeIdx, onPick) {
    const box = document.getElementById('navTrialBtns');
    if (!box) return;
    box.innerHTML = '';
    trials.forEach((t, i) => {
        const btn = document.createElement('button');
        btn.className = 'nav-trial-btn' + (i === activeIdx ? ' active' : '');
        // Trials are named {Subject}_{Trial}; the subject part is already
        // in the selector next to it, so show just the trial.
        btn.textContent = t.trial_name.includes('_')
            ? t.trial_name.split('_').slice(1).join('_')
            : t.trial_name;
        btn.title = `${t.trial_name} — ${t.frame_count} frames`;
        btn.addEventListener('click', () => onPick(i));
        box.appendChild(btn);
    });
}

/* Poll the queue and show a dot on the Jobs link while work is running. */
function startJobDot() {
    const link = document.querySelector('nav a[href="/jobs"]');
    if (!link) return;
    const dot = document.createElement('span');
    dot.className = 'nav-job-dot';
    dot.style.display = 'none';
    link.prepend(dot);

    async function poll() {
        try {
            const st = await (await fetch('/api/queue/state')).json();
            const n = (st.running || []).length;
            const q = (st.cpu_queue || []).length + (st.gpu_queue || []).length;
            dot.style.display = (n || q) ? 'inline-block' : 'none';
            dot.title = n ? `${n} running, ${q} queued` : `${q} queued`;
        } catch (e) { /* server restarting — try again next tick */ }
    }
    poll();
    setInterval(poll, 4000);
}

document.addEventListener('DOMContentLoaded', () => {
    buildHeader();
    startJobDot();
});
