/* Subjects page: the list, its per-trial progress, and sync. */

const subjectsPage = (() => {
    let subjects = [];

    async function load() {
        subjects = await (await fetch('/api/subjects')).json();
        render();
        // Per-trial detail is one request per subject, so fetch it after the
        // table is already on screen rather than making the page wait.
        for (const s of subjects) loadTrialStatus(s.id);
    }

    function render() {
        const table = document.getElementById('subjectTable');
        const hint = document.getElementById('emptyHint');
        const body = document.getElementById('subjectBody');

        if (!subjects.length) {
            table.style.display = 'none';
            hint.style.display = 'block';
            fetch('/api/settings/status').then(r => r.json()).then(st => {
                const el = document.getElementById('videoDirHint');
                if (el && st.video_dir) el.textContent = st.video_dir;
            }).catch(() => {});
            return;
        }

        hint.style.display = 'none';
        table.style.display = '';
        body.innerHTML = subjects.map(s => `
            <tr data-id="${s.id}">
                <td class="name-col"><a href="/label?subject=${s.id}"
                    style="color:var(--text);text-decoration:none;">${s.name}</a>
                    ${packageBadge(s)}</td>
                <td><span class="badge badge-${s.stage}">${s.stage}</span></td>
                <td>${s.is_package ? `${s.package.image_count} frames` : s.video_count}</td>
                <td>${s.labeled_frame_count || 0}</td>
                <td class="prog-col"><div class="trial-chips" id="chips-${s.id}">
                    <span style="font-size:10px;color:var(--text-muted);">…</span>
                </div></td>
                <td style="text-align:right;white-space:nowrap;">
                    <a class="btn btn-sm" href="/label?subject=${s.id}">Label</a>
                    <a class="btn btn-sm" href="/jobs?subject=${s.id}">Jobs</a>
                </td>
            </tr>
        `).join('');
    }

    /* A received package is not a recording; saying so on the row is what
     * stops it reading as a subject whose videos have gone missing. */
    function packageBadge(s) {
        if (!s.is_package) return '';
        const from = s.package.subject ? ` from ${s.package.subject}` : '';
        return `<span class="badge" style="background:var(--surface);`
            + `border:1px solid var(--border);color:var(--text-muted);`
            + `margin-left:6px;" title="A labeling package${from}, `
            + `${s.package.image_count} frames to label">package</span>`;
    }

    async function loadTrialStatus(subjectId) {
        const box = document.getElementById(`chips-${subjectId}`);
        if (!box) return;
        try {
            const d = await (await fetch(`/api/subjects/${subjectId}`)).json();
            const rows = d.trial_status || [];
            if (!rows.length) {
                box.innerHTML = '<span style="font-size:10px;color:var(--text-muted);">no trials found</span>';
                return;
            }
            box.innerHTML = rows.map(t => {
                const st = t.status || {};
                // One chip per trial, coloured by the furthest stage it has
                // reached — the quickest read of "what still needs doing".
                const cls = st.corrections ? 'corr'
                    : st.analysis ? 'pred'
                    : (st.mp_forward || st.mp_reverse || st.mp_static) ? 'mp' : '';
                const passes = ['mp_forward', 'mp_reverse', 'mp_static']
                    .filter(k => st[k]).length;
                const short = t.name.includes('_')
                    ? t.name.split('_').slice(1).join('_') : t.name;
                const title = [
                    `${t.name} (${t.frame_count} frames)`,
                    `MediaPipe passes: ${passes || 'none'}`,
                    st.mp_best ? 'best-per-frame fusion: yes' : 'best-per-frame fusion: no',
                    st.analysis ? 'analyzed: yes' : 'analyzed: no',
                    st.corrections ? 'corrections saved: yes' : 'corrections saved: no',
                ].join('\n');
                return `<span class="trial-chip ${cls}" title="${title}">${short}</span>`;
            }).join('');
        } catch (e) {
            box.innerHTML = '<span style="font-size:10px;color:var(--red);">failed to load</span>';
        }
    }

    async function sync() {
        const btn = document.getElementById('syncBtn');
        const status = document.getElementById('syncStatus');
        btn.disabled = true;
        status.textContent = 'Scanning…';
        try {
            const r = await (await fetch('/api/subjects/sync', { method: 'POST' })).json();
            status.textContent = `${r.total} subject(s): ${r.created} new, `
                + `${r.updated} updated, ${r.removed} removed`;
            await load();
        } catch (e) {
            status.textContent = 'Sync failed — see the server log';
        } finally {
            btn.disabled = false;
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        document.getElementById('syncBtn').addEventListener('click', sync);
        load();
    });

    return { load, sync };
})();
