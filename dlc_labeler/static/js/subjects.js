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
                    ${s.is_package
                        ? `<button class="btn btn-sm" data-import="${s.name}"
                             title="Map these labels back onto ${s.package.subject
                                 || 'their subject'}'s video frames">Import labels</button>`
                        : `<a class="btn btn-sm" href="/jobs?subject=${s.id}">Jobs</a>`}
                </td>
            </tr>
        `).join('');

        document.querySelectorAll('[data-import]').forEach(b =>
            b.addEventListener('click', () => importLabels(b, b.dataset.import)));
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

    /* Bringing a labeled package home.  Always a dry run first: the
     * person deserves to see how many frames land, and what collides
     * with labels already there, before anything is written. */
    async function importLabels(btn, packageName) {
        const label = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Checking…';
        try {
            const plan = await post(
                `/api/packages/${encodeURIComponent(packageName)}/import?dry_run=true`);
            const lines = [
                `${plan.imported} frame(s) would be added to ${plan.subject}.`,
                `${plan.labeled_images} of the package's images are labeled; `
                    + `${plan.unlabeled} are not.`,
            ];
            if (plan.conflicts.length) {
                lines.push('', `${plan.conflicts.length} frame(s) are already `
                    + 'labeled differently. Press OK to import everything else '
                    + 'and leave those alone.');
            }
            if (plan.missing_images.length) {
                lines.push('', `${plan.missing_images.length} image(s) were `
                    + 'deleted from the package and cannot be labeled.');
            }
            if (plan.unknown_trial.length) {
                lines.push('', `${plan.unknown_trial.length} image(s) do not `
                    + 'match any trial of this subject and will be skipped.');
            }
            if (plan.unknown_bodyparts.length) {
                lines.push('', 'Bodyparts this project does not have: '
                    + plan.unknown_bodyparts.join(', '));
            }
            if (!plan.imported && !plan.conflicts.length) {
                alert(lines.join('\n') + '\n\nNothing to import.');
                return;
            }
            if (!confirm(lines.join('\n'))) return;

            btn.textContent = 'Importing…';
            const done = await post(
                `/api/packages/${encodeURIComponent(packageName)}/import`);
            let msg = `Imported ${done.imported} frame(s) into ${done.subject}.`;
            if (done.conflicts.length) {
                msg += `\n\n${done.conflicts.length} frame(s) were left alone `
                    + 'because they are already labeled differently. Overwrite '
                    + 'them with the package\u2019s version?';
                if (confirm(msg)) {
                    const forced = await post(
                        `/api/packages/${encodeURIComponent(packageName)}`
                        + '/import?overwrite=true');
                    alert(`Overwrote ${forced.imported} frame(s).`);
                    return;
                }
            }
            alert(msg);
        } catch (e) {
            alert(`Could not import: ${e.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = label;
        }
    }

    async function post(url) {
        const r = await fetch(url, { method: 'POST' });
        if (!r.ok) {
            let detail = r.statusText;
            try { detail = (await r.json()).detail || detail; } catch (e) { /* not JSON */ }
            throw new Error(detail);
        }
        return r.json();
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
