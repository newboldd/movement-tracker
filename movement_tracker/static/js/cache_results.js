/* Cache Results: Jobs-page card with two collapsible jobs
   (Group Comparisons + Explore) that show + edit + rebuild the
   per-page static-cache config.  Source of truth lives in
   <DATA_DIR>/cache_config.json on the backend (via
   services/cache_config.py); the Results page reads the same file. */
(function () {
    const PAGES = [
        { id: 'group',   title: 'Group Comparisons' },
        { id: 'explore', title: 'Explore' },
    ];

    // ── State ──────────────────────────────────────────────────
    // ``serverCfg`` is the last config the backend persisted; we
    // compare against it to decide whether to show Revert+Cache.
    let serverCfg = null;
    let domains   = null;
    let labels    = null;
    let cacheInfo = null;
    let runs      = {};
    // ``localCfg[page]`` is what the UI currently shows — a mutable
    // copy of serverCfg[page] until the user clicks Revert/Cache.
    const localCfg = { group: null, explore: null };
    const expanded = new Set();

    // ── Helpers ────────────────────────────────────────────────
    function _fmtSize(n) {
        if (!n) return '0 B';
        if (n < 1024) return n + ' B';
        if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
        if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
        return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
    }
    function _arrEq(a, b) {
        if (!Array.isArray(a) || !Array.isArray(b)) return false;
        if (a.length !== b.length) return false;
        const s = new Set(a.map(x => JSON.stringify(x)));
        return b.every(x => s.has(JSON.stringify(x)));
    }
    function _cfgEq(a, b) {
        if (!a || !b) return false;
        return _arrEq(a.sources,    b.sources)
            && _arrEq(a.seq_modes,  b.seq_modes)
            && _arrEq(a.hand_trial, b.hand_trial);
    }
    function _cloneCfg(c) { return JSON.parse(JSON.stringify(c)); }

    // ── Fetch ──────────────────────────────────────────────────
    async function fetchConfig() {
        const data = await API.get('/api/cache-results/config');
        serverCfg = data.config;
        domains   = data.domains;
        labels    = data.labels;
        cacheInfo = data.cache;
        runs      = data.runs || {};
        for (const p of PAGES) localCfg[p.id] = _cloneCfg(serverCfg[p.id]);
        render();
    }

    // ── Render ─────────────────────────────────────────────────
    function render() {
        const totalLbl = document.getElementById('cacheTotalLabel');
        if (totalLbl) {
            totalLbl.textContent = cacheInfo
                ? `Cache size on disk: ${_fmtSize(cacheInfo.bytes)} · ${cacheInfo.files} files`
                : '';
        }
        const row = document.getElementById('cacheJobsRow');
        if (!row) return;
        row.innerHTML = '';
        for (const p of PAGES) row.appendChild(_buildJobCard(p));
    }

    function _buildJobCard(page) {
        const cfg = localCfg[page.id];
        const isExpanded = expanded.has(page.id);
        const dirty = !_cfgEq(cfg, serverCfg[page.id]);
        const run = runs[page.id];

        const card = document.createElement('div');
        card.style.cssText = 'border:1px solid var(--border);border-radius:var(--radius);'
                            + 'background:var(--bg);overflow:hidden;';

        // Header (always visible)
        const header = document.createElement('div');
        header.style.cssText = 'display:flex;align-items:center;gap:10px;padding:10px 14px;'
                              + 'cursor:pointer;user-select:none;background:var(--bg);';
        header.innerHTML = `
            <span style="font-weight:600;flex:0 0 auto;">${page.title}</span>
            <span style="font-size:11px;color:var(--text-muted);flex:1 1 auto;">
                ${cfg.sources.length} source(s) · ${cfg.seq_modes.length} seq mode(s) · ${cfg.hand_trial.length} (hand, trial) pair(s)
            </span>
            ${run ? _renderRunBadge(run) : ''}
            <span style="font-size:11px;color:var(--text-muted);flex:0 0 auto;">${isExpanded ? '▾' : '▸'}</span>
        `;
        header.addEventListener('click', () => {
            if (isExpanded) expanded.delete(page.id);
            else expanded.add(page.id);
            render();
        });
        card.appendChild(header);

        if (!isExpanded) return card;

        // Settings body
        const body = document.createElement('div');
        body.style.cssText = 'padding:12px 14px;border-top:1px solid var(--border);'
                            + 'display:flex;flex-direction:column;gap:12px;';

        body.appendChild(_buildCheckboxGroup('Sources', domains.sources, labels.sources,
            cfg.sources,
            (v, checked) => _toggleAxis(page.id, 'sources', v, checked)));

        body.appendChild(_buildCheckboxGroup('Sequence modes', domains.seq_modes, labels.seq_modes,
            cfg.seq_modes,
            (v, checked) => _toggleAxis(page.id, 'seq_modes', v, checked)));

        body.appendChild(_buildHandTrialGrid(page.id, cfg));

        // Action buttons (Revert + Cache) only when dirty
        const actions = document.createElement('div');
        actions.style.cssText = 'display:flex;gap:8px;align-items:center;'
                              + 'border-top:1px solid var(--border);padding-top:10px;';
        if (dirty) {
            const revert = document.createElement('button');
            revert.className = 'btn btn-sm';
            revert.textContent = 'Revert';
            revert.style.cssText = 'background:var(--blue);color:#fff;border-color:var(--blue);';
            revert.addEventListener('click', () => {
                localCfg[page.id] = _cloneCfg(serverCfg[page.id]);
                render();
            });
            actions.appendChild(revert);

            const cacheBtn = document.createElement('button');
            cacheBtn.className = 'btn btn-sm';
            cacheBtn.textContent = 'Cache';
            cacheBtn.style.cssText = 'background:var(--green);color:#fff;border-color:var(--green);';
            cacheBtn.addEventListener('click', () => _saveAndRebuild(page.id));
            actions.appendChild(cacheBtn);

            actions.appendChild(_textHint('Settings differ from the persisted config.'));
        } else {
            const rebuild = document.createElement('button');
            rebuild.className = 'btn btn-sm';
            rebuild.textContent = 'Rebuild cache';
            rebuild.title = 'Re-export every combo the current config covers';
            rebuild.addEventListener('click', () => _rebuildOnly(page.id));
            actions.appendChild(rebuild);
        }
        body.appendChild(actions);

        card.appendChild(body);
        return card;
    }

    function _textHint(txt) {
        const el = document.createElement('span');
        el.style.cssText = 'font-size:11px;color:var(--text-muted);';
        el.textContent = txt;
        return el;
    }

    function _renderRunBadge(run) {
        if (!run) return '';
        if (run.status === 'running' || run.status === 'starting') {
            const pct = run.n_total ? Math.round(100 * run.n_done / run.n_total) : 0;
            return `<span style="font-size:11px;color:var(--blue);">running · ${run.n_done}/${run.n_total} (${pct}%)</span>`;
        }
        if (run.status === 'done') {
            return `<span style="font-size:11px;color:var(--green);">last run wrote ${run.n_written}</span>`;
        }
        if (run.status === 'error') {
            return `<span style="font-size:11px;color:var(--red);">error: ${run.error || 'unknown'}</span>`;
        }
        return '';
    }

    function _buildCheckboxGroup(title, options, labelMap, selected, onToggle) {
        const wrap = document.createElement('div');
        const head = document.createElement('div');
        head.style.cssText = 'font-size:12px;font-weight:600;margin-bottom:4px;color:var(--text-muted);';
        head.textContent = title;
        wrap.appendChild(head);
        const grid = document.createElement('div');
        grid.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px 12px;';
        for (const opt of options) {
            const lbl = document.createElement('label');
            lbl.style.cssText = 'display:inline-flex;align-items:center;gap:4px;font-size:12px;cursor:pointer;';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = selected.includes(opt);
            cb.addEventListener('change', () => onToggle(opt, cb.checked));
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(' ' + (labelMap[opt] || opt)));
            grid.appendChild(lbl);
        }
        wrap.appendChild(grid);
        return wrap;
    }

    function _buildHandTrialGrid(pageId, cfg) {
        const wrap = document.createElement('div');
        const head = document.createElement('div');
        head.style.cssText = 'font-size:12px;font-weight:600;margin-bottom:4px;color:var(--text-muted);';
        head.textContent = 'Hand × Trial pairs';
        wrap.appendChild(head);

        const table = document.createElement('table');
        table.style.cssText = 'border-collapse:collapse;font-size:12px;';
        // Header row: trial labels
        const thead = document.createElement('thead');
        const trH = document.createElement('tr');
        const corner = document.createElement('th'); corner.textContent = '';
        corner.style.padding = '2px 8px'; trH.appendChild(corner);
        for (const t of domains.trials) {
            const th = document.createElement('th');
            th.textContent = labels.trials[t] || t;
            th.style.cssText = 'text-align:center;padding:2px 8px;color:var(--text-muted);font-weight:600;';
            trH.appendChild(th);
        }
        thead.appendChild(trH); table.appendChild(thead);
        // Body rows: one per hand
        const tbody = document.createElement('tbody');
        const pairKey = (h, t) => `${h}|${t}`;
        const pairSet = new Set(cfg.hand_trial.map(([h, t]) => pairKey(h, t)));
        for (const h of domains.hands) {
            const tr = document.createElement('tr');
            const th = document.createElement('th');
            th.textContent = labels.hands[h] || h;
            th.style.cssText = 'text-align:left;padding:2px 8px;color:var(--text-muted);font-weight:600;';
            tr.appendChild(th);
            for (const t of domains.trials) {
                const td = document.createElement('td');
                td.style.cssText = 'text-align:center;padding:2px 8px;';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = pairSet.has(pairKey(h, t));
                cb.addEventListener('change', () => _togglePair(pageId, h, t, cb.checked));
                td.appendChild(cb);
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        wrap.appendChild(table);
        return wrap;
    }

    function _toggleAxis(pageId, axis, value, checked) {
        const set = new Set(localCfg[pageId][axis]);
        if (checked) set.add(value);
        else set.delete(value);
        localCfg[pageId][axis] = [...set];
        render();
    }

    function _togglePair(pageId, hand, trial, checked) {
        const cur = localCfg[pageId].hand_trial;
        const key = JSON.stringify([hand, trial]);
        const has = cur.some(p => JSON.stringify(p) === key);
        if (checked && !has) cur.push([hand, trial]);
        else if (!checked && has) {
            localCfg[pageId].hand_trial = cur.filter(p => JSON.stringify(p) !== key);
        }
        render();
    }

    async function _saveAndRebuild(pageId) {
        const merged = { ...serverCfg, [pageId]: localCfg[pageId] };
        try {
            const r = await API.post('/api/cache-results/config', merged);
            serverCfg = r.config;
        } catch (e) {
            alert('Save failed: ' + e.message);
            return;
        }
        await _rebuildOnly(pageId);
    }

    async function _rebuildOnly(pageId) {
        try {
            await API.post(`/api/cache-results/rebuild/${pageId}`, {});
        } catch (e) {
            alert('Rebuild start failed: ' + e.message);
            return;
        }
        // Poll the config endpoint while the run is active.
        const tick = async () => {
            try {
                const d = await API.get('/api/cache-results/config');
                runs = d.runs || {};
                cacheInfo = d.cache;
                render();
                const st = runs[pageId]?.status;
                if (st === 'running' || st === 'starting') {
                    setTimeout(tick, 1000);
                }
            } catch (_) {}
        };
        tick();
    }

    // ── Init ───────────────────────────────────────────────────
    function init() {
        if (!document.getElementById('cacheJobsRow')) return;
        fetchConfig().catch(e => {
            console.warn('cache-results init failed', e);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
