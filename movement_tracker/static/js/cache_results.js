/* Cache Results: integrates with the existing Jobs-page step-button
   pattern.  When the user clicks the Group Comparison or Explore
   step button, selectStep() in remote.js hides the per-subject
   pickers + Launch button and invokes window.cacheResultsMount(page).
   This file renders the settings panel (sources / seq modes /
   hand-trial pairs) into #cacheStepBody, shows current cache size,
   exposes Revert / Cache, and polls progress while a rebuild is
   running.  Source of truth: <DATA_DIR>/cache_config.json on the
   backend (services/cache_config.py) — the same file the Results
   page consults for cache-hit checks. */
(function () {
    // ── State ──────────────────────────────────────────────────
    let serverCfg = null;   // last persisted config (per page)
    let domains   = null;   // every legal value per axis
    let labels    = null;   // human-friendly per-value labels
    let cacheInfo = null;   // { bytes, files }
    let runs      = {};     // per-page rebuild progress
    const localCfg = { group: null, explore: null };
    let mountedPage = null; // 'group' / 'explore' / null

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
        for (const id of ['group', 'explore']) {
            if (!localCfg[id]) localCfg[id] = _cloneCfg(serverCfg[id]);
        }
    }

    // ── Render ─────────────────────────────────────────────────
    function render() {
        const body = document.getElementById('cacheStepBody');
        if (!body || !mountedPage) return;
        body.innerHTML = '';

        const cfg = localCfg[mountedPage];
        const dirty = !_cfgEq(cfg, serverCfg[mountedPage]);
        const run = runs[mountedPage];

        // Header line: size on disk + run badge.
        const head = document.createElement('div');
        head.style.cssText = 'display:flex;align-items:center;gap:12px;margin-bottom:10px;';
        const title = document.createElement('span');
        title.style.cssText = 'font-weight:600;font-size:13px;';
        title.textContent = mountedPage === 'group' ? 'Group Comparison cache' : 'Explore cache';
        head.appendChild(title);
        const sizeLbl = document.createElement('span');
        sizeLbl.style.cssText = 'font-size:11px;color:var(--text-muted);flex:1 1 auto;';
        sizeLbl.textContent = cacheInfo
            ? `Cache on disk: ${_fmtSize(cacheInfo.bytes)} · ${cacheInfo.files} files (both pages combined)`
            : '';
        head.appendChild(sizeLbl);
        if (run) {
            const badge = document.createElement('span');
            badge.style.cssText = 'font-size:11px;';
            if (run.status === 'running' || run.status === 'starting') {
                const pct = run.n_total ? Math.round(100 * run.n_done / run.n_total) : 0;
                badge.style.color = 'var(--blue)';
                badge.textContent = `running · ${run.n_done}/${run.n_total} (${pct}%)`;
            } else if (run.status === 'done') {
                badge.style.color = 'var(--green)';
                badge.textContent = `last run wrote ${run.n_written}`;
            } else if (run.status === 'error') {
                badge.style.color = 'var(--red)';
                badge.textContent = `error: ${run.error || 'unknown'}`;
            }
            head.appendChild(badge);
        }
        body.appendChild(head);

        // Settings columns
        const grid = document.createElement('div');
        grid.style.cssText = 'display:flex;flex-direction:column;gap:12px;';

        grid.appendChild(_buildCheckboxGroup('Sources',
            domains.sources, labels.sources, cfg.sources,
            (v, on) => _toggleAxis('sources', v, on)));
        grid.appendChild(_buildCheckboxGroup('Sequence modes',
            domains.seq_modes, labels.seq_modes, cfg.seq_modes,
            (v, on) => _toggleAxis('seq_modes', v, on)));
        grid.appendChild(_buildHandTrialGrid(cfg));
        body.appendChild(grid);

        // Actions row
        const actions = document.createElement('div');
        actions.style.cssText = 'display:flex;gap:8px;align-items:center;margin-top:12px;';
        if (dirty) {
            const revert = document.createElement('button');
            revert.className = 'btn btn-sm';
            revert.textContent = 'Revert';
            revert.style.cssText = 'background:var(--blue);color:#fff;border-color:var(--blue);';
            revert.addEventListener('click', () => {
                localCfg[mountedPage] = _cloneCfg(serverCfg[mountedPage]);
                render();
            });
            actions.appendChild(revert);

            const cacheBtn = document.createElement('button');
            cacheBtn.className = 'btn btn-sm';
            cacheBtn.textContent = 'Cache';
            cacheBtn.style.cssText = 'background:var(--green);color:#fff;border-color:var(--green);';
            cacheBtn.addEventListener('click', _saveAndRebuild);
            actions.appendChild(cacheBtn);

            const hint = document.createElement('span');
            hint.style.cssText = 'font-size:11px;color:var(--text-muted);';
            hint.textContent = 'Settings differ from the persisted config.';
            actions.appendChild(hint);
        } else {
            const rebuild = document.createElement('button');
            rebuild.className = 'btn btn-sm';
            rebuild.textContent = 'Rebuild cache';
            rebuild.title = 'Re-export every combo the current config covers';
            rebuild.addEventListener('click', _rebuildOnly);
            actions.appendChild(rebuild);
        }
        body.appendChild(actions);
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

    function _buildHandTrialGrid(cfg) {
        const wrap = document.createElement('div');
        const head = document.createElement('div');
        head.style.cssText = 'font-size:12px;font-weight:600;margin-bottom:4px;color:var(--text-muted);';
        head.textContent = 'Hand × Trial pairs';
        wrap.appendChild(head);

        const table = document.createElement('table');
        table.style.cssText = 'border-collapse:collapse;font-size:12px;';
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
                cb.addEventListener('change', () => _togglePair(h, t, cb.checked));
                td.appendChild(cb);
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        wrap.appendChild(table);
        return wrap;
    }

    function _toggleAxis(axis, value, checked) {
        const set = new Set(localCfg[mountedPage][axis]);
        if (checked) set.add(value);
        else set.delete(value);
        localCfg[mountedPage][axis] = [...set];
        render();
    }

    function _togglePair(hand, trial, checked) {
        const cur = localCfg[mountedPage].hand_trial;
        const key = JSON.stringify([hand, trial]);
        const has = cur.some(p => JSON.stringify(p) === key);
        if (checked && !has) cur.push([hand, trial]);
        else if (!checked && has) {
            localCfg[mountedPage].hand_trial = cur.filter(p => JSON.stringify(p) !== key);
        }
        render();
    }

    async function _saveAndRebuild() {
        const merged = { ...serverCfg, [mountedPage]: localCfg[mountedPage] };
        try {
            const r = await API.post('/api/cache-results/config', merged);
            serverCfg = r.config;
        } catch (e) {
            alert('Save failed: ' + e.message);
            return;
        }
        await _rebuildOnly();
    }

    async function _rebuildOnly() {
        try {
            await API.post(`/api/cache-results/rebuild/${mountedPage}`, {});
        } catch (e) {
            alert('Rebuild start failed: ' + e.message);
            return;
        }
        const tick = async () => {
            try {
                const d = await API.get('/api/cache-results/config');
                runs = d.runs || {};
                cacheInfo = d.cache;
                render();
                const st = runs[mountedPage]?.status;
                if (st === 'running' || st === 'starting') {
                    setTimeout(tick, 1000);
                }
            } catch (_) {}
        };
        tick();
    }

    // ── Public entry point — selectStep() in remote.js calls this. ──
    window.cacheResultsMount = async function (page) {
        mountedPage = page;
        if (!serverCfg) {
            try { await fetchConfig(); }
            catch (e) { console.warn('cache-results fetch failed', e); return; }
        }
        render();
    };
})();
