/* Shared fetch wrappers for DLC Pipeline API */

// Static-export mode: when ``window.STATIC_RESULTS`` is set (the
// GitHub-Pages build), GET requests to ``/api/...`` are redirected to
// pre-exported JSON files under ``data/`` (path+query flattened to a
// filename — must match scripts/export_results_static.py).
function _staticApiUrl(url) {
    const base = window.STATIC_DATA_BASE || 'data';
    const key = String(url).replace(/^\//, '').replace(/[^a-zA-Z0-9]+/g, '_');
    return `${base}/${key}.json`;
}

const API = {
    async get(url) {
        if (window.STATIC_RESULTS) url = _staticApiUrl(url);
        const resp = await fetch(url);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    async post(url, body = null) {
        const opts = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        };
        if (body !== null) opts.body = JSON.stringify(body);
        const resp = await fetch(url, opts);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    async put(url, body) {
        const resp = await fetch(url, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    async patch(url, body) {
        const resp = await fetch(url, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    async del(url) {
        const resp = await fetch(url, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        return resp.json();
    },

    // SSE helper for live log streaming
    streamJobLog(jobId, onText, onDone) {
        const source = new EventSource(`/api/jobs/${jobId}/log-stream`);
        source.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.done) {
                source.close();
                if (onDone) onDone(data);
            } else if (data.text) {
                onText(data.text);
            }
        };
        source.onerror = () => {
            source.close();
            if (onDone) onDone({ done: true, status: 'error' });
        };
        return source;
    },

    // SSE helper for job progress streaming
    streamJob(jobId, onData, onDone) {
        const source = new EventSource(`/api/jobs/${jobId}/stream`);
        source.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onData(data);
            if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                source.close();
                if (onDone) onDone(data);
            }
        };
        source.onerror = () => {
            source.close();
            if (onDone) onDone({ status: 'error' });
        };
        return source;
    },
};
