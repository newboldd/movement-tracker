/* Shared fetch wrappers for DLC Pipeline API */

const API = {
    async get(url) {
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
