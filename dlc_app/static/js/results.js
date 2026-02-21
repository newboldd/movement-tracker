/* Results page: subject selection and results display */

let subjects = [];

async function loadSubjects() {
    try {
        subjects = await API.get('/api/subjects');
        const sel = document.getElementById('subjectSelect');
        subjects.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = `${s.name} (${s.stage})`;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load subjects:', e);
    }
}

async function loadResults(subjectId) {
    const content = document.getElementById('resultsContent');

    if (!subjectId) {
        content.innerHTML = '<div class="card" style="text-align:center;color:var(--text-muted);padding:40px;">Select a subject to view results</div>';
        return;
    }

    try {
        const results = await API.get(`/api/results/${subjectId}`);

        let html = `
            <div class="card">
                <h2>${results.subject}</h2>
                <div style="display:flex;gap:8px;align-items:center;margin-bottom:16px;">
                    <span class="badge badge-${results.stage}">${results.stage.replace(/_/g, ' ')}</span>
                    ${results.has_labels_v1 ? '<span style="color:var(--green);font-size:13px;">Has analysis outputs</span>' : ''}
                </div>
        `;

        if (results.csv_files && results.csv_files.length > 0) {
            html += `
                <h3 style="font-size:14px;color:var(--text-muted);margin:12px 0 8px;">CSV Outputs</h3>
                <div style="font-size:13px;">
                    ${results.csv_files.map(f => `<div style="padding:2px 0;">${f}</div>`).join('')}
                </div>
            `;
        }

        if (results.video_files && results.video_files.length > 0) {
            html += `
                <h3 style="font-size:14px;color:var(--text-muted);margin:12px 0 8px;">Labeled Videos</h3>
                <div style="font-size:13px;">
                    ${results.video_files.map(f => `<div style="padding:2px 0;">${f}</div>`).join('')}
                </div>
            `;
        }

        if (results.has_corrections) {
            html += `
                <h3 style="font-size:14px;color:var(--text-muted);margin:12px 0 8px;">Manual Corrections</h3>
                <div style="font-size:13px;">
                    ${results.correction_files.map(f => `<div style="padding:2px 0;">${f}</div>`).join('')}
                </div>
            `;
        }

        html += '</div>';
        content.innerHTML = html;

    } catch (e) {
        content.innerHTML = `<div class="card" style="color:var(--red);">${e.message}</div>`;
    }
}

document.getElementById('subjectSelect').addEventListener('change', (e) => {
    loadResults(e.target.value);
});

// Check URL params
const params = new URLSearchParams(window.location.search);
const subjectId = params.get('subject');

loadSubjects().then(() => {
    if (subjectId) {
        document.getElementById('subjectSelect').value = subjectId;
        loadResults(subjectId);
    }
});
