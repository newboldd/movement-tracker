/* Dashboard: subject table, job indicators, detail panel, diagnosis grouping */

let subjects = [];
let activeJobs = [];
let appStatus = { configured: true, has_calibration: false };
let calibrationNames = [];
let diagnosisGroups = ["Control", "MSA", "PD", "PSP"];
// ── Check settings status ────────────────────────────────
async function checkStatus() {
    try {
        appStatus = await API.get('/api/settings/status');
        const banner = document.getElementById('welcomeBanner');
        if (!appStatus.configured) {
            banner.style.display = 'block';
        } else {
            banner.style.display = 'none';
        }
    } catch (e) { /* ignore */ }
}

// ── Load diagnosis groups from settings ───────────────
async function loadDiagnosisGroups() {
    try {
        const settings = await API.get('/api/settings');
        diagnosisGroups = settings.diagnosis_groups || ["Control", "MSA", "PD", "PSP"];
    } catch (e) {
        diagnosisGroups = ["Control", "MSA", "PD", "PSP"];
    }
}

// ── Load subjects ─────────────────────────────────────
async function loadSubjects() {
    try {
        subjects = await API.get('/api/subjects');
        await loadDiagnosisGroups();
        renderDiagnosisGroups();
        populateStageFilter();
        loadJobs();
        // Load sparkline distance previews (non-blocking)
        loadSparklines();
    } catch (e) {
        document.getElementById('subjectTableBody').innerHTML =
            `<tr><td colspan="9" style="text-align:center;color:var(--red)">${e.message}</td></tr>`;
    }
}

async function loadSparklines() {
    try {
        // Load from file cache (instant if cached)
        const data = await API.get('/api/results/preview-distances');
        _drawSparklines(data.previews || {});

        // Background refresh: check for updates without blocking
        API.post('/api/results/preview-distances/refresh').then(res => {
            if (res.updated > 0) {
                // Some previews were updated — reload from cache
                API.get('/api/results/preview-distances').then(fresh => {
                    _drawSparklines(fresh.previews || {});
                });
            }
        }).catch(() => {});  // silently ignore refresh errors
    } catch (e) {
        console.log('Sparkline load failed:', e);
    }
}

function _drawSparklines(previews) {
    try {
        for (const [sid, info] of Object.entries(previews)) {
            const canvas = document.getElementById(`spark_${sid}`);
            if (!canvas) continue;

            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const w = canvas.clientWidth;
            const h = canvas.clientHeight;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            ctx.scale(dpr, dpr);

            const values = info.values || [];
            if (values.length < 2) continue;

            // Fixed y range: 0-200mm
            const yMin = 0, yMax = 200;

            // Draw line
            ctx.strokeStyle = '#4a9eff';
            ctx.lineWidth = 1;
            ctx.beginPath();
            let started = false;
            for (let i = 0; i < values.length; i++) {
                if (values[i] == null) { started = false; continue; }
                const x = (i / (values.length - 1)) * w;
                const y = h - ((values[i] - yMin) / (yMax - yMin)) * h;
                const cy = Math.max(0, Math.min(h, y));
                if (!started) { ctx.moveTo(x, cy); started = true; }
                else ctx.lineTo(x, cy);
            }
            ctx.stroke();
        }

        // Gray out canvases with no data
        subjects.forEach(s => {
            if (!previews[String(s.id)]) {
                const canvas = document.getElementById(`spark_${s.id}`);
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    const w = canvas.clientWidth;
                    const h = canvas.clientHeight;
                    const dpr = window.devicePixelRatio || 1;
                    canvas.width = w * dpr;
                    canvas.height = h * dpr;
                    ctx.scale(dpr, dpr);
                    ctx.fillStyle = 'var(--text-muted)';
                    ctx.font = '10px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillStyle = '#666';
                    ctx.fillText('No distance data', w / 2, h / 2 + 3);
                }
            }
        });
    } catch (e) {
        console.log('Sparkline draw failed:', e);
    }
}

// ── Render subjects grouped by diagnosis ──────────────
function renderDiagnosisGroups() {
    const container = document.getElementById('subjectTableBody');
    const filtered = getFilteredSubjects();

    document.getElementById('subjectCount').innerHTML =
        `Showing <strong>${filtered.length}</strong> of ${subjects.length} subjects`;
    const totalEl = document.getElementById('totalSubjectCount');
    if (totalEl) totalEl.textContent = `n = ${subjects.length}`;

    if (filtered.length === 0) {
        container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:40px;">No subjects found</div>';
        return;
    }

    // Group by diagnosis
    const grouped = {};
    diagnosisGroups.forEach(dg => {
        grouped[dg] = [];
    });

    filtered.forEach(s => {
        const group = s.group_label || s.diagnosis || "Control";
        if (!grouped[group]) {
            grouped[group] = [];
        }
        grouped[group].push(s);
    });

    let html = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; width: 100%;">';

    diagnosisGroups.forEach(diagnosis => {
        const groupSubjects = grouped[diagnosis] || [];
        html += `
            <div style="background: var(--bg-card); border-radius: var(--radius); padding: 12px; border: 1px solid var(--border);">
                <h3 style="margin: 0 0 12px 0; font-size: 14px; color: var(--text); display: flex; justify-content: space-between; align-items: center;">${diagnosis} <span style="font-weight: 400; font-size: 12px; color: var(--text-muted);">n=${groupSubjects.length}</span></h3>
                <div style="display: flex; flex-direction: column; gap: 6px;">
        `;

        if (groupSubjects.length === 0) {
            html += '<span style="font-size: 12px; color: var(--text-muted); text-align: center;">No subjects</span>';
        } else {
            groupSubjects.forEach(s => {
                const stageColor = `badge-${s.stage}`;
                html += `
                    <div style="padding: 8px; background: var(--bg); border-radius: 4px; border: 1px solid var(--border);">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-weight:600;font-size:13px;">${s.name}</span>
                            <div style="display:flex;align-items:center;gap:4px;">
                                ${(s.has_faces && !s.has_deident) ? `<a href="/deidentify?subject=${s.id}" onclick="sessionStorage.setItem('lastSubjectId','${s.id}')" style="color:var(--red);font-size:11px;text-decoration:none;cursor:pointer;" title="Needs deidentification">Deident</a>` : ''}
                                <span class="badge ${stageColor}" style="font-size:11px;">${stageLabel(s.stage)}</span>
                            </div>
                        </div>
                        <canvas id="spark_${s.id}" width="240" height="60"
                                style="width:100%;height:60px;cursor:pointer;display:block;margin-bottom:4px;"
                                onclick="sessionStorage.setItem('dlc_lastSubjectId','${s.id}');window.location.href='/results?subject=${s.id}&tab=individual&from=dashboard'"></canvas>
                        <div style="display:flex;gap:2px;flex-wrap:nowrap;">
                            <button class="btn btn-sm" style="white-space:nowrap;padding:2px 5px;font-size:11px;" onclick="showDetail(${s.id})">Info</button>
                            <button class="btn btn-sm" style="white-space:nowrap;padding:2px 5px;font-size:11px;" onclick="sessionStorage.setItem('lastSubjectId','${s.id}');window.location.href='/analyze?subject=${s.id}'">Detect</button>
                            <button class="btn btn-sm" style="white-space:nowrap;padding:2px 5px;font-size:11px;" onclick="openLabeling(${s.id})">DLC</button>
                            <button class="btn btn-sm" style="white-space:nowrap;padding:2px 5px;font-size:11px;" onclick="sessionStorage.setItem('lastSubjectId','${s.id}');window.location.href='/mano?subject=${s.id}'">MANO</button>
                            <button class="btn btn-sm" style="white-space:nowrap;padding:2px 5px;font-size:11px;" onclick="sessionStorage.setItem('lastSubjectId','${s.id}');window.location.href='/results?subject=${s.id}&from=dashboard'">Results</button>
                        </div>
                    </div>
                `;
            });
        }

        html += '</div></div>';
    });

    html += '</div>';
    container.innerHTML = html;
}

// ── Stage label shortening ────────────────────────────
function stageLabel(stage) {
    const overrides = { events_partial: 'events', events_complete: 'complete' };
    return (overrides[stage] || stage).replace(/_/g, ' ');
}

// Canonical stage order (matches pipeline progression)
const STAGE_ORDER = [
    "created", "videos_linked", "prelabeled", "labeling", "labeled",
    "committed", "training", "training_dataset_created", "trained",
    "cropping", "cropped", "analyzing", "analyzed",
    "triangulating", "triangulated", "refined", "corrected",
    "retraining", "retrained",
    "events_partial", "events_complete", "complete",
];

function populateStageFilter() {
    const present = new Set(subjects.map(s => s.stage));
    const sel = document.getElementById('stageFilter');
    sel.innerHTML = '<option value="">All stages</option>';
    // Add stages in pipeline order
    STAGE_ORDER.forEach(st => {
        if (present.has(st)) {
            sel.innerHTML += `<option value="${st}">${stageLabel(st)}</option>`;
        }
    });
    // Any stages not in the canonical list (e.g. error states) go at the end
    [...present].filter(st => !STAGE_ORDER.includes(st)).sort().forEach(st => {
        sel.innerHTML += `<option value="${st}">${stageLabel(st)}</option>`;
    });
}

function getFilteredSubjects() {
    const text = document.getElementById('filterInput').value.toLowerCase();
    const stage = document.getElementById('stageFilter').value;
    return subjects.filter(s => {
        if (text && !s.name.toLowerCase().includes(text)) return false;
        if (stage && s.stage !== stage) return false;
        return true;
    });
}

function renderTable() {
    const filtered = getFilteredSubjects();
    document.getElementById('subjectCount').innerHTML =
        `Showing <strong>${filtered.length}</strong> of ${subjects.length} subjects`;

    const tbody = document.getElementById('subjectTableBody');
    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--text-muted)">No subjects found</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(s => `
        <tr>
            <td class="name-col" onclick="showDetail(${s.id})">${s.name}</td>
            <td><button class="btn btn-sm" onclick="openFinal(${s.id})">View</button></td>
            <td><span class="badge badge-${s.stage}">${s.stage.replace(/_/g, ' ')}</span></td>
            <td>${getActions(s)}</td>
            <td>${s.video_count}</td>
            <td>${s.has_mediapipe ? '&check;' : '&mdash;'}</td>
            <td>${s.has_pose ? '&check;' : '&mdash;'}</td>
            <td>${s.has_blur
                ? '<span style="color:var(--green);">&check;</span>'
                : s.has_faces
                    ? '<a href="/deidentify?subject=' + s.id + '" class="btn btn-sm" style="padding:1px 6px;font-size:11px;text-decoration:none;">Deident</a>'
                    : '&mdash;'}</td>
            <td>${s.has_labels ? '&check;' : '&mdash;'}</td>
            <td>${s.has_snapshots ? '&check;' : '&mdash;'}</td>
        </tr>
    `).join('');
}

function getActions(subject) {
    const btns = [];
    const s = subject.stage;

    // Label / Restart button (always visible)
    btns.push(`<button class="btn btn-sm" onclick="openLabeling(${subject.id})">Label</button>`);

    // Stage-specific actions
    if (s === 'committed' || s === 'labeled') {
        btns.push(`<button class="btn btn-sm btn-primary" onclick="runStep(${subject.id}, 'train')">Train</button>`);
    }
    // Refine available for analyzed and beyond
    const refineStages = ['analyzed', 'refined', 'corrected'];
    if (refineStages.includes(s)) {
        btns.push(`<button class="btn btn-sm" onclick="openRefine(${subject.id})">Refine</button>`);
        btns.push(`<button class="btn btn-sm" onclick="openCorrections(${subject.id})">Corrections</button>`);
    }

    return btns.join(' ');
}

// ── Calibration helpers ──────────────────────────────
async function loadCalibrationNames() {
    try {
        const settings = await API.get('/api/settings');
        calibrationNames = Object.keys(settings.calibrations || {});
    } catch (e) { calibrationNames = []; }
}

async function updateSubjectCamera(subjectId, cameraName) {
    try {
        await API.patch(`/api/subjects/${subjectId}`, { camera_name: cameraName });
    } catch (e) {
        alert('Error updating camera: ' + e.message);
    }
}

async function updateSubjectDiagnosis(subjectId, group) {
    try {
        await API.patch(`/api/subjects/${subjectId}`, { group_label: group });
        // Refresh the subjects list to update the grouped view
        await loadSubjects();
    } catch (e) {
        alert('Error updating group: ' + e.message);
    }
}

async function onDiagnosisSelectChange(subjectId, sel) {
    if (sel.value !== '__new__') {
        updateSubjectDiagnosis(subjectId, sel.value);
        return;
    }
    const name = prompt('Enter new group name:');
    if (!name || !name.trim()) {
        sel.value = sel.dataset.prev || 'Control';
        return;
    }
    const trimmed = name.trim();
    if (!diagnosisGroups.includes(trimmed)) {
        diagnosisGroups.push(trimmed);
        try {
            const cfg = await API.get('/api/settings');
            cfg.diagnosis_groups = diagnosisGroups;
            await API.put('/api/settings', cfg);
        } catch { /* best-effort */ }
    }
    updateSubjectDiagnosis(subjectId, trimmed);
}

async function updateSubjectCameraMode(subjectId, cameraMode) {
    try {
        await API.patch(`/api/subjects/${subjectId}`, { camera_mode: cameraMode });
    } catch (e) {
        alert('Error updating camera mode: ' + e.message);
    }
}

async function updateNoFaceVideos(subjectId) {
    const checkboxes = document.querySelectorAll('.no-face-cb');
    const noFaceVideos = [];
    checkboxes.forEach(cb => {
        if (!cb.checked) noFaceVideos.push(cb.dataset.stem);
    });
    try {
        await API.patch(`/api/subjects/${subjectId}`, { no_face_videos: noFaceVideos });
    } catch (e) {
        alert('Error updating face status: ' + e.message);
    }
}

// ── Detail panel ─────────────────────────────────────
async function showDetail(subjectId) {
    try {
        const detail = await API.get(`/api/subjects/${subjectId}`);
        const panel = document.getElementById('detailPanel');
        document.getElementById('detailName').textContent = detail.name;

        const selStyle = 'padding:2px 6px;font-size:12px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);width:100%;';
        const inputStyle = 'padding:2px 6px;font-size:12px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);width:60px;';

        // Helper for editable clinical field
        function clinField(label, field, value, type) {
            const v = value || '';
            if (type === 'number') {
                return `<div class="info-row"><span class="label">${label}</span><input type="number" value="${v}" style="${inputStyle}" onchange="updateClinical(${detail.id},'${field}',this.value)"></div>`;
            } else if (type === 'select') {
                return ''; // handled separately
            }
            return `<div class="info-row"><span class="label">${label}</span><input type="text" value="${v}" style="${inputStyle}width:100px;" onchange="updateClinical(${detail.id},'${field}',this.value)"></div>`;
        }

        function selectField(label, field, value, options) {
            const opts = options.map(o => `<option value="${o}"${o === (value || '') ? ' selected' : ''}>${o}</option>`).join('');
            return `<div class="info-row"><span class="label">${label}</span><select style="${selStyle}width:auto;" onchange="updateClinical(${detail.id},'${field}',this.value)">${opts}</select></div>`;
        }

        function yesNoField(label, field, value) {
            return selectField(label, field, value, ['', 'Yes', 'No']);
        }

        // ── Trial checkmark chart ──
        const trialStatus = detail.trial_status || [];
        const features = [
            { key: 'deident', label: 'Deident' },
            { key: 'mp_hands', label: 'MP Hands' },
            { key: 'mp_pose', label: 'MP Pose' },
            { key: 'dlc_train', label: 'DLC Train' },
            { key: 'dlc_analysis', label: 'DLC Analyze' },
            { key: 'dlc_refine', label: 'DLC Refine' },
            { key: 'dlc_corrections', label: 'Corrections' },
            { key: 'events', label: 'Events' },
            { key: 'mano', label: 'MANO' },
        ];
        let trialChart = '';
        if (trialStatus.length > 0) {
            // Short trial labels (e.g. "L1" from "Con07_L1")
            const shortNames = trialStatus.map(t => {
                const parts = t.name.split('_');
                return parts.length > 1 ? parts.slice(1).join('_') : t.name;
            });
            trialChart = `<table style="font-size:11px;border-collapse:collapse;width:100%;">
                <tr><td></td>${shortNames.map(n => `<td style="text-align:center;padding:1px 4px;font-weight:600;">${n}</td>`).join('')}</tr>
                ${features.map(f => `<tr>
                    <td style="padding:1px 4px;color:var(--text-muted);white-space:nowrap;">${f.label}</td>
                    ${trialStatus.map(t => {
                        const val = t.status[f.key];
                        return `<td style="text-align:center;padding:1px 4px;">${val ? '<span style="color:var(--green);">&#10003;</span>' : '<span style="color:var(--text-muted);">&#8212;</span>'}</td>`;
                    }).join('')}
                </tr>`).join('')}
            </table>`;
        }

        document.getElementById('detailContent').innerHTML = `
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
                <!-- Column 1: Clinical -->
                <div class="detail-section">
                    <h3>Clinical</h3>
                    ${clinField('Age', 'age', detail.age, 'number')}
                    ${selectField('Sex', 'sex', detail.sex, ['', 'M', 'F', 'Other'])}
                    ${selectField('Handedness', 'handedness', detail.handedness, ['', 'R', 'L', 'Ambidextrous'])}
                    ${clinField('Diagnosis', 'diagnosis', detail.diagnosis)}
                    ${selectField('Laterality', 'laterality', detail.laterality, ['', 'R', 'L', 'b/l', 'B/L', 'L>R', 'R>L', 'none'])}
                    ${clinField('Duration (y)', 'disease_duration', detail.disease_duration, 'number')}
                    ${clinField('Sinemet', 'sinemet_schedule', detail.sinemet_schedule)}
                    ${clinField('Last Dose', 'last_dose', detail.last_dose)}
                    ${selectField('DBS', 'dbs', detail.dbs, ['', 'None', 'GPi (R)', 'GPi (L)', 'GPi (b/l)', 'STN (R)', 'STN (L)', 'STN (b/l)', 'VIM'])}
                    ${yesNoField('Fluctuations', 'fluctuations', detail.fluctuations)}
                    ${clinField('Tremor', 'tremor', detail.tremor)}
                    ${clinField('Dysmetria', 'dysmetria', detail.dysmetria)}
                    ${yesNoField('Myoclonus', 'myoclonus', detail.myoclonus)}
                    ${clinField('Other Meds', 'other_meds', detail.other_meds)}
                    ${clinField('Video Date', 'video_date', detail.video_date)}
                    <div style="display:flex;gap:8px;margin-top:4px;">
                        <label style="font-size:11px;display:flex;align-items:center;gap:4px;">
                            <input type="checkbox" ${detail.saa_flag ? 'checked' : ''} onchange="updateClinical(${detail.id},'saa_flag',this.checked?1:0)"> SAA
                        </label>
                        <label style="font-size:11px;display:flex;align-items:center;gap:4px;">
                            <input type="checkbox" ${detail.skin_biopsy_flag ? 'checked' : ''} onchange="updateClinical(${detail.id},'skin_biopsy_flag',this.checked?1:0)"> Skin Biopsy
                        </label>
                    </div>
                    <div style="margin-top:8px;">
                        <a class="btn btn-sm" href="/onboarding?subject=${detail.id}" style="text-decoration:none;width:100%;display:block;text-align:center;">Edit Subject</a>
                    </div>
                </div>

                <!-- Column 2: Trials checkmark chart -->
                <div class="detail-section">
                    <h3>Trials</h3>
                    ${trialChart || '<span style="font-size:12px;color:var(--text-muted);">No videos found</span>'}
                </div>

                <!-- Column 3: Controls -->
                <div class="detail-section">
                    <h3>Controls</h3>
                    <div class="info-row"><span class="label">Group</span>
                        <select id="diagnosisSelect" onchange="onDiagnosisSelectChange(${detail.id}, this)" style="${selStyle}width:auto;">
                            ${diagnosisGroups.map(dg => `<option value="${dg}"${dg === (detail.group_label || detail.diagnosis || 'Control') ? ' selected' : ''}>${dg}</option>`).join('')}
                            <option value="__new__">+ New…</option>
                        </select>
                    </div>
                    <div class="info-row"><span class="label">Video Type</span>
                        <select onchange="updateSubjectCameraMode(${detail.id}, this.value)" style="${selStyle}width:auto;">
                            <option value="single"${(detail.camera_mode || 'stereo') === 'single' ? ' selected' : ''}>Single</option>
                            <option value="stereo"${(detail.camera_mode || 'stereo') === 'stereo' ? ' selected' : ''}>Stereo</option>
                            <option value="multicam"${(detail.camera_mode || 'stereo') === 'multicam' ? ' selected' : ''}>Multicam</option>
                        </select>
                    </div>
                    <div class="info-row"><span class="label">Camera</span>
                        <select id="cameraSelect" onchange="updateSubjectCamera(${detail.id}, this.value)" style="${selStyle}width:auto;">
                            <option value="">None</option>
                            ${calibrationNames.map(n => `<option value="${n}"${n === detail.camera_name ? ' selected' : ''}>${n}</option>`).join('')}
                        </select>
                    </div>
                    <div style="margin-top:12px;">
                        <h3 style="margin-bottom:6px;">Pipeline</h3>
                        <div style="display:flex;gap:4px;flex-wrap:wrap;">
                            <button class="btn btn-sm" onclick="runStep(${detail.id}, 'mediapipe')">MediaPipe</button>
                            <button class="btn btn-sm" onclick="runStep(${detail.id}, 'pose')">Pose</button>
                            <button class="btn btn-sm" onclick="runStep(${detail.id}, 'deidentify')">Blur</button>
                            <button class="btn btn-sm btn-primary" onclick="runStep(${detail.id}, 'train')">Train</button>
                            <button class="btn btn-sm" onclick="runStep(${detail.id}, 'analyze_v1')">v1</button>
                            <button class="btn btn-sm" onclick="runStep(${detail.id}, 'analyze_v2')">v2</button>
                        </div>
                    </div>
                    <div style="margin-top:8px;">
                        <button class="btn btn-sm btn-danger" onclick="removeSubject(${detail.id}, '${detail.name}')" style="width:100%;">Remove Subject</button>
                    </div>
                </div>
            </div>

            <!-- Distance Trace (full width) -->
            <div class="detail-section" style="margin-top:12px;">
                <h3>Distance Trace</h3>
                <div id="detailDistancePlots" style="min-height:60px;">
                    <span style="font-size:12px;color:var(--text-muted);">Loading…</span>
                </div>
            </div>

            <!-- Recent Jobs (full width, under distance) -->
            ${detail.jobs.length > 0 ? `
            <div class="detail-section" style="margin-top:8px;">
                <h3>Recent Jobs</h3>
                ${detail.jobs.slice(0, 5).map(j => `
                    <div class="info-row">
                        <span><span class="job-indicator job-${j.status}"></span>${j.job_type}</span>
                        <span>${j.status} ${j.progress_pct ? `(${j.progress_pct.toFixed(0)}%)` : ''}</span>
                    </div>
                `).join('')}
            </div>` : ''}
        `;

        document.getElementById('detailOverlay').style.display = 'flex';

        // Load and render distance traces
        _loadDetailDistances(subjectId);
    } catch (e) {
        alert('Error loading subject: ' + e.message);
    }
}

async function updateClinical(subjectId, field, value) {
    try {
        const body = {};
        if (field === 'age' || field === 'disease_duration') {
            body[field] = value ? Number(value) : null;
        } else {
            body[field] = value || null;
        }
        await API.patch(`/api/subjects/${subjectId}`, body);
    } catch (e) {
        console.error('Failed to update clinical field:', e);
    }
}

function closeDetail() {
    document.getElementById('detailOverlay').style.display = 'none';
}

// ── Pipeline actions ─────────────────────────────────
async function runStep(subjectId, step) {
    try {
        const result = await API.post(`/api/subjects/${subjectId}/run-step`, { step });
        if (result.job_id) {
            // Start tracking job
            trackJob(result.job_id);
        }
        loadSubjects();
        showDetail(subjectId);
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

function trackJob(jobId) {
    const jobsCard = document.getElementById('jobsCard');
    jobsCard.style.display = 'block';

    API.streamJob(jobId,
        (data) => {
            updateJobDisplay(jobId, data);
        },
        (data) => {
            loadSubjects();
        }
    );
}

function updateJobDisplay(jobId, data) {
    const list = document.getElementById('jobsList');
    let el = document.getElementById(`job-${jobId}`);
    if (!el) {
        el = document.createElement('div');
        el.id = `job-${jobId}`;
        el.style.marginBottom = '8px';
        list.appendChild(el);
    }

    const remoteBadge = data.remote_host
        ? '<span class="badge" style="background:rgba(33,150,243,0.15);color:#2196f3;font-size:11px;">Remote</span>'
        : '';
    const errorMsg = (data.status === 'failed' && data.error_msg)
        ? `<div style="color:var(--red);font-size:12px;margin-top:4px;">${data.error_msg}</div>`
        : '';
    const logBtn = (data.status === 'running')
        ? `<button class="btn btn-sm" onclick="viewJobLogLive(${jobId})">View Log</button>`
        : (data.status === 'failed')
            ? `<button class="btn btn-sm" onclick="viewJobLog(${jobId})">View Log</button>`
            : '';

    el.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="job-indicator job-${data.status}"></span>
            <span style="font-weight:600;">${data.subject_name || '?'}</span>
            <span style="color:var(--text-muted);">${(data.job_type || '').replace(/_/g, ' ')}</span>
            ${remoteBadge}
            <div class="progress-bar" style="flex:1;max-width:200px;">
                <div class="fill" style="width:${data.progress_pct || 0}%"></div>
            </div>
            <span>${(data.progress_pct || 0).toFixed(0)}%</span>
            <span class="badge badge-${data.status}">${data.status}</span>
            ${data.status === 'running' ? `<button class="btn btn-sm btn-danger" onclick="cancelJob(${jobId})">Cancel</button>` : ''}
            ${logBtn}
        </div>
        ${errorMsg}
    `;
}

async function cancelJob(jobId) {
    try {
        await API.post(`/api/jobs/${jobId}/cancel`);
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

let _logStream = null;  // Active log SSE connection

async function viewJobLog(jobId) {
    const modal = document.getElementById('logModal');
    const pre = document.getElementById('logContent');
    pre.textContent = 'Loading...';
    modal.classList.add('active');
    try {
        const job = await API.get(`/api/jobs/${jobId}`);
        pre.textContent = (job.log_tail || []).join('') || 'No log output.';
        pre.scrollTop = pre.scrollHeight;
    } catch (e) {
        pre.textContent = 'Error fetching log: ' + e.message;
    }
}

function viewJobLogLive(jobId) {
    const modal = document.getElementById('logModal');
    const pre = document.getElementById('logContent');
    pre.textContent = '';
    modal.classList.add('active');

    _logStream = API.streamJobLog(jobId,
        (text) => {
            pre.textContent += text;
            pre.scrollTop = pre.scrollHeight;
        },
        (data) => {
            _logStream = null;
            if (data.status === 'failed') {
                pre.textContent += '\n\n--- Job failed ---\n';
            } else if (data.status === 'completed') {
                pre.textContent += '\n\n--- Job completed ---\n';
            }
            pre.scrollTop = pre.scrollHeight;
        }
    );
}

function closeLogModal() {
    if (_logStream) {
        _logStream.close();
        _logStream = null;
    }
    document.getElementById('logModal').classList.remove('active');
}

// ── Jobs polling ─────────────────────────────────────
async function loadJobs() {
    try {
        const jobs = await API.get('/api/jobs?status=running');
        if (jobs.length > 0) {
            document.getElementById('jobsCard').style.display = 'block';
            jobs.forEach(j => trackJob(j.id));
        }
    } catch (e) { /* ignore */ }
}

async function syncSubjects() {
    try {
        const result = await API.post('/api/subjects/sync');
        const parts = [`${result.created} new`, `${result.updated} updated`];
        if (result.removed) parts.push(`${result.removed} removed`);
        parts.push(`${result.total} total`);
        alert(`Sync complete: ${parts.join(', ')}`);
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function autoAssignDiagnosis() {
    if (!confirm('Auto-assign diagnosis groups to all subjects based on their names?\n\nThis will use name patterns (e.g., "Con" → Control, "MSA" → MSA, etc.) to assign subjects to groups.')) {
        return;
    }
    try {
        const result = await API.post('/api/subjects/auto-assign-diagnosis');
        alert(`Auto-assignment complete!\n\nUpdated: ${result.updated} subjects\nTotal subjects: ${result.total_subjects}`);
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function removeSubject(subjectId, subjectName) {
    if (!confirm(`Remove subject "${subjectName}"?\n\nThis will permanently delete:\n• All trial and deidentified videos\n• DLC directory (models, labels, config)\n• All database records\n\nThis cannot be undone.`)) {
        return;
    }
    try {
        const result = await API.del(`/api/subjects/${subjectId}`);
        alert(result.message);
        closeDetail();
        loadSubjects();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Refine navigation ────────────────────────────────
function openRefine(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    window.location.href = `/labeling-select?subject=${subjectId}&mode=refine`;
}

// ── Final (read-only) navigation ─────────────────────
function openFinal(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    window.location.href = `/labeling-select?subject=${subjectId}&mode=final`;
}

// ── Corrections navigation ───────────────────────────
function openCorrections(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    window.location.href = `/labeling-select?subject=${subjectId}&mode=corrections`;
}

// ── Labeling navigation ──────────────────────────────
function openLabeling(subjectId) {
    sessionStorage.setItem('dlc_lastSubjectId', String(subjectId));
    // Default to events mode; server _smart_mode handles subjects that aren't ready
    window.location.href = `/labeling-select?subject=${subjectId}`;
}

// ── Batch preprocessing ─────────────────────────────
function showBatchPreprocess() {
    document.getElementById('batchModal').classList.add('active');
}

function closeBatchModal() {
    document.getElementById('batchModal').classList.remove('active');
}

async function startBatchPreprocess() {
    const steps = [];
    if (document.getElementById('batchStepMp').checked) steps.push('mediapipe');
    if (document.getElementById('batchStepBlur').checked) steps.push('blur');

    if (steps.length === 0) {
        alert('Select at least one step');
        return;
    }

    closeBatchModal();

    try {
        const result = await API.post('/api/batch/preprocess', { steps, subjects: [] });
        if (result.job_id) {
            trackJob(result.job_id);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Event listeners ──────────────────────────────────
document.getElementById('filterInput').addEventListener('input', renderDiagnosisGroups);
document.getElementById('stageFilter').addEventListener('change', renderDiagnosisGroups);

// ── Videos panel ─────────────────────────────────────
async function loadVideosPanel() {
    const panel = document.getElementById('videosPanel');
    try {
        const data = await API.get('/api/video-tools/sources');
        const videos = data.videos || [];

        let html = '';

        if (videos.length === 0) {
            html += '<span style="color:var(--text-muted);font-size:13px;">No videos yet</span>';
        } else {
            videos.forEach(v => {
                const icon = v.source === 'sample' ? '📦' : '🎬';
                const label = v.source === 'sample' ? `${v.name} (sample)` : v.name;
                html += `<a href="/videos" class="btn btn-sm" onclick="sessionStorage.setItem('browse_video_path','${v.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}')" style="text-decoration:none;display:inline-flex;align-items:center;gap:4px;">${icon} ${label}</a>`;
            });
        }

        // Browse button always at the end
        html += `<a href="/videos" class="btn btn-sm btn-primary" style="text-decoration:none;">Browse…</a>`;

        panel.innerHTML = html;
    } catch (e) {
        panel.innerHTML = '<a href="/videos" class="btn btn-sm btn-primary" style="text-decoration:none;">Browse Videos</a>';
    }
}

// ── Detail panel distance traces ─────────────────────
async function _loadDetailDistances(subjectId) {
    const container = document.getElementById('detailDistancePlots');
    if (!container) return;

    try {
        const data = await API.get(`/api/results/${subjectId}/traces`);
        if (!data.trials || data.trials.length === 0) {
            container.innerHTML = '<span style="font-size:12px;color:var(--text-muted);">No distance data</span>';
            return;
        }

        container.innerHTML = '';

        data.trials.forEach((trial, idx) => {
            const div = document.createElement('div');
            div.id = `detailDist_${idx}`;
            div.style.height = '140px';
            container.appendChild(div);

            const fps = trial.fps || 60;
            const distances = trial.distances || [];
            const times = distances.map((_, i) => i / fps);

            const trace = {
                x: times,
                y: distances,
                type: 'scattergl',
                mode: 'lines',
                line: { color: '#4a9eff', width: 1 },
                name: trial.name,
            };

            const layout = {
                title: { text: trial.name, font: { size: 12 } },
                xaxis: { title: { text: 'Time (s)', font: { size: 10 } }, tickfont: { size: 9 } },
                yaxis: { title: { text: 'Distance (mm)', font: { size: 10 } }, tickfont: { size: 9 } },
                margin: { l: 45, r: 10, t: 25, b: 30 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: 'var(--text)' },
            };

            Plotly.newPlot(div, [trace], layout, {
                responsive: true,
                displayModeBar: false,
            });
        });
    } catch (e) {
        container.innerHTML = '<span style="font-size:12px;color:var(--text-muted);">Could not load distance data</span>';
    }
}

// ── Init ─────────────────────────────────────────────
checkStatus().then(async () => {
    await loadCalibrationNames();
    loadSubjects();
    loadVideosPanel();
});
