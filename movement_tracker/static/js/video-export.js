/**
 * Video Export Module
 *
 * Captures frames from a viewer's canvases and uploads to the server for
 * ffmpeg encoding into an MP4 video.  Works with both the MANO 3D viewer
 * and the label viewer via a common exportContext interface.
 *
 * Usage:
 *   VideoExport.open(viewer.getExportContext());
 */
window.VideoExport = (() => {
    const BATCH_SIZE = 100;
    const JPEG_QUALITY = 0.92;

    let modal = null;
    let els = {};       // modal sub-elements
    let ctx = null;     // current exportContext
    let abortCtrl = null;
    let exportId = null;
    let savedFrame = 0;

    // Save-to-location state
    let saveBrowserPath = '';
    let saveBrowserDirs = [];
    let saveBreadcrumbs = [];

    // ── Modal DOM ────────────────────────────────────────────

    function ensureModal() {
        if (modal) return;

        modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal" style="min-width:420px;max-width:520px;">
                <h3>Export Video</h3>

                <div class="form-group">
                    <label>Start Frame</label>
                    <div style="display:flex;gap:6px;">
                        <input type="number" id="vex-start" min="0" style="flex:1;">
                        <button class="btn btn-sm" id="vex-start-beg" title="First frame">&#x23EE;</button>
                        <button class="btn btn-sm" id="vex-start-cur" title="Use current frame">Current</button>
                        <button class="btn btn-sm" id="vex-start-end" title="Last frame">&#x23ED;</button>
                    </div>
                </div>

                <div class="form-group">
                    <label>End Frame</label>
                    <div style="display:flex;gap:6px;">
                        <input type="number" id="vex-end" min="0" style="flex:1;">
                        <button class="btn btn-sm" id="vex-end-beg" title="First frame">&#x23EE;</button>
                        <button class="btn btn-sm" id="vex-end-cur" title="Use current frame">Current</button>
                        <button class="btn btn-sm" id="vex-end-end" title="Last frame">&#x23ED;</button>
                    </div>
                </div>

                <div class="form-group">
                    <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;">
                        <input type="checkbox" id="vex-trace" checked>
                        Include distance trace
                    </label>
                </div>

                <div class="form-group" id="vex-progress-wrap" style="display:none;">
                    <label id="vex-status">Preparing...</label>
                    <div class="progress-bar" style="height:8px;margin-top:4px;">
                        <div class="progress-fill" id="vex-bar" style="width:0%;"></div>
                    </div>
                </div>

                <div id="vex-download-wrap" style="display:none;margin-top:12px;text-align:center;">
                    <a id="vex-download" class="btn btn-sm" style="display:inline-block;" download="export.mp4">
                        Download MP4
                    </a>
                </div>

                <div id="vex-save-wrap" style="display:none;margin-top:12px;">
                    <hr style="border:none;border-top:1px solid var(--border);margin:8px 0;">
                    <label style="font-weight:600;font-size:.9em;">Save to server location</label>
                    <div style="display:flex;gap:6px;margin-top:6px;">
                        <input type="text" id="vex-save-filename" placeholder="export.mp4" style="flex:1;">
                        <button class="btn btn-sm btn-primary" id="vex-save-btn">Save</button>
                    </div>
                    <div id="vex-save-browser" style="margin-top:8px;border:1px solid var(--border);border-radius:6px;max-height:200px;overflow-y:auto;display:none;">
                        <div id="vex-save-breadcrumbs" style="padding:4px 8px;font-size:.8em;border-bottom:1px solid var(--border);display:flex;flex-wrap:wrap;gap:2px;"></div>
                        <div id="vex-save-dirs" style="padding:4px 0;"></div>
                    </div>
                    <div style="display:flex;gap:6px;margin-top:6px;">
                        <button class="btn btn-sm" id="vex-save-browse" title="Choose directory">Browse...</button>
                        <button class="btn btn-sm" id="vex-save-mkdir" title="Create new folder" style="display:none;">+ New Folder</button>
                    </div>
                    <div id="vex-save-mkdir-row" style="display:none;margin-top:6px;display:none;">
                        <div style="display:flex;gap:6px;">
                            <input type="text" id="vex-save-mkdir-name" placeholder="New folder name" style="flex:1;">
                            <button class="btn btn-sm btn-primary" id="vex-save-mkdir-ok">Create</button>
                            <button class="btn btn-sm" id="vex-save-mkdir-cancel">Cancel</button>
                        </div>
                    </div>
                    <div id="vex-save-path-display" style="font-size:.8em;color:var(--text-muted);margin-top:4px;word-break:break-all;"></div>
                    <div id="vex-save-status" style="font-size:.85em;color:var(--green);margin-top:4px;display:none;"></div>
                </div>

                <div class="actions">
                    <button class="btn btn-sm" id="vex-cancel">Cancel</button>
                    <button class="btn btn-sm btn-primary" id="vex-export">Export</button>
                </div>
            </div>`;

        document.body.appendChild(modal);

        els = {
            start:       modal.querySelector('#vex-start'),
            end:         modal.querySelector('#vex-end'),
            startBeg:    modal.querySelector('#vex-start-beg'),
            startCur:    modal.querySelector('#vex-start-cur'),
            startEnd:    modal.querySelector('#vex-start-end'),
            endBeg:      modal.querySelector('#vex-end-beg'),
            endCur:      modal.querySelector('#vex-end-cur'),
            endEnd:      modal.querySelector('#vex-end-end'),
            trace:       modal.querySelector('#vex-trace'),
            progressWrap:modal.querySelector('#vex-progress-wrap'),
            status:      modal.querySelector('#vex-status'),
            bar:         modal.querySelector('#vex-bar'),
            downloadWrap:modal.querySelector('#vex-download-wrap'),
            download:    modal.querySelector('#vex-download'),
            saveWrap:    modal.querySelector('#vex-save-wrap'),
            saveFilename:modal.querySelector('#vex-save-filename'),
            saveBtn:     modal.querySelector('#vex-save-btn'),
            saveBrowser: modal.querySelector('#vex-save-browser'),
            saveBreadcrumbs: modal.querySelector('#vex-save-breadcrumbs'),
            saveDirs:    modal.querySelector('#vex-save-dirs'),
            saveBrowse:  modal.querySelector('#vex-save-browse'),
            saveMkdir:   modal.querySelector('#vex-save-mkdir'),
            saveMkdirRow:modal.querySelector('#vex-save-mkdir-row'),
            saveMkdirName:modal.querySelector('#vex-save-mkdir-name'),
            saveMkdirOk: modal.querySelector('#vex-save-mkdir-ok'),
            saveMkdirCancel:modal.querySelector('#vex-save-mkdir-cancel'),
            savePathDisplay: modal.querySelector('#vex-save-path-display'),
            saveStatus:  modal.querySelector('#vex-save-status'),
            cancel:      modal.querySelector('#vex-cancel'),
            exportBtn:   modal.querySelector('#vex-export'),
        };

        // Frame selection buttons
        els.startBeg.addEventListener('click', () => { if (ctx) els.start.value = 0; });
        els.startCur.addEventListener('click', () => { if (ctx) els.start.value = ctx.currentFrame; });
        els.startEnd.addEventListener('click', () => { if (ctx) els.start.value = Math.max(0, ctx.nFrames - 1); });
        els.endBeg.addEventListener('click', () => { if (ctx) els.end.value = 0; });
        els.endCur.addEventListener('click', () => { if (ctx) els.end.value = ctx.currentFrame; });
        els.endEnd.addEventListener('click', () => { if (ctx) els.end.value = Math.max(0, ctx.nFrames - 1); });

        // Save-to-location
        els.saveBrowse.addEventListener('click', () => {
            if (els.saveBrowser.style.display === 'none') {
                loadSaveDirs(saveBrowserPath || '');
                els.saveBrowser.style.display = '';
                els.saveMkdir.style.display = '';
            } else {
                els.saveBrowser.style.display = 'none';
                els.saveMkdir.style.display = 'none';
            }
        });
        els.saveMkdir.addEventListener('click', () => {
            els.saveMkdirRow.style.display = '';
            els.saveMkdirName.value = '';
            els.saveMkdirName.focus();
        });
        els.saveMkdirCancel.addEventListener('click', () => {
            els.saveMkdirRow.style.display = 'none';
        });
        els.saveMkdirOk.addEventListener('click', doMkdir);
        els.saveBtn.addEventListener('click', doSaveToLocation);

        els.cancel.addEventListener('click', close);
        els.exportBtn.addEventListener('click', startExport);
    }

    // ── Save-to-location directory browser ────────────────────

    async function loadSaveDirs(path) {
        try {
            const url = '/api/export-video/browse-dirs' + (path ? '?path=' + encodeURIComponent(path) : '');
            const resp = await fetch(url);
            if (!resp.ok) throw new Error('Failed to browse');
            const data = await resp.json();
            saveBrowserPath = data.path;
            saveBrowserDirs = data.dirs || [];
            saveBreadcrumbs = data.breadcrumbs || [];
            renderSaveBrowser();
            updateSavePathDisplay();
        } catch (err) {
            console.error('Browse dirs error:', err);
        }
    }

    function renderSaveBrowser() {
        // Breadcrumbs
        els.saveBreadcrumbs.innerHTML = '';
        saveBreadcrumbs.forEach((bc, i) => {
            if (i > 0) {
                const sep = document.createElement('span');
                sep.textContent = ' / ';
                sep.style.opacity = '0.5';
                els.saveBreadcrumbs.appendChild(sep);
            }
            const a = document.createElement('a');
            a.href = '#';
            a.textContent = bc.name;
            a.style.textDecoration = 'none';
            a.addEventListener('click', e => { e.preventDefault(); loadSaveDirs(bc.path); });
            els.saveBreadcrumbs.appendChild(a);
        });

        // Directory list
        els.saveDirs.innerHTML = '';
        if (saveBrowserDirs.length === 0) {
            const empty = document.createElement('div');
            empty.textContent = 'No subdirectories';
            empty.style.cssText = 'padding:8px 12px;font-size:.85em;color:var(--text-muted);';
            els.saveDirs.appendChild(empty);
        }
        saveBrowserDirs.forEach(d => {
            const row = document.createElement('div');
            row.style.cssText = 'padding:4px 12px;cursor:pointer;font-size:.85em;display:flex;align-items:center;gap:6px;';
            row.innerHTML = `<span style="opacity:0.5;">&#128193;</span> ${escHtml(d.name)}`;
            row.addEventListener('click', () => loadSaveDirs(d.path));
            row.addEventListener('mouseenter', () => { row.style.background = 'var(--bg-hover, #f0f0f0)'; });
            row.addEventListener('mouseleave', () => { row.style.background = ''; });
            els.saveDirs.appendChild(row);
        });
    }

    function updateSavePathDisplay() {
        if (saveBrowserPath) {
            els.savePathDisplay.textContent = 'Save to: ' + saveBrowserPath + '/';
        } else {
            els.savePathDisplay.textContent = '';
        }
    }

    async function doMkdir() {
        const name = els.saveMkdirName.value.trim();
        if (!name) return;
        if (!saveBrowserPath) { alert('Select a directory first'); return; }
        const newPath = saveBrowserPath + '/' + name;
        try {
            const resp = await fetch('/api/export-video/mkdir', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: newPath }),
            });
            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                alert(data.detail || 'Failed to create directory');
                return;
            }
            els.saveMkdirRow.style.display = 'none';
            // Refresh and navigate into new dir
            await loadSaveDirs(newPath);
        } catch (err) {
            alert('Error: ' + err.message);
        }
    }

    async function doSaveToLocation() {
        if (!exportId && !els.download.href) {
            alert('Export the video first, then save.');
            return;
        }
        if (!saveBrowserPath) {
            alert('Choose a directory using Browse first.');
            return;
        }
        const filename = els.saveFilename.value.trim() || 'export.mp4';
        const fullPath = saveBrowserPath + '/' + filename;

        // We need a valid exportId — if the export already finished and was downloaded,
        // exportId is null.  In that case, we need to re-upload. But we stored the
        // export_id before cleanup.  Let's use a different approach: save is only available
        // while exportId is still valid (before download cleanup).
        // Actually the encode endpoint does cleanup via background_tasks AFTER response,
        // and we set exportId = null after download blob. Let's keep exportId alive until
        // the modal closes or save is done.

        if (!exportId) {
            alert('Export session expired. Please export again.');
            return;
        }

        els.saveStatus.style.display = '';
        els.saveStatus.textContent = 'Saving...';
        els.saveStatus.style.color = 'var(--text-muted)';

        try {
            const resp = await fetch(`/api/export-video/${exportId}/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: fullPath }),
            });
            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                throw new Error(data.detail || 'Save failed');
            }
            els.saveStatus.textContent = 'Saved to ' + fullPath;
            els.saveStatus.style.color = 'var(--green)';
            exportId = null; // cleaned up by server
        } catch (err) {
            els.saveStatus.textContent = 'Error: ' + err.message;
            els.saveStatus.style.color = 'var(--red, #e53e3e)';
        }
    }

    function escHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    // ── Open / Close ─────────────────────────────────────────

    function open(exportContext) {
        if (!exportContext) { alert('Viewer not ready'); return; }
        ctx = exportContext;
        ensureModal();

        // Reset UI
        els.start.value = 0;
        els.start.max = Math.max(0, ctx.nFrames - 1);
        els.end.value = Math.max(0, ctx.nFrames - 1);
        els.end.max = Math.max(0, ctx.nFrames - 1);
        els.trace.checked = true;
        els.trace.parentElement.style.display = ctx.distanceCanvas ? '' : 'none';
        els.progressWrap.style.display = 'none';
        els.downloadWrap.style.display = 'none';
        els.saveWrap.style.display = 'none';
        els.saveBrowser.style.display = 'none';
        els.saveMkdir.style.display = 'none';
        els.saveMkdirRow.style.display = 'none';
        els.saveStatus.style.display = 'none';
        els.saveFilename.value = 'export.mp4';
        els.savePathDisplay.textContent = '';
        els.exportBtn.disabled = false;
        els.exportBtn.textContent = 'Export';
        els.bar.style.width = '0%';

        modal.classList.add('active');
    }

    function close() {
        if (abortCtrl) {
            abortCtrl.abort();
            abortCtrl = null;
        }
        if (exportId) {
            fetch(`/api/export-video/${exportId}`, { method: 'DELETE' }).catch(() => {});
            exportId = null;
        }
        if (modal) modal.classList.remove('active');
        // Restore frame position
        if (ctx && ctx.seekAndRender && savedFrame != null) {
            ctx.seekAndRender(savedFrame).catch(() => {});
        }
        ctx = null;
    }

    // ── Export pipeline ──────────────────────────────────────

    async function startExport() {
        if (!ctx) return;

        const startFrame = parseInt(els.start.value) || 0;
        const endFrame = parseInt(els.end.value) || 0;
        const includeTrace = els.trace.checked && !!ctx.distanceCanvas;

        if (endFrame < startFrame) {
            alert('End frame must be >= start frame');
            return;
        }

        const totalFrames = endFrame - startFrame + 1;
        savedFrame = ctx.currentFrame;

        // UI state: exporting
        els.exportBtn.disabled = true;
        els.exportBtn.textContent = 'Exporting...';
        els.progressWrap.style.display = 'block';
        els.downloadWrap.style.display = 'none';
        els.saveWrap.style.display = 'none';

        abortCtrl = new AbortController();

        try {
            // 1. Get composite dimensions
            const { width, height } = ctx.getCompositeSize();
            const traceH = includeTrace ? ctx.distanceCanvas.height : 0;
            const outW = width;
            const outH = height + traceH;

            // 2. Start server session
            const startResp = await fetch('/api/export-video/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fps: ctx.fps * (ctx.playbackRate || 1),
                    width: outW,
                    height: outH,
                    total_frames: totalFrames,
                }),
            });
            if (!startResp.ok) throw new Error('Failed to start export session');
            const { export_id } = await startResp.json();
            exportId = export_id;

            // 3. Capture & upload in batches
            const offscreen = document.createElement('canvas');
            offscreen.width = outW;
            offscreen.height = outH;
            const offCtx = offscreen.getContext('2d');

            for (let batchStart = startFrame; batchStart <= endFrame; batchStart += BATCH_SIZE) {
                if (abortCtrl.signal.aborted) throw new DOMException('Aborted', 'AbortError');

                const batchEnd = Math.min(batchStart + BATCH_SIZE - 1, endFrame);
                const formData = new FormData();
                formData.append('start_index', batchStart - startFrame);

                for (let f = batchStart; f <= batchEnd; f++) {
                    if (abortCtrl.signal.aborted) throw new DOMException('Aborted', 'AbortError');

                    // Seek + render all layers
                    await ctx.seekAndRender(f);

                    // Composite onto offscreen canvas
                    offCtx.fillStyle = '#000';
                    offCtx.fillRect(0, 0, outW, outH);

                    // Draw main canvas layers (video + 2D overlays)
                    for (const layer of ctx.canvasLayers) {
                        offCtx.drawImage(layer, 0, 0, width, height);
                    }

                    // Draw WebGL (Three.js) if present — must be same sync block
                    const webglCanvas = ctx.renderThreeJS();
                    if (webglCanvas) {
                        offCtx.drawImage(webglCanvas, 0, 0, width, height);
                    }

                    // Draw distance trace below main area
                    if (includeTrace) {
                        offCtx.drawImage(
                            ctx.distanceCanvas,
                            0, height,
                            width, traceH,
                        );
                    }

                    // Convert to JPEG blob
                    const blob = await new Promise(resolve =>
                        offscreen.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
                    const globalIdx = f - startFrame;
                    formData.append(
                        `frame_${f - batchStart}`,
                        blob,
                        `frame_${String(globalIdx).padStart(6, '0')}.jpg`,
                    );

                    // Progress
                    const pct = ((f - startFrame + 1) / totalFrames * 100).toFixed(0);
                    els.status.textContent = `Capturing frame ${f - startFrame + 1} / ${totalFrames}`;
                    els.bar.style.width = pct + '%';
                }

                // Upload batch
                els.status.textContent = `Uploading batch...`;
                const upResp = await fetch(`/api/export-video/${exportId}/frames`, {
                    method: 'POST',
                    body: formData,
                });
                if (!upResp.ok) throw new Error('Frame upload failed');
            }

            // 4. Encode — but DON'T download blob yet, keep exportId alive for save
            els.status.textContent = 'Encoding video...';
            els.bar.style.width = '100%';
            const encResp = await fetch(`/api/export-video/${exportId}/encode`, {
                method: 'POST',
            });
            if (!encResp.ok) throw new Error('Encoding failed');

            // 5. Show download + save options
            const mp4Blob = await encResp.blob();
            const url = URL.createObjectURL(mp4Blob);
            els.download.href = url;
            els.downloadWrap.style.display = 'block';
            els.saveWrap.style.display = '';
            els.saveStatus.style.display = 'none';
            els.status.textContent = 'Done!';
            // Note: exportId kept alive so "Save to location" can use it
            // Server cleanup happens via background_tasks after encode response,
            // but the MP4 file may be gone. We need to re-upload or change approach.
            // Actually the save endpoint copies from tmp_dir which is cleaned up.
            // Let's save the blob locally and upload it for save-to-location instead.
            _lastBlob = mp4Blob;
            // exportId is now invalid (server cleaned up after encode response)
            exportId = null;

        } catch (err) {
            if (err.name === 'AbortError') {
                els.status.textContent = 'Cancelled';
            } else {
                console.error('Export error:', err);
                els.status.textContent = 'Error: ' + err.message;
            }
        } finally {
            els.exportBtn.disabled = false;
            els.exportBtn.textContent = 'Export';
            abortCtrl = null;
        }
    }

    // We keep the blob so save-to-location can upload it
    let _lastBlob = null;

    // Override doSaveToLocation to upload from blob
    async function doSaveToLocation() {
        if (!_lastBlob) {
            alert('Export the video first, then save.');
            return;
        }
        if (!saveBrowserPath) {
            alert('Choose a directory using Browse first.');
            return;
        }
        const filename = els.saveFilename.value.trim() || 'export.mp4';
        const fullPath = saveBrowserPath + '/' + filename;

        els.saveStatus.style.display = '';
        els.saveStatus.textContent = 'Saving...';
        els.saveStatus.style.color = 'var(--text-muted)';

        try {
            // Upload the blob to a new export session, then save
            const formData = new FormData();
            formData.append('file', _lastBlob, filename);
            formData.append('path', fullPath);

            const resp = await fetch('/api/export-video/save-file', {
                method: 'POST',
                body: formData,
            });
            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                throw new Error(data.detail || 'Save failed');
            }
            els.saveStatus.textContent = 'Saved to ' + fullPath;
            els.saveStatus.style.color = 'var(--green)';
        } catch (err) {
            els.saveStatus.textContent = 'Error: ' + err.message;
            els.saveStatus.style.color = 'var(--red, #e53e3e)';
        }
    }

    return { open, close };
})();
