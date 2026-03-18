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

    // ── Modal DOM ────────────────────────────────────────────

    function ensureModal() {
        if (modal) return;

        modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal" style="min-width:360px;max-width:440px;">
                <h3>Export Video</h3>

                <div class="form-group">
                    <label>Start Frame</label>
                    <div style="display:flex;gap:6px;">
                        <input type="number" id="vex-start" min="0" style="flex:1;">
                        <button class="btn btn-sm" id="vex-start-cur" title="Use current frame">Current</button>
                    </div>
                </div>

                <div class="form-group">
                    <label>End Frame</label>
                    <div style="display:flex;gap:6px;">
                        <input type="number" id="vex-end" min="0" style="flex:1;">
                        <button class="btn btn-sm" id="vex-end-cur" title="Use current frame">Current</button>
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

                <div class="actions">
                    <button class="btn btn-sm" id="vex-cancel">Cancel</button>
                    <button class="btn btn-sm btn-primary" id="vex-export">Export</button>
                </div>
            </div>`;

        document.body.appendChild(modal);

        els = {
            start:       modal.querySelector('#vex-start'),
            end:         modal.querySelector('#vex-end'),
            startCur:    modal.querySelector('#vex-start-cur'),
            endCur:      modal.querySelector('#vex-end-cur'),
            trace:       modal.querySelector('#vex-trace'),
            progressWrap:modal.querySelector('#vex-progress-wrap'),
            status:      modal.querySelector('#vex-status'),
            bar:         modal.querySelector('#vex-bar'),
            downloadWrap:modal.querySelector('#vex-download-wrap'),
            download:    modal.querySelector('#vex-download'),
            cancel:      modal.querySelector('#vex-cancel'),
            exportBtn:   modal.querySelector('#vex-export'),
        };

        els.startCur.addEventListener('click', () => {
            if (ctx) els.start.value = ctx.currentFrame;
        });
        els.endCur.addEventListener('click', () => {
            if (ctx) els.end.value = ctx.currentFrame;
        });
        els.cancel.addEventListener('click', close);
        els.exportBtn.addEventListener('click', startExport);
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

            // 4. Encode
            els.status.textContent = 'Encoding video...';
            els.bar.style.width = '100%';
            const encResp = await fetch(`/api/export-video/${exportId}/encode`, {
                method: 'POST',
            });
            if (!encResp.ok) throw new Error('Encoding failed');

            // 5. Download
            const mp4Blob = await encResp.blob();
            const url = URL.createObjectURL(mp4Blob);
            els.download.href = url;
            els.downloadWrap.style.display = 'block';
            els.status.textContent = 'Done!';
            exportId = null; // server already cleaned up

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

    return { open, close };
})();
