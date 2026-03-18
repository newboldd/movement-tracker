"""Video export: accept JPEG frames from client, encode to MP4 with ffmpeg."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export-video", tags=["export"])

# ── In-memory registry of active exports ──────────────────────────────────

_active_exports: dict[str, dict[str, Any]] = {}
_STALE_SECONDS = 3600  # 1 hour


class ExportStartRequest(BaseModel):
    fps: float
    width: int
    height: int
    total_frames: int


def _cleanup_stale():
    """Remove exports older than _STALE_SECONDS."""
    now = time.time()
    stale = [eid for eid, meta in _active_exports.items()
             if now - meta["created"] > _STALE_SECONDS]
    for eid in stale:
        meta = _active_exports.pop(eid, None)
        if meta and os.path.isdir(meta["tmp_dir"]):
            shutil.rmtree(meta["tmp_dir"], ignore_errors=True)
            logger.info(f"Cleaned stale export {eid}")


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/start")
def start_export(req: ExportStartRequest) -> dict:
    """Create a new export session.  Returns {export_id}."""
    _cleanup_stale()

    export_id = uuid.uuid4().hex[:12]
    tmp_dir = tempfile.mkdtemp(prefix=f"dlc_export_{export_id}_")

    _active_exports[export_id] = {
        "tmp_dir": tmp_dir,
        "fps": req.fps,
        "width": req.width,
        "height": req.height,
        "total_frames": req.total_frames,
        "frames_received": 0,
        "created": time.time(),
    }

    logger.info(f"Export {export_id}: started ({req.total_frames} frames, "
                f"{req.width}x{req.height} @ {req.fps}fps)")
    return {"export_id": export_id}


@router.post("/{export_id}/frames")
async def upload_frames(export_id: str, request: Request) -> dict:
    """Upload a batch of JPEG frames via multipart form data.

    Expected fields:
      start_index: int  — global frame index of the first frame in this batch
      frame_0, frame_1, ...: JPEG blobs
    """
    meta = _active_exports.get(export_id)
    if not meta:
        raise HTTPException(404, "Export session not found")

    form = await request.form()
    start_index = int(form.get("start_index", 0))

    count = 0
    for key, value in form.items():
        if key == "start_index":
            continue
        if not hasattr(value, "read"):
            continue
        # key is "frame_N" where N is local batch index
        try:
            local_idx = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            local_idx = count
        global_idx = start_index + local_idx
        filename = f"frame_{global_idx:06d}.jpg"
        filepath = os.path.join(meta["tmp_dir"], filename)
        data = await value.read()
        with open(filepath, "wb") as f:
            f.write(data)
        count += 1

    meta["frames_received"] += count
    return {"received": count, "total_received": meta["frames_received"]}


@router.post("/{export_id}/encode")
def encode_export(export_id: str, background_tasks: BackgroundTasks):
    """Encode uploaded frames to MP4 and return the file for download."""
    meta = _active_exports.get(export_id)
    if not meta:
        raise HTTPException(404, "Export session not found")

    tmp_dir = meta["tmp_dir"]
    fps = meta["fps"]

    # Verify frames exist
    frame_files = sorted(Path(tmp_dir).glob("frame_*.jpg"))
    if not frame_files:
        raise HTTPException(400, "No frames uploaded")

    logger.info(f"Export {export_id}: encoding {len(frame_files)} frames. "
                f"First: {frame_files[0].name}, Last: {frame_files[-1].name}")

    output_path = os.path.join(tmp_dir, "export.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%06d.jpg"),
        # Pad to even dimensions (required by libx264 + yuv420p)
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg not found. Install FFmpeg and add to PATH.")

    if result.returncode != 0:
        # ffmpeg stderr starts with version/config info; the actual error is at the end
        stderr_tail = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
        logger.error(f"ffmpeg encode failed:\n{stderr_tail}")
        raise HTTPException(500, f"ffmpeg encode failed: {stderr_tail[:500]}")

    if not os.path.exists(output_path):
        raise HTTPException(500, "Encoding produced no output file")

    logger.info(f"Export {export_id}: encoded {len(frame_files)} frames → "
                f"{os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Schedule cleanup after the response is sent
    def cleanup():
        _active_exports.pop(export_id, None)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    background_tasks.add_task(cleanup)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="export.mp4",
    )


@router.delete("/{export_id}")
def cancel_export(export_id: str) -> dict:
    """Cancel and clean up an export session."""
    meta = _active_exports.pop(export_id, None)
    if meta and os.path.isdir(meta["tmp_dir"]):
        shutil.rmtree(meta["tmp_dir"], ignore_errors=True)
    return {"status": "cancelled"}
