"""Video tools API: probe, stream, trim, and subject onboarding pipeline."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel

from ..config import get_settings
from ..db import get_db_ctx
from ..services.jobs import registry

router = APIRouter(prefix="/api/video-tools", tags=["video-tools"])
logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────

class TrimRequest(BaseModel):
    source_path: str
    start_time: float
    end_time: float
    output_name: str


class SegmentDef(BaseModel):
    source_path: str
    start_time: float
    end_time: float
    trial_label: str  # e.g. "L1", "R2"
    camera_name: str | None = None  # e.g. "cam0" — set in multicam mode


class ProcessSubjectRequest(BaseModel):
    subject_name: str
    segments: list[SegmentDef]
    blur_faces: bool = True
    camera_mode: str = "stereo"
    camera_name: str | None = None  # camera setup name
    no_face_trials: list[str] = []  # trial labels that have no faces
    diagnosis: str | None = None    # subject group (e.g. Control, MSA, PD)


# ── Probe ─────────────────────────────────────────────────────────────────

@router.get("/probe")
def probe_video(path: str = Query(..., description="Path to video file")) -> dict:
    """Get video metadata: duration, fps, resolution, stereo detection.

    Uses OpenCV instead of ffprobe so no system-level FFmpeg install is needed
    just for probing videos.
    """
    import cv2

    if not Path(path).exists():
        raise HTTPException(404, f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, f"Cannot open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Try to get codec FourCC
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip() or "unknown"
    finally:
        cap.release()

    if width == 0 or height == 0:
        raise HTTPException(400, "No video stream found")

    # Stereo detection: aspect ratio ~2:1 suggests stereo
    aspect = width / height if height > 0 else 1
    is_stereo = aspect > 1.7  # stereo videos are ~2:1 or wider

    return {
        "path": path,
        "width": width,
        "height": height,
        "duration": round(duration, 3),
        "fps": round(fps, 2),
        "is_stereo": is_stereo,
        "codec": codec,
        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
    }


# ── Source videos (for dashboard panel) ───────────────────────────────────

@router.get("/sources")
def list_source_videos() -> dict:
    """List available source videos: sample data + previously used sources.

    Returns { videos: [{name, path, source}], browse_url: str }
    """
    import glob as _glob
    from ..config import DATA_DIR

    settings = get_settings()
    seen_paths: set[str] = set()
    videos: list[dict] = []

    # 1. Sample data
    sample_dir = DATA_DIR / "sample_data"
    if sample_dir.exists():
        for f in sorted(sample_dir.glob("*.mp4")):
            p = str(f)
            if p not in seen_paths:
                seen_paths.add(p)
                videos.append({"name": f.stem, "path": p, "source": "sample"})

    # 2. Previously used source videos (from segments table)
    with get_db_ctx() as db:
        rows = db.execute(
            "SELECT DISTINCT source_path FROM segments ORDER BY source_path"
        ).fetchall()
    for row in rows:
        p = row["source_path"]
        if p and p not in seen_paths and Path(p).exists():
            seen_paths.add(p)
            videos.append({
                "name": Path(p).stem,
                "path": p,
                "source": "recent",
            })

    return {"videos": videos}


# ── Stream ────────────────────────────────────────────────────────────────

@router.get("/stream")
def stream_video(path: str = Query(..., description="Path to video file")) -> FileResponse:
    """Stream a video file for HTML5 playback (supports range requests)."""
    if not Path(path).exists():
        raise HTTPException(404, f"Video not found: {path}")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=Path(path).name,
    )


# ── Trim ──────────────────────────────────────────────────────────────────

def _is_full_video(source_path: str, start_time: float, end_time: float,
                   tolerance: float = 0.5) -> bool:
    """Check if trim covers the entire video (within tolerance seconds)."""
    import cv2
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0
    return start_time <= tolerance and end_time >= (duration - tolerance)


def _copy_video(source_path: str, output_path: str, progress_callback=None):
    """Copy a video file without re-encoding."""
    import shutil
    if progress_callback:
        progress_callback(0)
    shutil.copy2(source_path, output_path)
    if progress_callback:
        progress_callback(100)
    return output_path


def _ffmpeg_trim(source_path: str, start_time: float, end_time: float,
                 output_path: str, progress_callback=None) -> str:
    """Trim a video segment using ffmpeg with frame-accurate re-encoding.

    Uses -ss before -i for fast keyframe seek, then re-encodes to get an exact
    cut.  Stream copy (-c copy) is avoided because it must start on a keyframe,
    which leaves black/blank frames at the beginning when the requested start
    doesn't coincide with one.

    Args:
        progress_callback: optional callable(pct: float) called with 0-100 during encode.
    """
    import re as _re

    duration = end_time - start_time
    duration_us = duration * 1_000_000

    from ..services.ffmpeg import get_ffmpeg_path

    try:
        ffmpeg = get_ffmpeg_path()
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_time:.3f}",
        "-i", source_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",  # no audio
        "-progress", "pipe:1",
        output_path,
    ]

    if not progress_callback:
        # Simple blocking path
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg trim failed: {result.stderr[:500]}")
        return output_path

    # Stream stdout for progress parsing
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    for line in proc.stdout:
        m = _re.match(r"out_time_us=(\d+)", line.strip())
        if m and duration_us > 0:
            pct = min(100.0, float(m.group(1)) / duration_us * 100)
            progress_callback(pct)

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"ffmpeg trim failed: {stderr[:500]}")

    return output_path


@router.post("/trim")
def trim_video(req: TrimRequest) -> dict:
    """Trim a video segment and save to output_name."""
    if not Path(req.source_path).exists():
        raise HTTPException(404, f"Source video not found: {req.source_path}")

    settings = get_settings()
    output_dir = settings.video_path
    output_path = str(output_dir / req.output_name)

    try:
        _ffmpeg_trim(req.source_path, req.start_time, req.end_time, output_path)
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return {
        "output_path": output_path,
        "duration": round(req.end_time - req.start_time, 3),
    }


# ── Process Subject Pipeline ─────────────────────────────────────────────

@router.post("/process-subject")
def process_subject(req: ProcessSubjectRequest) -> dict:
    """Full onboarding pipeline for a new subject.

    For each segment:
      1. Trim source video to segment
      2. Run face blur on trimmed segment
      3. Save to videos/{Subject}_{Trial}.mp4
      4. Clean up temp trimmed file

    Then create/update subject DB entry.
    Runs as a background job.
    """
    if not req.subject_name or not req.segments:
        raise HTTPException(400, "Subject name and at least one segment required")

    settings = get_settings()

    # Create job
    with get_db_ctx() as db:
        # Get or create subject
        subj = db.execute(
            "SELECT * FROM subjects WHERE name = ?", (req.subject_name,)
        ).fetchone()

        if not subj:
            dlc_dir = req.subject_name
            db.execute(
                """INSERT INTO subjects (name, stage, dlc_dir, camera_mode, camera_name, diagnosis)
                   VALUES (?, 'created', ?, ?, ?, ?)""",
                (req.subject_name, dlc_dir, req.camera_mode, req.camera_name,
                 req.diagnosis or 'Control'),
            )
            subj = db.execute(
                "SELECT * FROM subjects WHERE name = ?", (req.subject_name,)
            ).fetchone()
        else:
            # Update camera mode/name on existing subject
            updates = ["camera_mode = ?", "camera_name = ?", "updated_at = CURRENT_TIMESTAMP"]
            params = [req.camera_mode, req.camera_name]
            if req.diagnosis is not None:
                updates.append("diagnosis = ?")
                params.append(req.diagnosis)
            params.append(subj["id"])
            db.execute(
                f"UPDATE subjects SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )

        log_dir = settings.dlc_path / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"job_onboard_{subj['id']}.log")

        db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, log_path)
               VALUES (?, 'onboard', 'pending', ?)""",
            (subj["id"], log_path),
        )
        job = db.execute(
            "SELECT * FROM jobs WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subj["id"],),
        ).fetchone()

    # Save no_face_videos on subject
    if req.no_face_trials:
        import json as _json
        with get_db_ctx() as db:
            db.execute(
                "UPDATE subjects SET no_face_videos = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (_json.dumps(req.no_face_trials), subj["id"]),
            )

    # Run in background thread
    _do_process_subject(
        job_id=job["id"],
        subject_id=subj["id"],
        subject_name=req.subject_name,
        segments=[s.model_dump() for s in req.segments],
        log_path=log_path,
        blur_faces=req.blur_faces,
    )

    return {"job_id": job["id"], "status": "running", "subject_id": subj["id"]}


def _update_job_progress(job_id: int, pct: float, logfile=None, msg: str = ""):
    """Helper: update job progress in DB (and optionally log)."""
    with get_db_ctx() as db:
        db.execute("UPDATE jobs SET progress_pct = ? WHERE id = ?", (round(pct, 1), job_id))
    if logfile and msg:
        logfile.write(f"  {msg} ({pct:.0f}%)\n")
        logfile.flush()


def _do_process_subject(job_id: int, subject_id: int, subject_name: str,
                         segments: list[dict], log_path: str,
                         blur_faces: bool = True):
    """Background thread: trim each segment, optionally blur to deidentified/ dir."""

    cancel_event = registry.register_cancel_event(job_id)

    def _run():
        temp_trimmed = None
        try:
            with get_db_ctx() as db:
                db.execute(
                    "UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP, progress_pct = 1 WHERE id = ?",
                    (job_id,),
                )

            settings = get_settings()
            output_dir = settings.video_path
            os.makedirs(str(output_dir), exist_ok=True)

            total_segments = len(segments)
            seg_span = 100.0 / total_segments

            with open(log_path, "w") as logfile:
                for i, seg in enumerate(segments):
                    if cancel_event.is_set():
                        raise InterruptedError("Job cancelled")

                    trial = seg["trial_label"]
                    cam = seg.get("camera_name")
                    if cam:
                        output_name = f"{subject_name}_{trial}_{cam}.mp4"
                    else:
                        output_name = f"{subject_name}_{trial}.mp4"
                    output_path = str(output_dir / output_name)

                    seg_base = i * seg_span

                    # Progress allocation within each segment:
                    #   blur enabled:  trim 20%, blur 80%
                    #   blur disabled: trim 100%
                    if blur_faces:
                        trim_span = seg_span * 0.20
                        blur_start_pct = seg_base + trim_span
                        blur_span = seg_span * 0.80
                    else:
                        trim_span = seg_span

                    logfile.write(f"=== Segment {i+1}/{total_segments}: {trial} ===\n")
                    logfile.flush()
                    _update_job_progress(job_id, seg_base + 1, logfile, f"Starting {trial}")

                    def _trim_progress(pct, _base=seg_base, _span=trim_span):
                        if cancel_event.is_set():
                            raise InterruptedError("Job cancelled")
                        overall = _base + (pct / 100.0) * _span
                        _update_job_progress(job_id, overall)

                    if blur_faces:
                        # Trim to temp, then blur to deidentified/ dir
                        temp_trimmed = str(output_dir / f"_temp_trim_{subject_name}_{trial}.mp4")
                        full_video = _is_full_video(seg["source_path"], seg["start_time"], seg["end_time"])
                        if full_video:
                            logfile.write(f"  Copying full video {seg['source_path']} (no trim needed)\n")
                        else:
                            logfile.write(f"  Trimming {seg['source_path']} [{seg['start_time']:.1f}-{seg['end_time']:.1f}s]\n")
                        logfile.flush()

                        try:
                            if full_video:
                                _copy_video(seg["source_path"], temp_trimmed, _trim_progress)
                            else:
                                _ffmpeg_trim(seg["source_path"], seg["start_time"],
                                            seg["end_time"], temp_trimmed,
                                            progress_callback=_trim_progress)
                        except Exception as e:
                            logfile.write(f"  Trim failed: {e}\n")
                            temp_trimmed = None
                            continue

                        _update_job_progress(job_id, seg_base + trim_span, logfile, "Trim done")

                        logfile.write(f"  Running face blur...\n")
                        logfile.flush()

                        def _blur_progress(blur_pct, _blur_start=blur_start_pct, _blur_span=blur_span):
                            if cancel_event.is_set():
                                raise InterruptedError("Job cancelled")
                            overall = _blur_start + (blur_pct / 100.0) * _blur_span
                            _update_job_progress(job_id, overall)

                        # Also save the original (unblurred) trimmed segment
                        import shutil
                        shutil.move(temp_trimmed, output_path)
                        temp_trimmed = None

                        # Blur to deidentified/ subdir
                        deident_dir = Path(output_dir) / "deidentified"
                        deident_dir.mkdir(parents=True, exist_ok=True)
                        deident_path = str(deident_dir / output_name)
                        temp_blur = deident_path + ".tmp.mp4"

                        try:
                            from ..services.deidentify import deidentify_video
                            deidentify_video(
                                output_path, temp_blur,
                                progress_callback=_blur_progress,
                            )
                            os.replace(temp_blur, deident_path)
                            logfile.write(f"  Saved {deident_path}\n")
                        except ImportError:
                            logfile.write(f"  Deidentify not available, skipping blur\n")
                            if os.path.exists(temp_blur):
                                os.remove(temp_blur)
                        except InterruptedError:
                            if os.path.exists(temp_blur):
                                os.remove(temp_blur)
                            raise
                        except Exception as e:
                            logfile.write(f"  Blur failed: {e}\n")
                            if os.path.exists(temp_blur):
                                os.remove(temp_blur)
                    else:
                        # No blur — trim directly to output (or copy if full video)
                        full_video = _is_full_video(seg["source_path"], seg["start_time"], seg["end_time"])
                        if full_video:
                            logfile.write(f"  Copying full video {seg['source_path']} (no trim needed)\n")
                        else:
                            logfile.write(f"  Trimming {seg['source_path']} [{seg['start_time']:.1f}-{seg['end_time']:.1f}s] (no blur)\n")
                        logfile.flush()

                        try:
                            if full_video:
                                _copy_video(seg["source_path"], output_path, _trim_progress)
                            else:
                                _ffmpeg_trim(seg["source_path"], seg["start_time"],
                                            seg["end_time"], output_path,
                                            progress_callback=_trim_progress)
                        except Exception as e:
                            logfile.write(f"  Trim failed: {e}\n")
                            continue

                    pct = seg_base + seg_span
                    _update_job_progress(job_id, pct, logfile, f"Done {trial}")

            # Write .deidentified marker if blur was done
            if blur_faces:
                dlc_path = settings.dlc_path / subject_name
                dlc_path.mkdir(parents=True, exist_ok=True)
                (dlc_path / ".deidentified").write_text("")

            # Persist segment metadata for future editing
            with get_db_ctx() as db:
                for seg in segments:
                    trial = seg["trial_label"]
                    cam = seg.get("camera_name")
                    db.execute(
                        """INSERT OR REPLACE INTO segments
                           (subject_id, trial_label, source_path, start_time, end_time, camera_name)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (subject_id, trial, seg["source_path"],
                         seg["start_time"], seg["end_time"], cam),
                    )

            # Mark complete
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'completed', progress_pct = 100,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (job_id,),
                )
                db.execute(
                    "UPDATE subjects SET stage = 'created', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (subject_id,),
                )

        except InterruptedError:
            if temp_trimmed and os.path.exists(temp_trimmed):
                os.remove(temp_trimmed)
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'cancelled',
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (job_id,),
                )
        except Exception as e:
            if temp_trimmed and os.path.exists(temp_trimmed):
                os.remove(temp_trimmed)
            logger.exception(f"Job {job_id} onboarding failed")
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE jobs SET status = 'failed', error_msg = ?,
                       finished_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (str(e), job_id),
                )
        finally:
            registry.unregister_cancel_event(job_id)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
