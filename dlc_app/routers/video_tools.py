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


class ProcessSubjectRequest(BaseModel):
    subject_name: str
    segments: list[SegmentDef]
    blur_faces: bool = True


# ── Probe ─────────────────────────────────────────────────────────────────

def _ffprobe(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return json.loads(result.stdout)
    except FileNotFoundError:
        raise HTTPException(500, "ffprobe not found. Install FFmpeg and add to PATH.")


@router.get("/probe")
def probe_video(path: str = Query(..., description="Path to video file")) -> dict:
    """Get video metadata: duration, fps, resolution, stereo detection."""
    if not Path(path).exists():
        raise HTTPException(404, f"Video not found: {path}")

    info = _ffprobe(path)

    # Find video stream
    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise HTTPException(400, "No video stream found")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(info.get("format", {}).get("duration", 0))

    # Parse FPS from r_frame_rate (e.g. "30/1")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 30.0

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
        "codec": video_stream.get("codec_name", "unknown"),
        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
    }


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

def _ffmpeg_trim(source_path: str, start_time: float, end_time: float,
                 output_path: str) -> str:
    """Trim a video segment using ffmpeg with frame-accurate re-encoding.

    Uses -ss before -i for fast keyframe seek, then re-encodes to get an exact
    cut.  Stream copy (-c copy) is avoided because it must start on a keyframe,
    which leaves black/blank frames at the beginning when the requested start
    doesn't coincide with one.
    """
    duration = end_time - start_time

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", source_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",  # no audio
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg not found. Install FFmpeg and add to PATH.")

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr[:500]}")

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
                """INSERT INTO subjects (name, stage, dlc_dir)
                   VALUES (?, 'created', ?)""",
                (req.subject_name, dlc_dir),
            )
            subj = db.execute(
                "SELECT * FROM subjects WHERE name = ?", (req.subject_name,)
            ).fetchone()

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
                    output_name = f"{subject_name}_{trial}.mp4"
                    output_path = str(output_dir / output_name)

                    seg_base = i * seg_span
                    trim_pct = seg_base + seg_span * 0.05
                    blur_start = trim_pct
                    blur_span = seg_span * 0.95

                    logfile.write(f"=== Segment {i+1}/{total_segments}: {trial} ===\n")
                    logfile.flush()
                    _update_job_progress(job_id, seg_base + 1, logfile, f"Starting {trial}")

                    if blur_faces:
                        # Trim to temp, then blur to deidentified/ dir
                        temp_trimmed = str(output_dir / f"_temp_trim_{subject_name}_{trial}.mp4")
                        logfile.write(f"  Trimming {seg['source_path']} [{seg['start_time']:.1f}-{seg['end_time']:.1f}s]\n")
                        logfile.flush()

                        try:
                            _ffmpeg_trim(seg["source_path"], seg["start_time"],
                                        seg["end_time"], temp_trimmed)
                        except Exception as e:
                            logfile.write(f"  Trim failed: {e}\n")
                            temp_trimmed = None
                            continue

                        _update_job_progress(job_id, trim_pct, logfile, "Trim done")

                        logfile.write(f"  Running face blur...\n")
                        logfile.flush()

                        def _blur_progress(blur_pct, _blur_start=blur_start, _blur_span=blur_span):
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
                        # No blur — trim directly to output
                        logfile.write(f"  Trimming {seg['source_path']} [{seg['start_time']:.1f}-{seg['end_time']:.1f}s] (no blur)\n")
                        logfile.flush()

                        try:
                            _ffmpeg_trim(seg["source_path"], seg["start_time"],
                                        seg["end_time"], output_path)
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
