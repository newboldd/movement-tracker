"""Camera setups API: CRUD for camera calibration configurations + calibration execution."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config import get_settings, CALIBRATION_DIR
from ..db import get_db_ctx
from ..services.jobs import registry

router = APIRouter(prefix="/api/camera-setups", tags=["camera-setups"])
logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────

class CameraSetupCreate(BaseModel):
    name: str
    mode: str = "stereo"  # "stereo" or "multicam"
    camera_count: int = 2
    camera_names: List[str] = []


class CameraSetupResponse(BaseModel):
    id: int
    name: str
    mode: str
    camera_count: int
    camera_names: List[str]
    calibration_path: Optional[str]
    checkerboard_rows: Optional[int]
    checkerboard_cols: Optional[int]
    has_calibration: bool
    created_at: Optional[str]


class CalibrateFromVideoRequest(BaseModel):
    setup_id: int
    video_path: str
    start_time: float = 0.0
    end_time: float = 0.0  # 0 = full video
    checkerboard_rows: int = 9
    checkerboard_cols: int = 6
    square_size_mm: float = 25.0


class CalibrateFromImagesRequest(BaseModel):
    setup_id: int
    image_dir: str
    checkerboard_rows: int = 9
    checkerboard_cols: int = 6
    square_size_mm: float = 25.0


# ── CRUD ──────────────────────────────────────────────────────────────────

def _row_to_response(row: dict) -> dict:
    cam_names = json.loads(row.get("camera_names", "[]")) if isinstance(row.get("camera_names"), str) else row.get("camera_names", [])
    calib_path = row.get("calibration_path")
    return {
        **row,
        "camera_names": cam_names,
        "has_calibration": bool(calib_path and os.path.exists(calib_path)),
    }


@router.get("")
def list_camera_setups() -> List[dict]:
    """List all camera setups."""
    with get_db_ctx() as db:
        rows = db.execute("SELECT * FROM camera_setups ORDER BY name").fetchall()
    return [_row_to_response(r) for r in rows]


@router.post("", status_code=201)
def create_camera_setup(req: CameraSetupCreate) -> dict:
    """Create a new camera setup."""
    if req.mode not in ("stereo", "multicam"):
        raise HTTPException(400, "Mode must be 'stereo' or 'multicam'")
    if not req.name or not req.name.strip():
        raise HTTPException(400, "Name is required")

    cam_names = req.camera_names if req.camera_names else ["cam1", "cam2"]
    with get_db_ctx() as db:
        existing = db.execute(
            "SELECT id FROM camera_setups WHERE name = ?", (req.name,)
        ).fetchone()
        if existing:
            raise HTTPException(400, f"Camera setup '{req.name}' already exists")

        db.execute(
            """INSERT INTO camera_setups (name, mode, camera_count, camera_names)
               VALUES (?, ?, ?, ?)""",
            (req.name, req.mode, req.camera_count, json.dumps(cam_names)),
        )
        row = db.execute(
            "SELECT * FROM camera_setups WHERE name = ?", (req.name,)
        ).fetchone()
    return _row_to_response(row)


@router.get("/{setup_id}")
def get_camera_setup(setup_id: int) -> dict:
    """Get a single camera setup."""
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT * FROM camera_setups WHERE id = ?", (setup_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "Camera setup not found")
    return _row_to_response(row)


@router.delete("/{setup_id}")
def delete_camera_setup(setup_id: int) -> dict:
    """Delete a camera setup."""
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT * FROM camera_setups WHERE id = ?", (setup_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Camera setup not found")

        # Check if any subjects reference this setup
        subjects = db.execute(
            "SELECT id FROM subjects WHERE camera_name = ?", (row["name"],)
        ).fetchall()
        if subjects:
            raise HTTPException(
                400,
                f"Cannot delete: {len(subjects)} subject(s) use this camera setup"
            )

        db.execute("DELETE FROM camera_setups WHERE id = ?", (setup_id,))
    return {"deleted": True, "name": row["name"]}


# ── Calibration endpoints ─────────────────────────────────────────────────

@router.post("/calibrate-from-video")
def calibrate_from_video(req: CalibrateFromVideoRequest) -> dict:
    """Run stereo calibration from a stereo calibration video.

    Extracts checkerboard frames, detects corners in left/right halves,
    then runs OpenCV stereo calibration. Saves result as YAML.
    """
    with get_db_ctx() as db:
        setup = db.execute(
            "SELECT * FROM camera_setups WHERE id = ?", (req.setup_id,)
        ).fetchone()
    if not setup:
        raise HTTPException(404, "Camera setup not found")

    if not Path(req.video_path).exists():
        raise HTTPException(404, f"Video not found: {req.video_path}")

    # Create calibration output directory
    calib_dir = CALIBRATION_DIR
    calib_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(calib_dir / f"{setup['name']}_calibration.yaml")

    # Create a job for tracking
    settings = get_settings()
    log_dir = settings.dlc_path / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"calibration_{setup['name']}.log")

    with get_db_ctx() as db:
        # Use subject_id=0 as a sentinel for system jobs
        # First ensure we have at least one subject or use a dummy
        db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, log_path)
               VALUES (0, 'calibration', 'pending', ?)""",
            (log_path,),
        )
        job = db.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    _run_video_calibration(
        job_id=job["id"],
        setup_id=req.setup_id,
        setup_name=setup["name"],
        video_path=req.video_path,
        start_time=req.start_time,
        end_time=req.end_time,
        rows=req.checkerboard_rows,
        cols=req.checkerboard_cols,
        square_size=req.square_size_mm,
        output_path=output_path,
        log_path=log_path,
    )

    return {"job_id": job["id"], "status": "running", "output_path": output_path}


@router.post("/calibrate-from-images")
def calibrate_from_images(req: CalibrateFromImagesRequest) -> dict:
    """Run stereo calibration from paired calibration images.

    Expects a directory containing paired images:
      left_01.png + right_01.png  (or cam1_01.png + cam2_01.png)
    """
    with get_db_ctx() as db:
        setup = db.execute(
            "SELECT * FROM camera_setups WHERE id = ?", (req.setup_id,)
        ).fetchone()
    if not setup:
        raise HTTPException(404, "Camera setup not found")

    if not Path(req.image_dir).exists():
        raise HTTPException(404, f"Image directory not found: {req.image_dir}")

    calib_dir = CALIBRATION_DIR
    calib_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(calib_dir / f"{setup['name']}_calibration.yaml")

    settings = get_settings()
    log_dir = settings.dlc_path / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"calibration_{setup['name']}.log")

    with get_db_ctx() as db:
        db.execute(
            """INSERT INTO jobs (subject_id, job_type, status, log_path)
               VALUES (0, 'calibration', 'pending', ?)""",
            (log_path,),
        )
        job = db.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    cam_names = json.loads(setup["camera_names"]) if isinstance(setup["camera_names"], str) else setup["camera_names"]

    _run_image_calibration(
        job_id=job["id"],
        setup_id=req.setup_id,
        setup_name=setup["name"],
        image_dir=req.image_dir,
        cam_names=cam_names,
        rows=req.checkerboard_rows,
        cols=req.checkerboard_cols,
        square_size=req.square_size_mm,
        output_path=output_path,
        log_path=log_path,
    )

    return {"job_id": job["id"], "status": "running", "output_path": output_path}


# ── Background calibration workers ────────────────────────────────────────

def _update_job(job_id: int, **kwargs):
    with get_db_ctx() as db:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*kwargs.values(), job_id))


def _run_video_calibration(job_id, setup_id, setup_name, video_path,
                           start_time, end_time, rows, cols, square_size,
                           output_path, log_path):
    """Background: extract checkerboard frames from stereo video and calibrate."""
    import cv2
    import numpy as np

    cancel_event = registry.register_cancel_event(job_id)

    def _run():
        try:
            _update_job(job_id, status='running', progress_pct=1)

            with open(log_path, "w") as log:
                log.write(f"Stereo calibration from video: {video_path}\n")
                log.write(f"Checkerboard: {rows}x{cols}, square={square_size}mm\n")
                log.flush()

                # Open video
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                start_frame = int(start_time * fps) if start_time > 0 else 0
                end_frame = int(end_time * fps) if end_time > 0 else total_frames
                midline = width // 2

                # Sample frames (every ~0.5 seconds, max 200 frames)
                frame_range = end_frame - start_frame
                step = max(1, frame_range // 200)

                pattern_size = (cols, rows)  # OpenCV uses (cols, rows)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                obj_points = []  # 3D points in real world
                img_points_L = []  # 2D points in left image
                img_points_R = []  # 2D points in right image

                # Prepare object points
                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                objp *= square_size

                img_size = None
                found_count = 0
                checked_count = 0

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                for frame_idx in range(start_frame, end_frame, step):
                    if cancel_event.is_set():
                        raise InterruptedError("Cancelled")

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    checked_count += 1
                    left = frame[:, :midline]
                    right = frame[:, midline:]

                    gray_L = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                    gray_R = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

                    if img_size is None:
                        img_size = gray_L.shape[::-1]

                    ret_L, corners_L = cv2.findChessboardCorners(gray_L, pattern_size, None)
                    ret_R, corners_R = cv2.findChessboardCorners(gray_R, pattern_size, None)

                    if ret_L and ret_R:
                        corners_L = cv2.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
                        corners_R = cv2.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)

                        obj_points.append(objp)
                        img_points_L.append(corners_L)
                        img_points_R.append(corners_R)
                        found_count += 1

                    pct = min(80, (checked_count / max(1, frame_range // step)) * 80)
                    _update_job(job_id, progress_pct=round(pct, 1))

                    if found_count >= 80:  # enough frames
                        break

                cap.release()

                log.write(f"Found checkerboard in {found_count}/{checked_count} frames\n")
                log.flush()

                if found_count < 10:
                    raise ValueError(
                        f"Only found checkerboard in {found_count} frames "
                        f"(need at least 10). Try different checkerboard size or video."
                    )

                # Run stereo calibration
                log.write("Running stereo calibration...\n")
                log.flush()
                _update_job(job_id, progress_pct=85)

                flags = cv2.CALIB_FIX_INTRINSIC
                # First calibrate each camera individually
                ret_L, K1, dist1, _, _ = cv2.calibrateCamera(
                    obj_points, img_points_L, img_size, None, None
                )
                ret_R, K2, dist2, _, _ = cv2.calibrateCamera(
                    obj_points, img_points_R, img_size, None, None
                )

                log.write(f"Left camera RMS: {ret_L:.4f}\n")
                log.write(f"Right camera RMS: {ret_R:.4f}\n")
                log.flush()

                # Stereo calibration
                ret_stereo, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
                    obj_points, img_points_L, img_points_R,
                    K1, dist1, K2, dist2, img_size,
                    criteria=criteria,
                    flags=0,
                )

                log.write(f"Stereo RMS: {ret_stereo:.4f}\n")
                log.flush()
                _update_job(job_id, progress_pct=95)

                # Save calibration YAML
                _save_calibration_yaml(output_path, K1, dist1, K2, dist2, R, T)

                log.write(f"Saved calibration to {output_path}\n")
                log.write(f"Baseline: {np.linalg.norm(T):.2f}mm\n")
                log.flush()

            # Update camera setup with calibration path
            with get_db_ctx() as db:
                db.execute(
                    """UPDATE camera_setups SET calibration_path = ?,
                       checkerboard_rows = ?, checkerboard_cols = ?
                       WHERE id = ?""",
                    (output_path, rows, cols, setup_id),
                )

            # Register in settings.calibrations
            settings = get_settings()
            settings.calibrations[setup_name] = output_path
            settings.save()

            _update_job(job_id, status='completed', progress_pct=100)

        except InterruptedError:
            _update_job(job_id, status='cancelled')
        except Exception as e:
            logger.exception(f"Calibration job {job_id} failed")
            _update_job(job_id, status='failed', error_msg=str(e))
        finally:
            registry.unregister_cancel_event(job_id)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _run_image_calibration(job_id, setup_id, setup_name, image_dir,
                           cam_names, rows, cols, square_size,
                           output_path, log_path):
    """Background: calibrate from paired image directory."""
    import cv2
    import numpy as np

    cancel_event = registry.register_cancel_event(job_id)

    def _run():
        try:
            _update_job(job_id, status='running', progress_pct=1)

            with open(log_path, "w") as log:
                log.write(f"Stereo calibration from images: {image_dir}\n")
                log.write(f"Checkerboard: {rows}x{cols}, square={square_size}mm\n")
                log.flush()

                img_dir = Path(image_dir)

                # Find paired images (left_*.png + right_*.png or cam1_*.png + cam2_*.png)
                left_name = cam_names[0].lower() if cam_names else "left"
                right_name = cam_names[1].lower() if len(cam_names) > 1 else "right"

                pairs = _find_image_pairs(img_dir, left_name, right_name)

                if not pairs:
                    raise ValueError(
                        f"No paired images found in {image_dir}. "
                        f"Expected files matching {left_name}_*.png + {right_name}_*.png "
                        f"(or .jpg)"
                    )

                log.write(f"Found {len(pairs)} image pairs\n")
                log.flush()

                pattern_size = (cols, rows)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                objp = np.zeros((rows * cols, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                objp *= square_size

                obj_points = []
                img_points_L = []
                img_points_R = []
                img_size = None
                found_count = 0

                for i, (left_path, right_path) in enumerate(pairs):
                    if cancel_event.is_set():
                        raise InterruptedError("Cancelled")

                    img_L = cv2.imread(str(left_path))
                    img_R = cv2.imread(str(right_path))
                    if img_L is None or img_R is None:
                        continue

                    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
                    gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

                    if img_size is None:
                        img_size = gray_L.shape[::-1]

                    ret_L, corners_L = cv2.findChessboardCorners(gray_L, pattern_size, None)
                    ret_R, corners_R = cv2.findChessboardCorners(gray_R, pattern_size, None)

                    if ret_L and ret_R:
                        corners_L = cv2.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
                        corners_R = cv2.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)

                        obj_points.append(objp)
                        img_points_L.append(corners_L)
                        img_points_R.append(corners_R)
                        found_count += 1

                    pct = min(80, ((i + 1) / len(pairs)) * 80)
                    _update_job(job_id, progress_pct=round(pct, 1))

                log.write(f"Checkerboard detected in {found_count}/{len(pairs)} pairs\n")
                log.flush()

                if found_count < 5:
                    raise ValueError(
                        f"Only found checkerboard in {found_count} pairs "
                        f"(need at least 5). Check checkerboard dimensions."
                    )

                log.write("Running stereo calibration...\n")
                log.flush()
                _update_job(job_id, progress_pct=85)

                ret_L, K1, dist1, _, _ = cv2.calibrateCamera(
                    obj_points, img_points_L, img_size, None, None
                )
                ret_R, K2, dist2, _, _ = cv2.calibrateCamera(
                    obj_points, img_points_R, img_size, None, None
                )

                ret_stereo, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
                    obj_points, img_points_L, img_points_R,
                    K1, dist1, K2, dist2, img_size,
                    criteria=criteria,
                    flags=0,
                )

                log.write(f"Stereo RMS: {ret_stereo:.4f}\n")
                log.flush()
                _update_job(job_id, progress_pct=95)

                _save_calibration_yaml(output_path, K1, dist1, K2, dist2, R, T)
                log.write(f"Saved calibration to {output_path}\n")

            with get_db_ctx() as db:
                db.execute(
                    """UPDATE camera_setups SET calibration_path = ?,
                       checkerboard_rows = ?, checkerboard_cols = ?
                       WHERE id = ?""",
                    (output_path, rows, cols, setup_id),
                )

            settings = get_settings()
            settings.calibrations[setup_name] = output_path
            settings.save()

            _update_job(job_id, status='completed', progress_pct=100)

        except InterruptedError:
            _update_job(job_id, status='cancelled')
        except Exception as e:
            logger.exception(f"Calibration job {job_id} failed")
            _update_job(job_id, status='failed', error_msg=str(e))
        finally:
            registry.unregister_cancel_event(job_id)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _find_image_pairs(img_dir: Path, left_name: str, right_name: str) -> list[tuple]:
    """Find matching left/right image pairs in a directory."""
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    left_files = {}
    right_files = {}

    for f in sorted(img_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in extensions:
            continue
        name_lower = f.stem.lower()

        # Try prefix matching: left_01, right_01 or cam1_01, cam2_01
        for prefix in (left_name, "left", "l"):
            if name_lower.startswith(prefix):
                key = name_lower[len(prefix):].lstrip("_- ")
                left_files[key] = f
                break
        for prefix in (right_name, "right", "r"):
            if name_lower.startswith(prefix):
                key = name_lower[len(prefix):].lstrip("_- ")
                right_files[key] = f
                break

    # Match pairs by key
    pairs = []
    for key in sorted(left_files.keys()):
        if key in right_files:
            pairs.append((left_files[key], right_files[key]))

    return pairs


def _save_calibration_yaml(path: str, K1, dist1, K2, dist2, R, T):
    """Save stereo calibration matrices as OpenCV YAML."""
    import cv2
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("K1", K1)
    fs.write("dist1", dist1)
    fs.write("K2", K2)
    fs.write("dist2", dist2)
    fs.write("R", R)
    fs.write("T", T)
    fs.release()
