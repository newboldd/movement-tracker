"""Video frame extraction and stereo cropping."""
from __future__ import annotations

import json
import logging
import cv2
import numpy as np
from functools import lru_cache
from pathlib import Path

from ..config import get_settings, JPEG_QUALITY, FRAME_CACHE_SIZE
from ..db import get_db_ctx

logger = logging.getLogger(__name__)


_CV2_VERSION = cv2.__version__


def _compute_frame_offset(video_path: str, actual_count: int) -> int:
    """Compute offset between OpenCV frame indexing and browser video playback.

    Some videos contain disposable pre-roll packets with negative PTS that
    OpenCV decodes as regular frames but browsers skip.  This means OpenCV's
    frame N corresponds to the browser's frame N-offset.  The frontend needs
    this to seek to the correct video time for a given label frame.

    Uses ffprobe to count non-negative PTS video packets (fast — reads the
    container index, no decoding).  Returns 0 if ffprobe is unavailable.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries', 'packet=pts_time', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return 0
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        non_neg = sum(1 for l in lines if float(l) >= 0)
        offset = actual_count - non_neg
        if offset > 0:
            logger.info(
                f"{Path(video_path).name}: detected frame_offset={offset} "
                f"(opencv={actual_count}, browser={non_neg})"
            )
        return max(0, offset)
    except Exception as e:
        logger.debug(f"Could not compute frame offset for {video_path}: {e}")
        return 0


def _get_actual_frame_count(video_path: str) -> int:
    """Count actual decodable frames in a video, with persistent JSON cache.

    cv2.CAP_PROP_FRAME_COUNT often returns inflated counts (4–43 extra trailing
    frames that can't be decoded). This function reads until cap.read() fails
    and caches the result in <video_dir>/.frame_counts.json, keyed by filename
    and validated by file size and OpenCV version (different versions can decode
    different numbers of frames from the same file).

    Also computes and caches ``frame_offset`` — see :func:`_compute_frame_offset`.
    """
    p = Path(video_path)
    cache_path = p.parent / ".frame_counts.json"

    # Try cache
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    file_size = p.stat().st_size
    entry = cache.get(p.name)

    # Fast path: count + offset both cached and valid
    if (entry and entry.get("size") == file_size
            and entry.get("cv2_version") == _CV2_VERSION
            and "frame_offset" in entry):
        return entry["count"]

    # Semi-fast path: count is valid but frame_offset missing — only run ffprobe
    if (entry and entry.get("size") == file_size
            and entry.get("cv2_version") == _CV2_VERSION
            and "frame_offset" not in entry):
        actual = entry["count"]
        frame_offset = _compute_frame_offset(video_path, actual)
        entry["frame_offset"] = frame_offset
        try:
            cache_path.write_text(json.dumps(cache, indent=2))
        except OSError as e:
            logger.warning(f"Could not write frame count cache: {e}")
        return actual

    # Full cache miss — count actual decoded frames
    cap = cv2.VideoCapture(video_path)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual = 0
    for _ in range(reported + 100):  # overshoot in case reported is low
        if not cap.read()[0]:
            break
        actual += 1
    cap.release()

    if reported != actual:
        logger.warning(
            f"Frame count mismatch for {p.name}: "
            f"reported={reported}, actual={actual} (diff={reported - actual})"
        )

    # Compute browser-vs-OpenCV frame offset
    frame_offset = _compute_frame_offset(video_path, actual)

    # Save to cache
    cache[p.name] = {
        "size": file_size,
        "count": actual,
        "cv2_version": _CV2_VERSION,
        "frame_offset": frame_offset,
    }
    try:
        cache_path.write_text(json.dumps(cache, indent=2))
    except OSError as e:
        logger.warning(f"Could not write frame count cache: {e}")

    return actual


def _get_cached_frame_offset(video_path: str) -> int:
    """Read the cached frame_offset for a video (0 if unknown).

    Must be called after :func:`_get_actual_frame_count` has populated the cache.
    """
    p = Path(video_path)
    cache_path = p.parent / ".frame_counts.json"
    try:
        cache = json.loads(cache_path.read_text())
        return cache.get(p.name, {}).get("frame_offset", 0)
    except Exception:
        return 0


class VideoInfo:
    """Cached metadata about a video file."""
    def __init__(self, path: str):
        self.path = path
        cap = cv2.VideoCapture(path)
        self.frame_count = _get_actual_frame_count(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.midline = self.width // 2
        self.frame_offset = _get_cached_frame_offset(path)
        cap.release()


# Cache video info objects
_video_info_cache: dict[str, VideoInfo] = {}


def get_video_info(video_path: str) -> VideoInfo:
    """Get or create cached VideoInfo for a video file."""
    if video_path not in _video_info_cache:
        _video_info_cache[video_path] = VideoInfo(video_path)
    return _video_info_cache[video_path]


# Cache for no_face_videos per subject name — cleared on PATCH updates
_no_face_cache: dict[str, list[str]] = {}


def _get_no_face_videos(subject_name: str) -> list[str]:
    """Return list of video stems marked as no-face for this subject."""
    if subject_name in _no_face_cache:
        return _no_face_cache[subject_name]
    import json
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT no_face_videos FROM subjects WHERE name = ?",
            (subject_name,),
        ).fetchone()
    if row and row["no_face_videos"]:
        try:
            result = json.loads(row["no_face_videos"])
        except (json.JSONDecodeError, TypeError):
            result = []
    else:
        result = []
    _no_face_cache[subject_name] = result
    return result


def get_subject_videos(subject_name: str, *, prefer_deidentified: bool = False,
                       camera_name: str | None = None) -> list[str]:
    """Find all video files for a subject.

    In multicam mode with camera_name specified, returns only videos for
    that camera (e.g. Subject_Trial_CamName.mp4). Otherwise returns all
    videos matching Subject_*.mp4.

    Args:
        subject_name: Subject identifier (e.g. 'Con07')
        prefer_deidentified: If True, return deidentified versions when
            available. Default False — always returns originals so frame
            numbering stays consistent with labels in the database.
        camera_name: For multicam mode, filter to a specific camera's videos.
    """
    import glob
    settings = get_settings()
    video_dir = settings.video_path
    deident_dir = video_dir / "deidentified"

    pattern = str(video_dir / f"{subject_name}_*.mp4")
    videos = sorted(glob.glob(pattern))
    if not videos:
        all_vids = glob.glob(str(video_dir / "*.mp4"))
        prefix_lower = subject_name.lower() + "_"
        videos = sorted(
            v for v in all_vids
            if Path(v).name.lower().startswith(prefix_lower)
        )

    # In multicam mode, filter by camera name suffix if specified
    if camera_name and settings.default_camera_mode == "multicam":
        cam_suffix = f"_{camera_name}.mp4"
        cam_suffix_lower = cam_suffix.lower()
        videos = [v for v in videos if Path(v).name.lower().endswith(cam_suffix_lower)]

    # Optionally swap in deidentified versions
    if prefer_deidentified and deident_dir.is_dir():
        result = []
        for v in videos:
            deident_path = deident_dir / Path(v).name
            result.append(str(deident_path) if deident_path.exists() else v)
        return result

    return videos


def _group_multicam_videos(subject_name: str, videos: list[str],
                           camera_mode: str | None = None) -> list[dict]:
    """Group video files by trial, detecting per-camera files in multicam mode.

    Multicam naming convention: {Subject}_{Trial}_{CameraName}.mp4
    e.g. Con01_L1_cam0.mp4, Con01_L1_cam1.mp4

    Returns list of dicts:
        [{trial_name, cameras: [{name, path, idx}]}]

    In non-multicam mode, each video is its own trial with a single camera.

    Args:
        camera_mode: Override for camera mode. If None, uses global default.
    """
    settings = get_settings()
    mode = camera_mode or settings.default_camera_mode

    if mode != "multicam" or len(videos) <= 1:
        return [
            {"trial_name": Path(v).stem,
             "cameras": [{"name": "default", "path": v, "idx": 0}]}
            for v in videos
        ]

    from collections import defaultdict
    groups = defaultdict(list)

    prefix = subject_name + "_"
    prefix_lower = prefix.lower()

    for v in videos:
        stem = Path(v).stem
        # Remove subject prefix to get Trial_Camera or just Trial
        if stem.lower().startswith(prefix_lower):
            rest = stem[len(prefix):]  # e.g. "L1_cam0"
        else:
            rest = stem

        # Split on last underscore to separate trial and camera parts
        parts = rest.rsplit("_", 1)
        if len(parts) == 2:
            trial_part, cam_part = parts
            trial_stem = f"{subject_name}_{trial_part}"
            groups[trial_stem].append({"cam_name": cam_part, "path": v})
        else:
            # No underscore in rest — single file, its own trial
            groups[stem].append({"cam_name": "default", "path": v})

    # Only treat as multicam if at least one group has multiple files
    has_multicam = any(len(cams) > 1 for cams in groups.values())

    if not has_multicam:
        # Fall back to each file as its own trial
        return [
            {"trial_name": Path(v).stem,
             "cameras": [{"name": "default", "path": v, "idx": 0}]}
            for v in videos
        ]

    result = []
    for trial_stem, cams in sorted(groups.items()):
        cameras = [
            {"name": c["cam_name"], "path": c["path"], "idx": i}
            for i, c in enumerate(sorted(cams, key=lambda x: x["cam_name"]))
        ]
        result.append({"trial_name": trial_stem, "cameras": cameras})
    return result


def build_trial_map(subject_name: str, camera_mode: str | None = None) -> list[dict]:
    """Build a virtual timeline mapping for a subject's videos.

    Returns list of dicts with keys:
        video_path, trial_name, trial_stem, start_frame, end_frame,
        frame_count, fps, width, height, frame_offset, cameras

    In multicam mode, ``cameras`` contains [{name, path, idx}] for each
    camera file in the trial.  ``video_path`` is the primary (first) camera.

    Multiple trials are concatenated into a single frame index space.

    Args:
        camera_mode: Override for camera mode (per-subject). If None, uses global default.
    """
    videos = get_subject_videos(subject_name)
    settings = get_settings()

    # Group multicam files by trial
    grouped = _group_multicam_videos(subject_name, videos, camera_mode=camera_mode)

    trials = []
    offset = 0
    for group in grouped:
        primary_path = group["cameras"][0]["path"]
        info = get_video_info(primary_path)
        trials.append({
            "video_path": primary_path,
            "trial_name": group["trial_name"],
            "trial_stem": Path(primary_path).stem,
            "start_frame": offset,
            "end_frame": offset + info.frame_count - 1,
            "frame_count": info.frame_count,
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "frame_offset": info.frame_offset,
            "cameras": group["cameras"],
        })
        offset += info.frame_count
    return trials


def _resolve_frame(trials: list[dict], global_frame: int) -> tuple[str, int, dict]:
    """Convert global frame index to (video_path, local_frame_num, trial_dict)."""
    for trial in trials:
        if trial["start_frame"] <= global_frame <= trial["end_frame"]:
            local = global_frame - trial["start_frame"]
            return trial["video_path"], local, trial
    raise ValueError(f"Frame {global_frame} out of range")


# LRU cache for frame extraction — keyed by (video_path, frame_num, side)
@lru_cache(maxsize=FRAME_CACHE_SIZE)
def _extract_frame_cached(video_path: str, frame_num: int, side: str,
                          camera_mode: str | None = None) -> bytes:
    """Extract a single frame from video, optionally crop to left or right side, return JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_num} from {video_path}")

    settings = get_settings()
    mode = camera_mode or settings.default_camera_mode

    # Single camera or multicam mode: return full frame, no cropping
    # (multicam already resolved to the correct camera file before calling)
    if mode in ("single", "multicam"):
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return bytes(jpeg)

    # Stereo mode: crop to left or right half
    h, w = frame.shape[:2]
    midline = w // 2

    cam_names = settings.camera_names
    # First camera = left half, second camera = right half
    if len(cam_names) >= 2 and side == cam_names[1]:
        crop = frame[:, midline:]
    else:
        crop = frame[:, :midline]

    _, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return bytes(jpeg)


def _deidentified_path(video_path: str) -> str | None:
    """Return the deidentified version of a video if it exists."""
    p = Path(video_path)
    deident = p.parent / "deidentified" / p.name
    return str(deident) if deident.exists() else None


def _resolve_camera_path(trial: dict, side: str) -> str | None:
    """For multicam trials, find the video path for the given camera/side name.

    Returns None if this trial doesn't have multicam cameras or the side
    doesn't match any camera name.
    """
    cameras = trial.get("cameras", [])
    if len(cameras) <= 1:
        return None
    for cam in cameras:
        if cam["name"] == side:
            return cam["path"]
    return None


def extract_frame(subject_name: str, global_frame: int, side: str,
                  trials: list[dict] | None = None,
                  camera_mode: str | None = None) -> bytes:
    """Extract a frame for a subject at a global frame index.

    Args:
        subject_name: Subject identifier
        global_frame: Frame number in the virtual timeline
        side: Camera name (first or second from settings.camera_names),
              or multicam camera name
        trials: Pre-computed trial map (optional, computed if None)
        camera_mode: Per-subject camera mode (falls back to global default)

    Returns:
        JPEG bytes
    """
    if trials is None:
        trials = build_trial_map(subject_name, camera_mode=camera_mode)
    video_path, local_frame, trial = _resolve_frame(trials, global_frame)

    settings = get_settings()
    mode = camera_mode or settings.default_camera_mode

    # In multicam mode, resolve to the camera-specific file
    if mode == "multicam":
        cam_path = _resolve_camera_path(trial, side)
        if cam_path:
            video_path = cam_path

    # Optionally display deidentified version (frame numbering from originals)
    # Skip deidentified swap for videos marked as no-face
    if settings.prefer_deidentified:
        stem = Path(video_path).stem
        no_face = _get_no_face_videos(subject_name)
        if stem not in no_face:
            deident = _deidentified_path(video_path)
            if deident:
                video_path = deident

    return _extract_frame_cached(video_path, local_frame, side, camera_mode=mode)


def extract_frame_raw(video_path: str, frame_num: int, side: str,
                      camera_mode: str | None = None) -> np.ndarray:
    """Extract a frame as a raw numpy array (for saving PNGs on commit)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_num} from {video_path}")

    settings = get_settings()
    mode = camera_mode or settings.default_camera_mode

    # Single camera or multicam mode: return full frame (no cropping)
    if mode in ("single", "multicam"):
        return frame

    # Stereo mode: crop to left or right half
    h, w = frame.shape[:2]
    midline = w // 2

    cam_names = settings.camera_names
    if len(cam_names) >= 2 and side == cam_names[1]:
        return frame[:, midline:]
    else:
        return frame[:, :midline]


def get_total_frames(subject_name: str) -> int:
    """Get total frame count across all trials."""
    trials = build_trial_map(subject_name)
    if not trials:
        return 0
    return trials[-1]["end_frame"] + 1


def cache_all_frame_counts(video_dir: str) -> None:
    """Pre-populate the frame count cache for all .mp4 files in a directory."""
    import glob, time
    videos = sorted(glob.glob(str(Path(video_dir) / "*.mp4")))
    print(f"Caching frame counts for {len(videos)} videos in {video_dir}")
    for i, vpath in enumerate(videos):
        name = Path(vpath).name
        t0 = time.time()
        count = _get_actual_frame_count(vpath)
        dt = time.time() - t0
        marker = f"({dt:.1f}s)" if dt > 0.5 else "(cached)"
        print(f"  [{i+1}/{len(videos)}] {name}: {count} frames {marker}")
    print("Done.")
