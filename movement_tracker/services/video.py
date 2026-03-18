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


def get_subject_videos(subject_name: str, *, prefer_deidentified: bool = False) -> list[str]:
    """Find all stereo video files for a subject.

    Args:
        subject_name: Subject identifier (e.g. 'Con07')
        prefer_deidentified: If True, return deidentified versions when
            available. Default False — always returns originals so frame
            numbering stays consistent with labels in the database.
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

    # Optionally swap in deidentified versions
    if prefer_deidentified and deident_dir.is_dir():
        result = []
        for v in videos:
            deident_path = deident_dir / Path(v).name
            result.append(str(deident_path) if deident_path.exists() else v)
        return result

    return videos


def build_trial_map(subject_name: str) -> list[dict]:
    """Build a virtual timeline mapping: list of {video_path, trial_name, start_frame, end_frame, frame_count}.

    Multiple trials are concatenated into a single frame index space.
    """
    videos = get_subject_videos(subject_name)
    trials = []
    offset = 0
    for vpath in videos:
        info = get_video_info(vpath)
        trials.append({
            "video_path": vpath,
            "trial_name": Path(vpath).stem,
            "start_frame": offset,
            "end_frame": offset + info.frame_count - 1,
            "frame_count": info.frame_count,
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "frame_offset": info.frame_offset,
        })
        offset += info.frame_count
    return trials


def _resolve_frame(trials: list[dict], global_frame: int) -> tuple[str, int]:
    """Convert global frame index to (video_path, local_frame_num)."""
    for trial in trials:
        if trial["start_frame"] <= global_frame <= trial["end_frame"]:
            local = global_frame - trial["start_frame"]
            return trial["video_path"], local
    raise ValueError(f"Frame {global_frame} out of range")


# LRU cache for frame extraction — keyed by (video_path, frame_num, side)
@lru_cache(maxsize=FRAME_CACHE_SIZE)
def _extract_frame_cached(video_path: str, frame_num: int, side: str) -> bytes:
    """Extract a single frame from video, crop to left or right side, return JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_num} from {video_path}")

    h, w = frame.shape[:2]
    midline = w // 2

    settings = get_settings()
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


def extract_frame(subject_name: str, global_frame: int, side: str,
                  trials: list[dict] | None = None) -> bytes:
    """Extract a frame for a subject at a global frame index.

    Args:
        subject_name: Subject identifier
        global_frame: Frame number in the virtual timeline
        side: Camera name (first or second from settings.camera_names)
        trials: Pre-computed trial map (optional, computed if None)

    Returns:
        JPEG bytes
    """
    if trials is None:
        trials = build_trial_map(subject_name)
    video_path, local_frame = _resolve_frame(trials, global_frame)

    # Optionally display deidentified version (frame numbering from originals)
    # Skip deidentified swap for videos marked as no-face
    settings = get_settings()
    if settings.prefer_deidentified:
        stem = Path(video_path).stem
        no_face = _get_no_face_videos(subject_name)
        if stem not in no_face:
            deident = _deidentified_path(video_path)
            if deident:
                video_path = deident

    return _extract_frame_cached(video_path, local_frame, side)


def extract_frame_raw(video_path: str, frame_num: int, side: str) -> np.ndarray:
    """Extract a frame as a raw numpy array (for saving PNGs on commit)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_num} from {video_path}")

    h, w = frame.shape[:2]
    midline = w // 2

    settings = get_settings()
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
