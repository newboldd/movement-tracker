"""Video frame extraction and stereo cropping."""
from __future__ import annotations

import cv2
import numpy as np
from functools import lru_cache
from pathlib import Path

from ..config import get_settings, JPEG_QUALITY, FRAME_CACHE_SIZE


class VideoInfo:
    """Cached metadata about a video file."""
    def __init__(self, path: str):
        self.path = path
        cap = cv2.VideoCapture(path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.midline = self.width // 2
        cap.release()


# Cache video info objects
_video_info_cache: dict[str, VideoInfo] = {}


def get_video_info(video_path: str) -> VideoInfo:
    """Get or create cached VideoInfo for a video file."""
    if video_path not in _video_info_cache:
        _video_info_cache[video_path] = VideoInfo(video_path)
    return _video_info_cache[video_path]


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
