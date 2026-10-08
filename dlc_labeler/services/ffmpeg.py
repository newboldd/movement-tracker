"""Locate ffmpeg/ffprobe binaries.

Checks system PATH first, then falls back to the imageio-ffmpeg pip package
which bundles a static binary.  This means ffmpeg works on a fresh Windows
machine with no system-level installs — just ``pip install -r requirements.txt``.
"""
from __future__ import annotations

import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def get_ffmpeg_path() -> str:
    """Return the path to an ffmpeg executable."""
    # 1. System PATH
    path = shutil.which("ffmpeg")
    if path:
        return path

    # 2. imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass

    raise FileNotFoundError(
        "ffmpeg not found. Either install FFmpeg system-wide or "
        "run: pip install imageio-ffmpeg"
    )


@lru_cache(maxsize=1)
def get_ffprobe_path() -> str:
    """Return the path to an ffprobe executable."""
    # 1. System PATH
    path = shutil.which("ffprobe")
    if path:
        return path

    # 2. Derive from imageio-ffmpeg's ffmpeg binary
    #    imageio-ffmpeg ships ffmpeg; ffprobe lives next to it
    try:
        import imageio_ffmpeg
        from pathlib import Path
        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        # ffprobe is typically in the same directory
        ffprobe = ffmpeg.parent / ffmpeg.name.replace("ffmpeg", "ffprobe")
        if ffprobe.exists():
            return str(ffprobe)
    except (ImportError, RuntimeError):
        pass

    # ffprobe not available — callers should handle this gracefully
    raise FileNotFoundError(
        "ffprobe not found. Either install FFmpeg system-wide or "
        "run: pip install imageio-ffmpeg"
    )
