#!/usr/bin/env python3
"""MediaPipe job worker — a subprocess so jobs survive app restarts.

Usage:
    python worker.py --job-type mediapipe --subject Con01 --job-id 42 \\
        [--trial-idx 0] [--reverse] [--static-image-mode] [--no-bbox]

Progress is reported on stdout as ``PROGRESS:42.5``; the parent process
(``jobs.JobRegistry._monitor``) parses those and updates the DB.

DeepLabCut jobs do NOT come through here — they run
``services/dlc_pipeline.py``, which is standalone so it can execute under
a different interpreter than the web app.
"""
from __future__ import annotations

import argparse
import os
import sys

# Bootstrap sys.path so absolute-path invocations work under the portable
# Python that run.bat installs on Windows: its ``python3XX._pth`` file
# overrides sys.path AND ignores PYTHONPATH, so neither cwd nor
# PYTHONPATH gets us the ``dlc_labeler`` package.  Inserting PROJECT_DIR
# (the parent of the package directory) explicitly does.  Idempotent.
_PROJECT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def _progress_printer():
    """Return a progress callback that prints for the parent to parse."""
    def cb(pct: float):
        print(f"PROGRESS:{pct:.1f}", flush=True)
    return cb


def _load_crop_boxes(subject_name: str) -> dict | None:
    """Load the saved per-trial MediaPipe bounding boxes for a subject.

    Returned keyed by trial index with 'OS' / 'OD' entries, the shape
    ``run_mediapipe`` expects.  Returns None when the subject has no
    saved boxes, which makes MediaPipe run on the full camera half.
    """
    from dlc_labeler.config import get_settings
    from dlc_labeler.db import get_db_ctx

    cam_names = get_settings().camera_names or ["OS", "OD"]
    cam_os = cam_names[0]
    cam_od = cam_names[1] if len(cam_names) > 1 else cam_names[0]

    crop_boxes: dict = {}
    try:
        with get_db_ctx() as db:
            subj = db.execute("SELECT id FROM subjects WHERE name = ?",
                              (subject_name,)).fetchone()
            if not subj:
                return None
            rows = db.execute(
                "SELECT trial_idx, camera_name, x1, y1, x2, y2 FROM mp_crop_boxes "
                "WHERE subject_id = ? AND model_name = 'run-mediapipe'",
                (subj["id"],),
            ).fetchall()
        for r in rows:
            box = [r["x1"], r["y1"], r["x2"], r["y2"]]
            entry = crop_boxes.setdefault(r["trial_idx"], {})
            if r["camera_name"] == cam_os:
                entry["OS"] = box
            elif r["camera_name"] == cam_od:
                entry["OD"] = box
    except Exception as e:
        print(f"WARN: could not load mp_crop_boxes for {subject_name}: {e}",
              flush=True)
        return None

    return crop_boxes or None


def run_mediapipe(subject_name: str, static_image_mode: bool = False,
                  trial_idx: int | None = None, reverse: bool = False,
                  use_bbox: bool = True):
    from dlc_labeler.services.mediapipe_prelabel import run_mediapipe as _run_mp

    crop_boxes = _load_crop_boxes(subject_name) if use_bbox else None
    _run_mp(subject_name,
            progress_callback=_progress_printer(),
            crop_boxes=crop_boxes,
            static_image_mode=static_image_mode,
            trial_idx=trial_idx,
            reverse=reverse,
            use_bbox=use_bbox)


JOB_DISPATCH = {
    "mediapipe": lambda a: run_mediapipe(
        a.subject,
        static_image_mode=a.static_image_mode,
        trial_idx=a.trial_idx,
        reverse=a.reverse,
        use_bbox=not a.no_bbox,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="DLC Labeler job worker")
    parser.add_argument("--job-type", required=True, choices=list(JOB_DISPATCH))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--trial-idx", type=int, default=None,
                        help="Process only this trial, leaving other trials' "
                             "saved landmarks untouched")
    parser.add_argument("--static-image-mode", action="store_true",
                        help="Run the full palm detector on every frame "
                             "(no between-frame tracker)")
    parser.add_argument("--reverse", action="store_true",
                        help="Feed frames in reverse temporal order")
    parser.add_argument("--no-bbox", action="store_true",
                        help="Ignore saved crop boxes; use the full camera half")
    parser.add_argument("--data-dir", default=None,
                        help="DLC_DATA_DIR for this subprocess")

    args = parser.parse_args()

    if args.data_dir:
        os.environ["DLC_DATA_DIR"] = args.data_dir

    try:
        JOB_DISPATCH[args.job_type](args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
