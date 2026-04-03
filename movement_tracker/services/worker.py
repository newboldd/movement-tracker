#!/usr/bin/env python3
"""Universal local job worker — runs as a subprocess so jobs survive app restarts.

Usage:
    python -m movement_tracker.services.worker --job-type mediapipe --subject "Con01" --job-id 42 --log-path /tmp/job.log
    python -m movement_tracker.services.worker --job-type deidentify --subject "PD01" --job-id 43 --log-path /tmp/job.log --trial-idx 0

Progress is reported via stdout lines:  PROGRESS:42.5
The parent process (registry._monitor) parses these and updates the DB.
"""
from __future__ import annotations

import argparse
import os
import sys


def _progress_printer(job_id: int):
    """Return a progress callback that prints to stdout for the monitor to parse."""
    def cb(pct: float):
        print(f"PROGRESS:{pct:.1f}", flush=True)
    return cb


def run_mediapipe(subject_name: str, job_id: int):
    from movement_tracker.services.mediapipe_prelabel import run_mediapipe as _run_mp
    _run_mp(subject_name, progress_callback=_progress_printer(job_id))


def run_pose(subject_name: str, job_id: int):
    from movement_tracker.services.mediapipe_prelabel import run_pose_prelabels
    run_pose_prelabels(subject_name, progress_callback=_progress_printer(job_id))


def run_blur(subject_name: str, job_id: int):
    from movement_tracker.services.deidentify import deidentify_video_pipeline
    deidentify_video_pipeline(subject_name, progress_callback=_progress_printer(job_id))


def run_deidentify(subject_name: str, job_id: int, trial_idx: int | None = None):
    from movement_tracker.services.deidentify import render_with_blur_specs
    from movement_tracker.services.video import build_trial_map
    from movement_tracker.config import get_settings
    from movement_tracker.db import get_db_ctx
    import json

    settings = get_settings()
    with get_db_ctx() as db:
        subj = db.execute("SELECT * FROM subjects WHERE name = ?", (subject_name,)).fetchone()
        if not subj:
            raise ValueError(f"Subject not found: {subject_name}")

    camera_mode = subj.get("camera_mode") or "stereo"
    trials = build_trial_map(subject_name, camera_mode=camera_mode)

    if trial_idx is not None:
        trial_indices = [trial_idx]
    else:
        trial_indices = list(range(len(trials)))

    n_trials = len(trial_indices)
    for ti_pos, ti in enumerate(trial_indices):
        if ti >= len(trials):
            continue
        trial = trials[ti]

        # Load blur specs
        with get_db_ctx() as db:
            specs = db.execute(
                "SELECT * FROM blur_specs WHERE subject_id = ? AND trial_idx = ?",
                (subj["id"], ti),
            ).fetchall()
            face_rows = db.execute(
                "SELECT frame_num, x1, y1, x2, y2, side FROM face_detections WHERE subject_id = ? AND trial_idx = ?",
                (subj["id"], ti),
            ).fetchall()
            hs_row = db.execute(
                "SELECT * FROM blur_hand_settings WHERE subject_id = ? AND trial_idx = ?",
                (subj["id"], ti),
            ).fetchone()

        if not specs:
            continue

        blur_specs = [dict(s) for s in specs]

        # Build face detection lookup
        face_by_frame = {}
        for fd in face_rows:
            fn = int(fd["frame_num"])
            if fn not in face_by_frame:
                face_by_frame[fn] = []
            face_by_frame[fn].append(dict(fd))

        # Hand settings
        hand_mask_radius = 10
        hand_smooth = 10
        forearm_radius = 10
        forearm_extent = 0.7
        hand_smooth2 = 0
        dlc_radius = 10
        hand_temporal = 0
        hand_segments = []

        if hs_row:
            hand_mask_radius = hs_row.get("hand_mask_radius") or 10
            hand_smooth = hs_row.get("hand_smooth") or 10
            forearm_radius = hs_row.get("forearm_radius") or 10
            forearm_extent = hs_row.get("forearm_extent") or 0.7
            hand_smooth2 = hs_row.get("hand_smooth2") or 0
            dlc_radius = hs_row.get("dlc_radius") or 10
            hand_temporal = hs_row.get("hand_temporal") or 0
            seg_json = hs_row.get("segments_json", "[]")
            try:
                hand_segments = json.loads(seg_json) if isinstance(seg_json, str) else (seg_json or [])
            except (ValueError, TypeError):
                hand_segments = []

        # Output path
        output_dir = settings.video_path
        deident_dir = os.path.join(str(output_dir), "deidentified")
        os.makedirs(deident_dir, exist_ok=True)
        output_name = f"{trial['trial_name']}.mp4"
        output_path = os.path.join(deident_dir, output_name)

        base_pct = ti_pos * (100.0 / max(n_trials, 1))
        span = 100.0 / max(n_trials, 1)

        def progress_cb(pct, _base=base_pct, _span=span):
            overall = _base + (pct / 100.0) * _span
            print(f"PROGRESS:{overall:.1f}", flush=True)

        # Build hand_settings dict matching function signature
        hand_settings_dict = {
            "hand_mask_radius": hand_mask_radius,
            "hand_smooth": hand_smooth,
            "forearm_radius": forearm_radius,
            "forearm_extent": forearm_extent,
            "hand_smooth2": hand_smooth2,
            "dlc_radius": dlc_radius,
            "hand_temporal": hand_temporal,
            "segments_json": json.dumps(hand_segments),
        }

        # Face detections as flat list (function converts to by-frame internally)
        face_list = [dict(fd) for fd in face_rows]

        render_with_blur_specs(
            input_path=trial["video_path"],
            output_path=output_path,
            blur_specs=blur_specs,
            hand_settings=hand_settings_dict,
            face_detections=face_list,
            subject_name=subject_name,
            start_frame=trial["start_frame"],
            progress_callback=progress_cb,
        )

    print("PROGRESS:100.0", flush=True)


def run_vision(subject_name: str, job_id: int):
    try:
        from movement_tracker.services.vision_prelabel import run_vision_prelabels
        run_vision_prelabels(subject_name, progress_callback=_progress_printer(job_id))
    except ImportError:
        print("Apple Vision framework not available on this platform", file=sys.stderr)
        sys.exit(1)


JOB_DISPATCH = {
    "mediapipe": lambda args: run_mediapipe(args.subject, args.job_id),
    "pose": lambda args: run_pose(args.subject, args.job_id),
    "blur": lambda args: run_blur(args.subject, args.job_id),
    "deidentify": lambda args: run_deidentify(args.subject, args.job_id, args.trial_idx),
    "vision": lambda args: run_vision(args.subject, args.job_id),
}


def _parse_worker_progress(line: str):
    """Parse PROGRESS:42.5 lines from worker stdout."""
    line = line.strip()
    if line.startswith("PROGRESS:"):
        try:
            pct = float(line.split(":", 1)[1])
            return (pct, None, None)
        except (ValueError, IndexError):
            pass
    return None


# Export for use by local_executor
parse_worker_progress = _parse_worker_progress


def main():
    parser = argparse.ArgumentParser(description="Movement Tracker job worker")
    parser.add_argument("--job-type", required=True, choices=list(JOB_DISPATCH.keys()))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--trial-idx", type=int, default=None)

    # Set MT_DATA_DIR if passed (for subprocess to find DB/settings)
    parser.add_argument("--data-dir", default=None)

    args = parser.parse_args()

    if args.data_dir:
        os.environ["MT_DATA_DIR"] = args.data_dir

    handler = JOB_DISPATCH.get(args.job_type)
    if not handler:
        print(f"Unknown job type: {args.job_type}", file=sys.stderr)
        sys.exit(1)

    try:
        handler(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
