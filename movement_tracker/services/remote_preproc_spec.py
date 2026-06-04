"""Preproc plug-in for the generic remote batch framework.

Defines :data:`PREPROC_BATCH_SPEC` — a :class:`RemoteBatchJobSpec`
that wires the existing per-subject worker logic into
``services.remote_batch``.

NOT yet wired to any UI button or queue-manager dispatch path.
The legacy ``dispatch_remote_preproc_batch`` in ``services.remote``
remains the active code path for the "preproc" job type.  When
ready to migrate:

    from .remote_batch import dispatch_remote_batch, poll_remote_batch
    from .remote_preproc_spec import PREPROC_BATCH_SPEC, preproc_items_from_trials

    items = preproc_items_from_trials(trials_batch)
    result = dispatch_remote_batch(PREPROC_BATCH_SPEC, job_id, cfg,
                                    items, log_path, registry)

REMOTE LAYOUT (this spec)
-------------------------

Inputs / outputs colocated under ``subjects/<subject>/``:

    <work>/subjects/<sub>/
        videos/<v>.mp4
        mediapipe_prelabels.npz
        pose_prelabels.npz
        preproc/<trial>/
            camera_trajectory.npz
            background.npz
            background_refined.npz
            outlines.json
            stable.mp4                  (kept on remote, not downloaded)
            background_OS.png           (downloaded for visual review)
            background_OD.png           (downloaded for visual review)
            background_refined_OS.png   (downloaded)
            background_refined_OD.png   (downloaded)

Per-subject "item" granularity matches the legacy preproc — one
item per subject, all that subject's trials bake in sequence inside
the per-subject worker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .remote import RemoteConfig
from .remote_batch import (
    RemoteBatchJobSpec,
    subject_remote,
    subject_videos_remote,
)

# Per-trial output files downloaded back to local after a subject's
# bake completes.  stable.mp4 (~13 MB / 1100 frames) intentionally
# stays on the remote — it's an intermediate, not a deliverable.
_PREPROC_DOWNLOAD_FILES = (
    "camera_trajectory.npz",
    "outlines.json",
    "background_OS.png",
    "background_OD.png",
    "background_refined_OS.png",
    "background_refined_OD.png",
)


def _preproc_worker_source() -> str:
    """Import lazily so circular-import risk between remote.py and
    remote_batch.py / remote_preproc_spec.py stays nil.  The worker
    script is the 5-phase per-subject runner from remote.py."""
    from .remote import _REMOTE_PREPROC_WORKER
    return _REMOTE_PREPROC_WORKER


def _preproc_shared_modules() -> tuple[Path, ...]:
    """Local .py paths for the modules the worker imports.  Same set
    as the legacy ``dispatch_remote_preproc_batch`` shipped."""
    svc_dir = Path(__file__).parent
    return (
        svc_dir / "camera_motion.py",
        svc_dir / "background.py",
        svc_dir / "ffmpeg.py",
    )


def _preproc_item_uploads(item: dict, cfg: RemoteConfig, settings) -> list[tuple]:
    """Per-subject upload list: video files for every trial, plus
    mediapipe_prelabels.npz / pose_prelabels.npz if present locally.
    """
    sub = item["subject_name"]
    out: list[tuple] = []
    # Build trial map locally so we know which video files to ship.
    from .video import build_trial_map
    all_trials = build_trial_map(sub)
    sub_videos_remote = subject_videos_remote(cfg, sub)
    for t in item["trials"]:
        try:
            ti = int(t["trial_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if ti < 0 or ti >= len(all_trials):
            continue
        tr = all_trials[ti]
        vname = Path(tr["video_path"]).name
        local_v = settings.video_path / vname
        if local_v.exists():
            out.append((str(local_v), f"{sub_videos_remote}/{vname}"))
    # MP / pose npz (subject-level).
    sub_root_remote = subject_remote(cfg, sub)
    mp_local = settings.dlc_path / sub / "mediapipe_prelabels.npz"
    if mp_local.exists():
        out.append((str(mp_local), f"{sub_root_remote}/mediapipe_prelabels.npz"))
    pose_local = settings.dlc_path / sub / "pose_prelabels.npz"
    if pose_local.exists():
        out.append((str(pose_local), f"{sub_root_remote}/pose_prelabels.npz"))
    return out


def _preproc_item_spec(item: dict, cfg: RemoteConfig, settings) -> dict:
    """Per-subject bundle.json content the worker reads.  Mirrors the
    legacy bundle shape so the existing ``_REMOTE_PREPROC_WORKER``
    keeps working unchanged.
    """
    sub = item["subject_name"]
    sub_root_remote = subject_remote(cfg, sub)
    sub_videos_remote_ = subject_videos_remote(cfg, sub)

    from .video import build_trial_map
    all_trials = build_trial_map(sub)

    bundle_trials = []
    for t in item["trials"]:
        try:
            ti = int(t["trial_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if ti < 0 or ti >= len(all_trials):
            continue
        tr = all_trials[ti]
        vname = Path(tr["video_path"]).name
        bundle_trials.append({
            "trial_idx":          ti,
            "trial_name":         tr.get("trial_name", t.get("trial_name", "")),
            "video_name":         vname,
            "video_remote_path":  f"{sub_videos_remote_}/{vname}",
            "frame_count":        int(tr.get("frame_count", 0)),
            "start_frame":        int(tr.get("start_frame", 0)),
            "fps":                float(tr.get("fps", 30.0)),
        })
    # NOTE: ``data_dir`` is the directory the worker treats as its
    # local DLC root — the worker's ``_install_stubs`` builds a
    # SimpleNamespace where ``dlc_path = video_path = data_dir``,
    # then reads e.g. ``data_dir / <subject> / mediapipe_prelabels.npz``.
    # So the worker expects an EXTRA <subject>/ layer inside data_dir.
    #
    # New layout puts all subject files directly under
    # subjects/<sub>/.  To keep the worker code unchanged we set
    # ``data_dir = <work>/subjects`` so ``data_dir/<sub>/...`` resolves
    # correctly.  Outputs land at data_dir/<sub>/preproc/<trial>/,
    # which equals <work>/subjects/<sub>/preproc/<trial>/ — exactly
    # the layout this spec advertises.
    data_dir = f"{cfg.work_dir}/subjects"
    return {
        "subject_name": sub,
        "data_dir":     data_dir,
        "trials":       bundle_trials,
    }


def _preproc_item_id(item: dict) -> str:
    return str(item["subject_name"])


def _preproc_item_downloads(item: dict, cfg: RemoteConfig, settings) -> list[tuple]:
    """Per-subject output download list.  Each trial under
    ``subjects/<sub>/preproc/<trial>/`` contributes the
    ``_PREPROC_DOWNLOAD_FILES`` allow-list.  Trials whose outputs
    don't exist on the remote (worker bailed mid-bake) just get
    skipped silently — the scp returns non-zero and the file isn't
    created locally.
    """
    sub = item["subject_name"]
    sub_root_remote = subject_remote(cfg, sub)
    preproc_remote = f"{sub_root_remote}/preproc"
    local_root = settings.dlc_path / sub / "preproc"
    out: list[tuple] = []
    # We don't know the trial stems on the remote without an SSH
    # round-trip; instead enumerate every trial we sent in the
    # bundle.  Missing files get silently skipped by the scp.
    from .video import build_trial_map
    all_trials = build_trial_map(sub)
    for t in item["trials"]:
        try:
            ti = int(t["trial_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if ti < 0 or ti >= len(all_trials):
            continue
        stem = all_trials[ti].get("trial_name") or t.get("trial_name", "")
        if not stem:
            continue
        for fname in _PREPROC_DOWNLOAD_FILES:
            remote_p = f"{preproc_remote}/{stem}/{fname}"
            local_p  = local_root / stem / fname
            out.append((remote_p, str(local_p)))
    return out


def preproc_items_from_trials(trials_batch: list[dict]) -> list[dict]:
    """Group the flat ``trials_batch`` (per-trial dicts from the
    frontend) into one ``item`` per subject.

        items = preproc_items_from_trials(trials_batch)
        # → [
        #     {"subject_name": "Con01", "trials": [...]},
        #     {"subject_name": "Con02", "trials": [...]},
        #     ...
        #   ]
    """
    from collections import defaultdict
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for t in trials_batch:
        sn = t.get("subject_name")
        if sn:
            by_subj[sn].append(t)
    return [
        {"subject_name": sn, "trials": trs}
        for sn, trs in by_subj.items()
    ]


PREPROC_BATCH_SPEC = RemoteBatchJobSpec(
    job_type="preproc",
    # _preproc_worker_source is lazy — Python resolves it at first
    # access, after services.remote has finished importing.  Calling
    # at module-import time would risk circular import.
    worker_script_source="",   # populated below
    shared_module_paths=(),    # populated below
    item_uploads_fn=_preproc_item_uploads,
    item_spec_fn=_preproc_item_spec,
    item_id_fn=_preproc_item_id,
    item_downloads_fn=_preproc_item_downloads,
    display_label="preproc",
)


def _lazy_init():
    """Populate the worker source + module paths the first time the
    spec is used.  Keeps services.remote import-order independent
    of this module."""
    if not PREPROC_BATCH_SPEC.worker_script_source:
        PREPROC_BATCH_SPEC.worker_script_source = _preproc_worker_source()
    if not PREPROC_BATCH_SPEC.shared_module_paths:
        PREPROC_BATCH_SPEC.shared_module_paths = _preproc_shared_modules()


# Eager init at import time — safe because services.remote is
# already imported above (the RemoteBatchJobSpec import chain).
_lazy_init()
