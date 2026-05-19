"""Remote-result tracking and on-demand download.

Lets the running app:
  * record (per subject, per job_type) which remote outputs have been
    pulled to local — see ``remote_downloads`` table;
  * mark unwanted results as ``ignored`` so they don't keep nagging the
    user;
  * inspect the remote host for newly-available outputs (per-subject MP
    npz today; vision/pose/hrnet/deidentify will follow the same pattern);
  * download a single subject's outputs in-thread and record the row.

Currently MEDIAPIPE-ONLY.  Other job types will plug into ``OUTPUT_SPECS``
and ``download_one`` once the Jobs page UI surfaces them.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..config import get_settings
from ..db import get_db_ctx
from .remote import RemoteConfig, _scp_base_args, _ssh_base_args

logger = logging.getLogger(__name__)


# ─── Output specifications per job_type ────────────────────────────────

@dataclass
class OutputSpec:
    """One downloadable output produced by a remote job for one subject."""
    job_type: str
    label: str          # human-readable: "MediaPipe npz"
    remote_path: str    # absolute path on the remote host
    local_path: Path    # absolute path on local

    def to_row(self, subject_id: int, job_id: Optional[int],
               remote_size: Optional[int], local_size: Optional[int]) -> dict:
        return {
            "subject_id": subject_id,
            "job_id": job_id,
            "job_type": self.job_type,
            "output_path": str(self.local_path),
            "remote_size": remote_size,
            "local_size": local_size,
        }


def output_specs(job_type: str, subject_name: str,
                 remote_output_dir: str) -> list[OutputSpec]:
    """Return the list of expected outputs for a (job_type, subject) pair.

    MediaPipe outputs are now per-trial: one
    ``<remote>/<subject>/<trial_stem>/mediapipe.npz`` (and
    ``mediapipe_reverse.npz`` when the reverse pass ran).  We enumerate
    every trial in the local trial map; the SSH probe layer skips
    files that don't exist remotely, so listing both passes here is
    harmless even when only one ran.

    Legacy combined-file pull is still listed as a fallback spec so an
    older remote install that hasn't been migrated yet remains
    downloadable (the local startup migration then splits the combined
    file into per-trial files).
    """
    settings = get_settings()
    if job_type == "mediapipe":
        # Enumerate this subject's trials.  build_trial_map may fail
        # if the subject's videos aren't on this machine -- in that
        # case fall back to the legacy combined spec so the pull
        # still works.
        try:
            from .video import build_trial_map
            trials = build_trial_map(subject_name)
        except Exception:
            trials = []
        specs: list[OutputSpec] = []
        for t in trials:
            stem = t["trial_name"]
            for fname in ("mediapipe.npz", "mediapipe_reverse.npz",
                          "mediapipe_cropped.npz", "mediapipe_static.npz"):
                tag = ({"mediapipe.npz": "",
                        "mediapipe_reverse.npz": " reverse",
                        "mediapipe_cropped.npz": " cropped",
                        "mediapipe_static.npz": " static"})[fname]
                specs.append(OutputSpec(
                    job_type="mediapipe",
                    label=f"MediaPipe npz ({stem}{tag})",
                    remote_path=f"{remote_output_dir}/{subject_name}/{stem}/{fname}",
                    local_path=settings.dlc_path / subject_name / stem / fname,
                ))
        # Legacy combined-file fallback (only useful for older remote
        # installs that wrote the combined npz).  Keeps backward
        # compatibility until every remote host has been re-run.
        specs.append(OutputSpec(
            job_type="mediapipe",
            label="MediaPipe npz (legacy combined)",
            remote_path=f"{remote_output_dir}/{subject_name}/mediapipe_prelabels.npz",
            local_path=settings.dlc_path / subject_name / "mediapipe_prelabels.npz",
        ))
        return specs
    # Vision / pose / hrnet / deidentify will be added here.
    return []


# ─── Remote file probing ───────────────────────────────────────────────

def _norm_remote_path(p: str) -> str:
    """Normalize a remote path to use forward slashes.  Windows OpenSSH
    accepts ``C:/Users/...`` for SCP/SSH and is more reliable than mixed
    slashes (which ``os.path.exists`` is happy to accept but ``scp`` is
    sometimes not)."""
    return p.replace("\\", "/")


def _ssh_stat_quick(cfg: RemoteConfig, remote_path: str) -> Optional[int]:
    """Single-probe variant of ``_ssh_stat``.  Cheap — used by the
    pending-downloads listing endpoint where we only need to know the
    file is roughly available, not that it's stable.

    Uses ``cfg.python_executable`` (the explicit conda-env interpreter
    path) rather than bare ``python``.  Bare ``python`` only works
    when the user's default shell auto-activates a conda env, which
    nobody can rely on on Windows SSH sessions -- there ``python``
    typically isn't on PATH and the probe silently returned None
    while the file was actually present (which is what masked
    Con02's missing-MP-npz download)."""
    rp = _norm_remote_path(remote_path)
    # Wrap the script in literal double quotes so Windows OpenSSH +
    # PowerShell treat it as a single argument.  Without the outer
    # ``"..."`` PowerShell splits on ``;`` and reparses the raw-string
    # ``r'...'`` syntax, which made every probe fail silently on the
    # ``windows-dev`` host (re-download log: "npz not found for Con02"
    # while the file was actually there).  Mirrors the wrapping in
    # ``remote._py_cmd``.
    script = (
        f"\"import os,sys;p=r'{rp}';"
        "print(os.path.getsize(p) if os.path.exists(p) else -1)\""
    )
    cmd = _ssh_base_args(cfg) + [cfg.host, cfg.python_executable, "-c", script]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            return None
        out = proc.stdout.strip()
        if not out or out == "-1":
            return None
        return int(out)
    except (subprocess.TimeoutExpired, ValueError):
        return None


def _ssh_stat(cfg: RemoteConfig, remote_path: str) -> Optional[int]:
    """Return remote file size in bytes, or None if missing/unreachable.

    Polls the size twice with a 1-second gap and only returns when the
    two samples agree — guards against races where ``np.savez`` is
    mid-write and SCP would later see a partially-written file.

    Use ``_ssh_stat_quick`` for listing (no stability check needed).
    """
    s1 = _ssh_stat_quick(cfg, remote_path)
    if s1 is None:
        return None
    import time as _t
    _t.sleep(1.0)
    s2 = _ssh_stat_quick(cfg, remote_path)
    if s2 is None or s1 != s2:
        return None
    return s2


def _scp_pull(cfg: RemoteConfig, remote_path: str, local_path: Path) -> bool:
    """Download ``remote_path`` to ``local_path``.  Logs elapsed time and
    transfer size on completion so the per-file timing is visible in the
    Jobs log.  Returns True on success."""
    import time as _time
    local_path.parent.mkdir(parents=True, exist_ok=True)
    rp = _norm_remote_path(remote_path)
    cmd = _scp_base_args(cfg) + [
        f"{cfg.host}:{rp}", str(local_path),
    ]
    label = f"{Path(rp).parent.name}/{Path(rp).name}"
    logger.info(f"Downloading {label} …")
    t0 = _time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = _time.time() - t0
        if proc.returncode != 0:
            logger.warning(f"SCP failed: {remote_path} → {local_path}: "
                           f"{proc.stderr[-200:]}")
            return False
        size_mb = local_path.stat().st_size / 1e6 if local_path.exists() else 0.0
        rate = (size_mb / elapsed) if elapsed > 0.05 else 0.0
        logger.info(f"  Downloaded {label}: {size_mb:.1f} MB in {elapsed:.1f}s "
                    f"({rate:.1f} MB/s)")
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"SCP timeout: {remote_path}")
        return False


# ─── DB helpers ────────────────────────────────────────────────────────

def already_downloaded(subject_id: int, job_type: str,
                        output_path: str) -> Optional[dict]:
    """Return existing remote_downloads row for this (subject, type, path)
    or None.  Used to detect previously-completed downloads + ignored
    rows."""
    with get_db_ctx() as db:
        row = db.execute(
            "SELECT * FROM remote_downloads "
            "WHERE subject_id = ? AND job_type = ? AND output_path = ?",
            (subject_id, job_type, output_path),
        ).fetchone()
    return row


def record_download(spec: OutputSpec, subject_id: int, job_id: Optional[int],
                    remote_size: Optional[int]) -> None:
    """Insert (or upsert) a remote_downloads row after a successful pull."""
    local_size = (spec.local_path.stat().st_size
                  if spec.local_path.exists() else None)
    with get_db_ctx() as db:
        db.execute(
            "INSERT INTO remote_downloads "
            "(job_id, subject_id, job_type, output_path, remote_size, local_size) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(subject_id, job_type, output_path) DO UPDATE SET "
            "  job_id=excluded.job_id, "
            "  remote_size=excluded.remote_size, "
            "  local_size=excluded.local_size, "
            "  downloaded_at=CURRENT_TIMESTAMP, "
            "  ignored=0",
            (job_id, subject_id, spec.job_type, str(spec.local_path),
             remote_size, local_size),
        )


def mark_ignored(subject_id: int, job_type: str,
                  output_path: Optional[str] = None) -> None:
    """Mark a result as ignored — banner won't surface it again.

    If ``output_path`` is None, marks every output of (subject, job_type)
    as ignored (also creates rows for outputs we haven't tracked yet)."""
    with get_db_ctx() as db:
        if output_path is not None:
            db.execute(
                "INSERT INTO remote_downloads "
                "(subject_id, job_type, output_path, ignored) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(subject_id, job_type, output_path) DO UPDATE SET "
                "  ignored=1",
                (subject_id, job_type, output_path),
            )
        else:
            db.execute(
                "UPDATE remote_downloads SET ignored=1 "
                "WHERE subject_id=? AND job_type=?",
                (subject_id, job_type),
            )


# ─── Background-thread orchestration for batched downloads ────────────

import threading as _threading
import time as _time
from itertools import count as _count

_download_id_seq = _count(1)
_download_state: dict[int, dict] = {}    # id → progress dict
_download_lock = _threading.Lock()


def _set_download_state(did: int, **kwargs) -> None:
    with _download_lock:
        entry = _download_state.setdefault(did, {})
        entry.update(kwargs)


def get_download_state(active_only: bool = True) -> list[dict]:
    """Snapshot of all download tasks (or active-only) for the UI poll."""
    cutoff = _time.time() - 60     # keep finished entries 60 s for the UI
    with _download_lock:
        out = []
        for did, e in list(_download_state.items()):
            if e.get("status") in ("completed", "failed", "cancelled"):
                if e.get("finished_at") and (_time.time() - e["finished_at"]) > cutoff:
                    _download_state.pop(did, None)
                    continue
            if active_only and e.get("status") not in ("running", "completed", "failed"):
                continue
            out.append({**e, "download_id": did})
        return out


def start_background_download(job_ids: list[int], cfg: RemoteConfig,
                                remote_output_dir: str) -> int:
    """Launch a daemon thread that downloads outputs for ``job_ids`` one at
    a time.  Returns a ``download_id`` the UI can poll for progress."""
    from ..db import get_db_ctx

    download_id = next(_download_id_seq)
    _set_download_state(
        download_id,
        status="running",
        total=len(job_ids),
        completed=0,
        downloaded=0,
        skipped=0,
        current_label="",
        started_at=_time.time(),
        finished_at=None,
        error=None,
    )

    def _worker():
        try:
            with get_db_ctx() as db:
                placeholders = ",".join("?" * len(job_ids))
                jrows = db.execute(
                    f"SELECT j.id, j.job_type, j.subject_id, s.name AS subject_name "
                    f"FROM jobs j LEFT JOIN subjects s ON j.subject_id=s.id "
                    f"WHERE j.id IN ({placeholders})",
                    job_ids,
                ).fetchall()
            for j in jrows:
                if not j.get("subject_id") or not j.get("subject_name"):
                    _set_download_state(
                        download_id,
                        completed=_download_state[download_id]["completed"] + 1,
                    )
                    continue
                _set_download_state(
                    download_id,
                    current_label=f"{j['subject_name']} ({j['job_type']})",
                )
                ok_any = 0
                skip_any = 0
                for spec in output_specs(j["job_type"], j["subject_name"], remote_output_dir):
                    ok = download_one(spec, cfg, j["subject_id"], j["id"])
                    if ok: ok_any += 1
                    else:  skip_any += 1
                _set_download_state(
                    download_id,
                    completed=_download_state[download_id]["completed"] + 1,
                    downloaded=_download_state[download_id]["downloaded"] + ok_any,
                    skipped=_download_state[download_id]["skipped"] + skip_any,
                )
            _set_download_state(
                download_id, status="completed",
                current_label="",
                finished_at=_time.time(),
            )
        except Exception as exc:
            logger.exception(f"Background download {download_id} failed")
            _set_download_state(
                download_id, status="failed",
                error=str(exc), finished_at=_time.time(),
            )

    _threading.Thread(target=_worker, daemon=True,
                       name=f"download-{download_id}").start()
    return download_id


# ─── High-level API ────────────────────────────────────────────────────

def download_one(spec: OutputSpec, cfg: RemoteConfig,
                 subject_id: int, job_id: Optional[int]) -> bool:
    """Pull one output from the remote and record it.  Returns True on
    successful download or when an up-to-date local copy already exists."""
    remote_size = _ssh_stat(cfg, spec.remote_path)
    if remote_size is None:
        return False  # not on remote yet
    existing = already_downloaded(subject_id, spec.job_type, str(spec.local_path))
    if existing and existing.get("ignored"):
        return False  # user said don't pull this
    if (existing and not existing.get("ignored") and
            spec.local_path.exists() and
            existing.get("remote_size") == remote_size and
            spec.local_path.stat().st_size == remote_size):
        return True   # already up to date
    ok = _scp_pull(cfg, spec.remote_path, spec.local_path)
    if ok:
        record_download(spec, subject_id, job_id, remote_size)
        # If we just pulled a legacy combined MediaPipe npz, split it
        # into per-trial files immediately so downstream code sees the
        # new layout without waiting for the next app startup.
        if spec.job_type == "mediapipe" and spec.local_path.name in (
            "mediapipe_prelabels.npz", "mediapipe_reverse_prelabels.npz"
        ):
            try:
                _split_combined_mp_after_download(spec.local_path)
            except Exception as e:
                logger.warning(
                    f"Post-download MP split failed for {spec.local_path}: {e}"
                )
        # If we just pulled a per-trial MP source npz (any of the
        # four passes), rebuild the Combined fusion when at least
        # one OTHER pass already exists for the same trial.  Keeps
        # Combined current without waiting for a manual backfill
        # or a full local re-run.
        _mp_source_files = {
            "mediapipe.npz", "mediapipe_cropped.npz",
            "mediapipe_reverse.npz", "mediapipe_static.npz",
        }
        if spec.job_type == "mediapipe" and spec.local_path.name in _mp_source_files:
            try:
                trial_dir = spec.local_path.parent
                subject_name = trial_dir.parent.name
                trial_stem = trial_dir.name
                present = sum(1 for fn in _mp_source_files
                               if (trial_dir / fn).exists())
                if present >= 2:
                    from .mediapipe_prelabel import build_combined_mp_npz_for_trial
                    build_combined_mp_npz_for_trial(subject_name, trial_stem)
            except Exception as e:
                logger.warning(
                    f"Post-download Combined-MP rebuild failed for "
                    f"{spec.local_path}: {e}"
                )
    return ok


def _split_combined_mp_after_download(combined_path: Path) -> None:
    """Split a freshly-downloaded combined MP npz into per-trial files.

    Mirrors the startup migration in ``app._migrate_combined_mediapipe_npz``
    but scoped to one subject's one file.  Deletes the combined file on
    full success (matches the migration's contract)."""
    import numpy as np
    from .video import build_trial_map
    subject_dir = combined_path.parent
    subject_name = subject_dir.name
    per_trial_name = ("mediapipe_reverse.npz"
                      if "reverse" in combined_path.name
                      else "mediapipe.npz")
    data = np.load(str(combined_path))
    trials = build_trial_map(subject_name)
    if not trials:
        return
    all_written = True
    for trial in trials:
        stem = trial["trial_name"]
        out_dir = subject_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / per_trial_name
        if out_path.exists():
            continue
        sf = int(trial["start_frame"])
        slice_n = int(trial["frame_count"])
        ef = sf + slice_n - 1
        arrays: dict = {}
        for key in ("OS_landmarks", "OD_landmarks",
                    "OS_all_tracks", "OD_all_tracks",
                    "confidence_OS", "confidence_OD", "distances"):
            if key in data.files:
                src = data[key]
                end = min(sf + slice_n, src.shape[0])
                if end <= sf:
                    continue
                arrays[key] = src[sf:end]
        if not arrays:
            continue
        arrays["start_frame"] = sf
        arrays["total_frames"] = slice_n
        np.savez(str(out_path), **arrays)
    if all_written:
        try:
            combined_path.unlink()
        except OSError:
            pass


def pending_for_job(job_id: int, cfg: RemoteConfig,
                     remote_output_dir: str) -> list[dict]:
    """List per-output download status for one job.  Each entry:
        { subject_id, subject_name, job_type, label, remote_path,
          local_path, remote_size, downloaded, ignored, completed }

    Optimisation: skip the (slow) SSH probe for outputs we already have
    locally + recorded in ``remote_downloads`` with matching size.  Only
    probes the remote for the not-yet-downloaded case.  Uses the cheap
    single-probe variant since we only need "roughly available", not
    stability.
    """
    with get_db_ctx() as db:
        job = db.execute(
            "SELECT j.*, s.name AS subject_name FROM jobs j "
            "LEFT JOIN subjects s ON j.subject_id = s.id WHERE j.id=?",
            (job_id,),
        ).fetchone()
    if not job or not job.get("subject_id") or not job.get("subject_name"):
        return []
    out: list[dict] = []
    for spec in output_specs(job["job_type"], job["subject_name"], remote_output_dir):
        existing = already_downloaded(job["subject_id"], spec.job_type,
                                       str(spec.local_path))
        is_ignored = bool(existing and existing.get("ignored"))
        # Fast path: already-downloaded with sizes matching → no SSH.
        local_size = (spec.local_path.stat().st_size
                      if spec.local_path.exists() else None)
        already_local = bool(
            existing and not is_ignored
            and existing.get("local_size") is not None
            and local_size == existing.get("local_size")
        )
        if already_local:
            out.append({
                "subject_id": job["subject_id"],
                "subject_name": job["subject_name"],
                "job_id": job_id,
                "job_type": spec.job_type,
                "label": spec.label,
                "remote_path": spec.remote_path,
                "local_path": str(spec.local_path),
                "remote_size": existing.get("remote_size"),
                "downloaded": True,
                "ignored": False,
                "available": False,
            })
            continue
        # Otherwise, single-probe the remote.
        size = _ssh_stat_quick(cfg, spec.remote_path)
        out.append({
            "subject_id": job["subject_id"],
            "subject_name": job["subject_name"],
            "job_id": job_id,
            "job_type": spec.job_type,
            "label": spec.label,
            "remote_path": spec.remote_path,
            "local_path": str(spec.local_path),
            "remote_size": size,
            "downloaded": False,
            "ignored": is_ignored,
            "available": size is not None and not is_ignored,
        })
    return out
