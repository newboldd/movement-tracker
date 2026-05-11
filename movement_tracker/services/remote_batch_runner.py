#!/usr/bin/env python
"""Remote batch runner — owns a multi-trial HRnet batch on the GPU host.

Long-lived process launched by ``dispatch_remote_batch`` from the local
app.  Reads ``batch.json`` describing the trial list, runs HRnet inference
per trial against the *existing* per-trial work dirs under
``hrnet_jobs/<sub>_<stem>/`` (no duplicate video/MP storage).  Writes
per-trial ``done.flag`` markers under each trial's existing output dir
and a single ``batch_status.json`` heartbeat under ``batch_runs/<id>/``.

Survives local app restarts because it's launched detached
(``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``); the local poller can
disappear and reappear without affecting this loop.

Resume semantics:
    On startup, scans existing ``done.flag`` files and skips any trial
    whose flag exists.  A remote-side reboot or local restart that killed
    the runner can be recovered by simply re-launching this script with
    the same state_dir.

Layout::

    <work>/hrnet_jobs/<sub>_<stem>/         (existing per-trial dir)
        video.mp4                           (already uploaded)
        mediapipe_prelabels.npz             (already uploaded)
        remote_hrnet_script.py              (already uploaded)
        output/<stem>/
            hrnet_w18_heatmaps.npz          (created here on success)
            hand_crop.json
            hrnet_w18_mip.npz
            done.flag                       ("ok" / "failed: ...") — atomic

    <work>/batch_runs/<batch_id>/           (state only — no big files)
        batch.json                          (trial list spec)
        batch_status.json                   (heartbeat)
        runner.log                          (stdout/stderr of this process)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _atomic_write(path: str, content: str) -> None:
    """Write a file atomically — temp file in the same dir + os.replace."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_status(state_dir: str, status: str, n_done: int, n_total: int,
                   current: str | None = None) -> None:
    _atomic_write(
        os.path.join(state_dir, "batch_status.json"),
        json.dumps({
            "status": status,
            "n_done": n_done,
            "n_total": n_total,
            "current": current,
            "updated_at": time.time(),
            "pid": os.getpid(),
        }),
    )


def _existing_outcome(done_flag_path: str) -> str | None:
    """Return contents of an existing done.flag, or None when absent."""
    if not os.path.isfile(done_flag_path):
        return None
    try:
        with open(done_flag_path) as f:
            return f.read().strip() or "ok"
    except OSError:
        return None


def main(state_dir: str) -> int:
    print(f"[batch_runner] starting state_dir={state_dir} pid={os.getpid()}", flush=True)
    spec_path = os.path.join(state_dir, "batch.json")
    if not os.path.isfile(spec_path):
        print(f"[batch_runner] no batch.json at {spec_path}", flush=True)
        return 2
    with open(spec_path) as f:
        spec = json.load(f)

    trials = spec.get("trials") or []
    work_root = spec.get("work_root")           # <remote_work>/hrnet_jobs
    if not work_root:
        print("[batch_runner] missing work_root in batch.json", flush=True)
        _write_status(state_dir, "failed", 0, len(trials))
        return 3

    n_total = len(trials)
    if n_total == 0:
        _write_status(state_dir, "completed", 0, 0)
        return 0

    # Helper to compute the per-trial paths from a trial entry.
    def _paths(t):
        sub = t["subject"]
        stem = t["stem"]                          # "Con04_L1"
        outer = f"{sub}_{stem}"                   # "Con04_Con04_L1"
        trial_dir = os.path.join(work_root, outer)
        out_dir = os.path.join(trial_dir, "output", stem)
        # MP npz is shared per-subject (avoids duplicating the same file
        # in every trial dir).  Newer batches put the path in the trial
        # spec; older batches stored a per-trial copy at <trial>/mp.npz.
        mp_npz = (t.get("mp_path")
                  or os.path.join(trial_dir, "mediapipe_prelabels.npz"))
        return dict(
            sub=sub, stem=stem, trial_dir=trial_dir,
            out_dir=out_dir,
            video=os.path.join(trial_dir, "video.mp4"),
            mp_npz=mp_npz,
            script=os.path.join(trial_dir, "remote_hrnet_script.py"),
            done_flag=os.path.join(out_dir, "done.flag"),
        )

    # Initial heartbeat — count pre-existing done.flags as already done.
    n_done_initial = sum(
        1 for t in trials if _existing_outcome(_paths(t)["done_flag"]) is not None
    )
    _write_status(state_dir, "running", n_done_initial, n_total)
    n_done = n_done_initial

    for trial in trials:
        p = _paths(trial)

        # Cancel-flag check — local can write this to politely stop us.
        if os.path.exists(os.path.join(state_dir, "cancel.flag")):
            print(f"[batch_runner] cancel.flag detected; stopping at {p['stem']}", flush=True)
            _write_status(state_dir, "cancelled", n_done, n_total)
            return 4

        if _existing_outcome(p["done_flag"]) is not None:
            print(f"[batch_runner] skip {p['stem']} (prior outcome present)", flush=True)
            continue

        # Wait for the per-trial files to land.  The dispatcher overlaps
        # uploads with this loop — by the time we get to trial K, trial
        # K+1's video may still be in-flight.  Poll up to UPLOAD_WAIT_S
        # seconds (default 10 min, plenty for a 100 MB video on residential
        # upload speeds) before giving up on this trial.
        UPLOAD_WAIT_S = int(spec.get("upload_wait_s", 600))
        wait_targets = [
            ("video.mp4", p["video"]),
            ("mediapipe_prelabels.npz", p["mp_npz"]),
            ("remote_hrnet_script.py", p["script"]),
        ]
        deadline = time.time() + UPLOAD_WAIT_S
        first_log = True
        missing_after_wait = None
        while time.time() < deadline:
            missing = [name for name, fp in wait_targets if not os.path.isfile(fp)]
            if not missing:
                missing_after_wait = None
                break
            missing_after_wait = missing
            if os.path.exists(os.path.join(state_dir, "cancel.flag")):
                print(f"[batch_runner] cancel during wait at {p['stem']}", flush=True)
                _write_status(state_dir, "cancelled", n_done, n_total)
                return 4
            if first_log:
                print(f"[batch_runner] waiting for upload of {p['stem']} ({', '.join(missing)})", flush=True)
                first_log = False
            time.sleep(5)
        if missing_after_wait:
            msg = f"failed: missing after {UPLOAD_WAIT_S}s wait: {', '.join(missing_after_wait)}"
            print(f"[batch_runner] {p['stem']} {msg}", flush=True)
            _atomic_write(p["done_flag"], msg)
            n_done += 1
            continue
        # Files present — fall through to the existing safety re-checks.
        if not os.path.isfile(p["video"]):
            print(f"[batch_runner] missing video {p['video']}", flush=True)
            _atomic_write(p["done_flag"], "failed: video.mp4 not on remote")
            n_done += 1
            continue
        if not os.path.isfile(p["mp_npz"]):
            print(f"[batch_runner] missing MP npz {p['mp_npz']}", flush=True)
            _atomic_write(p["done_flag"], "failed: mediapipe_prelabels.npz not on remote")
            n_done += 1
            continue
        if not os.path.isfile(p["script"]):
            print(f"[batch_runner] missing script {p['script']}", flush=True)
            _atomic_write(p["done_flag"], "failed: remote_hrnet_script.py not on remote")
            n_done += 1
            continue

        # Import the inference script from THIS trial's dir.  We re-import
        # per trial so each trial uses the script that was uploaded with it
        # (in case they've drifted).  Cheap; the heavy work is GPU inference.
        if p["trial_dir"] not in sys.path:
            sys.path.insert(0, p["trial_dir"])
        try:
            if "remote_hrnet_script" in sys.modules:
                del sys.modules["remote_hrnet_script"]
            import remote_hrnet_script as inf
        except Exception as e:
            print(f"[batch_runner] cannot import remote_hrnet_script for {p['stem']}: {e}", flush=True)
            _atomic_write(p["done_flag"], f"failed: import error: {e}")
            n_done += 1
            continue

        os.makedirs(p["out_dir"], exist_ok=True)
        _write_status(state_dir, "running", n_done, n_total, current=p["stem"])
        print(f"[batch_runner] === {p['stem']} ({n_done+1}/{n_total}) ===", flush=True)

        # Per-trial status file lives in the batch state dir so the local
        # poller can peek at it alongside batch_status.json and report
        # fractional within-trial progress.  remote_hrnet_script.run()
        # writes {"status": "running", "progress_pct": <0..100>} updates
        # to this file each phase.
        per_trial_status = os.path.join(state_dir, "current_trial.json")
        from types import SimpleNamespace
        args = SimpleNamespace(
            video_path=p["video"],
            output_dir=os.path.join(p["trial_dir"], "output"),
            mediapipe_npz=p["mp_npz"],
            trial_name=p["stem"],
            trial_idx=trial.get("trial_idx", 0),
            start_frame=trial.get("start_frame", 0),
            n_frames=trial["n_frames"],
            camera_mode=trial.get("camera_mode", "stereo"),
            weights_url=spec.get("weights_url",
                "https://download.openmmlab.com/mmpose/hand/hrnetv2/"
                "hrnetv2_w18_coco_wholebody_hand_256x256-1c028db7_20210908.pth"),
            weights_path=os.path.join(p["trial_dir"], "output", ".models",
                                       "hrnet_w18_hand_256x256.pth"),
            bbox_os=trial.get("bbox_os"),
            bbox_od=trial.get("bbox_od"),
            status_file=per_trial_status,
            log_file=None,
        )

        try:
            inf.run(args)
            _atomic_write(p["done_flag"], "ok")
            n_done += 1
            print(f"[batch_runner] OK  {p['stem']}", flush=True)
        except Exception as e:
            tb = traceback.format_exc().splitlines()[-1]
            _atomic_write(p["done_flag"], f"failed: {e!s} ({tb})")
            n_done += 1
            print(f"[batch_runner] FAIL {p['stem']}: {e}", flush=True)
            traceback.print_exc()

        _write_status(state_dir, "running", n_done, n_total, current=p["stem"])

    _write_status(state_dir, "completed", n_done, n_total)
    print(f"[batch_runner] done — {n_done}/{n_total}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: remote_batch_runner.py <state_dir>", flush=True)
        sys.exit(2)
    try:
        sys.exit(main(sys.argv[1]))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
