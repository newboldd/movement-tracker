#!/usr/bin/env python3
"""Measure the residual offset between auto-detected events and the
saved (manually-curated) events, and write the calibration JSON read
by ``detect_events_from_peaks`` / ``auto_detect_from_distance`` at
``<MT_DATA_DIR>/auto_detect_calibration.json``.

Two phases:

1. Run ``detect_events_from_peaks`` with the user's saved peaks as
   anchors (matching the production workflow — peaks are manually
   corrected before opens / closes are detected) and with calibration
   *disabled*, so the measurements reflect the raw algorithm.

2. For each event type, pick the integer shift that MAXIMIZES the
   exact-match count.  For CLOSE we additionally sweep a velocity
   veto threshold; the wider shift + veto combination that pays off
   on more events wins.

Output JSON (richer than the older bare-int format):

    {
      "open":  {"shift": -1},
      "peak":  {"shift": 0},
      "close": {"shift": -3, "veto": {"side": "next", "thresh": 0.60}}
    }

The loader (``_load_calibration_offsets``) also accepts the legacy
``{"open": -2, "peak": 0, "close": -1}`` form for backwards compat.

Run:

    MT_DATA_DIR=~/data/movement-tracker python3 scripts/calibrate_auto_detect.py

Options:
    --tol N      Matching tolerance for (saved, auto) pairs (default 5).
    --print      Print the result instead of writing it.
    --subjects   Whitespace-separated subject names; otherwise all with events.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from movement_tracker.config import DATA_DIR
from movement_tracker.services.calibration import get_calibration_for_subject
from movement_tracker.services.metrics import (
    detect_events_from_peaks,
    build_joint_3d_trace,
    velocity_magnitude_3d,
    INDEX_TIP,
)
from movement_tracker.services.mediapipe_prelabel import (
    get_mediapipe_for_session,
    load_mediapipe_combined_prelabels,
    load_mediapipe_prelabels,
)
from movement_tracker.services.video import build_trial_map


def _read_saved_events(path: Path) -> dict:
    out = {"open": [], "peak": [], "close": []}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            etype, frame = row[0].strip(), row[1].strip()
            if etype in out:
                try:
                    out[etype].append(int(frame))
                except ValueError:
                    pass
    for k in out:
        out[k].sort()
    return out


def _build_trials(subject: str) -> list[dict]:
    trial_map = build_trial_map(subject)
    if not trial_map:
        return []
    return [{"start_frame": int(t["start_frame"]),
              "end_frame": int(t["end_frame"]),
              "trial_name": t.get("trial_name", "")} for t in trial_map]


def _match(saved: list[int], auto: list[int], tol: int):
    used = [False] * len(auto)
    offsets = []
    for s in saved:
        best_j, best_d = -1, tol + 1
        for j, a in enumerate(auto):
            if used[j]:
                continue
            d = a - s
            if abs(d) < abs(best_d):
                best_d = d
                best_j = j
        if best_j >= 0 and abs(best_d) <= tol:
            used[best_j] = True
            offsets.append(best_d)
    return offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    dlc_root = DATA_DIR / "dlc"
    if not dlc_root.is_dir():
        print(f"No dlc directory at {dlc_root}.", file=sys.stderr)
        sys.exit(1)

    subjects = args.subjects
    if not subjects:
        subjects = sorted(p.name for p in dlc_root.iterdir()
                           if p.is_dir() and (p / "events.csv").exists())
    print(f"Scanning {len(subjects)} subjects…", file=sys.stderr)

    # Collect per-event (saved_frame, auto_frame_unshifted, v_index_full)
    records = {"open": [], "close": []}
    n_subj = 0
    for subj in subjects:
        events_path = dlc_root / subj / "events.csv"
        saved = _read_saved_events(events_path)
        if not (saved["peak"] and (saved["open"] or saved["close"])):
            continue
        trials = _build_trials(subj)
        if not trials:
            continue
        # Load raw arrays directly (the labeler-facing helper aliases
        # them as thumb/index dicts; we need full 21-joint arrays).
        data = load_mediapipe_combined_prelabels(subj)
        if data is None:
            data = load_mediapipe_prelabels(subj)
        if data is None:
            continue
        distances = data.get("distances_clean")
        if distances is None or not np.any(~np.isnan(distances)):
            distances = data.get("distances")
        if distances is None:
            continue
        OS = data.get("OS_landmarks")
        OD = data.get("OD_landmarks")
        calib = None
        try:
            calib = get_calibration_for_subject(subj)
        except Exception:
            calib = None
        # Run detection with calibration disabled (zero shifts, no veto)
        # so the offsets we measure reflect the raw algorithm.
        auto = detect_events_from_peaks(
            list(distances), trials, saved["peak"],
            OS_landmarks=OS, OD_landmarks=OD, calib=calib,
            calibration={"open": 0, "peak": 0, "close": 0},
        )
        # Per-event records.  We also need v_index for the close sweep
        # so the adaptive veto can be evaluated post-hoc.
        v_index = None
        if OS is not None and OD is not None and calib is not None:
            try:
                idx3d = build_joint_3d_trace(OS, OD, INDEX_TIP, calib)
                v_index = velocity_magnitude_3d(idx3d, smooth_sigma=1.0)
            except Exception:
                v_index = None
        for etype in ("open", "close"):
            pairs = _match(saved[etype], auto[etype], args.tol)
            # We need (saved, auto) pairs not just offsets to evaluate
            # shifts — recover them via a parallel match.
            used = [False] * len(auto[etype])
            for s in saved[etype]:
                bj, bd = -1, args.tol + 1
                for j, a in enumerate(auto[etype]):
                    if used[j]: continue
                    d = a - s
                    if abs(d) < abs(bd):
                        bd = d; bj = j
                if bj >= 0 and abs(bd) <= args.tol:
                    used[bj] = True
                    records[etype].append({
                        "subject": subj,
                        "saved": int(s),
                        "auto": int(auto[etype][bj]),
                        "v": v_index,
                    })
        n_subj += 1

    print(f"\nUsable subjects: {n_subj}", file=sys.stderr)
    print(f"Open pairs: {len(records['open'])}  Close pairs: {len(records['close'])}\n",
          file=sys.stderr)

    def _ratio(v, f, side, window=5):
        if v is None: return np.nan
        n = len(v)
        if f < 1 or f >= n - 1: return np.nan
        fq = f - 1 if side == "prior" else f + 1
        if fq < 0 or fq >= n: return np.nan
        lo = max(0, f - window); hi = min(n, f + window + 1)
        lm = np.nanmax(v[lo:hi]) if hi > lo else np.nan
        if not np.isfinite(lm) or lm < 1.0: return np.nan
        vq = v[fq]
        if not np.isfinite(vq): return np.nan
        return float(vq / lm)

    def _exact_count(etype, shift, thresh=None):
        side = "prior" if etype == "open" else "next"
        n = 0
        for r in records[etype]:
            cand = r["auto"] + shift
            use_shift = True
            if thresh is not None:
                rr = _ratio(r["v"], cand, side)
                if np.isfinite(rr) and rr > thresh:
                    use_shift = False
            chosen = cand if use_shift else r["auto"]
            if chosen == r["saved"]:
                n += 1
        return n

    out = {}

    # OPEN — sweep shift only (no veto; the velocity-onset detector
    # plus a global shift already does well).
    print("OPEN  (no veto):", file=sys.stderr)
    best_shift, best_n = 0, _exact_count("open", 0)
    for s in range(-4, 5):
        c = _exact_count("open", s)
        marker = " *" if c > best_n else ""
        print(f"  shift {s:+d}: exact {c:5d}{marker}", file=sys.stderr)
        if c > best_n:
            best_shift, best_n = s, c
    out["open"] = {"shift": int(best_shift)}
    print(f"  → open shift = {best_shift:+d}  (exact {best_n})\n", file=sys.stderr)

    # CLOSE — sweep (shift, thresh).
    print("CLOSE  (veto on next-frame v_index ratio):", file=sys.stderr)
    print(f"  {'shift':>6}  " + "  ".join(f"t={t:.2f}" for t in
          [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]) + "  no-veto",
          file=sys.stderr)
    best = {"shift": 0, "thresh": None, "exact": _exact_count("close", 0)}
    for shift in range(-4, 4):
        row = [f"  {shift:+d}    "]
        for thresh in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, None):
            c = _exact_count("close", shift, thresh)
            row.append(f"{c:5d}")
            if c > best["exact"]:
                best = {"shift": shift, "thresh": thresh, "exact": c}
        print("  ".join(row), file=sys.stderr)

    close_entry: dict = {"shift": int(best["shift"])}
    if best["thresh"] is not None and best["exact"] > _exact_count("close", best["shift"], None):
        close_entry["veto"] = {"side": "next", "thresh": float(best["thresh"])}
        print(f"  → close shift = {best['shift']:+d}, veto thresh = "
              f"{best['thresh']:.2f}  (exact {best['exact']})\n", file=sys.stderr)
    else:
        print(f"  → close shift = {best['shift']:+d}  no veto  "
              f"(exact {best['exact']})\n", file=sys.stderr)
    out["close"] = close_entry

    # PEAK — saved peaks pass through unmodified.
    out["peak"] = {"shift": 0}

    if args.print:
        print(json.dumps(out, indent=2))
        return
    cal_path = DATA_DIR / "auto_detect_calibration.json"
    cal_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {cal_path}", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
