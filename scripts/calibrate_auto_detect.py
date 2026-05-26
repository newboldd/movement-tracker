#!/usr/bin/env python3
"""Measure the residual offset between auto-detected events and the
saved (manually-curated) events, and write per-event integer shifts to
``<MT_DATA_DIR>/auto_detect_calibration.json``.

``auto_detect_from_distance`` reads that file and applies the shifts
AFTER detection so the auto picks land on the same integer frame the
user typically chooses.

Run:

    MT_DATA_DIR=~/data/movement-tracker python3 scripts/calibrate_auto_detect.py

By default the script uses the file at ``<MT_DATA_DIR>``.  Pass
``--print`` to skip writing and just see the numbers.

The auto-detector is invoked with ``calibration_offsets={...zero...}``
so the measurement isn't biased by a previous calibration pass.

Matching tolerance is ``--tol`` frames (default 5).  Median offset
across all matched events per type goes into the JSON.
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
from movement_tracker.services.metrics import auto_detect_from_distance
from movement_tracker.services.mediapipe_prelabel import get_mediapipe_for_session
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
    """Build the [{start_frame, end_frame, ...}, ...] trial list the
    same way the labeling endpoints do."""
    trial_map = build_trial_map(subject)
    if not trial_map:
        return []
    trials = []
    for tm in trial_map:
        trials.append({
            "start_frame": int(tm["start_frame"]),
            "end_frame": int(tm["end_frame"]),
            "trial_name": tm.get("trial_name", ""),
        })
    return trials


def _match(saved: list[int], auto: list[int], tol: int):
    """Greedy 1-to-1 match within ±tol frames; return list of (auto−saved)
    offsets for matched pairs."""
    used = [False] * len(auto)
    offsets = []
    for s in saved:
        best_j = -1
        best_d = tol + 1
        for j, a in enumerate(auto):
            if used[j]:
                continue
            d = a - s
            if abs(d) < abs(best_d) or (abs(d) == abs(best_d) and best_j < 0):
                best_d = d
                best_j = j
        if best_j >= 0 and abs(best_d) <= tol:
            used[best_j] = True
            offsets.append(best_d)
    return offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=5,
                    help="Matching tolerance (frames) for (saved, auto) pairs.")
    ap.add_argument("--print", action="store_true",
                    help="Print the result instead of writing it.")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Limit to specific subject names (default: all).")
    args = ap.parse_args()

    dlc_root = DATA_DIR / "dlc"
    if not dlc_root.is_dir():
        print(f"No dlc directory at {dlc_root}; nothing to calibrate.",
              file=sys.stderr)
        sys.exit(1)

    subjects = args.subjects
    if not subjects:
        subjects = sorted(p.name for p in dlc_root.iterdir()
                           if p.is_dir() and (p / "events.csv").exists())
    print(f"Scanning {len(subjects)} subjects…", file=sys.stderr)

    pooled = {"open": [], "peak": [], "close": []}
    per_subject = {}
    n_match = {"open": 0, "peak": 0, "close": 0}

    for subj in subjects:
        events_path = dlc_root / subj / "events.csv"
        saved = _read_saved_events(events_path)
        if not any(saved.values()):
            continue
        trials = _build_trials(subj)
        if not trials:
            continue
        try:
            mp = get_mediapipe_for_session(subj, prefer_combined=True)
        except Exception as e:
            print(f"  ! {subj}: load failed ({e})", file=sys.stderr)
            continue
        if mp is None or "distances" not in mp:
            continue
        distances = mp.get("distances_clean")
        if distances is None or not np.any(~np.isnan(distances)):
            distances = mp["distances"]
        # ``pre_clean`` is True so we exercise the production path;
        # offsets={0,0,0} so we measure the raw residual.
        auto = auto_detect_from_distance(
            list(distances), trials,
            calibration_offsets={"open": 0, "peak": 0, "close": 0},
        )
        subj_off = {"open": [], "peak": [], "close": []}
        for etype in ("open", "peak", "close"):
            offs = _match(saved.get(etype, []), auto.get(etype, []), args.tol)
            pooled[etype].extend(offs)
            subj_off[etype] = offs
            n_match[etype] += len(offs)
        per_subject[subj] = subj_off

    out = {}
    print("\nResidual offsets (auto − saved):", file=sys.stderr)
    print(f"  {'event':<6} {'matches':>8} {'median':>7} {'mean':>7} {'IQR':>14} {'shift':>6}",
          file=sys.stderr)
    for etype in ("open", "peak", "close"):
        offs = np.asarray(pooled[etype], dtype=int)
        if offs.size == 0:
            shift = 0
            print(f"  {etype:<6} {0:>8} {'-':>7} {'-':>7} {'-':>14} {0:>6}",
                  file=sys.stderr)
        else:
            med = float(np.median(offs))
            mean = float(np.mean(offs))
            q1, q3 = float(np.percentile(offs, 25)), float(np.percentile(offs, 75))
            # Pick the integer shift that MAXIMIZES the at-exact-0 count
            # (mode-based, not median-based).  The median-based rule
            # over-corrects when the distribution is skewed: e.g. closes
            # with offsets concentrated at 0 but a long tail at +2/+3
            # have median +1, but shifting by −1 moves the cluster of
            # already-zero events to −1 and only the small tail to 0.
            counts = {}
            for o in offs:
                counts[int(o)] = counts.get(int(o), 0) + 1
            # Search the small integer range around the median; pick
            # the shift k that puts the most events at offset 0 (i.e.
            # the count at integer −k in the raw distribution).
            best_k, best_count = 0, counts.get(0, 0)
            for k in range(-5, 6):
                c = counts.get(-k, 0)
                if c > best_count:
                    best_count = c
                    best_k = k
            shift = best_k
            print(f"  {etype:<6} {offs.size:>8} {med:>+7.2f} {mean:>+7.2f}"
                  f"  [{q1:+.1f},{q3:+.1f}]  {shift:>+6d}",
                  file=sys.stderr)
        out[etype] = shift

    if args.print:
        print("\n" + json.dumps(out, indent=2))
        return

    cal_path = DATA_DIR / "auto_detect_calibration.json"
    cal_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {cal_path}", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
