#!/usr/bin/env python3
"""Calibrate ``detect_strategy_g`` (the production opens/closes detector)
against the manually-saved events across all subjects.

Pipeline per subject:

  1. Build trial map; load saved events (open / peak / close).
  2. Load combined-MP distances; for each trial, compute (or load
     cached) image metrics — reversal + motion_ssd.
  3. Sweep ``(ssd_search_radius, dist_guard_factor)``; inside each
     combination run ``detect_strategy_g`` with ``open_bias = 0`` and
     ``close_bias = 0``, anchored on the saved peaks; collect the
     per-event (auto − saved) offsets.
  4. For each (radius, guard) combo, the optimal integer bias is
     ``-mode(offsets)``.  Pick the (radius, guard) pair whose optimal
     biases yield the highest exact-match count.

Writes ``<MT_DATA_DIR>/strategy_g_calibration.json`` so the labeler /
calibration script can pick the tuned params up.  ``detect_strategy_g``
itself doesn't read this file directly — the wiring change is small
(read the JSON once on app start, fold its values into ``params``
before calling) but is left for a follow-up so this commit is a pure
measurement.

Run:

    MT_DATA_DIR=~/data/movement-tracker python3 scripts/calibrate_strategy_g.py
    MT_DATA_DIR=~/data/movement-tracker python3 scripts/calibrate_strategy_g.py --print

Use ``--quick`` for a smaller sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from movement_tracker.config import DATA_DIR, get_settings
from movement_tracker.services.metrics import (
    compute_trial_metrics,
    detect_strategy_g,
    get_cached_trial_metrics,
    save_trial_metrics_to_cache,
)
from movement_tracker.services.mediapipe_prelabel import (
    load_mediapipe_combined_prelabels,
    load_mediapipe_prelabels,
)
from movement_tracker.services.video import build_trial_map


# Sweep grids.
RADII = [1, 2, 3, 5, 7, 10]
GUARDS = [0.10, 0.15, 0.25, 0.50, 1.00]
BIAS_RANGE = range(-4, 5)         # -4 .. +4
TOL = 5                            # ±N-frame match tolerance


def _read_saved_events(p: Path) -> dict:
    out = {"open": [], "peak": [], "close": []}
    if not p.exists():
        return out
    with open(p, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() in out:
                try:
                    out[row[0].strip()].append(int(row[1].strip()))
                except ValueError:
                    pass
    for k in out:
        out[k].sort()
    return out


def _build_trials(subject: str) -> list[dict]:
    tm = build_trial_map(subject)
    if not tm:
        return []
    return [{"start_frame": int(t["start_frame"]),
              "end_frame": int(t["end_frame"]),
              "trial_name": t.get("trial_name", "")} for t in tm]


def _match_offsets(saved: list[int], auto: list[int], tol: int = TOL) -> list[int]:
    used = [False] * len(auto)
    offs: list[int] = []
    for s in saved:
        bj, bd = -1, tol + 1
        for j, a in enumerate(auto):
            if used[j]:
                continue
            d = a - s
            if abs(d) < abs(bd):
                bd, bj = d, j
        if bj >= 0 and abs(bd) <= tol:
            used[bj] = True
            offs.append(bd)
    return offs


def _best_bias(offsets: list[int]) -> tuple[int, int]:
    """Return (best_bias, exact_count_with_that_bias).  best_bias = -mode."""
    if not offsets:
        return 0, 0
    counts = Counter(offsets)
    # The bias that puts the most events at exactly 0 is ``-mode``.
    mode, mode_count = counts.most_common(1)[0]
    return -int(mode), int(mode_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="Don't write JSON.")
    ap.add_argument("--quick", action="store_true",
                    help="Quick sweep: just default radius/guard, sweep biases only.")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    settings = get_settings()
    cam_names = settings.camera_names
    dlc_root = DATA_DIR / "dlc"
    if not dlc_root.is_dir():
        print(f"No dlc dir at {dlc_root}.", file=sys.stderr)
        sys.exit(1)

    radii = [5] if args.quick else RADII
    guards = [0.25] if args.quick else GUARDS

    subjects = args.subjects or sorted(
        p.name for p in dlc_root.iterdir()
        if p.is_dir() and (p / "events.csv").exists()
    )
    print(f"Scanning {len(subjects)} subjects (radii {radii}, guards {guards})…",
          file=sys.stderr)

    # ── Phase 1: gather per-trial inputs (distances, saved peaks, metrics) ──
    trial_inputs: list[dict] = []
    for subj in subjects:
        saved = _read_saved_events(dlc_root / subj / "events.csv")
        if not saved["peak"] and not (saved["open"] or saved["close"]):
            continue
        trials = _build_trials(subj)
        if not trials:
            continue
        data = load_mediapipe_combined_prelabels(subj) or load_mediapipe_prelabels(subj)
        if data is None:
            continue
        distances = data.get("distances_clean")
        if distances is None or not np.any(~np.isnan(distances)):
            distances = data.get("distances")
        if distances is None:
            continue
        distances = list(distances)
        for tm in trials:
            sf, ef = tm["start_frame"], tm["end_frame"]
            trial_name = tm["trial_name"]
            cm = get_cached_trial_metrics(subj, trial_name)
            if cm is None:
                try:
                    cm = compute_trial_metrics(subj, tm, cam_names)
                    if cm:
                        save_trial_metrics_to_cache(subj, trial_name, cm)
                except Exception as e:
                    print(f"  ! {subj}/{trial_name}: metrics failed ({e})",
                          file=sys.stderr)
                    cm = None
            saved_peaks_local = [p - sf for p in saved["peak"] if sf <= p <= ef]
            saved_opens = [o for o in saved["open"] if sf <= o <= ef]
            saved_closes = [c for c in saved["close"] if sf <= c <= ef]
            trial_inputs.append({
                "subject": subj,
                "trial_name": trial_name,
                "sf": sf, "ef": ef,
                "trial_dist": distances[sf:ef + 1],
                "saved_peaks_local": saved_peaks_local,
                "saved_opens_global": saved_opens,
                "saved_closes_global": saved_closes,
                "reversal": (cm.get("reversal") if cm else None),
                # Prefer phase-correlation flow over raw MSD; fall back
                # to SSD for legacy cache entries that predate flow.
                "motion_ssd": ((cm.get("motion_flow") or cm.get("motion_ssd")) if cm else None),
                "per_cam_ssd": ((cm.get("per_cam_flow") or cm.get("per_cam_ssd")) if cm else None),
            })
    print(f"  collected {len(trial_inputs)} trials", file=sys.stderr)

    # ── Phase 2: sweep (radius, guard); within each, optimal biases come
    # from the offset histogram (running detect_strategy_g with bias 0).
    print(f"\n{'radius':>6} {'guard':>5}  {'open_b':>6} {'open_exact':>11}"
          f" {'close_b':>7} {'close_exact':>12}  {'sum_exact':>9}",
          file=sys.stderr)
    best = {
        "sum_exact": -1,
        "ssd_search_radius": 5, "dist_guard_factor": 0.25,
        "open_bias": 0, "close_bias": 0,
        "open_exact": 0, "close_exact": 0,
    }
    open_total = sum(len(t["saved_opens_global"]) for t in trial_inputs)
    close_total = sum(len(t["saved_closes_global"]) for t in trial_inputs)
    for r in radii:
        for g in guards:
            open_offs: list[int] = []
            close_offs: list[int] = []
            for t in trial_inputs:
                try:
                    det = detect_strategy_g(
                        t["trial_dist"],
                        t["reversal"], t["motion_ssd"], t["per_cam_ssd"],
                        cam_names, start_frame=t["sf"],
                        steps={
                            "use_reversal": t["reversal"] is not None,
                            "use_ssd": t["motion_ssd"] is not None,
                            "use_dist_guard": True,
                            "use_peak_guard": True,
                        },
                        params={
                            "ssd_search_radius": r,
                            "dist_guard_factor": g,
                            "open_bias": 0, "close_bias": 0,
                        },
                        peaks_only=False,
                        existing_peaks=(t["saved_peaks_local"] or None),
                    )
                except Exception:
                    continue
                open_offs += _match_offsets(t["saved_opens_global"], det.get("open", []), TOL)
                close_offs += _match_offsets(t["saved_closes_global"], det.get("close", []), TOL)
            ob, oe = _best_bias(open_offs)
            cb, ce = _best_bias(close_offs)
            tot = oe + ce
            print(f"{r:>6} {g:>5.2f}  {ob:>+6d} {oe:>5d} ({100*oe/max(1,open_total):4.1f}%)"
                  f" {cb:>+7d} {ce:>5d} ({100*ce/max(1,close_total):4.1f}%)"
                  f"  {tot:>9d}", file=sys.stderr)
            if tot > best["sum_exact"]:
                best.update({
                    "sum_exact": tot,
                    "ssd_search_radius": r, "dist_guard_factor": g,
                    "open_bias": ob, "close_bias": cb,
                    "open_exact": oe, "close_exact": ce,
                })

    # ── Phase 3: write the calibration ──
    out = {
        "ssd_search_radius": best["ssd_search_radius"],
        "dist_guard_factor": best["dist_guard_factor"],
        "open_bias": best["open_bias"],
        "close_bias": best["close_bias"],
        "_measured": {
            "saved_opens": open_total,
            "saved_closes": close_total,
            "open_exact": best["open_exact"],
            "close_exact": best["close_exact"],
        },
    }
    print(f"\nBEST: r={best['ssd_search_radius']} g={best['dist_guard_factor']:.2f}"
          f" open_bias={best['open_bias']:+d} close_bias={best['close_bias']:+d}"
          f"  → open {best['open_exact']}/{open_total} "
          f"({100*best['open_exact']/max(1,open_total):.1f}%), "
          f"close {best['close_exact']}/{close_total} "
          f"({100*best['close_exact']/max(1,close_total):.1f}%)",
          file=sys.stderr)

    if args.print:
        print(json.dumps(out, indent=2))
        return
    path = DATA_DIR / "strategy_g_calibration.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
