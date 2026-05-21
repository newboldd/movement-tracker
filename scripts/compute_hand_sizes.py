#!/usr/bin/env python3
"""Compute and store median hand size (mm) per hand for every subject.

Hand size = sum of the median Combined-MP 3D bone lengths over the 21-
landmark MediaPipe hand skeleton, computed separately for each hand's
trials (L/R).  Requires stereo calibration per subject (3D); subjects
without it get NULL.

Going forward this is refreshed automatically whenever Combined-MP is
regenerated (see build_combined_mp_npz_for_trial); this script is the
one-off backfill for existing data.

Usage:
    MT_DATA_DIR=~/data/movement-tracker python3 scripts/compute_hand_sizes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from movement_tracker.db import get_db_ctx, init_db
    from movement_tracker.services.mediapipe_prelabel import update_hand_sizes

    init_db()  # ensure hand_size columns exist
    with get_db_ctx() as db:
        names = [r["name"] for r in
                 db.execute("SELECT name FROM subjects ORDER BY name").fetchall()]

    print(f"Computing hand sizes for {len(names)} subjects…")
    n_l = n_r = 0
    for name in names:
        sizes = update_hand_sizes(name)
        l, r = sizes["left"], sizes["right"]
        if l is not None:
            n_l += 1
        if r is not None:
            n_r += 1
        ls = f"{l:.1f}" if l is not None else "—"
        rs = f"{r:.1f}" if r is not None else "—"
        print(f"  {name:10} L={ls:>7} mm   R={rs:>7} mm")

    print(f"\nStored: {n_l} left, {n_r} right hand sizes "
          f"(missing = no Combined-MP or no calibration).")


if __name__ == "__main__":
    main()
