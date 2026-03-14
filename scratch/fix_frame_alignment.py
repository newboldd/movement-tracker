"""Fix frame_num alignment for multi-trial labels after frame-counting change.

After commit fe5f793 (March 1, 2026) switched from CAP_PROP_FRAME_COUNT to
actual decoded frame counts, trial start offsets changed. Labels created before
the fix have frame_num values computed with the old (inflated) offsets. This
script recalculates them to match the new (actual) offsets.

For each affected label: new_frame_num = frame_num - cumulative_offset[trial_idx]
where cumulative_offset = old_start[trial] - new_start[trial]

Run from the dlc-labeler repo root:
    python scratch/fix_frame_alignment.py [--dry-run]
"""
import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from dlc_app.services.video import get_subject_videos, _get_actual_frame_count

DB_PATH = Path("dlc_app/dlc_app.db")
# Frame-counting fix was committed at 2026-03-01 22:57:23.
# Use midnight March 2 as cutoff — all sessions created before this definitely
# used old CAP_PROP_FRAME_COUNT offsets.
CUTOFF = "2026-03-02 00:00:00"


def compute_offsets(subject_name: str) -> dict[int, int]:
    """Compute per-trial cumulative offset (old_start - new_start).

    Returns {trial_idx: offset} for trials where offset != 0.
    """
    videos = get_subject_videos(subject_name)
    if not videos:
        return {}

    old_start = 0
    new_start = 0
    offsets = {}

    for i, v in enumerate(videos):
        cap = cv2.VideoCapture(v)
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        actual = _get_actual_frame_count(v)
        cum = old_start - new_start
        if i > 0 and cum != 0:
            offsets[i] = cum
        old_start += reported
        new_start += actual

    return offsets


def main():
    parser = argparse.ArgumentParser(description="Fix multi-trial frame alignment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying DB")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return 1

    # Backup database
    if not args.dry_run:
        backup = DB_PATH.with_suffix(f".db.bak-frame-fix-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(DB_PATH, backup)
        print(f"Database backed up to {backup}")

    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    # Find all subjects with labels in trial_idx > 0
    subjects = db.execute("""
        SELECT DISTINCT s.name
        FROM frame_labels fl
        JOIN label_sessions ls ON fl.session_id = ls.id
        JOIN subjects s ON ls.subject_id = s.id
        WHERE fl.trial_idx > 0
        ORDER BY s.name
    """).fetchall()
    subjects = [r["name"] for r in subjects]

    total_fixed = 0
    total_skipped = 0

    for subj in subjects:
        offsets = compute_offsets(subj)
        if not offsets:
            continue

        # Get all committed pre-fix sessions' frame_nums as reference set
        pre_fix_frames = set()
        pre_fix_rows = db.execute("""
            SELECT DISTINCT fl.frame_num, fl.trial_idx, fl.side
            FROM frame_labels fl
            JOIN label_sessions ls ON fl.session_id = ls.id
            JOIN subjects s ON ls.subject_id = s.id
            WHERE s.name = ? AND fl.trial_idx > 0
              AND ls.created_at < ?
        """, (subj, CUTOFF)).fetchall()
        for r in pre_fix_rows:
            pre_fix_frames.add((r["frame_num"], r["trial_idx"], r["side"]))

        # Get ALL labels for this subject with trial_idx > 0
        labels = db.execute("""
            SELECT fl.id, fl.frame_num, fl.trial_idx, fl.side,
                   ls.id as session_id, ls.created_at, ls.session_type
            FROM frame_labels fl
            JOIN label_sessions ls ON fl.session_id = ls.id
            JOIN subjects s ON ls.subject_id = s.id
            WHERE s.name = ? AND fl.trial_idx > 0
            ORDER BY fl.trial_idx, fl.frame_num
        """, (subj,)).fetchall()

        subj_fixed = 0
        subj_skipped = 0

        for label in labels:
            trial_idx = label["trial_idx"]
            offset = offsets.get(trial_idx, 0)
            if offset == 0:
                continue

            frame_num = label["frame_num"]
            created_at = label["created_at"]
            is_pre_fix = created_at < CUTOFF

            # Decide whether to apply correction
            if is_pre_fix:
                # Definitely created with old offsets
                apply = True
            else:
                # Post-fix session: only apply if this frame_num exists in
                # a pre-fix session (inherited/copied label)
                key = (frame_num, trial_idx, label["side"])
                apply = key in pre_fix_frames

            if not apply:
                if args.dry_run:
                    print(f"  SKIP {subj} id={label['id']} session={label['session_id']} "
                          f"type={label['session_type']} trial={trial_idx} "
                          f"frame={frame_num} (post-fix, new position)")
                subj_skipped += 1
                continue

            new_frame_num = frame_num - offset

            if args.dry_run:
                print(f"  FIX  {subj} id={label['id']} trial={trial_idx} "
                      f"frame={frame_num} -> {new_frame_num} (offset={offset})")
            else:
                db.execute(
                    "UPDATE frame_labels SET frame_num = ? WHERE id = ?",
                    (new_frame_num, label["id"]),
                )
            subj_fixed += 1

        if subj_fixed or subj_skipped:
            action = "Would fix" if args.dry_run else "Fixed"
            parts = [f"{action} {subj_fixed} labels"]
            if subj_skipped:
                parts.append(f"skipped {subj_skipped}")
            offset_str = ", ".join(f"t{t}:{o}" for t, o in sorted(offsets.items()))
            parts.append(f"offsets: {offset_str}")
            print(f"{subj}: {', '.join(parts)}")

        total_fixed += subj_fixed
        total_skipped += subj_skipped

    if not args.dry_run:
        db.commit()

    db.close()
    action = "Would fix" if args.dry_run else "Fixed"
    print(f"\nDone. {action} {total_fixed} labels, skipped {total_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
