"""Re-export committed labels to DLC format after frame alignment fix.

Regenerates PNGs, QC overlays, label_metadata.json, and CollectedData CSV
for each affected subject's latest committed initial session.

Run from the dlc-labeler repo root:
    python scratch/reexport_labels.py [--subjects Con09 MSA14 ...]
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dlc_app.config import get_settings
from dlc_app.services.labels import commit_labels_to_dlc

DB_PATH = Path("dlc_app/dlc_app.db")

# All subjects that had frame_num corrections applied
ALL_AFFECTED = [
    "Con03", "Con04", "Con05", "Con07", "Con08", "Con09",
    "MSA04", "MSA07", "MSA09", "MSA10", "MSA11", "MSA12",
    "MSA14", "MSA15", "MSA16", "MSA17", "MSA18",
    "PD01", "PD02", "PD03", "PD04", "PD05", "PD06", "PD07", "PD08", "PD09", "PD14",
    "PSP01", "PSP02",
]


def main():
    parser = argparse.ArgumentParser(description="Re-export committed labels to DLC")
    parser.add_argument("--subjects", nargs="*", help="Subjects to re-export (default: all affected)")
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else ALL_AFFECTED

    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    for subj in subjects:
        # Find the latest committed initial session
        session = db.execute("""
            SELECT ls.id
            FROM label_sessions ls
            JOIN subjects s ON ls.subject_id = s.id
            WHERE s.name = ? AND ls.session_type = 'initial' AND ls.status = 'committed'
            ORDER BY ls.id DESC
            LIMIT 1
        """, (subj,)).fetchone()

        if not session:
            print(f"{subj}: no committed initial session found, skipping")
            continue

        session_id = session["id"]

        # Get all labels for this session
        labels = db.execute("""
            SELECT frame_num, trial_idx, side, keypoints
            FROM frame_labels
            WHERE session_id = ?
            ORDER BY trial_idx, frame_num, side
        """, (session_id,)).fetchall()

        if not labels:
            print(f"{subj}: session {session_id} has no labels, skipping")
            continue

        # Convert to list of dicts
        session_labels = []
        for row in labels:
            kp = row["keypoints"]
            if isinstance(kp, str):
                kp = json.loads(kp)
            session_labels.append({
                "frame_num": row["frame_num"],
                "trial_idx": row["trial_idx"],
                "side": row["side"],
                "keypoints": kp,
            })

        t0 = time.time()
        try:
            result = commit_labels_to_dlc(subj, session_labels, iteration=1)
            dt = time.time() - t0
            print(f"{subj}: re-exported {result['frame_count']} frames from session {session_id} ({dt:.1f}s)")
        except Exception as e:
            print(f"{subj}: ERROR re-exporting session {session_id}: {e}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
