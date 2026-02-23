"""
Import existing DLC labels into the dlc-labeler SQLite database.

Reads label_metadata.json (frame_num/trial_idx/side mapping) and
CollectedData_labels.csv (keypoint coordinates) from each DLC project's
labeled-data directories, then creates committed label sessions in the DB.

Usage:
    python -m dlc_app.import_labels              # all projects
    python -m dlc_app.import_labels MSA18 Con01   # specific projects
    python -m dlc_app.import_labels --dry-run     # preview without writing

Requires label_metadata.json to exist in each labeled-data subdir.
Generate it with: scratch/generate_label_metadata.py (in the main project).
"""

import csv
import json
import sys
from pathlib import Path

from .config import get_settings
from .db import get_db, init_db
from .services.discovery import scan_all_subjects, infer_stage


def parse_collected_data_csv(csv_path: Path) -> dict:
    """Parse DLC CollectedData_labels.csv into {img_filename: {bodypart: [x, y]}}.

    DLC multi-header format:
      Row 0: scorer, , , scorer_name, ...
      Row 1: bodyparts, , , bp1, bp1, bp2, bp2, ...
      Row 2: coords, , , x, y, x, y, ...
      Data:  labeled-data, subdir, img.png, x1, y1, x2, y2, ...
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 4:
        return {}

    # Parse header to get bodypart names
    bp_row = rows[1]  # bodyparts row
    # Skip first 3 columns (labeled-data, subdir, filename)
    bodyparts = []
    for i in range(3, len(bp_row), 2):
        if i < len(bp_row):
            bodyparts.append(bp_row[i])

    # Parse data rows
    result = {}
    for row in rows[3:]:
        if len(row) < 4:
            continue
        img_filename = row[2]
        keypoints = {}
        for j, bp in enumerate(bodyparts):
            x_idx = 3 + j * 2
            y_idx = 3 + j * 2 + 1
            if x_idx < len(row) and y_idx < len(row):
                try:
                    x = float(row[x_idx]) if row[x_idx] else None
                    y = float(row[y_idx]) if row[y_idx] else None
                    if x is not None and y is not None:
                        keypoints[bp] = [x, y]
                except ValueError:
                    pass
        result[img_filename] = keypoints

    return result


def import_project(db, subject_name: str, dlc_path: Path, dry_run: bool = False) -> dict:
    """Import all labeled-data subdirs for a single DLC project.

    Combines labels from all subdirs into a single committed session.
    Later subdirs (alphabetically) override earlier ones at the same
    (frame_num, trial_idx, side) position.

    Returns dict with counts.
    """
    labeled_dir = dlc_path / "labeled-data"
    if not labeled_dir.exists():
        return {"skipped": True, "reason": "no labeled-data dir"}

    # Find subdirs with both label_metadata.json and CollectedData_labels.csv
    importable = []
    for subdir in sorted(labeled_dir.iterdir()):
        if not subdir.is_dir():
            continue
        meta_path = subdir / "label_metadata.json"
        csv_path = subdir / "CollectedData_labels.csv"
        if meta_path.exists() and csv_path.exists():
            importable.append(subdir)

    if not importable:
        return {"skipped": True, "reason": "no label_metadata.json found"}

    # Ensure subject exists in DB
    subj = db.execute(
        "SELECT * FROM subjects WHERE name = ?", (subject_name,)
    ).fetchone()

    if not subj:
        stage = infer_stage(dlc_path)
        if not dry_run:
            db.execute(
                """INSERT INTO subjects (name, stage, dlc_dir)
                   VALUES (?, ?, ?)""",
                (subject_name, stage, subject_name),
            )
            subj = db.execute(
                "SELECT * FROM subjects WHERE name = ?", (subject_name,)
            ).fetchone()
        else:
            print("  [DRY-RUN] Would create subject: %s (stage=%s)" % (subject_name, stage))

    # Collect all labels across subdirs, keyed by (frame_num, trial_idx, side)
    # Later subdirs override earlier ones (e.g., round2 overrides round1)
    all_labels = {}  # (frame_num, trial_idx, side) -> keypoints_json

    for subdir in importable:
        meta_path = subdir / "label_metadata.json"
        csv_path = subdir / "CollectedData_labels.csv"

        with open(meta_path) as f:
            metadata = json.load(f)
        keypoints_map = parse_collected_data_csv(csv_path)

        n_merged = 0
        for img_name, meta in metadata.items():
            kp = keypoints_map.get(img_name, {})
            if not kp:
                continue
            key = (meta["frame_num"], meta["trial_idx"], meta["side"])
            all_labels[key] = json.dumps(kp)
            n_merged += 1

        print("  %s: %d labels" % (subdir.name, n_merged))

    if not all_labels:
        return {"skipped": True, "reason": "no labels with both metadata and keypoints"}

    if dry_run:
        print("  [DRY-RUN] Would import %d unique labels total" % len(all_labels))
        return {"dirs": len(importable), "labels": len(all_labels)}

    # Check for existing committed session to avoid duplicates
    existing = db.execute(
        """SELECT ls.id, COUNT(fl.id) as label_count
           FROM label_sessions ls
           LEFT JOIN frame_labels fl ON fl.session_id = ls.id
           WHERE ls.subject_id = ? AND ls.status = 'committed'
           GROUP BY ls.id
           ORDER BY ls.committed_at DESC LIMIT 1""",
        (subj["id"],),
    ).fetchone()

    if existing and existing["label_count"] > 0:
        print(
            "  Skipping — already has %d committed labels (session %d)"
            % (existing["label_count"], existing["id"])
        )
        return {"dirs": 0, "labels": 0}

    # Create single committed session with all labels
    db.execute(
        """INSERT INTO label_sessions
           (subject_id, iteration, session_type, status, committed_at)
           VALUES (?, 1, 'initial', 'committed', CURRENT_TIMESTAMP)""",
        (subj["id"],),
    )
    session = db.execute(
        "SELECT * FROM label_sessions WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
        (subj["id"],),
    ).fetchone()

    for (frame_num, trial_idx, side), kp_json in all_labels.items():
        db.execute(
            """INSERT OR REPLACE INTO frame_labels
               (session_id, frame_num, trial_idx, side, keypoints, updated_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (session["id"], frame_num, trial_idx, side, kp_json),
        )

    print(
        "  -> imported %d labels into session %d"
        % (len(all_labels), session["id"])
    )
    return {"dirs": len(importable), "labels": len(all_labels)}


def main():
    dry_run = "--dry-run" in sys.argv
    requested = set(a for a in sys.argv[1:] if not a.startswith("-"))

    # Initialize DB
    init_db()
    db = get_db()

    settings = get_settings()
    dlc_dir = settings.dlc_path

    if not dlc_dir.exists():
        print("ERROR: DLC directory not found: %s" % dlc_dir)
        sys.exit(1)

    print("DLC dir: %s" % dlc_dir)
    print("Video dir: %s" % settings.video_path)
    if dry_run:
        print("MODE: dry-run (no changes will be made)\n")
    else:
        print()

    total_labels = 0
    total_projects = 0

    for entry in sorted(dlc_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "config.yaml").exists():
            continue
        name = entry.name
        if requested and name not in requested:
            continue

        # Skip label set bundles
        if "-labels-" in name:
            continue

        print("=== %s ===" % name)
        result = import_project(db, name, entry, dry_run=dry_run)

        if result.get("skipped"):
            print("  Skipped: %s" % result["reason"])
        else:
            total_labels += result.get("labels", 0)
            total_projects += 1 if result.get("labels", 0) > 0 else 0

    if not dry_run:
        db.commit()
    db.close()

    print("\n" + "=" * 50)
    print("  Imported %d labels across %d projects" % (total_labels, total_projects))
    if dry_run:
        print("  (DRY-RUN — no changes were made)")
    print("=" * 50)


if __name__ == "__main__":
    main()
