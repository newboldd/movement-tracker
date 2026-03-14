"""Generate QC overlay images for all subjects with committed labels.

For each labeled-data/{round}/ that has a CollectedData_labels.csv:
  - Extracts PNGs from source videos if they don't exist yet
  - Draws keypoint crosses and saves to labeled-data/{round}/qc/
  - Skips rounds that already have a complete qc/ directory

Run from the dlc-labeler repo root:
    python scratch/generate_qc_overlays.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dlc_app.config import get_settings
from dlc_app.services.video import build_trial_map, _resolve_frame, extract_frame_raw

_BP_COLORS = [
    (0, 0, 255),    # red (BGR)
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]
ARM = 12
THICKNESS = 2


def _parse_collected_data_csv(csv_path: Path) -> dict:
    """Parse CollectedData CSV -> {img_filename: {bodypart: [x, y]}}."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 4:
        return {}

    bodypart_row = rows[1]
    coord_row = rows[2]

    col_map = {}
    for col_idx in range(3, len(bodypart_row)):
        bp = bodypart_row[col_idx]
        coord = coord_row[col_idx]
        col_map.setdefault(bp, {})[coord] = col_idx

    result = {}
    for row in rows[3:]:
        img_filename = row[2]
        kp = {}
        for bp, cols in col_map.items():
            if "x" not in cols or "y" not in cols:
                continue
            try:
                x = float(row[cols["x"]])
                y = float(row[cols["y"]])
                kp[bp] = [x, y]
            except (IndexError, ValueError):
                pass
        result[img_filename] = kp

    return result


def _draw_overlay(frame: np.ndarray, keypoints: dict, bodyparts: list) -> np.ndarray:
    img = frame.copy()
    for i, bp in enumerate(bodyparts):
        coords = keypoints.get(bp)
        if not coords:
            continue
        x, y = int(round(coords[0])), int(round(coords[1]))
        color = _BP_COLORS[i % len(_BP_COLORS)]
        cv2.line(img, (x - ARM, y - ARM), (x + ARM, y + ARM), color, THICKNESS)
        cv2.line(img, (x - ARM, y + ARM), (x + ARM, y - ARM), color, THICKNESS)
        cv2.putText(img, bp, (x + ARM + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def _build_reported_offsets(trials: list) -> list:
    """Return old-style trial start offsets using CAP_PROP_FRAME_COUNT.

    Old metadata frame_nums were computed using CAP_PROP_FRAME_COUNT (which
    inflates frame counts by 4-43 trailing undecodable frames). This rebuilds
    the original offsets so we can recover correct local frame numbers.
    """
    offsets = []
    offset = 0
    for t in trials:
        offsets.append(offset)
        cap = cv2.VideoCapture(t["video_path"])
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        offset += reported
    return offsets


def _get_frame_array(img_name: str, meta: dict, trials: list, reported_offsets: list):
    """Extract a frame as numpy array using label metadata + trial map."""
    info = meta.get(img_name)
    if not info:
        return None

    side = info["side"]

    # Prefer explicit video_file + local_frame (newer metadata — always correct)
    if "video_file" in info and "local_frame" in info:
        settings = get_settings()
        video_path = settings.video_path / info["video_file"]
        if video_path.exists():
            return extract_frame_raw(str(video_path), info["local_frame"], side)

    # Old metadata: frame_num used CAP_PROP_FRAME_COUNT offsets.
    # Use trial_idx + reported offsets to recover the correct local frame.
    trial_idx = info.get("trial_idx", 0)
    if trial_idx < len(trials) and trial_idx < len(reported_offsets):
        local_frame = info["frame_num"] - reported_offsets[trial_idx]
        video_path = trials[trial_idx]["video_path"]
        if local_frame >= 0:
            try:
                return extract_frame_raw(video_path, local_frame, side)
            except Exception:
                pass

    # Final fallback: direct resolve (only correct for trial_idx=0)
    try:
        video_path, local_frame = _resolve_frame(trials, info["frame_num"])
        return extract_frame_raw(video_path, local_frame, side)
    except Exception:
        return None


def _meta_needs_reported_offsets(meta: dict) -> bool:
    """Return True if any entry lacks local_frame/video_file (old format)."""
    return any(
        "local_frame" not in info or "video_file" not in info
        for info in meta.values()
    )


def process_round(round_dir: Path, subject_name: str, bodyparts: list) -> tuple:
    """Generate PNGs and QC overlays for one labeled-data round dir.

    Returns (pngs_generated, qc_generated, skipped) counts.
    """
    csv_path = round_dir / "CollectedData_labels.csv"
    meta_path = round_dir / "label_metadata.json"
    if not csv_path.exists() or not meta_path.exists():
        return 0, 0, 0

    with open(meta_path) as f:
        meta = json.load(f)

    if not meta:
        return 0, 0, 0

    old_format = _meta_needs_reported_offsets(meta)

    expected = sorted(meta.keys())
    existing_pngs = {p.name for p in round_dir.glob("img*.png")}
    qc_dir = round_dir / "qc"
    existing_qc = {p.name for p in qc_dir.glob("img*.png")} if qc_dir.exists() else set()

    missing_pngs = [n for n in expected if n not in existing_pngs]
    # For old-format metadata, always regenerate QC (existing ones may use wrong frames)
    missing_qc = expected if old_format else [n for n in expected if n not in existing_qc]

    if not missing_pngs and not missing_qc:
        return 0, 0, len(expected)

    # Build trial map (needed for PNG extraction and old-format offset calculation)
    trials = []
    reported_offsets = []
    needs_trials = missing_pngs or old_format
    if needs_trials:
        try:
            trials = build_trial_map(subject_name)
            if old_format and trials:
                reported_offsets = _build_reported_offsets(trials)
        except Exception as e:
            print(f"  WARNING: could not build trial map for {subject_name}: {e}")

    keypoints = _parse_collected_data_csv(csv_path)
    qc_dir.mkdir(exist_ok=True)

    pngs_generated = 0
    qc_generated = 0

    for img_name in expected:
        png_path = round_dir / img_name

        # Generate PNG if missing
        if img_name in missing_pngs:
            if not trials:
                continue
            frame = _get_frame_array(img_name, meta, trials, reported_offsets)
            if frame is None:
                print(f"  WARNING: could not extract frame for {img_name}")
                continue
            cv2.imwrite(str(png_path), frame)
            pngs_generated += 1

        # Generate QC overlay if missing (or always for old-format metadata)
        if img_name in missing_qc:
            frame = cv2.imread(str(png_path))
            if frame is None:
                continue
            kp = keypoints.get(img_name, {})
            overlay = _draw_overlay(frame, kp, bodyparts)
            cv2.imwrite(str(qc_dir / img_name), overlay)
            qc_generated += 1

    return pngs_generated, qc_generated, len(expected) - len(missing_qc) if not old_format else 0


def main():
    settings = get_settings()
    dlc_root = settings.dlc_path
    bodyparts = settings.bodyparts

    subject_dirs = sorted(
        p for p in dlc_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    total_pngs = 0
    total_qc = 0
    total_skipped = 0

    for subj_dir in subject_dirs:
        labeled_root = subj_dir / "labeled-data"
        if not labeled_root.is_dir():
            continue

        subject_name = subj_dir.name
        subj_pngs = subj_qc = subj_skip = 0

        for round_dir in sorted(labeled_root.iterdir()):
            if not round_dir.is_dir():
                continue
            p, q, s = process_round(round_dir, subject_name, bodyparts)
            subj_pngs += p
            subj_qc += q
            subj_skip += s

        if subj_pngs or subj_qc or subj_skip:
            parts = []
            if subj_pngs:
                parts.append(f"{subj_pngs} PNGs extracted")
            if subj_qc:
                parts.append(f"{subj_qc} QC overlays generated")
            if subj_skip and not (subj_pngs or subj_qc):
                parts.append(f"already complete ({subj_skip})")
            elif subj_skip:
                parts.append(f"{subj_skip} already done")
            print(f"{subject_name}: {', '.join(parts)}")

        total_pngs += subj_pngs
        total_qc += subj_qc
        total_skipped += subj_skip

    print(f"\nDone. Extracted {total_pngs} PNGs, generated {total_qc} QC overlays, {total_skipped} already existed.")


if __name__ == "__main__":
    main()
