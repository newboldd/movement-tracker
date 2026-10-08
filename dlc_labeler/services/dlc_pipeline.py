#!/usr/bin/env python
"""Local DeepLabCut pipeline: train → crop → analyze, as one subprocess.

Run as a standalone script, deliberately with no ``dlc_labeler`` imports:
the interpreter that has ``deeplabcut`` installed may be a different venv
or conda environment from the one serving the web app (Settings → Python
executable).  Only ``deeplabcut`` and ``cv2`` are required, both of which
any working DLC environment has.

Progress is reported on stdout as ``PROGRESS:<pct>`` lines; the parent
process parses those (plus DeepLabCut's own ``Epoch X/Y`` output) and
writes them to the jobs table.

Usage:
    python dlc_pipeline.py --mode train \\
        --config-path <dlc>/<subject>/config.yaml \\
        --labels-dir  <dlc>/<subject>/labels_v1 \\
        --video-dir   <videos> \\
        --subject-name MSA01 \\
        --cam-names OS OD \\
        --net-type resnet_50

Modes:
    train    create training dataset from labeled-data/, train from
             scratch, then crop + analyze into --labels-dir.
    refine   register any new labeled-data/roundN/ directories in
             config.yaml, recreate the training dataset, resume training
             from the latest snapshot, then crop + analyze.
    analyze  crop + analyze only, reusing the existing trained model.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def progress(pct: float):
    """Report overall progress to the parent process."""
    print(f"PROGRESS:{pct:.1f}", flush=True)


# ── config.yaml fix-ups ─────────────────────────────────────────────────

def fix_project_path(config_path: str) -> None:
    """Point ``project_path`` at the directory config.yaml actually lives in.

    DLC stores an absolute path in the project config.  Copying a project
    between machines — or just moving the data directory — leaves that
    path stale, and every DLC call then fails on a path that doesn't
    exist.  Rewriting it on each run makes the projects portable, which
    is the whole point of handing these to someone else.
    """
    expected = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        lines = f.readlines()

    out, changed, i = [], False, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("project_path:"):
            value = line.split(":", 1)[1].strip()
            if value:
                current = value.rstrip("\n").rstrip("\\").rstrip("/")
            else:
                # Block-style value on the following line.
                i += 1
                current = lines[i].strip().rstrip("\\").rstrip("/") if i < len(lines) else ""
            if os.path.normpath(current) != os.path.normpath(expected):
                changed = True
            out.append(f"project_path: {expected}\n")
        else:
            out.append(line)
        i += 1

    if changed:
        with open(config_path, "w") as f:
            f.writelines(out)
        logger.info("Rewrote project_path -> %s", expected)


def ensure_all_rounds_in_config(config_path: str) -> list[str]:
    """Register every ``labeled-data/roundN/`` directory in ``video_sets``.

    ``create_training_dataset`` only reads labeled-data subdirectories
    whose name matches a video stem in ``video_sets``.  A refinement
    round that isn't registered has its labels silently ignored — the
    training set looks unchanged and the model never learns the
    corrections.  Returns the round names that were added.
    """
    labeled_data_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                    "labeled-data")
    if not os.path.isdir(labeled_data_dir):
        return []

    with open(config_path) as f:
        text = f.read()

    added: list[str] = []
    for rn in sorted(os.listdir(labeled_data_dir)):
        if not os.path.isdir(os.path.join(labeled_data_dir, rn)):
            continue
        if f"/{rn}.mp4:" in text or f"\\{rn}.mp4:" in text:
            continue
        m = re.search(r'(  .+[\\/])(\w+)(\.mp4:\s*\n\s+crop:\s*[^\n]+)', text)
        if m:
            new_entry = f"{m.group(1)}{rn}{m.group(3)}"
            text = text.replace(m.group(0), m.group(0) + "\n" + new_entry, 1)
            added.append(rn)

    if added:
        with open(config_path, "w") as f:
            f.write(text)
        logger.info("Registered new labeled-data rounds in config.yaml: %s",
                    ", ".join(added))
    return added


# ── Pre-flight: labeled-data PNGs ───────────────────────────────────────

def ensure_labeled_data_pngs(config_path: str, video_dir: str, subject_name: str):
    """Regenerate missing labeled-data PNGs from the source videos.

    The commit step writes both the PNGs and a ``label_metadata.json``
    sidecar naming the video file, local frame and camera half each PNG
    came from.  The PNGs are large and easy to lose (a cleaned data
    directory, a project copied without them); the sidecar makes them
    reproducible, so training never fails for want of image files that
    can be re-extracted in seconds.
    """
    import cv2

    labeled_data_root = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                     "labeled-data")
    if not os.path.isdir(labeled_data_root):
        return

    for subdir in sorted(os.listdir(labeled_data_root)):
        ld_path = os.path.join(labeled_data_root, subdir)
        meta_path = os.path.join(ld_path, "label_metadata.json")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        missing = [name for name in meta
                   if not os.path.isfile(os.path.join(ld_path, name))]
        if not missing:
            continue

        logger.info("Regenerating %d/%d PNGs in labeled-data/%s",
                    len(missing), len(meta), subdir)

        # Group by source video so each file is opened once.
        by_video: dict[str, list[tuple[str, int, str]]] = {}
        for img_name in missing:
            info = meta[img_name]
            side = info["side"]
            if "video_file" in info and "local_frame" in info:
                vpath = os.path.join(video_dir, info["video_file"])
                local_frame = info["local_frame"]
            else:
                # Pre-sidecar metadata: only a global frame number.
                videos = sorted(glob.glob(
                    os.path.join(video_dir, f"{subject_name}_*.mp4")))
                if not videos:
                    logger.warning("No video found for %s_* in %s",
                                   subject_name, video_dir)
                    break
                vpath = videos[0]
                local_frame = info["frame_num"]
            by_video.setdefault(vpath, []).append((img_name, local_frame, side))

        count = 0
        for vpath, frames in by_video.items():
            if not os.path.isfile(vpath):
                logger.warning("Video not found: %s", vpath)
                continue
            cap = cv2.VideoCapture(vpath)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if w == 0:
                logger.warning("Cannot open video: %s", vpath)
                cap.release()
                continue
            mid = w // 2
            for img_name, local_frame, side in sorted(frames, key=lambda x: x[1]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Cannot read frame %s from %s",
                                   local_frame, os.path.basename(vpath))
                    continue
                half = frame[:, :mid] if side == "OS" else frame[:, mid:]
                cv2.imwrite(os.path.join(ld_path, img_name), half)
                count += 1
            cap.release()

        logger.info("Extracted %d PNGs for labeled-data/%s", count, subdir)


# ── Crop ────────────────────────────────────────────────────────────────

def crop_stereo_videos(video_dir: str, subject_name: str, labels_dir: str,
                       cam_names: list[str], camera_mode: str = "stereo"):
    """Write one per-camera video per trial into ``labels_dir``.

    DLC analyses single-camera video, so a side-by-side stereo recording
    has to be split at the midline first.  Output names are
    ``{stem}_{cam}.mp4``, which is also what the CSV→trial matcher in
    ``dlc_predictions`` expects when it reads the results back.

    Already-cropped videos are skipped, so re-running analyze on a
    subject costs nothing extra.
    """
    import cv2

    os.makedirs(labels_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(video_dir, f"{subject_name}_*.mp4")))
    logger.info("Found %d source videos for %s", len(videos), subject_name)

    if camera_mode == "multicam":
        # Already per-camera on disk — copy (hardlink where possible) so
        # DLC writes its .h5 next to a file inside labels_dir.
        import shutil
        for vpath in videos:
            dest = os.path.join(labels_dir, os.path.basename(vpath))
            if os.path.exists(dest):
                continue
            try:
                os.link(vpath, dest)
            except OSError:
                shutil.copy2(vpath, dest)
        logger.info("Linked %d per-camera videos into %s", len(videos), labels_dir)
        return

    for vpath in videos:
        stem, ext = os.path.splitext(os.path.basename(vpath))

        if camera_mode == "single":
            out_paths = [os.path.join(labels_dir, f"{stem}_{cam_names[0]}{ext}")]
        else:
            out_paths = [os.path.join(labels_dir, f"{stem}_{cam_names[0]}{ext}")]
            if len(cam_names) > 1:
                out_paths.append(
                    os.path.join(labels_dir, f"{stem}_{cam_names[1]}{ext}"))

        if all(os.path.exists(p) for p in out_paths):
            logger.info("Skipping %s (already cropped)", stem)
            continue

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == 0 or h == 0:
            cap.release()
            logger.warning("Cannot read %s", vpath)
            continue

        mid = w // 2 if camera_mode == "stereo" else w
        codec = "avc1" if sys.platform == "darwin" else "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*codec)

        writer_L = cv2.VideoWriter(out_paths[0], fourcc, fps, (mid, h))
        writer_R = None
        if len(out_paths) > 1:
            writer_R = cv2.VideoWriter(out_paths[1], fourcc, fps, (w - mid, h))

        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer_L.write(frame[:, :mid])
            if writer_R is not None:
                writer_R.write(frame[:, mid:])

        cap.release()
        writer_L.release()
        if writer_R is not None:
            writer_R.release()
        logger.info("Cropped %s: %d frames", stem, n_frames)

    logger.info("Crop complete")


# ── Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(*, mode: str, config_path: str, labels_dir: str, video_dir: str,
                 subject_name: str, cam_names: list[str], shuffle: int,
                 net_type: str, camera_mode: str):
    # DLC links against its own OpenMP runtime; on machines that already
    # have one loaded (common with conda + MKL) the duplicate aborts the
    # process at import time.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from deeplabcut.core.engine import Engine
    import deeplabcut

    fix_project_path(config_path)

    if mode in ("train", "refine"):
        logger.info("=== Pre-flight: labeled-data PNGs ===")
        progress(1.0)
        ensure_labeled_data_pngs(config_path, video_dir, subject_name)

        if mode == "refine":
            ensure_all_rounds_in_config(config_path)

        # Build the .h5 files DLC reads from the committed CSVs.  The
        # commit step attempts this, but it runs in the web app's
        # interpreter, which may not have DeepLabCut — labeling a whole
        # subject before installing DLC is a normal thing to do.  Here we
        # are already inside the DLC environment, so it always works.
        logger.info("=== Converting CollectedData CSVs to H5 ===")
        progress(2.0)
        try:
            deeplabcut.convertcsv2h5(config_path, userfeedback=False)
        except Exception as e:
            # Not fatal on its own: if the H5 files are already present and
            # current, create_training_dataset will be happy regardless.
            logger.warning("convertcsv2h5 failed (%s); continuing", e)

        logger.info("=== Creating training dataset (net_type=%s) ===", net_type)
        progress(4.0)
        deeplabcut.create_training_dataset(
            config_path, net_type=net_type, engine=Engine.PYTORCH)

        # 6-75% is covered by the parent's Epoch X/Y parsing of DLC's own
        # output, so no PROGRESS line is emitted during training itself.
        logger.info("=== Training ===")
        progress(6.0)
        deeplabcut.train_network(config_path, shuffle=shuffle,
                                 engine=Engine.PYTORCH)
        logger.info("=== Training complete ===")
        progress(75.0)
    else:
        logger.info("=== Skipping training (analyze-only) ===")
        progress(5.0)

    logger.info("=== Cropping videos for analysis ===")
    progress(76.0)
    crop_stereo_videos(video_dir, subject_name, labels_dir, cam_names,
                       camera_mode=camera_mode)
    progress(80.0)

    logger.info("=== Analyzing ===")
    # DLC skips videos it believes are already analyzed by looking for
    # .h5 / .pickle next to them, so a re-run after retraining would
    # silently keep the old predictions.  Clear them first.
    for ext in ("*.h5", "*.pickle"):
        for old in glob.glob(os.path.join(labels_dir, ext)):
            logger.info("Removing stale analysis file: %s", os.path.basename(old))
            os.remove(old)

    video_files = sorted(glob.glob(os.path.join(labels_dir, "*.mp4")))
    logger.info("Analyzing %d videos: %s", len(video_files),
                [os.path.basename(v) for v in video_files])
    if not video_files:
        raise RuntimeError(
            f"No videos to analyze in {labels_dir} — check that "
            f"{subject_name}_*.mp4 exists in {video_dir}")

    deeplabcut.analyze_videos(config_path, video_files, shuffle=shuffle,
                              engine=Engine.PYTORCH)
    progress(95.0)

    deeplabcut.analyze_videos_converth5_to_csv(labels_dir)
    logger.info("=== All phases complete ===")
    progress(100.0)


def main():
    p = argparse.ArgumentParser(description="Local DeepLabCut pipeline")
    p.add_argument("--mode", required=True, choices=["train", "refine", "analyze"])
    p.add_argument("--config-path", required=True, help="DLC config.yaml")
    p.add_argument("--labels-dir", required=True,
                   help="Output directory for cropped videos + predictions")
    p.add_argument("--video-dir", required=True, help="Source video directory")
    p.add_argument("--subject-name", required=True)
    p.add_argument("--cam-names", nargs="+", default=["OS", "OD"])
    p.add_argument("--camera-mode", default="stereo",
                   choices=["stereo", "multicam", "single"])
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--net-type", default="resnet_50")
    args = p.parse_args()

    try:
        progress(0.0)
        run_pipeline(
            mode=args.mode,
            config_path=args.config_path,
            labels_dir=args.labels_dir,
            video_dir=args.video_dir,
            subject_name=args.subject_name,
            cam_names=args.cam_names,
            shuffle=args.shuffle,
            net_type=args.net_type,
            camera_mode=args.camera_mode,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
