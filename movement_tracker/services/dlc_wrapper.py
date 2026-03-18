"""DLC subprocess command builders."""
from __future__ import annotations

import re
from pathlib import Path

from ..config import get_settings

# Preamble for all DLC commands
_DLC_PREAMBLE = "from deeplabcut.core.engine import Engine; import deeplabcut; "


def fix_project_path(subject_name: str) -> str:
    """Ensure project_path in config.yaml points to correct directory. Returns config path."""
    settings = get_settings()
    dlc_dir = settings.dlc_path
    config_path = dlc_dir / subject_name / "config.yaml"
    expected_path = str(dlc_dir / subject_name)

    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml for {subject_name}")

    lines = config_path.read_text().splitlines(keepends=True)

    new_lines = []
    i = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("project_path:"):
            value = line.split(":", 1)[1].strip()
            if value:
                current_path = value.rstrip("\n").rstrip("\\").rstrip("/")
            else:
                i += 1
                if i < len(lines):
                    current_path = lines[i].strip().rstrip("\\").rstrip("/")
                else:
                    current_path = ""

            import os
            if os.path.normpath(current_path) != os.path.normpath(expected_path):
                changed = True
            new_lines.append(f"project_path: {expected_path}\n")
        else:
            new_lines.append(line)
        i += 1

    if changed:
        config_path.write_text("".join(new_lines))

    return str(config_path)


def ensure_all_rounds_in_config(config_path: str) -> list[str]:
    """Add video_sets entries for any labeled-data/roundN/ dirs missing from config.yaml.

    DLC's ``create_training_dataset`` only reads labeled-data subdirs that
    match a video stem in ``video_sets``.  When refinement adds a new round,
    its directory must be registered or the labels are silently ignored.

    Returns the list of round names that were added (empty if none).
    """
    cfg = Path(config_path)
    labeled_data_dir = cfg.parent / "labeled-data"
    if not labeled_data_dir.is_dir():
        return []

    text = cfg.read_text()
    added: list[str] = []

    for rd in sorted(labeled_data_dir.iterdir()):
        if not rd.is_dir():
            continue
        rn = rd.name
        # Check if already present (forward or backslash path separators)
        if f"/{rn}.mp4:" in text or f"\\{rn}.mp4:" in text:
            continue
        # Clone the first video_sets entry with the new round name
        m = re.search(r'(  .+[\\/])(\w+)(\.mp4:\s*\n\s+crop:\s*[^\n]+)', text)
        if m:
            new_entry = f"{m.group(1)}{rn}{m.group(3)}"
            text = text.replace(m.group(0), m.group(0) + "\n" + new_entry, 1)
            added.append(rn)

    if added:
        cfg.write_text(text)

    return added


def cmd_create_training_dataset(config_path: str) -> list[str]:
    """Build command to create DLC training dataset."""
    settings = get_settings()
    script = (
        f"{_DLC_PREAMBLE}"
        f"deeplabcut.create_training_dataset(r'{config_path}', net_type='{settings.dlc_net_type}', engine=Engine.PYTORCH)"
    )
    return [settings.python_executable, "-c", script]


def cmd_train_network(config_path: str) -> list[str]:
    """Build command to train DLC network."""
    settings = get_settings()
    script = (
        f"{_DLC_PREAMBLE}"
        f"deeplabcut.train_network(r'{config_path}', engine=Engine.PYTORCH)"
    )
    return [settings.python_executable, "-c", script]


def cmd_create_and_train(config_path: str) -> list[str]:
    """Build command to recreate the training dataset then resume training.

    Used by the Refine step: ``create_training_dataset`` picks up any new
    ``labeled-data/roundN/`` directories (e.g. round2 corrections), and
    ``train_network`` automatically resumes from the latest checkpoint.
    """
    settings = get_settings()
    script = (
        f"{_DLC_PREAMBLE}"
        f"deeplabcut.create_training_dataset(r'{config_path}', "
        f"net_type='{settings.dlc_net_type}', engine=Engine.PYTORCH); "
        f"deeplabcut.train_network(r'{config_path}', engine=Engine.PYTORCH)"
    )
    return [settings.python_executable, "-c", script]


def cmd_analyze_videos(config_path: str, video_dir: str) -> list[str]:
    """Build command to analyze videos with trained DLC model."""
    settings = get_settings()
    script = (
        f"{_DLC_PREAMBLE}"
        f"deeplabcut.analyze_videos(r'{config_path}', [r'{video_dir}'])"
    )
    return [settings.python_executable, "-c", script]


def cmd_create_labeled_video(config_path: str, video_dir: str) -> list[str]:
    """Build command to create labeled overlay videos."""
    settings = get_settings()
    script = (
        f"{_DLC_PREAMBLE}"
        f"deeplabcut.create_labeled_video(r'{config_path}', r'{video_dir}')"
    )
    return [settings.python_executable, "-c", script]


def cmd_triangulate(config_3d_path: str, video_dir: str) -> list[str]:
    """Build command for stereo triangulation."""
    settings = get_settings()
    script = (
        f"{_DLC_PREAMBLE}"
        f"deeplabcut.triangulate(r'{config_3d_path}', r'{video_dir}', "
        f"filterpredictions=False, save_as_csv=True)"
    )
    return [settings.python_executable, "-c", script]


def cmd_convert_h5_to_csv(video_dir: str) -> list[str]:
    """Build command to convert H5 results to CSV."""
    settings = get_settings()
    script = (
        f"{_DLC_PREAMBLE}"
        f"deeplabcut.analyze_videos_converth5_to_csv(r'{video_dir}')"
    )
    return [settings.python_executable, "-c", script]
