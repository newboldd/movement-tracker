"""Per-page static-cache configuration.

Single source of truth for which (source × seq_mode × hand × trial)
combinations are pre-computed and shipped to the public GitHub Pages
site.  Read by:

  * ``routers/results.py`` (``_GROUP_STATIC_*`` / ``_EXPLORE_STATIC_*``
    sets) to decide whether a request can be served from a cached
    JSON or has to fall through to live compute.
  * ``routers/cache_results.py`` to render the Jobs-page settings UI
    and to drive the rebuild button.
  * ``scripts/export_results_static.py`` so the public site exports
    exactly the combos the user has configured.

Stored as ``<DATA_DIR>/cache_config.json``.  Defaults match what
shipped before this module landed (so existing deploys are
unchanged on first read).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

_CACHE_PATH = DATA_DIR / "cache_config.json"

# Universe of values each axis can take — used by the UI to render
# checkbox grids and by validation when the user submits new settings.
OPTION_DOMAINS = {
    "sources":   ["auto", "corrections", "mp_combined", "mp_forward", "skeleton_v1", "skeleton_v2"],
    "seq_modes": ["none",
                  "linear_full", "linear_first10", "linear_multi",
                  "exp_full",    "exp_first10",    "exp_multi"],
    "hands":     ["more", "less", "L", "R", "average", "larger_se", "smaller_se"],
    "trials":    ["first", "last", "average"],
}

# Human-friendly labels for the UI.  Keys mirror the OPTION_DOMAINS
# values; entries omitted from the maps render as raw values.
OPTION_LABELS = {
    "sources": {
        "auto":         "Auto",
        "corrections":  "DLC (corrected)",
        "mp_combined":  "MP combined",
        "mp_forward":   "MP forward",
        "skeleton_v1":  "Skel fit v1",
        "skeleton_v2":  "Skel fit v2",
    },
    "seq_modes": {
        "none":            "None",
        "linear_full":     "Linear (full)",
        "linear_first10":  "Linear (first 10)",
        "linear_multi":    "Linear (multi-seq)",
        "exp_full":        "Exponential (full)",
        "exp_first10":     "Exponential (first 10)",
        "exp_multi":       "Exponential (multi-seq)",
    },
    "hands": {
        "more":       "More affected",
        "less":       "Less affected",
        "L":          "Left",
        "R":          "Right",
        "average":    "Average",
        "larger_se":  "Larger SE",
        "smaller_se": "Smaller SE",
    },
    "trials": {
        "first":   "First",
        "last":    "Last",
        "average": "Average",
    },
}

# Defaults shipped before the user customises.  These mirror the
# previous hard-coded ``_GROUP_STATIC_*`` / ``_EXPLORE_STATIC_*``
# sets so first-run behaviour is unchanged.
_DEFAULT_CFG: dict = {
    "group": {
        # Skeleton fit v1 is now a first-class source for head-to-head
        # comparison with DLC and MP on the Group Comparison page.
        "sources":   ["auto", "corrections", "mp_combined", "skeleton_v1"],
        "seq_modes": ["exp_full", "exp_first10", "exp_multi"],
        "hand_trial": [["more", "last"], ["less", "last"], ["average", "average"]],
    },
    "explore": {
        "sources":   ["auto", "corrections", "mp_combined", "skeleton_v1"],
        "seq_modes": ["exp_full", "exp_first10", "exp_multi"],
        "hand_trial": [["more", "last"], ["less", "last"], ["average", "average"]],
    },
}


def _validate_page_cfg(cfg: dict) -> dict:
    """Drop unknown values and de-dupe; returns a cleaned copy."""
    out = {}
    out["sources"] = sorted({s for s in cfg.get("sources", [])
                              if s in OPTION_DOMAINS["sources"]})
    out["seq_modes"] = sorted({s for s in cfg.get("seq_modes", [])
                                if s in OPTION_DOMAINS["seq_modes"]})
    pairs = []
    seen = set()
    for p in cfg.get("hand_trial", []):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        h, t = p[0], p[1]
        if h not in OPTION_DOMAINS["hands"]:  continue
        if t not in OPTION_DOMAINS["trials"]: continue
        key = (h, t)
        if key in seen: continue
        seen.add(key)
        pairs.append([h, t])
    out["hand_trial"] = pairs
    return out


def load_cache_config() -> dict:
    """Read the config from disk, falling back to the defaults."""
    if _CACHE_PATH.exists():
        try:
            raw = json.loads(_CACHE_PATH.read_text())
            cfg = {
                "group":   _validate_page_cfg(raw.get("group", {})),
                "explore": _validate_page_cfg(raw.get("explore", {})),
            }
            # Make sure each page has at least one value per axis so
            # the cache isn't a hard 404 wall.
            for page in ("group", "explore"):
                for axis in ("sources", "seq_modes", "hand_trial"):
                    if not cfg[page][axis]:
                        cfg[page][axis] = list(_DEFAULT_CFG[page][axis])
            return cfg
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("cache_config.json unreadable (%s); using defaults", e)
    return {k: {ax: list(v) if isinstance(v, list) else v
                for ax, v in pg.items()}
            for k, pg in _DEFAULT_CFG.items()}


def save_cache_config(cfg: dict) -> dict:
    """Persist ``cfg`` to disk after validation; returns the cleaned copy."""
    cleaned = {
        "group":   _validate_page_cfg(cfg.get("group", {})),
        "explore": _validate_page_cfg(cfg.get("explore", {})),
    }
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cleaned, indent=2) + "\n")
    return cleaned


def get_page_combos(page: str) -> dict:
    """Return ``{sources, seq_modes, hand_trial}`` for one page."""
    cfg = load_cache_config()
    return cfg.get(page, _DEFAULT_CFG.get(page, {}))


def cache_size_bytes() -> int:
    """Sum of sizes of all JSONs in the static-export data dir.
    Returns 0 if the dir doesn't exist yet."""
    site_data = Path(__file__).resolve().parents[2] / "site" / "data"
    if not site_data.is_dir():
        return 0
    total = 0
    for p in site_data.iterdir():
        if p.suffix == ".json":
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def cache_file_count() -> int:
    site_data = Path(__file__).resolve().parents[2] / "site" / "data"
    if not site_data.is_dir():
        return 0
    return sum(1 for p in site_data.iterdir() if p.suffix == ".json")
