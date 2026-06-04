"""Endpoints driving the Jobs-page 'Cache Results' section.

Two responsibilities:

  * Read / write the per-page cache-config JSON via
    ``services.cache_config``.
  * Trigger the static-export script for a single page (Group
    Comparison or Explore) without blocking the request.

The export runs in a background thread instead of a subprocess so it
shares the in-memory ``_EXPORT_MOVE_CACHE`` warmed by the live app —
much faster than a cold ``TestClient`` run.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException

from ..services.cache_config import (
    OPTION_DOMAINS,
    OPTION_LABELS,
    cache_file_count,
    cache_size_bytes,
    load_cache_config,
    save_cache_config,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cache-results", tags=["cache-results"])

# Track the most recent rebuild per page so the UI can show progress.
_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()


@router.get("/config")
def get_config() -> dict:
    """Return the live config plus the option universe + cache size."""
    return {
        "config": load_cache_config(),
        "domains": OPTION_DOMAINS,
        "labels":  OPTION_LABELS,
        "cache": {
            "bytes": cache_size_bytes(),
            "files": cache_file_count(),
        },
        "runs": _RUNS,
    }


@router.post("/config")
def post_config(body: dict = Body(...)) -> dict:
    """Persist new config.  Doesn't trigger a rebuild — that's a
    separate POST so the user can edit settings freely without
    triggering a long-running export until they hit Cache."""
    cleaned = save_cache_config(body)
    return {"config": cleaned}


def _run_export_for_page(page: str) -> None:
    """Drive the existing export-results-static script in-process.

    Uses TestClient against the live FastAPI app so the responses
    reuse the warm ``_EXPORT_MOVE_CACHE`` and avoid duplicating any
    of the iteration logic that lives in the script.
    """
    from fastapi.testclient import TestClient
    from ..app import app
    from ..services.cache_config import get_page_combos
    from pathlib import Path
    import json

    site_data = Path(__file__).resolve().parents[2] / "site" / "data"
    site_data.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)

    combos = get_page_combos(page)
    sources    = combos["sources"]
    hand_trial = combos["hand_trial"]

    endpoint = "group" if page == "group" else "explore"
    # seq_mode is no longer a cache axis — each cached file embeds
    # all sequence-effect models — so the fanout is just sources ×
    # hand_trial.  6× fewer files than the pre-refactor layout.
    n_total = len(sources) * len(hand_trial)
    n_done = 0
    n_written = 0

    def _flatten(url: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9]+", "_", url.lstrip("/")) + ".json"

    with _LOCK:
        _RUNS[page]["status"] = "running"
        _RUNS[page]["n_total"] = n_total
        _RUNS[page]["n_done"] = 0

    try:
        for src in sources:
            for hd, tr in hand_trial:
                url = (f"/api/results/{endpoint}?include_auto=true"
                       f"&source={src}"
                       f"&hand={hd}&trial={tr}")
                out = site_data / _flatten(url)
                resp = client.get(url)
                if resp.status_code == 200:
                    out.write_text(json.dumps(resp.json()))
                    n_written += 1
                n_done += 1
                with _LOCK:
                    _RUNS[page]["n_done"] = n_done

        with _LOCK:
            _RUNS[page]["status"] = "done"
            _RUNS[page]["n_written"] = n_written
            _RUNS[page]["finished"] = time.time()
    except Exception as e:
        logger.exception("cache rebuild for %s failed", page)
        with _LOCK:
            _RUNS[page]["status"] = "error"
            _RUNS[page]["error"] = str(e)


@router.post("/rebuild/{page}")
def rebuild(page: str) -> dict:
    """Kick off a cache rebuild for one page in a background thread."""
    if page not in ("group", "explore"):
        raise HTTPException(400, f"unknown page '{page}'")
    with _LOCK:
        cur = _RUNS.get(page)
        if cur and cur.get("status") == "running":
            return {"started": False, "run": cur}
        _RUNS[page] = {"status": "starting", "started": time.time(),
                       "n_total": 0, "n_done": 0, "n_written": 0}
    th = threading.Thread(target=_run_export_for_page, args=(page,), daemon=True)
    th.start()
    return {"started": True, "run": _RUNS[page]}
