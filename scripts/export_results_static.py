#!/usr/bin/env python3
"""Export the Results pages (Individual + Group Comparison) as a static
site that can be hosted on GitHub Pages — no server, no data drive.

The Results pages only ever issue four read-only GET endpoints:
    GET /api/subjects
    GET /api/results/group?include_auto=true&source=<src>&seq_mode=<mode>
    GET /api/results/{id}/traces?source=<src>
    GET /api/results/{id}/movements?source=<src>

This script drives the real FastAPI app (via TestClient, so the JSON is
identical to the live server), saves each response to ``data/`` with a
flattened filename, copies the static assets, and writes an
``index.html`` whose ``API.get`` is redirected to those files
(``window.STATIC_RESULTS``).

Usage:
    MT_DATA_DIR=~/data/movement-tracker python3 scripts/export_results_static.py [out_dir]

Default ``out_dir`` is ``site/`` at the repo root.  Then publish the
contents of that folder to GitHub Pages (see the printed instructions).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Source distances/sequence modes the dropdowns expose — must match the
# options in results.html.
SOURCES = ["auto", "corrections", "mp_combined", "mp_forward"]
SEQ_MODES = ["none", "linear_full", "linear_first10", "linear_multi",
             "exp_full", "exp_first10", "exp_multi"]
# Group-page hand + trial selectors.
HANDS = ["more", "less", "L", "R", "average"]
TRIALS = ["first", "last", "average"]


def _flatten(url: str) -> str:
    """Path+query → flat filename.  Mirrors _staticApiUrl in api.js."""
    return re.sub(r"[^A-Za-z0-9]+", "_", url.lstrip("/")) + ".json"


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO_ROOT / "site")
    data_dir = out_dir / "data"
    static_src = REPO_ROOT / "movement_tracker" / "static"
    data_dir.mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient
    from movement_tracker.app import app

    client = TestClient(app)
    written = 0
    resumed = 0
    # Resume support: skip URLs already exported (lets the run pick up
    # where it left off if interrupted, e.g. the laptop sleeping).  The
    # subjects list is always re-fetched (cheap) so we can enumerate.
    resume = "--fresh" not in sys.argv

    def dump(url: str, *, always: bool = False) -> dict | list | None:
        nonlocal written, resumed
        out_file = data_dir / _flatten(url)
        if resume and not always and out_file.exists() and out_file.stat().st_size > 0:
            resumed += 1
            try:
                return json.loads(out_file.read_text())
            except Exception:
                pass  # corrupt → re-fetch below
        resp = client.get(url)
        if resp.status_code != 200:
            print(f"  ! {url} -> HTTP {resp.status_code} (skipped)")
            return None
        payload = resp.json()
        out_file.write_text(json.dumps(payload))
        written += 1
        return payload

    print("Exporting API responses…")
    subjects = dump("/api/subjects") or []
    print(f"  {len(subjects)} subjects")

    # Group comparison: source × seq_mode × hand × trial (include_auto
    # always true on the group page).
    for src in SOURCES:
        for sm in SEQ_MODES:
            for hd in HANDS:
                for tr in TRIALS:
                    dump(f"/api/results/group?include_auto=true&source={src}"
                         f"&seq_mode={sm}&hand={hd}&trial={tr}")

    # Per-subject traces + movements for every source.
    for subj in subjects:
        sid = subj.get("id")
        if sid is None:
            continue
        for src in SOURCES:
            dump(f"/api/results/{sid}/traces?source={src}")
            dump(f"/api/results/{sid}/movements?source={src}")
        # Legacy no-source movements call (hidden tab) — alias to auto.
        dump(f"/api/results/{sid}/movements")

    print(f"  wrote {written} new JSON files ({resumed} already present) to {data_dir}")

    # ── Static assets ──────────────────────────────────────────────
    print("Copying static assets…")
    dest_static = out_dir / "static"
    if dest_static.exists():
        shutil.rmtree(dest_static)
    shutil.copytree(static_src, dest_static)

    # ── index.html (transformed results.html) ──────────────────────
    print("Building index.html…")
    html = (static_src / "results.html").read_text()

    # Absolute asset paths → relative (GitHub project pages live under
    # /<repo>/, so absolute /static/... would break).
    html = html.replace('href="/static/', 'href="static/')
    html = html.replace('src="/static/', 'src="static/')
    # Drop nav.js — its cross-page links and job polling don't apply to
    # the static results-only site.
    html = re.sub(r'\s*<script src="static/js/nav\.js[^"]*"></script>', "", html)
    # Strip the cross-page <nav> block (dead links on the static site).
    html = re.sub(r"<nav>.*?</nav>", "", html, flags=re.DOTALL)
    # Make the tab switcher visible so Individual/Group toggle without
    # the nav dropdown.
    html = html.replace('id="tabSwitcher" style="display:none;"',
                        'id="tabSwitcher" style="display:flex;gap:6px;"')
    # Title link → relative.
    html = html.replace('href="/" style="font-size:18px;"',
                        'href="index.html" style="font-size:18px;"')
    # Turn on static mode before any app script runs.
    html = html.replace(
        "</head>",
        "    <script>window.STATIC_RESULTS = true;</script>\n</head>",
    )

    (out_dir / "index.html").write_text(html)

    # GitHub Pages: disable Jekyll so files/folders are served verbatim.
    (out_dir / ".nojekyll").write_text("")

    print(f"\nDone → {out_dir}")
    print(
        "\nPublish to GitHub Pages:\n"
        f"  1. cd {out_dir}\n"
        "  2. git init && git add -A && git commit -m 'Results site'\n"
        "  3. Create a new GitHub repo (e.g. movement-results) and:\n"
        "       git branch -M main\n"
        "       git remote add origin git@github.com:<you>/<repo>.git\n"
        "       git push -u origin main\n"
        "  4. Repo → Settings → Pages → Source: Deploy from a branch,\n"
        "     branch 'main' / root.  Site appears at\n"
        "     https://<you>.github.io/<repo>/\n"
    )


if __name__ == "__main__":
    main()
