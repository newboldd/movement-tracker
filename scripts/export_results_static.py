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

# Combos shipped to the public site come from cache_config.json
# (managed via the Jobs page).  Lazy-load below in main() so the
# config is read fresh on every run.
def _load_combos():
    from movement_tracker.services.cache_config import get_page_combos
    g = get_page_combos("group")
    e = get_page_combos("explore")
    return (
        g["sources"], g["seq_modes"], [(h, t) for h, t in g["hand_trial"]],
        e["sources"], e["seq_modes"], [(h, t) for h, t in e["hand_trial"]],
    )


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

    def _stale(url: str, payload) -> bool:
        """Decide whether an already-cached payload is missing fields
        that were added after it was first written — in which case we
        force a re-fetch.  Add new sentinels here as the schema evolves."""
        if not isinstance(payload, dict):
            return False
        subjs = payload.get("subjects")
        if not isinstance(subjs, list) or not subjs:
            return False
        s0 = subjs[0]
        if "/api/results/explore" in url:
            vars0 = s0.get("vars") or {}
            return "movement_similarity" not in vars0
        if "/api/results/group" in url:
            return "movement_similarity" not in s0
        return False

    def dump(url: str, *, always: bool = False) -> dict | list | None:
        nonlocal written, resumed
        out_file = data_dir / _flatten(url)
        if resume and not always and out_file.exists() and out_file.stat().st_size > 0:
            try:
                cached = json.loads(out_file.read_text())
                if not _stale(url, cached):
                    resumed += 1
                    return cached
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

    (G_SOURCES, G_SEQ_MODES, G_HAND_TRIAL,
     E_SOURCES, E_SEQ_MODES, E_HAND_TRIAL) = _load_combos()

    # Group comparison: source × seq_mode × explicit (hand, trial)
    # pairs (include_auto always true on the group page).
    for src in G_SOURCES:
        for sm in G_SEQ_MODES:
            for hd, tr in G_HAND_TRIAL:
                dump(f"/api/results/group?include_auto=true&source={src}"
                     f"&seq_mode={sm}&hand={hd}&trial={tr}")

    # Explore (Variable Explorer) — same shape, narrower set.
    for src in E_SOURCES:
        for sm in E_SEQ_MODES:
            for hd, tr in E_HAND_TRIAL:
                dump(f"/api/results/explore?include_auto=true&source={src}"
                     f"&seq_mode={sm}&hand={hd}&trial={tr}")

    # Per-subject traces + movements — once per source the user picked
    # for either page (union, since traces are page-agnostic).
    per_subj_sources = sorted(set(G_SOURCES) | set(E_SOURCES))
    for subj in subjects:
        sid = subj.get("id")
        if sid is None:
            continue
        for src in per_subj_sources:
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

    # ── Page HTMLs (results → index.html, plus explore.html) ────────
    def _transform(src_html: str, page: str) -> str:
        """Apply the common HTML transforms used by both static pages."""
        h = src_html
        # Absolute → relative asset paths.
        h = h.replace('href="/static/', 'href="static/')
        h = h.replace('src="/static/', 'src="static/')
        # Drop nav.js + the <nav> block (cross-page links don't apply).
        h = re.sub(r'\s*<script src="static/js/nav\.js[^"]*"></script>', "", h)
        h = re.sub(r"<nav>.*?</nav>", "", h, flags=re.DOTALL)
        # Title → relative link to results.
        h = h.replace('href="/" style="font-size:18px;"',
                      'href="index.html" style="font-size:18px;"')
        # results.html has a hidden tab switcher — surface it on the static site.
        if page == "results":
            h = h.replace('id="tabSwitcher" style="display:none;"',
                          'id="tabSwitcher" style="display:flex;gap:6px;"')
        # Small fixed-position cross-page link.
        link = ('<a href="explore.html"', '→ Explore') if page == "results" \
            else ('<a href="index.html"', '← Results')
        cross = (link[0] + ' style="position:fixed;top:10px;right:14px;z-index:99;'
                 'background:#fff;padding:4px 10px;border:1px solid #ddd;'
                 'border-radius:4px;text-decoration:none;color:#444;'
                 'font-size:13px;">' + link[1] + '</a>')
        h = re.sub(r'(<body[^>]*>)', r'\1' + cross, h, count=1)
        # Turn on static mode before any app script runs.
        h = h.replace("</head>",
                      "    <script>window.STATIC_RESULTS = true;</script>\n</head>")
        return h

    print("Building index.html…")
    html = _transform((static_src / "results.html").read_text(), "results")

    # Simple client-side password gate.  SHA-256 of the password is
    # hardcoded; the user types it once per tab (sessionStorage).  Not
    # cryptographically strong — anyone can view-source and brute-force
    # — but it's a low-friction "are you supposed to be here" barrier in
    # front of non-identifying research data.
    gate = (
        "<style>#__gate{position:fixed;inset:0;background:#0b1220;"
        "display:flex;align-items:center;justify-content:center;z-index:99999;"
        "font:14px system-ui,sans-serif;color:#e7eaf0;}"
        "#__gate form{background:#141c2e;padding:24px 28px;border-radius:8px;"
        "box-shadow:0 6px 24px rgba(0,0,0,.4);display:flex;flex-direction:column;gap:10px;min-width:260px;}"
        "#__gate input{padding:8px 10px;border:1px solid #2a3550;border-radius:4px;"
        "background:#0b1220;color:#e7eaf0;font-size:14px;}"
        "#__gate button{padding:8px 12px;border:0;border-radius:4px;background:#3b82f6;"
        "color:#fff;font-size:14px;cursor:pointer;}"
        "#__gate .err{color:#f87171;min-height:1em;font-size:12px;}"
        "html.__locked body{visibility:hidden;}</style>"
        "<script>(function(){"
        "var H='bf37c7c208717d6de100ce851b48273f8d5d945c8dc64fc64372087ffcf88ba9';"
        "if(sessionStorage.getItem('__unlocked')==='1')return;"
        "document.documentElement.classList.add('__locked');"
        "async function sha(s){var b=new TextEncoder().encode(s);"
        "var h=await crypto.subtle.digest('SHA-256',b);"
        "return Array.from(new Uint8Array(h)).map(x=>x.toString(16).padStart(2,'0')).join('');}"
        "document.addEventListener('DOMContentLoaded',function(){"
        "var g=document.createElement('div');g.id='__gate';"
        "g.innerHTML='<form><div>Password required</div>"
        "<input type=password autofocus autocomplete=off>"
        "<button type=submit>Unlock</button><div class=err></div></form>';"
        "document.documentElement.appendChild(g);"
        "g.querySelector('form').addEventListener('submit',async function(e){"
        "e.preventDefault();var v=g.querySelector('input').value;"
        "var h=await sha(v);"
        "if(h===H){sessionStorage.setItem('__unlocked','1');"
        "document.documentElement.classList.remove('__locked');g.remove();}"
        "else{g.querySelector('.err').textContent='Incorrect';"
        "g.querySelector('input').select();}});});})();</script>"
    )
    html = html.replace("</head>", gate + "\n</head>")
    (out_dir / "index.html").write_text(html)

    # Explore page: same transforms + same gate.
    print("Building explore.html…")
    explore_html = _transform((static_src / "explore.html").read_text(), "explore")
    explore_html = explore_html.replace("</head>", gate + "\n</head>")
    (out_dir / "explore.html").write_text(explore_html)

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
