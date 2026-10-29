"""Copy the newest storefront page and its 3D assets into app/static.

The backend serves app/static/index.html. The versioned pages live in
../frontend, and the one to serve is the highest version by the numbers in its
name (v15_gemini_v3 beats v15_gemini_v2, v16_* beats v15_*), never the file
date. start_stylemate.command does this on a laptop; on a host this script is
the build step, so a pushed new version is what goes live.

    python backend/sync_static.py

Standard library only.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
FRONTEND = BACKEND.parent / "frontend"
STATIC = BACKEND / "app" / "static"
ASSETS = ("three.module.min.js", "OrbitControls.js", "stylemate-room.js")


def highest(kind: str) -> Path | None:
    best, best_key = None, (-1, -1)
    for f in FRONTEND.glob(f"*_{kind}_v*.html"):
        m = re.search(rf"_v(\d+)_{kind}_v(\d+)\.html$", f.name)
        key = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if key > best_key:
            best, best_key = f, key
    return best


def main() -> int:
    page = highest("gemini") or highest("ollama")
    if not page:
        print(f"sync_static: no versioned page in {FRONTEND}; serving app/static as committed")
        return 0
    STATIC.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(page, STATIC / "index.html")
    print(f"sync_static: serving {page.name}")
    for a in ASSETS:
        src = FRONTEND / a
        if src.is_file():
            shutil.copyfile(src, STATIC / a)
    missing = [a for a in ASSETS if not (STATIC / a).is_file()]
    if missing:
        print("sync_static: 3D assets missing - the fitting room will not load: " + ", ".join(missing))
        return 1
    print("sync_static: 3D assets present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
