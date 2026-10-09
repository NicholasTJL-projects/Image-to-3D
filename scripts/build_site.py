"""Assemble the static site deployed to Vercel (or any static host) into ./site.

    python scripts/build_site.py --gallery /path/to/gallery_assets

The site has two parts:
* ``/``      the results gallery (meshes, splats and depth maps in interactive viewers)
* ``/app/``  the web app's front end; without a backend it shows how to run one

The gallery page source lives in site/gallery.html; the app front end is copied from the
package so it never drifts from what the server serves.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "image_to_3d" / "web" / "static"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", help="folder with manifest.json and assets/ for the gallery (optional)")
    args = ap.parse_args()
    site = ROOT / "site"
    app = site / "app"
    if app.exists():
        shutil.rmtree(app)
    shutil.copytree(STATIC, app)
    (app / "config.js").write_text(
        "// Set to the URL of a hosted Image-to-3D API to make this page work on a static host.\n"
        "window.IMAGE_TO_3D_API = '';\n")
    # the gallery shares the viewer libraries with the app
    for name in ("three.module.js", "three.core.js", "OrbitControls.js", "GLTFLoader.js", "gaussian-splats-3d.module.js"):
        shutil.copy(STATIC / "vendor" / name, site / name)
    (site / "utils").mkdir(exist_ok=True)
    for f in (STATIC / "vendor" / "utils").glob("*.js"):
        shutil.copy(f, site / "utils" / f.name)
    if args.gallery:
        g = Path(args.gallery)
        shutil.copy(g / "manifest.json", site / "manifest.json")
        if (site / "assets").exists():
            shutil.rmtree(site / "assets")
        shutil.copytree(g / "assets", site / "assets")
    body = (site / "gallery.html").read_text()
    (site / "index.html").write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<style>[hidden]{display:none!important}</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    print(f"site assembled in {site}")


if __name__ == "__main__":
    main()
