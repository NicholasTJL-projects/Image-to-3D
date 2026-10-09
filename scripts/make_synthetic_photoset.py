"""Render a synthetic multi-angle photo set for testing the multi-view pipeline.

Renders a richly textured scene (box, torus knot, cylinder on a patterned floor) with Three.js
in headless Chromium from two rings of camera positions, and writes JPEGs that COLMAP can
pose without any real footage.

    pip install playwright && playwright install chromium
    python scripts/make_synthetic_photoset.py out/photos --count 36
    image-to-3d capture out/photos out/ws && image-to-3d sfm out/ws --matcher exhaustive
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="folder for photo_###.jpg")
    ap.add_argument("--count", type=int, default=36, help="total photos (2/3 low ring, 1/3 high ring)")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"), help="executable path override")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    n_low = args.count * 2 // 3
    n_high = args.count - n_low
    try:
        with sync_playwright() as p:
            launch = dict(headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            if args.chromium:
                launch["executable_path"] = args.chromium
            browser = p.chromium.launch(**launch)
            page = browser.new_page(viewport={"width": 800, "height": 600})
            page.goto(f"http://127.0.0.1:{args.port}/scripts/synthetic_photoset/scene.html")
            page.wait_for_function("window.ready === true")
            k = 0
            for ring, n in ((0, n_low), (1, n_high)):
                for i in range(n):
                    data = page.evaluate(f"shoot({i}, {n}, {ring})")
                    (out / f"photo_{k:03d}.jpg").write_bytes(base64.b64decode(data.split(",", 1)[1]))
                    k += 1
            browser.close()
    finally:
        srv.shutdown()
    print(f"wrote {k} photos to {out}")


if __name__ == "__main__":
    main()
