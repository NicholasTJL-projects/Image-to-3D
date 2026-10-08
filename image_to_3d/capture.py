"""Stage 1: turn a 2D capture (video file, webcam stream or photo folder) into a
clean set of frames for reconstruction.

Good input for SfM / Gaussian Splatting means:
* many views (60-300) with ~70% overlap between neighbours,
* sharp frames (we score blur with the variance of the Laplacian and drop the worst),
* a static scene, constant exposure, no rolling-shutter whip pans.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def blur_score(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian: higher means sharper. Typical sharp photos > 100."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_max(image: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale so the longest side is at most ``max_side`` (keeps aspect ratio)."""
    if not max_side:
        return image
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class CaptureConfig:
    every_nth: int = 1          # keep one frame every n (video/webcam)
    max_frames: int | None = None
    max_side: int | None = 1600  # resize so the longest edge is <= this (None keeps original)
    min_blur: float = 0.0        # drop frames with Laplacian variance below this
    keep_sharpest_ratio: float = 1.0  # after min_blur, keep only this fraction of sharpest frames
    jpeg_quality: int = 95


def iter_video_frames(source: str | int, every_nth: int = 1, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video path or a webcam index."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    try:
        i = kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_nth == 0:
                yield frame
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def iter_folder_images(folder: str | Path) -> Iterator[np.ndarray]:
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                yield img


def select_frames(frames: Iterable[np.ndarray], cfg: CaptureConfig) -> list[np.ndarray]:
    """Apply resizing and blur filtering; returns the frames to keep, in order."""
    scored: list[tuple[float, int, np.ndarray]] = []
    for idx, f in enumerate(frames):
        f = resize_max(f, cfg.max_side)
        s = blur_score(f)
        if s >= cfg.min_blur:
            scored.append((s, idx, f))
    if cfg.keep_sharpest_ratio < 1.0 and scored:
        n_keep = max(1, int(round(len(scored) * cfg.keep_sharpest_ratio)))
        scored = sorted(scored, key=lambda t: -t[0])[:n_keep]
        scored.sort(key=lambda t: t[1])  # restore temporal order
    return [f for _, _, f in scored]


def capture(source: str | int, out_dir: str | Path, cfg: CaptureConfig | None = None, *, clean: bool = False) -> list[Path]:
    """Extract frames from ``source`` into ``out_dir/`` as ``frame_00000.jpg``...

    ``source`` may be a video file, a webcam index (``0``), or a folder of photos.
    """
    cfg = cfg or CaptureConfig()
    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, str) and Path(source).is_dir():
        frames = iter_folder_images(source)
    else:
        src: str | int = int(source) if isinstance(source, str) and source.isdigit() else source
        frames = iter_video_frames(src, cfg.every_nth, cfg.max_frames)

    kept = select_frames(frames, cfg)
    paths = []
    for i, f in enumerate(kept):
        p = out_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        paths.append(p)
    return paths
