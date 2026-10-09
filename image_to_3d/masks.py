"""Object-only mode for the multi-view path: per-photo subject masks and visual-hull pruning.

Gaussian Splatting reproduces every pixel of every photo, background included. A plain
background has no texture to pin its depth, so it turns into floating blobs. Masking the
subject in each photo lets the trainer (a) composite the background to a random colour so
off-object Gaussians are driven transparent, and (b) delete Gaussians that project outside
the silhouette in several views (the visual hull).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .scene import Scene


def compute_masks(scene: Scene, out_dir: str | Path, *, segmenter: str = "auto",
                  progress=None) -> dict[str, Path]:
    """Segment the subject in every camera's image; returns ``{camera name: mask png}``.

    Masks are cached in ``out_dir`` so a re-run is free.
    """
    from .single_image import segment, tighten_mask

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for i, cam in enumerate(scene.cameras):
        dst = out / (Path(cam.name).stem + ".png")
        paths[cam.name] = dst
        if dst.exists():
            continue
        if progress:
            progress(f"segmenting photo {i + 1}/{len(scene.cameras)}", i / max(1, len(scene.cameras)))
        bgr = cv2.imread(cam.image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(cam.image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask, _ = segment(rgb, segmenter)
        mask = tighten_mask(mask)
        cv2.imwrite(str(dst), (mask * 255).astype(np.uint8))
    return paths


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read a mask png as float32 in [0,1], optionally resized to ``(width, height)``."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if size is not None and (m.shape[1], m.shape[0]) != size:
        m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    return m.astype(np.float32) / 255.0


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate((mask > 0.5).astype(np.uint8), k).astype(np.float32)


def outside_votes(points: np.ndarray, cameras: list[Camera], masks: list[np.ndarray], *,
                  margin_frac: float = 0.015) -> np.ndarray:
    """Count, per point, the cameras in whose (dilated) mask it does NOT fall.

    The subject is in frame in every photo, so a point behind a camera or outside its image is
    also "outside" for that camera. ``margin_frac`` dilates each mask by that fraction of the
    image width, so segmentation jitter does not eat the subject.
    """
    pts = np.asarray(points, dtype=np.float64)
    votes = np.zeros(len(pts), dtype=np.int32)
    for cam, mask in zip(cameras, masks):
        h, w = mask.shape
        m = dilate_mask(mask, int(round(margin_frac * w)))
        uv, z = cam.project(pts)
        u = np.round(uv[:, 0] * (w / cam.width) - 0.5).astype(int)
        v = np.round(uv[:, 1] * (h / cam.height) - 0.5).astype(int)
        inside_img = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        out = np.ones(len(pts), dtype=bool)  # off-frame or behind the camera counts as outside
        out[inside_img] = m[v[inside_img], u[inside_img]] < 0.5
        votes += out
    return votes


def hull_filter(points: np.ndarray, cameras: list[Camera], masks: list[np.ndarray], *,
                min_votes: int = 2, margin_frac: float = 0.015) -> np.ndarray:
    """Boolean keep-mask: points seen outside the subject in fewer than ``min_votes`` views."""
    return outside_votes(points, cameras, masks, margin_frac=margin_frac) < min_votes
