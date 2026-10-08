"""Synthetic scenes for tests and the ``demo`` command: a known Gaussian cloud and
a ring of cameras around it, so the whole pipeline (minus COLMAP) can run without
any real footage."""

from __future__ import annotations

import numpy as np

from .camera import Camera
from .gaussians import GaussianCloud, inverse_sigmoid, rgb_to_sh
from .scene import Scene


def ring_cameras(n: int, radius: float = 4.0, height: float = 0.8, *, width: int = 64, height_px: int = 64,
                 focal: float | None = None, target=(0.0, 0.0, 0.0)) -> list[Camera]:
    """``n`` cameras on a circle in the XZ plane, all looking at ``target``."""
    focal = focal or 0.9 * width
    cams = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        eye = np.array([radius * np.cos(ang), -height, radius * np.sin(ang)])  # -Y is "up" (image rows grow down)
        cams.append(Camera.look_at(eye, np.asarray(target, dtype=float), width=width, height=height_px,
                                   focal=focal, name=f"cam_{i:03d}"))
    return cams


def blob_cloud(n: int = 200, seed: int = 0, extent: float = 1.0, scale: float = 0.12, opacity: float = 0.8) -> GaussianCloud:
    """Random anisotropic Gaussians with colours derived from position."""
    rng = np.random.default_rng(seed)
    means = rng.uniform(-extent, extent, size=(n, 3))
    log_scales = np.log(scale * rng.uniform(0.6, 1.4, size=(n, 3)))
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    rgb = (means / extent + 1) / 2  # position -> colour, so views are distinguishable
    return GaussianCloud(
        means=means, log_scales=log_scales, quats=quats,
        logit_opacity=np.full((n, 1), inverse_sigmoid(np.array(opacity))),
        features_dc=rgb_to_sh(rgb),
    )


def synthetic_scene(n_gaussians: int = 200, n_cameras: int = 24, *, width: int = 64, height: int = 64,
                    seed: int = 0, point_noise: float = 0.05) -> tuple[GaussianCloud, Scene]:
    """Return the ground-truth cloud and a Scene whose sparse points are a noisy
    copy of the Gaussian centres (imitating what SfM would give us)."""
    gt = blob_cloud(n_gaussians, seed=seed)
    rng = np.random.default_rng(seed + 1)
    pts = gt.means + rng.normal(scale=point_noise, size=gt.means.shape)
    rgb = (gt.colors * 255).astype(np.uint8)
    cams = ring_cameras(n_cameras, width=width, height_px=height)
    return gt, Scene(cams, pts, rgb)
