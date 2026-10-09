"""Glue between the stages: a ``workspace`` folder holds everything for one capture.

    workspace/
      images/                 frames written by `capture`
      colmap/                 COLMAP database + raw sparse model (`sfm`)
      undistorted/            pinhole images + model used for training (`sfm`)
      scene/                  camera poses + sparse points in our own format (`sfm` / `init`)
      output/init.ply         Gaussians seeded from the sparse points (`init`)
      output/point_cloud.ply  trained Gaussians (`train`)
      renders/                images rendered by `render`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .colmap_io import load_colmap_scene
from .gaussians import GaussianCloud
from .scene import Scene

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read a YAML config; missing file -> the packaged default."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def load_scene(workspace: str | Path, *, sparse: str | Path | None = None, images: str | Path | None = None) -> Scene:
    """Load the scene from ``workspace/scene`` or convert a COLMAP model into one."""
    workspace = Path(workspace)
    if sparse is not None:
        images = Path(images) if images else workspace / "images"
        scene = load_colmap_scene(sparse, images)
        scene.save(workspace / "scene")
        return scene
    if (workspace / "scene" / "cameras.json").exists():
        return Scene.load(workspace / "scene")
    for sp, im in ((workspace / "undistorted" / "sparse", workspace / "undistorted" / "images"),
                   (workspace / "colmap" / "sparse" / "0", workspace / "images"),
                   (workspace / "sparse" / "0", workspace / "images"),
                   (workspace / "sparse", workspace / "images")):
        if (sp / "cameras.bin").exists() or (sp / "cameras.txt").exists():
            scene = load_colmap_scene(sp, im)
            scene.save(workspace / "scene")
            return scene
    raise FileNotFoundError(f"No scene found in {workspace}. Run `image-to-3d sfm` first or pass --sparse.")


def init_gaussians(scene: Scene, workspace: str | Path, *, initial_opacity: float = 0.1, sh_degree: int = 0,
                   random_points: int = 0, seed: int = 0, masks: dict | None = None,
                   hull_min_votes: int = 2) -> GaussianCloud:
    """Seed Gaussians from the sparse SfM points (optionally padded with random points).

    With ``masks`` (``{camera name: mask path or array}``, see :mod:`image_to_3d.masks`) only
    points inside the subject's visual hull are kept, so the background never gets seeded.
    """
    xyz, rgb = scene.points_xyz, scene.points_rgb
    if masks and len(xyz):
        from .masks import hull_filter, load_mask

        cams = [c for c in scene.cameras if c.name in masks]
        ms = [masks[c.name] if isinstance(masks[c.name], np.ndarray) else load_mask(masks[c.name]) for c in cams]
        keep = hull_filter(xyz, cams, ms, min_votes=hull_min_votes)
        if keep.sum() >= 10:
            xyz, rgb = xyz[keep], rgb[keep]
    if random_points > 0 or len(xyz) == 0:
        rng = np.random.default_rng(seed)
        n = random_points or 10_000
        centre, radius = scene.centroid, scene.radius
        extra = centre + rng.uniform(-radius, radius, size=(n, 3))
        xyz = np.concatenate([xyz, extra], 0) if len(xyz) else extra
        rgb = np.concatenate([rgb, np.full((n, 3), 128, np.uint8)], 0) if len(rgb) else np.full((n, 3), 128, np.uint8)
    cloud = GaussianCloud.from_points(xyz, rgb, initial_opacity=initial_opacity, sh_degree=sh_degree)
    out = Path(workspace) / "output"
    out.mkdir(parents=True, exist_ok=True)
    cloud.save_ply(out / "init.ply")
    return cloud


def orbit_cameras(scene: Scene, n: int = 60, *, elevation_deg: float = 10.0, width: int | None = None,
                  height: int | None = None):
    """Cameras on a circle around the scene, for turntable renders.

    The orbit centre is the median of the sparse points (robust to outliers) or, without points,
    the point closest to all camera viewing rays. The radius is the median camera distance to it
    and "up" is the average camera up vector, so the orbit sits where the real cameras were.
    """
    from .camera import Camera

    ref = scene.cameras[0]
    width = width or ref.width
    height = height or ref.height
    focal = ref.fx * width / ref.width
    centers = np.stack([c.center for c in scene.cameras])
    fwds = np.stack([c.forward for c in scene.cameras])

    if len(scene.points_xyz) >= 10:
        target = np.median(scene.points_xyz, axis=0)
    else:  # least-squares intersection of the viewing rays
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for c0, d in zip(centers, fwds):
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ c0
        target = np.linalg.lstsq(A, b, rcond=None)[0] if np.linalg.matrix_rank(A) == 3 else centers.mean(0)

    up = -np.mean([c.R.T @ np.array([0.0, 1.0, 0.0]) for c in scene.cameras], axis=0)  # image rows grow down
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, -1.0, 0.0])
    up /= np.linalg.norm(up)
    radius = float(np.median(np.linalg.norm(centers - target, axis=1)))

    # orthonormal basis in the plane perpendicular to `up`, starting at the first camera's azimuth
    a = centers[0] - target
    a -= up * (a @ up)
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(up, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    bvec = np.cross(up, a)
    cams = []
    el = np.deg2rad(elevation_deg)
    for i in range(n):
        ang = 2 * np.pi * i / n
        eye = target + radius * (np.cos(el) * (np.cos(ang) * a + np.sin(ang) * bvec) + np.sin(el) * up)
        cams.append(Camera.look_at(eye, target, up=up, width=width, height=height, focal=focal, name=f"orbit_{i:04d}"))
    return cams
