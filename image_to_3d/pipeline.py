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
                   random_points: int = 0, seed: int = 0) -> GaussianCloud:
    """Seed Gaussians from the sparse SfM points (optionally padded with random points)."""
    xyz, rgb = scene.points_xyz, scene.points_rgb
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
    """Cameras on a circle through the training rig, for turntable renders."""
    from .camera import Camera

    ref = scene.cameras[0]
    width = width or ref.width
    height = height or ref.height
    focal = ref.fx * width / ref.width
    centre = scene.centroid
    # average viewing target: where the cameras look, ~ scene radius ahead of the rig centroid
    fwd = np.mean([c.forward for c in scene.cameras], axis=0)
    target = centre + fwd * scene.radius if np.linalg.norm(fwd) > 1e-3 else centre
    up = -np.mean([c.R.T @ np.array([0, 1.0, 0]) for c in scene.cameras], axis=0)
    up /= np.linalg.norm(up)
    radius = np.linalg.norm(centre - target) or scene.radius
    # orthonormal basis in the plane perpendicular to `up`
    a = centre - target
    a -= up * (a @ up)
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(up, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    cams = []
    el = np.deg2rad(elevation_deg)
    for i in range(n):
        ang = 2 * np.pi * i / n
        eye = target + radius * (np.cos(el) * (np.cos(ang) * a + np.sin(ang) * b) + np.sin(el) * up)
        cams.append(Camera.look_at(eye, target, up=-up, width=width, height=height, focal=focal, name=f"orbit_{i:04d}"))
    return cams
