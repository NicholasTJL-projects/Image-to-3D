"""A reconstructed scene: posed cameras + sparse points, independent of COLMAP."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import Camera, qvec2rotmat, rotmat2qvec


@dataclass
class Scene:
    cameras: list[Camera]
    points_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    points_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))

    def __len__(self) -> int:
        return len(self.cameras)

    @property
    def centroid(self) -> np.ndarray:
        centers = np.stack([c.center for c in self.cameras])
        return centers.mean(axis=0)

    @property
    def radius(self) -> float:
        """Radius of the camera rig around its centroid; sets the 3DGS spatial lr scale."""
        centers = np.stack([c.center for c in self.cameras])
        return float(np.linalg.norm(centers - self.centroid, axis=1).max()) * 1.1

    def split(self, every_nth: int = 8) -> tuple["Scene", "Scene"]:
        """Deterministic train / test split by holding out every n-th camera."""
        train = [c for i, c in enumerate(self.cameras) if i % every_nth != 0]
        test = [c for i, c in enumerate(self.cameras) if i % every_nth == 0]
        if not train:  # tiny scenes: train on everything
            train = list(self.cameras)
        return (
            Scene(train, self.points_xyz, self.points_rgb),
            Scene(test, self.points_xyz, self.points_rgb),
        )

    # -- serialisation (a small JSON + NPZ pair, so no COLMAP is needed later) --
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cams = []
        for c in self.cameras:
            cams.append(
                {
                    "name": c.name,
                    "image_path": c.image_path,
                    "width": c.width,
                    "height": c.height,
                    "K": c.K.tolist(),
                    "qvec": rotmat2qvec(c.R).tolist(),
                    "t": c.t.tolist(),
                }
            )
        (path / "cameras.json").write_text(json.dumps(cams, indent=1))
        np.savez_compressed(path / "points.npz", xyz=self.points_xyz, rgb=self.points_rgb)

    @staticmethod
    def load(path: str | Path) -> "Scene":
        path = Path(path)
        cams = json.loads((path / "cameras.json").read_text())
        cameras = [
            Camera(
                width=c["width"],
                height=c["height"],
                K=np.array(c["K"]),
                R=qvec2rotmat(np.array(c["qvec"])),
                t=np.array(c["t"]),
                image_path=c.get("image_path"),
                name=c.get("name", ""),
            )
            for c in cams
        ]
        pts = np.load(path / "points.npz")
        return Scene(cameras, pts["xyz"], pts["rgb"])
