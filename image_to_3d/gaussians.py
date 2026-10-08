"""The 3D Gaussian Splatting scene representation.

Each Gaussian is defined by
* ``means``        (N,3)  centre in world space
* ``log_scales``   (N,3)  log of the per-axis standard deviation
* ``quats``        (N,4)  rotation as a unit quaternion ``(w, x, y, z)``
* ``logit_opacity``(N,1)  opacity before the sigmoid
* ``features_dc``  (N,3)  degree-0 spherical-harmonic colour coefficients
* ``features_rest``(N,K,3) higher-order SH coefficients (K = (deg+1)^2 - 1)

Storage follows the reference implementation's PLY layout so files are
interchangeable with the official viewers (SIBR, antimatter15/splat, SuperSplat...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814  # 1 / (2*sqrt(pi))


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


def sh_to_rgb(sh: np.ndarray) -> np.ndarray:
    return np.asarray(sh, dtype=np.float64) * SH_C0 + 0.5


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 1e-6, 1 - 1e-6)
    return np.log(y / (1 - y))


def quats_to_rotmats(q: np.ndarray) -> np.ndarray:
    """``(N,4)`` quaternions (w,x,y,z) -> ``(N,3,3)`` rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class GaussianCloud:
    means: np.ndarray
    log_scales: np.ndarray
    quats: np.ndarray
    logit_opacity: np.ndarray
    features_dc: np.ndarray
    features_rest: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.means)
        self.means = np.asarray(self.means, dtype=np.float32).reshape(n, 3)
        self.log_scales = np.asarray(self.log_scales, dtype=np.float32).reshape(n, 3)
        self.quats = np.asarray(self.quats, dtype=np.float32).reshape(n, 4)
        self.logit_opacity = np.asarray(self.logit_opacity, dtype=np.float32).reshape(n, 1)
        self.features_dc = np.asarray(self.features_dc, dtype=np.float32).reshape(n, 3)
        if self.features_rest is None:
            self.features_rest = np.zeros((n, 0, 3), dtype=np.float32)
        self.features_rest = np.asarray(self.features_rest, dtype=np.float32).reshape(n, -1, 3)

    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        k = self.features_rest.shape[1] + 1
        return int(round(np.sqrt(k))) - 1

    # -- derived quantities ---------------------------------------------
    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.log_scales)

    @property
    def opacities(self) -> np.ndarray:
        return sigmoid(self.logit_opacity)[:, 0]

    @property
    def colors(self) -> np.ndarray:
        """View-independent base colour in [0,1], ``(N,3)``."""
        return np.clip(sh_to_rgb(self.features_dc), 0.0, 1.0)

    def covariances(self) -> np.ndarray:
        """World-space 3x3 covariance per Gaussian: ``R S S^T R^T``."""
        R = quats_to_rotmats(self.quats)
        S = self.scales.astype(np.float64)
        RS = R * S[:, None, :]
        return RS @ RS.transpose(0, 2, 1)

    # -- construction ---------------------------------------------------
    @staticmethod
    def from_points(
        xyz: np.ndarray,
        rgb: np.ndarray | None = None,
        *,
        initial_opacity: float = 0.1,
        sh_degree: int = 0,
        min_scale: float = 1e-4,
    ) -> "GaussianCloud":
        """Initialise one isotropic Gaussian per SfM point.

        The scale is the mean distance to the 3 nearest neighbours, which is how
        the reference implementation seeds the scene.
        """
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = len(xyz)
        if n == 0:
            raise ValueError("Cannot initialise Gaussians from an empty point cloud")
        if rgb is None:
            rgb = np.full((n, 3), 0.5)
        rgb = np.asarray(rgb, dtype=np.float64).reshape(n, 3)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        if n >= 4:
            tree = cKDTree(xyz)
            dist, _ = tree.query(xyz, k=4)
            mean_dist = np.sqrt(np.clip((dist[:, 1:] ** 2).mean(axis=1), min_scale**2, None))
        else:
            mean_dist = np.full(n, 0.1)
        log_scales = np.repeat(np.log(mean_dist)[:, None], 3, axis=1)
        quats = np.zeros((n, 4))
        quats[:, 0] = 1.0
        k_rest = (sh_degree + 1) ** 2 - 1
        return GaussianCloud(
            means=xyz,
            log_scales=log_scales,
            quats=quats,
            logit_opacity=np.full((n, 1), inverse_sigmoid(np.array(initial_opacity))),
            features_dc=rgb_to_sh(rgb),
            features_rest=np.zeros((n, k_rest, 3)),
        )

    def subset(self, mask: np.ndarray) -> "GaussianCloud":
        return GaussianCloud(
            self.means[mask], self.log_scales[mask], self.quats[mask],
            self.logit_opacity[mask], self.features_dc[mask], self.features_rest[mask],
        )

    # -- PLY I/O (reference-implementation compatible) --------------------
    def save_ply(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self)
        rest = self.features_rest.transpose(0, 2, 1).reshape(n, -1)  # (N, 3*K): channel-major like the reference
        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        names += [f"f_rest_{i}" for i in range(rest.shape[1])]
        names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        data = np.concatenate(
            [self.means, np.zeros((n, 3), np.float32), self.features_dc, rest,
             self.logit_opacity, self.log_scales, self.quats], axis=1,
        ).astype(np.float32)
        arr = np.empty(n, dtype=[(nm, "f4") for nm in names])
        for i, nm in enumerate(names):
            arr[nm] = data[:, i]
        PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))

    @staticmethod
    def load_ply(path: str | Path) -> "GaussianCloud":
        v = PlyData.read(str(path))["vertex"]
        n = v.count
        rest_names = sorted((nm for nm in v.data.dtype.names if nm.startswith("f_rest_")),
                            key=lambda s: int(s.split("_")[-1]))
        rest = np.stack([v[nm] for nm in rest_names], axis=1) if rest_names else np.zeros((n, 0))
        rest = rest.reshape(n, 3, -1).transpose(0, 2, 1)
        return GaussianCloud(
            means=np.stack([v["x"], v["y"], v["z"]], axis=1),
            log_scales=np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
            quats=np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
            logit_opacity=np.asarray(v["opacity"])[:, None],
            features_dc=np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1),
            features_rest=rest,
        )

    def export_point_cloud(self, path: str | Path, min_opacity: float = 0.3) -> None:
        """Write Gaussian centres as a plain coloured point cloud (viewable anywhere)."""
        keep = self.opacities >= min_opacity
        xyz = self.means[keep]
        rgb = (self.colors[keep] * 255).astype(np.uint8)
        arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
