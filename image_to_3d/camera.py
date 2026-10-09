"""Pinhole camera model shared by every stage of the pipeline.

Conventions
-----------
* ``R`` and ``t`` map world coordinates to camera coordinates: ``x_cam = R @ x_world + t``
  (the same convention COLMAP uses).
* The camera looks down +Z, +X is right, +Y is down (OpenCV convention).
* ``K`` is the 3x3 intrinsic matrix with ``fx, fy, cx, cy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion ``(w, x, y, z)`` -> 3x3 rotation matrix (COLMAP ordering)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion ``(w, x, y, z)``."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


@dataclass
class Camera:
    """A posed pinhole camera and the image it captured."""

    width: int
    height: int
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 world->camera rotation
    t: np.ndarray  # 3 world->camera translation
    image_path: str | None = None
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    # -- intrinsics -----------------------------------------------------
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    # -- extrinsics -----------------------------------------------------
    @property
    def world_to_camera(self) -> np.ndarray:
        """4x4 matrix mapping world points into the camera frame."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    @property
    def camera_to_world(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R.T
        M[:3, 3] = -self.R.T @ self.t
        return M

    @property
    def center(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction (world frame, unit length)."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    # -- projection -----------------------------------------------------
    def to_camera(self, xyz: np.ndarray) -> np.ndarray:
        """World points ``(N,3)`` -> camera-frame points ``(N,3)``."""
        xyz = np.asarray(xyz, dtype=np.float64)
        return xyz @ self.R.T + self.t

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points ``(N,3)`` -> pixel coords ``(N,2)`` and depths ``(N,)``.

        Points behind the camera get a non-positive depth; callers decide how to
        treat them.
        """
        cam = self.to_camera(xyz)
        z = cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.fx * cam[:, 0] / z_safe + self.cx
        v = self.fy * cam[:, 1] / z_safe + self.cy
        return np.stack([u, v], axis=1), z

    def scaled(self, factor: float) -> "Camera":
        """Return a copy with the image resolution multiplied by ``factor``."""
        K = self.K.copy()
        K[0, :] *= factor
        K[1, :] *= factor
        return Camera(
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
            K=K,
            R=self.R.copy(),
            t=self.t.copy(),
            image_path=self.image_path,
            name=self.name,
            extra=dict(self.extra),
        )

    # -- constructors ---------------------------------------------------
    @staticmethod
    def look_at(
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = (0.0, -1.0, 0.0),
        *,
        width: int,
        height: int,
        focal: float,
        name: str = "",
    ) -> "Camera":
        """Build a camera at ``eye`` looking towards ``target``.

        ``up`` defaults to -Y because image rows grow downwards in the OpenCV
        convention, so world "up" maps to image "up".
        """
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(-up, z)  # image-right = (-up) x forward, with up pointing to -Y rows
        if np.linalg.norm(x) < 1e-9:
            x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_c2w = np.stack([x, y, z], axis=1)  # columns are camera axes in world coords
        R = R_c2w.T
        t = -R @ eye
        K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]])
        return Camera(width=width, height=height, K=K, R=R, t=t, name=name)
