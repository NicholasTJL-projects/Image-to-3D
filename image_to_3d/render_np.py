"""Reference NumPy rasteriser for Gaussian clouds (forward pass only).

Not fast, but dependency-free and exact: it is the ground truth for the tests,
the preview renderer when PyTorch is not installed, and the implementation to
read if you want to understand how splatting works.

For each Gaussian we
1. transform the mean into the camera frame and project it,
2. push the 3D covariance through the projection's Jacobian to get a 2D
   covariance (the EWA splatting approximation),
3. alpha-composite front-to-back inside a 3-sigma bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .gaussians import GaussianCloud


@dataclass
class ProjectedGaussians:
    means2d: np.ndarray    # (M,2) pixel coordinates
    cov2d: np.ndarray      # (M,2,2)
    depths: np.ndarray     # (M,)
    radii: np.ndarray      # (M,) bounding radius in pixels
    index: np.ndarray      # (M,) index into the source cloud, sorted front-to-back


def project_gaussians(cloud: GaussianCloud, cam: Camera, *, near: float = 0.01, dilation: float = 0.3) -> ProjectedGaussians:
    """Project all Gaussians in front of the camera; result is depth-sorted."""
    R, t = cam.R, cam.t
    means_cam = cloud.means.astype(np.float64) @ R.T + t
    z = means_cam[:, 2]
    visible = z > near
    idx = np.nonzero(visible)[0]
    mc = means_cam[idx]
    z = z[idx]
    x, y = mc[:, 0], mc[:, 1]

    # Clamp the view ray to the frustum (+30%) before building the Jacobian, as the
    # reference implementation does; it keeps the linearisation sane at the edges.
    tan_x = 1.3 * (cam.width / (2 * cam.fx))
    tan_y = 1.3 * (cam.height / (2 * cam.fy))
    xz = np.clip(x / z, -tan_x, tan_x) * z
    yz = np.clip(y / z, -tan_y, tan_y) * z

    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = cam.fx / z
    J[:, 0, 2] = -cam.fx * xz / z**2
    J[:, 1, 1] = cam.fy / z
    J[:, 1, 2] = -cam.fy * yz / z**2

    cov3d = cloud.covariances()[idx]
    JW = J @ R[None]
    cov2d = JW @ cov3d @ JW.transpose(0, 2, 1)
    cov2d[:, 0, 0] += dilation
    cov2d[:, 1, 1] += dilation

    u = cam.fx * x / z + cam.cx
    v = cam.fy * y / z + cam.cy
    means2d = np.stack([u, v], axis=1)

    # radius from the largest eigenvalue of the 2D covariance
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    lambda_max = mid + np.sqrt(np.clip(mid**2 - det, 0.1, None))
    radii = np.ceil(3.0 * np.sqrt(lambda_max))

    order = np.argsort(z, kind="stable")
    return ProjectedGaussians(means2d[order], cov2d[order], z[order], radii[order], idx[order])


def render(
    cloud: GaussianCloud,
    cam: Camera,
    *,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    return_depth: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise ``cloud`` from ``cam``.

    Returns an ``(H,W,3)`` float image in [0,1]; with ``return_depth`` also the
    ``(H,W)`` expected depth and ``(H,W)`` accumulated alpha.
    """
    H, W = cam.height, cam.width
    color = np.zeros((H, W, 3))
    depth = np.zeros((H, W))
    T = np.ones((H, W))  # remaining transmittance

    proj = project_gaussians(cloud, cam)
    colors = cloud.colors[proj.index]
    opac = cloud.opacities[proj.index]

    for k in range(len(proj.index)):
        if T.min() < 1e-4 and k > 0 and (T < 1e-4).all():
            break
        mx, my = proj.means2d[k]
        r = proj.radii[k]
        x0, x1 = int(max(0, np.floor(mx - r))), int(min(W, np.ceil(mx + r) + 1))
        y0, y1 = int(max(0, np.floor(my - r))), int(min(H, np.ceil(my + r) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        cov = proj.cov2d[k]
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
        if det <= 0:
            continue
        a, b, c = cov[1, 1] / det, -cov[0, 1] / det, cov[0, 0] / det  # inverse covariance
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs + 0.5 - mx
        dy = ys + 0.5 - my
        power = -0.5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy)
        alpha = np.minimum(0.99, opac[k] * np.exp(power))
        alpha[(power > 0) | (alpha < 1.0 / 255.0)] = 0.0
        Tk = T[y0:y1, x0:x1]
        w = alpha * Tk
        color[y0:y1, x0:x1] += w[..., None] * colors[k]
        depth[y0:y1, x0:x1] += w * proj.depths[k]
        T[y0:y1, x0:x1] = Tk * (1.0 - alpha)

    acc = 1.0 - T
    color += T[..., None] * np.asarray(background, dtype=np.float64)
    if return_depth:
        depth = np.where(acc > 1e-6, depth / np.maximum(acc, 1e-6), 0.0)
        return color, depth, acc
    return color


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
