"""Differentiable Gaussian rasteriser in pure PyTorch.

The maths mirrors :mod:`image_to_3d.render_np`; the difference is that every
step is a tensor op so autograd can back-propagate the photometric loss into
the Gaussian parameters. Pixels are processed in tiles and, within a tile, all
candidate Gaussians are evaluated densely and composited with a cumulative
product, which is what makes the operation differentiable without a custom
CUDA kernel.

It runs on CPU or any torch device. It is roughly two orders of magnitude slower
than the CUDA rasterisers (``gsplat``, ``diff-gaussian-rasterization``); use
:func:`render` with ``backend="gsplat"`` when that package and a GPU are available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .camera import Camera

SH_C0 = 0.28209479177387814


def quats_to_rotmats(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
        ],
        1,
    )


def covariances_3d(log_scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    R = quats_to_rotmats(quats)
    RS = R * torch.exp(log_scales)[:, None, :]
    return RS @ RS.transpose(1, 2)


@dataclass
class TorchCamera:
    """Camera tensors on the right device."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: torch.Tensor  # 3x3
    t: torch.Tensor  # 3

    @staticmethod
    def from_camera(cam: Camera, device, dtype=torch.float32) -> "TorchCamera":
        return TorchCamera(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy,
                           torch.as_tensor(cam.R, dtype=dtype, device=device),
                           torch.as_tensor(cam.t, dtype=dtype, device=device))


@dataclass
class RenderOutput:
    image: torch.Tensor      # (H,W,3)
    alpha: torch.Tensor      # (H,W)
    depth: torch.Tensor      # (H,W)
    means2d: torch.Tensor    # (N,2) screen-space means with retained grad (for densification)
    radii: torch.Tensor      # (N,) 0 for culled Gaussians
    visible: torch.Tensor    # (N,) bool


def project(means, log_scales, quats, cam: TorchCamera, near: float = 0.01, dilation: float = 0.3):
    means_cam = means @ cam.R.T + cam.t
    x, y, z = means_cam.unbind(1)
    visible = z > near
    z_safe = torch.where(visible, z, torch.ones_like(z))

    tan_x = 1.3 * cam.width / (2 * cam.fx)
    tan_y = 1.3 * cam.height / (2 * cam.fy)
    xz = torch.clamp(x / z_safe, -tan_x, tan_x) * z_safe
    yz = torch.clamp(y / z_safe, -tan_y, tan_y) * z_safe

    zero = torch.zeros_like(z)
    J = torch.stack(
        [
            torch.stack([cam.fx / z_safe, zero, -cam.fx * xz / z_safe**2], 1),
            torch.stack([zero, cam.fy / z_safe, -cam.fy * yz / z_safe**2], 1),
        ],
        1,
    )  # (N,2,3)
    JW = J @ cam.R
    cov2d = JW @ covariances_3d(log_scales, quats) @ JW.transpose(1, 2)
    cov2d = cov2d + torch.diag(torch.tensor([dilation, dilation], device=means.device, dtype=means.dtype))

    means2d = torch.stack([cam.fx * x / z_safe + cam.cx, cam.fy * y / z_safe + cam.cy], 1)

    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    lambda_max = mid + torch.sqrt(torch.clamp(mid**2 - det, min=0.1))
    radii = torch.ceil(3.0 * torch.sqrt(lambda_max))
    # cull Gaussians that are behind the camera or whose bounding box misses the image
    inside = ((means2d[:, 0] + radii) > 0) & ((means2d[:, 0] - radii) < cam.width) & \
             ((means2d[:, 1] + radii) > 0) & ((means2d[:, 1] - radii) < cam.height)
    visible = visible & inside & (det > 0)
    radii = torch.where(visible, radii, torch.zeros_like(radii))
    return means2d, cov2d, z, radii, visible


def rasterize_torch(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    logit_opacity: torch.Tensor,
    colors: torch.Tensor,
    cam: TorchCamera,
    *,
    background: torch.Tensor | None = None,
    tile: int = 32,
) -> RenderOutput:
    """Render ``(H,W,3)`` from raw Gaussian parameters. ``colors`` are RGB in [0,1]."""
    device, dtype = means.device, means.dtype
    H, W = cam.height, cam.width
    bg = torch.zeros(3, device=device, dtype=dtype) if background is None else background.to(device, dtype)

    means2d_all, cov2d, depth_all, radii, visible = project(means, log_scales, quats, cam)
    means2d_all.retain_grad() if means2d_all.requires_grad else None

    idx = torch.nonzero(visible, as_tuple=False)[:, 0]
    out_img = torch.empty(H, W, 3, device=device, dtype=dtype)
    out_alpha = torch.empty(H, W, device=device, dtype=dtype)
    out_depth = torch.empty(H, W, device=device, dtype=dtype)
    if idx.numel() == 0:
        out_img[:] = bg
        out_alpha.zero_()
        out_depth.zero_()
        return RenderOutput(out_img, out_alpha, out_depth, means2d_all, radii, visible)

    order = torch.argsort(depth_all[idx])
    idx = idx[order]
    means2d = means2d_all[idx]
    cov = cov2d[idx]
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    inv_a, inv_b, inv_c = cov[:, 1, 1] / det, -cov[:, 0, 1] / det, cov[:, 0, 0] / det
    opac = torch.sigmoid(logit_opacity[idx]).reshape(-1)
    col = colors[idx]
    dep = depth_all[idx]
    rad = radii[idx]
    bb_min = means2d - rad[:, None]
    bb_max = means2d + rad[:, None]

    rows, cols = [], []
    for y0 in range(0, H, tile):
        y1 = min(H, y0 + tile)
        row_imgs, row_alphas, row_depths = [], [], []
        for x0 in range(0, W, tile):
            x1 = min(W, x0 + tile)
            sel = (bb_max[:, 0] > x0) & (bb_min[:, 0] < x1) & (bb_max[:, 1] > y0) & (bb_min[:, 1] < y1)
            sel_idx = torch.nonzero(sel, as_tuple=False)[:, 0]
            th, tw = y1 - y0, x1 - x0
            if sel_idx.numel() == 0:
                row_imgs.append(bg.expand(th, tw, 3))
                row_alphas.append(torch.zeros(th, tw, device=device, dtype=dtype))
                row_depths.append(torch.zeros(th, tw, device=device, dtype=dtype))
                continue
            ys = torch.arange(y0, y1, device=device, dtype=dtype) + 0.5
            xs = torch.arange(x0, x1, device=device, dtype=dtype) + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            dx = gx.reshape(1, -1) - means2d[sel_idx, 0:1]  # (M,P)
            dy = gy.reshape(1, -1) - means2d[sel_idx, 1:2]
            power = -0.5 * (inv_a[sel_idx, None] * dx * dx + 2 * inv_b[sel_idx, None] * dx * dy + inv_c[sel_idx, None] * dy * dy)
            alpha = torch.clamp(opac[sel_idx, None] * torch.exp(power), max=0.99)
            alpha = torch.where((power > 0) | (alpha < 1 / 255.0), torch.zeros_like(alpha), alpha)
            trans = torch.cumprod(1 - alpha, dim=0)
            T = torch.cat([torch.ones_like(trans[:1]), trans[:-1]], 0)  # exclusive
            w = alpha * T  # (M,P)
            img = (w[..., None] * col[sel_idx, None, :]).sum(0) + trans[-1][:, None] * bg
            acc = w.sum(0)
            d = (w * dep[sel_idx, None]).sum(0) / acc.clamp_min(1e-6)
            row_imgs.append(img.reshape(th, tw, 3))
            row_alphas.append(acc.reshape(th, tw))
            row_depths.append(d.reshape(th, tw))
        rows.append(torch.cat(row_imgs, 1))
        cols.append((torch.cat(row_alphas, 1), torch.cat(row_depths, 1)))
    image = torch.cat(rows, 0)
    alpha = torch.cat([a for a, _ in cols], 0)
    depth = torch.cat([d for _, d in cols], 0)
    return RenderOutput(image, alpha, depth, means2d_all, radii, visible)


def render_gsplat(means, log_scales, quats, logit_opacity, colors, cam: TorchCamera, *, background=None) -> RenderOutput:
    """Same interface, backed by the CUDA ``gsplat`` rasteriser (GPU only)."""
    import gsplat  # noqa: F401  (optional dependency)

    device = means.device
    viewmat = torch.eye(4, device=device, dtype=means.dtype)
    viewmat[:3, :3] = cam.R
    viewmat[:3, 3] = cam.t
    K = torch.tensor([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], device=device, dtype=means.dtype)
    bg = torch.zeros(1, 3, device=device) if background is None else background.to(device).reshape(1, 3)
    rgbd, alphas, meta = gsplat.rasterization(
        means, quats / quats.norm(dim=1, keepdim=True), torch.exp(log_scales),
        torch.sigmoid(logit_opacity).reshape(-1), colors, viewmat[None], K[None],
        cam.width, cam.height, render_mode="RGB+ED", backgrounds=bg,
    )
    means2d = meta["means2d"]
    if means2d.dim() == 3:
        means2d = means2d[0]
    radii = meta["radii"]
    if radii.dim() == 3:
        radii = radii[0]
    if radii.dim() == 2:
        radii = radii.max(dim=1).values
    return RenderOutput(rgbd[0, ..., :3], alphas[0, ..., 0], rgbd[0, ..., 3], means2d, radii.float(), radii > 0)


def render(params: dict, cam: TorchCamera, *, background=None, backend: str = "torch") -> RenderOutput:
    """Render from a dict of parameter tensors (keys as in :class:`GaussianCloud`)."""
    colors = torch.clamp(params["features_dc"] * SH_C0 + 0.5, 0.0, 1.0)
    fn = render_gsplat if backend == "gsplat" else rasterize_torch
    return fn(params["means"], params["log_scales"], params["quats"], params["logit_opacity"], colors, cam,
              background=background)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM between two ``(H,W,3)`` images in [0,1] (Gaussian window)."""
    c1, c2 = 0.01**2, 0.03**2
    coords = torch.arange(window, device=img1.device, dtype=img1.dtype) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None].expand(3, 1, window, window).contiguous()

    def filt(x):
        return torch.nn.functional.conv2d(x, kernel, padding=window // 2, groups=3)

    a = img1.permute(2, 0, 1)[None]
    b = img2.permute(2, 0, 1)[None]
    mu_a, mu_b = filt(a), filt(b)
    sa = filt(a * a) - mu_a**2
    sb = filt(b * b) - mu_b**2
    sab = filt(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sa + sb + c2)
    return (num / den).mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
