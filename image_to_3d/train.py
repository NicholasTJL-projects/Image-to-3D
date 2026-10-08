"""Stage 4: optimise a Gaussian cloud against the captured images.

This is a compact re-implementation of the 3D Gaussian Splatting training loop
(Kerbl et al. 2023): photometric L1 + D-SSIM loss, per-parameter Adam learning
rates, adaptive density control (clone / split / prune) and periodic opacity
resets. The renderer is pluggable: the pure-PyTorch rasteriser works anywhere,
``gsplat`` is used for real scenes on a GPU.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .camera import Camera
from .gaussians import GaussianCloud
from .scene import Scene


@dataclass
class TrainConfig:
    iterations: int = 7000
    device: str = "auto"          # "auto" picks cuda if available
    backend: str = "auto"         # "torch" | "gsplat" | "auto" (gsplat when importable + cuda)
    downscale: int = 1            # train on images downscaled by this integer factor
    seed: int = 0
    background: str = "black"     # "black" | "white" | "random"

    # learning rates (means lr is multiplied by the scene radius, as in the reference)
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_features: float = 2.5e-3
    lr_opacity: float = 0.05
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lambda_dssim: float = 0.2

    # adaptive density control
    densify: bool = True
    densify_from: int = 500
    densify_until: int | None = None   # default: iterations // 2
    densify_interval: int = 100
    densify_grad_threshold: float = 2e-4
    percent_dense: float = 0.01
    prune_opacity: float = 0.005
    opacity_reset_interval: int = 3000
    max_gaussians: int | None = None   # safety cap, useful on CPU

    # bookkeeping
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 0          # 0 = only at the end
    test_every_nth: int = 8            # hold out every n-th camera for PSNR

    @staticmethod
    def from_dict(d: dict) -> "TrainConfig":
        known = {k: v for k, v in d.items() if k in TrainConfig.__dataclass_fields__}
        return TrainConfig(**known)


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("Training needs PyTorch: pip install 'image-to-3d[train]'  (or pip install torch)") from e
    import torch
    return torch


def load_image(cam: Camera, downscale: int = 1) -> np.ndarray:
    """Read the camera's image as ``(H,W,3)`` float RGB in [0,1] at the training resolution."""
    if cam.image_path is None:
        raise ValueError(f"camera {cam.name!r} has no image_path")
    img = cv2.imread(cam.image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(cam.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target = (cam.width // downscale, cam.height // downscale)
    if img.shape[1] != target[0] or img.shape[0] != target[1]:
        img = cv2.resize(img, target, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


class Trainer:
    def __init__(self, scene: Scene, cloud: GaussianCloud, cfg: TrainConfig,
                 images: dict[str, np.ndarray] | None = None, out_dir: str | Path | None = None,
                 log: Callable[[str], None] = print):
        torch = _torch()
        self.torch = torch
        self.cfg = cfg
        self.log = log
        self.out_dir = Path(out_dir) if out_dir else None
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self.backend = cfg.backend
        if self.backend == "auto":
            self.backend = "torch"
            if self.device.type == "cuda":
                try:
                    import gsplat  # noqa: F401
                    self.backend = "gsplat"
                except ImportError:
                    pass

        train_scene, test_scene = scene.split(cfg.test_every_nth)
        self.scene_radius = max(scene.radius, 1e-3)
        ds = cfg.downscale
        self.train_cams = [c.scaled(1.0 / ds) if ds != 1 else c for c in train_scene.cameras]
        self.test_cams = [c.scaled(1.0 / ds) if ds != 1 else c for c in test_scene.cameras]

        def _img(cam: Camera, original: Camera):
            if images is not None and original.name in images:
                arr = images[original.name]
                if arr.shape[0] != cam.height or arr.shape[1] != cam.width:
                    arr = cv2.resize(arr, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
                return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=self.device)
            return torch.as_tensor(load_image(original, ds), device=self.device)

        self.train_images = [_img(c, o) for c, o in zip(self.train_cams, train_scene.cameras)]
        self.test_images = [_img(c, o) for c, o in zip(self.test_cams, test_scene.cameras)]

        from .render_torch import TorchCamera
        self.train_tcams = [TorchCamera.from_camera(c, self.device) for c in self.train_cams]
        self.test_tcams = [TorchCamera.from_camera(c, self.device) for c in self.test_cams]

        # parameters
        self.params = {
            "means": self._param(cloud.means),
            "log_scales": self._param(cloud.log_scales),
            "quats": self._param(cloud.quats),
            "logit_opacity": self._param(cloud.logit_opacity),
            "features_dc": self._param(cloud.features_dc),
        }
        self.sh_degree = cloud.sh_degree
        self._build_optimizer()
        n = len(cloud)
        self.grad_accum = torch.zeros(n, device=self.device)
        self.grad_count = torch.zeros(n, device=self.device)
        self.max_radii = torch.zeros(n, device=self.device)
        self.step = 0

    # -- helpers ----------------------------------------------------------
    def _param(self, arr):
        return self.torch.nn.Parameter(self.torch.as_tensor(np.asarray(arr, dtype=np.float32), device=self.device))

    def _lr_groups(self) -> dict[str, float]:
        c = self.cfg
        return {
            "means": c.lr_means * self.scene_radius,
            "log_scales": c.lr_scales,
            "quats": c.lr_quats,
            "logit_opacity": c.lr_opacity,
            "features_dc": c.lr_features,
        }

    def _build_optimizer(self):
        torch = self.torch
        groups = [{"params": [p], "lr": lr, "name": name} for (name, p), lr in
                  zip(self.params.items(), self._lr_groups().values())]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def _means_lr(self, step: int) -> float:
        """Log-linear decay from ``lr_means`` to ``lr_means_final`` over training."""
        c = self.cfg
        t = min(1.0, step / max(1, c.iterations))
        lr = math.exp(math.log(c.lr_means) * (1 - t) + math.log(c.lr_means_final) * t)
        return lr * self.scene_radius

    def _background(self):
        torch = self.torch
        if self.cfg.background == "white":
            return torch.ones(3, device=self.device)
        if self.cfg.background == "random":
            return torch.rand(3, device=self.device)
        return torch.zeros(3, device=self.device)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def cloud(self) -> GaussianCloud:
        p = {k: v.detach().cpu().numpy() for k, v in self.params.items()}
        n = len(p["means"])
        return GaussianCloud(p["means"], p["log_scales"], p["quats"], p["logit_opacity"], p["features_dc"],
                             np.zeros((n, (self.sh_degree + 1) ** 2 - 1, 3), np.float32))

    # -- optimizer state surgery (needed when Gaussians are added/removed) --
    def _replace_params(self, new: dict, keep_mask=None, extra: dict | None = None):
        """Rebuild parameters: keep rows where ``keep_mask`` is True, then append ``extra`` rows.

        Adam moments are carried over for the kept rows and zeroed for new ones,
        exactly as the reference implementation does.
        """
        torch = self.torch
        for group in self.optimizer.param_groups:
            name = group["name"]
            old_p = group["params"][0]
            state = self.optimizer.state.pop(old_p, None)
            tensor = new[name]
            if keep_mask is not None:
                tensor = tensor[keep_mask]
            if extra is not None and name in extra:
                tensor = torch.cat([tensor, extra[name]], 0)
            p = torch.nn.Parameter(tensor.detach().contiguous())
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    m = state[key]
                    if keep_mask is not None:
                        m = m[keep_mask]
                    if extra is not None and name in extra:
                        m = torch.cat([m, torch.zeros_like(extra[name])], 0)
                    state[key] = m
                self.optimizer.state[p] = state
            group["params"] = [p]
            self.params[name] = p
        n = self.num_gaussians
        self.grad_accum = torch.zeros(n, device=self.device)
        self.grad_count = torch.zeros(n, device=self.device)
        self.max_radii = torch.zeros(n, device=self.device)

    # -- adaptive density control ----------------------------------------
    def _densify_and_prune(self, cam_w: int, cam_h: int):
        torch = self.torch
        c = self.cfg
        p = self.params
        grads = self.grad_accum / self.grad_count.clamp_min(1)
        scales = torch.exp(p["log_scales"].detach())
        max_scale = scales.max(dim=1).values
        high_grad = grads >= c.densify_grad_threshold
        extent = c.percent_dense * self.scene_radius

        # clone: small Gaussians in under-reconstructed regions move a copy along the gradient
        clone = high_grad & (max_scale <= extent)
        # split: big Gaussians in over-reconstructed regions become two smaller ones
        split = high_grad & (max_scale > extent)

        n_before = self.num_gaussians
        extra = {k: [] for k in p}
        if clone.any():
            for k in p:
                extra[k].append(p[k].detach()[clone])
        if split.any():
            n_split = int(split.sum())
            means, logs, quats = p["means"].detach()[split], p["log_scales"].detach()[split], p["quats"].detach()[split]
            from .render_torch import quats_to_rotmats
            R = quats_to_rotmats(quats)
            for _ in range(2):
                samples = torch.randn(n_split, 3, device=self.device) * torch.exp(logs)
                offs = (R @ samples[..., None])[..., 0]
                extra["means"].append(means + offs)
                extra["log_scales"].append(logs - math.log(1.6))
                extra["quats"].append(quats)
                extra["logit_opacity"].append(p["logit_opacity"].detach()[split])
                extra["features_dc"].append(p["features_dc"].detach()[split])
        extra_t = {k: torch.cat(v, 0) for k, v in extra.items() if v}

        opac = torch.sigmoid(p["logit_opacity"].detach()).reshape(-1)
        prune = opac < c.prune_opacity
        if self.step > c.opacity_reset_interval:
            big_world = max_scale > 0.1 * self.scene_radius
            big_screen = self.max_radii > 0.2 * max(cam_w, cam_h)
            prune = prune | big_world | big_screen
        keep = ~(prune | split)  # split parents are removed

        if c.max_gaussians is not None and extra_t:
            room = max(0, c.max_gaussians - int(keep.sum()))
            if room < len(extra_t["means"]):
                sel = torch.randperm(len(extra_t["means"]), device=self.device)[:room]
                extra_t = {k: v[sel] for k, v in extra_t.items()}

        self._replace_params(p, keep_mask=keep, extra=extra_t if extra_t else None)
        self.log(f"[{self.step}] densify: {n_before} -> {self.num_gaussians} gaussians "
                 f"(clone {int(clone.sum())}, split {int(split.sum())}, prune {int(prune.sum())})")

    def _reset_opacity(self):
        torch = self.torch
        p = self.params["logit_opacity"]
        new_val = torch.minimum(p.detach(), torch.full_like(p, math.log(0.01 / 0.99)))
        state = self.optimizer.state.get(p)
        if state is not None:
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
        p.data.copy_(new_val)

    # -- the loop ----------------------------------------------------------
    def train_step(self) -> float:
        torch = self.torch
        from .render_torch import render, ssim

        c = self.cfg
        self.step += 1
        for g in self.optimizer.param_groups:
            if g["name"] == "means":
                g["lr"] = self._means_lr(self.step)

        i = int(self.rng.integers(len(self.train_cams)))
        tcam, gt = self.train_tcams[i], self.train_images[i]
        out = render(self.params, tcam, background=self._background(), backend=self.backend)
        img = out.image
        l1 = torch.abs(img - gt).mean()
        loss = (1 - c.lambda_dssim) * l1 + c.lambda_dssim * (1 - ssim(img, gt))
        loss.backward()

        with torch.no_grad():
            if c.densify and self.step < (c.densify_until or c.iterations // 2) and out.means2d.grad is not None:
                vis = out.visible
                # the reference thresholds gradients of NDC coordinates: scale pixel grads accordingly
                g2 = out.means2d.grad[vis] * torch.tensor([tcam.width / 2, tcam.height / 2], device=self.device)
                self.grad_accum[vis] += g2.norm(dim=1)
                self.grad_count[vis] += 1
                self.max_radii[vis] = torch.maximum(self.max_radii[vis], out.radii[vis])

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # keep quaternions normalised so scales stay meaningful
            q = self.params["quats"]
            q.data.div_(q.data.norm(dim=1, keepdim=True).clamp_min(1e-8))

            if c.densify and c.densify_from <= self.step < (c.densify_until or c.iterations // 2):
                if self.step % c.densify_interval == 0:
                    self._densify_and_prune(tcam.width, tcam.height)
                if c.opacity_reset_interval and self.step % c.opacity_reset_interval == 0:
                    self._reset_opacity()
        return float(loss.item())

    @property
    def _no_grad(self):
        return self.torch.no_grad

    def evaluate(self) -> float:
        """Mean PSNR over the held-out cameras (or the training cameras if none)."""
        from .render_torch import psnr, render
        torch = self.torch
        cams = self.test_tcams or self.train_tcams
        imgs = self.test_images or self.train_images
        with torch.no_grad():
            vals = [psnr(render(self.params, c, backend=self.backend).image, im) for c, im in zip(cams, imgs)]
        return float(np.mean(vals)) if vals else float("nan")

    def train(self) -> GaussianCloud:
        c = self.cfg
        self.log(f"training {self.num_gaussians} gaussians for {c.iterations} iters on {self.device} "
                 f"({self.backend} backend, {len(self.train_cams)} train / {len(self.test_cams)} test views, "
                 f"{self.train_cams[0].width}x{self.train_cams[0].height})")
        t0 = time.time()
        ema = None
        for _ in range(c.iterations):
            loss = self.train_step()
            ema = loss if ema is None else 0.9 * ema + 0.1 * loss
            if self.step % c.log_every == 0 or self.step == c.iterations:
                self.log(f"[{self.step}/{c.iterations}] loss {ema:.4f}  gaussians {self.num_gaussians}  "
                         f"{(time.time() - t0) / self.step:.3f}s/it")
            if c.eval_every and self.step % c.eval_every == 0:
                self.log(f"[{self.step}] test PSNR {self.evaluate():.2f} dB")
            if self.out_dir and c.checkpoint_every and self.step % c.checkpoint_every == 0:
                self.cloud().save_ply(self.out_dir / f"point_cloud_{self.step:06d}.ply")
        final = self.cloud()
        if self.out_dir:
            final.save_ply(self.out_dir / "point_cloud.ply")
        self.log(f"done: {self.num_gaussians} gaussians, final test PSNR {self.evaluate():.2f} dB, "
                 f"{time.time() - t0:.1f}s")
        return final


def train(scene: Scene, cloud: GaussianCloud, cfg: TrainConfig | None = None, *, images=None, out_dir=None,
          log=print) -> GaussianCloud:
    return Trainer(scene, cloud, cfg or TrainConfig(), images=images, out_dir=out_dir, log=log).train()


def config_to_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
