"""Monocular depth estimation backends for the single-image path.

``estimate_depth(rgb, backend)`` returns a *relative inverse depth* map in [0, 1]
(1 = nearest) at the input resolution. Backends:

* ``midas_small``   MiDaS v2.1 small (EfficientNet-Lite3 encoder, ~21M params). The
                    network definition below is a trimmed copy of the MIT-licensed
                    MiDaS code (Intel ISL, https://github.com/isl-org/MiDaS) built on
                    timm's backbone, so no torch.hub access is needed. Weights are
                    downloaded once from the MiDaS GitHub release into the cache dir.
* ``depth_anything`` Depth Anything V2 (small) through Hugging Face ``transformers``,
                    if that package and hub access are available. Better quality.
* ``inflate``       Model-free fallback: puffs the foreground mask like a pillow
                    (distance transform). Keeps the app working with no weights.
"""

from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MIDAS_SMALL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
MIDAS_SMALL_SHA256_PREFIX = None  # release asset is served by GitHub; size check only
MIDAS_SMALL_SIZE = 85_761_505

BACKENDS = ("midas_small", "depth_anything", "inflate")


def cache_dir() -> Path:
    d = Path(os.environ.get("IMAGE_TO_3D_CACHE", Path.home() / ".cache" / "image_to_3d"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def midas_weights_path(download: bool = True) -> Path:
    path = cache_dir() / "midas_v21_small_256.pt"
    if path.exists() and path.stat().st_size == MIDAS_SMALL_SIZE:
        return path
    if not download:
        raise FileNotFoundError(path)
    tmp = path.with_suffix(".part")
    print(f"downloading MiDaS small weights -> {path}", flush=True)
    urllib.request.urlretrieve(MIDAS_SMALL_URL, tmp)
    if tmp.stat().st_size != MIDAS_SMALL_SIZE:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("MiDaS weight download was truncated")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# MiDaS small network (adapted from isl-org/MiDaS, MIT License, (c) 2019 Intel ISL)
# ---------------------------------------------------------------------------
def _build_midas_small():
    import timm
    import torch
    import torch.nn as nn

    class Interpolate(nn.Module):
        def __init__(self, scale_factor, mode, align_corners=False):
            super().__init__()
            self.scale_factor, self.mode, self.align_corners = scale_factor, mode, align_corners

        def forward(self, x):
            return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                                             align_corners=self.align_corners)

    class ResidualConvUnit(nn.Module):
        def __init__(self, features, activation):
            super().__init__()
            self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
            self.activation = activation

        def forward(self, x):
            out = self.conv1(self.activation(x))
            out = self.conv2(self.activation(out))
            return out + x

    class FeatureFusionBlock(nn.Module):
        def __init__(self, features, activation, expand=False, align_corners=True):
            super().__init__()
            self.align_corners = align_corners
            out_features = features // 2 if expand else features
            self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True)
            self.resConfUnit1 = ResidualConvUnit(features, activation)
            self.resConfUnit2 = ResidualConvUnit(features, activation)

        def forward(self, *xs):
            output = xs[0]
            if len(xs) == 2:
                output = output + self.resConfUnit1(xs[1])
            output = self.resConfUnit2(output)
            output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
            return self.out_conv(output)

    class MidasNetSmall(nn.Module):
        def __init__(self, features=64):
            super().__init__()
            effnet = timm.create_model("tf_efficientnet_lite3", pretrained=False)
            self.pretrained = nn.Module()
            # The original wraps gen-efficientnet's (conv_stem, bn1, act1, blocks...). timm folds the
            # activation into bn1, so an Identity keeps the Sequential indices (and state-dict keys) aligned.
            self.pretrained.layer1 = nn.Sequential(effnet.conv_stem, effnet.bn1, nn.Identity(), *effnet.blocks[0:2])
            self.pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
            self.pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
            self.pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

            f1, f2, f3, f4 = features, features * 2, features * 4, features * 8
            self.scratch = nn.Module()
            for i, (cin, cout) in enumerate(zip([32, 48, 136, 384], [f1, f2, f3, f4]), start=1):
                setattr(self.scratch, f"layer{i}_rn", nn.Conv2d(cin, cout, 3, 1, 1, bias=False))
            act = nn.ReLU(False)
            self.scratch.activation = act
            self.scratch.refinenet4 = FeatureFusionBlock(f4, act, expand=True)
            self.scratch.refinenet3 = FeatureFusionBlock(f3, act, expand=True)
            self.scratch.refinenet2 = FeatureFusionBlock(f2, act, expand=True)
            self.scratch.refinenet1 = FeatureFusionBlock(f1, act, expand=False)
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(features, features // 2, 3, 1, 1),
                Interpolate(scale_factor=2, mode="bilinear"),
                nn.Conv2d(features // 2, 32, 3, 1, 1),
                act,
                nn.Conv2d(32, 1, 1, 1, 0),
                nn.ReLU(True),
                nn.Identity(),
            )

        def forward(self, x):
            l1 = self.pretrained.layer1(x)
            l2 = self.pretrained.layer2(l1)
            l3 = self.pretrained.layer3(l2)
            l4 = self.pretrained.layer4(l3)
            p4 = self.scratch.refinenet4(self.scratch.layer4_rn(l4))
            p3 = self.scratch.refinenet3(p4, self.scratch.layer3_rn(l3))
            p2 = self.scratch.refinenet2(p3, self.scratch.layer2_rn(l2))
            p1 = self.scratch.refinenet1(p2, self.scratch.layer1_rn(l1))
            return self.scratch.output_conv(p1).squeeze(1)

    return MidasNetSmall()


@lru_cache(maxsize=1)
def load_midas_small(device: str = "cpu"):
    import torch

    model = _build_midas_small()
    state = torch.load(midas_weights_path(), map_location="cpu")
    if "optimizer" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _midas_preprocess(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize so the longer side is ``size`` (dims multiples of 32), ImageNet-normalise, CHW."""
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw = max(32, int(np.floor(w * scale / 32)) * 32)
    nh = max(32, int(np.floor(h * scale / 32)) * 32)
    img = cv2.resize(rgb.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_CUBIC)
    img = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def depth_midas_small(rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch

    model = load_midas_small(device)
    x = torch.from_numpy(_midas_preprocess(rgb))[None].to(device)
    with torch.no_grad():
        pred = model(x)
        pred = torch.nn.functional.interpolate(pred[None], size=rgb.shape[:2], mode="bicubic",
                                               align_corners=False)[0, 0]
    return pred.cpu().numpy()


def depth_depth_anything(rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    from PIL import Image
    from transformers import pipeline  # optional dependency, needs hub access for the weights

    pipe = _depth_anything_pipe(device)
    out = pipe(Image.fromarray(rgb))
    pred = np.asarray(out["predicted_depth"], dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    return cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)


@lru_cache(maxsize=1)
def _depth_anything_pipe(device: str):
    from transformers import pipeline

    return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if device.startswith("cuda") else -1)


def depth_inflate(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Model-free pseudo depth: the further a pixel is from the silhouette, the closer it is."""
    h, w = rgb.shape[:2]
    if mask is None:
        mask = np.ones((h, w), np.uint8)
    m = (mask > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros((h, w), np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dist = np.sqrt(dist / max(dist.max(), 1e-6))  # sqrt -> rounded, pillow-like profile
    return dist.astype(np.float32)


def normalise_inverse_depth(inv: np.ndarray, mask: np.ndarray | None = None, clip_pct: float = 1.0) -> np.ndarray:
    """Robustly rescale a relative inverse-depth map to [0, 1] using the masked pixels."""
    inv = inv.astype(np.float32)
    sel = inv[mask > 0.5] if mask is not None and (mask > 0.5).any() else inv.ravel()
    lo, hi = np.percentile(sel, clip_pct), np.percentile(sel, 100 - clip_pct)
    if hi - lo < 1e-6:
        return np.full_like(inv, 0.5)
    return np.clip((inv - lo) / (hi - lo), 0.0, 1.0)


def backend_available(name: str) -> bool:
    if name == "inflate":
        return True
    if name == "midas_small":
        try:
            import timm  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return True
    if name == "depth_anything":
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True
    return False


def estimate_depth(rgb: np.ndarray, backend: str = "midas_small", *, mask: np.ndarray | None = None,
                   device: str = "cpu") -> np.ndarray:
    """Return relative inverse depth in [0,1] (1 = nearest) for an ``(H,W,3)`` uint8 RGB image."""
    if backend == "midas_small":
        inv = depth_midas_small(rgb, device)
    elif backend == "depth_anything":
        inv = depth_depth_anything(rgb, device)
    elif backend == "inflate":
        return depth_inflate(rgb, mask)
    else:
        raise ValueError(f"unknown depth backend {backend!r}; choose from {BACKENDS}")
    return normalise_inverse_depth(inv, mask)
