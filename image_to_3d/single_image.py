"""Single-image reconstruction: one photo in, a textured 3D mesh and a Gaussian splat out.

Steps
-----
1. **Segment** the subject (``rembg`` U^2-Net when installed, otherwise OpenCV GrabCut seeded
   from the image border) so only the object is reconstructed. Optional: keep the whole
   frame for a 2.5D relief.
2. **Estimate depth** with a monocular network (:mod:`image_to_3d.depth`).
3. **Back-project** every pixel through a pinhole camera into 3D, mapping the relative inverse
   depth onto a depth range controlled by ``relief``.
4. **Mesh** the pixel grid, cutting triangles across depth discontinuities and outside the
   mask, texture it with the photo, optionally mirror it to give the object a back, and
   export GLB (viewable in any glTF viewer / Three.js).
5. **Splat**: the same points as 3D Gaussians in the reference PLY layout, so the result plugs
   into the Gaussian Splatting tooling in this repo and into web splat viewers.

A single image carries no information about the hidden sides of an object, so the output is
a relief / "2.5D" model, not a full scan. Use the multi-view pipeline for that.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import depth as depth_mod
from .gaussians import GaussianCloud, inverse_sigmoid, rgb_to_sh


@dataclass
class SingleImageConfig:
    depth_backend: str = "auto"          # auto | depth_anything | midas_small | inflate
    remove_background: bool = True
    segmenter: str = "auto"              # auto | rembg | grabcut | none
    max_side: int = 512                  # working resolution for depth / mesh grid
    fov_deg: float = 50.0                # assumed horizontal field of view of the photo
    relief: float = 0.35                 # depth range as a fraction of the object's width
    edge_threshold: float = 0.25         # cut triangles whose depth jump exceeds this * depth range (only real cliffs)
    mirror_back: bool = True             # mirror the relief to close the back of a cut-out object
    smooth_depth: int = 3                # bilateral smoothing passes on the depth map (0 = off)
    splat_stride: int = 1                # keep every n-th pixel as a Gaussian
    device: str = "cpu"

    @staticmethod
    def from_dict(d: dict) -> "SingleImageConfig":
        known = {k: v for k, v in d.items() if k in SingleImageConfig.__dataclass_fields__}
        return SingleImageConfig(**known)


@dataclass
class SingleImageResult:
    rgb: np.ndarray            # (H,W,3) uint8 working-resolution image
    mask: np.ndarray           # (H,W) float32 in [0,1]
    inverse_depth: np.ndarray  # (H,W) float32 in [0,1], 1 = near
    points: np.ndarray         # (H,W,3) float32 back-projected positions (y up, z towards viewer)
    vertices: np.ndarray       # (V,3)
    faces: np.ndarray          # (F,3)
    uvs: np.ndarray            # (V,2)
    cloud: GaussianCloud
    timings: dict

    def save(self, out_dir: str | Path, *, cfg: SingleImageConfig | None = None) -> dict[str, Path]:
        """Write every artefact into ``out_dir`` and return the paths."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        cv2.imwrite(str(out / "input.png"), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        paths["input"] = out / "input.png"
        cv2.imwrite(str(out / "mask.png"), (self.mask * 255).astype(np.uint8))
        paths["mask"] = out / "mask.png"
        depth_vis = cv2.applyColorMap((self.inverse_depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_vis[self.mask < 0.5] = 0
        cv2.imwrite(str(out / "depth.png"), depth_vis)
        paths["depth"] = out / "depth.png"
        cv2.imwrite(str(out / "depth16.png"), (self.inverse_depth * 65535).astype(np.uint16))
        paths["depth16"] = out / "depth16.png"
        export_glb(self, out / "model.glb")
        paths["glb"] = out / "model.glb"
        self.cloud.save_ply(out / "splat.ply")
        paths["splat"] = out / "splat.ply"
        meta = {
            "vertices": int(len(self.vertices)), "faces": int(len(self.faces)), "gaussians": int(len(self.cloud)),
            "resolution": [int(self.rgb.shape[1]), int(self.rgb.shape[0])], "timings": self.timings,
            "config": asdict(cfg) if cfg else None,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        paths["meta"] = out / "meta.json"
        return paths


# --------------------------------------------------------------------------- loading
def load_image(path_or_bytes, max_side: int) -> np.ndarray:
    """Decode an image file / bytes to RGB uint8, EXIF-rotated, longest side <= ``max_side``."""
    from PIL import Image, ImageOps

    img = Image.open(path_or_bytes) if not isinstance(path_or_bytes, (bytes, bytearray)) else Image.open(
        __import__("io").BytesIO(path_or_bytes))
    img = ImageOps.exif_transpose(img)
    alpha = None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        alpha = np.asarray(img)[..., 3]
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    if alpha is not None and (alpha < 250).any():
        return np.dstack([rgb, alpha])
    return rgb


# --------------------------------------------------------------------------- segmentation
def segment_grabcut(rgb: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Foreground mask from GrabCut seeded with 'border = background, centre = probably foreground'."""
    h, w = rgb.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    b = max(2, int(0.04 * min(h, w)))
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = cv2.GC_BGD
    cy0, cy1, cx0, cx1 = int(0.25 * h), int(0.75 * h), int(0.25 * w), int(0.75 * w)
    mask[cy0:cy1, cx0:cx1] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.ones((h, w), np.float32)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    if fg.sum() < 0.01 * h * w:  # GrabCut gave up: keep everything rather than nothing
        return np.ones((h, w), np.float32)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.0)


def segment_rembg(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image
    from rembg import remove

    out = remove(Image.fromarray(rgb), only_mask=True, session=_rembg_session())
    return np.asarray(out).astype(np.float32) / 255.0


_REMBG_SESSION = None


def _rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session("u2net")
    return _REMBG_SESSION


def segmenter_available(name: str) -> bool:
    if name == "rembg":
        try:
            import rembg  # noqa: F401
            return True
        except ImportError:
            return False
    return name in ("grabcut", "none", "auto")


def segment(rgb: np.ndarray, method: str = "auto") -> tuple[np.ndarray, str]:
    if method == "none":
        return np.ones(rgb.shape[:2], np.float32), "none"
    if method == "auto":
        method = "rembg" if segmenter_available("rembg") else "grabcut"
    if method == "rembg":
        try:
            m = segment_rembg(rgb)
            if (m > 0.5).sum() >= 0.005 * m.size:
                return m, "rembg"
        except Exception as e:  # model download failed etc. -> fall back
            print(f"rembg failed ({e}); falling back to GrabCut", flush=True)
    return segment_grabcut(rgb), "grabcut"


def tighten_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and fill holes: one clean subject."""
    binary = (mask > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)
    # fill holes: flood from the border on the inverted mask
    inv = 1 - binary
    flood = inv.copy()
    h, w = inv.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[seed[1], seed[0]] == 1:
            cv2.floodFill(flood, ff_mask, seed, 2)
    holes = (inv == 1) & (flood != 2)
    binary[holes] = 1
    # soft edges inside the component survive; filled holes become solid foreground
    return np.where(binary > 0, np.maximum(mask, 0.6), 0.0).astype(np.float32)


# --------------------------------------------------------------------------- geometry
def back_project(inverse_depth: np.ndarray, fov_deg: float, relief: float) -> np.ndarray:
    """Map pixels + relative inverse depth to 3D points.

    Returns ``(H,W,3)`` with x right, y up, z towards the viewer (glTF convention), centred so
    the object's width is ~1 unit. The depth range (near..far) is ``relief`` * width.
    """
    h, w = inverse_depth.shape
    f = (w / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    z_near = 1.0
    span = relief * (w / f) * z_near  # world width at z_near is w/f * z_near
    z_far = z_near + span
    inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
    z = 1.0 / (inverse_depth * (inv_near - inv_far) + inv_far)
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    x = (us - w / 2) * z / f
    y = -(vs - h / 2) * z / f
    pts = np.dstack([x, y, -z]).astype(np.float32)
    return pts


def build_mesh(points: np.ndarray, mask: np.ndarray, *, edge_threshold: float, mirror_back: bool
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate the pixel grid. Returns vertices (V,3), faces (F,3), uvs (V,2)."""
    h, w = mask.shape
    valid = mask > 0.5
    if valid.sum() < 4:
        raise ValueError("The subject mask is empty; try disabling background removal.")
    index = -np.ones((h, w), np.int64)
    index[valid] = np.arange(int(valid.sum()))
    verts = points[valid]
    us, vs = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    uvs = np.stack([us[valid], 1.0 - vs[valid]], 1).astype(np.float32)

    z = points[..., 2]
    z_range = float(z[valid].max() - z[valid].min()) if valid.any() else 1.0
    thresh = edge_threshold * max(z_range, 1e-6)

    # quad corners: a=(y,x) b=(y,x+1) c=(y+1,x) d=(y+1,x+1)
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    za, zb, zc, zd = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    zs = np.stack([za, zb, zc, zd])
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0) & ((zs.max(0) - zs.min(0)) < thresh)
    # counter-clockwise seen from +z (the viewer): a -> c -> b and b -> c -> d
    f1 = np.stack([a[ok], c[ok], b[ok]], 1)
    f2 = np.stack([b[ok], c[ok], d[ok]], 1)
    faces = np.concatenate([f1, f2], 0)

    if mirror_back:
        # reflect about the plane just behind the farthest point so the object gets a back
        z_back = float(verts[:, 2].min()) - 0.002 * z_range
        back = verts.copy()
        back[:, 2] = 2 * z_back - verts[:, 2]
        n = len(verts)
        back_faces = faces[:, ::-1] + n
        verts = np.concatenate([verts, back], 0)
        uvs = np.concatenate([uvs, uvs], 0)
        faces = np.concatenate([faces, back_faces], 0)

    # normalise: centre and scale the width to 1
    centre = (verts.max(0) + verts.min(0)) / 2
    extent = verts.max(0) - verts.min(0)
    scale = 1.0 / max(float(extent[0]), 1e-6)
    verts = ((verts - centre) * scale).astype(np.float32)
    return verts, faces.astype(np.int64), uvs


def build_splat(points: np.ndarray, rgb: np.ndarray, mask: np.ndarray, *, stride: int, fov_deg: float,
                normalise_like: tuple[np.ndarray, float] | None = None) -> GaussianCloud:
    """One Gaussian per (strided) foreground pixel, sized to its footprint."""
    h, w = mask.shape
    f = (w / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    sel = (mask > 0.5)
    sel[::1, ::1] &= True
    grid = np.zeros_like(sel)
    grid[::stride, ::stride] = True
    sel &= grid
    pts = points[sel]
    cols = rgb[sel].astype(np.float32) / 255.0
    z = -pts[:, 2]
    footprint = z / f * stride  # world size of one (strided) pixel
    if normalise_like is not None:
        centre, scale = normalise_like
        pts = (pts - centre) * scale
        footprint = footprint * scale
    scales = np.stack([footprint * 0.7, footprint * 0.7, footprint * 0.35], 1)
    n = len(pts)
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0
    alpha = np.clip(mask[sel], 0.05, 0.99)
    return GaussianCloud(
        means=pts, log_scales=np.log(np.clip(scales, 1e-6, None)), quats=quats,
        logit_opacity=inverse_sigmoid(alpha)[:, None], features_dc=rgb_to_sh(cols),
    )


def export_glb(result: SingleImageResult, path: str | Path) -> None:
    """Textured glTF binary via trimesh."""
    import trimesh
    from PIL import Image

    tex = Image.fromarray(result.rgb)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9,
                                                   doubleSided=True)
    visual = trimesh.visual.TextureVisuals(uv=result.uvs, material=material)
    mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, visual=visual, process=False)
    mesh.export(str(path), file_type="glb")


# --------------------------------------------------------------------------- pipeline
def reconstruct(image, cfg: SingleImageConfig | None = None, *, progress: Callable[[str, float], None] | None = None
                ) -> SingleImageResult:
    """Run the whole single-image pipeline. ``image`` is a path, bytes or an RGB array."""
    cfg = cfg or SingleImageConfig()
    report = progress or (lambda stage, frac: None)
    timings: dict[str, float] = {}
    t0 = time.time()

    report("loading", 0.02)
    if isinstance(image, np.ndarray):
        rgb = image
        if max(rgb.shape[:2]) > cfg.max_side:
            s = cfg.max_side / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    else:
        rgb = load_image(image, cfg.max_side)
    alpha = None
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        alpha = rgb[..., 3].astype(np.float32) / 255.0
        rgb = np.ascontiguousarray(rgb[..., :3])
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    timings["load"] = time.time() - t0

    report("segmenting", 0.1)
    t = time.time()
    if alpha is not None:  # the upload already has transparency: trust it
        mask, used = alpha, "alpha"
    elif cfg.remove_background:
        mask, used = segment(rgb, cfg.segmenter)
        mask = tighten_mask(mask)
    else:
        mask, used = np.ones(rgb.shape[:2], np.float32), "none"
    timings["segment"] = time.time() - t

    report("estimating depth", 0.35)
    t = time.time()
    backend = depth_mod.default_backend() if cfg.depth_backend == "auto" else cfg.depth_backend
    if not depth_mod.backend_available(backend):
        backend = depth_mod.default_backend()
    try:
        inv = depth_mod.estimate_depth(rgb, backend, mask=mask, device=cfg.device)
    except Exception as e:  # no weights / no network: fall back down the list, then inflate
        print(f"depth backend {backend} failed ({e})", flush=True)
        inv = None
        for alt in depth_mod.BACKENDS[depth_mod.BACKENDS.index(backend) + 1:]:
            if not depth_mod.backend_available(alt):
                continue
            try:
                inv = depth_mod.estimate_depth(rgb, alt, mask=mask, device=cfg.device)
                backend = alt
                break
            except Exception as e2:
                print(f"depth backend {alt} failed ({e2})", flush=True)
        if inv is None:
            backend = "inflate"
            inv = depth_mod.estimate_depth(rgb, "inflate", mask=mask)
    if backend == "midas_small" and cfg.remove_background:
        # MiDaS small blurs silhouettes; push cut-out edges back so they don't flare towards the viewer
        inv = np.minimum(inv, 0.35 + 0.65 * depth_mod.depth_inflate(rgb, mask) ** 0.5)
    for _ in range(cfg.smooth_depth):
        inv = cv2.bilateralFilter(inv.astype(np.float32), 7, 0.08, 5)
    inv = np.clip(inv, 0, 1).astype(np.float32)
    timings["depth"] = time.time() - t
    timings["depth_backend"] = backend  # type: ignore[assignment]
    timings["segmenter"] = used  # type: ignore[assignment]

    report("building mesh", 0.7)
    t = time.time()
    pts = back_project(inv, cfg.fov_deg, cfg.relief)
    verts, faces, uvs = build_mesh(pts, mask, edge_threshold=cfg.edge_threshold, mirror_back=cfg.mirror_back)
    # reuse the same normalisation for the splat
    raw = pts[mask > 0.5]
    if cfg.mirror_back:
        z_back = float(raw[:, 2].min()) - 0.002 * float(raw[:, 2].max() - raw[:, 2].min())
        both = np.concatenate([raw, np.column_stack([raw[:, 0], raw[:, 1], 2 * z_back - raw[:, 2]])], 0)
    else:
        both = raw
    centre = (both.max(0) + both.min(0)) / 2
    scale = 1.0 / max(float((both.max(0) - both.min(0))[0]), 1e-6)
    cloud = build_splat(pts, rgb, mask, stride=cfg.splat_stride, fov_deg=cfg.fov_deg, normalise_like=(centre, scale))
    timings["mesh"] = time.time() - t
    timings["total"] = time.time() - t0
    report("done", 1.0)
    return SingleImageResult(rgb=rgb, mask=mask.astype(np.float32), inverse_depth=inv, points=pts, vertices=verts,
                             faces=faces, uvs=uvs, cloud=cloud, timings=timings)
