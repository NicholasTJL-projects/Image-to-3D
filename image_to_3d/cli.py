"""Command line entry point: ``image-to-3d <stage> ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from . import pipeline
from .capture import CaptureConfig, capture
from .gaussians import GaussianCloud
from .scene import Scene


def _cfg_section(args, name: str) -> dict:
    return dict(pipeline.load_config(getattr(args, "config", None)).get(name, {}) or {})


def _override(section: dict, args, keys: list[str]) -> dict:
    for k in keys:
        v = getattr(args, k, None)
        if v is not None:
            section[k] = v
    return section


# ---------------------------------------------------------------- stages
def cmd_capture(args):
    cfg = CaptureConfig(**{k: v for k, v in _override(_cfg_section(args, "capture"), args,
                        ["every_nth", "max_frames", "max_side", "min_blur", "keep_sharpest_ratio"]).items()
                           if k in CaptureConfig.__dataclass_fields__})
    paths = capture(args.source, Path(args.workspace) / "images", cfg, clean=args.clean)
    print(f"wrote {len(paths)} frames to {Path(args.workspace) / 'images'}")
    if len(paths) < 20:
        print("warning: fewer than 20 frames; SfM usually needs 40+ overlapping views", file=sys.stderr)


def cmd_sfm(args):
    from .colmap import run_sfm

    cfg = _override(_cfg_section(args, "sfm"), args, ["matcher", "camera_model"])
    if args.no_gpu:
        cfg["use_gpu"] = False
    scene = run_sfm(args.workspace, images_dir=args.images, binary=args.colmap_bin,
                    **{k: v for k, v in cfg.items() if k in ("matcher", "camera_model", "single_camera", "use_gpu")})
    print(f"SfM done: {len(scene)} posed cameras, {len(scene.points_xyz)} sparse points -> {Path(args.workspace) / 'scene'}")


def cmd_init(args):
    scene = pipeline.load_scene(args.workspace, sparse=args.sparse, images=args.images)
    cfg = _override(_cfg_section(args, "init"), args, ["initial_opacity", "sh_degree", "random_points"])
    cloud = pipeline.init_gaussians(scene, args.workspace, **cfg)
    print(f"initialised {len(cloud)} gaussians from {len(scene.points_xyz)} points -> {Path(args.workspace) / 'output' / 'init.ply'}")


def cmd_train(args):
    from .train import TrainConfig, train

    ws = Path(args.workspace)
    scene = pipeline.load_scene(ws)
    init_ply = Path(args.init) if args.init else ws / "output" / "init.ply"
    if init_ply.exists():
        cloud = GaussianCloud.load_ply(init_ply)
    else:
        cloud = pipeline.init_gaussians(scene, ws, **_cfg_section(args, "init"))
    cfg_d = _override(_cfg_section(args, "train"), args,
                      ["iterations", "device", "backend", "downscale", "background", "max_gaussians", "log_every",
                       "eval_every", "checkpoint_every", "seed"])
    if args.no_densify:
        cfg_d["densify"] = False
    cfg = TrainConfig.from_dict(cfg_d)
    out = train(scene, cloud, cfg, out_dir=ws / "output")
    print(f"saved {len(out)} gaussians to {ws / 'output' / 'point_cloud.ply'}")


def _write_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def cmd_render(args):
    ws = Path(args.workspace)
    ply = Path(args.ply) if args.ply else ws / "output" / "point_cloud.ply"
    if not ply.exists():
        ply = ws / "output" / "init.ply"
    cloud = GaussianCloud.load_ply(ply)
    scene = pipeline.load_scene(ws)
    rcfg = _override(_cfg_section(args, "render"), args, ["orbit_frames", "elevation_deg", "background"])
    bg = {"white": (1, 1, 1), "black": (0, 0, 0)}.get(rcfg.get("background", "black"), (0, 0, 0))
    ds = args.downscale or 1
    cams = scene.cameras if args.views == "train" else pipeline.orbit_cameras(
        scene, int(rcfg.get("orbit_frames", 60)), elevation_deg=float(rcfg.get("elevation_deg", 10)))
    cams = [c.scaled(1.0 / ds) if ds != 1 else c for c in cams]
    out_dir = ws / "renders" / args.views
    renderer = _make_renderer(args.backend, cloud, bg)
    print(f"rendering {len(cams)} views of {len(cloud)} gaussians with {renderer.__name__} -> {out_dir}")
    frames = []
    for i, cam in enumerate(cams):
        img = renderer(cam)
        _write_image(out_dir / f"{cam.name or f'view_{i:04d}'}.png", img)
        frames.append(img)
    if args.video:
        vp = out_dir.with_suffix(".mp4")
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"wrote {vp}")


def _make_renderer(backend: str, cloud: GaussianCloud, bg):
    if backend in ("auto", "torch"):
        try:
            import torch
            from .render_torch import TorchCamera, render as trender

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            params = {k: torch.as_tensor(getattr(cloud, k), device=device)
                      for k in ("means", "log_scales", "quats", "logit_opacity", "features_dc")}
            bgt = torch.tensor(bg, dtype=torch.float32, device=device)

            def torch_renderer(cam):
                with torch.no_grad():
                    return trender(params, TorchCamera.from_camera(cam, device), background=bgt).image.cpu().numpy()
            return torch_renderer
        except ImportError:
            if backend == "torch":
                raise
    from .render_np import render as nrender

    def numpy_renderer(cam):
        return nrender(cloud, cam, background=bg)
    return numpy_renderer


def cmd_info(args):
    cloud = GaussianCloud.load_ply(args.ply)
    s = cloud.scales
    print(f"{args.ply}: {len(cloud)} gaussians, SH degree {cloud.sh_degree}")
    print(f"  extent   min {cloud.means.min(0)}  max {cloud.means.max(0)}")
    print(f"  opacity  mean {cloud.opacities.mean():.3f}  >0.5: {(cloud.opacities > 0.5).mean() * 100:.1f}%")
    print(f"  scale    median {np.median(s):.4f}  max {s.max():.4f}")


def cmd_export(args):
    GaussianCloud.load_ply(args.ply).export_point_cloud(args.out, min_opacity=args.min_opacity)
    print(f"wrote {args.out}")


def cmd_demo(args):
    """Synthetic end-to-end run: render ground truth, initialise from noisy points, train, compare."""
    from .render_np import psnr, render as nrender
    from .synthetic import synthetic_scene

    ws = Path(args.workspace)
    gt, scene = synthetic_scene(args.gaussians, args.cameras, width=args.size, height=args.size, seed=args.seed)
    (ws / "images").mkdir(parents=True, exist_ok=True)
    images = {}
    for cam in scene.cameras:
        img = nrender(gt, cam)
        cam.image_path = str(ws / "images" / f"{cam.name}.png")
        _write_image(Path(cam.image_path), img)
        images[cam.name] = img.astype(np.float32)
    scene.save(ws / "scene")
    gt.save_ply(ws / "output" / "ground_truth.ply")
    cloud = pipeline.init_gaussians(scene, ws, initial_opacity=0.5)
    before = np.mean([psnr(nrender(cloud, c), images[c.name]) for c in scene.cameras[::8]])
    print(f"initial PSNR {before:.2f} dB with {len(cloud)} gaussians")

    from .train import TrainConfig, train

    cfg = TrainConfig(iterations=args.iterations, device=args.device or "auto", densify=not args.no_densify,
                      densify_from=100, densify_interval=100, opacity_reset_interval=0, log_every=50,
                      eval_every=0, max_gaussians=4 * args.gaussians, lr_means=4e-3, lr_means_final=1e-4)
    out = train(scene, cloud, cfg, images=images, out_dir=ws / "output")
    after = np.mean([psnr(nrender(out, c), images[c.name]) for c in scene.cameras[::8]])
    print(f"final PSNR {after:.2f} dB with {len(out)} gaussians (ground truth has {len(gt)})")


def cmd_run(args):
    """capture -> sfm -> init -> train -> render, in one go."""
    cmd_capture(args)
    cmd_sfm(args)
    cmd_init(args)
    cmd_train(args)
    args.views, args.ply, args.video, args.backend = "orbit", None, True, "auto"
    args.downscale = args.downscale or 2
    cmd_render(args)


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="image-to-3d", description="2D captures -> 3D Gaussian Splatting scene")
    p.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    def ws(sp):
        sp.add_argument("workspace", help="folder that holds everything for this capture")

    s = sub.add_parser("capture", help="extract sharp frames from a video / webcam / photo folder")
    s.add_argument("source", help="video file, webcam index (e.g. 0) or folder of photos")
    ws(s)
    s.add_argument("--every-nth", dest="every_nth", type=int)
    s.add_argument("--max-frames", dest="max_frames", type=int)
    s.add_argument("--max-side", dest="max_side", type=int)
    s.add_argument("--min-blur", dest="min_blur", type=float)
    s.add_argument("--keep-sharpest", dest="keep_sharpest_ratio", type=float)
    s.add_argument("--clean", action="store_true", help="delete existing images/ first")
    s.set_defaults(func=cmd_capture)

    s = sub.add_parser("sfm", help="run COLMAP structure-from-motion on workspace/images")
    ws(s)
    s.add_argument("--images", help="image folder (default workspace/images)")
    s.add_argument("--matcher", choices=["sequential", "exhaustive"])
    s.add_argument("--camera-model", dest="camera_model")
    s.add_argument("--no-gpu", action="store_true")
    s.add_argument("--colmap-bin", dest="colmap_bin", default="colmap")
    s.set_defaults(func=cmd_sfm)

    s = sub.add_parser("init", help="seed Gaussians from the sparse point cloud")
    ws(s)
    s.add_argument("--sparse", help="COLMAP sparse model folder (if SfM was run elsewhere)")
    s.add_argument("--images", help="image folder matching --sparse")
    s.add_argument("--initial-opacity", dest="initial_opacity", type=float)
    s.add_argument("--sh-degree", dest="sh_degree", type=int)
    s.add_argument("--random-points", dest="random_points", type=int)
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("train", help="optimise the Gaussians against the images (needs torch)")
    ws(s)
    s.add_argument("--init", help="starting PLY (default workspace/output/init.ply)")
    s.add_argument("--iterations", type=int)
    s.add_argument("--device")
    s.add_argument("--backend", choices=["auto", "torch", "gsplat"])
    s.add_argument("--downscale", type=int)
    s.add_argument("--background", choices=["black", "white", "random"])
    s.add_argument("--max-gaussians", dest="max_gaussians", type=int)
    s.add_argument("--no-densify", dest="no_densify", action="store_true")
    s.add_argument("--log-every", dest="log_every", type=int)
    s.add_argument("--eval-every", dest="eval_every", type=int)
    s.add_argument("--checkpoint-every", dest="checkpoint_every", type=int)
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("render", help="render the trained scene from the training or an orbit of cameras")
    ws(s)
    s.add_argument("--ply", help="PLY to render (default workspace/output/point_cloud.ply)")
    s.add_argument("--views", choices=["orbit", "train"], default="orbit")
    s.add_argument("--orbit-frames", dest="orbit_frames", type=int)
    s.add_argument("--elevation-deg", dest="elevation_deg", type=float)
    s.add_argument("--background", choices=["black", "white"])
    s.add_argument("--downscale", type=int)
    s.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    s.add_argument("--video", action="store_true", help="also write an mp4")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("info", help="print statistics of a Gaussian PLY")
    s.add_argument("ply")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("export", help="export Gaussian centres as a coloured point cloud PLY")
    s.add_argument("ply")
    s.add_argument("out")
    s.add_argument("--min-opacity", dest="min_opacity", type=float, default=0.3)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("demo", help="synthetic end-to-end run without any footage or COLMAP")
    ws(s)
    s.add_argument("--gaussians", type=int, default=150)
    s.add_argument("--cameras", type=int, default=24)
    s.add_argument("--size", type=int, default=64)
    s.add_argument("--iterations", type=int, default=300)
    s.add_argument("--device")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-densify", dest="no_densify", action="store_true")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("run", help="capture + sfm + init + train + render")
    s.add_argument("source")
    ws(s)
    s.add_argument("--every-nth", dest="every_nth", type=int)
    s.add_argument("--max-frames", dest="max_frames", type=int)
    s.add_argument("--max-side", dest="max_side", type=int)
    s.add_argument("--min-blur", dest="min_blur", type=float)
    s.add_argument("--keep-sharpest", dest="keep_sharpest_ratio", type=float)
    s.add_argument("--clean", action="store_true")
    s.add_argument("--images", default=None)
    s.add_argument("--matcher", choices=["sequential", "exhaustive"])
    s.add_argument("--camera-model", dest="camera_model")
    s.add_argument("--no-gpu", action="store_true")
    s.add_argument("--colmap-bin", dest="colmap_bin", default="colmap")
    s.add_argument("--sparse", default=None)
    s.add_argument("--init", default=None)
    s.add_argument("--iterations", type=int)
    s.add_argument("--device")
    s.add_argument("--backend", choices=["auto", "torch", "gsplat"])
    s.add_argument("--downscale", type=int)
    s.add_argument("--background", choices=["black", "white", "random"])
    s.add_argument("--max-gaussians", dest="max_gaussians", type=int)
    s.add_argument("--no-densify", dest="no_densify", action="store_true")
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_run, initial_opacity=None, sh_degree=None, random_points=None, log_every=None,
                   eval_every=None, checkpoint_every=None, orbit_frames=None, elevation_deg=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
