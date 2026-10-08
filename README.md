# Image-to-3D

Reconstruct a 3D scene from ordinary 2D captures (a phone video, a webcam stream or a folder
of photos) using **3D Gaussian Splatting**.

```
2D frames  ──►  camera poses + sparse points  ──►  Gaussians  ──►  trained scene  ──►  renders
 capture            sfm (COLMAP)                    init          train (PyTorch)      render
```

The result is a `point_cloud.ply` in the standard 3DGS layout, so it opens in any splat viewer
(SuperSplat, antimatter15/splat, SIBR, Polycam...) and can be rendered from new viewpoints here.

## Install

```bash
pip install -e ".[train,dev]"             # core + PyTorch + pytest
# for real scenes on an NVIDIA GPU, add the CUDA rasteriser:
pip install gsplat
```

Structure-from-Motion uses the external [COLMAP](https://colmap.github.io/install.html) binary
(`apt install colmap`, `brew install colmap`, or the Windows release). Only the `sfm` stage needs it.

## Quick start

Check the whole pipeline without any footage or COLMAP:

```bash
image-to-3d demo workspaces/demo --iterations 300
```

This renders a synthetic Gaussian scene from 24 cameras, seeds new Gaussians from noisy points,
trains them against the renders and prints the PSNR before and after.

A real capture:

```bash
image-to-3d capture my_video.mp4 workspaces/desk     # extract sharp frames -> images/
image-to-3d sfm     workspaces/desk                  # COLMAP poses + sparse points -> scene/
image-to-3d init    workspaces/desk                  # seed Gaussians -> output/init.ply
image-to-3d train   workspaces/desk --iterations 7000
image-to-3d render  workspaces/desk --views orbit --video   # turntable -> renders/orbit.mp4
```

or everything at once: `image-to-3d run my_video.mp4 workspaces/desk` /
`scripts/run_pipeline.sh my_video.mp4 workspaces/desk`.

Settings live in `configs/default.yaml`; every value can be overridden with a CLI flag
(`image-to-3d train --help`).

## Capturing good input

Gaussian Splatting only reconstructs what the cameras saw, and COLMAP only recovers poses when
neighbouring frames overlap. For a hand-held phone video:

* Walk slowly around the object in a full circle, then a second pass higher or lower.
  60 to 300 frames with roughly 70 % overlap between neighbours is the target.
* Keep the scene static and the lighting constant. Lock exposure and focus if the camera app allows it.
* Avoid motion blur: `capture` scores each frame with the variance of the Laplacian and drops
  the blurriest ones (`--min-blur`, `--keep-sharpest`).
* Matte, textured surfaces reconstruct well. Glass, mirrors and plain white walls do not.
* Use `--matcher exhaustive` for an unordered photo set, `sequential` (default) for video.

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|--------------|
| capture | `image_to_3d/capture.py` | Reads a video, webcam or folder with OpenCV, resizes to `max_side`, filters blurry frames, writes `images/frame_*.jpg`. |
| sfm | `image_to_3d/colmap.py` | Runs COLMAP feature extraction, matching, mapping and undistortion. `colmap_io.py` parses the `.bin`/`.txt` models into a `Scene` (pinhole `Camera`s + coloured points). |
| init | `image_to_3d/pipeline.py` | One isotropic Gaussian per SfM point; scale from the mean distance to the 3 nearest neighbours, colour from the point, opacity 0.1. |
| train | `image_to_3d/train.py` | 3DGS optimisation: L1 + D-SSIM loss, per-parameter Adam, densification (clone / split), pruning, opacity resets, held-out PSNR. |
| render | `image_to_3d/render_torch.py`, `render_np.py` | Rasterises the Gaussians from the training cameras or an orbit; optional mp4. |

### Renderers

* `render_np.py` is a forward-only NumPy implementation. It is the reference the tests check
  against and the fallback when PyTorch is missing.
* `render_torch.py` is a differentiable pure-PyTorch rasteriser (tile-based, dense compositing
  with a cumulative product). It runs on CPU or GPU and is what `train` uses by default.
* `backend: gsplat` switches training and rendering to the CUDA kernels from
  [gsplat](https://github.com/nerfstudio-project/gsplat), which is one to two orders of magnitude
  faster. Use it for any real scene; the pure-PyTorch path is for development, tests and small
  scenes.

### Output

```
workspaces/<name>/
  images/                 extracted frames
  colmap/                 COLMAP database + raw sparse model
  undistorted/            pinhole images + model used for training
  scene/                  cameras.json + points.npz (our own pose format)
  output/init.ply         initial Gaussians
  output/point_cloud.ply  trained Gaussians (3DGS PLY, viewer compatible)
  renders/orbit/*.png     turntable frames, renders/orbit.mp4
```

`image-to-3d info output/point_cloud.ply` prints statistics;
`image-to-3d export output/point_cloud.ply scene_points.ply` writes a plain coloured point cloud
for tools that do not understand splats.

## Development

```bash
pytest -q                 # unit tests (renderer, PLY/COLMAP I/O, capture, training smoke test)
image-to-3d demo /tmp/d   # end-to-end synthetic run
```

Conventions: cameras follow COLMAP/OpenCV (`x_cam = R x_world + t`, +Z forward, +Y down),
quaternions are `(w, x, y, z)`, Gaussian scales are stored as logs and opacities as logits.

## References

* Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* SIGGRAPH 2023.
* Schönberger, Frahm. *Structure-from-Motion Revisited.* CVPR 2016 (COLMAP).
* Zwicker et al. *EWA Splatting.* IEEE TVCG 2002 (the 2D covariance projection).
