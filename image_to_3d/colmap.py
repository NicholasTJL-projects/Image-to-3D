"""Thin wrapper around the COLMAP command line for Structure-from-Motion.

COLMAP is an external binary (https://colmap.github.io). Install it with
``apt install colmap`` / ``brew install colmap`` or the Windows release zip.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .colmap_io import load_colmap_scene
from .scene import Scene


def colmap_available(binary: str = "colmap") -> bool:
    return shutil.which(binary) is not None


def colmap_has_cuda(binary: str = "colmap") -> bool:
    """COLMAP builds without CUDA refuse ``--SiftExtraction.use_gpu 1``; detect that from the banner."""
    try:
        out = subprocess.run([binary, "-h"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    banner = (out.stdout + out.stderr).lower()
    return "with cuda" in banner and "without cuda" not in banner


def _run(cmd: list[str], log: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if log is not None:
        with open(log, "a") as fh:
            fh.write("$ " + " ".join(cmd) + "\n" + result.stdout + result.stderr + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP step failed ({result.returncode}):\n{result.stderr[-4000:]}")


def run_sfm(
    workspace: str | Path,
    *,
    images_dir: str | Path | None = None,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    matcher: str = "sequential",
    use_gpu: bool | None = None,
    binary: str = "colmap",
) -> Scene:
    """Run feature extraction, matching, mapping and undistortion.

    Layout produced inside ``workspace``::

        images/              input frames (from ``capture``)
        colmap/database.db
        colmap/sparse/0/     raw sparse model (with distortion)
        undistorted/images/  undistorted frames  <- used for training
        undistorted/sparse/  PINHOLE model matching those frames

    Returns the undistorted :class:`Scene`.
    """
    if not colmap_available(binary):
        raise RuntimeError(
            "COLMAP binary not found. Install it (https://colmap.github.io/install.html) "
            "or pass --colmap-bin. Alternatively run SfM elsewhere and point `init` at the sparse/ folder."
        )
    workspace = Path(workspace)
    images_dir = Path(images_dir) if images_dir else workspace / "images"
    colmap_dir = workspace / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db = colmap_dir / "database.db"
    log = colmap_dir / "colmap.log"
    if use_gpu is None:  # auto: only when the binary was built with CUDA
        use_gpu = colmap_has_cuda(binary)
    gpu = "1" if use_gpu else "0"

    _run(
        [
            binary, "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--SiftExtraction.use_gpu", gpu,
        ],
        log,
    )
    if matcher == "sequential":  # video frames: neighbours overlap, cheap and reliable
        _run([binary, "sequential_matcher", "--database_path", str(db),
              "--SequentialMatching.overlap", "10", "--SequentialMatching.loop_detection", "0",
              "--SiftMatching.use_gpu", gpu], log)
    elif matcher == "exhaustive":  # unordered photo sets
        _run([binary, "exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu], log)
    else:
        raise ValueError("matcher must be 'sequential' or 'exhaustive'")

    sparse = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    _run([binary, "mapper", "--database_path", str(db), "--image_path", str(images_dir),
          "--output_path", str(sparse)], log)
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError("COLMAP mapper produced no model: not enough overlap between frames.")
    # the largest model (most images) is the one we want
    best = max(models, key=lambda p: (p / "images.bin").stat().st_size if (p / "images.bin").exists() else 0)

    undist = workspace / "undistorted"
    _run([binary, "image_undistorter", "--image_path", str(images_dir), "--input_path", str(best),
          "--output_path", str(undist), "--output_type", "COLMAP"], log)
    scene = load_colmap_scene(undist / "sparse", undist / "images")
    scene.save(workspace / "scene")
    return scene
