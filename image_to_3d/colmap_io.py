"""Readers for COLMAP's sparse reconstruction files (``.bin`` and ``.txt``).

Only the subset of the format needed to build a :class:`~image_to_3d.scene.Scene`
is handled: camera intrinsics, image poses and 3D points with colours.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .camera import Camera, qvec2rotmat
from .scene import Scene

# model id -> (name, number of parameters)   (see colmap/src/colmap/sensor/models.h)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}


def intrinsics_from_params(model: str, params: np.ndarray) -> np.ndarray:
    """Build a pinhole ``K`` from a COLMAP camera model's parameter vector.

    Distortion coefficients are ignored; run ``colmap image_undistorter`` first
    (``image_to_3d.colmap.run_sfm`` does) so the images match a pure pinhole.
    """
    params = np.asarray(params, dtype=np.float64)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# binary readers
# --------------------------------------------------------------------------
def _read(fh, fmt: str):
    fmt = "<" + fmt  # little-endian, no padding
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise EOFError("truncated COLMAP binary file")
    return struct.unpack(fmt, data)


def read_cameras_bin(path: Path) -> dict[int, dict]:
    cameras = {}
    with open(path, "rb") as fh:
        (n,) = _read(fh, "Q")
        for _ in range(n):
            cam_id, model_id, w, h = _read(fh, "iiQQ")
            name, num_params = CAMERA_MODELS[model_id]
            params = np.array(_read(fh, "d" * num_params))
            cameras[cam_id] = {"model": name, "width": int(w), "height": int(h), "params": params}
    return cameras


def read_images_bin(path: Path) -> dict[int, dict]:
    images = {}
    with open(path, "rb") as fh:
        (n,) = _read(fh, "Q")
        for _ in range(n):
            (img_id,) = _read(fh, "i")
            qvec = np.array(_read(fh, "dddd"))
            tvec = np.array(_read(fh, "ddd"))
            (cam_id,) = _read(fh, "i")
            name = b""
            while True:
                c = fh.read(1)
                if c == b"\x00" or c == b"":
                    break
                name += c
            (n_pts,) = _read(fh, "Q")
            fh.read(24 * n_pts)  # x, y, point3D_id per 2D observation; not needed
            images[img_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name.decode("utf-8")}
    return images


def read_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with open(path, "rb") as fh:
        (n,) = _read(fh, "Q")
        for _ in range(n):
            _pid, x, y, z, r, g, b, _err = _read(fh, "QdddBBBd")
            (track_len,) = _read(fh, "Q")
            fh.read(8 * track_len)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3)


# --------------------------------------------------------------------------
# text readers
# --------------------------------------------------------------------------
def _data_lines(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def read_cameras_txt(path: Path) -> dict[int, dict]:
    cameras = {}
    for line in _data_lines(path):
        parts = line.split()
        cam_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        cameras[cam_id] = {"model": model, "width": w, "height": h, "params": np.array(parts[4:], dtype=np.float64)}
    return cameras


def read_images_txt(path: Path) -> dict[int, dict]:
    images = {}
    # two lines per image: the pose line, then the (possibly empty) 2D-point line
    lines = [ln.strip() for ln in path.read_text().splitlines() if not ln.strip().startswith("#")]
    while lines and not lines[-1]:
        lines.pop()
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        images[int(parts[0])] = {
            "qvec": np.array(parts[1:5], dtype=np.float64),
            "tvec": np.array(parts[5:8], dtype=np.float64),
            "camera_id": int(parts[8]),
            "name": " ".join(parts[9:]),
        }
    return images


def read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    for line in _data_lines(path):
        parts = line.split()
        xyz.append(tuple(float(v) for v in parts[1:4]))
        rgb.append(tuple(int(v) for v in parts[4:7]))
    return np.array(xyz, dtype=np.float64).reshape(-1, 3), np.array(rgb, dtype=np.uint8).reshape(-1, 3)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def read_model(sparse_dir: str | Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Read ``cameras``, ``images`` and ``points3D`` from a COLMAP sparse directory."""
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "cameras.bin").exists():
        cams = read_cameras_bin(sparse_dir / "cameras.bin")
        imgs = read_images_bin(sparse_dir / "images.bin")
        xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    elif (sparse_dir / "cameras.txt").exists():
        cams = read_cameras_txt(sparse_dir / "cameras.txt")
        imgs = read_images_txt(sparse_dir / "images.txt")
        xyz, rgb = read_points3d_txt(sparse_dir / "points3D.txt")
    else:
        raise FileNotFoundError(f"No COLMAP model (cameras.bin/.txt) found in {sparse_dir}")
    return cams, imgs, xyz, rgb


def load_colmap_scene(sparse_dir: str | Path, images_dir: str | Path) -> Scene:
    """Convert a COLMAP sparse model into a :class:`Scene` with pinhole cameras."""
    sparse_dir, images_dir = Path(sparse_dir), Path(images_dir)
    cams, imgs, xyz, rgb = read_model(sparse_dir)
    cameras: list[Camera] = []
    for img_id in sorted(imgs, key=lambda i: imgs[i]["name"]):
        im = imgs[img_id]
        cam = cams[im["camera_id"]]
        cameras.append(
            Camera(
                width=cam["width"],
                height=cam["height"],
                K=intrinsics_from_params(cam["model"], cam["params"]),
                R=qvec2rotmat(im["qvec"]),
                t=im["tvec"],
                image_path=str(images_dir / im["name"]),
                name=im["name"],
                extra={"colmap_image_id": img_id, "colmap_model": cam["model"]},
            )
        )
    return Scene(cameras, xyz, rgb)


def write_model_txt(scene: Scene, out_dir: str | Path) -> None:
    """Write a :class:`Scene` as a COLMAP text model (handy for tests and for
    feeding other 3DGS trainers that only read COLMAP)."""
    from .camera import rotmat2qvec

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cameras.txt", "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, c in enumerate(scene.cameras, start=1):
            fh.write(f"{i} PINHOLE {c.width} {c.height} {c.fx} {c.fy} {c.cx} {c.cy}\n")
    with open(out_dir / "images.txt", "w") as fh:
        fh.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, c in enumerate(scene.cameras, start=1):
            q = rotmat2qvec(c.R)
            name = c.name or Path(c.image_path or f"image_{i:05d}.png").name
            fh.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {c.t[0]} {c.t[1]} {c.t[2]} {i} {name}\n\n")
    with open(out_dir / "points3D.txt", "w") as fh:
        fh.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, c) in enumerate(zip(scene.points_xyz, scene.points_rgb), start=1):
            fh.write(f"{i} {p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])} 0.0\n")
