import struct

import numpy as np

from image_to_3d import colmap_io
from image_to_3d.synthetic import synthetic_scene


def test_text_model_roundtrip(tmp_path):
    _, scene = synthetic_scene(n_gaussians=20, n_cameras=5)
    for c in scene.cameras:
        c.image_path = str(tmp_path / "images" / f"{c.name}.png")
    colmap_io.write_model_txt(scene, tmp_path / "sparse")
    loaded = colmap_io.load_colmap_scene(tmp_path / "sparse", tmp_path / "images")
    assert len(loaded) == len(scene)
    for a, b in zip(scene.cameras, loaded.cameras):
        assert np.allclose(a.K, b.K)
        assert np.allclose(a.R, b.R, atol=1e-9)
        assert np.allclose(a.t, b.t, atol=1e-9)
        assert (a.width, a.height) == (b.width, b.height)
    assert np.allclose(loaded.points_xyz, scene.points_xyz)
    assert np.array_equal(loaded.points_rgb, scene.points_rgb)


def _write_bin_model(d, cams, imgs, pts):
    with open(d / "cameras.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(cams)))
        for cid, (model_id, w, h, params) in cams.items():
            fh.write(struct.pack("<iiQQ", cid, model_id, w, h))
            fh.write(struct.pack("<" + "d" * len(params), *params))
    with open(d / "images.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(imgs)))
        for iid, (q, t, cid, name, pts2d) in imgs.items():
            fh.write(struct.pack("<i", iid))
            fh.write(struct.pack("<dddd", *q))
            fh.write(struct.pack("<ddd", *t))
            fh.write(struct.pack("<i", cid))
            fh.write(name.encode() + b"\x00")
            fh.write(struct.pack("<Q", len(pts2d)))
            for x, y, pid in pts2d:
                fh.write(struct.pack("<ddq", x, y, pid))
    with open(d / "points3D.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(pts)))
        for pid, (xyz, rgb, track) in pts.items():
            fh.write(struct.pack("<QdddBBBd", pid, *xyz, *rgb, 0.5))
            fh.write(struct.pack("<Q", len(track)))
            for a, b in track:
                fh.write(struct.pack("<ii", a, b))


def test_binary_model(tmp_path):
    cams = {1: (2, 640, 480, [500.0, 320.0, 240.0, -0.01])}  # SIMPLE_RADIAL
    imgs = {
        7: ([1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3], 1, "b.jpg", [(1.0, 2.0, 3), (4.0, 5.0, -1)]),
        3: ([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, "a.jpg", []),
    }
    pts = {3: ((1.0, 2.0, 3.0), (10, 20, 30), [(7, 0)]), 9: ((-1.0, 0.0, 0.5), (0, 0, 255), [])}
    _write_bin_model(tmp_path, cams, imgs, pts)
    scene = colmap_io.load_colmap_scene(tmp_path, tmp_path / "images")
    assert [c.name for c in scene.cameras] == ["a.jpg", "b.jpg"]  # sorted by name
    assert scene.cameras[1].fx == 500.0 and scene.cameras[1].cx == 320.0
    assert np.allclose(scene.cameras[1].t, [0.1, 0.2, 0.3])
    assert np.allclose(scene.cameras[0].R, np.diag([1.0, -1.0, -1.0]))
    assert scene.points_xyz.shape == (2, 3)
    assert set(map(tuple, scene.points_rgb.tolist())) == {(10, 20, 30), (0, 0, 255)}
