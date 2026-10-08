import numpy as np
import pytest

from image_to_3d.depth import depth_inflate, normalise_inverse_depth
from image_to_3d.gaussians import GaussianCloud
from image_to_3d.single_image import SingleImageConfig, back_project, build_mesh, reconstruct, segment_grabcut, tighten_mask


def disc_image(h=96, w=128, with_alpha=False):
    import cv2

    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (30, 60, 90)
    cv2.circle(img, (w // 2, h // 2), 30, (220, 120, 40), -1)
    if with_alpha:
        a = np.zeros((h, w), np.uint8)
        cv2.circle(a, (w // 2, h // 2), 30, 255, -1)
        return np.dstack([img, a])
    return img


FAST = SingleImageConfig(depth_backend="inflate", segmenter="grabcut", max_side=128, smooth_depth=1)


def test_inflate_depth_peaks_in_the_middle():
    mask = np.zeros((50, 80), np.float32)
    mask[10:40, 20:60] = 1
    d = depth_inflate(np.zeros((50, 80, 3), np.uint8), mask)
    assert d.max() == pytest.approx(1.0) and d[0, 0] == 0
    assert d[25, 40] > d[12, 22]


def test_normalise_inverse_depth_uses_mask():
    inv = np.zeros((10, 10), np.float32)
    inv[2:8, 2:8] = np.linspace(5, 9, 36).reshape(6, 6)
    mask = (inv > 0).astype(np.float32)
    out = normalise_inverse_depth(inv, mask, clip_pct=0)
    assert out[2, 2] == pytest.approx(0.0) and out[7, 7] == pytest.approx(1.0)
    assert out[0, 0] == 0.0  # outside pixels clip to the far end


def test_grabcut_finds_the_disc():
    m = segment_grabcut(disc_image())
    assert m.shape == (96, 128)
    assert m[48, 64] > 0.5 and m[2, 2] < 0.5


def test_tighten_mask_fills_holes_and_drops_islands():
    mask = np.zeros((60, 60), np.float32)
    mask[10:50, 10:50] = 1
    mask[25:30, 25:30] = 0      # hole
    mask[2:5, 55:58] = 1        # island
    out = tighten_mask(mask)
    assert out[27, 27] > 0.4 and out[3, 56] == 0


def test_back_project_geometry():
    inv = np.zeros((20, 40), np.float32)
    pts = back_project(inv, fov_deg=60.0, relief=0.5)
    assert pts.shape == (20, 40, 3)
    assert np.allclose(pts[..., 2], pts[0, 0, 2])  # constant depth plane
    assert pts[10, 39, 0] > pts[10, 0, 0]          # x grows to the right
    assert pts[0, 20, 1] > pts[19, 20, 1]          # y grows upwards
    near = back_project(np.ones_like(inv), 60.0, 0.5)
    assert near[0, 0, 2] > pts[0, 0, 2]            # inverse depth 1 is closer to the viewer (z towards +)


def test_build_mesh_cuts_edges_and_mirrors():
    inv = np.zeros((30, 30), np.float32)
    inv[:, 15:] = 1.0  # a depth cliff down the middle
    mask = np.ones((30, 30), np.float32)
    pts = back_project(inv, 50.0, 0.5)
    v, f, uv = build_mesh(pts, mask, edge_threshold=0.1, mirror_back=False)
    assert len(v) == 900 and len(uv) == 900
    # no triangle may span the cliff: all vertex columns of a face lie on the same side
    cols = (np.arange(900) % 30)[f]
    assert not np.any((cols.min(1) < 15) & (cols.max(1) >= 15))
    v2, f2, _ = build_mesh(pts, mask, edge_threshold=0.1, mirror_back=True)
    assert len(v2) == 1800 and len(f2) == 2 * len(f)
    assert np.isclose(v2[:, 0].max() - v2[:, 0].min(), 1.0)  # width normalised to 1


def test_reconstruct_end_to_end(tmp_path):
    res = reconstruct(disc_image(), FAST)
    assert res.timings["depth_backend"] == "inflate"
    assert len(res.faces) > 100 and res.cloud is not None and len(res.cloud) > 100
    paths = res.save(tmp_path, cfg=FAST)
    assert paths["glb"].stat().st_size > 1000 and paths["splat"].exists() and paths["meta"].exists()
    back = GaussianCloud.load_ply(paths["splat"])
    assert len(back) == len(res.cloud)
    import trimesh

    mesh = trimesh.load(str(paths["glb"]), force="mesh")
    assert len(mesh.faces) == len(res.faces)


def test_reconstruct_honours_alpha_channel():
    res = reconstruct(disc_image(with_alpha=True), FAST)
    assert res.timings["segmenter"] == "alpha"
    assert res.mask[48, 64] > 0.9 and res.mask[0, 0] == 0


def test_relief_changes_depth_extent():
    flat = reconstruct(disc_image(), SingleImageConfig(**{**FAST.__dict__, "relief": 0.1, "mirror_back": False}))
    deep = reconstruct(disc_image(), SingleImageConfig(**{**FAST.__dict__, "relief": 0.8, "mirror_back": False}))
    assert np.ptp(deep.vertices[:, 2]) > 3 * np.ptp(flat.vertices[:, 2])


def test_midas_small_if_weights_cached():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from image_to_3d import depth as d

    try:
        d.midas_weights_path(download=False)
    except FileNotFoundError:
        pytest.skip("MiDaS weights not cached")
    inv = d.estimate_depth(disc_image(), "midas_small")
    assert inv.shape == (96, 128) and 0 <= inv.min() and inv.max() <= 1
