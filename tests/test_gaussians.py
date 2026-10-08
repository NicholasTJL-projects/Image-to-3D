import numpy as np

from image_to_3d.gaussians import GaussianCloud, rgb_to_sh, sh_to_rgb
from image_to_3d.synthetic import blob_cloud


def test_sh_colour_roundtrip():
    rgb = np.random.default_rng(0).uniform(size=(10, 3))
    assert np.allclose(sh_to_rgb(rgb_to_sh(rgb)), rgb)


def test_from_points_scales_follow_neighbour_distance():
    grid = np.stack(np.meshgrid(*[np.arange(5.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.2
    cloud = GaussianCloud.from_points(grid, np.full((len(grid), 3), 200))
    assert len(cloud) == len(grid)
    assert np.allclose(cloud.scales, 0.2, atol=1e-3)  # neighbours are 0.2 apart
    assert np.allclose(cloud.colors, 200 / 255, atol=1e-5)
    assert np.allclose(cloud.opacities, 0.1, atol=1e-5)


def test_ply_roundtrip(tmp_path):
    cloud = blob_cloud(50, seed=3)
    cloud.features_rest = np.random.default_rng(1).normal(size=(50, 3, 3)).astype(np.float32)  # SH degree 1
    cloud.save_ply(tmp_path / "c.ply")
    back = GaussianCloud.load_ply(tmp_path / "c.ply")
    assert back.sh_degree == 1
    for k in ("means", "log_scales", "quats", "logit_opacity", "features_dc", "features_rest"):
        assert np.allclose(getattr(cloud, k), getattr(back, k), atol=1e-6), k


def test_covariance_is_rotated_diag():
    cloud = blob_cloud(5, seed=0)
    cov = cloud.covariances()
    assert cov.shape == (5, 3, 3)
    assert np.allclose(cov, cov.transpose(0, 2, 1))
    eig = np.sort(np.linalg.eigvalsh(cov), axis=1)
    assert np.allclose(eig, np.sort(cloud.scales.astype(np.float64) ** 2, axis=1), rtol=1e-4)


def test_export_point_cloud(tmp_path):
    cloud = blob_cloud(20)
    cloud.export_point_cloud(tmp_path / "pts.ply")
    assert (tmp_path / "pts.ply").stat().st_size > 0
