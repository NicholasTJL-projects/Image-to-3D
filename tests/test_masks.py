import numpy as np
import pytest

from image_to_3d import pipeline
from image_to_3d.masks import dilate_mask, hull_filter, outside_votes
from image_to_3d.render_np import render
from image_to_3d.scene import Scene
from image_to_3d.synthetic import synthetic_scene


def _scene_with_masks():
    gt, scene = synthetic_scene(n_gaussians=40, n_cameras=8, width=48, height=48, point_noise=0.02)
    masks = []
    for cam in scene.cameras:
        _, _, acc = render(gt, cam, return_depth=True)
        masks.append((acc > 0.3).astype(np.float32))
    return gt, scene, masks


def test_dilate_grows_mask():
    m = np.zeros((20, 20), np.float32)
    m[8:12, 8:12] = 1
    assert dilate_mask(m, 2).sum() > m.sum()
    assert np.array_equal(dilate_mask(m, 0), m)


def test_hull_filter_keeps_object_points_and_drops_outliers():
    gt, scene, masks = _scene_with_masks()
    outliers = np.array([[3.0, 0.0, 0.0], [0.0, 2.5, 0.0], [-2.0, -2.0, 1.0]])
    pts = np.concatenate([gt.means, outliers], 0)
    keep = hull_filter(pts, scene.cameras, masks, min_votes=2, margin_frac=0.02)
    assert keep[: len(gt.means)].mean() > 0.9
    assert not keep[len(gt.means):].any()
    votes = outside_votes(outliers, scene.cameras, masks)
    assert (votes >= 2).all()


def test_init_gaussians_with_masks_filters_points(tmp_path):
    gt, scene, masks = _scene_with_masks()
    outliers = np.random.default_rng(0).uniform(2.0, 3.0, size=(30, 3))
    scene = Scene(scene.cameras, np.concatenate([gt.means, outliers], 0),
                  np.concatenate([(gt.colors * 255).astype(np.uint8), np.zeros((30, 3), np.uint8)], 0))
    cloud = pipeline.init_gaussians(scene, tmp_path, masks={c.name: m for c, m in zip(scene.cameras, masks)})
    assert len(gt.means) * 0.9 <= len(cloud) < len(gt.means) + 5


def test_object_only_training_prunes_background():
    torch = pytest.importorskip("torch")
    from image_to_3d.gaussians import GaussianCloud
    from image_to_3d.train import TrainConfig, train

    gt, scene, masks = _scene_with_masks()
    images = {c.name: render(gt, c).astype(np.float32) for c in scene.cameras}
    bg_pts = np.random.default_rng(1).uniform(2.0, 3.0, size=(25, 3))
    init = GaussianCloud.from_points(np.concatenate([gt.means, bg_pts], 0), None, initial_opacity=0.5)
    cfg = TrainConfig(iterations=12, device="cpu", densify=False, object_only=True, hull_prune_interval=6,
                      log_every=1000, eval_every=6, test_every_nth=4)
    out = train(scene, init, cfg, images=images, masks={c.name: m for c, m in zip(scene.cameras, masks)},
                log=lambda s: None)
    assert len(out) <= len(gt.means) + 2  # background seeds removed by the hull prune
