import numpy as np
import pytest

torch = pytest.importorskip("torch")

from image_to_3d import render_np  # noqa: E402
from image_to_3d.gaussians import GaussianCloud  # noqa: E402
from image_to_3d.render_torch import TorchCamera, render, ssim  # noqa: E402
from image_to_3d.synthetic import synthetic_scene  # noqa: E402
from image_to_3d.train import TrainConfig, train  # noqa: E402


def _params(cloud):
    return {k: torch.as_tensor(getattr(cloud, k)) for k in ("means", "log_scales", "quats", "logit_opacity", "features_dc")}


def test_torch_renderer_matches_numpy():
    gt, scene = synthetic_scene(n_gaussians=60, n_cameras=3, width=40, height=40)
    for cam in scene.cameras:
        ref = render_np.render(gt, cam)
        out = render(_params(gt), TorchCamera.from_camera(cam, "cpu"))
        assert np.allclose(out.image.numpy(), ref, atol=2e-3)
        assert out.visible.sum() > 0


def test_torch_renderer_tiles_are_seamless():
    gt, scene = synthetic_scene(n_gaussians=60, n_cameras=1, width=70, height=50)  # not a multiple of the tile size
    ref = render_np.render(gt, scene.cameras[0])
    out = render(_params(gt), TorchCamera.from_camera(scene.cameras[0], "cpu"))
    assert out.image.shape == (50, 70, 3)
    assert np.allclose(out.image.numpy(), ref, atol=2e-3)


def test_gradients_flow_to_all_parameters():
    gt, scene = synthetic_scene(n_gaussians=20, n_cameras=1, width=24, height=24)
    params = {k: v.clone().requires_grad_(True) for k, v in _params(gt).items()}
    out = render(params, TorchCamera.from_camera(scene.cameras[0], "cpu"))
    out.image.sum().backward()
    for k, v in params.items():
        assert v.grad is not None and torch.isfinite(v.grad).all(), k
    assert out.means2d.grad is not None


def test_ssim_identity():
    img = torch.rand(20, 20, 3)
    assert ssim(img, img).item() == pytest.approx(1.0, abs=1e-5)
    assert ssim(img, torch.zeros_like(img)).item() < 0.5


def test_training_improves_psnr(tmp_path):
    gt, scene = synthetic_scene(n_gaussians=40, n_cameras=8, width=32, height=32, point_noise=0.08)
    images = {c.name: render_np.render(gt, c).astype(np.float32) for c in scene.cameras}
    init = GaussianCloud.from_points(scene.points_xyz, scene.points_rgb, initial_opacity=0.5)
    before = np.mean([render_np.psnr(render_np.render(init, c), images[c.name]) for c in scene.cameras])
    cfg = TrainConfig(iterations=60, device="cpu", densify=True, densify_from=20, densify_interval=20,
                      opacity_reset_interval=0, log_every=1000, eval_every=0, lr_means=5e-3, lr_means_final=1e-3,
                      test_every_nth=4, max_gaussians=200)
    out = train(scene, init, cfg, images=images, out_dir=tmp_path, log=lambda s: None)
    after = np.mean([render_np.psnr(render_np.render(out, c), images[c.name]) for c in scene.cameras])
    assert after > before + 1.0
    assert (tmp_path / "point_cloud.ply").exists()
    assert len(out) <= 200


def test_load_image_matches_scaled_camera(tmp_path):
    import cv2

    from image_to_3d.camera import Camera
    from image_to_3d.train import load_image

    cv2.imwrite(str(tmp_path / "odd.png"), np.zeros((599, 799, 3), np.uint8))
    cam = Camera.look_at([0, 0, -3.0], [0, 0, 0.0], width=799, height=599, focal=700, name="odd")
    cam.image_path = str(tmp_path / "odd.png")
    small = cam.scaled(0.25)
    img = load_image(cam, 4)
    assert img.shape[:2] == (small.height, small.width) == (150, 200)
