import numpy as np

from image_to_3d.camera import Camera
from image_to_3d.gaussians import GaussianCloud, inverse_sigmoid, rgb_to_sh
from image_to_3d.render_np import project_gaussians, psnr, render
from image_to_3d.synthetic import synthetic_scene


def single_gaussian(opacity=0.9, scale=0.1, colour=(1.0, 0.0, 0.0)):
    return GaussianCloud(
        means=np.zeros((1, 3)), log_scales=np.full((1, 3), np.log(scale)), quats=np.array([[1.0, 0, 0, 0]]),
        logit_opacity=np.array([[inverse_sigmoid(np.array(opacity))]]), features_dc=rgb_to_sh(np.array([colour])),
    )


def test_single_gaussian_peaks_at_centre():
    cam = Camera.look_at([0, 0, -3.0], [0, 0, 0.0], width=65, height=65, focal=100)
    img, depth, acc = render(single_gaussian(), cam, return_depth=True)
    cy, cx = np.unravel_index(np.argmax(img[..., 0]), img.shape[:2])
    assert (cy, cx) == (32, 32)
    assert np.isclose(img[32, 32, 0], 0.9, atol=0.02)   # opacity at the centre
    assert img[32, 32, 1] == 0 and img[32, 32, 2] == 0  # pure red
    assert np.isclose(depth[32, 32], 3.0, atol=0.05)
    assert acc[0, 0] == 0.0 and img[0, 0].sum() == 0.0  # corners untouched


def test_background_shows_through():
    cam = Camera.look_at([0, 0, -3.0], [0, 0, 0.0], width=33, height=33, focal=100)
    img = render(single_gaussian(opacity=0.5, colour=(0, 0, 1.0)), cam, background=(1.0, 1.0, 1.0))
    assert np.allclose(img[0, 0], [1, 1, 1])
    assert np.isclose(img[16, 16, 0], 0.5, atol=0.02)  # half red-channel background through alpha 0.5


def test_front_to_back_compositing():
    # two gaussians on the optical axis: the nearer green one should dominate the pixel
    cloud = GaussianCloud(
        means=np.array([[0, 0, 0.0], [0, 0, -1.0]]), log_scales=np.full((2, 3), np.log(0.1)),
        quats=np.array([[1.0, 0, 0, 0]] * 2), logit_opacity=np.full((2, 1), inverse_sigmoid(np.array(0.8))),
        features_dc=rgb_to_sh(np.array([[1.0, 0, 0], [0, 1.0, 0]])),
    )
    cam = Camera.look_at([0, 0, -4.0], [0, 0, 0.0], width=33, height=33, focal=100)
    img = render(cloud, cam)
    assert img[16, 16, 1] > 0.75 and img[16, 16, 0] < 0.2
    proj = project_gaussians(cloud, cam)
    assert list(proj.index) == [1, 0]  # sorted by depth


def test_gaussian_behind_camera_is_culled():
    cam = Camera.look_at([0, 0, -3.0], [0, 0, 0.0], width=16, height=16, focal=20)
    cloud = single_gaussian()
    cloud.means[:] = [0, 0, -10.0]
    assert len(project_gaussians(cloud, cam).index) == 0
    assert render(cloud, cam).sum() == 0


def test_anisotropic_gaussian_is_elongated():
    cloud = single_gaussian()
    cloud.log_scales[:] = np.log([0.5, 0.05, 0.05])
    cam = Camera.look_at([0, 0, -3.0], [0, 0, 0.0], width=65, height=65, focal=100)
    img = render(cloud, cam)[..., 0]
    assert (img[32, :] > 0.1).sum() > 3 * (img[:, 32] > 0.1).sum()


def test_synthetic_views_differ_and_psnr_sane():
    gt, scene = synthetic_scene(n_gaussians=50, n_cameras=4, width=32, height=32)
    a = render(gt, scene.cameras[0])
    b = render(gt, scene.cameras[2])
    assert a.shape == (32, 32, 3)
    assert psnr(a, a) == float("inf")
    assert psnr(a, b) < 30
