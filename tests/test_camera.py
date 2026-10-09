import numpy as np

from image_to_3d.camera import Camera, qvec2rotmat, rotmat2qvec


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = qvec2rotmat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        q2 = rotmat2qvec(R)
        assert np.allclose(q, q2, atol=1e-9) or np.allclose(q, -q2, atol=1e-9)


def test_look_at_projects_target_to_principal_point():
    cam = Camera.look_at([3.0, -1.0, 2.0], [0.0, 0.0, 0.0], width=200, height=100, focal=150)
    uv, z = cam.project(np.zeros((1, 3)))
    assert np.allclose(uv[0], [100, 50])
    assert z[0] > 0
    assert np.allclose(cam.center, [3.0, -1.0, 2.0])
    assert np.allclose(cam.forward, -cam.center / np.linalg.norm(cam.center))


def test_world_up_is_image_up():
    cam = Camera.look_at([0.0, 0.0, -5.0], [0.0, 0.0, 0.0], width=100, height=100, focal=100)
    uv, _ = cam.project(np.array([[0.0, -1.0, 0.0]]))  # a point above the target (-Y is up)
    assert uv[0, 1] < 50  # appears in the upper half of the image


def test_scaled_camera():
    cam = Camera.look_at([0, 0, -5.0], [0, 0, 0.0], width=100, height=60, focal=80)
    half = cam.scaled(0.5)
    assert (half.width, half.height) == (50, 30)
    uv1, _ = cam.project(np.array([[0.3, 0.2, 1.0]]))
    uv2, _ = half.project(np.array([[0.3, 0.2, 1.0]]))
    assert np.allclose(uv1 * 0.5, uv2)
