import numpy as np

from image_to_3d import pipeline
from image_to_3d.cli import main
from image_to_3d.scene import Scene
from image_to_3d.synthetic import synthetic_scene


def test_scene_save_load_and_split(tmp_path):
    _, scene = synthetic_scene(n_gaussians=10, n_cameras=16)
    scene.save(tmp_path / "scene")
    back = Scene.load(tmp_path / "scene")
    assert len(back) == 16 and np.allclose(back.points_xyz, scene.points_xyz)
    assert np.allclose(back.cameras[3].R, scene.cameras[3].R)
    train, test = scene.split(8)
    assert (len(train), len(test)) == (14, 2)


def test_load_scene_from_colmap_txt_and_init(tmp_path):
    from image_to_3d.colmap_io import write_model_txt

    _, scene = synthetic_scene(n_gaussians=30, n_cameras=4)
    write_model_txt(scene, tmp_path / "sparse" / "0")
    loaded = pipeline.load_scene(tmp_path)
    assert len(loaded) == 4 and (tmp_path / "scene" / "cameras.json").exists()
    cloud = pipeline.init_gaussians(loaded, tmp_path, random_points=10)
    assert len(cloud) == 40 and (tmp_path / "output" / "init.ply").exists()


def test_orbit_cameras_look_at_scene(tmp_path):
    _, scene = synthetic_scene(n_gaussians=10, n_cameras=12)
    cams = pipeline.orbit_cameras(scene, 8)
    assert len(cams) == 8
    for c in cams:
        uv, z = c.project(np.zeros((1, 3)))
        assert z[0] > 0 and 0 <= uv[0, 0] <= c.width and 0 <= uv[0, 1] <= c.height


def test_orbit_cameras_target_point_cloud_not_ray_overshoot():
    # downward-looking ring (like a hand-held capture): the orbit must aim at the points, not below them
    from image_to_3d.synthetic import ring_cameras

    cams = ring_cameras(12, radius=3.0, height=2.0, target=(0.0, -0.5, 0.0))
    pts = np.random.default_rng(0).normal(scale=0.3, size=(200, 3)) + np.array([0.0, -0.5, 0.0])
    scene = Scene(cams, pts, np.zeros((200, 3), np.uint8))
    for c in pipeline.orbit_cameras(scene, 6):
        uv, z = c.project(np.array([[0.0, -0.5, 0.0]]))
        assert z[0] > 0
        assert abs(uv[0, 0] - c.width / 2) < 1 and abs(uv[0, 1] - c.height / 2) < 1
        assert abs(np.linalg.norm(c.center - np.array([0.0, -0.5, 0.0])) - np.linalg.norm(cams[0].center - np.array([0.0, -0.5, 0.0]))) < 0.3
        # same orientation as the real cameras: camera "up" (-Y axis in world) points the same way
        real_up = -cams[0].R.T @ np.array([0.0, 1.0, 0.0])
        orbit_up = -c.R.T @ np.array([0.0, 1.0, 0.0])
        assert real_up @ orbit_up > 0.7  # same hemisphere; the orbit pitches less than the steep real cameras


def test_cli_init_info_export_render(tmp_path, capsys):
    from image_to_3d.colmap_io import write_model_txt

    _, scene = synthetic_scene(n_gaussians=30, n_cameras=4, width=24, height=24)
    write_model_txt(scene, tmp_path / "sparse")
    assert main(["init", str(tmp_path)]) == 0
    assert main(["info", str(tmp_path / "output" / "init.ply")]) == 0
    assert "30 gaussians" in capsys.readouterr().out
    assert main(["export", str(tmp_path / "output" / "init.ply"), str(tmp_path / "pts.ply"), "--min-opacity", "0"]) == 0
    assert main(["render", str(tmp_path), "--views", "orbit", "--orbit-frames", "3", "--backend", "numpy"]) == 0
    assert len(list((tmp_path / "renders" / "orbit").glob("*.png"))) == 3
