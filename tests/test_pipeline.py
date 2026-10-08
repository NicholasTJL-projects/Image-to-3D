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
