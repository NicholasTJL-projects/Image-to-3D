import numpy as np

from image_to_3d.coverage import assess_coverage
from image_to_3d.scene import Scene
from image_to_3d.synthetic import ring_cameras


def _scene(cams, n_pts=3000):
    pts = np.random.default_rng(0).normal(scale=0.3, size=(n_pts, 3))
    return Scene(cams, pts, np.zeros((n_pts, 3), np.uint8))


def test_full_two_ring_capture_is_good():
    cams = ring_cameras(24, radius=3.0, height=1.0) + ring_cameras(12, radius=3.5, height=2.5)
    c = assess_coverage(_scene(cams))
    assert c["verdict"] == "good" and c["azimuth_coverage_deg"] > 300 and c["elevation_range_deg"] > 15


def test_one_sided_capture_is_flagged():
    cams = [c for c in ring_cameras(24, radius=3.0, height=1.0)][:6]  # a 75-degree arc
    c = assess_coverage(_scene(cams, n_pts=400), photos_submitted=10)
    assert c["verdict"] == "poor"
    text = " ".join(c["advice"])
    assert "could not be matched" in text and "circle" in text and "textured surface" in text


def test_too_few_cameras():
    c = assess_coverage(_scene(ring_cameras(1)))
    assert c["verdict"] == "poor" and c["cameras_posed"] == 1
