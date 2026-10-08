import cv2
import numpy as np

from image_to_3d.capture import CaptureConfig, blur_score, capture, resize_max, select_frames


def _sharp(h=64, w=96):
    return np.kron(np.random.default_rng(0).integers(0, 255, (h // 8, w // 8, 3)), np.ones((8, 8, 1))).astype(np.uint8)


def test_blur_score_orders_sharpness():
    sharp = _sharp()
    blurry = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert blur_score(sharp) > blur_score(blurry) * 2


def test_resize_max():
    img = np.zeros((300, 600, 3), np.uint8)
    out = resize_max(img, 200)
    assert out.shape[:2] == (100, 200)
    assert resize_max(img, 1000).shape == img.shape


def test_select_frames_filters_and_keeps_order():
    sharp = _sharp()
    blurry = cv2.GaussianBlur(sharp, (31, 31), 12)
    frames = [sharp, blurry, sharp, blurry, sharp]
    kept = select_frames(frames, CaptureConfig(min_blur=blur_score(blurry) + 1, max_side=None))
    assert len(kept) == 3
    kept = select_frames(frames, CaptureConfig(keep_sharpest_ratio=0.6, max_side=None))
    assert len(kept) == 3 and all(np.array_equal(k, sharp) for k in kept)


def test_capture_from_video_and_folder(tmp_path):
    video = tmp_path / "in.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    for _ in range(12):
        vw.write(_sharp())
    vw.release()
    paths = capture(str(video), tmp_path / "frames", CaptureConfig(every_nth=3, max_side=None, jpeg_quality=90))
    assert len(paths) == 4 and paths[0].name == "frame_00000.jpg"
    again = capture(str(tmp_path / "frames"), tmp_path / "frames2", CaptureConfig(max_side=48))
    assert len(again) == 4
    assert cv2.imread(str(again[0])).shape[:2] == (32, 48)
