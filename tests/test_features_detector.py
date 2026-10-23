"""Tests for feature extraction, letterboxing, IoU, NMS and output decoding."""

import numpy as np
import pytest
from PIL import Image

from porksort.detector import box_iou, decode, letterbox, nms
from porksort.features import FeatureConfig, extract, group_slices, load_rgb, resize
from porksort.perf import conv2d_flops, mlp_flops, roofline


# ---------- features ----------

def test_feature_length_is_independent_of_input_size():
    rng = np.random.default_rng(0)
    a = extract(rng.integers(0, 255, (540, 720, 3), dtype=np.uint8))
    b = extract(rng.integers(0, 255, (100, 133, 3), dtype=np.uint8))
    assert a.shape == b.shape
    assert sum(s.stop - s.start for s in group_slices().values()) == a.size
    assert np.isfinite(a).all()


def test_features_deterministic_and_histograms_normalised():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (54, 72, 3), dtype=np.uint8)
    f1, f2 = extract(img), extract(img)
    assert np.array_equal(f1, f2)
    sl = group_slices()
    assert f1[sl["colour"]].sum() == pytest.approx(1.0)
    assert f1[sl["texture"]].sum() == pytest.approx(1.0)


def test_feature_errors(tmp_path):
    with pytest.raises(ValueError, match="RGB array"):
        extract(np.zeros((10, 10)))
    with pytest.raises(ValueError, match="unknown interpolation"):
        resize(np.zeros((10, 10, 3), np.uint8), (5, 5), "magic")
    with pytest.raises(ValueError, match="unknown feature group"):
        extract(np.zeros((10, 10, 3), np.uint8), FeatureConfig(groups=("nope",)))
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(ValueError, match="unsupported image type"):
        load_rgb(tmp_path / "x.txt")
    (tmp_path / "bad.jpg").write_text("not a jpeg")
    with pytest.raises(ValueError, match="could not decode"):
        load_rgb(tmp_path / "bad.jpg")
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "missing.jpg")


def test_load_rgb_converts_grey(tmp_path):
    Image.fromarray(np.full((20, 30), 128, np.uint8)).save(tmp_path / "g.png")
    a = load_rgb(tmp_path / "g.png")
    assert a.shape == (20, 30, 3) and a.dtype == np.uint8


# ---------- detector maths ----------

def test_letterbox_720x540():
    img = np.zeros((540, 720, 3), np.uint8)
    img[:, :, 0] = 255
    out, r, (dx, dy) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == pytest.approx(640 / 720)
    assert (dx, dy) == (0.0, 80.0)
    assert (out[:80] == 114).all() and (out[-80:] == 114).all()
    assert (out[80:560, :, 0] == 255).all()


def test_box_iou_known_values():
    a = np.array([[0, 0, 2, 2]], float)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou(a, np.array([[3, 3, 4, 4]], float))[0, 0] == 0.0
    # overlap of 2 on an area-4 box and an area-4 box: 2 / (4 + 4 - 2) = 1/3
    assert box_iou(a, np.array([[1, 0, 3, 2]], float))[0, 0] == pytest.approx(1 / 3)


def test_nms_suppresses_overlaps_only():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    scores = np.array([0.9, 0.8, 0.7])
    assert list(nms(boxes, scores, 0.5)) == [0, 2]
    assert list(nms(boxes, scores, 0.95)) == [0, 1, 2]


def test_decode_round_trip():
    """Manufactured solution: encode a known box as network output, decode it back."""
    W, H, S = 720, 540, 640
    _, r, pad = letterbox(np.zeros((H, W, 3), np.uint8), S)
    # true box in normalised image coords
    xc, yc, w, h = 0.25, 0.6, 0.1, 0.2
    cx, cy = xc * W * r + pad[0], yc * H * r + pad[1]
    bw, bh = w * W * r, h * H * r
    out = np.zeros((1, 9, 3), np.float32)
    out[0, :4, 0] = [cx, cy, bw, bh]
    out[0, 4 + 3, 0] = 0.9            # class 3 = unsealed
    out[0, :4, 1] = [cx + 1, cy, bw, bh]
    out[0, 4 + 3, 1] = 0.5            # duplicate, should be removed by NMS
    out[0, :4, 2] = [cx, cy, bw, bh]
    out[0, 4 + 1, 2] = 0.0005         # below confidence threshold
    boxes = decode(out, r, pad, (W, H), conf_thr=0.001, iou_thr=0.6)
    assert len(boxes) == 1
    bx = boxes[0]
    assert bx.cls == "unsealed" and bx.conf == pytest.approx(0.9)
    assert (bx.xc, bx.yc, bx.w, bx.h) == pytest.approx((xc, yc, w, h), abs=1e-5)


def test_decode_rejects_wrong_class_count():
    with pytest.raises(ValueError, match="classes"):
        decode(np.zeros((1, 6, 4)), 1.0, (0, 0), (10, 10), 0.1, 0.5)


# ---------- performance helpers ----------

def test_conv_flops_against_direct_count():
    """Count multiply-adds of a naive convolution loop and compare with the formula."""
    c_in, c_out, k, h, w = 3, 4, 3, 5, 6
    macs = 0
    for _ in range(c_out):
        for _ in range(h):
            for _ in range(w):
                macs += c_in * k * k
    assert conv2d_flops(c_in, c_out, k, k, h, w, bias=False) == 2 * macs
    # depthwise: groups = channels
    assert conv2d_flops(8, 8, 3, 3, 4, 4, groups=8, bias=False) == 2 * 8 * 16 * 9
    with pytest.raises(ValueError):
        conv2d_flops(3, 4, 3, 3, 1, 1, groups=2)


def test_mlp_flops_and_roofline():
    assert mlp_flops([10, 5, 1]) == (2 * 10 * 5 + 5) + (2 * 5 + 1)
    assert roofline(0.5, peak_flops=100.0, bandwidth=50.0) == 25.0   # memory-bound
    assert roofline(10.0, peak_flops=100.0, bandwidth=50.0) == 100.0  # compute-bound
