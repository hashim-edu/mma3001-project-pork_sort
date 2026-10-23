"""End-to-end test of the ONNX detector and FLOP profiler on a tiny synthetic network.

The network has the same input and output layout as an exported YOLOv8 model
(1x3xSxS in, 1x(4+5)xN out) but random weights, so it checks the plumbing,
not accuracy. Skipped if the ``onnx`` package is not installed.
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from porksort.detector import OnnxYoloDetector  # noqa: E402
from porksort.perf import conv2d_flops, onnx_conv_profile  # noqa: E402

S = 64


def make_tiny_yolo(path, dtype=TensorProto.FLOAT):
    rng = np.random.default_rng(0)
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    W = numpy_helper.from_array(rng.normal(0, 0.5, (9, 3, 3, 3)).astype(np_dtype), "W")
    B = numpy_helper.from_array(np.zeros(9, np_dtype), "B")
    shape = numpy_helper.from_array(np.array([1, 9, -1], np.int64), "shape")
    scale = numpy_helper.from_array(np.array(S, np_dtype).reshape(1), "scale")
    nodes = [
        helper.make_node("Conv", ["images", "W", "B"], ["c"], name="conv0", kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1], strides=[8, 8]),
        helper.make_node("Sigmoid", ["c"], ["s"]),
        helper.make_node("Reshape", ["s", "shape"], ["r"]),
        helper.make_node("Mul", ["r", "scale"], ["output0"]),
    ]
    g = helper.make_graph(nodes, "tiny",
                          [helper.make_tensor_value_info("images", dtype, [1, 3, S, S])],
                          [helper.make_tensor_value_info("output0", dtype, [1, 9, None])],
                          initializer=[W, B, shape, scale])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def test_detector_runs_and_returns_valid_boxes(tmp_path):
    p = tmp_path / "tiny.onnx"
    make_tiny_yolo(p)
    det = OnnxYoloDetector(p, conf=0.0, iou=0.5)
    assert det.size == S and det.dtype == np.float32
    rgb = np.random.default_rng(1).integers(0, 255, (54, 72, 3), dtype=np.uint8)
    boxes = det.detect(rgb)
    assert boxes, "expected some boxes at conf=0"
    for b in boxes:
        assert 0 <= b.xc <= 1 and 0 <= b.yc <= 1 and 0 < b.w <= 1 and 0 < b.h <= 1
        assert 0 <= b.conf <= S  # scores were multiplied by S in this toy graph
    assert set(det.last_timing) == {"preprocess", "inference", "postprocess"}


def test_fp16_model_input_is_cast(tmp_path):
    p = tmp_path / "tiny16.onnx"
    make_tiny_yolo(p, TensorProto.FLOAT16)
    det = OnnxYoloDetector(p, conf=0.0)
    assert det.dtype == np.float16
    x, _, _ = det.preprocess(np.zeros((54, 72, 3), np.uint8))
    assert x.dtype == np.float16 and x.shape == (1, 3, S, S)


def test_conv_profile_matches_formula(tmp_path):
    p = tmp_path / "tiny.onnx"
    make_tiny_yolo(p)
    rows = onnx_conv_profile(p)
    assert len(rows) == 1
    r = rows[0]
    assert (r["c_in"], r["c_out"], r["k"], r["h_out"], r["w_out"]) == (3, 9, 3, 8, 8)
    assert r["flops"] == conv2d_flops(3, 9, 3, 3, 8, 8)
    # bytes: weights 9*3*3*3 + bias 9 + input 3*64*64 + output 9*8*8, 4 bytes each
    assert r["bytes"] == 4 * (243 + 9 + 3 * 64 * 64 + 9 * 64)


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        OnnxYoloDetector(tmp_path / "none.onnx")
