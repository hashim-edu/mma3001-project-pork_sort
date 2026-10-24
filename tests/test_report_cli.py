"""Tests for reports and the CLI's handling of good and bad inputs (detector mocked)."""

import json

import numpy as np
from PIL import Image

import porksort.detector as detector_mod
from porksort import cli
from porksort.dataset import Box
from porksort.report import InspectionReport, annotate, input_warnings, write_report


class FakeDetector:
    """Stands in for the ONNX model: one unsealed box at 0.8 on every frame."""

    def __init__(self, *a, **k):
        self.last_timing = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}

    def detect(self, rgb):
        return [Box("unsealed", 0.5, 0.5, 0.4, 0.4, 0.8), Box("wrinkle", 0.2, 0.2, 0.1, 0.1, 0.05)]


def test_input_warnings():
    assert input_warnings(np.full((540, 720, 3), 100, np.uint8)) == ["frame is nearly uniform; camera may be blocked or the frame blank"]
    w = input_warnings(np.random.default_rng(0).integers(0, 10, (100, 100, 3), dtype=np.uint8))
    assert any("differs" in x for x in w) and any("very dark" in x for x in w)
    assert input_warnings(np.random.default_rng(0).integers(0, 255, (540, 720, 3), dtype=np.uint8)) == []


def test_write_report_and_annotate(tmp_path):
    rgb = np.zeros((54, 72, 3), np.uint8)
    rep = InspectionReport("f.jpg", "FAIL", 0.8, 0.3,
                           [{"class": "unsealed", "confidence": 0.8, "bbox_px": [10, 10, 40, 40], "area_frac": 0.1}])
    p = write_report(rep, rgb, tmp_path)
    data = json.loads(p.read_text())
    assert data["status"] == "FAIL" and data["defects"][0]["class"] == "unsealed"
    assert (tmp_path / "f.png").exists()
    assert annotate(rgb, rep).size == (72, 54)


def test_cli_inspect_mixed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(detector_mod, "OnnxYoloDetector", FakeDetector)
    img_dir = tmp_path / "in"
    img_dir.mkdir()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (540, 720, 3), dtype=np.uint8)).save(img_dir / "a.jpg")
    (img_dir / "notes.txt").write_text("not an image")
    (img_dir / "broken.jpg").write_text("corrupt")
    weights = tmp_path / "w.onnx"
    weights.write_text("")
    out = tmp_path / "out"
    code = cli.main(["inspect", str(img_dir), "--weights", str(weights), "--threshold", "0.3", "--out", str(out)])
    assert code == 2  # some inputs were errors
    a = json.loads((out / "a.json").read_text())
    assert a["status"] == "FAIL"
    assert [d["class"] for d in a["defects"]] == ["unsealed"]  # wrinkle at 0.05 is below threshold
    assert json.loads((out / "notes.json").read_text())["status"] == "ERROR"
    assert json.loads((out / "broken.json").read_text())["status"] == "ERROR"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"] == {"PASS": 0, "FAIL": 1, "ERROR": 2}
    assert (out / "index.html").exists()


def test_cli_bad_arguments(tmp_path):
    assert cli.main(["inspect", str(tmp_path), "--threshold", "1.7", "--weights", str(tmp_path / "w.onnx")]) == 1
    assert cli.main(["inspect", str(tmp_path / "empty_dir_missing"), "--threshold", "0.5",
                     "--weights", str(tmp_path / "w.onnx")]) == 1


def test_resolve_threshold_by_weights_and_cost_ratio(tmp_path):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    w = run / "weights" / "best_fp32_480.onnx"
    w.write_text("")
    (run / "threshold.json").write_text(json.dumps({"best_fp32_480.onnx": {"1": 0.13, "10": 0.007}}))
    assert cli.resolve_threshold("auto", w, 10) == 0.007
    assert cli.resolve_threshold("auto", w, 1.0) == 0.13
    assert cli.resolve_threshold("0.4", w) == 0.4
    import pytest
    with pytest.raises(ValueError, match="no threshold"):
        cli.resolve_threshold("auto", w, 7)
