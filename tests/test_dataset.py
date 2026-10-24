"""Tests for label parsing and the image index."""

import pytest

from porksort.dataset import Box, build_index, frame_number, is_fail, parse_label_file, source_id


def test_parse_valid_and_empty(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5 0.2 0.1\n3 0.1 0.2 0.05 0.05\n")
    boxes = parse_label_file(f)
    assert [b.cls for b in boxes] == ["packaging-error", "unsealed"]
    assert boxes[0].area == pytest.approx(0.02)
    (tmp_path / "e.txt").write_text("")
    assert parse_label_file(tmp_path / "e.txt") == []


@pytest.mark.parametrize(
    "line, msg",
    [
        ("1 0.5 0.5 0.2", "expected 5 fields"),
        ("9 0.5 0.5 0.2 0.1", "unknown class id"),
        ("1 1.5 0.5 0.2 0.1", r"outside \[0, 1\]"),
        ("x 0.5 0.5 0.2 0.1", "non-numeric"),
    ],
)
def test_parse_rejects_bad_lines(tmp_path, line, msg):
    f = tmp_path / "bad.txt"
    f.write_text(line + "\n")
    with pytest.raises(ValueError, match=msg):
        parse_label_file(f)


def test_source_and_frame():
    assert source_id("frame_with_red_10049_jpg.rf.abc.jpg") == "frame_with_red_10049_jpg"
    assert frame_number("frame_with_red_10049_jpg") == 10049
    assert frame_number("random") is None


def test_is_fail():
    assert not is_fail([])
    assert is_fail([Box("wrinkle", 0.5, 0.5, 0.1, 0.1)])


def test_box_to_pixels():
    b = Box("unsealed", 0.5, 0.5, 0.5, 0.5)
    assert b.to_pixels(720, 540) == (180.0, 135.0, 540.0, 405.0)


def test_build_index(tiny_dataset):
    idx = build_index(tiny_dataset)
    assert set(idx["rf_split"]) == {"train", "valid", "test"}
    # 120 train frames x 3 copies + 20 + 20
    assert len(idx) == 120 * 3 + 20 + 20
    assert idx.groupby("source").size().max() == 3
    assert set(idx["copy"]) == {0, 1, 2}
    # every 4th frame is PASS
    first = idx[idx["copy"] == 0]
    assert (~first["fail"]).sum() == 30 + 5 + 5


def test_build_index_errors(tmp_path, tiny_dataset):
    with pytest.raises(FileNotFoundError, match="no train/valid/test"):
        build_index(tmp_path / "nothing")
    lbl = next((tiny_dataset / "test" / "labels").iterdir())
    lbl.unlink()
    with pytest.raises(FileNotFoundError, match="no label file"):
        build_index(tiny_dataset)
