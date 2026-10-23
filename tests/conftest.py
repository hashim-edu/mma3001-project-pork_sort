"""Shared fixtures: a tiny synthetic dataset in Roboflow YOLOv8 layout.

Tests never need the real dataset, so they run on any machine (and in CI)
in a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _write_image(path: Path, rng: np.random.Generator, size=(72, 54)) -> None:
    a = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(a).save(path)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Dataset with 3 splits; train frames have 3 augmented copies each.

    Frame numbers run 0, 10, 20, ... so blocks and gaps can be tested.
    Every 4th frame is a PASS (empty label); others have one box whose class
    cycles through the five classes.
    """
    rng = np.random.default_rng(0)
    layout = {"train": range(0, 1200, 10), "valid": range(1205, 1405, 10), "test": range(1405, 1605, 10)}
    for split, frames in layout.items():
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        for k, f in enumerate(frames):
            copies = 3 if split == "train" else 1
            for c in range(copies):
                stem = f"frame_with_red_{f}_jpg.rf.{c:02d}{k:04d}"
                _write_image(tmp_path / split / "images" / f"{stem}.jpg", rng)
                label = "" if k % 4 == 0 else f"{k % 5} 0.5 0.5 0.2 0.1\n"
                (tmp_path / split / "labels" / f"{stem}.txt").write_text(label)
    return tmp_path
