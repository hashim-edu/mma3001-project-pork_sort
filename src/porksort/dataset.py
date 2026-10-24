"""Reading the Roboflow YOLOv8 export and building an image index.

The export has one ``.txt`` label file per image. Each line is one bounding
box in normalised YOLO format::

    <class_id> <x_centre> <y_centre> <width> <height>

with all four coordinates in [0, 1] relative to the image size. An empty file
means the annotator found no defect, which this project treats as a PASS frame.

Roboflow names augmented copies ``<source>.rf.<hash>.jpg``. All copies of one
source frame must stay in the same split, otherwise the model is tested on a
brightness-shifted copy of an image it trained on. `source_id` recovers the
source name so the splitting code can group copies.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, DEFAULT_DATA_DIR, DEFECT_CLASSES

_FRAME_RE = re.compile(r"frame_with_red_(\d+)_jpg$")
SPLITS: tuple[str, ...] = ("train", "valid", "test")


@dataclass(frozen=True)
class Box:
    """One annotated or predicted bounding box.

    Attributes:
        cls: Class name, one of `porksort.config.CLASS_NAMES`.
        xc: Box centre x as a fraction of image width, in [0, 1].
        yc: Box centre y as a fraction of image height, in [0, 1].
        w: Box width as a fraction of image width, in (0, 1].
        h: Box height as a fraction of image height, in (0, 1].
        conf: Confidence in [0, 1]. Ground-truth boxes use 1.0.
    """

    cls: str
    xc: float
    yc: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def area(self) -> float:
        """Box area as a fraction of the image area."""
        return self.w * self.h

    def to_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Convert to corner coordinates in pixels.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            ``(x0, y0, x1, y1)``: the top-left and bottom-right corners in pixels.
        """
        return (
            (self.xc - self.w / 2) * width,
            (self.yc - self.h / 2) * height,
            (self.xc + self.w / 2) * width,
            (self.yc + self.h / 2) * height,
        )


def parse_label_file(path: str | os.PathLike) -> list[Box]:
    """Parse one YOLO label file.

    Args:
        path: Path to a ``.txt`` label file.

    Returns:
        The boxes in the file, possibly an empty list (a PASS frame).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line does not have 5 fields, the class id is not a
            known class, or a coordinate is outside [0, 1]. The message names
            the file and line so a bad annotation can be found and fixed.
    """
    boxes: list[Box] = []
    text = Path(path).read_text()
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        # Blank lines are allowed; some export tools end the file with one.
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
        try:
            cid = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: non-numeric field") from exc
        # Validate here, so a bad annotation fails loudly with its file and line
        # instead of silently becoming a wrong label later.
        if not 0 <= cid < len(CLASS_NAMES):
            raise ValueError(f"{path}:{lineno}: unknown class id {cid}")
        for name, v in (("xc", xc), ("yc", yc), ("w", w), ("h", h)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{path}:{lineno}: {name}={v} outside [0, 1]")
        boxes.append(Box(CLASS_NAMES[cid], xc, yc, w, h))
    return boxes


def source_id(filename: str) -> str:
    """Return the source-frame name shared by all augmented copies.

    Args:
        filename: Image file name or path.

    Returns:
        The part before ``.rf.`` for a Roboflow copy, otherwise the file stem.

    Examples:
        >>> source_id("frame_with_red_10049_jpg.rf.cd786c4f.jpg")
        'frame_with_red_10049_jpg'
        >>> source_id("my_photo.jpg")
        'my_photo'
    """
    stem = Path(filename).name
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return Path(stem).stem


def frame_number(source: str) -> int | None:
    """Extract the video frame number from a source name.

    Args:
        source: Source-frame name from `source_id`.

    Returns:
        The frame number, or None if the name does not follow the export's
        ``frame_with_red_<n>_jpg`` pattern.

    Examples:
        >>> frame_number("frame_with_red_10049_jpg")
        10049
        >>> frame_number("other") is None
        True
    """
    m = _FRAME_RE.search(source)
    return int(m.group(1)) if m else None


def is_fail(boxes: list[Box]) -> bool:
    """Ground-truth label of one frame.

    Args:
        boxes: Annotated boxes of the frame.

    Returns:
        True (FAIL) if any box is a defect class, False (PASS) otherwise.
    """
    return any(b.cls in DEFECT_CLASSES for b in boxes)


def build_index(root: str | os.PathLike | None = None) -> pd.DataFrame:
    """Index every image in the export, one row per image file.

    Args:
        root: Folder that contains ``train/``, ``valid/`` and ``test/``.
            Defaults to the ``PORKSORT_DATA`` environment variable, then
            `porksort.config.DEFAULT_DATA_DIR`.

    Returns:
        DataFrame with columns ``rf_split``, ``image``, ``label``, ``source``,
        ``frame``, ``copy`` (0, 1, 2 ... within a source), ``n_boxes``,
        ``fail`` (bool), ``classes`` ("|"-joined class names present) and one
        count column ``n_<class>`` per class. Rows are sorted by split then
        filename, so the index is deterministic.

    Raises:
        FileNotFoundError: If ``root`` has none of the expected split folders,
            or an image has no matching label file.
    """
    root = Path(root or os.environ.get("PORKSORT_DATA", DEFAULT_DATA_DIR))
    rows = []
    found_any = False
    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            continue
        found_any = True
        for img in sorted(img_dir.iterdir()):
            # Skip hidden files such as macOS ._frame.jpg metadata.
            if img.name.startswith("."):
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                raise FileNotFoundError(f"no label file for {img}")
            boxes = parse_label_file(lbl)
            src = source_id(img.name)
            row = {
                "rf_split": split,
                "image": str(img),
                "label": str(lbl),
                "source": src,
                "frame": frame_number(src),
                "n_boxes": len(boxes),
                "fail": is_fail(boxes),
                "classes": "|".join(sorted({b.cls for b in boxes})),
            }
            for c in CLASS_NAMES:
                row[f"n_{c}"] = sum(b.cls == c for b in boxes)
            rows.append(row)
    if not found_any:
        raise FileNotFoundError(
            f"no train/valid/test folders under {root}. Unzip the Roboflow export there or set PORKSORT_DATA."
        )
    df = pd.DataFrame(rows)
    # Number the augmented copies of each source frame 0, 1, 2 ...; evaluation keeps copy 0 only.
    df["copy"] = df.groupby("source").cumcount()
    return df


def unique_sources(index: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per source frame, for splitting and audits.

    Args:
        index: Output of `build_index`.

    Returns:
        The first copy (by file name) of each source frame, re-indexed from 0.
    """
    return index.sort_values(["source", "image"]).drop_duplicates("source").reset_index(drop=True)
