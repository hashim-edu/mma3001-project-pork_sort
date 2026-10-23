"""Project-wide constants.

Everything that defines *what the system means* by a defect lives here, so a
change of inspection policy is a one-line change that the tests can see.
"""

from __future__ import annotations

from pathlib import Path

#: Class names in the order of the YOLO class ids in ``data.yaml``.
CLASS_NAMES: tuple[str, ...] = (
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
)

#: Classes that make a frame FAIL inspection. All five annotated classes are
#: defects, so any one of them fails the frame.
DEFECT_CLASSES: frozenset[str] = frozenset(CLASS_NAMES)

#: Size of every image in the Roboflow v4 export (width, height) in pixels.
#: Roboflow resized with "fill (with centre crop)", so this is fixed.
IMAGE_SIZE: tuple[int, int] = (720, 540)

#: Image file extensions the system accepts.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

#: Default cost ratio used to choose the decision threshold: shipping a
#: defective pack (a false PASS) is treated as this many times worse than
#: rejecting a good pack (a false FAIL).
DEFAULT_COST_RATIO: float = 10.0

#: Root of the repository (two levels above this file).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default location of the unzipped Roboflow export. Override with the
#: ``PORKSORT_DATA`` environment variable or the ``--data`` CLI option.
DEFAULT_DATA_DIR: Path = REPO_ROOT / "data" / "pork_v4"

#: Where scripts write metrics, tables and figures.
RESULTS_DIR: Path = REPO_ROOT / "results"
