"""The pass/fail rule and how its threshold is chosen.

Every model in the project, from the decision tree to YOLO, is reduced to one
number per frame: a *defect score* in [0, 1], where higher means more likely
to be defective. The frame FAILs when the score reaches the threshold::

    FAIL  if  score >= threshold
    PASS  otherwise

For a detector the score is the highest confidence of any defect box. For a
classifier it is the predicted probability (or a squashed decision value) of
the FAIL class.

The threshold is not 0.5 by default. It is chosen on the validation set to
minimise expected cost, where a false PASS (a defective pack shipped) costs
``cost_ratio`` times a false FAIL (a good pack rejected and re-checked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .config import DEFAULT_COST_RATIO, DEFECT_CLASSES
from .dataset import Box


@dataclass
class Decision:
    """Outcome of inspecting one frame.

    Attributes:
        status: ``"PASS"`` or ``"FAIL"``.
        score: Defect score in [0, 1] that was compared with the threshold.
        threshold: Threshold that was applied.
        defects: Boxes at or above the threshold (empty for PASS).
    """

    status: str
    score: float
    threshold: float
    defects: list[Box] = field(default_factory=list)


def frame_score(boxes: Iterable[Box], defect_classes: frozenset[str] = DEFECT_CLASSES) -> float:
    """Highest confidence among defect boxes, or 0.0 if there are none."""
    return max((b.conf for b in boxes if b.cls in defect_classes), default=0.0)


def decide(boxes: Sequence[Box], threshold: float, defect_classes: frozenset[str] = DEFECT_CLASSES) -> Decision:
    """Apply the pass/fail rule to the detections of one frame.

    Args:
        boxes: Detections for the frame, each with a confidence.
        threshold: Confidence at or above which a defect box fails the frame.
        defect_classes: Classes that count as defects.

    Raises:
        ValueError: If ``threshold`` is outside [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    defects = [b for b in boxes if b.cls in defect_classes and b.conf >= threshold]
    score = frame_score(boxes, defect_classes)
    return Decision("FAIL" if defects else "PASS", score, threshold, sorted(defects, key=lambda b: -b.conf))


def expected_cost(y_fail: np.ndarray, scores: np.ndarray, threshold: float,
                  cost_ratio: float = DEFAULT_COST_RATIO) -> float:
    """Mean cost per frame at a threshold.

    Cost is ``cost_ratio`` for each false PASS (defective frame scored below
    the threshold) and 1 for each false FAIL (good frame at or above it).
    """
    y = np.asarray(y_fail, dtype=bool)
    pred = np.asarray(scores) >= threshold
    false_pass = np.sum(y & ~pred)
    false_fail = np.sum(~y & pred)
    return float((cost_ratio * false_pass + false_fail) / len(y))


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive distinct scores, plus both extremes.

    Every distinct way of splitting the frames is represented exactly once, so
    searching these candidates finds the true minimum of the cost on the data.
    """
    s = np.unique(np.clip(np.asarray(scores, dtype=float), 0.0, 1.0))
    mids = (s[:-1] + s[1:]) / 2 if len(s) > 1 else np.array([])
    return np.concatenate([[0.0], mids, [np.nextafter(s[-1], 2.0) if len(s) else 1.0]]).clip(0.0, 1.0)


def choose_threshold(y_fail: np.ndarray, scores: np.ndarray,
                     cost_ratio: float = DEFAULT_COST_RATIO) -> tuple[float, float]:
    """Threshold that minimises expected cost on labelled data.

    Ties are broken towards the lower threshold, which fails more frames,
    because a false PASS is the more expensive mistake.

    Returns:
        ``(threshold, cost)``.
    """
    y = np.asarray(y_fail, dtype=bool)
    if y.size == 0:
        raise ValueError("cannot choose a threshold from zero frames")
    cands = candidate_thresholds(scores)
    costs = np.array([expected_cost(y, scores, t, cost_ratio) for t in cands])
    i = int(np.flatnonzero(costs == costs.min())[0])
    return float(cands[i]), float(costs[i])
