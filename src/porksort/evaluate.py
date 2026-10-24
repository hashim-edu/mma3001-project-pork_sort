"""Metrics for the pass/fail decision.

FAIL is the positive class throughout, because catching defects is the job.
The two errors that matter on a packing line are:

- **escape rate** (false PASS rate) = FN / (TP + FN): the share of defective
  frames the system lets through. This is the food-safety and recall risk.
- **false reject rate** = FP / (FP + TN): the share of good frames sent for
  re-inspection. This is the throughput and labour cost.

Accuracy is reported but not used to choose models, because 80 to 85 % of
frames are defective and a model that always says FAIL already scores that.

Proportions estimated from a few dozen PASS frames are uncertain, so every
rate is given with a Wilson score 95 % interval. The Wilson interval is used
instead of the usual ``p ± 1.96 sqrt(p(1-p)/n)`` because that formula
collapses to zero width when p = 0 or 1 and can leave [0, 1] for small n.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .decision import expected_cost


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Standard normal quantile (1.96 for 95 %).

    Returns:
        ``(lower, upper)``; ``(0.0, 1.0)`` when ``n == 0``.

    Raises:
        ValueError: If ``k`` is outside [0, n].

    Examples:
        >>> lo, hi = wilson_ci(0, 10)
        >>> round(lo, 3), round(hi, 3)
        (0.0, 0.278)
    """
    if n == 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    denom = 1 + z * z / n
    # The centre is pulled towards 0.5 and the half-width stays positive at p = 0 or 1,
    # which is what the normal approximation gets wrong.
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def confusion(y_fail, pred_fail) -> dict[str, int]:
    """Confusion counts with FAIL as the positive class.

    Args:
        y_fail: True labels, True = defective frame.
        pred_fail: Predicted labels, True = FAIL.

    Returns:
        Counts ``tp``, ``fn``, ``fp`` and ``tn``.

    Raises:
        ValueError: If the two arrays have different shapes.
    """
    y = np.asarray(y_fail, dtype=bool)
    p = np.asarray(pred_fail, dtype=bool)
    if y.shape != p.shape:
        raise ValueError("y_fail and pred_fail must have the same shape")
    return {
        "tp": int(np.sum(y & p)),
        "fn": int(np.sum(y & ~p)),
        "fp": int(np.sum(~y & p)),
        "tn": int(np.sum(~y & ~p)),
    }


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"value": (k / n) if n else float("nan"), "ci95": [lo, hi], "k": k, "n": n}


def passfail_metrics(y_fail, scores, threshold: float, cost_ratio: float) -> dict:
    """All pass/fail metrics at one threshold.

    Args:
        y_fail: True labels, True = defective frame.
        scores: Defect scores in [0, 1].
        threshold: Frames with score >= threshold are predicted FAIL.
        cost_ratio: Cost of a false PASS relative to a false FAIL.

    Returns:
        Dict with the confusion counts, escape rate, false reject rate, FAIL
        recall, PASS recall, FAIL precision, accuracy, balanced accuracy,
        expected cost, ROC AUC and average precision (threshold-free).
    """
    y = np.asarray(y_fail, dtype=bool)
    s = np.asarray(scores, dtype=float)
    c = confusion(y, s >= threshold)
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    out = {
        "n": int(len(y)),
        "n_fail": int(y.sum()),
        "n_pass": int((~y).sum()),
        "threshold": float(threshold),
        "confusion": c,
        "escape_rate": _rate(fn, tp + fn),
        "false_reject_rate": _rate(fp, fp + tn),
        "fail_recall": _rate(tp, tp + fn),
        "pass_recall": _rate(tn, fp + tn),
        "fail_precision": _rate(tp, tp + fp),
        "accuracy": _rate(tp + tn, len(y)),
        "expected_cost": expected_cost(y, s, threshold, cost_ratio),
        "cost_ratio": float(cost_ratio),
    }
    out["balanced_accuracy"] = 0.5 * (out["fail_recall"]["value"] + out["pass_recall"]["value"])
    # ROC AUC is undefined when only one class is present.
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["average_precision"] = float(average_precision_score(y, s))
    else:
        out["roc_auc"] = out["average_precision"] = float("nan")
    return out


def bootstrap_auc(y_fail, scores, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95 % interval for ROC AUC.

    Frames are resampled with replacement ``n_boot`` times and the AUC is
    recomputed on each resample; resamples that contain only one class are
    skipped.

    Args:
        y_fail: True labels, True = defective frame.
        scores: Defect scores, one per frame.
        n_boot: Number of resamples.
        seed: Seed for the random number generator, so the interval repeats.

    Returns:
        ``(lower, upper)``: the 2.5th and 97.5th percentiles of the resampled
        AUCs.
    """
    y = np.asarray(y_fail, dtype=bool)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        # Resample frames with replacement.
        i = rng.integers(0, len(y), len(y))
        # Skip resamples that contain a single class (AUC undefined).
        if 0 < y[i].sum() < len(i):
            aucs.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
