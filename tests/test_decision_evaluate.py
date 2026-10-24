"""Tests for the pass/fail rule, threshold search and metrics."""

import math

import numpy as np
import pytest

from porksort.dataset import Box
from porksort.decision import candidate_thresholds, choose_threshold, decide, expected_cost, frame_score
from porksort.evaluate import confusion, passfail_metrics, wilson_ci


def b(cls, conf):
    return Box(cls, 0.5, 0.5, 0.1, 0.1, conf)


def test_decide_rule():
    assert decide([], 0.3).status == "PASS"
    assert decide([b("unsealed", 0.29)], 0.3).status == "PASS"
    d = decide([b("unsealed", 0.30), b("wrinkle", 0.9), b("loose-meat", 0.1)], 0.3)
    assert d.status == "FAIL"
    assert [x.cls for x in d.defects] == ["wrinkle", "unsealed"]  # sorted by confidence
    assert d.score == pytest.approx(0.9)


def test_decide_respects_defect_set_and_range():
    assert decide([b("wrinkle", 0.9)], 0.5, defect_classes=frozenset({"unsealed"})).status == "PASS"
    assert frame_score([b("wrinkle", 0.9)], frozenset({"unsealed"})) == 0.0
    with pytest.raises(ValueError):
        decide([], 1.5)


def test_choose_threshold_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    t, c = choose_threshold(y, s, cost_ratio=10)
    assert c == 0.0
    assert 0.3 < t <= 0.6


def test_choose_threshold_matches_dense_search():
    """Manufactured check: the candidate search equals brute force on a fine grid."""
    rng = np.random.default_rng(3)
    y = rng.random(300) < 0.8
    s = np.clip(rng.normal(0.3 + 0.3 * y, 0.2), 0, 1)
    for ratio in (1, 5, 20):
        _, c = choose_threshold(y, s, ratio)
        grid = np.linspace(0, 1, 20001)
        c_grid = min(expected_cost(y, s, t, ratio) for t in grid)
        assert c <= c_grid + 1e-12


def test_higher_cost_ratio_never_raises_threshold():
    rng = np.random.default_rng(4)
    y = rng.random(400) < 0.8
    s = np.clip(rng.normal(0.4 + 0.2 * y, 0.2), 0, 1)
    ts = [choose_threshold(y, s, r)[0] for r in (1, 3, 10, 30, 100)]
    assert all(a >= b for a, b in zip(ts[:-1], ts[1:], strict=True))


def test_candidates_cover_extremes():
    c = candidate_thresholds(np.array([0.2, 0.2, 0.6]))
    assert c[0] == 0.0 and c[-1] > 0.6 and 0.2 < c[1] < 0.6


def test_wilson_known_values():
    lo, hi = wilson_ci(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    # symmetry: CI for k/n mirrors CI for (n-k)/n
    a, b2 = wilson_ci(3, 17), wilson_ci(14, 17)
    assert a[0] == pytest.approx(1 - b2[1]) and a[1] == pytest.approx(1 - b2[0])
    with pytest.raises(ValueError):
        wilson_ci(5, 3)


def test_confusion_and_metrics():
    y = np.array([1, 1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.7, 0.1])
    assert confusion(y, s >= 0.5) == {"tp": 2, "fn": 1, "fp": 1, "tn": 1}
    m = passfail_metrics(y, s, 0.5, cost_ratio=10)
    assert m["escape_rate"]["value"] == pytest.approx(1 / 3)
    assert m["escape_rate"]["value"] + m["fail_recall"]["value"] == pytest.approx(1.0)
    assert m["false_reject_rate"]["value"] == pytest.approx(0.5)
    assert m["expected_cost"] == pytest.approx((10 * 1 + 1) / 5)
    assert not math.isnan(m["roc_auc"])
