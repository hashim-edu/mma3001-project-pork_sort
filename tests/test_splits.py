"""Tests for the blocked split: no overlap, the buffer gap holds, one copy for evaluation."""

import numpy as np
import pytest

from porksort.dataset import build_index
from porksort.splits import (
    BlockedSplitConfig,
    apply_split,
    blocked_split,
    leakage_report,
    load_split,
    roboflow_split,
)

CFG = BlockedSplitConfig(block_size=100, gap=25, val_frac=0.15, test_frac=0.15, n_trials=50, seed=1)


def test_blocked_split_properties(tiny_dataset):
    idx = build_index(tiny_dataset)
    a = blocked_split(idx, CFG)
    assert a["source"].is_unique
    assert set(a["split"]) <= {"train", "val", "test", "buffer"}
    kept = a[a["split"] != "buffer"]
    # whole blocks go to one split (ignoring buffered frames)
    assert (kept.groupby("block")["split"].nunique() == 1).all()
    # no train frame within `gap` of a test frame, none of val within gap of test
    df = apply_split(idx, a)
    assert leakage_report(df, gaps=(CFG.gap,))["within"][CFG.gap] == 0
    te = df.loc[df.split == "test", "frame"].to_numpy()
    va = df.loc[df.split == "val", "frame"].to_numpy()
    if len(te) and len(va):
        assert np.abs(te[:, None] - va[None, :]).min() > CFG.gap


def test_apply_split_keeps_one_copy_for_eval(tiny_dataset):
    idx = build_index(tiny_dataset)
    df = apply_split(idx, blocked_split(idx, CFG))
    ev = df[df["split"].isin(["val", "test"])]
    assert ev["source"].is_unique
    assert (ev["copy"] == 0).all()
    tr = df[df["split"] == "train"]
    assert tr.groupby("source").size().max() == 3


def test_blocked_split_is_deterministic(tiny_dataset):
    idx = build_index(tiny_dataset)
    a, b = blocked_split(idx, CFG), blocked_split(idx, CFG)
    assert (a["split"] == b["split"]).all()


def test_bad_fractions(tiny_dataset):
    idx = build_index(tiny_dataset)
    with pytest.raises(ValueError):
        blocked_split(idx, BlockedSplitConfig(val_frac=0.5, test_frac=0.45))


def test_roboflow_split_and_load_split(tiny_dataset, tmp_path):
    idx = build_index(tiny_dataset)
    rf = roboflow_split(idx)
    assert set(rf["split"]) == {"train", "val", "test"}
    with pytest.raises(ValueError, match="unknown split"):
        load_split(idx, "random")
    csv = tmp_path / "split.csv"
    blocked_split(idx, CFG).to_csv(csv, index=False)
    assert "split" in load_split(idx, "blocked", csv_path=csv)


def test_leakage_report_counts():
    import pandas as pd

    df = pd.DataFrame({"split": ["train", "train", "test", "test"], "frame": [0, 100, 1, 50]})
    r = leakage_report(df, gaps=(1, 10, 50))
    assert r["within"] == {1: 1, 10: 1, 50: 2}
