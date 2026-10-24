"""Train/validation/test splits.

Two splits are supported:

``roboflow``
    The split shipped with the export. Its test frames are interleaved with
    training frames from the same conveyor video, so a test tray is often the
    same physical tray as a training tray a few frames earlier or later. Scores
    on it are expected to be optimistic.

``blocked``
    Frames are grouped into contiguous blocks of ``block_size`` frame numbers,
    and whole blocks are assigned to one split. Any frame within ``gap`` frame
    numbers of a frame in a higher-priority split (test > val > train) is
    dropped into a ``buffer`` set that is not used. A tray therefore cannot
    appear in both training and test data unless it stays in view for more
    than ``gap`` frame numbers.

Block assignment is random but chosen from ``n_trials`` seeds to keep the
PASS rate and the common defect rates of val and test close to the whole
dataset. The choice uses labels only, never model results, so it does not
leak test performance into the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import unique_sources

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


@dataclass(frozen=True)
class BlockedSplitConfig:
    """Settings for `blocked_split`.

    Attributes:
        block_size: Width of a block in frame numbers.
        gap: Frames closer than this to a higher-priority split are buffered.
        val_frac: Target fraction of source frames in validation.
        test_frac: Target fraction of source frames in test.
        n_trials: Number of random block assignments to compare.
        seed: Seed for the random number generator.
    """

    block_size: int = 200
    gap: int = 30
    val_frac: float = 0.15
    test_frac: float = 0.15
    n_trials: int = 500
    seed: int = 0


def roboflow_split(index: pd.DataFrame) -> pd.DataFrame:
    """Use the split shipped with the Roboflow export.

    Args:
        index: Output of `porksort.dataset.build_index`.

    Returns:
        A copy of ``index`` with a ``split`` column (``train``, ``val`` or ``test``).
    """
    out = index.copy()
    out["split"] = out["rf_split"].map({"train": "train", "valid": "val", "test": "test"})
    return out


def _balance_score(src: pd.DataFrame, assign: pd.Series) -> float:
    """How far val and test label rates are from the whole dataset (lower is better)."""
    cols = ["fail", "has_packaging-error", "has_unsealed", "has_rare"]
    overall = src[cols].mean()
    score = 0.0
    for part in ("val", "test"):
        sub = src[assign == part]
        if len(sub) == 0:
            return np.inf
        score += float((sub[cols].mean() - overall).abs().sum())
        # every rare class should appear at least once in val and in test
        if sub["has_rare"].sum() < 2:
            score += 1.0
    return score


def blocked_split(index: pd.DataFrame, cfg: BlockedSplitConfig = BlockedSplitConfig()) -> pd.DataFrame:
    """Assign each source frame to train, val, test or buffer.

    Args:
        index: Output of `porksort.dataset.build_index`.
        cfg: Block size, gap and target fractions.

    Returns:
        DataFrame with one row per source and columns ``source``, ``frame``,
        ``block`` and ``split`` (``train``/``val``/``test``/``buffer``).

    Raises:
        ValueError: If the fractions do not leave room for a training set.
    """
    if cfg.val_frac + cfg.test_frac >= 0.9:
        raise ValueError("val_frac + test_frac must be below 0.9")
    src = unique_sources(index)[["source", "frame", "fail", "classes"]].copy()
    # Sources without a frame number get their own block far from the others.
    missing = src["frame"].isna()
    src.loc[missing, "frame"] = 10**9 + np.arange(missing.sum()) * (cfg.block_size + cfg.gap + 1)
    src["frame"] = src["frame"].astype(np.int64)
    src["block"] = src["frame"] // cfg.block_size
    src["has_packaging-error"] = src["classes"].str.contains("packaging-error")
    src["has_unsealed"] = src["classes"].str.contains("unsealed")
    src["has_rare"] = src["classes"].str.contains("loose-meat|twisted-meat|wrinkle")

    blocks = np.sort(src["block"].unique())
    sizes = src.groupby("block").size().reindex(blocks).to_numpy()
    n = len(src)
    rng = np.random.default_rng(cfg.seed)
    best, best_score = None, np.inf
    for _ in range(cfg.n_trials):
        order = rng.permutation(len(blocks))
        cum = np.cumsum(sizes[order])
        part = np.where(
            cum <= cfg.test_frac * n,
            "test",
            np.where(cum <= (cfg.test_frac + cfg.val_frac) * n, "val", "train"),
        )
        block_to_part = dict(zip(blocks[order], part, strict=True))
        assign = src["block"].map(block_to_part)
        s = _balance_score(src, assign)
        if s < best_score:
            best, best_score = assign, s
    src["split"] = best

    # Buffer: drop frames that sit within `gap` of a higher-priority split.
    frames = src["frame"].to_numpy()
    prio = src["split"].map(SPLIT_PRIORITY).to_numpy()
    order = np.argsort(frames)
    keep = np.ones(n, dtype=bool)
    sorted_frames = frames[order]
    for i in range(n):
        lo = np.searchsorted(sorted_frames, frames[i] - cfg.gap, side="left")
        hi = np.searchsorted(sorted_frames, frames[i] + cfg.gap, side="right")
        neighbours = order[lo:hi]
        if np.any(prio[neighbours] < prio[i]):
            keep[i] = False
    src.loc[~keep, "split"] = "buffer"
    return src[["source", "frame", "block", "split"]].reset_index(drop=True)


def apply_split(index: pd.DataFrame, assignment: pd.DataFrame, eval_one_copy: bool = True) -> pd.DataFrame:
    """Attach a source-level assignment to the per-image index.

    Args:
        index: Output of `porksort.dataset.build_index`.
        assignment: Output of `blocked_split`.
        eval_one_copy: Keep only the first copy of each source in val and test,
            so every evaluation frame is counted once. Training keeps all
            augmented copies.

    Returns:
        ``index`` with ``split`` and ``block`` columns, buffer rows removed.
    """
    out = index.merge(assignment[["source", "split", "block"]], on="source", how="left")
    out = out[out["split"] != "buffer"]
    if eval_one_copy:
        out = out[(out["split"] == "train") | (out["copy"] == 0)]
    return out.reset_index(drop=True)


def load_split(index: pd.DataFrame, name: str, csv_path=None) -> pd.DataFrame:
    """Return ``index`` with a ``split`` column for a named split.

    Args:
        index: Output of `porksort.dataset.build_index`.
        name: ``"roboflow"`` or ``"blocked"``.
        csv_path: Assignment CSV for the blocked split. Defaults to the
            committed ``splits/blocked_split.csv``, so every run and every
            machine (including Colab) uses the same frames.

    Returns:
        Per-image index with ``split`` and ``block`` columns. For the blocked
        split, buffer frames are removed and validation and test keep one copy
        of each frame (see `apply_split`).

    Raises:
        ValueError: For an unknown split name.
        FileNotFoundError: If the blocked split CSV is missing.
    """
    if name == "roboflow":
        out = roboflow_split(index)
        out["block"] = out["frame"] // BlockedSplitConfig().block_size
        return out
    if name == "blocked":
        from .config import REPO_ROOT

        path = Path(csv_path) if csv_path else REPO_ROOT / "splits" / "blocked_split.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/01_data_audit.py")
        return apply_split(index, pd.read_csv(path))
    raise ValueError(f"unknown split {name!r}; use 'roboflow' or 'blocked'")


def export_yolo_dataset(df: pd.DataFrame, out_dir, class_names, link: bool = True) -> Path:
    """Write a YOLO-format folder tree for a split so Ultralytics can train on it.

    Args:
        df: Per-image index with a ``split`` column (``train``/``val``/``test``).
        out_dir: Destination folder. Created if missing.
        class_names: Class names in id order, written to ``data.yaml``.
        link: Symlink images and labels instead of copying (saves disk).

    Returns:
        Path to the written ``data.yaml``.
    """
    import shutil

    out = Path(out_dir)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)
    for row in df.itertuples():
        for kind, src in (("images", row.image), ("labels", row.label)):
            dst = out / row.split / kind / Path(src).name
            if dst.exists() or dst.is_symlink():
                continue
            if link:
                dst.symlink_to(Path(src).resolve())
            else:
                shutil.copy2(src, dst)
    yaml = out / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml.write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n{names}\n"
    )
    return yaml


def leakage_report(df: pd.DataFrame, gaps=(1, 5, 10, 30)) -> dict:
    """Count test frames that have a training frame within each gap.

    Args:
        df: Per-image index with ``split`` and ``frame`` columns.
        gaps: Frame-number distances to report.

    Returns:
        ``{"n_test": int, "within": {gap: count}}``.
    """
    train = np.sort(df.loc[df["split"] == "train", "frame"].dropna().unique())
    test = df.loc[df["split"] == "test", "frame"].dropna().unique()
    if len(train) == 0 or len(test) == 0:
        return {"n_test": int(len(test)), "within": {g: 0 for g in gaps}}
    pos = np.searchsorted(train, test)
    left = train[np.clip(pos - 1, 0, len(train) - 1)]
    right = train[np.clip(pos, 0, len(train) - 1)]
    dist = np.minimum(np.abs(test - left), np.abs(test - right))
    return {"n_test": int(len(test)), "within": {int(g): int((dist <= g).sum()) for g in gaps}}
