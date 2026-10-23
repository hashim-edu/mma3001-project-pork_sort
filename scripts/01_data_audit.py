"""Data audit: class balance, PASS/FAIL balance, leakage, and the blocked split.

Run from the repo root::

    python scripts/01_data_audit.py

Writes:
    results/data_audit.json            counts, box sizes, leakage, split sizes
    results/figures/audit_*.png        figures used in the report
    splits/blocked_split.csv           the source -> split assignment (committed)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from porksort.config import CLASS_NAMES, REPO_ROOT, RESULTS_DIR  # noqa: E402
from porksort.dataset import build_index, parse_label_file, unique_sources  # noqa: E402
from porksort.splits import (  # noqa: E402
    BlockedSplitConfig,
    apply_split,
    blocked_split,
    leakage_report,
    roboflow_split,
)

FIG = RESULTS_DIR / "figures"


def thumb(path: str, size=(48, 36)) -> np.ndarray:
    """Z-normalised grayscale thumbnail used to measure frame similarity."""
    a = np.asarray(Image.open(path).convert("L").resize(size, Image.BILINEAR), dtype=np.float64).ravel()
    return (a - a.mean()) / (a.std() + 1e-12)


def similarity_vs_gap(src, max_gap=120, n_random=2000, seed=0):
    """Mean thumbnail correlation of frame pairs as a function of frame gap.

    Same-tray frames look alike, so correlation stays high while the tray is
    in view and falls to the random-pair level once it has left. The gap at
    which the curve meets the random level is how far apart frames must be
    before they stop sharing a tray.
    """
    src = src.dropna(subset=["frame"]).sort_values("frame")
    frames = src["frame"].astype(int).to_numpy()
    T = np.stack([thumb(p) for p in src["image"]])
    sums, counts = np.zeros(max_gap + 1), np.zeros(max_gap + 1)
    for i in range(len(frames)):
        j = i + 1
        while j < len(frames) and frames[j] - frames[i] <= max_gap:
            d = frames[j] - frames[i]
            sums[d] += float(T[i] @ T[j]) / T.shape[1]
            counts[d] += 1
            j += 1
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(frames), (2, n_random))
    far = np.abs(frames[a] - frames[b]) > 1000
    rand = np.array([float(T[i] @ T[j]) / T.shape[1] for i, j in zip(a[far], b[far])])
    with np.errstate(invalid="ignore"):
        mean = sums / counts
    return mean, counts, float(rand.mean()), float(rand.std())


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    idx = build_index()
    src = unique_sources(idx)
    out: dict = {"n_images": int(len(idx)), "n_sources": int(len(src))}

    # Per-split counts (as exported)
    per_split = {}
    for s, g in idx.groupby("rf_split"):
        per_split[s] = {
            "images": int(len(g)),
            "sources": int(g["source"].nunique()),
            "pass_images": int((~g["fail"]).sum()),
            "fail_images": int(g["fail"].sum()),
            "images_with_class": {c: int((g[f"n_{c}"] > 0).sum()) for c in CLASS_NAMES},
            "boxes_per_class": {c: int(g[f"n_{c}"].sum()) for c in CLASS_NAMES},
        }
    out["rf_splits"] = per_split
    out["copies_per_source"] = {int(k): int(v) for k, v in idx.groupby("source").size().value_counts().items()}
    out["sources_in_multiple_rf_splits"] = int((idx.groupby("source")["rf_split"].nunique() > 1).sum())

    # Box sizes per class (fraction of image area), from source frames only
    areas = {c: [] for c in CLASS_NAMES}
    for lbl in src["label"]:
        for b in parse_label_file(lbl):
            areas[b.cls].append(b.area)
    out["box_area_frac"] = {
        c: {"n": len(v), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90))} for c, v in areas.items() if v
    }

    # Leakage of the Roboflow split
    rf = roboflow_split(idx)
    out["leakage_roboflow"] = leakage_report(rf)

    # Similarity vs frame gap, to justify the buffer size
    mean, counts, r_mu, r_sd = similarity_vs_gap(src)
    ok = counts >= 5
    gaps = np.arange(len(mean))
    # Dominant period of the similarity curve = spacing between trays on the
    # conveyor, in frame numbers. Found from the FFT of the detrended curve.
    g, m = gaps[ok][1:], mean[ok][1:]
    spec = np.abs(np.fft.rfft(m - m.mean()))
    freqs = np.fft.rfftfreq(len(m), d=float(np.median(np.diff(g))))
    k = int(np.argmax(spec[1:]) + 1)
    period = float(1.0 / freqs[k])
    out["similarity_random_pairs"] = {"mean": r_mu, "std": r_sd}
    out["tray_pitch_frames_fft"] = period
    out["similarity_by_gap"] = {int(a): float(b) for a, b in zip(g, m)}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(g, m, ".-", lw=1, label="mean correlation of pairs at this gap")
    ax.axhline(r_mu, color="0.5", lw=1, label="mean of random far-apart pairs")
    ax.axvline(30, color="C3", ls="--", lw=1, label="chosen buffer gap = 30")
    ax.set_xlabel("frame-number gap between two frames")
    ax.set_ylabel("thumbnail correlation")
    ax.set_title(f"Similarity repeats every ~{period:.1f} frames (tray pitch)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "audit_similarity_vs_gap.png", dpi=150)
    plt.close(fig)

    # Blocked split
    cfg = BlockedSplitConfig()
    assign = blocked_split(idx, cfg)
    (REPO_ROOT / "splits").mkdir(exist_ok=True)
    assign.to_csv(REPO_ROOT / "splits" / "blocked_split.csv", index=False)
    bl = apply_split(idx, assign)
    out["blocked_config"] = cfg.__dict__
    out["blocked_sources_buffered"] = int((assign["split"] == "buffer").sum())
    out["blocked_splits"] = {
        s: {"images": int(len(g)), "sources": int(g["source"].nunique()),
            "pass_images": int((~g["fail"]).sum()), "fail_images": int(g["fail"].sum()),
            "images_with_class": {c: int((g[f"n_{c}"] > 0).sum()) for c in CLASS_NAMES}}
        for s, g in bl.groupby("split")
    }
    out["leakage_blocked"] = leakage_report(bl)

    # Figure: distance from each test frame to the nearest training frame
    def nearest(df):
        tr = np.sort(df.loc[df.split == "train", "frame"].dropna().unique())
        te = df.loc[df.split == "test", "frame"].dropna().unique()
        pos = np.searchsorted(tr, te)
        return np.minimum(np.abs(te - tr[np.clip(pos - 1, 0, len(tr) - 1)]),
                          np.abs(te - tr[np.clip(pos, 0, len(tr) - 1)]))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.arange(0, 120, 4)
    ax.hist(np.clip(nearest(rf), 0, 118), bins=bins, alpha=0.7, label="Roboflow test split")
    ax.hist(np.clip(nearest(bl), 0, 118), bins=bins, alpha=0.7, label="blocked test split")
    ax.set_xlabel("frame gap to nearest training frame (clipped at 118)")
    ax.set_ylabel("test frames")
    ax.set_title("Test-to-train proximity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "audit_test_train_gap.png", dpi=150)
    plt.close(fig)

    # Figure: class presence per split
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    labels = ["PASS"] + list(CLASS_NAMES)
    width = 0.27
    for k, s in enumerate(["train", "valid", "test"]):
        d = per_split[s]
        vals = [d["pass_images"] / d["images"]] + [d["images_with_class"][c] / d["images"] for c in CLASS_NAMES]
        ax.bar(np.arange(len(labels)) + (k - 1) * width, vals, width, label=f"{s} (n={d['images']})")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, fontsize=8)
    ax.set_ylabel("fraction of images")
    ax.set_title("Frames containing each label (Roboflow split)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "audit_class_presence.png", dpi=150)
    plt.close(fig)

    (RESULTS_DIR / "data_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("leakage_roboflow", "leakage_blocked", "blocked_splits",
                                          "tray_pitch_frames_fft")}, indent=1))


if __name__ == "__main__":
    main()
