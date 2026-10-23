"""Report figures that are not produced by the analysis scripts.

    python scripts/06_figures.py

- ``results/figures/pipeline.png``: system architecture (training and inspection).
- ``results/figures/examples_blocked_test.png``: blocked-test frames with ground
  truth and YOLO detections, including the worst escapes and false rejects
  (needs the YOLO results).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from porksort.config import RESULTS_DIR  # noqa: E402

FIG = RESULTS_DIR / "figures"


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec="0.3", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1, color="0.3"))


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    data, train, infer, out = "#e8eef7", "#fdf1dc", "#e6f4ea", "#f7e3e3"
    ax.text(0.1, 4.4, "Development (once)", fontsize=9, weight="bold")
    box(ax, 0.1, 3.4, 1.6, 0.8, "Roboflow v4 export\n3,549 frames, 5 classes\nYOLO boxes", data)
    box(ax, 2.0, 3.4, 1.7, 0.8, "Index + PASS/FAIL label\n(FAIL = any defect box)\ngroup augmented copies", data)
    box(ax, 4.0, 3.4, 1.8, 0.8, "Blocked split\n200-frame blocks\n30-frame buffer", data)
    box(ax, 6.1, 3.85, 1.8, 0.55, "YOLOv8n fine-tune\n(Colab T4) -> ONNX", train)
    box(ax, 6.1, 3.2, 1.8, 0.55, "Features + LR / tree /\nSVM / MLP (grouped CV)", train)
    box(ax, 8.2, 3.4, 1.7, 0.8, "Threshold on val\nmin expected cost\n(false PASS = 10)", train)
    for a, b in ((1.7, 2.0), (3.7, 4.0), (7.9, 8.2)):
        arrow(ax, a, 3.8, b, 3.8)
    arrow(ax, 5.8, 3.8, 6.1, 4.1)
    arrow(ax, 5.8, 3.8, 6.1, 3.5)

    ax.text(0.1, 2.75, "Inspection (every frame)", fontsize=9, weight="bold")
    box(ax, 0.1, 1.5, 1.6, 0.9, "Camera frame\nRGB 8-bit\n720 x 540 px", data)
    box(ax, 2.0, 1.5, 1.7, 0.9, "Input checks\ntype, decode, size,\nexposure, blank", data)
    box(ax, 4.0, 1.95, 1.8, 0.6, "Letterbox 640 x 640\n/255 normalise", infer)
    box(ax, 6.1, 1.95, 1.8, 0.6, "YOLOv8n (ONNX Runtime)\ndecode + NMS -> boxes", infer)
    box(ax, 4.0, 1.15, 1.8, 0.6, "Resize 240 x 180\nhandcrafted features", infer)
    box(ax, 6.1, 1.15, 1.8, 0.6, "Classifier\n-> P(defect)", infer)
    box(ax, 8.2, 1.5, 1.7, 0.9, "Defect score s\nFAIL if s >= threshold\nelse PASS", out)
    box(ax, 8.2, 0.15, 1.7, 0.95, "Report\nJSON + annotated PNG\nbatch index.html", out)
    box(ax, 2.0, 0.15, 1.7, 0.95, "ERROR report\n(unreadable or\nunsupported input)", out)
    arrow(ax, 1.7, 1.95, 2.0, 1.95)
    arrow(ax, 3.7, 1.95, 4.0, 2.25)
    arrow(ax, 3.7, 1.95, 4.0, 1.45)
    arrow(ax, 5.8, 2.25, 6.1, 2.25)
    arrow(ax, 5.8, 1.45, 6.1, 1.45)
    arrow(ax, 7.9, 2.25, 8.2, 2.05)
    arrow(ax, 7.9, 1.45, 8.2, 1.85)
    arrow(ax, 9.05, 1.5, 9.05, 1.1)
    arrow(ax, 2.85, 1.5, 2.85, 1.1)
    arrow(ax, 9.05, 3.4, 9.05, 2.4)
    ax.text(7.05, 0.75, "main model: YOLOv8n\nbaselines: classical path", ha="center", fontsize=7, color="0.35")
    fig.tight_layout()
    fig.savefig(FIG / "pipeline.png", dpi=180)
    plt.close(fig)


def examples():
    """Grid of blocked-test frames: GT (dashed) vs YOLO (solid), sorted by error type."""
    run = RESULTS_DIR / "yolo" / "yolov8n_blocked"
    if not (run / "predictions.json").exists():
        print("examples skipped: no YOLO predictions")
        return
    import numpy as np
    from PIL import Image, ImageDraw

    from porksort.dataset import Box, build_index, parse_label_file
    from porksort.decision import frame_score
    from porksort.report import COLOURS
    from porksort.splits import load_split

    preds = json.loads((run / "predictions.json").read_text())
    thr = json.loads((run / "threshold.json").read_text())["threshold"]
    df = load_split(build_index(), "blocked")
    te = df[df.split == "test"].copy()
    te["score"] = [frame_score([Box(**b) for b in preds[Path(p).name]]) for p in te["image"]]
    te["pred_fail"] = te["score"] >= thr
    groups = {
        "correct FAIL": te[te.fail & te.pred_fail].sort_values("score", ascending=False).head(3),
        "correct PASS": te[~te.fail & ~te.pred_fail].sort_values("score").head(3),
        "escape (defect passed)": te[te.fail & ~te.pred_fail].sort_values("score").head(3),
        "false reject (good failed)": te[~te.fail & te.pred_fail].sort_values("score", ascending=False).head(3),
    }
    rows = [(g, r) for g, d in groups.items() for r in d.itertuples()]
    if not rows:
        return
    ncol = 3
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 2.7))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (g, r) in zip(np.ravel(axs), rows):
        im = Image.open(r.image).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in parse_label_file(r.label):
            x0, y0, x1, y1 = b.to_pixels(*im.size)
            for k in range(0, int(x1 - x0), 12):  # dashed ground truth
                d.line([x0 + k, y0, min(x0 + k + 6, x1), y0], fill="white", width=3)
                d.line([x0 + k, y1, min(x0 + k + 6, x1), y1], fill="white", width=3)
            for k in range(0, int(y1 - y0), 12):
                d.line([x0, y0 + k, x0, min(y0 + k + 6, y1)], fill="white", width=3)
                d.line([x1, y0 + k, x1, min(y0 + k + 6, y1)], fill="white", width=3)
        for b in (Box(**x) for x in preds[Path(r.image).name]):
            if b.conf >= thr:
                d.rectangle(b.to_pixels(*im.size), outline=COLOURS[b.cls], width=4)
        ax.imshow(im)
        gt = r.classes.replace("|", ", ") or "no defect"
        ax.set_title(f"{g}\nGT: {gt}; score {r.score:.2f}", fontsize=7)
    fig.suptitle(f"Blocked test frames: dashed white = ground truth, solid = YOLO boxes >= threshold {thr:.2f}",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "examples_blocked_test.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    pipeline()
    examples()
    print("figures written to", FIG)
