"""Report figures that are not produced by the analysis scripts.

    python scripts/06_figures.py

- ``results/figures/pipeline.png``: system architecture (training and inspection).
- ``results/figures/examples_blocked_test.jpg``: blocked-test frames with ground
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
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="0.3", lw=1)
    )
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
    box(
        ax,
        2.0,
        3.4,
        1.7,
        0.8,
        "Index + PASS/FAIL label\n(FAIL = any defect box)\ngroup augmented copies",
        data,
    )
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
    ax.text(
        7.05, 0.75, "main model: YOLOv8n\nbaselines: classical path", ha="center", fontsize=7, color="0.35"
    )
    fig.tight_layout()
    fig.savefig(FIG / "pipeline.png", dpi=180)
    plt.close(fig)


def examples():
    """Grid of blocked-test frames: GT (dashed) vs YOLO (solid), sorted by error type."""
    run = RESULTS_DIR / "yolo" / "yolov8n_blocked"
    if not (run / "predictions_best_fp32_480.json").exists():
        print("examples skipped: no YOLO predictions")
        return
    import numpy as np
    from PIL import Image, ImageDraw

    from porksort.dataset import Box, build_index, parse_label_file
    from porksort.decision import frame_score
    from porksort.report import COLOURS
    from porksort.splits import load_split

    # Selected model (ONNX, 480 px input) at the cost ratio 1 operating point: at r = 10 its
    # validation threshold fails every frame, which would show no PASS decisions at all.
    preds = json.loads((run / "predictions_best_fp32_480.json").read_text())
    thr = json.loads((run / "threshold.json").read_text())["best_fp32_480.onnx"]["1"]
    df = load_split(build_index(), "blocked")
    te = df[df.split == "test"].copy()
    te["score"] = [frame_score([Box(**b) for b in preds[Path(p).name]]) for p in te["image"]]
    te["pred_fail"] = te["score"] >= thr
    groups = {
        "correct FAIL": te[te.fail & te.pred_fail].sort_values("score", ascending=False).head(1),
        "correct PASS": te[~te.fail & ~te.pred_fail].sort_values("score").head(1),
        "escape (defect passed)": te[te.fail & ~te.pred_fail].sort_values("score").head(2),
        "false reject (good failed)": te[~te.fail & te.pred_fail]
        .sort_values("score", ascending=False)
        .head(2),
    }
    rows = [(g, r) for g, d in groups.items() for r in d.itertuples()]
    if not rows:
        return
    ncol = 3
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 2.7))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (g, r) in zip(np.ravel(axs), rows, strict=False):
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
    fig.suptitle(
        f"Blocked test frames, YOLOv8n at 480 px, threshold {thr:.2f} (cost ratio 1): "
        "dashed white = ground truth, solid = detections at or above threshold",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIG / "examples_blocked_test.jpg", dpi=120, pil_kwargs={"quality": 82})
    plt.close(fig)


# M4 performance-core peak from the unit's formula: cores x clock x fp32 lanes x FMA x FP pipes.
# The 4 pipes per core is an estimate from microarchitecture analyses, not an Apple figure.
M4_PEAK_GFLOPS = 4 * 4.4 * 4 * 2 * 4  # = 563.2
M4_BW_GBS = 120.0
T4_PEAK_GFLOPS, T4_BW_GBS = 8100.0, 320.0


def roofline_fig():
    """Roofline of the M4 CPU and Colab T4 with YOLOv8n's layers and the measured M4 point."""
    import numpy as np

    prof_f = RESULTS_DIR / "yolo_layer_profile.json"
    mac_f = RESULTS_DIR / "performance_mac_vm.json"
    if not (prof_f.exists() and mac_f.exists()):
        print("roofline skipped")
        return
    prof = json.loads(prof_f.read_text())
    mac = json.loads(mac_f.read_text())["yolo"]
    ai_net = mac["arithmetic_intensity_whole_net"]
    achieved = next(v["achieved_gflops_per_s"] for v in mac["variants"] if v.get("achieved_gflops_per_s"))
    ai = np.logspace(-1, 3, 200)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.loglog(
        ai,
        np.minimum(M4_PEAK_GFLOPS, ai * M4_BW_GBS),
        color="C0",
        lw=2,
        label="M4, 4 performance cores (est.)",
    )
    ax.loglog(
        ai,
        np.minimum(T4_PEAK_GFLOPS, ai * T4_BW_GBS),
        color="0.5",
        lw=1.2,
        ls="--",
        label="Colab T4 GPU (fp32)",
    )
    for r in prof:
        ax.axvline(r["ai"], ymin=0, ymax=0.04, color="0.6", lw=0.6)
    ax.plot(
        [ai_net],
        [achieved],
        "o",
        color="C3",
        ms=7,
        label=f"YOLOv8n on M4: {achieved:.0f} GFLOP/s at {ai_net:.0f} FLOP/byte",
    )
    ax.axvline(M4_PEAK_GFLOPS / M4_BW_GBS, color="C0", lw=0.8, ls=":")
    ax.text(
        M4_PEAK_GFLOPS / M4_BW_GBS * 1.1,
        2,
        f"M4 balance {M4_PEAK_GFLOPS / M4_BW_GBS:.1f}",
        fontsize=7,
        color="C0",
    )
    ax.set_xlabel("arithmetic intensity (FLOP/byte); ticks = individual conv layers")
    ax.set_ylabel("attainable GFLOP/s")
    ax.set_title("YOLOv8n is compute-bound on the M4 and runs at about a quarter of the roof", fontsize=9)
    ax.set_ylim(1, 2e4)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "roofline.png", dpi=150)
    plt.close(fig)


def yolo_resolution_fig():
    """Validation and test AUC (from the container run) against M4 CPU time, per input size."""
    cont_f = RESULTS_DIR / "performance_container.json"
    mac_f = RESULTS_DIR / "performance_mac_vm.json"
    if not (cont_f.exists() and mac_f.exists()):
        return
    acc = {
        v["imgsz"]: v
        for v in json.loads(cont_f.read_text())["yolo"]["variants"]
        if v["precision"] == "fp32" and "val_auc" in v
    }
    tm = {
        v["imgsz"]: v["inference_ms"]["median"]
        for v in json.loads(mac_f.read_text())["yolo"]["variants"]
        if v["precision"] == "fp32" and v["threads"] == "default"
    }
    sizes = sorted(set(acc) & set(tm))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    a1.plot(sizes, [acc[s]["val_auc"] for s in sizes], "o-", color="C0", label="validation")
    a1.plot(sizes, [acc[s]["test_auc"] for s in sizes], "s--", color="0.5", label="test")
    a1.set_ylim(0.80, 0.92)
    a1.set_xlabel("network input size (px)")
    a1.set_ylabel("ROC AUC")
    a1.legend(fontsize=7)
    a1.set_title("Accuracy barely changes with input size", fontsize=9)
    a2.plot(sizes, [tm[s] for s in sizes], "o-", color="C3")
    a2.set_xlabel("network input size (px)")
    a2.set_ylabel("M4 CPU inference time (ms)")
    a2.set_title("Time grows with pixel count", fontsize=9)
    for s in sizes:
        a2.annotate(f"{tm[s]:.0f}", (s, tm[s]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    fig.suptitle("YOLOv8n trained at 640 px, run at other input sizes", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "yolo_resolution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    pipeline()
    examples()
    roofline_fig()
    yolo_resolution_fig()
    print("figures written to", FIG)
