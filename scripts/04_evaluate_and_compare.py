"""Pass/fail evaluation of every model on both splits, plus YOLO verification.

Run from the repo root after scripts 01 and 02 (and the Colab notebook, if the
YOLO results are available)::

    python scripts/04_evaluate_and_compare.py

For each (split, model):
    - threshold chosen on validation at the default cost ratio (10);
    - test metrics at that threshold, with Wilson 95 % intervals;
    - threshold and test errors across cost ratios 1 to 50;
    - test ROC AUC of PASS frames against frames containing each defect class
      (which defects each model can actually see).

For YOLO only (needs ``results/yolo/yolov8n_<split>/``):
    - ``threshold.json`` next to the weights, used by the CLI;
    - verification: our ONNX Runtime pipeline vs Ultralytics' own predictions;
    - fp16 vs fp32 ONNX: score differences and decision flips.

Writes ``results/comparison.csv``, ``results/comparison.md``,
``results/yolo_verification.json`` and figures in ``results/figures/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from porksort.config import CLASS_NAMES, DEFAULT_COST_RATIO, RESULTS_DIR  # noqa: E402
from porksort.dataset import Box, build_index  # noqa: E402
from porksort.decision import choose_threshold, frame_score  # noqa: E402
from porksort.evaluate import bootstrap_auc, passfail_metrics  # noqa: E402
from porksort.splits import load_split  # noqa: E402

FIG = RESULTS_DIR / "figures"
COST_RATIOS = (1, 2, 3, 5, 10, 20, 50)
LABEL = {"majority": "Majority baseline", "colour_lr": "Colour-histogram LR", "logreg": "Logistic regression",
         "tree": "Decision tree", "svm": "SVM (RBF)", "mlp": "MLP", "yolov8n": "YOLOv8n detector"}


def yolo_scores(run_dir: Path, parts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame] | None:
    """Frame scores from the Colab predictions (max defect confidence per frame)."""
    f = run_dir / "predictions.json"
    if not f.exists():
        return None
    preds = json.loads(f.read_text())
    out = {}
    for sp in ("val", "test"):
        d = parts[sp]
        names = d["image"].map(lambda p: Path(p).name)
        missing = [n for n in names if n not in preds]
        if missing:
            raise ValueError(f"{len(missing)} {sp} frames missing from {f}; was it run on this split?")
        s = [frame_score([Box(**b) for b in preds[n]]) for n in names]
        out[sp] = pd.DataFrame({"image": names, "y_fail": d["fail"].astype(int).to_numpy(), "score": s})
    return out


def evaluate_scores(val: pd.DataFrame, test: pd.DataFrame, test_classes: pd.Series) -> dict:
    thr, _ = choose_threshold(val["y_fail"], val["score"], DEFAULT_COST_RATIO)
    m = passfail_metrics(test["y_fail"], test["score"], thr, DEFAULT_COST_RATIO)
    if test["score"].nunique() > 1:
        m["roc_auc_ci95"] = list(bootstrap_auc(test["y_fail"], test["score"]))
    sweep = []
    for r in COST_RATIOS:
        t, _ = choose_threshold(val["y_fail"], val["score"], r)
        mm = passfail_metrics(test["y_fail"], test["score"], t, r)
        sweep.append({"cost_ratio": r, "threshold": t, "escape_rate": mm["escape_rate"]["value"],
                      "false_reject_rate": mm["false_reject_rate"]["value"], "expected_cost": mm["expected_cost"]})
    per_class = {}
    is_pass = test["y_fail"].to_numpy() == 0
    for c in CLASS_NAMES:
        has = test_classes.str.contains(c).to_numpy()
        if has.sum() >= 2 and is_pass.sum() >= 2:
            sel = is_pass | has
            per_class[c] = {"n": int(has.sum()),
                            "auc_vs_pass": float(roc_auc_score(has[sel], test["score"].to_numpy()[sel])),
                            "frame_recall_at_thr": float((test["score"].to_numpy()[has] >= thr).mean())}
    return {"threshold": thr, "test": m, "cost_sweep": sweep, "per_class": per_class}


def verify_onnx(run_dir: Path, parts: dict, preds: dict, n: int = 60) -> dict:
    """Compare our ONNX pipeline with Ultralytics on the first n test frames."""
    from porksort.detector import OnnxYoloDetector, box_iou
    from porksort.features import load_rgb

    out = {}
    w32 = run_dir / "weights" / "best_fp32.onnx"
    w16 = run_dir / "weights" / "best_fp16.onnx"
    if not w32.exists():
        return {"skipped": "no best_fp32.onnx"}
    det32 = OnnxYoloDetector(w32, conf=0.001, iou=0.6)
    det16 = OnnxYoloDetector(w16, conf=0.001, iou=0.6) if w16.exists() else None
    imgs = parts["test"]["image"].tolist()[:n]
    d_score, matched_iou, n_ul, n_match, d16, flips16 = [], [], 0, 0, [], 0
    thr = json.loads((run_dir / "threshold.json").read_text())["threshold"]
    for p in imgs:
        rgb = load_rgb(p)
        ours = det32.detect(rgb)
        ul = [Box(**b) for b in preds[Path(p).name]]
        d_score.append(abs(frame_score(ours) - frame_score(ul)))
        # match Ultralytics boxes above 0.25 to ours of the same class by IoU
        for u in (b for b in ul if b.conf >= 0.25):
            n_ul += 1
            cand = [o for o in ours if o.cls == u.cls]
            if not cand:
                continue
            A = np.array([u.to_pixels(720, 540)])
            B = np.array([o.to_pixels(720, 540) for o in cand])
            best = box_iou(A, B)[0].max()
            matched_iou.append(float(best))
            n_match += best >= 0.9
        if det16 is not None:
            s16 = frame_score(det16.detect(rgb))
            s32 = frame_score(ours)
            d16.append(abs(s16 - s32))
            flips16 += (s16 >= thr) != (s32 >= thr)
    out["onnx_vs_ultralytics"] = {
        "frames": len(imgs), "frame_score_abs_diff_max": float(np.max(d_score)),
        "frame_score_abs_diff_mean": float(np.mean(d_score)),
        "ultralytics_boxes_conf_ge_0.25": n_ul, "matched_iou_ge_0.9": int(n_match),
        "matched_iou_median": float(np.median(matched_iou)) if matched_iou else None,
    }
    if det16 is not None:
        out["fp16_vs_fp32"] = {"frames": len(imgs), "score_abs_diff_max": float(np.max(d16)),
                               "score_abs_diff_mean": float(np.mean(d16)), "decision_flips": int(flips16)}
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    index = build_index()
    rows, roc_data, all_results, verification = [], {}, {}, {}
    for split in ("blocked", "roboflow"):
        df = load_split(index, split)
        parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("val", "test")}
        scores: dict[str, dict[str, pd.DataFrame]] = {}
        cdir = RESULTS_DIR / "classical" / split
        for name in LABEL:
            fv, ft = cdir / f"{name}_val_scores.csv", cdir / f"{name}_test_scores.csv"
            if fv.exists() and ft.exists():
                scores[name] = {"val": pd.read_csv(fv), "test": pd.read_csv(ft)}
        run = RESULTS_DIR / "yolo" / f"yolov8n_{split}"
        ys = yolo_scores(run, parts)
        if ys is not None:
            scores["yolov8n"] = ys
        for name, sc in scores.items():
            assert (sc["test"]["image"].to_numpy() == parts["test"]["image"].map(lambda p: Path(p).name).to_numpy()).all()
            res = evaluate_scores(sc["val"], sc["test"], parts["test"]["classes"])
            all_results[f"{split}/{name}"] = res
            t = res["test"]
            rows.append({"split": split, "model": LABEL[name], "test_n": t["n"], "test_pass_n": t["n_pass"],
                         "roc_auc": t["roc_auc"], "auc_ci_lo": t.get("roc_auc_ci95", [np.nan] * 2)[0],
                         "auc_ci_hi": t.get("roc_auc_ci95", [np.nan] * 2)[1],
                         "avg_precision": t["average_precision"], "threshold": res["threshold"],
                         "escape_rate": t["escape_rate"]["value"], "escape_ci_hi": t["escape_rate"]["ci95"][1],
                         "false_reject_rate": t["false_reject_rate"]["value"],
                         "balanced_accuracy": t["balanced_accuracy"], "expected_cost": t["expected_cost"]})
            if split == "blocked":
                roc_data[name] = roc_curve(sc["test"]["y_fail"], sc["test"]["score"])
            if name == "yolov8n":
                (run / "threshold.json").write_text(json.dumps(
                    {"threshold": res["threshold"], "cost_ratio": DEFAULT_COST_RATIO, "chosen_on": f"{split} val"}, indent=1))
                preds = json.loads((run / "predictions.json").read_text())
                verification[split] = verify_onnx(run, parts, preds)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    fmt = table.copy()
    for c in fmt.columns:
        if fmt[c].dtype == float:
            fmt[c] = fmt[c].map(lambda v: f"{v:.3f}")
    (RESULTS_DIR / "comparison.md").write_text(fmt.to_markdown(index=False) if hasattr(fmt, "to_markdown") else fmt.to_string())
    (RESULTS_DIR / "comparison_details.json").write_text(json.dumps(all_results, indent=1, default=float))
    if verification:
        (RESULTS_DIR / "yolo_verification.json").write_text(json.dumps(verification, indent=1))

    # ROC curves on the blocked test split
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (fpr, tpr, _) in roc_data.items():
        if name == "majority":
            continue
        auc = table[(table.split == "blocked") & (table.model == LABEL[name])]["roc_auc"].iloc[0]
        ax.plot(fpr, tpr, lw=2 if name == "yolov8n" else 1.2, label=f"{LABEL[name]} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="chance")
    ax.set_xlabel("false reject rate (good frames failed)")
    ax.set_ylabel("defect catch rate (1 - escape rate)")
    ax.set_title("ROC, blocked test split")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_blocked_test.png", dpi=150)
    plt.close(fig)

    # Leakage: AUC on Roboflow test vs blocked test, per model
    piv = table.pivot(index="model", columns="split", values="roc_auc").dropna()
    piv = piv.drop(index=LABEL["majority"], errors="ignore")
    if len(piv) and {"blocked", "roboflow"} <= set(piv.columns):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(piv))
        ax.bar(x - 0.2, piv["roboflow"], 0.4, label="Roboflow split (leaky)")
        ax.bar(x + 0.2, piv["blocked"], 0.4, label="blocked split")
        ax.set_xticks(x, piv.index, rotation=15, fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("test ROC AUC")
        ax.set_title("Effect of the split on measured performance")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "leakage_auc.png", dpi=150)
        plt.close(fig)

    # Cost-ratio sensitivity for the best blocked model
    best = table[table.split == "blocked"].sort_values("roc_auc", ascending=False).iloc[0]
    key = next(k for k, v in LABEL.items() if v == best["model"])
    sw = pd.DataFrame(all_results[f"blocked/{key}"]["cost_sweep"])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(sw.cost_ratio, sw.escape_rate, "o-", label="escape rate (defects passed)")
    ax.plot(sw.cost_ratio, sw.false_reject_rate, "s-", label="false reject rate (good failed)")
    ax.set_xscale("log")
    ax.set_xlabel("cost of a false PASS / cost of a false FAIL")
    ax.set_ylabel("rate on blocked test")
    ax.set_title(f"Threshold sensitivity to cost ratio: {best['model']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cost_ratio_sensitivity.png", dpi=150)
    plt.close(fig)

    print(table[["split", "model", "roc_auc", "auc_ci_lo", "auc_ci_hi", "threshold", "escape_rate",
                 "false_reject_rate", "expected_cost"]].round(3).to_string(index=False))
    if verification:
        print(json.dumps(verification, indent=1))


if __name__ == "__main__":
    main()
