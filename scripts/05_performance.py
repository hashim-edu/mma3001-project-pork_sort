"""Performance, optimisation and sensitivity analysis.

Run from the repo root::

    python scripts/05_performance.py --machine container      # everything
    python scripts/05_performance.py --machine mac --sections timing

Sections:

``profile``     line-by-line profile of feature extraction (line_profiler) and
                time per feature group.
``ablation``    accuracy lost by dropping each feature group (logistic
                regression, blocked split), so a slow group can be removed only
                if it is not paying for itself.
``resolution``  classical pipeline: working resolution x interpolation method
                -> accuracy and time.
``yolo``        YOLO: FLOPs and arithmetic intensity from the ONNX graph
                (checked against Ultralytics), fp32 vs fp16, thread count,
                input size 320/480/640/800 -> accuracy and time.
``timing``      the timing-only parts of ``profile`` and ``yolo`` (for running
                on another machine).
``repeat``      repeatability of the MLP across 5 random seeds.

Writes ``results/performance_<machine>.json`` and figures.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from porksort import features as F  # noqa: E402
from porksort.config import DEFAULT_COST_RATIO, RESULTS_DIR  # noqa: E402
from porksort.dataset import build_index  # noqa: E402
from porksort.decision import choose_threshold, frame_score  # noqa: E402
from porksort.evaluate import passfail_metrics  # noqa: E402
from porksort.perf import onnx_conv_profile, time_call  # noqa: E402
from porksort.splits import load_split  # noqa: E402

FIG = RESULTS_DIR / "figures"


def parts_blocked(index, train_one_copy=False):
    df = load_split(index, "blocked")
    if train_one_copy:
        df = df[(df.split != "train") | (df["copy"] == 0)]
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}


def fit_logreg(Xtr, ytr, C=0.001):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(Xtr, ytr)


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


# --------------------------------------------------------------------- profile
def section_profile(index, out):
    paths = index["image"].iloc[:30].tolist()
    imgs = [F.load_rgb(p) for p in paths]
    cfg = F.FeatureConfig()
    out["load_ms"] = time_call(lambda: F.load_rgb(paths[0]), repeats=30)
    out["extract_ms"] = time_call(lambda: [F.extract(im, cfg) for im in imgs], repeats=3, warmup=1)
    out["extract_ms"] = {k: v / len(imgs) if k != "repeats" else v for k, v in out["extract_ms"].items()}
    # resize + colour conversion happen once per call, so time them on their own
    small = F.resize(imgs[0], cfg.work_size, cfg.interpolation)
    per_group = {"resize": time_call(lambda: F.resize(imgs[0], cfg.work_size, cfg.interpolation))["median"],
                 "rgb2hsv": time_call(lambda: F.rgb2hsv(small))["median"]}
    base = time_call(lambda: F.extract(imgs[0], F.FeatureConfig(groups=("meat",))))["median"]
    for g in F.ALL_GROUPS:
        per_group[g] = time_call(lambda: F.extract(imgs[0], F.FeatureConfig(groups=(g,))))["median"]
    per_group["_note"] = "each group timed alone; includes resize + rgb2hsv + grey (about %.2f ms)" % base
    out["group_ms"] = per_group
    try:
        from line_profiler import LineProfiler

        lp = LineProfiler()
        lp.add_function(F.extract)
        wrapped = lp(F.extract)
        for im in imgs[:10]:
            wrapped(im, cfg)
        buf = io.StringIO()
        lp.print_stats(stream=buf, output_unit=1e-3)
        (RESULTS_DIR / "profile_features.txt").write_text(buf.getvalue())
        out["line_profile"] = "results/profile_features.txt"
    except ImportError:
        out["line_profile"] = "line_profiler not installed"


# -------------------------------------------------------------------- ablation
def section_ablation(index, out):
    cfg = F.FeatureConfig()
    X = F.extract_many(index["image"], cfg, cache_dir=RESULTS_DIR / "cache")
    row = {p: i for i, p in enumerate(index["image"])}
    P = parts_blocked(index)
    Xs = {s: X[[row[p] for p in d["image"]]] for s, d in P.items()}
    ys = {s: d["fail"].to_numpy().astype(int) for s, d in P.items()}
    sl = F.group_slices(cfg)
    res = {}
    for drop in (None,) + F.ALL_GROUPS:
        cols = np.concatenate([np.arange(s.start, s.stop) for g, s in sl.items() if g != drop])
        m = fit_logreg(Xs["train"][:, cols], ys["train"])
        res[drop or "none"] = {"n_features": int(len(cols)),
                               "val_auc": auc(ys["val"], m.predict_proba(Xs["val"][:, cols])[:, 1]),
                               "test_auc": auc(ys["test"], m.predict_proba(Xs["test"][:, cols])[:, 1])}
    out["ablation_logreg_drop_group"] = res


# ------------------------------------------------------------------ resolution
def section_resolution(index, out):
    P = parts_blocked(index, train_one_copy=True)
    sizes = [(120, 90), (240, 180), (480, 360)]
    methods = ["nearest", "bilinear", "bicubic", "box"]
    img0 = F.load_rgb(P["test"]["image"].iloc[0])
    res = []
    for ws in sizes:
        for meth in methods:
            cfg = F.FeatureConfig(work_size=ws, interpolation=meth)
            X = {s: F.extract_many(d["image"], cfg, cache_dir=RESULTS_DIR / "cache") for s, d in P.items()}
            y = {s: d["fail"].to_numpy().astype(int) for s, d in P.items()}
            m = fit_logreg(X["train"], y["train"])
            res.append({"work_size": list(ws), "interpolation": meth,
                        "val_auc": auc(y["val"], m.predict_proba(X["val"])[:, 1]),
                        "test_auc": auc(y["test"], m.predict_proba(X["test"])[:, 1]),
                        "extract_ms": time_call(lambda: F.extract(img0, cfg), repeats=20)["median"]})
            print("resolution", ws, meth, round(res[-1]["val_auc"], 3), round(res[-1]["extract_ms"], 1), "ms")
    out["classical_resolution"] = res
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    for meth in methods:
        r = [x for x in res if x["interpolation"] == meth]
        axs[0].plot([x["work_size"][0] for x in r], [x["val_auc"] for x in r], "o-", label=meth)
        axs[1].plot([x["work_size"][0] for x in r], [x["extract_ms"] for x in r], "o-", label=meth)
    axs[0].set_xlabel("working width (px)"); axs[0].set_ylabel("validation ROC AUC")
    axs[1].set_xlabel("working width (px)"); axs[1].set_ylabel("feature time per frame (ms)")
    axs[0].legend(fontsize=7, title="interpolation", title_fontsize=7)
    fig.suptitle("Classical pipeline: resolution and interpolation", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "classical_resolution.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ yolo
def section_yolo(index, out, accuracy=True):
    from porksort.detector import OnnxYoloDetector

    run = RESULTS_DIR / "yolo" / "yolov8n_blocked"
    w = run / "weights"
    if not (w / "best_fp32.onnx").exists():
        out["yolo"] = "skipped: no results/yolo/yolov8n_blocked/weights/best_fp32.onnx"
        return
    y = {}
    # FLOPs and arithmetic intensity from the graph
    prof = onnx_conv_profile(w / "best_fp32.onnx")
    total_flops = sum(r["flops"] for r in prof)
    total_bytes = sum(r["bytes"] for r in prof)
    y["conv_layers"] = len(prof)
    y["gflops_conv_640"] = total_flops / 1e9
    y["mbytes_conv_640"] = total_bytes / 1e6
    y["arithmetic_intensity_whole_net"] = total_flops / total_bytes
    ais = np.array([r["ai"] for r in prof])
    y["ai_per_layer"] = {"min": float(ais.min()), "median": float(np.median(ais)), "max": float(ais.max())}
    envf = run / "env.json"
    if envf.exists():
        y["ultralytics_model_info"] = json.loads(envf.read_text()).get("model_info")
        y["colab_gpu_predict_ms_median"] = json.loads(envf.read_text()).get("gpu_predict_ms_median")
    (RESULTS_DIR / "yolo_layer_profile.json").write_text(json.dumps(prof, indent=0))

    P = parts_blocked(index)
    img0 = F.load_rgb(P["test"]["image"].iloc[0])
    thr = json.loads((run / "threshold.json").read_text())["threshold"] if (run / "threshold.json").exists() else 0.25
    variants = [("fp32", 640, "best_fp32.onnx"), ("fp16", 640, "best_fp16.onnx"),
                ("fp32", 320, "best_fp32_320.onnx"), ("fp32", 480, "best_fp32_480.onnx"),
                ("fp32", 800, "best_fp32_800.onnx")]
    res = []
    for prec, size, fn in variants:
        if not (w / fn).exists():
            continue
        for threads in ((None, 1) if (prec, size) == ("fp32", 640) else (None,)):
            det = OnnxYoloDetector(w / fn, conf=0.001, threads=threads)
            x, _, _ = det.preprocess(img0)
            t_inf = time_call(lambda: det.raw(x), repeats=30)
            t_all = time_call(lambda: det.detect(img0), repeats=20)
            r = {"precision": prec, "imgsz": size, "threads": threads or "default",
                 "inference_ms": t_inf, "end_to_end_ms": t_all["median"]}
            if size == 640 and prec == "fp32" and threads is None:
                r["achieved_gflops_per_s"] = y["gflops_conv_640"] / (t_inf["median"] / 1e3)
            if accuracy and threads is None:
                sc = {}
                for sp in ("val", "test"):
                    sc[sp] = np.array([frame_score(det.detect(F.load_rgb(p))) for p in P[sp]["image"]])
                yv, yt = P["val"]["fail"].to_numpy(), P["test"]["fail"].to_numpy()
                t, _ = choose_threshold(yv, sc["val"], DEFAULT_COST_RATIO)
                m = passfail_metrics(yt, sc["test"], t, DEFAULT_COST_RATIO)
                r.update({"val_auc": auc(yv, sc["val"]), "test_auc": m["roc_auc"], "threshold": t,
                          "test_escape": m["escape_rate"]["value"], "test_false_reject": m["false_reject_rate"]["value"]})
            res.append(r)
            print("yolo", prec, size, threads, round(t_inf["median"], 1), "ms", r.get("val_auc"))
    y["variants"] = res
    y["pass_fail_threshold_used_for_640"] = thr
    out["yolo"] = y
    acc = [r for r in res if "val_auc" in r and r["precision"] == "fp32"]
    if acc:
        acc.sort(key=lambda r: r["imgsz"])
        fig, ax1 = plt.subplots(figsize=(5.5, 3.2))
        ax1.plot([r["imgsz"] for r in acc], [r["test_auc"] for r in acc], "o-", color="C0", label="test ROC AUC")
        ax1.set_xlabel("network input size (px)")
        ax1.set_ylabel("test ROC AUC", color="C0")
        ax2 = ax1.twinx()
        ax2.plot([r["imgsz"] for r in acc], [r["inference_ms"]["median"] for r in acc], "s--", color="C3")
        ax2.set_ylabel("CPU inference time (ms)", color="C3")
        ax1.set_title("YOLOv8n: input resolution trade-off (trained at 640)")
        fig.tight_layout()
        fig.savefig(FIG / "yolo_resolution.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------- repeat
def section_repeat(index, out):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = F.FeatureConfig()
    X = F.extract_many(index["image"], cfg, cache_dir=RESULTS_DIR / "cache")
    row = {p: i for i, p in enumerate(index["image"])}
    P = parts_blocked(index)
    Xs = {s: X[[row[p] for p in d["image"]]] for s, d in P.items()}
    ys = {s: d["fail"].to_numpy().astype(int) for s, d in P.items()}
    summ = json.loads((RESULTS_DIR / "classical" / "blocked" / "summary.json").read_text())
    bp = summ["models"]["mlp"]["best_params"]
    aucs, iters = [], []
    for seed in range(5):
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=tuple(bp["clf__hidden_layer_sizes"]),
                          alpha=bp["clf__alpha"], max_iter=400, early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=15, random_state=seed)).fit(Xs["train"], ys["train"])
        aucs.append(auc(ys["test"], m.predict_proba(Xs["test"])[:, 1]))
        iters.append(int(m[-1].n_iter_))
    out["mlp_seed_repeatability"] = {"test_auc": aucs, "mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1)),
                                     "n_iter": iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default=platform.node() or "unknown")
    ap.add_argument("--sections", nargs="+", default=["profile", "ablation", "resolution", "yolo", "repeat"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    index = build_index()
    out = {"machine": args.machine, "platform": platform.platform(), "python": platform.python_version(),
           "cpu_count": os.cpu_count()}
    secs = set(args.sections)
    if "timing" in secs:
        secs |= {"profile", "yolo_timing"}
    t0 = time.time()
    if "profile" in secs:
        section_profile(index, out)
    if "ablation" in secs:
        section_ablation(index, out)
    if "resolution" in secs:
        section_resolution(index, out)
    if "yolo" in secs:
        section_yolo(index, out, accuracy=True)
    elif "yolo_timing" in secs:
        section_yolo(index, out, accuracy=False)
    if "repeat" in secs:
        section_repeat(index, out)
    out["script_seconds"] = time.time() - t0
    f = RESULTS_DIR / f"performance_{args.machine}.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(out)
    f.write_text(json.dumps(old, indent=1))
    print("wrote", f)


if __name__ == "__main__":
    main()
