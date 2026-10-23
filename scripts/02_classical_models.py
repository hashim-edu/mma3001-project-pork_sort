"""Train and evaluate the baseline and classical models on both splits.

Run from the repo root::

    python scripts/02_classical_models.py [--splits blocked roboflow] [--jobs 2]

Protocol, identical for every model:

1. Hyperparameters chosen by 5-fold StratifiedGroupKFold cross-validation on
   the training set, scored by ROC AUC. Groups are 200-frame blocks, so the
   three augmented copies of a frame and its neighbouring frames never sit on
   both sides of a fold.
2. The best configuration is refitted on the whole training set.
3. The PASS/FAIL threshold is chosen on the validation set by minimum expected
   cost (false PASS = 10, false FAIL = 1).
4. The test set is scored once, at that threshold.

Writes ``results/classical/<split>/summary.json`` and per-image score CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from porksort.config import DEFAULT_COST_RATIO, RESULTS_DIR  # noqa: E402
from porksort.dataset import build_index  # noqa: E402
from porksort.decision import choose_threshold  # noqa: E402
from porksort.evaluate import bootstrap_auc, passfail_metrics  # noqa: E402
from porksort.features import FeatureConfig, extract_many, group_slices, load_rgb, extract  # noqa: E402
from porksort.models import MODEL_NAMES, defect_score, make_model  # noqa: E402
from porksort.splits import load_split  # noqa: E402


def feature_matrix(index: pd.DataFrame, cfg: FeatureConfig, jobs: int) -> np.ndarray:
    """Features for every image in the index (cached)."""
    t = time.perf_counter()
    X = extract_many(index["image"], cfg, n_jobs=jobs, cache_dir=RESULTS_DIR / "cache")
    print(f"features {X.shape} in {time.perf_counter() - t:.1f} s")
    return X


def time_inference(model, x_row: np.ndarray, repeats: int = 200) -> float:
    """Median wall time in ms to score one frame (features already computed)."""
    for _ in range(10):
        defect_score(model, x_row)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        defect_score(model, x_row)
        ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["blocked", "roboflow"])
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--cost-ratio", type=float, default=DEFAULT_COST_RATIO)
    args = ap.parse_args()

    cfg = FeatureConfig()
    index = build_index()
    X_all = feature_matrix(index, cfg, args.jobs)
    row_of = {p: i for i, p in enumerate(index["image"])}
    colour = group_slices(cfg)["colour"]

    # Feature extraction cost per frame (single process, median of 30 frames)
    sample = index["image"].iloc[:30].tolist()
    ts = []
    for p in sample:
        t = time.perf_counter()
        extract(load_rgb(p), cfg)
        ts.append(time.perf_counter() - t)
    feat_ms = 1000 * float(np.median(ts))

    for split_name in args.splits:
        out_dir = RESULTS_DIR / "classical" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        df = load_split(index, split_name)
        parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
        X = {s: X_all[[row_of[p] for p in d["image"]]] for s, d in parts.items()}
        y = {s: d["fail"].to_numpy().astype(int) for s, d in parts.items()}
        groups = parts["train"]["block"].to_numpy()
        summary = {"split": split_name, "feature_config": cfg.__dict__, "feature_ms_per_frame": feat_ms,
                   "n": {s: int(len(d)) for s, d in parts.items()}, "models": {}}

        for name in args.models:
            pipe, grid = make_model(name, colour_slice=colour)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                if grid:
                    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=args.jobs, refit=True)
                    gs.fit(X["train"], y["train"], groups=groups)
                    model, best = gs.best_estimator_, gs.best_params_
                    i = gs.best_index_
                    cv_auc = (float(gs.cv_results_["mean_test_score"][i]), float(gs.cv_results_["std_test_score"][i]))
                else:
                    model, best, cv_auc = pipe.fit(X["train"], y["train"]), {}, (0.5, 0.0)
            fit_s = time.perf_counter() - t0
            n_conv_warn = sum(issubclass(x.category, ConvergenceWarning) for x in w)

            s_val = defect_score(model, X["val"])
            s_test = defect_score(model, X["test"])
            thr, val_cost = choose_threshold(y["val"], s_val, args.cost_ratio)
            m_val = passfail_metrics(y["val"], s_val, thr, args.cost_ratio)
            m_test = passfail_metrics(y["test"], s_test, thr, args.cost_ratio)
            m_test["roc_auc_ci95"] = list(bootstrap_auc(y["test"], s_test)) if name != "majority" else [0.5, 0.5]

            clf = model[-1]
            extra = {}
            if hasattr(clf, "n_iter_"):
                extra["n_iter"] = np.asarray(clf.n_iter_).tolist()
            if hasattr(clf, "loss_curve_"):
                extra["loss_curve"] = [float(v) for v in clf.loss_curve_]
            if hasattr(clf, "get_depth"):
                extra["depth"], extra["leaves"] = int(clf.get_depth()), int(clf.get_n_leaves())
            if hasattr(clf, "n_support_"):
                extra["n_support"] = clf.n_support_.tolist()

            summary["models"][name] = {
                "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
                "cv_roc_auc_mean_std": cv_auc, "fit_seconds_incl_search": fit_s,
                "convergence_warnings": n_conv_warn, "threshold": thr,
                "val": m_val, "test": m_test,
                "predict_ms_per_frame": time_inference(model, X["test"][:1]),
                **extra,
            }
            pd.DataFrame({"image": parts["val"]["image"].map(lambda p: Path(p).name), "y_fail": y["val"],
                          "score": s_val}).to_csv(out_dir / f"{name}_val_scores.csv", index=False)
            pd.DataFrame({"image": parts["test"]["image"].map(lambda p: Path(p).name), "y_fail": y["test"],
                          "score": s_test}).to_csv(out_dir / f"{name}_test_scores.csv", index=False)
            print(f"[{split_name}] {name:9s} cv_auc={cv_auc[0]:.3f}±{cv_auc[1]:.3f} "
                  f"test_auc={m_test['roc_auc']:.3f} thr={thr:.3f} "
                  f"escape={m_test['escape_rate']['value']:.3f} false_reject={m_test['false_reject_rate']['value']:.3f} "
                  f"fit={fit_s:.1f}s {best}")

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
