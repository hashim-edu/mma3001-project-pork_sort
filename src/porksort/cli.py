"""Command-line interface.

Inspect one image, several images, or a folder::

    python -m porksort inspect path/to/frames --out reports/
    python -m porksort inspect frame.jpg --weights results/yolo/yolov8n_blocked/weights/best_fp32.onnx

Options:
    --weights    ONNX model (default: the blocked-split YOLOv8n model).
    --threshold  Pass/fail threshold in [0, 1], or ``auto`` (default) to use
                 the cost-optimal threshold saved next to the weights.
    --out        Output folder for per-frame JSON/PNG and the batch index.html.

Exit code is 0 if every frame was inspected (PASS or FAIL), 2 if any frame
was an ERROR (unreadable or unsupported input), 1 for bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import IMAGE_EXTENSIONS, REPO_ROOT
from .decision import decide
from .features import load_rgb
from .report import InspectionReport, batch_summary, defects_to_dicts, input_warnings, write_report

DEFAULT_WEIGHTS = REPO_ROOT / "results" / "yolo" / "yolov8n_blocked" / "weights" / "best_fp32.onnx"


def resolve_threshold(arg: str, weights: Path) -> float:
    """Parse ``--threshold``; ``auto`` reads ``threshold.json`` from the run folder."""
    if arg != "auto":
        t = float(arg)
        if not 0.0 <= t <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return t
    f = weights.parent.parent / "threshold.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found; run scripts/04_evaluate_yolo.py or pass --threshold")
    return float(json.loads(f.read_text())["threshold"])


def collect_inputs(paths: list[str]) -> list[Path]:
    """Expand folders into their files (sorted). Non-images are kept so they are reported as ERROR."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.is_file() and not q.name.startswith("."))
        else:
            out.append(p)
    return out


def inspect_paths(paths: list[Path], weights: Path, threshold: float, out_dir: Path) -> list[InspectionReport]:
    """Inspect each path and write its report. Never raises for a bad image."""
    from .detector import OnnxYoloDetector

    det = OnnxYoloDetector(weights, conf=min(0.01, threshold))
    reports = []
    for p in paths:
        t0 = time.perf_counter()
        try:
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"not an image file ({p.suffix or 'no extension'})")
            rgb = load_rgb(p)
        except (FileNotFoundError, ValueError) as exc:
            rep = InspectionReport(p.name, "ERROR", None, threshold, model=weights.name, warnings=[str(exc)])
            write_report(rep, None, out_dir)
            reports.append(rep)
            continue
        t_load = time.perf_counter()
        boxes = det.detect(rgb)
        d = decide(boxes, threshold)
        timing = {"load": 1e3 * (t_load - t0), **det.last_timing}
        timing["total"] = 1e3 * (time.perf_counter() - t0)
        rep = InspectionReport(p.name, d.status, round(d.score, 4), threshold,
                               defects_to_dicts(d.defects, rgb.shape[1], rgb.shape[0]), model=weights.name,
                               width=rgb.shape[1], height=rgb.shape[0],
                               timing_ms={k: round(v, 2) for k, v in timing.items()},
                               warnings=input_warnings(rgb))
        write_report(rep, rgb, out_dir)
        reports.append(rep)
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="porksort", description="Pass/fail inspection of pork rasher trays")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("inspect", help="inspect images and write reports")
    ins.add_argument("paths", nargs="+", help="image files or folders")
    ins.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ins.add_argument("--threshold", default="auto")
    ins.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    try:
        thr = resolve_threshold(args.threshold, args.weights)
        paths = collect_inputs(args.paths)
        if not paths:
            raise ValueError("no input files found")
        reports = inspect_paths(paths, args.weights, thr, args.out)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    page = batch_summary(reports, args.out)
    counts = {s: sum(r.status == s for r in reports) for s in ("PASS", "FAIL", "ERROR")}
    print(f"{len(reports)} frames: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['ERROR']} ERROR -> {page}")
    return 2 if counts["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
