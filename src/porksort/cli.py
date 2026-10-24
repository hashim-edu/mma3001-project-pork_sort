"""Command-line interface.

Inspect one image, several images, or a folder::

    python -m porksort inspect path/to/frames --out reports/
    python -m porksort inspect frame.jpg --weights results/yolo/yolov8n_blocked/weights/best_fp32.onnx

Options:
    --weights    ONNX model (default: the blocked-split YOLOv8n model).
    --threshold  Pass/fail threshold in [0, 1], or ``auto`` (default) to use
                 the threshold chosen on validation data for ``--cost-ratio``.
    --cost-ratio Cost of a false PASS relative to a false FAIL (default 10).
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

#: Selected model: YOLOv8n trained at 640 px, run at 480 px (same validation AUC, about half the time).
DEFAULT_WEIGHTS = REPO_ROOT / "results" / "yolo" / "yolov8n_blocked" / "weights" / "best_fp32_480.onnx"


def resolve_threshold(arg: str, weights: Path, cost_ratio: float = 10.0) -> float:
    """Turn the ``--threshold`` argument into a number.

    ``auto`` looks up the threshold chosen on validation data for this weights
    file and cost ratio in ``threshold.json`` (written by
    ``scripts/04_evaluate_and_compare.py``), one folder above the weights.

    Args:
        arg: ``"auto"`` or a number in [0, 1] as text.
        weights: ONNX model file; its name selects the entry in ``threshold.json``.
        cost_ratio: Cost of a false PASS relative to a false FAIL. Used only with
            ``auto``; thresholds exist for 1, 2, 3, 5, 10, 20 and 50.

    Returns:
        The threshold in [0, 1].

    Raises:
        ValueError: If the threshold is not a number, is outside [0, 1], or no
            threshold was chosen for that weights file and cost ratio.
        FileNotFoundError: If ``threshold.json`` is missing.

    Examples:
        >>> resolve_threshold("0.25", Path("model.onnx"))
        0.25
    """
    if arg != "auto":
        t = float(arg)
        if not 0.0 <= t <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return t
    f = weights.parent.parent / "threshold.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found; run scripts/04_evaluate_and_compare.py or pass --threshold")
    table = json.loads(f.read_text()).get(weights.name, {})
    key = str(int(cost_ratio)) if float(cost_ratio).is_integer() else str(cost_ratio)
    if key not in table:
        raise ValueError(
            f"no threshold for {weights.name} at cost ratio {cost_ratio}; available: {sorted(table)}"
        )
    return float(table[key])


def collect_inputs(paths: list[str]) -> list[Path]:
    """Expand folders into the files they contain.

    Args:
        paths: Image files and folders, as given on the command line.

    Returns:
        Files in the order given, each folder expanded to its files sorted by
        name (hidden files skipped). Non-image files are kept so that they are
        reported as ERROR rather than silently ignored.
    """
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.is_file() and not q.name.startswith("."))
        else:
            out.append(p)
    return out


def inspect_paths(
    paths: list[Path], weights: Path, threshold: float, out_dir: Path
) -> list[InspectionReport]:
    """Inspect each file and write its report.

    Args:
        paths: Files to inspect.
        weights: ONNX model file.
        threshold: Pass/fail threshold in [0, 1].
        out_dir: Folder for the per-frame JSON and PNG reports, created if missing.

    Returns:
        One `porksort.report.InspectionReport` per path, in input order. A file
        that is not a supported image, or cannot be decoded, gets status ERROR
        with the reason in ``warnings``; it never stops the batch.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    from .detector import OnnxYoloDetector

    # Keep the detector's own cut-off at or below the threshold, so no box that
    # could fail the frame is dropped before the decision.
    det = OnnxYoloDetector(weights, conf=min(0.01, threshold))
    reports = []
    for p in paths:
        t0 = time.perf_counter()
        try:
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"not an image file ({p.suffix or 'no extension'})")
            rgb = load_rgb(p)
        # A bad file becomes an ERROR report and the batch carries on.
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
        rep = InspectionReport(
            p.name,
            d.status,
            round(d.score, 4),
            threshold,
            defects_to_dicts(d.defects, rgb.shape[1], rgb.shape[0]),
            model=weights.name,
            width=rgb.shape[1],
            height=rgb.shape[0],
            timing_ms={k: round(v, 2) for k, v in timing.items()},
            warnings=input_warnings(rgb),
        )
        write_report(rep, rgb, out_dir)
        reports.append(rep)
    return reports


def main(argv: list[str] | None = None) -> int:
    """Run the command-line tool.

    Args:
        argv: Arguments without the program name, for example
            ``["inspect", "frames/", "--cost-ratio", "1"]``. ``None`` reads
            ``sys.argv``.

    Returns:
        Exit code: 0 if every frame was inspected, 2 if any frame was an ERROR,
        1 for an invalid threshold, a missing model or threshold file, or no
        input files. Malformed options are rejected by argparse, which exits
        with code 2.
    """
    ap = argparse.ArgumentParser(prog="porksort", description="Pass/fail inspection of pork rasher trays")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("inspect", help="inspect images and write reports")
    ins.add_argument("paths", nargs="+", help="image files or folders")
    ins.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ins.add_argument("--threshold", default="auto", help="number in [0, 1], or 'auto' (from validation)")
    ins.add_argument(
        "--cost-ratio",
        type=float,
        default=10.0,
        help="cost of a false PASS relative to a false FAIL, for --threshold auto (1, 2, 3, 5, 10, 20, 50)",
    )
    ins.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    try:
        thr = resolve_threshold(args.threshold, args.weights, args.cost_ratio)
        paths = collect_inputs(args.paths)
        if not paths:
            raise ValueError("no input files found")
        reports = inspect_paths(paths, args.weights, thr, args.out)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    page = batch_summary(reports, args.out)
    counts = {s: sum(r.status == s for r in reports) for s in ("PASS", "FAIL", "ERROR")}
    summary = ", ".join(f"{counts[s]} {s}" for s in ("PASS", "FAIL", "ERROR"))
    print(f"{len(reports)} frames: {summary} -> {page}")
    # Exit code 2 tells a calling script that some inputs need attention.
    return 2 if counts["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
