"""Inspection reports: one JSON and one annotated image per frame, plus a batch summary.

Report fields (``InspectionReport``):

========== ================================================================
image      file name of the inspected frame
status     ``PASS``, ``FAIL``, or ``ERROR`` (input could not be inspected)
score      defect score in [0, 1] (highest defect confidence)
threshold  score at or above which the frame fails
defects    list of ``{class, confidence, bbox_px [x0, y0, x1, y1], area_frac}``
model      model file used
timing_ms  ``load``, ``preprocess``, ``inference``, ``postprocess``, ``total``
warnings   anything unusual about the input (wrong size, near-blank frame ...)
========== ================================================================
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import __version__
from .dataset import Box

COLOURS = {"loose-meat": (255, 200, 0), "packaging-error": (0, 200, 255), "twisted-meat": (255, 0, 255),
           "unsealed": (60, 255, 60), "wrinkle": (255, 140, 0)}


@dataclass
class InspectionReport:
    """Result of inspecting one frame. See the module docstring for fields."""

    image: str
    status: str
    score: float | None
    threshold: float | None
    defects: list[dict] = field(default_factory=list)
    model: str = ""
    width: int | None = None
    height: int | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    porksort_version: str = __version__

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(asdict(self), indent=1)


def defects_to_dicts(boxes: list[Box], width: int, height: int) -> list[dict]:
    """Convert boxes to the report's defect entries (pixel corners, area fraction)."""
    return [{"class": b.cls, "confidence": round(b.conf, 4),
             "bbox_px": [round(v, 1) for v in b.to_pixels(width, height)],
             "area_frac": round(b.area, 5)} for b in boxes]


def input_warnings(rgb: np.ndarray, expected_wh=(720, 540)) -> list[str]:
    """Checks that flag frames unlike the training data (they are still inspected)."""
    w = []
    h, wd = rgb.shape[:2]
    if (wd, h) != tuple(expected_wh):
        w.append(f"size {wd}x{h} differs from the {expected_wh[0]}x{expected_wh[1]} training frames; "
                 "result is less reliable")
    mean = float(rgb.mean())
    if mean < 15:
        w.append(f"frame is very dark (mean level {mean:.0f}/255); check lighting")
    if mean > 240:
        w.append(f"frame is very bright (mean level {mean:.0f}/255); check exposure")
    if float(rgb.std()) < 3:
        w.append("frame is nearly uniform; camera may be blocked or the frame blank")
    return w


def annotate(rgb: np.ndarray, rep: InspectionReport) -> Image.Image:
    """Draw defect boxes and a PASS/FAIL banner on a copy of the frame."""
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    for dft in rep.defects:
        col = COLOURS.get(dft["class"], (255, 255, 255))
        d.rectangle(dft["bbox_px"], outline=col, width=3)
        x0, y0 = dft["bbox_px"][:2]
        d.text((x0 + 3, max(0, y0 - 12)), f"{dft['class']} {dft['confidence']:.2f}", fill=col)
    banner = {"PASS": (30, 140, 60), "FAIL": (190, 30, 30)}.get(rep.status, (90, 90, 90))
    d.rectangle([0, 0, im.width, 26], fill=banner)
    score = "n/a" if rep.score is None else f"{rep.score:.3f}"
    thr = "n/a" if rep.threshold is None else f"{rep.threshold:.3f}"
    d.text((8, 7), f"{rep.status}   score {score}  threshold {thr}   {len(rep.defects)} defect(s)", fill="white")
    return im


def write_report(rep: InspectionReport, rgb: np.ndarray | None, out_dir: str | Path) -> Path:
    """Write ``<stem>.json`` and, if a frame is given, ``<stem>.png`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(rep.image).stem
    (out / f"{stem}.json").write_text(rep.to_json())
    if rgb is not None:
        annotate(rgb, rep).save(out / f"{stem}.png")
    return out / f"{stem}.json"


def batch_summary(reports: list[InspectionReport], out_dir: str | Path) -> Path:
    """Write ``summary.json`` and a self-contained ``index.html`` table for a batch."""
    out = Path(out_dir)
    n = len(reports)
    counts = {s: sum(r.status == s for r in reports) for s in ("PASS", "FAIL", "ERROR")}
    by_class: dict[str, int] = {}
    for r in reports:
        for c in {d["class"] for d in r.defects}:
            by_class[c] = by_class.get(c, 0) + 1
    totals = [r.timing_ms.get("total") for r in reports if r.timing_ms.get("total") is not None]
    summary = {"n_frames": n, "counts": counts, "frames_with_class": by_class,
               "median_total_ms": float(np.median(totals)) if totals else None,
               "model": reports[0].model if reports else "", "porksort_version": __version__}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    rows = []
    for r in reports:
        stem = Path(r.image).stem
        img = f'<img src="{html.escape(stem)}.png" width="240">' if r.status != "ERROR" else ""
        defects = ", ".join(f"{d['class']} ({d['confidence']:.2f})" for d in r.defects) or "none"
        warn = "<br>".join(html.escape(w) for w in r.warnings)
        score = "" if r.score is None else f"{r.score:.3f}"
        rows.append(f"<tr class='{r.status}'><td>{img}</td><td>{html.escape(r.image)}</td>"
                    f"<td><b>{r.status}</b></td><td>{score}</td><td>{html.escape(defects)}</td><td>{warn}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>PorkSort inspection report</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px;vertical-align:top}}
tr.PASS td:nth-child(3){{color:#1e8c3c}}tr.FAIL td:nth-child(3){{color:#be1e1e}}tr.ERROR td:nth-child(3){{color:#666}}</style>
</head><body><h1>PorkSort inspection report</h1>
<p>{n} frames: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['ERROR']} ERROR.
Model: {html.escape(summary['model'])}. Median time per frame: {summary['median_total_ms'] or 0:.1f} ms.</p>
<table><tr><th>frame</th><th>file</th><th>status</th><th>score</th><th>defects</th><th>warnings</th></tr>
{''.join(rows)}</table></body></html>"""
    (out / "index.html").write_text(page)
    return out / "index.html"
