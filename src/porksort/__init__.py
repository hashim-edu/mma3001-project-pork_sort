"""PorkSort: automated pass/fail inspection of packaged pork rasher trays.

The package turns an RGB image of a pork rasher tray on a conveyor into an
inspection report: a PASS/FAIL decision, the defects found, where they are,
and how long the inspection took.

Modules
-------
- `porksort.config`: class names, defect set and default settings.
- `porksort.dataset`: reading YOLO labels and building an image index.
- `porksort.splits`: the Roboflow split and a leakage-resistant blocked split.
- `porksort.features`: handcrafted image features for the classical models.
- `porksort.models`: baseline and classical classifiers.
- `porksort.decision`: the frame-level pass/fail rule and threshold selection.
- `porksort.evaluate`: metrics, confidence intervals and cost analysis.
- `porksort.detector`: YOLOv8 wrapper that returns detections.
- `porksort.perf`: timing, FLOP counting and arithmetic intensity.
- `porksort.report`: per-image JSON/PNG reports and a batch HTML summary.
- `porksort.cli`: command-line entry point (`python -m porksort ...`).
"""

__version__ = "0.1.0"
