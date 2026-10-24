# PorkSort

Automated PASS/FAIL inspection of packaged pork rasher trays from conveyor camera frames.
MMA3001 Numerical Methods and Machine Learning, individual project (Dataset 3), Monash University, 2026.
Author: Hashim Navas.

PorkSort takes an RGB frame of a sealed pork rasher tray on a conveyor and produces an inspection
report: **PASS** or **FAIL**, which defects were found (loose meat, twisted meat, unsealed film,
wrinkles, other packaging errors), where they are, a defect score, and how long the inspection took.
It is a decision-support tool: FAIL frames are routed to an operator for re-inspection.

## Results at a glance

Blocked test split (174 frames, 34 PASS), thresholds chosen on validation data:

| Model | Test ROC AUC (95% CI) | Notes |
|---|---|---|
| Logistic regression on handcrafted features | 0.739 (0.649 to 0.823) | best classical model |
| YOLOv8n, ONNX, 480 px input (selected) | 0.887 (0.810 to 0.947) | +0.148 over logistic regression (paired 95% CI 0.068 to 0.235) |

- At the design cost ratio (a missed defect costs 10 re-checks) no model does materially better than
  re-checking every pack (the best passes 2 of 34 good test frames); the selected detector pays off at a
  cost ratio of 1, where it passes 56% of good frames and lets 3.6% of defective frames through.
- Apple M4 CPU: 36.7 ms inference, 41 ms end to end per frame at 480 px.
- The provided Roboflow split leaks (test frames sit within 1 to 30 frames of training frames) and
  inflates classical AUC by 0.07 to 0.10; all headline results use the blocked split.

Full tables: `results/comparison.md`, `results/comparison_details.json`, `results/performance_*.json`.

## Inputs and outputs

| | |
|---|---|
| **Input** | RGB image file (`.jpg`, `.jpeg`, `.png`, `.bmp`), 8 bits per channel. Designed for 720 x 540 px top-down conveyor frames like the training data. Other sizes are accepted (letterboxed) but flagged. Grey or RGBA images are converted to RGB. |
| **Output (per frame)** | `<frame>.json` and an annotated `<frame>.png`. JSON fields: `status` (PASS/FAIL/ERROR), `score` (defect score in [0, 1]), `threshold`, `defects` (class, confidence in [0, 1], pixel box `[x0, y0, x1, y1]`, area as a fraction of the frame), `timing_ms`, `warnings`. |
| **Output (per batch)** | `summary.json` (counts, frames per defect class, median time) and `index.html` (table with thumbnails). |
| **Invalid input** | Non-image or corrupt files get `status: ERROR` with the reason and the batch continues. Exit code 2 if any frame was an ERROR, 1 for bad arguments. Very dark, very bright, near-blank or wrongly sized frames are inspected but carry a warning. |

Decision rule: a frame FAILs if any defect is detected with confidence at or above the threshold. The
threshold is chosen on validation data to minimise expected cost for a given cost ratio r (cost of a
false PASS divided by the cost of a false FAIL); the thresholds are stored in
`results/yolo/yolov8n_blocked/threshold.json`.

## Requirements

- Python 3.10 or newer (developed on 3.13, timed on 3.10). No GPU is needed for inspection or for the
  analysis scripts; only YOLO training needs one, and it runs on Google Colab.
- Python packages: `requirements.txt` (runtime plus the test, documentation and profiling tools).
  `pip install -e ".[dev]"` installs the same development tools from `pyproject.toml`.
- About 1 GB of disk for the dataset, the feature cache and the model files, plus the Python packages.

## Quick start

```bash
git clone https://github.com/hashim-edu/mma3001-project-pork_sort.git
cd mma3001-project-pork_sort
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Inspect frames with the selected model (YOLOv8n ONNX, 480 px) at the design cost ratio (r = 10)
python -m porksort inspect path/to/frames --out reports/
# The operating point where the detector pays off (r = 1)
python -m porksort inspect path/to/frames --cost-ratio 1 --out reports/
open reports/index.html
```

## Reproducing the results

1. **Data.** Download "Pork Rasher Error(Packaging)" version 4 in YOLOv8 format from
   [Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4)
   (CC BY 4.0) and unzip it to `data/pork_v4/` (or set `PORKSORT_DATA` to its folder).
2. `python scripts/01_data_audit.py`: class balance, leakage analysis, tray pitch, builds `splits/blocked_split.csv`.
3. `python scripts/02_classical_models.py`: baseline, logistic regression, decision tree, SVM and MLP on both splits.
4. `notebooks/03_train_yolo_colab.ipynb` on Google Colab (T4 GPU): trains YOLOv8n and exports ONNX at
   320, 480, 640 and 800 px plus fp16. Unzip its output into `results/yolo/yolov8n_blocked/`.
5. `python scripts/04_evaluate_and_compare.py`: pass/fail evaluation of every model, thresholds per cost
   ratio, check of the ONNX pipeline against Ultralytics, fp16 against fp32.
6. `python scripts/05_performance.py`: profiling, feature ablation and optimisation, resolution and
   interpolation study, YOLO FLOPs, arithmetic intensity, timing (`--machine mac --sections timing` for
   another machine).
7. `python scripts/06_figures.py`: pipeline diagram, example frames, roofline and resolution figures.
8. `python scripts/07_render_report.py report/report.md report/report.pdf`: renders the written report.

The repository keeps the selected model (`best_fp32_480.onnx`) and the PyTorch weights (`best.pt`). The
other ONNX exports (640 px fp32 and fp16, 320 px, 800 px) are regenerated by the notebook's export cell;
the ONNX predictions they produced are cached as JSON in `results/yolo/yolov8n_blocked/`, so script 04
reproduces every table without them. All random choices are seeded.

## Repository layout

```
src/porksort/      package (see docs/api/ for the API reference)
  config.py        class names, defect set, defaults
  dataset.py       YOLO label parsing, image index, source-frame grouping
  splits.py        Roboflow split, blocked split with buffer, leakage report
  features.py      handcrafted features for the classical models
  models.py        baseline and classical classifiers
  decision.py      pass/fail rule, cost-based threshold
  evaluate.py      escape rate, false reject rate, Wilson intervals, AUC
  detector.py      YOLOv8 ONNX inference: letterbox, decode, NMS
  perf.py          timing, FLOP counting, arithmetic intensity, roofline
  report.py        JSON/PNG/HTML inspection reports
  cli.py           command-line interface
scripts/           numbered analysis scripts that write to results/
notebooks/         Colab training notebook
splits/            committed blocked split (source frame -> split)
results/           metrics (JSON/CSV), profiles, figures, YOLO run outputs
report/            written report (Markdown source and PDF)
tests/             pytest suite (synthetic data, no dataset needed)
docs/              HTML API documentation (pdoc) and pytest HTML test report
```

## Tests, documentation and code style

```bash
pytest --doctest-modules src tests --html=docs/test_report.html --self-contained-html
pdoc src/porksort -o docs/api --docformat google
ruff check src tests scripts && ruff format --check src tests scripts
```

52 tests (46 unit tests and 6 docstring examples), including manufactured-solution checks: decoding a
hand-encoded box, the convolution FLOP formula against a loop count, the threshold search against brute
force, and the optimised histogram against NumPy's reference. Open `docs/api/index.html` for the API
documentation and `docs/test_report.html` for the latest test report.

Every public function has a Google-style docstring giving its arguments, return value and the errors it
raises. Code is formatted and linted with ruff (settings in `pyproject.toml`, 110-character lines).

## Data

"Pork Rasher Error(Packaging)" v4, Roboflow Universe workspace `hello-2aqe0`, licensed CC BY 4.0.
The images are not redistributed in this repository.

## AI use

Claude (Anthropic) was used to write most of the code, run the analyses and draft the report, under the
author's direction. All decisions, results and claims were checked by the author. See the AI reflection
in the written report.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the original authors. The trained model files in
`results/yolo/` were produced with Ultralytics YOLOv8, which is licensed under AGPL-3.0; those files are
covered by that licence.
