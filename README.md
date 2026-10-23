# PorkSort

Automated PASS/FAIL inspection of packaged pork rasher trays from conveyor camera frames.
MMA3001 Numerical Methods and Machine Learning, individual project (Dataset 3), Monash University, 2026.

PorkSort takes an RGB frame of a sealed pork rasher tray on a conveyor and produces an inspection
report: **PASS** or **FAIL**, which defects were found (loose meat, twisted meat, unsealed film,
wrinkles, other packaging errors), where they are, a confidence score, and how long the inspection
took. The aim is to support, not replace, a line operator: frames that FAIL are routed to manual
re-inspection.

## Inputs and outputs

| | |
|---|---|
| **Input** | RGB image file (`.jpg`, `.jpeg`, `.png`, `.bmp`), 8-bit per channel. Designed for 720 x 540 px top-down conveyor frames like the training data. Other sizes are accepted (letterboxed) but flagged as less reliable. Grey or RGBA images are converted to RGB. |
| **Output (per frame)** | `<frame>.json` report and `<frame>.png` annotated image. JSON fields: `status` (PASS/FAIL/ERROR), `score` (defect score in [0, 1]), `threshold`, `defects` (class, confidence in [0, 1], pixel box `[x0, y0, x1, y1]`, area as a fraction of the frame), `timing_ms`, `warnings`. |
| **Output (per batch)** | `summary.json` (counts, frames per defect class, median time) and `index.html` (table with thumbnails). |
| **Invalid input** | Non-image or corrupt files get `status: ERROR` with the reason, and the batch continues. The CLI exits with code 2 if any frame was an ERROR, 1 for bad arguments. Very dark, very bright, near-blank or wrongly sized frames are inspected but carry a warning. |

Decision rule: a frame FAILs if any defect is detected with confidence at or above the threshold.
The threshold is chosen on validation data to minimise expected cost, with a defective pack shipped
(false PASS) costed at 10 times a good pack rejected (false FAIL).

## Quick start

```bash
git clone <this repo> && cd porksort
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Inspect frames (uses the trained YOLOv8n ONNX model in results/yolo/)
python -m porksort inspect path/to/frames --out reports/
open reports/index.html
```

## Reproducing the results

1. **Data.** Download "Pork Rasher Error(Packaging)" version 4 in YOLOv8 format from
   [Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4)
   (CC BY 4.0) and unzip it to `data/pork_v4/` (or set `PORKSORT_DATA` to its folder).
2. `python scripts/01_data_audit.py`: class balance, leakage analysis, builds `splits/blocked_split.csv`.
3. `python scripts/02_classical_models.py`: baseline, logistic regression, decision tree, SVM and MLP on both splits.
4. `notebooks/03_train_yolo_colab.ipynb` on Google Colab (T4 GPU): trains YOLOv8n, exports ONNX. Unzip its output into `results/yolo/`.
5. `python scripts/04_evaluate_yolo.py`: pass/fail evaluation of YOLO, threshold choice, comparison with Ultralytics' own predictions.
6. `python scripts/05_performance.py`: timing, profiling, FLOPs and arithmetic intensity, resolution, interpolation and precision sensitivity.

All random choices are seeded. The blocked split is committed so every machine uses the same frames.

## Repository layout

```
src/porksort/      package (see docs/ for the API reference)
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
results/           metrics (JSON/CSV) and figures used in the report
tests/             pytest suite (synthetic data, no dataset needed)
docs/              HTML API documentation (pdoc) and test report
```

## Tests and documentation

```bash
pytest --doctest-modules src tests --html=docs/test_report.html --self-contained-html
pdoc src/porksort -o docs/api --docformat google
```

## Data

"Pork Rasher Error(Packaging)" v4, Roboflow Universe workspace `hello-2aqe0`, licensed CC BY 4.0.
The images are not redistributed in this repository.

## AI use

Claude (Anthropic) was used to help design, write and test the code and documentation. All design
decisions, results and claims were checked by the author. See the AI reflection in the written report.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from the original authors.
