# PorkSort: Pass/Fail Inspection of Pork Rasher Packaging

Oct 25, 2026 · Hashim Navas

Hashim Navas, student ID \[add\] · MMA3001 Numerical Methods and Machine Learning, Semester 2 2026, Monash University · Individual project, Dataset 3 · Repository: github.com/hashim-edu/mma3001-project-pork\_sort

## How to use this scaffold

- **Limit.** The brief allows 5 to 10 pages; references and appendices do not count. With 11 pt Times and 20 mm margins, about 4,000 words of body text plus the four figures and three tables below fills about 9 pages.
- **Each section lists** what the brief requires, a word budget, the facts to use (with the results file each comes from), and the figures and tables to place. Write your own text in place of the prompts, then delete the prompts.
- **The brief's eight required items** map to sections 1 to 7 plus the References.
- **Quote numbers as given here,** so the report matches the repository.
- **Ready-made material.** The tables, figure captions, formula, reference list and appendices are factual material from the code and results. Check them and keep, edit or drop them.

## Summary (about 150 to 200 words)

**Write:** one paragraph covering the problem, the input and output, the selected method, the headline result with its uncertainty, whether it meets your acceptance criteria, and the speed.

**Key facts:**

- Selected model: YOLOv8n, fine-tuned on 869 distinct training frames, run with ONNX Runtime on a laptop CPU at a 480 px input.
- Blocked test split, 174 frames (34 PASS): ROC AUC 0.887 (95% CI 0.810 to 0.947), 0.148 above logistic regression (paired 95% CI 0.068 to 0.235).
- At cost ratio r = 10, no model does materially better than re-checking every pack. At r = 1 the detector passes 56% of good frames (19 of 34) and lets 3.6% of defective frames through (5 of 140).
- 41 ms per frame end to end on the Apple M4 CPU.

## 1 Engineering problem (about 250 to 300 words)

**Brief requires:** the engineering context; the problem being addressed; why it matters; the intended use of the solution; the limitations of the selected scope.

**Rubric, excellent band:** the problem is precisely defined, its importance and intended use are explicit, and the scope and limitations are clearly justified.

**Key facts:**

- Tray-sealing lines run at about 120 packs per minute \[1\]; recent machines are rated up to 200 \[2\]. That leaves 300 to 500 ms per pack.
- Unsealed or damaged film risks spoilage and a recall; loose or twisted meat leads to customer complaints. Consistent visual inspection at two to three packs a second is hard for people over a shift.
- The Dataset 3 brief poses an open problem (a report on meat and packaging quality) and asks for a narrower, achievable one. Yours: decide from one overhead RGB frame whether the visible tray should PASS or FAIL packaging inspection, and show where any defect is.
- Intended use: decision support. Every FAIL frame goes to an operator with its defect boxes, and the operator makes the final call. The system never rejects product itself.
- Scope limits and why:
  - Per frame, not per tray: frames carry no tray identifier, and a tray stays in view for about 12 frames.
  - Packaging and placement only: the labels cover five defects (unsealed film, other packaging errors, loose meat, twisted meat, wrinkles) and nothing about meat quality.
  - One line, one camera, one lighting setup.
- FAIL means any of the five defects is present, all treated as equally serious, because the safe response to an unconfirmed defect is a human check.

**Tip:** a one-sentence problem statement set apart as a quote makes the scope easy to see.

## 2 Inputs, outputs and data (about 450 to 550 words plus the table)

**Brief requires:** the input data accepted; its format, units and domain; the outputs produced; the meaning and intended use of each output; how invalid, missing or unsupported inputs are handled. Later in the document, the inputs and outputs of the functions you developed (Appendix B covers this; point to it).

**Rubric, excellent band:** inputs and outputs fully specified, including format, units, domains, meaning and intended use.

### 2.1 Top-level inputs and outputs

Ready-made table (check and keep):

| Item | Format and units | Domain | Meaning and use |
| --- | --- | --- | --- |
| Input frame | JPEG, PNG or BMP; RGB, 8 bits per channel | 720 x 540 px overhead conveyor frame. Other sizes are letterboxed and flagged | One camera frame of the line |
| Defect score s | Dimensionless | 0 to 1 | Highest confidence among detected defect boxes; ranks how suspicious the frame is |
| Threshold τ | Dimensionless | 0 to 1, fixed from validation data for a cost ratio r | The frame FAILs when s ≥ τ |
| Status | Text | PASS, FAIL or ERROR | Operator action: let through, re-inspect, or check the camera or file |
| Defects | Class; confidence 0 to 1; box corners in px; area as a fraction of the frame | Five classes | What is wrong and where to look |
| Timing | Milliseconds | Positive | Load, preprocess, inference, postprocess and total time; checks the line budget |
| Warnings | Text | Wrong size, very dark (mean below 15/255), very bright (above 240/255), near-uniform | The input is unlike the training data |

**Key facts for the text around it:**

- Each frame produces a JSON report and an annotated PNG; a batch adds `summary.json` and an HTML index.
- Invalid input never stops a batch: a non-image or undecodable file gets status ERROR with the reason. Grey and RGBA images are converted to RGB.
- Exit codes: 0 all inspected, 2 if any frame was an ERROR, 1 for invalid arguments. The test `test_cli_inspect_mixed_inputs` checks this with good, corrupt and non-image files.
- Function-level inputs and outputs: Appendix B, plus the docstrings collected in `docs/api/`.

### 2.2 The data

**Key facts** (from `results/data_audit.json`):

- Roboflow "Pork Rasher Error(Packaging)" export, version 4, CC BY 4.0 \[3\]: 3,549 images from 1,272 distinct frames of one overhead conveyor video, with boxes for five defect classes.
- Already resized to 720 x 540 px by centre crop (tray edges can be cut off). Pixel values are normalised inside the pipeline.
- Each training frame has three copies with exposure shifted by up to ±10%.
- FAIL if a frame has any defect box, which makes PASS the minority: about 20% of images (684 of 3,349 training, 14 of 120 validation, 12 of 80 test).
- Twisted meat, loose meat and wrinkle each appear in fewer than 70 training images; wrinkle appears in no test frame.
- Defect sizes differ greatly: the median packaging-error box covers 0.38% of the frame (about 37 x 37 px), the median unsealed box 10.9%.

### 2.3 Leakage in the provided split, and the blocked split

**Key facts** (from `results/data_audit.json` and `splits/blocked_split.csv`):

- All 80 provided test frames have a training frame within 30 frame numbers, and 23 within one (Figure C1). Neighbouring frames show the same tray slightly moved, so testing on them partly tests on trays seen in training: leakage \[4\].
- Thumbnail correlation repeats every 11.9 frames, found from the FFT of the similarity-against-gap curve (Figure C2). That is the tray spacing, so frames 30 or more apart (about 2.5 trays) show different trays.
- Blocked split: whole blocks of 200 consecutive frame numbers go to train, validation or test \[5\]. Any frame within 30 frame numbers of a higher-priority split (test, then validation, then train) is dropped.
- The block assignment is the best of 500 random assignments, judged on the validation and test PASS rate and class mix only, never on model results.
- Result: 869 training frames (2,433 images with copies), 192 validation (37 PASS) and 174 test (34 PASS). Validation and test use one copy of each frame.

**Write:** explain in your own words why neighbouring frames inflate test scores.

## 3 Computational approach (about 600 to 700 words plus Figure 1)

**Brief requires:** how the method works, at an appropriate level; why it was selected; its important assumptions; its settings or parameters; how it was implemented; its known limitations.

**Key fact for the opening:** every model is reduced to one defect score s in \[0, 1\] per frame, and the frame FAILs when s ≥ τ, so all models are judged by the same rule.

&#91;image: Figure 1\]

*Figure 1. PorkSort architecture. Development (top) runs once; inspection (bottom) runs on every frame. The classical path is the baseline family the selected YOLOv8n detector was compared against.*

### 3.1 Classical models on handcrafted features

**Key facts:**

- Raw pixels are not usable: over a million values per frame and only 869 distinct training frames.
- Each frame is resized to 240 x 180 px and summarised by 1,439 features: HSV colour histogram (128), meat-pixel share, centroid and spread (5), glare on a 4 x 3 grid (12), Sobel edge strength on a 6 x 4 grid (24), local binary pattern texture (10) \[6\] and HOG (1,260) \[7\]. Edges and HOG rest on finite-difference image gradients.
- Classifiers: logistic regression, decision tree, RBF support vector machine and multilayer perceptron, the classification counterparts of unit topics 5.1 to 5.4 \[8\]; scikit-learn \[9\] and scikit-image \[10\].
- Standardisation for the logistic regression, SVM and MLP: without it, large-range features dominate the SVM's distances and make gradient descent badly conditioned. Classes weighted by inverse frequency.
- Hyperparameters by five-fold cross-validation on the training set, scored by ROC AUC, with folds grouped by 200-frame block so copies of a frame never sit on both sides. Chosen settings in Appendix D.

### 3.2 YOLOv8n detector

**Key facts:**

- Why: the smallest defects are about 37 px wide and global features miss them (Section 4). YOLOv8n \[11\], \[12\] is the smallest YOLOv8 model, fast enough for a CPU, its pretrained weights transfer to a small dataset, and its boxes explain each FAIL.
- How it works: a convolutional backbone gives feature maps at three scales; the head predicts a box and five class scores at each of 8,400 positions (640 px input). Boxes below a confidence threshold are dropped. Non-maximum suppression keeps the highest-scoring box and deletes same-class boxes overlapping it with IoU above 0.6.
- Training: COCO-pretrained, 640 px input with frames letterboxed to 640 x 480 plus grey padding, up to 60 epochs, early stopping after 20 without improvement, batch 16, seed 0, default augmentation, free Colab T4 GPU. Stopped at epoch 39 after 22 minutes, keeping the epoch-19 weights (best validation mAP).
- Implementation: exported to ONNX and run with ONNX Runtime on a CPU \[13\]. Letterboxing, decoding and NMS are in NumPy (`src/porksort/detector.py`), so inspection needs neither PyTorch nor a GPU. Deployed at a 480 px input (Section 5.3).

### 3.3 Choosing the threshold by cost

Ready-made formula:

```latex
C(\tau) = \frac{r \cdot \mathrm{FN}(\tau) + \mathrm{FP}(\tau)}{N}
```

**Key facts:**

- FN = defective frames passed, FP = good frames failed, N = all frames. A shipped defect risks a recall; a rejected good pack costs one re-check.
- r = 10: a shipped defect counts as ten re-checks (Section 5.4 tests this).
- Candidates are the midpoints between consecutive distinct scores, so the search finds the exact minimum on the data. Ties go to the lower threshold, because a false PASS costs more.
- Assumptions: annotations correct and complete for what is visible; new frames from the same line, camera and lighting; the five classes equally serious; each frame judged on its own.

**Write:** why 0.5 is not a meaningful threshold here, in your own words.

## 4 Validation and verification (about 800 to 900 words plus Tables 1 and 2, Figures 2 and 3)

**Brief requires:** what was validated; why the method is appropriate; which metrics or acceptance criteria were used; whether validation influenced model selection; what the validation cannot establish.

**Rubric, excellent band:** comparison with ground truth, withheld data or benchmarks; identification of unsupported inputs and failure modes; honest limitations.

### 4.1 What was validated and verified

**Key facts** (from `results/yolo_verification.json` and `tests/`):

- Validation: each of the 174 withheld blocked-test frames' PASS/FAIL decision compared with its label, once, after every choice was fixed.
- Verification: 52 automated tests (Appendix E). Four compare with answers known in advance:
  - the decoder recovers a hand-encoded box to within 1e-5;
  - the FLOP formula matches a direct loop count;
  - the threshold search matches a 20,001-point brute-force search;
  - the optimised histogram is bit-identical to NumPy's `histogramdd`.
- Benchmark against Ultralytics' own predictions on 60 test frames: 34 of the 41 boxes it found at confidence ≥ 0.25 matched at IoU ≥ 0.9 (median 0.94); frame scores differed by 0.04 on average.
- Two experiments placed the difference in the input, not the decoding: OpenCV resizing changed the mean difference only from 0.0392 to 0.0389, while moving the padding from centred to top-aligned shifted scores by 0.024. Ultralytics feeds a 640 x 480 rectangle; the exported model needs a padded square.

**Write:** why validation and verification are different questions.

### 4.2 Why this method is appropriate, and the acceptance criteria

**Key facts:**

- Withheld data is the only direct test on unseen frames; the blocked split keeps every test tray out of training.
- FAIL is the positive class. Primary metric: escape rate FN / (TP + FN). Secondary: false reject rate FP / (FP + TN). ROC AUC (area under the ROC curve, trapezoidal rule) measures ranking across thresholds.
- Accuracy not used: always answering FAIL scores 80.5%.
- Wilson score 95% intervals \[14\] for rates (valid at 0% and 100%, unlike the normal approximation); AUC intervals from 2,000 bootstrap resamples.
- Acceptance criteria, set before any detector result: escape rate with an upper 95% bound of at most 5%, and a false reject rate of at most 30%.
- Influence on selection: grouped cross-validation chose the classical hyperparameters; the validation set chose the threshold, input size and final model; the test set was used once.

### 4.3 Results

*Table 1. Blocked test set (174 frames, 34 PASS), thresholds chosen on validation at r = 10. Intervals are 95%.*

| Model | ROC AUC (95% CI) | Threshold | Escape rate | False reject rate | Cost per frame |
| --- | --- | --- | --- | --- | --- |
| Always FAIL (baseline) | 0.500 | 0 | 0.000 | 1.000 | 0.195 |
| Colour-histogram logistic regression | 0.739 (0.652 to 0.822) | 0.205 | 0.000 | 0.941 | 0.184 |
| Logistic regression | 0.739 (0.649 to 0.823) | 0 | 0.000 | 1.000 | 0.195 |
| Decision tree | 0.717 (0.621 to 0.807) | 0.147 | 0.029 | 0.971 | 0.420 |
| SVM (RBF) | 0.724 (0.633 to 0.806) | 0 | 0.000 | 1.000 | 0.195 |
| MLP | 0.717 (0.618 to 0.805) | 0 | 0.000 | 1.000 | 0.195 |
| YOLOv8n, 640 px input | 0.859 (0.773 to 0.926) | 0.004 | 0.000 | 1.000 | 0.195 |
| **YOLOv8n, 480 px input (selected)** | **0.887 (0.810 to 0.947)** | 0.007 | 0.000 | 1.000 | 0.195 |

&#91;image: Figure 2\]

*Figure 2. ROC curves on the blocked test set. The selected detector (blue) dominates every classical model (grey).*

*Table 2. Selected detector: operating points chosen on validation, evaluated on test.*

| Cost ratio r | Threshold | Escape rate (95% CI) | False reject rate (95% CI) | Cost per frame (re-check all: 0.195) |
| --- | --- | --- | --- | --- |
| 1 | 0.128 | 3.6% (1.5% to 8.1%) | 44% (29% to 61%) | 0.115 |
| 2 to 50 | 0.007 | 0% | 100% | 0.195 |

&#91;image: Figure 3\]

*Figure 3. Blocked test frames at the r = 1 threshold. Dashed white boxes are labels; solid boxes are detections.*

**Key facts** (from `results/comparison_details.json` and `results/paired_auc_selected_vs_logreg.json`):

- Ranking: the detector's AUC beats logistic regression's by 0.148 on the same frames (paired bootstrap 95% CI 0.068 to 0.235), so the gain is not sampling noise.
- Per class, scored against the PASS frames:
  - detector: 0.95 unsealed, 0.84 packaging error, 0.71 loose meat (6 frames), 0.97 twisted meat (4 frames);
  - logistic regression: 0.86, 0.67, 0.51 and 0.49, so global features see the large defect but not the small ones.
- At r = 10: with 80% of frames defective, passing a good frame saves one re-check while each escape costs ten, and the lowest-scoring validation frames still include defective ones (five wrinkle frames; wrinkle has only three training frames). The detector's threshold fails every frame. The best any model does is the colour-histogram model, passing 2 of 34 good frames. No model meets the acceptance criteria.
- At r = 1: 19 of 34 good frames passed, 5 of 140 defective frames escaped, cost per frame cut by 41%.
- Failure modes: most escapes are small packaging errors at tray corners and loose pieces at the frame edge. Several false rejects are trays whose film looks lifted but carries no label (possible label ambiguity).
- Effect of the split: on the leaky Roboflow split, logistic regression scores 0.808 and the SVM 0.826, against 0.739 and 0.724 on the blocked split, an overstatement of 0.07 to 0.10. No second detector was trained on the Roboflow split.

### 4.4 What the validation cannot establish

**Key facts:**

- Performance on another line, camera, lighting setup or product.
- Behaviour per tray: a defect in one frame may be out of view in the next.
- Recall for rare classes: wrinkle has no test frames; loose and twisted meat have four to six.
- The effect of label errors.
- Whether r = 10 matches the plant's real costs.

## 5 Alternatives, optimisation and performance (about 750 to 850 words plus Table 3 and Figure 4)

**Brief requires:** credible alternatives compared (accuracy, robustness, time, memory, complexity, interpretability, scalability, suitability); profilers, timing, arithmetic intensity and FLOP estimates to explain performance; a comparison of the baseline with the improved or selected solution, and a justification of the final design.

**Rubric, excellent band:** clear baseline, justified selection, quantitative analysis, evidence-supported improvements, thoughtful trade-offs, conclusions grounded in results.

### 5.1 Comparison of alternatives

*Table 3. Alternatives compared on the brief's criteria. Times are per frame on the M4 laptop CPU unless marked.*

| Model | Blocked test AUC | Time per frame | Model size | Interpretability | Localises defects |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.739 | 2.8 ms features (optimised) + 0.25 ms (container) | about 35 kB | High: one weight per feature | No |
| Decision tree (depth 5) | 0.717 | features + 0.11 ms (container) | under 10 kB | High: readable rules | No |
| SVM (RBF) | 0.724 | features + 2.0 ms (container) | 1,606 support vectors | Low | No |
| MLP (64 hidden units) | 0.717 | features + 0.31 ms (container) | about 0.75 MB | Low | No |
| **YOLOv8n, 480 px** | **0.887** | 36.7 ms inference, 41 ms end to end | 12.7 MB (fp32), 6.4 MB (fp16) | Medium: boxes show why | Yes |

**Write:** justify the choice of YOLOv8n at 480 px: highest AUC with a paired interval excluding zero, the only model that localises defects, fast enough. Name its costs (larger model, GPU training, less transparent), and say it is the best available, not one that meets the criteria.

### 5.2 Profiling and optimising the classical pipeline

**Key facts** (from `results/profile_features_mac_vm.txt` and `results/performance_mac_vm.json`):

- line\_profiler on the M4: RGB-to-HSV conversion 36% of `features.extract`, colour histogram 14%, texture 13%, HOG 11%, resizing 10%.
- Change 1: PIL's 8-bit HSV conversion (in C) instead of scikit-image's float64 one. Within 1/255, far finer than the 1/8 and 1/4 bins; tested.
- Change 2: bin index computed directly and counted with `bincount`; bit-identical to `histogramdd`; tested.
- Change 3: ablation. Dropping colour cut validation AUC by 0.026 and dropping HOG by 0.036; each other group changed it by under 0.003, so those four groups were removed.
- Change 4: resizing is 2D interpolation. Across three sizes and four methods, box (area-averaging) filtering was best at every size: nearest-neighbour aliases the belt's fine weave into noise (Figure C4). Box filtering at 120 x 90 px was chosen.
- Overall: 13.9 to 9.0 to 4.9 to 2.8 ms per frame (4.9 times faster); validation AUC 0.718 to 0.758, test AUC 0.739 to 0.748.

### 5.3 YOLOv8n: FLOPs, roofline, precision and input size

&#91;image: Figure 4\]

*Figure 4. Roofline for the M4 and the T4. Ticks mark individual convolution layers; the red point is the measured whole-network performance on the M4.*

**Key facts** (from `results/yolo_layer_profile.json` and `results/performance_mac_vm.json`):

- FLOPs per convolution: 2 x C\_out x H\_out x W\_out x (C\_in / groups) x k². Total 8.64 GFLOPs per 640 px frame, within 5% of Ultralytics' 8.2.
- Bytes moved if each layer reads weights and input and writes output once: 153 MB per frame. Arithmetic intensity 57 FLOP/byte (layers range 0.5 to 174).
- Roofline \[15\]:
  - M4 peak from the unit's formula: 4 cores x 4.4 GHz x 4 lanes x 2 (FMA) x 4 pipes ≈ 563 GFLOP/s \[16\]; four pipes is an estimate. With about 120 GB/s, the machine balance is 4.7 FLOP/byte.
  - T4: 8.1 TFLOP/s and 320 GB/s, a balance of about 25 \[17\].
  - So the network is compute-bound on both. Measured 140 GFLOP/s on the M4, about a quarter of the roof. Likely reasons: the virtual machine's cores may not all be performance cores, small layers use the vector units poorly, and the count omits activations, concatenation and decoding.
- Half precision: file 12.8 to 6.4 MB; 640 px inference 61.9 to 42.9 ms. The largest score change over 60 frames was 0.0008, consistent with fp16 unit roundoff 2^-11 ≈ 0.0005; no decision changed. fp16 was exported only at 640 px, so the deployed 480 px model is fp32.
- Threads: 207 ms on one thread, 62 ms on four (3.3 times faster).
- Input size (Figure C5): 480 px was chosen on validation, the same AUC as 640 px within noise at 59% of the time. Its higher test AUC (0.887 against 0.859) is not a claimed gain. 800 px did not help the small defects, which points to training at a higher resolution.

| Input size | 320 px | 480 px | 640 px | 800 px |
| --- | --- | --- | --- | --- |
| Validation AUC | 0.848 | 0.875 | 0.870 | 0.851 |
| Inference time per frame (M4) | 17.7 ms | 36.7 ms | 61.9 ms | 98.5 ms |

### 5.4 Line budget, sensitivity and repeatability

**Key facts:**

- One frame per pack: 300 to 500 ms available; 41 ms is 7 to 12 times inside.
- Every frame (11.9 per tray): 25 to 42 ms available. 480 px fits only at 120 packs per minute; 320 px (21 ms) fits both speeds for 0.027 less validation AUC.
- Cost ratio swept from 1 to 50 (Figure C6): the detector's operating point changes only between r = 1 and r = 2; the classical thresholds do not move, so their weakness is ranking, not the choice of r.
- Repeatability: MLP test AUC over five seeds 0.716 ± 0.008, converging in 50 to 70 iterations under early stopping. YOLOv8n was trained once.

**Write:** one paragraph comparing the baseline with the selected solution and justifying the final design.

## 6 Limitations and lessons learned (about 250 to 300 words)

**Brief requires:** the limitations and the lessons learned.

**Key facts, limitations:**

- Does not meet the acceptance criteria: at r = 10 the detector is a prioritisation aid with boxes, not an automatic gate.
- Per-frame decisions: a frame can be labelled PASS while the defect is just out of view; a deployed system should track trays.
- Rare classes: wrinkle has three training frames and no test frames.
- One recording with an unknown frame rate; 32 frame names carry two different images, suggesting two recording sessions.
- One YOLO training run, so its run-to-run variation is unknown.

**Key facts, lessons to choose from:** the split decides what is measured; ranking is not deciding; match the method to the scale of the defect; profile before optimising.

**Key facts, next steps:** train at about 960 px for small packaging errors; more rare-class data or one merged placement class; track trays across frames; several seeds; measure the plant's real costs.

**Write:** pick the lessons that genuinely changed how you think, and say why in your own words.

## 7 AI use and reflection (about 350 to 400 words)

**Brief requires (each point):** which AI tools were used; what for; the approximate level of AI contribution; why AI was used for those tasks; how AI-generated material was checked; which important decisions remained yours; errors, limitations or unhelpful suggestions; how AI use affected your understanding or workflow.

**Rubric, excellent band:** fully disclosed, critically evaluated, clearly subordinate to your own work, with you clearly driving the process.

**Facts you can draw on:**

- Tools: Claude (Anthropic) through the Claude app, linked to your laptop and a cloud workspace; Google Colab for the training GPU.
- What the AI did: wrote most of the code (package, scripts, tests, Colab notebook), ran the data audit and analyses, made the figures, produced a first full draft of the report, then this scaffold and the appendix tables.
- What you did: chose the dataset and the per-frame PASS/FAIL goal with defect locations; required methods from the unit outline and running without a GPU; chose Colab and ran the training, reporting its errors back; checked the draft against the brief and found it over the page limit; wrote this report from the scaffold. \[Add anything else you did.\]
- Proposed by the AI and reviewed and accepted by you: the blocked split, the cost-based threshold, r = 10 and the acceptance criteria.
- Checks: 52 tests (four against known answers); the Ultralytics comparison; inspecting frames and every figure; tracing numbers to `results/`.
- AI errors caught:
  - an uninformative ±2 standard deviation band (replaced by an FFT);
  - a Colab notebook that failed because the folder name hid the package;
  - an out-of-date code copy, caught by file sizes;
  - two bugs in its own tests;
  - an assumed 30 fps frame rate;
  - an overstated r = 10 claim;
  - a first draft over the page limit and missing function-level inputs and outputs.

**Write:** your own estimate of the AI's contribution, and honestly how it affected your understanding and workflow.

## References

**Reference bank.** These are the sources behind the key facts. Cite the ones you use, number them in order of first citation (IEEE style), and delete any you do not cite.

1. "Proseal launch for GT2X Twin E-seal," *BakeryandSnacks.com*, Jul. 2012. \[Online\]. Available: [bakeryandsnacks.com](https://www.bakeryandsnacks.com/Article/2012/07/27/Proseal-launch-for-GT2X-Twin-E-seal/)
2. "Harpak ULMA launches high speed grind line tray sealing application powered by Mondini Sinfonia," *Business Wire*, Jan. 2026. \[Online\]. Available: [businesswire.com](https://www.businesswire.com/news/home/20260115931807/en/Harpak-ULMA-Launches-High-Speed-Grind-Line-Tray-Sealing-Application-Powered-by-Mondini-Sinfonia)
3. "Pork Rasher Error(Packaging)," version 4, Roboflow Universe, workspace hello-2aqe0, licensed CC BY 4.0, 2025. \[Online\]. Available: [universe.roboflow.com](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging)
4. S. Kaufman, S. Rosset, C. Perlich and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, 2012.
5. D. R. Roberts *et al.*, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913-929, 2017.
6. T. Ojala, M. Pietikäinen and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 24, no. 7, pp. 971-987, 2002.
7. N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. IEEE CVPR*, 2005, pp. 886-893.
8. MMA3001 teaching team, *MMA3001 Numerical Methods and Machine Learning unit notes*, Weeks 2.1, 2.2, 3.5 and 5.1 to 5.6, Monash University, 2026.
9. F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.
10. S. van der Walt *et al.*, "scikit-image: Image processing in Python," *PeerJ*, vol. 2, e453, 2014.
11. J. Redmon, S. Divvala, R. Girshick and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE CVPR*, 2016, pp. 779-788.
12. G. Jocher, A. Chaurasia and J. Qiu, "Ultralytics YOLOv8," 2023. \[Online\]. Available: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
13. ONNX Runtime developers, "ONNX Runtime," Microsoft. \[Online\]. Available: [onnxruntime.ai](https://onnxruntime.ai)
14. E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209-212, 1927.
15. S. Williams, A. Waterman and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," *Commun. ACM*, vol. 52, no. 4, pp. 65-76, 2009.
16. "Apple Silicon: M4 chip specs," *Low End Mac*, 2025. \[Online\]. Available: [lowendmac.com](https://lowendmac.com/2025/apple-silicon-m4-chip-specs/)
17. NVIDIA, "NVIDIA T4 Tensor Core GPU." \[Online\]. Available: [nvidia.com](https://www.nvidia.com/en-au/data-center/tesla-t4/)

**Appendices** (not counted in the page limit). Appendices A to F are factual reference material generated from the code and results. Check each one, then keep, edit or drop it. Update Appendix E's test count if you add tests.

## Appendix A: Repository guide

Repository: github.com/hashim-edu/mma3001-project-pork\_sort. The README explains the purpose, inputs and outputs, installation, and how to reproduce every result: install with `pip install -r requirements.txt` then `pip install -e .`, unzip the Roboflow export to `data/pork_v4/`, and run `01_data_audit.py`, `02_classical_models.py`, the Colab notebook `03_train_yolo_colab.ipynb`, `04_evaluate_and_compare.py`, `05_performance.py` and `06_figures.py` in order. The inspection tool runs with `python -m porksort inspect <frames> --out reports/`; `--cost-ratio 1` selects the threshold chosen on validation for r = 1. Every public function has a Google-style docstring giving its arguments, return value and errors (collected in the HTML documentation), and the code is formatted and linted with ruff.

| Folder | Contents |
| --- | --- |
| `src/porksort/` | Package: dataset, splits, features, models, decision, evaluate, detector, perf, report, cli |
| `scripts/` | Numbered analysis scripts that write to `results/` |
| `notebooks/` | Colab training notebook |
| `splits/` | The committed blocked split, so every machine uses the same frames |
| `results/` | Metrics (JSON, CSV), profiles and figures used in this report |
| `tests/` | pytest suite on synthetic data; needs no dataset |
| `docs/` | HTML API documentation (pdoc, Google-style docstrings) and the pytest HTML test report |

## Appendix B: Inputs and outputs of the main functions

Boxes are `Box` records: class name, centre x and y, width and height as fractions of the image in \[0, 1\], and confidence in \[0, 1\]. Full docstrings are in `docs/api/`.

| Function | Inputs | Output | Invalid input |
| --- | --- | --- | --- |
| `dataset.parse_label_file(path)` | YOLO label file; one line per box: class id, cx, cy, w, h in \[0, 1\] | List of `Box` | `FileNotFoundError`; `ValueError` naming the file and line for a wrong field count, unknown class or coordinate outside \[0, 1\] |
| `dataset.build_index(root)` | Export folder with train, valid and test images and labels | DataFrame, one row per image: path, provided split, source frame, frame number, boxes, FAIL label | `FileNotFoundError` if the folders or a label file are missing |
| `splits.blocked_split(index, cfg)` | Image index; `BlockedSplitConfig` (block 200 frames, gap 30, validation and test fractions 0.15, 500 trials, seed 0) | Each source frame assigned to train, val, test or buffer | `ValueError` if the fractions leave no training set |
| `features.load_rgb(path)` | Image file: .jpg, .jpeg, .png or .bmp | uint8 array (H, W, 3); grey and RGBA converted to RGB | `FileNotFoundError`; `ValueError` if unsupported or undecodable |
| `features.extract(rgb, cfg)` | uint8 RGB array of any size; `FeatureConfig` (working size, interpolation, feature groups, HSV method) | float vector of fixed length for a config: 1,439 for all groups, 1,388 for colour and HOG | `ValueError` if not (H, W, 3) |
| `models.make_model(name)` | `majority`, `colour_lr`, `logreg`, `tree`, `svm` or `mlp` | Untrained scikit-learn pipeline and its hyperparameter grid | `ValueError` for an unknown name |
| `models.defect_score(model, X)` | Fitted pipeline; feature matrix (n, d) | n defect scores in \[0, 1\] |  |
| `detector.letterbox(rgb, size, fill)` | uint8 RGB frame; square side in px; padding grey level (114) | Padded (size, size, 3) array, scale r, padding (dx, dy) in px |  |
| `detector.decode(output, r, pad, img_wh, conf_thr, iou_thr)` | Raw network output (1, 9, anchors): cx, cy, w, h in letterboxed px and five class scores in \[0, 1\]; r and pad from `letterbox`; frame size; thresholds | Up to 300 `Box` in normalised frame coordinates, after per-class NMS |  |
| `detector.nms(boxes, scores, iou_thr)` | Boxes (N, 4) as corner coordinates; scores (N,); IoU threshold | Indices of kept boxes, highest score first |  |
| `detector.OnnxYoloDetector(path, conf, iou, threads).detect(rgb)` | ONNX model file; uint8 RGB frame | List of `Box`; per-stage times in ms in `last_timing` | `FileNotFoundError` if the model is missing |
| `decision.decide(boxes, threshold)` | Detected boxes; τ in \[0, 1\] | `Decision`: FAIL if the frame score, the highest defect confidence (0 if none), is at least τ; the score; the defects | `ValueError` if τ is outside \[0, 1\] |
| `decision.choose_threshold(y_fail, scores, cost_ratio)` | Labels (True = defective), scores in \[0, 1\], cost ratio r | (τ, minimum expected cost per frame), ties to the lower τ |  |
| `evaluate.wilson_ci(k, n)` | Count k of n trials | 95% interval (lower, upper) in \[0, 1\] |  |
| `evaluate.passfail_metrics(y_fail, scores, threshold, cost_ratio)` | Labels, scores, τ, r | Dict: confusion counts, escape and false reject rates with intervals, expected cost, ROC AUC |  |
| `evaluate.bootstrap_auc(y_fail, scores, n_boot, seed)` | Labels, scores, resamples (2,000), seed | Percentile 95% interval for ROC AUC |  |
| `perf.conv2d_flops(...)` | Input and output channels, kernel size, output height and width, groups | FLOPs of one convolution (integer) |  |
| `perf.onnx_conv_profile(path)` | ONNX model file | Per convolution: shape, FLOPs, bytes moved, arithmetic intensity (FLOP/byte) |  |
| `report.input_warnings(rgb)` | uint8 RGB frame | Warnings for wrong size, very dark, very bright or near-uniform frames |  |
| `cli.resolve_threshold(arg, weights, cost_ratio)` | `auto` or a number; model file; r | τ, read from `threshold.json` for that model and r | `ValueError` if outside \[0, 1\] or none stored; `FileNotFoundError` if the file is missing |
| `cli.main(argv)` | `inspect <files or folders> [--weights] [--threshold] [--cost-ratio] [--out]` | Per-frame JSON and PNG, batch summary and HTML index; exit code 0, 1 (bad arguments) or 2 (an input was an ERROR) | Bad files become ERROR reports and the batch continues |

## Appendix C: Additional figures

&#91;image: Figure C1\]

*Figure C1. Gap from each test frame to its nearest training frame. Every provided test frame sits within 30 frames of a training frame; the blocked split removes this.*

&#91;image: Figure C2\]

*Figure C2. Correlation of frame thumbnails against frame gap. The curve repeats every 11.9 frames, the spacing between trays; the 30-frame buffer spans about 2.5 trays.*

&#91;image: Figure C3\]

*Figure C3. Share of frames containing each label in the provided splits. PASS is a minority and three classes are rare.*

&#91;image: Figure C4\]

*Figure C4. Classical pipeline: validation AUC (left) and feature time (right) against working width and interpolation method. Box filtering is best at every size; time grows with pixel count.*

&#91;image: Figure C5\]

*Figure C5. YOLOv8n trained at 640 px and run at other input sizes: accuracy barely changes, while time grows with pixel count.*

&#91;image: Figure C6\]

*Figure C6. Selected detector: test escape rate and false reject rate at the threshold chosen on validation for each cost ratio. Only at r = 1 does the threshold pass any frames.*

## Appendix D: Chosen settings

| Model | Settings chosen by grouped cross-validation |
| --- | --- |
| Colour-histogram logistic regression | C = 0.01, balanced class weights |
| Logistic regression | C = 0.001, balanced class weights |
| Decision tree | maximum depth 5, at least 20 samples per leaf (18 leaves) |
| SVM | RBF kernel, C = 3, gamma = 1e-4 (1,606 support vectors) |
| MLP | one hidden layer of 64 units, L2 penalty alpha = 1.0, early stopping |
| YOLOv8n | 640 px input, up to 60 epochs, patience 20, batch 16, seed 0, COCO-pretrained; stopped at epoch 39 with the weights from epoch 19; deployed at a 480 px input |
| Decision threshold | chosen on validation for each cost ratio; r = 10 by default |

## Appendix E: Verification tests

| Test file | Tests | What they establish |
| --- | --- | --- |
| `test_dataset.py` | 10 | Label files are parsed and malformed ones rejected with the file and line; augmented copies group to one source frame |
| `test_splits.py` | 6 | Whole blocks stay together, no test frame lies within the gap of a training frame, evaluation uses one copy per frame, the split is reproducible |
| `test_decision_evaluate.py` | 8 | The PASS/FAIL rule; the threshold search equals a brute-force search and never rises as the cost ratio rises; Wilson intervals match known values |
| `test_features_detector.py` | 13 | Feature vectors have a fixed length; letterbox, IoU and NMS give known answers; the decoder recovers a hand-encoded box; the FLOP formula matches a loop count; the optimised histogram and HSV conversion match the references |
| `test_onnx_pipeline.py` | 4 | A small synthetic ONNX network runs end to end in fp32 and fp16, and the graph FLOP profiler matches the formula |
| `test_report_cli.py` | 5 | Reports are written; corrupt, non-image and invalid inputs become ERROR reports without stopping a batch; thresholds are looked up by model and cost ratio |
| Doctests in `src/` | 6 | Worked examples in the docstrings |

## Appendix F: Environment

Development and classical models: Python 3.13, scikit-learn 1.9.1, scikit-image 0.26.0, NumPy 2.5.3, pandas 3.0.5, ONNX Runtime 1.29.0, Pillow 12.3.0. Timing on the target laptop: Apple MacBook with M4 chip, in a 4-core Linux virtual machine with Python 3.10. Training: Google Colab with an NVIDIA T4 GPU, Ultralytics 8.4.185, PyTorch 2.11.0 (CUDA 13.0), Python 3.13.
