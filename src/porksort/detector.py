"""YOLOv8 defect detector running on ONNX Runtime (CPU).

Training happens on Colab with Ultralytics (``notebooks/03_train_yolo_colab.ipynb``),
which exports the trained network to ONNX. Inference here uses only
``onnxruntime`` and NumPy, so the inspection tool runs on a laptop without
PyTorch or a GPU.

Pipeline for one frame::

    RGB (H, W, 3) uint8
      -> letterbox: scale by r = min(S/W, S/H), pad to S x S with grey 114
      -> float32 in [0, 1], CHW, batch of 1           (normalisation)
      -> network -> (1, 4 + n_classes, n_anchors)      (8,400 anchors at S = 640)
      -> keep anchors whose best class score >= conf
      -> per-class non-maximum suppression at IoU >= iou
      -> undo the letterbox, convert to normalised (xc, yc, w, h)

The output tensor layout and the letterbox follow Ultralytics YOLOv8. Our
output is checked against Ultralytics' own predictions in
``scripts/04_evaluate_and_compare.py`` (verification against an established
implementation).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES
from .dataset import Box


def letterbox(rgb: np.ndarray, size: int = 640, fill: int = 114) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio, then pad to a square.

    Args:
        rgb: (H, W, 3) uint8 image.
        size: Side length S of the square network input.
        fill: Grey level used for padding.

    Returns:
        ``(padded, r, (dx, dy))`` where ``padded`` is (S, S, 3) uint8, ``r``
        the scale factor and ``(dx, dy)`` the left and top padding in pixels.
    """
    h, w = rgb.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)) if (nw, nh) != (w, h) else rgb
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (float(left), float(top))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between each box in ``a`` (N, 4) and each in ``b`` (M, 4), xyxy format."""
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy non-maximum suppression.

    Repeatedly keeps the highest-scoring box and removes every remaining box
    that overlaps it with IoU above ``iou_thr``.

    Returns:
        Indices of kept boxes, highest score first.
    """
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=int)


def decode(output: np.ndarray, r: float, pad: tuple[float, float], img_wh: tuple[int, int],
           conf_thr: float, iou_thr: float, class_names=CLASS_NAMES, max_det: int = 300) -> list[Box]:
    """Turn the raw network output into boxes in normalised image coordinates.

    Args:
        output: (1, 4 + n_classes, n_anchors) array; rows are cx, cy, w, h in
            letterboxed pixels followed by per-class scores in [0, 1].
        r, pad: Scale and padding returned by `letterbox`.
        img_wh: Original image (width, height).
        conf_thr: Minimum class score to keep an anchor.
        iou_thr: IoU above which overlapping boxes of the same class are merged.
        class_names: Class names in id order.
        max_det: Maximum boxes returned.
    """
    pred = np.asarray(output, dtype=np.float32)[0].T  # (n_anchors, 4 + nc)
    nc = len(class_names)
    if pred.shape[1] != 4 + nc:
        raise ValueError(f"output has {pred.shape[1] - 4} classes, expected {nc}")
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(1)
    conf = cls_scores[np.arange(len(cls)), cls]
    m = conf >= conf_thr
    if not m.any():
        return []
    xywh, cls, conf = pred[m, :4].astype(np.float64), cls[m], conf[m].astype(np.float64)
    xyxy = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                            xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2])
    # Per-class NMS by shifting each class's boxes far apart (boxes of
    # different classes then never overlap).
    offset = cls[:, None] * 10_000.0
    keep = nms(xyxy + offset, conf, iou_thr)[:max_det]
    W, H = img_wh
    dx, dy = pad
    out = []
    for k in keep:
        x0, y0, x1, y1 = xyxy[k]
        x0, x1 = np.clip([(x0 - dx) / r, (x1 - dx) / r], 0, W)
        y0, y1 = np.clip([(y0 - dy) / r, (y1 - dy) / r], 0, H)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(Box(class_names[int(cls[k])], float((x0 + x1) / 2 / W), float((y0 + y1) / 2 / H),
                       float((x1 - x0) / W), float((y1 - y0) / H), float(conf[k])))
    return out


class OnnxYoloDetector:
    """YOLOv8 detector backed by an exported ONNX model.

    Args:
        onnx_path: Path to ``best_fp32.onnx`` or ``best_fp16.onnx``.
        conf: Minimum confidence for a box to be returned. Keep this low
            (0.001 to 0.05) and apply the pass/fail threshold afterwards.
        iou: NMS IoU threshold (0.6 matches the Colab evaluation).
        threads: ONNX Runtime intra-op threads (None = library default).

    Raises:
        FileNotFoundError: If the model file is missing.
    """

    def __init__(self, onnx_path: str | Path, conf: float = 0.01, iou: float = 0.6, threads: int | None = None):
        import onnxruntime as ort

        p = Path(onnx_path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; train on Colab and copy the weights into results/yolo/")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        self.conf, self.iou = conf, iou
        self.path = p
        self.last_timing: dict[str, float] = {}

    def preprocess(self, rgb: np.ndarray):
        """Letterbox and normalise to a (1, 3, S, S) tensor in [0, 1]."""
        lb, r, pad = letterbox(rgb, self.size)
        x = (lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).astype(self.dtype)
        return x, r, pad

    def raw(self, x: np.ndarray) -> np.ndarray:
        """Run the network on a preprocessed tensor."""
        return self.session.run(None, {self.input_name: x})[0]

    def detect(self, rgb: np.ndarray) -> list[Box]:
        """Detect defects in one RGB frame. Timing is stored in `last_timing` (ms)."""
        t0 = time.perf_counter()
        x, r, pad = self.preprocess(rgb)
        t1 = time.perf_counter()
        y = self.raw(x)
        t2 = time.perf_counter()
        boxes = decode(y, r, pad, (rgb.shape[1], rgb.shape[0]), self.conf, self.iou)
        t3 = time.perf_counter()
        self.last_timing = {"preprocess": 1e3 * (t1 - t0), "inference": 1e3 * (t2 - t1), "postprocess": 1e3 * (t3 - t2)}
        return boxes
