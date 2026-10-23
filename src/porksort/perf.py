"""Performance analysis: timing, FLOP counts, arithmetic intensity and the roofline.

The unit's roofline model says a kernel's attainable speed is

    attainable FLOP/s = min(peak FLOP/s, arithmetic intensity x memory bandwidth)

where arithmetic intensity (AI) = FLOPs performed / bytes moved to and from
memory. A low-AI kernel is memory-bound (limited by bandwidth), a high-AI
kernel is compute-bound (limited by the arithmetic units).

For a convolution layer with ``C_in`` input channels, ``C_out`` output
channels, a ``k_h x k_w`` kernel, ``groups`` groups and an output of
``H_out x W_out``, each output value needs ``(C_in / groups) * k_h * k_w``
multiply-adds, and one multiply-add is 2 FLOPs::

    FLOPs = 2 * C_out * H_out * W_out * (C_in / groups) * k_h * k_w

Bytes moved are estimated as weights + input activations + output
activations, each read or written once at the tensor's element size. Real
hardware reuses data in cache, so this is an upper bound on traffic and a
lower bound on AI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np


def time_call(fn: Callable[[], object], repeats: int = 50, warmup: int = 5) -> dict[str, float]:
    """Time a zero-argument callable.

    Returns:
        ``median``, ``p10``, ``p90`` and ``mean`` wall time in milliseconds.
        The median is the headline figure because it ignores one-off stalls.
    """
    for _ in range(warmup):
        fn()
    ts = np.empty(repeats)
    for i in range(repeats):
        t = time.perf_counter()
        fn()
        ts[i] = time.perf_counter() - t
    ts *= 1e3
    return {"median": float(np.median(ts)), "p10": float(np.percentile(ts, 10)),
            "p90": float(np.percentile(ts, 90)), "mean": float(ts.mean()), "repeats": repeats}


def conv2d_flops(c_in: int, c_out: int, k_h: int, k_w: int, h_out: int, w_out: int,
                 groups: int = 1, bias: bool = True) -> int:
    """FLOPs of one 2D convolution (2 per multiply-add, plus 1 per output for bias)."""
    if c_in % groups or c_out % groups:
        raise ValueError("channels must be divisible by groups")
    macs = c_out * h_out * w_out * (c_in // groups) * k_h * k_w
    return 2 * macs + (c_out * h_out * w_out if bias else 0)


def dense_flops(n_in: int, n_out: int, bias: bool = True) -> int:
    """FLOPs of a fully connected layer."""
    return 2 * n_in * n_out + (n_out if bias else 0)


def mlp_flops(layer_sizes: list[int]) -> int:
    """FLOPs of an MLP forward pass for one sample (activations ignored)."""
    return sum(dense_flops(a, b) for a, b in zip(layer_sizes[:-1], layer_sizes[1:]))


def svm_rbf_flops(n_support: int, n_features: int) -> int:
    """Approximate FLOPs of one RBF-SVM prediction.

    Each support vector needs a squared distance (3 FLOPs per feature:
    subtract, square, add) and one multiply-add for the dual coefficient. The
    ``exp`` is counted as one operation although it costs more in practice.
    """
    return n_support * (3 * n_features + 3)


def onnx_conv_profile(onnx_path: str | Path, bytes_per_element: int | None = None) -> list[dict]:
    """Per-layer FLOPs and bytes for every Conv node in an ONNX model.

    Uses ONNX shape inference to find each layer's output size.

    Args:
        onnx_path: Exported model.
        bytes_per_element: Override element size (default: 2 for fp16 models, else 4).

    Returns:
        List of dicts with ``name``, ``c_in``, ``c_out``, ``k``, ``groups``,
        ``h_out``, ``w_out``, ``flops``, ``bytes`` and ``ai`` (FLOPs per byte).
    """
    import onnx
    from onnx import numpy_helper, shape_inference

    model = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
        elem = vi.type.tensor_type.elem_type
        if bytes_per_element is None and elem == onnx.TensorProto.FLOAT16:
            bytes_per_element = 2
    bpe = bytes_per_element or 4
    rows = []
    for node in g.node:
        if node.op_type != "Conv":
            continue
        w = inits.get(node.input[1])
        if w is None:
            continue
        c_out, c_in_g, k_h, k_w = w.shape
        groups = next((a.i for a in node.attribute if a.name == "group"), 1)
        out = shapes.get(node.output[0])
        inp = shapes.get(node.input[0])
        if not out or len(out) != 4 or not inp:
            continue
        _, _, h_out, w_out = out
        c_in = c_in_g * groups
        has_bias = len(node.input) > 2
        flops = conv2d_flops(c_in, c_out, k_h, k_w, h_out, w_out, groups, has_bias)
        n_in = int(np.prod(inp[1:]))
        n_out = c_out * h_out * w_out
        nbytes = bpe * (w.size + (c_out if has_bias else 0) + n_in + n_out)
        rows.append({"name": node.name, "c_in": int(c_in), "c_out": int(c_out), "k": int(k_h),
                     "groups": int(groups), "h_out": int(h_out), "w_out": int(w_out),
                     "flops": int(flops), "bytes": int(nbytes), "ai": flops / nbytes})
    return rows


def roofline(ai: float, peak_flops: float, bandwidth: float) -> float:
    """Attainable FLOP/s for arithmetic intensity ``ai`` (FLOPs/byte)."""
    return min(peak_flops, ai * bandwidth)
