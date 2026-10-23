"""Handcrafted image features for the classical (non-deep) models.

The classical models cannot look at raw pixels directly: 720 x 540 x 3 is
over a million numbers per frame and only about 870 distinct training frames
exist. Instead each frame is summarised by a fixed-length vector built from
groups of features that target what a defect changes:

``colour``  joint HSV histogram (8 x 4 x 4 = 128 bins). Meat, tray film and
            belt have different hues; loose or twisted meat changes the
            amount and spread of meat colour.
``meat``    fraction, centroid and spread of meat-coloured pixels (5 values).
            Locates the tray in the frame.
``glare``   fraction of bright, unsaturated pixels on a 4 x 3 grid
            (12 values). Lidding film that is wrinkled or not sealed reflects
            light differently from film stretched flat over the tray.
``edges``   mean Sobel gradient magnitude on a 6 x 4 grid (24 values).
``texture`` uniform local binary pattern histogram (10 values).
``hog``     histogram of oriented gradients on a 128 x 96 grey image
            (1,260 values). Captures shape and edge layout.

All groups are computed after resizing the frame to ``work_size`` with a
chosen interpolation method. Resizing is 2D interpolation of the pixel grid,
so the interpolation method and working resolution are tested as sensitivity
parameters (see ``scripts/05_performance.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel

from .config import IMAGE_EXTENSIONS

INTERPOLATION = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
}

ALL_GROUPS = ("colour", "meat", "glare", "edges", "texture", "hog")


@dataclass(frozen=True)
class FeatureConfig:
    """Settings that define a feature vector.

    Attributes:
        work_size: (width, height) the frame is resized to before extraction.
        interpolation: Key of `INTERPOLATION` used for that resize.
        groups: Feature groups to include, in order.
        hog_size: (width, height) of the grey image used for HOG.
        hsv: ``"skimage"`` (float64 reference conversion) or ``"pil"`` (fast
            8-bit conversion in C). The 8-bit result is quantised to steps of
            1/255, far finer than the 8 x 4 x 4 histogram bins; see
            ``scripts/05_performance.py`` for the measured effect.
    """

    work_size: tuple[int, int] = (240, 180)
    interpolation: str = "bilinear"
    groups: tuple[str, ...] = ALL_GROUPS
    hog_size: tuple[int, int] = (128, 96)
    hsv: str = "skimage"

    def key(self) -> str:
        """Short hash used to name feature caches."""
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 array (H, W, 3).

    Grey, RGBA and palette images are converted to RGB. EXIF orientation is
    not applied because the Roboflow export already stripped it.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the extension is not a supported image type or the
            file cannot be decoded as an image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image type {p.suffix!r}; expected one of {sorted(IMAGE_EXTENSIONS)}")
    try:
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"))
    except Exception as exc:  # PIL raises several types for corrupt files
        raise ValueError(f"could not decode {p}: {exc}") from exc


def resize(rgb: np.ndarray, size: tuple[int, int], method: str = "bilinear") -> np.ndarray:
    """Resize an RGB array to ``size`` = (width, height) with a named method."""
    if method not in INTERPOLATION:
        raise ValueError(f"unknown interpolation {method!r}; choose from {sorted(INTERPOLATION)}")
    return np.asarray(Image.fromarray(rgb).resize(size, INTERPOLATION[method]))


def _grid_mean(a: np.ndarray, nx: int, ny: int) -> np.ndarray:
    h, w = a.shape
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    return np.array([a[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean() for j in range(ny) for i in range(nx)])


def to_hsv(rgb: np.ndarray, method: str = "skimage") -> np.ndarray:
    """RGB uint8 -> HSV float64 in [0, 1] with the chosen implementation."""
    if method == "skimage":
        return rgb2hsv(rgb)
    if method == "pil":
        return np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float64) / 255.0
    raise ValueError(f"unknown hsv method {method!r}; use 'skimage' or 'pil'")


def hsv_histogram(hsv: np.ndarray, bins: tuple[int, int, int] = (8, 4, 4)) -> np.ndarray:
    """Normalised joint HSV histogram with equal-width bins on [0, 1].

    Computes each pixel's bin index directly and counts with ``np.bincount``,
    which gives the same counts as ``np.histogramdd`` with
    ``range=((0, 1),) * 3`` (a value of exactly 1.0 goes in the last bin) but
    avoids its per-dimension sorting.
    """
    flat = np.zeros(hsv.shape[:2], dtype=np.int64)
    for i, b in enumerate(bins):
        flat = flat * b + np.minimum((hsv[..., i] * b).astype(np.int64), b - 1)
    hist = np.bincount(flat.ravel(), minlength=int(np.prod(bins))).astype(np.float64)
    return hist / hist.sum()


def meat_mask(hsv: np.ndarray) -> np.ndarray:
    """Boolean mask of meat-coloured pixels (red to pink, saturated, not dark)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h < 0.07) | (h > 0.93)) & (s > 0.25) & (v > 0.25)


def extract(rgb: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Feature vector for one RGB frame.

    Args:
        rgb: (H, W, 3) uint8 array, any size. It is resized to ``cfg.work_size``.
        cfg: Feature settings.

    Returns:
        1-D float64 array. Its length depends only on ``cfg``.

    Raises:
        ValueError: If ``rgb`` is not an (H, W, 3) array.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got shape {rgb.shape}")
    small = resize(rgb, cfg.work_size, cfg.interpolation)
    hsv = to_hsv(small, cfg.hsv)
    grey = small.astype(np.float64) @ np.array([0.299, 0.587, 0.114]) / 255.0
    parts: list[np.ndarray] = []
    for g in cfg.groups:
        if g == "colour":
            parts.append(hsv_histogram(hsv))
        elif g == "meat":
            m = meat_mask(hsv)
            frac = m.mean()
            if m.any():
                yy, xx = np.nonzero(m)
                cx, cy = xx.mean() / m.shape[1], yy.mean() / m.shape[0]
                sx, sy = xx.std() / m.shape[1], yy.std() / m.shape[0]
            else:
                cx = cy = sx = sy = 0.0
            parts.append(np.array([frac, cx, cy, sx, sy]))
        elif g == "glare":
            glare = (hsv[..., 2] > 0.85) & (hsv[..., 1] < 0.2)
            parts.append(_grid_mean(glare.astype(float), 4, 3))
        elif g == "edges":
            parts.append(_grid_mean(sobel(grey), 6, 4))
        elif g == "texture":
            lbp = local_binary_pattern((grey * 255).astype(np.uint8), P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
            parts.append(hist / hist.sum())
        elif g == "hog":
            hg = np.asarray(Image.fromarray((grey * 255).astype(np.uint8)).resize(cfg.hog_size, INTERPOLATION[cfg.interpolation]))
            parts.append(hog(hg, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                             block_norm="L2-Hys", feature_vector=True))
        else:
            raise ValueError(f"unknown feature group {g!r}")
    return np.concatenate(parts).astype(np.float64)


def group_slices(cfg: FeatureConfig = FeatureConfig()) -> dict[str, slice]:
    """Column range of each feature group in the vector from `extract`."""
    dummy = np.zeros((cfg.work_size[1], cfg.work_size[0], 3), dtype=np.uint8)
    out, start = {}, 0
    for g in cfg.groups:
        n = len(extract(dummy, FeatureConfig(cfg.work_size, cfg.interpolation, (g,), cfg.hog_size, cfg.hsv)))
        out[g] = slice(start, start + n)
        start += n
    return out


def extract_many(paths, cfg: FeatureConfig = FeatureConfig(), n_jobs: int = -1,
                 cache_dir: str | Path | None = None) -> np.ndarray:
    """Feature matrix for many images, optionally cached on disk.

    Args:
        paths: Image paths. The cache is keyed by file names and ``cfg``.
        cfg: Feature settings.
        n_jobs: Parallel workers for joblib (-1 = all cores).
        cache_dir: If given, load from / save to ``<cache_dir>/<hash>.npy``.

    Returns:
        (n_images, n_features) float64 array in the order of ``paths``.
    """
    from joblib import Parallel, delayed

    paths = [str(p) for p in paths]
    cache = None
    if cache_dir is not None:
        h = hashlib.sha1((cfg.key() + "|".join(Path(p).name for p in paths)).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"feat_{h}.npy"
        if cache.exists():
            return np.load(cache)
    X = np.stack(Parallel(n_jobs=n_jobs)(delayed(_extract_path)(p, cfg) for p in paths))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    return X


def _extract_path(path: str, cfg: FeatureConfig) -> np.ndarray:
    return extract(load_rgb(path), cfg)
