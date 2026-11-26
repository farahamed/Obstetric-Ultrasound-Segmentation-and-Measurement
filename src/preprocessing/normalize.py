"""Intensity normalization helpers for grayscale obstetric ultrasound images."""

from __future__ import annotations

import numpy as np


def normalize_intensity(
    image: np.ndarray,
    method: str = "robust_minmax",
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> np.ndarray:
    """Normalize a 2D grayscale ultrasound image into float32 range [0.0, 1.0].

    Args:
        image: 2D grayscale array.
        method:
            - 'minmax': standard scaling dividing uint8 range or image [min, max] to [0, 1].
            - 'robust_minmax': clips extreme hot pixels/text markers at [lower_pct, upper_pct]
              computed over non-zero scan pixels, then rescales to [0, 1].
            - 'zscore_clipped': zero-mean/unit-std over non-zero scan pixels clipped to [-3, 3]
              and mapped to [0, 1].
    """
    arr = image.astype(np.float32)

    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    # Restrict statistics to active ultrasound sector (exclude pure-black zero padding)
    nonzero = arr[arr > 0]
    ref = nonzero if nonzero.size > 100 else arr.ravel()

    if method == "robust_minmax":
        lo = float(np.percentile(ref, lower_pct))
        hi = float(np.percentile(ref, upper_pct))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        # Keep pure-black ultrasound sector border at 0
        out[arr == 0] = 0.0
        return out.astype(np.float32)

    if method == "zscore_clipped":
        mean, std = float(ref.mean()), float(ref.std())
        if std < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        z = np.clip((arr - mean) / std, -3.0, 3.0)
        out = (z + 3.0) / 6.0
        out[arr == 0] = 0.0
        return out.astype(np.float32)

    raise ValueError(f"Unsupported normalization method: {method}")


def to_uint8(norm_image: np.ndarray) -> np.ndarray:
    """Convert a [0.0, 1.0] float image cleanly into uint8 [0, 255] for lossless PNG storage."""
    return np.clip(np.rint(norm_image * 255.0), 0, 255).astype(np.uint8)
