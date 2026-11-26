"""Postprocessing utilities: morphological cleanup, ellipse fitting, and Ramanujan HC (mm)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


@dataclass
class EllipseMeasurement:
    """Result of contour extraction and least-squares ellipse fitting on a binary mask."""

    clean_mask: np.ndarray  # uint8 {0, 1}
    ellipse_mask: np.ndarray  # uint8 {0, 1}
    center_px: Tuple[float, float]
    axes_px: Tuple[float, float]  # full (axis1, axis2) in pixels
    angle_deg: float
    semi_major_mm: float
    semi_minor_mm: float
    hc_mm: float
    success: bool


def ramanujan_ellipse_perimeter(a: float, b: float) -> float:
    """Compute ellipse circumference using Ramanujan's second approximation.

    Args:
        a: Semi-major axis length (mm).
        b: Semi-minor axis length (mm).

    Returns:
        Perimeter in millimeters.
    """
    a = abs(float(a))
    b = abs(float(b))
    if a + b <= 0.0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return float(np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h))))


def clean_binary_mask(
    prob_or_mask: np.ndarray,
    threshold: float = 0.5,
    keep_largest_cc: bool = True,
    fill_holes: bool = True,
    morph_close_kernel: int = 5,
) -> np.ndarray:
    """Clean a predicted probability map or binary mask for contour/ellipse extraction.

    Steps:
      1. Binarize using ``threshold`` (handles both [0, 1] float probs and {0, 255} uint8 masks).
      2. Optional morphological closing to bridge small acoustic-shadowing gaps.
      3. Fill internal holes via ``binary_fill_holes``.
      4. Retain only the largest foreground connected component to remove stray islands.
    """
    if prob_or_mask.dtype == np.uint8 and prob_or_mask.max() > 1:
        bin_mask = (prob_or_mask > 127).astype(np.uint8)
    else:
        bin_mask = (prob_or_mask >= threshold).astype(np.uint8)

    if bin_mask.sum() == 0:
        return bin_mask

    if morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel)
        )
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    if fill_holes:
        bin_mask = binary_fill_holes(bin_mask).astype(np.uint8)

    if keep_largest_cc:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            bin_mask = (labels == largest_label).astype(np.uint8)

    return bin_mask


def fit_ellipse_and_measure_hc(
    prob_or_mask: np.ndarray,
    pixel_size_x_mm: float,
    pixel_size_y_mm: Optional[float] = None,
    threshold: float = 0.5,
    morph_close_kernel: int = 5,
) -> EllipseMeasurement:
    """Clean predicted mask, fit a least-squares ellipse, and compute physical HC (mm).

    Fits the ellipse in pixel space (for rasterizing ``ellipse_mask``) and in physical
    millimeter coordinates ``(x * pixel_size_x_mm, y * pixel_size_y_mm)`` so both
    isotropic native images and anisotropically resized images yield unbiased HC (mm).
    """
    if pixel_size_y_mm is None:
        pixel_size_y_mm = pixel_size_x_mm

    clean_mask = clean_binary_mask(
        prob_or_mask,
        threshold=threshold,
        keep_largest_cc=True,
        fill_holes=True,
        morph_close_kernel=morph_close_kernel,
    )

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return EllipseMeasurement(
            clean_mask=clean_mask,
            ellipse_mask=np.zeros_like(clean_mask),
            center_px=(0.0, 0.0),
            axes_px=(0.0, 0.0),
            angle_deg=0.0,
            semi_major_mm=0.0,
            semi_minor_mm=0.0,
            hc_mm=float("nan"),
            success=False,
        )

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        return EllipseMeasurement(
            clean_mask=clean_mask,
            ellipse_mask=clean_mask.copy(),
            center_px=(0.0, 0.0),
            axes_px=(0.0, 0.0),
            angle_deg=0.0,
            semi_major_mm=0.0,
            semi_minor_mm=0.0,
            hc_mm=float("nan"),
            success=False,
        )

    # Pixel-space least-squares ellipse fit (for overlay and geometric mask interpolation)
    ellipse_px = cv2.fitEllipse(largest_contour)
    (cx, cy), (w_px, h_px), angle_deg = ellipse_px

    ellipse_mask = np.zeros_like(clean_mask, dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse_px, color=1, thickness=-1)

    # Physical-space least-squares ellipse fit (mm coordinates -> exact semi-axes)
    pts_mm = largest_contour.reshape(-1, 2).astype(np.float32)
    pts_mm[:, 0] *= float(pixel_size_x_mm)
    pts_mm[:, 1] *= float(pixel_size_y_mm)
    (_, (axis1_mm, axis2_mm), _) = cv2.fitEllipse(pts_mm.reshape(-1, 1, 2))

    semi_major_mm = max(axis1_mm, axis2_mm) / 2.0
    semi_minor_mm = min(axis1_mm, axis2_mm) / 2.0
    hc_mm = ramanujan_ellipse_perimeter(semi_major_mm, semi_minor_mm)

    return EllipseMeasurement(
        clean_mask=clean_mask,
        ellipse_mask=ellipse_mask,
        center_px=(float(cx), float(cy)),
        axes_px=(float(w_px), float(h_px)),
        angle_deg=float(angle_deg),
        semi_major_mm=float(semi_major_mm),
        semi_minor_mm=float(semi_minor_mm),
        hc_mm=float(hc_mm),
        success=True,
    )
