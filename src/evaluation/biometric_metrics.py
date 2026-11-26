"""Biometric segmentation and head circumference evaluation metrics (DSC, HD, AD, MD)."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def dice_coefficient_pct(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute binary Dice Similarity Coefficient (DSC) in percent [0, 100]."""
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return 100.0
    inter = int(np.logical_and(p, g).sum())
    return float(100.0 * (2.0 * inter) / denom)


def _extract_boundary_points_mm(
    mask: np.ndarray,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
) -> np.ndarray:
    """Extract 1-pixel boundary coordinates scaled into physical millimeters."""
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return np.empty((0, 2), dtype=np.float64)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(bin_mask, kernel)
    boundary = (bin_mask - eroded) > 0
    if not boundary.any():
        boundary = bin_mask > 0
    ys, xs = np.nonzero(boundary)
    pts = np.column_stack([xs * float(pixel_size_x_mm), ys * float(pixel_size_y_mm)])
    return pts.astype(np.float64)


def hausdorff_distance_mm(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pixel_size_x_mm: float,
    pixel_size_y_mm: Optional[float] = None,
) -> Dict[str, float]:
    """Compute exact symmetric Hausdorff Distance (HD) and 95th-percentile HD95 in mm."""
    if pixel_size_y_mm is None:
        pixel_size_y_mm = pixel_size_x_mm

    pts_p = _extract_boundary_points_mm(pred_mask, pixel_size_x_mm, pixel_size_y_mm)
    pts_g = _extract_boundary_points_mm(gt_mask, pixel_size_x_mm, pixel_size_y_mm)

    if len(pts_p) == 0 or len(pts_g) == 0:
        return {"hd_mm": float("nan"), "hd95_mm": float("nan")}

    d_p_to_g, _ = cKDTree(pts_g).query(pts_p, k=1)
    d_g_to_p, _ = cKDTree(pts_p).query(pts_g, k=1)

    hd = float(max(d_p_to_g.max(), d_g_to_p.max()))
    hd95 = float(max(np.percentile(d_p_to_g, 95), np.percentile(d_g_to_p, 95)))
    return {"hd_mm": hd, "hd95_mm": hd95}


def compute_sample_biometric_metrics(
    pred_clean_mask: np.ndarray,
    pred_ellipse_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_hc_mm: float,
    gt_hc_mm: float,
    pixel_size_x_mm: float,
    pixel_size_y_mm: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all clinical biometric metrics for one ultrasound image.

    Returns DSC (%), HD (mm), HD95 (mm) for both the cleaned U-Net mask and the
    ellipse-fitted mask, plus signed Mean Difference (MD = pred - gt) and
    Absolute Difference (AD = |pred - gt|) in mm.
    """
    dsc_clean = dice_coefficient_pct(pred_clean_mask, gt_mask)
    dsc_ellipse = dice_coefficient_pct(pred_ellipse_mask, gt_mask)

    hd_clean = hausdorff_distance_mm(
        pred_clean_mask, gt_mask, pixel_size_x_mm, pixel_size_y_mm
    )
    hd_ellipse = hausdorff_distance_mm(
        pred_ellipse_mask, gt_mask, pixel_size_x_mm, pixel_size_y_mm
    )

    md_mm = float(pred_hc_mm - gt_hc_mm)
    ad_mm = float(abs(md_mm))

    return {
        "dsc_clean_pct": round(dsc_clean, 4),
        "dsc_ellipse_pct": round(dsc_ellipse, 4),
        "hd_clean_mm": round(hd_clean["hd_mm"], 4),
        "hd95_clean_mm": round(hd_clean["hd95_mm"], 4),
        "hd_ellipse_mm": round(hd_ellipse["hd_mm"], 4),
        "hd95_ellipse_mm": round(hd_ellipse["hd95_mm"], 4),
        "md_mm": round(md_mm, 4),
        "ad_mm": round(ad_mm, 4),
    }


def assign_trimester_group(hc_mm: float) -> str:
    """Map fetal HC (mm) into approximate clinical trimester strata."""
    if hc_mm < 100.0:
        return "1st Trimester (<100 mm)"
    if hc_mm <= 250.0:
        return "2nd Trimester (100-250 mm)"
    return "3rd Trimester (>250 mm)"


def summarize_biometric_results(pred_df: pd.DataFrame, label: str) -> Dict[str, object]:
    """Aggregate mean/SD biometric metrics across an evaluated split."""
    return {
        "model": label,
        "n_images": int(len(pred_df)),
        "dsc_clean_mean_pct": round(float(pred_df["dsc_clean_pct"].mean()), 2),
        "dsc_clean_std_pct": round(float(pred_df["dsc_clean_pct"].std()), 2),
        "dsc_ellipse_mean_pct": round(float(pred_df["dsc_ellipse_pct"].mean()), 2),
        "dsc_ellipse_std_pct": round(float(pred_df["dsc_ellipse_pct"].std()), 2),
        "hd_ellipse_mean_mm": round(float(pred_df["hd_ellipse_mm"].mean()), 2),
        "hd_ellipse_std_mm": round(float(pred_df["hd_ellipse_mm"].std()), 2),
        "hd95_ellipse_mean_mm": round(float(pred_df["hd95_ellipse_mm"].mean()), 2),
        "ad_mean_mm": round(float(pred_df["ad_mm"].mean()), 2),
        "ad_std_mm": round(float(pred_df["ad_mm"].std()), 2),
        "ad_median_mm": round(float(pred_df["ad_mm"].median()), 2),
        "md_mean_mm": round(float(pred_df["md_mm"].mean()), 2),
        "md_std_mm": round(float(pred_df["md_mm"].std()), 2),
    }


def summarize_by_trimester(pred_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Aggregate DSC (%), HD (mm), AD (mm), and MD (mm) broken down by trimester stage."""
    df = pred_df.copy()
    df["trimester"] = df["gt_hc_mm"].apply(assign_trimester_group)
    order = [
        "1st Trimester (<100 mm)",
        "2nd Trimester (100-250 mm)",
        "3rd Trimester (>250 mm)",
    ]
    rows = []
    for tri in order:
        sub = df[df["trimester"] == tri]
        if sub.empty:
            continue
        rows.append(
            {
                "model": label,
                "trimester": tri,
                "n_images": int(len(sub)),
                "dsc_ellipse_pct": round(float(sub["dsc_ellipse_pct"].mean()), 2),
                "hd_ellipse_mm": round(float(sub["hd_ellipse_mm"].mean()), 2),
                "ad_mm": round(float(sub["ad_mm"].mean()), 2),
                "md_mm": round(float(sub["md_mm"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)
