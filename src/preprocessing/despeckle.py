"""Edge-preserving speckle reduction filters and quantitative ultrasound quality metrics."""

from __future__ import annotations

import time
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd


def bilateral_despeckle(
    image_u8: np.ndarray,
    d: int = 7,
    sigma_color: float = 25.0,
    sigma_space: float = 25.0,
) -> np.ndarray:
    """Edge-preserving bilateral filter (spatial + intensity range Gaussian weights)."""
    return cv2.bilateralFilter(image_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def anisotropic_diffusion_despeckle(
    image_u8: np.ndarray,
    n_iter: int = 12,
    kappa: float = 18.0,
    gamma: float = 0.15,
    option: int = 1,
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion for edge-preserving ultrasound speckle reduction.

    Diffuses homogeneous speckled tissue while inhibiting diffusion across strong
    hyperechoic cranial skull boundaries (high gradient magnitude).
    """
    img = image_u8.astype(np.float32).copy()
    k2 = float(kappa) ** 2

    for _ in range(n_iter):
        # 4-connected directional finite differences (N, S, E, W)
        north = np.zeros_like(img)
        south = np.zeros_like(img)
        east = np.zeros_like(img)
        west = np.zeros_like(img)

        north[1:, :] = img[:-1, :] - img[1:, :]
        south[:-1, :] = img[1:, :] - img[:-1, :]
        east[:, :-1] = img[:, 1:] - img[:, :-1]
        west[:, 1:] = img[:, :-1] - img[:, 1:]

        if option == 1:
            # Favors high-contrast edges (ideal for bright fetal skull contour)
            cN = np.exp(-((north ** 2) / k2))
            cS = np.exp(-((south ** 2) / k2))
            cE = np.exp(-((east ** 2) / k2))
            cW = np.exp(-((west ** 2) / k2))
        else:
            # Favors wide regions over smaller ones
            cN = 1.0 / (1.0 + (north ** 2) / k2)
            cS = 1.0 / (1.0 + (south ** 2) / k2)
            cE = 1.0 / (1.0 + (east ** 2) / k2)
            cW = 1.0 / (1.0 + (west ** 2) / k2)

        img += gamma * (cN * north + cS * south + cE * east + cW * west)

    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def nlm_despeckle(
    image_u8: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """Edge-preserving Non-Local Means denoising using patch self-similarity."""
    return cv2.fastNlMeansDenoising(
        image_u8,
        None,
        h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )


FILTER_REGISTRY = {
    "bilateral": bilateral_despeckle,
    "anisotropic_diffusion": anisotropic_diffusion_despeckle,
    "nlm": nlm_despeckle,
}


def apply_despeckle(image_u8: np.ndarray, method: str = "anisotropic_diffusion") -> np.ndarray:
    """Dispatch edge-preserving speckle-reduction filter by name."""
    if method not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown despeckle method '{method}'. Choose from {list(FILTER_REGISTRY)}."
        )
    return FILTER_REGISTRY[method](image_u8)


def compute_ultrasound_quality_metrics(
    raw_u8: np.ndarray,
    filtered_u8: np.ndarray,
    mask_u8: np.ndarray,
) -> Dict[str, float]:
    """Compute ENL, skull-ring CNR, and Edge Preservation Index (EPI) on one image.

    Regions derived from the ground-truth solid fetal head mask:
      - Homogeneous interior (eroded mask) -> Equivalent Number of Looks (ENL = mu^2 / var)
      - Hyperechoic skull ring (boundary band of mask) vs. surrounding tissue band -> CNR
      - High-pass Laplacian edge correlation on skull ring -> Edge Preservation Index (EPI)
    """
    raw = raw_u8.astype(np.float64)
    flt = filtered_u8.astype(np.float64)
    fg = (mask_u8 > 127).astype(np.uint8)

    # 1. Homogeneous fetal brain interior (eroded away from skull bone)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    interior = cv2.erode(fg, kernel_erode) > 0
    if interior.sum() < 50:
        interior = fg > 0

    mu_in = flt[interior].mean()
    var_in = flt[interior].var() + 1e-6
    enl = float((mu_in ** 2) / var_in)

    # 2. Skull boundary ring vs. adjacent background ring (excluding zero border)
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    skull_ring = (cv2.dilate(fg, k_ring) - cv2.erode(fg, k_ring)) > 0
    outer_bg = ((cv2.dilate(fg, k_outer) - cv2.dilate(fg, k_ring)) > 0) & (raw > 5)

    if skull_ring.sum() > 20 and outer_bg.sum() > 20:
        mu_s, std_s = flt[skull_ring].mean(), flt[skull_ring].std()
        mu_b, std_b = flt[outer_bg].mean(), flt[outer_bg].std()
        cnr = float(abs(mu_s - mu_b) / np.sqrt(std_s ** 2 + std_b ** 2 + 1e-6))
    else:
        cnr = float("nan")

    # 3. Edge Preservation Index (EPI) along the skull boundary band
    lap_raw = cv2.Laplacian(raw, cv2.CV_64F)
    lap_flt = cv2.Laplacian(flt, cv2.CV_64F)
    r_vals = lap_raw[skull_ring]
    f_vals = lap_flt[skull_ring]
    r_c = r_vals - r_vals.mean()
    f_c = f_vals - f_vals.mean()
    denom = np.sqrt(np.sum(r_c ** 2) * np.sum(f_c ** 2)) + 1e-9
    epi = float(np.sum(r_c * f_c) / denom)

    # 4. Speckle variance reduction ratio inside interior (higher = smoother speckle)
    raw_var = raw[interior].var() + 1e-6
    var_reduction_pct = float(100.0 * (1.0 - var_in / raw_var))

    return {
        "enl": enl,
        "cnr": cnr,
        "epi": epi,
        "interior_var_reduction_pct": var_reduction_pct,
    }


def benchmark_despeckle_filters(
    images_u8: List[np.ndarray],
    masks_u8: List[np.ndarray],
) -> pd.DataFrame:
    """Benchmark all registered edge-preserving filters (+ raw baseline) on a sample batch."""
    rows = []

    # Raw baseline
    raw_metrics = [
        compute_ultrasound_quality_metrics(im, im, mk)
        for im, mk in zip(images_u8, masks_u8)
    ]
    rows.append(
        {
            "filter": "raw_normalized",
            "enl_mean": round(float(np.nanmean([m["enl"] for m in raw_metrics])), 3),
            "cnr_mean": round(float(np.nanmean([m["cnr"] for m in raw_metrics])), 4),
            "epi_mean": round(float(np.nanmean([m["epi"] for m in raw_metrics])), 4),
            "interior_var_reduction_pct": 0.0,
            "runtime_ms_per_img": 0.0,
        }
    )

    for name, fn in FILTER_REGISTRY.items():
        t0 = time.perf_counter()
        metrics = []
        for im, mk in zip(images_u8, masks_u8):
            flt = fn(im)
            metrics.append(compute_ultrasound_quality_metrics(im, flt, mk))
        elapsed_ms = 1000.0 * (time.perf_counter() - t0) / max(len(images_u8), 1)
        rows.append(
            {
                "filter": name,
                "enl_mean": round(float(np.nanmean([m["enl"] for m in metrics])), 3),
                "cnr_mean": round(float(np.nanmean([m["cnr"] for m in metrics])), 4),
                "epi_mean": round(float(np.nanmean([m["epi"] for m in metrics])), 4),
                "interior_var_reduction_pct": round(
                    float(np.nanmean([m["interior_var_reduction_pct"] for m in metrics])), 2
                ),
                "runtime_ms_per_img": round(elapsed_ms, 2),
            }
        )

    return pd.DataFrame(rows)
