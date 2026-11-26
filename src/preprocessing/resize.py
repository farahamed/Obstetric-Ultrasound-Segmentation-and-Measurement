"""Image and mask dimension standardization with exact anisotropic mm/px recalibration."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from src.config import DEFAULT_TARGET_SIZE


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    orig_pixel_size_mm: float,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Resize grayscale ultrasound image and binary mask to target (H, W) and recalibrate spacing.

    Uses ``cv2.INTER_AREA`` for anti-aliased downsampling of ultrasound intensities and
    ``cv2.INTER_NEAREST`` for binary masks so label values stay strictly binary.

    Args:
        image: 2D grayscale ultrasound array of shape (H_orig, W_orig).
        mask: 2D binary mask array of shape (H_orig, W_orig) with values in {0, 255} or {0, 1}.
        orig_pixel_size_mm: Original isotropic pixel spacing in mm/px from HC18 CSV.
        target_size: Desired (target_height, target_width).

    Returns:
        resized_img: 2D uint8 ultrasound image of shape (target_height, target_width).
        resized_mask: 2D uint8 binary mask of shape (target_height, target_width) in {0, 255}.
        calibration: Dictionary with original/target dimensions, scale factors, and
            exact recalibrated ``pixel_size_x_mm`` and ``pixel_size_y_mm``.
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size

    interp_img = (
        cv2.INTER_AREA if (target_h < orig_h or target_w < orig_w) else cv2.INTER_LINEAR
    )
    resized_img = cv2.resize(image, (target_w, target_h), interpolation=interp_img)

    bin_mask = ((mask > 127).astype(np.uint8) * 255) if mask.max() > 1 else (mask.astype(np.uint8) * 255)
    resized_mask = cv2.resize(bin_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    resized_mask = ((resized_mask > 127).astype(np.uint8) * 255)

    # Physical width = orig_w * orig_pixel_size_mm = target_w * pixel_size_x_mm
    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)
    pixel_size_x_mm = float(orig_pixel_size_mm) * (orig_w / float(target_w))
    pixel_size_y_mm = float(orig_pixel_size_mm) * (orig_h / float(target_h))
    effective_pixel_size_mm = float(np.sqrt(pixel_size_x_mm * pixel_size_y_mm))

    calibration = {
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "scale_x": round(scale_x, 6),
        "scale_y": round(scale_y, 6),
        "pixel_size_x_mm": round(pixel_size_x_mm, 8),
        "pixel_size_y_mm": round(pixel_size_y_mm, 8),
        "effective_pixel_size_mm": round(effective_pixel_size_mm, 8),
    }
    return resized_img, resized_mask, calibration
