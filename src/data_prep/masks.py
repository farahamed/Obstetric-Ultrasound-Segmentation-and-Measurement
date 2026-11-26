"""Convert hollow HC18 ellipse contour annotations into solid binary masks."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

from src.config import COL_FILENAME, INTERIM_DATA_DIR, RAW_TRAIN_DIR


def fill_annotation_contour(annotation_gray: np.ndarray) -> Tuple[np.ndarray, str]:
    """Flood-fill a hollow HC18 ellipse contour into a solid binary mask ({0, 1}).

    For closed contours fully inside the frame, ``scipy.ndimage.binary_fill_holes``
    fills the interior exactly. For the few HC18 annotations where the ellipse is
    clipped by the top or bottom image border (leaving an open C-arc), naive
    flood-fill fails to fill the interior; in those cases, we fit the exact ellipse
    parameters with ``cv2.fitEllipse`` on the contour pixels and rasterize the
    filled ellipse clipped naturally to the image frame.

    Returns:
        mask_bin: uint8 array with values in {0, 1}.
        method: 'flood_fill' or 'ellipse_fit_border_repair'.
    """
    bin_contour = (annotation_gray > 127).astype(np.uint8)
    contour_px = int(bin_contour.sum())
    if contour_px == 0:
        raise ValueError("Empty annotation image encountered.")

    filled = binary_fill_holes(bin_contour).astype(np.uint8)
    filled_px = int(filled.sum())

    # If the interior failed to fill (e.g., border-clipped open arc), fit exact ellipse
    if filled_px <= 2 * contour_px:
        ys, xs = np.nonzero(bin_contour)
        pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        if len(pts) < 5:
            raise ValueError("Too few contour points to fit repair ellipse.")
        ellipse = cv2.fitEllipse(pts)
        repaired = np.zeros_like(bin_contour, dtype=np.uint8)
        cv2.ellipse(repaired, ellipse, color=1, thickness=-1)
        # Ensure original annotation contour pixels are included
        repaired = np.maximum(repaired, bin_contour)
        return repaired, "ellipse_fit_border_repair"

    return filled, "flood_fill"


def generate_solid_masks(
    cleaned_df: pd.DataFrame,
    raw_train_dir: Path = RAW_TRAIN_DIR,
    output_dir: Path = INTERIM_DATA_DIR,
) -> pd.DataFrame:
    """Generate and save solid binary masks (0/255 PNG) for all cleaned samples.

    Adds ``mask_filename``, ``mask_fill_method``, ``mask_area_px``, and ``fg_ratio``
    columns to ``cleaned_df`` and updates ``data/interim/cleaned_metadata.csv``.
    """
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_filenames = []
    fill_methods = []
    mask_areas = []
    fg_ratios = []

    for _, row in cleaned_df.iterrows():
        us_name = row[COL_FILENAME]
        ann_name = row["annotation_filename"]
        ann_path = raw_train_dir / ann_name
        ann_gray = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        if ann_gray is None:
            raise IOError(f"Failed to read annotation: {ann_path}")

        mask_bin, method = fill_annotation_contour(ann_gray)
        area_px = int(mask_bin.sum())
        total_px = int(mask_bin.size)
        fg_ratio = area_px / total_px

        # Sanity bounds: fetal head masks in HC18 occupy roughly 1% to 85% of image area
        if not (0.005 <= fg_ratio <= 0.90):
            raise ValueError(
                f"Suspicious filled mask area for {ann_name}: "
                f"{area_px}/{total_px} ({fg_ratio:.4f}) via {method}"
            )

        mask_name = us_name.replace(".png", "_mask.png")
        mask_path = masks_dir / mask_name
        cv2.imwrite(str(mask_path), mask_bin * 255)

        mask_filenames.append(mask_name)
        fill_methods.append(method)
        mask_areas.append(area_px)
        fg_ratios.append(round(fg_ratio, 6))

    updated_df = cleaned_df.copy()
    updated_df["mask_filename"] = mask_filenames
    updated_df["mask_fill_method"] = fill_methods
    updated_df["mask_area_px"] = mask_areas
    updated_df["fg_ratio"] = fg_ratios

    updated_df.to_csv(output_dir / "cleaned_metadata.csv", index=False)
    return updated_df
