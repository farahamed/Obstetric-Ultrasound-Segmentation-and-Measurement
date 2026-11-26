"""End-to-end runner for resizing, calibration, normalization, and despeckling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    COL_FILENAME,
    COL_HC_MM,
    COL_PIXEL_SIZE,
    DEFAULT_TARGET_SIZE,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_TRAIN_DIR,
    SPLITS_DIR,
)
from src.preprocessing.despeckle import (
    FILTER_REGISTRY,
    apply_despeckle,
    benchmark_despeckle_filters,
)
from src.preprocessing.normalize import normalize_intensity, to_uint8
from src.preprocessing.resize import resize_image_and_mask


def _ramanujan_ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan's second approximation for ellipse perimeter given semi-axes a, b."""
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    return float(np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h))))


def _hc_mm_from_resized_mask(
    mask_u8: np.ndarray,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
) -> float:
    """Compute physical HC (mm) by fitting an ellipse to the resized mask in mm coordinates."""
    cnts, _ = cv2.findContours((mask_u8 > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return float("nan")
    largest = max(cnts, key=cv2.contourArea)
    if len(largest) < 5:
        return float("nan")
    # Convert contour points (x, y) into physical millimeter coordinates before fitting
    pts_mm = largest.reshape(-1, 2).astype(np.float32)
    pts_mm[:, 0] *= float(pixel_size_x_mm)
    pts_mm[:, 1] *= float(pixel_size_y_mm)
    (_, (axis1, axis2), _) = cv2.fitEllipse(pts_mm.reshape(-1, 1, 2))
    return _ramanujan_ellipse_perimeter(axis1 / 2.0, axis2 / 2.0)


def run_preprocessing_pipeline(
    despeckle_method: str = "anisotropic_diffusion",
    norm_method: str = "robust_minmax",
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    raw_train_dir: Path = RAW_TRAIN_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    splits_dir: Path = SPLITS_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process all 994 cleaned images & masks into standardized raw-normalized & despeckled sets.

    Outputs saved under ``data/processed/``:
      - ``images_raw_norm/``: Resized & normalized ultrasound PNGs (control arm).
      - ``images_despeckled/``: Resized, edge-preserving despeckled & normalized PNGs (experimental arm).
      - ``masks/``: Resized solid binary masks (strictly {0, 255}).
      - ``processed_metadata.csv``: Full manifest with split, paths, and recalibrated mm/px.
      - ``filter_benchmark.csv``: Quantitative ENL / CNR / EPI / runtime benchmark on validation split.
    """
    raw_norm_dir = processed_dir / "images_raw_norm"
    despeckled_dir = processed_dir / "images_despeckled"
    masks_out_dir = processed_dir / "masks"
    for d in (raw_norm_dir, despeckled_dir, masks_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load all splits so every sample carries its verified leakage-free split label
    split_dfs = []
    for s in ("train", "val", "test"):
        sdf = pd.read_csv(splits_dir / f"{s}.csv")
        sdf["split"] = s
        split_dfs.append(sdf)
    full_df = pd.concat(split_dfs, ignore_index=True)

    interim_masks_dir = interim_dir / "masks"
    records = []
    val_imgs_for_bench = []
    val_masks_for_bench = []

    for _, row in full_df.iterrows():
        us_name = row[COL_FILENAME]
        mask_name = row["mask_filename"]
        orig_ps = float(row[COL_PIXEL_SIZE])

        us_raw = cv2.imread(str(raw_train_dir / us_name), cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(str(interim_masks_dir / mask_name), cv2.IMREAD_GRAYSCALE)
        if us_raw is None or mask_raw is None:
            raise IOError(f"Failed loading image/mask pair for {us_name}")

        # 1. Resize & recalibrate anisotropic pixel sizes
        us_resized, mask_resized, cal = resize_image_and_mask(
            us_raw, mask_raw, orig_pixel_size_mm=orig_ps, target_size=target_size
        )

        # 2. Control arm: robust intensity normalization only
        us_raw_norm = to_uint8(normalize_intensity(us_resized, method=norm_method))

        # 3. Experimental arm: edge-preserving speckle reduction -> robust intensity normalization
        us_filtered = apply_despeckle(us_resized, method=despeckle_method)
        us_despeckled_norm = to_uint8(normalize_intensity(us_filtered, method=norm_method))

        cv2.imwrite(str(raw_norm_dir / us_name), us_raw_norm)
        cv2.imwrite(str(despeckled_dir / us_name), us_despeckled_norm)
        cv2.imwrite(str(masks_out_dir / mask_name), mask_resized)

        # Collect validation images for objective filter benchmarking (never touching test set)
        if row["split"] == "val":
            val_imgs_for_bench.append(us_raw_norm)
            val_masks_for_bench.append(mask_resized)

        # Verify physical HC consistency after resizing + pixel-size recalibration
        recal_hc_mm = _hc_mm_from_resized_mask(
            mask_resized, cal["pixel_size_x_mm"], cal["pixel_size_y_mm"]
        )

        rec = row.to_dict()
        rec.update(cal)
        rec["despeckle_method"] = despeckle_method
        rec["norm_method"] = norm_method
        rec["raw_norm_relpath"] = f"images_raw_norm/{us_name}"
        rec["despeckled_relpath"] = f"images_despeckled/{us_name}"
        rec["mask_relpath"] = f"masks/{mask_name}"
        rec["resized_gt_hc_mm"] = round(recal_hc_mm, 3)
        rec["resize_hc_abs_err_mm"] = round(abs(recal_hc_mm - float(row[COL_HC_MM])), 3)
        records.append(rec)

    processed_df = pd.DataFrame(records)
    processed_df.to_csv(processed_dir / "processed_metadata.csv", index=False)

    # Benchmark all 3 edge-preserving filters on a stratified sample of the validation split
    bench_df = benchmark_despeckle_filters(val_imgs_for_bench, val_masks_for_bench)
    bench_df.to_csv(processed_dir / "filter_benchmark.csv", index=False)

    return processed_df, bench_df


def plot_filter_comparison(
    processed_df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    save_path: Path = FIGURES_DIR / "preprocessing_filter_comparison.png",
) -> Path:
    """Plot Raw Normalized vs Bilateral vs Anisotropic Diffusion vs NLM + skull intensity profile."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick a representative mid-trimester validation image with clear skull boundaries
    val_df = processed_df[processed_df["split"] == "val"].copy()
    val_df["dist_to_median"] = (val_df[COL_HC_MM] - 175.0).abs()
    sample = val_df.sort_values("dist_to_median").iloc[0]

    raw_norm = cv2.imread(str(processed_dir / sample["raw_norm_relpath"]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(processed_dir / sample["mask_relpath"]), cv2.IMREAD_GRAYSCALE)

    variants = {
        "Raw Normalized": raw_norm,
        "Bilateral Filter": to_uint8(normalize_intensity(FILTER_REGISTRY["bilateral"](raw_norm))),
        "Anisotropic Diffusion\n(Perona-Malik)": to_uint8(
            normalize_intensity(FILTER_REGISTRY["anisotropic_diffusion"](raw_norm))
        ),
        "Non-Local Means\n(NLM)": to_uint8(normalize_intensity(FILTER_REGISTRY["nlm"](raw_norm))),
    }

    # Find skull bounding box for zoomed crop and horizontal profile row across head center
    ys, xs = np.nonzero(mask > 127)
    cy = int(np.mean(ys))
    x0 = max(0, int(xs.min()) - 20)
    x1 = min(raw_norm.shape[1], int(xs.max()) + 20)
    y0 = max(0, int(ys.min()) - 15)
    y1 = min(raw_norm.shape[0], int(ys.max()) + 15)

    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.1, 0.9], hspace=0.28, wspace=0.12)

    for c, (title, img) in enumerate(variants.items()):
        # Row 0: full resized image
        ax0 = fig.add_subplot(gs[0, c])
        ax0.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax0.axhline(cy, color="#ff4d4d", linestyle="--", linewidth=0.9, alpha=0.85)
        ax0.set_title(title, fontsize=10.5, fontweight="bold")
        ax0.axis("off")

        # Row 1: zoomed crop around fetal skull
        ax1 = fig.add_subplot(gs[1, c])
        ax1.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        ax1.set_title("Zoomed Skull Region", fontsize=9)
        ax1.axis("off")

    # Row 2: 1D intensity profile across skull wall (row cy)
    ax_prof = fig.add_subplot(gs[2, :])
    x_coords = np.arange(x0, x1)
    colors = ["#444444", "#1b9e77", "#d95f02", "#7570b3"]
    for (title, img), col in zip(variants.items(), colors):
        label = title.replace("\n", " ")
        lw = 1.1 if "Raw" in label else 1.7
        alpha = 0.65 if "Raw" in label else 0.95
        ax_prof.plot(x_coords, img[cy, x0:x1], label=label, color=col, linewidth=lw, alpha=alpha)

    ax_prof.set_title(
        f"Horizontal Intensity Profile Across Fetal Skull (y={cy}, {sample[COL_FILENAME]}, HC={sample[COL_HC_MM]:.1f} mm)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_prof.set_xlabel("X Coordinate (px)")
    ax_prof.set_ylabel("Normalized Intensity (0-255)")
    ax_prof.legend(ncol=4, loc="upper right", fontsize=9)
    ax_prof.grid(alpha=0.3)

    fig.suptitle(
        "Preprocessing: Edge-Preserving Speckle Reduction Comparison",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_calibration_check(
    processed_df: pd.DataFrame,
    save_path: Path = FIGURES_DIR / "preprocessing_calibration_check.png",
) -> Path:
    """Verify that resized masks + recalibrated mm/px accurately preserve ground-truth HC (mm)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gt = processed_df[COL_HC_MM].values
    est = processed_df["resized_gt_hc_mm"].values
    err = est - gt
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    axes[0].scatter(gt, est, s=12, alpha=0.55, color="#2b8cbe", edgecolors="none")
    lims = [min(gt.min(), est.min()) - 10, max(gt.max(), est.max()) + 10]
    axes[0].plot(lims, lims, "r--", linewidth=1.3, label="Ideal (y = x)")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_title(
        f"HC Calibration Check After 256×384 Resizing\n(MAE = {mae:.3f} mm, Max |Err| = {max_err:.3f} mm)",
        fontweight="bold",
    )
    axes[0].set_xlabel("Ground-Truth CSV HC (mm)")
    axes[0].set_ylabel("Resized Mask Ellipse-Fit HC (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(err, bins=35, color="#66c2a5", edgecolor="black", linewidth=0.5)
    axes[1].axvline(0.0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_title(
        f"Resizing Calibration Residual Distribution\n(Mean = {err.mean():+.3f} mm, SD = {err.std():.3f} mm)",
        fontweight="bold",
    )
    axes[1].set_xlabel("Estimated HC − Ground-Truth HC (mm)")
    axes[1].set_ylabel("Number of Images")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_all(despeckle_method: str = "anisotropic_diffusion") -> None:
    """Execute full preprocessing pipeline, benchmark, and plots."""
    print(
        f"[1/3] Running preprocessing (target={DEFAULT_TARGET_SIZE}, "
        f"norm=robust_minmax, despeckle={despeckle_method})..."
    )
    processed_df, bench_df = run_preprocessing_pipeline(despeckle_method=despeckle_method)
    print(f"      Processed {len(processed_df)} image/mask pairs into {PROCESSED_DATA_DIR}")
    print(
        f"      Resizing HC calibration check: MAE = {processed_df['resize_hc_abs_err_mm'].mean():.3f} mm, "
        f"Median |Err| = {processed_df['resize_hc_abs_err_mm'].median():.3f} mm, "
        f"95th pct = {processed_df['resize_hc_abs_err_mm'].quantile(0.95):.3f} mm"
    )

    print("\n[2/3] Validation-set edge-preserving filter benchmark (n=148):")
    print(bench_df.to_string(index=False))

    print("\n[3/3] Generating preprocessing verification figures...")
    fig1 = plot_filter_comparison(processed_df)
    fig2 = plot_calibration_check(processed_df)
    print(f"      Saved: {fig1}")
    print(f"      Saved: {fig2}")
    print("Preprocessing pipeline complete.")


if __name__ == "__main__":
    run_all()
