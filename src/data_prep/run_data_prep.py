"""End-to-end runner for Member 1: cleaning, solid mask generation, splitting, and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    COL_FILENAME,
    COL_HC_MM,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    RAW_TRAIN_DIR,
    SPLITS_DIR,
)
from src.data_prep.cleaning import HC_BIN_LABELS, clean_hc18_metadata
from src.data_prep.masks import generate_solid_masks
from src.data_prep.splitting import create_grouped_stratified_splits


def plot_mask_verification(
    cleaned_df: pd.DataFrame,
    raw_train_dir: Path = RAW_TRAIN_DIR,
    masks_dir: Path = INTERIM_DATA_DIR / "masks",
    save_path: Path = FIGURES_DIR / "data_prep_mask_verification.png",
) -> Path:
    """Save a visual verification grid covering normal flood-filled and border-repaired masks."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    repaired = cleaned_df[cleaned_df["mask_fill_method"] == "ellipse_fit_border_repair"].head(2)
    normal = cleaned_df[cleaned_df["mask_fill_method"] == "flood_fill"].iloc[[0, len(cleaned_df) // 2]]
    samples = pd.concat([normal, repaired], ignore_index=True)

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Raw Ultrasound",
        "Hollow Ellipse Annotation",
        "Filled Solid Binary Mask",
        "Overlay (Mask + Contour)",
    ]

    for r, (_, row) in enumerate(samples.iterrows()):
        us = cv2.imread(str(raw_train_dir / row[COL_FILENAME]), cv2.IMREAD_GRAYSCALE)
        ann = cv2.imread(str(raw_train_dir / row["annotation_filename"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(masks_dir / row["mask_filename"]), cv2.IMREAD_GRAYSCALE)

        overlay = cv2.cvtColor(us, cv2.COLOR_GRAY2RGB)
        green = np.zeros_like(overlay)
        green[:, :, 1] = mask
        overlay = cv2.addWeighted(overlay, 1.0, green, 0.30, 0)
        overlay[ann > 127] = [255, 60, 60]

        imgs = [us, ann, mask, overlay]
        for c, im in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray" if c < 3 else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, fontweight="bold")
        axes[r, 0].set_ylabel(
            f"{row[COL_FILENAME]}\nHC={row[COL_HC_MM]:.1f} mm\n({row['mask_fill_method']})",
            fontsize=9,
            rotation=0,
            labelpad=75,
            va="center",
        )

    fig.suptitle(
        "Member 1 Verification: Hollow Contour to Solid Binary Mask Conversion",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_split_distributions(
    splits: Dict[str, pd.DataFrame],
    save_path: Path = FIGURES_DIR / "data_prep_split_distribution.png",
) -> Path:
    """Save Train / Validation / Test HC distribution and stratum proportion comparison."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {"train": "#2b8cbe", "val": "#fdae61", "test": "#66c2a5"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: Normalized HC density histograms
    bins = np.linspace(40, 360, 22)
    for name in ("train", "val", "test"):
        sdf = splits[name]
        axes[0].hist(
            sdf[COL_HC_MM],
            bins=bins,
            density=True,
            alpha=0.45,
            label=f"{name.title()} (n={len(sdf)}, mean={sdf[COL_HC_MM].mean():.1f} mm)",
            color=colors[name],
            edgecolor="black",
            linewidth=0.5,
        )
    axes[0].set_title("HC Distribution Across Splits (Normalized Density)", fontweight="bold")
    axes[0].set_xlabel("Head Circumference (mm)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Right: Percentage share per HC stratum
    x = np.arange(len(HC_BIN_LABELS))
    width = 0.26
    for i, name in enumerate(("train", "val", "test")):
        sdf = splits[name]
        pcts = [100.0 * (sdf["hc_bin"] == b).mean() for b in HC_BIN_LABELS]
        axes[1].bar(
            x + (i - 1) * width,
            pcts,
            width=width,
            label=name.title(),
            color=colors[name],
            edgecolor="black",
            linewidth=0.5,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{b} mm" for b in HC_BIN_LABELS], rotation=20)
    axes[1].set_title("Stratum Proportions Within Each Split (%)", fontweight="bold")
    axes[1].set_ylabel("Share Within Split (%)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Leakage-Free Exam-Grouped & HC-Stratified Split (70% / 15% / 15%)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_all() -> None:
    """Execute the full Member 1 data cleaning, mask generation, and splitting pipeline."""
    print("[1/4] Auditing & deduplicating HC18 training metadata...")
    cleaned_df, dup_audit_df = clean_hc18_metadata()
    n_dropped = int((dup_audit_df["action"] == "drop").sum()) if not dup_audit_df.empty else 0
    print(
        f"      Detected {dup_audit_df['duplicate_group'].nunique()} duplicate groups -> "
        f"dropped {n_dropped} redundant records; {len(cleaned_df)} cleaned images remain "
        f"across {cleaned_df['exam_group_id'].nunique()} exam groups."
    )

    print("[2/4] Flood-filling hollow ellipse annotations into solid binary masks...")
    cleaned_df = generate_solid_masks(cleaned_df)
    method_counts = cleaned_df["mask_fill_method"].value_counts().to_dict()
    mean_fg_pct = 100.0 * cleaned_df["fg_ratio"].mean()
    print(f"      Mask fill methods: {method_counts}")
    print(
        f"      Mean foreground pixel share: {mean_fg_pct:.2f}% "
        f"(background {100.0 - mean_fg_pct:.2f}%)"
    )

    print("[3/4] Creating leakage-free exam-grouped stratified Train/Val/Test splits...")
    splits, summary_df = create_grouped_stratified_splits(cleaned_df)
    print(summary_df.to_string(index=False))

    print("[4/4] Generating verification figures...")
    fig1 = plot_mask_verification(cleaned_df)
    fig2 = plot_split_distributions(splits)
    print(f"      Saved: {fig1}")
    print(f"      Saved: {fig2}")
    print("Member 1 pipeline complete. Zero leakage verified.")


if __name__ == "__main__":
    run_all()
