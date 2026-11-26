"""Leakage-free exam-grouped, HC-stratified Train / Validation / Test splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COL_FILENAME,
    COL_HC_MM,
    COL_PIXEL_SIZE,
    RANDOM_SEED,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data_prep.cleaning import HC_BIN_LABELS, assign_hc_bins


def create_grouped_stratified_splits(
    cleaned_df: pd.DataFrame,
    splits_dir: Path = SPLITS_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Split cleaned HC18 samples into leakage-free Train / Val / Test manifests.

    Splitting is performed strictly at the ``exam_group_id`` level so that all
    multi-plane ultrasound scans from the same exam stay in a single split,
    and stratified by each exam's mean head-circumference size stratum.

    Raises AssertionError if any exam ID, filename, or pixel hash overlaps across splits.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate one stratification label per unique exam_group_id
    exam_table = (
        cleaned_df.groupby("exam_group_id", as_index=False)
        .agg(
            n_images=(COL_FILENAME, "count"),
            mean_hc_mm=(COL_HC_MM, "mean"),
        )
    )
    exam_table["exam_hc_bin"] = assign_hc_bins(exam_table["mean_hc_mm"])

    # Stage 1: Train vs. Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_exams, temp_exams = train_test_split(
        exam_table,
        test_size=temp_ratio,
        stratify=exam_table["exam_hc_bin"],
        random_state=random_seed,
    )

    # Stage 2: Val vs. Test
    rel_test_ratio = test_ratio / temp_ratio
    val_exams, test_exams = train_test_split(
        temp_exams,
        test_size=rel_test_ratio,
        stratify=temp_exams["exam_hc_bin"],
        random_state=random_seed,
    )

    exam_to_split = {}
    for eid in train_exams["exam_group_id"]:
        exam_to_split[int(eid)] = "train"
    for eid in val_exams["exam_group_id"]:
        exam_to_split[int(eid)] = "val"
    for eid in test_exams["exam_group_id"]:
        exam_to_split[int(eid)] = "test"

    full_df = cleaned_df.copy()
    full_df["split"] = full_df["exam_group_id"].map(exam_to_split)

    splits = {
        name: full_df[full_df["split"] == name].sort_values("seq_id").reset_index(drop=True)
        for name in ("train", "val", "test")
    }

    # Strict anti-leakage assertions
    _verify_no_leakage(splits, expected_total=len(cleaned_df))

    # Save CSV manifests
    for name, sdf in splits.items():
        sdf.to_csv(splits_dir / f"{name}.csv", index=False)

    # Save concise summary table
    summary_rows = []
    total_images = len(full_df)
    for name in ("train", "val", "test"):
        sdf = splits[name]
        row = {
            "split": name,
            "n_images": len(sdf),
            "image_share_pct": round(100.0 * len(sdf) / total_images, 2),
            "n_exams": int(sdf["exam_group_id"].nunique()),
            "hc_mean_mm": round(float(sdf[COL_HC_MM].mean()), 2),
            "hc_median_mm": round(float(sdf[COL_HC_MM].median()), 2),
            "hc_std_mm": round(float(sdf[COL_HC_MM].std()), 2),
            "pixel_size_mean_mm": round(float(sdf[COL_PIXEL_SIZE].mean()), 5),
        }
        if "fg_ratio" in sdf.columns:
            mean_fg = float(sdf["fg_ratio"].mean())
            row["fg_pixel_pct"] = round(100.0 * mean_fg, 2)
            row["recommended_bce_pos_weight"] = round((1.0 - mean_fg) / mean_fg, 3)
        for b in HC_BIN_LABELS:
            row[f"bin_{b}"] = int((sdf["hc_bin"] == b).sum())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(splits_dir / "split_summary.csv", index=False)
    return splits, summary_df


def _verify_no_leakage(splits: Dict[str, pd.DataFrame], expected_total: int) -> None:
    """Assert zero overlap across splits by exam_group_id, filename, and pixel hashes."""
    total = sum(len(sdf) for sdf in splits.values())
    assert total == expected_total, f"Row count mismatch: {total} != {expected_total}"

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    check_cols = ["exam_group_id", COL_FILENAME, "us_pixel_sha256", "ann_pixel_sha256"]

    for a, b in pairs:
        for col in check_cols:
            if col in splits[a].columns and col in splits[b].columns:
                overlap = set(splits[a][col]).intersection(set(splits[b][col]))
                assert len(overlap) == 0, (
                    f"DATA LEAKAGE DETECTED between '{a}' and '{b}' on '{col}': "
                    f"{len(overlap)} overlapping entries ({list(overlap)[:5]})"
                )
