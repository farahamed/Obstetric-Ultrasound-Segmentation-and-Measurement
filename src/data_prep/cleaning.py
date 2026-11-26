"""Data cleaning, duplicate auditing, and multi-plane exam grouping for HC18."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import (
    COL_FILENAME,
    COL_HC_MM,
    COL_PIXEL_SIZE,
    INTERIM_DATA_DIR,
    RAW_TRAIN_CSV,
    RAW_TRAIN_DIR,
)

# HC size bins aligned with clinical growth stages / Phase 1 presentation
HC_BIN_EDGES = [0.0, 100.0, 150.0, 200.0, 250.0, 300.0, 1000.0]
HC_BIN_LABELS = ["<100", "100-150", "150-200", "200-250", "250-300", ">300"]


def parse_filename_parts(filename: str) -> Tuple[int, int]:
    """Extract sequential numeric prefix and multi-plane scan index from an HC18 filename.

    Examples:
        '12_HC.png'  -> (12, 1)
        '11_2HC.png' -> (11, 2)
        '17_3HC.png' -> (17, 3)
    """
    seq_match = re.match(r"^(\d+)_", filename)
    if not seq_match:
        raise ValueError(f"Unexpected HC18 filename format: {filename}")
    seq_id = int(seq_match.group(1))

    scan_match = re.search(r"_(\d+)HC\.png$", filename)
    scan_num = int(scan_match.group(1)) if scan_match else 1
    return seq_id, scan_num


def annotation_filename_for(ultrasound_filename: str) -> str:
    """Return the corresponding hollow ellipse annotation filename."""
    return ultrasound_filename.replace(".png", "_Annotation.png")


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_pixel_sha256(path: Path) -> Tuple[str, int, int]:
    """Compute SHA-256 hex digest of decoded grayscale pixel array and return (hash, width, height)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
    return digest, int(w), int(h)


def assign_exam_groups(df: pd.DataFrame) -> pd.Series:
    """Reconstruct multi-plane exam groups from sequential HC18 filenames.

    In this HC18 release, filenames are sorted lexicographically ('_2HC', '_3HC', ..., '_HC')
    and assigned sequential numeric prefixes (1..999). Multi-plane scans from the same exam
    appear as consecutive runs of scan_num > 1 ending with scan_num == 1 ('_HC.png').
    If an exam's numeric prefix itself repeats (e.g., original Zenodo '010_HC' & '010_2HC'),
    same-prefix rows are grouped directly.
    """
    # Check whether seq_id already repeats across multi-plane scans
    if df["seq_id"].nunique() < len(df):
        return df["seq_id"].astype(int)

    ordered = df.sort_values("seq_id")
    group_map = {}
    current_group = 1
    for idx, row in ordered.iterrows():
        group_map[idx] = current_group
        if row["scan_num"] == 1:
            current_group += 1

    return pd.Series(group_map).reindex(df.index).astype(int)


def assign_hc_bins(hc_series: pd.Series) -> pd.Series:
    """Assign each sample to a head-circumference size stratum for balanced splitting."""
    return pd.cut(
        hc_series,
        bins=HC_BIN_EDGES,
        labels=HC_BIN_LABELS,
        right=False,
    ).astype(str)


def clean_hc18_metadata(
    raw_train_dir: Path = RAW_TRAIN_DIR,
    raw_train_csv: Path = RAW_TRAIN_CSV,
    output_dir: Path = INTERIM_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit HC18 training files, resolve duplicates, and export cleaned metadata.

    Duplicate detection checks:
      1. Exact raw-byte SHA-256 duplicates of ultrasound images or annotations.
      2. Exact decoded-pixel SHA-256 duplicates of ultrasound images or annotations
         (catches pixel-identical images re-encoded with different PNG headers).

    Within each duplicate group, one canonical record is retained (preferring scan_num == 1
    '_HC.png', then lowest seq_id) and redundant copies are dropped before splitting.

    Returns:
        cleaned_df: Deduplicated metadata DataFrame with exam_group_id and image dimensions.
        duplicate_audit_df: Full audit table of detected duplicate groups and actions taken.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_train_csv)

    required_cols = [COL_FILENAME, COL_PIXEL_SIZE, COL_HC_MM]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in {raw_train_csv}: {missing_cols}")

    if df[required_cols].isna().any().any():
        raise ValueError("Unexpected missing values found in HC18 CSV.")

    # Parse filename components and verify files exist
    parsed = df[COL_FILENAME].apply(parse_filename_parts)
    df["seq_id"] = [p[0] for p in parsed]
    df["scan_num"] = [p[1] for p in parsed]
    df["annotation_filename"] = df[COL_FILENAME].apply(annotation_filename_for)

    widths, heights = [], []
    us_file_hashes, us_pixel_hashes = [], []
    ann_file_hashes, ann_pixel_hashes = [], []

    for _, row in df.iterrows():
        us_path = raw_train_dir / row[COL_FILENAME]
        ann_path = raw_train_dir / row["annotation_filename"]
        if not us_path.is_file():
            raise FileNotFoundError(f"Missing ultrasound image: {us_path}")
        if not ann_path.is_file():
            raise FileNotFoundError(f"Missing annotation image: {ann_path}")

        us_file_hashes.append(file_sha256(us_path))
        ann_file_hashes.append(file_sha256(ann_path))

        us_px_hash, w_us, h_us = image_pixel_sha256(us_path)
        ann_px_hash, w_ann, h_ann = image_pixel_sha256(ann_path)
        if (w_us, h_us) != (w_ann, h_ann):
            raise ValueError(
                f"Dimension mismatch between {row[COL_FILENAME]} ({w_us}x{h_us}) "
                f"and {row['annotation_filename']} ({w_ann}x{h_ann})"
            )

        us_pixel_hashes.append(us_px_hash)
        ann_pixel_hashes.append(ann_px_hash)
        widths.append(w_us)
        heights.append(h_us)

    df["width"] = widths
    df["height"] = heights
    df["us_file_sha256"] = us_file_hashes
    df["us_pixel_sha256"] = us_pixel_hashes
    df["ann_file_sha256"] = ann_file_hashes
    df["ann_pixel_sha256"] = ann_pixel_hashes
    df["exam_group_id"] = assign_exam_groups(df)

    # Group duplicates via Union-Find across ultrasound pixel hash and annotation pixel hash
    parent = {i: i for i in df.index}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for col in ("us_file_sha256", "us_pixel_sha256", "ann_file_sha256", "ann_pixel_sha256"):
        buckets = defaultdict(list)
        for idx, val in df[col].items():
            buckets[val].append(idx)
        for idxs in buckets.values():
            if len(idxs) > 1:
                for other in idxs[1:]:
                    union(idxs[0], other)

    clusters = defaultdict(list)
    for idx in df.index:
        clusters[find(idx)].append(idx)

    dup_clusters = [sorted(idxs) for idxs in clusters.values() if len(idxs) > 1]
    dup_clusters.sort(key=lambda idxs: df.loc[idxs[0], "seq_id"])

    audit_rows = []
    drop_indices = set()

    for group_num, idxs in enumerate(dup_clusters, start=1):
        sub = df.loc[idxs].copy()
        # Deterministic canonical selection: prefer scan_num == 1 ('_HC.png'), then lowest seq_id
        sub_sorted = sub.sort_values(
            by=["scan_num", "seq_id"],
            key=lambda s: (s != 1) if s.name == "scan_num" else s,
        )
        keep_idx = sub_sorted.index[0]

        # Determine match type for clear reporting
        ann_file_exact = sub["ann_file_sha256"].nunique() == 1
        us_file_exact = sub["us_file_sha256"].nunique() == 1
        ann_px_exact = sub["ann_pixel_sha256"].nunique() == 1
        us_px_exact = sub["us_pixel_sha256"].nunique() == 1

        if us_file_exact and ann_file_exact:
            match_type = "exact_file_ultrasound_and_annotation"
        elif ann_file_exact:
            match_type = "exact_file_annotation"
        elif us_px_exact:
            match_type = "exact_pixel_ultrasound"
        elif ann_px_exact:
            match_type = "exact_pixel_annotation"
        else:
            match_type = "mixed_duplicate_cluster"

        # Compute mean abs pixel diff against kept ultrasound image
        kept_img = cv2.imread(
            str(raw_train_dir / df.loc[keep_idx, COL_FILENAME]), cv2.IMREAD_GRAYSCALE
        ).astype(np.float32)

        for idx in idxs:
            action = "keep" if idx == keep_idx else "drop"
            if action == "drop":
                drop_indices.add(idx)
            curr_img = cv2.imread(
                str(raw_train_dir / df.loc[idx, COL_FILENAME]), cv2.IMREAD_GRAYSCALE
            ).astype(np.float32)
            mad = float(np.mean(np.abs(curr_img - kept_img))) if curr_img.shape == kept_img.shape else float("nan")

            audit_rows.append(
                {
                    "duplicate_group": group_num,
                    "match_type": match_type,
                    "action": action,
                    COL_FILENAME: df.loc[idx, COL_FILENAME],
                    "annotation_filename": df.loc[idx, "annotation_filename"],
                    "seq_id": int(df.loc[idx, "seq_id"]),
                    "exam_group_id": int(df.loc[idx, "exam_group_id"]),
                    COL_PIXEL_SIZE: float(df.loc[idx, COL_PIXEL_SIZE]),
                    COL_HC_MM: float(df.loc[idx, COL_HC_MM]),
                    "us_mad_vs_kept": round(mad, 4),
                    "ann_file_sha256_short": df.loc[idx, "ann_file_sha256"][:12],
                    "ann_pixel_sha256_short": df.loc[idx, "ann_pixel_sha256"][:12],
                }
            )

    duplicate_audit_df = pd.DataFrame(audit_rows)
    cleaned_df = df.drop(index=sorted(drop_indices)).reset_index(drop=True)
    cleaned_df["hc_bin"] = assign_hc_bins(cleaned_df[COL_HC_MM])

    duplicate_audit_df.to_csv(output_dir / "duplicate_audit.csv", index=False)
    cleaned_df.to_csv(output_dir / "cleaned_metadata.csv", index=False)

    return cleaned_df, duplicate_audit_df
