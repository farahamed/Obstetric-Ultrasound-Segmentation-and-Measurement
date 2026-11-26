"""End-to-end test-set evaluation: pixel metrics, literature comparison, and error analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import REPORTS_DIR
from src.evaluation.pixel_metrics import (
    PixelEvalResult,
    evaluate_arm_pixel_metrics,
    plot_confusion_matrices,
    plot_literature_comparison,
    plot_roc_curves,
    results_to_dataframe,
)
from src.postprocessing.run_postprocessing import run_all as run_biometrics


def build_literature_comparison_table(
    pixel_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pixel and biometric results with van den Heuvel (2018) and Sobhaninia (2019)."""
    raw_px = pixel_df[pixel_df["image_arm"] == "raw_norm"].iloc[0]
    dsp_px = pixel_df[pixel_df["image_arm"] == "despeckled"].iloc[0]
    raw_bio = biometric_df[biometric_df["model"].str.contains("raw_norm")].iloc[0]
    dsp_bio = biometric_df[biometric_df["model"].str.contains("despeckled")].iloc[0]

    rows = [
        {
            "method": "van den Heuvel et al. (2018)",
            "pipeline": "Haar features + Random Forest + Hough + Ellipse fit",
            "eval_set": "Official HC18 Test (n=335)",
            "dsc_pct": 97.00,
            "hd_mm": None,
            "ad_mm": 2.80,
            "md_mm": None,
            "sensitivity_pct": None,
            "specificity_pct": None,
            "auc_roc": None,
        },
        {
            "method": "Sobhaninia et al. (2019)",
            "pipeline": "Multi-task Link-Net CNN + Ellipse Tuner",
            "eval_set": "Official HC18 Test (n=335)",
            "dsc_pct": 96.84,
            "hd_mm": None,
            "ad_mm": 2.12,
            "md_mm": None,
            "sensitivity_pct": None,
            "specificity_pct": None,
            "auc_roc": None,
        },
        {
            "method": "Baseline U-Net (Raw Normalized)",
            "pipeline": "4-stage U-Net (492k params) + cv2.fitEllipse",
            "eval_set": f"Internal Test (n={int(raw_bio['n_images'])})",
            "dsc_pct": float(raw_bio["dsc_ellipse_mean_pct"]),
            "hd_mm": float(raw_bio["hd_ellipse_mean_mm"]),
            "ad_mm": float(raw_bio["ad_mean_mm"]),
            "md_mm": float(raw_bio["md_mean_mm"]),
            "sensitivity_pct": float(raw_px["sensitivity_pct"]),
            "specificity_pct": float(raw_px["specificity_pct"]),
            "auc_roc": float(raw_px["auc_roc"]),
        },
        {
            "method": "Baseline U-Net (Despeckled)",
            "pipeline": "Perona-Malik + 4-stage U-Net + cv2.fitEllipse",
            "eval_set": f"Internal Test (n={int(dsp_bio['n_images'])})",
            "dsc_pct": float(dsp_bio["dsc_ellipse_mean_pct"]),
            "hd_mm": float(dsp_bio["hd_ellipse_mean_mm"]),
            "ad_mm": float(dsp_bio["ad_mean_mm"]),
            "md_mm": float(dsp_bio["md_mean_mm"]),
            "sensitivity_pct": float(dsp_px["sensitivity_pct"]),
            "specificity_pct": float(dsp_px["specificity_pct"]),
            "auc_roc": float(dsp_px["auc_roc"]),
        },
    ]
    return pd.DataFrame(rows)


def build_error_analysis_table(
    raw_pred_df: pd.DataFrame,
    dsp_pred_df: pd.DataFrame,
    top_k: int = 8,
) -> pd.DataFrame:
    """Identify the highest-AD test cases across both arms for qualitative failure analysis."""
    merged = raw_pred_df[
        ["filename", "trimester", "gt_hc_mm", "pred_hc_mm", "ad_mm", "md_mm", "dsc_ellipse_pct", "hd_ellipse_mm"]
    ].merge(
        dsp_pred_df[
            ["filename", "pred_hc_mm", "ad_mm", "md_mm", "dsc_ellipse_pct", "hd_ellipse_mm"]
        ],
        on="filename",
        suffixes=("_raw", "_dsp"),
    )
    merged["max_ad_mm"] = merged[["ad_mm_raw", "ad_mm_dsp"]].max(axis=1)
    return merged.sort_values("max_ad_mm", ascending=False).head(top_k).reset_index(drop=True)


def run_all(split: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute full pixel-level evaluation, literature comparison, error analysis, and plots."""
    # Ensure biometric summary CSVs exist
    bio_sum_path = REPORTS_DIR / f"{split}_biometric_summary.csv"
    tri_path = REPORTS_DIR / f"{split}_biometric_by_trimester.csv"
    raw_pred_path = REPORTS_DIR / f"{split}_biometric_predictions_raw_norm.csv"
    dsp_pred_path = REPORTS_DIR / f"{split}_biometric_predictions_despeckled.csv"

    if not (bio_sum_path.is_file() and tri_path.is_file() and raw_pred_path.is_file() and dsp_pred_path.is_file()):
        run_biometrics(split=split)

    biometric_df = pd.read_csv(bio_sum_path)
    trimester_df = pd.read_csv(tri_path)
    raw_pred_df = pd.read_csv(raw_pred_path)
    dsp_pred_df = pd.read_csv(dsp_pred_path)

    pixel_results: Dict[str, PixelEvalResult] = {}
    for arm in ("raw_norm", "despeckled"):
        print(f"Computing native-resolution pixel classification metrics & ROC for [{arm}] on '{split}'...")
        pixel_results[arm] = evaluate_arm_pixel_metrics(image_arm=arm, split=split)

    pixel_df = results_to_dataframe(pixel_results)
    lit_df = build_literature_comparison_table(pixel_df, biometric_df)
    err_df = build_error_analysis_table(raw_pred_df, dsp_pred_df, top_k=8)

    pixel_df.to_csv(REPORTS_DIR / f"{split}_pixel_metrics_summary.csv", index=False)
    lit_df.to_csv(REPORTS_DIR / f"{split}_literature_comparison.csv", index=False)
    err_df.to_csv(REPORTS_DIR / f"{split}_error_analysis.csv", index=False)

    print(f"\n=== Pixel-Level Classification Metrics ({split.title()} Split) ===")
    print(pixel_df.to_string(index=False))

    print("\n=== Full Comparison vs. Reference Literature Baselines ===")
    print(lit_df.to_string(index=False))

    print("\n=== Top-8 Largest-Error Test Cases (Error Analysis) ===")
    print(err_df.to_string(index=False))

    fig_cm = plot_confusion_matrices(pixel_results)
    fig_roc = plot_roc_curves(pixel_results)
    fig_lit = plot_literature_comparison(biometric_df, trimester_df)
    print(f"\nSaved confusion matrix: {fig_cm}")
    print(f"Saved ROC curves: {fig_roc}")
    print(f"Saved literature comparison: {fig_lit}")

    return pixel_df, lit_df, err_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pixel & literature evaluation.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()
    run_all(split=args.split)
