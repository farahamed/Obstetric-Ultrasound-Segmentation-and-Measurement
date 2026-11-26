"""Run postprocessing (ellipse fitting + Ramanujan HC) and biometric evaluation on Test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import (
    COL_FILENAME,
    COL_HC_MM,
    COL_PIXEL_SIZE,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_TRAIN_DIR,
    REPORTS_DIR,
)
from src.evaluation.biometric_metrics import (
    assign_trimester_group,
    compute_sample_biometric_metrics,
    summarize_biometric_results,
    summarize_by_trimester,
)
from src.models.trainer import CHECKPOINTS_DIR, load_trained_model
from src.postprocessing.ellipse_fitting import fit_ellipse_and_measure_hc


@torch.no_grad()
def evaluate_arm_biometrics(
    image_arm: str,
    split: str = "test",
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run checkpoint inference, upsample probabilities to native resolution, fit ellipses, and score.

    Upsampling predicted probability maps bilinearly back to each scan's native original
    resolution ``(orig_height, orig_width)`` allows direct comparison against the exact
    full-resolution ground-truth masks using native isotropic ``pixel size (mm/px)``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoints_dir / f"baseline_unet_{image_arm}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, ckpt = load_trained_model(ckpt_path, device=device)
    cfg = ckpt["config"]
    in_h, in_w = int(cfg["input_height"]), int(cfg["input_width"])

    full_df = pd.read_csv(processed_dir / "processed_metadata.csv")
    split_df = full_df[full_df["split"] == split].reset_index(drop=True)
    img_col = "raw_norm_relpath" if image_arm == "raw_norm" else "despeckled_relpath"
    interim_masks_dir = interim_dir / "masks"

    rows = []
    for _, row in split_df.iterrows():
        img_proc = cv2.imread(str(processed_dir / row[img_col]), cv2.IMREAD_GRAYSCALE)
        img_net = cv2.resize(img_proc, (in_w, in_h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img_net.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        prob_net = torch.sigmoid(model(x))[0, 0].cpu().numpy()

        # Upsample soft probabilities back to native resolution for exact biometric measurement
        orig_w, orig_h = int(row["orig_width"]), int(row["orig_height"])
        prob_native = cv2.resize(prob_net, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        gt_native = (
            cv2.imread(str(interim_masks_dir / row["mask_filename"]), cv2.IMREAD_GRAYSCALE) > 127
        ).astype(np.uint8)
        native_ps = float(row[COL_PIXEL_SIZE])
        gt_hc_mm = float(row[COL_HC_MM])

        meas = fit_ellipse_and_measure_hc(
            prob_native,
            pixel_size_x_mm=native_ps,
            pixel_size_y_mm=native_ps,
            threshold=threshold,
            morph_close_kernel=7,
        )

        metrics = compute_sample_biometric_metrics(
            pred_clean_mask=meas.clean_mask,
            pred_ellipse_mask=meas.ellipse_mask,
            gt_mask=gt_native,
            pred_hc_mm=meas.hc_mm,
            gt_hc_mm=gt_hc_mm,
            pixel_size_x_mm=native_ps,
            pixel_size_y_mm=native_ps,
        )

        rows.append(
            {
                COL_FILENAME: row[COL_FILENAME],
                "mask_filename": row["mask_filename"],
                "exam_group_id": int(row["exam_group_id"]),
                "trimester": assign_trimester_group(gt_hc_mm),
                "orig_width": orig_w,
                "orig_height": orig_h,
                COL_PIXEL_SIZE: native_ps,
                "gt_hc_mm": gt_hc_mm,
                "pred_hc_mm": round(meas.hc_mm, 3),
                "semi_major_mm": round(meas.semi_major_mm, 3),
                "semi_minor_mm": round(meas.semi_minor_mm, 3),
                "ellipse_fit_ok": bool(meas.success),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def plot_ellipse_overlays(
    pred_dfs: Dict[str, pd.DataFrame],
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    interim_dir: Path = INTERIM_DATA_DIR,
    raw_train_dir: Path = RAW_TRAIN_DIR,
    save_path: Path = FIGURES_DIR / "postprocessing_ellipse_overlays.png",
) -> Path:
    """Plot qualitative Test-set ellipse overlays across trimesters plus largest-error case."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_arm = "raw_norm" if "raw_norm" in pred_dfs else next(iter(pred_dfs))
    ref_df = pred_dfs[ref_arm].sort_values("gt_hc_mm").reset_index(drop=True)

    # Select representative 1st, 2nd, 3rd trimester scans + hardest (max AD) scan
    chosen_names = [
        ref_df.iloc[len(ref_df) // 12][COL_FILENAME],
        ref_df.iloc[len(ref_df) // 2][COL_FILENAME],
        ref_df.iloc[int(len(ref_df) * 0.90)][COL_FILENAME],
        ref_df.sort_values("ad_mm", ascending=False).iloc[0][COL_FILENAME],
    ]
    row_tags = ["1st Trimester", "2nd Trimester", "3rd Trimester", "Highest-Error Case"]

    full_meta = pd.read_csv(processed_dir / "processed_metadata.csv").set_index(COL_FILENAME)
    arms = list(pred_dfs.keys())
    loaded = {arm: load_trained_model(checkpoints_dir / f"baseline_unet_{arm}.pt", device=device) for arm in arms}

    n_cols = 1 + len(arms)
    fig, axes = plt.subplots(len(chosen_names), n_cols, figsize=(4.8 * n_cols, 3.4 * len(chosen_names)))
    arm_labels = {"raw_norm": "Raw Normalized + Ellipse Fit", "despeckled": "Despeckled + Ellipse Fit"}

    for r, (fname, rtag) in enumerate(zip(chosen_names, row_tags)):
        meta = full_meta.loc[fname]
        orig_w, orig_h = int(meta["orig_width"]), int(meta["orig_height"])
        ps = float(meta[COL_PIXEL_SIZE])
        gt_hc = float(meta[COL_HC_MM])

        raw_us = cv2.imread(str(raw_train_dir / fname), cv2.IMREAD_GRAYSCALE)
        gt_mask = (
            cv2.imread(str(interim_dir / "masks" / meta["mask_filename"]), cv2.IMREAD_GRAYSCALE) > 127
        ).astype(np.uint8)

        axes[r, 0].imshow(raw_us, cmap="gray")
        axes[r, 0].contour(gt_mask, levels=[0.5], colors="#00e676", linewidths=1.7)
        axes[r, 0].set_ylabel(
            f"{rtag}\n{fname}\nGT HC={gt_hc:.1f} mm",
            fontsize=9,
            rotation=0,
            labelpad=65,
            va="center",
        )
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Ultrasound + Ground-Truth (Green)", fontweight="bold", fontsize=10.5)

        for c_off, arm in enumerate(arms):
            model, ckpt = loaded[arm]
            cfg = ckpt["config"]
            in_h, in_w = int(cfg["input_height"]), int(cfg["input_width"])
            img_col = "raw_norm_relpath" if arm == "raw_norm" else "despeckled_relpath"
            img_proc = cv2.imread(str(processed_dir / meta[img_col]), cv2.IMREAD_GRAYSCALE)
            img_net = cv2.resize(img_proc, (in_w, in_h), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(img_net.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_net = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            prob_native = cv2.resize(prob_net, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            meas = fit_ellipse_and_measure_hc(prob_native, pixel_size_x_mm=ps, morph_close_kernel=7)
            row_stats = pred_dfs[arm].set_index(COL_FILENAME).loc[fname]

            ax = axes[r, 1 + c_off]
            ax.imshow(raw_us, cmap="gray")
            ax.contour(gt_mask, levels=[0.5], colors="#00e676", linewidths=1.4)
            ax.contour(meas.clean_mask, levels=[0.5], colors="#00e5ff", linewidths=1.0, linestyles=":")
            ax.contour(meas.ellipse_mask, levels=[0.5], colors="#ff5252", linewidths=1.5, linestyles="--")
            ax.text(
                0.02,
                0.96,
                f"Pred={row_stats['pred_hc_mm']:.1f} mm (AD={row_stats['ad_mm']:.2f} mm)\n"
                f"DSC={row_stats['dsc_ellipse_pct']:.2f}% | HD={row_stats['hd_ellipse_mm']:.2f} mm",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="yellow",
                fontsize=8.2,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.65, pad=2.5),
            )
            ax.axis("off")
            if r == 0:
                ax.set_title(arm_labels.get(arm, arm), fontweight="bold", fontsize=10.5)

    fig.suptitle(
        "Test-Set Postprocessing: Ground Truth (Green), Cleaned Mask (Cyan), Fitted Ellipse (Red Dashed)",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_hc_agreement_and_bland_altman(
    pred_dfs: Dict[str, pd.DataFrame],
    save_path: Path = FIGURES_DIR / "biometric_bland_altman_and_scatter.png",
) -> Path:
    """Plot HC Scatter (Predicted vs. Ground Truth) and Bland-Altman agreement for each arm."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arms = list(pred_dfs.keys())
    arm_titles = {"raw_norm": "Raw Normalized", "despeckled": "Despeckled (Perona-Malik)"}
    arm_colors = {"raw_norm": "#2b8cbe", "despeckled": "#d95f02"}

    fig, axes = plt.subplots(len(arms), 2, figsize=(12.8, 4.7 * len(arms)))
    if len(arms) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, arm in enumerate(arms):
        df = pred_dfs[arm]
        gt = df["gt_hc_mm"].values
        pr = df["pred_hc_mm"].values
        diff = pr - gt
        avg = 0.5 * (pr + gt)
        md = float(diff.mean())
        sd = float(diff.std())
        ad = float(np.abs(diff).mean())
        corr = float(np.corrcoef(gt, pr)[0, 1])
        col = arm_colors.get(arm, "#2b8cbe")
        lbl = arm_titles.get(arm, arm)

        # Left: Scatter plot
        ax0 = axes[r, 0]
        ax0.scatter(gt, pr, s=22, alpha=0.65, color=col, edgecolors="white", linewidth=0.3)
        lims = [40, 360]
        ax0.plot(lims, lims, "k--", linewidth=1.1, label="Identity (y = x)")
        ax0.set_xlim(lims)
        ax0.set_ylim(lims)
        ax0.set_title(
            f"{lbl} — HC Agreement (r = {corr:.4f}, AD = {ad:.2f} mm)",
            fontweight="bold",
        )
        ax0.set_xlabel("Ground-Truth HC (mm)")
        ax0.set_ylabel("Predicted HC (mm)")
        ax0.legend(loc="upper left")
        ax0.grid(alpha=0.3)

        # Right: Bland-Altman plot
        ax1 = axes[r, 1]
        ax1.scatter(avg, diff, s=22, alpha=0.65, color=col, edgecolors="white", linewidth=0.3)
        ax1.axhline(md, color="#d73027", linewidth=1.5, label=f"MD = {md:+.2f} mm")
        ax1.axhline(md + 1.96 * sd, color="#444444", linestyle="--", linewidth=1.1, label=f"+1.96 SD = {md + 1.96*sd:+.2f} mm")
        ax1.axhline(md - 1.96 * sd, color="#444444", linestyle="--", linewidth=1.1, label=f"-1.96 SD = {md - 1.96*sd:+.2f} mm")
        ax1.axhline(0.0, color="gray", linestyle=":", linewidth=0.9)
        ax1.set_title(f"{lbl} — Bland-Altman Plot (n = {len(df)})", fontweight="bold")
        ax1.set_xlabel("Mean of Predicted & Ground-Truth HC (mm)")
        ax1.set_ylabel("Predicted − Ground-Truth HC (mm)")
        ax1.legend(loc="upper right", fontsize=8.5)
        ax1.grid(alpha=0.3)

    fig.suptitle(
        "Held-Out Test Set Biometric HC Measurement Agreement",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_all(split: str = "test") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Run ellipse fitting and biometric evaluation on both trained arms."""
    arms = ["raw_norm", "despeckled"]
    pred_dfs: Dict[str, pd.DataFrame] = {}
    summary_rows = []
    trimester_tables = []

    for arm in arms:
        print(f"Evaluating [{arm}] on '{split}' split (native resolution + cv2.fitEllipse + Ramanujan HC)...")
        pdf = evaluate_arm_biometrics(image_arm=arm, split=split)
        pred_dfs[arm] = pdf
        pdf.to_csv(REPORTS_DIR / f"{split}_biometric_predictions_{arm}.csv", index=False)
        summary_rows.append(summarize_biometric_results(pdf, label=f"BaselineUNet ({arm})"))
        trimester_tables.append(summarize_by_trimester(pdf, label=f"BaselineUNet ({arm})"))

    summary_df = pd.DataFrame(summary_rows)
    trimester_df = pd.concat(trimester_tables, ignore_index=True)

    summary_df.to_csv(REPORTS_DIR / f"{split}_biometric_summary.csv", index=False)
    trimester_df.to_csv(REPORTS_DIR / f"{split}_biometric_by_trimester.csv", index=False)

    print(f"\n=== Biometric Summary ({split.title()} Split) ===")
    print(summary_df.to_string(index=False))
    print(f"\n=== Biometric Breakdown by Trimester ({split.title()} Split) ===")
    print(trimester_df.to_string(index=False))

    fig1 = plot_ellipse_overlays(pred_dfs)
    fig2 = plot_hc_agreement_and_bland_altman(pred_dfs)
    print(f"\nSaved ellipse overlays: {fig1}")
    print(f"Saved Bland-Altman & scatter plot: {fig2}")

    return pred_dfs, summary_df, trimester_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run postprocessing and biometric evaluation.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()
    run_all(split=args.split)
