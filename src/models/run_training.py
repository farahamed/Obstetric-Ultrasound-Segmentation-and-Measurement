"""CLI entry point for training the Baseline U-Net on raw-normalized and/or despeckled inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import RANDOM_SEED, REPORTS_DIR
from src.models.trainer import (
    TrainConfig,
    plot_training_curves,
    plot_validation_predictions,
    train_baseline_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Baseline U-Net for fetal head ultrasound segmentation."
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="both",
        choices=["raw_norm", "despeckled", "both"],
        help="Input preprocessing arm to train on (default: both).",
    )
    parser.add_argument("--epochs", type=int, default=18, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--base-channels", type=int, default=8, help="Base U-Net channel width.")
    parser.add_argument("--height", type=int, default=128, help="Network input height.")
    parser.add_argument("--width", type=int, default=192, help="Network input width.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial AdamW learning rate.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    return parser.parse_args()


def run_all(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    arms = ["raw_norm", "despeckled"] if args.arm == "both" else [args.arm]
    trained_models = {}
    histories = {}
    summary_rows = []

    for arm in arms:
        cfg = TrainConfig(
            image_arm=arm,
            input_height=args.height,
            input_width=args.width,
            base_channels=args.base_channels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            early_stopping_patience=args.patience,
            seed=args.seed,
        )
        model, hdf, ckpt_path = train_baseline_model(cfg)
        trained_models[arm] = model
        histories[arm] = hdf

        best_idx = int(hdf["val_dice"].idxmax())
        summary_rows.append(
            {
                "image_arm": arm,
                "best_epoch": int(hdf.loc[best_idx, "epoch"]),
                "best_val_dice_pct": round(float(hdf.loc[best_idx, "val_dice"]) * 100.0, 2),
                "best_val_iou_pct": round(float(hdf.loc[best_idx, "val_iou"]) * 100.0, 2),
                "best_val_loss": round(float(hdf.loc[best_idx, "val_loss"]), 5),
                "checkpoint": str(ckpt_path.name),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORTS_DIR / "baseline_training_summary.csv", index=False)

    print("\n=== Baseline Training Summary (Validation Split) ===")
    print(summary_df.to_string(index=False))

    curve_fig = plot_training_curves(histories)
    pred_fig = plot_validation_predictions(
        trained_models, input_size=(args.height, args.width)
    )
    print(f"\nSaved learning curves: {curve_fig}")
    print(f"Saved qualitative predictions: {pred_fig}")


if __name__ == "__main__":
    run_all()
