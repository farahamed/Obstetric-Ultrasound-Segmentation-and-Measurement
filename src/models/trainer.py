"""Training loop, checkpointing, inference, and learning-curve plotting for baseline models."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RANDOM_SEED, REPORTS_DIR
from src.models.dataset import build_dataloaders
from src.models.losses import DiceBCELoss, batch_dice_and_iou
from src.models.unet import BaselineUNet, count_parameters

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


@dataclass
class TrainConfig:
    """Hyperparameters and experiment metadata saved inside every checkpoint."""

    image_arm: str = "raw_norm"  # 'raw_norm' or 'despeckled'
    input_height: int = 128
    input_width: int = 192
    base_channels: int = 8
    bottleneck_dropout: float = 0.15
    batch_size: int = 16
    epochs: int = 18
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    use_stratum_sampler: bool = True
    early_stopping_patience: int = 7
    seed: int = RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Run one training or validation epoch and return mean loss, Dice, and IoU."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            bs = imgs.size(0)

            logits = model(imgs)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            dice, iou = batch_dice_and_iou(logits, masks)
            total_loss += float(loss.item()) * bs
            total_dice += dice * bs
            total_iou += iou * bs
            total_samples += bs

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "dice": total_dice / denom,
        "iou": total_iou / denom,
    }


def train_baseline_model(
    cfg: TrainConfig,
    processed_dir: Path = PROCESSED_DATA_DIR,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> Tuple[BaselineUNet, pd.DataFrame, Path]:
    """Train BaselineUNet end-to-end, save best checkpoint by validation Dice, and log history."""
    seed_everything(cfg.seed)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (cfg.input_height, cfg.input_width)

    loaders, _, pos_weight = build_dataloaders(
        image_arm=cfg.image_arm,
        batch_size=cfg.batch_size,
        input_size=input_size,
        use_stratum_sampler=cfg.use_stratum_sampler,
        processed_dir=processed_dir,
        seed=cfg.seed,
    )

    model = BaselineUNet(
        in_channels=1,
        out_channels=1,
        base_channels=cfg.base_channels,
        bottleneck_dropout=cfg.bottleneck_dropout,
    ).to(device)

    criterion = DiceBCELoss(
        pos_weight=pos_weight,
        dice_weight=cfg.dice_weight,
        bce_weight=cfg.bce_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-5
    )

    n_params = count_parameters(model)
    print(
        f"\n=== Training BaselineUNet [{cfg.image_arm}] | device={device} | "
        f"input={input_size} | params={n_params:,} | pos_weight={pos_weight:.3f} ==="
    )

    best_val_dice = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    ckpt_path = checkpoints_dir / f"baseline_unet_{cfg.image_arm}.pt"
    history_rows: List[Dict[str, float]] = []

    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.perf_counter()
        train_m = _run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer)
        val_m = _run_epoch(model, loaders["val"], criterion, device, optimizer=None)
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        ep_sec = time.perf_counter() - t_ep

        history_rows.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": round(train_m["loss"], 5),
                "train_dice": round(train_m["dice"], 5),
                "train_iou": round(train_m["iou"], 5),
                "val_loss": round(val_m["loss"], 5),
                "val_dice": round(val_m["dice"], 5),
                "val_iou": round(val_m["iou"], 5),
                "epoch_sec": round(ep_sec, 2),
            }
        )

        improved = val_m["dice"] > best_val_dice
        if improved:
            best_val_dice = val_m["dice"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "pos_weight": pos_weight,
                    "best_epoch": best_epoch,
                    "best_val_dice": best_val_dice,
                    "best_val_iou": val_m["iou"],
                    "best_val_loss": val_m["loss"],
                    "n_params": n_params,
                },
                ckpt_path,
            )
            tag = " *best*"
        else:
            epochs_no_improve += 1
            tag = ""

        print(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} ({ep_sec:4.1f}s) | "
            f"train loss={train_m['loss']:.4f} dice={train_m['dice']*100:.2f}% | "
            f"val loss={val_m['loss']:.4f} dice={val_m['dice']*100:.2f}% iou={val_m['iou']*100:.2f}%{tag}"
        )

        if epochs_no_improve >= cfg.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch}).")
            break

    total_sec = time.perf_counter() - t_start
    print(
        f"Finished [{cfg.image_arm}] in {total_sec:.1f}s | "
        f"Best Val Dice = {best_val_dice*100:.2f}% (epoch {best_epoch}) -> {ckpt_path}"
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(reports_dir / f"training_history_{cfg.image_arm}.csv", index=False)

    # Reload best weights before returning
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model, history_df, ckpt_path


def load_trained_model(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[BaselineUNet, Dict]:
    """Load a saved BaselineUNet checkpoint and return (model, checkpoint_dict)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BaselineUNet(
        in_channels=1,
        out_channels=1,
        base_channels=int(cfg["base_channels"]),
        bottleneck_dropout=float(cfg["bottleneck_dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def plot_training_curves(
    histories: Dict[str, pd.DataFrame],
    save_path: Path = FIGURES_DIR / "baseline_training_curves.png",
) -> Path:
    """Plot training/validation Dice + Weighted BCE loss and Dice score curves across arms."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    palette = {"raw_norm": "#2b8cbe", "despeckled": "#d95f02"}
    labels = {"raw_norm": "Raw Normalized", "despeckled": "Despeckled (Perona-Malik)"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for arm, hdf in histories.items():
        col = palette.get(arm, "#444444")
        lbl = labels.get(arm, arm)
        ep = hdf["epoch"].values

        axes[0].plot(ep, hdf["train_loss"], linestyle="--", alpha=0.6, color=col, label=f"{lbl} (Train)")
        axes[0].plot(ep, hdf["val_loss"], linewidth=2.0, color=col, label=f"{lbl} (Val)")

        axes[1].plot(ep, hdf["train_dice"] * 100.0, linestyle="--", alpha=0.6, color=col, label=f"{lbl} (Train)")
        axes[1].plot(ep, hdf["val_dice"] * 100.0, linewidth=2.0, color=col, label=f"{lbl} (Val)")

        best_idx = int(hdf["val_dice"].idxmax())
        axes[1].scatter(
            [hdf.loc[best_idx, "epoch"]],
            [hdf.loc[best_idx, "val_dice"] * 100.0],
            color=col,
            s=45,
            zorder=5,
        )

    axes[0].set_title("Combined Loss (Soft Dice + Weighted BCE)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Segmentation Overlap (Dice Score %)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Similarity Coefficient (%)")
    axes[1].legend(fontsize=8.5, loc="lower right")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Baseline U-Net Training & Validation Learning Curves",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


@torch.no_grad()
def plot_validation_predictions(
    models: Dict[str, BaselineUNet],
    input_size: Tuple[int, int] = (128, 192),
    processed_dir: Path = PROCESSED_DATA_DIR,
    save_path: Path = FIGURES_DIR / "baseline_val_predictions.png",
) -> Path:
    """Plot qualitative validation predictions across 1st, 2nd, and 3rd trimester scans."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = pd.read_csv(processed_dir / "processed_metadata.csv")
    val_df = full_df[full_df["split"] == "val"].sort_values("head circumference (mm)").reset_index(drop=True)

    # Pick 1st trimester (low HC), 2nd trimester (median HC), and 3rd trimester (high HC)
    sample_indices = [len(val_df) // 10, len(val_df) // 2, int(len(val_df) * 0.90)]
    arms = list(models.keys())
    n_cols = 2 + len(arms)

    fig, axes = plt.subplots(len(sample_indices), n_cols, figsize=(3.8 * n_cols, 3.2 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    arm_titles = {
        "raw_norm": "Pred: Raw Normalized",
        "despeckled": "Pred: Despeckled",
    }

    import cv2

    for r, idx in enumerate(sample_indices):
        row = val_df.iloc[idx]
        th, tw = input_size
        raw_img = cv2.resize(
            cv2.imread(str(processed_dir / row["raw_norm_relpath"]), cv2.IMREAD_GRAYSCALE),
            (tw, th),
            interpolation=cv2.INTER_AREA,
        )
        gt_mask = (
            cv2.resize(
                cv2.imread(str(processed_dir / row["mask_relpath"]), cv2.IMREAD_GRAYSCALE),
                (tw, th),
                interpolation=cv2.INTER_NEAREST,
            )
            > 127
        ).astype(np.float32)

        axes[r, 0].imshow(raw_img, cmap="gray")
        axes[r, 0].set_ylabel(
            f"{row['filename']}\nHC={row['head circumference (mm)']:.1f} mm",
            fontsize=9,
            rotation=0,
            labelpad=58,
            va="center",
        )
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

        axes[r, 1].imshow(raw_img, cmap="gray")
        axes[r, 1].contour(gt_mask, levels=[0.5], colors="#00e676", linewidths=1.6)
        axes[r, 1].axis("off")

        if r == 0:
            axes[r, 0].set_title("Input Ultrasound", fontweight="bold", fontsize=10.5)
            axes[r, 1].set_title("Ground-Truth Contour", fontweight="bold", fontsize=10.5)

        for c_off, arm in enumerate(arms):
            rel = row["raw_norm_relpath"] if arm == "raw_norm" else row["despeckled_relpath"]
            im_in = cv2.resize(
                cv2.imread(str(processed_dir / rel), cv2.IMREAD_GRAYSCALE),
                (tw, th),
                interpolation=cv2.INTER_AREA,
            )
            t_in = torch.from_numpy(im_in.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            model = models[arm].to(device)
            model.eval()
            prob = torch.sigmoid(model(t_in))[0, 0].cpu().numpy()
            pred = (prob >= 0.5).astype(np.float32)
            dice = (2.0 * (pred * gt_mask).sum() + 1e-7) / (pred.sum() + gt_mask.sum() + 1e-7)

            ax = axes[r, 2 + c_off]
            ax.imshow(im_in, cmap="gray")
            ax.contour(gt_mask, levels=[0.5], colors="#00e676", linewidths=1.3)
            ax.contour(pred, levels=[0.5], colors="#ff5252", linewidths=1.3, linestyles="--")
            ax.text(
                6,
                16,
                f"DSC={dice*100:.1f}%",
                color="yellow",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, pad=1.5),
            )
            ax.axis("off")
            if r == 0:
                ax.set_title(arm_titles.get(arm, arm), fontweight="bold", fontsize=10.5)

    fig.suptitle(
        "Baseline U-Net Validation Predictions (Green = Ground Truth, Red Dashed = Prediction)",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
