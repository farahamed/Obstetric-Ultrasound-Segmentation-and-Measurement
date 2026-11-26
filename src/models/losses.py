"""Imbalance-aware segmentation losses (Soft Dice + Weighted BCE) and batch overlap metrics."""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """Differentiable per-sample Soft Dice Loss operating on raw logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Soft Dice Loss + Class-Weighted BCEWithLogitsLoss.

    Addresses pixel-level foreground/background class imbalance (~30.6% head vs. 69.4%
    background in HC18 training masks) by pairing overlap-driven Soft Dice with
    positive-weighted binary cross-entropy.
    """

    def __init__(
        self,
        pos_weight: float = 2.269,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice = SoftDiceLoss(smooth=smooth)
        self.register_buffer("pos_weight_tensor", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight_tensor
        )
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def batch_dice_and_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[float, float]:
    """Compute mean per-image binary Dice (DSC) and IoU over a batch of logits."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    dims = (1, 2, 3)
    inter = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    targ_sum = torch.sum(targets, dim=dims)

    dice = ((2.0 * inter + eps) / (pred_sum + targ_sum + eps)).mean().item()
    union = pred_sum + targ_sum - inter
    iou = ((inter + eps) / (union + eps)).mean().item()
    return float(dice), float(iou)
