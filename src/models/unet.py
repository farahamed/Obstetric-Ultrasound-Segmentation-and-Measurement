"""Lightweight 2D Baseline U-Net architecture for fetal head ultrasound segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d 3x3 -> BatchNorm2d -> ReLU) blocks."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """MaxPool2d downsampling followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Bilinear upsampling + skip concatenation + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    """Lightweight 4-stage 2D U-Net for binary fetal head segmentation.

    Default channel width ``base_channels=16`` produces ~1.94M parameters (channels:
    16 -> 32 -> 64 -> 128 -> bottleneck 256), offering fast convergence on CPU or GPU
    while capturing multi-scale cranial context across all three trimesters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        bottleneck_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.enc4 = DownBlock(c * 4, c * 8)

        self.bottleneck = DownBlock(c * 8, c * 16, dropout=bottleneck_dropout)

        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)

        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck(s4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
