"""Surface Normals Estimation with Uncertainty (NU-Net).

Bae et al., ICCV 2021.
Paper: https://arxiv.org/abs/2106.10457
Source: https://github.com/baegwangbin/surface_normal_uncertainty

This method estimates per-pixel surface normals together with aleatoric
uncertainty (concentration parameter kappa of the von Mises-Fisher distribution
on the unit sphere).

Architecture:
  1. Encoder-decoder backbone (here: compact U-Net style) -- shared features.
  2. Normal head -- outputs 3-channel unnormalized surface normal, L2-normalized
     to the unit sphere at inference.
  3. Uncertainty head -- outputs a scalar kappa >= 0 per pixel via softplus;
     higher kappa = lower uncertainty.

The key distinctive primitive is the dual-head structure where one head predicts
the direction (unit normal) and the other predicts confidence (kappa), trained
jointly under a negative log-likelihood of the von Mises-Fisher distribution.

Compact config: 3-level encoder-decoder, base_ch=32, input (1,3,64,64).
Outputs: normals (1,3,H,W) + kappa (1,1,H,W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Compact U-Net backbone
# ---------------------------------------------------------------------------


class UNetBackbone(nn.Module):
    """3-level U-Net backbone with skip connections; returns final decoder features."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        bc = base_ch
        # Encoder
        self.enc1 = DoubleConv(in_ch, bc)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(bc, bc * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(bc * 2, bc * 4)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(bc * 4, bc * 8)
        # Decoder
        self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, 2)
        self.dec3 = DoubleConv(bc * 8, bc * 4)
        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, 2)
        self.dec2 = DoubleConv(bc * 4, bc * 2)
        self.up1 = nn.ConvTranspose2d(bc * 2, bc, 2, 2)
        self.dec1 = DoubleConv(bc * 2, bc)
        self.out_ch = bc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1  # (B, bc, H, W)


# ---------------------------------------------------------------------------
# Full model: shared backbone + dual heads
# ---------------------------------------------------------------------------


class SurfaceNormalUncertainty(nn.Module):
    """Surface normal estimation with aleatoric uncertainty.

    Input:  (B, 3, H, W) RGB image.
    Output: (normal (B, 3, H, W) L2-normalized, kappa (B, 1, H, W) >= 0)
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.backbone = UNetBackbone(in_ch=3, base_ch=base_ch)
        feat_ch = base_ch

        # Normal head: predict 3-ch direction, normalize to unit sphere
        self.normal_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 3, 1),
        )

        # Uncertainty head: predict kappa >= 0 (softplus-activated scalar)
        self.kappa_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)

        # Normal: unnormalized -> L2 normalize to unit sphere
        normal_raw = self.normal_head(feat)
        normal = F.normalize(normal_raw, p=2, dim=1)

        # Kappa: uncertainty concentration (>0 via softplus)
        kappa = F.softplus(self.kappa_head(feat))

        return normal, kappa


# ---------------------------------------------------------------------------
# Builder + example
# ---------------------------------------------------------------------------


def build_surface_normal_uncertainty() -> nn.Module:
    """Build compact NU-Net surface normal + uncertainty estimator."""
    return SurfaceNormalUncertainty(base_ch=32)


def example_input_snu() -> torch.Tensor:
    """(1, 3, 64, 64) RGB image."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Surface Normal Uncertainty (dual-head U-Net: normal + kappa concentration, von Mises-Fisher)",
        "build_surface_normal_uncertainty",
        "example_input_snu",
        "2021",
        "DC",
    ),
]
