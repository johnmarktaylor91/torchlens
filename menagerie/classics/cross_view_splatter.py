"""Cross-View Splatter: feed-forward pixel-aligned Gaussian-splat predictor.

Niantic Spatial, 2026.
Paper: https://arxiv.org/abs/2605.19656

Cross-View Splatter predicts a 3D Gaussian-splat scene in a single forward pass
by fusing a ground-level image with a georeferenced satellite / BEV image.  Its
DISTINCTIVE mechanism:

  - Two input branches: a GROUND image and a SATELLITE (bird's-eye) image, each
    encoded by a small U-Net / conv encoder.
  - A CROSS-BRANCH fusion aligns the ground and bird's-eye feature maps (here a
    feature concatenation + fusion conv; cross-view attention is the alternative
    described in the paper).
  - A per-pixel head outputs pixel-aligned 3D Gaussian PARAMETER maps:
    xyz offset (3) + scale (3) + rotation quaternion (4) + opacity (1) +
    color (3) = 14 channels per pixel.

RASTERIZER-FREE FAITHFUL CORE: the final gsplat CUDA rasterization (turning the
Gaussian params into an image) is out of scope -- it is a fixed differentiable
renderer, not a learned module.  We faithfully reimplement the ENCODER that
predicts the per-pixel Gaussian parameters and STOP at the param map (we do not
rasterize).  The architecture captured is exactly the feed-forward
pixel-aligned Gaussian predictor with ground+satellite fusion.

Modest width; forward() returns the 14-channel Gaussian-param map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv -> BN -> ReLU x2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UNetEncoder(nn.Module):
    """Small U-Net encoder: per-branch features, returns full-res feature map."""

    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, base)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(base, base * 2)
        self.down2 = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out_ch = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return d1  # (B, base, H, W)


class CrossViewSplatter(nn.Module):
    """Cross-View Splatter: ground+satellite encoders -> Gaussian-param map."""

    def __init__(self, base: int = 32, gaussian_channels: int = 14) -> None:
        super().__init__()
        self.ground_encoder = _UNetEncoder(3, base)
        self.satellite_encoder = _UNetEncoder(3, base)
        # Cross-branch fusion (concat ground + bird's-eye features + fuse conv).
        self.fusion = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        # Per-pixel Gaussian-parameter head:
        # xyz(3) + scale(3) + rot quat(4) + opacity(1) + color(3) = 14.
        self.gaussian_head = nn.Conv2d(base, gaussian_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 3, H, W) -- [ground, satellite] stacked on dim 1.
        ground = x[:, 0]
        satellite = x[:, 1]
        gf = self.ground_encoder(ground)  # (B, base, H, W)
        sf = self.satellite_encoder(satellite)  # (B, base, H, W)
        # Align satellite features to ground resolution if needed.
        if sf.shape[-2:] != gf.shape[-2:]:
            sf = F.interpolate(sf, size=gf.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat([gf, sf], dim=1))
        params = self.gaussian_head(fused)  # (B, 14, H, W)
        # Return the per-pixel Gaussian-parameter map (no rasterization).
        return params


def build_cross_view_splatter() -> nn.Module:
    """Build Cross-View Splatter (rasterizer-free; Gaussian-param map output)."""
    return CrossViewSplatter(base=32, gaussian_channels=14)


def example_input() -> torch.Tensor:
    """Example stacked ``(1, 2, 3, 32, 32)`` = [ground, satellite] views."""
    return torch.randn(1, 2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Cross-View Splatter (feed-forward pixel-aligned Gaussian predictor)",
        "build_cross_view_splatter",
        "example_input",
        "2026",
        "DC",
    ),
]
