"""SHM: Semantic Human Matting.

Chen et al., ACM MM 2018.
Paper: https://arxiv.org/abs/1809.01354
Source: https://github.com/lizhengwei1992/Semantic_Human_Matting

SHM (Semantic Human Matting) performs trimap-free alpha-matte estimation in
two coupled sub-networks:

  T-Net  -- a semantic segmentation network that produces a soft trimap:
             foreground / background / unknown probabilities.  Implemented
             here as a compact U-Net style encoder-decoder with output
             3-channel trimap softmax.

  M-Net  -- a matting network that takes the original image concatenated with
             the T-Net trimap (6 channels total) and predicts a fine alpha
             matte.  Also a compact U-Net with skip connections; output is a
             single-channel sigmoid alpha.

  Fusion -- the final alpha is a weighted combination of the T-Net hard
             decisions and the M-Net soft alpha:
               alpha = T_fg * 1 + T_bg * 0 + T_unk * M_alpha

Distinctive primitive: the two-stage trimap-based fusion -- T-Net trimap
gates the M-Net prediction so the matting network focuses on unknown regions.

Compact config: 3-level U-Net with base_ch=16; inputs are (1,3,64,64).
Output: alpha matte (1,1,64,64).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Compact U-Net encoder-decoder
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    """3-level U-Net: encoder -> bottleneck -> decoder with skip connections."""

    def __init__(self, in_ch: int, out_ch: int, base_ch: int = 16) -> None:
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        # Head
        self.head = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# T-Net and M-Net wrappers
# ---------------------------------------------------------------------------


class TNet(nn.Module):
    """T-Net: predict soft trimap (fg/bg/unknown) from RGB image."""

    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        self.unet = UNetSmall(in_ch=3, out_ch=3, base_ch=base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)  ->  trimap: (B, 3, H, W) softmax
        return F.softmax(self.unet(x), dim=1)


class MNet(nn.Module):
    """M-Net: predict alpha matte from image + trimap (6-channel input)."""

    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        self.unet = UNetSmall(in_ch=6, out_ch=1, base_ch=base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, H, W) = image + trimap  ->  alpha: (B, 1, H, W)
        return torch.sigmoid(self.unet(x))


# ---------------------------------------------------------------------------
# Full SHM model
# ---------------------------------------------------------------------------


class SemanticHumanMatting(nn.Module):
    """SHM: T-Net trimap estimator + M-Net alpha predictor + fusion.

    Input:  (B, 3, H, W) RGB image.
    Output: (B, 1, H, W) alpha matte.
    """

    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        self.tnet = TNet(base_ch)
        self.mnet = MNet(base_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trimap from T-Net: channels [fg, bg, unknown]
        trimap = self.tnet(x)  # (B, 3, H, W)

        # M-Net alpha from image + trimap
        m_alpha = self.mnet(torch.cat([x, trimap], dim=1))  # (B, 1, H, W)

        # Fusion: alpha = T_fg + T_unk * m_alpha
        t_fg = trimap[:, 0:1]
        t_unk = trimap[:, 2:3]
        alpha = t_fg + t_unk * m_alpha
        alpha = alpha.clamp(0, 1)
        return alpha


# ---------------------------------------------------------------------------
# Builder + example
# ---------------------------------------------------------------------------


def build_semantic_human_matting() -> nn.Module:
    """Build compact SHM model."""
    return SemanticHumanMatting(base_ch=16)


def example_input_shm() -> torch.Tensor:
    """(1, 3, 64, 64) RGB image."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Semantic Human Matting (T-Net trimap + M-Net alpha + fusion, trimap-free matting)",
        "build_semantic_human_matting",
        "example_input_shm",
        "2018",
        "DC",
    ),
]
