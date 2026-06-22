"""RFR-Net -- Recurrent Feature Reasoning for Image Inpainting.

Li et al., CVPR 2020.
Paper: https://arxiv.org/abs/2008.03737
Source: https://github.com/jingyuanli001/RFR-Inpainting

RFR-Net recurrently refines features via a Knowledge-Consistent Attention (KCA)
module: a self-attention mechanism on feature tokens that propagates information
from valid regions into hole regions. The loop is unrolled for a fixed number of
steps (2 here) for tracing compatibility.

Input: (B, 4, H, W) -- RGB(3) + binary mask(1).
Output: (B, 3, H, W) inpainted image.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KCAModule(nn.Module):
    """Knowledge-Consistent Attention: scaled dot-product self-attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        d = max(channels // 8, 1)
        self.q = nn.Conv2d(channels, d, 1)
        self.k = nn.Conv2d(channels, d, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        d = self.q.out_channels
        q = self.q(feat).view(B, d, -1).permute(0, 2, 1)  # (B, HW, d)
        k = self.k(feat).view(B, d, -1)  # (B, d, HW)
        v = self.v(feat).view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        attn = torch.softmax(torch.bmm(q, k) / (d**0.5), dim=-1)  # (B, HW, HW)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return feat + self.out(out)


class RFRNet(nn.Module):
    """Compact RFR-Net with 2 unrolled recurrent steps.

    Input: (B, 4, H, W) -- RGB(3) + mask(1).
    Output: (B, 3, H, W) inpainted image.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
        )
        # Shared KCA module (applied twice with separate refine convs)
        self.kca = KCAModule(64)
        self.refine1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.refine2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        img = x_in[:, :3]
        mask = x_in[:, 3:]

        feat = self.enc(x_in)  # (B, 64, 16, 16)

        # Step 1
        feat = self.kca(feat)
        feat = self.refine1(feat)
        # Step 2
        feat = self.kca(feat)
        feat = self.refine2(feat)

        out = self.dec(feat)  # (B, 3, 64, 64)

        # Blend: use known pixels from input
        mask_full = F.interpolate(mask, size=(64, 64), mode="nearest")
        return out * (1 - mask_full) + img * mask_full


def build_rfrnet_inpainting() -> nn.Module:
    """Build compact RFR-Net inpainting network."""
    return RFRNet()


def example_input() -> torch.Tensor:
    """Example (image + mask) tensor ``(1, 4, 64, 64)``."""
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "RFR-Net Inpainting (recurrent feature reasoning)",
        "build_rfrnet_inpainting",
        "example_input",
        "2020",
        "DC",
    ),
]
