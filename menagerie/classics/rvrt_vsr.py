"""RVRT -- Recurrent Video Restoration Transformer with Guided Deformable Attention.

Liang et al., NeurIPS 2022.
Paper: https://arxiv.org/abs/2206.02146
Source: https://github.com/JingyunLiang/RVRT

RVRT applies a guided deformable alignment (simplified here via grid_sample
with learned offsets) followed by window self-attention blocks for spatial
feature refinement. Two recurrent steps are unrolled for tracing compatibility.

Input: (B, 9, H, W) -- 3 concatenated LR frames (3ch each).
Output: (B, 3, H_sr, W_sr) upsampled x4 HR frame via pixel_shuffle.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedDeformAlign(nn.Module):
    """Simplified guided deformable alignment via grid_sample + learned offsets."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.offset_conv = nn.Conv2d(channels * 2, 2, 3, padding=1)

    def forward(self, feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        concat = torch.cat([feat, ref_feat], dim=1)
        offsets = self.offset_conv(concat)  # (B, 2, H, W)

        # Build normalized sampling grid
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        norm_offsets = offsets.permute(0, 2, 3, 1) / max(H, W)
        grid = base_grid + norm_offsets
        aligned = F.grid_sample(ref_feat, grid, align_corners=True, padding_mode="border")
        return feat + aligned


class WindowAttn(nn.Module):
    """Window self-attention block (ws x ws windows)."""

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 4) -> None:
        super().__init__()
        self.ws = window_size
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.ws
        nH, nW = H // ws, W // ws

        # Partition into windows
        x_w = (
            x.reshape(B, C, nH, ws, nW, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(B * nH * nW, ws * ws, C)
        )
        # Self-attention
        x_n = self.norm(x_w)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x_w = x_w + attn_out
        x_w = x_w + self.ffn(self.norm2(x_w))

        # Unpartition
        x_out = x_w.reshape(B, nH, nW, ws, ws, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x_out


class RVRTSimple(nn.Module):
    """Compact RVRT with 2 unrolled recurrent GDA + window-attn steps.

    Input: (B, 9, H, W) -- 3-frame LR video.
    Output: (B, 3, 4H, 4W) -- x4 super-resolved single frame via pixel_shuffle.
    """

    def __init__(self) -> None:
        super().__init__()
        C = 32
        self.feat_extract = nn.Conv2d(9, C, 3, padding=1)
        self.gda1 = GuidedDeformAlign(C)
        self.attn1 = WindowAttn(C, num_heads=4, window_size=4)
        self.gda2 = GuidedDeformAlign(C)
        self.attn2 = WindowAttn(C, num_heads=4, window_size=4)
        self.upsample = nn.Sequential(
            nn.Conv2d(C, 3 * 16, 3, padding=1),
            nn.PixelShuffle(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feat_extract(x)  # (B, C, H, W)

        # Step 1
        feat = self.gda1(feat, feat)
        feat = self.attn1(feat)
        # Step 2
        feat = self.gda2(feat, feat)
        feat = self.attn2(feat)

        return self.upsample(feat)  # (B, 3, 4H, 4W)


def build_rvrt_vsr_x4() -> nn.Module:
    """Build compact RVRT x4 video super-resolution network."""
    return RVRTSimple()


def example_input() -> torch.Tensor:
    """Example 3-frame LR video tensor ``(1, 9, 32, 32)``."""
    return torch.randn(1, 9, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "RVRT VSR x4 (recurrent video restoration transformer)",
        "build_rvrt_vsr_x4",
        "example_input",
        "2022",
        "DC",
    ),
]
