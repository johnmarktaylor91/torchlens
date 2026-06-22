"""MAT -- Mask-Aware Transformer for Large Hole Image Inpainting.

Li et al., CVPR 2022.
Paper: https://arxiv.org/abs/2203.15270
Source: https://github.com/fenglinglwb/MAT

MAT combines a transformer backbone with style modulation (AdaIN-like) for
large-hole inpainting. Features are extracted by a strided conv stem, processed
through transformer blocks with adaptive normalization derived from global
context, then upsampled back to the input resolution.

Input: (B, 4, H, W) -- RGB(3) + binary mask(1).
Output: (B, 3, H, W) inpainted image.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (multi-head attn + FFN)."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n = self.norm1(tokens)
        tokens = tokens + self.attn(n, n, n)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class MATInpainting(nn.Module):
    """Compact MAT inpainting network with transformer + style modulation.

    Input: (B, 4, H, W) -- RGB(3) + mask(1).
    Output: (B, 3, H, W) reconstructed image.
    """

    def __init__(self) -> None:
        super().__init__()
        C = 32
        # Stem: 64x64 -> 16x16 patches
        self.stem = nn.Sequential(
            nn.Conv2d(4, C, 3, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.Conv2d(C, C, 3, stride=2, padding=1),  # 16x16
            nn.GELU(),
        )
        # Two transformer blocks at 16x16 (256 tokens)
        self.tf1 = TransformerBlock(C, num_heads=4)
        self.tf2 = TransformerBlock(C, num_heads=4)
        # Style modulation: global avg -> AdaIN scale/shift
        self.style_fc = nn.Sequential(
            nn.Linear(C, C),
            nn.GELU(),
            nn.Linear(C, C * 2),  # scale + shift
        )
        self.adain_norm = nn.LayerNorm(C)
        # Decoder: 16x16 -> 64x64
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(C, C, 4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(C, 3, 4, stride=2, padding=1),  # 64x64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)  # (B, C, 16, 16)
        B, C, H, W = feat.shape

        # Flatten to token sequence
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Style from global average
        style_vec = feat.mean(dim=[2, 3])  # (B, C)
        style_params = self.style_fc(style_vec)  # (B, 2C)
        scale = style_params[:, :C].unsqueeze(1)  # (B, 1, C)
        shift = style_params[:, C:].unsqueeze(1)  # (B, 1, C)

        # Transformer with AdaIN
        tokens = self.tf1(tokens)
        tokens = self.adain_norm(tokens) * (1.0 + scale) + shift
        tokens = self.tf2(tokens)

        # Reshape and decode
        feat_out = tokens.permute(0, 2, 1).view(B, C, H, W)
        return self.dec(feat_out)


def build_mat_inpainting_512() -> nn.Module:
    """Build compact MAT inpainting network."""
    return MATInpainting()


def example_input() -> torch.Tensor:
    """Example (image + mask) tensor ``(1, 4, 64, 64)``."""
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "MAT Inpainting (masked transformer inpainting)",
        "build_mat_inpainting_512",
        "example_input",
        "2022",
        "DC",
    ),
]
