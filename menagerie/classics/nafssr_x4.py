"""NAFSSR x4: stereo image super-resolution using NAFNet blocks.

Paper: "NAFSSR: Stereo Image Super-Resolution Using NAFNet", Chu et al.,
CVPRW NTIRE 2022.

NAFSSR extends NAFNet from single-image restoration to stereo super-resolution
by inserting Stereo Cross Attention Modules (SCAM).  Each view is processed by
activation-free NAFBlocks, then bidirectional scaled dot-product cross-attention
lets the left and right views exchange complementary epipolar information before
late pixel-shuffle x4 reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics.nafnet_width32_sidd_denoise import LayerNorm2d, NAFBlock


class SCAM(nn.Module):
    """Stereo Cross Attention Module with bidirectional view exchange."""

    def __init__(self, channels: int) -> None:
        """Initialize stereo attention projections."""

        super().__init__()
        self.norm_l = LayerNorm2d(channels)
        self.norm_r = LayerNorm2d(channels)
        self.q_l = nn.Conv2d(channels, channels, 1)
        self.k_l = nn.Conv2d(channels, channels, 1)
        self.v_l = nn.Conv2d(channels, channels, 1)
        self.q_r = nn.Conv2d(channels, channels, 1)
        self.k_r = nn.Conv2d(channels, channels, 1)
        self.v_r = nn.Conv2d(channels, channels, 1)
        self.beta_l = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta_r = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply scaled dot-product attention over flattened spatial positions."""

        batch, channels, height, width = q.shape
        q_flat = q.flatten(2).transpose(1, 2)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q_flat, k_flat) / (channels**0.5), dim=-1)
        return torch.bmm(attn, v_flat).transpose(1, 2).view(batch, channels, height, width)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange information between left and right feature maps."""

        left_n = self.norm_l(left)
        right_n = self.norm_r(right)
        left_from_right = self._attend(self.q_l(left_n), self.k_r(right_n), self.v_r(right_n))
        right_from_left = self._attend(self.q_r(right_n), self.k_l(left_n), self.v_l(left_n))
        return left + left_from_right * self.beta_l, right + right_from_left * self.beta_r


class NAFSSRCompact(nn.Module):
    """Compact stereo NAFSSR x4 model."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize compact NAFSSR."""

        super().__init__()
        self.intro = nn.Conv2d(3, channels, 3, padding=1)
        self.left_block = NAFBlock(channels)
        self.right_block = NAFBlock(channels)
        self.scam = SCAM(channels)
        self.fuse = NAFBlock(channels)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve stereo input of shape ``(B, 2, 3, H, W)``."""

        left = self.left_block(self.intro(x[:, 0]))
        right = self.right_block(self.intro(x[:, 1]))
        left, right = self.scam(left, right)
        left_out = self.up(self.fuse(left)) + F.interpolate(
            x[:, 0], scale_factor=4.0, mode="bilinear"
        )
        right_out = self.up(self.fuse(right)) + F.interpolate(
            x[:, 1], scale_factor=4.0, mode="bilinear"
        )
        return torch.stack([left_out, right_out], dim=1)


def build() -> nn.Module:
    """Build compact NAFSSR x4."""

    return NAFSSRCompact()


def example_input() -> torch.Tensor:
    """Return a small stereo RGB pair."""

    return torch.randn(1, 2, 3, 12, 12)


MENAGERIE_ENTRIES = [
    ("NAFSSR x4 (NAFBlocks + stereo cross attention)", "build", "example_input", "2022", "E7")
]
