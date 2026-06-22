"""CRAFT x4 image super-resolution transformer.

Paper: Li et al. 2024, "CRAFT: Cross Aggregation Transformer for Image
Restoration." This is distinct from the CRAFT scene-text detector. The compact
classic keeps CRAFT-SR's defining pieces: shallow feature extraction,
window-style spatial attention, channel attention, cross aggregation of the two
branches, residual groups, and pixel-shuffle x4 reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAggregationBlock(nn.Module):
    """CRAFT-style spatial/channel cross aggregation block."""

    def __init__(self, channels: int = 32, heads: int = 4) -> None:
        """Initialize attention, channel gate, and feed-forward branches.

        Parameters
        ----------
        channels:
            Feature channel count.
        heads:
            Number of spatial-attention heads.
        """

        super().__init__()
        self.norm_spatial = nn.LayerNorm(channels)
        self.spatial_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        self.mix = nn.Conv2d(channels * 2, channels, 1)
        self.ff = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1), nn.GELU(), nn.Conv2d(channels * 2, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross aggregation to an image feature map.

        Parameters
        ----------
        x:
            Feature map ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        spatial, _ = self.spatial_attn(
            self.norm_spatial(tokens),
            self.norm_spatial(tokens),
            self.norm_spatial(tokens),
            need_weights=False,
        )
        spatial_map = spatial.transpose(1, 2).reshape(batch, channels, height, width)
        channel_map = x * self.channel_gate(x)
        fused = x + self.mix(torch.cat([spatial_map, channel_map], dim=1))
        return fused + self.ff(fused)


class CRAFTSRx4(nn.Module):
    """Compact CRAFT x4 super-resolution network."""

    def __init__(self, channels: int = 32, blocks: int = 2) -> None:
        """Initialize CRAFT-SR stem, body, and pixel-shuffle upsampler.

        Parameters
        ----------
        channels:
            Feature channel count.
        blocks:
            Number of cross-aggregation blocks.
        """

        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[CrossAggregationBlock(channels) for _ in range(blocks)])
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Super-resolve a low-resolution RGB image by x4.

        Parameters
        ----------
        image:
            Low-resolution RGB image ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Super-resolved RGB image ``(B, 3, 4H, 4W)``.
        """

        shallow = self.stem(image)
        features = shallow + self.body_tail(self.body(shallow))
        return self.upsample(features)


def build() -> nn.Module:
    """Build compact CRAFT x4 super-resolution model.

    Returns
    -------
    nn.Module
        Random-initialized CRAFT-SR model.
    """

    return CRAFTSRx4()


def example_input() -> torch.Tensor:
    """Create a low-resolution RGB input.

    Returns
    -------
    torch.Tensor
        Image tensor of shape ``(1, 3, 16, 16)``.
    """

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [("craft_sr_x4", "build", "example_input", "2024", "SR")]
