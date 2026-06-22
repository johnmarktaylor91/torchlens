"""nnFormer: interleaved 3D transformer for volumetric segmentation.

Paper: nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer,
Zhou et al. 2021/2023.

The distinctive mechanisms are a U-shaped 3D encoder-decoder with interleaved
convolution and transformer blocks, local/global volume self-attention, and skip
attention instead of simple U-Net concatenation.  This compact version keeps those
primitives on a tiny volume.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VolumeAttentionBlock(nn.Module):
    """Window-free compact volume self-attention block."""

    def __init__(self, channels: int, heads: int = 2) -> None:
        """Initialize 3D transformer components.

        Parameters
        ----------
        channels:
            Feature channel width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply volume self-attention to flattened voxels.

        Parameters
        ----------
        x:
            Volume feature map ``(batch, channels, depth, height, width)``.

        Returns
        -------
        torch.Tensor
            Updated volume feature map.
        """

        batch, channels, depth, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_in = self.norm(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).view(batch, channels, depth, height, width)


class SkipAttention(nn.Module):
    """nnFormer-style attention gate for encoder skip features."""

    def __init__(self, channels: int) -> None:
        """Initialize skip gate projections.

        Parameters
        ----------
        channels:
            Number of skip/gating channels.
        """

        super().__init__()
        self.skip_proj = nn.Conv3d(channels, channels, 1)
        self.gate_proj = nn.Conv3d(channels, channels, 1)
        self.score = nn.Conv3d(channels, channels, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Gate skip features with decoder context.

        Parameters
        ----------
        skip:
            Encoder skip tensor.
        gate:
            Decoder gate tensor.

        Returns
        -------
        torch.Tensor
            Attention-weighted skip tensor.
        """

        gate = F.interpolate(gate, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        score = torch.sigmoid(self.score(F.gelu(self.skip_proj(skip) + self.gate_proj(gate))))
        return skip * score


class NNFormerVolumetric(nn.Module):
    """Compact U-shaped 3D transformer segmentation model."""

    def __init__(self, in_channels: int = 1, classes: int = 3, channels: int = 12) -> None:
        """Initialize compact nnFormer.

        Parameters
        ----------
        in_channels:
            Input modalities.
        classes:
            Segmentation classes.
        channels:
            Base channel width.
        """

        super().__init__()
        self.embed = nn.Sequential(nn.Conv3d(in_channels, channels, 3, padding=1), nn.GELU())
        self.local_attn = VolumeAttentionBlock(channels)
        self.down = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 3, stride=2, padding=1), nn.GELU()
        )
        self.bottleneck = VolumeAttentionBlock(channels * 2)
        self.up = nn.ConvTranspose3d(channels * 2, channels, 2, stride=2)
        self.skip_attn = SkipAttention(channels)
        self.fuse = nn.Sequential(nn.Conv3d(channels * 2, channels, 3, padding=1), nn.GELU())
        self.global_attn = VolumeAttentionBlock(channels)
        self.out = nn.Conv3d(channels, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment a 3D volume.

        Parameters
        ----------
        x:
            Input volume ``(batch, channels, depth, height, width)``.

        Returns
        -------
        torch.Tensor
            Segmentation logits.
        """

        skip = self.local_attn(self.embed(x))
        low = self.bottleneck(self.down(skip))
        up = self.up(low)
        attended = self.skip_attn(skip, up)
        fused = self.fuse(torch.cat([up, attended], dim=1))
        return self.out(self.global_attn(fused))


def build() -> nn.Module:
    """Build compact nnFormer.

    Returns
    -------
    nn.Module
        Random-init volumetric segmenter.
    """

    return NNFormerVolumetric()


def example_input() -> torch.Tensor:
    """Create a tiny 3D medical volume.

    Returns
    -------
    torch.Tensor
        Example volume ``(1, 1, 8, 12, 12)``.
    """

    return torch.randn(1, 1, 8, 12, 12)


MENAGERIE_ENTRIES = [
    (
        "nnformer_volumetric",
        "build",
        "example_input",
        "2021",
        "medical/segmentation",
    ),
]
