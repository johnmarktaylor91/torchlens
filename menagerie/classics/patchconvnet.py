"""PatchConvNet: convolutional patch stem with attention-based aggregation.

Touvron et al., 2021, arXiv:2112.13692.
Paper: Augmenting Convolutional networks with attention-based aggregation.

PatchConvNet keeps a simple patch-based convolutional trunk and replaces final
average pooling with a learned class token that aggregates patches through
attention.  This compact version keeps the patch conv blocks and late attention
pooling while using small widths and images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchConvBlock(nn.Module):
    """Residual convolutional patch block."""

    def __init__(self, channels: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual patch-convolution block.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        y = self.bn(self.dw(x))
        y = self.pw2(F.gelu(self.pw1(y)))
        return x + y


class AttentionAggregator(nn.Module):
    """Single class-token attention aggregator for image patches."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize the attention pooling head.

        Parameters
        ----------
        dim:
            Token dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Aggregate patch tokens with a learned class token.

        Parameters
        ----------
        tokens:
            Patch tokens ``(B, N, D)``.

        Returns
        -------
        torch.Tensor
            Aggregated class token ``(B, D)``.
        """

        cls = self.cls.expand(tokens.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.norm(seq)
        return self.attn(seq[:, :1], seq, seq, need_weights=False)[0].squeeze(1)


class PatchConvNet(nn.Module):
    """Compact PatchConvNet classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize patch embedding, convolutional trunk, and attention head.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.stem = nn.Conv2d(3, 48, 4, stride=4)
        self.blocks = nn.ModuleList([PatchConvBlock(48) for _ in range(4)])
        self.agg = AttentionAggregator(48)
        self.head = nn.Linear(48, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Image tensor ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        tokens = x.flatten(2).transpose(1, 2)
        return self.head(self.agg(tokens))


def build_patchconvnet() -> nn.Module:
    """Build the compact PatchConvNet.

    Returns
    -------
    nn.Module
        Random-init PatchConvNet classifier.
    """

    return PatchConvNet()


def example_input() -> torch.Tensor:
    """Create a compact image input.

    Returns
    -------
    torch.Tensor
        Image tensor ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("PatchConvNet", "build_patchconvnet", "example_input", "2021", "DC")]
