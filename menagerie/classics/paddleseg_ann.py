"""Asymmetric Non-local Network for semantic segmentation.

Zhu et al. (ICCV 2019), "Asymmetric Non-local Neural Networks for Semantic
Segmentation."  ANN reduces non-local attention cost with pyramid-sampled
key/value features.  Its AFNB fuses low/high-level features by asymmetric
non-local attention, and APNB refines a feature map with the same sampled
non-local primitive.  This compact version keeps AFNB, APNB, and a small
segmentation head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        stride:
            Spatial stride.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.net(x)


class AsymmetricNonLocal(nn.Module):
    """Pyramid-sampled asymmetric non-local attention."""

    def __init__(self, channels: int, samples: tuple[int, ...] = (1, 2, 4)) -> None:
        """Initialize sampled query/key/value projections.

        Parameters
        ----------
        channels:
            Feature channel count.
        samples:
            Pyramid pooling output sizes for keys and values.
        """

        super().__init__()
        self.samples = samples
        hidden = max(4, channels // 2)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def _sample(self, x: torch.Tensor) -> torch.Tensor:
        """Collect pyramid-sampled spatial tokens.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Concatenated sampled tokens.
        """

        tokens = [F.adaptive_avg_pool2d(x, size).flatten(2) for size in self.samples]
        return torch.cat(tokens, dim=-1)

    def forward(self, query_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric non-local attention.

        Parameters
        ----------
        query_feat:
            Dense query feature map.
        context_feat:
            Feature map used for sampled keys and values.

        Returns
        -------
        torch.Tensor
            Residually refined query features.
        """

        bsz, channels, height, width = query_feat.shape
        q = self.query(query_feat).flatten(2).transpose(1, 2)
        k = self._sample(self.key(context_feat))
        v = self._sample(self.value(context_feat)).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) / (k.shape[1] ** 0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(bsz, channels, height, width)
        return query_feat + self.out(out)


class CompactANN(nn.Module):
    """Compact ANN segmentation model with AFNB and APNB."""

    def __init__(self, classes: int = 7, width: int = 16) -> None:
        """Initialize the model.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        width:
            Base channel width.
        """

        super().__init__()
        self.low = ConvBNReLU(3, width, stride=2)
        self.high = ConvBNReLU(width, width * 2, stride=2)
        self.high_proj = nn.Conv2d(width * 2, width, 1)
        self.afnb = AsymmetricNonLocal(width)
        self.apnb = AsymmetricNonLocal(width)
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment an image.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Segmentation logits at input resolution.
        """

        low = self.low(x)
        high = self.high_proj(self.high(low))
        high_up = F.interpolate(high, size=low.shape[2:], mode="bilinear")
        fused = self.afnb(low, high_up)
        refined = self.apnb(fused, fused)
        return F.interpolate(self.head(refined), size=x.shape[2:], mode="bilinear")


def build() -> nn.Module:
    """Build the compact ANN model.

    Returns
    -------
    nn.Module
        Random-init ANN in evaluation mode.
    """

    return CompactANN().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 48, 48)``.
    """

    return torch.randn(1, 3, 48, 48)


MENAGERIE_ENTRIES = [
    ("paddleseg_ann", "build", "example_input", "2019", "DC"),
]
