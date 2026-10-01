"""Coordinate Attention for efficient mobile networks.

Paper: Coordinate Attention for Efficient Mobile Network Design.
Hou, Zhou, and Feng, CVPR 2021.

Coordinate attention factorizes global pooling into height-wise and width-wise
1D encodings.  The two encodings share a bottleneck transform, then produce
direction-aware attention maps that retain positional information.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HardSwish(nn.Module):
    """MobileNetV3-style hard-swish activation used by coordinate attention."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply hard-swish activation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Tensor
            Activated tensor.
        """
        return x * F.relu6(x + 3.0) / 6.0


class CoordinateAttention(nn.Module):
    """Coordinate attention block with separate height and width gates."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        """Initialize shared bottleneck and directional projections.

        Parameters
        ----------
        channels:
            Number of input and output channels.
        reduction:
            Bottleneck reduction ratio.
        """
        super().__init__()
        hidden = max(8, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            HardSwish(),
        )
        self.height_proj = nn.Conv2d(hidden, channels, kernel_size=1)
        self.width_proj = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply coordinate attention.

        Parameters
        ----------
        x:
            Feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Direction-aware reweighted feature map.
        """
        height = x.shape[-2]
        width = x.shape[-1]
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        pooled = torch.cat([pooled_h, pooled_w], dim=2)
        encoded = self.shared(pooled)
        encoded_h, encoded_w = torch.split(encoded, [height, width], dim=2)
        gate_h = torch.sigmoid(self.height_proj(encoded_h))
        gate_w = torch.sigmoid(self.width_proj(encoded_w.transpose(2, 3)))
        return x * gate_h * gate_w


class CoordinateAttentionDemoNet(nn.Module):
    """Compact mobile-style CNN with coordinate attention blocks."""

    def __init__(self, channels: int = 16, num_classes: int = 5) -> None:
        """Initialize the demonstration classifier.

        Parameters
        ----------
        channels:
            Width of the convolutional trunk.
        num_classes:
            Number of output logits.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            HardSwish(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            HardSwish(),
        )
        self.attn = CoordinateAttention(channels)
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits through coordinate attention.

        Parameters
        ----------
        x:
            RGB image tensor with shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = self.stem(x)
        x = self.project(self.attn(self.depthwise(x)))
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.classifier(x)


def build() -> nn.Module:
    """Build a compact coordinate-attention demonstration network.

    Returns
    -------
    nn.Module
        Random-initialized coordinate-attention demo network.
    """
    return CoordinateAttentionDemoNet()


def example_input() -> Tensor:
    """Return a small traceable image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("Coordinate Attention (efficient mobile attention)", "build", "example_input", "2021", "DC")
]
