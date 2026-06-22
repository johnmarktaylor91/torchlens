"""ECBSR: edge-oriented convolution blocks for real-time super-resolution.

Paper: "Edge-oriented Convolution Block for Real-time Super Resolution on
Mobile Devices", Zhang et al., ACM MM 2021.

The compact reconstruction keeps ECB's training-time multi-branch block: a
normal 3x3 branch, channel expansion/squeeze branch, and fixed first/second
order edge-derivative branches, followed by a pixel-shuffle SR head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    """Fixed edge-derivative depthwise convolution branch."""

    def __init__(self, channels: int, kernel: torch.Tensor) -> None:
        """Initialize a fixed edge branch."""

        super().__init__()
        self.channels = channels
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fixed edge extraction followed by learnable mixing."""

        return self.mix(F.conv2d(x, self.kernel, padding=1, groups=self.channels))


class ECB(nn.Module):
    """Edge-oriented convolution block."""

    def __init__(self, channels: int) -> None:
        """Initialize ECB branches."""

        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.expand = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1), nn.ReLU(), nn.Conv2d(channels * 2, channels, 1)
        )
        self.edge1 = EdgeConv(channels, sobel_x)
        self.edge2 = EdgeConv(channels, laplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse regular, expanded, and edge-oriented branches."""

        return x + F.relu(self.conv3(x) + self.expand(x) + self.edge1(x) + self.edge2(x))


class ECBSRCompact(nn.Module):
    """Compact ECBSR x4 network."""

    def __init__(self, channels: int = 16, blocks: int = 3) -> None:
        """Initialize ECBSR."""

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[ECB(channels) for _ in range(blocks)])
        self.tail = nn.Sequential(nn.Conv2d(channels, 3 * 16, 3, padding=1), nn.PixelShuffle(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image by 4x."""

        return self.tail(self.body(self.head(x)))


def build_basicsr_ecbsr_x4() -> nn.Module:
    """Build compact ECBSR x4."""

    return ECBSRCompact()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "ECBSR x4 (edge-oriented convolution block SR)",
        "build_basicsr_ecbsr_x4",
        "example_input",
        "2021",
        "E7",
    )
]
