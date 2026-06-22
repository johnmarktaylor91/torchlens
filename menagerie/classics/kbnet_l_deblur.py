"""KBNet: Kernel Basis Network for Image Restoration.

Zhang et al., 2023.
Paper: https://arxiv.org/abs/2303.02881

KBNet is defined by kernel basis attention (KBA): learnable kernel bases are
mixed with pixel-wise coefficients to form adaptive local aggregation kernels.
The MFF block fuses this pixel-adaptive branch with spatial-invariant depthwise
convolution and channel attention.  This compact variant preserves those
primitives for deblurring.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelBasisAttention(nn.Module):
    """Pixel-adaptive kernel-basis attention."""

    def __init__(self, channels: int, num_bases: int = 4, kernel_size: int = 3) -> None:
        """Initialize kernel bases and coefficient predictor."""

        super().__init__()
        self.channels = channels
        self.num_bases = num_bases
        self.kernel_size = kernel_size
        self.bases = nn.Parameter(
            torch.randn(num_bases, channels, kernel_size * kernel_size) * 0.02
        )
        self.coeff = nn.Conv2d(channels, num_bases, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate local patches with pixel-wise mixtures of learned bases."""

        b, c, h, w = x.shape
        patches = F.unfold(x, self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.view(b, c, self.kernel_size * self.kernel_size, h, w)
        coeff = torch.softmax(self.coeff(x), dim=1)
        kernels = torch.einsum("bnhw,nck->bckhw", coeff, self.bases)
        return (patches * kernels).sum(dim=2)


class MFFBlock(nn.Module):
    """Multi-axis feature fusion block with KBA, depthwise, and channel paths."""

    def __init__(self, channels: int) -> None:
        """Initialize the three KBNet feature axes."""

        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.kba = KernelBasisAttention(channels)
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse pixel-adaptive, spatial-invariant, and channel-wise features."""

        y = self.norm(x)
        adaptive = self.kba(y)
        spatial = self.depthwise(y)
        channel = y * self.channel(y)
        return x + self.fuse(torch.cat([adaptive, spatial, channel], dim=1))


class KBNet(nn.Module):
    """Compact KBNet restoration network."""

    def __init__(self, channels: int = 24, blocks: int = 3) -> None:
        """Initialize encoder, MFF trunk, and residual RGB head."""

        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[MFFBlock(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore an RGB image with residual prediction."""

        return x + self.head(self.blocks(self.stem(x)))


def build_kbnet_l_deblur() -> nn.Module:
    """Build compact KBNet-L deblurring reconstruction."""

    return KBNet(channels=24, blocks=3)


def build_kbnet_s_denoise() -> nn.Module:
    """Build compact KBNet-S denoising reconstruction."""

    return KBNet(channels=16, blocks=2)


def example_input() -> torch.Tensor:
    """Return a small RGB restoration input."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "KBNet-L Deblur (kernel basis attention restoration)",
        "build_kbnet_l_deblur",
        "example_input",
        "2023",
        "image-restoration/deblurring",
    ),
]
