"""SAFMN: spatially-adaptive feature modulation network for efficient SR.

Paper: "Spatially-Adaptive Feature Modulation for Efficient Image
Super-Resolution", Sun et al., ICCV 2023.

SAFMN uses Feature Mixing Modules made from a spatially-adaptive feature
modulation layer and a convolutional channel mixer.  SAFM splits channels,
extracts multi-scale spatial context, and uses it to modulate the original
features before x4 upsampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAFM(nn.Module):
    """Spatially-adaptive feature modulation layer."""

    def __init__(self, channels: int, splits: int = 4) -> None:
        """Initialize SAFM.

        Parameters
        ----------
        channels:
            Feature channel count.
        splits:
            Number of channel groups/scales.
        """

        super().__init__()
        self.splits = splits
        group_channels = channels // splits
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(group_channels, group_channels, 3, padding=1, groups=group_channels)
                for _ in range(splits)
            ]
        )
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate features with multi-scale spatial context.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Spatially modulated tensor.
        """

        h, w = x.shape[-2:]
        pieces = torch.chunk(x, self.splits, dim=1)
        outputs = []
        for idx, (piece, branch) in enumerate(zip(pieces, self.branches, strict=True)):
            if idx == 0:
                context = branch(piece)
            else:
                scale = 2**idx
                pooled = F.adaptive_avg_pool2d(piece, (max(1, h // scale), max(1, w // scale)))
                context = F.interpolate(branch(pooled), size=(h, w), mode="nearest")
            outputs.append(context)
        gate = F.gelu(self.project(torch.cat(outputs, dim=1)))
        return x * gate


class CCM(nn.Module):
    """Convolutional channel mixer used after SAFM."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        """Initialize CCM.

        Parameters
        ----------
        channels:
            Feature channel count.
        expansion:
            Hidden expansion ratio.
        """

        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix local context and channels.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Mixed tensor.
        """

        return self.net(x)


class FMM(nn.Module):
    """SAFMN feature mixing module."""

    def __init__(self, channels: int) -> None:
        """Initialize FMM.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.safm = SAFM(channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ccm = CCM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SAFM and CCM residual sublayers.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output feature tensor.
        """

        y = x + self.safm(self.norm1(x))
        return y + self.ccm(self.norm2(y))


class SAFMNCompact(nn.Module):
    """Compact SAFMN x4 super-resolution model."""

    def __init__(self, channels: int = 24, blocks: int = 3) -> None:
        """Initialize compact SAFMN.

        Parameters
        ----------
        channels:
            Feature width.
        blocks:
            Number of FMM blocks.
        """

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[FMM(channels) for _ in range(blocks)])
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image by 4x.

        Parameters
        ----------
        x:
            Low-resolution RGB image.

        Returns
        -------
        torch.Tensor
            Reconstructed RGB image.
        """

        feat = self.head(x)
        return self.up(self.body_tail(self.blocks(feat)) + feat)


def build_safmn_x4() -> nn.Module:
    """Build compact SAFMN x4.

    Returns
    -------
    nn.Module
        Random-init SAFMN reconstruction.
    """

    return SAFMNCompact()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image.

    Returns
    -------
    torch.Tensor
        Example image tensor.
    """

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "SAFMN x4 (spatially-adaptive feature modulation SR)",
        "build_safmn_x4",
        "example_input",
        "2023",
        "E7",
    )
]
