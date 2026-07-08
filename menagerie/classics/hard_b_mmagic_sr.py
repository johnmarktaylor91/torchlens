"""Compact MMagic super-resolution classics for hard-b residual rows.

Paper: RDN, "Residual Dense Network for Image Super-Resolution" (Zhang et al.,
CVPR 2018); SRCNN, "Image Super-Resolution Using Deep Convolutional Networks"
(Dong et al., TPAMI 2016).

These random-init PyTorch modules keep the load-bearing structure of the
MMagic/OpenMMLab recipes without depending on mmcv/mmagic: RDN uses residual
dense blocks, dense feature fusion, global residual learning, and late
pixel-shuffle upsampling; SRCNN upsamples first, then applies the canonical
9-5-5 convolutional reconstruction stack.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual dense block with local feature fusion."""

    def __init__(self, channels: int = 32, growth: int = 16, layers: int = 4) -> None:
        """Initialize a residual dense block.

        Parameters
        ----------
        channels:
            Input and output feature channel count.
        growth:
            Channels added by each dense convolution.
        layers:
            Number of densely connected convolutions.
        """

        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels + index * growth, growth, 3, padding=1) for index in range(layers)]
        )
        self.local_fusion = nn.Conv2d(channels + layers * growth, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dense feature extraction with a local residual skip.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Refined feature map.
        """

        features = [x]
        for conv in self.convs:
            features.append(F.relu(conv(torch.cat(features, dim=1))))
        return x + self.local_fusion(torch.cat(features, dim=1))


class RDNCompact(nn.Module):
    """Residual Dense Network for image super-resolution."""

    def __init__(self, channels: int = 32, blocks: int = 3, scale: int = 2) -> None:
        """Initialize compact RDN.

        Parameters
        ----------
        channels:
            Trunk feature width.
        blocks:
            Number of residual dense blocks.
        scale:
            Pixel-shuffle upsampling factor.
        """

        super().__init__()
        self.shallow1 = nn.Conv2d(3, channels, 3, padding=1)
        self.shallow2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualDenseBlock(channels) for _ in range(blocks)])
        self.global_fusion = nn.Sequential(
            nn.Conv2d(channels * blocks, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image with residual dense features.

        Parameters
        ----------
        x:
            Low-resolution RGB image.

        Returns
        -------
        torch.Tensor
            Super-resolved RGB image.
        """

        shallow = self.shallow1(x)
        feat = self.shallow2(shallow)
        block_outputs = []
        for block in self.blocks:
            feat = block(feat)
            block_outputs.append(feat)
        fused = self.global_fusion(torch.cat(block_outputs, dim=1)) + shallow
        return self.up(fused)


class SRCNNCompact(nn.Module):
    """SRCNN 9-5-5 convolutional super-resolution network."""

    def __init__(self, scale: int = 4) -> None:
        """Initialize compact SRCNN.

        Parameters
        ----------
        scale:
            Bicubic pre-upsampling factor.
        """

        super().__init__()
        self.scale = scale
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample first, then reconstruct with the SRCNN stack.

        Parameters
        ----------
        x:
            Low-resolution RGB image.

        Returns
        -------
        torch.Tensor
            Super-resolved RGB image.
        """

        upsampled = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.features(upsampled)


def build_rdn_x2() -> nn.Module:
    """Build compact RDN x2 super-resolution model.

    Returns
    -------
    nn.Module
        Random-initialized RDN module.
    """

    return RDNCompact(scale=2).eval()


def build_srcnn_x4() -> nn.Module:
    """Build compact SRCNN x4 super-resolution model.

    Returns
    -------
    nn.Module
        Random-initialized SRCNN module.
    """

    return SRCNNCompact(scale=4).eval()


def example_input() -> torch.Tensor:
    """Return a compact low-resolution RGB image.

    Returns
    -------
    torch.Tensor
        Low-resolution image tensor.
    """

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "RDN x2 (residual dense network super-resolution)",
        "build_rdn_x2",
        "example_input",
        "2018",
        "E5",
    ),
    (
        "SRCNN x4 (9-5-5 convolutional super-resolution)",
        "build_srcnn_x4",
        "example_input",
        "2016",
        "E4",
    ),
]
