"""AlphaPose FastPose-DUC: ResNet backbone with dense upsampling convolution.

AlphaPose's FastPose network uses a ResNet feature extractor followed by DUC
(dense upsampling convolution / pixel shuffle) modules to predict human-pose
heatmaps efficiently. This compact reconstruction keeps the backbone-to-DUC
heatmap path without external AlphaPose dependencies.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DUC(nn.Module):
    """Dense upsampling convolution block."""

    def __init__(self, in_channels: int, out_channels: int, upscale: int = 2) -> None:
        """Initialize convolution and pixel shuffle upsampler.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels after upsampling.
        upscale:
            Pixel-shuffle scale factor.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * upscale * upscale, 3, padding=1),
            nn.BatchNorm2d(out_channels * upscale * upscale),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Upsample features with DUC.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Upsampled feature map.
        """
        return self.block(x)


class FastPoseDUC(nn.Module):
    """Compact FastPose DUC keypoint heatmap network."""

    def __init__(self, keypoints: int = 17) -> None:
        """Initialize ResNet-like stem and DUC head.

        Parameters
        ----------
        keypoints:
            Number of pose heatmap channels.
        """
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.duc1 = DUC(96, 64)
        self.duc2 = DUC(64, 32)
        self.heatmaps = nn.Conv2d(32, keypoints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict keypoint heatmaps from a person crop.

        Parameters
        ----------
        image:
            Person crop tensor with shape ``(batch, 3, 64, 64)``.

        Returns
        -------
        Tensor
            Heatmap tensor with shape ``(batch, keypoints, 32, 32)``.
        """
        features = self.backbone(image)
        features = self.duc1(features)
        features = self.duc2(features)
        return self.heatmaps(features)


def build() -> nn.Module:
    """Build a compact AlphaPose FastPose-DUC model.

    Returns
    -------
    nn.Module
        Random-initialized pose model.
    """
    return FastPoseDUC()


def example_input() -> Tensor:
    """Return a small RGB person crop.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("AlphaPose-FastPose-DUC", "build", "example_input", "2023", "DC"),
]
