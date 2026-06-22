"""NanoTrack mobile Siamese tracker, compact random-init reconstruction.

NanoTrack in HonglinChu/SiamTrackers is a lightweight embedded/mobile tracker
in the SiamBAN/LightTrack lineage. Its load-bearing pieces are a very small
mobile convolutional backbone shared by template and search crops, depthwise
cross-correlation, and separate dense classification/regression heads. This
module keeps those primitives with inverted-residual mobile blocks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand: int = 2) -> None:
        """Initialize the mobile block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        stride:
            Depthwise stride.
        expand:
            Expansion ratio.
        """
        super().__init__()
        hidden = in_channels * expand
        self.use_residual = stride == 1 and in_channels == out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the inverted residual block.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        Tensor
            Output feature tensor.
        """
        out = self.net(x)
        return x + out if self.use_residual else out


class NanoBackbone(nn.Module):
    """Tiny shared mobile backbone for NanoTrack template/search crops."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize the mobile feature extractor.

        Parameters
        ----------
        channels:
            Output channel count.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            InvertedResidual(16, 16, stride=1),
            InvertedResidual(16, 24, stride=2),
            InvertedResidual(24, channels, stride=2),
            InvertedResidual(channels, channels, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode a tracking crop.

        Parameters
        ----------
        x:
            RGB crop.

        Returns
        -------
        Tensor
            Feature map.
        """
        return self.net(x)


def depthwise_xcorr(template: Tensor, search: Tensor) -> Tensor:
    """Depthwise cross-correlate template and search features.

    Parameters
    ----------
    template:
        Template features.
    search:
        Search features.

    Returns
    -------
    Tensor
        Correlation feature map.
    """
    batch, channels, ht, wt = template.shape
    _, _, hs, ws = search.shape
    search_grouped = search.reshape(1, batch * channels, hs, ws)
    kernels = template.reshape(batch * channels, 1, ht, wt)
    out = F.conv2d(search_grouped, kernels, groups=batch * channels)
    return out.reshape(batch, channels, out.shape[-2], out.shape[-1])


class NanoTrackMobile(nn.Module):
    """Mobile Siamese tracker with classification and box heads."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize backbone, neck, and dense heads.

        Parameters
        ----------
        channels:
            Correlation channel count.
        """
        super().__init__()
        self.backbone = NanoBackbone(channels)
        self.neck = nn.Conv2d(channels, channels, 1)
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, 2, 1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, 4, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Run NanoTrack on stacked template and search crops.

        Parameters
        ----------
        inputs:
            Tensor with template and search stacked on channel axis, shape
            ``(batch, 6, height, width)``.

        Returns
        -------
        Tensor
            Classification logits and positive box distances.
        """
        template, search = inputs[:, :3], inputs[:, 3:]
        z = self.neck(self.backbone(template))
        x = self.neck(self.backbone(search))
        corr = depthwise_xcorr(z, x)
        cls = self.cls_head(corr)
        box = F.softplus(self.box_head(corr))
        return torch.cat((cls, box), dim=1)


def build() -> nn.Module:
    """Build a compact NanoTrack-Mobile model.

    Returns
    -------
    nn.Module
        Random-init NanoTrack reconstruction.
    """
    return NanoTrackMobile()


def example_input() -> Tensor:
    """Return stacked template/search crops.

    Returns
    -------
    Tensor
        Input tensor of shape ``(1, 6, 64, 64)``.
    """
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    ("NanoTrack-Mobile", "build", "example_input", "2021", "vision/tracking"),
    ("SiamTrackers-NanoTrack", "build", "example_input", "2021", "vision/tracking"),
]
