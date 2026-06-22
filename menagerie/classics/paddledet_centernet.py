"""PaddleDetection CenterNet: anchor-free object detection as points.

Zhou et al. (2019), "Objects as Points".  CenterNet represents each object by
its center keypoint and predicts three dense heads from a shared feature map:
class heatmap, box size, and local center offset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetBackbone(nn.Module):
    """Compact downsample/upsample backbone for dense CenterNet heads."""

    def __init__(self, width: int = 24) -> None:
        """Initialize the backbone.

        Parameters
        ----------
        width:
            Base channel count.
        """

        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=False),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the dense prediction feature map.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Dense feature map at output stride two.
        """

        return self.up(self.down(x))


class CenterHead(nn.Module):
    """Small convolutional prediction head."""

    def __init__(self, channels: int, out_channels: int) -> None:
        """Initialize a CenterNet prediction head.

        Parameters
        ----------
        channels:
            Input feature channels.
        out_channels:
            Number of prediction channels.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the prediction head.

        Parameters
        ----------
        x:
            Dense feature map.

        Returns
        -------
        torch.Tensor
            Dense prediction map.
        """

        return self.net(x)


class PaddleDetCenterNet(nn.Module):
    """Compact CenterNet detector."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize CenterNet.

        Parameters
        ----------
        classes:
            Number of object classes.
        """

        super().__init__()
        self.backbone = CenterNetBackbone()
        self.heatmap = CenterHead(24, classes)
        self.size = CenterHead(24, 2)
        self.offset = CenterHead(24, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict heatmap, size, and offset maps.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            CenterNet dense detection maps.
        """

        feat = self.backbone(x)
        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "size": F.softplus(self.size(feat)),
            "offset": self.offset(feat),
        }


def build() -> nn.Module:
    """Build the compact PaddleDetection CenterNet model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetCenterNet().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_centernet", "build", "example_input", "2019", "DC"),
]
