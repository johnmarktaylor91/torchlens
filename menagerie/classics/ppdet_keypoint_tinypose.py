"""TinyPose compact PaddleDetection exact-name reconstruction.

Paper: PaddleDetection TinyPose lightweight top-down pose estimator.

TinyPose is a mobile top-down human-pose model: a lightweight convolutional
backbone feeds an upsampling heatmap head, with offset/refinement predictions
for joint localization.  This compact version keeps those primitives.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DepthwiseBlock(nn.Module):
    """Mobile depthwise-separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize depthwise and pointwise convolutions.

        Parameters
        ----------
        in_channels:
            Input feature channels.
        out_channels:
            Output feature channels.
        stride:
            Depthwise stride.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the depthwise-separable block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """

        return self.net(x)


class TinyPose(nn.Module):
    """Compact TinyPose heatmap and offset estimator."""

    def __init__(self, joints: int = 8, width: int = 24) -> None:
        """Initialize lightweight backbone and pose heads.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Feature width.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.SiLU(),
            DepthwiseBlock(width, width, stride=2),
            DepthwiseBlock(width, width * 2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.heatmap = nn.Conv2d(width, joints, 1)
        self.offset = nn.Conv2d(width, joints * 2, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict keypoint heatmaps and local offsets.

        Parameters
        ----------
        image:
            Cropped person image tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Joint heatmaps and x/y offsets.
        """

        feat = self.deconv(self.backbone(image))
        return self.heatmap(feat), torch.tanh(self.offset(feat))


def build() -> nn.Module:
    """Build a compact random-init TinyPose model.

    Returns
    -------
    nn.Module
        Dependency-free TinyPose-style model.
    """

    return TinyPose().eval()


def example_input() -> Tensor:
    """Return a small cropped-person RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 48)``.
    """

    return torch.randn(1, 3, 64, 48)


MENAGERIE_ENTRIES = [("ppdet_keypoint_tinypose", "build", "example_input", "2021", "DET")]
