"""RetinaNet compact PaddleDetection-style one-stage detector.

Lin et al., 2017, "Focal Loss for Dense Object Detection".  RetinaNet combines
a feature pyramid backbone with separate dense classification and box-regression
subnets.  PaddleDetection exposes RetinaNet variants with this same inference
primitive; focal loss is a training criterion and is therefore noted but not
executed in the traced forward.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RetinaNet(nn.Module):
    """Compact RetinaNet with FPN and dense heads."""

    def __init__(self, channels: int = 32, classes: int = 20, anchors: int = 3) -> None:
        """Initialize backbone, FPN, and subnets.

        Parameters
        ----------
        channels:
            Feature width.
        classes:
            Number of object classes.
        anchors:
            Anchors per location.
        """
        super().__init__()
        self.c3 = nn.Sequential(nn.Conv2d(3, channels, 3, stride=2, padding=1), nn.ReLU())
        self.c4 = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU())
        self.c5 = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU())
        self.lat = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(3)])
        self.cls = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, anchors * classes, 3, padding=1),
        )
        self.box = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, anchors * 4, 3, padding=1),
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict dense anchor classes and boxes.

        Parameters
        ----------
        image:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Flattened class logits and anchor-box deltas.
        """
        c3 = self.c3(image)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.lat[2](c5)
        p4 = self.lat[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        cls = self.cls(p3).flatten(2).transpose(1, 2)
        box = self.box(p3).flatten(2).transpose(1, 2)
        return cls, box


def build() -> nn.Module:
    """Build compact RetinaNet.

    Returns
    -------
    nn.Module
        Random-initialized detector.
    """
    return RetinaNet().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_retinanet", "build", "example_input", "2017", "DC"),
]
