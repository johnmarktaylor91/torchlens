"""PP-YOLOE compact object detector.

Xu et al., 2022, "PP-YOLOE: An evolved version of YOLO".  PP-YOLOE keeps a
YOLO-style one-stage detector but replaces hand-crafted anchor design with an
anchor-free decoupled head, CSPRepResNet backbone, path aggregation neck, and
task-aligned assignment during training.  This module traces the compact
inference graph: CSPRep blocks, PAN fusion, and decoupled class/box heads.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RepBlock(nn.Module):
    """CSPRep-style residual convolution block."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual rep block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """
        return F.silu(self.bn(self.conv3(x) + self.conv1(x)) + x)


class PPYOLOE(nn.Module):
    """Compact PP-YOLOE detector."""

    def __init__(self, width: int = 40, classes: int = 20) -> None:
        """Initialize backbone, PAN neck, and heads.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of classes.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width)
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width)
        )
        self.pan3 = RepBlock(width)
        self.pan4 = RepBlock(width)
        self.cls_tower = nn.Sequential(RepBlock(width), nn.Conv2d(width, classes, 1))
        self.box_tower = nn.Sequential(RepBlock(width), nn.Conv2d(width, 4, 1))
        self.obj_head = nn.Conv2d(width, 1, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict anchor-free class, box, and objectness maps.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, normalized box maps, and objectness logits.
        """
        x = self.stem(image)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p4 = self.pan4(c4 + F.interpolate(c5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.pan3(c3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        cls = self.cls_tower(p3).flatten(2).transpose(1, 2)
        box = torch.sigmoid(self.box_tower(p3)).flatten(2).transpose(1, 2)
        obj = self.obj_head(p3).flatten(2).transpose(1, 2)
        return cls, box, obj


def build() -> nn.Module:
    """Build a compact PP-YOLOE detector.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """
    return PPYOLOE().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_ppyoloe", "build", "example_input", "2022", "DC"),
]
