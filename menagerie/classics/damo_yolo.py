"""DAMO-YOLO compact detector with RepGFPN and ZeroHead.

DAMO-YOLO (Xu et al., 2022) extends the YOLO detector line with a NAS-derived
backbone, an efficient reparameterized generalized FPN (RepGFPN), and a
lightweight ZeroHead prediction head. This compact version keeps those
load-bearing detector primitives while using a small CPU-friendly backbone.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RepBlock(nn.Module):
    """Training-time reparameterizable convolution block."""

    def __init__(self, channels: int) -> None:
        """Initialize parallel 3x3, 1x1, and identity branches.

        Parameters
        ----------
        channels:
            Feature channel count.
        """
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the multi-branch reparameterized block.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Activated feature map.
        """
        return F.silu(self.bn(self.conv3(x) + self.conv1(x) + x))


class TinyNasBackbone(nn.Module):
    """Small NAS-style multi-scale convolutional backbone."""

    def __init__(self, width: int = 24) -> None:
        """Initialize staged feature extractor.

        Parameters
        ----------
        width:
            Base channel count.
        """
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), nn.SiLU())
        self.stage2 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.SiLU(), RepBlock(width * 2)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.SiLU(), RepBlock(width * 4)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, 3, stride=2, padding=1), nn.SiLU(), RepBlock(width * 8)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return multi-scale backbone features.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Three feature scales.
        """
        x = self.stem(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class RepGFPN(nn.Module):
    """Efficient generalized FPN with reparameterized fusion blocks."""

    def __init__(self, width: int = 24) -> None:
        """Initialize lateral projections and fusion blocks.

        Parameters
        ----------
        width:
            Base channel count.
        """
        super().__init__()
        self.lat3 = nn.Conv2d(width * 2, width * 2, 1)
        self.lat4 = nn.Conv2d(width * 4, width * 2, 1)
        self.lat5 = nn.Conv2d(width * 8, width * 2, 1)
        self.fuse4 = RepBlock(width * 2)
        self.fuse3 = RepBlock(width * 2)
        self.down3 = nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1)
        self.out4 = RepBlock(width * 2)

    def forward(self, feats: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Fuse backbone scales top-down and bottom-up.

        Parameters
        ----------
        feats:
            Backbone feature tuple.

        Returns
        -------
        tuple[Tensor, Tensor]
            Refined FPN features.
        """
        p3, p4, p5 = feats
        p5 = self.lat5(p5)
        p4 = self.fuse4(self.lat4(p4) + F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3 = self.fuse3(self.lat3(p3) + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))
        p4 = self.out4(p4 + self.down3(p3))
        return p3, p4


class ZeroHead(nn.Module):
    """DAMO-YOLO lightweight task-projection prediction head."""

    def __init__(self, channels: int, classes: int = 3) -> None:
        """Initialize prediction projection.

        Parameters
        ----------
        channels:
            Input channel count.
        classes:
            Number of classes.
        """
        super().__init__()
        self.pred = nn.Conv2d(channels, classes + 4 + 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict class logits, boxes, and objectness.

        Parameters
        ----------
        x:
            FPN feature map.

        Returns
        -------
        Tensor
            Dense detection tensor.
        """
        pred = self.pred(x)
        cls = pred[:, :3]
        box = F.softplus(pred[:, 3:7])
        obj = torch.sigmoid(pred[:, 7:8])
        return torch.cat((cls, box, obj), dim=1)


class DAMOYOLO(nn.Module):
    """Compact DAMO-YOLO detector."""

    def __init__(self) -> None:
        """Initialize backbone, RepGFPN, and ZeroHead."""
        super().__init__()
        self.backbone = TinyNasBackbone()
        self.neck = RepGFPN()
        self.head = ZeroHead(48)

    def forward(self, x: Tensor) -> Tensor:
        """Run dense object detection.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        Tensor
            Concatenated multi-scale predictions.
        """
        p3, p4 = self.neck(self.backbone(x))
        p4 = F.interpolate(self.head(p4), size=p3.shape[-2:], mode="nearest")
        return torch.cat((self.head(p3), p4), dim=1)


def build() -> nn.Module:
    """Build compact DAMO-YOLO.

    Returns
    -------
    nn.Module
        Random-init detector.
    """
    return DAMOYOLO()


def example_input() -> Tensor:
    """Return a small image input.

    Returns
    -------
    Tensor
        RGB image tensor.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("DAMO-YOLO", "build", "example_input", "2022", "vision/detection")]
