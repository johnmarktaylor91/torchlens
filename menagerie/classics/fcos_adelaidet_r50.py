"""AdelaiDet FCOS R50: anchor-free ResNet-FPN detector.

FCOS (Tian et al., 2019) predicts boxes per feature-map location with no anchor
boxes. The AdelaiDet/FCOS R50 family uses a ResNet-50 style backbone, an FPN,
and shared fully-convolutional classification, box-regression, and centerness
heads over pyramid levels.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    """Compact ResNet bottleneck block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize bottleneck convolutions.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        stride:
            Spatial stride for the 3x3 convolution.
        """
        super().__init__()
        mid = out_channels // 4
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual bottleneck block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """
        return F.relu(self.body(x) + self.downsample(x))


class MiniResNetFPN(nn.Module):
    """Small ResNet-FPN backbone for FCOS."""

    def __init__(self, width: int = 32, fpn_channels: int = 32) -> None:
        """Initialize backbone stages and lateral FPN projections.

        Parameters
        ----------
        width:
            Stem channel width.
        fpn_channels:
            Number of channels in all FPN levels.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.c3 = nn.Sequential(Bottleneck(width, width * 4), Bottleneck(width * 4, width * 4))
        self.c4 = nn.Sequential(
            Bottleneck(width * 4, width * 8, stride=2), Bottleneck(width * 8, width * 8)
        )
        self.c5 = nn.Sequential(
            Bottleneck(width * 8, width * 16, stride=2), Bottleneck(width * 16, width * 16)
        )
        self.lateral3 = nn.Conv2d(width * 4, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(width * 8, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(width * 16, fpn_channels, 1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return FPN feature maps P3-P5.

        Parameters
        ----------
        x:
            Image tensor with shape ``(batch, 3, height, width)``.

        Returns
        -------
        list[Tensor]
            Pyramid feature maps ordered from high to low resolution.
        """
        stem = self.stem(x)
        c3 = self.c3(stem)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [self.smooth3(p3), self.smooth4(p4), self.smooth5(p5)]


class FCOSHead(nn.Module):
    """Shared FCOS classification, box, and centerness towers."""

    def __init__(self, channels: int = 32, classes: int = 5) -> None:
        """Initialize prediction towers.

        Parameters
        ----------
        channels:
            FPN channel width.
        classes:
            Number of object classes.
        """
        super().__init__()
        tower = []
        for _ in range(2):
            tower.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GroupNorm(4, channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.cls_tower = nn.Sequential(*tower)
        self.box_tower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(channels, classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(channels, 1, 3, padding=1)
        self.scales = nn.Parameter(torch.ones(3))

    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Predict FCOS outputs on each pyramid level.

        Parameters
        ----------
        features:
            FPN feature maps.

        Returns
        -------
        tuple[list[Tensor], list[Tensor], list[Tensor]]
            Class logits, positive box distances, and centerness logits.
        """
        logits: list[Tensor] = []
        boxes: list[Tensor] = []
        centers: list[Tensor] = []
        for level, feature in enumerate(features):
            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)
            logits.append(self.cls_logits(cls_feature))
            boxes.append(torch.exp(self.scales[level] * self.bbox_pred(box_feature)))
            centers.append(self.centerness(box_feature))
        return logits, boxes, centers


class FCOSAdelaiDetR50(nn.Module):
    """Compact AdelaiDet FCOS R50 reconstruction."""

    def __init__(self) -> None:
        """Initialize backbone and dense FCOS head."""
        super().__init__()
        self.backbone = MiniResNetFPN()
        self.head = FCOSHead()

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run FCOS and concatenate pyramid predictions.

        Parameters
        ----------
        image:
            Image tensor with shape ``(batch, 3, 96, 96)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Flattened class, box, and centerness predictions.
        """
        cls, box, ctr = self.head(self.backbone(image))
        flat_cls = torch.cat([item.flatten(2).transpose(1, 2) for item in cls], dim=1)
        flat_box = torch.cat([item.flatten(2).transpose(1, 2) for item in box], dim=1)
        flat_ctr = torch.cat([item.flatten(2).transpose(1, 2) for item in ctr], dim=1)
        return flat_cls, flat_box, flat_ctr


def build() -> nn.Module:
    """Build a compact FCOS AdelaiDet R50 model.

    Returns
    -------
    nn.Module
        Random-initialized detector.
    """
    return FCOSAdelaiDetR50()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 96, 96)``.
    """
    return torch.randn(1, 3, 96, 96)


MENAGERIE_ENTRIES = [
    ("fcos_adelaidet_r50", "build", "example_input", "2019", "DC"),
]
