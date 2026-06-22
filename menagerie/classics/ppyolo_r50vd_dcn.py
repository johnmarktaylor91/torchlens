"""PP-YOLO R50vd-DCN compact random-init reconstruction.

Paper: PP-YOLO: An Effective and Efficient Implementation of Object Detector
(Long et al., 2020).

The faithful primitives are a ResNet50-vd-style detection backbone with
deformable-conv offsets in the later stage, CoordConv/SPP/FPN neck features,
and YOLO heads with IoU-aware and grid-sensitive outputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CoordConv(nn.Module):
    """Coordinate-channel convolution used in PP-YOLO heads."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize convolution after appending x/y coordinates."""

        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Append normalized coordinate maps and convolve."""

        bsz, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing="ij",
        )
        coord = torch.stack([xx, yy], dim=0).expand(bsz, -1, -1, -1)
        return self.conv(torch.cat([x, coord], dim=1))


class DeformableStage(nn.Module):
    """Small deformable-convolution surrogate with learned offsets."""

    def __init__(self, channels: int) -> None:
        """Initialize offset predictor and sampled convolution."""

        super().__init__()
        self.offset = nn.Conv2d(channels, 2, 3, padding=1)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Sample features with learned offsets before convolution."""

        bsz, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1)
        warped = F.grid_sample(x, grid + 0.1 * torch.tanh(self.offset(x)).permute(0, 2, 3, 1))
        return F.silu(self.conv(warped))


class PPYOLO(nn.Module):
    """Compact PP-YOLO detector with R50vd-DCN-style components."""

    def __init__(self, classes: int = 10, width: int = 24) -> None:
        """Initialize vd stem, deformable stage, SPP/FPN, and YOLO heads."""

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.c4 = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.SiLU())
        self.dcn = DeformableStage(width * 2)
        self.spp = nn.Conv2d(width * 8, width * 2, 1)
        self.coord = CoordConv(width * 3, width * 2)
        self.cls = nn.Conv2d(width * 2, classes, 1)
        self.box = nn.Conv2d(width * 2, 4, 1)
        self.obj = nn.Conv2d(width * 2, 1, 1)
        self.iou = nn.Conv2d(width * 2, 1, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict PP-YOLO class, grid-sensitive box, and IoU-aware objectness."""

        c3 = self.stem(image)
        c4 = self.dcn(self.c4(c3))
        pools = [
            c4,
            F.max_pool2d(c4, 5, stride=1, padding=2),
            F.max_pool2d(c4, 9, stride=1, padding=4),
        ]
        pools.append(F.max_pool2d(c4, 13, stride=1, padding=6))
        p4 = self.spp(torch.cat(pools, dim=1))
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        feat = F.silu(self.coord(torch.cat([c3, p4_up], dim=1)))
        grid_box = torch.sigmoid(self.box(feat)) * 1.05 - 0.025
        iou_aware = torch.sigmoid(self.obj(feat)) * torch.sigmoid(self.iou(feat))
        return self.cls(feat), grid_box, iou_aware


def build() -> nn.Module:
    """Build a compact random-init PP-YOLO R50vd-DCN detector."""

    return PPYOLO().eval()


def example_input() -> Tensor:
    """Return a small RGB image."""

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("ppyolo_r50vd_dcn", "build", "example_input", "2020", "DC"),
]
