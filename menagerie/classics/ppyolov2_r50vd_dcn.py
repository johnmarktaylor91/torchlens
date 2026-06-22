"""PP-YOLOv2 R50vd-DCN compact random-init reconstruction.

Paper: PP-YOLOv2: A Practical Object Detector (Huang et al., 2021).

PP-YOLOv2 builds on PP-YOLO with R50vd-DCN, PAN-style path aggregation,
Mish/DropBlock-style regularized features, IoU-aware YOLO heads, and refined
training/inference components.  This compact model keeps the architectural
neck/head refinements in a dependency-free form.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .ppyolo_r50vd_dcn import DeformableStage


class DropBlockLite(nn.Module):
    """Deterministic DropBlock-like feature attenuation for traced inference."""

    def __init__(self, block: int = 3) -> None:
        """Initialize pooling block size."""

        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        """Attenuate locally dominant activations in a DropBlock-like pattern."""

        mask = torch.sigmoid(F.avg_pool2d(x.abs(), self.block, stride=1, padding=self.block // 2))
        return x * (1.0 - 0.1 * mask)


class PPYOLOv2(nn.Module):
    """Compact PP-YOLOv2 detector with PAN and IoU-aware heads."""

    def __init__(self, classes: int = 10, width: int = 24) -> None:
        """Initialize backbone, deformable stage, PAN neck, and detection heads."""

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.Mish()
        )
        self.c4 = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.Mish())
        self.c5 = nn.Sequential(
            DeformableStage(width * 2),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.Mish(),
        )
        self.top = nn.Conv2d(width * 4, width * 2, 1)
        self.pan_down = nn.Conv2d(width * 4, width * 2, 3, stride=2, padding=1)
        self.drop = DropBlockLite()
        self.cls = nn.Conv2d(width * 2, classes, 1)
        self.box = nn.Conv2d(width * 2, 4, 1)
        self.iou_obj = nn.Conv2d(width * 2, 2, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict PP-YOLOv2 class, box, and IoU-aware objectness maps."""

        c3 = self.stem(image)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        top = F.interpolate(self.top(c5), size=c4.shape[-2:], mode="nearest")
        pan = self.drop(F.mish(top + c4))
        bottom = self.drop(F.mish(self.pan_down(torch.cat([pan, c4], dim=1)) + self.top(c5)))
        box = torch.sigmoid(self.box(bottom)) * 1.1 - 0.05
        obj, iou = self.iou_obj(bottom).chunk(2, dim=1)
        return self.cls(bottom), box, torch.sigmoid(obj) * torch.sigmoid(iou)


def build() -> nn.Module:
    """Build a compact random-init PP-YOLOv2 R50vd-DCN detector."""

    return PPYOLOv2().eval()


def example_input() -> Tensor:
    """Return a small RGB image."""

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("ppyolov2_r50vd_dcn", "build", "example_input", "2021", "DC"),
]
