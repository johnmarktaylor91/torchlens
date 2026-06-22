"""PP-Vehicle compact PaddleDetection pipeline reconstruction.

Source: PaddleDetection PP-Vehicle pipeline.

PP-Vehicle wraps vehicle detection with traffic-scene analytics such as color,
type, plate/attribute cues, and trajectory behavior.  This compact
random-initialized torch version keeps a shared detector feature path and
separate vehicle attribute and motion heads.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from menagerie.classics.paddledet_ppyoloe import RepBlock


class PPVehicle(nn.Module):
    """Compact PP-Vehicle detection and attribute pipeline."""

    def __init__(self, width: int = 32, vehicle_classes: int = 4) -> None:
        """Initialize detector, attribute, and traffic-behavior heads.

        Parameters
        ----------
        width:
            Feature width.
        vehicle_classes:
            Number of vehicle detection classes.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )
        self.stage = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            RepBlock(width),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            RepBlock(width),
        )
        self.neck = RepBlock(width)
        self.det_cls = nn.Conv2d(width, vehicle_classes, 1)
        self.det_box = nn.Conv2d(width, 4, 1)
        self.color = nn.Linear(width, 6)
        self.vehicle_type = nn.Linear(width, 5)
        self.plate_presence = nn.Linear(width, 1)
        self.motion = nn.Linear(width, 3)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run compact PP-Vehicle detection and analytics.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            Detection logits, boxes, color, type, plate, and behavior logits.
        """

        feat = self.neck(self.stage(self.stem(image)))
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        det_cls = self.det_cls(feat).flatten(2).transpose(1, 2)
        det_box = torch.sigmoid(self.det_box(feat)).flatten(2).transpose(1, 2)
        color = self.color(pooled)
        kind = self.vehicle_type(pooled)
        plate = torch.sigmoid(self.plate_presence(pooled))
        motion = self.motion(pooled)
        return det_cls, det_box, color, kind, plate, motion


def build() -> nn.Module:
    """Build a compact random-init PP-Vehicle pipeline.

    Returns
    -------
    nn.Module
        Dependency-free PP-Vehicle-style model.
    """

    return PPVehicle().eval()


def example_input() -> Tensor:
    """Return a small RGB frame.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ppdet_ppvehicle", "build", "example_input", "2022", "DET")]
