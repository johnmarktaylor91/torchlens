"""PP-YOLOE-R compact oriented-object detector.

Paper: PP-YOLOE-R / PaddleDetection rotated PP-YOLOE.

The distinctive primitive is PP-YOLOE's CSPRep/PAN anchor-free detector with an
oriented-box head.  The compact model predicts center/size plus sin-cos angle
coordinates so the traced graph contains the rotated-box regression path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from menagerie.classics.paddledet_ppyoloe import RepBlock


class PPYOLOERotated(nn.Module):
    """Compact PP-YOLOE-R detector with angle-aware regression."""

    def __init__(self, width: int = 32, classes: int = 8) -> None:
        """Initialize CSPRep backbone, PAN neck, and oriented head.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of oriented detection classes.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.SiLU()
        )
        self.c3 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.c4 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.c5 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.pan = RepBlock(width)
        self.cls = nn.Conv2d(width, classes, 1)
        self.box = nn.Conv2d(width, 4, 1)
        self.angle = nn.Conv2d(width, 2, 1)
        self.obj = nn.Conv2d(width, 1, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict class logits, oriented boxes, and objectness logits.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, rotated-box tensor, and objectness logits.
        """

        x = self.stem(image)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p = self.pan(
            c3
            + F.interpolate(
                c4 + F.interpolate(c5, size=c4.shape[-2:], mode="nearest"),
                size=c3.shape[-2:],
                mode="nearest",
            )
        )
        box = torch.sigmoid(self.box(p))
        angle_vec = F.normalize(self.angle(p), dim=1)
        oriented = torch.cat([box, angle_vec], dim=1).flatten(2).transpose(1, 2)
        return (
            self.cls(p).flatten(2).transpose(1, 2),
            oriented,
            self.obj(p).flatten(2).transpose(1, 2),
        )


def build() -> nn.Module:
    """Build a compact random-init PP-YOLOE-R detector.

    Returns
    -------
    nn.Module
        Dependency-free oriented-object detector.
    """

    return PPYOLOERotated().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ppdet_rotate_ppyoloe_r", "build", "example_input", "2022", "DET")]
