"""YOLOX compact PaddleDetection-style anchor-free detector.

Ge et al., 2021, "YOLOX: Exceeding YOLO Series in 2021".  YOLOX modernizes the
YOLO family with an anchor-free formulation, decoupled classification and box
heads, and SimOTA label assignment during training.  The traced forward keeps
the CSP-like backbone, PAN fusion, and decoupled anchor-free heads.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class YOLOX(nn.Module):
    """Compact YOLOX detector."""

    def __init__(self, width: int = 32, classes: int = 20) -> None:
        """Initialize backbone, PAN, and heads.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of classes.
        """
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), nn.SiLU())
        self.c3 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())
        self.c4 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())
        self.c5 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())
        self.pan = nn.Conv2d(width, width, 3, padding=1)
        self.cls_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, classes, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 4, 1)
        )
        self.obj_head = nn.Conv2d(width, 1, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict anchor-free detections.

        Parameters
        ----------
        image:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, boxes, and objectness logits.
        """
        c3 = self.c3(self.stem(image))
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        feat = F.silu(
            self.pan(
                c3
                + F.interpolate(
                    c4 + F.interpolate(c5, size=c4.shape[-2:], mode="nearest"),
                    size=c3.shape[-2:],
                    mode="nearest",
                )
            )
        )
        return (
            self.cls_head(feat).flatten(2).transpose(1, 2),
            torch.sigmoid(self.reg_head(feat)).flatten(2).transpose(1, 2),
            self.obj_head(feat).flatten(2).transpose(1, 2),
        )


def build() -> nn.Module:
    """Build compact YOLOX.

    Returns
    -------
    nn.Module
        Random-initialized detector.
    """
    return YOLOX().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_yolox", "build", "example_input", "2021", "DC"),
]
