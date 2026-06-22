"""PaddleDetection FCOS: fully convolutional anchor-free detection.

Tian et al. (ICCV 2019), "FCOS: Fully Convolutional One-Stage Object
Detection".  FCOS predicts class scores, per-pixel ``l/t/r/b`` box distances,
and centerness without anchor boxes or proposal stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddleDetFCOS(nn.Module):
    """Compact FCOS detector."""

    def __init__(self, classes: int = 5, channels: int = 32) -> None:
        """Initialize FCOS.

        Parameters
        ----------
        classes:
            Number of object classes.
        channels:
            Feature channel count.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.cls_tower = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(False))
        self.box_tower = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(False))
        self.cls = nn.Conv2d(channels, classes, 3, padding=1)
        self.ltrb = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict dense FCOS outputs.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Class logits, l/t/r/b distances, and centerness.
        """

        feat = self.backbone(x)
        cls_feat = self.cls_tower(feat)
        box_feat = self.box_tower(feat)
        return {
            "cls_logits": self.cls(cls_feat),
            "ltrb": F.softplus(self.ltrb(box_feat)),
            "centerness": torch.sigmoid(self.centerness(box_feat)),
        }


def build() -> nn.Module:
    """Build the compact PaddleDetection FCOS model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetFCOS().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("paddledet_fcos", "build", "example_input", "2019", "DC")]
