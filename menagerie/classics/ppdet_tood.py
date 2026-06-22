"""TOOD compact PaddleDetection exact-name reconstruction.

Paper: TOOD: Task-aligned One-stage Object Detection.

TOOD aligns classification and localization by learning task-interactive
features, task-decomposition branches, and an alignment score.  This compact
model traces those inference primitives directly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TOODHead(nn.Module):
    """Task-aligned one-stage detection head."""

    def __init__(self, channels: int, classes: int) -> None:
        """Initialize task-interactive and decomposed branches.

        Parameters
        ----------
        channels:
            Feature width.
        classes:
            Number of classes.
        """

        super().__init__()
        self.interact = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )
        self.cls_decomp = nn.Conv2d(channels, channels, 1)
        self.reg_decomp = nn.Conv2d(channels, channels, 1)
        self.align = nn.Conv2d(channels, 1, 3, padding=1)
        self.cls = nn.Conv2d(channels, classes, 1)
        self.box = nn.Conv2d(channels, 4, 1)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict task-aligned class, box, and alignment maps.

        Parameters
        ----------
        feat:
            Backbone feature map.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, normalized boxes, and alignment logits.
        """

        shared = self.interact(feat)
        cls_feat = F.silu(self.cls_decomp(shared))
        reg_feat = F.silu(self.reg_decomp(shared))
        alignment = torch.sigmoid(self.align(cls_feat * reg_feat))
        cls = self.cls(cls_feat) * alignment
        box = torch.sigmoid(self.box(reg_feat))
        return cls.flatten(2).transpose(1, 2), box.flatten(2).transpose(1, 2), alignment


class TOOD(nn.Module):
    """Compact TOOD detector."""

    def __init__(self, width: int = 32, classes: int = 8) -> None:
        """Initialize backbone and task-aligned head.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of classes.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.head = TOODHead(width, classes)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run TOOD one-stage detection.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Detection outputs.
        """

        return self.head(self.backbone(image))


def build() -> nn.Module:
    """Build a compact random-init TOOD detector.

    Returns
    -------
    nn.Module
        Dependency-free TOOD-style detector.
    """

    return TOOD().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ppdet_tood", "build", "example_input", "2021", "DET")]
