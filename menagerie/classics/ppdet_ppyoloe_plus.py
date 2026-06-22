"""PP-YOLOE+ compact PaddleDetection exact-name reconstruction.

Paper: PP-YOLOE: An Evolved Version of YOLO; PP-YOLOE+ PaddleDetection update.

PP-YOLOE+ keeps the PP-YOLOE anchor-free detector family and emphasizes a
CSPRepResNet/PAN backbone-neck with an efficient task-aligned head.  This
compact version makes the PP-YOLOE+ deployment primitive explicit by adding
distribution-bin box regression to the decoupled class/objectness branch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from menagerie.classics.paddledet_ppyoloe import RepBlock


class PPYOLOEPlus(nn.Module):
    """Compact PP-YOLOE+ detector with ET-head and distribution bins."""

    def __init__(self, width: int = 32, classes: int = 10, bins: int = 8) -> None:
        """Initialize CSPRep backbone, PAN neck, and task-aligned head.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of detection classes.
        bins:
            Number of distribution-regression bins per box side.
        """

        super().__init__()
        self.bins = bins
        self.register_buffer("bin_values", torch.arange(bins, dtype=torch.float32))
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )
        self.c3 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.c4 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.c5 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.fuse4 = RepBlock(width)
        self.fuse3 = RepBlock(width)
        self.share = RepBlock(width)
        self.cls = nn.Conv2d(width, classes, 1)
        self.obj = nn.Conv2d(width, 1, 1)
        self.dist = nn.Conv2d(width, 4 * bins, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict classes, distributional boxes, and task alignment scores.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, expected box distances, and alignment logits.
        """

        x = self.stem(image)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p4 = self.fuse4(c4 + F.interpolate(c5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.fuse3(c3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        feat = self.share(p3)
        cls = self.cls(feat).flatten(2).transpose(1, 2)
        obj = self.obj(feat).flatten(2).transpose(1, 2)
        bsz, _, height, width = feat.shape
        dist = self.dist(feat).view(bsz, 4, self.bins, height, width)
        box = (torch.softmax(dist, dim=2) * self.bin_values.view(1, 1, -1, 1, 1)).sum(dim=2)
        box = box.flatten(2).transpose(1, 2) / float(self.bins)
        alignment = torch.sigmoid(obj) * torch.sigmoid(cls).amax(dim=-1, keepdim=True)
        return cls, box, alignment


def build() -> nn.Module:
    """Build a compact random-init PP-YOLOE+ detector.

    Returns
    -------
    nn.Module
        Dependency-free PP-YOLOE+ detector.
    """

    return PPYOLOEPlus().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ppdet_ppyoloe_plus", "build", "example_input", "2022", "DET")]
