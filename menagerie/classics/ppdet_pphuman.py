"""PP-Human compact PaddleDetection pipeline reconstruction.

Source: PaddleDetection PP-Human pipeline.

PP-Human is not just a detector checkpoint: it is a pedestrian-analysis
pipeline around PaddleDetection, combining person detection with auxiliary
attribute, keypoint, and re-identification heads.  This compact model keeps a
shared PP-YOLOE-like feature extractor and traces those task-specific branches.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from menagerie.classics.paddledet_ppyoloe import RepBlock


class PPHuman(nn.Module):
    """Compact PP-Human multi-task perception pipeline."""

    def __init__(self, width: int = 32, classes: int = 2, joints: int = 6) -> None:
        """Initialize shared detector and human-analysis heads.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Person/background detection classes.
        joints:
            Number of compact pose heatmaps.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )
        self.c3 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.c4 = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), RepBlock(width))
        self.pan = RepBlock(width)
        self.det_cls = nn.Conv2d(width, classes, 1)
        self.det_box = nn.Conv2d(width, 4, 1)
        self.pose_up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.pose = nn.Conv2d(width, joints, 1)
        self.attribute = nn.Linear(width, 5)
        self.reid = nn.Linear(width, 16)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run compact PP-Human detection, pose, attributes, and ReID.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            Detection logits, boxes, pose heatmaps, attributes, and embeddings.
        """

        c3 = self.c3(self.stem(image))
        c4 = self.c4(c3)
        feat = self.pan(c3 + F.interpolate(c4, size=c3.shape[-2:], mode="nearest"))
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        det_cls = self.det_cls(feat).flatten(2).transpose(1, 2)
        det_box = torch.sigmoid(self.det_box(feat)).flatten(2).transpose(1, 2)
        pose = self.pose(F.silu(self.pose_up(feat)))
        attr = torch.sigmoid(self.attribute(pooled))
        reid = F.normalize(self.reid(pooled), dim=-1)
        return det_cls, det_box, pose, attr, reid


def build() -> nn.Module:
    """Build a compact random-init PP-Human pipeline.

    Returns
    -------
    nn.Module
        Dependency-free PP-Human-style model.
    """

    return PPHuman().eval()


def example_input() -> Tensor:
    """Return a small RGB frame.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ppdet_pphuman", "build", "example_input", "2022", "DET")]
