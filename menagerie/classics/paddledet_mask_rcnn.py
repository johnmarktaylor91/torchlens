"""Mask R-CNN compact PaddleDetection-style instance segmenter.

He et al., 2017, "Mask R-CNN".  Mask R-CNN extends Faster R-CNN with ROIAlign
and a parallel FCN mask branch.  This compact module reuses the traceable
Faster R-CNN proposal/class/box core and adds a small per-proposal mask head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .paddledet_faster_rcnn import FasterRCNN


class MaskRCNN(FasterRCNN):
    """Compact Mask R-CNN with an added mask branch."""

    def __init__(self, channels: int = 32, proposals: int = 8, classes: int = 20) -> None:
        """Initialize two-stage detector and mask head.

        Parameters
        ----------
        channels:
            Feature width.
        proposals:
            Number of proposal summaries.
        classes:
            Number of classes.
        """
        super().__init__(channels, proposals, classes)
        self.mask = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, classes * 8 * 8)
        )
        self.classes = classes

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run detection and mask heads.

        Parameters
        ----------
        image:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            RPN boxes, class logits, box deltas, and per-class masks.
        """
        feat = self.backbone(image)
        rpn_feat = torch.relu(self.rpn(feat))
        obj = self.rpn_obj(rpn_feat)
        rpn_box = self.rpn_box(rpn_feat).flatten(2).transpose(1, 2)
        roi = self.roi_fc(self.roi_features(feat, obj))
        masks = self.mask(roi).reshape(image.shape[0], self.proposals, self.classes, 8, 8)
        return rpn_box, self.cls(roi), self.box(roi), masks


def build() -> nn.Module:
    """Build compact Mask R-CNN.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """
    return MaskRCNN().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_mask_rcnn", "build", "example_input", "2017", "DC"),
]
