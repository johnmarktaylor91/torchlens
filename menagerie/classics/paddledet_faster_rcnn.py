"""Faster R-CNN compact PaddleDetection-style two-stage detector.

Ren et al., 2015, "Faster R-CNN: Towards Real-Time Object Detection with
Region Proposal Networks".  The distinctive inference graph is shared CNN
features, an RPN objectness/box branch, ROI feature extraction, and a second
stage classifier/regressor.  This compact version uses fixed differentiable ROI
summaries from top objectness locations so it stays traceable without custom ops.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FasterRCNN(nn.Module):
    """Compact Faster R-CNN with RPN and ROI heads."""

    def __init__(self, channels: int = 32, proposals: int = 8, classes: int = 20) -> None:
        """Initialize detector components.

        Parameters
        ----------
        channels:
            Feature width.
        proposals:
            Number of top proposals to summarize.
        classes:
            Number of object classes.
        """
        super().__init__()
        self.proposals = proposals
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.rpn = nn.Conv2d(channels, channels, 3, padding=1)
        self.rpn_obj = nn.Conv2d(channels, 3, 1)
        self.rpn_box = nn.Conv2d(channels, 12, 1)
        self.roi_fc = nn.Sequential(nn.Linear(channels, channels), nn.ReLU())
        self.cls = nn.Linear(channels, classes)
        self.box = nn.Linear(channels, classes * 4)

    def roi_features(self, feat: Tensor, objectness: Tensor) -> Tensor:
        """Pool top objectness locations into proposal features.

        Parameters
        ----------
        feat:
            Shared convolutional feature map.
        objectness:
            RPN objectness map.

        Returns
        -------
        Tensor
            Proposal feature tensor.
        """
        scores = objectness.flatten(2).max(dim=1).values
        idx = torch.topk(scores, self.proposals, dim=1).indices
        flat = feat.flatten(2).transpose(1, 2)
        gather = idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        return torch.gather(flat, 1, gather)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run RPN and second-stage ROI heads.

        Parameters
        ----------
        image:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            RPN boxes, ROI class logits, and ROI box deltas.
        """
        feat = self.backbone(image)
        rpn_feat = F.relu(self.rpn(feat))
        obj = self.rpn_obj(rpn_feat)
        rpn_box = self.rpn_box(rpn_feat).flatten(2).transpose(1, 2)
        roi = self.roi_fc(self.roi_features(feat, obj))
        return rpn_box, self.cls(roi), self.box(roi)


def build() -> nn.Module:
    """Build compact Faster R-CNN.

    Returns
    -------
    nn.Module
        Random-initialized detector.
    """
    return FasterRCNN().eval()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_faster_rcnn", "build", "example_input", "2015", "DC"),
]
