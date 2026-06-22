"""Tiny LVIS Mask R-CNN with FPN, RPN, RoIAlign, box, and mask heads.

Paper: Gupta et al. 2019, "LVIS: A Dataset for Large Vocabulary Instance
Segmentation"; He et al. 2017, "Mask R-CNN."
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TinyBackboneFPN(nn.Module):
    """Small convolutional backbone with two FPN levels."""

    def __init__(self, channels: int = 12) -> None:
        """Initialize backbone and lateral FPN projections.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.c2 = nn.Sequential(nn.Conv2d(3, channels, 3, stride=2, padding=1), nn.ReLU())
        self.c3 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), nn.ReLU()
        )
        self.p2 = nn.Conv2d(channels, channels, 1)
        self.p3 = nn.Conv2d(channels * 2, channels, 1)
        self.smooth = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create two FPN feature maps.

        Parameters
        ----------
        image:
            Input RGB image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            FPN levels ``p2`` and ``p3``.
        """

        c2 = self.c2(image)
        c3 = self.c3(c2)
        p3 = self.p3(c3)
        up = F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth(self.p2(c2) + up)
        return p2, p3


class TinyRPN(nn.Module):
    """Region proposal branch shared across FPN levels."""

    def __init__(self, channels: int = 12) -> None:
        """Initialize objectness and box-delta heads.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.objectness = nn.Conv2d(channels, 3, 1)
        self.box_delta = nn.Conv2d(channels, 12, 1)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run RPN on both FPN levels.

        Parameters
        ----------
        features:
            FPN levels.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Objectness logits and box deltas.
        """

        obj_maps = []
        box_maps = []
        for feature in features:
            hidden = F.relu(self.conv(feature))
            obj_maps.append(self.objectness(hidden).flatten(2))
            box_maps.append(self.box_delta(hidden).flatten(2))
        return torch.cat(obj_maps, dim=2), torch.cat(box_maps, dim=2).transpose(1, 2)


class TinyLVISMaskRCNN(nn.Module):
    """Compact random-init LVIS Mask R-CNN detector."""

    def __init__(self, classes: int = 10, proposals: int = 2, channels: int = 12) -> None:
        """Initialize detector components.

        Parameters
        ----------
        classes:
            Number of LVIS categories retained for the tiny head.
        proposals:
            Number of fixed proposals.
        channels:
            Feature width.
        """

        super().__init__()
        self.classes = classes
        self.proposals = proposals
        self.backbone = TinyBackboneFPN(channels)
        self.rpn = TinyRPN(channels)
        boxes = torch.tensor(
            [[-0.75, -0.75, -0.15, -0.15], [0.15, 0.15, 0.75, 0.75]], dtype=torch.float32
        )
        self.register_buffer("boxes", boxes)
        self.box_head = nn.Sequential(nn.Linear(channels * 2 * 2, 32), nn.ReLU())
        self.classifier = nn.Linear(32, classes)
        self.box_regressor = nn.Linear(32, classes * 4)
        self.mask_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, classes, 1),
        )

    def roi_align(self, feature: torch.Tensor) -> torch.Tensor:
        """Sample fixed proposal windows with bilinear RoIAlign.

        Parameters
        ----------
        feature:
            FPN feature map used for RoIAlign.

        Returns
        -------
        torch.Tensor
            RoI features with shape ``(batch, proposals, channels, 2, 2)``.
        """

        batch = feature.shape[0]
        pooled = []
        base_y = torch.linspace(0.0, 1.0, 2, device=feature.device, dtype=feature.dtype)
        base_x = torch.linspace(0.0, 1.0, 2, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(base_y, base_x, indexing="ij")
        for proposal in range(self.proposals):
            box = self.boxes[proposal]
            x_grid = box[0] + (box[2] - box[0]) * xx
            y_grid = box[1] + (box[3] - box[1]) * yy
            grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
            pooled.append(F.grid_sample(feature, grid, align_corners=False))
        return torch.stack(pooled, dim=1)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run tiny LVIS Mask R-CNN inference heads.

        Parameters
        ----------
        image:
            Input RGB image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            RPN boxes, class logits, box deltas, and per-class masks.
        """

        p2, p3 = self.backbone(image)
        _objectness, rpn_boxes = self.rpn((p2, p3))
        roi = self.roi_align(p2)
        flat_roi = roi.flatten(2)
        box_features = self.box_head(flat_roi)
        mask_features = roi.reshape(image.shape[0] * self.proposals, roi.shape[2], 2, 2)
        masks = self.mask_head(mask_features).reshape(
            image.shape[0], self.proposals, self.classes, 2, 2
        )
        return rpn_boxes, self.classifier(box_features), self.box_regressor(box_features), masks


def build() -> nn.Module:
    """Build a compact LVIS Mask R-CNN.

    Returns
    -------
    nn.Module
        Random-initialized tiny detector.
    """

    return TinyLVISMaskRCNN().eval()


def example_input() -> torch.Tensor:
    """Create a low-resolution RGB image.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("d2_lvisv0.5_instancesegmentation__mask_rcnn", "build", "example_input", "2019", "DC"),
    ("d2_lvisv1_instancesegmentation__mask_rcnn", "build", "example_input", "2020", "DC"),
    # X-101 backbone config variants render the same Mask-RCNN architecture (tiny config)
    (
        "d2_lvisv0.5_instancesegmentation__mask_rcnn_X_101_32x8d_FPN_1x",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
    (
        "d2_lvisv1_instancesegmentation__mask_rcnn_X_101_32x8d_FPN_1x",
        "build",
        "example_input",
        "2020",
        "DC",
    ),
]
