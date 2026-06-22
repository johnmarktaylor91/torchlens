"""PaddleDetection Cascade R-CNN: FPN/RPN with cascaded box heads.

Cai and Vasconcelos (CVPR 2018), "Cascade R-CNN: Delving Into High Quality
Object Detection".  Cascade R-CNN uses a proposal generator followed by a
sequence of detection heads trained at increasing IoU thresholds; each stage
refines boxes for the next stage.

This compact reconstruction keeps the defining pieces used by PaddleDetection:
a convolutional backbone, top-down FPN fusion, RPN objectness/regression heads,
and three sequential ROI heads whose class and box outputs are fed forward as
refined proposal features.  ROI extraction is represented by adaptive pooling
over the FPN feature map to keep the graph deterministic and package-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBackbone(nn.Module):
    """Small convolutional backbone producing two feature levels."""

    def __init__(self, width: int = 24) -> None:
        """Initialize the backbone.

        Parameters
        ----------
        width:
            Base channel count.
        """

        super().__init__()
        self.c3 = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower and higher-level feature maps.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            C3 and C4 feature maps.
        """

        c3 = self.c3(x)
        return c3, self.c4(c3)


class FPN(nn.Module):
    """Two-level feature pyramid network."""

    def __init__(self, in_low: int = 24, in_high: int = 48, out_channels: int = 32) -> None:
        """Initialize lateral and smoothing convolutions.

        Parameters
        ----------
        in_low:
            Lower-level input channels.
        in_high:
            Higher-level input channels.
        out_channels:
            FPN output channels.
        """

        super().__init__()
        self.lat_low = nn.Conv2d(in_low, out_channels, 1)
        self.lat_high = nn.Conv2d(in_high, out_channels, 1)
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse feature levels top-down.

        Parameters
        ----------
        feats:
            C3 and C4 feature maps.

        Returns
        -------
        torch.Tensor
            Fused pyramid feature map.
        """

        low, high = feats
        high = self.lat_high(high)
        low = self.lat_low(low) + F.interpolate(high, size=low.shape[-2:], mode="nearest")
        return F.relu(self.smooth(low))


class RPNHead(nn.Module):
    """Region proposal head with objectness and anchor-box deltas."""

    def __init__(self, channels: int = 32, anchors: int = 3) -> None:
        """Initialize the RPN head.

        Parameters
        ----------
        channels:
            Feature channels.
        anchors:
            Number of anchors per spatial location.
        """

        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.objectness = nn.Conv2d(channels, anchors, 1)
        self.box_delta = nn.Conv2d(channels, anchors * 4, 1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict objectness and proposal deltas.

        Parameters
        ----------
        feat:
            FPN feature map.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            RPN objectness and box-delta maps.
        """

        hidden = F.relu(self.conv(feat))
        return self.objectness(hidden), self.box_delta(hidden)


class CascadeStage(nn.Module):
    """One Cascade R-CNN detection refinement stage."""

    def __init__(self, in_dim: int = 32 * 4 * 4, hidden: int = 64, classes: int = 5) -> None:
        """Initialize the ROI classification/regression head.

        Parameters
        ----------
        in_dim:
            Flattened pooled feature dimension.
        hidden:
            Hidden layer dimension.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cls = nn.Linear(hidden, classes)
        self.box = nn.Linear(hidden, classes * 4)
        self.refine = nn.Linear(classes * 4, in_dim)

    def forward(self, roi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one cascade stage and produce refined ROI features.

        Parameters
        ----------
        roi:
            Flattened ROI feature tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Class logits, box deltas, and refined ROI features.
        """

        hidden = F.relu(self.fc1(roi))
        hidden = F.relu(self.fc2(hidden))
        boxes = self.box(hidden)
        return self.cls(hidden), boxes, roi + 0.05 * torch.tanh(self.refine(boxes))


class PaddleDetCascadeRCNN(nn.Module):
    """Compact Cascade R-CNN detector."""

    def __init__(self) -> None:
        """Initialize the compact detector."""

        super().__init__()
        self.backbone = TinyBackbone()
        self.fpn = FPN()
        self.rpn = RPNHead()
        self.stages = nn.ModuleList([CascadeStage() for _ in range(3)])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run backbone, RPN, and cascaded ROI heads.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            RPN predictions and three-stage cascade predictions.
        """

        feat = self.fpn(self.backbone(x))
        rpn_logits, rpn_boxes = self.rpn(feat)
        roi = F.adaptive_avg_pool2d(feat, (4, 4)).flatten(1)
        outputs: dict[str, torch.Tensor] = {"rpn_logits": rpn_logits, "rpn_boxes": rpn_boxes}
        for idx, stage in enumerate(self.stages):
            cls, box, roi = stage(roi)
            outputs[f"stage{idx + 1}_cls"] = cls
            outputs[f"stage{idx + 1}_box"] = box
        return outputs


def build() -> nn.Module:
    """Build the compact PaddleDetection Cascade R-CNN model.

    Returns
    -------
    nn.Module
        Random-init detector in evaluation mode.
    """

    return PaddleDetCascadeRCNN().eval()


def example_input() -> torch.Tensor:
    """Return a small image batch for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_cascade_rcnn", "build", "example_input", "2018", "DC"),
]
