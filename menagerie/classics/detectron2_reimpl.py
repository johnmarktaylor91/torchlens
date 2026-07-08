"""Detectron2-style ResNet/FPN detectors.

Paper: Ren et al. 2015, "Faster R-CNN"; He et al. 2017, "Mask R-CNN"; Cai and
Vasconcelos 2018, "Cascade R-CNN"; Lin et al. 2017, "Focal Loss for Dense Object
Detection"; Lin et al. 2017, "Feature Pyramid Networks".

This compact Torch-only implementation preserves the traced detector structure
behind the missing Detectron2 model-zoo rows: a ResNet-like convolutional
backbone, optional FPN pyramid, RPN objectness/box heads, ROI-style box heads,
optional mask head, cascade refinement stages, and RetinaNet dense class/box
heads. It uses random synthetic image tensors rather than Detectron2's
``list[dict(image=...)]`` wrapper so TorchLens can validate it in the base
environment.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualStage(nn.Module):
    """Small ResNet stage with residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int) -> None:
        """Initialize a residual stage.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        stride:
            Stride of the first block.
        depth:
            Number of residual blocks.
        """
        super().__init__()
        blocks = []
        current = in_channels
        for index in range(depth):
            block_stride = stride if index == 0 else 1
            blocks.append(self._make_block(current, out_channels, block_stride))
            current = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual stage.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Stage output.
        """
        return self.blocks(x)

    @staticmethod
    def _make_block(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Create one residual block.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        stride:
            First convolution stride.

        Returns
        -------
        nn.Module
            Residual block.
        """

        class Block(nn.Module):
            """Residual block closure."""

            def __init__(self) -> None:
                """Initialize residual block layers."""
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.skip = (
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride)
                    if in_channels != out_channels or stride != 1
                    else nn.Identity()
                )

            def forward(self, x: Tensor) -> Tensor:
                """Apply the residual block.

                Parameters
                ----------
                x:
                    Feature map.

                Returns
                -------
                Tensor
                    Block output.
                """
                y = F.relu(self.bn1(self.conv1(x)))
                y = self.bn2(self.conv2(y))
                return F.relu(y + self.skip(x))

        return Block()


class ResNetBackbone(nn.Module):
    """Compact C4/C5 ResNet backbone."""

    def __init__(self, depth: int = 50) -> None:
        """Initialize backbone stages.

        Parameters
        ----------
        depth:
            ResNet depth marker; ``101`` uses deeper final stages.
        """
        super().__init__()
        stage_depth = 3 if depth >= 101 else 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.res2 = ResidualStage(32, 48, stride=1, depth=2)
        self.res3 = ResidualStage(48, 64, stride=2, depth=2)
        self.res4 = ResidualStage(64, 96, stride=2, depth=stage_depth)
        self.res5 = ResidualStage(96, 128, stride=2, depth=stage_depth)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return multi-scale backbone features.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        dict[str, Tensor]
            Feature maps keyed by stage.
        """
        c1 = self.stem(x)
        c2 = self.res2(c1)
        c3 = self.res3(c2)
        c4 = self.res4(c3)
        c5 = self.res5(c4)
        return {"c3": c3, "c4": c4, "c5": c5}


class FPN(nn.Module):
    """Feature Pyramid Network neck."""

    def __init__(self, out_channels: int = 64) -> None:
        """Initialize lateral and output convolutions.

        Parameters
        ----------
        out_channels:
            Pyramid feature width.
        """
        super().__init__()
        self.lateral3 = nn.Conv2d(64, out_channels, 1)
        self.lateral4 = nn.Conv2d(96, out_channels, 1)
        self.lateral5 = nn.Conv2d(128, out_channels, 1)
        self.out3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features: dict[str, Tensor]) -> list[Tensor]:
        """Build top-down pyramid features.

        Parameters
        ----------
        features:
            Backbone features.

        Returns
        -------
        list[Tensor]
            Pyramid features from fine to coarse.
        """
        p5 = self.lateral5(features["c5"])
        p4 = self.lateral4(features["c4"]) + F.interpolate(p5, size=features["c4"].shape[-2:])
        p3 = self.lateral3(features["c3"]) + F.interpolate(p4, size=features["c3"].shape[-2:])
        return [self.out3(p3), self.out4(p4), self.out5(p5)]


class RPNHead(nn.Module):
    """Region proposal network head."""

    def __init__(self, channels: int = 64, anchors: int = 3) -> None:
        """Initialize RPN convolutions.

        Parameters
        ----------
        channels:
            Feature channel count.
        anchors:
            Anchors per spatial location.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.objectness = nn.Conv2d(channels, anchors, 1)
        self.box_delta = nn.Conv2d(channels, anchors * 4, 1)

    def forward(self, features: list[Tensor]) -> Tensor:
        """Compute RPN objectness and box deltas.

        Parameters
        ----------
        features:
            Pyramid features.

        Returns
        -------
        Tensor
            Concatenated proposal summary.
        """
        summaries = []
        for feature in features:
            hidden = F.relu(self.conv(feature))
            objectness = self.objectness(hidden).mean(dim=(2, 3))
            box_delta = self.box_delta(hidden).mean(dim=(2, 3))
            summaries.append(torch.cat((objectness, box_delta), dim=1))
        return torch.cat(summaries, dim=1)


class ROIHeads(nn.Module):
    """ROI box, mask, and cascade heads over pooled pyramid features."""

    def __init__(self, channels: int = 64, mode: str = "faster") -> None:
        """Initialize ROI heads.

        Parameters
        ----------
        channels:
            Pyramid feature channel count.
        mode:
            Detector mode.
        """
        super().__init__()
        self.mode = mode
        self.box = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 84),
        )
        self.cascade2 = nn.Linear(84, 84)
        self.cascade3 = nn.Linear(84, 84)
        self.mask = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, 80, 1),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        """Compute ROI-style predictions.

        Parameters
        ----------
        features:
            Pyramid features.

        Returns
        -------
        Tensor
            Prediction tensor.
        """
        pooled_map = F.adaptive_avg_pool2d(features[0], (7, 7))
        pooled = pooled_map.mean(dim=(2, 3))
        box_logits = self.box(pooled)
        outputs = [box_logits]
        if self.mode == "cascade":
            refined = F.relu(self.cascade2(box_logits))
            outputs.append(self.cascade3(refined))
        if self.mode == "mask":
            outputs.append(self.mask(pooled_map).mean(dim=(2, 3)))
        return torch.cat(outputs, dim=1)


class RetinaNetHead(nn.Module):
    """Dense RetinaNet classification and box-regression head."""

    def __init__(self, channels: int = 64, anchors: int = 9, classes: int = 80) -> None:
        """Initialize dense prediction towers.

        Parameters
        ----------
        channels:
            Pyramid feature channel count.
        anchors:
            Anchors per location.
        classes:
            Number of classes.
        """
        super().__init__()
        tower = []
        for _ in range(3):
            tower.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=False)])
        self.tower = nn.Sequential(*tower)
        self.cls = nn.Conv2d(channels, anchors * classes, 3, padding=1)
        self.box = nn.Conv2d(channels, anchors * 4, 3, padding=1)

    def forward(self, features: list[Tensor]) -> Tensor:
        """Compute dense RetinaNet summaries.

        Parameters
        ----------
        features:
            Pyramid features.

        Returns
        -------
        Tensor
            Dense prediction summary.
        """
        summaries = []
        for feature in features:
            hidden = self.tower(feature)
            summaries.append(
                torch.cat(
                    (self.cls(hidden).mean(dim=(2, 3)), self.box(hidden).mean(dim=(2, 3))), dim=1
                )
            )
        return torch.cat(summaries, dim=1)


class Detectron2LikeDetector(nn.Module):
    """Compact Detectron2-style detector family."""

    def __init__(self, mode: str, depth: int = 50, use_fpn: bool = True) -> None:
        """Initialize backbone, neck, and task heads.

        Parameters
        ----------
        mode:
            Detector mode: ``"faster"``, ``"mask"``, ``"cascade"``, ``"retina"``, or ``"rpn"``.
        depth:
            ResNet depth marker.
        use_fpn:
            Whether to use FPN features or a single C4-style feature.
        """
        super().__init__()
        self.mode = mode
        self.use_fpn = use_fpn
        self.backbone = ResNetBackbone(depth=depth)
        self.fpn = FPN() if use_fpn else nn.Conv2d(96, 64, 1)
        self.rpn = RPNHead()
        self.roi = ROIHeads(
            mode="cascade" if mode == "cascade" else "mask" if mode == "mask" else "faster"
        )
        self.retina = RetinaNetHead()

    def forward(self, images: Tensor) -> Tensor:
        """Run the detector on image tensors.

        Parameters
        ----------
        images:
            Tensor with shape ``(batch, 3, height, width)``.

        Returns
        -------
        Tensor
            Concatenated detector predictions.
        """
        features = self.backbone(images)
        if self.use_fpn:
            pyramid = self.fpn(features)
        else:
            pyramid = [self.fpn(features["c4"])]
        if self.mode == "retina":
            return self.retina(pyramid)
        proposals = self.rpn(pyramid)
        if self.mode == "rpn":
            return proposals
        return torch.cat((proposals, self.roi(pyramid)), dim=1)


def build_detector(mode: str, depth: int = 50, use_fpn: bool = True) -> nn.Module:
    """Build a compact Detectron2-style detector.

    Parameters
    ----------
    mode:
        Detector mode.
    depth:
        ResNet depth marker.
    use_fpn:
        Whether to use FPN features.

    Returns
    -------
    nn.Module
        Detector model.
    """
    return Detectron2LikeDetector(mode=mode, depth=depth, use_fpn=use_fpn)
