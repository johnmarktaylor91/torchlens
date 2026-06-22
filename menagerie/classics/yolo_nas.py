"""YOLO-NAS / PP-YOLOE compact detectors.

YOLO-NAS was released in Deci-AI SuperGradients (2023).  Public descriptions
identify the inference architecture as a YOLO-style detector found by AutoNAC:
quantization-friendly RepVGG basic blocks are composed into QSP/QCI stages, a
PAN/FPN neck fuses multi-scale features, and an efficient decoupled detection
head predicts boxes/classes/objectness.  SuperGradients also ships PP-YOLOE
variants, whose distinctive public architecture is a CSP/ELAN-like backbone
with an FPN/PAN neck and decoupled anchor-free YOLO heads.

This module keeps the faithful inference primitives while using tiny widths and
small inputs so the graph renders in the base TorchLens environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch norm, and SiLU activation."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, stride: int = 1) -> None:
        """Initialize a YOLO-style convolution block.

        Parameters
        ----------
        c_in:
            Input channels.
        c_out:
            Output channels.
        k:
            Kernel size.
        stride:
            Spatial stride.
        """

        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block.

        Parameters
        ----------
        x:
            Input image feature map.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return F.silu(self.bn(self.conv(x)))


class RepBlock(nn.Module):
    """Training-time RepVGG block used by quantization-friendly YOLO-NAS blocks."""

    def __init__(self, channels: int) -> None:
        """Initialize parallel 3x3, 1x1, and identity branches.

        Parameters
        ----------
        channels:
            Number of input/output channels.
        """

        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.id_bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the unfused RepVGG branches.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            RepVGG feature map.
        """

        return F.silu(self.conv3(x) + self.conv1(x) + self.id_bn(x))


class QSPBlock(nn.Module):
    """YOLO-NAS QSP-style partial stage with RepVGG blocks."""

    def __init__(self, channels: int, depth: int) -> None:
        """Initialize a split-transform-concat stage.

        Parameters
        ----------
        channels:
            Stage channels.
        depth:
            Number of RepVGG transforms in the active branch.
        """

        super().__init__()
        half = channels // 2
        self.left = ConvBNAct(channels, half, 1)
        self.right = ConvBNAct(channels, half, 1)
        self.blocks = nn.Sequential(*[RepBlock(half) for _ in range(depth)])
        self.fuse = ConvBNAct(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-stage partial RepVGG processing.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Stage output.
        """

        y = self.blocks(self.left(x))
        return self.fuse(torch.cat([y, self.right(x)], dim=1))


class DecoupledHead(nn.Module):
    """YOLO-NAS/PP-YOLOE decoupled box, objectness, and class head."""

    def __init__(self, channels: int, classes: int) -> None:
        """Initialize one detection head branch.

        Parameters
        ----------
        channels:
            Input feature channels.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.box = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 4, 1))
        self.obj = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, 1, 1))
        self.cls = nn.Sequential(ConvBNAct(channels, channels), nn.Conv2d(channels, classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict dense detection logits at one feature scale.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Concatenated box/object/class logits.
        """

        return torch.cat([self.box(x), self.obj(x), self.cls(x)], dim=1)


class YOLONAS(nn.Module):
    """Compact YOLO-NAS detector with RepVGG-QSP stages and PAN neck."""

    def __init__(
        self, width: int = 16, depth: int = 1, classes: int = 5, pose: bool = False
    ) -> None:
        """Initialize the compact detector.

        Parameters
        ----------
        width:
            Base channel count.
        depth:
            Repetition count for QSP stages.
        classes:
            Number of classes.
        pose:
            Whether to append a keypoint heatmap branch.
        """

        super().__init__()
        self.pose = pose
        self.stem = ConvBNAct(3, width, 3, 2)
        self.s2 = nn.Sequential(ConvBNAct(width, width * 2, 3, 2), QSPBlock(width * 2, depth))
        self.s3 = nn.Sequential(ConvBNAct(width * 2, width * 4, 3, 2), QSPBlock(width * 4, depth))
        self.s4 = nn.Sequential(ConvBNAct(width * 4, width * 8, 3, 2), QSPBlock(width * 8, depth))
        self.lateral4 = ConvBNAct(width * 8, width * 4, 1)
        self.pan3 = QSPBlock(width * 8, depth)
        self.down = ConvBNAct(width * 8, width * 8, 3, 2)
        self.pan4 = QSPBlock(width * 16, depth)
        self.head3 = DecoupledHead(width * 8, classes)
        self.head4 = DecoupledHead(width * 16, classes)
        self.pose_head = nn.Conv2d(width * 8, 17 * 2, 1) if pose else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run YOLO-style multi-scale detection.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Low-scale detections, high-scale detections, and optional pose logits.
        """

        p2 = self.s2(self.stem(x))
        p3 = self.s3(p2)
        p4 = self.s4(p3)
        up = F.interpolate(self.lateral4(p4), size=p3.shape[-2:], mode="nearest")
        n3 = self.pan3(torch.cat([p3, up], dim=1))
        n4 = self.pan4(torch.cat([p4, self.down(n3)], dim=1))
        pose = self.pose_head(n3)
        if not self.pose:
            pose = pose.mean(dim=(2, 3), keepdim=True)
        return self.head3(n3), self.head4(n4), pose


class PPYOLOE(nn.Module):
    """Compact PP-YOLOE-style CSP detector."""

    def __init__(self, width: int = 16, depth: int = 1, classes: int = 5) -> None:
        """Initialize PP-YOLOE compact detector.

        Parameters
        ----------
        width:
            Base channel count.
        depth:
            Number of partial-stage blocks.
        classes:
            Number of classes.
        """

        super().__init__()
        self.model = YOLONAS(width=width, depth=depth, classes=classes, pose=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PP-YOLOE-style detection.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Detection logits.
        """

        return self.model(x)


def build_yolo_nas_s() -> nn.Module:
    """Build YOLO-NAS-S compact detector.

    Returns
    -------
    nn.Module
        Random-init compact YOLO-NAS-S.
    """

    return YOLONAS(width=12, depth=1)


def build_yolo_nas_m() -> nn.Module:
    """Build YOLO-NAS-M compact detector.

    Returns
    -------
    nn.Module
        Random-init compact YOLO-NAS-M.
    """

    return YOLONAS(width=16, depth=1)


def build_yolo_nas_l() -> nn.Module:
    """Build YOLO-NAS-L compact detector.

    Returns
    -------
    nn.Module
        Random-init compact YOLO-NAS-L.
    """

    return YOLONAS(width=20, depth=2)


def build_yolo_nas_pose_n() -> nn.Module:
    """Build YOLO-NAS-pose-N compact detector.

    Returns
    -------
    nn.Module
        Random-init compact YOLO-NAS pose N.
    """

    return YOLONAS(width=8, depth=1, pose=True)


def build_yolo_nas_pose_s() -> nn.Module:
    """Build YOLO-NAS-pose-S compact detector.

    Returns
    -------
    nn.Module
        Random-init compact YOLO-NAS pose S.
    """

    return YOLONAS(width=12, depth=1, pose=True)


def build_yolo_nas_pose_m() -> nn.Module:
    """Build YOLO-NAS-pose-M compact detector.

    Returns
    -------
    nn.Module
        Random-init compact YOLO-NAS pose M.
    """

    return YOLONAS(width=16, depth=1, pose=True)


def build_yolo_nas_pose_l() -> nn.Module:
    """Build YOLO-NAS-pose-L compact detector.

    Returns
    -------
    nn.Module
        Random-init compact YOLO-NAS pose L.
    """

    return YOLONAS(width=20, depth=2, pose=True)


def build_ppyoloe_s() -> nn.Module:
    """Build PP-YOLOE-S compact detector.

    Returns
    -------
    nn.Module
        Random-init compact PP-YOLOE-S.
    """

    return PPYOLOE(width=12, depth=1)


def build_ppyoloe_x() -> nn.Module:
    """Build PP-YOLOE-X compact detector.

    Returns
    -------
    nn.Module
        Random-init compact PP-YOLOE-X.
    """

    return PPYOLOE(width=20, depth=2)


def example_input() -> torch.Tensor:
    """Create a small RGB image input.

    Returns
    -------
    torch.Tensor
        Image tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "YOLO-NAS-S (RepVGG-QSP detector, PAN neck, decoupled head)",
        "build_yolo_nas_s",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "YOLO-NAS-M (RepVGG-QSP detector, PAN neck, decoupled head)",
        "build_yolo_nas_m",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "YOLO-NAS-L (RepVGG-QSP detector, PAN neck, decoupled head)",
        "build_yolo_nas_l",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "YOLO-NAS-pose-N (RepVGG-QSP detector with keypoint head)",
        "build_yolo_nas_pose_n",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "YOLO-NAS-pose-S (RepVGG-QSP detector with keypoint head)",
        "build_yolo_nas_pose_s",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "YOLO-NAS-pose-M (RepVGG-QSP detector with keypoint head)",
        "build_yolo_nas_pose_m",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "YOLO-NAS-pose-L (RepVGG-QSP detector with keypoint head)",
        "build_yolo_nas_pose_l",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "SuperGradients PP-YOLOE-S (CSP/PAN decoupled detector)",
        "build_ppyoloe_s",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "SuperGradients PP-YOLOE-X (CSP/PAN decoupled detector)",
        "build_ppyoloe_x",
        "example_input",
        "2022",
        "DC",
    ),
]
