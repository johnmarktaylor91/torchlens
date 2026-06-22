"""YOLOX with CSPDarknet backbone and decoupled anchor-free head.

Ge et al., 2021.
Paper: https://arxiv.org/abs/2107.08430

YOLOX modernizes the YOLO family with a CSPDarknet backbone, SPP bottleneck,
PAN/FPN-style feature fusion, and a decoupled anchor-free detection head with
separate classification, objectness, and box-regression branches.  This compact
model keeps those primitives and emits dense grid predictions at one scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        kernel:
            Kernel size.
        stride:
            Stride.
        """

        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.block(x)


class Bottleneck(nn.Module):
    """YOLOX residual bottleneck."""

    def __init__(self, channels: int) -> None:
        """Initialize a residual bottleneck.

        Parameters
        ----------
        channels:
            Channel count.
        """

        super().__init__()
        hidden = channels // 2
        self.net = nn.Sequential(ConvBNAct(channels, hidden, 1), ConvBNAct(hidden, channels, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual bottleneck.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        return x + self.net(x)


class CSPLayer(nn.Module):
    """Cross Stage Partial layer used in CSPDarknet."""

    def __init__(self, in_ch: int, out_ch: int, blocks: int = 1) -> None:
        """Initialize split, residual, and merge paths.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        blocks:
            Number of residual blocks on the processed branch.
        """

        super().__init__()
        hidden = out_ch // 2
        self.left = ConvBNAct(in_ch, hidden, 1)
        self.right = ConvBNAct(in_ch, hidden, 1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden) for _ in range(blocks)])
        self.merge = ConvBNAct(hidden * 2, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run CSP split-transform-concat.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            CSP output.
        """

        return self.merge(torch.cat([self.blocks(self.left(x)), self.right(x)], dim=1))


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling bottleneck from YOLOX."""

    def __init__(self, channels: int) -> None:
        """Initialize SPP branches.

        Parameters
        ----------
        channels:
            Input and output channels.
        """

        super().__init__()
        hidden = channels // 2
        self.pre = ConvBNAct(channels, hidden, 1)
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in (5, 9, 13)]
        )
        self.post = ConvBNAct(hidden * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-kernel SPP.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            SPP output.
        """

        x = self.pre(x)
        return self.post(torch.cat([x, *[pool(x) for pool in self.pools]], dim=1))


class YOLOXDarknetCompact(nn.Module):
    """Compact YOLOX detector with CSPDarknet and decoupled head."""

    def __init__(self, classes: int = 3) -> None:
        """Initialize the compact YOLOX model.

        Parameters
        ----------
        classes:
            Number of object classes.
        """

        super().__init__()
        self.stem = ConvBNAct(3, 16, 3, stride=2)
        self.dark2 = nn.Sequential(ConvBNAct(16, 32, 3, stride=2), CSPLayer(32, 32))
        self.dark3 = nn.Sequential(ConvBNAct(32, 64, 3, stride=2), CSPLayer(64, 64, blocks=2))
        self.spp = SPPBottleneck(64)
        self.reduce = ConvBNAct(64, 32, 1)
        self.pan = CSPLayer(64, 32)
        self.cls_tower = nn.Sequential(ConvBNAct(32, 32, 3), ConvBNAct(32, 32, 3))
        self.reg_tower = nn.Sequential(ConvBNAct(32, 32, 3), ConvBNAct(32, 32, 3))
        self.cls_pred = nn.Conv2d(32, classes, 1)
        self.obj_pred = nn.Conv2d(32, 1, 1)
        self.box_pred = nn.Conv2d(32, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict anchor-free dense detections.

        Parameters
        ----------
        x:
            RGB image tensor with shape ``(batch, 3, 64, 64)``.

        Returns
        -------
        torch.Tensor
            Dense predictions with channels ``box4 + objectness + classes``.
        """

        c2 = self.dark2(self.stem(x))
        c3 = self.spp(self.dark3(c2))
        up = torch.nn.functional.interpolate(self.reduce(c3), size=c2.shape[-2:], mode="nearest")
        feat = self.pan(torch.cat([up, c2], dim=1))
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)
        pred = torch.cat(
            [self.box_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], 1
        )
        return pred.flatten(2).transpose(1, 2)


def build() -> nn.Module:
    """Build compact YOLOX-Darknet.

    Returns
    -------
    nn.Module
        Random-init YOLOX model.
    """

    return YOLOXDarknetCompact()


def example_input() -> torch.Tensor:
    """Create a small RGB detector input.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("megvii_yolox_darknet", "build", "example_input", "2021", "DC"),
]
