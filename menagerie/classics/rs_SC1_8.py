# SOURCE: vendored from https://github.com/suvojit-0x55aa/A2S2K-ResNet @ 60b71b7 (A2S2KResNet/A2S2KResNet.py)
"""Vendored A2S2K-ResNet hyperspectral image classifier."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class EcaLayer(nn.Module):
    """Efficient channel attention layer from A2S2K-ResNet."""

    def __init__(self, channel: int, k_size: int = 3) -> None:
        """Initialize the ECA layer.

        Parameters
        ----------
        channel
            Number of channels in the input feature map.
        k_size
            1D attention kernel size.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run efficient channel attention.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reweighted tensor.
        """

        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(-1, -3).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Residual(nn.Module):
    """Residual 3D convolution block from A2S2K-ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        padding: tuple[int, int, int],
        use_1x1conv: bool = False,
        stride: int = 1,
        start_block: bool = False,
        end_block: bool = False,
    ) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        in_channels
            Input channel count.
        out_channels
            Output channel count.
        kernel_size
            3D convolution kernel size.
        padding
            3D convolution padding.
        use_1x1conv
            Whether to define a projection branch.
        stride
            Convolution stride.
        start_block
            Whether this is the first residual block.
        end_block
            Whether this is the last residual block.
        """

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride
        )
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        self.ecalayer = EcaLayer(out_channels)
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Parameters
        ----------
        X
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        identity = X

        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        out = self.ecalayer(out)
        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)

        return out


class S3KAIResNet(nn.Module):
    """Attention-based adaptive spectral-spatial kernel ResNet."""

    def __init__(self, band: int, classes: int, reduction: int, kernel_size: int = 24) -> None:
        """Initialize A2S2K-ResNet.

        Parameters
        ----------
        band
            Number of hyperspectral bands.
        classes
            Number of output classes.
        reduction
            Attention reduction ratio.
        kernel_size
            Number of spectral-spatial kernel channels.
        """

        super().__init__()
        self.name = "SSRN"
        self.kernel_size = kernel_size
        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=kernel_size,
            kernel_size=(1, 1, 7),
            stride=(1, 1, 2),
            padding=0,
        )
        self.conv3x3 = nn.Conv3d(
            in_channels=1,
            out_channels=kernel_size,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 2),
            padding=(1, 1, 0),
        )

        self.batch_norm1x1 = nn.Sequential(
            nn.BatchNorm3d(kernel_size, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True)
        )
        self.batch_norm3x3 = nn.Sequential(
            nn.BatchNorm3d(kernel_size, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(kernel_size, band // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv_ex = nn.Conv3d(band // reduction, kernel_size, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = Residual(kernel_size, kernel_size, (1, 1, 7), (0, 0, 3), start_block=True)
        self.res_net2 = Residual(kernel_size, kernel_size, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(kernel_size, kernel_size, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(kernel_size, kernel_size, (3, 3, 1), (1, 1, 0), end_block=True)

        kernel_3d = math.ceil((band - 6) / 2)
        self.conv2 = nn.Conv3d(
            in_channels=kernel_size,
            out_channels=128,
            padding=(0, 0, 0),
            kernel_size=(1, 1, kernel_3d),
            stride=(1, 1, 1),
        )
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=kernel_size,
            padding=(0, 0, 0),
            kernel_size=(3, 3, 128),
            stride=(1, 1, 1),
        )
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(kernel_size, eps=0.001, momentum=0.1, affine=True), nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(nn.Linear(kernel_size, classes))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Run A2S2K-ResNet.

        Parameters
        ----------
        X
            Hyperspectral patch tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x_1x1 = self.conv1x1(X)
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)
        x_3x3 = self.conv3x3(X)
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)

        x1 = torch.cat([x_3x3, x_1x1], dim=1)
        U = torch.sum(x1, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat(
            [self.conv_ex(Z).unsqueeze(dim=1), self.conv_ex(Z).unsqueeze(dim=1)], dim=1
        )
        attention_vector = self.softmax(attention_vector)
        V = (x1 * attention_vector).sum(dim=1)

        x2 = self.res_net1(V)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        return self.full_connection(x4)


def build_a2s2k_resnet() -> S3KAIResNet:
    """Build a traceable A2S2K-ResNet model.

    Returns
    -------
    S3KAIResNet
        Model instance.
    """

    return S3KAIResNet(band=24, classes=8, reduction=2, kernel_size=24)


def example_input_a2s2k_resnet() -> torch.Tensor:
    """Return a sample hyperspectral patch.

    Returns
    -------
    torch.Tensor
        Example input.
    """

    return torch.randn(1, 1, 9, 9, 24)


MENAGERIE_ENTRIES: list[tuple[str, str, str, int, str]] = [
    ("A2S2K-ResNet", "build_a2s2k_resnet", "example_input_a2s2k_resnet", 2020, "SC1-008"),
]
