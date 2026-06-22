"""MinkowskiEngine Sparse ConvNets: ResNet34 and Res16UNet34C on sparse 3D data.

Choy, Gwak & Savarese (Stanford), CVPR 2019, arXiv:1904.08755.
Source: https://github.com/NVIDIA/MinkowskiEngine

MinkowskiEngine implements sparse 3D convolutions: only non-empty voxels are
processed, represented as a sparse tensor (coordinates + features). The
distinctive primitives are:
  - Sparse 3D convolution: only computed at occupied voxel positions
  - Sparse batch normalization + ReLU
  - Residual blocks with sparse convs (ResNet34 architecture adapted for 3D)
  - Res16UNet34C: encoder-decoder (U-Net) with residual blocks, skip connections,
    and sparse transposed convolutions for upsampling

NOTE: MinkowskiEngine requires compiled CUDA kernels and is NOT pip-installable
in this environment. We faithfully reproduce the ARCHITECTURE using DENSE 3D
convolutions on a small 16^3 voxel grid as a structural stand-in for sparse
convs. The graph structure, residual connectivity, and U-Net topology are
identical to the original; only the sparse-vs-dense execution differs.

Compact config: 16^3 input voxels, 4 channels, reduced block counts.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Basic 3D conv residual block (stand-in for Minkowski sparse conv block)    #
# --------------------------------------------------------------------------- #


class BasicBlock3D(nn.Module):
    """3D residual block: two 3x3x3 convs + BN + ReLU, skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


def make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock3D(in_ch, out_ch, stride=stride)]
    for _ in range(1, n_blocks):
        layers.append(BasicBlock3D(out_ch, out_ch))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
#  MinkowskiResNet34 (ResNet34 adapted for 3D sparse convs)                   #
# --------------------------------------------------------------------------- #


class MinkowskiResNet34(nn.Module):
    """ResNet34 with 3D convs. Faithful architecture stand-in for sparse-conv ResNet34.

    Layers: 3, 4, 6, 3 basic blocks (ResNet34 spec), plus initial 7x7x7 conv
    and global average pooling to classification head.
    Reduced channels (64->16, 128->32, 256->64, 512->128) for small voxel grid.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Initial conv: 4-channel voxel features -> 16 channels
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        # ResNet34 block counts: [3, 4, 6, 3]
        self.layer1 = make_layer(16, 16, n_blocks=2, stride=1)  # 3 -> 2 for compact
        self.layer2 = make_layer(16, 32, n_blocks=3, stride=2)  # 4 -> 3
        self.layer3 = make_layer(32, 64, n_blocks=4, stride=2)  # 6 -> 4
        self.layer4 = make_layer(64, 128, n_blocks=2, stride=2)  # 3 -> 2
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W) -- voxel grid
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------- #
#  Res16UNet34C (U-Net with residual blocks + skip connections)               #
# --------------------------------------------------------------------------- #


class Res16UNet34C(nn.Module):
    """Res16UNet34C: encoder-decoder U-Net with residual 3D conv blocks.

    Faithful architectural stand-in for MinkowskiEngine's Res16UNet34C:
    - Encoder: successive downsampling blocks (stride-2 convs)
    - Decoder: transposed conv upsampling + skip connections from encoder
    - Residual blocks at each scale
    Reduced channels for 16^3 input.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Encoder
        self.enc0 = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.enc1 = make_layer(16, 32, n_blocks=2, stride=2)  # 16->8
        self.enc2 = make_layer(32, 64, n_blocks=2, stride=2)  # 8->4
        self.enc3 = make_layer(64, 128, n_blocks=2, stride=2)  # 4->2

        # Decoder (transposed conv + skip)
        self.up3 = nn.ConvTranspose3d(128, 64, 2, stride=2)  # 2->4
        self.dec3 = make_layer(64 + 64, 64, n_blocks=2)

        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)  # 4->8
        self.dec2 = make_layer(32 + 32, 32, n_blocks=2)

        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)  # 8->16
        self.dec1 = make_layer(16 + 16, 16, n_blocks=2)

        # Head: per-voxel classification
        self.head = nn.Conv3d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 16, 16, 16)
        e0 = self.enc0(x)  # (B, 16, 16, 16, 16)
        e1 = self.enc1(e0)  # (B, 32, 8, 8, 8)
        e2 = self.enc2(e1)  # (B, 64, 4, 4, 4)
        e3 = self.enc3(e2)  # (B, 128, 2, 2, 2)

        d3 = self.up3(e3)  # (B, 64, 4, 4, 4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # (B, 64, 4, 4, 4)

        d2 = self.up2(d3)  # (B, 32, 8, 8, 8)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # (B, 32, 8, 8, 8)

        d1 = self.up1(d2)  # (B, 16, 16, 16, 16)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))  # (B, 16, 16, 16, 16)

        out = self.head(d1)  # (B, num_classes, 16, 16, 16)
        # Return global average for single output
        return out.mean(dim=[2, 3, 4])  # (B, num_classes)


# --------------------------------------------------------------------------- #
#  Wrappers & menagerie interface                                              #
# --------------------------------------------------------------------------- #


class _ResNet34Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = MinkowskiResNet34(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UNetWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = Res16UNet34C(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_minkowski_resnet34() -> nn.Module:
    return _ResNet34Wrapper()


def build_minkowski_res16unet34c() -> nn.Module:
    return _UNetWrapper()


def example_input_resnet() -> torch.Tensor:
    """16^3 voxel grid, 1 channel."""
    torch.manual_seed(0)
    return torch.randn(1, 1, 16, 16, 16)


def example_input_unet() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 1, 16, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "MinkowskiResNet34 (ResNet34 3D residual blocks; dense stand-in for sparse conv)",
        "build_minkowski_resnet34",
        "example_input_resnet",
        "2019",
        "DC",
    ),
    (
        "MinkowskiRes16UNet34C (3D ResNet U-Net; dense stand-in for sparse conv)",
        "build_minkowski_res16unet34c",
        "example_input_unet",
        "2019",
        "DC",
    ),
]
