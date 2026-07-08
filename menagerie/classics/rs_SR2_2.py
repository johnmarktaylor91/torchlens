# FAITHFUL REIMPLEMENTATION from Kabir, Lee & Lee, "Deep Learning-Based Detection
# of Aflatoxin B1 Contamination in Almonds Using Hyperspectral Imaging: A Focus on
# Optimized 3D Inception-ResNet Model" (Toxins 17(4):156, 2025,
# https://doi.org/10.3390/toxins17040156, open access) and its attention-guided
# successor Kabir, Lee & Lee, "Squeeze-Excitation Attention-Guided 3D Inception
# ResNet for Aflatoxin B1 Classification in Almonds Using Hyperspectral Imaging"
# (AGIR-3DNet, Toxins 18(2):76, https://www.mdpi.com/2072-6651/18/2/76).
#
# GitHub code search (`gh search code`/`gh search repos`) and web search found no
# public repository for either paper; both are describe-only. The base paper's
# Section 3 ("Research Methodology") gives an exact block-level architecture
# description, transcribed below:
#
#   3.1 3D ResNet: 3D conv stem [3x3x3, stride 2x2x2] -> BN -> ReLU, followed by
#   bottleneck residual blocks, each [1x1x1] (reduce) -> [3x3x3] (extract) ->
#   [1x1x1] (restore), with y_residual = x + F(x) (Eq. 3), then global average
#   pooling (Eq. 4) -> FC -> softmax (Eq. 5).
#
#   3.2 3D Inception: initial [3x3x3] conv followed by [3x3x1] then [1x1x1],
#   stride [2x2x2]. Three Inception blocks (A, B, C). Inception block A has 4
#   parallel branches (Eq. 6): branch 1/2 = [1x1x1]; branch 3 = [1x1x1] ->
#   [5x5x5]; branch 4 = [1x1x1] -> [3x3x3] -> [3x3x3]; outputs concatenated along
#   the channel/depth dimension. A [3x3x1] max-pool + [1x1x1] bottleneck handles
#   downsampling between blocks. Blocks B/C use factorized [7x1x1]/[1x7x1]
#   kernels in place of the cubic kernels "to improve computational cost and
#   gradient flow".
#
#   3.3 3D Inception-ResNet: Inception-ResNet blocks combine multi-branch
#   Inception-style feature extraction with a residual connection (Eq. 7-8):
#   y_mixed = concat( F_1x1x1(x), F_3x3x3(F_1x1x1(x)), F_3x3x3(F_3x3x3(F_1x1x1(x))) ),
#   y_residual = alpha * y_mixed + x, where alpha scales the Inception branch's
#   contribution before the residual add (mirroring the original 2D
#   Inception-ResNet-v2's linear projection + scaled residual pattern) --
#   requires a channel-matching linear (no-activation) projection of the
#   concatenated branches back to the block's input channel count, which this
#   module implements as `proj` below.
#
# This module implements one Inception-ResNet-A block, one reduction block
# (strided conv, per "two different reduction blocks" between Inception stages),
# one Inception-ResNet-B block using the paper's [7x1x1]/[1x7x1] factorized
# kernels, a second reduction block, and one Inception-ResNet-C block, followed
# by global average pooling and an FC+softmax classification head -- the same
# stage sequence as the paper's Fig. 4 framework, at a tiny random-init scale
# (the paper's own Deep/Lightweight variants are 824/381 layers; this menagerie
# entry captures the architecture family, not the full block-repeat depth).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


class ConvBNReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _same_pad(k):
    return tuple(v // 2 for v in k) if isinstance(k, tuple) else k // 2


class InceptionResNetBlockA(nn.Module):
    """Inception-ResNet-A block (paper Eq. 6-8): branches with [1x1x1],
    [1x1x1]->[3x3x3], and [1x1x1]->[3x3x3]->[3x3x3], concatenated, linearly
    projected back to the input channel count, and added to the input with a
    learned residual scale alpha."""

    def __init__(self, channels, branch_channels=8, alpha=0.17):
        super().__init__()
        self.branch0 = ConvBNReLU3D(channels, branch_channels, 1)
        self.branch1 = nn.Sequential(
            ConvBNReLU3D(channels, branch_channels, 1),
            ConvBNReLU3D(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch2 = nn.Sequential(
            ConvBNReLU3D(channels, branch_channels, 1),
            ConvBNReLU3D(branch_channels, branch_channels, 3, padding=1),
            ConvBNReLU3D(branch_channels, branch_channels, 3, padding=1),
        )
        self.proj = nn.Conv3d(branch_channels * 3, channels, kernel_size=1)
        self.alpha = alpha
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mixed = torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], dim=1)
        y_residual = x + self.alpha * self.proj(mixed)
        return self.relu(y_residual)


class InceptionResNetBlockB(nn.Module):
    """Inception-ResNet-B block using the paper's factorized [7x1x1]/[1x7x1]
    kernels "to improve computational cost and gradient flow" in the deeper
    Inception-ResNet stages."""

    def __init__(self, channels, branch_channels=8, alpha=0.10):
        super().__init__()
        self.branch0 = ConvBNReLU3D(channels, branch_channels, 1)
        self.branch1 = nn.Sequential(
            ConvBNReLU3D(channels, branch_channels, 1),
            ConvBNReLU3D(branch_channels, branch_channels, (1, 7, 1), padding=(0, 3, 0)),
            ConvBNReLU3D(branch_channels, branch_channels, (7, 1, 1), padding=(3, 0, 0)),
        )
        self.proj = nn.Conv3d(branch_channels * 2, channels, kernel_size=1)
        self.alpha = alpha
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mixed = torch.cat([self.branch0(x), self.branch1(x)], dim=1)
        y_residual = x + self.alpha * self.proj(mixed)
        return self.relu(y_residual)


class ReductionBlock3D(nn.Module):
    """Strided-conv reduction block between Inception-ResNet stages (paper:
    "two different reduction blocks" handle downsampling between blocks A/B/C)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBNReLU3D(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Inception3DResNet(nn.Module):
    """3D Inception-ResNet (Kabir, Lee & Lee 2025): 3D conv stem, Inception-
    ResNet-A block, reduction, Inception-ResNet-B block (factorized kernels),
    reduction, Inception-ResNet-C block, global average pooling, FC+softmax
    classification head (contaminated vs. non-contaminated)."""

    def __init__(self, in_channels=1, stem_channels=8, n_classes=2):
        super().__init__()
        # Stem: [3x3x3] stride 2 -> [3x3x1] -> [1x1x1], per Sec. 3.2.
        self.stem = nn.Sequential(
            ConvBNReLU3D(in_channels, stem_channels, 3, stride=2, padding=1),
            ConvBNReLU3D(stem_channels, stem_channels, (3, 3, 1), padding=(1, 1, 0)),
            ConvBNReLU3D(stem_channels, stem_channels, 1),
        )
        self.block_a = InceptionResNetBlockA(stem_channels)
        self.reduction1 = ReductionBlock3D(stem_channels, stem_channels * 2)
        self.block_b = InceptionResNetBlockB(stem_channels * 2)
        self.reduction2 = ReductionBlock3D(stem_channels * 2, stem_channels * 4)
        self.block_c = InceptionResNetBlockA(stem_channels * 4)  # Block C: same
        # multi-branch+residual template as Block A per Eq. 7-8, at the
        # network's final (coarsest) stage.
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(stem_channels * 4, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block_a(x)
        x = self.reduction1(x)
        x = self.block_b(x)
        x = self.reduction2(x)
        x = self.block_c(x)
        x = self.gap(x).flatten(1)
        return torch.softmax(self.fc(x), dim=-1)


def build_3d_inception_resnet_aflatoxin():
    return Inception3DResNet(in_channels=1, stem_channels=8, n_classes=2)


def example_input_3d_inception_resnet_aflatoxin():
    # (N, C, spectral_depth, H, W) hyperspectral datacube tile, tiny for tracing
    # (the paper's real ROI is 150x100x224; this menagerie entry captures the
    # architecture family, not the full-resolution input).
    return (torch.rand(1, 1, 16, 16, 16),)


MENAGERIE_ENTRIES = [
    (
        "3D Inception-ResNet for aflatoxin detection",
        "build_3d_inception_resnet_aflatoxin",
        "example_input_3d_inception_resnet_aflatoxin",
        2025,
        "reimpl-pytorch",
    ),
]
