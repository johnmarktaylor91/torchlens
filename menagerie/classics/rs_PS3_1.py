# FAITHFUL REIMPLEMENTATION from Zhao, Yuan, Zhang, Sun 2022 "A Transfer Learning
# Framework with a One-Dimensional Deep Subdomain Adaptation Network for Bearing Fault
# Diagnosis under Different Working Conditions", Sensors 22(9):3282 (PMC8876626,
# doi:10.3390/s22093282) (no public code)
#
# 1D-LDSAN: a lightweight 1D-CNN feature extractor (inspired by MobileNetV2's inverted
# residual / linear bottleneck / depthwise-separable blocks, adapted to 1D raw vibration
# signals) feeding a shared FC classifier; trained with cross-entropy on the source domain
# plus a Local Maximum Mean Discrepancy (LMMD) subdomain-alignment loss between source and
# target domain features (Zhu et al. 2020 DSAN-style subdomain adaptation, applied to a 1D
# lightweight backbone rather than a 2D ResNet-50 image backbone). No code release exists
# for this paper (Data Availability Statement: "Not applicable"; no GitHub/supplementary
# code link); the paper's Table 1 ("Details of feature extraction module") gives an EXACT
# per-block spec (block type, kernel size, stride, output channels, output length at every
# stage), which is transcribed layer-for-layer below:
#
#   Block                    Layer        Kernel/stride         Output (len x ch)
#   Input                                                        1024 x 1
#   Regular Conv             ConvBNReLU6  k=4, s=4               256 x 6
#   Separable Block          ConvBNReLU6  k=3 (depthwise), s=1   256 x 6
#                             ConvBN       k=1 (pointwise), s=1   256 x 16
#   Inverted Bottleneck #1   ConvBNReLU6  k=1 (expand),   s=1     256 x 96
#                             ConvBNReLU6  k=3 (depthwise), s=2   128 x 96
#                             ConvBN       k=1 (project),  s=1    128 x 24
#   Inverted Bottleneck #2   ConvBNReLU6  k=1 (expand),   s=1     128 x 144
#                             ConvBNReLU6  k=3 (depthwise), s=2    64 x 144
#                             ConvBN       k=1 (project),  s=1     64 x 32
#   Separable Block          ConvBNReLU6  k=3 (depthwise), s=1     64 x 32
#                             ConvBN       k=1 (pointwise), s=1     64 x 48
#   Regular Conv             ConvBNReLU6  k=1, s=1                 64 x 64
#   Avg Pooling                                                     1 x 64
#
# The classification/adaptation module (Section 3.1.2) is described as a shared FC
# classifier "whose number of input neurons is the same as the number of extracted
# features" (i.e. 64) feeding the fault-class logits (10 classes for the CWRU bearing
# dataset, Table 2); the LMMD subdomain-alignment loss (Eq. 4-5) is a training-time
# objective computed on paired source/target batches and has no learnable parameters of
# its own, so it is not part of this module's forward architecture (this mirrors how the
# generic DSAN family's LMMD loss is applied externally to two forward passes' features in
# the reference DSAN implementations, e.g. jindongwang/transferlearning's DSAN, which target
# a different 2D ResNet-50 backbone and are not reused here since the backbone architecture
# in this paper is a distinct 1D lightweight design).
"""1D-LDSAN: 1D lightweight (MobileNetV2-style) CNN backbone + FC classifier for bearing
fault diagnosis, trained with a subdomain-adaptation (LMMD) transfer-learning objective."""

import torch
from torch import nn

MENAGERIE_ZOO = "reimpl-pytorch"


class ConvBNAct1d(nn.Module):
    """`ConvBNReLU6` / `ConvBN` block from Table 1 (Conv1d + BatchNorm1d + optional
    ReLU6)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, groups=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SeparableBlock1d(nn.Module):
    """Depthwise (k=3) + pointwise (k=1) separable convolution block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = ConvBNAct1d(
            in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch, act=True
        )
        self.pointwise = ConvBNAct1d(in_ch, out_ch, kernel_size=1, stride=1, act=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class InvertedBottleneck1d(nn.Module):
    """Expand (k=1) -> depthwise (k=3, stride) -> project (k=1) linear-bottleneck block."""

    def __init__(self, in_ch, expand_ch, out_ch, stride):
        super().__init__()
        self.expand = ConvBNAct1d(in_ch, expand_ch, kernel_size=1, stride=1, act=True)
        self.depthwise = ConvBNAct1d(
            expand_ch, expand_ch, kernel_size=3, stride=stride, groups=expand_ch, act=True
        )
        self.project = ConvBNAct1d(expand_ch, out_ch, kernel_size=1, stride=1, act=False)

    def forward(self, x):
        return self.project(self.depthwise(self.expand(x)))


class LDSAN1DFeatureExtractor(nn.Module):
    """Table 1's feature extraction module: 2 regular conv blocks + 4 unique
    (separable / inverted-bottleneck) blocks, ending in global average pooling."""

    def __init__(self):
        super().__init__()
        self.regular_conv1 = ConvBNAct1d(1, 6, kernel_size=4, stride=4, act=True)
        self.separable1 = SeparableBlock1d(6, 16, stride=1)
        self.inverted_bottleneck1 = InvertedBottleneck1d(16, 96, 24, stride=2)
        self.inverted_bottleneck2 = InvertedBottleneck1d(24, 144, 32, stride=2)
        self.separable2 = SeparableBlock1d(32, 48, stride=1)
        self.regular_conv2 = ConvBNAct1d(48, 64, kernel_size=1, stride=1, act=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.regular_conv1(x)
        x = self.separable1(x)
        x = self.inverted_bottleneck1(x)
        x = self.inverted_bottleneck2(x)
        x = self.separable2(x)
        x = self.regular_conv2(x)
        x = self.avg_pool(x)
        return x.flatten(1)  # [B, 64]


class OneDimLDSAN(nn.Module):
    """1D-LDSAN: shared feature extractor + FC classifier (the LMMD subdomain-adaptation
    loss is a training-time objective over paired source/target features, not part of the
    forward architecture -- see module header)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = LDSAN1DFeatureExtractor()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


def build_1d_ldsan():
    torch.manual_seed(0)
    return OneDimLDSAN(num_classes=10)


def example_input_1d_ldsan():
    torch.manual_seed(0)
    # real input size from the paper: 1024-sample raw vibration signal segment, 1 channel.
    return (torch.randn(2, 1, 1024),)


MENAGERIE_ENTRIES = [
    (
        "1D-LDSAN",
        "build_1d_ldsan",
        "example_input_1d_ldsan",
        2022,
        "reimpl",
    ),
]
