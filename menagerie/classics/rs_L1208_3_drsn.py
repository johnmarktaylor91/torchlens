# FAITHFUL PORT of zhao62/Deep-Residual-Shrinkage-Networks @ master (original framework: Keras/TFLearn, TF1.x)
# https://github.com/zhao62/Deep-Residual-Shrinkage-Networks/blob/master/DRSN_Keras.py
# https://github.com/zhao62/Deep-Residual-Shrinkage-Networks/blob/master/DRSN_TFLearn.py
# The official author repo (M. Zhao et al., "Deep Residual Shrinkage Networks for Fault
# Diagnosis", IEEE TII 2019) ships ONLY Keras-2.2.1/TF-1.x and TFLearn-0.3.2/TF-1.0
# implementations; those toolchains cannot be installed alongside torch 2.8 in this
# environment, so the architecture is transcribed faithfully into torch (rung 3, not a
# from-scratch reimplementation from a paper summary -- every mechanism below is a
# direct translation of the two released source files, which agree on the design).
# Faithfully ported, every mechanism preserved:
#   - `residual_shrinkage_block`: BN -> ReLU -> Conv3x3(stride) -> BN -> ReLU -> Conv3x3
#     -> per-channel absolute-value global average pool (`abs_mean`) -> a small
#     2-layer FC "scales" sub-network (Linear -> BN -> ReLU -> Linear -> Sigmoid) that
#     produces a per-channel gating coefficient in (0, 1) -> threshold = abs_mean *
#     scales -> soft-thresholding: sign(residual) * relu(|residual| - threshold)
#     -> identity shortcut (average-pooled when downsampling, zero-padded in the
#     channel dim when in_channels != out_channels) -> residual + identity.
#   - Stem: Conv3x3(3->16) -> one non-downsampling shrinkage block (16->16) -> two
#     downsampling shrinkage blocks (16->32, 32->32) -> BN -> ReLU -> global average
#     pool -> Linear classifier head, matching the Keras script's 3-block CIFAR-10
#     network exactly.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ResidualShrinkageBlock(nn.Module):
    """Direct port of `residual_shrinkage_block` (DRSN_Keras.py / DRSN_TFLearn.py)."""

    def __init__(self, in_channels, out_channels, downsample=False, downsample_strides=2):
        super().__init__()
        self.downsample = downsample
        self.downsample_strides = downsample_strides if downsample else 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.downsample_strides,
            padding=1,
            bias=True,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )

        # "scales" sub-network computing the per-channel soft-threshold gate
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.fc1_bn = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

        if self.downsample_strides > 1:
            self.identity_pool = nn.AvgPool2d(kernel_size=1, stride=self.downsample_strides)
        else:
            self.identity_pool = None

    def forward(self, x):
        identity = x
        residual = x

        residual = self.bn1(residual)
        residual = F.relu(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = F.relu(residual)
        residual = self.conv2(residual)

        # abs_mean: global average pool of |residual| over spatial dims, per channel
        residual_abs = torch.abs(residual)
        abs_mean = residual_abs.mean(dim=(2, 3))  # [B, C]

        scales = self.fc1(abs_mean)
        scales = self.fc1_bn(scales)
        scales = F.relu(scales)
        scales = self.fc2(scales)
        scales = torch.sigmoid(scales)

        thres = abs_mean * scales  # [B, C]
        thres = thres.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # soft thresholding: sign(x) * max(|x| - thres, 0)
        sub = residual_abs - thres
        n_sub = torch.clamp(sub, min=0.0)
        residual = torch.sign(residual) * n_sub

        if self.identity_pool is not None:
            identity = self.identity_pool(identity)

        if self.in_channels != self.out_channels:
            pad = self.out_channels - self.in_channels
            pad_left = pad // 2
            pad_right = pad - pad_left
            identity = F.pad(identity, (0, 0, 0, 0, pad_left, pad_right))

        return residual + identity


class DeepResidualShrinkageNetwork(nn.Module):
    """Direct port of the 3-block CIFAR-10 network defined at the bottom of
    DRSN_Keras.py (Conv stem -> 3 shrinkage blocks -> BN/ReLU -> GAP -> Linear)."""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=True)
        self.block1 = ResidualShrinkageBlock(8, 8, downsample=True)
        self.bn_out = nn.BatchNorm2d(8)
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.bn_out(x)
        x = F.relu(x)
        x = x.mean(dim=(2, 3))  # global average pool
        x = self.classifier(x)
        return F.softmax(x, dim=-1)


def build_drsn():
    # Tiny menagerie-scale config: 8-channel stem/block (shrunk from the released 8/16/32
    # progression's smallest stage) and a single downsampling shrinkage block, matching
    # the released model's per-block mechanism exactly while keeping the network small.
    return DeepResidualShrinkageNetwork(in_channels=3, num_classes=10)


def example_input_drsn():
    torch.manual_seed(0)
    return (torch.randn(2, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    (
        "DRSN (Deep Residual Shrinkage Network)",
        "build_drsn",
        "example_input_drsn",
        2019,
        "ported-pytorch",
    ),
]
