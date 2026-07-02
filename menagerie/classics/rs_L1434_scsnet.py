# SOURCE: vendored from stephenllh/bcs-unet @ be534a25e28cbe3501278d0ee6e2417b2cd737d3
# https://raw.githubusercontent.com/stephenllh/bcs-unet/be534a25e28cbe3501278d0ee6e2417b2cd737d3/src/benchmark/scsnet/net.py
# https://raw.githubusercontent.com/stephenllh/bcs-unet/be534a25e28cbe3501278d0ee6e2417b2cd737d3/src/benchmark/scsnet/learner.py
#
# SCSNet (Shi, Song, Zhu, "Image Compressed Sensing Using Convolutional Neural
# Network", CVPR 2019 / IEEE TIP 2019, wzhshi/SCSNet) -- block-based compressive
# sensing (CS) with a learned linear sampling operator, an "initial reconstruction"
# subnetwork, and a "deep reconstruction" subnetwork. The official repo
# (wzhshi/SCSNet) ships only MatConvNet (.m) code and pretrained .mat weights --
# MATLAB, not runnable in a PyTorch env. `stephenllh/bcs-unet` (a benchmark suite
# comparing block-CS reconstruction networks, including SCSNet, ReconNet, and the
# author's own BCS-UNet) contains a real, independent PyTorch reimplementation of
# SCSNet's two subnetworks matching the CVPR2019 architecture: `SCSNetInit`
# implements the sampling + initial-reconstruction stage as a single learned 1x1
# Conv2d "sampling matrix" (in_channels = round(sampling_ratio * block_size**2))
# whose block_size**2 output channels are folded back into full-resolution pixel
# blocks (the paper's block-wise linear measurement + initial reconstruction
# fused into one convolutional layer); `SCSNetDeep` is the non-linear deep
# reconstruction network -- a 16-layer Conv2d+BatchNorm+ReLU stack (128 -> 32
# channels, 13 middle 32-channel blocks, 32 -> 128, 128 -> 1) with a residual
# (sigmoid(x + out)) connection around the whole stack, matching the paper's
# deep-reconstruction residual-refinement design. `SCSNetLearner.forward` in
# learner.py establishes the exact `net2(net1(x))` cascade used here.
#
# Transcribed verbatim from net.py -- every Conv2d/BatchNorm2d/ReLU layer and its
# arguments, the block-permute reshape in `SCSNetInit._permute`, and the residual
# sigmoid in `SCSNetDeep.forward` are unchanged. Only the pytorch_lightning
# `SCSNetLearner` training wrapper (loss/optimizer/metric plumbing) is dropped
# since it is not part of the traced architecture; its `forward` composition
# (`net2(net1(inputs))`) is preserved verbatim in `SCSNet.forward` below.

import torch
import torch.nn as nn


class SCSNetInit(nn.Module):
    """The "initial reconstruction network" of SCSNet"""

    def __init__(self, in_channels, block_size=4):
        super().__init__()
        self.block_size = block_size
        self.conv = nn.Conv2d(in_channels, block_size**2, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        out = self._permute(x)
        return out

    def _permute(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H, W, self.block_size, self.block_size)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        out = x.view(-1, 1, H * self.block_size, W * self.block_size)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class SCSNetDeep(nn.Module):
    """The "deep reconstruction network" of SCSNet"""

    def __init__(self):
        super().__init__()
        middle_convs = [
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3) for _ in range(13)
        ]
        self.convs = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=128, kernel_size=3),
            ConvBlock(in_channels=128, out_channels=32, kernel_size=3),
            *middle_convs,
            ConvBlock(in_channels=32, out_channels=128, kernel_size=3),
            nn.Conv2d(
                in_channels=128,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        out = self.convs(x)
        return torch.sigmoid(x + out)


class SCSNet(nn.Module):
    """Cascade of the sampling/initial-reconstruction and deep-reconstruction
    subnetworks, matching `SCSNetLearner.forward` (`net2(net1(inputs))`) in the
    source repo's learner.py."""

    def __init__(self, sampling_ratio, block_size=4):
        super().__init__()
        in_channels = int(sampling_ratio * block_size**2)
        self.net1 = SCSNetInit(in_channels, block_size=block_size)
        self.net2 = SCSNetDeep()

    def forward(self, x):
        return self.net2(self.net1(x))


def build_scsnet():
    torch.manual_seed(0)
    # sampling_ratio=0.25, block_size=4 -> in_channels=4 (matches the repo's
    # `int(config["sampling_ratio"] * 16)` convention with block_size=4).
    model = SCSNet(sampling_ratio=0.25, block_size=4)
    model.eval()
    return model


def example_input_scsnet():
    torch.manual_seed(0)
    # Small block-CS measurement tensor: (batch, in_channels=4, H, W).
    return torch.randn(1, 4, 8, 8)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SCSNet", "build_scsnet", "example_input_scsnet", 2019, MENAGERIE_ZOO),
]
