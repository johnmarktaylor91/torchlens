# SOURCE: vendored from deepconsc/SplitSR @ master
# https://raw.githubusercontent.com/deepconsc/SplitSR/master/model/splitsr.py
# https://raw.githubusercontent.com/deepconsc/SplitSR/master/modules/blocks.py
#
# Liu, Chu, Chen, Shen, Liu 2021 "SplitSR: An End-to-End Approach to Super-Resolution on
# Mobile Devices" (ACM IMWUT/UbiComp 2021). Split-computation super-resolution CNN: a
# SplitSRBlock partial-channel convolution (only an alpha-fraction of channels pass
# through a conv+BN+ReLU each block, the rest are passed through unchanged) alternates
# with full ResidualBlocks, followed by a pixel-shuffle upsampler. `SplitSR`,
# `SplitSRBlock`, `ResidualBlock`, `Upsample`, and `MeanShift` are copied verbatim from
# model/splitsr.py + modules/blocks.py (only the `__main__` demo blocks, which contain a
# syntax error `&&` instead of `and` in the upstream repo, are omitted).
"""SplitSR: split-computation super-resolution CNN for mobile devices (4x upsampling)."""

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from modules/blocks.py ---
class SplitSRBlock(nn.Module):
    def __init__(self, channels, kernel, alpha):
        super(SplitSRBlock, self).__init__()
        self.alpharatio = int(channels * alpha)
        self.channels = channels
        self.conv = nn.Conv2d(
            in_channels=self.alpharatio,
            out_channels=self.alpharatio,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
            bias=True,
        )
        self.batchnorm = nn.BatchNorm2d(self.alpharatio)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        active, passive = x[:, : self.alpharatio], x[:, self.alpharatio :]
        active = self.conv(active)  # In: (1, 64 * a, W, H) | Out: (1, 64 * a, W, H)
        active = self.batchnorm(active)
        active = self.relu(active)
        x = torch.cat([passive, active], dim=1)  # Out: (1, 64, W, H)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels * 4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=True,
        )
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)  # In: (1, 64, W, H) | Out: (1, 256, W, H)
        x = self.pixelshuffle(x)  # In: (1, 256, W*2, H*2) | Out: (1, 64, W*2, H*2)

        x = self.conv(x)  # In: (1, 64, W*2, H*2) | Out: (1, 256, W*2, H*2)
        x = self.pixelshuffle(x)  # In: (1, 256, W*4, H*4) | Out: (1, 64, W*4, H*4)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=True,
        )
        self.batchnorm = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.batchnorm(x)
        x += residual
        x = self.relu(x)

        return x


class MeanShift(nn.Conv2d):
    def __init__(self, coeff):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor([1.0, 1.0, 1.0])
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = coeff * 255 * torch.Tensor([0.4488, 0.4371, 0.4040])
        self.bias.data.div_(std)
        self.requires_grad = False


# --- vendored from model/splitsr.py ---
class SplitSR(nn.Module):
    def __init__(self):
        super(SplitSR, self).__init__()

        self.ResidualGroup = nn.Sequential(
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.250),
            SplitSRBlock(channels=64, kernel=3, alpha=0.250),
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.250),
            SplitSRBlock(channels=64, kernel=3, alpha=0.250),
        )
        self.conv_head = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True
        )
        self.conv_back = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2, bias=True
        )
        self.upsample = Upsample(64)
        self.MeanSubstract = MeanShift(-1)
        self.MeanAdd = MeanShift(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.MeanSubstract(x)
        x = self.conv_head(x)

        x = self.ResidualGroup(x)
        x = self.ResidualGroup(x)
        x = self.ResidualGroup(x)
        x = self.ResidualGroup(x)
        x = self.ResidualGroup(x)
        x = self.ResidualGroup(x)

        x = self.upsample(x)
        x = self.conv_back(x)
        x = self.MeanAdd(x)

        return x


def build_splitsr():
    # real architecture from model/splitsr.py: fixed 64-channel ResidualGroup (2x
    # ResidualBlock + 4x SplitSRBlock alpha=0.25) run 6 times, then a 4x pixel-shuffle
    # upsampler. No size hyperparameters to shrink -- this is the paper's actual config.
    model = SplitSR()
    model.eval()
    return model


def example_input_splitsr():
    torch.manual_seed(0)
    # real repo's own __main__ demo uses a 96x96 LR patch.
    return (torch.randn(1, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    ("SplitSR", "build_splitsr", "example_input_splitsr", 2021, "vendored"),
]
