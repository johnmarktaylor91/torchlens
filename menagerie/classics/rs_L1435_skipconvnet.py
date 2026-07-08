# SOURCE: vendored from zehuachenImperial/SkipConvNet @ b51da03677d98472ececfe939bb3c32f2c0140ba
# (model.py, classes Conv2d / ConvTranspose2d / SkipBlock / SkipConnection / SkipConvNet --
# "SkipConvNet: Skip Convolutional Neural Network for Speech Dereverberation using Optimally
# Smoothed Spectral Mapping", Interspeech 2020). The repo's README/queue notes an "AEC" alias
# but the paper and this architecture are for speech dereverberation, not acoustic-echo
# cancellation; the network itself is unchanged either way. 8-level encoder-decoder U-Net over
# log-magnitude spectrogram "images": each encoder stage is a strided Conv2d (with
# normal-std-0.02 weight init / zero bias init, as in the source) followed by BatchNorm; a
# "SkipConnection" (a stack of pre-activation SkipBlock residual conv layers) is applied to
# every encoder feature map before it is concatenated into the matching decoder stage, so the
# U-Net long-range skips are themselves small residual sub-networks rather than raw
# concatenation. The original class subclasses `pytorch_lightning.LightningModule` and defines
# `training_step`/`validation_step`/`test_step`/`configure_optimizers`/`*_dataloader` methods
# that construct a `SpecImages` `DataLoader` for the original spectrogram-image dataset; none
# of that touches the forward architecture, so it is dropped here and the class subclasses
# plain `nn.Module` instead. Every conv/skip/batchnorm layer and the exact forward control flow
# (including which decoder stage concatenates which encoder skip-path output) is unchanged.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class Conv2d(nn.Module):
    """
    Convolutional Module with weights initialized with normal distribution and weights to zeros
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=2):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
            bias=True,
        )
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2d(nn.Module):
    """
    Transpose Convolution Module with weights initialized with normal distribution and weights to zeros
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(ConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=True
        )
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class SkipBlock(nn.Module):
    """
    Each SkipBlock is a Activation -> Convolutions + Residual Connection followed by a normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(SkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.conv1.bias)
        self.norm = nn.BatchNorm2d(in_channels)
        self.lRelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.norm(self.conv1(self.lRelu(x)) + self.lRelu(x))


class SkipConnection(nn.Module):
    """
    SkipConnection is a concatenations of SkipBlocks
    """

    def __init__(self, in_channels, num_convblocks):
        super(SkipConnection, self).__init__()
        self.skip_blocks = [
            SkipBlock(in_channels, in_channels, kernel_size=3, padding=1)
            for k in range(num_convblocks)
        ]
        self.skip_path = nn.Sequential(*self.skip_blocks)

    def forward(self, x):
        return self.skip_path(x)


class SkipConvNet(nn.Module):
    """
    Proposed: SkipConvNet (Interspeech 2020)
    """

    def __init__(self):
        super(SkipConvNet, self).__init__()
        self.modelName = "SkipConvNet"

        self.dconv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.skip1 = SkipConnection(in_channels=64, num_convblocks=8)

        self.dconv2 = Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.dBNorm2 = nn.BatchNorm2d(128)
        self.skip2 = SkipConnection(in_channels=128, num_convblocks=8)

        self.dconv3 = Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.dBNorm3 = nn.BatchNorm2d(256)
        self.skip3 = SkipConnection(in_channels=256, num_convblocks=4)

        self.dconv4 = Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=2)
        self.dBNorm4 = nn.BatchNorm2d(512)
        self.skip4 = SkipConnection(in_channels=512, num_convblocks=4)

        self.dconv5 = Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.dBNorm5 = nn.BatchNorm2d(512)
        self.skip5 = SkipConnection(in_channels=512, num_convblocks=2)

        self.dconv6 = Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.dBNorm6 = nn.BatchNorm2d(512)
        self.skip6 = SkipConnection(in_channels=512, num_convblocks=2)

        self.dconv7 = Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.dBNorm7 = nn.BatchNorm2d(512)
        self.skip7 = SkipConnection(in_channels=512, num_convblocks=1)

        self.dconv8 = Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2)

        self.uconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.uBNorm1 = nn.BatchNorm2d(512)

        self.uconv2 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.uBNorm2 = nn.BatchNorm2d(512)

        self.uconv3 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.uBNorm3 = nn.BatchNorm2d(512)

        self.uconv4 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.uBNorm4 = nn.BatchNorm2d(512)

        self.uconv5 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=256, kernel_size=2, stride=2
        )
        self.uBNorm5 = nn.BatchNorm2d(256)

        self.uconv6 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=2, stride=2)
        self.uBNorm6 = nn.BatchNorm2d(128)

        self.uconv7 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2)
        self.uBNorm7 = nn.BatchNorm2d(64)

        self.uconv8 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=2, stride=2)

        self.lRelu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # +++++++++++++++++++ Squeezing Path  +++++++++++++++++++++ #
        d1 = self.dconv1(x)
        d2 = self.dBNorm2(self.dconv2(self.lRelu(d1)))
        d3 = self.dBNorm3(self.dconv3(self.lRelu(d2)))
        d4 = self.dBNorm4(self.dconv4(self.lRelu(d3)))
        d5 = self.dBNorm5(self.dconv5(self.lRelu(d4)))
        d6 = self.dBNorm6(self.dconv6(self.lRelu(d5)))
        d7 = self.dBNorm7(self.dconv7(self.lRelu(d6)))
        d8 = self.dconv8(self.lRelu(d7))

        # +++++++++++++++++++ Expanding Path  +++++++++++++++++++++ #
        u1 = self.drop(self.uBNorm1(self.uconv1(self.relu(d8))))
        u2 = self.drop(self.uBNorm2(self.uconv2(self.relu(torch.cat((u1, self.skip7(d7)), 1)))))
        u3 = self.drop(self.uBNorm3(self.uconv3(self.relu(torch.cat((u2, self.skip6(d6)), 1)))))
        u4 = self.uBNorm4(self.uconv4(self.relu(torch.cat((u3, self.skip5(d5)), 1))))
        u5 = self.uBNorm5(self.uconv5(self.relu(torch.cat((u4, self.skip4(d4)), 1))))
        u6 = self.uBNorm6(self.uconv6(self.relu(torch.cat((u5, self.skip3(d3)), 1))))
        u7 = self.uBNorm7(self.uconv7(self.relu(torch.cat((u6, self.skip2(d2)), 1))))
        u8 = self.uconv8(self.relu(torch.cat((u7, self.skip1(d1)), 1)))
        Output = self.tanh(u8)
        return Output


def build_skipconvnet():
    model = SkipConvNet()
    model.eval()
    return model


def example_input_skipconvnet():
    # Original demo/summary uses (1, 256, 256) spectrogram "images"; 8 strided
    # (stride-2) conv stages need spatial dims divisible by 2**8=256.
    return torch.randn(1, 1, 256, 256)


MENAGERIE_ENTRIES = [
    ("SkipConvNet", "build_skipconvnet", "example_input_skipconvnet", 2020, MENAGERIE_ZOO),
]
