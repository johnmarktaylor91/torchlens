# SOURCE: vendored from omegahh/DeepHiC @ a53135a131f0cb628cd3aae2c9edb36b7d6964e2
# https://github.com/omegahh/DeepHiC/blob/a53135a131f0cb628cd3aae2c9edb36b7d6964e2/models/deephic.py
#
# DeepHiC: a generative adversarial network for enhancing Hi-C data resolution
# (Hong et al., PLOS Computational Biology 2020). This file vendors the real
# Generator/Discriminator nn.Module classes from models/deephic.py verbatim
# (only import path is unchanged -- no architectural edits).

import torch
import torch.nn as nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class residualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # a swish layer here
        self.conv2 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = swish(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual


class Generator(nn.Module):
    def __init__(self, scale_factor, in_channel=3, resblock_num=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=9, stride=1, padding=4)
        # have a swish here in forward

        resblocks = [residualBlock(64) for _ in range(resblock_num)]
        self.resblocks = nn.Sequential(*resblocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # have a swish here in forward

        self.conv3 = nn.Conv2d(64, in_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        emb = swish(self.conv1(x))
        x = self.resblocks(emb)
        x = swish(self.bn2(self.conv2(x)))
        x = self.conv3(x + emb)
        return (torch.tanh(x) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        # Replaced original paper FC layers with FCN
        self.conv7 = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.size(0)

        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))

        x = self.conv7(x)
        x = self.avgpool(x)
        return torch.sigmoid(x.view(batch_size))


MENAGERIE_ZOO = "vendored-pytorch"


def build_deephic_generator():
    # chunk=40 (default training resolution in DeepHiC's train.py), 1-channel
    # Hi-C contact-map patches, resblock_num=5 (paper default).
    return Generator(scale_factor=1, in_channel=1, resblock_num=5)


def example_input_deephic_generator():
    return torch.randn(1, 1, 40, 40)


def build_deephic_discriminator():
    return Discriminator(in_channel=1)


def example_input_deephic_discriminator():
    return torch.randn(1, 1, 40, 40)


MENAGERIE_ENTRIES = [
    (
        "DeepHiC-Generator",
        "build_deephic_generator",
        "example_input_deephic_generator",
        2020,
        MENAGERIE_ZOO,
    ),
    (
        "DeepHiC-Discriminator",
        "build_deephic_discriminator",
        "example_input_deephic_discriminator",
        2020,
        MENAGERIE_ZOO,
    ),
]
