# SOURCE: vendored from zoonono/RAKI-PyTorch @ master
# https://raw.githubusercontent.com/zoonono/RAKI-PyTorch/master/Network.py
#
# RAKI (Robust Artificial-neural-network for k-space Interpolation), PyTorch port of
# the original scan-specific CNN for MRI parallel-imaging k-space interpolation
# (Akcakaya et al., MRM 2019). The original repo wraps the network inside a
# `RAKINetwork` training-harness class whose `build_network()` constructs the actual
# `nn.Sequential` k-space CNN (real/imag-stacked channel-in -> 3 conv+ReLU layers ->
# R*2*channels-out acceleration-fold expansion). `SpatialNetwork.build_network()` is a
# second scan-specific variant that ends in `PixelShuffle` instead of a plain conv.
# Both `nn.Sequential` topologies (channel counts, kernel sizes, paddings,
# 'replicate' padding_mode) are transcribed verbatim from `Network.py`; only the
# non-architectural training/eval/IO harness (SummaryWriter, optimizer, checkpoint
# save/load, per-epoch training loop) is dropped since it is not part of the traced
# forward pass. Wrapped here as thin `nn.Module`s so the real `nn.Sequential` network
# is directly traceable.

import torch
import torch.nn as nn


class RAKINetworkModule(nn.Module):
    """The base RAKI k-space-interpolation CNN (`RAKINetwork.build_network`)."""

    def __init__(self, channels_in=8, acceleration_rate=2):
        super().__init__()
        self.channels_in = channels_in
        self.R = acceleration_rate
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.channels_in,
                out_channels=128,
                kernel_size=[3, 5],
                padding=[1, 2],
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=[1, 1],
                padding=[0, 0],
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=[1, 3],
                padding=[0, 1],
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=self.R * 2 * self.channels_in,
                kernel_size=[3, 3],
                padding=[1, 1],
                padding_mode="replicate",
            ),
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)


class SpatialNetworkModule(nn.Module):
    """The `SpatialNetwork.build_network` variant (PixelShuffle-based upsampling)."""

    def __init__(self, channels_in=8, acceleration_rate=2):
        super().__init__()
        self.channels_in = channels_in
        self.R = acceleration_rate
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.channels_in,
                out_channels=64,
                kernel_size=[3, 3],
                padding=[1, 1],
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=[3, 3],
                padding=[1, 1],
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=[3, 3],
                padding=[1, 1],
                padding_mode="replicate",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=2 * self.channels_in * self.R**2,
                kernel_size=[3, 3],
                padding=[1, 1],
                padding_mode="replicate",
            ),
            nn.PixelShuffle(self.R),
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)


def build_raki():
    torch.manual_seed(0)
    model = RAKINetworkModule(channels_in=8, acceleration_rate=2)
    model.eval()
    return model


def example_input_raki():
    torch.manual_seed(0)
    # Real/imag-stacked k-space patch: 2*channels_in=16 input channels over a small
    # kx x ky ACS-region-sized window.
    return torch.randn(1, 16, 16, 16)


def build_raki_spatial():
    torch.manual_seed(0)
    model = SpatialNetworkModule(channels_in=8, acceleration_rate=2)
    model.eval()
    return model


def example_input_raki_spatial():
    torch.manual_seed(0)
    return torch.randn(1, 16, 16, 16)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RAKI", "build_raki", "example_input_raki", 2019, MENAGERIE_ZOO),
    ("RAKI-Spatial", "build_raki_spatial", "example_input_raki_spatial", 2019, MENAGERIE_ZOO),
]
