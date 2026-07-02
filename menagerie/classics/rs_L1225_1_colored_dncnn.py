# SOURCE: vendored from Intelligent-Sensing/D-VDAMP @ master
# https://github.com/Intelligent-Sensing/D-VDAMP/blob/master/train/model.py
#
# Metzler & Wetzstein, "D-VDAMP: Denoising-Based Approximate Message Passing for
# Compressive MRI," ICASSP 2021. `Colored_DnCNN` is the paper's colored-noise
# denoiser CNN: a DnCNN-style stack of 3x3 conv layers (first: conv+ReLU; middle
# num_layers-2: conv+BatchNorm+ReLU; last: conv), except that at EVERY layer the
# per-wavelet-subband noise standard-deviation vector `std` is broadcast to
# spatial maps and concatenated onto the running feature map before the conv
# (this is what makes it "colored" rather than plain white-noise DnCNN). The
# network predicts the residual noise and returns `x - noise`. Transcribed
# verbatim from `train/model.py`; the identical architecture (module-for-module)
# also appears as `ColoredDnCNN` in `algorithm/denoiser.py` with a per-sample
# (unbatched) forward signature -- this file keeps the batched `train/model.py`
# version. Only edit: none to the architecture; added build_/example_input_
# staging helpers below.

import torch
from torch import nn
from torch.nn import functional as F


class Colored_DnCNN(nn.Module):
    """DnCNN but with 1xHxW tensors of estimated standard deviation concatenated
    to the input before every convolution layer."""

    def __init__(self, channels=1, num_layers=20, std_channels=13):
        super(Colored_DnCNN, self).__init__()

        self.num_layers = num_layers

        # Fixed parameters
        kernel_size = 3
        padding = 1
        features = 64

        conv_layers = []
        bn_layers = []

        self.first_conv = nn.Conv2d(
            in_channels=channels + std_channels,
            out_channels=features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        for _ in range(num_layers - 2):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=features + std_channels,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            bn_layers.append(nn.BatchNorm2d(features))
        self.last_conv = nn.Conv2d(
            in_channels=features + std_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self._initialize_weights()

    def forward(self, x, std):
        _, _, H, W = x.shape
        std_channels = self._generate_std_channels(std, H, W)
        noise = torch.cat((x, std_channels), dim=1)
        noise = F.relu(self.first_conv(noise))
        for i in range(self.num_layers - 2):
            noise = torch.cat((noise, std_channels), dim=1)
            noise = F.relu(self.bn_layers[i](self.conv_layers[i](noise)))
        noise = torch.cat((noise, std_channels), dim=1)
        noise = self.last_conv(noise)
        out = x - noise
        return out

    def _generate_std_channels(self, std, H, W):
        N, concat_channels = std.shape
        std_channels = std.reshape(N, concat_channels, 1, 1).repeat(1, 1, H, W)
        return std_channels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def build_colored_dncnn():
    """Tiny-config Colored DnCNN: 6 conv layers (vs paper default 20), 13-band
    wavelet-subband std vector (Haar level-4 decomposition, matching the
    paper's trained checkpoints)."""
    return Colored_DnCNN(channels=1, num_layers=6, std_channels=13)


def example_input_colored_dncnn():
    batch_size = 2
    x = torch.randn(batch_size, 1, 32, 32)
    std = torch.rand(batch_size, 13) * 0.1
    return (x, std)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("D-VDAMP Colored DnCNN", "build_colored_dncnn", "example_input_colored_dncnn", 2021, "MRI"),
]
