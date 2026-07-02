# SOURCE: vendored from https://github.com/Intelligent-Sensing/D-VDAMP @ master (0405a68d)
# (train/model.py :: Colored_DnCNN)
#
# D-VDAMP: Denoising-Based Approximate Message Passing for Compressive MRI
# (Metzler & Wetzstein, ICASSP 2021). https://github.com/Intelligent-Sensing/D-VDAMP
#
# The D-VDAMP "algorithm" itself (algorithm/dvdamp.py :: dvdamp()) is a plain
# numpy/scipy iterative signal-processing routine (wavelet-domain VDAMP with a
# denoiser subroutine) -- not an nn.Module. The one trainable neural network in
# the repo is Colored_DnCNN, the denoiser plugged into that loop via
# algorithm/denoiser.py's DnCNN_denoiser wrapper. Colored_DnCNN is DnCNN with a
# per-subband noise-standard-deviation map concatenated onto the input of every
# convolution layer (the "colored" noise conditioning described in the paper).
# The class below is copied verbatim from train/model.py; only the import block
# was trimmed (torch/nn/functional only, no other repo modules needed).

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class Colored_DnCNN(nn.Module):
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


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing.
#
# The real training script (train/main.py) constructs Colored_DnCNN with the repo
# defaults (channels=1, num_layers=20, std_channels=13 -- one std map per wavelet
# subband up to 4 decomposition levels) and feeds it single-channel image patches
# plus the matching per-subband std vector. We keep the repo defaults and a small
# 33x33 spatial size (the model is fully convolutional, so any HxW works).
# ---------------------------------------------------------------------------
_CHANNELS = 1
_NUM_LAYERS = 20
_STD_CHANNELS = 13
_H = 33
_W = 33
_BATCH = 2


def build_dvdamp_denoiser():
    torch.manual_seed(0)
    model = Colored_DnCNN(channels=_CHANNELS, num_layers=_NUM_LAYERS, std_channels=_STD_CHANNELS)
    model.eval()
    return model


def example_input_dvdamp_denoiser():
    torch.manual_seed(0)
    x = torch.randn(_BATCH, _CHANNELS, _H, _W)
    std = torch.rand(_BATCH, _STD_CHANNELS)
    return (x, std)


MENAGERIE_ENTRIES = [
    (
        "D-VDAMP-ColoredDnCNN",
        "build_dvdamp_denoiser",
        "example_input_dvdamp_denoiser",
        2021,
        MENAGERIE_ZOO,
    ),
]
