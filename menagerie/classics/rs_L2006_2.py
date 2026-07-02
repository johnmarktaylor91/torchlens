# SOURCE: vendored from https://github.com/opendr-eu/opendr @ master
# (src/opendr/perception/speech_recognition/edgespeechnets/algorithm/models.py)
#
# EdgeSpeechNets (Lin, Chwyl & Wong, 2018/arXiv:1810.08559) are compact
# residual CNNs for on-device keyword-spotting. The reference implementation
# named in the queue (AmirLavasani/EdgeSpeechNets) has no code (README only);
# the OpenDR European Project toolkit vendors the actual working PyTorch
# implementation used by its `EdgeSpeechNetLearner`, which is what is copied
# below verbatim (only the relative "from .xxx" import is inlined since this
# is now a single-file module; no other logic changed).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class EdgeSpeechNet(nn.Module):
    def __init__(self, target_classes_n):
        super().__init__()
        self.encoder = None
        self._make_encoder()
        self.decoder = nn.Linear(in_features=45, out_features=target_classes_n)

    def _make_encoder(self):
        layers = []
        for entry in self.__class__.architecture:
            layer, kwargs = entry
            layers.append(layer(**kwargs))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.squeeze(2).squeeze(2)
        x = self.decoder(x)
        return F.log_softmax(x, dim=1)


class ESNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode="zeros"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            bias=False,
            padding=1,
            padding_mode=padding_mode,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class ESNResBlock(nn.Module):
    def __init__(self, in_out_channels, mid_channels, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.prebn = nn.BatchNorm2d(in_out_channels)
        self.conv1 = ESNConv2d(in_channels=in_out_channels, out_channels=mid_channels)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.conv2 = ESNConv2d(in_channels=mid_channels, out_channels=in_out_channels)

    def forward(self, x):
        residual_input = x
        if not self.first:
            x = self.prebn(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x + residual_input
        return x


class EdgeSpeechNetA(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 39}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 20, "first": True}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 15}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 25}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 22}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 22}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 25}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 45}),
    ]


class EdgeSpeechNetB(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 30}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 8, "first": True}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 9}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 11}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 10}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 8}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 11}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 45}),
    ]


class EdgeSpeechNetC(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 24}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6, "first": True}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 9}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 12}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 5}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 2}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 45}),
    ]


class EdgeSpeechNetD(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 45}),
        (nn.AvgPool2d, {"kernel_size": 2}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 30, "first": True}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 33}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 35}),
    ]


# Real construction: MFCC-spectrogram-shaped input, 1 input channel, a small
# number of target keyword classes (OpenDR's default speech-commands setup
# uses 20-35 classes; 12 keeps the trace tiny while still real).
def build_edgespeechnet_a():
    return EdgeSpeechNetA(target_classes_n=12)


def build_edgespeechnet_d():
    return EdgeSpeechNetD(target_classes_n=12)


def example_edgespeechnet_input():
    # (batch, channels=1, n_mfcc, time) - matches get_mfcc()'s (n_mfcc, length)
    # cepstrogram before the batch/channel dims are added for the encoder.
    return torch.randn(1, 1, 24, 51)


MENAGERIE_ENTRIES = [
    ("EdgeSpeechNetA", "build_edgespeechnet_a", "example_edgespeechnet_input", "2018", "AUD"),
    ("EdgeSpeechNetD", "build_edgespeechnet_d", "example_edgespeechnet_input", "2018", "AUD"),
]
