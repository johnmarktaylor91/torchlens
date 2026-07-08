# SOURCE: vendored from PRBonn/lidar-bonnetal @ 99b827f0228ff0e997473ac8e2cecbaa4af7d7c7
# (train/backbones/darknet.py, train/tasks/semantic/decoders/darknet.py,
#  train/tasks/semantic/modules/segmentator.py)
"""RangeNet++ (Milioto et al. 2019, IROS): range-image CNN for LiDAR
semantic segmentation.

The official repo projects LiDAR point clouds onto a spherical range image
and runs a Darknet-style encoder/decoder CNN (a YOLOv3-derived backbone,
adapted with configurable output-stride and skip connections) followed by a
lightweight segmentation head; a KNN post-processing step (not part of the
learned network) projects predictions back to 3D. The encoder (``Backbone``)
and decoder (``Decoder``) below are copied verbatim from
``train/backbones/darknet.py`` and
``train/tasks/semantic/decoders/darknet.py``, and ``Segmentator`` (the
top-level module tying backbone + decoder + segmentation head together) is
adapted from ``train/tasks/semantic/modules/segmentator.py``: the upstream
class dynamically imports the backbone/decoder modules via
``imp.load_source`` and optionally attaches a dense-CRF post-processing
layer and pretrained-weight loading -- none of that is architecture, so it
is replaced here with direct construction of the same ``Backbone``/
``Decoder``/head Sequential (CRF is off by default in every shipped config,
e.g. ``train/tasks/semantic/config/arch/darknet53.yaml`` sets
``post.CRF.use: False``, so it is omitted). ``forward`` is unchanged:
backbone -> decoder -> conv head -> softmax.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# train/backbones/darknet.py (encoder)
# --------------------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
    """Class for DarknetSeg. Subclasses PyTorch's own "nn" module"""

    def __init__(self, params):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.OS = params["OS"]
        self.layers = params["extra"]["layers"]

        # input depth calc
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)

        # stride play
        self.strides = [2, 2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s

        # make the new stride
        if self.OS > current_os:
            pass
        else:
            # redo strides according to needed stride
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break

        # check that darknet exists
        assert self.layers in model_blocks.keys()

        # generate layers depending on darknet type
        self.blocks = model_blocks[self.layers]

        # input layer
        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        # encoder
        self.enc1 = self._make_enc_layer(
            BasicBlock, [32, 64], self.blocks[0], stride=self.strides[0], bn_d=self.bn_d
        )
        self.enc2 = self._make_enc_layer(
            BasicBlock, [64, 128], self.blocks[1], stride=self.strides[1], bn_d=self.bn_d
        )
        self.enc3 = self._make_enc_layer(
            BasicBlock, [128, 256], self.blocks[2], stride=self.strides[2], bn_d=self.bn_d
        )
        self.enc4 = self._make_enc_layer(
            BasicBlock, [256, 512], self.blocks[3], stride=self.strides[3], bn_d=self.bn_d
        )
        self.enc5 = self._make_enc_layer(
            BasicBlock, [512, 1024], self.blocks[4], stride=self.strides[4], bn_d=self.bn_d
        )

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 1024

    # make layer useful function
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
        layers = []

        #  downsample
        layers.append(
            (
                "conv",
                nn.Conv2d(
                    planes[0],
                    planes[1],
                    kernel_size=3,
                    stride=[1, stride],
                    dilation=1,
                    padding=1,
                    bias=False,
                ),
            )
        )
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), block(inplanes, planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # store for skip connections
        skips = {}
        os = 1

        # first layer
        x, skips, os = self.run_layer(x, self.conv1, skips, os)
        x, skips, os = self.run_layer(x, self.bn1, skips, os)
        x, skips, os = self.run_layer(x, self.relu1, skips, os)

        # all encoder blocks with intermediate dropouts
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc5, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        return x, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth


# --------------------------------------------------------------------------
# train/tasks/semantic/decoders/darknet.py (decoder)
# --------------------------------------------------------------------------


class DecoderBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(DecoderBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class Decoder(nn.Module):
    """Class for DarknetSeg. Subclasses PyTorch's own "nn" module"""

    def __init__(self, params, stub_skips, OS=32, feature_depth=1024):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]

        # stride play
        self.strides = [2, 2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break

        # decoder
        self.dec5 = self._make_dec_layer(
            DecoderBasicBlock,
            [self.backbone_feature_depth, 512],
            bn_d=self.bn_d,
            stride=self.strides[0],
        )
        self.dec4 = self._make_dec_layer(
            DecoderBasicBlock, [512, 256], bn_d=self.bn_d, stride=self.strides[1]
        )
        self.dec3 = self._make_dec_layer(
            DecoderBasicBlock, [256, 128], bn_d=self.bn_d, stride=self.strides[2]
        )
        self.dec2 = self._make_dec_layer(
            DecoderBasicBlock, [128, 64], bn_d=self.bn_d, stride=self.strides[3]
        )
        self.dec1 = self._make_dec_layer(
            DecoderBasicBlock, [64, 32], bn_d=self.bn_d, stride=self.strides[4]
        )

        # layer list to execute with skips
        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 32

    def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2):
        layers = []

        #  downsample
        if stride == 2:
            layers.append(
                (
                    "upconv",
                    nn.ConvTranspose2d(
                        planes[0], planes[1], kernel_size=[1, 4], stride=[1, 2], padding=[0, 1]
                    ),
                )
            )
        else:
            layers.append(("conv", nn.Conv2d(planes[0], planes[1], kernel_size=3, padding=1)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        layers.append(("residual", block(planes[1], planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os].detach()  # add skip
        x = feats
        return x, skips, os

    def forward(self, x, skips):
        os = self.backbone_OS

        # run layers
        x, skips, os = self.run_layer(x, self.dec5, skips, os)
        x, skips, os = self.run_layer(x, self.dec4, skips, os)
        x, skips, os = self.run_layer(x, self.dec3, skips, os)
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        x, skips, os = self.run_layer(x, self.dec1, skips, os)

        x = self.dropout(x)

        return x

    def get_last_depth(self):
        return self.last_channels


# --------------------------------------------------------------------------
# train/tasks/semantic/modules/segmentator.py (top-level module), adapted:
# CRF branch + dynamic imp.load_source + checkpoint loading removed (not
# architecture; CRF is off in every shipped config).
# --------------------------------------------------------------------------


class Segmentator(nn.Module):
    def __init__(self, ARCH, nclasses):
        super().__init__()
        self.ARCH = ARCH
        self.nclasses = nclasses

        self.backbone = Backbone(params=self.ARCH["backbone"])

        # do a pass of the backbone to initialize the skip connections
        stub = torch.zeros(
            (
                1,
                self.backbone.get_input_depth(),
                self.ARCH["dataset"]["sensor"]["img_prop"]["height"],
                self.ARCH["dataset"]["sensor"]["img_prop"]["width"],
            )
        )
        _, stub_skips = self.backbone(stub)

        self.decoder = Decoder(
            params=self.ARCH["decoder"],
            stub_skips=stub_skips,
            OS=self.ARCH["backbone"]["OS"],
            feature_depth=self.backbone.get_last_depth(),
        )

        self.head = nn.Sequential(
            nn.Dropout2d(p=ARCH["head"]["dropout"]),
            nn.Conv2d(
                self.decoder.get_last_depth(), self.nclasses, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x, mask=None):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        y = F.softmax(y, dim=1)
        return y


def build_rangenetpp() -> nn.Module:
    """Build a tiny RangeNet++ ``Segmentator`` (Darknet21 encoder/decoder).

    Uses the smaller ``layers=21`` Darknet variant and a small range-image
    size so the module traces quickly; the architecture (Darknet encoder
    with configurable output-stride + skip-connected Darknet decoder +
    conv/softmax segmentation head) is unchanged from the real
    ``train/tasks/semantic/modules/segmentator.py``.

    Returns
    -------
    nn.Module
        Random-initialized RangeNet++ ``Segmentator`` (5-channel range image
        input: range + xyz + remission, 20-class KITTI-style output).
    """
    arch = {
        "backbone": {
            "input_depth": {"range": True, "xyz": True, "remission": True},
            "dropout": 0.01,
            "bn_d": 0.01,
            "OS": 32,
            "extra": {"layers": 21},
        },
        "decoder": {
            "dropout": 0.01,
            "bn_d": 0.01,
        },
        "head": {
            "dropout": 0.01,
        },
        "dataset": {
            "sensor": {
                "img_prop": {"height": 8, "width": 32},
            },
        },
    }
    return Segmentator(arch, nclasses=20)


def example_input_rangenetpp() -> torch.Tensor:
    """Return a small 5-channel range image matching ``build_rangenetpp``.

    Returns
    -------
    torch.Tensor
        ``(batch, channels, height, width)`` range-image tensor of shape
        ``(1, 5, 8, 32)`` (range, x, y, z, remission channels).
    """
    return torch.randn(1, 5, 8, 32)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RangeNet++", "build_rangenetpp", "example_input_rangenetpp", 2019, "rangenetpp"),
]
