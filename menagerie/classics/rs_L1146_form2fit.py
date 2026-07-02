# SOURCE: vendored from https://github.com/kevinzakka/form2fit @ 099a4ceac0
# Form2Fit: "Form2Fit: Learning Shape Priors for Generalizable Assembly from
# Disassembly" (Zakka et al., ICRA 2020).
#
# Vendored real repo code: form2fit/code/ml/models/{resnet.py,fcn.py,
# suction.py,placement.py,base.py}, combined into one file with only import
# paths flattened (the originals use `from form2fit.code.ml.models import
# resnet` package-relative imports that don't resolve outside the repo tree).
# No layer or dataflow was changed.
#
# NOTE: the repo's third head, CorrespondenceNet, additionally depends on
# `cv2.warpAffine` mid-forward-pass and the non-pip-installable `walle`
# robotics-utils package (kevinzakka's own internal library, not on PyPI) for
# `walle.core.RotationMatrix` -- not vendorable without those non-base deps.
# SuctionNet and PlacementNet share the identical FCNet backbone (the same
# dense encoder-decoder architecture) and require only base torch/torchvision,
# so they are vendored here as the representative Form2Fit architecture.

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class BaseModel(torch.nn.Module):
    """An abstract base class for a deep neural network."""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters())


def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    import numpy as np

    kernel_size = np.asarray((3, 3))

    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    full_padding = (upsampled_kernel_size - 1) // 2
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)


class FCNet(BaseModel):
    """A fully-convolutional network with an encoder-decoder architecture."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    ),
                    ("enc-norm0", nn.BatchNorm2d(64)),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    ("enc-pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn2",
                        BasicBlock(
                            64,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                nn.BatchNorm2d(128),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc-pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn4",
                        BasicBlock(
                            128,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                nn.BatchNorm2d(256),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "enc-resn5",
                        BasicBlock(
                            256,
                            512,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                                nn.BatchNorm2d(512),
                            ),
                            dilation=1,
                        ),
                    ),
                ]
            )
        )

        self._decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-resn0",
                        BasicBlock(
                            512,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                nn.BatchNorm2d(256),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-resn1",
                        BasicBlock(
                            256,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                nn.BatchNorm2d(128),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("dec-upsm2", Interpolate(scale_factor=2, mode="bilinear", align_corners=True)),
                    (
                        "dec-resn3",
                        BasicBlock(
                            128,
                            64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                nn.BatchNorm2d(64),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("dec-upsm4", Interpolate(scale_factor=2, mode="bilinear", align_corners=True)),
                    ("dec-conv5", nn.Conv2d(64, out_channels, kernel_size=1, stride=1, bias=True)),
                ]
            )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_enc = self._encoder(x)
        out_dec = self._decoder(out_enc)
        return out_dec


class SuctionNet(BaseModel):
    """The suction-point prediction network (dense heightmap -> per-pixel
    graspability heatmap)."""

    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels
        self._fcn = FCNet(num_channels, 1)

    def forward(self, x):
        return self._fcn(x)


def build_form2fit_suctionnet():
    # Real repo default: 4-channel input (RGB-D heightmap, color + depth).
    return SuctionNet(num_channels=4)


def example_input_form2fit_suctionnet():
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Form2Fit",
        build_form2fit_suctionnet,
        example_input_form2fit_suctionnet,
        2020,
        MENAGERIE_ZOO,
    ),
]
