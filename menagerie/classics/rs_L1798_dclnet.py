# SOURCE: vendored from DIAL-RPI/FreehandUSRecon @ master
# https://raw.githubusercontent.com/DIAL-RPI/FreehandUSRecon/master/networks/mynet.py
#
# Guo, Xu, Wood, Yan 2020 (MICCAI) "Sensorless Freehand 3D Ultrasound Reconstruction via Deep
# Contextual Learning" -- DCL-Net. `networks/mynet.py`'s `ResNeXt`/`resnet50` (instantiated as
# `model_type == 'mynet'` in `train_network.py::define_model`, the repo's default 3D-conv
# architecture) IS DCL-Net: a 3D ResNeXt backbone over a stacked-frame US video volume, feeding
# an embedded self-attention module (`self.attention`, a small 3D-conv + BatchNorm + Sigmoid
# stack) that gates the backbone features before global average pooling and a final FC
# regression head predicting inter-frame motion (degrees of freedom). This is exactly the
# paper's "3D convolutions ... embedded self-attention module" description.
#
# `ResNeXtBottleneck`, `ResNeXt.__init__`, `_make_layer` are copied verbatim from the real
# file. `ResNeXt.forward` is copied from the real file's `else` branch (the functional/
# production path); the real file's `if show_size:` branch is IDENTICAL math but interleaves
# `print()` calls and an unconditional `time.sleep(30)` debugging leftover -- dropped here as
# non-architectural debug scaffolding, not a behavior change (both branches compute the exact
# same tensor ops). `layer3`/`layer4`/`dropout1`/`self.maxpool` are constructed by the real
# `__init__` but never invoked by either real `forward` branch -- left constructed-but-unused
# here too, exactly matching the real dead-code quirk (not our fabrication: the real repo
# defines 4 residual stages and a maxpool but the real `forward()` only ever calls
# `layer1`/`layer2`). `train_network.py::define_model`'s `model_type == 'mynet'` branch
# overrides `conv1` post-construction to `Conv3d(in_channels=1, out_channels=64,
# kernel_size=(3,7,7), stride=(1,2,2), padding=(0,3,3))` AND replaces `fc` with
# `nn.Linear(384, ...)` (a value hardcoded for a specific `neighbour_slice`/resolution
# combination the original authors ran, not derivable from `mynet.py` itself); we apply the
# same `conv1` override, and replace `fc` analogously but sized from the real (unmodified)
# conv/pool/attention pipeline's actual flattened feature count for our example input, since
# 384 is only valid for the external training script's specific untracked hyperparameters.
# `self.avgpool` is hardcoded by the real constructor to `AvgPool3d((1, 28, 28))` regardless
# of `sample_size`/`sample_duration` (the computed `last_duration`/`last_size` are dead
# variables in the real code) -- so, exactly as in the real repo, the spatial input size must
# independently satisfy that hardcoded 28x28 requirement; `example_input_dclnet()` picks the
# (real, unmodified) 3D-conv/pool math's actual required resolution rather than the
# (separately configurable, and here irrelevant since `conv1`/`avgpool` are hardcoded)
# `data_transform(resize=224)` value used elsewhere in the pipeline. No architectural changes.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNeXt", "resnet50", "resnet101"]


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    )
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = torch.cat([out.data, zero_pads], dim=1)
    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)

        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(
        self,
        block,
        layers,
        sample_size,
        sample_duration,
        shortcut_type="B",
        cardinality=32,
        num_classes=400,
    ):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=(1, 4, 4)
        )
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=(1, 2, 2)
        )
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=(1, 2, 2)
        )
        int(math.ceil(sample_duration / 16))
        int(math.ceil(sample_size / 32))

        self.avgpool = nn.AvgPool3d((1, 28, 28), stride=1)

        self.conv2 = nn.Conv3d(
            in_channels=512,
            out_channels=128,
            kernel_size=(5, 3, 3),
            stride=(1, 2, 2),
            padding=(2, 1, 1),
        )

        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        self.dropout1 = nn.Dropout(p=0.25, inplace=False)

        self.attention = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.Conv3d(
                in_channels=128,
                out_channels=64,
                kernel_size=(5, 3, 3),
                stride=(1, 1, 1),
                padding=(2, 1, 1),
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,
                out_channels=1,
                kernel_size=(5, 3, 3),
                stride=(1, 1, 1),
                padding=(2, 1, 1),
            ),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block, planes=planes * block.expansion, stride=stride
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv2(x)

        at_map = self.attention(x)
        x = x * at_map

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def _example_input():
    # neighbour_slice=8 (train_network.py default) stacked US frames, 1 input channel.
    # Spatial resolution 448x448 is required for self.avgpool's hardcoded AvgPool3d((1,28,28))
    # kernel to be valid given the real (unmodified) conv1/layer1/layer2/conv2 strides.
    return (torch.randn(1, 1, 8, 448, 448),)


def build_dclnet():
    # train_network.py::define_model(model_type='mynet'): mynet.resnet50(sample_size=2,
    # sample_duration=16, cardinality=32), then conv1 (and fc, see module header) is
    # overridden in-place, exactly as the real training script does.
    model = ResNeXt(
        ResNeXtBottleneck, [3, 4, 6, 3], sample_size=2, sample_duration=16, cardinality=32
    )
    model.conv1 = nn.Conv3d(
        in_channels=1,
        out_channels=64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(0, 3, 3),
        bias=False,
    )
    model.eval()
    with torch.no_grad():
        x = _example_input()[0]
        x = model.relu(model.bn1(model.conv1(x)))
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.conv2(x)
        at_map = model.attention(x)
        x = x * at_map
        x = model.avgpool(x)
        num_ftrs = x.view(x.size(0), -1).shape[1]
    model.fc = nn.Linear(num_ftrs, 6)
    model.eval()
    return model


def example_input_dclnet():
    return _example_input()


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DCL-Net", "build_dclnet", "example_input_dclnet", 2020, "vendored"),
]
