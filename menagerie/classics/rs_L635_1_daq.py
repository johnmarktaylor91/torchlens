# SOURCE: vendored from kiharalab/DAQ @ 16c9287c61affd13a99395e115c0af0dadb218ee
# https://raw.githubusercontent.com/kiharalab/DAQ/main/models/resnet.py
#
# Terashi & Kihara et al. "Residue-Wise Local Quality Assessment for Protein Complex
# Cryo-EM Structures" DAQ (Nature Methods 2022). The real model class is `ResNet_custom`,
# a 3D ResNeXt-style (cardinality-grouped Conv3d) CNN with a single conv1/bn1/maxpool stem
# over 4 residual stages (Bottleneck blocks with a `dropout` after every conv, matching
# the actual code) and 3 separate linear heads (`fc1`/`fc2`/`fc3` over 20/6/3 output
# classes -- amino-acid identity, secondary-structure class, and DAQ backbone/sidechain
# quality bins respectively). `predict/predict_trimmap.py` constructs the real model via
# `resnet18(sample_size=params['voxel_size'])`, feeding a single-channel voxelized cryo-EM
# density cube; `voxel_size` defaults to 11 per `ops/argparser.py`. Copied verbatim from
# the real `models/resnet.py`; only the CUDA branch inside `downsample_basic_block` (dead
# on CPU, `isinstance(out.data, torch.cuda.FloatTensor)` never True here) is untouched --
# no architecture lines were changed.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def downsample_basic_block(x, planes, stride):
    """
    :param x:
    :param planes: required output, specify new channel
    :param stride:
    :return:
    x format: [batch_size,channel,dimension0,dimension1,dimension2]
    """
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        """
        :param inplanes: int in number of filters
        :param planes: out number of filters
        :param cardinality:
        :param stride:
        :param downsample: specify downsample or not
        """
        super(Bottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out


class ResNet_custom(nn.Module):
    def __init__(
        self, block, layers, sample_size, shortcut_type="B", cardinality=32, num_classes=[20, 6, 3]
    ):
        """
        :param block: choose bloce
        :param layers: 4-length list: combine 4 differen layer with different specifications
        :param sample_size:
        :param sample_duration:
        :param shortcut_type:
        :param cardinality:
        :param num_classes:num of classes that we need to predict
        """
        self.inplanes = 64
        super(ResNet_custom, self).__init__()
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False
        )  # Out input channel size is 1, our map
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_size, last_size, last_size), stride=1)
        self.fc1 = nn.Linear(cardinality * 32 * block.expansion, num_classes[0])
        self.fc2 = nn.Linear(cardinality * 32 * block.expansion, num_classes[1])
        self.fc3 = nn.Linear(cardinality * 32 * block.expansion, num_classes[2])

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        """
        :param block: residue block define before
        :param planes: number of filters
        :param blocks: number of times use this block
        :param shortcut_type: specify use downsample or not
        :param cardinality:
        :param stride: stride in CNN
        :return:
        """
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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)

        return x1, x2, x3


def resnet18(**kwargs):
    """Constructs a ResNet-18 model. We will use this,
    actually 6*3+2=20
    """
    model = ResNet_custom(Bottleneck, [1, 2, 2, 1], **kwargs)
    return model


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_daq_resnet18() -> nn.Module:
    # sample_size mirrors the real `predict_trimmap` call `resnet18(sample_size=...)`;
    # the real default voxel_size is 11 (ops/argparser.py --voxel_size default=11).
    # cardinality is left at the real default (32): the fc heads are hardcoded to
    # `cardinality * 32 * block.expansion` in the real code (not derived from
    # `self.inplanes`/layer4's actual output channels), so shrinking cardinality breaks
    # the real forward pass with a shape mismatch -- this is architecture-as-written, not
    # a staging choice. layers=[1,2,2,1] (the real `resnet18` block-count config, already
    # the smallest offered) plus the small voxel_size keeps the trace fast regardless.
    model = ResNet_custom(Bottleneck, [1, 2, 2, 1], sample_size=11, cardinality=32)
    model.eval()
    return model


def example_input_daq_resnet18():
    # (batch, 1, voxel, voxel, voxel) single-channel voxelized cryo-EM density cube,
    # exactly what predict_trimmap.py builds into `input_use` before `model(input_data)`.
    return (torch.randn(2, 1, 11, 11, 11),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DAQ", "build_daq_resnet18", "example_input_daq_resnet18", 2022, "vendored"),
]
