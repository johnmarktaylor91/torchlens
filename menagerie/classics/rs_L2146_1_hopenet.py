# SOURCE: vendored from natanielruiz/deep-head-pose @ master
# https://raw.githubusercontent.com/natanielruiz/deep-head-pose/master/code/hopenet.py
#
# Ruiz, Chong, Rehg, 2018 (CVPR-W) "Fine-Grained Head Pose Estimation Without
# Keypoints". Hopenet is a ResNet50 trunk (built here from the real, unmodified
# `torchvision.models.resnet.Bottleneck` block -- exactly what the official repo's
# `test_hopenet.py` passes in: `hopenet.Hopenet(torchvision.models.resnet.Bottleneck,
# [3, 4, 6, 3], 66)`) with three parallel fully-connected heads (fc_yaw / fc_pitch /
# fc_roll) that classify each Euler angle into discretized bins; at inference the
# softmax-weighted expectation over bins gives a continuous angle (bin-and-delta
# classification+regression). The three-head split off a shared ResNet trunk is
# Hopenet's architectural contribution, so this is vendored (real code), not built
# from a stock library class.
#
# `code/hopenet.py` is reproduced verbatim below (only the unused `Variable` import
# is dropped since it is a no-op alias for `torch.Tensor` in modern torch).

import math

import torch.nn as nn
from torchvision.models.resnet import Bottleneck

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# code/hopenet.py :: Hopenet (verbatim, minus the vestigial ResNet/AlexNet
# sibling classes in the same file, which are unrelated baselines also defined
# in the repo but not part of the Hopenet architecture itself)
# ============================================================================


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_hopenet():
    # Real repo default (test_hopenet.py): ResNet50 trunk (torchvision's real,
    # unmodified Bottleneck block) with 66 angle bins.
    model = Hopenet(Bottleneck, [3, 4, 6, 3], 66)
    model.eval()
    return model


def example_input_hopenet():
    import torch

    torch.manual_seed(0)
    # Real repo feeds 224x224 RGB crops (see datasets.py transforms.Scale(224)).
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    ("HopNet", build_hopenet, example_input_hopenet, 2018, "vendored-pytorch"),
]
