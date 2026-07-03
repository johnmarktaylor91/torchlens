# SOURCE: vendored from Ahmednull/L2CS-Net @ main
# https://raw.githubusercontent.com/Ahmednull/L2CS-Net/main/l2cs/model.py
#
# Abdelrahman, Hempel, Khalifa, Al-Hamadi 2022 "L2CS-Net: Fine-Grained Gaze Estimation
# in Unconstrained Environments" (arxiv 2203.03339) -- dual-branch (yaw/pitch) gaze
# estimation head on top of a real torchvision ResNet backbone (Bottleneck blocks built
# from the actual torchvision.models.resnet.Bottleneck class, per the official
# l2cs/utils.py::getArch ResNet50 config). The L2CS class itself is copied verbatim from
# l2cs/model.py; only the constructor call below picks the torchvision Bottleneck block
# (ResNet50 config: [3, 4, 6, 3]) as the official repo does.
"""L2CS-Net: dual-branch (yaw/pitch) gaze estimation on a torchvision ResNet backbone."""

import math

import torch
import torch.nn as nn
import torchvision

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from l2cs/model.py ---
class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

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

        # gaze
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze


def build_l2cs_net():
    # ResNet50 config, matching l2cs/utils.py::getArch default branch. Real
    # torchvision.models.resnet.Bottleneck block, tiny (2,2,2,2) layer counts for a
    # fast random-init trace (the real repo default is [3, 4, 6, 3] / full ResNet50).
    return L2CS(torchvision.models.resnet.Bottleneck, [2, 2, 2, 2], num_bins=28)


def example_input_l2cs_net():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ENTRIES = [
    ("L2CS-Net", "build_l2cs_net", "example_input_l2cs_net", 2022, "vendored"),
]
