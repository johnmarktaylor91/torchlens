# SOURCE: vendored from dotchen/LearningByCheating @ release-0.9.6
# (bird_view/models/birdview.py, bird_view/models/common.py, bird_view/models/resnet.py)
"""Learning by Cheating / LBC (CoRL 2019) -- privileged BEV teacher.

LBC trains a privileged bird's-eye-view "cheating" agent with full simulator state, then
distills it into a camera-only student via DAgger. `BirdViewPolicyModelSS` is the real
privileged-teacher policy network: a ResNet-18 stem over a 7-channel rendered BEV image,
late-fused with ego velocity, deconvolved back up to 48x48, and read out through FOUR
independent per-high-level-command branches (left/right/straight/follow), each a spatial-
softmax heatmap head (`common.SpatialSoftmax`) that regresses 5 future waypoint locations;
`common.select_branch` then gathers the branch matching the driver's one-hot command. This
branch-conditioned spatial-softmax waypoint head (not a generic classifier/regressor) is the
genuine LBC architecture contribution, vendored verbatim below (only `cv2`/`.agent`/
`.controller` imports dropped -- those back the closed-loop CARLA control-agent wrapper
`BirdViewAgent`, not the traceable policy network itself). The repo's own `bird_view/models/
resnet.py` fork (`ResNet` with configurable `input_channel`/`bias_first`, feature-map-only
`forward`, no avgpool/fc) is vendored verbatim too since it differs architecturally from
stock torchvision resnet (7-channel stem, first-conv `bias_first` toggle).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# bird_view/models/resnet.py -- vendored verbatim
# ---------------------------------------------------------------------------

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        input_channel=7,
        num_classes=1000,
        zero_init_residual=False,
        bias_first=True,
    ):
        super().__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=bias_first
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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

        return x


model_funcs = {
    "resnet18": (BasicBlock, [2, 2, 2, 2], -1),
    "resnet34": (BasicBlock, [3, 4, 6, 3], 512),
    "resnet50": (Bottleneck, [3, 4, 6, 3], -1),
    "resnet101": (Bottleneck, [3, 4, 23, 3], -1),
    "resnet152": (Bottleneck, [3, 8, 36, 3], -1),
}


def get_resnet(model_name="resnet18", pretrained=False, **kwargs):
    block, layers, c_out = model_funcs[model_name]
    model = ResNet(block, layers, **kwargs)

    if pretrained and kwargs.get("input_channel", 3) == 3:
        url = model_urls[model_name]
        model.load_state_dict(model_zoo.load_url(url))

    return model, c_out


# ---------------------------------------------------------------------------
# bird_view/models/common.py -- vendored (only the pieces BirdViewPolicyModelSS needs)
# ---------------------------------------------------------------------------


def select_branch(branches, one_hot):
    shape = branches.size()

    for i, s in enumerate(shape[2:]):
        one_hot = torch.stack([one_hot for _ in range(s)], dim=i + 2)

    return torch.sum(one_hot * branches, dim=1)


class ResnetBase(nn.Module):
    def __init__(self, backbone, input_channel=3, bias_first=True, pretrained=False):
        super().__init__()

        conv, c = get_resnet(
            backbone, input_channel=input_channel, bias_first=bias_first, pretrained=pretrained
        )

        self.conv = conv
        self.c = c

        self.backbone = backbone
        self.input_channel = input_channel
        self.bias_first = bias_first


class SpatialSoftmax(nn.Module):
    # Source: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
    def __init__(self, height, width, channel, temperature=None, data_format="NCHW"):
        super().__init__()

        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height), np.linspace(-1.0, 1.0, self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        if self.data_format == "NHWC":
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(torch.autograd.Variable(self.pos_x) * weight, dim=1, keepdim=True)
        expected_y = torch.sum(torch.autograd.Variable(self.pos_y) * weight, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)

        return feature_keypoints


# ---------------------------------------------------------------------------
# bird_view/models/birdview.py -- vendored verbatim (agent/control-loop wrapper dropped)
# ---------------------------------------------------------------------------

STEPS = 5
SPEED_STEPS = 3
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5


def spatial_softmax_base():
    return nn.Sequential(
        nn.BatchNorm2d(640),
        nn.ConvTranspose2d(640, 256, 3, 2, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
        nn.ReLU(True),
    )


class BirdViewPolicyModelSS(ResnetBase):
    def __init__(self, backbone="resnet18", input_channel=7, n_step=5, all_branch=False, **kwargs):
        super().__init__(backbone=backbone, input_channel=input_channel, bias_first=False)

        self.deconv = spatial_softmax_base()
        self.location_pred = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, STEPS, 1, 1, 0),
                    SpatialSoftmax(48, 48, STEPS),
                )
                for i in range(COMMANDS)
            ]
        )

        self.all_branch = all_branch

    def forward(self, bird_view, velocity, command):
        h = self.conv(bird_view)
        b, c, kh, kw = h.size()

        # Late fusion for velocity
        velocity = velocity[..., None, None, None].repeat((1, 128, kh, kw))

        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)

        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)

        location_pred = select_branch(location_preds, command)

        if self.all_branch:
            return location_pred, location_preds

        return location_pred


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------

_B = 1
_BEV_SIZE = 192  # matches repo's CROP_SIZE; the ResNet-18 stem/deconv stack requires
# this exact spatial scale to line up with the hardcoded SpatialSoftmax(48, 48, ...) heads


def build_lbc_birdview():
    model = BirdViewPolicyModelSS(backbone="resnet18", input_channel=7, all_branch=False)
    model.eval()
    return model


def example_input_lbc_birdview():
    bird_view = torch.rand(_B, 7, _BEV_SIZE, _BEV_SIZE)
    velocity = torch.rand(_B)
    # one-hot high-level command over the 4 branches (left/right/straight/follow)
    command = F.one_hot(torch.tensor([0]), COMMANDS).float()
    return (bird_view, velocity, command)


MENAGERIE_ENTRIES = [
    (
        "LBC-BirdViewPolicySS",
        build_lbc_birdview,
        example_input_lbc_birdview,
        2019,
        "vendored-pytorch",
    ),
]
