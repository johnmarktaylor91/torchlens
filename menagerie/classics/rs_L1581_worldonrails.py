# SOURCE: vendored from dotchen/WorldOnRails @ release (files: rails/models/main_model.py,
# rails/models/ego_model.py, common/resnet.py, common/normalize.py, common/segmentation.py --
# fetched 2026-07-02).
#
# This vendors the REAL "World on Rails" perception+control model: the `CameraModel`
# (two-camera ResNet18/34 backbones + segmentation decoder heads + speed-conditioned discrete
# action head, from `rails/models/main_model.py`) and the `EgoModel` (differentiable kinematic
# bicycle model used as the learned forward dynamics / "world model", from
# `rails/models/ego_model.py`). Together these form the coupled world-model + policy that gives
# the CoRL 2021 paper its name. `common/resnet.py` is the standard torchvision-style ResNet
# feature extractor (unmodified architecture, avgpool/fc unused/commented as in upstream) used
# as CameraModel's backbone; kept verbatim since it is imported directly by main_model.py.
#
# Code below is the upstream source with only mechanical edits: cross-file imports flattened
# into this single module, `torch.hub.load_state_dict_from_url` pretrained-download path
# disabled (random init only, as required for a trace/recipe), everything else untouched.

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# common/resnet.py (standard torchvision-style ResNet; unmodified architecture)
# ---------------------------------------------------------------------------
def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_channels: int = 3,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    # NOTE: pretrained-download path intentionally disabled here (random init only);
    # upstream calls torch.hub.load_state_dict_from_url(model_urls[arch]).
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


# ---------------------------------------------------------------------------
# common/normalize.py
# ---------------------------------------------------------------------------
class Normalize(nn.Module):
    """ImageNet normalization"""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

    def forward(self, x):
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]


# ---------------------------------------------------------------------------
# common/segmentation.py
# ---------------------------------------------------------------------------
class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_labels):
        super().__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, num_labels, 1, 1, 0),
        )

    def forward(self, x):
        return self.upconv(x)


# ---------------------------------------------------------------------------
# rails/models/main_model.py :: CameraModel (the "World on Rails" perception + policy net)
# ---------------------------------------------------------------------------
def action_logits(raw_logits, num_steers, num_throts):
    steer_logits = raw_logits[..., :num_steers]
    throt_logits = raw_logits[..., num_steers : num_steers + num_throts]
    brake_logits = raw_logits[..., -1:]

    steer_logits = steer_logits.repeat(1, 1, 1, num_throts)
    throt_logits = throt_logits.repeat_interleave(num_steers, -1)

    act_logits = torch.cat([steer_logits + throt_logits, brake_logits], dim=-1)

    return act_logits


class CameraModel(nn.Module):
    def __init__(self, config, num_cmds=6):
        super().__init__()

        # Configs
        self.num_cmds = num_cmds
        self.num_steers = config["num_steers"]
        self.num_throts = config["num_throts"]
        self.num_speeds = config["num_speeds"]
        self.num_labels = len(config["seg_channels"])
        self.all_speeds = config["all_speeds"]
        self.two_cam = config["use_narr_cam"]

        self.backbone_wide = resnet34(pretrained=config["imagenet_pretrained"])
        self.seg_head_wide = SegmentationHead(512, self.num_labels + 1)
        if self.two_cam:
            self.backbone_narr = resnet18(pretrained=config["imagenet_pretrained"])
            self.seg_head_narr = SegmentationHead(512, self.num_labels + 1)
            self.bottleneck_narr = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(True),
            )

        if self.all_speeds:
            self.num_acts = (
                self.num_cmds * self.num_speeds * (self.num_steers + self.num_throts + 1)
            )
        else:
            self.num_acts = self.num_cmds * (self.num_steers + self.num_throts + 1)
            self.spd_encoder = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(True),
                nn.Linear(64, 64),
                nn.ReLU(True),
            )

        self.wide_seg_head = SegmentationHead(512, self.num_labels + 1)
        self.act_head = nn.Sequential(
            nn.Linear(512 + (0 if self.all_speeds else 64) + (64 if self.two_cam else 0), 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, self.num_acts),
        )

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, wide_rgb, narr_rgb, spd=None):
        assert (self.all_speeds and spd is None) or (not self.all_speeds and spd is not None)

        wide_embed = self.backbone_wide(self.normalize(wide_rgb / 255.0))
        wide_seg_output = self.seg_head_wide(wide_embed)

        if self.two_cam:
            narr_embed = self.backbone_narr(self.normalize(narr_rgb / 255.0))
            narr_seg_output = self.seg_head_narr(narr_embed)
            embed = torch.cat(
                [
                    wide_embed.mean(dim=[2, 3]),
                    self.bottleneck_narr(narr_embed.mean(dim=[2, 3])),
                ],
                dim=1,
            )
        else:
            embed = wide_embed.mean(dim=[2, 3])

        # Action logits
        if self.all_speeds:
            act_output = self.act_head(embed).view(
                -1, self.num_cmds, self.num_speeds, self.num_steers + self.num_throts + 1
            )
            act_output = action_logits(act_output, self.num_steers, self.num_throts)
        else:
            act_output = self.act_head(
                torch.cat([embed, self.spd_encoder(spd[:, None])], dim=1)
            ).view(-1, self.num_cmds, 1, self.num_steers + self.num_throts + 1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts).squeeze(2)

        if self.two_cam:
            return act_output, wide_seg_output, narr_seg_output
        else:
            return act_output, wide_seg_output


# ---------------------------------------------------------------------------
# rails/models/ego_model.py :: EgoModel (learned kinematic-bicycle world/dynamics model)
# ---------------------------------------------------------------------------
class EgoModel(nn.Module):
    def __init__(self, dt=1.0 / 4):
        super().__init__()

        self.dt = dt

        # Kinematic bicycle model
        self.front_wb = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.rear_wb = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.steer_gain = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.brake_accel = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.throt_accel = nn.Sequential(
            nn.Linear(1, 1, bias=False),
        )

    def forward(self, locs, yaws, spds, acts):
        """only plannar"""

        steer = acts[..., 0:1]
        throt = acts[..., 1:2]
        brake = acts[..., 2:3].byte()

        accel = torch.where(brake, self.brake_accel.expand(*brake.size()), self.throt_accel(throt))
        wheel = self.steer_gain * steer

        beta = torch.atan(self.rear_wb / (self.front_wb + self.rear_wb) * torch.tan(wheel))

        next_locs = (
            locs + spds * torch.cat([torch.cos(yaws + beta), torch.sin(yaws + beta)], -1) * self.dt
        )
        next_yaws = yaws + spds / self.rear_wb * torch.sin(beta) * self.dt
        next_spds = spds + accel * self.dt

        return next_locs, next_yaws, F.relu(next_spds)


# ---------------------------------------------------------------------------
# Staging module build/example-input functions
# ---------------------------------------------------------------------------
def build_worldonrails_camera():
    """Tiny-size real CameraModel (two-camera wide+narrow ResNet perception+control net)."""
    config = dict(
        num_steers=9,
        num_throts=3,
        num_speeds=4,
        seg_channels=[4, 6, 7, 8, 10],
        imagenet_pretrained=False,
        all_speeds=True,
        use_narr_cam=True,
    )
    return CameraModel(config)


def example_input_worldonrails_camera():
    wide_rgb = torch.zeros(1, 3, 32, 96)
    narr_rgb = torch.zeros(1, 3, 32, 96)
    return wide_rgb, narr_rgb


class EgoModelTraceWrapper(nn.Module):
    """Wraps EgoModel so it can be traced from a single packed tensor input."""

    def __init__(self, net: EgoModel):
        super().__init__()
        self.net = net

    def forward(self, state):
        locs = state[..., 0:2]
        yaws = state[..., 2:3]
        spds = state[..., 3:4]
        acts = state[..., 4:7]
        next_locs, next_yaws, next_spds = self.net(locs, yaws, spds, acts)
        return torch.cat([next_locs, next_yaws, next_spds], dim=-1)


def build_worldonrails_ego():
    return EgoModelTraceWrapper(EgoModel())


def example_input_worldonrails_ego():
    # [locs(2), yaws(1), spds(1), acts(steer, throt, brake)(3)] packed along last dim.
    state = torch.zeros(4, 7)
    state[:, 3] = 1.0  # nonzero speed
    state[:, 6] = 0.0  # brake=0 (falsy .byte()) to avoid NaN branch of torch.where
    return (state,)


MENAGERIE_ENTRIES = [
    (
        "World on Rails (CameraModel)",
        build_worldonrails_camera,
        example_input_worldonrails_camera,
        2021,
        MENAGERIE_ZOO,
    ),
    (
        "World on Rails (EgoModel kinematic world-model)",
        build_worldonrails_ego,
        example_input_worldonrails_ego,
        2021,
        MENAGERIE_ZOO,
    ),
]
