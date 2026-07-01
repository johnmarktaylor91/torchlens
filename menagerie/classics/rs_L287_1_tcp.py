# SOURCE: vendored from https://github.com/OpenDriveLab/TCP @ main
# TCP/resnet.py (modified torchvision ResNet returning penultimate feature map)
# and TCP/model.py (TCP dual-branch trajectory + control prediction model), lightly
# adapted only to (a) drop the `pretrained=True` ImageNet download and (b) drop the
# unused CARLA-specific PID controller / inference-only helper methods that depend on
# numpy control loops with no tensor ops (process_action / control_pid / get_action).
# Architecture (ResNet-34 perception backbone, GRUCell trajectory + control decoders,
# attention-weighted feature re-pooling, dual value/policy heads) is unchanged.
from collections import deque
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

MENAGERIE_ZOO = "vendored-pytorch"

# --- from TCP/resnet.py (verbatim architecture, pretrained-download path unused) ---


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
        super().__init__()
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


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
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
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock],
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

    def _forward_impl(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_layer4 = self.layer4(x)

        x = self.avgpool(x_layer4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, x_layer4

    def forward(self, x: Tensor):
        return self._forward_impl(x)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 model (pretrained weight download disabled for staging)."""
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# --- from TCP/model.py (TCP), with `pretrained=True` -> `pretrained=False` and the
# CARLA-inference-only numpy helper methods (process_action / control_pid / get_action /
# PIDController) dropped since they contain no tensor ops relevant to tracing. ---


class TCPConfig:
    """Minimal stand-in for TCP.config.GlobalConfig (only the fields TCP.__init__ reads)."""

    def __init__(self, pred_len: int = 4):
        self.pred_len = pred_len


class TCP(nn.Module):
    def __init__(self, config: TCPConfig):
        super().__init__()
        self.config = config

        self.perception = resnet34(pretrained=False)

        self.measurements = nn.Sequential(
            nn.Linear(1 + 2 + 6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.join_traj = nn.Sequential(
            nn.Linear(128 + 1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.join_ctrl = nn.Sequential(
            nn.Linear(128 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.value_branch_traj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.value_branch_ctrl = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        dim_out = 2

        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
        )
        self.decoder_ctrl = nn.GRUCell(input_size=256 + 4, hidden_size=256)
        self.output_ctrl = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
        self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())

        self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
        self.output_traj = nn.Linear(256, 2)

        self.init_att = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 29 * 8),
            nn.Softmax(1),
        )

        self.wp_att = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 29 * 8),
            nn.Softmax(1),
        )

        self.merge = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, img, state, target_point):
        feature_emb, cnn_feature = self.perception(img)
        outputs = {}
        outputs["pred_speed"] = self.speed_branch(feature_emb)
        measurement_feature = self.measurements(state)

        j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))
        outputs["pred_value_traj"] = self.value_branch_traj(j_traj)
        outputs["pred_features_traj"] = j_traj
        z = j_traj
        output_wp = list()
        traj_hidden_state = list()

        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)

        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.decoder_traj(x_in, z)
            traj_hidden_state.append(z)
            dx = self.output_traj(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        outputs["pred_wp"] = pred_wp

        traj_hidden_state = torch.stack(traj_hidden_state, dim=1)
        init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)
        feature_emb = torch.sum(cnn_feature * init_att, dim=(2, 3))
        j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
        outputs["pred_value_ctrl"] = self.value_branch_ctrl(j_ctrl)
        outputs["pred_features_ctrl"] = j_ctrl
        policy = self.policy_head(j_ctrl)
        outputs["mu_branches"] = self.dist_mu(policy)
        outputs["sigma_branches"] = self.dist_sigma(policy)

        x = j_ctrl
        mu = outputs["mu_branches"]
        sigma = outputs["sigma_branches"]
        future_feature, future_mu, future_sigma = [], [], []

        h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, mu, sigma], dim=1)
            h = self.decoder_ctrl(x_in, h)
            wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
            new_feature_emb = torch.sum(cnn_feature * wp_att, dim=(2, 3))
            merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
            dx = self.output_ctrl(merged_feature)
            x = dx + x

            policy = self.policy_head(x)
            mu = self.dist_mu(policy)
            sigma = self.dist_sigma(policy)
            future_feature.append(x)
            future_mu.append(mu)
            future_sigma.append(sigma)

        outputs["future_feature"] = future_feature
        outputs["future_mu"] = future_mu
        outputs["future_sigma"] = future_sigma
        return outputs


def build_tcp() -> nn.Module:
    return TCP(TCPConfig(pred_len=4))


def example_input_tcp():
    # TCPAgent's front camera is 900x256 (W x H) per leaderboard/team_code/tcp_agent.py
    # sensors(); the hardcoded 8*29 attention grids in init_att/wp_att below assume this
    # exact resnet34-layer4 output spatial size (8, 29), so the input must stay non-square.
    img = torch.randn(1, 3, 256, 900)
    state = torch.randn(1, 9)  # 1 (speed) + 2 (target_point) + 6 (one-hot command)
    target_point = torch.randn(1, 2)
    return (img, state, target_point)


MENAGERIE_ENTRIES = [
    ("TCP", "build_tcp", "example_input_tcp", 2022, "vendored-pytorch"),
]
