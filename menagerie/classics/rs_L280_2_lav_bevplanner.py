# SOURCE: vendored from dotchen/LAV @ main
# (lav/models/bev_planner.py, lav/models/resnet.py)
"""LAV (CVPR 2022) -- Learning from All Vehicles.

LAV is a multi-component autonomous-driving system: a LiDAR/BEV perceiver (PointPillar +
sparse-conv backbone, needs `torch_scatter`), an RGB semantic-segmentation branch (ERFNet),
and the planning core -- `BEVPlanner`. `BEVPlanner` is the architecturally novel,
self-contained piece that gives LAV its name: it is trained not just to plan the ego
vehicle's trajectory from a bird's-eye-view feature crop, but to *cast* short-horizon
trajectories for every nearby vehicle under every one of `num_cmds` discrete high-level
commands (the "learning from all vehicles" auxiliary supervision), via a per-command bank
of GRU casting heads (`cast_grus`/`cast_mlps`) feeding a second GRU-based iterative planner
(`plan_gru`/`plan_mlp`) that refines the plan conditioned on a next-waypoint target `nxp`
cropped out of the BEV feature map with a custom affine-grid `crop_feature` (spatial
attention/ROI-align-style module, not a generic op). The vehicle-centric CNN backbone
(`bev_conv_emb`) is the repo's own `resnet18` fork with an added `num_channels` constructor
arg (arbitrary BEV-channel input instead of fixed RGB-3), vendored verbatim below from
`lav/models/resnet.py`; both files import only base `torch`, so this is a straight rung-2
vendor with no dependency substitution needed.

`BEVPlanner.forward` has one genuine data-dependent branch (whether any other vehicles are
present in `typs`, guarding a `torch.multinomial` random-subsample of nearby cars for
memory reasons at training time); the menagerie example input sets `typs` to all-zero so the
deterministic "no other vehicles in view" branch is exercised on every run.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# lav/models/resnet.py -- vendored verbatim (torchvision-style resnet fork with a
# `num_channels` constructor arg for arbitrary-channel BEV input)
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
        super().__init__()
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
                "replace_stride_with_dilation should be None or a 3-element tuple, got "
                f"{replace_stride_with_dilation}"
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
    if pretrained:
        raise NotImplementedError("pretrained weights not used for menagerie tracing")
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


# ---------------------------------------------------------------------------
# lav/models/bev_planner.py -- vendored verbatim
# ---------------------------------------------------------------------------


class BEVPlanner(nn.Module):
    def __init__(
        self,
        pixels_per_meter=2,
        crop_size=64,
        x_offset=0,
        y_offset=0.75,
        feature_x_jitter=1,
        feature_angle_jitter=10,
        num_plan=10,
        k=16,
        num_out_feature=64,
        num_cmds=6,
        max_num_cars=5,
        num_plan_iter=1,
    ):
        super().__init__()

        self.num_cmds = num_cmds
        self.num_plan = num_plan
        self.num_plan_iter = num_plan_iter
        self.max_num_cars = max_num_cars

        self.num_out_feature = num_out_feature

        self.pixels_per_meter = pixels_per_meter
        self.crop_size = crop_size

        self.feature_x_jitter = feature_x_jitter
        self.feature_angle_jitter = np.deg2rad(feature_angle_jitter)

        self.offset_x = nn.Parameter(torch.tensor(x_offset).float(), requires_grad=False)
        self.offset_y = nn.Parameter(torch.tensor(y_offset).float(), requires_grad=False)

        self.bev_conv_emb = nn.Sequential(
            resnet18(num_channels=5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.plan_gru = nn.GRU(4, 512, batch_first=True)
        self.plan_mlp = nn.Linear(512, 2)

        self.cast_grus = nn.ModuleList(
            [nn.GRU(512, 64, batch_first=True) for _ in range(self.num_cmds)]
        )
        self.cast_mlps = nn.ModuleList([nn.Linear(64, 2) for _ in range(self.num_cmds)])
        self.cast_cmd_pred = nn.Sequential(
            nn.Linear(512, self.num_cmds),
            nn.Sigmoid(),
        )

    def infer(self, bev, nxps):
        cropped_ego_bev = self.crop_feature(
            bev,
            torch.zeros((1, 2), dtype=bev.dtype, device=bev.device),
            torch.zeros((1,), dtype=bev.dtype, device=bev.device),
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size * 2,
        )

        ego_bev_embd = self.bev_conv_emb(cropped_ego_bev)

        ego_cast_locs = self.cast(ego_bev_embd)
        ego_plan_locs = self.plan(
            ego_bev_embd,
            nxps,
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size * 2,
        )
        ego_cast_cmds = self.cast_cmd_pred(ego_bev_embd)

        return ego_plan_locs, ego_cast_locs, ego_cast_cmds

    def forward(self, bev, ego_locs, locs, oris, nxps, typs):
        ego_oris = oris[:, :1]

        locs = locs[:, 1:]
        oris = oris[:, 1:]
        typs = typs[:, 1:] == 1  # 1 is for vehicles

        N = locs.size(1)

        # Only pick the good ones.
        typs = filter_cars(ego_locs, locs, typs)

        # Other vehicles
        if int(typs.float().sum()) > 0:
            # Guard against OOM: randomly sample cars to train on
            typs = random_sample(typs, size=self.max_num_cars)

            # Flatten the locs
            flat_bev = bev.expand(N, *bev.size()).permute(1, 0, 2, 3, 4).contiguous()[typs]

            flat_locs = (locs[:, :, 1:] - locs[:, :, :1])[typs]
            flat_rel_loc0 = (locs[:, :, 0] - ego_locs[:, None, 0])[typs]
            flat_rel_ori0 = (oris - ego_oris)[typs]

            K = flat_locs.size(0)

            locs_jitter = (torch.rand((K, 2)) * 2 - 1).float().to(
                locs.device
            ) * self.feature_x_jitter
            locs_jitter[:, 1] = 0
            oris_jitter = (torch.rand((K,)) * 2 - 1).float().to(
                oris.device
            ) * self.feature_angle_jitter

            cropped_other_bev = self.crop_feature(
                flat_bev,
                flat_rel_loc0 + locs_jitter,
                flat_rel_ori0 + oris_jitter,
                pixels_per_meter=self.pixels_per_meter,
                crop_size=self.crop_size * 2,
            )

            other_locs = transform_points(
                flat_locs - locs_jitter[:, None], -flat_rel_ori0 - oris_jitter
            )

            other_bev_embd = self.bev_conv_emb(cropped_other_bev)

            other_cast_locs = self.cast(other_bev_embd)
            other_cast_cmds = self.cast_cmd_pred(other_bev_embd)

        else:
            dtype = bev.dtype
            device = bev.device

            other_locs = torch.zeros((N, self.num_plan, 2), dtype=dtype, device=device)

            other_cast_locs = torch.zeros(
                (N, self.num_cmds, self.num_plan, 2), dtype=dtype, device=device
            )
            other_cast_cmds = torch.zeros((N, self.num_cmds), dtype=dtype, device=device)

        B = bev.size(0)

        cropped_ego_bev = self.crop_feature(
            bev,
            torch.zeros((B, 2), dtype=bev.dtype, device=bev.device),
            torch.zeros((B,), dtype=bev.dtype, device=bev.device),
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size * 2,
        )

        ego_bev_embd = self.bev_conv_emb(cropped_ego_bev)

        ego_cast_locs = self.cast(ego_bev_embd)
        ego_plan_locs = self.plan(
            ego_bev_embd,
            nxps,
            cast_locs=ego_cast_locs,
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size * 2,
        )
        ego_cast_cmds = self.cast_cmd_pred(ego_bev_embd)

        return (
            other_locs,
            other_cast_locs,
            other_cast_cmds,
            ego_plan_locs,
            ego_cast_locs,
            ego_cast_cmds,
        )

    def _plan(self, embd, nxp, cast_locs, pixels_per_meter=4, crop_size=96):
        B = embd.size(0)

        h0, u0 = embd, nxp * pixels_per_meter / crop_size * 2 - 1

        self.plan_gru.flatten_parameters()

        locs = []
        for i in range(self.num_cmds):
            u = torch.cat(
                [
                    u0.expand(self.num_plan, B, -1).permute(1, 0, 2),
                    cast_locs[:, i],
                ],
                dim=2,
            )
            out, _ = self.plan_gru(u, h0[None])
            locs.append(torch.cumsum(self.plan_mlp(out), dim=1))

        return torch.stack(locs, dim=1) + cast_locs

    def plan(self, embd, nxp, cast_locs=None, pixels_per_meter=4, crop_size=96):
        if cast_locs is None:
            plan_loc = self.cast(embd).detach()
        else:
            plan_loc = cast_locs.detach()

        plan_locs = []
        for _ in range(self.num_plan_iter):
            plan_loc = self._plan(
                embd, nxp, plan_loc, pixels_per_meter=pixels_per_meter, crop_size=crop_size
            )
            plan_locs.append(plan_loc)

        return torch.stack(plan_locs, dim=1)

    def cast(self, embd):
        B = embd.size(0)

        u = embd.expand(self.num_plan, B, -1).permute(1, 0, 2)

        locs = []
        for gru, mlp in zip(self.cast_grus, self.cast_mlps):
            gru.flatten_parameters()
            out, _ = gru(u)
            locs.append(torch.cumsum(mlp(out), dim=1))

        return torch.stack(locs, dim=1)

    def crop_feature(self, features, rel_locs, rel_oris, pixels_per_meter=4, crop_size=96):
        B, C, H, W = features.size()

        # ERROR proof hack...
        rel_locs = rel_locs.view(-1, 2)

        rel_locs = (
            rel_locs
            * pixels_per_meter
            / torch.tensor([H / 2, W / 2]).type_as(rel_locs).to(rel_locs.device)
        )

        cos = torch.cos(rel_oris)
        sin = torch.sin(rel_oris)

        rel_x = rel_locs[..., 0]
        rel_y = rel_locs[..., 1]

        k = crop_size / H

        rot_x_offset = -k * self.offset_x * cos + k * self.offset_y * sin + self.offset_x
        rot_y_offset = -k * self.offset_x * sin - k * self.offset_y * cos + self.offset_y

        theta = torch.stack(
            [
                torch.stack([k * cos, k * -sin, rot_x_offset + rel_x], dim=-1),
                torch.stack([k * sin, k * cos, rot_y_offset + rel_y], dim=-1),
            ],
            dim=-2,
        )

        grids = F.affine_grid(theta, torch.Size((B, C, crop_size, crop_size)), align_corners=True)

        cropped_features = F.grid_sample(features, grids, align_corners=True)

        return cropped_features


def transform_points(locs, oris):
    cos, sin = torch.cos(oris), torch.sin(oris)
    R = torch.stack(
        [
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1),
        ],
        dim=-2,
    )

    return locs @ R


def filter_cars(ego_locs, locs, typs):
    # We don't care about cars behind us ;)
    rel_locs = locs[:, :, 0] - ego_locs[:, 0:1]

    return typs & (rel_locs[..., 1] < 0)


def random_sample(binaries, size):
    cut_binaries = torch.zeros_like(binaries)
    for i in range(binaries.size(0)):
        if binaries[i].sum() <= size:
            cut_binaries[i] = binaries[i]
        else:
            nonzero = torch.nonzero(binaries[i]).squeeze(1)
            nonzero_idx = torch.multinomial(torch.ones_like(nonzero).float(), size)
            nonzero = nonzero[nonzero_idx]
            cut_binaries[i, nonzero] = binaries[i, nonzero]

    return cut_binaries


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------

_B = 1
_T = 11  # num_plan (10) + 1
_MAX_OBJS = 3  # real repo default max_objs is much larger; shrunk for a fast tiny trace
_H = _W = 64  # real repo crop_size*2 default is 128/192; shrunk for a fast tiny trace


def build_lav_bevplanner():
    model = BEVPlanner(
        pixels_per_meter=2,
        crop_size=_H // 2,
        num_plan=10,
        num_cmds=6,
        max_num_cars=5,
        num_plan_iter=1,
    )
    model.eval()
    return model


def example_input_lav_bevplanner():
    bev = torch.rand(_B, 5, _H, _W)
    ego_locs = torch.randn(_B, _T, 2)
    locs = torch.randn(_B, _MAX_OBJS, _T, 2)
    oris = torch.randn(_B, _MAX_OBJS)
    nxps = torch.randn(_B, 2)
    # all-zero typs -> deterministic "no other vehicles in view" forward branch
    typs = torch.zeros(_B, _MAX_OBJS, dtype=torch.int64)
    return (bev, ego_locs, locs, oris, nxps, typs)


MENAGERIE_ENTRIES = [
    (
        "LAV-BEVPlanner",
        build_lav_bevplanner,
        example_input_lav_bevplanner,
        2022,
        "vendored-pytorch",
    ),
]
