# SOURCE: vendored from topazape/ST-ResNet @ master
# https://raw.githubusercontent.com/topazape/ST-ResNet/master/stresnet/models/stresnet.py
# https://raw.githubusercontent.com/topazape/ST-ResNet/master/stresnet/models/resunit.py
#
# ST-ResNet (AAAI 2017): Zhang, Zheng, Qi, "Deep Spatio-Temporal Residual
# Networks for Citywide Crowd Flows Prediction". Modern PyTorch reimplementation
# of the original architecture (closeness/period/trend residual CNN branches
# fused with learned per-pixel weights + optional external-factor branch).
#
# Vendored verbatim (only trivial edit: inlined the `from stresnet.models import
# ResUnit` package-relative import since we vendor both files into one module).
# Architecture is untouched.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class ResUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x


class STResNet(nn.Module):
    def __init__(
        self,
        len_closeness: int,
        len_period: int,
        len_trend: int,
        external_dim: Optional[int],
        nb_flow: int,
        map_height: int,
        map_width: int,
        nb_residual_unit: int,
    ) -> None:
        super().__init__()
        self.external_dim = external_dim
        self.map_height = map_height
        self.map_width = map_width
        self.nb_flow = nb_flow
        self.nb_residual_unit = nb_residual_unit

        # models
        self.c_net = self._create_timenet(len_closeness)
        self.p_net = self._create_timenet(len_period)
        self.t_net = self._create_timenet(len_trend)
        if self.external_dim:
            # in/out flows * (len_closeness + len_period + len_trend)
            nb_total_flows = self.nb_flow * (len_closeness + len_period + len_trend)
            self.e_net = self._create_extnet(self.external_dim, nb_total_flows=nb_total_flows)

        # for fusion
        self.W_c = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_p = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_t = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )

    def _create_extnet(self, ext_dim: int, nb_total_flows: int) -> nn.Sequential:
        ext_net = nn.Sequential(
            nn.Linear(ext_dim, nb_total_flows),
            nn.ReLU(inplace=True),
            # flatten in/out flow * grid_height * grid_width
            nn.Linear(nb_total_flows, self.nb_flow * self.map_height * self.map_width),
        )
        return ext_net

    def _create_timenet(self, length: int) -> nn.Sequential:
        time_net = nn.Sequential()
        time_net.add_module(
            "Conv1",
            nn.Conv2d(
                in_channels=(length * self.nb_flow),
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        for i in range(self.nb_residual_unit):
            time_net.add_module(f"ResUnit{i + 1}", ResUnit(in_channels=64, out_channels=64))

        time_net.add_module(
            "Conv2",
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding="same"),
        )
        return time_net

    def forward(
        self,
        xc: torch.Tensor,
        xp: torch.Tensor,
        xt: torch.Tensor,
        ext: Optional[torch.Tensor],
    ) -> torch.Tensor:
        c_out = self.c_net(xc)
        p_out = self.p_net(xp)
        t_out = self.t_net(xt)

        if self.external_dim:
            e_out = self.e_net(ext).view(-1, self.nb_flow, self.map_width, self.map_height)
            # fusion with ext data
            res = self.W_c.unsqueeze(0) * c_out
            res += self.W_p.unsqueeze(0) * p_out
            res += self.W_t.unsqueeze(0) * t_out
            res += e_out
        else:
            res = self.W_c.unsqueeze(0) * c_out
            res += self.W_p.unsqueeze(0) * p_out
            res += self.W_t.unsqueeze(0) * t_out

        return torch.tanh(res)


# --- Menagerie staging wrapper --------------------------------------------------
#
# Real constructor signature needs closeness/period/trend history lengths, a
# grid map size, and residual-unit depth. Sized small for a fast trace;
# external_dim branch is exercised since it is real architecture (not a stub).

_LEN_CLOSENESS = 3
_LEN_PERIOD = 2
_LEN_TREND = 2
_EXTERNAL_DIM = 8
_NB_FLOW = 2
_MAP_HEIGHT = 8
_MAP_WIDTH = 8
_NB_RESIDUAL_UNIT = 2


def build_st_resnet():
    return STResNet(
        len_closeness=_LEN_CLOSENESS,
        len_period=_LEN_PERIOD,
        len_trend=_LEN_TREND,
        external_dim=_EXTERNAL_DIM,
        nb_flow=_NB_FLOW,
        map_height=_MAP_HEIGHT,
        map_width=_MAP_WIDTH,
        nb_residual_unit=_NB_RESIDUAL_UNIT,
    )


def example_input_st_resnet():
    xc = torch.randn(1, _LEN_CLOSENESS * _NB_FLOW, _MAP_HEIGHT, _MAP_WIDTH)
    xp = torch.randn(1, _LEN_PERIOD * _NB_FLOW, _MAP_HEIGHT, _MAP_WIDTH)
    xt = torch.randn(1, _LEN_TREND * _NB_FLOW, _MAP_HEIGHT, _MAP_WIDTH)
    ext = torch.randn(1, _EXTERNAL_DIM)
    return xc, xp, xt, ext


MENAGERIE_ENTRIES = [
    ("ST-ResNet", build_st_resnet, example_input_st_resnet, 2017, "CODE"),
]
