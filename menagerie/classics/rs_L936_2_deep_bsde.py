# SOURCE: vendored from skyhehe123/DeepBSDE-pytorch @ master
#   (repo: https://github.com/skyhehe123/DeepBSDE-pytorch)
#   solver.py (Dense, Subnetwork, FeedForwardModel) + equation.py (Equation,
#   AllenCahn.f_th/g_th), copied verbatim.
#
# DeepBSDE (Han, Jentzen & E, "Solving high-dimensional partial differential
# equations using deep learning", PNAS 2018) reformulates a high-dimensional
# parabolic PDE as a backward stochastic differential equation (BSDE) and
# approximates the unknown gradient process Z_t at each of the
# `num_time_interval` discretized timesteps with its own small feedforward
# subnetwork (`Subnetwork`: BatchNorm1d -> stack of `Dense`
# Linear+BatchNorm1d+ReLU blocks -> final linear, no activation). The main
# `FeedForwardModel.forward` recursively updates a running scalar value
# estimate `y` and its gradient estimate `z` across timesteps using the
# per-step subnetwork outputs and the Brownian increments `dw`, exactly as
# in the original TensorFlow `frankhan91/DeepBSDE` (queue notes: "original
# repo TF; multiple PyTorch reimplementations exist") -- this repo is one of
# those confirmed PyTorch reimplementations, a real from-scratch
# `nn.Module` reproduction (not a base-lib class), so this is rung 2
# (vendor), not rung 1. The specific PDE used here is `AllenCahn`
# (`f_th`/`g_th` from equation.py) matching this repo's shipped training
# config. Code copied verbatim except: (1) the two hardcoded `.cuda()`
# calls in `FeedForwardModel.forward` are made device-agnostic (follow the
# input tensor's device instead of unconditionally moving to GPU -- the
# repo assumes a CUDA machine; this environment may not have one, and the
# architecture itself does not require CUDA) -- everything else (subnetwork
# structure, recursive y/z BSDE update, loss computation) is untouched;
# (2) `equation.py`'s `Equation.sample()` (numpy RNG data generation, not
# part of the traced nn.Module graph) is dropped -- `example_input_` below
# generates the same-shaped (dw, x) tensors directly with torch.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

MENAGERIE_ZOO = "vendored-pytorch"

TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


# ---------------------------------------------------------------------------
# Verbatim from equation.py @ master (base Equation + AllenCahn f_th/g_th)
# ---------------------------------------------------------------------------
class AllenCahnEquation:
    """The AllenCahn PDE generator/terminal functions used by this repo's
    shipped training config (config.py: AllenCahnConfig)."""

    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t

    def f_th(self, t, x, y, z):
        return y - torch.pow(y, 3)

    def g_th(self, t, x):
        return 0.5 / (1 + 0.2 * torch.sum(x**2, dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# Verbatim from solver.py @ master
# ---------------------------------------------------------------------------
class Dense(nn.Module):
    def __init__(self, cin, cout, batch_norm=True, activate=True):
        super(Dense, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.activate = activate
        if batch_norm:
            self.bn = nn.BatchNorm1d(cout, eps=EPSILON, momentum=MOMENTUM)
        else:
            self.bn = None
        nn.init.normal_(self.linear.weight, std=5.0 / np.sqrt(cin + cout))

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate:
            x = F.relu(x)
        return x


class Subnetwork(nn.Module):
    def __init__(self, config):
        super(Subnetwork, self).__init__()
        self._config = config
        self.bn = nn.BatchNorm1d(config.dim, eps=EPSILON, momentum=MOMENTUM)
        self.layers = [
            Dense(config.num_hiddens[i - 1], config.num_hiddens[i])
            for i in range(1, len(config.num_hiddens) - 1)
        ]
        self.layers += [Dense(config.num_hiddens[-2], config.num_hiddens[-1], activate=False)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.bn(x)
        x = self.layers(x)
        return x


class FeedForwardModel(nn.Module):
    """The fully connected neural network model."""

    def __init__(self, config, bsde):
        super(FeedForwardModel, self).__init__()
        self._config = config
        self._bsde = bsde

        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time

        self._y_init = Parameter(torch.Tensor([1]))
        self._y_init.data.uniform_(self._config.y_init_range[0], self._config.y_init_range[1])
        self._subnetworkList = nn.ModuleList(
            [Subnetwork(config) for _ in range(self._num_time_interval - 1)]
        )

    def forward(self, x, dw):
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        # NOTE: the original repo hardcodes `.cuda()` here for both z_init
        # and all_one_vec; made device-agnostic (follow `dw`'s device)
        # instead -- everything else in this method is verbatim.
        device = dw.device
        z_init = torch.zeros([1, self._dim]).uniform_(-0.1, 0.1).to(TH_DTYPE).to(device)

        all_one_vec = torch.ones((dw.shape[0], 1), dtype=TH_DTYPE, device=device)
        y = all_one_vec * self._y_init

        z = torch.matmul(all_one_vec, z_init)

        for t in range(0, self._num_time_interval - 1):
            y = y - self._bsde.delta_t * (self._bsde.f_th(time_stamp[t], x[:, :, t], y, z))
            add = torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            y = y + add
            z = self._subnetworkList[t](x[:, :, t + 1]) / self._dim
        # terminal time
        y = (
            y
            - self._bsde.delta_t
            * self._bsde.f_th(
                time_stamp[-1],
                x[:, :, -2],
                y,
                z,
            )
            + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)
        )

        delta = y - self._bsde.g_th(self._total_time, x[:, :, -1])

        # use linear approximation outside the clipped range
        loss = torch.mean(
            torch.where(
                torch.abs(delta) < DELTA_CLIP,
                delta**2,
                2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP**2,
            )
        )
        return loss, self._y_init


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
class _TinyConfig:
    """Tiny stand-in for config.py's AllenCahnConfig, shrunk for a fast
    staging trace (real repo default: dim=100, num_time_interval=20)."""

    n_layer = 4
    y_init_range = [0.3, 0.6]
    dim = 8
    num_hiddens = [8, 18, 18, 8]


_DIM = 8
_TOTAL_TIME = 0.3
_NUM_TIME_INTERVAL = 4


def build_deep_bsde():
    bsde = AllenCahnEquation(dim=_DIM, total_time=_TOTAL_TIME, num_time_interval=_NUM_TIME_INTERVAL)
    model = FeedForwardModel(_TinyConfig(), bsde)
    model.eval()
    return model


def example_input_deep_bsde():
    """A (dw, x) pair with the real `(num_sample, dim, num_time_interval[+1])`
    shapes `Equation.sample()` produces: `dw` is the Brownian increments
    over `num_time_interval` steps, `x` is the forward SDE path sampled at
    `num_time_interval + 1` points (matching AllenCahnEquation's random
    walk), fed directly to `FeedForwardModel.forward(x, dw)`."""
    torch.manual_seed(0)
    num_sample = 4
    dw = torch.randn(num_sample, _DIM, _NUM_TIME_INTERVAL) * np.sqrt(
        _TOTAL_TIME / _NUM_TIME_INTERVAL
    )
    x = torch.zeros(num_sample, _DIM, _NUM_TIME_INTERVAL + 1)
    sigma = np.sqrt(2.0)
    for i in range(_NUM_TIME_INTERVAL):
        x[:, :, i + 1] = x[:, :, i] + sigma * dw[:, :, i]
    return (x, dw)


MENAGERIE_ENTRIES = [
    (
        "DeepBSDE",
        build_deep_bsde,
        example_input_deep_bsde,
        2018,
        "CODE",
    ),
]
