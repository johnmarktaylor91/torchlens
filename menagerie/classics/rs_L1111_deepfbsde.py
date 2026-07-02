# SOURCE: vendored from https://github.com/zhoufy20/deep-bsde-pytorch @ main
# (train.py, classes `Dense`, `Subnetwork`, `ForwardModel`)
#
# Deep BSDE Solver (Han, Jentzen, E 2018, PNAS 115(34):8505-8510,
# doi.org/10.1073/pnas.1718942115): reformulates a high-dimensional
# parabolic PDE (here the Allen-Cahn equation) as a backward stochastic
# differential equation and approximates the unknown gradient process Z_t at
# each time step with a small feedforward subnetwork, chaining these
# subnetworks across the discretized time interval; the network is trained
# so that the terminal-time BSDE simulation matches the PDE's known terminal
# condition. `ForwardModel` (the real "deep BSDE" network: a
# `nn.ModuleList` of per-timestep `Subnetwork` MLPs, each `Dense`
# Linear+BatchNorm1d+ReLU block) is copied verbatim below; only the
# `equation`/`default_parameters` imports were replaced by a tiny in-file
# equation stub exposing the same `eqn_dim` / `eqn_num_time_interval` /
# `eqn_total_time` / `eqn_delta_t` / `f_th` / `g_th` surface `ForwardModel`
# reads (the real `AllenCahn` equation class from the repo's `equation.py`,
# shrunk to a smaller dimension for a fast CPU trace).

import numpy as np
import torch

MENAGERIE_ZOO = "vendored-pytorch"

MOMENTUM = 0.99
EPSILON = 1e-6


# ---------------------------------------------------------------------------
# From train.py, verbatim.
# ---------------------------------------------------------------------------
class Dense(torch.nn.Module):
    def __init__(self, cin, cout, batch_norm=True, activate=True):
        super(Dense, self).__init__()
        self.cout = cout
        self.linear = torch.nn.Linear(cin, cout)
        self.activate = activate
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(cout, eps=EPSILON, momentum=MOMENTUM)
        else:
            self.bn = None
        torch.nn.init.normal_(self.linear.weight, std=5.0 / np.sqrt(cin + cout))

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate:
            x = torch.nn.functional.relu(x)
        return x


class Subnetwork(torch.nn.Module):
    def __init__(self, config):
        super(Subnetwork, self).__init__()
        self._config = config
        self.bn = torch.nn.BatchNorm1d(config["dim"], eps=EPSILON, momentum=MOMENTUM)
        self.layers = [
            Dense(config["num_hiddens"][i - 1], config["num_hiddens"][i])
            for i in range(1, len(config["num_hiddens"]) - 1)
        ]
        self.layers += [Dense(config["num_hiddens"][-2], config["num_hiddens"][-1], activate=False)]
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.bn(x)
        x = self.layers(x)
        return x


class ForwardModel(torch.nn.Module):
    def __init__(self, config, bsde):
        super(ForwardModel, self).__init__()
        self.config = config
        self.bsde = bsde
        self.device = config["default_Config"]["device"]
        self.TH_DTYPE = config["default_Config"]["TH_DTYPE"]
        self.DELTA_CLIP = config["default_Config"]["DELTA_CLIP"]

        # make sure consistent with FBSDE equation
        self.dim = bsde.eqn_dim

        self.num_time_interval = bsde.eqn_num_time_interval
        self.total_time = bsde.eqn_total_time
        self.y_init = torch.nn.Parameter(torch.Tensor([1]))
        self.y_init.data.uniform_(self.config["y_init_range"][0], self.config["y_init_range"][1])
        self.subnetworkList = torch.nn.ModuleList(
            [Subnetwork(config) for _ in range(self.num_time_interval - 1)]
        )

    def forward(self, x, dw):
        time_stamp = np.arange(0, self.bsde.eqn_num_time_interval) * self.bsde.eqn_delta_t
        z_init = (torch.zeros([1, self.dim]).uniform_(-0.1, 0.1).to(self.TH_DTYPE)).to(self.device)
        all_one_vec = torch.ones((dw.shape[0], 1), dtype=self.TH_DTYPE).to(self.device)

        y = all_one_vec * self.y_init
        z = torch.matmul(all_one_vec, z_init)

        for t in range(0, self.num_time_interval - 1):
            y = y - self.bsde.eqn_delta_t * (self.bsde.f_th(time_stamp[t], x[:, :, t], y, z))
            add = torch.sum(z * dw[:, :, t], dim=1, keepdim=True)
            y = y + add
            z = self.subnetworkList[t](x[:, :, t + 1]) / self.dim

        # terminal time
        y = (
            y
            - self.bsde.eqn_delta_t * self.bsde.f_th(time_stamp[-1], x[:, :, -2], y, z)
            + torch.sum(z * dw[:, :, -1], dim=1, keepdim=True)
        )

        criterion = torch.nn.MSELoss()
        loss = criterion(y, self.bsde.g_th(self.bsde.eqn_total_time, x[:, :, -1]))
        return loss, self.y_init


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
#
# `ForwardModel.__init__`/`forward` only ever read `bsde.eqn_dim`,
# `.eqn_num_time_interval`, `.eqn_total_time`, `.eqn_delta_t`, and call
# `.f_th(t, x, y, z)` / `.g_th(t, x)`. `_AllenCahnStub` reproduces those five
# members using the REAL AllenCahn equation from equation.py verbatim
# (f_th = y - y**3, g_th = 0.5 / (1 + 0.2*sum(x**2))) so the traced network
# exercises the actual per-op math the paper's flagship Allen-Cahn benchmark
# uses, without depending on the repo's `equation.py` import path.
# ---------------------------------------------------------------------------
class _AllenCahnStub:
    def __init__(self, dim, total_time, num_time_interval):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = total_time / num_time_interval

    @property
    def eqn_dim(self):
        return self._dim

    @property
    def eqn_num_time_interval(self):
        return self._num_time_interval

    @property
    def eqn_total_time(self):
        return self._total_time

    @property
    def eqn_delta_t(self):
        return self._delta_t

    def f_th(self, t, x, y, z):
        # Real AllenCahn.f_th, verbatim.
        return y - torch.pow(y, 3)

    def g_th(self, t, x):
        # Real AllenCahn.g_th, verbatim.
        return 0.5 / (1 + 0.2 * torch.sum(x**2, dim=1, keepdim=True))


_DIM = 6
_NUM_TIME_INTERVAL = 4
_BATCH = 4


class ForwardModelWrapped(torch.nn.Module):
    """Thin adapter so tl.trace() sees a plain (x, dw) -> Tensor forward
    instead of a (loss, y_init) tuple; the real ForwardModel.forward is
    called unmodified, we just select the scalar loss for the trace output.
    """

    def __init__(self, forward_model):
        super().__init__()
        self.forward_model = forward_model

    def forward(self, x, dw):
        loss, _y_init = self.forward_model(x, dw)
        return loss


def build_deepfbsde_forwardmodel():
    torch.manual_seed(0)
    config = {
        "default_Config": {
            "device": "cpu",
            "TH_DTYPE": torch.float32,
            "DELTA_CLIP": 50,
        },
        "dim": _DIM,
        "num_hiddens": [_DIM, _DIM + 2, _DIM + 2, _DIM],
        "y_init_range": [0.3, 0.6],
    }
    bsde = _AllenCahnStub(dim=_DIM, total_time=0.3, num_time_interval=_NUM_TIME_INTERVAL)
    inner = ForwardModel(config, bsde)
    inner.train()  # BatchNorm1d requires batch>1 in train mode; real solve() also trains
    model = ForwardModelWrapped(inner)
    return model


def example_input_deepfbsde_forwardmodel():
    torch.manual_seed(0)
    # x: [num_sample, dim, num_time_interval+1] simulated forward SDE paths
    # dw: [num_sample, dim, num_time_interval] Brownian increments
    x = torch.randn(_BATCH, _DIM, _NUM_TIME_INTERVAL + 1)
    dw = torch.randn(_BATCH, _DIM, _NUM_TIME_INTERVAL) * np.sqrt(0.3 / _NUM_TIME_INTERVAL)
    return (x, dw)


MENAGERIE_ENTRIES = [
    (
        "DeepBSDE-Solver-AllenCahn",
        "build_deepfbsde_forwardmodel",
        "example_input_deepfbsde_forwardmodel",
        2018,
        MENAGERIE_ZOO,
    ),
]
