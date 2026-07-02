# SOURCE: vendored from https://github.com/EnyanDai/GANF @ main
# (models/GANF.py + models/NF.py [MAF/MADE/MaskedLinear/BatchNorm/
#  FlowSequential only])
#
# GANF (Dai & Chen, ICLR 2022, "Graph-Augmented Normalizing Flow for
# Multivariate Time Series Anomaly Detection") couples an LSTM temporal
# encoder with a graph-convolution module (`GNN`, learned adjacency `A`
# applied via `torch.einsum`) whose output conditions a Masked Autoregressive
# Flow (MAF, built from MADE blocks) that scores per-timestep log-likelihood.
# This vendors the real repo's `GNN`/`GANF` classes verbatim from
# `models/GANF.py`, plus the real `MAF`/`MADE`/`MaskedLinear`/`BatchNorm`/
# `FlowSequential`/`create_masks` classes from `models/NF.py` that `GANF`
# constructs internally (`model="MAF"` branch; the `RealNVP` alternative
# branch is present in the real repo but not exercised on this path). No
# layer, mechanism, or dataflow was changed; only the demo `#%%` cell
# markers were dropped and the two files' imports were flattened into one
# module (no relative `models.*` package).

import copy
import math

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/NF.py (MADE / MaskedLinear / BatchNorm / FlowSequential / MAF)
# ---------------------------------------------------------------------------
def create_masks(input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    if input_size > 1:
        if input_order == "sequential":
            degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                degrees += [torch.arange(hidden_size) % (input_size - 1)]
            degrees += (
                [torch.arange(input_size) % input_size - 1]
                if input_degrees is None
                else [input_degrees % input_size - 1]
            )

        elif input_order == "random":
            degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
                degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += (
                [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
                if input_degrees is None
                else [input_degrees - 1]
            )
    else:
        degrees += [torch.zeros([1]).long()]
        for _ in range(n_hidden + 1):
            degrees += [torch.zeros([hidden_size]).long()]
        degrees += [torch.zeros([input_size]).long()]
    # construct masks
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """MADE building block layer"""

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer("mask", mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(
                torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size)
            )

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size is not None) * ", cond_features={}".format(self.cond_label_size)


class BatchNorm(nn.Module):
    """RealNVP BatchNorm layer"""

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(
                0
            )  # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """Container for layers of a normalizing flow"""

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="relu",
        input_order="sequential",
        input_degrees=None,
    ):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        # setup activation
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        Dd = u.shape[1]  # noqa: F841 (unused in real repo too)
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MAF(nn.Module):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="relu",
        input_order="sequential",
        batch_norm=True,
    ):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


# ---------------------------------------------------------------------------
# models/GANF.py
# ---------------------------------------------------------------------------
class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """

    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        # A: K X K
        # H: N X K X L X D
        h_n = self.lin_n(torch.einsum("nkld,kj->njld", h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class GANF(nn.Module):
    def __init__(
        self, n_blocks, input_size, hidden_size, n_hidden, dropout=0.1, model="MAF", batch_norm=True
    ):
        super(GANF, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout
        )
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        # model="MAF" branch only (the real repo's alternative RealNVP branch
        # is not exercised here; MAF is the paper's reported configuration)
        self.nf = MAF(
            n_blocks,
            input_size,
            hidden_size,
            n_hidden,
            cond_label_size=hidden_size,
            batch_norm=batch_norm,
            activation="tanh",
        )

    def forward(self, x, A):
        return self.test(x, A).mean()

    def test(self, x, A):
        # x: N X K X L X D
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # reshape: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, A)

        # reshape N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, h).reshape([full_shape[0], -1])
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def locate(self, x, A):
        # x: N X K X L X D
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # reshape: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, A)

        # reshape N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, h).reshape([full_shape[0], full_shape[1], -1])
        log_prob = log_prob.mean(dim=2)

        return log_prob


def build_ganf():
    # n_blocks=1 (single MADE flow block), input_size=3 (sensor feature
    # dim D), hidden_size=8, n_hidden=1 -- smallest sizes that still exercise
    # every real module (LSTM -> GNN graph-conv -> MAF/MADE flow).
    return GANF(
        n_blocks=1,
        input_size=3,
        hidden_size=8,
        n_hidden=1,
        dropout=0.0,
        model="MAF",
        batch_norm=True,
    )


def example_input_ganf():
    # x: N=2 windows X K=4 sensors/nodes X L=6 timesteps X D=3 features,
    # matching the real repo's documented `x` layout in GANF.test/.forward.
    x = torch.randn(2, 4, 6, 3)
    # A: K X K learned/precomputed adjacency (row-stochastic here, matching
    # how eval_water.py / train_water.py normalize the graph before GANF).
    A = torch.softmax(torch.randn(4, 4), dim=1)
    return (x, A)


MENAGERIE_ENTRIES = [
    (
        "GANF",
        build_ganf,
        example_input_ganf,
        2022,
        MENAGERIE_ZOO,
    ),
]
