# SOURCE: vendored from https://github.com/zqhang/MTGFLOW @ main
# (models/MTGFLOW.py::GNN/ScaleDotProductAttention/MTGFLOW + models/NF.py::MaskedLinear/
#  LinearMaskedCoupling/BatchNorm/FlowSequential/MADE/MAF)
#
# MTGFlow (Zhou et al., AAAI 2023, "Detecting Multivariate Time Series Anomalies with
# Zero Known Label"): an unsupervised multivariate time-series anomaly detector that
# combines a per-sensor LSTM encoder, a learned dynamic-graph attention module (a
# scaled-dot-product "attention" over the flattened window that produces a KxK sensor
# adjacency matrix), a graph neural network (GNN) that propagates the LSTM hidden states
# along that learned graph, and a conditional Masked Autoregressive Flow (MAF, from
# Papamakarios et al. 2017) that models the density of the graph-conditioned
# representations. Anomaly score = negative log-likelihood under the flow.
#
# Vendored real repo code verbatim: GNN, ScaleDotProductAttention, MTGFLOW (models/
# MTGFLOW.py) and MaskedLinear, LinearMaskedCoupling, BatchNorm, FlowSequential, MADE,
# MAF (models/NF.py) -- every Linear/LSTM/masking/coupling layer, shape, and the
# graph-conditioned log_prob math is unchanged from the original. Only non-architectural
# scaffolding was dropped: the unused `interpolate`/`plot_attention` helper functions in
# MTGFLOW.py (dead code / matplotlib debug plotting, never called from MTGFLOW.forward),
# the unrelated `test`/`MAF_Full`/`MADE_Full`/`create_masks_pmu` variants (alternate
# unconditional-flow code paths not used by the real MTGFLOW training entry point in
# main.py, which always constructs `MTGFLOW(..., model="MAF")`), and the module-level
# `#%%` Jupyter cell markers / `from cgitb import reset` / `from turtle import forward,
# shape` dead imports (unused, artifacts of the authors' notebook workflow).

import math

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


# --------------------------------------------------------------------------------
# models/MTGFLOW.py
# --------------------------------------------------------------------------------


class GNN(nn.Module):
    """
    The GNN module applied in GANF/MTGFlow: propagates per-sensor hidden states
    along the learned adjacency matrix `A`, with a causal (lag-1) recurrent term.
    """

    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        # A: K X K
        # H: N X K X L X D
        h_n = self.lin_n(torch.einsum("nkld,nkj->njld", h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class ScaleDotProductAttention(nn.Module):
    """
    Compute scale dot product attention over the flattened (window*input_size)
    sensor representation, producing a learned KxK sensor adjacency ("graph").
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor: [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0], shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        score = self.dropout(self.softmax(score))

        return score, k


# --------------------------------------------------------------------------------
# models/NF.py
# --------------------------------------------------------------------------------


def create_masks(input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None):
    # MADE paper sec 4: degrees of connections between layers -- ensure at most
    # in_degree - 1 connections.
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

    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """MADE building block layer."""

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


class BatchNorm(nn.Module):
    """RealNVP BatchNorm layer."""

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
            self.batch_var = x.var(0)  # note: MAF paper uses biased variance estimate

            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

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
    """Container for layers of a normalizing flow."""

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
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        net = []
        for m in masks[1:-1]:
            net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian


class MAF(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_sensor,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="relu",
        input_order="sequential",
        batch_norm=True,
        mode="rand",
    ):
        super().__init__()
        if mode == "zero":
            self.register_buffer("base_dist_mean", torch.zeros(n_sensor, 1))
            self.register_buffer("base_dist_var", torch.ones(n_sensor, 1))
        elif mode == "rand":
            self.register_buffer("base_dist_mean", torch.randn(n_sensor, 1))
            self.register_buffer("base_dist_var", torch.ones(n_sensor, 1))
        else:
            raise AttributeError("no choice")

        modules = []
        self.input_size = input_size
        self.input_degrees = None
        for _ in range(n_blocks):
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

    def base_dist_logp(self, z, k, window_size):
        N = z.shape[0] // k // window_size
        logp = -0.5 * (z - self.base_dist_mean.repeat_interleave(window_size, 0).repeat(N, 1)) ** 2
        return logp

    def forward(self, x, y=None):
        return self.net(x, y)

    def log_prob(self, x, k, window_size, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        C = u.shape[1]
        return (
            torch.sum(self.base_dist_logp(u, k, window_size) + sum_log_abs_det_jacobians, dim=1)
            + C * _GCONST_
        )


# --------------------------------------------------------------------------------
# models/MTGFLOW.py :: MTGFLOW (top-level model)
# --------------------------------------------------------------------------------


class MTGFLOW(nn.Module):
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        window_size,
        n_sensor,
        dropout=0.1,
        model="MAF",
        batch_norm=True,
    ):
        super(MTGFLOW, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout
        )
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model == "MAF":
            self.nf = MAF(
                n_blocks,
                n_sensor,
                input_size,
                hidden_size,
                n_hidden,
                cond_label_size=hidden_size,
                batch_norm=batch_norm,
                activation="tanh",
            )

        self.attention = ScaleDotProductAttention(window_size * input_size)

    def forward(self, x):
        return self.test(x).mean()

    def test(self, x):
        # x: N X K X L X D
        full_shape = x.shape
        graph, _ = self.attention(x)
        self.graph = graph
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # reshape: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshape N*K*L, H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0], -1])
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def get_graph(self):
        return self.graph


def build_mtgflow():
    # Tiny config matching the real repo's default MAF/GNN/LSTM sizing shape
    # (main.py defaults: n_blocks=1, input_size=1, hidden_size=32, n_hidden=1,
    # window_size=60) but with a small window_size/n_sensor for a fast trace.
    return MTGFLOW(
        n_blocks=1,
        input_size=1,
        hidden_size=8,
        n_hidden=1,
        window_size=6,
        n_sensor=5,
        dropout=0.0,
        model="MAF",
        batch_norm=True,
    )


def example_input_mtgflow():
    # x: N x K x L x D = (batch, n_sensor, window_size, input_size)
    return torch.randn(2, 5, 6, 1)


MENAGERIE_ENTRIES = [
    ("MTGFlow", build_mtgflow, example_input_mtgflow, 2023, MENAGERIE_ZOO),
]
