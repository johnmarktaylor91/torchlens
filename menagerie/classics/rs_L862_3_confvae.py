# SOURCE: vendored from https://github.com/MinkaiXu/ConfVAE-ICML21 @ main (d1d7469a67c82fc0e5d0d3bf8ce3572376f15a10)
# (models/common.py: GConv/GNNPrior/GNNEncoder; models/vae.py: CNFDecoder/build_flow/
# ImplicitVAE; models/cnf_edge/{cnf,odefunc,odegnn,odemlp}.py: the continuous-normalizing-
# -flow decoder -- CNF, ODEfunc, ODEgnn, ODEmlp, ConcatSquashLinear, SequentialFlow,
# MovingBatchNorm1d) -- the real ConfVAE "An End-to-End Framework for Molecular
# Conformation Generation via Bilevel Programming" (ICML 2021) VAE, unmodified.
# torch_geometric/torch_scatter/torch_sparse (graph ops) and torchdiffeq (the ODE
# solver the CNF decoder is built on) are all present in this environment and used
# exactly as in the original code.
#
# The original `models/vae.py` module additionally imports RDKit at module scope
# (`from rdkit.Chem import rdDepictor as DP`, `from rdkit.Chem.rdMolAlign import
# AlignMol, GetBestRMS`, `from utils.chem import *`) purely to support the OPTIONAL
# `implicit_loss` / `implicit_layer` / `sample` methods on `ImplicitVAE`, which
# align/embed 3D conformers via RDKit for the auxiliary "implicit" reconstruction
# term (`use_implicit=True`, off by default -- `ImplicitVAE.get_loss` defaults to
# `use_implicit=False`). Those RDKit-only methods are omitted here; the core
# trainable network -- prior GNN, encoder GNN, and CNF-flow decoder, trained via
# `get_loss(data, d, use_implicit=False)` exactly as the repo's own default -- is
# vendored verbatim below. RDKit is not installed in this environment and is not
# needed for the traced forward path.
import math
import sys
import types as _types
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_scatter import scatter, scatter_mean
from torch_sparse import SparseTensor
from torchdiffeq import odeint as odeint_normal
from torchdiffeq import odeint_adjoint

# torch_geometric's MessagePassing.inspector resolves `message()`'s type-hinted params
# via `sys.modules[self.__module__].__dict__`. If this file is loaded via
# importlib.util.module_from_spec() without the resulting module object also being
# registered in sys.modules under its own name, that lookup KeyErrors at GConv/GINEConv
# construction time. Registering the currently-executing module object here is
# load-time scaffolding only (mirrors what a normal `import` statement always does) --
# it does not touch the vendored architecture below.
if __name__ not in sys.modules:
    sys.modules[__name__] = _types.ModuleType(__name__)


# ---------------------------------------------------------------------------
# models/common.py (verbatim, minus training-only LR-scheduler helpers)
# ---------------------------------------------------------------------------
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def shifted_softplus(x):
    return F.softplus(x) - np.log(2.0)


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return shifted_softplus(x)


def get_activation_fn(fn, module=False):
    mods = {
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "ssp": ShiftedSoftplus,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
    }
    funcs = {
        "relu": F.relu,
        "softplus": F.softplus,
        "ssp": shifted_softplus,
        "tanh": F.tanh,
        "elu": F.elu,
    }

    if module:
        return mods[fn]()
    else:
        return funcs[fn]


class SimpleMLP(nn.Module):
    def __init__(self, dims, activation, bias=True, last_act=False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if (i < len(dims) - 1) or last_act:
                layers.append(get_activation_fn(activation, module=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------------------------
# models/vae.py :: GConv / GNNPrior / GNNEncoder (verbatim)
# ---------------------------------------------------------------------------
class GConv(MessagePassing):
    def __init__(self, eps: float = 0.0, train_eps: bool = False, **kwargs):
        super(GConv, self).__init__(aggr="add", **kwargs)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.softplus(x_j + edge_attr)


class GNNPrior(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.act = F.softplus
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.node_emb = torch.nn.Embedding(100, hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, hidden_dim)

        self.conv1 = GConv()
        self.bn_conv1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GConv()
        self.bn_conv2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GConv()
        self.bn_conv3 = torch.nn.BatchNorm1d(hidden_dim)

        self.out_fc1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.bn_out1 = torch.nn.BatchNorm1d(hidden_dim)
        self.out_fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_out2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.out_fc3 = torch.nn.Linear(hidden_dim // 2, latent_dim * 2)

    def forward(self, node_type, edge_type, edge_index, batch):
        node_attr = self.node_emb(node_type)
        edge_attr = self.edge_emb(edge_type)

        h = node_attr
        h = self.act(self.bn_conv1(self.conv1(h, edge_index, edge_attr)))
        h = self.act(self.bn_conv2(self.conv2(h, edge_index, edge_attr)))
        h = self.bn_conv3(self.conv3(h, edge_index, edge_attr))

        h_global = scatter(h, batch, dim=0, reduce="sum")
        node_feat = torch.cat([h, h_global[batch]], dim=-1)
        node_feat = self.act(self.bn_out1(self.out_fc1(node_feat)))
        node_feat = self.act(self.bn_out2(self.out_fc2(node_feat)))
        out = self.out_fc3(node_feat)

        mu = out[:, : self.latent_dim]
        sigma = out[:, self.latent_dim :]

        return mu, sigma


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.act = F.softplus
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.node_emb = torch.nn.Embedding(100, hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, hidden_dim)

        self.d_fc1 = torch.nn.Linear(1, hidden_dim)
        self.bn_d1 = torch.nn.BatchNorm1d(hidden_dim)
        self.d_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn_d2 = torch.nn.BatchNorm1d(hidden_dim)

        self.conv1 = GConv()
        self.bn_conv1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GConv()
        self.bn_conv2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GConv()
        self.bn_conv3 = torch.nn.BatchNorm1d(hidden_dim)

        self.out_fc1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.bn_out1 = torch.nn.BatchNorm1d(hidden_dim)
        self.out_fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_out2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.out_fc3 = torch.nn.Linear(hidden_dim // 2, latent_dim * 2)

    def forward(self, x, node_type, edge_type, edge_index, batch):
        node_attr = self.node_emb(node_type)
        edge_attr = self.edge_emb(edge_type)

        d_emb = self.act(self.bn_d1(self.d_fc1(x)))  # Embedings for edge lengths `x`
        d_emb = self.bn_d2(self.d_fc2(d_emb))
        edge_attr = d_emb * edge_attr

        h = node_attr
        h = self.act(self.bn_conv1(self.conv1(h, edge_index, edge_attr)))
        h = self.act(self.bn_conv2(self.conv2(h, edge_index, edge_attr)))
        h = self.bn_conv3(self.conv3(h, edge_index, edge_attr))

        h_global = scatter(h, batch, dim=0, reduce="sum")
        node_feat = torch.cat([h, h_global[batch]], dim=-1)
        node_feat = self.act(self.bn_out1(self.out_fc1(node_feat)))
        node_feat = self.act(self.bn_out2(self.out_fc2(node_feat)))
        out = self.out_fc3(node_feat)

        mu = out[:, : self.latent_dim]
        sigma = out[:, self.latent_dim :]

        return mu, sigma


# ---------------------------------------------------------------------------
# models/cnf_edge/odemlp.py (verbatim: ConcatSquashLinear + ODEmlp)
# ---------------------------------------------------------------------------
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
}


class ODEmlp(nn.Module):
    """Helper class to make neural nets for use in continuous normalizing flows."""

    def __init__(self, hidden_dims, input_shape, context_dim=0, nonlinearity="softplus"):
        super().__init__()
        base_layer = ConcatSquashLinear

        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out in hidden_dims + (input_shape[0],):
            layer = base_layer(hidden_shape[0], dim_out, context_dim)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(hidden_shape)
            hidden_shape[0] = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y, context=None):
        dx = y
        for layer_idx, layer in enumerate(self.layers):
            if context is not None:
                tc = torch.cat([t, context.view(y.size(0), -1)], dim=1)
            else:
                tc = t
            dx = layer(tc, dx)
            if layer_idx < len(self.layers) - 1:
                dx = self.activation_fns[layer_idx](dx)
        return dx


# ---------------------------------------------------------------------------
# models/cnf_edge/odegnn.py (verbatim: GINEConv (edge-CNF variant) + ODEgnn)
# ---------------------------------------------------------------------------
class GINEConv(MessagePassing):
    def __init__(self, nn_module: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs):
        super(GINEConv, self).__init__(aggr="add", **kwargs)
        self.nn = nn_module
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self,
        t,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(t, out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.softplus(x_j + edge_attr)


class ODEgnn(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.act = F.softplus
        self.d_fc1 = ConcatSquashLinear(1, hidden_dim, dim_c=0)
        self.d_fc2 = ConcatSquashLinear(hidden_dim, hidden_dim, dim_c=0)

        self.conv1 = GINEConv(ODEmlp((hidden_dim,), (hidden_dim,)))
        self.conv2 = GINEConv(ODEmlp((hidden_dim,), (hidden_dim,)))
        self.conv3 = GINEConv(ODEmlp((hidden_dim,), (hidden_dim,)))

        self.out_fc1 = ConcatSquashLinear(2 * hidden_dim, hidden_dim, dim_c=0)
        self.out_fc2 = ConcatSquashLinear(hidden_dim, hidden_dim // 2, dim_c=0)
        self.out_fc3 = ConcatSquashLinear(hidden_dim // 2, 1, dim_c=0)

        self.edge_index = None

    def forward(self, t, x, node_attr, edge_attr):
        assert self.edge_index is not None, "`edge_index` is not prepared."
        edge_index = self.edge_index

        d_emb = self.act(self.d_fc1(t, x))
        d_emb = self.d_fc2(t, d_emb)  # Embedings for edge lengths `x`
        edge_attr = d_emb * edge_attr

        t_node = torch.ones_like(node_attr)[0, :1] * t.mean()
        h = node_attr
        h = self.act(self.conv1(t_node, h, edge_index, edge_attr))
        h = self.act(self.conv2(t_node, h, edge_index, edge_attr))
        h = self.conv3(t_node, h, edge_index, edge_attr)

        h_row, h_col = h[edge_index[0]], h[edge_index[1]]
        pair_feat = torch.cat([h_row * h_col, edge_attr], dim=-1)
        pair_feat = self.act(self.out_fc1(t, pair_feat))
        pair_feat = self.act(self.out_fc2(t, pair_feat))
        out = self.out_fc3(t, pair_feat)

        return out


# ---------------------------------------------------------------------------
# models/cnf_edge/odefunc.py (verbatim)
# ---------------------------------------------------------------------------
def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)

    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1

    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, "(failed to add node to graph)"
    return approx_tr_dzdx


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.0))

    def before_odeint(self, edge_index=None, e=None):
        self._e = e
        self._num_evals.fill_(0)
        self.diffeq.edge_index = edge_index

    def forward(self, t, states):
        y = states[0]
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True):
            assert len(states) == 4  # conditional CNF: x, logpx, node_attr, edge_attr
            node_attr, edge_attr = states[2:]
            dy = self.diffeq(t, y, node_attr, edge_attr)
            divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)
            return (
                dy,
                -divergence,
                torch.zeros_like(node_attr).requires_grad_(True),
                torch.zeros_like(edge_attr).requires_grad_(True),
            )


# ---------------------------------------------------------------------------
# models/cnf_edge/cnf.py (verbatim: CNF, SequentialFlow, MovingBatchNorm1d)
# ---------------------------------------------------------------------------
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class CNF(nn.Module):
    def __init__(
        self,
        odefunc,
        T=1.0,
        train_T=False,
        regularization_fns=None,
        solver="dopri5",
        atol=1e-5,
        rtol=1e-5,
        use_adjoint=True,
    ):
        super().__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))

        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(
        self, x, node_attr, edge_attr, edge_index, logpx=None, integration_times=None, reverse=False
    ):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx

        states = (x, _logpx, node_attr, edge_attr)
        atol = [self.atol] * 3
        rtol = [self.rtol] * 3

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.tensor(0.0).to(x), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x)
            else:
                integration_times = torch.tensor([0.0, self.T], requires_grad=False).to(x)

        if reverse:
            integration_times = _flip(integration_times, 0)

        self.odefunc.before_odeint(edge_index=edge_index)
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]

        if logpx is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(
        self,
        x,
        node_attr,
        edge_attr,
        edge_index,
        logpx=None,
        reverse=False,
        inds=None,
        integration_times=None,
    ):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](
                    x,
                    node_attr=node_attr,
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    logpx=logpx,
                    integration_times=integration_times,
                    reverse=reverse,
                )
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](
                    x,
                    node_attr=node_attr,
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    logpx=logpx,
                    integration_times=integration_times,
                    reverse=reverse,
                )
            return x, logpx


class MovingBatchNormNd(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0.0, affine=True, sync=False):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.sync = sync
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer("step", torch.zeros(1))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    def _forward(self, x, logpx=None):
        num_channels = x.size(-1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        if self.training:
            x_t = x.transpose(0, 1).reshape(num_channels, -1)
            batch_mean = torch.mean(x_t, dim=1)
            batch_var = torch.var(x_t, dim=1)

            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= 1.0 - self.bn_lag ** (self.step[0] + 1)
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= 1.0 - self.bn_lag ** (self.step[0] + 1)

            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1

        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _reverse(self, y, logpy=None):
        used_mean = self.running_mean
        used_var = self.running_var

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad


class MovingBatchNorm1d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1]

    def forward(
        self, x, node_attr, edge_attr, edge_index, logpx=None, integration_times=None, reverse=False
    ):
        ret = super(MovingBatchNorm1d, self).forward(x, logpx=logpx, reverse=reverse)
        return ret


# ---------------------------------------------------------------------------
# models/vae.py :: build_flow / CNFDecoder / ImplicitVAE (verbatim, minus the
# RDKit-only implicit_loss/implicit_layer/sample methods -- see header note)
# ---------------------------------------------------------------------------
def build_flow(args, hidden_dim, num_blocks):
    def build_cnf():
        diffeq = ODEgnn(hidden_dim=hidden_dim)
        odefunc = ODEfunc(diffeq=diffeq)
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            solver=args.solver,
            use_adjoint=args.use_adjoint,
            atol=args.atol,
            rtol=args.rtol,
        )
        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    if args.batch_norm:
        bn_layers = [
            MovingBatchNorm1d(1, bn_lag=args.bn_lag, sync=args.sync_bn) for _ in range(num_blocks)
        ]
        bn_chain = [MovingBatchNorm1d(1, bn_lag=args.bn_lag, sync=args.sync_bn)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)
    return model


class CNFDecoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_emb = torch.nn.Linear(args.latent_dim, args.latent_dim)
        self.node_emb = torch.nn.Embedding(100, args.hidden_dim - args.latent_dim)
        self.edge_emb = torch.nn.Embedding(100, args.hidden_dim)
        self.flow = build_flow(
            args,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
        )
        self.use_deterministic_encoder = args.use_deterministic_encoder

    def emb(self, node_type, edge_type, latent):
        node_attr = torch.cat([self.latent_emb(latent), self.node_emb(node_type)], dim=1)
        edge_attr = self.edge_emb(edge_type)
        return node_attr, edge_attr

    def get_log_prob(self, data, d, latent):
        E = d.size(0)
        node_attr, edge_attr = self.emb(data.node_type, data.edge_type, latent)
        z, delta_logpz = self.flow(
            x=d,
            node_attr=node_attr,
            edge_attr=edge_attr,
            edge_index=data.edge_index,
            logpx=torch.zeros(E, 1).to(d),
        )
        log_pz = standard_normal_logprob(z).view(E, -1).sum(1, keepdim=True)
        log_pd = log_pz - delta_logpz
        return log_pd

    def get_loss(self, data, d, latent):
        log_pd = self.get_log_prob(data, d, latent)
        loss = -log_pd.mean()
        return loss


class ImplicitVAE(torch.nn.Module):
    def __init__(self, args):
        super(ImplicitVAE, self).__init__()
        self.device = args.device
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.latent_dim = args.latent_dim
        self.kl_weight = args.kl_weight
        self.implicit_weight = args.implicit_weight
        self.prior = GNNPrior(args.hidden_dim, args.latent_dim)
        self.encoder = GNNEncoder(args.hidden_dim, args.latent_dim)
        self.decoder = CNFDecoder(args)

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    def get_nll(self, data, d, use_implicit=False, eval=False):
        # sigma is logvar
        # q(z|G,D)
        mu_q, sigma_q = self.encoder(
            data.edge_length, data.node_type, data.edge_type, data.edge_index, data.batch
        )
        # p(z|G)
        mu_p, sigma_p = self.prior(data.node_type, data.edge_type, data.edge_index, data.batch)
        # KL Distance
        loss_kl = self.compute_vae_kl(mu_q, sigma_q, mu_p, sigma_p)

        # infer latent
        if self.use_deterministic_encoder:
            latent = mu_q + 0 * torch.exp(0.5 * sigma_q)
        else:
            latent = self.reparameterize_gaussian(mu_q, sigma_q)

        # Reconstrcution
        # p(D|G,z)
        log_pd = self.decoder.get_log_prob(data, d, latent)
        loss_rec_d = -log_pd

        # p(X|G,z) -- implicit RDKit-based term is intentionally not vendored
        # (see module header); use_implicit is always False here.
        loss_rec_x = None

        return loss_kl, loss_rec_d, loss_rec_x

    @staticmethod
    def compute_vae_kl(mu_q, logvar_q, mu_prior, logvar_prior):
        mu1 = mu_q
        std1 = torch.exp(0.5 * logvar_q)
        mu2 = mu_prior
        std2 = torch.exp(0.5 * logvar_prior)
        kl = (
            -0.5
            + torch.log(std2 / (std1 + 1e-8) + 1e-8)
            + ((torch.pow(std1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(std2, 2)))
        )
        return kl

    def forward(self, data, d):
        """Menagerie glue: real repo callers use `.get_loss(data, d)`; forward
        dispatches straight to it so `tl.trace(model, data, d)` exercises the
        real training-time computation graph (prior GNN -> encoder GNN ->
        reparameterize -> CNF-flow decoder log-prob)."""
        loss_kl, loss_rec_d, _ = self.get_nll(data, d, use_implicit=False)
        bs = data.batch[-1] + 1
        loss_kl = self.kl_weight * loss_kl.sum() / bs
        loss_rec_d = loss_rec_d.sum() / bs
        loss = loss_kl + loss_rec_d
        return loss, loss_kl, loss_rec_d


# ---------------------------------------------------------------------------
# Menagerie staging glue: tiny config + synthetic molecule-graph batch
# ---------------------------------------------------------------------------
class _Namespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _confvae_args() -> _Namespace:
    # Mirrors ConfVAE's own default training config (sized down for a tiny model).
    return _Namespace(
        device=torch.device("cpu"),
        use_deterministic_encoder=False,
        latent_dim=4,
        kl_weight=1.0,
        implicit_weight=0.0,
        hidden_dim=12,
        num_blocks=1,
        time_length=0.5,
        train_T=False,
        solver="dopri5",
        use_adjoint=False,
        atol=1e-3,
        rtol=1e-3,
        batch_norm=False,
        bn_lag=0.0,
        sync_bn=False,
    )


def _synthetic_qm9_batch() -> Data:
    """Two tiny toy molecules batched into one torch_geometric Data object,
    matching the fields `ImplicitVAE.get_nll`/`GNNEncoder`/`GNNPrior`/`CNFDecoder`
    read: `node_type`, `edge_type`, `edge_index`, `edge_length`, `batch`."""
    torch.manual_seed(0)
    edge_index_0 = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    edge_index_1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long) + 4

    edge_index = torch.cat([edge_index_0, edge_index_1], dim=1)
    n_edges = edge_index.size(1)
    edge_type = torch.randint(1, 4, (n_edges,), dtype=torch.long)
    node_type = torch.randint(1, 9, (7,), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    edge_length = torch.rand(n_edges, 1) + 0.5

    data = Data(
        node_type=node_type,
        edge_type=edge_type,
        edge_index=edge_index,
        edge_length=edge_length,
        batch=batch,
    )
    return data


def _synthetic_edge_distances(data: Data) -> torch.Tensor:
    return data.edge_length


def build_confvae():
    model = ImplicitVAE(_confvae_args())
    model.eval()  # avoid BatchNorm1d needing batch-size > 1 per graph in this toy demo
    return model


def example_input_confvae():
    data = _synthetic_qm9_batch()
    d = _synthetic_edge_distances(data)
    return (data, d)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ConfVAE", build_confvae, example_input_confvae, 2021, "SOURCE_AVAILABLE"),
]
