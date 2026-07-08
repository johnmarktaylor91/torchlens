# ruff: noqa: E741  (verbatim upstream uses single-letter `l` for spherical-harmonic degree)
# SOURCE: vendored from mzjb/DeepH-pack @ main
# https://github.com/mzjb/DeepH-pack/blob/main/deeph/model.py
# https://github.com/mzjb/DeepH-pack/blob/main/deeph/from_se3_transformer/representations.py
# https://github.com/mzjb/DeepH-pack/blob/main/deeph/from_schnetpack/acsf.py
# https://github.com/mzjb/DeepH-pack/blob/main/deeph/from_PyG_future/graph_norm.py
# https://github.com/mzjb/DeepH-pack/blob/main/deeph/from_PyG_future/diff_group_norm.py
# https://github.com/mzjb/DeepH-pack/blob/main/deeph/from_HermNet/rmnet.py
#
# DeepH (Deep Hamiltonian): equivariant graph neural network that predicts DFT tight-binding
# Hamiltonian matrix elements between atom-pair "edges" of a crystal graph (Li, Zhang, Yang
# et al., Nature Computational Science 2022, https://github.com/mzjb/DeepH-pack). This file
# inlines the real `HGNN` entry point from `deeph/model.py` verbatim, plus its four small
# vendored-by-upstream submodules (`from_se3_transformer/representations.py` for the
# spherical-harmonics basis, `from_schnetpack/acsf.py` for the Gaussian distance basis,
# `from_PyG_future/{graph_norm,diff_group_norm}.py` for GraphNorm/DiffGroupNorm, and
# `from_HermNet/rmnet.py` for the PAINN-style RBF/cutoff/ShiftedSoftplus helpers) --
# only change: the four submodules' relative imports are collapsed into this one file
# (base-library imports -- torch/torch_geometric/torch_scatter/numpy/scipy -- hoisted and
# deduplicated). No architectural change: same CGConv/GAT_Crystal/PAINN message-passing
# blocks, same 5-layer MPLayer stack, same edge-readout head.
import os
from math import ceil, sqrt
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import comb
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.inits import ones as pyg_ones
from torch_geometric.nn.inits import zeros as pyg_zeros
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from torch_geometric.nn.norm import InstanceNorm, LayerNorm, PairNorm
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_add, scatter_mean

MENAGERIE_ZOO = "vendored-pytorch"


# --- deeph/from_se3_transformer/representations.py (verbatim spherical-harmonics basis) ---
def semifactorial(x):
    """Compute the semifactorial function x!!."""
    y = 1.0
    for n in range(x, 1, -2):
        y *= n
    return y


def pochhammer(x, k):
    """Compute the pochhammer symbol (x)_k."""
    xf = float(x)
    for n in range(x + 1, x + k):
        xf *= n
    return xf


def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase."""
    m_abs = abs(m)
    if m_abs > l:
        return torch.zeros_like(x)

    yold = ((-1) ** m_abs * semifactorial(2 * m_abs - 1)) * torch.pow(1 - x * x, m_abs / 2)

    if m_abs != l:
        y = x * (2 * m_abs + 1) * yold
    else:
        y = yold

    for i in range(m_abs + 2, l + 1):
        tmp = y
        y = ((2 * i - 1) / (i - m_abs)) * x * y
        y -= ((i + m_abs - 1) / (i - m_abs)) * yold
        yold = tmp

    if m < 0:
        y *= (-1) ** m / pochhammer(l + m + 1, -2 * m)

    return y


def tesseral_harmonics(l, m, theta=0.0, phi=0.0):
    """Tesseral spherical harmonic with Condon-Shortley phase."""
    assert abs(m) <= l, "absolute value of order m must be <= degree l"

    N = np.sqrt((2 * l + 1) / (4 * np.pi))
    leg = lpmv(l, abs(m), torch.cos(theta))
    if m == 0:
        return N * leg
    elif m > 0:
        Y = torch.cos(m * phi) * leg
    else:
        Y = torch.sin(abs(m) * phi) * leg
    N *= np.sqrt(2.0 / pochhammer(l - abs(m) + 1, 2 * abs(m)))
    Y *= N
    return Y


class SphericalHarmonics:
    def __init__(self):
        self.leg = {}

    def clear(self):
        self.leg = {}

    def negative_lpmv(self, l, m, y):
        """Compute negative order coefficients"""
        if m < 0:
            y *= (-1) ** m / pochhammer(l + m + 1, -2 * m)
        return y

    def lpmv(self, l, m, x):
        """Associated Legendre function including Condon-Shortley phase."""
        m_abs = abs(m)
        if (l, m) in self.leg:
            return self.leg[(l, m)]
        elif m_abs > l:
            return None
        elif l == 0:
            self.leg[(l, m)] = torch.ones_like(x)
            return self.leg[(l, m)]

        if m_abs == l:
            y = (-1) ** m_abs * semifactorial(2 * m_abs - 1)
            y *= torch.pow(1 - x * x, m_abs / 2)
            self.leg[(l, m)] = self.negative_lpmv(l, m, y)
            return self.leg[(l, m)]
        else:
            self.lpmv(l - 1, m, x)

        y = ((2 * l - 1) / (l - m_abs)) * x * self.lpmv(l - 1, m_abs, x)
        if l - m_abs > 1:
            y -= ((l + m_abs - 1) / (l - m_abs)) * self.leg[(l - 2, m_abs)]

        if m < 0:
            y = self.negative_lpmv(l, m, y)
        self.leg[(l, m)] = y

        return self.leg[(l, m)]

    def get_element(self, l, m, theta, phi):
        """Tesseral spherical harmonic with Condon-Shortley phase."""
        assert abs(m) <= l, "absolute value of order m must be <= degree l"

        N = np.sqrt((2 * l + 1) / (4 * np.pi))
        leg = self.lpmv(l, abs(m), torch.cos(theta))
        if m == 0:
            return N * leg
        elif m > 0:
            Y = torch.cos(m * phi) * leg
        else:
            Y = torch.sin(abs(m) * phi) * leg
        N *= np.sqrt(2.0 / pochhammer(l - abs(m) + 1, 2 * abs(m)))
        Y *= N
        return Y

    def get(self, l, theta, phi, refresh=True):
        """Tesseral harmonic with Condon-Shortley phase."""
        results = []
        if refresh:
            self.clear()
        for m in range(-l, l + 1):
            results.append(self.get_element(l, m, theta, phi))
        return torch.stack(results, -1)


# --- deeph/from_schnetpack/acsf.py (verbatim Gaussian distance basis) ---
def gaussian_smearing(distances, offset, widths, centered=False):
    if not centered:
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances[..., None] - offset
    else:
        coeff = -0.5 / torch.pow(offset, 2)
        diff = distances[..., None]
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianBasis(nn.Module):
    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values."""
        return gaussian_smearing(distances, self.offsets, self.width, centered=self.centered)


# --- deeph/from_PyG_future/graph_norm.py (verbatim GraphNorm, arXiv:2009.03294) ---
class GraphNorm(torch.nn.Module):
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.Tensor(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        pyg_ones(self.weight)
        pyg_zeros(self.bias)
        pyg_ones(self.mean_scale)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_size = int(batch.max()) + 1

        mean = scatter_mean(x, batch, dim=0, dim_size=batch_size)[batch]
        out = x - mean * self.mean_scale
        var = scatter_mean(out.pow(2), batch, dim=0, dim_size=batch_size)
        std = (var + self.eps).sqrt()[batch]
        return self.weight * out / std + self.bias

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels})"


# --- deeph/from_PyG_future/diff_group_norm.py (verbatim DiffGroupNorm, arXiv:2006.06972) ---
class DiffGroupNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        groups,
        lamda=0.01,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups
        self.lamda = lamda

        self.lin = nn.Linear(in_channels, groups, bias=False)
        self.norm = nn.BatchNorm1d(groups * in_channels, eps, momentum, affine, track_running_stats)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        F_, G = self.in_channels, self.groups

        s = self.lin(x).softmax(dim=-1)  # [N, G]
        out = s.unsqueeze(-1) * x.unsqueeze(-2)  # [N, G, F]
        out = self.norm(out.view(-1, G * F_)).view(-1, G, F_).sum(-2)  # [N, F]

        return x + self.lamda * out

    def __repr__(self):
        return "{}({}, groups={})".format(self.__class__.__name__, self.in_channels, self.groups)


# --- deeph/from_HermNet/rmnet.py (verbatim PAINN-style RBF/cutoff/activation helpers) ---
_eps = 1e-3


class RBF(nn.Module):
    """Radial basis function. A modified version of feature engineering in DimeNet, used in PAINN."""

    def __init__(self, rc: float, l: int):
        super().__init__()
        self.rc = rc
        self.l = l

    def forward(self, x: Tensor):
        ls = torch.arange(1, self.l + 1).float().to(x.device)
        norm = torch.sqrt((x**2).sum(dim=-1) + _eps).unsqueeze(-1)
        return torch.sin(np.pi / self.rc * norm @ ls.unsqueeze(0)) / norm


class cosine_cutoff(nn.Module):
    """Cutoff function in https://aip.scitation.org/doi/pdf/10.1063/1.3553717."""

    def __init__(self, rc: float):
        super().__init__()
        self.rc = rc

    def forward(self, x: Tensor):
        norm = torch.norm(x, dim=-1, keepdim=True) + _eps
        return 0.5 * (torch.cos(np.pi * norm / self.rc) + 1)


class ShiftedSoftplus(nn.Module):
    def __init__(self, beta=1, shift=2, threshold=20):
        super().__init__()

        self.shift = shift
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, inputs):
        return self.softplus(inputs) - np.log(float(self.shift))


# --- deeph/model.py (verbatim architecture) ---
class ExpBernsteinBasis(nn.Module):
    def __init__(self, K, gamma, cutoff, trainable=True):
        super().__init__()
        self.K = K
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.gamma = torch.tensor(gamma)
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("comb_k", torch.Tensor(comb(K - 1, np.arange(K))))

    def forward(self, distances):
        f_zero = torch.zeros_like(distances)
        f_cut = torch.where(
            distances < self.cutoff,
            torch.exp(-(distances**2) / (self.cutoff**2 - distances**2)),
            f_zero,
        )
        x = torch.exp(-self.gamma * distances)
        out = []
        for k in range(self.K):
            out.append((x**k) * ((1 - x) ** (self.K - 1 - k)))
        out = torch.stack(out, dim=-1)
        out = out * self.comb_k[None, :] * f_cut[:, None]
        return out


def get_spherical_from_cartesian(cartesian, cartesian_x=1, cartesian_y=2, cartesian_z=0):
    spherical = torch.zeros_like(cartesian[..., 0:2])
    r_xy = cartesian[..., cartesian_x] ** 2 + cartesian[..., cartesian_y] ** 2
    spherical[..., 0] = torch.atan2(torch.sqrt(r_xy), cartesian[..., cartesian_z])
    spherical[..., 1] = torch.atan2(cartesian[..., cartesian_y], cartesian[..., cartesian_x])
    return spherical


class SphericalHarmonicsBasis(nn.Module):
    def __init__(self, num_l=5):
        super().__init__()
        self.num_l = num_l

    def forward(self, edge_attr):
        r_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        r_vec_sp = get_spherical_from_cartesian(r_vec)
        sph_harm_func = SphericalHarmonics()

        angular_expansion = []
        for l in range(self.num_l):
            angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
        angular_expansion = torch.cat(angular_expansion, dim=-1)

        return angular_expansion


class CGConv(MessagePassing):
    """Extended from pytorch_geometric's CGConv (MIT License, Matthias Fey)."""

    def __init__(
        self,
        channels: Union[int, Tuple[int, int]],
        dim: int = 0,
        aggr: str = "add",
        normalization: str = None,
        bias: bool = True,
        if_exp: bool = False,
        **kwargs,
    ):
        super().__init__(aggr=aggr, flow="source_to_target", **kwargs)
        self.channels = channels
        self.dim = dim
        self.normalization = normalization
        self.if_exp = if_exp

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = nn.Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = nn.Linear(sum(channels) + dim, channels[1], bias=bias)
        if self.normalization == "BatchNorm":
            self.bn = nn.BatchNorm1d(channels[1], track_running_stats=True)
        elif self.normalization == "LayerNorm":
            self.ln = LayerNorm(channels[1])
        elif self.normalization == "PairNorm":
            self.pn = PairNorm(channels[1])
        elif self.normalization == "InstanceNorm":
            self.instance_norm = InstanceNorm(channels[1])
        elif self.normalization == "GraphNorm":
            self.gn = GraphNorm(channels[1])
        elif self.normalization == "DiffGroupNorm":
            self.group_norm = DiffGroupNorm(channels[1], 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError(f"Unknown normalization function: {normalization}")

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.normalization == "BatchNorm":
            self.bn.reset_parameters()

    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor,
        batch,
        distance,
        size: Size = None,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, distance=distance, size=size)
        if self.normalization == "BatchNorm":
            out = self.bn(out)
        elif self.normalization == "LayerNorm":
            out = self.ln(out, batch)
        elif self.normalization == "PairNorm":
            out = self.pn(out, batch)
        elif self.normalization == "InstanceNorm":
            out = self.instance_norm(out, batch)
        elif self.normalization == "GraphNorm":
            out = self.gn(out, batch)
        elif self.normalization == "DiffGroupNorm":
            out = self.group_norm(out)
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor, distance) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-(distance**n) / sigma**n / 2).view(-1, 1)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.channels}, dim={self.dim})"


class GAT_Crystal(MessagePassing):
    def __init__(
        self,
        in_features,
        out_features,
        edge_dim,
        heads,
        concat=False,
        normalization: str = None,
        dropout=0,
        bias=True,
        **kwargs,
    ):
        super().__init__(node_dim=0, aggr="add", flow="target_to_source", **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.neg_slope = 0.2
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(heads)
        self.W = nn.Parameter(torch.Tensor(in_features + edge_dim, heads * out_features))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.normalization = normalization
        if self.normalization == "BatchNorm":
            self.bn = nn.BatchNorm1d(out_features, track_running_stats=True)
        elif self.normalization == "LayerNorm":
            self.ln = LayerNorm(out_features)
        elif self.normalization == "PairNorm":
            self.pn = PairNorm(out_features)
        elif self.normalization == "InstanceNorm":
            self.instance_norm = InstanceNorm(out_features)
        elif self.normalization == "GraphNorm":
            self.gn = GraphNorm(out_features)
        elif self.normalization == "DiffGroupNorm":
            self.group_norm = DiffGroupNorm(out_features, 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError(f"Unknown normalization function: {normalization}")

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, batch, distance):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.normalization == "BatchNorm":
            out = self.bn(out)
        elif self.normalization == "LayerNorm":
            out = self.ln(out, batch)
        elif self.normalization == "PairNorm":
            out = self.pn(out, batch)
        elif self.normalization == "InstanceNorm":
            out = self.instance_norm(out, batch)
        elif self.normalization == "GraphNorm":
            out = self.gn(out, batch)
        elif self.normalization == "DiffGroupNorm":
            out = self.group_norm(out)
        return out

    def message(self, edge_index_i, x_i, x_j, size_i, index, ptr: OptTensor, edge_attr):
        x_i = torch.cat([x_i, edge_attr], dim=-1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        x_i = F.softplus(torch.matmul(x_i, self.W))
        x_j = F.softplus(torch.matmul(x_j, self.W))
        x_i = x_i.view(-1, self.heads, self.out_features)
        x_j = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1))
        alpha = F.softplus(self.bn1(alpha))

        alpha = softmax(alpha, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out, x):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class PaninnNodeFea:
    def __init__(self, node_fea_s, node_fea_v=None):
        self.node_fea_s = node_fea_s
        if node_fea_v is None:
            self.node_fea_v = torch.zeros(
                node_fea_s.shape[0],
                node_fea_s.shape[1],
                3,
                dtype=node_fea_s.dtype,
                device=node_fea_s.device,
            )
        else:
            self.node_fea_v = node_fea_v

    def __add__(self, other):
        return PaninnNodeFea(self.node_fea_s + other.node_fea_s, self.node_fea_v + other.node_fea_v)


class PAINN(nn.Module):
    def __init__(self, in_features, edge_dim, rc: float, l: int, normalization):
        super().__init__()
        self.ms1 = nn.Linear(in_features, in_features)
        self.ssp = ShiftedSoftplus()
        self.ms2 = nn.Linear(in_features, in_features * 3)

        self.rbf = RBF(rc, l)
        self.mv = nn.Linear(l, in_features * 3)
        self.fc = cosine_cutoff(rc)

        self.us1 = nn.Linear(in_features * 2, in_features)
        self.us2 = nn.Linear(in_features, in_features * 3)

        self.normalization = normalization
        if self.normalization == "BatchNorm":
            self.bn = nn.BatchNorm1d(in_features, track_running_stats=True)
        elif self.normalization == "LayerNorm":
            self.ln = LayerNorm(in_features)
        elif self.normalization == "PairNorm":
            self.pn = PairNorm(in_features)
        elif self.normalization == "InstanceNorm":
            self.instance_norm = InstanceNorm(in_features)
        elif self.normalization == "GraphNorm":
            self.gn = GraphNorm(in_features)
        elif self.normalization == "DiffGroupNorm":
            self.group_norm = DiffGroupNorm(in_features, 128)
        elif self.normalization is None or self.normalization == "None":
            pass
        else:
            raise ValueError(f"Unknown normalization function: {normalization}")

    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor,
        batch,
        edge_vec,
    ) -> torch.Tensor:
        r = torch.sqrt((edge_vec**2).sum(dim=-1) + _eps).unsqueeze(-1)
        sj = x.node_fea_s[edge_index[1, :]]
        vj = x.node_fea_v[edge_index[1, :]]

        phi = self.ms2(self.ssp(self.ms1(sj)))
        w = self.fc(r) * self.mv(self.rbf(r))
        v_, s_, r_ = torch.chunk(phi * w, 3, dim=-1)

        ds_update = s_
        dv_update = vj * v_.unsqueeze(-1) + r_.unsqueeze(-1) * (edge_vec / r).unsqueeze(1)

        ds = scatter(ds_update, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce="mean")
        dv = scatter(dv_update, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce="mean")
        x = x + PaninnNodeFea(ds, dv)

        sj = x.node_fea_s[edge_index[1, :]]
        vj = x.node_fea_v[edge_index[1, :]]
        norm = torch.sqrt((vj**2).sum(dim=-1) + _eps)
        s = torch.cat([norm, sj], dim=-1)
        sj = self.us2(self.ssp(self.us1(s)))

        uv = scatter(vj, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce="mean")
        norm = torch.sqrt((uv**2).sum(dim=-1) + _eps).unsqueeze(-1)
        s_ = scatter(sj, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce="mean")
        avv, asv, ass = torch.chunk(s_, 3, dim=-1)

        ds = ((uv / norm) ** 2).sum(dim=-1) * asv + ass
        dv = uv * avv.unsqueeze(-1)

        if self.normalization == "BatchNorm":
            ds = self.bn(ds)
        elif self.normalization == "LayerNorm":
            ds = self.ln(ds, batch)
        elif self.normalization == "PairNorm":
            ds = self.pn(ds, batch)
        elif self.normalization == "InstanceNorm":
            ds = self.instance_norm(ds, batch)
        elif self.normalization == "GraphNorm":
            ds = self.gn(ds, batch)
        elif self.normalization == "DiffGroupNorm":
            ds = self.group_norm(ds)

        x = x + PaninnNodeFea(ds, dv)

        return x


class MPLayer(nn.Module):
    def __init__(
        self,
        in_atom_fea_len,
        in_edge_fea_len,
        out_edge_fea_len,
        if_exp,
        if_edge_update,
        normalization,
        atom_update_net,
        gauss_stop,
        output_layer=False,
    ):
        super().__init__()
        if atom_update_net == "CGConv":
            self.cgconv = CGConv(
                channels=in_atom_fea_len,
                dim=in_edge_fea_len,
                aggr="add",
                normalization=normalization,
                if_exp=if_exp,
            )
        elif atom_update_net == "GAT":
            self.cgconv = GAT_Crystal(
                in_features=in_atom_fea_len,
                out_features=in_atom_fea_len,
                edge_dim=in_edge_fea_len,
                heads=3,
                normalization=normalization,
            )
        elif atom_update_net == "PAINN":
            self.cgconv = PAINN(
                in_features=in_atom_fea_len,
                edge_dim=in_edge_fea_len,
                rc=gauss_stop,
                l=64,
                normalization=normalization,
            )

        self.if_edge_update = if_edge_update
        self.atom_update_net = atom_update_net
        if if_edge_update:
            if output_layer:
                self.e_lin = nn.Sequential(
                    nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                    nn.SiLU(),
                    nn.Linear(128, out_edge_fea_len),
                )
            else:
                self.e_lin = nn.Sequential(
                    nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                    nn.SiLU(),
                    nn.Linear(128, out_edge_fea_len),
                    nn.SiLU(),
                )

    def forward(self, atom_fea, edge_idx, edge_fea, batch, distance, edge_vec):
        if self.atom_update_net == "PAINN":
            atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, edge_vec)
            atom_fea_s = atom_fea.node_fea_s
        else:
            atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance)
            atom_fea_s = atom_fea
        if self.if_edge_update:
            row, col = edge_idx
            edge_fea = self.e_lin(torch.cat([atom_fea_s[row], atom_fea_s[col], edge_fea], dim=-1))
            return atom_fea, edge_fea
        else:
            return atom_fea


class LCMPLayer(nn.Module):
    def __init__(
        self,
        in_atom_fea_len,
        in_edge_fea_len,
        out_edge_fea_len,
        num_l,
        normalization: str = None,
        bias: bool = True,
        if_exp: bool = False,
    ):
        super().__init__()
        self.in_atom_fea_len = in_atom_fea_len
        self.normalization = normalization
        self.if_exp = if_exp

        self.lin_f = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)
        self.lin_s = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)
        self.bn = nn.BatchNorm1d(in_atom_fea_len, track_running_stats=True)

        self.e_lin = nn.Sequential(
            nn.Linear(in_edge_fea_len + in_atom_fea_len * 2 - num_l**2, 128),
            nn.SiLU(),
            nn.Linear(128, out_edge_fea_len),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.normalization == "BatchNorm":
            self.bn.reset_parameters()

    def forward(
        self,
        atom_fea,
        edge_fea,
        sub_atom_idx,
        sub_edge_idx,
        sub_edge_ang,
        sub_index,
        distance,
        huge_structure,
        output_final_layer_neuron,
    ):
        if huge_structure:
            sub_graph_batch_num = 8

            sub_graph_num = sub_atom_idx.shape[0]
            sub_graph_batch_size = ceil(sub_graph_num / sub_graph_batch_num)

            num_edge = edge_fea.shape[0]
            vf_update = (
                torch.zeros((num_edge * 2, self.in_atom_fea_len))
                .type(torch.get_default_dtype())
                .to(atom_fea.device)
            )
            for sub_graph_batch_index in range(sub_graph_batch_num):
                if sub_graph_batch_index == sub_graph_batch_num - 1:
                    sub_graph_idx = slice(
                        sub_graph_batch_size * sub_graph_batch_index, sub_graph_num
                    )
                else:
                    sub_graph_idx = slice(
                        sub_graph_batch_size * sub_graph_batch_index,
                        sub_graph_batch_size * (sub_graph_batch_index + 1),
                    )

                sub_atom_idx_batch = sub_atom_idx[sub_graph_idx]
                sub_edge_idx_batch = sub_edge_idx[sub_graph_idx]
                sub_edge_ang_batch = sub_edge_ang[sub_graph_idx]
                sub_index_batch = sub_index[sub_graph_idx]

                z = torch.cat(
                    [
                        atom_fea[sub_atom_idx_batch][:, 0, :],
                        atom_fea[sub_atom_idx_batch][:, 1, :],
                        edge_fea[sub_edge_idx_batch],
                        sub_edge_ang_batch,
                    ],
                    dim=-1,
                )
                out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

                if self.if_exp:
                    sigma = 3
                    n = 2
                    out = out * torch.exp(-(distance[sub_edge_idx_batch] ** n) / sigma**n / 2).view(
                        -1, 1
                    )

                vf_update += scatter_add(out, sub_index_batch, dim=0, dim_size=num_edge * 2)

            if self.normalization == "BatchNorm":
                vf_update = self.bn(vf_update)
            vf_update = vf_update.reshape(num_edge, 2, -1)
            if output_final_layer_neuron != "":
                final_layer_neuron = (
                    torch.cat([vf_update[:, 0, :], vf_update[:, 1, :], edge_fea], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                np.save(
                    os.path.join(output_final_layer_neuron, "final_layer_neuron.npy"),
                    final_layer_neuron,
                )
            out = self.e_lin(torch.cat([vf_update[:, 0, :], vf_update[:, 1, :], edge_fea], dim=-1))

            return out

        num_edge = edge_fea.shape[0]
        z = torch.cat(
            [
                atom_fea[sub_atom_idx][:, 0, :],
                atom_fea[sub_atom_idx][:, 1, :],
                edge_fea[sub_edge_idx],
                sub_edge_ang,
            ],
            dim=-1,
        )
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-(distance[sub_edge_idx] ** n) / sigma**n / 2).view(-1, 1)

        out = scatter_add(out, sub_index, dim=0)
        if self.normalization == "BatchNorm":
            out = self.bn(out)
        out = out.reshape(num_edge, 2, -1)
        if output_final_layer_neuron != "":
            final_layer_neuron = (
                torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1).detach().cpu().numpy()
            )
            np.save(
                os.path.join(output_final_layer_neuron, "final_layer_neuron.npy"),
                final_layer_neuron,
            )
        out = self.e_lin(torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1))
        return out


class MultipleLinear(nn.Module):
    def __init__(
        self, num_linear: int, in_fea_len: int, out_fea_len: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.num_linear = num_linear
        self.out_fea_len = out_fea_len
        self.weight = nn.Parameter(torch.Tensor(num_linear, in_fea_len, out_fea_len))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, out_fea_len))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, batch_edge: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(input, self.weight)

        if self.bias is not None:
            output += self.bias[:, None, :]
        return output


class HGNN(nn.Module):
    """Hamiltonian Graph Neural Network -- the real DeepH entry point (deeph/model.py)."""

    def __init__(
        self,
        num_species,
        in_atom_fea_len,
        in_edge_fea_len,
        num_orbital,
        distance_expansion,
        gauss_stop,
        if_exp,
        if_MultipleLinear,
        if_edge_update,
        if_lcmp,
        normalization,
        atom_update_net,
        separate_onsite,
        trainable_gaussians,
        type_affine,
        num_l=5,
    ):
        super().__init__()
        self.num_species = num_species
        self.embed = nn.Embedding(num_species + 5, in_atom_fea_len)

        if type_affine:
            self.type_affine = nn.Embedding(
                num_species**2,
                2,
                _weight=torch.stack(
                    [torch.ones(num_species**2), torch.zeros(num_species**2)], dim=-1
                ),
            )
        else:
            self.type_affine = None

        if if_edge_update or (if_edge_update is False and if_lcmp is False):
            distance_expansion_len = in_edge_fea_len
        else:
            distance_expansion_len = in_edge_fea_len - num_l**2
        if distance_expansion == "GaussianBasis":
            self.distance_expansion = GaussianBasis(
                0.0, gauss_stop, distance_expansion_len, trainable=trainable_gaussians
            )
        elif distance_expansion == "BesselBasis":
            self.distance_expansion = BesselBasisLayer(
                distance_expansion_len, gauss_stop, envelope_exponent=5
            )
        elif distance_expansion == "ExpBernsteinBasis":
            self.distance_expansion = ExpBernsteinBasis(
                K=distance_expansion_len, gamma=0.5, cutoff=gauss_stop, trainable=True
            )
        else:
            raise ValueError(f"Unknown distance expansion function: {distance_expansion}")

        self.if_MultipleLinear = if_MultipleLinear
        self.if_edge_update = if_edge_update
        self.if_lcmp = if_lcmp
        self.atom_update_net = atom_update_net
        self.separate_onsite = separate_onsite

        if if_lcmp is True:
            mp_output_edge_fea_len = in_edge_fea_len - num_l**2
        else:
            assert if_MultipleLinear is False
            mp_output_edge_fea_len = in_edge_fea_len

        if if_edge_update is True:
            self.mp1 = MPLayer(
                in_atom_fea_len,
                in_edge_fea_len,
                in_edge_fea_len,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp2 = MPLayer(
                in_atom_fea_len,
                in_edge_fea_len,
                in_edge_fea_len,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp3 = MPLayer(
                in_atom_fea_len,
                in_edge_fea_len,
                in_edge_fea_len,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp4 = MPLayer(
                in_atom_fea_len,
                in_edge_fea_len,
                in_edge_fea_len,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp5 = MPLayer(
                in_atom_fea_len,
                in_edge_fea_len,
                mp_output_edge_fea_len,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
        else:
            self.mp1 = MPLayer(
                in_atom_fea_len,
                distance_expansion_len,
                None,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp2 = MPLayer(
                in_atom_fea_len,
                distance_expansion_len,
                None,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp3 = MPLayer(
                in_atom_fea_len,
                distance_expansion_len,
                None,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp4 = MPLayer(
                in_atom_fea_len,
                distance_expansion_len,
                None,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )
            self.mp5 = MPLayer(
                in_atom_fea_len,
                distance_expansion_len,
                None,
                if_exp,
                if_edge_update,
                normalization,
                atom_update_net,
                gauss_stop,
            )

        if if_lcmp is True:
            if self.if_MultipleLinear is True:
                self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
                self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
                self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
            else:
                self.lcmp = LCMPLayer(
                    in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp
                )
        else:
            self.mp_output = MPLayer(
                in_atom_fea_len,
                in_edge_fea_len,
                num_orbital,
                if_exp,
                if_edge_update=True,
                normalization=normalization,
                atom_update_net=atom_update_net,
                gauss_stop=gauss_stop,
                output_layer=True,
            )

    def forward(
        self,
        atom_attr,
        edge_idx,
        edge_attr,
        batch,
        sub_atom_idx=None,
        sub_edge_idx=None,
        sub_edge_ang=None,
        sub_index=None,
        huge_structure=False,
        output_final_layer_neuron="",
    ):
        batch_edge = batch[edge_idx[0]]
        atom_fea0 = self.embed(atom_attr)
        distance = edge_attr[:, 0]
        edge_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        if self.type_affine is None:
            edge_fea0 = self.distance_expansion(distance)
        else:
            affine_coeff = self.type_affine(
                self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]]
            )
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])
        if self.atom_update_net == "PAINN":
            atom_fea0 = PaninnNodeFea(atom_fea0)

        if self.if_edge_update is True:
            atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            if self.if_lcmp is True:
                if self.atom_update_net == "PAINN":
                    atom_fea_s = atom_fea.node_fea_s
                else:
                    atom_fea_s = atom_fea
                out = self.lcmp(
                    atom_fea_s,
                    edge_fea,
                    sub_atom_idx,
                    sub_edge_idx,
                    sub_edge_ang,
                    sub_index,
                    distance,
                    huge_structure,
                    output_final_layer_neuron,
                )
            else:
                atom_fea, edge_fea = self.mp_output(
                    atom_fea, edge_idx, edge_fea, batch, distance, edge_vec
                )
                out = edge_fea
        else:
            atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp4(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            if self.atom_update_net == "PAINN":
                atom_fea_s = atom_fea.node_fea_s
            else:
                atom_fea_s = atom_fea
            if self.if_lcmp is True:
                out = self.lcmp(
                    atom_fea_s,
                    edge_fea0,
                    sub_atom_idx,
                    sub_edge_idx,
                    sub_edge_ang,
                    sub_index,
                    distance,
                    huge_structure,
                    output_final_layer_neuron,
                )
            else:
                atom_fea, edge_fea = self.mp_output(
                    atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec
                )
                out = edge_fea

        if self.if_MultipleLinear is True:
            out = self.multiple_linear1(F.silu(out), batch_edge)
            out = self.multiple_linear2(F.silu(out), batch_edge)
            out = out.T

        return out


def build_deeph():
    # Tiny menagerie-scale HGNN config using the real class's simpler (but fully
    # legitimate, code-supported) if_lcmp=False path -- same MPLayer/CGConv message-
    # passing stack as the repo's default if_lcmp=True config, just skipping the extra
    # local-coordinate-message-passing sub-graph readout stage (mp_output edge-readout
    # head instead of LCMPLayer). atom_update_net="CGConv" is kernel.py's own fallback
    # default. Real hyperparameter names/roles preserved; widths/depth shrunk.
    return HGNN(
        num_species=10,
        in_atom_fea_len=16,
        in_edge_fea_len=16,
        num_orbital=4,
        distance_expansion="GaussianBasis",
        gauss_stop=6.0,
        if_exp=True,
        if_MultipleLinear=False,
        if_edge_update=True,
        if_lcmp=False,
        normalization="LayerNorm",
        atom_update_net="CGConv",
        separate_onsite=False,
        trainable_gaussians=False,
        type_affine=False,
        num_l=5,
    )


def example_input_deeph():
    # Small synthetic crystal-graph batch matching HGNN.forward's real (atom_attr,
    # edge_idx, edge_attr, batch) signature: edge_attr columns are
    # [distance, sender_xyz(3), receiver_xyz(3)] per deeph/model.py's own slicing
    # (edge_attr[:, 1:4] - edge_attr[:, 4:7] = edge_vec).
    torch.manual_seed(0)
    num_atoms = 6
    num_species = 10

    atom_attr = torch.randint(0, num_species, (num_atoms,))
    edges = torch.tensor(
        [[i, j] for i in range(num_atoms) for j in range(num_atoms) if i != j], dtype=torch.long
    )
    edge_idx = edges.t().contiguous()

    sender_xyz = torch.rand(edge_idx.shape[1], 3) * 3.0
    receiver_xyz = torch.rand(edge_idx.shape[1], 3) * 3.0
    distance = torch.norm(sender_xyz - receiver_xyz, dim=-1, keepdim=True) + 0.1
    edge_attr = torch.cat([distance, sender_xyz, receiver_xyz], dim=-1)

    batch = torch.zeros(num_atoms, dtype=torch.long)

    return (atom_attr, edge_idx, edge_attr, batch)


MENAGERIE_ENTRIES = [
    (
        "DeepH",
        "build_deeph",
        "example_input_deeph",
        2022,
        MENAGERIE_ZOO,
    ),
]
