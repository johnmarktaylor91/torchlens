# SOURCE: vendored from YKQ98/Matformer @ main
#
# Matformer (Yan, Liu, Lin, Ji. 2022, NeurIPS, "Periodic Graph Transformers
# for Crystal Material Property Prediction") -- a periodic graph transformer
# for crystal-structure property prediction. `MatformerConv` (real file:
# matformer/models/transformer.py) is a custom `torch_geometric`
# `MessagePassing` layer combining multi-head key/query/value attention with
# an edge-feature-conditioned message-update gate; `Matformer` (real file:
# matformer/models/pyg_att.py) stacks 5 `MatformerConv` layers over an
# RBF-expanded-distance edge embedding, then reads out per-crystal scalar
# properties via mean-pool + MLP head.
#
# Vendored files (copied close to verbatim; only cosmetic import/config
# changes noted below):
#   https://raw.githubusercontent.com/YKQ98/Matformer/main/matformer/models/transformer.py
#   https://raw.githubusercontent.com/YKQ98/Matformer/main/matformer/models/utils.py  (RBFExpansion only)
#   https://raw.githubusercontent.com/YKQ98/Matformer/main/matformer/features.py       (angle_emb_mp + its
#       DimeNet-style spherical/Bessel basis helpers, ported verbatim from the same repo file)
#   https://raw.githubusercontent.com/YKQ98/Matformer/main/matformer/models/pyg_att.py (Matformer)
#
# Deviations (packaging only, NOT architecture): the real `Matformer.__init__`
# takes a `MatformerConfig(BaseSettings)` pydantic-v1-style settings object
# (`matformer/utils.py`, `from pydantic import BaseSettings as
# PydanticBaseSettings`) -- pydantic v1's `BaseSettings` was removed from
# core pydantic in v2 (moved to the separate `pydantic-settings` package,
# not installed here), so `MatformerConfig` is replaced with a plain
# `dataclasses.dataclass` holding the exact same field names/defaults; no
# architectural code changed. `matformer.models.utils.softmax`/`softmax_vec`
# (jit-scripted CSR/COO softmax helpers) are UNUSED by `MatformerConv.message`
# (it never calls them -- attention weights are produced via elementwise
# query*key, not the sparse-softmax helper) and are dropped; everything that
# IS reached by `Matformer.forward` -> `MatformerConv.forward/message` is
# vendored intact, including the (real-code) unused-but-constructed
# `self.sbf = angle_emb_mp(...)`, `self.edge_init`, `self.lattice_*` branches.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import sympy as sym
import torch
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy import special as sp
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import scatter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# matformer/features.py -- DimeNet-style spherical/Bessel basis functions
# (verbatim; only dependency is sympy/scipy/numpy/torch, all base-lib).
# ---------------------------------------------------------------------------


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    return zerosj


def spherical_bessel_formulas(n):
    x = sym.symbols("x")
    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(k, m):
    return (
        (2 * k + 1) * np.math.factorial(k - abs(m)) / (4 * np.pi * np.math.factorial(k + abs(m)))
    ) ** 0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    z = sym.symbols("z")
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(
                ((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0]) / j
            )
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify((2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] - (i + j - 1) * P_l_m[j - 2][i])
                        / (j - i)
                    )

    return P_l_m


def real_sph_harm(l, zero_m_only=False, spherical_coordinates=True):  # noqa: E741
    """Computes formula strings of the real part of the spherical harmonics up
    to order l (excluded)."""
    if not zero_m_only:
        x = sym.symbols("x")
        y = sym.symbols("y")
        S_m = [x * 0]
        C_m = [1 + 0 * x]
        for i in range(1, l):
            x = sym.symbols("x")
            y = sym.symbols("y")
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols("theta")
        z = sym.symbols("z")
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols("phi")
            for i in range(len(S_m)):
                S_m[i] = (
                    S_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )
            for i in range(len(C_m)):
                C_m[i] = (
                    C_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )

    Y_func_l_m = [["0"] * (2 * j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j]
                )
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j]
                )

    return Y_func_l_m


class angle_emb_mp(torch.nn.Module):
    def __init__(self, num_spherical=3, num_radial=30, cutoff=8.0, envelope_exponent=5):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x, sph1=sph1: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


# ---------------------------------------------------------------------------
# matformer/models/utils.py -- RBFExpansion (verbatim).
# ---------------------------------------------------------------------------


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))

        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


# ---------------------------------------------------------------------------
# matformer/models/transformer.py -- MatformerConv (verbatim).
# ---------------------------------------------------------------------------


class MatformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels)
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None
        )

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.concat:
            out = self.lin_concate(out)

        out = F.silu(self.bn(out))  # after norm and silu

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        return out

    def message(
        self,
        query_i: Tensor,
        key_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        value_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        query_i = torch.cat((query_i, query_i, query_i), dim=-1)
        key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
        alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(
            self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels))
        )
        out = self.msg_layer(out)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"


# ---------------------------------------------------------------------------
# matformer/models/pyg_att.py -- Matformer (verbatim, config plumbing only
# swapped from broken pydantic-v1 `BaseSettings` to a plain dataclass; see
# header note above).
# ---------------------------------------------------------------------------


class MatformerConfig:
    """Hyperparameter schema for the Matformer model (plain-class stand-in
    for the real repo's pydantic-v1 `BaseSettings`-derived `MatformerConfig`;
    same field names/defaults, no architectural change). A `dataclasses`
    `@dataclass` is deliberately avoided here: `dataclass()` looks up
    `sys.modules[cls.__module__]` at decoration time, which is `None` for a
    module loaded via `importlib.util.spec_from_file_location` before it is
    registered in `sys.modules` -- a plain `__init__` sidesteps that."""

    def __init__(
        self,
        name: str = "matformer",
        conv_layers: int = 5,
        edge_layers: int = 0,
        atom_input_features: int = 92,
        edge_features: int = 128,
        triplet_input_features: int = 40,
        node_features: int = 128,
        fc_layers: int = 1,
        fc_features: int = 128,
        output_features: int = 1,
        node_layer_head: int = 4,
        edge_layer_head: int = 4,
        nn_based: bool = False,
        link: str = "identity",
        zero_inflated: bool = False,
        use_angle: bool = False,
        angle_lattice: bool = False,
        classification: bool = False,
    ):
        self.name = name
        self.conv_layers = conv_layers
        self.edge_layers = edge_layers
        self.atom_input_features = atom_input_features
        self.edge_features = edge_features
        self.triplet_input_features = triplet_input_features
        self.node_features = node_features
        self.fc_layers = fc_layers
        self.fc_features = fc_features
        self.output_features = output_features
        self.node_layer_head = node_layer_head
        self.edge_layer_head = edge_layer_head
        self.nn_based = nn_based
        self.link = link
        self.zero_inflated = zero_inflated
        self.use_angle = use_angle
        self.angle_lattice = angle_lattice
        self.classification = classification


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, config: MatformerConfig = MatformerConfig()):
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.zero_inflated = config.zero_inflated
        self.atom_embedding = nn.Linear(config.atom_input_features, config.node_features)
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )
        self.angle_lattice = config.angle_lattice
        if self.angle_lattice:  # module not used
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features),
                nn.Linear(config.edge_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features),
            )

            self.lattice_angle = nn.Sequential(
                RBFExpansion(vmin=-1, vmax=1.0, bins=config.triplet_input_features),
                nn.Linear(config.triplet_input_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features),
            )

            self.lattice_emb = nn.Sequential(
                nn.Linear(config.node_features * 6, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features),
            )

            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(config.node_features * 2, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features),
            )

        self.edge_init = nn.Sequential(  # module not used
            nn.Linear(3 * config.node_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )

        self.sbf = angle_emb_mp(num_spherical=3, num_radial=40, cutoff=8.0)  # module not used

        self.angle_init_layers = nn.Sequential(  # module not used
            nn.Linear(120, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(
                    in_channels=config.node_features,
                    out_channels=config.node_features,
                    heads=config.node_layer_head,
                    edge_dim=config.node_features,
                )
                for _ in range(config.conv_layers)
            ]
        )

        self.edge_update_layers = nn.ModuleList(  # module not used
            [
                MatformerConv(
                    in_channels=config.node_features,
                    out_channels=config.node_features,
                    heads=config.edge_layer_head,
                    edge_dim=config.node_features,
                )
                for _ in range(config.edge_layers)
            ]
        )

        self.fc = nn.Sequential(nn.Linear(config.node_features, config.fc_features), nn.SiLU())
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(config.fc_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, data) -> torch.Tensor:
        data, ldata, lattice = data
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)

        edge_features = self.rbf(edge_feat)
        if self.angle_lattice:  # module not used
            lattice_len = torch.norm(lattice, dim=-1)
            lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128)
            cos1 = self.lattice_angle(
                torch.clamp(
                    torch.sum(lattice[:, 0, :] * lattice[:, 1, :], dim=-1)
                    / (torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 1, :], dim=-1)),
                    -1,
                    1,
                ).unsqueeze(-1)
            ).view(-1, 128)
            cos2 = self.lattice_angle(
                torch.clamp(
                    torch.sum(lattice[:, 0, :] * lattice[:, 2, :], dim=-1)
                    / (torch.norm(lattice[:, 0, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)),
                    -1,
                    1,
                ).unsqueeze(-1)
            ).view(-1, 128)
            cos3 = self.lattice_angle(
                torch.clamp(
                    torch.sum(lattice[:, 1, :] * lattice[:, 2, :], dim=-1)
                    / (torch.norm(lattice[:, 1, :], dim=-1) * torch.norm(lattice[:, 2, :], dim=-1)),
                    -1,
                    1,
                ).unsqueeze(-1)
            ).view(-1, 128)
            lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
            node_features = self.lattice_atom_emb(
                torch.cat((node_features, lattice_emb[data.batch]), dim=-1)
            )

        for att_layer in self.att_layers:
            node_features = att_layer(node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        if self.angle_lattice:
            features += lattice_emb

        features = self.fc(features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)
        if self.classification:
            out = self.softmax(out)

        return torch.squeeze(out)


# ---------------------------------------------------------------------------
# Staging build/example-input helpers.
# ---------------------------------------------------------------------------


# torch_geometric's `MessagePassing.__init__` (base class of `MatformerConv`)
# eagerly inspects `self.message`'s type hints via
# `sys.modules[cls.__module__].__dict__` (torch_geometric/inspector.py
# `Inspector._globals`). When this file is loaded standalone via
# `importlib.util.spec_from_file_location` (as opposed to a real package
# import), `__name__`/`__module__` is not yet registered in `sys.modules`,
# so that lookup raises `KeyError`. This is generic `torch_geometric`
# machinery unrelated to Matformer's architecture -- register this module
# under its own `__name__` defensively so direct-file validation works the
# same as a real package import.
def _ensure_module_registered():
    """torch_geometric's `MessagePassing.__init__` (base class of
    `MatformerConv`) eagerly inspects `self.message`'s type hints via
    `sys.modules[cls.__module__].__dict__`
    (torch_geometric/inspector.py `Inspector._globals`). When this file is
    loaded standalone via `importlib.util.spec_from_file_location` (as
    opposed to a real package import), `sys.modules[__name__]` is not
    registered by that loading path, so the lookup raises `KeyError`. This
    is generic `torch_geometric` machinery unrelated to Matformer's
    architecture -- self-register a module object exposing this file's
    current globals so direct-file validation behaves like a real package
    import. No-op once this file is `import`ed normally (module import
    always registers `sys.modules[__name__]` itself)."""
    import sys
    import types

    if __name__ in sys.modules:
        return
    stub = types.ModuleType(__name__)
    stub.__dict__.update(globals())
    sys.modules[__name__] = stub


_ensure_module_registered()


def build_matformer():
    """Builds the real Matformer model at tiny dims (real defaults are
    node_features=128, conv_layers=5, atom_input_features=92; here scaled
    down to node_features=8 / conv_layers=2 / atom_input_features=4 for a
    fast trace, `edge_layers=0` and `angle_lattice=False` match the real
    default config so the unused-in-forward submodules stay unused)."""
    config = MatformerConfig(
        name="matformer",
        conv_layers=2,
        edge_layers=0,
        atom_input_features=4,
        edge_features=8,
        triplet_input_features=8,
        node_features=8,
        fc_layers=1,
        fc_features=8,
        output_features=1,
        node_layer_head=2,
        edge_layer_head=2,
        link="identity",
        angle_lattice=False,
        classification=False,
    )
    return Matformer(config)


def example_input_matformer():
    """Builds a tiny real `torch_geometric`-style crystal graph batch: 6 atom
    nodes, a small ring of periodic-image edges, matching the
    `(data, ldata, lattice)` tuple `Matformer.forward` unpacks (`ldata`/
    `lattice` are only read when `angle_lattice=True`, so they are passed as
    placeholders here, exactly as `angle_lattice=False` leaves them
    unused)."""
    from types import SimpleNamespace

    n_nodes = 6
    n_edges = 12
    x = torch.randn(n_nodes, 4)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)
    edge_attr = torch.randn(n_edges, 3)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    data = SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    ldata = None
    lattice = None
    return ((data, ldata, lattice),)


MENAGERIE_ENTRIES = [
    ("Matformer", "build_matformer", "example_input_matformer", 2022, "vendored-pytorch"),
]
