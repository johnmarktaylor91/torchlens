# ruff: noqa: E741  (verbatim upstream uses single-letter `l` for spherical-harmonic degree)
# SOURCE: vendored from divelab/DIG @ dig-stable
# https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/method/comenet/comenet.py
# https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/method/comenet/features.py
#
# ComENet: "Towards Complete and Efficient Message Passing for 3D Molecular Graphs"
# (Wang et al., NeurIPS 2022, https://arxiv.org/abs/2206.08515). This file inlines the
# two upstream source files verbatim (only change: relative import `from .features import
# angle_emb, torsion_emb` collapsed since both live in this one module; base-library
# imports -- torch/torch_geometric/torch_cluster/torch_scatter/sympy/numpy/scipy -- are
# hoisted to the top). No architectural change: same torsion/angle spherical-Bessel basis,
# EdgeGraphConv + GraphNorm interaction blocks, and scatter-pooled energy readout as the
# real divelab/DIG implementation.
import math
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from scipy import special as sp
from scipy.optimize import brentq
from torch import Tensor, nn
from torch.nn import Embedding
from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm, inits
from torch_geometric.nn.models.schnet import GaussianSmearing  # noqa: F401  (upstream import, unused directly here but kept for parity)
from torch_scatter import scatter, scatter_min

try:
    import sympy as sym
except ImportError:
    sym = None

MENAGERIE_ZOO = "vendored-pytorch"


# --- dig/threedgraph/method/comenet/features.py (verbatim spherical-Bessel basis helpers) ---
def Jn(r, n):
    """numerical spherical bessel functions of order n"""
    return sp.spherical_jn(n, r)


def Jn_zeros(n, k):
    """Compute the first k zeros of the spherical bessel functions up to order n (excluded)"""
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
    """Computes the sympy formulas for the spherical bessel functions up to order n (excluded)"""
    x = sym.symbols("x")
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """Compute the sympy formulas for the normalized and rescaled spherical bessel functions."""
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


def sph_harm_prefactor(l, m):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m."""
    return (
        (2 * l + 1) / (4 * np.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m))
    ) ** 0.5


def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded)."""
    z = sym.symbols("z")
    P_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            P_l_m[1][0] = z
            for l in range(2, L):
                P_l_m[l][0] = sym.simplify(
                    ((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l
                )
            return P_l_m
        else:
            for l in range(1, L):
                P_l_m[l][l] = sym.simplify((1 - 2 * l) * (1 - z**2) ** 0.5 * P_l_m[l - 1][l - 1])

            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sym.simplify((2 * m + 1) * z * P_l_m[m][m])

            for l in range(2, L):
                for m in range(l - 1):
                    P_l_m[l][m] = sym.simplify(
                        ((2 * l - 1) * z * P_l_m[l - 1][m] - (l + m - 1) * P_l_m[l - 2][m])
                        / (l - m)
                    )

            if not pos_m_only:
                for l in range(1, L):
                    for m in range(1, l + 1):
                        P_l_m[l][-m] = sym.simplify(
                            (-1) ** m * math.factorial(l - m) / math.factorial(l + m) * P_l_m[l][m]
                        )

            return P_l_m


def real_sph_harm(L, spherical_coordinates, zero_m_only=True):
    """Computes formula strings of the real part of the spherical harmonics up to degree L."""
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [[0] * (2 * l + 1) for l in range(L)]

    if spherical_coordinates:
        theta = sym.symbols("theta")
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))

    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1, L):
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(
                    2**0.5 * (-1) ** m * sph_harm_prefactor(l, m) * P_l_m[l][m] * sym.cos(m * phi)
                )
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(
                    2**0.5 * (-1) ** m * sph_harm_prefactor(l, -m) * P_l_m[l][m] * sym.sin(m * phi)
                )

        if not spherical_coordinates:
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m


class angle_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(num_spherical, spherical_coordinates=True, zero_m_only=True)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0
        for l in range(len(Y_lm)):
            if l == 0:
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(lambda theta: torch.zeros_like(theta) + first_sph(theta))
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], bessel_formulas[l][n], modules))

    def forward(self, dist, angle):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf.view(-1, n, k) * sbf.view(-1, n, 1)).view(-1, n * k)
        return out


class torsion_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super().__init__()
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(num_spherical, spherical_coordinates=True, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        for l in range(len(Y_lm)):
            for m in range(len(Y_lm[l])):
                if l == 0:
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(
                        lambda theta, phi: torch.zeros_like(theta) + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(sym.lambdify([theta, phi], Y_lm[l][m], modules))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], bessel_formulas[l][j], modules))

        self.register_buffer("degreeInOrder", torch.arange(num_spherical) * 2 + 1, persistent=False)

    def forward(self, dist, theta, phi):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(theta, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        rbf = rbf.view((-1, n, k)).repeat_interleave(self.degreeInOrder, dim=1).view((-1, n**2 * k))
        sbf = sbf.repeat_interleave(k, dim=1)
        out = rbf * sbf
        return out


# --- dig/threedgraph/method/comenet/comenet.py (verbatim architecture) ---
def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        weight_initializer="glorot",
        bias_initializer="zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == "glorot":
                inits.glorot(self.weight)
            elif self.weight_initializer == "glorot_orthogonal":
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == "uniform":
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == "kaiming_uniform":
                inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer '{self.weight_initializer}' is not supported"
                )

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == "zeros":
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer '{self.bias_initializer}' is not supported"
                )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bias=False, act=False):
        super().__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(GraphConv):
    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        middle_channels,
        num_radial,
        num_spherical,
        num_layers,
        output_channels,
        act=swish,
    ):
        super().__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        self.lin_feature1 = TwoLayerLinear(
            num_radial * num_spherical**2, middle_channels, hidden_channels
        )
        self.lin_feature2 = TwoLayerLinear(
            num_radial * num_spherical, middle_channels, hidden_channels
        )

        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()
        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin_cat.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.lin(x))

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))

        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        return h


class ComENet(nn.Module):
    r"""The ComENet from "ComENet: Towards Complete and Efficient Message Passing for 3D
    Molecular Graphs" (Wang et al., NeurIPS 2022). Embeds atoms, builds a radius graph,
    computes complete local-frame torsion/angle/distance features via a spherical-Bessel
    basis, refines through EdgeGraphConv + GraphNorm interaction blocks, then scatter-pools
    to a per-graph molecular property.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: 8.0)
        num_layers (int, optional): Number of building blocks. (default: 4)
        hidden_channels (int, optional): Hidden embedding size. (default: 256)
        middle_channels (int, optional): Middle embedding size for the two layer linear block. (default: 256)
        out_channels (int, optional): Size of each output sample. (default: 1)
        num_radial (int, optional): Number of radial basis functions. (default: 3)
        num_spherical (int, optional): Number of spherical harmonics. (default: 2)
        num_output_layers (int, optional): Number of linear layers for the output blocks. (default: 3)
    """

    def __init__(
        self,
        cutoff=8.0,
        num_layers=4,
        hidden_channels=256,
        middle_channels=64,
        out_channels=1,
        num_radial=3,
        num_spherical=2,
        num_output_layers=3,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act

        self.feature1 = torsion_emb(
            num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff
        )
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.emb = EmbeddingBlock(hidden_channels, act)

        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def _forward(self, data):
        batch = data.batch
        z = data.z.long()
        pos = data.pos
        num_nodes = z.size(0)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        x = self.emb(z)

        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        n0 = n0[i]
        n1 = n1[i]

        n0_j = n0_j[j]
        n1_j = n1_j[j]

        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref],
        )

        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        for lin in self.lins:
            x = self.act(lin(x))
        x = self.lin_out(x)

        energy = scatter(x, batch, dim=0)
        return energy

    def forward(self, batch_data):
        return self._forward(batch_data)


def build_comenet():
    # Tiny menagerie-scale config: small hidden/middle widths, 2 interaction layers,
    # num_radial/num_spherical kept at real defaults (algorithm-required small constants).
    return ComENet(
        cutoff=8.0,
        num_layers=2,
        hidden_channels=16,
        middle_channels=8,
        out_channels=1,
        num_radial=3,
        num_spherical=2,
        num_output_layers=2,
    )


def example_input_comenet():
    torch.manual_seed(0)
    from torch_geometric.data import Data

    num_nodes = 10
    z = torch.randint(1, 10, (num_nodes,))
    pos = torch.randn(num_nodes, 3)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(z=z, pos=pos, batch=batch)
    return (data,)


MENAGERIE_ENTRIES = [
    (
        "ComENet",
        "build_comenet",
        "example_input_comenet",
        2022,
        MENAGERIE_ZOO,
    ),
]
