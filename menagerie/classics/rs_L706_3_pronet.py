# SOURCE: vendored from divelab/DIG @ 21476b0 (dig-stable)
# Files: dig/threedgraph/method/pronet/features.py, dig/threedgraph/method/pronet/pronet.py
# ProNet: "Learning Protein Representations via Complete 3D Graph Networks" (hierarchical
# amino-acid / backbone / all-atom level protein 3D graph network). Vendored verbatim from
# the DIG library (divelab/DIG), the two source files flattened into one module (the
# package-relative `from .features import ...` in pronet.py is replaced by the merged
# in-module definitions below; no architecture changes). Uses torch_geometric's
# MessagePassing/radius_graph plus torch_scatter.scatter and torch_sparse.matmul, all of
# which are installed in this env.

# Based on the code from: https://github.com/TUM-DAML/gemnet_pytorch
# https://github.com/TUM-DAML/gemnet_pytorch/blob/master/gemnet/model/layers/basis_utils.py
# https://github.com/TUM-DAML/gemnet_pytorch/blob/master/gemnet/model/layers/basis_layers.py

import torch
import sympy as sym
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp


def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
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
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols("x")
    # j_i = (-x)^i * (1/x * d/dx)^î * sin(x)/x
    j = [sym.sin(x) / x]  # j_0
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    Returns:
        bess_basis: list
            Bessel basis formulas taking in a single argument x.
            Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = (
            1 / np.array(normalizer_tmp) ** 0.5
        )  # sqrt(2/(j_l+1)**2) , sqrt(1/c**3) not taken into account yet
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


def sph_harm_prefactor(l, m):  # noqa: E741 -- matches original repo variable naming (l = degree)
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.
    Parameters
    ----------
        l: int
            Degree of the spherical harmonic. l >= 0
        m: int
            Order of the spherical harmonic. -l <= m <= l
    Returns
    -------
        factor: float
    """
    # sqrt((2*l+1)/4*pi * (l-m)!/(l+m)! )
    return (
        (2 * l + 1) / (4 * np.pi) * np.math.factorial(l - abs(m)) / np.math.factorial(l + abs(m))
    ) ** 0.5


def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True):
    """Computes string formulas of the associated legendre polynomials up to degree L (excluded).
    Parameters
    ----------
        L: int
            Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.
    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).
    """
    # calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html
    z = sym.symbols("z")
    P_l_m = [
        [0] * (2 * l + 1) for l in range(L)
    ]  # for order l: -l <= m <= l  # noqa: E741 -- matches original repo variable naming (l = degree)

    P_l_m[0][0] = 1
    if L > 0:
        if zero_m_only:
            # m = 0
            P_l_m[1][0] = z
            for l in range(2, L):  # noqa: E741 -- matches original repo variable naming (l = degree)
                P_l_m[l][0] = sym.simplify(
                    ((2 * l - 1) * z * P_l_m[l - 1][0] - (l - 1) * P_l_m[l - 2][0]) / l
                )
            return P_l_m
        else:
            # for m >= 0
            for l in range(1, L):  # noqa: E741 -- matches original repo variable naming (l = degree)
                P_l_m[l][l] = sym.simplify(
                    (1 - 2 * l) * (1 - z**2) ** 0.5 * P_l_m[l - 1][l - 1]
                )  # P_00, P_11, P_22, P_33

            for m in range(0, L - 1):
                P_l_m[m + 1][m] = sym.simplify(
                    (2 * m + 1) * z * P_l_m[m][m]
                )  # P_10, P_21, P_32, P_43

            for l in range(2, L):  # noqa: E741 -- matches original repo variable naming (l = degree)
                for m in range(l - 1):  # P_20, P_30, P_31
                    P_l_m[l][m] = sym.simplify(
                        ((2 * l - 1) * z * P_l_m[l - 1][m] - (l + m - 1) * P_l_m[l - 2][m])
                        / (l - m)
                    )

            if not pos_m_only:
                # for m < 0: P_l(-m) = (-1)^m * (l-m)!/(l+m)! * P_lm
                for l in range(1, L):  # noqa: E741 -- matches original repo variable naming (l = degree)
                    for m in range(1, l + 1):  # P_1(-1), P_2(-1) P_2(-2)
                        P_l_m[l][-m] = sym.simplify(
                            (-1) ** m
                            * np.math.factorial(l - m)
                            / np.math.factorial(l + m)
                            * P_l_m[l][m]
                        )

            return P_l_m


def real_sph_harm(L, spherical_coordinates, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.
    Parameters
    ----------
        L: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        spherical_coordinates: bool
            - True: Expects the input of the formula strings to be phi and theta.
            - False: Expects the input of the formula strings to be x, y and z.
        zero_m_only: bool
            If True only calculate the harmonics where m=0.
    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l in range(L)]  # noqa: E741 -- matches original repo variable naming (l = degree)
    else:
        Y_l_m = [
            [0] * (2 * l + 1) for l in range(L)
        ]  # for order l: -l <= m <= l  # noqa: E741 -- matches original repo variable naming (l = degree)

    # convert expressions to spherical coordiantes
    if spherical_coordinates:
        # replace z by cos(theta)
        theta = sym.symbols("theta")
        for l in range(L):  # noqa: E741 -- matches original repo variable naming (l = degree)
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l in range(L):  # noqa: E741 -- matches original repo variable naming (l = degree)
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1, L):  # noqa: E741 -- matches original repo variable naming (l = degree)
            # m > 0
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(
                    2**0.5 * (-1) ** m * sph_harm_prefactor(l, m) * P_l_m[l][m] * sym.cos(m * phi)
                )
            # m < 0
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(
                    2**0.5 * (-1) ** m * sph_harm_prefactor(l, -m) * P_l_m[l][m] * sym.sin(m * phi)
                )

        # convert expressions to cartesian coordinates
        if not spherical_coordinates:
            # replace phi by atan2(y,x)
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l in range(L):  # noqa: E741 -- matches original repo variable naming (l = degree)
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m


class d_angle_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(d_angle_emb, self).__init__()
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
        for l in range(len(Y_lm)):  # noqa: E741 -- matches original repo variable naming (l = degree)
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


class d_theta_phi_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(d_theta_phi_emb, self).__init__()
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
        for l in range(len(Y_lm)):  # noqa: E741 -- matches original repo variable naming (l = degree)
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


"""
This is an implementation of ProNet model

"""

from torch_geometric.nn import inits, MessagePassing  # noqa: E402 -- import placement matches original repo file layout
from torch_geometric.nn import radius_graph  # noqa: E402 -- import placement matches original repo file layout


from torch_scatter import scatter  # noqa: E402 -- import placement matches original repo file layout
from torch_sparse import matmul  # noqa: E402 -- import placement matches original repo file layout

import torch  # noqa: E402 -- import placement matches original repo file layout
from torch import nn  # noqa: E402 -- import placement matches original repo file layout
from torch.nn import Embedding  # noqa: E402 -- import placement matches original repo file layout
import torch.nn.functional as F  # noqa: E402 -- import placement matches original repo file layout


num_aa_type = 26
num_side_chain_embs = 8
num_bb_embs = 6


def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):
    """
    A linear method encapsulation similar to PyG's

    Parameters
    ----------
    in_channels (int)
    out_channels (int)
    bias (int)
    weight_initializer (string): (glorot or zeros)
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer="glorot"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == "glorot":
            inits.glorot(self.weight)
        elif self.weight_initializer == "zeros":
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):
    """
    A layer with two linear modules

    Parameters
    ----------
    in_channels (int)
    middle_channels (int)
    out_channels (int)
    bias (bool)
    act (bool)
    """

    def __init__(self, in_channels, middle_channels, out_channels, bias=False, act=False):
        super(TwoLinear, self).__init__()
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


class EdgeGraphConv(MessagePassing):
    """
    Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

    The difference is that this module performs Hadamard product between node feature and edge feature

    Parameters
    ----------
    in_channels (int)
    out_channels (int)
    """

    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        output_channels,
        num_radial,
        num_spherical,
        num_layers,
        mid_emb,
        act=swish,
        num_pos_emb=16,
        dropout=0,
        level="allatom",
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin_feature0 = TwoLinear(num_radial * num_spherical**2, mid_emb, hidden_channels)
        if level == "aminoacid":
            self.lin_feature1 = TwoLinear(num_radial * num_spherical, mid_emb, hidden_channels)
        elif level == "backbone" or level == "allatom":
            self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(3 * hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))

        feature0 = self.lin_feature0(feature0)
        h0 = self.conv0(x_lin_1, edge_index, feature0)
        h0 = self.lin0(h0)
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)

        h = torch.cat((h0, h1, h2), 1)
        for lin in self.lins_cat:
            h = self.act(lin(h))

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h))
        h = self.final(h)
        return h


class ProNet(nn.Module):
    r"""
     The ProNet from the "Learning Protein Representations via Complete 3D Graph Networks" paper.

    Args:
        level: (str, optional): The level of protein representations. It could be :obj:`aminoacid`, obj:`backbone`, and :obj:`allatom`. (default: :obj:`aminoacid`)
        num_blocks (int, optional): Number of building blocks. (default: :obj:`4`)
        hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
        out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
        mid_emb (int, optional): Embedding size used for geometric features. (default: :obj:`64`)
        num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
        num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`)
        max_num_neighbors (int, optional): Max number of neighbors during graph construction. (default: :obj:`32`)
        int_emb_layers (int, optional): Number of embedding layers in the interaction block. (default: :obj:`3`)
        out_layers (int, optional): Number of layers for features after interaction blocks. (default: :obj:`2`)
        num_pos_emb (int, optional): Number of positional embeddings. (default: :obj:`16`)
        dropout (float, optional): Dropout. (default: :obj:`0`)
        data_augment_eachlayer (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to the node features before each interaction block. (default: :obj:`False`)
        euler_noise (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to Euler angles. (default: :obj:`False`)

    """

    def __init__(
        self,
        level="aminoacid",
        num_blocks=4,
        hidden_channels=128,
        out_channels=1,
        mid_emb=64,
        num_radial=6,
        num_spherical=2,
        cutoff=10.0,
        max_num_neighbors=32,
        int_emb_layers=3,
        out_layers=2,
        num_pos_emb=16,
        dropout=0,
        data_augment_eachlayer=False,
        euler_noise=False,
    ):
        super(ProNet, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.act = swish

        self.feature0 = d_theta_phi_emb(
            num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff
        )
        self.feature1 = d_angle_emb(
            num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff
        )

        if level == "aminoacid":
            self.embedding = Embedding(num_aa_type, hidden_channels)
        elif level == "backbone":
            self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs, hidden_channels)
        elif level == "allatom":
            self.embedding = torch.nn.Linear(
                num_aa_type + num_bb_embs + num_side_chain_embs, hidden_channels
            )
        else:
            print("No supported model!")

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_layers=int_emb_layers,
                    mid_emb=mid_emb,
                    act=self.act,
                    num_pos_emb=num_pos_emb,
                    dropout=dropout,
                    level=level,
                )
                for _ in range(num_blocks)
            ]
        )

        self.lins_out = torch.nn.ModuleList()
        for _ in range(out_layers - 1):
            self.lins_out.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self, edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, batch_data):
        z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs

        device = z.device

        if self.level == "aminoacid":
            x = self.embedding(z)
        elif self.level == "backbone":
            x = torch.cat(
                [torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs], dim=1
            )
            x = self.embedding(x)
        elif self.level == "allatom":
            x = torch.cat(
                [
                    torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()),
                    bb_embs,
                    side_chain_embs,
                ],
                dim=1,
            )
            x = self.embedding(x)
        else:
            print("No supported model!")

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)
        j, i = edge_index

        # Calculate distances.
        dist = (pos[i] - pos[j]).norm(dim=1)

        num_nodes = len(z)

        # Calculate angles theta and phi.
        refi0 = (i - 1) % num_nodes
        refi1 = (i + 1) % num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
        theta = torch.atan2(b, a)

        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / (
            (pos[refi0] - pos[i]).norm(dim=-1)
        )
        phi = torch.atan2(b, a)

        feature0 = self.feature0(dist, theta, phi)

        if self.level == "backbone" or self.level == "allatom":
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7

            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z)

            angle1 = torch.atan2(
                (torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1) / Or1_z_length,
                (Or1_x * Or1_Or2_N).sum(dim=-1),
            )
            angle2 = torch.atan2(
                torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1)
            )
            angle3 = torch.atan2(
                (torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1) / Or2_z_length,
                (Or1_Or2_N * Or2_x).sum(dim=-1),
            )

            if self.euler_noise:
                euler_noise = torch.clip(
                    torch.empty(3, len(angle1)).to(device).normal_(mean=0.0, std=0.025),
                    min=-0.1,
                    max=0.1,
                )
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]

            feature1 = torch.cat(
                (
                    self.feature1(dist, angle1),
                    self.feature1(dist, angle2),
                    self.feature1(dist, angle3),
                ),
                1,
            )

        elif self.level == "aminoacid":
            refi = (i - 1) % num_nodes

            refj0 = (j - 1) % num_nodes
            refj = (j - 1) % num_nodes
            refj1 = (j + 1) % num_nodes

            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]

            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i])
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j])
            a = (plane1 * plane2).sum(dim=-1)
            b = (torch.cross(plane1, plane2) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)

            feature1 = self.feature1(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            if self.data_augment_eachlayer:
                # add gaussian noise to features
                gaussian_noise = torch.clip(
                    torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1
                )
                x += gaussian_noise
            x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)

        y = scatter(x, batch, dim=0)

        for lin in self.lins_out:
            y = self.relu(lin(y))
            y = self.dropout(y)
        y = self.lin_out(y)
        return y

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


MENAGERIE_ZOO = "vendored-pytorch"


def build_pronet():
    import torch

    torch.manual_seed(0)
    # real constructor defaults: level='aminoacid', num_blocks=4, hidden_channels=128,
    # num_radial=6, num_spherical=2, cutoff=10.0 (see run_ProNet.py hyperparameters);
    # shrunk to tiny dims/blocks for a fast CPU trace, same aminoacid-level architecture.
    return ProNet(
        level="aminoacid",
        num_blocks=1,
        hidden_channels=16,
        out_channels=4,
        mid_emb=8,
        num_radial=3,
        num_spherical=2,
        cutoff=10.0,
        max_num_neighbors=8,
        int_emb_layers=2,
        out_layers=1,
        num_pos_emb=8,
        dropout=0.0,
    )


def example_input_pronet():
    import torch
    from torch_geometric.data import Data

    torch.manual_seed(0)
    # real input is a torch_geometric Data/Batch object with per-residue amino-acid type
    # (x), CA/N/C backbone coordinates, backbone/side-chain embedding placeholders, and a
    # batch-assignment vector -- built by the repo's dataset loaders (fold/EC benchmarks).
    n = 12
    x = torch.randint(0, 20, (n,))
    coords_ca = torch.randn(n, 3) * 5
    coords_n = coords_ca + torch.randn(n, 3) * 0.5
    coords_c = coords_ca + torch.randn(n, 3) * 0.5
    batch = torch.zeros(n, dtype=torch.long)
    data = Data(
        x=x,
        coords_ca=coords_ca,
        coords_n=coords_n,
        coords_c=coords_c,
        bb_embs=torch.zeros(n, 6),
        side_chain_embs=torch.zeros(n, 8),
        batch=batch,
    )
    return (data,)


MENAGERIE_ENTRIES = [
    ("ProNet", "build_pronet", "example_input_pronet", 2023, "SOURCE_AVAILABLE"),
]
