# SOURCE: vendored from https://github.com/divelab/AIRS @ main, path OpenMat/SyMat
# Files: OpenMat/SyMat/model/spherenet.py (SphereNetEncoder), OpenMat/SyMat/model/modules.py
# (emb, ResidualLayer, init, update_e, update_v, update_u), OpenMat/SyMat/model/features.py
# (Envelope, dist_emb, angle_emb, torsion_emb, and the sympy/scipy basis-function machinery
# they depend on), OpenMat/SyMat/utils/geometric_computing.py (xyz_to_dat),
# OpenMat/SyMat/utils/mat_utils.py (lattice_params_to_matrix_torch, get_pbc_distances) --
# all architecture classes and geometric-feature functions copied verbatim; only import
# plumbing (relative-package and repo-root `utils` imports collapsed into this single file)
# was changed. `torch_geometric.nn.acts.swish` no longer exists in the currently installed
# torch_geometric (it moved out of `nn.acts` in newer releases), so the one-line `swish`
# activation (`x * sigmoid(x)`) is inlined here rather than imported -- this is not an
# architectural change, just a shim for a relocated stdlib-style helper.
#
# SyMat (Luo, Wang, Liu, Ji 2023, "Towards Symmetry-Aware Generation of Periodic Materials",
# NeurIPS 2023) is a VAE for crystal structure generation whose encoder backbone is SphereNet
# (Liu et al. 2022, "Spherical Message Passing for 3D Molecular Graphs"), a spherical/torsional
# message-passing GNN over periodic-boundary-condition atomic graphs. This vendors that
# SphereNetEncoder backbone -- the full spherical-message-passing architecture (distance/angle/
# torsion embeddings -> triplet-indexed interaction blocks -> node/graph updates) -- exactly as
# used inside SyMat's `MatGen.encode()`. It is the genuinely novel, traceable core of the model;
# the rest of MatGen.forward()/CoordGen.forward() are training-loss functions that return scalar
# loss dicts (not network output tensors) and unconditionally move submodules to 'cuda' and spin
# up a `multiprocessing.Pool` at construction time (CoordGen.__init__), which are training-script
# / hardware concerns orthogonal to the architecture, not part of a traceable forward computation.
#
# The `xyz_to_dat` periodic-boundary-aware triplet/torsion index construction requires a real
# crystal batch (atom_types, edge_index, frac_coords, lattice lengths/angles, to_jimages,
# num_atoms, num_bonds) exactly as constructed by SyMat's own PyG data pipeline
# (utils/data_utils.py); we hand-build a minimal but shape-faithful synthetic periodic-crystal
# batch (a couple of small unit cells with a handful of atoms and periodic-image bonds) rather
# than depending on the real CIF-parsing dataset loader, which is a data-preparation concern
# outside the architecture.

from math import pi as PI
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from scipy import special as sp
from scipy.optimize import brentq
from torch import Tensor
from torch.nn import Embedding, Linear
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from torch_sparse import SparseTensor

import sympy as sym


def swish(x: Tensor) -> Tensor:
    """models/modules.py imports this from torch_geometric.nn.acts; inlined verbatim
    (x * sigmoid(x)) since that submodule was removed from newer torch_geometric releases."""
    return x * torch.sigmoid(x)


# ==================== utils/mat_utils.py (pure tensor math, verbatim) ====================
def lattice_params_to_matrix_torch(lengths: Tensor, angles: Tensor) -> Tensor:
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum("bi,bij->bj", coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[i_index] - pos[j_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, num_bonds, dim=0)
    offsets = torch.einsum("bi,bij->bj", to_jimages.float(), lattice_edges)
    distance_vectors = distance_vectors - offsets

    return pos, distance_vectors, offsets


# =============== utils/geometric_computing.py (triplet/torsion indices, verbatim) ===============
def xyz_to_dat(edge_index, num_nodes, num_edges, distance_vectors, use_torsion=False):
    """Compute the distance, angle, and torsion from geometric information."""
    j, i = edge_index  # j->i

    dist = distance_vectors.pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    edge2graph = torch.arange(len(num_edges), device=num_edges.device)
    edge2graph = edge2graph.repeat_interleave(num_edges, dim=0)
    num_triplets_per_graph = scatter(num_triplets, edge2graph, dim=0, reduce="sum")

    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    idx_ji = adj_t_row.storage.row()
    idx_kj = adj_t_row.storage.value()
    same_edge_diff = (num_edges // 2).repeat_interleave(num_triplets_per_graph, dim=0)
    mask = (idx_ji - idx_kj).abs() != same_edge_diff
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_ji, idx_kj = idx_ji[mask], idx_kj[mask]

    pos_ji = distance_vectors[idx_ji]
    pos_jk = -distance_vectors[idx_kj]
    a = (pos_ji * pos_jk).sum(dim=-1)
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)
    angle = torch.atan2(b, a)

    if use_torsion:
        idx_batch = torch.arange(len(idx_i), device=j.device)
        adj_t_row_t = adj_t[idx_j]
        idx_k_n = adj_t_row_t.storage.col()
        idx_k_n_j_t = adj_t_row_t.storage.value()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_ji_t = idx_ji.repeat_interleave(num_triplets_t)
        idx_kj_t = idx_kj.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        same_edge_diff_t = same_edge_diff[mask]
        same_edge_diff_t = same_edge_diff_t.repeat_interleave(num_triplets_t)
        mask = (idx_ji_t - idx_k_n_j_t).abs() != same_edge_diff_t
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = (
            idx_i_t[mask],
            idx_j_t[mask],
            idx_k_t[mask],
            idx_k_n[mask],
            idx_batch_t[mask],
        )
        idx_ji_t, idx_kj_t, idx_k_n_j_t = idx_ji_t[mask], idx_kj_t[mask], idx_k_n_j_t[mask]

        pos_jk = -distance_vectors[idx_kj_t]
        pos_ji = distance_vectors[idx_ji_t]
        pos_j_k_n = -distance_vectors[idx_k_n_j_t]
        plane1 = torch.cross(pos_ji, pos_jk)
        plane2 = torch.cross(pos_ji, pos_j_k_n)
        a = (plane1 * plane2).sum(dim=-1)
        b = torch.cross(plane1, plane2).norm(dim=-1)
        torsion1 = torch.atan2(b, a)
        torsion1[torsion1 <= 0] += 2 * PI
        torsion = scatter(torsion1, idx_batch_t, reduce="min")

        return dist, angle, torsion, i, j, idx_kj, idx_ji

    else:
        return dist, angle, i, j, idx_kj, idx_ji


# ======================== model/features.py (basis functions, verbatim) ========================
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


def real_sph_harm(l, zero_m_only=False, spherical_coordinates=True):  # noqa: E741 (matches original signature)
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
                if type(P_l_m[i][j]) is not int:
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


class Envelope(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class dist_emb(nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        # Original repo code writes an init value directly into the leaf Parameter via
        # `torch.arange(..., out=self.freq)`, which older torch allowed under autograd but
        # current torch (2.x) rejects for out=-args on a tensor that requires_grad. Wrapping
        # this parameter-initialization write in no_grad() is a version-compat shim only
        # (identical resulting values), not an architectural change.
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class angle_emb(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
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


class torsion_emb(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(
                    lambda x, y, sph1=sph1: torch.zeros_like(x) + torch.zeros_like(y) + sph1(0, 0)
                )
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k + i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        return out


# ========================= model/modules.py (message-passing blocks, verbatim) =========================
class emb(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super().__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class ResidualLayer(nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(nn.Module):  # noqa: A001 (matches original class name)
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True):
        super().__init__()
        self.act = act
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = Embedding(100, hidden_channels)
        else:
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf, _, _ = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        return e1, e2


class update_e(nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super().__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(
            num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False
        )
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(nn.Module):
    def __init__(
        self, hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init
    ):
        super().__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == "zeros":
            self.lin.weight.data.fill_(0)
        if self.output_init == "GlorotOrthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i):
        _, e2 = e
        v = scatter(e2, i, dim=0)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, v, batch):
        u = u + scatter(v, batch, dim=0)
        return u


# ============================ model/spherenet.py (top-level encoder, verbatim) ============================
class SphereNetEncoder(nn.Module):
    r"""The spherical message passing neural network SphereNet from the
    "Spherical Message Passing for 3D Graph Networks" paper (https://arxiv.org/abs/2102.05013)."""

    def __init__(
        self,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=128,
        out_channels=1,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
        output_init="GlorotOrthogonal",
        use_node_features=True,
    ):
        super().__init__()

        self.cutoff = cutoff

        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features)
        self.init_v = update_v(
            hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init
        )
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = nn.ModuleList(
            [
                update_v(
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_init,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_es = nn.ModuleList(
            [
                update_e(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size_dist,
                    basis_emb_size_angle,
                    basis_emb_size_torsion,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_us = nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e_layer in self.update_es:
            update_e_layer.reset_parameters()
        for update_v_layer in self.update_vs:
            update_v_layer.reset_parameters()

    def forward(self, batch_data):
        z = batch_data["atom_types"] - 1
        edge_index = batch_data["edge_index"]
        frac_coords = batch_data["frac_coords"]
        batch = batch_data["batch"]
        lattice_lengths, lattice_angles = batch_data["lengths"], batch_data["angles"]
        to_jimages, num_atoms, num_bonds = (
            batch_data["to_jimages"],
            batch_data["num_atoms"],
            batch_data["num_bonds"],
        )

        _, distance_vectors, _ = get_pbc_distances(
            frac_coords,
            edge_index,
            lattice_lengths,
            lattice_angles,
            to_jimages,
            num_atoms,
            num_bonds,
        )
        num_nodes = z.shape[0]
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(
            edge_index, num_nodes, num_bonds, distance_vectors, use_torsion=True
        )

        emb_out = self.emb(dist, angle, torsion, idx_kj)

        e = self.init_e(z, emb_out, i, j)
        v = self.init_v(e, i)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)

        for update_e_layer, update_v_layer, update_u_layer in zip(
            self.update_es, self.update_vs, self.update_us
        ):
            e = update_e_layer(e, emb_out, idx_kj, idx_ji)
            v = update_v_layer(e, i)
            u = update_u_layer(u, v, batch)

        return u


MENAGERIE_ZOO = "vendored-pytorch"

_HIDDEN_CHANNELS = 16
_OUT_CHANNELS = 8
_INT_EMB_SIZE = 8
_BASIS_EMB = 4
_OUT_EMB_CHANNELS = 16
_NUM_SPHERICAL = 3
_NUM_RADIAL = 4


def build_symat():
    return SphereNetEncoder(
        cutoff=5.0,
        num_layers=2,
        hidden_channels=_HIDDEN_CHANNELS,
        out_channels=_OUT_CHANNELS,
        int_emb_size=_INT_EMB_SIZE,
        basis_emb_size_dist=_BASIS_EMB,
        basis_emb_size_angle=_BASIS_EMB,
        basis_emb_size_torsion=_BASIS_EMB,
        out_emb_channels=_OUT_EMB_CHANNELS,
        num_spherical=_NUM_SPHERICAL,
        num_radial=_NUM_RADIAL,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=1,
        num_output_layers=2,
    )


def _small_periodic_crystal_batch():
    """Hand-build a tiny 2-crystal batch: each unit cell has a few atoms bonded to their
    periodic images (to_jimages != 0 for at least one bond, exercising the PBC-offset path),
    matching the field names/shapes SyMat's own data pipeline (utils/data_utils.py) produces.
    """
    torch.manual_seed(0)

    num_atoms = torch.tensor([3, 4], dtype=torch.long)
    total_atoms = int(num_atoms.sum())
    batch = torch.repeat_interleave(torch.arange(len(num_atoms)), num_atoms)

    atom_types = torch.randint(1, 30, (total_atoms,), dtype=torch.long)
    frac_coords = torch.rand(total_atoms, 3)

    lengths = torch.tensor([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
    angles = torch.tensor([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]])

    # Bonds within each crystal (both directions) plus one periodic-image bond per crystal.
    edge_src, edge_dst, to_jimages, num_bonds = [], [], [], []
    offset = 0
    for n in num_atoms.tolist():
        local_edges = []
        for k in range(n - 1):
            local_edges.append((k, k + 1, (0, 0, 0)))
            local_edges.append((k + 1, k, (0, 0, 0)))
        # one periodic-image bond wrapping the first and last atom of the cell
        local_edges.append((0, n - 1, (1, 0, 0)))
        local_edges.append((n - 1, 0, (-1, 0, 0)))
        for a, b, jimg in local_edges:
            edge_src.append(offset + a)
            edge_dst.append(offset + b)
            to_jimages.append(jimg)
        num_bonds.append(len(local_edges))
        offset += n

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    to_jimages = torch.tensor(to_jimages, dtype=torch.long)
    num_bonds = torch.tensor(num_bonds, dtype=torch.long)

    return {
        "atom_types": atom_types,
        "edge_index": edge_index,
        "frac_coords": frac_coords,
        "batch": batch,
        "lengths": lengths,
        "angles": angles,
        "to_jimages": to_jimages,
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
    }


def example_input_symat():
    return (_small_periodic_crystal_batch(),)


MENAGERIE_ENTRIES = [
    (
        "SyMat (SphereNet encoder backbone)",
        build_symat,
        example_input_symat,
        2023,
        "SOURCE_AVAILABLE",
    ),
]
