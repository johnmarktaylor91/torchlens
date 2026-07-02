# SOURCE: vendored from divelab/GraphBP @ main
#   (repo: https://github.com/divelab/GraphBP)
#   GraphBP/model/graphbp.py (GraphBP.__init__ + GraphBP.forward only) +
#   GraphBP/model/net_utils.py (ST_Net_Exp / Rescale / MLP / LB2 /
#   flow_forward) + GraphBP/model/schnet.py (SchNet / GaussianSmearing /
#   ShiftedSoftplus / CFConv / InteractionBlock) + GraphBP/model/features.py
#   (dist_emb / angle_emb / Envelope / bessel_basis / real_sph_harm helpers),
#   copied verbatim (imports only adjusted to be self-contained in this
#   single file; the unused `from torch_geometric.nn.acts import swish`
#   import in net_utils.py is dropped -- that submodule was removed from
#   modern torch_geometric and `swish` is never called by any vendored
#   function).
#
# GraphBP (Liu, Wang, Ji, ICLR 2022, "Generating 3D Molecules for Target
# Protein Binding") is a fragment-free autoregressive 3D ligand generator
# conditioned on a protein binding pocket: a shared protein/ligand atom
# embedding feeds a SchNet-style continuous-filter message-passing encoder
# over the joint protein+ligand point cloud, whose per-atom hidden states
# drive a focus-atom / contact-atom classifier (`focus_mlp` / `contact_mlp`)
# and four autoregressive normalizing-flow heads (`node_flow_layers`,
# `dist_flow_layers`, `angle_flow_layers`, `torsion_flow_layers`, all
# `ST_Net_Exp` affine-coupling blocks) that place the next ligand atom's
# type, bond distance, bond angle, and dihedral torsion relative to the
# already-placed atoms, using DimeNet-style spherical-Bessel distance/angle
# embeddings (`dist_emb`/`angle_emb`) to condition the distance->angle->
# torsion cascade. Only `GraphBP.forward` (SchNet encode -> focus/contact
# scores -> 4-stage flow_forward cascade) is vendored here -- the real
# architecture's shared training-time forward pass. `GraphBP.generate`
# (autoregressive sampling loop with `flow_reverse` + `dattoxyz` internal-
# coordinate placement) and `utils/bond_adding.py` (post-hoc RDKit bond
# perception) are inference/generation orchestration, not exercised by a
# single forward-pass trace, so they are not vendored; no architecture code
# was rewritten.

from math import pi as PI

import numpy as np
import sympy as sym
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special as sp
from scipy.optimize import brentq
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import MessagePassing, radius_graph

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim from model/net_utils.py
# ---------------------------------------------------------------------------


class ST_Net_Exp(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers  # unused
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim * 2, bias=bias)
        self.rescale1 = Rescale()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.0)
            nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        """
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        """
        x = self.linear2(self.tanh(self.linear1(x)))
        s = x[:, : self.output_dim]
        t = x[:, self.output_dim :]
        s = self.rescale1(torch.tanh(s))
        return s, t


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            raise RuntimeError("Rescale factor has NaN entries")

        x = torch.exp(self.weight) * x
        return x


def init_layer(layer: torch.nn.Linear, w_scale=1.0) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)  # type: ignore
    torch.nn.init.constant_(layer.bias.data, 0)
    return layer


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units=128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_units)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_units, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x).view(-1)


class LB2(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, bias=False):
        super(LB2, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_units, bias=bias)
        self.lin2 = nn.Linear(hidden_units, output_dim, bias=bias)

    def forward(self, x):
        return self.lin2(self.lin1(x))


def flow_forward(flow_layers, x, feat):
    for i in range(len(flow_layers)):
        s, t = flow_layers[i](feat)
        s = s.exp()
        x = (x + t) * s

        if i == 0:
            x_log_jacob = (torch.abs(s) + 1e-20).log()
        else:
            x_log_jacob += (torch.abs(s) + 1e-20).log()
    return x, x_log_jacob


# ---------------------------------------------------------------------------
# Verbatim from model/schnet.py
# ---------------------------------------------------------------------------


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNet(torch.nn.Module):
    def __init__(
        self,
        num_node_types,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
    ):
        super(SchNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff

        self.embedding = Embedding(num_node_types, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()

    def forward(self, z, pos, batch):
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h


# ---------------------------------------------------------------------------
# Verbatim from model/features.py
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
    """
    Computes formula strings of the the real part of the spherical harmonics
    up to order l (excluded). Variables are either cartesian coordinates
    x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
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


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
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


class dist_emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(dist_emb, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class angle_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super(angle_emb, self).__init__()
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
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj=None):
        dist = dist / self.cutoff

        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        if idx_kj is None:  # Use for encoding in generative modeling
            out = (rbf.view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        else:  # Use for SphereNet physical representation
            out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


# ---------------------------------------------------------------------------
# Verbatim from model/graphbp.py (GraphBP.__init__ + GraphBP.forward only)
# ---------------------------------------------------------------------------


class GraphBP(nn.Module):
    def __init__(
        self,
        cutoff,
        num_node_types,
        num_lig_node_types,
        num_interactions,
        num_filters,
        num_gaussians,
        hidden_channels,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_flow_layers,
        deq_coeff=0.9,
        use_gpu=False,
    ):
        super(GraphBP, self).__init__()
        self.use_gpu = use_gpu
        self.num_node_types = num_node_types
        self.num_lig_node_types = num_lig_node_types

        self.feat_net = SchNet(
            num_node_types, hidden_channels, num_filters, num_interactions, num_gaussians, cutoff
        )

        node_feat_dim, dist_feat_dim, angle_feat_dim, torsion_feat_dim = (
            hidden_channels,
            hidden_channels,
            hidden_channels * 2,
            hidden_channels * 3,
        )

        self.node_flow_layers = nn.ModuleList(
            [
                ST_Net_Exp(node_feat_dim, num_lig_node_types, hid_dim=hidden_channels, bias=True)
                for _ in range(num_flow_layers)
            ]
        )
        self.dist_flow_layers = nn.ModuleList(
            [
                ST_Net_Exp(dist_feat_dim, 1, hid_dim=hidden_channels, bias=True)
                for _ in range(num_flow_layers)
            ]
        )
        self.angle_flow_layers = nn.ModuleList(
            [
                ST_Net_Exp(angle_feat_dim, 1, hid_dim=hidden_channels, bias=True)
                for _ in range(num_flow_layers)
            ]
        )
        self.torsion_flow_layers = nn.ModuleList(
            [
                ST_Net_Exp(torsion_feat_dim, 1, hid_dim=hidden_channels, bias=True)
                for _ in range(num_flow_layers)
            ]
        )
        self.focus_mlp = MLP(hidden_channels)
        self.contact_mlp = MLP(hidden_channels)
        self.deq_coeff = deq_coeff

        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent=5)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent=5)

        self.dist_lb2 = LB2(num_radial, basis_emb_size, hidden_channels)
        self.angle_lb2 = LB2(num_spherical * num_radial, basis_emb_size, hidden_channels)

        if use_gpu:
            self.feat_net = self.feat_net.to("cuda")
            self.node_flow_layers = self.node_flow_layers.to("cuda")
            self.dist_flow_layers = self.dist_flow_layers.to("cuda")
            self.angle_flow_layers = self.angle_flow_layers.to("cuda")
            self.torsion_flow_layers = self.torsion_flow_layers.to("cuda")
            self.focus_mlp = self.focus_mlp.to("cuda")
            self.contact_mlp = self.contact_mlp.to("cuda")
            self.dist_lb2 = self.dist_lb2.to("cuda")
            self.angle_lb2 = self.angle_lb2.to("cuda")
            self.dist_emb = self.dist_emb.to("cuda")
            self.angle_emb = self.angle_emb.to("cuda")

    def forward(self, data_batch):
        z, pos, batch = data_batch["atom_type"], data_batch["position"], data_batch["batch"]
        node_feat = self.feat_net(z, pos, batch)
        focus_score = self.focus_mlp(node_feat[~data_batch["rec_mask"]])
        contact_score = self.contact_mlp(node_feat[data_batch["contact_y_or_n"]])

        new_atom_type, focus = data_batch["new_atom_type"], data_batch["focus"]
        x_z = F.one_hot(new_atom_type, num_classes=self.num_lig_node_types).float()
        x_z += self.deq_coeff * torch.rand(x_z.size(), device=x_z.device)

        local_node_type_feat = node_feat[focus[:, 0]]
        node_latent, node_log_jacob = flow_forward(self.node_flow_layers, x_z, local_node_type_feat)
        node_type_emb_block = self.feat_net.embedding
        node_type_emb = node_type_emb_block(new_atom_type)
        node_emb = node_feat * node_type_emb[batch]

        c1_focus, c2_c1_focus = data_batch["c1_focus"], data_batch["c2_c1_focus"]
        dist, angle, torsion = (
            data_batch["new_dist"],
            data_batch["new_angle"],
            data_batch["new_torsion"],
        )

        local_dist_feat = node_emb[focus[:, 0]]
        dist_latent, dist_log_jacob = flow_forward(self.dist_flow_layers, dist, local_dist_feat)

        # d --> theta
        dist_emb_out = self.dist_lb2(self.dist_emb(dist.squeeze()[batch].to(torch.float)))
        node_emb = node_emb * dist_emb_out  # [N, hidden] * [N, hidden]

        node_emb_clone = node_emb.clone()  # Avoid changing node_emb in-place
        local_angle_feat = torch.cat(
            (node_emb_clone[c1_focus[:, 1]], node_emb_clone[c1_focus[:, 0]]), dim=1
        )
        angle_latent, angle_log_jacob = flow_forward(
            self.angle_flow_layers, angle, local_angle_feat
        )

        # d, theta --> phi
        dist_angle_emd = self.angle_lb2(
            self.angle_emb(
                dist.squeeze()[batch].to(torch.float), angle.squeeze()[batch].to(torch.float)
            )
        )
        node_emb = node_emb * dist_angle_emd

        local_torsion_feat = torch.cat(
            (node_emb[c2_c1_focus[:, 2]], node_emb[c2_c1_focus[:, 1]], node_emb[c2_c1_focus[:, 0]]),
            dim=1,
        )
        torsion_latent, torsion_log_jacob = flow_forward(
            self.torsion_flow_layers, torsion, local_torsion_feat
        )

        return (
            (node_latent, node_log_jacob),
            focus_score,
            contact_score,
            (dist_latent, dist_log_jacob),
            (angle_latent, angle_log_jacob),
            (torsion_latent, torsion_log_jacob),
        )


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_graphbp():
    """Tiny-size real GraphBP (SchNet pocket/ligand encoder + focus/contact
    classifiers + 4-stage autoregressive flow cascade), CPU-only
    (use_gpu=False, matching the repo's own supported constructor flag)."""
    return GraphBP(
        cutoff=10.0,
        num_node_types=6,
        num_lig_node_types=5,
        num_interactions=2,
        num_filters=16,
        num_gaussians=10,
        hidden_channels=16,
        basis_emb_size=8,
        num_spherical=2,
        num_radial=4,
        num_flow_layers=2,
        use_gpu=False,
    )


def example_input_graphbp():
    """Tiny 2-complex batch matching GraphBP.forward's data_batch dict
    contract: each complex has 6 protein/receptor atoms + 3 already-placed
    ligand atoms, with one new atom being autoregressively placed per
    complex (focus on ligand atom 0 of each complex; c1/c2 context atoms
    among that complex's existing 9 nodes). Two molecules (rather than one)
    keep `new_dist`/`new_angle`/`new_torsion` -- which the real forward
    squeezes from [num_mols, 1] to a 1-D per-molecule vector before
    broadcasting to nodes via `batch` -- from squeezing to a 0-D scalar."""
    torch.manual_seed(0)
    n_rec, n_lig_placed = 6, 3
    n_per_mol = n_rec + n_lig_placed
    num_mols = 2
    n_total = n_per_mol * num_mols

    atom_type = torch.cat(
        [
            torch.cat([torch.randint(0, 6, (n_rec,)), torch.randint(0, 5, (n_lig_placed,))])
            for _ in range(num_mols)
        ]
    )
    position = torch.randn(n_total, 3)
    batch = torch.arange(num_mols).repeat_interleave(n_per_mol)
    rec_mask = torch.cat(
        [
            torch.cat(
                [torch.ones(n_rec, dtype=torch.bool), torch.zeros(n_lig_placed, dtype=torch.bool)]
            )
            for _ in range(num_mols)
        ]
    )
    contact_y_or_n = torch.zeros(n_total, dtype=torch.bool)
    contact_y_or_n[0] = True
    contact_y_or_n[n_per_mol] = True

    new_atom_type = torch.tensor([2, 3])
    mol_offsets = torch.arange(num_mols) * n_per_mol
    focus = (mol_offsets + n_rec).unsqueeze(1)  # focus on first placed ligand atom, per mol
    c1_focus = torch.stack([mol_offsets + n_rec, mol_offsets + n_rec + 1], dim=1)
    c2_c1_focus = torch.stack(
        [mol_offsets + n_rec, mol_offsets + n_rec + 1, mol_offsets + n_rec + 2], dim=1
    )
    new_dist = torch.tensor([[1.5], [1.3]])
    new_angle = torch.tensor([[1.9], [2.1]])
    new_torsion = torch.tensor([[0.7], [0.4]])

    data_batch = {
        "atom_type": atom_type,
        "position": position,
        "batch": batch,
        "rec_mask": rec_mask,
        "contact_y_or_n": contact_y_or_n,
        "new_atom_type": new_atom_type,
        "focus": focus,
        "c1_focus": c1_focus,
        "c2_c1_focus": c2_c1_focus,
        "new_dist": new_dist,
        "new_angle": new_angle,
        "new_torsion": new_torsion,
    }
    return (data_batch,)


MENAGERIE_ENTRIES = [
    (
        "GraphBP",
        build_graphbp,
        example_input_graphbp,
        2022,
        "CODE",
    ),
]
