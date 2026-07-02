# SOURCE: vendored from ixsluo/CrystalFlow @ main
#   (paper: arXiv:2412.11693 "CrystalFlow: A Flow-Based Generative Model for
#   Crystalline Materials", code: https://github.com/ixsluo/CrystalFlow)
#
# CSPNet is the equivariant graph-neural-network denoiser/vector-field network
# that CrystalFlow's flow-matching sampler calls at every ODE integration step
# to predict the lattice and fractional-coordinate velocity fields. This file
# vendors the ACTUAL model classes from diffcsp/pl_modules/cspnet.py verbatim
# (CrystalFlow's repo keeps the historical "diffcsp" package name from the
# authors' earlier DiffCSP work; CSPNet itself is the real CrystalFlow network).
#
# Only import paths were adjusted (the original file imports several pure
# torch/numpy geometry helpers from diffcsp/common/data_utils.py, a much larger
# module that also pulls in heavy non-base deps such as pymatgen/pyxtal used
# only for dataset preprocessing; those specific helper functions are copied
# here verbatim since they have no such heavy dependencies themselves).
#
# CSPNet.gen_edges() supports 3 edge styles in the original source: 'fc'
# (fully-connected within each crystal), 'knn'/'knn_cart', and 'knn_frac'.
# This staged module keeps the 'fc' branch verbatim (CrystalFlow's own
# default config, see conf/model/decoder/cspnet.yaml) and omits the two knn
# branches together with the symmetric-edge-reordering helpers
# (select_symmetric_edges/reorder_symmetric_edges/repeat_blocks) that only
# those branches call, since they require radius_graph_pbc (periodic-boundary
# neighbor search) that is unnecessary for a random-init capture and was
# excluded to keep this staged module dependency-minimal. CSPNet, CSPLayer,
# and all embedding submodules are otherwise transcribed verbatim.
#
# No architectural code was rewritten or approximated.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse
from einops import rearrange, repeat  # noqa: F401  (kept for source fidelity)

MENAGERIE_ZOO = "vendored-pytorch"

MAX_ATOMIC_NUM = 100


# ---------------------------------------------------------------------------
# Verbatim helpers from diffcsp/common/data_utils.py (pure torch/numpy, no
# heavy deps) needed by CSPNet's default 'fc' (fully-connected) edge style.
# ---------------------------------------------------------------------------


def lattice_params_to_matrix_torch(lengths, angles):
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


def get_reciprocal_lattice_torch(L):
    V = torch.det(L)[:, None] + 1e-9  # (B, 1)
    a0 = L[:, 0, :]  # (B, 3)
    a1 = L[:, 1, :]
    a2 = L[:, 2, :]
    cross_a12 = torch.cross(a1, a2, dim=1)  # (B, 3)
    cross_a20 = torch.cross(a2, a0, dim=1)
    cross_a01 = torch.cross(a0, a1, dim=1)
    b0 = (2 * math.pi * cross_a12 / V)[:, None, :]  # (B, 1, 3)
    b1 = (2 * math.pi * cross_a20 / V)[:, None, :]  # (B, 1, 3)
    b2 = (2 * math.pi * cross_a01 / V)[:, None, :]  # (B, 1, 3)
    b = torch.cat([b0, b1, b2], dim=1)  # (B, 3, 3)
    return b


# ---------------------------------------------------------------------------
# Verbatim CSPNet architecture from diffcsp/pl_modules/cspnet.py
# ---------------------------------------------------------------------------


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PeriodicNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, metrics, scaled_r):
        """
        Parameters
        ----------
        metrics: torch.Tensor
            Metrics matrix, symmetric. Shape of (E, 3, 3).
        scaled_r: torch.Tensor
            Vectors in fractional coordinates of the lattice cell. Shape of (E, 3).
        Results
        -------
        norm: torch.Tensor, shape (E,)
        """
        a = 1 - torch.cos(2 * math.pi * scaled_r)
        b = torch.sin(2 * math.pi * scaled_r)
        cos_term = torch.einsum("em,emn,en->e", a, metrics, a)
        sin_term = torch.einsum("em,emn,en->e", b, metrics, b)
        return (1 / (2 * math.pi)) * torch.sqrt(cos_term + sin_term)


class RecSinusoidsEmbedding(nn.Module):
    def __init__(self, n_millers=10):
        super().__init__()
        self.n_millers = n_millers
        self.millerindices = torch.cartesian_prod(
            torch.arange(self.n_millers),
            torch.arange(self.n_millers),
            torch.arange(self.n_millers),
        ).to(torch.get_default_dtype())
        self.dim = 2 * self.millerindices.shape[0]

    def forward(self, frac_diff, lattice):
        cart_diff = torch.einsum("ei,eij->ej", frac_diff, lattice)  # (E,3)
        R = get_reciprocal_lattice_torch(lattice)
        hb = torch.einsum("mi,eij->emj", self.millerindices.to(R.device), R)  # (E, M, 3)
        hbX = torch.einsum("emj,ej->em", hb, cart_diff)  # (E, M)
        emb = torch.cat([hbX.cos(), hbX.sin()], dim=-1)  # (E, 2M)
        return emb


class CSPLayer(nn.Module):
    """Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        lattice_dim=9,
        act_fn=nn.SiLU(),
        dis_emb=None,
        rec_emb=None,
        na_emb=None,
        periodic_norm=None,
        ln=False,
        ip=True,
        use_angles=False,
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.rec_emb = rec_emb
        self.na_emb = na_emb
        self.periodic_norm = periodic_norm
        self.ip = ip
        self.use_angles = use_angles

        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        if rec_emb is not None:
            self.dis_dim += rec_emb.dim
        if na_emb is not None:
            self.dis_dim += na_emb.embedding_dim
        if use_angles:
            self.dis_dim += 3
        if periodic_norm is not None:
            self.dis_dim += 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + lattice_dim + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), act_fn, nn.Linear(hidden_dim, hidden_dim), act_fn
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices_rep,
        edge_index,
        edge2graph,
        num_atoms,
        frac_diff=None,
        lattices_mat=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        inputs = [hi, hj]

        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0

        if self.rec_emb is not None:
            rec_diff = self.rec_emb(frac_diff, lattices_mat[edge2graph])
            inputs.append(rec_diff)

        if self.dis_emb is not None:
            frac_diff_emb = self.dis_emb(frac_diff)
            inputs.append(frac_diff_emb)

        if self.na_emb is not None:
            na_emb = self.na_emb(num_atoms.repeat_interleave(num_atoms, dim=0))[edge_index[0]]
            inputs.append(na_emb)

        if self.periodic_norm is not None:
            metrics = (lattices_mat.transpose(-1, -2) @ lattices_mat)[edge2graph]
            norm = self.periodic_norm(metrics, frac_diff)
            norm = norm.view(-1, 1)
            inputs.append(norm)

        if self.use_angles:  # angles between edges and lattices vectors
            cart_diff = torch.einsum("ei,eij->ej", frac_diff, lattices_mat[edge2graph])  # (E,3)
            inner_dot = torch.einsum("ei,eji->ej", cart_diff, lattices_mat[edge2graph])  # (E,3)
            norm = torch.linalg.norm(inner_dot, axis=1)[:, None]  # (E,1)
            cos_angles = inner_dot / norm
            cos_angles = torch.where(cos_angles.isnan(), 0, cos_angles)
            inputs.append(cos_angles)

        if self.ip:
            lattice_ips = lattices_rep @ lattices_rep.transpose(-1, -2)
        else:
            lattice_ips = lattices_rep
        lattice_ips_flatten = torch.flatten(lattice_ips, start_dim=1)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        inputs.append(lattice_ips_flatten_edges)

        edges_input = torch.cat(inputs, dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(
            edge_features, edge_index[0], dim=0, reduce="mean", dim_size=node_features.shape[0]
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features,
        frac_coords,
        lattices_rep,
        edge_index,
        edge2graph,
        num_atoms,
        frac_diff=None,
        lattices_mat=None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features,
            frac_coords,
            lattices_rep,
            edge_index,
            edge2graph,
            num_atoms,
            frac_diff,
            lattices_mat,
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):
    """CrystalFlow's equivariant vector-field network: the real model called
    at every flow-matching ODE step to predict lattice/coordinate velocities."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        lattice_dim=9,
        cemb_dim=1,
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",  # fractional distance embedding
        num_freqs=10,
        rec_emb="none",  # reciprocal distance embedding
        num_millers=5,
        periodic_norm=False,
        na_emb=0,  # number of atoms embedding
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        use_angles=False,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
        type_encoding=None,
    ):
        super(CSPNet, self).__init__()

        self.ip = ip
        self.smooth = smooth
        self.type_encoding = type_encoding
        if self.type_encoding is None:
            if self.smooth:
                self.node_embedding = nn.Linear(MAX_ATOMIC_NUM, hidden_dim)
            else:
                self.node_embedding = nn.Embedding(MAX_ATOMIC_NUM, hidden_dim)
        else:
            self.node_embedding = nn.Linear(self.type_encoding.out_dim, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)  # no trainable params
        elif dis_emb == "none":
            self.dis_emb = None
        else:
            raise ValueError(f"Unknown fractional distance embedding: {dis_emb=}")
        if rec_emb == "sin":
            self.rec_emb = RecSinusoidsEmbedding(n_millers=num_millers)  # no trainable params
        elif rec_emb == "none":
            self.rec_emb = None
        else:
            raise ValueError(f"Unknown reciprocal distance embedding: {rec_emb=}")
        if na_emb > 0:
            self.na_emb = nn.Embedding(
                max_atoms, na_emb
            )  # with trainable params, but same in each layer
        else:
            self.na_emb = None
        if periodic_norm:
            self.periodic_norm = PeriodicNorm()
        else:
            self.periodic_norm = None

        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i,
                CSPLayer(
                    hidden_dim=hidden_dim,
                    lattice_dim=lattice_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    rec_emb=self.rec_emb,
                    na_emb=self.na_emb,
                    periodic_norm=self.periodic_norm,
                    ln=ln,
                    ip=ip,
                    use_angles=use_angles,
                ),
            )
            self.add_module(
                "cemb_mixin_%d" % i,
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
            self.add_module(
                "cemb_adapter_%d" % i,
                nn.Sequential(
                    nn.Linear(cemb_dim, hidden_dim),
                    self.act_fn,
                    nn.Linear(hidden_dim, hidden_dim),
                    self.act_fn,
                ),
            )

        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, lattice_dim, bias=False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            type_out_dim = (
                MAX_ATOMIC_NUM if self.type_encoding is None else self.type_encoding.out_dim
            )
            self.type_out = nn.Linear(hidden_dim, type_out_dim)
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0

        else:
            raise ValueError(
                f"Unknown type of edge style: {self.edge_style} "
                "(this vendored module ships the 'fc' path only; 'knn'/"
                "'knn_cart'/'knn_frac' require radius_graph_pbc, which is "
                "not needed for a random-init trace and was left out to "
                "keep the staged module dependency-minimal)."
            )

    def forward(
        self,
        t,
        atom_types,
        frac_coords,
        lattices_rep,
        num_atoms,
        node2graph,
        lattices_mat=None,
        cemb=None,
        guide_indicator=None,
    ):
        if lattices_mat is None:
            lattices_mat = lattices_rep
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices_mat, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            if cemb is not None:
                cemb_mixin = self._modules["cemb_mixin_%d" % i]
                cemb_adapter = self._modules["cemb_adapter_%d" % i]
                cemb_bias = (
                    cemb_mixin(cemb_adapter(cemb)) * guide_indicator[:, None]
                ).repeat_interleave(num_atoms, dim=0)
                node_features = node_features + cemb_bias
            csp_layer = self._modules["csp_layer_%d" % i]
            node_features = csp_layer(
                node_features,
                frac_coords,
                lattices_rep,
                edges,
                edge2graph,
                num_atoms=num_atoms,
                frac_diff=frac_diff,
                lattices_mat=lattices_mat,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(lattices_rep.shape)
        if self.ip:
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices_rep)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_crystalflow_cspnet():
    """Tiny-size real CSPNet (CrystalFlow's flow-matching denoiser network)."""
    return CSPNet(
        hidden_dim=32,
        latent_dim=1,
        lattice_dim=9,
        cemb_dim=1,
        num_layers=2,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=4,
        rec_emb="none",
        periodic_norm=False,
        na_emb=0,
        edge_style="fc",
        ln=False,
        ip=True,
        use_angles=False,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
    )


def example_input_crystalflow_cspnet():
    """A tiny 2-crystal batch (mirrors CrystalFlow's PyG-style batched graph
    inputs to CSPNet.forward): timestep embedding, atom types, fractional
    coords, lattice matrices, atom counts per crystal, and node->graph index.
    """
    torch.manual_seed(0)
    num_atoms = torch.tensor([3, 2], dtype=torch.long)
    n_total = int(num_atoms.sum().item())
    n_graphs = num_atoms.shape[0]

    t = torch.rand(n_graphs, 1)
    atom_types = torch.randint(1, MAX_ATOMIC_NUM, (n_total,), dtype=torch.long)
    frac_coords = torch.rand(n_total, 3)

    lengths = torch.rand(n_graphs, 3) + 3.0
    angles = torch.full((n_graphs, 3), 90.0)
    lattices_rep = lattice_params_to_matrix_torch(lengths, angles)

    node2graph = torch.repeat_interleave(torch.arange(n_graphs), num_atoms)

    return (t, atom_types, frac_coords, lattices_rep, num_atoms, node2graph)


MENAGERIE_ENTRIES = [
    (
        "CrystalFlow-CSPNet",
        build_crystalflow_cspnet,
        example_input_crystalflow_cspnet,
        2024,
        "CODE",
    ),
]
