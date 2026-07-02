# SOURCE: vendored from HannesStark/FlowSite @ main (models/flowsite_model.py,
# models/invariant_layers.py, models/pytorch_modules.py, utils/simdesign_utils.py).
# FlowSite (ICML 2024, arxiv:2310.05764) is a conditional generative model for
# protein-pocket / binding-site design: a PiFold-style invariant residue-level
# GNN (`InvariantLayer`/`PiFoldConv`/`PiFoldEmbedder`, adapted by the original
# authors from A4Bio/PiFold) that jointly refines receptor-residue and
# ligand-atom node/edge embeddings across cross-attention message-passing
# layers, then decodes per-residue amino-acid identity logits (and optional
# side-chain torsion angles). The real repo also supports an equivariant
# Tensor-Field-Network (e3nn) refinement branch gated by `args.use_tfn`; this
# staging module exercises the model with `use_tfn=False, use_inv=True`, an
# officially supported configuration (see the repo's own
# `assert args.use_tfn or args.use_inv` in `FlowSiteModel.__init__`), so we
# vendor only the invariant-GNN branch classes (`FlowSiteModel`,
# `LigEdgeBuilder`, `CrossEdgeBuilder`, `build_cg_general`,
# `InvariantLayer`/`PiFoldConv`/`PiFoldEmbedder` and their node modules, plus
# `pytorch_modules.py`'s `Linear`/`Encoder`/`GaussianSmearing`-adjacent
# helpers and `simdesign_utils.py`'s backbone-geometry feature functions) --
# every class here is transcribed verbatim from the official repo, with only
# the TFN/e3nn branch, the rdkit/Bio.PDB-dependent data *preprocessing*
# pipeline (`utils/featurize.py`, `datasets/complex_dataset.py`), and
# time-conditioning (unused when `time_condition_inv=False`, the default)
# omitted since they are not exercised by this configuration. The small
# `atom_features_list['residues_canonical']` / `get_feature_dims()`-derived
# dimension is reproduced verbatim (it is a pure python constant, not an
# architectural choice) so we avoid importing the rdkit-gated
# `utils/featurize.py` module just for a dimension count.
"""FlowSite invariant-GNN branch (real repo PyTorch + torch_geometric code, vendored)."""

import time
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from torch_geometric.data import HeteroData
from torch_geometric.utils import unbatch
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# models/pytorch_modules.py (verbatim, minus the unused deepspeed LayerNorm
# bfloat16 branch import guard which is retained but never triggered on CPU
# float32 tensors)
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, emb_dim, feature_dims):
        # first element of feature_dims is a list with the length of each categorical feature
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()
        self.num_categorical_features = len(feature_dims)
        for i, dim in enumerate(feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
        return x_embedding


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    from scipy.stats import truncnorm

    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = np.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """A Linear layer with built-in nonstandard initializations (openfold-derived)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")


# ---------------------------------------------------------------------------
# utils/simdesign_utils.py (verbatim; from jingraham/neurips19-graph-protein-design)
# ---------------------------------------------------------------------------


def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor


def _normalize(tensor, dim=-1):
    return nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def cal_dihedral(X, eps=1e-7):
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_2 = U[:, 2:, :]

    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_2), dim=-1)

    cosD = (n_0 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

    v = _normalize(torch.cross(n_0, n_1), dim=-1)
    D = torch.sign((-v * u_1).sum(-1)) * torch.acos(cosD)

    return D


def _dihedrals(X, dihedral_type=0, eps=1e-7):
    B, N, _, _ = X.shape
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    D = cal_dihedral(X)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    dX = X[:, 1:, :] - X[:, :-1, :]
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    cosD = (u_0 * u_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    D_features = torch.cat((Dihedral_Angle_features, Angle_features), 2)
    return D_features


def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0.0, 20.0, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def _get_rbf(A, B, E_idx=None, num_rbf=16):
    if E_idx is not None:
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = _rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, :, None, :]) ** 2, -1) + 1e-6)
        RBF_A_B = _rbf(D_A_B, num_rbf)
    return RBF_A_B


def _orientations_coarse_gl_tuple(X, E_idx, eps=1e-6):
    V = X.clone()
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:, :-2, :], U[:, 1:-1, :]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)

    n_0 = n_0[:, ::3, :]
    b_1 = b_1[:, ::3, :]
    X = X[:, ::3, :]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0, 0, 0, 1), "constant", 0)

    Q_neighbors = gather_nodes(Q, E_idx)
    X_neighbors = gather_nodes(V[:, :, 1, :], E_idx)
    N_neighbors = gather_nodes(V[:, :, 0, :], E_idx)
    C_neighbors = gather_nodes(V[:, :, 2, :], E_idx)
    O_neighbors = gather_nodes(V[:, :, 3, :], E_idx)

    Q = Q.view(list(Q.shape[:2]) + [3, 3]).unsqueeze(2)
    Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:3]) + [3, 3])

    dX = (
        torch.stack([X_neighbors, N_neighbors, C_neighbors, O_neighbors], dim=3)
        - X[:, :, None, None, :]
    )
    dU = torch.matmul(Q[:, :, :, None, :, :], dX[..., None]).squeeze(-1)
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K, -1)
    R = torch.matmul(Q.transpose(-1, -2), Q_neighbors)
    q = _quaternions(R)

    dX_inner = V[:, :, [0, 2, 3], :] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B, N, -1)
    return V_direct, E_direct, q


def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(
        torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1))
    )

    def _R(i, j):
        return R[:, :, :, i, j]

    signs = torch.sign(
        torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1)
    )
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q


# ---------------------------------------------------------------------------
# models/invariant_layers.py (verbatim; PiFold-style invariant GNN, adapted by
# the FlowSite authors from A4Bio/PiFold)
# ---------------------------------------------------------------------------


class InvariantLayer(nn.Module):
    def __init__(self, args, device, update_edges=True):
        super().__init__()
        self.args = args
        self.device = device
        fold_dim = args.fold_dim
        self.rec2rec = PiFoldConv(
            args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges
        )
        if not self.args.ignore_lig:
            self.lig2rec = PiFoldConv(
                args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges
            )
            self.rec2lig = PiFoldConv(
                args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges
            )
            self.lig2lig = PiFoldConv(
                args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges
            )

    def forward(
        self, data, rec_idx, rec_na, rec_ea, lig_idx, lig_na, lig_ea, cross_idx, cross_ea, temb
    ):
        if self.args.time_condition_inv and self.args.time_condition_repeat:
            rec_nap = rec_na + temb[data["protein"].batch.long()]
            lig_nap = lig_na + temb[data["ligand"].batch.long()]
            lig_eap = lig_ea + temb[data["ligand"].batch.long()[lig_idx[0].long()]]
            rec_eap = rec_ea + temb[data["protein"].batch.long()[rec_idx[0].long()]]
            cross_eap = cross_ea + temb[data["ligand"].batch.long()[cross_idx[0].long()]]
        else:
            rec_nap, lig_nap, lig_eap, rec_eap, cross_eap = rec_na, lig_na, lig_ea, rec_ea, cross_ea

        if not self.args.ignore_lig:
            rec2lig_na, rec2lig_ea = self.rec2lig(
                rec_nap, lig_nap, cross_eap, cross_idx, data["ligand"].batch
            )
            if self.args.inv_straight_combine:
                lig_nap = lig_nap + rec2lig_na
            lig2lig_na, lig2lig_ea = self.lig2lig(
                lig_nap, lig_nap, lig_eap, lig_idx, data["ligand"].batch
            )
            if self.args.inv_straight_combine:
                lig_nap = lig_nap + lig2lig_na
            lig2rec_na, lig2rec_ea = self.lig2rec(
                lig_nap, rec_nap, cross_eap, cross_idx.flip(0), data["protein"].batch
            )
            if self.args.inv_straight_combine:
                rec_nap = rec_nap + lig2rec_na
        rec2rec_na, rec2rec_ea = self.rec2rec(
            rec_nap, rec_nap, rec_eap, rec_idx, data["protein"].batch
        )

        rec_na = rec2rec_na
        rec_ea = rec2rec_ea
        if not self.args.ignore_lig:
            rec_na = rec_na + lig2rec_na
            lig_na = lig2lig_na + rec2lig_na
            lig_ea = lig2lig_ea
            cross_ea = lig2rec_ea + rec2lig_ea
        return rec_na, rec_ea, lig_na, lig_ea, cross_ea


# The following classes were adapted from https://github.com/A4Bio/PiFold/blob/main/methods/prodesign_module.py


class PiFoldConv(nn.Module):
    def __init__(self, args, num_hidden, num_in, dropout=0.1, scale=30, update_edges=True):
        super().__init__()
        self.args = args
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads=4)
        self.update_edegs = update_edges

        if update_edges:
            self.edge_net = EdgeMLP(num_hidden, num_in)

        if args.node_context or args.edge_context:
            self.context = Context(
                num_hidden, num_in, node_context=args.node_context, edge_context=args.edge_context
            )

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4), nn.ReLU(), nn.Linear(num_hidden * 4, num_hidden)
        )

    def forward(self, src_na, dst_na, ea, edge_idx, batch_id):
        dh = self.attention(src_na, dst_na, ea, edge_idx)

        dst_na_update = self.norm[0](dst_na + self.dropout(dh))
        dh = self.dense(dst_na_update)
        dst_na_update = self.norm[1](dst_na_update + self.dropout(dh))

        if self.update_edegs:
            ea = self.edge_net(src_na, dst_na_update, ea, edge_idx)

        if self.args.node_context or self.args.edge_context:
            dst_na_update, ea = self.context(dst_na_update, ea, edge_idx, batch_id)
        return dst_na_update, ea


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp

        self.W_V = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
        )
        self.Bias = nn.Sequential(
            nn.Linear(num_hidden * 3, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_heads),
        )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, src_na, dst_na, ea, edge_idx):
        dst_idx = edge_idx[0]
        src_idx = edge_idx[1]

        d = int(self.num_hidden / self.num_heads)

        w = self.Bias(torch.cat([src_na[src_idx], ea, dst_na[dst_idx]], dim=-1)).view(
            ea.shape[0], self.num_heads, 1
        )
        attend_logits = w / np.sqrt(d)

        V = self.W_V(torch.cat([src_na[src_idx], ea], dim=-1)).view(-1, self.num_heads, d)
        attend = scatter_softmax(attend_logits, index=dst_idx, dim=0)
        dst_na_update = scatter_sum(attend * V, dst_idx, dim=0, dim_size=len(dst_na)).view(
            [-1, self.num_hidden]
        )

        if self.output_mlp:
            dst_na_update = self.W_O(dst_na_update)
        else:
            dst_na_update = dst_na_update
        return dst_na_update


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, src_na, dst_na, ea, edge_idx):
        dst_idx = edge_idx[0]
        src_idx = edge_idx[1]

        h_EV = torch.cat([src_na[src_idx], ea, dst_na[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        ea_update = self.norm(ea + self.dropout(h_message))
        return ea_update


class Context(nn.Module):
    def __init__(self, num_hidden, num_in, scale=30, node_context=False, edge_context=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context
        if self.node_context:
            self.V_MLP_g = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.Sigmoid(),
            )

        if self.edge_context:
            self.E_MLP = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
            )

            self.E_MLP_g = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.Sigmoid(),
            )

    def forward(self, dst_na, ea, edge_idx, batch_id):
        if self.node_context:
            c_V = scatter_mean(dst_na, batch_id, dim=0)
            dst_na = dst_na * self.V_MLP_g(c_V[batch_id])

        if self.edge_context:
            c_V = scatter_mean(dst_na, batch_id, dim=0)
            ea = ea * self.E_MLP_g(c_V[batch_id[edge_idx[0]]])

        return dst_na, ea


# ---------------------------------------------------------------------------
# models/tfn_layers.py -- only the (base-lib) GaussianSmearing distance
# embedding is needed by the invariant-only (use_tfn=False) configuration
# exercised here; the e3nn tensor-product refinement classes in the real
# tfn_layers.py are not imported since that branch is disabled.
# ---------------------------------------------------------------------------


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1) + 1e-6
        return torch.exp(self.coeff * torch.pow(dist, 2))


# ---------------------------------------------------------------------------
# models/flowsite_model.py (verbatim, invariant-only slice): FlowSiteModel
# forward path with use_tfn=False, LigEdgeBuilder, CrossEdgeBuilder,
# build_cg_general, PiFoldEmbedder.
# ---------------------------------------------------------------------------

# utils/featurize.py::atom_features_list['residues_canonical'] and
# get_feature_dims() -- reproduced verbatim as pure python constants (no
# rdkit/Bio.PDB call involved) to avoid importing the rdkit-gated featurize
# module just for a dimension count.
_RESIDUES_CANONICAL = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "misc",
]  # fmt: skip
_ATOMIC_NUM = list(range(1, 75)) + ["misc"]
_CHIRALITY = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
_DEGREE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"]
_NUMRING = [0, 1, 2, "misc"]
_IMPLICIT_VALENCE = [0, 1, 2, 3, 4, 5, 6, "misc"]
_FORMAL_CHARGE = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"]
_NUMH = [0, 1, 2, 3, 4, "misc"]
_HYBRIDIZATION = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"]
_IS_AROMATIC = [False, True]
_IS_IN_RING5 = [False, True]
_IS_IN_RING6 = [False, True]
_BOND_TYPE = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]
_BOND_STEREO = ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"]
_IS_CONJUGATED = [False, True]


def get_feature_dims():
    node_feature_dims = [
        len(_ATOMIC_NUM), len(_CHIRALITY), len(_DEGREE), len(_NUMRING),
        len(_IMPLICIT_VALENCE), len(_FORMAL_CHARGE), len(_NUMH),
        len(_HYBRIDIZATION), len(_IS_AROMATIC), len(_IS_IN_RING5), len(_IS_IN_RING6),
    ]  # fmt: skip
    edge_attribute_dims = [len(_BOND_TYPE), len(_BOND_STEREO), len(_IS_CONJUGATED), 1]
    lig_bond_feature_dims = [len(_BOND_TYPE), len(_BOND_STEREO), len(_IS_CONJUGATED)]
    rec_feature_dims = [len(_RESIDUES_CANONICAL)]
    return node_feature_dims, edge_attribute_dims, lig_bond_feature_dims, rec_feature_dims


def build_cg_general(pos, edge_attr, edge_index, batch=None, radius=None):
    if isinstance(pos, tuple):
        pos1, pos2 = pos
        batch1, batch2 = batch or (None, None)
    else:
        pos1 = pos2 = pos
        batch1 = batch2 = batch

    if radius is not None:
        if isinstance(radius, tuple):
            radius1, radius2 = radius
        else:
            radius1 = radius2 = radius

        radius_edge_idx = torch_cluster.radius(
            pos1 / radius1, pos2 / radius2, 1.0, batch1, batch2, max_num_neighbors=10000
        )

        if edge_index is not None:
            edge_index = torch.cat([edge_index, radius_edge_idx], 1).long()
        else:
            edge_index = radius_edge_idx

        if edge_attr is not None:
            edge_attr = F.pad(edge_attr, (0, 0, 0, radius_edge_idx.shape[-1]))

    return edge_index, edge_attr


class LigEdgeBuilder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.lig_radius_embedder = nn.Sequential(
            GaussianSmearing(0.0, args.protein_radius, args.radius_emb_dim),
            Linear(
                args.radius_emb_dim, args.fold_dim, init="relu" if args.fancy_init else "default"
            ),
            nn.ReLU(),
            Linear(args.fold_dim, args.fold_dim, init="final" if args.fancy_init else "default"),
        )

    def forward(self, data, lig_ea):
        edge_index, edge_attr = build_cg_general(
            pos=data["ligand"].pos,
            edge_attr=lig_ea,
            edge_index=data["ligand", "bond_edge", "ligand"].edge_index,
            batch=data["ligand"].batch,
            radius=self.args.lig_radius,
        )
        src, dst = edge_index
        edge_vec = data["ligand"].pos[dst.long()] - data["ligand"].pos[src.long()]

        edge_attr = edge_attr + self.lig_radius_embedder(edge_vec.norm(dim=-1))
        return edge_index, edge_attr


class CrossEdgeBuilder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.cross_rbf = GaussianSmearing(0.0, args.protein_radius, args.radius_emb_dim)
        self.cross_attr_embedder = nn.Sequential(
            Linear(
                args.radius_emb_dim * 5,
                args.fold_dim,
                init="relu" if args.fancy_init else "default",
            ),
            nn.ReLU(),
            Linear(args.fold_dim, args.fold_dim, init="final" if args.fancy_init else "default"),
        )

    def forward(self, data):
        cross_idx, _ = build_cg_general(
            pos=(data["protein"].pos, data["ligand"].pos),
            edge_attr=None,
            edge_index=None,
            batch=(data["protein"].batch, data["ligand"].batch),
            radius=self.args.cross_radius,
        )
        src, dst = cross_idx
        edge_vec = data["ligand"].pos[src.long()] - data["protein"].pos[dst.long()]
        edge_vec_cb = data["ligand"].pos[src.long()] - data["protein"].pos_Cb[dst.long()]
        edge_vec_c = data["ligand"].pos[src.long()] - data["protein"].pos_C[dst.long()]
        edge_vec_o = data["ligand"].pos[src.long()] - data["protein"].pos_O[dst.long()]
        edge_vec_n = data["ligand"].pos[src.long()] - data["protein"].pos_N[dst.long()]
        edge_attr = torch.cat(
            [
                self.cross_rbf(edge_vec.norm(dim=-1)),
                self.cross_rbf(edge_vec_cb.norm(dim=-1)),
                self.cross_rbf(edge_vec_c.norm(dim=-1)),
                self.cross_rbf(edge_vec_o.norm(dim=-1)),
                self.cross_rbf(edge_vec_n.norm(dim=-1)),
            ],
            dim=-1,
        )
        return cross_idx, self.cross_attr_embedder(edge_attr)


class PiFoldEmbedder(nn.Module):
    def __init__(self, args, device, dim=None):
        super().__init__()
        self.args = args
        self.device = device
        fold_dim = args.fold_dim if dim is None else dim

        self.top_k = args.k_neighbors
        self.num_rbf = 16

        self.virtual_atoms = nn.Parameter(torch.rand(self.args.virtual_num, 3, device=self.device))

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num > 0:
                pair_num += self.args.virtual_num * (self.args.virtual_num - 1)
            node_in += pair_num * self.num_rbf
        if self.args.node_angle:
            node_in += 12
        if self.args.node_direct:
            node_in += 9

        edge_in = 0
        if self.args.edge_dist:
            pair_num = 16
            if self.args.virtual_num > 0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num * (self.args.virtual_num - 1)
            edge_in += pair_num * self.num_rbf
        if self.args.edge_angle:
            edge_in += 4
        if self.args.edge_direct:
            edge_in += 12

        self.node_embedding = nn.Linear(node_in, fold_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in, fold_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(fold_dim)
        self.norm_edges = nn.BatchNorm1d(fold_dim)

        self.W_v = nn.Sequential(
            nn.Linear(fold_dim, fold_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fold_dim),
            nn.Linear(fold_dim, fold_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(fold_dim),
            nn.Linear(fold_dim, fold_dim, bias=True),
        )
        self.W_e = nn.Linear(fold_dim, fold_dim, bias=True)
        self._init_params()

    def forward(self, data):
        start = time.time()
        unbatched_pos = unbatch(data["protein"].pos, data["protein"].batch)
        pos_N = data["protein"].pos_N
        pos_C = data["protein"].pos_C
        pos_O = data["protein"].pos_O
        unbatched_pos_N = unbatch(pos_N, data["protein"].batch)
        unbatched_pos_C = unbatch(pos_C, data["protein"].batch)
        unbatched_pos_O = unbatch(pos_O, data["protein"].batch)

        lengths = np.array([len(b) for b in unbatched_pos], dtype=np.int32)
        L_max = max(lengths)
        B = len(unbatched_pos)
        X = torch.zeros([B, L_max, 4, 3], device=self.device)
        for i, (pos_Ca, pos_C, pos_N, pos_O) in enumerate(
            zip(unbatched_pos, unbatched_pos_C, unbatched_pos_N, unbatched_pos_O)
        ):
            x = torch.stack([pos_N, pos_Ca, pos_C, pos_O], 1)
            l = len(pos_Ca)  # noqa: E741
            x_pad = torch.from_numpy(
                np.pad(
                    x.detach().cpu().numpy(),
                    [[0, L_max - l], [0, 0], [0, 0]],
                    "constant",
                    constant_values=(np.nan,),
                )
            ).to(self.device)
            X[i, :, :, :] = x_pad

        mask = torch.isfinite(torch.sum(X, (2, 3))).float()
        numbers = torch.sum(mask, axis=1).long()
        pos_new = torch.zeros_like(X) + torch.nan
        for i, n in enumerate(numbers):
            pos_new[i, :n, ::] = X[i][mask[i] == 1]
        pos = pos_new
        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X, (2, 3))).float()
        pos[isnan] = 0.0
        data.logs["padding_time"] = time.time() - start

        mask_bool = mask == 1
        B, N, _, _ = pos.shape
        X_ca = pos[:, :, 1, :]
        D_neighbors, rec_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), rec_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(
            -1, x.shape[-1]
        )  # noqa: E731
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(
            -1, x.shape[-1]
        )  # noqa: E731

        V_angles = _dihedrals(pos, 0)
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(pos, rec_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        atom_N = pos[:, :, 0, :]
        atom_Ca = pos[:, :, 1, :]
        atom_C = pos[:, :, 2, :]
        atom_O = pos[:, :, 3, :]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        atoms = {"N": atom_N, "Ca": atom_Ca, "C": atom_C, "O": atom_O}
        if self.args.virtual_num > 0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                atoms["v" + str(i)] = (
                    virtual_atoms[i][0] * a
                    + virtual_atoms[i][1] * b
                    + virtual_atoms[i][2] * c
                    + 1 * atom_Ca
                )

        node_list = ["Ca-N", "Ca-C", "Ca-O", "N-C", "N-O", "O-C"]
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split("-")
            node_dist.append(
                node_mask_select(_get_rbf(atoms[atom1], atoms[atom2], None, self.num_rbf).squeeze())
            )

        if self.args.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                for j in range(0, i):
                    node_dist.append(
                        node_mask_select(
                            _get_rbf(
                                atoms["v" + str(i)], atoms["v" + str(j)], None, self.num_rbf
                            ).squeeze()
                        )
                    )
                    node_dist.append(
                        node_mask_select(
                            _get_rbf(
                                atoms["v" + str(j)], atoms["v" + str(i)], None, self.num_rbf
                            ).squeeze()
                        )
                    )
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

        pair_lst = [
            "Ca-Ca", "Ca-C", "C-Ca", "Ca-N", "N-Ca", "Ca-O", "O-Ca", "C-C",
            "C-N", "N-C", "C-O", "O-C", "N-N", "N-O", "O-N", "O-O",
        ]  # fmt: skip

        edge_dist = []
        for pair in pair_lst:
            atom1, atom2 = pair.split("-")
            rbf = _get_rbf(atoms[atom1], atoms[atom2], rec_idx, self.num_rbf)
            edge_dist.append(edge_mask_select(rbf))

        if self.args.virtual_num > 0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(
                    edge_mask_select(
                        _get_rbf(atoms["v" + str(i)], atoms["v" + str(i)], rec_idx, self.num_rbf)
                    )
                )
                for j in range(0, i):
                    edge_dist.append(
                        edge_mask_select(
                            _get_rbf(
                                atoms["v" + str(i)], atoms["v" + str(j)], rec_idx, self.num_rbf
                            )
                        )
                    )
                    edge_dist.append(
                        edge_mask_select(
                            _get_rbf(
                                atoms["v" + str(j)], atoms["v" + str(i)], rec_idx, self.num_rbf
                            )
                        )
                    )

        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        rec_na = []
        if self.args.node_dist:
            rec_na.append(V_dist)
        if self.args.node_angle:
            rec_na.append(V_angles)
        if self.args.node_direct:
            rec_na.append(V_direct)

        rec_ea = []
        if self.args.edge_dist:
            rec_ea.append(E_dist)
        if self.args.edge_angle:
            rec_ea.append(E_angles)
        if self.args.edge_direct:
            rec_ea.append(E_direct)

        rec_na = torch.cat(rec_na, dim=-1)
        rec_ea = torch.cat(rec_ea, dim=-1)

        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B, 1, 1) + rec_idx
        src = torch.masked_select(src, mask_attend).view(1, -1)
        dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(1, -1, 1).expand_as(
            mask_attend
        )
        dst = torch.masked_select(dst, mask_attend).view(1, -1)
        rec_idx = torch.cat((dst, src), dim=0).long()

        sparse_idx = mask.nonzero()
        batch_id = sparse_idx[:, 0]

        rec_na = self.W_v(self.norm_nodes(self.node_embedding(rec_na)))
        rec_ea = self.W_e(self.norm_edges(self.edge_embedding(rec_ea)))
        assert all(batch_id == data["protein"].batch)
        return rec_na, rec_ea, rec_idx

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == "virtual_atoms":
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = (1.0 - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(
            D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx


class FlowSiteModel(nn.Module):
    """FlowSiteModel forward path with use_tfn=False, use_inv=True (real repo, verbatim invariant branch)."""

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        fold_dim = args.fold_dim
        num_inv_layers = args.num_inv_layers
        atom_feature_dims, edge_feature_dims, lig_bond_feature_dims, rec_feature_dims = (
            get_feature_dims()
        )

        assert args.use_tfn or args.use_inv, (
            "Must use at least one of tfn or inv, otherwise this model will do nothing"
        )
        assert not (args.use_tfn and args.ignore_lig), (
            "Tensorfield always uses lig so ignore_lig does not work with it"
        )
        if not args.ignore_lig or args.lig2d_mpnn:
            self.lig_node_embedder = nn.Sequential(
                Encoder(emb_dim=fold_dim, feature_dims=atom_feature_dims),
                nn.ReLU(),
                Linear(fold_dim, fold_dim, init="final" if args.fancy_init else "default"),
            )

            self.lig_edge_embedder = nn.Sequential(
                Encoder(emb_dim=fold_dim, feature_dims=lig_bond_feature_dims),
                nn.ReLU(),
                Linear(fold_dim, fold_dim, init="final" if args.fancy_init else "default"),
            )
        if not args.ignore_lig:
            self.lig_edge_builder = LigEdgeBuilder(args, device)
            self.cross_edge_builder = CrossEdgeBuilder(args, device)

        if self.args.use_inv:
            self.inv_embedder = PiFoldEmbedder(args, device)
            self.inv_layers = nn.Sequential(
                *[
                    InvariantLayer(args, device, update_edges=(i + 1 < num_inv_layers))
                    for i in range(num_inv_layers)
                ]
            )

        self.decoder = nn.Linear(fold_dim, len(_RESIDUES_CANONICAL))
        if self.args.num_angle_pred > 0:
            self.angle_linear = Linear(fold_dim, fold_dim)
            self.angle_linear_skip = Linear(fold_dim, fold_dim)
            self.angle_decoder1 = nn.Sequential(
                Linear(fold_dim, fold_dim), nn.ReLU(), Linear(fold_dim, fold_dim), nn.ReLU()
            )
            self.angle_decoder2 = nn.Sequential(
                Linear(fold_dim, fold_dim), nn.ReLU(), Linear(fold_dim, fold_dim, nn.ReLU())
            )
            self.angle_predictor = nn.Linear(fold_dim, self.args.num_angle_pred * 2)

        assert not ((args.time_condition_inv or args.time_condition_tfn) and args.ignore_lig)

    def forward(self, data, x_self=None, x_prior=None):
        if self.args.ignore_lig:
            lig_na, lig_ea, lig_idx, cross_idx, cross_ea = None, None, None, None, None
        else:
            lig_na = self.lig_node_embedder(data["ligand"].feat)
            lig_ea = self.lig_edge_embedder(data["ligand", "bond_edge", "ligand"].edge_attr)
            lig_idx, lig_ea = self.lig_edge_builder(data, lig_ea)
            cross_idx, cross_ea = self.cross_edge_builder(data)

        rec_na, rec_ea, rec_idx = self.inv_embedder(data)

        temb_inv = None
        for inv_layer in self.inv_layers:
            rec_na, rec_ea, lig_na, lig_ea, cross_ea = inv_layer(
                data,
                rec_idx,
                rec_na,
                rec_ea,
                lig_idx,
                lig_na,
                lig_ea,
                cross_idx,
                cross_ea,
                temb_inv,
            )

        logits = self.decoder(rec_na)
        if self.args.num_angle_pred > 0:
            angle_na = self.angle_linear(rec_na) + self.angle_linear_skip(rec_na)
            angle_na = angle_na + self.angle_decoder1(angle_na)
            angle_na = angle_na + self.angle_decoder2(angle_na)
            angles = self.angle_predictor(angle_na).reshape(-1, self.args.num_angle_pred, 2)
        else:
            angles = None

        return logits, torch.stack([data["ligand"].pos]), angles


# ---------------------------------------------------------------------------
# Staging build/example helpers: a tiny synthetic protein-pocket + ligand
# HeteroData complex matching the real repo's node/edge-type schema (built
# from scratch here since the real featurization pipeline needs rdkit/
# Bio.PDB structure files, which is a data-preprocessing concern, not
# architecture). use_tfn=False / use_inv=True is the invariant-GNN-only
# configuration exercised; virtual_num=0 keeps the PiFoldEmbedder feature
# arithmetic simple for a minimal random-init trace.
# ---------------------------------------------------------------------------


def _build_args():
    return SimpleNamespace(
        fold_dim=16,
        num_inv_layers=2,
        use_tfn=False,
        use_inv=True,
        ignore_lig=False,
        lig2d_mpnn=False,
        fancy_init=False,
        self_condition_inv=False,
        time_condition_inv=False,
        time_condition_tfn=False,
        time_condition_repeat=False,
        inv_dropout=0.0,
        node_context=False,
        edge_context=False,
        inv_straight_combine=False,
        node_dist=1,
        node_angle=1,
        node_direct=1,
        edge_dist=1,
        edge_angle=1,
        edge_direct=1,
        virtual_num=0,
        k_neighbors=4,
        protein_radius=15.0,
        radius_emb_dim=8,
        lig_radius=15.0,
        cross_radius=20.0,
        num_angle_pred=0,
    )


def build_flowsite():
    args = _build_args()
    return FlowSiteModel(args, device="cpu")


def example_input_flowsite():
    torch.manual_seed(0)
    # PiFoldEmbedder pads+stacks the receptor batch into a dense [B, L_max, 4, 3]
    # tensor and calls .squeeze() on per-pair RBF tensors -- a real-repo quirk
    # that only round-trips correctly for B >= 2 (B=1 makes .squeeze() also
    # collapse the batch dim). We use 2 receptor complexes of equal residue
    # count so the real code path is exercised faithfully without hitting
    # that pre-existing single-example edge case.
    n_complexes = 2
    n_res = 10  # receptor residues per complex
    n_atoms = 6  # ligand heavy atoms per complex
    n_bonds = 6  # ligand bonds per complex (directed, small ring-ish graph)

    data = HeteroData()
    data.logs = {}

    # protein backbone atom positions (N, CA, C, O per residue)
    ca = torch.randn(n_complexes * n_res, 3) * 5
    data["protein"].pos = ca
    data["protein"].pos_N = ca + torch.randn(n_complexes * n_res, 3) * 0.3
    data["protein"].pos_C = ca + torch.randn(n_complexes * n_res, 3) * 0.3
    data["protein"].pos_O = ca + torch.randn(n_complexes * n_res, 3) * 0.3
    data["protein"].pos_Cb = ca + torch.randn(n_complexes * n_res, 3) * 0.3
    data["protein"].batch = torch.repeat_interleave(torch.arange(n_complexes), n_res)
    # dense radius_graph placeholder edge_index (unused directly by
    # PiFoldEmbedder, which builds its own top-k neighbor graph, but kept
    # for schema completeness matching the real repo's HeteroData).
    src, dst = torch.meshgrid(torch.arange(n_res), torch.arange(n_res), indexing="ij")
    mask = src != dst
    data["protein", "radius_graph", "protein"].edge_index = torch.stack(
        [src[mask], dst[mask]], dim=0
    )

    # ligand atom features (11 categorical columns matching atom_feature_dims)
    feat_dims = [len(_ATOMIC_NUM), len(_CHIRALITY), len(_DEGREE), len(_NUMRING), len(_IMPLICIT_VALENCE),
                 len(_FORMAL_CHARGE), len(_NUMH), len(_HYBRIDIZATION), len(_IS_AROMATIC), len(_IS_IN_RING5),
                 len(_IS_IN_RING6)]  # fmt: skip
    n_lig_atoms = n_complexes * n_atoms
    lig_feat = torch.stack([torch.randint(0, d, (n_lig_atoms,)) for d in feat_dims], dim=1)
    data["ligand"].feat = lig_feat
    data["ligand"].pos = torch.randn(n_lig_atoms, 3) * 2
    data["ligand"].batch = torch.repeat_interleave(torch.arange(n_complexes), n_atoms)

    lig_src = torch.repeat_interleave(torch.arange(n_complexes), n_bonds) * n_atoms + torch.randint(
        0, n_atoms, (n_complexes * n_bonds,)
    )
    lig_dst = torch.repeat_interleave(torch.arange(n_complexes), n_bonds) * n_atoms + torch.randint(
        0, n_atoms, (n_complexes * n_bonds,)
    )
    data["ligand", "bond_edge", "ligand"].edge_index = torch.stack([lig_src, lig_dst], dim=0)
    bond_feat_dims = [len(_BOND_TYPE), len(_BOND_STEREO), len(_IS_CONJUGATED)]
    data["ligand", "bond_edge", "ligand"].edge_attr = torch.stack(
        [torch.randint(0, d, (n_complexes * n_bonds,)) for d in bond_feat_dims], dim=1
    )

    return (data,)


MENAGERIE_ENTRIES = [
    ("FlowSite", "build_flowsite", "example_input_flowsite", 2024, "vendored-pytorch"),
]
