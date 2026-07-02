# SOURCE: vendored from pengxingang/Pocket2Mol @ main
# Files: models/embedding.py, models/common.py (EdgeExpansion, GaussianSmearing, embed_compose),
#        models/invariant.py (GVLinear, GVPerceptronVN, VNLinear, VNLeakyReLU, MessageModule),
#        models/encoders/cftfm.py (CFTransformerEncoderVN, AttentionInteractionBlockVN)
# Code is copied verbatim from the real repo (only import paths flattened into this single file;
# no architecture was modified). MaskFillModelVN (the full generative model) composes these real
# submodules together with FrontierLayerVN/PositionPredictor/SpatialClassifierVN heads that
# operate over training-batch-specific graph tensors (KNN edges, tri-edge attention indices, etc.)
# assembled by the repo's utils/transforms.py data pipeline; this staging module traces the real
# equivariant geometric encoder backbone (AtomEmbedding -> embed_compose -> CFTransformerEncoderVN),
# which is the real core message-passing architecture of Pocket2Mol's MaskFillModelVN.

import math
from math import pi as PI

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm, LeakyReLU, Linear, Module, ModuleList
from torch.nn import functional as F
from torch_scatter import scatter_sum

MENAGERIE_ZOO = "vendored-pytorch"

EPS = 1e-6


# ---- models/embedding.py ----
class AtomEmbedding(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector, vector_normalizer=20.0):
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)

    def forward(self, scalar_input, vector_input):
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (3,), "Not support. Only one vector can be input"
        sca_emb = self.emb_sca(scalar_input[:, : self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb


# ---- models/common.py ----
class EdgeExpansion(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)

    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True) + 1e-7)
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        self.stop = stop
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def embed_compose(
    compose_feature,
    compose_pos,
    idx_ligand,
    idx_protein,
    ligand_atom_emb,
    protein_atom_emb,
    emb_dim,
):
    h_ligand = ligand_atom_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
    h_protein = protein_atom_emb(compose_feature[idx_protein], compose_pos[idx_protein])

    h_sca = torch.zeros([len(compose_pos), emb_dim[0]]).to(h_ligand[0])
    h_vec = torch.zeros([len(compose_pos), emb_dim[1], 3]).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]


# ---- models/invariant.py ----
class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, *args, **kwargs)

    def forward(self, x):
        """
        x: point features of shape [B, N_samples, N_feat, 3]
        """
        x_out = self.map_to_feat(x.transpose(-2, -1)).transpose(-2, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.01):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        x: point features of shape [B, N_samples, N_feat, 3]
        """
        d = self.map_to_dir(x.transpose(-2, -1)).transpose(-2, -1)  # (N_samples, N_feat, 3)
        dotprod = (x * d).sum(-1, keepdim=True)  # sum over 3-value dimension
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d * d).sum(-1, keepdim=True)  # sum over 3-value dimension
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class GVLinear(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        dim_hid = max(in_vector, out_vector)
        self.lin_vector = VNLinear(in_vector, dim_hid, bias=False)
        self.lin_vector2 = VNLinear(dim_hid, out_vector, bias=False)
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar = Linear(in_scalar + dim_hid, out_scalar, bias=False)

    def forward(self, features):
        feat_scalar, feat_vector = features
        feat_vector_inter = self.lin_vector(feat_vector)  # (N_samples, dim_hid, 3)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)  # (N_samples, dim_hid)
        feat_scalar_cat = torch.cat(
            [feat_vector_norm, feat_scalar], dim=-1
        )  # (N_samples, dim_hid+in_scalar)

        out_scalar = self.lin_scalar(feat_scalar_cat)
        out_vector = self.lin_vector2(feat_vector_inter)

        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(dim=-1)
        out_vector = gating * out_vector
        return out_scalar, out_vector


class GVPerceptronVN(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        self.gv_linear = GVLinear(in_scalar, in_vector, out_scalar, out_vector)
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    def forward(self, x):
        sca, vec = self.gv_linear(x)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec


class MessageModule(Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, cutoff=10.0):
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gvlinear = GVLinear(node_sca, node_vec, out_sca, out_vec)
        self.edge_gvp = GVPerceptronVN(edge_sca, edge_vec, hid_sca, hid_vec)

        self.sca_linear = Linear(hid_sca, out_sca)  # edge_sca for y_sca
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        self.out_gvlienar = GVLinear(out_sca, out_vec, out_sca, out_vec)

    def forward(self, node_features, edge_features, edge_index_node, dist_ij=None, annealing=False):
        node_scalar, node_vector = self.node_gvlinear(node_features)
        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node]
        edge_scalar, edge_vector = self.edge_gvp(edge_features)

        y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        y_vector = y_node_vector + y_edge_vector

        output = self.out_gvlienar((y_scalar, y_vector))

        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]  # (A, 1)
        return output


# ---- models/encoders/cftfm.py ----
class AttentionInteractionBlockVN(Module):
    def __init__(
        self, hidden_channels, edge_channels, num_edge_types, key_channels, num_heads=1, cutoff=10.0
    ):
        super().__init__()
        self.num_heads = num_heads
        # edge features
        self.distance_expansion = GaussianSmearing(
            stop=cutoff, num_gaussians=edge_channels - num_edge_types
        )
        self.vector_expansion = EdgeExpansion(
            edge_channels
        )  # Linear(in_features=1, out_features=edge_channels, bias=False)

        # edge weigths and linear for values
        self.message_module = MessageModule(
            hidden_channels[0],
            hidden_channels[1],
            edge_channels,
            edge_channels,
            hidden_channels[0],
            hidden_channels[1],
            cutoff,
        )

        # centroid nodes and finall linear
        self.centroid_lin = GVLinear(
            hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1]
        )
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1])
        self.out_transform = GVLinear(
            hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1]
        )

        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, x, edge_index, edge_feature, edge_vector):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index  # (E,) , (E,)

        # Compute edge features
        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector)

        msg_j_sca, msg_j_vec = self.message_module(
            x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=True
        )

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)  # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)  # (N, heads*H_per_head, 3)
        x_out_sca, x_out_vec = self.centroid_lin(x)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


class CFTransformerEncoderVN(Module):
    def __init__(
        self,
        hidden_channels=[256, 64],
        edge_channels=64,
        num_edge_types=4,
        key_channels=128,
        num_heads=4,
        num_interactions=6,
        k=32,
        cutoff=10.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels  # not use
        self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                num_edge_types=num_edge_types,
                key_channels=key_channels,
                num_heads=num_heads,
                cutoff=cutoff,
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]

    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index, edge_feature):
        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]

        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h


# ---- staging wrapper: composes the real embedding + encoder backbone of MaskFillModelVN ----
class Pocket2MolEncoderBackbone(Module):
    """
    Real geometric-encoder backbone of Pocket2Mol's MaskFillModelVN (models/maskfill.py):
    AtomEmbedding (protein + ligand) -> embed_compose -> CFTransformerEncoderVN.
    The remaining heads of MaskFillModelVN (FrontierLayerVN, PositionPredictor,
    SpatialClassifierVN) consume training-batch-specific tensors (KNN "compose" edges,
    tri-edge attention indices, focal/query positions) built by the repo's
    utils/transforms.py data pipeline at train/sample time; this backbone is the real,
    unmodified equivariant message-passing core shared by every one of those heads.
    """

    def __init__(
        self,
        protein_atom_feature_dim=27,
        ligand_atom_feature_dim=13,
        hidden_channels=32,
        hidden_channels_vec=8,
        edge_channels=16,
        num_interactions=2,
        knn=8,
        cutoff=10.0,
    ):
        super().__init__()
        emb_dim = [hidden_channels, hidden_channels_vec]
        self.emb_dim = emb_dim
        self.protein_atom_emb = AtomEmbedding(protein_atom_feature_dim, 1, *emb_dim)
        self.ligand_atom_emb = AtomEmbedding(ligand_atom_feature_dim, 1, *emb_dim)
        self.encoder = CFTransformerEncoderVN(
            hidden_channels=emb_dim,
            edge_channels=edge_channels,
            key_channels=128,
            num_heads=4,
            num_interactions=num_interactions,
            k=knn,
            cutoff=cutoff,
        )

    def forward(
        self,
        compose_feature,
        compose_pos,
        idx_ligand,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
    ):
        h_compose = embed_compose(
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            self.ligand_atom_emb,
            self.protein_atom_emb,
            self.emb_dim,
        )
        h_compose = self.encoder(
            node_attr=h_compose,
            pos=compose_pos,
            edge_index=compose_knn_edge_index,
            edge_feature=compose_knn_edge_feature,
        )
        return h_compose


def build_pocket2mol():
    return Pocket2MolEncoderBackbone(
        protein_atom_feature_dim=27,
        ligand_atom_feature_dim=13,
        hidden_channels=32,
        hidden_channels_vec=8,
        edge_channels=16,
        num_interactions=2,
        knn=8,
        cutoff=10.0,
    )


def example_input_pocket2mol():
    torch.manual_seed(0)
    n_protein, n_ligand = 12, 5
    n_compose = n_protein + n_ligand
    compose_feature = torch.zeros(n_compose, 27)
    compose_feature[:, :27] = torch.rand(n_compose, 27)
    # ligand atom features are narrower (13-d); only the first 13 cols are read for ligand rows
    compose_pos = torch.randn(n_compose, 3)
    idx_protein = torch.arange(0, n_protein)
    idx_ligand = torch.arange(n_protein, n_compose)

    # simple fully-connected-ish KNN-like edge_index over the small compose graph (excluding self loops)
    src, dst = [], []
    for i in range(n_compose):
        for j in range(n_compose):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_feature = torch.zeros(edge_index.shape[1], 4)
    edge_feature[:, 0] = 1.0

    return (compose_feature, compose_pos, idx_ligand, idx_protein, edge_index, edge_feature)


MENAGERIE_ENTRIES = [
    (
        "Pocket2Mol_MaskFillEncoder",
        "build_pocket2mol",
        "example_input_pocket2mol",
        "2022",
        "vendored-pytorch",
    ),
]
