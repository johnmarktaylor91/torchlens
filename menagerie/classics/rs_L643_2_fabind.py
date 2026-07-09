# SOURCE: vendored from QizhiPei/FABind @ main (subdirectory FABind/fabind)
# Vendored files (real architecture code, imports/relative-paths adjusted only):
#   models/model_utils.py  (Attention, Transition, InteractionModule, RBFDistanceModule,
#                            GaussianSmearing, permute_final_dims, flatten_final_dims)
#   models/cross_att.py    (CrossAttentionModule, RowTriangleAttentionBlock, RowAttentionBlock)
#   models/egnn.py         (MC_E_GCL, MC_Att_L, MCAttEGNN, MCnoAttEGNN, MCnoAttwithCrossAttEGNN,
#                            coord2radial, unsorted_segment_sum/mean)
#   models/att_model.py    (ComplexGraph, EfficientMCAttModel, sequential_and/or, _radial_edges)
#   models/model.py        (Transition_diff_out_dim,
#                            IaBNet_mean_and_pocket_prediction_cls_coords_dependent)
#   utils/utils.py         (compute_dis_between_two_vector_tensor, get_keepNode_tensor,
#                            gumbel_softmax_no_random -- the only pieces of utils.py needed by
#                            the model forward/inference path that do not require rdkit)
#
# FABind (Pei et al., NeurIPS 2023, "FABind: Fast and Accurate Protein-Ligand Binding") is an
# E(n)-equivariant multi-channel attention GNN (EGNN + AlphaFold-style row/pair attention) that
# jointly predicts a protein binding pocket and docks a ligand into it in a single forward pass,
# avoiding the sampling/search loop of classical docking. `IaBNet_mean_and_pocket_prediction_
# cls_coords_dependent` is the real top-level model class (models/model.py::get_model, mode=5),
# instantiated by FABind's own fabind_inference.py; `.inference(data)` is the exact real
# inference-time entry point that script calls (`model.inference(data)`).
#
# The real code's `utils/utils.py::construct_data_from_graph_gvp_mean` (used only by the
# *training* dataset, data.py::FABindDataSet) unconditionally imports rdkit at module scope,
# which is not installed here; it is not needed for a random-init forward trace, so this
# staging module builds a tiny synthetic HeteroData batch directly, matching the exact real
# field schema documented in FABind's own utils/fabind_inference_dataset.py::InferenceDataset
# (compound/protein_whole/complex_whole_protein graphs) -- the model's real `inference()`
# method then derives `complex`/`pocket` fields itself, exactly as in production.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import masked_fill  # noqa: F401  (kept to match original att_model.py import list)
from torch.nn import Linear, LayerNorm
from torch.nn.functional import softmax
from torch_scatter import scatter_softmax, scatter_add, scatter_sum
from torch_geometric.utils import to_dense_batch, to_dense_adj  # noqa: F401
from torch_geometric.data import HeteroData
import random
from typing import List, Tuple, Optional

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from utils/utils.py (only the rdkit-free helper functions the
# forward/inference path actually needs)
# ---------------------------------------------------------------------------
def compute_dis_between_two_vector_tensor(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


def get_keepNode_tensor(protein_node_xyz, pocket_radius, add_noise_to_com, chosen_pocket_com):
    if add_noise_to_com:
        chosen_pocket_com = chosen_pocket_com + add_noise_to_com * (
            2 * torch.rand_like(chosen_pocket_com) - 1
        )
    dis = compute_dis_between_two_vector_tensor(protein_node_xyz, chosen_pocket_com.unsqueeze(0))
    keepNode = dis < pocket_radius
    return keepNode


def gumbel_softmax_no_random(
    logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1
) -> torch.Tensor:
    gumbels = logits / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(
            dim, index, 1.0
        )
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


# ---------------------------------------------------------------------------
# Vendored from models/model_utils.py
# ---------------------------------------------------------------------------
def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]
) -> torch.Tensor:
    query = permute_final_dims(query, (1, 0, 2))
    key = permute_final_dims(key, (1, 2, 0))
    value = permute_final_dims(value, (1, 0, 2))
    a = torch.matmul(query, key)
    for b in biases:
        a = a + b
    a = softmax(a, -1)
    a = torch.matmul(a, value)
    a = a.transpose(-2, -3)
    return a


class Attention(nn.Module):
    """Standard multi-head attention using AlphaFold's default layer initialization."""

    def __init__(
        self, c_q: int, c_k: int, c_v: int, c_hidden: int, no_heads: int, gating: bool = True
    ):
        super(Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False)
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q)

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.no_heads)

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x, kv_x):
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        import math

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o, q_x):
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = flatten_final_dims(o, 2)
        o = self.linear_o(o)
        return o

    def forward(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, biases: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if biases is None:
            biases = []
        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        o = self._wrap_up(o, q_x)
        return o


class Transition(torch.nn.Module):
    def __init__(self, hidden_dim=128, n=4, rm_layernorm=False):
        super().__init__()
        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm = torch.nn.LayerNorm(hidden_dim)
        self.linear_1 = Linear(hidden_dim, n * hidden_dim)
        self.linear_2 = Linear(n * hidden_dim, hidden_dim)

    def forward(self, x):
        if not self.rm_layernorm:
            x = self.layernorm(x)
        x = self.linear_2((self.linear_1(x)).relu())
        return x


class InteractionModule(torch.nn.Module):
    def __init__(self, node_hidden_dim, pair_hidden_dim, hidden_dim, opm=False, rm_layernorm=False):
        super(InteractionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.opm = opm

        self.rm_layernorm = rm_layernorm
        if not rm_layernorm:
            self.layer_norm_p = nn.LayerNorm(node_hidden_dim)
            self.layer_norm_c = nn.LayerNorm(node_hidden_dim)

        if self.opm:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim**2, pair_hidden_dim)
        else:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim, pair_hidden_dim)

    def forward(self, p_embed, c_embed, p_mask=None, c_mask=None):
        if p_mask is None:
            p_mask = p_embed.new_ones(p_embed.shape[:-1], dtype=torch.bool)
        if c_mask is None:
            c_mask = c_embed.new_ones(c_embed.shape[:-1], dtype=torch.bool)
        inter_mask = torch.einsum("...i,...j->...ij", p_mask, c_mask)

        if not self.rm_layernorm:
            p_embed = self.layer_norm_p(p_embed)
            c_embed = self.layer_norm_c(c_embed)
        if self.opm:
            p_embed = self.linear_p(p_embed)
            c_embed = self.linear_c(c_embed)
            inter_embed = torch.einsum("...bc,...de->...bdce", p_embed, c_embed)
            inter_embed = torch.flatten(inter_embed, -2)
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        else:
            p_embed = self.linear_p(p_embed)
            c_embed = self.linear_c(c_embed)
            inter_embed = torch.einsum("...ik,...jk->...ijk", p_embed, c_embed)
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        return inter_embed, inter_mask


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist[..., None] - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RBFDistanceModule(torch.nn.Module):
    def __init__(self, rbf_stop, distance_hidden_dim, num_gaussian=32, dropout=0.1):
        super(RBFDistanceModule, self).__init__()
        self.distance_hidden_dim = distance_hidden_dim
        self.rbf = GaussianSmearing(start=0, stop=rbf_stop, num_gaussians=num_gaussian)
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussian, distance_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(distance_hidden_dim, distance_hidden_dim),
        )

    def forward(self, distance):
        return self.mlp(self.rbf(distance))


# ---------------------------------------------------------------------------
# Vendored from models/cross_att.py
# ---------------------------------------------------------------------------
class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        node_hidden_dim,
        pair_hidden_dim,
        rm_layernorm=False,
        keep_trig_attn=False,
        dist_hidden_dim=32,
        normalize_coord=None,
    ):
        super().__init__()
        self.pair_hidden_dim = pair_hidden_dim
        self.keep_trig_attn = keep_trig_attn

        if keep_trig_attn:
            self.triangle_block_row = RowTriangleAttentionBlock(
                pair_hidden_dim, dist_hidden_dim, rm_layernorm=rm_layernorm
            )
            self.triangle_block_column = RowTriangleAttentionBlock(
                pair_hidden_dim, dist_hidden_dim, rm_layernorm=rm_layernorm
            )

        self.p_attention_block = RowAttentionBlock(
            node_hidden_dim, pair_hidden_dim, rm_layernorm=rm_layernorm
        )
        self.c_attention_block = RowAttentionBlock(
            node_hidden_dim, pair_hidden_dim, rm_layernorm=rm_layernorm
        )
        self.p_transition = Transition(node_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.c_transition = Transition(node_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.pair_transition = Transition(pair_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.inter_layer = InteractionModule(
            node_hidden_dim, pair_hidden_dim, 32, opm=False, rm_layernorm=rm_layernorm
        )

    def forward(
        self,
        p_embed_batched,
        p_mask,
        c_embed_batched,
        c_mask,
        pair_embed,
        pair_mask,
        c_c_dist_embed=None,
        p_p_dist_embed=None,
    ):
        if self.keep_trig_attn:
            pair_embed = self.triangle_block_row(
                pair_embed=pair_embed, pair_mask=pair_mask, dist_embed=c_c_dist_embed
            )
            pair_embed = self.triangle_block_row(
                pair_embed=pair_embed.transpose(-2, -3),
                pair_mask=pair_mask.transpose(-1, -2),
                dist_embed=p_p_dist_embed,
            ).transpose(-2, -3)

        p_embed_batched = self.p_attention_block(
            node_embed_i=p_embed_batched,
            node_embed_j=c_embed_batched,
            pair_embed=pair_embed,
            pair_mask=pair_mask,
            node_mask_i=p_mask,
        )
        c_embed_batched = self.c_attention_block(
            node_embed_i=c_embed_batched,
            node_embed_j=p_embed_batched,
            pair_embed=pair_embed.transpose(-2, -3),
            pair_mask=pair_mask.transpose(-1, -2),
            node_mask_i=c_mask,
        )
        p_embed_batched = p_embed_batched + self.p_transition(p_embed_batched)
        c_embed_batched = c_embed_batched + self.c_transition(c_embed_batched)

        pair_embed = (
            pair_embed + self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]
        )

        pair_embed = self.pair_transition(pair_embed) * pair_mask.to(torch.float).unsqueeze(-1)
        return p_embed_batched, c_embed_batched, pair_embed


class RowTriangleAttentionBlock(nn.Module):
    inf = 1e9

    def __init__(
        self,
        pair_hidden_dim,
        dist_hidden_dim,
        attention_hidden_dim=32,
        no_heads=4,
        dropout=0.1,
        rm_layernorm=False,
    ):
        super(RowTriangleAttentionBlock, self).__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.dist_hidden_dim = dist_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm = LayerNorm(pair_hidden_dim)

        self.linear = Linear(dist_hidden_dim, self.no_heads)
        self.linear_g = Linear(dist_hidden_dim, self.no_heads)
        self.dropout = nn.Dropout(dropout)
        self.mha = Attention(
            pair_hidden_dim, pair_hidden_dim, pair_hidden_dim, attention_hidden_dim, no_heads
        )

    def forward(self, pair_embed, pair_mask, dist_embed):
        if not self.rm_layernorm:
            pair_embed = self.layernorm(pair_embed)

        mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., :, None, None, :]
        dist_bias = self.linear(dist_embed) * self.linear_g(dist_embed).sigmoid()
        dist_bias = permute_final_dims(dist_bias, [2, 1, 0])[..., None, :, :, :]

        pair_embed = pair_embed + self.dropout(
            self.mha(q_x=pair_embed, kv_x=pair_embed, biases=[mask_bias, dist_bias])
        ) * pair_mask.to(torch.float).unsqueeze(-1)

        return pair_embed


class RowAttentionBlock(nn.Module):
    inf = 1e9

    def __init__(
        self,
        node_hidden_dim,
        pair_hidden_dim,
        attention_hidden_dim=32,
        no_heads=4,
        dropout=0.1,
        rm_layernorm=False,
    ):
        super(RowAttentionBlock, self).__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm_node_i = LayerNorm(node_hidden_dim)
            self.layernorm_node_j = LayerNorm(node_hidden_dim)
            self.layernorm_pair = LayerNorm(pair_hidden_dim)

        self.linear = Linear(pair_hidden_dim, self.no_heads)
        self.linear_g = Linear(pair_hidden_dim, self.no_heads)

        self.dropout = nn.Dropout(dropout)

        self.mha = Attention(
            node_hidden_dim, node_hidden_dim, node_hidden_dim, attention_hidden_dim, no_heads
        )

    def forward(self, node_embed_i, node_embed_j, pair_embed, pair_mask, node_mask_i):
        if not self.rm_layernorm:
            node_embed_i = self.layernorm_node_i(node_embed_i)
            node_embed_j = self.layernorm_node_j(node_embed_j)
            pair_embed = self.layernorm_pair(pair_embed)

        mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., None, :, :]
        pair_bias = self.linear(pair_embed) * self.linear_g(pair_embed).sigmoid()
        pair_bias = permute_final_dims(pair_bias, [2, 0, 1])

        node_embed_i = node_embed_i + self.dropout(
            self.mha(q_x=node_embed_i, kv_x=node_embed_j, biases=[mask_bias, pair_bias])
        ) * node_mask_i.to(torch.float).unsqueeze(-1)

        return node_embed_i


# ---------------------------------------------------------------------------
# Vendored from models/egnn.py
# ---------------------------------------------------------------------------
class MC_E_GCL(nn.Module):
    """Multi-Channel E(n) Equivariant Convolutional Layer."""

    def __init__(
        self,
        args,
        input_nf,
        output_nf,
        hidden_nf,
        n_channel,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        dropout=0.1,
        coord_change_maximum=10,
    ):
        super(MC_E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.args = args
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.dropout = nn.Dropout(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + n_channel**2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, output_nf)
        )

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())
        self.coord_change_maximum = coord_change_maximum

    def edge_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)

        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)

        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter: %s" % self.coords_agg)
        coord = coord + agg.clamp(-self.coord_change_maximum, self.coord_change_maximum)
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_id=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, self.args.rm_F_norm, batch_id=batch_id, norm_type=self.args.norm_type
        )

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord


class MC_Att_L(nn.Module):
    """Multi-Channel Attention Layer."""

    def __init__(
        self,
        args,
        input_nf,
        output_nf,
        hidden_nf,
        n_channel,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        dropout=0.1,
        coord_change_maximum=10,
        opm=False,
        normalize_coord=None,
    ):
        super().__init__()
        self.args = args

        self.hidden_nf = hidden_nf

        self.dropout = nn.Dropout(dropout)

        self.linear_q = nn.Linear(input_nf, hidden_nf)
        self.linear_kv = nn.Linear(input_nf + n_channel**2 + edges_in_d, hidden_nf * 2)

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.coord_change_maximum = coord_change_maximum

        if args.add_cross_attn_layer and args.explicit_pair_embed:
            self.cross_attn_module = CrossAttentionModule(
                node_hidden_dim=input_nf,
                pair_hidden_dim=input_nf,
                rm_layernorm=args.rm_layernorm,
                keep_trig_attn=args.keep_trig_attn,
                dist_hidden_dim=input_nf,
                normalize_coord=normalize_coord,
            )
        elif args.add_cross_attn_layer and not args.explicit_pair_embed:
            raise AssertionError
        elif args.add_cross_attn_layer and not args.add_attn_pair_bias:
            raise AssertionError

        if args.add_attn_pair_bias:
            self.inter_layer = InteractionModule(
                input_nf, output_nf, hidden_nf, opm=opm, rm_layernorm=args.rm_layernorm
            )
            self.attn_bias_proj = nn.Linear(hidden_nf, 1)

    def att_model(self, h, edge_index, radial, edge_attr, pair_embed=None):
        row, col = edge_index
        source, target = h[row], h[col]

        q = self.linear_q(source)
        n_channel = radial.shape[1]
        radial = radial.reshape(radial.shape[0], n_channel * n_channel)
        if edge_attr is not None:
            target_feat = torch.cat([radial, target, edge_attr], dim=1)
        else:
            target_feat = torch.cat([radial, target], dim=1)
        kv = self.linear_kv(target_feat)
        k, v = kv[..., 0::2], kv[..., 1::2]

        if self.args.add_attn_pair_bias:
            attn_bias = self.attn_bias_proj(pair_embed).squeeze(-1)
            alpha = torch.sum(q * k, dim=1) + attn_bias
        else:
            alpha = torch.sum(q * k, dim=1)

        alpha = scatter_softmax(alpha, row)

        return alpha, v

    def node_model(self, h, edge_index, att_weight, v):
        row, _ = edge_index
        agg = unsorted_segment_sum(att_weight * v, row, h.shape[0])
        agg = self.dropout(agg)
        return h + agg

    def coord_model(self, coord, edge_index, coord_diff, att_weight, v):
        row, _ = edge_index
        coord_v = att_weight * self.coord_mlp(v)
        trans = coord_diff * coord_v.unsqueeze(-1)
        agg = unsorted_segment_sum(trans, row, coord.size(0))
        coord = coord + agg.clamp(-self.coord_change_maximum, self.coord_change_maximum)
        return coord

    def trio_encoder(
        self,
        h,
        edge_index,
        coord,
        pair_embed_batched=None,
        pair_mask=None,
        batch_id=None,
        segment_id=None,
        reduced_tuple=None,
        LAS_mask=None,
        p_p_dist_embed=None,
        c_c_dist_embed=None,
    ):
        row, col = edge_index
        c_batch = batch_id[segment_id == 0]
        p_batch = batch_id[segment_id == 1]
        c_embed = h[segment_id == 0]
        p_embed = h[segment_id == 1]
        p_embed_batched, p_mask = to_dense_batch(p_embed, p_batch)
        c_embed_batched, c_mask = to_dense_batch(c_embed, c_batch)
        if self.args.add_cross_attn_layer:
            p_embed_batched, c_embed_batched, pair_embed_batched = self.cross_attn_module(
                p_embed_batched,
                p_mask,
                c_embed_batched,
                c_mask,
                pair_embed_batched,
                pair_mask,
                p_p_dist_embed=p_p_dist_embed,
                c_c_dist_embed=c_c_dist_embed,
            )
            for i in range(batch_id.max() + 1):
                if i == 0:
                    new_h = torch.cat(
                        (c_embed_batched[i][c_mask[i]], p_embed_batched[i][p_mask[i]]), dim=0
                    )
                else:
                    new_sample = torch.cat(
                        (c_embed_batched[i][c_mask[i]], p_embed_batched[i][p_mask[i]]), dim=0
                    )
                    new_h = torch.cat((new_h, new_sample), dim=0)
        else:
            new_h = h
            if self.args.explicit_pair_embed:
                pair_embed_batched = (
                    pair_embed_batched
                    + self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]
                )
            else:
                pair_embed_batched = self.inter_layer(
                    p_embed_batched, c_embed_batched, p_mask, c_mask
                )[0]

        compound_offset_in_batch = c_mask.sum(1)
        reduced_inter_edges_batchid, reduced_inter_edge_offsets = reduced_tuple
        reduced_row = row[row < col] - reduced_inter_edge_offsets
        reduced_col = (
            col[row < col]
            - reduced_inter_edge_offsets
            - compound_offset_in_batch[reduced_inter_edges_batchid]
        )
        first_part = pair_embed_batched[reduced_inter_edges_batchid, reduced_col, reduced_row]
        reduced_row = (
            row[row > col]
            - reduced_inter_edge_offsets
            - compound_offset_in_batch[reduced_inter_edges_batchid]
        )
        reduced_col = col[row > col] - reduced_inter_edge_offsets
        second_part = pair_embed_batched[reduced_inter_edges_batchid, reduced_row, reduced_col]
        for i in range(reduced_inter_edges_batchid.max() + 1):
            if i == 0:
                pair_offset = torch.cat(
                    (
                        first_part[reduced_inter_edges_batchid == i],
                        second_part[reduced_inter_edges_batchid == i],
                    ),
                    dim=0,
                )
            else:
                new_sample = torch.cat(
                    (
                        first_part[reduced_inter_edges_batchid == i],
                        second_part[reduced_inter_edges_batchid == i],
                    ),
                    dim=0,
                )
                pair_offset = torch.cat((pair_offset, new_sample), dim=0)
        return new_h, pair_embed_batched, pair_offset

    def forward(
        self,
        h,
        edge_index,
        coord,
        edge_attr=None,
        segment_id=None,
        batch_id=None,
        reduced_tuple=None,
        pair_embed_batched=None,
        pair_mask=None,
        LAS_mask=None,
        p_p_dist_embed=None,
        c_c_dist_embed=None,
    ):
        if self.args.add_attn_pair_bias:
            h, pair_embed_batched, pair_offset_embed = self.trio_encoder(
                h,
                edge_index,
                coord,
                pair_embed_batched=pair_embed_batched,
                pair_mask=pair_mask,
                batch_id=batch_id,
                segment_id=segment_id,
                reduced_tuple=reduced_tuple,
                LAS_mask=LAS_mask,
                p_p_dist_embed=p_p_dist_embed,
                c_c_dist_embed=c_c_dist_embed,
            )
        else:
            pair_offset_embed = None

        radial, coord_diff = coord2radial(
            edge_index, coord, self.args.rm_F_norm, batch_id=batch_id, norm_type=self.args.norm_type
        )
        att_weight, v = self.att_model(
            h, edge_index, radial, edge_attr, pair_embed=pair_offset_embed
        )

        flat_att_weight = att_weight
        att_weight = att_weight.unsqueeze(-1)
        h = self.node_model(h, edge_index, att_weight, v)
        coord = self.coord_model(coord, edge_index, coord_diff, att_weight, v)

        return h, coord, flat_att_weight


class MCAttEGNN(nn.Module):
    def __init__(
        self,
        args,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        n_channel,
        in_edge_nf=0,
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        dropout=0.1,
        dense=False,
        normalize_coord=None,
        unnormalize_coord=None,
        geometry_reg_step_size=0.001,
    ):
        super().__init__()
        self.args = args

        self.geometry_reg_step_size = geometry_reg_step_size
        self.geom_reg_steps = 1

        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        self.normalize_coord = normalize_coord
        self.unnormalize_coord = unnormalize_coord
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(
                f"gcl_{i}",
                MC_E_GCL(
                    args,
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    n_channel,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    dropout=dropout,
                    coord_change_maximum=self.normalize_coord(10),
                ),
            )
            self.add_module(
                f"att_{i}",
                MC_Att_L(
                    args,
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    n_channel,
                    edges_in_d=0,
                    act_fn=act_fn,
                    dropout=dropout,
                    coord_change_maximum=self.normalize_coord(10),
                    opm=args.opm,
                    normalize_coord=normalize_coord,
                ),
            )
        self.out_layer = MC_E_GCL(
            args,
            self.hidden_nf,
            self.hidden_nf,
            self.hidden_nf,
            n_channel,
            edges_in_d=in_edge_nf,
            act_fn=act_fn,
            residual=residual,
            coord_change_maximum=self.normalize_coord(10),
        )

    def forward(
        self,
        h,
        x,
        ctx_edges,
        att_edges,
        LAS_edge_list,
        batched_complex_coord_LAS,
        segment_id=None,
        batch_id=None,
        reduced_tuple=None,
        pair_embed_batched=None,
        pair_mask=None,
        LAS_mask=None,
        p_p_dist_embed=None,
        c_c_dist_embed=None,
        mask=None,
        ctx_edge_attr=None,
        att_edge_attr=None,
        return_attention=False,
    ):
        h = self.linear_in(h)
        h = self.dropout(h)
        x = x.clone()

        ctx_states, ctx_coords, atts = [], [], []
        for i in range(0, self.n_layers):
            h, coord = self._modules[f"gcl_{i}"](
                h, ctx_edges, x, batch_id=batch_id, edge_attr=ctx_edge_attr
            )
            if self.args.fix_pocket:
                x[mask] = coord[mask]
            else:
                x = coord
            ctx_states.append(h)
            ctx_coords.append(x)
            if self.args.add_attn_pair_bias:
                if self.args.explicit_pair_embed:
                    h, coord, att = self._modules[f"att_{i}"](
                        h,
                        att_edges,
                        x,
                        edge_attr=att_edge_attr,
                        segment_id=segment_id,
                        batch_id=batch_id,
                        reduced_tuple=reduced_tuple,
                        pair_embed_batched=pair_embed_batched,
                        pair_mask=pair_mask,
                        LAS_mask=LAS_mask,
                        p_p_dist_embed=p_p_dist_embed,
                        c_c_dist_embed=c_c_dist_embed,
                    )
                else:
                    h, coord, att = self._modules[f"att_{i}"](
                        h,
                        att_edges,
                        x,
                        edge_attr=att_edge_attr,
                        segment_id=segment_id,
                        batch_id=batch_id,
                        reduced_tuple=reduced_tuple,
                    )
            else:
                h, coord, att = self._modules[f"att_{i}"](
                    h, att_edges, x, batch_id=batch_id, edge_attr=att_edge_attr
                )

            if self.args.fix_pocket:
                x[mask] = coord[mask]
            else:
                x = coord
            atts.append(att)

            if not self.args.rm_LAS_constrained_optim:
                x.squeeze_(1)
                batched_complex_coord_LAS.squeeze_(1)
                for step in range(self.geom_reg_steps):
                    LAS_cur_squared = torch.sum(
                        (x[LAS_edge_list[0]] - x[LAS_edge_list[1]]) ** 2, dim=1
                    )
                    LAS_true_squared = torch.sum(
                        (
                            batched_complex_coord_LAS[LAS_edge_list[0]]
                            - batched_complex_coord_LAS[LAS_edge_list[1]]
                        )
                        ** 2,
                        dim=1,
                    )
                    grad_squared = 2 * (x[LAS_edge_list[0]] - x[LAS_edge_list[1]])
                    LAS_force = 2 * (LAS_cur_squared - LAS_true_squared)[:, None] * grad_squared
                    LAS_delta_coord = scatter_add(
                        src=LAS_force, index=LAS_edge_list[1], dim=0, dim_size=x.shape[0]
                    )

                    x = x + (LAS_delta_coord * self.geometry_reg_step_size).clamp(
                        min=self.normalize_coord(-15), max=self.normalize_coord(15)
                    )
                x.unsqueeze_(1)

        h, coord = self.out_layer(h, ctx_edges, x, batch_id=batch_id, edge_attr=ctx_edge_attr)
        if self.args.fix_pocket:
            x[mask] = coord[mask]
        else:
            x = coord
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        if return_attention:
            return h, x, atts
        else:
            return h, x


def coord2radial(edge_index, coord, rm_F_norm, batch_id=None, norm_type=None):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))
    if not rm_F_norm:
        if norm_type == "all_sample":
            radial = F.normalize(radial, dim=0)
        elif norm_type == "per_sample":
            edge_batch_id = batch_id[row]
            norm_for_each_sample = scatter_sum(src=(radial**2), index=edge_batch_id, dim=0).sqrt()
            norm_for_each_edge = norm_for_each_sample[edge_batch_id]
            radial = radial / norm_for_each_edge
        elif norm_type == "4_sample":
            shrink_batch_id = batch_id // 4
            edge_batch_id = shrink_batch_id[row]
            norm_for_each_sample = scatter_sum(src=(radial**2), index=edge_batch_id, dim=0).sqrt()
            norm_for_each_edge = norm_for_each_sample[edge_batch_id]
            radial = radial / norm_for_each_edge

    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


# ---------------------------------------------------------------------------
# Vendored from models/att_model.py
# ---------------------------------------------------------------------------
def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def _radial_edges(X, src_dst, cutoff):
    dist = X[:, 0][src_dst]
    dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1)
    src_dst = src_dst[dist <= cutoff]
    src_dst = src_dst.transpose(0, 1)
    return src_dst


class ComplexGraph(nn.Module):
    def __init__(
        self, args, inter_cutoff=10, intra_cutoff=8, normalize_coord=None, unnormalize_coord=None
    ):
        super().__init__()
        self.args = args
        self.inter_cutoff = normalize_coord(inter_cutoff)
        self.intra_cutoff = normalize_coord(intra_cutoff)

    @torch.no_grad()
    def construct_edges(self, X, batch_id, segment_ids, is_global):
        """Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch"""
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]

        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0
        row, col = torch.nonzero(same_bid).T
        col = col + offsets[batch_id[row]]

        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))

        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = sequential_and(row_seg == col_seg, row_seg == 1, not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        ctx_edges = _radial_edges(
            X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=self.intra_cutoff
        )

        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges = _radial_edges(
            X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=self.inter_cutoff
        )
        if inter_edges.shape[1] == 0:
            inter_edges = torch.tensor(
                [[inter_all_row[0], inter_all_col[0]], [inter_all_col[0], inter_all_row[0]]],
                device=inter_all_row.device,
            )
        reduced_inter_edge_batchid = batch_id[inter_edges[0][inter_edges[0] < inter_edges[1]]]
        reduced_inter_edge_offsets = offsets.gather(-1, reduced_inter_edge_batchid)

        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])
        select_edges = torch.logical_and(row_global, col_global)
        global_global = torch.stack([row[select_edges], col[select_edges]])

        space_edge_num = ctx_edges.shape[1] + global_normal.shape[1] + global_global.shape[1]  # noqa: F841 (unused in original source; kept verbatim)
        ctx_edges = torch.cat([ctx_edges, global_normal, global_global], dim=1)

        if self.args.add_attn_pair_bias:
            return ctx_edges, inter_edges, (reduced_inter_edge_batchid, reduced_inter_edge_offsets)
        else:
            return ctx_edges, inter_edges, None

    def forward(self, X, batch_id, segment_id, is_global):
        return self.construct_edges(X, batch_id, segment_id, is_global)


class EfficientMCAttModel(nn.Module):
    def __init__(
        self,
        args,
        embed_size,
        hidden_size,
        n_channel,
        n_edge_feats=0,
        n_layers=5,
        dropout=0.1,
        n_iter=5,
        dense=False,
        inter_cutoff=10,
        intra_cutoff=8,
        normalize_coord=None,
        unnormalize_coord=None,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.args = args
        self.random_n_iter = args.random_n_iter
        # NOTE: only the MCAttEGNN ("default") backbone is vendored; the
        # ablation_no_attention / ablation_no_attention_with_cross_attn code paths
        # (MCnoAttEGNN / MCnoAttwithCrossAttEGNN) are inert with the real default args.
        self.gnn = MCAttEGNN(
            args,
            embed_size,
            hidden_size,
            hidden_size,
            n_channel,
            n_edge_feats,
            n_layers=n_layers,
            residual=True,
            dropout=dropout,
            dense=dense,
            normalize_coord=normalize_coord,
            unnormalize_coord=unnormalize_coord,
            geometry_reg_step_size=args.geometry_reg_step_size,
        )

        self.extract_edges = ComplexGraph(
            args,
            inter_cutoff=inter_cutoff,
            intra_cutoff=intra_cutoff,
            normalize_coord=normalize_coord,
            unnormalize_coord=unnormalize_coord,
        )

        if args.explicit_pair_embed:
            self.inter_layer = InteractionModule(
                hidden_size, hidden_size, hidden_size, rm_layernorm=args.rm_layernorm
            )
        if args.keep_trig_attn:
            f = normalize_coord
            self.p_p_dist_layer = RBFDistanceModule(
                rbf_stop=f(32), distance_hidden_dim=hidden_size, num_gaussian=32
            )
            self.c_c_dist_layer = RBFDistanceModule(
                rbf_stop=f(16), distance_hidden_dim=hidden_size, num_gaussian=32
            )

    def forward(
        self,
        X,
        H,
        batch_id,
        segment_id,
        mask,
        is_global,
        compound_edge_index,
        LAS_edge_index,
        batched_complex_coord_LAS,
        LAS_mask=None,
    ):
        if self.args.keep_trig_attn:
            s_coord = X.squeeze(1).clone().detach()
            c_batch = batch_id[segment_id == 0]
            p_batch = batch_id[segment_id == 1]
            c_coord = s_coord[segment_id == 0]
            p_coord = s_coord[segment_id == 1]
            p_coord_batched, p_coord_mask = to_dense_batch(p_coord, p_batch)
            c_coord_batched, c_coord_mask = to_dense_batch(c_coord, c_batch)
            p_p_dist = torch.cdist(
                p_coord_batched, p_coord_batched, compute_mode="donot_use_mm_for_euclid_dist"
            )
            c_c_dist = torch.cdist(
                c_coord_batched, c_coord_batched, compute_mode="donot_use_mm_for_euclid_dist"
            )
            p_p_dist_mask = torch.einsum("...i, ...j->...ij", p_coord_mask, p_coord_mask)
            c_c_diag_mask = torch.diag_embed(c_coord_mask)
            c_c_dist_mask = torch.logical_or(LAS_mask, c_c_diag_mask)
            p_p_dist[~p_p_dist_mask] = 1e6
            c_c_dist[~c_c_dist_mask] = 1e6
            p_p_dist_embed = self.p_p_dist_layer(p_p_dist)
            c_c_dist_embed = self.c_c_dist_layer(c_c_dist)
        else:
            p_p_dist_embed = None
            c_c_dist_embed = None

        if self.args.explicit_pair_embed:
            c_batch = batch_id[segment_id == 0]
            p_batch = batch_id[segment_id == 1]
            c_embed = H[segment_id == 0]
            p_embed = H[segment_id == 1]
            p_embed_batched, p_mask = to_dense_batch(p_embed, p_batch)
            c_embed_batched, c_mask = to_dense_batch(c_embed, c_batch)
            pair_embed_batched, pair_mask = self.inter_layer(
                p_embed_batched, c_embed_batched, p_mask, c_mask
            )
            pair_embed_batched = pair_embed_batched * pair_mask.to(torch.float).unsqueeze(-1)
        else:
            pair_embed_batched, pair_mask = None, None

        if self.training and self.random_n_iter:
            iter_i = random.randint(1, self.n_iter)
        else:
            iter_i = self.n_iter

        for r in range(iter_i):
            if self.args.refine == "stack":
                with torch.no_grad():
                    ctx_edges, inter_edges, reduced_tuple = self.extract_edges(
                        X, batch_id, segment_id, is_global
                    )
                    ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                H, Z = self.gnn(
                    H,
                    X,
                    ctx_edges,
                    inter_edges,
                    LAS_edge_index,
                    batched_complex_coord_LAS,
                    segment_id=segment_id,
                    batch_id=batch_id,
                    reduced_tuple=reduced_tuple,
                    pair_embed_batched=pair_embed_batched,
                    pair_mask=pair_mask,
                    LAS_mask=LAS_mask,
                    p_p_dist_embed=p_p_dist_embed,
                    c_c_dist_embed=c_c_dist_embed,
                    mask=mask,
                )
                X[mask] = Z[mask]

            elif self.args.refine == "refine_coord":
                if r < iter_i - 1:
                    with torch.no_grad():
                        ctx_edges, inter_edges, reduced_tuple = self.extract_edges(
                            X, batch_id, segment_id, is_global
                        )
                        ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                        _, Z = self.gnn(
                            H,
                            X,
                            ctx_edges,
                            inter_edges,
                            LAS_edge_index,
                            batched_complex_coord_LAS,
                            segment_id=segment_id,
                            batch_id=batch_id,
                            reduced_tuple=reduced_tuple,
                            pair_embed_batched=pair_embed_batched,
                            pair_mask=pair_mask,
                            LAS_mask=LAS_mask,
                            p_p_dist_embed=p_p_dist_embed,
                            c_c_dist_embed=c_c_dist_embed,
                            mask=mask,
                        )
                        X[mask] = Z[mask]
                else:
                    with torch.no_grad():
                        ctx_edges, inter_edges, reduced_tuple = self.extract_edges(
                            X, batch_id, segment_id, is_global
                        )
                        ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                    H, Z = self.gnn(
                        H,
                        X,
                        ctx_edges,
                        inter_edges,
                        LAS_edge_index,
                        batched_complex_coord_LAS,
                        segment_id=segment_id,
                        batch_id=batch_id,
                        reduced_tuple=reduced_tuple,
                        pair_embed_batched=pair_embed_batched,
                        pair_mask=pair_mask,
                        LAS_mask=LAS_mask,
                        p_p_dist_embed=p_p_dist_embed,
                        c_c_dist_embed=c_c_dist_embed,
                        mask=mask,
                    )
                    X[mask] = Z[mask]
        return X, H


# ---------------------------------------------------------------------------
# Vendored from models/model.py
# ---------------------------------------------------------------------------
class Transition_diff_out_dim(torch.nn.Module):
    def __init__(self, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n * embedding_channels)
        self.linear2 = Linear(n * embedding_channels, out_channels)
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=0.001)

    def forward(self, z):
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z


class IaBNet_mean_and_pocket_prediction_cls_coords_dependent(torch.nn.Module):
    """The real FABind top-level model (models/model.py::get_model, mode=5)."""

    def __init__(self, args, embedding_channels=128, pocket_pred_embedding_channels=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.args = args
        self.coordinate_scale = args.coordinate_scale
        self.normalize_coord = lambda x: x / self.coordinate_scale
        self.unnormalize_coord = lambda x: x * self.coordinate_scale
        self.stage_prob = args.stage_prob

        n_channel = 1
        self.complex_model = EfficientMCAttModel(
            args,
            embedding_channels,
            embedding_channels,
            n_channel,
            n_edge_feats=0,
            n_layers=args.mean_layers,
            n_iter=args.n_iter,
            inter_cutoff=args.inter_cutoff,
            intra_cutoff=args.intra_cutoff,
            normalize_coord=self.normalize_coord,
            unnormalize_coord=self.unnormalize_coord,
        )

        self.pocket_pred_model = EfficientMCAttModel(
            args,
            pocket_pred_embedding_channels,
            pocket_pred_embedding_channels,
            n_channel,
            n_edge_feats=0,
            n_layers=args.pocket_pred_layers,
            n_iter=args.pocket_pred_n_iter,
            inter_cutoff=args.inter_cutoff,
            intra_cutoff=args.intra_cutoff,
            normalize_coord=self.normalize_coord,
            unnormalize_coord=self.unnormalize_coord,
        )

        self.protein_to_pocket = Transition_diff_out_dim(
            embedding_channels=embedding_channels, n=4, out_channels=1
        )

        self.glb_c = nn.Parameter(torch.ones(1, embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, embedding_channels))
        if args.use_esm2_feat:
            protein_hidden = 1280
        else:
            protein_hidden = 15
        if args.esm2_concat_raw:
            protein_hidden = 1295
        self.protein_linear_whole_protein = nn.Linear(protein_hidden, embedding_channels)
        self.compound_linear_whole_protein = nn.Linear(56, embedding_channels)

        self.embedding_shrink = nn.Linear(embedding_channels, pocket_pred_embedding_channels)
        self.embedding_enlarge = nn.Linear(pocket_pred_embedding_channels, embedding_channels)

        self.distmap_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.ReLU(),
            nn.Linear(embedding_channels, 1),
        )

        torch.nn.init.xavier_uniform_(self.protein_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.compound_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_shrink.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_enlarge.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[0].weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[2].weight, gain=0.001)

    def inference(self, data):
        """Real inference-time forward path (called by FABind's own fabind_inference.py
        as `model.inference(data)`)."""
        compound_batch = data["compound"].batch
        protein_batch_whole = data["protein_whole"].batch
        complex_batch_whole_protein = data["complex_whole_protein"].batch

        batched_complex_coord_whole_protein = self.normalize_coord(
            data["complex_whole_protein"].node_coords.unsqueeze(-2)
        )
        batched_complex_coord_LAS_whole_protein = self.normalize_coord(
            data["complex_whole_protein"].node_coords_LAS.unsqueeze(-2)
        )
        batched_compound_emb_whole_protein = self.compound_linear_whole_protein(
            data["compound"].node_feats
        )
        batched_protein_emb_whole_protein = self.protein_linear_whole_protein(
            data["protein_whole"].node_feats
        )

        for i in range(complex_batch_whole_protein.max() + 1):
            if i == 0:
                new_samples_whole_protein = torch.cat(
                    (
                        self.glb_c,
                        batched_compound_emb_whole_protein[compound_batch == i],
                        self.glb_p,
                        batched_protein_emb_whole_protein[protein_batch_whole == i],
                    ),
                    dim=0,
                )
            else:
                new_sample_whole_protein = torch.cat(
                    (
                        self.glb_c,
                        batched_compound_emb_whole_protein[compound_batch == i],
                        self.glb_p,
                        batched_protein_emb_whole_protein[protein_batch_whole == i],
                    ),
                    dim=0,
                )
                new_samples_whole_protein = torch.cat(
                    (new_samples_whole_protein, new_sample_whole_protein), dim=0
                )

        new_samples_whole_protein = self.embedding_shrink(new_samples_whole_protein)

        complex_coords_whole_protein, complex_out_whole_protein = self.pocket_pred_model(
            batched_complex_coord_whole_protein,
            new_samples_whole_protein,
            batch_id=complex_batch_whole_protein,
            segment_id=data["complex_whole_protein"].segment,
            mask=data["complex_whole_protein"].mask,
            is_global=data["complex_whole_protein"].is_global,
            compound_edge_index=data[
                "complex_whole_protein", "c2c", "complex_whole_protein"
            ].edge_index,
            LAS_edge_index=data["complex_whole_protein", "LAS", "complex_whole_protein"].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
            LAS_mask=None,
        )

        complex_out_whole_protein = self.embedding_enlarge(complex_out_whole_protein)

        compound_flag_whole_protein = torch.logical_and(
            data["complex_whole_protein"].segment == 0, ~data["complex_whole_protein"].is_global
        )
        compound_out_whole_protein = complex_out_whole_protein[compound_flag_whole_protein]
        protein_flag_whole_protein = torch.logical_and(
            data["complex_whole_protein"].segment == 1, ~data["complex_whole_protein"].is_global
        )
        protein_out_whole_protein = complex_out_whole_protein[protein_flag_whole_protein]
        protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(
            protein_out_whole_protein, protein_batch_whole
        )
        pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole

        protein_coords_batched_whole, protein_coords_mask_whole = to_dense_batch(
            data.node_xyz_whole, protein_batch_whole
        )

        pred_pocket_center = torch.zeros((pocket_cls_pred.shape[0], 3)).to(pocket_cls_pred.device)
        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        for i, j in enumerate(batch_len):
            pred_index_bool = pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1
            if pred_index_bool.sum() != 0:
                pred_pocket_center[i] = protein_coords_batched_whole.detach()[i][:j][
                    pred_index_bool
                ].mean(dim=0)
            else:
                pred_index_true = pocket_cls_pred[i][:j].sigmoid().unsqueeze(-1)
                pred_index_false = 1.0 - pred_index_true
                pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
                pred_index_log_prob = torch.log(pred_index_prob)
                pred_index_one_hot = gumbel_softmax_no_random(
                    pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard
                )
                pred_index_one_hot_true = pred_index_one_hot[:, 1].unsqueeze(-1)
                pred_pocket_center_gumbel = (
                    pred_index_one_hot_true * protein_coords_batched_whole[i][:j]
                )
                pred_pocket_center[i] = pred_pocket_center_gumbel.sum(
                    dim=0
                ) / pred_index_one_hot_true.sum(dim=0)

        batched_compound_emb = compound_out_whole_protein
        data["complex"].node_coords = torch.tensor([], device=compound_batch.device)
        data["complex"].node_coords_LAS = torch.tensor([], device=compound_batch.device)
        data["complex"].segment = torch.tensor([], device=compound_batch.device)
        data["complex"].mask = torch.tensor([], device=compound_batch.device)
        data["complex"].is_global = torch.tensor([], device=compound_batch.device)
        complex_batch = torch.tensor([], device=compound_batch.device)
        pocket_batch = torch.tensor([], device=compound_batch.device)
        data["complex", "c2c", "complex"].edge_index = torch.tensor(
            [], device=compound_batch.device
        )
        data["complex", "LAS", "complex"].edge_index = torch.tensor(
            [], device=compound_batch.device
        )
        pocket_coords_concats = torch.tensor([], device=compound_batch.device)
        dis_map = torch.tensor([], device=compound_batch.device)

        for i in range(pred_pocket_center.shape[0]):
            protein_i = data.node_xyz_whole[protein_batch_whole == i].detach()
            keepNode = get_keepNode_tensor(
                protein_i, self.args.pocket_radius, None, pred_pocket_center[i].detach()
            )
            if keepNode.sum() < 5:
                keepNode[:100] = True
            pocket_emb = protein_out_batched_whole[i][protein_out_mask_whole[i]][keepNode]
            if i == 0:
                new_samples = torch.cat(
                    (self.glb_c, batched_compound_emb[compound_batch == i], self.glb_p, pocket_emb),
                    dim=0,
                )
            else:
                new_sample = torch.cat(
                    (self.glb_c, batched_compound_emb[compound_batch == i], self.glb_p, pocket_emb),
                    dim=0,
                )
                new_samples = torch.cat((new_samples, new_sample), dim=0)

            pocket_coords = protein_coords_batched_whole[i][protein_coords_mask_whole[i]][keepNode]
            pocket_coords_concats = torch.cat((pocket_coords_concats, pocket_coords), dim=0)

            data["complex"].node_coords = torch.cat(
                (
                    data["complex"].node_coords,
                    torch.zeros((1, 3), device=compound_batch.device),
                    data["compound"].node_coords[compound_batch == i]
                    - data["compound"].node_coords[compound_batch == i].mean(dim=0).reshape(1, 3)
                    + pocket_coords.mean(dim=0).reshape(1, 3),
                    torch.zeros((1, 3), device=compound_batch.device),
                    pocket_coords,
                ),
                dim=0,
            ).float()

            if (
                self.args.compound_coords_init_mode == "redocking"
                or self.args.compound_coords_init_mode == "redocking_no_rotate"
            ):
                data["complex"].node_coords_LAS = torch.cat(
                    (
                        data["complex"].node_coords_LAS,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data["compound"].node_coords[compound_batch == i],
                        torch.zeros((1, 3), device=compound_batch.device),
                        torch.zeros_like(pocket_coords),
                    ),
                    dim=0,
                ).float()
            else:
                data["complex"].node_coords_LAS = torch.cat(
                    (
                        data["complex"].node_coords_LAS,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data["compound"].rdkit_coords[compound_batch == i],
                        torch.zeros((1, 3), device=compound_batch.device),
                        torch.zeros_like(pocket_coords),
                    ),
                    dim=0,
                ).float()

            n_protein = pocket_emb.shape[0]
            n_compound = batched_compound_emb[compound_batch == i].shape[0]
            segment = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            segment[n_compound + 1 :] = 1
            data["complex"].segment = torch.cat((data["complex"].segment, segment), dim=0)
            mask = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            mask[: n_compound + 2] = 1
            data["complex"].mask = torch.cat((data["complex"].mask, mask.bool()), dim=0)
            is_global = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            is_global[0] = 1
            is_global[n_compound + 1] = 1
            data["complex"].is_global = torch.cat(
                (data["complex"].is_global, is_global.bool()), dim=0
            )

            data["complex", "c2c", "complex"].edge_index = torch.cat(
                (
                    data["complex", "c2c", "complex"].edge_index,
                    data["compound_atom_edge_list"]
                    .x[data["compound_atom_edge_list"].batch == i]
                    .t()
                    + complex_batch.shape[0],
                ),
                dim=1,
            )
            data["complex", "LAS", "complex"].edge_index = torch.cat(
                (
                    data["complex", "LAS", "complex"].edge_index,
                    data["LAS_edge_list"].x[data["LAS_edge_list"].batch == i].t()
                    + complex_batch.shape[0],
                ),
                dim=1,
            )

            complex_batch = torch.cat(
                (
                    complex_batch,
                    torch.ones((n_compound + n_protein + 2), device=compound_batch.device) * i,
                ),
                dim=0,
            )
            pocket_batch = torch.cat(
                (pocket_batch, torch.ones((n_protein), device=compound_batch.device) * i), dim=0
            )

            dis_map_i = torch.cdist(
                pocket_coords, data["compound"].node_coords[compound_batch == i].to(torch.float32)
            ).flatten()
            dis_map_i[dis_map_i > 10] = 10
            dis_map = torch.cat((dis_map, dis_map_i), dim=0)

        batched_complex_coord = self.normalize_coord(data["complex"].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS = self.normalize_coord(
            data["complex"].node_coords_LAS.unsqueeze(-2)
        )
        complex_batch = complex_batch.to(torch.int64)
        pocket_batch = pocket_batch.to(torch.int64)
        pocket_coords_batched, _ = to_dense_batch(
            self.normalize_coord(pocket_coords_concats), pocket_batch
        )
        data["complex", "c2c", "complex"].edge_index = data[
            "complex", "c2c", "complex"
        ].edge_index.to(torch.int64)
        data["complex", "LAS", "complex"].edge_index = data[
            "complex", "LAS", "complex"
        ].edge_index.to(torch.int64)
        data["complex"].segment = data["complex"].segment.to(torch.bool)
        data["complex"].mask = data["complex"].mask.to(torch.bool)
        data["complex"].is_global = data["complex"].is_global.to(torch.bool)
        data["complex"].batch = complex_batch

        complex_coords, complex_out = self.complex_model(
            batched_complex_coord,
            new_samples,
            batch_id=complex_batch,
            segment_id=data["complex"].segment,
            mask=data["complex"].mask,
            is_global=data["complex"].is_global,
            compound_edge_index=data["complex", "c2c", "complex"].edge_index,
            LAS_edge_index=data["complex", "LAS", "complex"].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None,
        )

        compound_flag = torch.logical_and(data["complex"].segment == 0, ~data["complex"].is_global)
        compound_coords_out = complex_coords[compound_flag].squeeze(-2)
        compound_coords_out = self.unnormalize_coord(compound_coords_out)

        return compound_coords_out, compound_batch


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
class _FABindArgs:
    """Real argparse defaults from FABind/fabind/fabind_inference.py (mode=5 config)."""

    def __init__(self):
        self.mode = 5
        self.gs_tau = 1
        self.gs_hard = False
        self.hidden_size = 32  # shrunk from real default 256 for a smoke-sized model
        self.pocket_pred_hidden_size = 16  # shrunk from real default 128
        self.mean_layers = 2  # shrunk from real default 4/5-equivalent config
        self.n_iter = 2  # shrunk from real default 8
        self.pocket_pred_layers = 1
        self.pocket_pred_n_iter = 1
        self.inter_cutoff = 10.0
        self.intra_cutoff = 8.0
        self.coordinate_scale = 5.0
        self.geometry_reg_step_size = 0.001
        self.pocket_radius = 20.0
        self.stage_prob = 0.5
        self.use_esm2_feat = False
        self.esm2_concat_raw = False
        self.compound_coords_init_mode = "pocket_center_rdkit"
        self.random_n_iter = False
        self.refine = "refine_coord"
        self.rm_F_norm = False
        self.norm_type = "per_sample"
        self.fix_pocket = False
        self.rm_LAS_constrained_optim = False
        self.add_attn_pair_bias = False
        self.explicit_pair_embed = False
        self.opm = False
        self.add_cross_attn_layer = False
        self.rm_layernorm = False
        self.keep_trig_attn = False
        self.ablation_no_attention = False
        self.ablation_no_attention_with_cross_attn = False
        self.train_pred_pocket_noise = None


class _FABindInferenceWrapper(nn.Module):
    """Thin call-convention adapter only: `forward` forwards straight to the real
    `IaBNet_...cls_coords_dependent.inference(data)` method (the exact real entry point
    FABind's own fabind_inference.py calls), so `tl.trace(model, (data,))`'s implicit
    `model(*args)` reaches the unmodified real inference path."""

    def __init__(self, real_model):
        super().__init__()
        self.real_model = real_model

    def forward(self, data):
        return self.real_model.inference(data)


def build_fabind():
    args = _FABindArgs()
    real_model = IaBNet_mean_and_pocket_prediction_cls_coords_dependent(
        args,
        embedding_channels=args.hidden_size,
        pocket_pred_embedding_channels=args.pocket_pred_hidden_size,
    )
    return _FABindInferenceWrapper(real_model)


def example_input_fabind():
    """Builds a tiny synthetic HeteroData batch matching the real field schema of
    FABind's own utils/fabind_inference_dataset.py::InferenceDataset.get() (batch size 1,
    3 protein residues, 4 compound atoms)."""
    torch.manual_seed(0)
    n_protein_whole = 3
    n_compound = 4
    protein_hidden = 15  # matches args.use_esm2_feat=False branch (protein_hidden=15)

    protein_node_xyz = torch.randn(n_protein_whole, 3)
    protein_esm_feature = torch.randn(n_protein_whole, protein_hidden)
    compound_node_features = torch.randn(n_compound, 56)
    rdkit_coords = torch.randn(n_compound, 3)
    # a tiny fully-connected (minus self loops) local-atom-strain (LAS) edge index over the
    # compound atoms, matching the real (2, E) long edge_index convention.
    src, dst = torch.meshgrid(torch.arange(n_compound), torch.arange(n_compound), indexing="ij")
    mask = src != dst
    LAS_edge_index = torch.stack([src[mask], dst[mask]], dim=0).long()
    input_atom_edge_list = torch.cat(
        [LAS_edge_index.t().float(), torch.ones(LAS_edge_index.shape[1], 1)], dim=1
    )

    coords_init = rdkit_coords - rdkit_coords.mean(dim=0)

    data = HeteroData()
    data.coord_offset = protein_node_xyz.mean(dim=0).unsqueeze(0)
    protein_node_xyz = protein_node_xyz - protein_node_xyz.mean(dim=0)

    data["compound"].node_feats = compound_node_features.float()
    data["compound", "LAS", "compound"].edge_index = LAS_edge_index
    data["compound"].node_coords = coords_init
    data["compound"].rdkit_coords = coords_init
    data["compound_atom_edge_list"].x = (
        input_atom_edge_list[:, :2].long().contiguous() + 1
    ).clone()
    data["LAS_edge_list"].x = (LAS_edge_index + 1).clone().t()

    data.node_xyz_whole = protein_node_xyz
    data.idx = 0

    data["complex_whole_protein"].node_coords = torch.cat(
        (
            torch.zeros(1, 3),
            coords_init - coords_init.mean(dim=0),
            torch.zeros(1, 3),
            protein_node_xyz,
        ),
        dim=0,
    ).float()
    data["complex_whole_protein"].node_coords_LAS = torch.cat(
        (torch.zeros(1, 3), rdkit_coords, torch.zeros(1, 3), torch.zeros_like(protein_node_xyz)),
        dim=0,
    ).float()

    segment = torch.zeros(n_protein_whole + n_compound + 2)
    segment[n_compound + 1 :] = 1
    data["complex_whole_protein"].segment = segment
    mask_t = torch.zeros(n_protein_whole + n_compound + 2)
    mask_t[: n_compound + 2] = 1
    data["complex_whole_protein"].mask = mask_t.bool()
    is_global = torch.zeros(n_protein_whole + n_compound + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex_whole_protein"].is_global = is_global.bool()

    data["complex_whole_protein", "c2c", "complex_whole_protein"].edge_index = (
        input_atom_edge_list[:, :2].long().t().contiguous() + 1
    )
    data["complex_whole_protein", "LAS", "complex_whole_protein"].edge_index = LAS_edge_index + 1

    data["protein_whole"].node_feats = protein_esm_feature

    # torch_geometric batches single-graph HeteroData via a length-1 DataLoader-style
    # collate, giving every node store a zero `.batch` vector for batch_size=1.
    data["compound"].batch = torch.zeros(n_compound, dtype=torch.long)
    data["protein_whole"].batch = torch.zeros(n_protein_whole, dtype=torch.long)
    data["complex_whole_protein"].batch = torch.zeros(
        n_protein_whole + n_compound + 2, dtype=torch.long
    )
    data["compound_atom_edge_list"].batch = torch.zeros(
        input_atom_edge_list.shape[0], dtype=torch.long
    )
    data["LAS_edge_list"].batch = torch.zeros(LAS_edge_index.shape[1], dtype=torch.long)

    return (data,)


MENAGERIE_ENTRIES = [
    ("FABind", "build_fabind", "example_input_fabind", 2023, MENAGERIE_ZOO),
]
