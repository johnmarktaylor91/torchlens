# SOURCE: vendored from resistzzz/FABFlex @ main (subdirectory fabflex)
# Vendored files (real architecture code, imports/relative-paths adjusted only):
#   models/model_utils.py  (InteractionModule, Attention, MLP, MLPwithLastAct, MLPwoBias,
#                            permute_final_dims, flatten_final_dims, _attention)
#   models/cross_att.py    (CrossAttentionModule, RowAttentionBlock)
#   models/egnn.py         (MC_E_GCL, MC_Att_L, MCAttEGNN, coord2radial,
#                            unsorted_segment_sum/mean)
#   models/attn_model.py   (ComplexGraph, EfficientMCAttModel, sequential_and, _radial_edges)
#   models/model.py        (FABindPlus -- the real top-level model class)
#   utils/utils.py         (compute_dis_between_two_vector_tensor, get_keepNode_tensor,
#                            gumbel_softmax_no_random -- pure-torch helper functions with no
#                            rdkit/scipy dependency)
#
# FABFlex (Zhou et al., ICLR 2025, "Fast and Accurate Blind Flexible Docking") extends
# FABind/FABind+ with an explicit protein-side flexibility prediction: it jointly predicts a
# binding pocket, a data-driven pocket radius, and docks a ligand while also refining the
# local protein backbone/sidechain coordinates (`pro_mask` alongside `lig_mask`) inside the
# same E(n)-equivariant multi-channel attention GNN (EGNN + AlphaFold-style row/pair cross
# attention) used by FABind, unlike FABind's fixed-protein single `mask`. `FABindPlus` is the
# real top-level model class (models/model.py::get_model); `.inference(data, stage=2)` is the
# exact real zero-ground-truth inference entry point used by FABFlex's own
# inference_without_post_optim.py (`model.inference(data, stage=2)`).
import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.functional import softmax
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add, scatter_softmax, scatter_sum

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from utils/utils.py (pure-torch helpers only; no rdkit/scipy needed
# for the inference() forward path)
# ---------------------------------------------------------------------------
def compute_dis_between_two_vector_tensor(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


def get_keepNode_tensor(protein_node_xyz, pocket_radius, chosen_pocket_com):
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
class InteractionModule(nn.Module):
    def __init__(self, node_hidden_dim, pair_hidden_dim, hidden_dim, opm=False, rm_layernorm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.opm = opm

        self.rm_layernorm = rm_layernorm
        if not rm_layernorm:
            self.layer_norm_p = nn.LayerNorm(node_hidden_dim)
            self.layer_norm_c = nn.LayerNorm(node_hidden_dim)

        self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
        self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, pair_hidden_dim)

    def forward(self, p_embed, c_embed, p_mask=None, c_mask=None):
        if p_mask is None:
            p_mask = p_embed.new_ones(p_embed.shape[:-1], dtype=torch.bool)
        if c_mask is None:
            c_mask = c_embed.new_ones(c_embed.shape[:-1], dtype=torch.bool)

        if not self.rm_layernorm:
            p_embed = self.layer_norm_p(p_embed)
            c_embed = self.layer_norm_c(c_embed)

        inter_mask = torch.einsum("...i,...j->...ij", p_mask, c_mask)
        p_embed = self.linear_p(p_embed)
        c_embed = self.linear_c(c_embed)
        inter_embed = torch.einsum("...ik,...jk->...ijk", p_embed, c_embed)
        inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        return inter_embed, inter_mask


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds]).contiguous()


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
    distance=None,
    dis_pair_type=None,
    mha_permu=False,
) -> torch.Tensor:
    query = permute_final_dims(query, (1, 0, 2))
    key = permute_final_dims(key, (1, 2, 0))
    value = permute_final_dims(value, (1, 0, 2))
    a = torch.matmul(query, key)
    for b in biases:
        a = a + b
    if dis_pair_type == "add":
        safe_inverse_distance = distance
        if mha_permu:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 0, 1))
        else:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 1, 0))
        a = a + safe_inverse_distance
    a = softmax(a, -1)
    if dis_pair_type == "mul":
        safe_inverse_distance = distance
        if mha_permu:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 0, 1))
        else:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 1, 0))
        a = a * safe_inverse_distance
    a = torch.matmul(a, value)
    a = a.transpose(-2, -3)
    return a


class Attention(nn.Module):
    """Standard multi-head attention using AlphaFold's default layer initialization."""

    def __init__(
        self,
        args,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        mha_permu=False,
    ):
        super().__init__()
        self.args = args
        self.mha_permu = mha_permu

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = nn.Linear(self.c_q, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = nn.Linear(self.c_k, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = nn.Linear(self.c_v, self.c_hidden * self.no_heads, bias=False)
        self.linear_o = nn.Linear(self.c_hidden * self.no_heads, self.c_q)

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(self.c_q, self.c_hidden * self.no_heads)

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = flatten_final_dims(o, 2)
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        distance=None,
    ) -> torch.Tensor:
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(
            q,
            k,
            v,
            biases,
            distance=distance,
            dis_pair_type=self.args.rel_dis_pair_bias,
            mha_permu=self.mha_permu,
        )
        o = self._wrap_up(o, q_x)

        return o


class MLPwithLastAct(nn.Module):
    def __init__(self, args, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.args = args
        if self.args.use_ln_mlp:
            self.layernorm = nn.LayerNorm(embedding_channels)
        if args.dropout > 0:
            self.dropout1 = nn.Dropout(args.dropout)
            self.dropout2 = nn.Dropout(args.dropout)
        self.linear1 = nn.Linear(embedding_channels, int(n * embedding_channels))
        self.linear2 = nn.Linear(int(n * embedding_channels), out_channels)

    def forward(self, z):
        if self.args.use_ln_mlp:
            z = self.layernorm(z)

        if self.args.dropout > 0:
            z = self.dropout2(self.linear2(self.dropout1(self.linear1(z).relu())).relu())
        else:
            z = self.linear2(self.linear1(z).relu()).relu()
        return z


class MLPwoBias(nn.Module):
    def __init__(self, args, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.args = args
        if self.args.use_ln_mlp:
            self.layernorm = nn.LayerNorm(embedding_channels)
        if args.dropout > 0:
            self.dropout = nn.Dropout(args.dropout)
        self.linear1 = nn.Linear(embedding_channels, n * embedding_channels)
        self.linear2 = nn.Linear(n * embedding_channels, out_channels, bias=False)

    def forward(self, z):
        if self.args.use_ln_mlp:
            z = self.layernorm(z)
        if self.args.dropout > 0:
            z = self.linear2(self.dropout(self.linear1(z).relu()))
        else:
            z = self.linear2(self.linear1(z).relu())
        return z


class MLP(nn.Module):
    def __init__(self, args, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.args = args
        if self.args.use_ln_mlp:
            self.layernorm = nn.LayerNorm(embedding_channels)
        if args.dropout > 0:
            self.dropout = nn.Dropout(args.dropout)
        self.linear1 = nn.Linear(embedding_channels, n * embedding_channels)
        self.linear2 = nn.Linear(n * embedding_channels, out_channels)

    def forward(self, z):
        if self.args.use_ln_mlp:
            z = self.layernorm(z)

        if self.args.dropout > 0:
            z = self.linear2(self.dropout(self.linear1(z).relu()))
        else:
            z = self.linear2(self.linear1(z).relu())
        return z


# ---------------------------------------------------------------------------
# Vendored from models/cross_att.py
# ---------------------------------------------------------------------------
class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        args,
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
        self.p_attention_block = RowAttentionBlock(
            args,
            node_hidden_dim,
            pair_hidden_dim,
            no_heads=args.mha_heads,
            rm_layernorm=rm_layernorm,
            mha_permu=True,
        )
        self.c_attention_block = RowAttentionBlock(
            args,
            node_hidden_dim,
            pair_hidden_dim,
            no_heads=args.mha_heads,
            rm_layernorm=rm_layernorm,
            mha_permu=False,
        )
        self.p_transition = MLPwithLastAct(
            args,
            embedding_channels=node_hidden_dim,
            n=args.mlp_hidden_scale,
            out_channels=node_hidden_dim,
        )
        self.c_transition = MLPwithLastAct(
            args,
            embedding_channels=node_hidden_dim,
            n=args.mlp_hidden_scale,
            out_channels=node_hidden_dim,
        )
        self.pair_transition = MLPwithLastAct(
            args,
            embedding_channels=pair_hidden_dim,
            n=args.mlp_hidden_scale,
            out_channels=pair_hidden_dim,
        )
        self.inter_layer = InteractionModule(
            node_hidden_dim, pair_hidden_dim, 32, opm=False, rm_layernorm=rm_layernorm
        )

    def forward(
        self, p_embed_batched, p_mask, c_embed_batched, c_mask, pair_embed, pair_mask, distance=None
    ):
        p_embed_batched = self.p_attention_block(
            node_embed_i=p_embed_batched,
            node_embed_j=c_embed_batched,
            pair_embed=pair_embed,
            pair_mask=pair_mask,
            node_mask_i=p_mask,
            distance=distance,
        )
        c_embed_batched = self.c_attention_block(
            node_embed_i=c_embed_batched,
            node_embed_j=p_embed_batched,
            pair_embed=pair_embed.transpose(-2, -3),
            pair_mask=pair_mask.transpose(-1, -2),
            node_mask_i=c_mask,
            distance=distance,
        )
        p_embed_batched = p_embed_batched + self.p_transition(p_embed_batched)
        c_embed_batched = c_embed_batched + self.c_transition(c_embed_batched)

        pair_embed = (
            pair_embed + self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]
        )

        pair_embed = self.pair_transition(pair_embed) * pair_mask.to(torch.float).unsqueeze(-1)
        return p_embed_batched, c_embed_batched, pair_embed


class RowAttentionBlock(nn.Module):
    def __init__(
        self,
        args,
        node_hidden_dim,
        pair_hidden_dim,
        attention_hidden_dim=32,
        no_heads=4,
        dropout=0.1,
        rm_layernorm=False,
        mha_permu=False,
    ):
        super().__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.inf = 1e9

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm_node_i = LayerNorm(node_hidden_dim)
            self.layernorm_node_j = LayerNorm(node_hidden_dim)
            self.layernorm_pair = LayerNorm(pair_hidden_dim)

        self.linear = nn.Linear(pair_hidden_dim, self.no_heads)
        self.linear_g = nn.Linear(pair_hidden_dim, self.no_heads)

        self.dropout = nn.Dropout(dropout)

        self.mha = Attention(
            args,
            node_hidden_dim,
            node_hidden_dim,
            node_hidden_dim,
            attention_hidden_dim,
            no_heads,
            mha_permu=mha_permu,
        )

    def forward(
        self, node_embed_i, node_embed_j, pair_embed, pair_mask, node_mask_i, distance=None
    ):
        if not self.rm_layernorm:
            node_embed_i = self.layernorm_node_i(node_embed_i)
            node_embed_j = self.layernorm_node_j(node_embed_j)
            pair_embed = self.layernorm_pair(pair_embed)

        mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., None, :, :]
        pair_bias = self.linear(pair_embed) * self.linear_g(pair_embed).sigmoid()
        pair_bias = permute_final_dims(pair_bias, [2, 0, 1])

        node_embed_i = node_embed_i + self.dropout(
            self.mha(
                q_x=node_embed_i,
                kv_x=node_embed_j,
                biases=[mask_bias, pair_bias],
                distance=distance,
            )
        ) * node_mask_i.to(torch.float).unsqueeze(-1)

        return node_embed_i


# ---------------------------------------------------------------------------
# Vendored from models/egnn.py
# ---------------------------------------------------------------------------
def coord2radial(edge_index, coord, batch_id=None, norm_type=None):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))
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


class MC_E_GCL(nn.Module):
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
        super().__init__()
        input_edge = input_nf * 2
        self.args = args
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.coord_change_maximum = coord_change_maximum
        self.epsilon = 1e-8

        self.edge_mlp = MLPwithLastAct(
            args,
            embedding_channels=input_edge + n_channel**2 + edges_in_d,
            out_channels=hidden_nf,
            n=args.mlp_hidden_scale,
        )
        self.node_mlp = MLPwithLastAct(
            args,
            embedding_channels=hidden_nf + input_nf,
            out_channels=output_nf,
            n=args.mlp_hidden_scale,
        )
        self.coord_mlp = MLPwoBias(
            args, embedding_channels=hidden_nf, out_channels=n_channel, n=args.mlp_hidden_scale
        )
        torch.nn.init.xavier_uniform_(self.coord_mlp.linear2.weight, gain=0.001)

    def edge_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)
        if edge_attr is None:
            out = torch.cat((source, target, radial), dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        return out

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

    def node_model(self, x, edge_index, edge_feat, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_id=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, batch_id=batch_id, norm_type=self.args.norm_type
        )

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord


class MC_Att_L(nn.Module):
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

        self.coord_mlp = MLPwoBias(
            args, embedding_channels=hidden_nf, out_channels=n_channel, n=args.mlp_hidden_scale
        )
        torch.nn.init.xavier_uniform_(self.coord_mlp.linear2.weight, gain=0.001)
        self.coord_change_maximum = coord_change_maximum

        self.cross_attn_module = CrossAttentionModule(
            args,
            node_hidden_dim=input_nf,
            pair_hidden_dim=input_nf,
            rm_layernorm=args.rm_layernorm,
            keep_trig_attn=args.keep_trig_attn,
            dist_hidden_dim=input_nf,
            normalize_coord=normalize_coord,
        )

        if args.add_attn_pair_bias:
            self.inter_layer = InteractionModule(
                input_nf, output_nf, hidden_nf, opm=opm, rm_layernorm=args.rm_layernorm
            )
            self.attn_bias_proj = nn.Linear(hidden_nf, 1)

    def att_model(self, h, edge_index, radial, edge_attr=None, pair_embed=None):
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

        p_embed_batched, c_embed_batched, pair_embed_batched = self.cross_attn_module(
            p_embed_batched, p_mask, c_embed_batched, c_mask, pair_embed_batched, pair_mask
        )
        new_h = torch.cat((c_embed_batched[0][c_mask[0]], p_embed_batched[0][p_mask[0]]), dim=0)
        for i in range(1, batch_id.max() + 1):
            new_sample = torch.cat(
                (c_embed_batched[i][c_mask[i]], p_embed_batched[i][p_mask[i]]), dim=0
            )
            new_h = torch.cat((new_h, new_sample), dim=0)

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

        pair_offset = torch.cat(
            (
                first_part[reduced_inter_edges_batchid == 0],
                second_part[reduced_inter_edges_batchid == 0],
            ),
            dim=0,
        )
        for i in range(1, reduced_inter_edges_batchid.max() + 1):
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

        radial, coord_diff = coord2radial(
            edge_index, coord, batch_id=batch_id, norm_type=self.args.norm_type
        )
        att_weight, v = self.att_model(
            h, edge_index, radial, edge_attr, pair_embed=pair_offset_embed
        )

        flat_att_weight = att_weight
        att_weight = att_weight.unsqueeze(-1)
        h = self.node_model(h, edge_index, att_weight, v)

        coord = self.coord_model(coord, edge_index, coord_diff, att_weight, v)
        return h, coord, flat_att_weight, pair_embed_batched


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

        self.dropout = nn.Dropout(args.dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        self.normalize_coord = normalize_coord
        self.unnormalize_coord = unnormalize_coord

        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                MC_E_GCL(
                    args,
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
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
                    normalize_coord=normalize_coord,
                ),
            )
        self.out_layer = MC_E_GCL(
            args,
            hidden_nf,
            hidden_nf,
            hidden_nf,
            n_channel,
            edges_in_d=in_edge_nf,
            act_fn=act_fn,
            residual=residual,
            dropout=dropout,
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
            x = coord

            ctx_states.append(h)
            ctx_coords.append(x)

            h, coord, att, pair_embed_batched = self._modules[f"att_{i}"](
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
            x = coord
            atts.append(att)

            x.squeeze_(1)
            batched_complex_coord_LAS.squeeze_(1)
            for step in range(self.geom_reg_steps):
                LAS_cur_squared = torch.sum((x[LAS_edge_list[0]] - x[LAS_edge_list[1]]) ** 2, dim=1)
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
        x = coord
        ctx_states.append(h)
        ctx_coords.append(x)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x, pair_embed_batched


# ---------------------------------------------------------------------------
# Vendored from models/attn_model.py
# ---------------------------------------------------------------------------
def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
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
        N, max_n = batch_id.shape[0], lengths.max().item()
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)

        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]

        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[gni, lengths[batch_id] - 1] = 1
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

        ctx_edges = torch.cat([ctx_edges, global_normal, global_global], dim=1)

        return ctx_edges, inter_edges, (reduced_inter_edge_batchid, reduced_inter_edge_offsets)

    def forward(self, X, batch_id, segment_ids, is_global):
        return self.construct_edges(X, batch_id, segment_ids, is_global)


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
        self.inter_layer = InteractionModule(
            hidden_size, hidden_size, hidden_size, rm_layernorm=args.rm_layernorm
        )

    def forward(
        self,
        X,
        H,
        batch_id,
        segment_id,
        lig_mask,
        pro_mask,
        is_global,
        compound_edge_index,
        LAS_edge_index,
        batched_complex_coord_LAS,
        LAS_mask=None,
        flag=None,
    ):
        p_p_dist_embed = None
        c_c_dist_embed = None
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

        if self.training and self.random_n_iter:
            iter_i = random.randint(1, self.n_iter)
        else:
            iter_i = self.n_iter

        for r in range(iter_i):
            if r < iter_i - 1:
                with torch.no_grad():
                    ctx_edges, inter_edges, reduced_tuple = self.extract_edges(
                        X, batch_id, segment_id, is_global
                    )
                    ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                    _, Z, _ = self.gnn(
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
                        mask=lig_mask,
                    )
                    if flag == 1:
                        X[lig_mask] = Z[lig_mask]
                    elif flag == 2:
                        X[pro_mask] = Z[pro_mask]
                    else:
                        X = Z
            else:
                with torch.no_grad():
                    ctx_edges, inter_edges, reduced_tuple = self.extract_edges(
                        X, batch_id, segment_id, is_global
                    )
                    ctx_edges = torch.cat((compound_edge_index, ctx_edges), dim=1)
                H, Z, pair_embed_batched = self.gnn(
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
                    mask=lig_mask,
                )
                if flag == 1:
                    X[lig_mask] = Z[lig_mask]
                elif flag == 2:
                    X[pro_mask] = Z[pro_mask]
                else:
                    X = Z
        return X, H, pair_embed_batched


# ---------------------------------------------------------------------------
# Vendored from models/model.py
# ---------------------------------------------------------------------------
class FABindPlus(nn.Module):
    """The real FABFlex top-level model (models/model.py::get_model)."""

    def __init__(self, args, embedding_channels=128, pocket_pred_embedding_channels=128):
        super().__init__()
        self.args = args
        self.coordinate_scale = args.coord_scale
        self.normalize_coord = lambda x: x / self.coordinate_scale
        self.unnormalize_coord = lambda x: x * self.coordinate_scale

        self.glb_c = nn.Parameter(torch.ones(1, embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, embedding_channels))
        protein_hidden = 1280  # hard-coded for ESM2 feature
        compound_hidden = 56  # hard-coded for hand-crafted feature

        self.protein_linear_whole_protein = nn.Linear(protein_hidden, embedding_channels)
        self.compound_linear_whole_protein = nn.Linear(compound_hidden, embedding_channels)
        self.embedding_shrink = nn.Linear(embedding_channels, pocket_pred_embedding_channels)
        self.embedding_enlarge = nn.Linear(pocket_pred_embedding_channels, embedding_channels)

        n_channel = 1
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
        self.pocket_radius_head = MLP(
            args,
            embedding_channels=embedding_channels,
            n=self.args.mlp_hidden_scale,
            out_channels=1,
        )
        self.protein_to_pocket = MLP(
            args,
            embedding_channels=embedding_channels,
            n=self.args.mlp_hidden_scale,
            out_channels=1,
        )

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
        self.distmap_mlp = MLP(
            args,
            embedding_channels=embedding_channels,
            n=self.args.mlp_hidden_scale,
            out_channels=1,
        )

        torch.nn.init.xavier_uniform_(self.protein_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.compound_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_shrink.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_enlarge.weight, gain=0.001)

    def inference(self, data, stage=1, flag=2):
        """Real zero-ground-truth inference-time forward path (called by FABFlex's own
        inference_without_post_optim.py as `model.inference(data, stage=2)`)."""
        keepNode_less_5 = 0
        compound_batch = data["compound"].batch
        pocket_batch = data["pocket"].batch
        complex_batch = data["complex"].batch
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
        _, complex_out_whole_protein, _ = self.pocket_pred_model(
            batched_complex_coord_whole_protein,
            new_samples_whole_protein,
            batch_id=complex_batch_whole_protein,
            segment_id=data["complex_whole_protein"].segment,
            lig_mask=data["complex_whole_protein"].lig_mask,
            pro_mask=data["complex_whole_protein"].pro_mask,
            is_global=data["complex_whole_protein"].is_global,
            compound_edge_index=data[
                "complex_whole_protein", "c2c", "complex_whole_protein"
            ].edge_index,
            LAS_edge_index=data["complex_whole_protein", "LAS", "complex_whole_protein"].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
            LAS_mask=None,
            flag=flag,
        )

        complex_out_whole_protein = self.embedding_enlarge(complex_out_whole_protein)

        compound_flag_whole_protein = torch.logical_and(
            data["complex_whole_protein"].segment == 0, ~data["complex_whole_protein"].is_global
        )
        compound_out_whole_protein = complex_out_whole_protein[compound_flag_whole_protein]

        compound_in_complex_whole_batch = complex_batch_whole_protein[compound_flag_whole_protein]
        compound_emb_batch, compound_emb_mask = to_dense_batch(
            compound_out_whole_protein, compound_in_complex_whole_batch
        )
        pocket_radius_pred = self.pocket_radius_head(compound_emb_batch.sum(dim=1)).relu()

        protein_flag_whole_protein = torch.logical_and(
            data["complex_whole_protein"].segment == 1, ~data["complex_whole_protein"].is_global
        )
        protein_out_whole_protein = complex_out_whole_protein[protein_flag_whole_protein]
        protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(
            protein_out_whole_protein, protein_batch_whole
        )
        pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole
        pocket_cls, _ = to_dense_batch(data.pocket_idx, protein_batch_whole)

        pocket_coords_batched, _ = to_dense_batch(
            self.normalize_coord(data.input_pocket_node_xyz), pocket_batch
        )
        protein_coords_batched_whole, protein_coords_mask_whole = to_dense_batch(
            data.input_protein_node_xyz, protein_batch_whole
        )

        pred_index_true = pocket_cls_pred.sigmoid().unsqueeze(-1)
        pred_index_false = 1.0 - pred_index_true
        pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
        pred_index_prob = torch.clamp(pred_index_prob, min=1e-6, max=1 - 1e-6)
        pred_index_log_prob = torch.log(pred_index_prob)
        if self.pocket_pred_model.training:
            pred_index_one_hot = F.gumbel_softmax(
                pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard
            )
        else:
            pred_index_one_hot = gumbel_softmax_no_random(
                pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard
            )
        pred_index_one_hot_true = (pred_index_one_hot[:, :, 1] * protein_out_mask_whole).unsqueeze(
            -1
        )
        pred_pocket_center_gumbel = pred_index_one_hot_true * protein_coords_batched_whole
        pred_pocket_center = pred_pocket_center_gumbel.sum(dim=1) / pred_index_one_hot_true.sum(
            dim=1
        )

        if stage == 1:
            gt_pocket_batch = pocket_batch.clone()
            pocket_center_bias = torch.zeros_like(pred_pocket_center, device=compound_batch.device)
            batched_compound_emb = compound_out_whole_protein
            batched_pocket_emb = protein_out_whole_protein[data["pocket"].keepNode]
            for i in range(complex_batch.max() + 1):
                if self.args.shift_coord:
                    num_compound_atoms = data["compound"].node_feats[compound_batch == i].shape[0]
                    temp_coords = data["complex"].node_coords[complex_batch == i]
                    temp_coords[1 : num_compound_atoms + 1] = temp_coords[
                        1 : num_compound_atoms + 1
                    ] - temp_coords[1 : num_compound_atoms + 1].mean(dim=0)
                    temp_coords[num_compound_atoms + 2 :] = temp_coords[
                        num_compound_atoms + 2 :
                    ] - data.pocket_residue_center[i].unsqueeze(0)
                    data["complex"].node_coords[complex_batch == i] = temp_coords
                    data.coords[compound_batch == i] = data.coords[
                        compound_batch == i
                    ] - data.pocket_residue_center[i].unsqueeze(0)
                    data.pocket_node_xyz[pocket_batch == i] = data.pocket_node_xyz[
                        pocket_batch == i
                    ] - data.pocket_residue_center[i].unsqueeze(0)

                if i == 0:
                    new_samples = torch.cat(
                        (
                            self.glb_c,
                            batched_compound_emb[compound_batch == i],
                            self.glb_p,
                            batched_pocket_emb[pocket_batch == i],
                        ),
                        dim=0,
                    )
                else:
                    new_sample = torch.cat(
                        (
                            self.glb_c,
                            batched_compound_emb[compound_batch == i],
                            self.glb_p,
                            batched_pocket_emb[pocket_batch == i],
                        ),
                        dim=0,
                    )
                    new_samples = torch.cat((new_samples, new_sample), dim=0)
            dis_map = data.dis_map

            batched_complex_coord = self.normalize_coord(data["complex"].node_coords.unsqueeze(-2))
            batched_complex_coord_LAS = self.normalize_coord(
                data["complex"].node_coords_LAS.unsqueeze(-2)
            )
        else:
            batched_compound_emb = compound_out_whole_protein
            data["complex"].node_coords = torch.tensor([], device=compound_batch.device)
            data["complex"].node_coords_LAS = torch.tensor([], device=compound_batch.device)
            data["complex"].segment = torch.tensor([], device=compound_batch.device)
            data["complex"].lig_mask = torch.tensor([], device=compound_batch.device)
            data["complex"].pro_mask = torch.tensor([], device=compound_batch.device)
            data["complex"].is_global = torch.tensor([], device=compound_batch.device)
            complex_batch = torch.tensor([], device=compound_batch.device)
            gt_pocket_batch = pocket_batch.clone()
            pocket_batch = torch.tensor([], device=compound_batch.device)
            data["complex", "c2c", "complex"].edge_index = torch.tensor(
                [], device=compound_batch.device
            )
            data["complex", "LAS", "complex"].edge_index = torch.tensor(
                [], device=compound_batch.device
            )
            pocket_coords_concats = torch.tensor([], device=compound_batch.device)
            dis_map = torch.tensor([], device=compound_batch.device)
            data["pocket"].keepNode = torch.tensor(
                [], device=compound_batch.device, dtype=torch.bool
            )
            pocket_node_xyz_concate = torch.tensor([], device=compound_batch.device)
            pocket_center_bias = torch.zeros_like(pred_pocket_center, device=compound_batch.device)
            for i in range(pred_pocket_center.shape[0]):
                protein_i = data.input_protein_node_xyz[protein_batch_whole == i].detach()
                if self.args.pocket_radius_buffer <= 2.0:
                    pocket_radius = (pocket_radius_pred[i] * self.args.pocket_radius_buffer).item()
                else:
                    pocket_radius = (pocket_radius_pred[i] + self.args.pocket_radius_buffer).item()
                if pocket_radius < self.args.min_pocket_radius:
                    pocket_radius = self.args.min_pocket_radius
                if self.args.force_fix_radius:
                    pocket_radius = self.args.pocket_radius
                keepNode = get_keepNode_tensor(
                    protein_i, pocket_radius, pred_pocket_center[i].detach()
                )
                if keepNode.sum() < 5:
                    keepNode[:100] = True
                    keepNode_less_5 += 1
                data["pocket"].keepNode = torch.cat((data["pocket"].keepNode, keepNode), dim=0)
                pocket_emb = protein_out_batched_whole[i][protein_out_mask_whole[i]][keepNode]
                if i == 0:
                    new_samples = torch.cat(
                        (
                            self.glb_c,
                            batched_compound_emb[compound_batch == i],
                            self.glb_p,
                            pocket_emb,
                        ),
                        dim=0,
                    )
                else:
                    new_sample = torch.cat(
                        (
                            self.glb_c,
                            batched_compound_emb[compound_batch == i],
                            self.glb_p,
                            pocket_emb,
                        ),
                        dim=0,
                    )
                    new_samples = torch.cat((new_samples, new_sample), dim=0)

                pocket_coords = protein_coords_batched_whole[i][protein_coords_mask_whole[i]][
                    keepNode
                ]
                pocket_coords_center = pocket_coords.mean(dim=0).reshape(1, 3)
                gt_pocket_coords = data.protein_node_xyz[protein_batch_whole == i][keepNode.bool()]
                if self.args.shift_coord:
                    pocket_coords = pocket_coords - pocket_coords_center
                    gt_pocket_coords = gt_pocket_coords - pocket_coords_center
                    data.coords[compound_batch == i] = (
                        data.coords[compound_batch == i] - pocket_coords_center
                    )
                    pocket_center_bias[i] = pocket_coords_center.squeeze()

                pocket_coords_concats = torch.cat((pocket_coords_concats, pocket_coords), dim=0)
                pocket_node_xyz_concate = torch.cat(
                    (pocket_node_xyz_concate, gt_pocket_coords), dim=0
                )

                data["complex"].node_coords = torch.cat(
                    (
                        data["complex"].node_coords,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data["compound"].node_coords[compound_batch == i]
                        - data["compound"]
                        .node_coords[compound_batch == i]
                        .mean(dim=0)
                        .reshape(1, 3)
                        + pocket_coords.mean(dim=0).reshape(1, 3),
                        torch.zeros((1, 3), device=compound_batch.device),
                        pocket_coords,
                    ),
                    dim=0,
                ).float()

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
                lig_mask = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                lig_mask[: n_compound + 2] = 1
                data["complex"].lig_mask = torch.cat(
                    (data["complex"].lig_mask, lig_mask.bool()), dim=0
                )
                pro_mask = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                pro_mask[0] = 1
                pro_mask[n_compound + 1 :] = 1
                data["complex"].pro_mask = torch.cat(
                    (data["complex"].pro_mask, pro_mask.bool()), dim=0
                )
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

                dis_map_i = torch.cdist(pocket_coords, data.coords[compound_batch == i]).flatten()
                dis_map_i[dis_map_i > self.args.dis_map_thres] = self.args.dis_map_thres
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
            data["complex"].lig_mask = data["complex"].lig_mask.to(torch.bool)
            data["complex"].pro_mask = data["complex"].pro_mask.to(torch.bool)
            data["complex"].is_global = data["complex"].is_global.to(torch.bool)
            data.pocket_node_xyz = pocket_node_xyz_concate
            data.dis_map = dis_map
            data["complex"].batch = complex_batch
            data["pocket"].batch = pocket_batch

        complex_coords, complex_out, pair_embed_batched = self.complex_model(
            batched_complex_coord,
            new_samples,
            batch_id=complex_batch,
            segment_id=data["complex"].segment,
            lig_mask=data["complex"].lig_mask,
            pro_mask=data["complex"].pro_mask,
            is_global=data["complex"].is_global,
            compound_edge_index=data["complex", "c2c", "complex"].edge_index,
            LAS_edge_index=data["complex", "LAS", "complex"].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None,
            flag=flag,
        )
        compound_flag = torch.logical_and(data["complex"].segment == 0, ~data["complex"].is_global)
        protein_flag = torch.logical_and(data["complex"].segment == 1, ~data["complex"].is_global)
        pocket_out = complex_out[protein_flag]
        pocket_coords_out = complex_coords[protein_flag].squeeze(-2)
        compound_out = complex_out[compound_flag]
        compound_coords_out = complex_coords[compound_flag].squeeze(-2)

        _, pocket_out_mask = to_dense_batch(pocket_out, pocket_batch)
        _, compound_out_mask = to_dense_batch(compound_out, compound_batch)
        compound_coords_out_batched, _ = to_dense_batch(compound_coords_out, compound_batch)
        pocket_coords_out_batched, _ = to_dense_batch(pocket_coords_out, pocket_batch)
        holo_compound_coords_batched, _ = to_dense_batch(
            self.normalize_coord(data.coords), compound_batch
        )

        if flag == 1:
            pocket_com_dis_map = torch.cdist(pocket_coords_batched, compound_coords_out_batched)
        elif flag == 2:
            pocket_com_dis_map = torch.cdist(
                pocket_coords_out_batched, holo_compound_coords_batched
            )
        else:
            pocket_com_dis_map = torch.cdist(pocket_coords_out_batched, compound_coords_out_batched)

        z = pair_embed_batched[:, 1:, 1:, ...]
        z_mask = torch.einsum("bi,bj->bij", pocket_out_mask, compound_out_mask)

        b = self.distmap_mlp(z).squeeze(-1)
        y_pred = b[z_mask]
        y_pred = y_pred.sigmoid() * self.args.dis_map_thres

        y_pred_by_coords = pocket_com_dis_map[z_mask]
        y_pred_by_coords = self.unnormalize_coord(y_pred_by_coords)
        y_pred_by_coords = torch.clamp(y_pred_by_coords, 0, self.args.dis_map_thres)

        compound_coords_out = self.unnormalize_coord(compound_coords_out)
        pocket_coords_out = self.unnormalize_coord(pocket_coords_out)

        return (
            compound_coords_out,
            compound_batch,
            pocket_coords_out,
            pocket_batch,
            pocket_cls_pred,
            pocket_cls,
            protein_out_mask_whole,
            protein_coords_batched_whole,
            pred_pocket_center,
            pocket_radius_pred,
            gt_pocket_batch,
        )


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
class _FABFlexArgs:
    """Real argparse defaults from FABFlex/fabflex/utils/parsing.py."""

    def __init__(self):
        self.pocket_flag = 1
        self.hidden_size = 32  # shrunk from real default 512 for a smoke-sized model
        self.dropout = 0.1
        self.pocket_pred_hidden_size = 16  # shrunk from real default 128
        self.coord_scale = 5.0
        self.mean_layers = 2  # shrunk from real default 5
        self.pocket_pred_layers = 1
        self.inter_cutoff = 10.0
        self.intra_cutoff = 8.0
        self.n_iter = 1
        self.pocket_pred_n_iter = 1
        self.random_n_iter = False  # deterministic for a single smoke trace
        self.norm_type = "per_sample"
        self.geometry_reg_step_size = 0.001
        self.geometry_reg_step = 1
        self.mha_heads = 2  # shrunk from real default 4 (must divide hidden_size)
        self.rel_dis_pair_bias = "no"
        self.mlp_hidden_scale = 1
        self.gs_tau = 1
        self.gs_hard = False
        self.add_attn_pair_bias = True
        self.rm_layernorm = False
        self.use_ln_mlp = False
        self.keep_trig_attn = False
        self.force_fix_radius = False
        self.pocket_radius_buffer = 5.0
        self.min_pocket_radius = 20.0
        self.pocket_radius = 20
        self.shift_coord = False
        self.dis_map_thres = 15.0


def build_fabflex():
    args = _FABFlexArgs()
    real_model = FABindPlus(
        args,
        embedding_channels=args.hidden_size,
        pocket_pred_embedding_channels=args.pocket_pred_hidden_size,
    )
    return _FABFlexInferenceWrapper(real_model)


class _FABFlexInferenceWrapper(nn.Module):
    """Thin call-convention adapter only: `forward` forwards straight to the real
    `FABindPlus.inference(data, stage=2)` method (the exact real entry point FABFlex's own
    inference_without_post_optim.py calls), so `tl.trace(model, (data,))`'s implicit
    `model(*args)` reaches the unmodified real inference path."""

    def __init__(self, real_model):
        super().__init__()
        self.real_model = real_model

    def forward(self, data):
        return self.real_model.inference(data, stage=2)


def example_input_fabflex():
    """Builds a tiny synthetic HeteroData batch matching the real field schema consumed by
    FABindPlus.inference(data, stage=2) (batch size 1, 3 protein residues, 4 compound atoms),
    mirroring the construction in FABFlex's own utils/utils.py::construct_data (real field
    names: node_coords/node_coords_LAS/segment/lig_mask/pro_mask/is_global on
    complex_whole_protein, node_feats on compound/protein_whole, plus the
    input_protein_node_xyz/input_pocket_node_xyz/pocket_idx/protein_node_xyz/coords/pocket
    fields the real inference() forward reads directly off `data`)."""
    torch.manual_seed(0)
    n_protein_whole = 3
    n_compound = 4
    protein_hidden = 1280  # real hard-coded ESM2 feature width

    protein_node_xyz = torch.randn(n_protein_whole, 3)
    protein_esm_feature = torch.randn(n_protein_whole, protein_hidden)
    compound_node_features = torch.randn(n_compound, 56)
    rdkit_coords = torch.randn(n_compound, 3)
    src, dst = torch.meshgrid(torch.arange(n_compound), torch.arange(n_compound), indexing="ij")
    mask = src != dst
    LAS_edge_index = torch.stack([src[mask], dst[mask]], dim=0).long()
    input_atom_edge_list = torch.cat(
        [LAS_edge_index.t().float(), torch.ones(LAS_edge_index.shape[1], 1)], dim=1
    )

    coords_init = rdkit_coords - rdkit_coords.mean(dim=0)

    data = HeteroData()
    protein_node_xyz = protein_node_xyz - protein_node_xyz.mean(dim=0)

    data["compound"].node_feats = compound_node_features.float()
    data["compound", "LAS", "compound"].edge_index = LAS_edge_index
    data["compound"].node_coords = coords_init
    data["compound"].rdkit_coords = coords_init
    data["compound_atom_edge_list"].x = (
        input_atom_edge_list[:, :2].long().contiguous() + 1
    ).clone()
    data["LAS_edge_list"].x = (LAS_edge_index + 1).clone().t()

    # real top-level `data.*` fields read directly by inference()
    data.input_protein_node_xyz = protein_node_xyz
    data.protein_node_xyz = protein_node_xyz
    data.pocket_idx = torch.zeros(n_protein_whole)
    data.coords = coords_init

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
    lig_mask_t = torch.zeros(n_protein_whole + n_compound + 2)
    lig_mask_t[: n_compound + 2] = 1
    data["complex_whole_protein"].lig_mask = lig_mask_t.bool()
    pro_mask_t = torch.zeros(n_protein_whole + n_compound + 2)
    pro_mask_t[0] = 1
    pro_mask_t[n_compound + 1 :] = 1
    data["complex_whole_protein"].pro_mask = pro_mask_t.bool()
    is_global = torch.zeros(n_protein_whole + n_compound + 2)
    is_global[0] = 1
    is_global[n_compound + 1] = 1
    data["complex_whole_protein"].is_global = is_global.bool()

    data["complex_whole_protein", "c2c", "complex_whole_protein"].edge_index = (
        input_atom_edge_list[:, :2].long().t().contiguous() + 1
    )
    data["complex_whole_protein", "LAS", "complex_whole_protein"].edge_index = LAS_edge_index + 1

    data["protein_whole"].node_feats = protein_esm_feature

    # pocket-store fields (`.batch` for stage==2's `to_dense_batch(data.input_pocket_node_xyz, pocket_batch)`)
    data.input_pocket_node_xyz = protein_node_xyz
    data["pocket"].batch = torch.zeros(n_protein_whole, dtype=torch.long)

    data["compound"].batch = torch.zeros(n_compound, dtype=torch.long)
    data["protein_whole"].batch = torch.zeros(n_protein_whole, dtype=torch.long)
    data["complex_whole_protein"].batch = torch.zeros(
        n_protein_whole + n_compound + 2, dtype=torch.long
    )
    data["compound_atom_edge_list"].batch = torch.zeros(
        input_atom_edge_list.shape[0], dtype=torch.long
    )
    data["LAS_edge_list"].batch = torch.zeros(LAS_edge_index.shape[1], dtype=torch.long)
    # data['complex'].batch is populated by the real inference() code itself (stage=2 branch
    # rebuilds `complex`/`pocket` from `complex_whole_protein` predictions); provide an empty
    # placeholder store so `data['complex']` / `data['pocket']` HeteroData node types exist.
    data["complex"].x = torch.zeros(0, 1)
    data["complex"].batch = torch.zeros(0, dtype=torch.long)
    data["pocket"].x = torch.zeros(n_protein_whole, 1)

    return (data,)


MENAGERIE_ENTRIES = [
    ("FABFlex", "build_fabflex", "example_input_fabflex", 2025, MENAGERIE_ZOO),
]
