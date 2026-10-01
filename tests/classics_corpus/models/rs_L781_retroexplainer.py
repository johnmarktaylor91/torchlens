# SOURCE: vendored from wangyu-sd/RetroExplainer @ main
# Files: model/Embeddings.py, model/Modules.py
#
# RetroExplainer / RetroAGT (Wang et al., Nature Machine Intelligence 2023): a chemical-
# knowledge-guided molecular-assembly Transformer for retrosynthesis prediction. This module
# vendors the model's real, novel encoder architecture verbatim: `RetroAGTEmbeddingLayer`
# jointly embeds atom features (`AtomFeaEmbedding`, categorical atom-property embeddings plus
# a Gaussian radial-basis atom-charge feature) and bond/path features (`EdgeEmbedding`, a
# multi-hop bond-type path encoding plus a Gaussian radial-basis interatomic-distance feature,
# producing a per-head attention-bias tensor), prepending a learned "graph token" (a virtual
# whole-molecule node, [CLS]-style). `RetroAGTEncoder` stacks `RetroAGTEncoderLayer` blocks
# (pre/post-LN Transformer blocks using a from-scratch `MultiHeadAtomAttention` that consumes
# the per-head edge-bias tensor as an additive attention bias -- graph-aware self-attention).
# `MultiHeadAtomAdj` (a from-scratch bilinear multi-head pairwise-adjacency head, reusing the
# same in-projection/attention machinery without softmax normalization) predicts the
# reaction-center bond-change logits, exactly mirroring the encoder->rc_adj_fn path inside
# the real top-level `RetroAGT.forward` in model/RetroAGT.py.
#
# The full `RetroAGT` (a `pytorch_lightning.LightningModule`) is NOT vendored here: its
# `__init__` unconditionally does `assert os.path.isfile(leaving_group_path)` and
# `torch.load(leaving_group_path)` for a pretrained leaving-group vocabulary file that ships
# with model checkpoints, not the repo -- an external-data dependency that cannot be satisfied
# with a tiny random-init config. `rdkit` (used only by the downstream candidate-ranking /
# beam-search `run()` method, not the encoder) is also not a base dependency. The vendored
# submodules below are the real, architecturally-novel contribution and are wired together
# exactly as the real `RetroAGT.forward` calls them for its `rc_adj_prob` output (shared
# embedding -> shared encoder -> reaction-center adjacency head).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

PI = 3.14159
A = (2 * PI) ** 0.5


# --- model/Embeddings.py ---
class RetroAGTEmbeddingLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        max_paths,
        n_graph_type,
        max_single_hop,
        atom_dim,
        total_degree,
        formal_charge,
        hybrid,
        exp_valance,
        hydrogen,
        aromatic,
        ring,
        n_layers,
        need_graph_token,
        n_rxn_cnt,
        use_3d_info=False,
        dropout=0.0,
        known_rxn_type=True,
        known_rxn_cnt=True,
        use_dist_adj=True,
    ):
        super(RetroAGTEmbeddingLayer, self).__init__()

        self.atom_encoder = AtomFeaEmbedding(
            d_model,
            atom_dim,
            total_degree,
            formal_charge,
            hybrid,
            exp_valance,
            hydrogen,
            aromatic,
            ring,
            n_layers,
            need_graph_token,
            known_rxn_type,
            known_rxn_cnt,
            n_rxn_cnt=n_rxn_cnt,
        )
        self.edge_encoder = EdgeEmbedding(
            d_model,
            n_head,
            max_paths,
            n_graph_type,
            max_single_hop,
            n_layers,
            need_graph_token,
            use_3d_info=use_3d_info,
            known_rxn_type=known_rxn_type,
            known_rxn_cnt=known_rxn_cnt,
            use_dist_adj=use_dist_adj,
            n_rxn_cnt=n_rxn_cnt,
        )

    def forward(
        self, atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist3d_adj=None, contrast=False
    ):
        return self.atom_encoder(atom_fea, center_cnt, rxn_type, contrast), self.edge_encoder(
            bond_adj, dist_adj, center_cnt, rxn_type, dist3d_adj, contrast
        )


class AtomFeaEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        atom_dim=65,
        total_degree=20,
        formal_charge=8,
        hybrid=10,
        exp_valance=8,
        hydrogen=8,
        aromatic=2,
        ring=9,
        n_layers=1,
        need_graph_token=True,
        known_rxn_type=True,
        known_rxn_cnt=True,
        n_rxn_cnt=10,
    ):
        super(AtomFeaEmbedding, self).__init__()
        self.known_rxn_cnt = known_rxn_cnt
        self.known_rxn_type = known_rxn_type

        self.atom_encoders = nn.ModuleList(
            [
                nn.Embedding(atom_dim + 1, d_model, padding_idx=0),  # 65
                nn.Embedding(total_degree + 1, d_model, padding_idx=0),  # 20
                nn.Embedding(hybrid + 1, d_model, padding_idx=0),  # 10
                nn.Embedding(hydrogen + 1, d_model, padding_idx=0),  # 10
                nn.Embedding(aromatic + 1, d_model, padding_idx=0),  # 2
                nn.Embedding(ring + 1, d_model, padding_idx=0),  # 9
                nn.Embedding(30, d_model, padding_idx=0),
                nn.Embedding(5, d_model, padding_idx=0),
                GaussianAtomLayer(d_model, means=(-1, 1), stds=(0.1, 10)),
            ]
        )
        self.need_graph_token = need_graph_token

        if need_graph_token:
            self.graph_token = nn.Embedding(1, d_model)
            if known_rxn_cnt:
                self.cnt_token = nn.Embedding(n_rxn_cnt, d_model)
            if known_rxn_type:
                self.type_token = nn.Embedding(10, d_model)
            self.contrast_token = nn.Embedding(1, d_model)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, atom_fea, center_cnt=None, rxn_type=None, contrast=False):
        """
        :param atom_fea: [bsz, n_fea_type, n_atom]
        :return: [bsz, n_atom + I, d_model] I = 1 if need_graph_token else 0
        """
        bsz, n_fea_type, n_atom = atom_fea.size()
        out = self.atom_encoders[-1](atom_fea[:, -1])

        for idx in range(n_fea_type - 1):
            out += self.atom_encoders[idx](atom_fea[:, idx].int())

        if self.need_graph_token:
            graph_token = self.graph_token.weight[0].clone()
            if contrast:
                graph_token += self.contrast_token.weight[0]

            graph_token = graph_token.view(1, 1, -1).repeat(bsz, 1, 1)

            if self.known_rxn_type:
                graph_token += self.type_token(rxn_type).view(bsz, 1, -1)
            if self.known_rxn_cnt:
                graph_token += self.cnt_token(center_cnt).view(bsz, 1, -1)

            out = torch.cat([graph_token, out], dim=1)

        return out


class EdgeEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head=16,
        max_paths=50,
        n_graph_type=6,
        max_single_hop=4,
        n_layers=1,
        need_graph_token=True,
        use_3d_info=False,
        known_rxn_type=True,
        known_rxn_cnt=True,
        use_dist_adj=True,
        n_rxn_cnt=10,
    ):
        super().__init__()
        self.known_rxn_cnt = known_rxn_cnt
        self.known_rxn_type = known_rxn_type
        self.num_heads = n_head
        self.embed_dim = embed_dim
        self.n_graph_type = n_graph_type
        self.max_single_hop = max_single_hop
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        self.max_paths = max_paths
        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.edge_encoders = nn.ModuleList(
            [nn.Embedding(max_paths + 1, n_head, padding_idx=0) for _ in range(n_graph_type)]
        )
        self.spatial_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))
        if self.use_3d_info:
            self.spatial3d_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))

        self.need_graph_token = need_graph_token
        if need_graph_token:
            self.graph_token = nn.Embedding(1, n_head)
            if known_rxn_cnt:
                self.cnt_token = nn.Embedding(n_rxn_cnt, n_head)
            if known_rxn_type:
                self.type_token = nn.Embedding(10, n_head)
            self.contrast_token = nn.Embedding(1, n_head)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(
        self, bond_adj, dist_adj, center_cnt=None, rxn_type=None, dist3d_adj=None, contrast=False
    ):
        """
        :param bond_adj: [bsz, n_atom, n_atom]
        :param dist_adj: [bsz, n_atom, n_atom]
        :param rxn_type: [bsz]
        :return: attention bias with mask [bsz*n_head, n_atom, n_atom]
        """
        bsz, n_atom, _ = bond_adj.size()
        comb_embed = 0
        if self.use_dist_adj and dist_adj is not None:
            comb_embed += self.spatial_encoder(dist_adj)
        if self.use_3d_info and dist3d_adj is not None:
            comb_embed += self.spatial3d_encoder(dist3d_adj)

        if self.max_single_hop > 0:
            for i in range(self.n_graph_type):
                j_hop_embed = bond_adj.long()
                # decode to multi sense embedding
                j_hop_embed = torch.where(j_hop_embed > 0, ((j_hop_embed - 1) >> i) % 2, 0).float()
                base_hop_embed = j_hop_embed
                comb_embed += self.edge_encoders[i](j_hop_embed.int())
                for j in range(1, self.max_single_hop):
                    j_hop_embed = torch.bmm(j_hop_embed, base_hop_embed)
                    j_hop_embed = self.max_paths - torch.relu(self.max_paths - j_hop_embed)
                    comb_embed += self.edge_encoders[i](j_hop_embed.int())

        comb_embed = comb_embed.permute(0, 3, 1, 2)
        mask = torch.where(bond_adj != 0, 0.0, -torch.inf)

        if self.need_graph_token:
            graph_token = self.graph_token.weight[0].clone()
            if contrast:
                graph_token += self.contrast_token.weight[0]
            graph_token = graph_token.view(1, self.num_heads, 1, 1).repeat(bsz, 1, 1, 1)

            if self.known_rxn_type:
                graph_token += self.type_token(rxn_type).view(bsz, self.num_heads, 1, 1)
            if self.known_rxn_cnt:
                graph_token += self.cnt_token(center_cnt).view(bsz, self.num_heads, 1, 1)

            comb_embed = torch.cat(
                [graph_token.expand(bsz, self.num_heads, n_atom, 1), comb_embed], dim=-1
            )
            comb_embed = torch.cat(
                [graph_token.expand(bsz, self.num_heads, 1, n_atom + 1), comb_embed], dim=-2
            )

            mask = torch.cat([torch.zeros_like(mask[:, :, 0]).unsqueeze(-1), mask], dim=-1)
            mask = torch.cat([torch.zeros_like(mask[:, 0, :]).unsqueeze(-2), mask], dim=-2)

            n_atom += 1

        mask = mask.unsqueeze(1).expand(bsz, self.num_heads, n_atom, n_atom)

        return (comb_embed + mask).reshape(bsz * self.num_heads, n_atom, n_atom)


def gaussian(x, mean, std):
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (A * std)


class GaussianAtomLayer(nn.Module):
    def __init__(self, d_model=128, means=(0, 3), stds=(0.1, 10)):
        super().__init__()
        self.d_model = d_model
        self.means = nn.Embedding(1, d_model)
        self.stds = nn.Embedding(1, d_model)
        self.mul = nn.Embedding(1, 1)
        self.bias = nn.Embedding(1, 1)
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, self.d_model)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)

    def forward(self, x):
        """
        :param x: [bsz, n_atom]
        :return: [bsz, n_atom, d_model]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))


class GaussianBondLayer(nn.Module):
    def __init__(self, nhead=16, means=(0, 3), stds=(0.1, 10)):
        super().__init__()
        self.nhead = nhead
        self.means = nn.Embedding(1, nhead)
        self.stds = nn.Embedding(1, nhead)
        self.mul = nn.Embedding(1, 1)
        self.bias = nn.Embedding(1, 1)
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.nhead)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)

    def forward(self, x):
        """
        :param x: [bsz, n_atom, n_atom]
        :return: [bsz, n_atom, n_atom, nhead]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


# --- model/Modules.py ---
class RetroAGTEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(RetroAGTEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, attn_mask=None):
        output = x
        for mod in self.layers:
            output = mod(output, attn_mask)

        if self.norm is not None and self.num_layers > 0:
            output = self.norm(output)

        return output


class RetroAGTEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.gelu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
    ) -> None:
        super(RetroAGTEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAtomAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x, attn_mask=None):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class MultiHeadAtomAdj(nn.Module):
    def __init__(
        self, embed_dim, num_heads, bias=True, add_zero_attn=False, batch_first=False
    ) -> None:
        super(MultiHeadAtomAdj, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.in_proj_weight = nn.Parameter(torch.empty((2 * embed_dim, embed_dim)))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(2 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)

    def forward(self, query, key, attn_mask=None):
        if self.batch_first:
            query, key = [x.transpose(1, 0) for x in (query, key)]
        q, k = _in_projection_packed(query, key, None, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)

        _, adj = _scaled_dot_product_atom_attention(q, k, None, 0.0, attn_mask)

        return adj.view(bsz, self.num_heads, q_d, k_d).permute(0, 2, 3, 1).contiguous()


class MultiHeadAtomAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, bias=True, add_zero_attn=False, batch_first=False
    ) -> None:
        super(MultiHeadAtomAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)

        dropout_p = self.dropout if self.training else 0.0
        attn_output, attn_output_weights = _scaled_dot_product_atom_attention(
            q, k, v, dropout_p, attn_mask
        )

        attn_output = attn_output.transpose(0, 1).contiguous().view(q_d, bsz, self.embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, q_d, k_d)
            return attn_output, attn_output_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attn_output, None


def _scaled_dot_product_atom_attention(q, k, v=None, dropout_p=0.0, attn_mask=None):
    B, Q, D = q.size()
    q = q / math.sqrt(D)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    if v is not None:
        attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.bmm(attn, v) if v is not None else None
    return output, attn


def _in_projection_packed(q, k, v=None, w=None, b=None):
    E = q.size(-1)
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    elif v is not None:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    elif q is k:
        return F.linear(q, w, b).chunk(2, dim=-1)
    else:
        w_q, w_k = w.split([E, E])
        if b is None:
            b_q = b_k = None
        else:
            b_q, b_k = b.split([E, E])
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# --- staging wrapper: mirrors the real RetroAGT.forward's shared embedding -> shared
# encoder -> reaction-center adjacency head path (model/RetroAGT.py lines ~204-214), the
# part of the model that needs no external leaving-group vocabulary / rdkit dependency. ---
class RetroAGTCore(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=8,
        num_shared_layer=2,
        dim_feedforward=128,
        dropout=0.0,
        max_paths=50,
        n_graph_type=6,
        max_single_hop=4,
        atom_dim=90,
        total_degree=50,
        formal_charge=30,
        hybrid=10,
        exp_valance=20,
        hydrogen=20,
        aromatic=2,
        ring=10,
        n_layers=1,
        n_rxn_cnt=7,
        batch_first=True,
        known_rxn_type=True,
        known_rxn_cnt=True,
        norm_first=False,
        activation="gelu",
        use_3d_info=False,
        use_dist_adj=True,
    ):
        super().__init__()
        self.emb = RetroAGTEmbeddingLayer(
            d_model,
            nhead,
            max_paths,
            n_graph_type,
            max_single_hop,
            atom_dim,
            total_degree,
            formal_charge,
            hybrid,
            exp_valance,
            hydrogen,
            aromatic,
            ring,
            n_layers,
            need_graph_token=True,
            use_3d_info=use_3d_info,
            dropout=dropout,
            known_rxn_type=known_rxn_type,
            n_rxn_cnt=n_rxn_cnt,
            known_rxn_cnt=known_rxn_cnt,
            use_dist_adj=use_dist_adj,
        )
        encoder_layer = RetroAGTEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.shared_encoder = RetroAGTEncoder(
            encoder_layer, num_shared_layer, nn.LayerNorm(d_model)
        )
        self.rc_adj_fn = MultiHeadAtomAdj(d_model, nhead, batch_first=batch_first)
        self.rc_out_fn = nn.Sequential(nn.Linear(nhead, 1))

    def forward(self, atom_fea, bond_adj, dist_adj, center_cnt, rxn_type):
        shared_atom_fea, masked_adj = self.emb(atom_fea, bond_adj, dist_adj, center_cnt, rxn_type)
        shared_atom_fea = self.shared_encoder(shared_atom_fea, masked_adj)
        rc_fea = shared_atom_fea

        rc_prob = self.rc_adj_fn(rc_fea[:, 1:], rc_fea[:, 1:], None)
        rc_adj_prob = self.rc_out_fn(rc_prob).squeeze(-1) + torch.where(bond_adj > 1, 0.0, -10.0)
        return rc_adj_prob


def build_retroexplainer():
    return RetroAGTCore(
        d_model=64,
        nhead=8,
        num_shared_layer=2,
        dim_feedforward=128,
        max_paths=50,
        n_graph_type=6,
        max_single_hop=2,
        atom_dim=90,
        total_degree=50,
        formal_charge=30,
        hybrid=10,
        exp_valance=20,
        hydrogen=20,
        aromatic=2,
        ring=10,
        n_rxn_cnt=7,
    )


def example_input_retroexplainer():
    bsz, n_atom, n_fea_type = 2, 8, 9
    atom_fea = torch.zeros(bsz, n_fea_type, n_atom, dtype=torch.long)
    atom_fea[:, 0] = torch.randint(1, 20, (bsz, n_atom))  # atomic_num idx (padding_idx=0)
    atom_fea[:, 1] = torch.randint(1, 5, (bsz, n_atom))  # degree
    atom_fea[:, 2] = torch.randint(1, 5, (bsz, n_atom))  # hybridization
    atom_fea[:, 3] = torch.randint(1, 5, (bsz, n_atom))  # num_hs
    atom_fea[:, 4] = torch.randint(1, 2, (bsz, n_atom))  # aromatic
    atom_fea[:, 5] = torch.randint(1, 5, (bsz, n_atom))  # ring
    atom_fea[:, 6] = torch.randint(1, 20, (bsz, n_atom))  # formal charge idx
    atom_fea[:, 7] = torch.randint(1, 4, (bsz, n_atom))  # extra idx
    atom_fea_float = atom_fea.clone().float()
    atom_fea_float[:, 8] = torch.rand(bsz, n_atom) * 2 - 1  # Gaussian charge feature (real-valued)
    bond_adj = (torch.rand(bsz, n_atom, n_atom) > 0.6).float()
    bond_adj = torch.triu(bond_adj, diagonal=1)
    bond_adj = bond_adj + bond_adj.transpose(1, 2)
    dist_adj = torch.rand(bsz, n_atom, n_atom) * 3
    dist_adj = (dist_adj + dist_adj.transpose(1, 2)) / 2
    center_cnt = torch.randint(0, 7, (bsz,))
    rxn_type = torch.randint(0, 10, (bsz,))
    return (atom_fea_float, bond_adj, dist_adj, center_cnt, rxn_type)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RetroExplainer", "build_retroexplainer", "example_input_retroexplainer", 2023, "vendored"),
]
