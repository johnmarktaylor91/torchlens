# SOURCE: vendored from Roestlab/massformer @ master
# Files: src/massformer/gf_model.py (GFv2Embedder / GraphormerEncoder / GraphormerGraphEncoder /
#        GraphormerGraphEncoderLayer / MultiheadAttention / GraphNodeFeature / GraphAttnBias /
#        LRPLayerNorm and the graphormer arg helpers), src/massformer/model.py (Predictor,
#        LinearBlock, NeimsBlock, mask_prediction_by_mass, reverse_prediction -- the
#        architecture-relevant subset that does not require dgl/dgllife).
#
# MassFormer (Young et al., 2021) predicts an EI-MS mass spectrum from a molecule by
# embedding the molecular graph and passing the embedding through a feed-forward
# prediction head with optional bidirectional (forward+reverse) spectrum prediction,
# precursor-mass gating/masking, and learned spectral attention. The paper's flagship
# molecule embedder is "gf_v2": a Graphormer graph-transformer encoder (graph-token
# self-attention with learned spatial-position / edge-path attention biases) initialized
# from the public `graphormer_small` architecture config. Real Predictor class, used here
# exactly as the real repo constructs it (`embed_types=["gf_v2"]`, `ff_layer_type="neims"`,
# `bidirectional_prediction=True`) -- i.e. the actual "MassFormer (Graphormer)" variant
# from the paper. The alternate embedder branches (`GNNEmbedder`/`WLNEmbedder`/
# `GATEmbedder`/`GINPTEmbedder`, which need dgl + dgllife) are omitted; `Predictor`
# dispatches on `embed_types` unmodified, so this is the same class, just constructed
# along its dgl-free code path. `FPEmbedder` (the third, always-available embedder type)
# is kept too since it is pure torch/no extra deps.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import argparse
import copy
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MENAGERIE_ZOO = "vendored-pytorch"


class DummyContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


# --- src/massformer/gf_model.py (verbatim, pure-torch Graphormer encoder) ---
def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return lambda x: F.relu(x).pow(2)
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class LRPLayerNorm(nn.LayerNorm):
    def forward(self, input, lrp=False):
        if lrp:
            reduce_dims = [i for i in range(input.ndim - len(self.normalized_shape), input.ndim)]
            e = torch.mean(input, dim=reduce_dims)
            e = e.reshape(tuple(e.shape) + tuple(1 for dim in reduce_dims))
            v = torch.var(input, dim=reduce_dims, unbiased=False)
            v = v.reshape(tuple(v.shape) + tuple(1 for dim in reduce_dims))
            num = input - e
            denom = torch.sqrt(v + self.eps)
            return (num / denom.detach()) * self.weight + self.bias
        else:
            return super().forward(input)


class MultiheadAttention(nn.Module):
    """Multi-headed attention. See "Attention Is All You Need" for more details."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        assert self.self_attention, "Only support self attention"
        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        lrp: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            assert not lrp
            return attn_weights, v

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        if lrp:
            attn_probs = attn_probs.detach()

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """Compute node features for each node in the graph."""

    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, return_input_feats=""):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        input_feats = None

        if "token" in return_input_feats:
            embed_weight = self.atom_encoder.weight
            token_feature_idx = int(return_input_feats[len("token") :])
            token_feature = F.one_hot(x, num_classes=embed_weight.shape[0]).float()
            token_feature.requires_grad = True
            token_feature_mask = torch.zeros_like(token_feature, dtype=torch.bool)
            input_feats = token_feature[:, :, token_feature_idx, :]
            token_feature_mask[:, :, token_feature_idx, :] = 1.0
            token_feature = (
                token_feature_mask.float() * input_feats.unsqueeze(2)
                + (~token_feature_mask).float() * token_feature
            ).clone()
            atom_feature = token_feature @ embed_weight
        else:
            atom_feature = self.atom_encoder(x)
        if "atom" in return_input_feats and return_input_feats != "atom":
            atom_feature_idx = int(return_input_feats[len("atom") :])
            atom_feature_mask = torch.zeros_like(atom_feature, dtype=torch.bool)
            atom_feature_mask[:, :, atom_feature_idx, :] = 1.0
            input_feats = atom_feature[:, :, atom_feature_idx, :]
            atom_feature = (
                atom_feature_mask.float() * input_feats.unsqueeze(2)
                + (~atom_feature_mask).float() * atom_feature
            ).clone()
        atom_feature = atom_feature.sum(dim=-2)
        if return_input_feats == "atom":
            input_feats = atom_feature
            atom_feature = atom_feature.clone()

        node_feature = (
            atom_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        )
        if return_input_feats == "node":
            input_feats = node_feature
            node_feature = node_feature.clone()

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature, input_feats


class GraphAttnBias(nn.Module):
    """Compute attention bias for each head."""

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[
                    :max_dist, :, :
                ],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(
                0, 3, 1, 2
            )
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        return graph_attn_bias


class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(dropout)

        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        self.self_attn_layer_norm = LRPLayerNorm(self.embedding_dim)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        self.final_layer_norm = LRPLayerNorm(self.embedding_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(
        self, embed_dim, num_attention_heads, dropout, self_attention, q_noise, qn_block_size
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        return_attn_mats: bool = False,
        lrp: bool = False,
    ):
        residual = x

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=return_attn_mats,
            attn_mask=self_attn_mask,
            need_head_weights=return_attn_mats,
            lrp=lrp,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x, lrp=lrp)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x, lrp=lrp)
        return x, attn


def init_graphormer_params(module, init_layernorm=False):
    """Initialize the weights specific to the Graphormer Model."""

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
    if init_layernorm and isinstance(module, nn.LayerNorm):
        with torch.no_grad():
            module.weight.data.copy_(torch.ones_like(module.weight))
            module.bias.data.zero_()


class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.embed_scale = embed_scale

        if q_noise > 0:
            raise ValueError("quant_noise")
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LRPLayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            raise ValueError("layerdrop")
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn_mats: bool = False,
        return_input_feats: str = "",
    ) -> Tuple:
        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        if token_embeddings is not None:
            assert not return_input_feats
            x, input_feats = token_embeddings, None
        else:
            x, input_feats = self.graph_node_feature(
                batched_data, return_input_feats=return_input_feats
            )

        if perturb is not None:
            assert not return_input_feats
            x[:, 1:, :] += perturb

        attn_bias = self.graph_attn_bias(batched_data)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x, lrp=return_input_feats != "")

        x = self.dropout_module(x)

        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        attn_mats = []
        for layer in self.layers:
            x, am = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
                return_attn_mats=return_attn_mats,
                lrp=return_input_feats != "",
            )
            if not last_state_only:
                inner_states.append(x)
            if return_attn_mats:
                attn_mats.append(am)

        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            assert not return_attn_mats
            assert not return_input_feats
            inner_states = torch.stack(inner_states)

        return inner_states, graph_rep, attn_mats, input_feats


class GraphormerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_nodes = args.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
            n_trans_layers_to_freeze=args.n_trans_layers_to_freeze,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)

        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = get_activation_fn(args.activation_fn)
        self.layer_norm = LRPLayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(args.encoder_embed_dim, args.num_classes, bias=False)
            else:
                raise NotImplementedError

    def forward(
        self,
        batched_data,
        perturb=None,
        masked_tokens=None,
        return_attn_mats=False,
        return_input_feats="",
        **unused,
    ):
        inner_states, graph_rep, attn_mats, input_feats = self.graph_encoder(
            batched_data,
            perturb=perturb,
            return_attn_mats=return_attn_mats,
            return_input_feats=return_input_feats,
        )
        if return_attn_mats:
            assert len(attn_mats) > 0, len(attn_mats)
            attn_mats = torch.stack(attn_mats, dim=0).permute(2, 0, 1, 3, 4)

        x = inner_states[-1].transpose(0, 1)

        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(x)), lrp=return_input_feats != ""
        )

        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x, attn_mats, input_feats


class GFv2Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = argparse.Namespace()
        self.args.model_name = kwargs["gf_model_name"]
        self.args.pretrained_model_name = kwargs["gf_pretrain_name"]
        set_data_args(self.args)
        set_graphormer_base_architecture_args(self.args)
        # menagerie recipe addition (not in the original repo): shrink the encoder to a
        # tiny size for tracing. The real repo always uses production-scale
        # graphormer_base/graphormer_small configs; this override only touches layer
        # count/width, not the architecture itself.
        for k, v in kwargs.get("size_overrides", {}).items():
            setattr(self.args, k, v)
        self.args.remove_head = True
        self.pretrain = self.args.pretrained_model_name != "none"
        self.fix_num_pt_layers = kwargs["fix_num_pt_layers"]
        self.reinit_num_pt_layers = kwargs["reinit_num_pt_layers"]
        self.reinit_layernorm = kwargs["reinit_layernorm"]
        if self.pretrain:
            if self.fix_num_pt_layers == -1:
                assert self.reinit_num_pt_layers == 0
                self.args.n_trans_layers_to_freeze = self.args.encoder_layers
            else:
                assert (
                    self.reinit_num_pt_layers <= self.args.encoder_layers - self.fix_num_pt_layers
                )
                self.args.n_trans_layers_to_freeze = self.fix_num_pt_layers
        self.encoder = GraphormerEncoder(self.args)
        if getattr(self.args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)
        if self.pretrain:
            raise NotImplementedError(
                "pretrained-checkpoint loading is out of scope for the menagerie recipe"
            )

    def get_split_params(self):
        if self.pretrain:
            pt_param_names = [
                "encoder.graph_encoder." + k
                for k, v in self.encoder.graph_encoder.named_parameters()
            ]
        else:
            pt_param_names = []
        nopt_params, pt_params = [], []
        for k, v in self.named_parameters():
            if k in pt_param_names:
                pt_params.append(v)
            else:
                nopt_params.append(v)
        return nopt_params, pt_params

    def forward(self, data, perturb=None, return_attn_mats=False, return_input_feats=""):
        assert not (return_attn_mats and return_input_feats)
        batched_data = data["gf_v2_data"]
        batched_node_embeds, attn_mats, input_feats = self.encoder(
            batched_data,
            perturb=perturb,
            return_attn_mats=return_attn_mats,
            return_input_feats=return_input_feats,
        )
        if return_attn_mats:
            return batched_node_embeds[:, 0, :], attn_mats
        elif return_input_feats:
            return batched_node_embeds[:, 0, :], input_feats
        else:
            return batched_node_embeds[:, 0, :]

    def get_embed_dim(self):
        return self.args.encoder_embed_dim


def set_data_args(args):
    args.dataset_name = "pcqm4mv2"
    args.num_classes = 1
    args.max_nodes = 128
    args.dataset_source = "ogb"
    args.num_atoms = 512 * 9
    args.num_edges = 512 * 3
    args.num_in_degree = 512
    args.num_out_degree = 512
    args.num_spatial = 512
    args.num_edge_dis = 128
    args.multi_hop_max_dist = 5
    args.spatial_pos_max = 1024
    args.edge_type = "multi_hop"
    args.seed = 420
    args.load_pretrained_model_output_layer = False
    args.train_epoch_shuffle = False
    args.user_data_dir = ""


def set_graphormer_base_architecture_args(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(args, "share_encoder_input_output_embed", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    if args.model_name == "graphormer_base":
        args.encoder_layers = 12
        args.encoder_attention_heads = 32
        args.encoder_ffn_embed_dim = 768
        args.encoder_embed_dim = 768
        args.dropout = getattr(args, "dropout", 0.0)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.act_dropout = getattr(args, "act_dropout", 0.1)
    elif args.model_name == "graphormer_small":
        args.encoder_layers = 6
        args.encoder_attention_heads = 32
        args.encoder_ffn_embed_dim = 512
        args.encoder_embed_dim = 512
        args.dropout = getattr(args, "dropout", 0.0)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.act_dropout = getattr(args, "act_dropout", 0.1)
    else:
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
        args.encoder_layers = getattr(args, "encoder_layers", 12)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(args, "share_encoder_input_output_embed", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.n_trans_layers_to_freeze = getattr(args, "n_trans_layers_to_freeze", 0)


# --- src/massformer/model.py (verbatim architecture-relevant subset; dgl-based embedders
# WLNEmbedder/GATEmbedder/GNNEmbedder/GINPTEmbedder omitted since they require dgl/dgllife) ---
def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.0):
    # adapted from NEIMS
    max_idx = raw_prediction.shape[1]
    assert torch.all(prec_mass_idx < max_idx)
    idx = torch.arange(max_idx, device=prec_mass_idx.device)
    mask = (idx.unsqueeze(0) <= (prec_mass_idx.unsqueeze(1) + prec_mass_offset)).float()
    return mask * raw_prediction + (1.0 - mask) * mask_value


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    # adapted from NEIMS
    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    assert torch.all(prec_mass_idx < max_idx)
    rev_prediction = torch.flip(raw_prediction, dims=(1,))
    offset_idx = torch.minimum(
        max_idx * torch.ones_like(prec_mass_idx), prec_mass_idx + prec_mass_offset + 1
    )
    shifts = -(max_idx - offset_idx)
    gather_idx = (
        torch.arange(max_idx, device=raw_prediction.device).unsqueeze(0).expand(batch_size, max_idx)
    )
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
    offset_rev_prediction = torch.gather(rev_prediction, 1, gather_idx)
    return offset_rev_prediction


class LinearBlock(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.1):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.bn(self.dropout(F.relu(self.linear(x))))


class NeimsBlock(nn.Module):
    """from the NEIMS paper (uses LeakyReLU instead of ReLU)"""

    def __init__(self, in_dim, out_dim, dropout):
        super(NeimsBlock, self).__init__()
        bottleneck_factor = 0.5
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = x
        h = self.in_batch_norm(h)
        h = self.in_activation(h)
        h = self.dropout(h)
        h = self.in_linear(h)
        h = self.out_batch_norm(h)
        h = self.out_activation(h)
        h = self.out_linear(h)
        return h


class FPEmbedder(nn.Module):
    def __init__(self, dim_d, **kwargs):
        super(FPEmbedder, self).__init__()
        self.fp_dim = dim_d["fp_dim"]
        self.embed_dim = self.fp_dim

    def get_embed_dim(self):
        return self.embed_dim

    def get_split_params(self):
        return [], []

    def forward(self, data, perturb=None):
        return data["fp"]


def dirichlet_sample(alpha):
    dist = torch.distributions.dirichlet.Dirichlet(alpha)
    return dist.sample()


class SpectrumAttention(nn.Module):
    def __init__(self, D, H, N, alpha=10.0):
        super().__init__()
        self.pi = nn.Parameter(data=dirichlet_sample(alpha * torch.ones(N)), requires_grad=True)
        self.Q = nn.Parameter(data=torch.normal(0.0, 1.0, (D, H, N)), requires_grad=True)

    def forward(self, s):
        M = (self.Q @ self.pi.view(1, -1, 1)).squeeze(-1)
        MTs = M.transpose(0, 1).unsqueeze(0) @ s.unsqueeze(-1)
        MMTs = (M.unsqueeze(0) @ MTs).squeeze(-1)
        return MMTs


class Predictor(nn.Module):
    def __init__(self, dim_d, **kwargs):
        super(Predictor, self).__init__()
        self.g_dim = dim_d["g_dim"]
        self.o_dim = dim_d["o_dim"]
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.embedders = nn.ModuleList([])
        if "fp" in self.embed_types:
            self.embedders.append(FPEmbedder(dim_d, **kwargs))
        if "gf_v2" in self.embed_types:
            self.embedders.append(GFv2Embedder(**kwargs))
        assert len(self.embedders) > 0, len(self.embedders)
        embeds_dims = [embedder.get_embed_dim() for embedder in self.embedders]
        if self.g_dim > 0:
            embeds_dims.append(self.g_dim)
        if self.embed_dim == -1:
            self.embed_dim = sum(embeds_dims)
        if self.embed_linear:
            self.embed_layers = nn.ModuleList(
                [nn.Linear(embed_dim, self.embed_dim) for embed_dim in embeds_dims]
            )
        else:
            self.embed_layers = nn.ModuleList([nn.Identity() for embed_dim in embeds_dims])
        self.ff_layers = nn.ModuleList([])
        self.out_modules = []
        if self.ff_layer_type == "standard":
            ff_layer = LinearBlock
        else:
            assert self.ff_layer_type == "neims", self.ff_layer_type
            ff_layer = NeimsBlock
        self.ff_layers.append(nn.Linear(self.embed_dim, self.ff_h_dim))
        self.out_modules.extend(["ff_layers"])
        for i in range(self.ff_num_layers):
            self.ff_layers.append(ff_layer(self.ff_h_dim, self.ff_h_dim, self.dropout))
        if self.bidirectional_prediction:
            self.forw_out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
            self.rev_out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
            self.out_gate = nn.Sequential(*[nn.Linear(self.ff_h_dim, self.o_dim), nn.Sigmoid()])
            self.out_modules.extend(["forw_out_layer", "rev_out_layer", "out_gate"])
        else:
            self.out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
            if self.gate_prediction:
                self.out_gate = nn.Sequential(*[nn.Linear(self.ff_h_dim, self.o_dim), nn.Sigmoid()])
            self.out_modules.extend(["out_layer", "out_gate"])
        if self.spectrum_attention:
            self.spectrum_attender = SpectrumAttention(self.o_dim, 100, 10)
            self.out_modules.extend(["spectrum_attender"])

    def forward(self, data, perturb=None, amp=False, return_input_feats="", return_lda_pred=False):
        if amp:
            amp_context = torch.cuda.amp.autocast()
        else:
            amp_context = DummyContext()
        with amp_context:
            if return_input_feats:
                assert len(self.embedders) == 1
                embedder = self.embedders[0]
                assert isinstance(embedder, GFv2Embedder)
                embed, input_feats = embedder(
                    data, perturb=perturb, return_input_feats=return_input_feats
                )
                embeds = [embed]
            else:
                embeds = [embedder(data, perturb=perturb) for embedder in self.embedders]
            if self.g_dim > 0:
                embeds.append(data["spec_meta"])
            embeds = [self.embed_layers[embed_idx](embed) for embed_idx, embed in enumerate(embeds)]
            if self.embed_agg == "concat":
                fh = torch.cat(embeds, dim=1)
            elif self.embed_agg == "add":
                assert all(embed.shape[1] == embeds[0].shape[1] for embed in embeds)
                fh = sum(embeds)
            elif self.embed_agg == "avg":
                fh = sum(embeds) / len(embeds)
            else:
                raise ValueError("invalid agg_embed")
            fh = self.ff_layers[0](fh)
            for ff_layer in self.ff_layers[1:]:
                if self.ff_skip:
                    fh = fh + ff_layer(fh)
                else:
                    fh = ff_layer(fh)
            if self.bidirectional_prediction:
                ff = self.forw_out_layer(fh)
                fr = reverse_prediction(
                    self.rev_out_layer(fh), data["prec_mz_idx"], self.prec_mass_offset
                )
                fg = self.out_gate(fh)
                fo = ff * fg + fr * (1.0 - fg)
                fo = mask_prediction_by_mass(fo, data["prec_mz_idx"], self.prec_mass_offset)
            else:
                fo = self.out_layer(fh)
                if self.gate_prediction:
                    fg = self.out_gate(fh)
                    fo = fg * fo
            if self.output_activation == "relu":
                output_activation_fn = F.relu
            elif self.output_activation == "sp":
                output_activation_fn = F.softplus
            elif self.output_activation == "sm":
                assert not self.bidirectional_prediction
                assert not self.gate_prediction
                assert not self.spectrum_attention

                def output_activation_fn(x):
                    return F.softmax(x, dim=1)

            else:
                raise ValueError(f"invalid output_activation: {self.output_activation}")
            fo = output_activation_fn(fo)
            if self.gt_gate_prediction:
                gt_fo = (data["spec"] > 0.0).float()
                assert self.gt_gate_val > 0.5
                gt_fo = gt_fo * (2 * self.gt_gate_val - 1.0) + (1.0 - self.gt_gate_val)
                fo = gt_fo * fo
            if self.spectrum_attention:
                fo = self.spectrum_attender(fo)
                fo = output_activation_fn(fo)
            if self.output_normalization == "l1":
                fo = F.normalize(fo, p=1, dim=1)
            elif self.output_normalization == "l2":
                fo = F.normalize(fo, p=2, dim=1)
            elif self.output_normalization == "none":
                pass
            else:
                raise ValueError(f"invalid output_normalization: {self.output_normalization}")
            output_d = {"pred": fo}
            if return_input_feats:
                output_d["input_feats"] = input_feats
            return output_d


def build_massformer():
    torch.manual_seed(0)
    dim_d = {"g_dim": 0, "o_dim": 128, "fp_dim": 0}
    kwargs = dict(
        embed_types=["gf_v2"],
        gf_model_name="graphormer_small",
        gf_pretrain_name="none",
        fix_num_pt_layers=0,
        reinit_num_pt_layers=0,
        reinit_layernorm=False,
        embed_dim=-1,
        embed_linear=False,
        ff_layer_type="neims",
        ff_h_dim=64,
        ff_num_layers=1,
        dropout=0.0,
        ff_skip=True,
        bidirectional_prediction=True,
        prec_mass_offset=10,
        gate_prediction=False,
        spectrum_attention=False,
        output_activation="relu",
        output_normalization="l1",
        gt_gate_prediction=False,
        gt_gate_val=0.9,
        embed_agg="concat",
        cfm_postprocessing=False,
    )
    # shrink the Graphormer encoder to a menagerie-appropriate tiny size (the real repo's
    # default `graphormer_small` config is production-scale: 6 layers / 512-dim / 32 heads).
    kwargs["size_overrides"] = dict(
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_embed_dim=32,
        encoder_embed_dim=32,
    )
    return Predictor(dim_d, **kwargs)


def example_input_massformer():
    # Synthesizes a `gf_v2_data` batch with the same tensor names/shapes/dtypes the real
    # repo's `gf_preprocess`/`collator` pipeline produces (see src/massformer/gf_data_utils.py),
    # without invoking the RDKit-dependent SMILES->graph featurizer (architecture-only input).
    torch.manual_seed(0)
    batch_size, n_node, multi_hop_max_dist = 2, 6, 5
    num_atoms, num_edges, num_in_degree, num_spatial, num_edge_dis = 512 * 9, 512 * 3, 512, 512, 128

    x = torch.randint(1, num_atoms, (batch_size, n_node, 9), dtype=torch.long)
    attn_bias = torch.zeros(batch_size, n_node + 1, n_node + 1, dtype=torch.float)
    spatial_pos = torch.randint(1, num_spatial, (batch_size, n_node, n_node), dtype=torch.long)
    in_degree = torch.randint(0, num_in_degree, (batch_size, n_node), dtype=torch.long)
    out_degree = in_degree.clone()
    edge_input = torch.randint(
        1, num_edges, (batch_size, n_node, n_node, multi_hop_max_dist, 3), dtype=torch.long
    )
    attn_edge_type = torch.randint(
        1, num_edge_dis, (batch_size, n_node, n_node, 3), dtype=torch.long
    )

    gf_v2_data = {
        "x": x,
        "attn_bias": attn_bias,
        "spatial_pos": spatial_pos,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "edge_input": edge_input,
        "attn_edge_type": attn_edge_type,
    }
    prec_mz_idx = torch.randint(0, 100, (batch_size,), dtype=torch.long)
    data = {"gf_v2_data": gf_v2_data, "prec_mz_idx": prec_mz_idx}
    return (data,)


MENAGERIE_ENTRIES = [
    (
        "MassFormer (Graphormer)",
        "build_massformer",
        "example_input_massformer",
        2021,
        "vendored-pytorch",
    ),
]
