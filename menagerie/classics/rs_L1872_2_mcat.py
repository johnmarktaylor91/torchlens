# SOURCE: vendored from https://github.com/mahmoodlab/MCAT @ master
# (models/model_coattn.py::MCAT_Surv + its bundled `multi_head_attention_forward`/
#  `MultiheadAttention` patch, and models/model_utils.py::{Attn_Net_Gated, SNN_Block}).
# Class bodies are copied verbatim (only import paths adjusted to be self-contained
# in one file; `torch.nn.modules.linear._LinearWithBias`, removed from modern torch,
# is replaced with the behaviorally-identical `nn.Linear(embed_dim, embed_dim)`
# (bias=True by default) -- the SAME substitution the real class made internally,
# not an architectural change). MCAT ("Multimodal Co-Attention Transformer",
# ICCV 2021) is a whole-slide-image + genomic co-attention survival model: WSI
# patch features are FC-projected, six genomic-pathway signatures are each run
# through their own SNN (self-normalizing) block, then a genomic-query /
# histology-key-value multihead co-attention (the repo's own patched
# `MultiheadAttention`, which additionally returns raw per-head attention maps)
# fuses the two modalities; each fused stream is further encoded by its own
# TransformerEncoder + gated-attention pooling head, concatenated (or bilinearly
# fused), and mapped to a 4-way discrete-time hazard classifier. `fusion="concat"`
# is used here (the repo's `bilinear` fusion path hardcodes `torch.cuda.FloatTensor`
# and is GPU-only bookkeeping, not additional architecture -- concat is the other
# real, CPU-safe fusion mode already in the repo). This co-attention + dual
# transformer + SNN-pathway fusion topology is not any base-lib class, so it is
# vendored in full rather than approximated.
"""Vendored MCAT (Multimodal Co-Attention Transformer) survival model definition."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import dropout, linear, pad, softmax
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/model_utils.py
# ---------------------------------------------------------------------------


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)"""
    return nn.Sequential(nn.Linear(dim1, dim2), nn.ELU(), nn.AlphaDropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""Attention Network with Sigmoid Gating (3 fc layers)"""
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


# ---------------------------------------------------------------------------
# models/model_coattn.py :: repo-patched multi_head_attention_forward /
# MultiheadAttention (returns raw, un-averaged per-head attention weights via
# `need_raw`, used for the co-attention step).
# ---------------------------------------------------------------------------


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        if need_raw:
            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_raw
        else:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information from different
    representation subspaces. See reference: Attention Is All You Need"""

    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        # NOTE: real code used `torch.nn.modules.linear._LinearWithBias`,
        # which no longer exists in modern torch; `nn.Linear(..., bias=True)`
        # (the default) is behaviorally identical -- not an architecture change.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        need_raw=True,
        attn_mask=None,
    ):
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
            )


# ---------------------------------------------------------------------------
# models/model_coattn.py :: MCAT_Surv
# ---------------------------------------------------------------------------


class MCAT_Surv(nn.Module):
    def __init__(
        self,
        fusion="concat",
        omic_sizes=[100, 200, 300, 400, 500, 600],
        n_classes=4,
        model_size_wsi: str = "small",
        model_size_omic: str = "small",
        dropout=0.25,
    ):
        super(MCAT_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {"small": [256, 256], "big": [1024, 1024, 1024, 256]}

        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation="relu"
        )
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(
            L=size[2], D=size[2], dropout=dropout, n_classes=1
        )
        self.path_rho = nn.Sequential(
            *[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation="relu"
        )
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(
            L=size[2], D=size[2], dropout=dropout, n_classes=1
        )
        self.omic_rho = nn.Sequential(
            *[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        ### Fusion Layer -- "concat" (the CPU-safe real fusion mode; see header note)
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(256 * 2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()]
            )
        else:
            self.mm = None

        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, x_path, x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6):
        x_omic = [x_omic1, x_omic2, x_omic3, x_omic4, x_omic5, x_omic6]

        h_path_bag = self.wsi_net(x_path).unsqueeze(
            1
        )  ### path embeddings are fed through a FC layer
        h_omic = [
            self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)
        ]  ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(
            1
        )  ### omic embeddings are stacked (to be used in co-attention)

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag)

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze()

        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()

        h = self.mm(torch.cat([h_path, h_omic], axis=0))

        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny WSI bag size for fast tracing).
# ---------------------------------------------------------------------------

_OMIC_SIZES = [8, 8, 8, 8, 8, 8]


def build_mcat():
    return MCAT_Surv(
        fusion="concat",
        omic_sizes=_OMIC_SIZES,
        n_classes=4,
        model_size_wsi="small",
        model_size_omic="small",
        dropout=0.0,
    )


def example_input_mcat():
    # Real usage always has batch_size=1 (variable-size WSI bags); the repo's
    # own `collate_MIL_survival_sig` torch.cat's 1D per-sample omic tensors
    # along dim=0, so each x_omic_i arriving in forward() is 1D [size_i], not
    # [1, size_i] -- matching torch.stack(h_omic).unsqueeze(1) being 3D like
    # h_path_bag (see header note).
    n_patches = 6  # small WSI patch bag
    x_path = torch.randn(n_patches, 1024)
    x_omics = tuple(torch.randn(size) for size in _OMIC_SIZES)
    return (x_path,) + x_omics


MENAGERIE_ENTRIES = [
    ("MCAT", "build_mcat", "example_input_mcat", 2021, "vendored-pytorch"),
]
