# SOURCE: vendored from https://github.com/Shujun-He/RibonanzaNet @ 991e9c5cb430b98bc9d0ee6979dea480b2598e06
#
# RibonanzaNet (Shujun He et al.) -- a 1D dilated-conv + pairwise-triangle-attention
# transformer for per-nucleotide RNA chemical-mapping reactivity prediction, winning
# entry of the Stanford OpenVaccine "Ribonanza" Kaggle competition. Vendored verbatim
# (architecture-relevant classes only) from the repo's own files:
#   https://raw.githubusercontent.com/Shujun-He/RibonanzaNet/991e9c5cb430b98bc9d0ee6979dea480b2598e06/Network.py
#   https://raw.githubusercontent.com/Shujun-He/RibonanzaNet/991e9c5cb430b98bc9d0ee6979dea480b2598e06/dropout.py
#
# What is kept: MultiHeadAttention, ConvTransformerEncoderLayer (conv + pairwise-biased
# self-attention + AlphaFold-style TriangleMultiplicativeModule/TriangleAttention pair
# update), Outer_Product_Mean, relpos, and the top-level RibonanzaNet module -- every
# mechanism in the real architecture, transcribed unmodified.
#
# What is dropped (data plumbing, not architecture): matplotlib debug-plot calls inside
# ScaledDotProductAttention.forward were commented-out in the original and are omitted
# here; the `Config`/YAML loader from Functions.py is replaced with a plain kwargs
# `types.SimpleNamespace`-style Config class (identical field semantics, same
# `Config(**entries)` constructor signature used by the original `load_config_from_yaml`).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math
from functools import partialmethod
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from dropout.py (Copyright 2021 AlQuraishi Laboratory, Apache-2.0)
# ---------------------------------------------------------------------------
class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    """Convenience class for rowwise dropout as described in subsection 1.11.6."""

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """Convenience class for columnwise dropout as described in subsection 1.11.6."""

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


# ---------------------------------------------------------------------------
# from Network.py
# ---------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, attn_mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn + mask  # this is actually the bias

        if attn_mask is not None:
            attn = attn.float().masked_fill(attn_mask == -1, float("-1e-9"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, src_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        if src_mask is not None:
            src_mask[src_mask == 0] = -1
            src_mask = src_mask.unsqueeze(-1).float()
            attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1)).unsqueeze(1)
            q, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class TriangleMultiplicativeModule(nn.Module):
    def __init__(self, *, dim, hidden_dim=None, mix="ingoing"):
        super().__init__()
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, src_mask=None):
        src_mask = src_mask.unsqueeze(-1).float()
        mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))
        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"
        if exists(mask):
            mask = rearrange(mask, "b i j -> b i j ()")

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise="row"):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.to_out = nn.Linear(n_heads * dim, in_dim)

    def forward(self, z, src_mask):
        src_mask[src_mask == 0] = -1
        src_mask = src_mask.unsqueeze(-1).float()
        attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))

        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(lambda x: rearrange(x, "b i j (h d)->b i j h d", h=self.n_heads), (q, k, v))
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** 0.5
        if wise == "row":
            eq_attn = "brihd,brjhd->brijh"
            eq_multi = "brijh,brjhd->brihd"
            b = rearrange(b, "b i j (r h)->b r i j h", r=1)
            softmax_dim = 3
            attn_mask = rearrange(attn_mask, "b i j->b 1 i j 1")
        elif wise == "col":
            eq_attn = "bilhd,bjlhd->bijlh"
            eq_multi = "bijlh,bjlhd->bilhd"
            b = rearrange(b, "b i j (l h)->b i j l h", l=1)
            softmax_dim = 2
            attn_mask = rearrange(attn_mask, "b i j->b i j 1 1")
        else:
            raise ValueError("wise should be col or row!")
        logits = torch.einsum(eq_attn, q, k) / scale + b
        logits = logits.masked_fill(attn_mask == -1, float("-1e-9"))
        attn = logits.softmax(softmax_dim)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, "b i j h d-> b i j (h d)")
        z_ = self.to_out(out)
        return z_


class Outer_Product_Mean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super(Outer_Product_Mean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(self, seq_rep, pair_rep=None):
        seq_rep = self.proj_down1(seq_rep)
        outer_product = torch.einsum("bid,bjc -> bijcd", seq_rep, seq_rep)
        outer_product = rearrange(outer_product, "b i j c d -> b i j (c d)")
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product = outer_product + pair_rep

        return outer_product


class relpos(nn.Module):
    def __init__(self, dim=64):
        super(relpos, self).__init__()
        self.linear = nn.Linear(17, dim)

    def forward(self, src):
        L = src.shape[1]
        res_id = torch.arange(L).to(src.device).unsqueeze(0)
        device = res_id.device
        bin_values = torch.arange(-8, 9, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(8, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p


class ConvTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        pairwise_dimension,
        use_triangular_attention,
        dropout=0.1,
        k=3,
    ):
        super(ConvTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        self.conv = nn.Conv1d(d_model, d_model, k, padding=k // 2)

        self.triangle_update_out = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="outgoing"
        )
        self.triangle_update_in = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="ingoing"
        )

        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutRowwise(dropout)

        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="row"
            )
            self.triangle_attention_in = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="col"
            )

            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=d_model, pairwise_dim=pairwise_dimension
        )

        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension),
        )

    def forward(self, src, pairwise_features, src_mask=None, return_aw=False):
        src = src * src_mask.float().unsqueeze(-1)

        res = src  # noqa: F841 -- unused in the original repo code too; kept for fidelity
        src = src + self.conv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.norm3(src)

        pairwise_bias = self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(
            0, 3, 1, 2
        )
        src2, attention_weights = self.self_attn(
            src, src, src, mask=pairwise_bias, src_mask=src_mask
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        pairwise_features = pairwise_features + self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pair_dropout_out(
            self.triangle_update_out(pairwise_features, src_mask)
        )
        pairwise_features = pairwise_features + self.pair_dropout_in(
            self.triangle_update_in(pairwise_features, src_mask)
        )
        if self.use_triangular_attention:
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(
                self.triangle_attention_out(pairwise_features, src_mask)
            )
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(
                self.triangle_attention_in(pairwise_features, src_mask)
            )
        pairwise_features = pairwise_features + self.pair_transition(pairwise_features)
        if return_aw:
            return src, pairwise_features, attention_weights
        else:
            return src, pairwise_features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class RibonanzaNet(nn.Module):
    def __init__(self, config):
        super(RibonanzaNet, self).__init__()
        self.config = config
        nhid = config.ninp * 4

        self.transformer_encoder = []
        for i in range(config.nlayers):
            if i != config.nlayers - 1:
                k = config.k
            else:
                k = 1
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp,
                    nhead=config.nhead,
                    dim_feedforward=nhid,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dropout=config.dropout,
                    k=k,
                )
            )
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp, config.nclass)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=config.ninp, pairwise_dim=config.pairwise_dimension
        )
        self.pos_encoder = relpos(config.pairwise_dimension)

    def forward(self, src, src_mask=None, return_aw=False):
        B, L = src.shape
        src = self.encoder(src).reshape(B, L, -1)

        pairwise_features = self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pos_encoder(src)

        attention_weights = []
        for i, layer in enumerate(self.transformer_encoder):
            if src_mask is not None:
                if return_aw:
                    src, aw = layer(src, pairwise_features, src_mask, return_aw=return_aw)
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(
                        src, pairwise_features, src_mask, return_aw=return_aw
                    )
            else:
                if return_aw:
                    src, aw = layer(src, pairwise_features, return_aw=return_aw)
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(src, pairwise_features, return_aw=return_aw)
        output = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0

        if return_aw:
            return output, attention_weights
        else:
            return output


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
class Config:
    """Plain kwargs config object, equivalent to Functions.py's `Config(**entries)`
    used by the original `load_config_from_yaml`."""

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries


def build_ribonanzanet():
    config = Config(
        ntoken=5,
        nclass=2,
        ninp=64,
        nhead=4,
        nlayers=2,
        k=3,
        pairwise_dimension=32,
        use_triangular_attention=True,
        dropout=0.0,
    )
    return RibonanzaNet(config)


def example_input_ribonanzanet():
    # (src, src_mask): 4 nucleotide tokens (A/C/G/U) + padding_idx=4, batch of 2,
    # sequence length 24, matching the real repo's `x=torch.ones(4,128).long()` demo.
    src = torch.randint(0, 4, (2, 24)).long()
    src_mask = torch.ones(2, 24).long()
    return (src, src_mask)


def _forward_ribonanzanet(model, inputs):
    src, src_mask = inputs
    return model(src, src_mask=src_mask)


MENAGERIE_ENTRIES = [
    ("RibonanzaNet", "build_ribonanzanet", "example_input_ribonanzanet", 2024, "vendored-pytorch"),
]
