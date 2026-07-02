# SOURCE: vendored from https://github.com/lucidrains/protein-bert-pytorch @ main
# File: protein_bert_pytorch/protein_bert_pytorch.py
#
# ProteinBERT (Brandes, Ofer, Peleg, Rappoport, Linial 2022) -- a BERT-style protein
# language model jointly modeling local per-residue sequence representations and a global
# GO/EC-annotation representation, fused every block via cross-attention. The official repo
# (nadavbra/protein_bert) is TensorFlow/Keras and cannot run in this base torch environment.
# lucidrains' `protein-bert-pytorch` is a faithful, widely-used PyTorch port of the same
# architecture (GlobalLinearSelfAttention, dual narrow/wide dilated Conv1d branches per
# local block, bidirectional local<->global CrossAttention, global GELU dense/attend stack)
# and is itself real, published torch code with only base deps (torch, einops). Vendored
# verbatim -- only the core `ProteinBERT` model (+ its `Layer`/`GlobalLinearSelfAttention`/
# `CrossAttention`/`Residual` building blocks) is kept; the `PretrainingWrapper` masking/loss
# wrapper (a training utility, not architecture) is dropped.
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn

MENAGERIE_ZOO = "vendored-pytorch"


# helpers


def exists(val):
    return val is not None


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class GlobalLinearSelfAttention(nn.Module):
    def __init__(self, *, dim, dim_head, heads):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, feats, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(feats).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, "b n -> b () n ()")
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        if exists(mask):
            v = v.masked_fill(~mask, 0.0)

        context = einsum("b h n d, b h n e -> b h d e", k, v)
        out = einsum("b h d e, b h n d -> b h n e", context, q)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_keys,
        dim_out,
        heads,
        dim_head=64,
        qk_activation=nn.Tanh(),
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.qk_activation = qk_activation

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_keys, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim_out)

        self.null_key = nn.Parameter(torch.randn(dim_head))
        self.null_value = nn.Parameter(torch.randn(dim_head))

    def forward(self, x, context, mask=None, context_mask=None):
        b, h, device = x.shape[0], self.heads, x.device

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        null_k, null_v = map(
            lambda t: repeat(t, "d -> b h () d", b=b, h=h), (self.null_key, self.null_value)
        )
        k = torch.cat((null_k, k), dim=-2)
        v = torch.cat((null_v, v), dim=-2)

        q, k = map(lambda t: self.qk_activation(t), (q, k))

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(mask) or exists(context_mask):
            i, j = sim.shape[-2:]

            if not exists(mask):
                mask = torch.ones(b, i, dtype=torch.bool, device=device)

            if exists(context_mask):
                context_mask = F.pad(context_mask, (1, 0), value=True)
            else:
                context_mask = torch.ones(b, j, dtype=torch.bool, device=device)

            mask = rearrange(mask, "b i -> b () i ()") * rearrange(context_mask, "b j -> b () () j")
            sim.masked_fill_(~mask, max_neg_value(sim))

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Layer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_global,
        narrow_conv_kernel=9,
        wide_conv_kernel=9,
        wide_conv_dilation=5,
        attn_heads=8,
        attn_dim_head=64,
        attn_qk_activation=nn.Tanh(),
        local_to_global_attn=False,
        local_self_attn=False,
        glu_conv=False,
    ):
        super().__init__()

        self.seq_self_attn = (
            GlobalLinearSelfAttention(dim=dim, dim_head=attn_dim_head, heads=attn_heads)
            if local_self_attn
            else None
        )

        conv_mult = 2 if glu_conv else 1

        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding=narrow_conv_kernel // 2),
            nn.GELU() if not glu_conv else nn.GLU(dim=1),
        )

        wide_conv_padding = (
            wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)
        ) // 2

        self.wide_conv = nn.Sequential(
            nn.Conv1d(
                dim,
                dim * conv_mult,
                wide_conv_kernel,
                dilation=wide_conv_dilation,
                padding=wide_conv_padding,
            ),
            nn.GELU() if not glu_conv else nn.GLU(dim=1),
        )

        self.local_to_global_attn = local_to_global_attn

        if local_to_global_attn:
            self.extract_global_info = CrossAttention(
                dim=dim,
                dim_keys=dim_global,
                dim_out=dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
            )
        else:
            self.extract_global_info = nn.Sequential(
                Reduce("b n d -> b d", "mean"),
                nn.Linear(dim_global, dim),
                nn.GELU(),
                Rearrange("b d -> b () d"),
            )

        self.local_norm = nn.LayerNorm(dim)

        self.local_feedforward = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim),
        )

        self.global_attend_local = CrossAttention(
            dim=dim_global,
            dim_out=dim_global,
            dim_keys=dim,
            heads=attn_heads,
            dim_head=attn_dim_head,
            qk_activation=attn_qk_activation,
        )

        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU(),
        )

        self.global_norm = nn.LayerNorm(dim_global)

        self.global_feedforward = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Linear(dim_global, dim_global),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global),
        )

    def forward(self, tokens, annotation, mask=None):
        if self.local_to_global_attn:
            global_info = self.extract_global_info(tokens, annotation, mask=mask)
        else:
            global_info = self.extract_global_info(annotation)

        # process local (protein sequence)

        global_linear_attn = self.seq_self_attn(tokens) if exists(self.seq_self_attn) else 0

        conv_input = rearrange(tokens, "b n d -> b d n")

        if exists(mask):
            conv_input_mask = rearrange(mask, "b n -> b () n")
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.0)

        narrow_out = self.narrow_conv(conv_input)
        narrow_out = rearrange(narrow_out, "b d n -> b n d")
        wide_out = self.wide_conv(conv_input)
        wide_out = rearrange(wide_out, "b d n -> b n d")

        tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn
        tokens = self.local_norm(tokens)

        tokens = self.local_feedforward(tokens)

        # process global (annotations)

        annotation = self.global_attend_local(annotation, tokens, context_mask=mask)
        annotation = self.global_dense(annotation)
        annotation = self.global_norm(annotation)
        annotation = self.global_feedforward(annotation)

        return tokens, annotation


# main model


class ProteinBERT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens=26,
        num_annotation=8943,
        dim=512,
        dim_global=256,
        depth=6,
        narrow_conv_kernel=9,
        wide_conv_kernel=9,
        wide_conv_dilation=5,
        attn_heads=8,
        attn_dim_head=64,
        attn_qk_activation=nn.Tanh(),
        local_to_global_attn=False,
        local_self_attn=False,
        num_global_tokens=1,
        glu_conv=False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.num_global_tokens = num_global_tokens
        self.to_global_emb = nn.Linear(num_annotation, num_global_tokens * dim_global)

        self.layers = nn.ModuleList(
            [
                Layer(
                    dim=dim,
                    dim_global=dim_global,
                    narrow_conv_kernel=narrow_conv_kernel,
                    wide_conv_dilation=wide_conv_dilation,
                    wide_conv_kernel=wide_conv_kernel,
                    attn_qk_activation=attn_qk_activation,
                    local_to_global_attn=local_to_global_attn,
                    local_self_attn=local_self_attn,
                    glu_conv=glu_conv,
                )
                for layer in range(depth)
            ]
        )

        self.to_token_logits = nn.Linear(dim, num_tokens)

        self.to_annotation_logits = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.Linear(dim_global, num_annotation),
        )

    def forward(self, seq, annotation, mask=None):
        tokens = self.token_emb(seq)

        annotation = self.to_global_emb(annotation)
        annotation = rearrange(annotation, "b (n d) -> b n d", n=self.num_global_tokens)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation, mask=mask)

        tokens = self.to_token_logits(tokens)
        annotation = self.to_annotation_logits(annotation)
        return tokens, annotation


def build_proteinbert():
    torch.manual_seed(0)
    # real defaults: dim=512, dim_global=256, depth=6, num_annotation=8943 (GO+Pfam);
    # shrunk to a tiny config for a fast CPU trace, same architecture (per-block dual
    # dilated Conv1d branches + bidirectional local<->global CrossAttention fusion).
    return ProteinBERT(
        num_tokens=26,
        num_annotation=32,
        dim=16,
        dim_global=8,
        depth=2,
        attn_heads=2,
        attn_dim_head=8,
    )


def example_input_proteinbert():
    torch.manual_seed(0)
    batch, seq_len, num_annotation = 1, 16, 32
    seq = torch.randint(0, 26, (batch, seq_len))
    annotation = torch.randint(0, 2, (batch, num_annotation)).float()
    return (seq, annotation)


MENAGERIE_ENTRIES = [
    ("ProteinBERT", "build_proteinbert", "example_input_proteinbert", 2022, "SOURCE_AVAILABLE"),
]
