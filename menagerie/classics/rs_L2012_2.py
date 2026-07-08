# FAITHFUL PORT of https://github.com/microsoft/archai @ neurips-lts branch (original
# framework: PyTorch, inside the `archai` package which is not installed / not reasonably
# installable as a dependency alongside base torch here)
#
# "Lite Transformer Search" (LTS) -- Javaheripi et al., "LTS: A DNN Efficient Architecture
# Design Method", NeurIPS 2021 (training-free NAS using decoder parameter count as a
# perplexity proxy, run directly on the target device CPU) -- searches over a fixed backbone
# architecture: NVIDIA's Memory Transformer (`MemTransformerLM`, the Transformer-XL decoder
# used as archai's NLP search space). This module transcribes the actual repo code 1:1 (not
# a paper paraphrase), at the search space's own default `attn_type=0` (relative partial
# learnable attention -- the architecture published as the base case in all of archai's NLP
# NAS configs):
#   archai/nlp/models/mem_transformer/model_mem_transformer.py
#     (`MemTransformerLM`, `RelPartialLearnableDecoderLayer`,
#      `RelPartialLearnableMultiHeadAttn`, `PositionwiseFF`, `PositionalEmbedding`)
#   archai/nlp/models/model_utils/adaptive_embedding.py   (`AdaptiveEmbedding`)
#   archai/nlp/models/model_utils/proj_adaptive_softmax.py (`ProjectedAdaptiveLogSoftmax`,
#     non-adaptive / `n_clusters == 0` path, i.e. `cutoffs=[n_token]`)
#   archai/nlp/models/model_base.py                        (`ArchaiModel` -- a thin
#     `torch.nn.Module` subclass adding only introspection helpers, no forward-path change)
#
# `archai` itself (its NAS search-space enumeration, mixed-precision/ONNX export utilities,
# `LogUniformSampler`-based sampled softmax used only for `sample_softmax > 0` -- off by
# default, `primer_conv`/`primer_square` Primer-EZ ablation switches -- off by default, and
# `map_to_list`/`ArchaiModel` bookkeeping helpers) is not installed and far too large a
# framework to vendor as a dependency for one model, so the architecture is transcribed here
# as self-contained torch: the same segment-level-recurrence memory transformer core
# (`AdaptiveEmbedding` -> stack of `RelPartialLearnableDecoderLayer` -- each a relative
# partial-learnable multi-head self-attention with sinusoidal relative positional encoding,
# per-layer learnable `r_w_bias`/`r_r_bias` biases, and a position-wise feed-forward block --
# -> `ProjectedAdaptiveLogSoftmax` output head), the same non-adaptive (`div_val=1`,
# `n_clusters=0`, single-cluster) softmax path taken when `n_token` is small (the search
# space's own default), and the same causal `dec_attn_mask` construction with an explicit
# `mems` memory bank. `attn_type in {1,2,3}` (learnable / absolute embeddings, alternate
# search-space variants), `sample_softmax`, `primer_conv`/`primer_square` Primer-EZ variants,
# and `past_key_values` incremental-decoding caching are NAS-searchable ablations / serving
# concerns off the base architecture's default forward path and are not ported.
#
# Repo: https://github.com/microsoft/archai @ neurips-lts

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class AdaptiveEmbedding(nn.Module):
    """Faithful port of archai's `AdaptiveEmbedding` at the non-adaptive (`div_val=1`) path
    (single embedding table, optional output projection to `d_proj`).
    """

    def __init__(self, n_token, d_embed, d_proj):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj**0.5

        self.emb_layers = nn.ModuleList([nn.Embedding(n_token, d_embed)])
        self.emb_projs = nn.ParameterList()
        if d_proj != d_embed:
            self.emb_projs.append(nn.Parameter(torch.zeros(d_proj, d_embed)))

    def forward(self, inp):
        embed = self.emb_layers[0](inp)
        if self.d_proj != self.d_embed:
            embed = F.linear(embed, self.emb_projs[0])
        embed = embed * self.emb_scale
        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):
    """Faithful port of archai's `ProjectedAdaptiveLogSoftmax` at the non-adaptive
    (`n_clusters == 0`, single output layer) path -- the branch taken by the search space's
    default small-vocabulary configs.
    """

    def __init__(self, n_token, d_embed, d_proj, out_layer_weight=None):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.out_proj = None
        if d_proj != d_embed:
            self.out_proj = nn.Parameter(torch.zeros(d_proj, d_embed))

        self.out_layer_bias = nn.Parameter(torch.zeros(n_token))
        if out_layer_weight is not None:
            self.out_layer_weight = out_layer_weight
        else:
            self.out_layer_weight = nn.Parameter(torch.zeros(n_token, d_embed))

    def _compute_logit(self, hidden):
        if self.out_proj is None:
            return F.linear(hidden, self.out_layer_weight, bias=self.out_layer_bias)
        logit = torch.einsum("bd,de,ev->bv", (hidden, self.out_proj, self.out_layer_weight.t()))
        return logit + self.out_layer_bias

    def forward(self, hidden):
        """hidden: [len*bsz x d_proj] -> log_probs: [len*bsz x n_token]."""
        logit = self._compute_logit(hidden)
        return F.log_softmax(logit, dim=-1)


class PositionalEmbedding(nn.Module):
    """Faithful port of `model_mem_transformer.PositionalEmbedding` (sinusoidal, relative)."""

    def __init__(self, demb):
        super().__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    """Faithful port of `model_mem_transformer.PositionwiseFF`."""

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super().__init__()
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    """Faithful port of `model_mem_transformer.RelPartialLearnableMultiHeadAttn` (the
    Transformer-XL relative partial-learnable self-attention: content-based term AC +
    position-based term BD via `_rel_shift`, no `mems`/`past_key_values`/`primer_ez` for a
    plain single-pass forward).
    """

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.0, pre_lnorm=False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head**0.5)

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_heads = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)
        r_head_k = self.r_net(r)

        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        rw_head_q = w_head_q + r_w_bias
        ac = torch.einsum("ibnd,jbnd->bnij", (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias
        bd = torch.einsum("ibnd,jnd->bnij", (rr_head_q, r_head_k))
        bd = self._rel_shift(bd)

        attn_score = add_and_scale(ac, bd, self.scale)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score = attn_score.masked_fill(attn_mask[None, None, :, :], float("-inf"))
            elif attn_mask.dim() == 3:
                attn_score = attn_score.masked_fill(attn_mask[:, None, :, :], float("-inf"))

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum("bnij,jbnd->ibnd", (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)
        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    """Faithful port of `model_mem_transformer.RelPartialLearnableDecoderLayer`."""

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0.0, pre_lnorm=False):
        super().__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm
        )
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask)
        output = self.pos_ff(output)
        return output


def _init_weight(weight, std=0.02):
    nn.init.normal_(weight, 0.0, std)


def _weights_init(m, std=0.02, proj_std=0.01):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            _init_weight(m.weight, std)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("AdaptiveEmbedding") != -1:
        for p in m.emb_projs:
            if p is not None:
                nn.init.normal_(p, 0.0, proj_std)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            _init_weight(m.weight, std)
    elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
        if m.out_proj is not None:
            nn.init.normal_(m.out_proj, 0.0, proj_std)
        _init_weight(m.out_layer_weight, std)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, std)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MemTransformerLM(nn.Module):
    """Faithful port of archai's `MemTransformerLM` at `attn_type=0` (the search space's
    default relative partial-learnable attention), non-adaptive softmax (`div_val=1`,
    single-cluster output), tied input/output embeddings (`tie_weight=True`, the default).
    """

    def __init__(
        self,
        n_token,
        n_layer=4,
        n_head=4,
        d_model=32,
        d_head=8,
        d_inner=64,
        dropout=0.1,
        dropatt=0.0,
        tie_weight=True,
        d_embed=32,
        pre_lnorm=False,
        tgt_len=16,
        mem_len=0,
        same_length=False,
        clamp_len=-1,
    ):
        super().__init__()
        self.n_token = n_token
        self.d_model = d_model
        self.n_head = [n_head] * n_layer
        self.d_head = [d_head] * n_layer
        self.n_layer = n_layer
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = 0
        self.max_klen = tgt_len + self.ext_len + mem_len
        self.same_length = same_length
        self.clamp_len = clamp_len

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm
                )
                for _ in range(n_layer)
            ]
        )

        self.pos_emb = PositionalEmbedding(d_model)
        for i in range(n_layer):
            setattr(self, f"r_w_bias_{i}", nn.Parameter(torch.zeros(n_head, d_head)))
            setattr(self, f"r_r_bias_{i}", nn.Parameter(torch.zeros(n_head, d_head)))

        out_layer_weight = self.word_emb.emb_layers[0].weight if tie_weight else None
        self.crit = ProjectedAdaptiveLogSoftmax(
            n_token, d_embed, d_model, out_layer_weight=out_layer_weight
        )

        self.apply(functools.partial(_weights_init, std=0.02, proj_std=0.01))
        self.word_emb.apply(functools.partial(_weights_init, std=0.02, proj_std=0.01))

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len - 1
            mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
            dec_attn_mask = (
                torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)
            ).bool()
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1 + mlen).bool()

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq = pos_seq.clamp(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        for i, layer in enumerate(self.layers):
            core_out = layer(
                core_out,
                pos_emb,
                getattr(self, f"r_w_bias_{i}"),
                getattr(self, f"r_r_bias_{i}"),
                dec_attn_mask=dec_attn_mask,
            )

        core_out = self.drop(core_out)
        return core_out

    def forward(self, input_ids):
        """input_ids: [batch, seq_len] -> log_probs: [batch, seq_len, n_token]."""
        input_ids = input_ids.t()  # -> [seq_len, batch]
        hidden = self._forward(input_ids)
        tgt_len = input_ids.size(0)
        pred_hid = hidden[-tgt_len:]
        log_probs = self.crit(pred_hid.reshape(-1, pred_hid.size(-1)))
        log_probs = log_probs.view(tgt_len, input_ids.size(1), -1).transpose(0, 1)
        return log_probs


def build_mem_transformer_lts():
    # LTS's own NAS search-space defaults are much larger (n_layer up to 16, d_model up to
    # 1024, n_token in the hundreds of thousands for WikiText-103 / lm1b); shrunk here for a
    # fast trace with a small vocab that legitimately hits the non-adaptive softmax path the
    # search space itself falls back to below its default cutoffs -- architecture (relative
    # partial-learnable attention + position-wise FFN stack, tied adaptive embedding /
    # softmax) unchanged.
    return MemTransformerLM(
        n_token=200,
        n_layer=2,
        n_head=4,
        d_model=32,
        d_head=8,
        d_inner=64,
        dropout=0.0,
        tgt_len=8,
        mem_len=0,
    )


def example_input_mem_transformer_lts():
    batch, seq_len, n_token = 2, 8, 200
    return torch.randint(0, n_token, (batch, seq_len))


MENAGERIE_ENTRIES = [
    (
        "Lite Transformer Search (Memory Transformer / Transformer-XL NAS backbone)",
        build_mem_transformer_lts,
        example_input_mem_transformer_lts,
        2021,
        MENAGERIE_ZOO,
    ),
]
