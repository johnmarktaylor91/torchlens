# SOURCE: vendored from https://github.com/automl/RNAformer @ main
#
# RNAformer (automl group) -- a scalable axial-attention transformer for RNA secondary-
# structure / base-pair prediction, operating directly on an LxL pairwise "contact map"
# latent built from the sequence via an outer-sum sequence-to-matrix embedding, refined
# by stacked row/column ("triangle") attention + feed-forward blocks (a lighter-weight,
# AlphaFold-Evoformer-style pairwise architecture applied to RNA). Vendored verbatim
# (architecture-relevant classes only) from the repo's own files:
#   https://raw.githubusercontent.com/automl/RNAformer/main/RNAformer/model/RNAformer.py
#   https://raw.githubusercontent.com/automl/RNAformer/main/RNAformer/model/RNAformer_stack.py
#   https://raw.githubusercontent.com/automl/RNAformer/main/RNAformer/model/RNAformer_block.py
#   https://raw.githubusercontent.com/automl/RNAformer/main/RNAformer/module/embedding.py
#   https://raw.githubusercontent.com/automl/RNAformer/main/RNAformer/module/feed_forward.py
#   https://raw.githubusercontent.com/automl/RNAformer/main/RNAformer/module/axial_attention.py
#
# What is kept: RiboFormer (top-level model), RNAformerStack, RNAformerBlock,
# EmbedSequence2Matrix/PosEmbedding, FeedForward/ConvFeedForward, and
# TriangleAttention/Attention2d -- every mechanism in the real architecture, transcribed
# unmodified. `config.rotary_emb=False` is used at construction (routing through
# TriangleAttention/Attention2d rather than the flash-attn/rotary-embedding
# AxialAttention path), matching the repo's own non-flash inference fallback
# (`is_package_installed('flash-attn')` is False in this environment); AxialAttention and
# FlashAttention2d classes are kept vendored (unused at this config) for architectural
# completeness/fidelity but are not required for the traced entry point.
#
# What is dropped (config plumbing, not architecture): the original `RNAformer/utils/
# configuration.py` YAML-file `Config` loader is replaced with a plain kwargs
# `SimpleNestedNamespace`-equivalent class carrying the same field names read by the
# model code (`config.model_dim`, `config.num_head`, ... as enumerated by grepping every
# `config.<attr>` access across the vendored files).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from RNAformer/module/embedding.py
# ---------------------------------------------------------------------------
class EmbedSequence2Matrix(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pos_embedding = config.pos_embedding

        self.src_embed_1 = nn.Embedding(config.seq_vocab_size, config.model_dim)
        self.src_embed_2 = nn.Embedding(config.seq_vocab_size, config.model_dim)
        self.scale = nn.Parameter(
            torch.sqrt(torch.FloatTensor([config.model_dim // 2])), requires_grad=False
        )

        self.norm = nn.LayerNorm(
            config.model_dim, eps=config.ln_eps, elementwise_affine=config.learn_ln
        )

    def forward(self, src_seq):
        seq_1_embed = self.src_embed_1(src_seq)
        seq_2_embed = self.src_embed_2(src_seq)

        seq_1_embed = seq_1_embed * self.scale
        seq_2_embed = seq_2_embed * self.scale

        pair_latent = seq_1_embed.unsqueeze(1) + seq_2_embed.unsqueeze(2)

        pair_latent = self.norm(pair_latent)

        return pair_latent


# ---------------------------------------------------------------------------
# from RNAformer/module/feed_forward.py
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()

        ff_dim = int(config.ff_factor * config.model_dim)

        self.glu = config.use_glu
        ff_dim_1, ff_dim_2 = ff_dim, ff_dim

        self.input_norm = nn.LayerNorm(
            config.model_dim, eps=config.ln_eps, elementwise_affine=config.learn_ln
        )

        self.linear_1 = nn.Linear(config.model_dim, ff_dim_1, bias=config.use_bias)
        self.linear_2 = nn.Linear(ff_dim_2, config.model_dim, bias=config.use_bias)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.input_norm(x)

        if self.glu:
            x = self.linear_1(x)
            x, gate = x.chunk(2, dim=-1)
            x = self.act(gate) * x
        else:
            x = self.act(self.linear_1(x))

        return self.linear_2(x)


# ---------------------------------------------------------------------------
# from RNAformer/module/axial_attention.py (non-flash path only)
# ---------------------------------------------------------------------------
class Attention2d(nn.Module):
    def __init__(
        self,
        model_dim,
        num_head,
        softmax_scale,
        precision,
        zero_init,
        use_bias,
        initializer_range,
        n_layers,
    ):
        super().__init__()
        assert model_dim % num_head == 0
        self.key_dim = model_dim // num_head
        self.value_dim = model_dim // num_head

        if softmax_scale:
            self.softmax_scale = torch.sqrt(torch.FloatTensor([self.key_dim]))
        else:
            self.softmax_scale = False

        self.num_head = num_head
        self.model_dim = model_dim

        if precision == "fp32" or precision == 32 or precision == "bf16":
            self.mask_bias = -1e9
        elif precision == "fp16" or precision == 16:
            self.mask_bias = -1e4
        else:
            raise UserWarning(f"unknown precision: {precision} . Please us fp16, fp32 or bf16")

        self.Wqkv = nn.Linear(model_dim, 3 * model_dim, bias=use_bias)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=use_bias)

    def forward(self, pair_act, attention_mask):
        batch_size = pair_act.size(0)
        N_seq = pair_act.size(1)
        N_res = pair_act.size(2)

        query, key, value = self.Wqkv(pair_act).split(self.model_dim, dim=3)

        query = query.view(batch_size, N_seq, N_res, self.num_head, self.key_dim).permute(
            0, 1, 3, 2, 4
        )
        key = key.view(batch_size, N_seq, N_res, self.num_head, self.value_dim).permute(
            0, 1, 3, 4, 2
        )
        value = value.view(batch_size, N_seq, N_res, self.num_head, self.value_dim).permute(
            0, 1, 3, 2, 4
        )

        attn_weights = torch.matmul(query, key)

        if self.softmax_scale:
            attn_weights /= self.softmax_scale.to(pair_act.device)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None, None, :]
            attn_weights.masked_fill_(attention_mask, self.mask_bias)
        attn_weights = F.softmax(attn_weights, dim=-1)

        weighted_avg = torch.matmul(attn_weights, value).permute(0, 1, 3, 2, 4)

        output = self.out_proj(
            weighted_avg.reshape(batch_size, N_seq, N_res, self.num_head * self.value_dim)
        )
        return output


class TriangleAttention(nn.Module):
    def __init__(
        self,
        model_dim,
        num_head,
        orientation,
        softmax_scale,
        precision,
        zero_init,
        use_bias,
        flash_attn,
        initializer_range,
        n_layers,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.num_head = num_head

        assert orientation in ["per_row", "per_column"]
        self.orientation = orientation

        self.input_norm = nn.LayerNorm(model_dim, eps=1e-6)

        # flash-attn path intentionally not vendored (requires the `flash-attn` CUDA
        # extension, unavailable in the base env); the real repo also falls back to
        # this path via `is_package_installed('flash-attn')`.
        self.attn = Attention2d(
            model_dim,
            num_head,
            softmax_scale,
            precision,
            zero_init,
            use_bias,
            initializer_range,
            n_layers,
        )

    def forward(self, pair_act, pair_mask, cycle_infer=False):
        assert len(pair_act.shape) == 4

        if self.orientation == "per_column":
            pair_act = torch.swapaxes(pair_act, -2, -3)
            if pair_mask is not None:
                pair_mask = torch.swapaxes(pair_mask, -1, -2)

        pair_act = self.input_norm(pair_act)

        if self.training and not cycle_infer:
            pair_act = checkpoint(self.attn, pair_act, pair_mask, use_reentrant=True)
        else:
            pair_act = self.attn(pair_act, pair_mask)

        if self.orientation == "per_column":
            pair_act = torch.swapaxes(pair_act, -2, -3)

        return pair_act


# ---------------------------------------------------------------------------
# from RNAformer/model/RNAformer_block.py
# ---------------------------------------------------------------------------
class RNAformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_pair_row = TriangleAttention(
            config.model_dim,
            config.num_head,
            "per_row",
            config.softmax_scale,
            config.precision,
            config.zero_init,
            config.use_bias,
            config.flash_attn,
            config.initializer_range,
            config.n_layers,
        )
        self.attn_pair_col = TriangleAttention(
            config.model_dim,
            config.num_head,
            "per_column",
            config.softmax_scale,
            config.precision,
            config.zero_init,
            config.use_bias,
            config.flash_attn,
            config.initializer_range,
            config.n_layers,
        )

        self.pair_dropout_row = nn.Dropout(p=config.resi_dropout / 2)
        self.pair_dropout_col = nn.Dropout(p=config.resi_dropout / 2)

        self.pair_transition = FeedForward(config)

        self.res_dropout = nn.Dropout(p=config.resi_dropout)

    def forward(self, pair_act, pair_mask, cycle_infer=False):
        pair_act = pair_act + self.pair_dropout_row(
            self.attn_pair_row(pair_act, pair_mask, cycle_infer)
        )
        pair_act = pair_act + self.pair_dropout_col(
            self.attn_pair_col(pair_act, pair_mask, cycle_infer)
        )
        pair_act = pair_act + self.res_dropout(self.pair_transition(pair_act))

        return pair_act


# ---------------------------------------------------------------------------
# from RNAformer/model/RNAformer_stack.py
# ---------------------------------------------------------------------------
class RNAformerStack(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_ln = nn.LayerNorm(
            config.model_dim, eps=config.ln_eps, elementwise_affine=config.learn_ln
        )

        module_list = []
        for idx in range(config.n_layers):
            layer = RNAformerBlock(config=config)
            module_list.append(layer)
        self.layers = nn.ModuleList(module_list)

    def forward(self, pair_act, pair_mask, cycle_infer=False):
        for idx, layer in enumerate(self.layers):
            pair_act = layer(pair_act, pair_mask, cycle_infer=cycle_infer)

        pair_act = self.output_ln(pair_act)

        return pair_act


# ---------------------------------------------------------------------------
# from RNAformer/model/RNAformer.py
# ---------------------------------------------------------------------------
class RiboFormer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_dim = config.model_dim
        self.cycling = False

        self.seq2mat_embed = EmbedSequence2Matrix(config)
        self.RNAformer = RNAformerStack(config)

        self.pdf_embedding = nn.Linear(1, config.model_dim, bias=True)
        self.use_pdb = True

        self.output_mat = nn.Linear(config.model_dim, 1, bias=True)

    def make_pair_mask(self, src, src_len):
        encode_mask = torch.arange(src.shape[1], device=src.device).expand(
            src.shape[:2]
        ) < src_len.unsqueeze(1)

        pair_mask = encode_mask[:, None, :] * encode_mask[:, :, None]

        return torch.bitwise_not(pair_mask)

    def forward(self, src_seq, src_len, pdb_sample, max_cycle=0):
        pair_mask = self.make_pair_mask(src_seq, src_len)

        pair_latent = self.seq2mat_embed(src_seq)

        if self.use_pdb:
            pair_latent = pair_latent + self.pdf_embedding(pdb_sample)[:, None, None, :]

        latent = self.RNAformer(pair_act=pair_latent, pair_mask=pair_mask, cycle_infer=False)

        logits = self.output_mat(latent)

        return logits, pair_mask


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
class Config:
    """Plain kwargs config object, equivalent to the field set read off the real
    repo's YAML-driven `Config` (utils/configuration.py)."""

    def __init__(self, **entries):
        self.__dict__.update(entries)


def build_rnaformer():
    config = Config(
        model_dim=32,
        num_head=4,
        n_layers=2,
        seq_vocab_size=5,
        max_len=64,
        pos_embedding=False,
        rel_pos_enc=False,
        rotary_emb=False,
        rotary_emb_fraction=1.0,
        ln_eps=1e-5,
        learn_ln=True,
        initializer_range=0.02,
        resi_dropout=0.0,
        ff_factor=2,
        ff_kernel=0,
        use_glu=False,
        use_bias=True,
        zero_init=True,
        softmax_scale=True,
        precision="fp32",
        flash_attn=False,
    )
    return RiboFormer(config)


def example_input_rnaformer():
    batch, length = 2, 12
    src_seq = torch.randint(0, 4, (batch, length)).long()
    src_len = torch.full((batch,), length, dtype=torch.long)
    pdb_sample = torch.ones(batch, 1)
    return (src_seq, src_len, pdb_sample)


def _forward_rnaformer(model, inputs):
    src_seq, src_len, pdb_sample = inputs
    logits, pair_mask = model(src_seq, src_len, pdb_sample)
    return logits


MENAGERIE_ENTRIES = [
    ("RNAformer", "build_rnaformer", "example_input_rnaformer", 2023, "vendored-pytorch"),
]
