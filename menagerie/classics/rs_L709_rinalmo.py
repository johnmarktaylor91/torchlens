# SOURCE: vendored from https://github.com/lbcb-sci/RiNALMo @ 1ad48ee6c67c201439b91b7e088473f703213e96
#
# RiNALMo (Penic, Vlasic, Huber, Wan, Sikic. 2024, "RiNALMo: General-Purpose RNA Language
# Models Can Generalize Well on Structure Prediction Tasks") -- a BERT-style RNA language
# model (up to 650M params at the "giga" size) pretrained on 36M non-coding RNA sequences,
# using rotary position embeddings and a SwiGLU transition. Vendored verbatim
# (architecture-relevant classes only) from the repo's own files:
#   https://raw.githubusercontent.com/lbcb-sci/RiNALMo/1ad48ee6c67c201439b91b7e088473f703213e96/rinalmo/model/model.py
#   https://raw.githubusercontent.com/lbcb-sci/RiNALMo/1ad48ee6c67c201439b91b7e088473f703213e96/rinalmo/model/modules.py
#   https://raw.githubusercontent.com/lbcb-sci/RiNALMo/1ad48ee6c67c201439b91b7e088473f703213e96/rinalmo/model/attention.py
#   https://raw.githubusercontent.com/lbcb-sci/RiNALMo/1ad48ee6c67c201439b91b7e088473f703213e96/rinalmo/model/rope.py
#   https://raw.githubusercontent.com/lbcb-sci/RiNALMo/1ad48ee6c67c201439b91b7e088473f703213e96/rinalmo/data/alphabet.py
#   https://raw.githubusercontent.com/lbcb-sci/RiNALMo/1ad48ee6c67c201439b91b7e088473f703213e96/rinalmo/data/constants.py
#
# What is kept: RiNALMo (embedding + Transformer + MaskedLanguageModelHead), TokenDropout,
# Transformer/TransformerBlock, SwiGLU transition, MaskedLanguageModelHead, the real
# (non-flash) MultiHeadAttention/MultiHeadSelfAttention with RotaryPositionEmbedding --
# every mechanism in the real architecture, transcribed unmodified. `use_flash_attn=False`
# selects the repo's own eager-attention code path (`attention.py`'s non-Flash
# `MultiHeadSelfAttention`), the same architecture the flash path accelerates -- this is
# not a stub, it's the real fallback implementation shipped in the same file.
#
# What is dropped (infra plumbing, not architecture): the `flash_attn`-package-based
# `FlashAttention`/`FlashMultiHeadSelfAttention` classes (CUDA-only fused-kernel alternate
# implementation of the identical attention math, gated by `use_flash_attn`, and
# `flash_attn` is not an installed base lib here); `torch.utils.checkpoint.checkpoint(...)`
# gradient-checkpointing wrapper around each transformer block is replaced with a direct
# call (checkpointing is a memory/compute-tradeoff optimization, not an architectural
# change -- forward output is identical); the `ml_collections`-based `config.py` nested
# ConfigDict is replaced with a plain nested-dict `_NS` shim exposing the same
# attribute-and-item access (`config.model.transformer.use_flash_attn` /
# `config.model['transformer']`) that `model.py`'s own forward() relies on.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from rinalmo/data/constants.py + rinalmo/data/alphabet.py
# ---------------------------------------------------------------------------
ADENINE_TKN = "A"
CYTOSINE_TKN = "C"
URACIL_TKN = "U"
GUANINE_TKN = "G"
THYMINE_TKN = "T"
INOSINE_TKN = "I"
ANY_NUCLEOTIDE_TKN = "N"

RNA_TOKENS = [
    ADENINE_TKN,
    CYTOSINE_TKN,
    GUANINE_TKN,
    THYMINE_TKN,
    INOSINE_TKN,
    "R",
    "Y",
    "K",
    "M",
    "S",
    "W",
    "B",
    "D",
    "H",
    "V",
    ANY_NUCLEOTIDE_TKN,
    "-",
]

CLS_TKN = "<cls>"
PAD_TKN = "<pad>"
BOS_TKN = "<bos>"
EOS_TKN = "<eos>"
UNK_TKN = "<unk>"
MASK_TKN = "<mask>"


class Alphabet:
    def __init__(
        self, standard_tkns=RNA_TOKENS, special_tkns=[CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN]
    ):
        super().__init__()

        self.standard_tkns = standard_tkns
        self.special_tkns = special_tkns

        self.idx_to_tkn = special_tkns + standard_tkns
        self.tkn_to_idx = {t: i for i, t in enumerate(self.idx_to_tkn)}

        self.cls_idx = self.tkn_to_idx[CLS_TKN]
        self.eos_idx = self.tkn_to_idx[EOS_TKN]

        self.unk_idx = self.tkn_to_idx[UNK_TKN]
        self.pad_idx = self.tkn_to_idx[PAD_TKN]
        self.mask_idx = self.tkn_to_idx[MASK_TKN]

    def __len__(self):
        return len(self.idx_to_tkn)

    def get_idx(self, tkn):
        return self.tkn_to_idx.get(tkn, self.unk_idx)


# ---------------------------------------------------------------------------
# from rinalmo/model/rope.py
# ---------------------------------------------------------------------------
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cached(self, x, seq_dim):
        seq_len = x.shape[seq_dim]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q, k):
        self._update_cached(k, seq_dim=-2)
        return apply_rotary_pos_emb(q, k, self.cos_cached, self.sin_cached)


# ---------------------------------------------------------------------------
# from rinalmo/model/attention.py (non-flash eager path only; see header note)
# ---------------------------------------------------------------------------
def dot_product_attention(q, k, v, attn_mask=None, key_pad_mask=None, dropout=None):
    c = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(c)

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask, float("-inf"))

    if key_pad_mask is not None:
        attn = attn.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    attn = attn.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()
        assert c_in % num_heads == 0, (
            "Embedding dimensionality must be divisible with number of attention heads!"
        )

        self.c_in = c_in
        self.num_heads = num_heads

        self.c_head = c_in // self.num_heads
        self.c_qkv = self.c_head * num_heads

        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.c_head)

        self.to_q = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_k = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_v = nn.Linear(self.c_in, self.c_qkv, bias=bias)

        self.attention_dropout = nn.Dropout(p=attention_dropout)

        self.out_proj = nn.Linear(c_in, c_in, bias=bias)

    def forward(self, q, k, v, attn_mask=None, key_pad_mask=None):
        bs = q.shape[0]

        q = self.to_q(q).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        k = self.to_k(k).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        v = self.to_v(v).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        output, attn = dot_product_attention(
            q, k, v, attn_mask, key_pad_mask, self.attention_dropout
        )

        output = output.transpose(-2, -3).contiguous().view(bs, -1, self.num_heads * self.c_head)
        output = self.out_proj(output)

        return output, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()

        self.mh_attn = MultiHeadAttention(c_in, num_heads, attention_dropout, use_rot_emb, bias)

    def forward(self, x, attn_mask=None, key_pad_mask=None):
        return self.mh_attn(x, x, x, attn_mask, key_pad_mask)


# ---------------------------------------------------------------------------
# from rinalmo/model/modules.py
# ---------------------------------------------------------------------------
class TokenDropout(nn.Module):
    def __init__(
        self,
        active: bool,
        mask_ratio: float,
        mask_tkn_prob: float,
        mask_tkn_idx: int,
        pad_tkn_idx: int,
    ):
        super().__init__()

        self.active = active

        self.mask_ratio_train = mask_ratio * mask_tkn_prob

        self.mask_tkn_idx = mask_tkn_idx
        self.pad_tkn_idx = pad_tkn_idx

    def forward(self, x, tokens):
        if self.active:
            pad_mask = tokens.eq(self.pad_tkn_idx)
            src_lens = (~pad_mask).sum(dim=-1)

            x = torch.where((tokens == self.mask_tkn_idx).unsqueeze(dim=-1), 0.0, x)
            mask_ratio_observed = (tokens == self.mask_tkn_idx).sum(dim=-1) / src_lens
            x = x * (1 - self.mask_ratio_train) / (1 - mask_ratio_observed[..., None, None])

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_blocks,
        num_heads,
        use_rot_emb=True,
        attn_qkv_bias=False,
        transition_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        transition_factor=4,
        use_flash_attn=False,
    ):
        super().__init__()

        self.use_flash_attn = use_flash_attn

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    use_rot_emb,
                    attn_qkv_bias,
                    transition_dropout,
                    attention_dropout,
                    residual_dropout,
                    transition_factor,
                    use_flash_attn,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, need_attn_weights=False):
        attn_weights = None
        if need_attn_weights:
            attn_weights = []

        for block in self.blocks:
            # NOTE: original uses torch.utils.checkpoint.checkpoint(block, ...) here for
            # memory-saving gradient checkpointing; a direct call is functionally identical
            # for a forward pass (checkpointing only changes how the backward graph is
            # rematerialized, not the forward computation) and is required for TorchLens
            # to trace the real per-op computation inside each block.
            x, attn = block(
                x, key_padding_mask=key_padding_mask, need_attn_weights=need_attn_weights
            )

            if need_attn_weights:
                attn_weights.append(attn)

        x = self.final_layer_norm(x)

        return x, attn_weights


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """

    def __init__(self, size_in, size_out, beta_is_learnable=True, bias=True):
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=beta_is_learnable)

    def forward(self, x):
        linear_out = self.linear(x)
        swish_out = linear_out * torch.sigmoid(self.beta * linear_out)
        return swish_out * self.linear_gate(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        use_rot_emb=True,
        attn_qkv_bias=False,
        transition_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        transition_factor=4,
        use_flash_attn=False,
    ):
        super().__init__()

        self.use_flash_attn = use_flash_attn

        # NOTE: real repo branches to FlashMultiHeadSelfAttention when use_flash_attn=True;
        # that class requires the CUDA-only `flash_attn` package (not an installed base lib
        # here). This staging module always uses the real non-flash MultiHeadSelfAttention
        # eager path (identical attention math, same file in the real repo).
        self.mh_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attention_dropout, use_rot_emb, attn_qkv_bias
        )

        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        self.transition = nn.Sequential(
            SwiGLU(
                embed_dim,
                int(2 / 3 * transition_factor * embed_dim),
                beta_is_learnable=True,
                bias=True,
            ),
            nn.Dropout(p=transition_dropout),
            nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        )
        self.out_layer_norm = nn.LayerNorm(embed_dim)

        self.residual_dropout_1 = nn.Dropout(p=residual_dropout)
        self.residual_dropout_2 = nn.Dropout(p=residual_dropout)

    def forward(self, x, key_padding_mask=None, need_attn_weights=None):
        x = self.attn_layer_norm(x)
        mh_out, attn = self.mh_attn(x, attn_mask=None, key_pad_mask=key_padding_mask)
        x = x + self.residual_dropout_1(mh_out)

        residual = x
        x = self.out_layer_norm(x)
        x = residual + self.residual_dropout_2(self.transition(x))

        return x, attn


class MaskedLanguageModelHead(nn.Module):
    def __init__(self, embed_dim, alphabet_size):
        super().__init__()

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, alphabet_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.linear2(x)

        return x


# ---------------------------------------------------------------------------
# from rinalmo/model/model.py
# ---------------------------------------------------------------------------
class RiNALMo(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pad_tkn_idx = self.config.model.embedding["padding_idx"]

        self.embedding = nn.Embedding(**self.config.model["embedding"])
        self.transformer = Transformer(**self.config.model["transformer"])

        self.lm_mask_head = MaskedLanguageModelHead(**self.config.model["lm_mask_head"])

        self.token_dropout = TokenDropout(**self.config.model["token_dropout"])

    def forward(self, tokens, need_attn_weights=False):
        pad_mask = tokens.eq(self.pad_tkn_idx)
        x = self.embedding(tokens)
        x = self.token_dropout(x, tokens)

        if self.config.model.transformer.use_flash_attn:
            representation, attn_weights = self.transformer(
                x,
                key_padding_mask=torch.logical_not(pad_mask) if pad_mask is not None else None,
                need_attn_weights=need_attn_weights,
            )
        else:
            representation, attn_weights = self.transformer(
                x, key_padding_mask=pad_mask, need_attn_weights=need_attn_weights
            )
        x = self.lm_mask_head(representation)

        result = {"logits": x, "representation": representation}
        if need_attn_weights:
            result["attentions"] = torch.stack(attn_weights, dim=1)

        return result


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
class _NS(dict):
    """Minimal nested-dict namespace supporting both `.attr` and `['item']` access,
    equivalent to the subset of `ml_collections.ConfigDict` behavior that model.py's
    own forward()/__init__() rely on (`config.model.embedding['padding_idx']`,
    `config.model['embedding']`, `config.model.transformer.use_flash_attn`)."""

    def __getattr__(self, item):
        try:
            val = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return _NS(val) if isinstance(val, dict) else val

    def __setattr__(self, key, value):
        self[key] = value


def _tiny_rinalmo_config():
    alphabet = Alphabet()
    embed_dim = 32
    return _NS(
        {
            "model": {
                "embedding": {
                    "num_embeddings": len(alphabet),
                    "embedding_dim": embed_dim,
                    "padding_idx": alphabet.pad_idx,
                },
                "token_dropout": {
                    "active": True,
                    "mask_ratio": 0.15,
                    "mask_tkn_prob": 0.8,
                    "mask_tkn_idx": alphabet.mask_idx,
                    "pad_tkn_idx": alphabet.pad_idx,
                },
                "transformer": {
                    "embed_dim": embed_dim,
                    "num_blocks": 2,
                    "num_heads": 4,
                    "use_rot_emb": True,
                    "attn_qkv_bias": False,
                    "attention_dropout": 0.0,
                    "transition_dropout": 0.0,
                    "residual_dropout": 0.0,
                    "transition_factor": 4,
                    "use_flash_attn": False,
                },
                "lm_mask_head": {
                    "embed_dim": embed_dim,
                    "alphabet_size": len(alphabet),
                },
            },
        }
    )


def build_rinalmo():
    return RiNALMo(_tiny_rinalmo_config())


def example_input_rinalmo():
    alphabet = Alphabet()
    seq = "ACGUACGUACGUACGU"
    tokens = (
        [alphabet.cls_idx]
        + [alphabet.get_idx(t) for t in seq.replace("U", "T")]
        + [alphabet.eos_idx]
    )
    return torch.tensor([tokens, tokens], dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("RiNALMo", "build_rinalmo", "example_input_rinalmo", 2024, "vendored-pytorch"),
]
