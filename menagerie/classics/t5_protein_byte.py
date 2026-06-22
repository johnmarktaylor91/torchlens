"""T5-family encoder/encoder-decoder variants: ByT5 (byte-level) and Ankh (protein).

ByT5 (Xue et al. 2021, arXiv:2105.13626) is a token-free T5 that operates directly
on UTF-8 bytes -- same T5 encoder-decoder architecture but with a tiny byte
"vocabulary" (256 + special) and a *heavier encoder, lighter decoder* split.

Ankh (Elnaggar et al. 2023, arXiv:2301.06568) is a protein language model: a T5
*encoder* (the generative-decoder half is dropped for representation use) trained
on amino-acid sequences with a gated-GELU feed-forward.

Both are faithful T5 stacks.  The distinctive T5 ingredients reproduced here:
  - relative-position bias added to attention logits (learned, bucketed, shared
    across layers in a stack)
  - RMSNorm (T5LayerNorm, no bias / no mean-subtraction) pre-norm blocks
  - no scaling of attention scores by 1/sqrt(d) (T5 folds it into init)
  - gated-GELU feed-forward (the "v1.1"/ByT5/Ankh FFN: two input projections,
    one GELU-gated)

Random init, CPU, forward-only.  Inputs are token-id sequences (we embed
random ids) -- no tokenizer required.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class T5LayerNorm(nn.Module):
    """RMSNorm without mean-subtraction or bias (T5 layer norm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


def _relative_position_bucket(
    rel_pos: torch.Tensor, bidirectional: bool, num_buckets: int = 32, max_distance: int = 128
) -> torch.Tensor:
    ret = torch.zeros_like(rel_pos)
    n = -rel_pos
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = n.abs()
    else:
        n = n.clamp_min(0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_large = (
        max_exact
        + (
            torch.log(n.float().clamp_min(1) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
    )
    val_large = val_large.clamp_max(num_buckets - 1)
    ret += torch.where(is_small, n, val_large)
    return ret


class T5Attention(nn.Module):
    def __init__(
        self, d_model: int, d_kv: int, n_heads: int, has_rel_bias: bool, bidirectional: bool
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_kv = d_kv
        inner = d_kv * n_heads
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        self.bidirectional = bidirectional
        self.has_rel_bias = has_rel_bias
        if has_rel_bias:
            self.rel_bias = nn.Embedding(32, n_heads)

    def _bias(self, qlen: int, klen: int, device) -> torch.Tensor:
        ctx = torch.arange(qlen, device=device)[:, None]
        mem = torch.arange(klen, device=device)[None, :]
        rel = mem - ctx
        bucket = _relative_position_bucket(rel, self.bidirectional)
        vals = self.rel_bias(bucket)  # (q,k,h)
        return vals.permute(2, 0, 1).unsqueeze(0)  # (1,h,q,k)

    def forward(
        self, x: torch.Tensor, kv: torch.Tensor | None = None, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        kv = x if kv is None else kv
        S = kv.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.d_kv).transpose(1, 2)
        k = self.k(kv).view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.v(kv).view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2))  # NO 1/sqrt(d) -- T5 convention
        if bias is not None:
            scores = scores + bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.o(out)


class T5FFGatedGelu(nn.Module):
    """v1.1 / ByT5 / Ankh gated-GELU feed-forward."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(F.gelu(self.wi_0(x)) * self.wi_1(x))


class T5EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_kv: int, d_ff: int, n_heads: int, first: bool) -> None:
        super().__init__()
        self.ln1 = T5LayerNorm(d_model)
        self.attn = T5Attention(d_model, d_kv, n_heads, has_rel_bias=first, bidirectional=True)
        self.ln2 = T5LayerNorm(d_model)
        self.ff = T5FFGatedGelu(d_model, d_ff)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), bias=bias)
        x = x + self.ff(self.ln2(x))
        return x


class T5DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_kv: int, d_ff: int, n_heads: int, first: bool) -> None:
        super().__init__()
        self.ln1 = T5LayerNorm(d_model)
        self.self_attn = T5Attention(
            d_model, d_kv, n_heads, has_rel_bias=first, bidirectional=False
        )
        self.ln2 = T5LayerNorm(d_model)
        self.cross_attn = T5Attention(
            d_model, d_kv, n_heads, has_rel_bias=False, bidirectional=False
        )
        self.ln3 = T5LayerNorm(d_model)
        self.ff = T5FFGatedGelu(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, enc: torch.Tensor, self_bias: torch.Tensor, causal: torch.Tensor
    ) -> torch.Tensor:
        sb = self_bias + causal
        x = x + self.self_attn(self.ln1(x), bias=sb)
        x = x + self.cross_attn(self.ln2(x), kv=enc)
        x = x + self.ff(self.ln3(x))
        return x


class T5Stack(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int,
        d_kv: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        decoder: bool,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.embed = nn.Embedding(vocab, d_model)
        Layer = T5DecoderLayer if decoder else T5EncoderLayer
        self.layers = nn.ModuleList(
            [Layer(d_model, d_kv, d_ff, n_heads, first=(i == 0)) for i in range(n_layers)]
        )
        self.final_ln = T5LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, enc: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(ids)
        T = ids.shape[1]
        bias = (
            self.layers[0].self_attn._bias(T, T, ids.device)
            if self.decoder
            else self.layers[0].attn._bias(T, T, ids.device)
        )
        if self.decoder:
            causal = torch.full((T, T), float("-inf"), device=ids.device).triu(1)[None, None]
            for layer in self.layers:
                x = layer(x, enc, bias, causal)
        else:
            for layer in self.layers:
                x = layer(x, bias)
        return self.final_ln(x)


class ByT5Small(nn.Module):
    """ByT5: byte-level T5 encoder-decoder (heavy encoder, light decoder)."""

    def __init__(
        self,
        vocab: int = 384,
        d_model: int = 256,
        d_kv: int = 64,
        d_ff: int = 512,
        n_heads: int = 6,
    ) -> None:
        super().__init__()
        # ByT5 uses a deeper encoder than decoder; scaled down for the atlas.
        self.encoder = T5Stack(vocab, d_model, d_kv, d_ff, n_heads, n_layers=4, decoder=False)
        self.decoder = T5Stack(vocab, d_model, d_kv, d_ff, n_heads, n_layers=2, decoder=True)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) byte ids in [0, vocab); split into enc/dec halves
        enc_ids = x
        dec_ids = x[:, : x.shape[1] // 2]
        enc = self.encoder(enc_ids)
        dec = self.decoder(dec_ids, enc)
        return self.lm_head(dec)


class AnkhEncoder(nn.Module):
    """Ankh protein LM: T5 encoder only (representation model), gated-GELU FFN."""

    def __init__(
        self, vocab: int = 32, d_model: int = 256, d_kv: int = 64, d_ff: int = 768, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.encoder = T5Stack(vocab, d_model, d_kv, d_ff, n_heads, n_layers=4, decoder=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_byt5() -> nn.Module:
    return ByT5Small()


def build_ankh() -> nn.Module:
    return AnkhEncoder()


def example_input() -> torch.Tensor:
    """Byte-id sequence ``(1, 48)`` for ByT5."""
    return torch.randint(0, 384, (1, 48))


def example_input_ankh() -> torch.Tensor:
    """Amino-acid id sequence ``(1, 64)`` for Ankh."""
    return torch.randint(0, 32, (1, 64))


MENAGERIE_ENTRIES = [
    ("ByT5 (token-free byte-level T5 seq2seq)", "build_byt5", "example_input", "2021", "DC"),
    ("Ankh (protein T5 encoder language model)", "build_ankh", "example_input_ankh", "2023", "DC"),
]
