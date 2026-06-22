"""SEDD: Score Entropy Discrete Diffusion.

Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the
Data Distribution." arXiv:2310.16834 (NeurIPS 2024 oral).
Source: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

Distinctive primitive:
  A diffusion-transformer (DiT-like) operating on DISCRETE token sequences.
  The model predicts score-entropy ratios s(x, t, theta) -- a vector over
  vocabulary for each token, representing the ratio of forward-diffusion
  transition probabilities. This is conditioned on continuous time t via a
  sinusoidal time embedding injected as adaLN-Zero modulation (scale+shift to
  LayerNorm, scale to FFN/attn outputs) -- the same conditioning used in
  DiT, but applied to discrete token sequences.

  Network structure:
    token embed (vocab -> d_model)
  + sinusoidal time embed -> MLP -> adaLN params
  + transformer blocks with adaLN-Zero (time-conditioned LN)
  + output head: (d_model -> vocab) projecting score-entropy ratios

Faithful-compact simplifications:
  - 2 transformer layers, d_model=64, 4 heads.
  - adaLN-Zero: single MLP per block (2*d_model scale+shift for LN,
    2 extra scale factors for attn/ffn outputs).
  - Vocabulary = 32, sequence length = 8.
  - No masking-specific noise schedule (just the network structure).
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal embedding of scalar time t in [0, 1]."""

    def __init__(self, d: int) -> None:
        super().__init__()
        assert d % 2 == 0
        self.d = d

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float  ->  (B, d)"""
        half = self.d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freqs[None]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d)


class AdaLNZero(nn.Module):
    """adaLN-Zero conditioning: time embedding -> (alpha, beta, gamma_attn, gamma_ffn).

    Following DiT: LayerNorm(x) is scaled/shifted by alpha/beta (from time);
    attn and FFN outputs are multiplied by gate gamma (initialized near zero).
    """

    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        # 6 * d_model params: alpha_attn, beta_attn, gamma_attn,
        #                     alpha_ffn,  beta_ffn,  gamma_ffn
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model),
        )
        # Initialize output near zero so gates start near 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """c: (B, d_cond) -> 6 tensors each (B, d_model)"""
        out = self.mlp(c)  # (B, 6*d_model)
        return out.chunk(6, dim=-1)


class SEDDBlock(nn.Module):
    """One SEDD transformer block with adaLN-Zero time conditioning."""

    def __init__(self, d_model: int, n_heads: int, d_cond: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln = AdaLNZero(d_model, d_cond)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model),  c: (B, d_cond)  ->  (B, L, d_model)"""
        B, L, D = x.shape
        alpha_a, beta_a, gamma_a, alpha_f, beta_f, gamma_f = self.adaln(c)

        # unsqueeze to (B, 1, d_model) for broadcast over L
        def us(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1)

        # --- Self-attention with adaLN conditioning ---
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + us(alpha_a)) + us(beta_a)

        qkv = self.qkv(x_norm).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)  # each (B, L, n_heads, d_head)
        scale = math.sqrt(self.d_head)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) / scale  # (B,L,L,n_heads)
        attn = F.softmax(attn, dim=2)
        out = torch.einsum("bijh,bjhd->bihd", attn, v)  # (B,L,n_heads,d_head)
        out = out.reshape(B, L, D)
        out = self.proj(out)
        x = x + us(gamma_a) * out

        # --- FFN with adaLN ---
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + us(alpha_f)) + us(beta_f)
        ffn_out = self.ffn(x_norm)
        x = x + us(gamma_f) * ffn_out
        return x


class SEDDScoreNet(nn.Module):
    """SEDD score-entropy network.

    Input: token sequence (B, L) int + time t (B,) float.
    Output: score-entropy ratios (B, L, vocab_size) float.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_time: int = 64,
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(d_time),
            nn.Linear(d_time, 4 * d_time),
            nn.SiLU(),
            nn.Linear(4 * d_time, 4 * d_time),
        )
        d_cond = 4 * d_time
        self.blocks = nn.ModuleList([SEDDBlock(d_model, n_heads, d_cond) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) int64,  t: (B,) float  ->  (B, L, vocab_size)"""
        x = self.tok_embed(tokens)  # (B, L, d_model)
        c = self.time_embed(t)  # (B, d_cond)
        for blk in self.blocks:
            x = blk(x, c)
        x = self.norm_out(x)
        return self.head(x)  # (B, L, vocab_size)


def build_sedd() -> nn.Module:
    return SEDDScoreNet(vocab_size=32, d_model=64, n_heads=4, n_layers=2, d_time=32)


def example_input_sedd() -> list[torch.Tensor]:
    """Batch=1, seq_len=8 discrete tokens + scalar time."""
    torch.manual_seed(3)
    tokens = torch.randint(0, 32, (1, 8))
    t = torch.rand(1)
    return [tokens, t]


MENAGERIE_ENTRIES = [
    (
        "SEDD (Score Entropy Discrete Diffusion)",
        "build_sedd",
        "example_input_sedd",
        "2024",
        "DC",
    ),
]
