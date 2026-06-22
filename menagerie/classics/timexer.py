"""TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.

Jin et al., NeurIPS 2024.
Paper: https://arxiv.org/abs/2402.19072
Source: https://github.com/thuml/TimeXer

TimeXer extends the Transformer for forecasting when both an endogenous target series
and exogenous covariates are available.  The key architectural primitives are:

  1. Endogenous path:
     - Patch embedding: split the endogenous series into non-overlapping patches
       and project each patch to d_model.
     - A learned global token (cls-like) is prepended to attend across all patches.
     - Self-attention over (global_token || patch_tokens).

  2. Exogenous path:
     - Each exogenous variate is embedded as a single token (linear over full length).
     - These become "variate tokens".

  3. Cross-attention: endogenous patch tokens attend TO exogenous variate tokens
     (endo queries, exo keys/values).  The global token also participates.

  4. Feed-forward + prediction head on the global token (or patch-average) -> forecast.

Simplified for compact graph:
  - 1 exogenous variate, 3 patch tokens, 1 global token, 1 transformer layer.
  - Input: endogenous (1, 32, 1) + exogenous (1, 32, 1). Output: (1, 8, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split sequence into non-overlapping patches and project to d_model."""

    def __init__(self, patch_len: int, d_model: int, n_vars: int = 1) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * n_vars, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        # Ensure T divisible by patch_len
        n_patches = T // self.patch_len
        x = x[:, : n_patches * self.patch_len, :]
        x = x.reshape(B, n_patches, self.patch_len * C)
        return self.proj(x)  # (B, n_patches, d_model)


class TimeXerBlock(nn.Module):
    """One TimeXer block: self-attn over endo tokens + cross-attn to exo tokens + FFN."""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Self-attention (endo patches + global token attend to each other)
        self.norm1 = nn.LayerNorm(d_model)
        self.q_self = nn.Linear(d_model, d_model)
        self.k_self = nn.Linear(d_model, d_model)
        self.v_self = nn.Linear(d_model, d_model)
        self.out_self = nn.Linear(d_model, d_model)

        # Cross-attention: endo -> exo
        self.norm2 = nn.LayerNorm(d_model)
        self.q_cross = nn.Linear(d_model, d_model)
        self.k_cross = nn.Linear(d_model, d_model)
        self.v_cross = nn.Linear(d_model, d_model)
        self.out_cross = nn.Linear(d_model, d_model)

        # Feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, Nq, D = q.shape
        Nk = k.shape[1]
        q = q.reshape(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)
        return out.transpose(1, 2).reshape(B, Nq, D)

    def forward(
        self,
        endo_tokens: torch.Tensor,
        exo_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # endo_tokens: (B, N_endo, d_model)  -- global + patches
        # exo_tokens:  (B, N_exo, d_model)

        # Self-attention over endo tokens
        normed = self.norm1(endo_tokens)
        q = self.q_self(normed)
        k = self.k_self(normed)
        v = self.v_self(normed)
        endo_tokens = endo_tokens + self.out_self(self._attn(q, k, v))

        # Cross-attention: endo queries, exo keys/values
        normed = self.norm2(endo_tokens)
        q = self.q_cross(normed)
        k = self.k_cross(exo_tokens)
        v = self.v_cross(exo_tokens)
        endo_tokens = endo_tokens + self.out_cross(self._attn(q, k, v))

        # Feed-forward
        normed = self.norm3(endo_tokens)
        endo_tokens = endo_tokens + self.ff2(F.gelu(self.ff1(normed)))

        return endo_tokens


class TimeXer(nn.Module):
    """TimeXer: endogenous patches + global token + exogenous variate tokens (random-init reimpl).

    Compact: 1 block, patch_len=8, 1 exo var, 3 patches + 1 global token.
    Input: tuple (endo, exo) where endo: (B, L, 1) and exo: (B, L, n_exo).
    Output: (B, pred_len, 1) forecast of endogenous variable.
    """

    def __init__(
        self,
        seq_len: int = 32,
        pred_len: int = 8,
        patch_len: int = 8,
        n_endo: int = 1,
        n_exo: int = 1,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        n_patches = seq_len // patch_len

        # Endogenous: patch embedding
        self.patch_embed = PatchEmbedding(patch_len, d_model, n_endo)

        # Learnable global token
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Exogenous: each variable is embedded as a single variate token (linear over full length)
        self.exo_embed = nn.Linear(seq_len * n_exo, d_model)

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + n_patches, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([TimeXerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)

        # Prediction head: use global token output
        self.head = nn.Linear(d_model, pred_len * n_endo)
        self.n_endo = n_endo

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        endo, exo = x
        # endo: (B, L, n_endo), exo: (B, L, n_exo)
        B = endo.shape[0]

        # --- Endogenous path: patch tokens ---
        patches = self.patch_embed(endo)  # (B, n_patches, d_model)

        # Prepend global token
        global_tok = self.global_token.expand(B, 1, -1)
        endo_tokens = torch.cat([global_tok, patches], dim=1)  # (B, 1+n_patches, d_model)
        endo_tokens = endo_tokens + self.pos_embed

        # --- Exogenous path: variate tokens ---
        B, L, n_exo = exo.shape
        exo_flat = exo.reshape(B, L * n_exo)
        exo_tokens = self.exo_embed(exo_flat).unsqueeze(1)  # (B, 1, d_model)

        # --- Transformer blocks: self-attn over endo + cross-attn to exo ---
        for blk in self.blocks:
            endo_tokens = blk(endo_tokens, exo_tokens)

        endo_tokens = self.norm(endo_tokens)

        # Prediction from global token (index 0)
        global_out = endo_tokens[:, 0, :]  # (B, d_model)
        pred = self.head(global_out)  # (B, pred_len * n_endo)
        return pred.reshape(B, self.pred_len, self.n_endo)


def build_timexer() -> nn.Module:
    return TimeXer(
        seq_len=32,
        pred_len=8,
        patch_len=8,
        n_endo=1,
        n_exo=1,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=1,
    )


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Endogenous (1, 32, 1) + exogenous (1, 32, 1) series."""
    return (torch.randn(1, 32, 1), torch.randn(1, 32, 1))


MENAGERIE_ENTRIES = [
    (
        "TimeXer (endogenous patch tokens + global token + exogenous variate cross-attention)",
        "build_timexer",
        "example_input",
        "2024",
        "DC",
    ),
]
