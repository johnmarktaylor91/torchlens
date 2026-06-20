"""STARFlow / TARFlow: Transformer AutoRegressive Flow.

Zhai et al. (Apple), 2025.
Paper: https://arxiv.org/abs/2506.06276
Source: https://github.com/apple/ml-tarflow (and ml-starflow)

TARFlow / STARFlow is a normalizing flow for generative modeling whose coupling
transforms are produced by a CAUSAL (autoregressive) Transformer operating over
a sequence of latent tokens.  Its DISTINCTIVE mechanism:

  - The latent (here, flattened latent patches) is a sequence of T tokens.
  - A causal Transformer attends over the tokens with a strictly lower-triangular
    mask, so token i's affine-coupling parameters (scale s_i, shift t_i) depend
    only on tokens < i.
  - Each flow step transforms the sequence autoregressively:
        z_i = (x_i - t_i) * exp(-s_i)
    which is invertible by construction (the AR Jacobian is triangular).
  - STARFlow uses a DEEP-SHALLOW design: one deep causal-transformer block stack
    captures most of the modeling capacity, followed by a few shallow blocks that
    refine; each block is one autoregressive flow transform.

This faithful reimplementation captures the AR-transformer-parameterized
normalizing-flow coupling and the deep+shallow block layout at modest width
(embed_dim=128, seq=16 tokens).  forward() returns the transformed latent.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention (lower-triangular mask)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class _CausalBlock(nn.Module):
    """Pre-norm causal transformer block."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _ARFlowTransform(nn.Module):
    """One autoregressive-flow transform parameterized by a causal Transformer.

    A stack of causal blocks predicts per-token (scale, shift); applies the
    affine coupling  z = (x - t) * exp(-s)  autoregressively.  Because the
    transformer is causal, the params for token i depend only on tokens < i, so
    the transform is exactly invertible (triangular Jacobian).
    """

    def __init__(self, token_dim: int, embed_dim: int, depth: int, num_heads: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(token_dim, embed_dim)
        self.blocks = nn.ModuleList([_CausalBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # Produce 2 * token_dim params per token: scale s and shift t.
        self.param_head = nn.Linear(embed_dim, 2 * token_dim)
        self.token_dim = token_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, token_dim).  Shift the input right by one so token i is
        # conditioned only on tokens < i (standard AR-flow convention).
        B, T, D = x.shape
        x_shifted = torch.zeros_like(x)
        x_shifted[:, 1:, :] = x[:, :-1, :]
        h = self.in_proj(x_shifted)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        params = self.param_head(h)  # (B, T, 2*token_dim)
        s, t = params[..., : self.token_dim], params[..., self.token_dim :]
        s = torch.tanh(s)  # bound the log-scale for numerical stability
        z = (x - t) * torch.exp(-s)
        return z


class STARFlow(nn.Module):
    """STARFlow / TARFlow: deep+shallow stack of AR-transformer flow transforms."""

    def __init__(
        self,
        token_dim: int = 8,
        embed_dim: int = 128,
        deep_blocks: int = 4,
        shallow_blocks: int = 2,
        deep_depth: int = 4,
        shallow_depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        # Deep stack: each transform is a deep causal-transformer flow step.
        self.deep = nn.ModuleList(
            [
                _ARFlowTransform(token_dim, embed_dim, deep_depth, num_heads)
                for _ in range(deep_blocks)
            ]
        )
        # Shallow stack: refinement flow steps with fewer causal blocks.
        self.shallow = nn.ModuleList(
            [
                _ARFlowTransform(token_dim, embed_dim, shallow_depth, num_heads)
                for _ in range(shallow_blocks)
            ]
        )
        # Permute token-feature dims between flow steps (standard flow trick) so
        # successive AR transforms mix all coordinates.
        self.register_buffer("perm", torch.randperm(token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, token_dim) latent token sequence.
        z = x
        for tf in self.deep:
            z = tf(z)
            z = z[..., self.perm]
        for tf in self.shallow:
            z = tf(z)
            z = z[..., self.perm]
        return z


def build_starflow() -> nn.Module:
    """Build STARFlow / TARFlow (AR-transformer normalizing flow, deep+shallow)."""
    return STARFlow(
        token_dim=8,
        embed_dim=128,
        deep_blocks=4,
        shallow_blocks=2,
        deep_depth=4,
        shallow_depth=2,
        num_heads=4,
    )


def example_input() -> torch.Tensor:
    """Example latent token sequence ``(1, 16, 8)`` = batch 1, 16 tokens, dim 8."""
    return torch.randn(1, 16, 8)


MENAGERIE_ENTRIES = [
    (
        "STARFlow / TARFlow (transformer autoregressive normalizing flow)",
        "build_starflow",
        "example_input",
        "2025",
        "DC",
    ),
]
