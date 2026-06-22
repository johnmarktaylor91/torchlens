"""VAR: Visual Autoregressive Modeling via Next-Scale Prediction.

Tian et al., NeurIPS 2024 (Best Paper).
Paper: https://arxiv.org/abs/2404.02905
Source: https://github.com/FoundationVision/VAR

Distinctive paradigm -- NEXT-SCALE (coarse-to-fine) autoregression:
  Classic image AR predicts the next TOKEN in raster order over a single-scale
  token grid. VAR instead represents an image as a PYRAMID of multi-scale token
  maps (1x1, 2x2, 3x3, ..., 16x16) produced by a multi-scale VQ-VAE, and the
  transformer predicts the NEXT SCALE (the next, higher-resolution token map)
  given all coarser scales already generated.

  Each generated scale's tokens are embedded, summed with a per-scale level
  embedding, concatenated into one long sequence, and a block-wise-causal
  transformer (a token may attend within its own scale and to all coarser
  scales) predicts the token logits for the next scale. Class conditioning is
  injected as the start token. This module reproduces:
    - multi-scale token embedding + per-scale level embeddings,
    - the coarse-to-fine concatenated token sequence,
    - a block-causal transformer producing next-scale logits.

Faithful compact random-init reimplementation: small patch_nums pyramid,
small vocab/depth/width, takes class-label + (random) multi-scale token maps
internally so it is forward-able from a single class-label tensor. arXiv:2404.02905.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _AdaLNBlock(nn.Module):
    """Transformer block with adaptive layer-norm conditioning (VAR uses AdaLN)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        scale, shift = self.ada(cond).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class VAR(nn.Module):
    """Visual Autoregressive transformer: next-scale prediction over a token pyramid."""

    def __init__(
        self,
        vocab: int = 256,
        dim: int = 96,
        depth: int = 4,
        heads: int = 4,
        num_classes: int = 16,
        patch_nums: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8),
    ) -> None:
        super().__init__()
        self.patch_nums = patch_nums
        self.vocab = vocab
        self.num_scales = len(patch_nums)
        self.token_embed = nn.Embedding(vocab, dim)
        self.class_embed = nn.Embedding(num_classes, dim)
        total_tokens = sum(p * p for p in patch_nums)
        # per-scale level embeddings broadcast over each scale's tokens
        self.level_embed = nn.Parameter(torch.zeros(self.num_scales, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, dim))
        self.blocks = nn.ModuleList([_AdaLNBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, vocab)
        # block-wise-causal mask: token can attend within its scale and to coarser scales
        self.register_buffer("attn_mask", self._build_block_causal_mask(), persistent=False)
        # per-scale token-count index for level-embedding broadcast
        scale_ids = []
        for s, p in enumerate(patch_nums):
            scale_ids += [s] * (p * p)
        self.register_buffer("scale_ids", torch.tensor(scale_ids), persistent=False)

    def _build_block_causal_mask(self) -> torch.Tensor:
        sizes = [p * p for p in self.patch_nums]
        total = sum(sizes)
        mask = torch.full((total, total), float("-inf"))
        starts = [0]
        for s in sizes[:-1]:
            starts.append(starts[-1] + s)
        for i, (st, sz) in enumerate(zip(starts, sizes)):
            # tokens at scale i may attend to everything up to and including scale i
            allowed_end = st + sz
            mask[st : st + sz, :allowed_end] = 0.0
        return mask

    def forward(self, class_label: torch.Tensor) -> torch.Tensor:
        b = class_label.shape[0]
        total = self.pos_embed.shape[1]
        # synthesize random multi-scale token maps (random-init atlas; the structure is the point)
        tokens = torch.randint(0, self.vocab, (b, total), device=class_label.device)
        x = self.token_embed(tokens) + self.pos_embed
        x = x + self.level_embed[self.scale_ids].unsqueeze(0)
        cond = self.class_embed(class_label)  # (B, dim) AdaLN conditioning
        for blk in self.blocks:
            x = blk(x, cond, self.attn_mask)
        x = self.norm(x)
        return self.head(x)  # (B, total_tokens, vocab) next-scale logits


class _VARWrapper(nn.Module):
    """Single-tensor wrapper: input is the class-label id."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, class_label: torch.Tensor) -> torch.Tensor:
        return self.model(class_label)


def build_var() -> nn.Module:
    """Build a compact VAR (next-scale autoregressive image transformer)."""
    return _VARWrapper(
        VAR(vocab=256, dim=96, depth=4, heads=4, num_classes=16, patch_nums=(1, 2, 3, 4, 5, 6, 8))
    )


def example_input() -> torch.Tensor:
    """Example class label ``(1,)`` int64 for VAR."""
    return torch.zeros(1, dtype=torch.int64)


MENAGERIE_ENTRIES = [
    (
        "VAR (Visual Autoregressive, next-scale coarse-to-fine prediction)",
        "build_var",
        "example_input",
        "2024",
        "DC",
    ),
]
