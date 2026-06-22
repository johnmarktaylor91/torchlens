"""AMFormer: arithmetic feature-interaction Transformer for tabular learning.

Paper: "Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning",
AAAI 2024.

AMFormer augments tabular token attention with parallel additive and
multiplicative interaction branches plus learnable prompts.  This compact
random-init reconstruction keeps those distinctive ingredients while reducing
the number of fields, prompts, and layers for TorchLens rendering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArithmeticBlock(nn.Module):
    """Parallel additive and multiplicative token interaction block."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        """Initialize additive attention and multiplicative gates.

        Parameters
        ----------
        dim:
            Token embedding width.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.additive = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mul_left = nn.Linear(dim, dim)
        self.mul_right = nn.Linear(dim, dim)
        self.mix = nn.Linear(dim * 2, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply additive attention and multiplicative feature interactions.

        Parameters
        ----------
        tokens:
            Tabular field and prompt tokens, shape ``(batch, fields, dim)``.

        Returns
        -------
        torch.Tensor
            Updated token tensor with the same shape.
        """

        z = self.norm(tokens)
        add, _ = self.additive(z, z, z, need_weights=False)
        mul = torch.tanh(self.mul_left(z)) * torch.sigmoid(self.mul_right(z))
        tokens = tokens + self.mix(torch.cat((add, mul), dim=-1))
        return tokens + self.ffn(tokens)


class AMFormer(nn.Module):
    """Compact AMFormer classifier over numerical tabular features."""

    def __init__(self, n_features: int = 12, dim: int = 32, n_prompts: int = 3) -> None:
        """Initialize feature tokenizers, prompts, arithmetic blocks, and head.

        Parameters
        ----------
        n_features:
            Number of scalar tabular columns.
        dim:
            Token embedding width.
        n_prompts:
            Number of learnable prompt tokens.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, dim))
        self.prompts = nn.Parameter(torch.randn(n_prompts, dim) * 0.02)
        self.blocks = nn.ModuleList([ArithmeticBlock(dim) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify a batch of tabular feature vectors.

        Parameters
        ----------
        features:
            Numeric features with shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Class logits with shape ``(batch, 4)``.
        """

        tokens = features.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        prompts = self.prompts.unsqueeze(0).expand(features.shape[0], -1, -1)
        tokens = torch.cat((prompts, tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0])


def build_amformer() -> nn.Module:
    """Build a compact AMFormer model.

    Returns
    -------
    nn.Module
        Random-initialized AMFormer.
    """

    return AMFormer()


def example_input() -> torch.Tensor:
    """Return a small tabular batch.

    Returns
    -------
    torch.Tensor
        Input tensor with shape ``(2, 12)``.
    """

    return torch.randn(2, 12)


MENAGERIE_ENTRIES = [
    ("AMFormer", "build_amformer", "example_input", "2024", "tabular"),
]
