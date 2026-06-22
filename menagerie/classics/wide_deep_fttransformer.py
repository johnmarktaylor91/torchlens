"""WideDeep FT-Transformer tabular model.

Gorishniy et al., 2021, Revisiting Deep Learning Models for Tabular Data.
Paper: https://openreview.net/forum?id=i_Q1yrOegLY

FT-Transformer tokenizes each numerical/categorical feature into a feature
token, prepends a CLS token, and applies self-attention over features of the
same row.  This module mirrors the WideDeep library's FTTransformer-style deep
tower with compact feature tokenization and a transformer encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Tokenize scalar tabular features into transformer tokens."""

    def __init__(self, n_features: int, dim: int) -> None:
        """Initialize per-feature affine token parameters.

        Parameters
        ----------
        n_features:
            Number of numerical features.
        dim:
            Token embedding dimension.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize features and prepend CLS.

        Parameters
        ----------
        x:
            Numerical features with shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Tokens with shape ``(batch, n_features + 1, dim)``.
        """

        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class CompactFTTransformer(nn.Module):
    """Compact FT-Transformer classifier for tabular data."""

    def __init__(self, n_features: int = 8, dim: int = 24, heads: int = 4) -> None:
        """Initialize tokenizer, transformer encoder, and head.

        Parameters
        ----------
        n_features:
            Number of scalar features.
        dim:
            Token width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify one tabular row batch.

        Parameters
        ----------
        x:
            Numerical features with shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Logits from the CLS token.
        """

        tokens = self.tokenizer(x)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


def build() -> nn.Module:
    """Build a compact FT-Transformer.

    Returns
    -------
    nn.Module
        Random-init FT-Transformer.
    """

    return CompactFTTransformer()


def example_input() -> torch.Tensor:
    """Create a tabular input batch.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(2, 8)``.
    """

    return torch.randn(2, 8)


MENAGERIE_ENTRIES = [
    ("WideDeep-FTTransformer", "build", "example_input", "2021", "DC"),
]
