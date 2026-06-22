"""RealMLP tabular model.

Paper: Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data,
Holzmuller, Grinsztajn, and Steinwart 2024.

RealMLP is an MLP strengthened for tabular data with robust continuous-feature
scaling plus smooth clipping, numerical embeddings, categorical embeddings,
diagonal feature weighting, pre-activation normalization, residual hidden
blocks, and a calibrated output head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RealMLPBlock(nn.Module):
    """Pre-normalized gated residual MLP block."""

    def __init__(self, width: int) -> None:
        """Initialize a residual block.

        Parameters
        ----------
        width:
            Feature width.
        """

        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated residual transformation.

        Parameters
        ----------
        x:
            Hidden feature tensor.

        Returns
        -------
        torch.Tensor
            Residually updated hidden tensor.
        """

        z = self.norm(x)
        return x + self.out(F.silu(self.gate(z)) * self.value(z))


class NumericalEmbedding(nn.Module):
    """Periodic numerical embeddings with learned per-feature frequencies."""

    def __init__(self, features: int, emb_dim: int = 4) -> None:
        """Initialize numerical embedding parameters.

        Parameters
        ----------
        features:
            Number of continuous features.
        emb_dim:
            Embedding dimensions per feature.
        """

        super().__init__()
        self.freq = nn.Parameter(torch.linspace(0.5, 2.0, emb_dim).repeat(features, 1))
        self.phase = nn.Parameter(torch.zeros(features, emb_dim))
        self.diagonal = nn.Parameter(torch.ones(features, emb_dim * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scaled numerical features with a diagonal weighting layer.

        Parameters
        ----------
        x:
            Continuous features of shape ``(batch, features)``.

        Returns
        -------
        torch.Tensor
            Flattened numerical embeddings.
        """

        angles = x.unsqueeze(-1) * self.freq.unsqueeze(0) + self.phase.unsqueeze(0)
        embedded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return (embedded * self.diagonal.unsqueeze(0)).flatten(1)


class RealMLP(nn.Module):
    """Compact RealMLP with continuous and categorical tabular inputs."""

    def __init__(
        self, continuous: int = 6, cat_cards: tuple[int, ...] = (5, 7), width: int = 32
    ) -> None:
        """Initialize the tabular MLP.

        Parameters
        ----------
        continuous:
            Number of continuous features.
        cat_cards:
            Cardinalities for categorical fields.
        width:
            Hidden width.
        """

        super().__init__()
        self.register_buffer("median", torch.linspace(-0.5, 0.5, continuous))
        self.register_buffer("iqr", torch.linspace(0.8, 1.2, continuous))
        self.num_embedding = NumericalEmbedding(continuous)
        self.embeddings = nn.ModuleList([nn.Embedding(card, 4) for card in cat_cards])
        self.in_proj = nn.Linear(continuous * 8 + 4 * len(cat_cards), width)
        self.blocks = nn.Sequential(RealMLPBlock(width), RealMLPBlock(width))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict tabular logits.

        Parameters
        ----------
        data:
            Tuple of continuous features ``(batch, C)`` and categorical ids ``(batch, K)``.

        Returns
        -------
        torch.Tensor
            Calibrated logits.
        """

        continuous, categories = data
        continuous = (continuous - self.median) / self.iqr.clamp_min(1e-3)
        continuous = continuous / torch.sqrt(1.0 + (continuous / 3.0).pow(2))
        numerical = self.num_embedding(continuous)
        embedded = [emb(categories[:, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat([numerical, *embedded], dim=-1)
        return self.head(self.blocks(F.silu(self.in_proj(x)))) / self.temperature.abs().clamp_min(
            0.1
        )


def build() -> nn.Module:
    """Build compact RealMLP.

    Returns
    -------
    nn.Module
        Random-init RealMLP model.
    """

    return RealMLP()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return continuous and categorical tabular features.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Continuous and categorical feature tensors.
    """

    return torch.randn(3, 6), torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)


MENAGERIE_ENTRIES = [("RealMLP", "build", "example_input", "2024", "tabular/mlp")]
