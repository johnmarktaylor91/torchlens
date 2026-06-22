"""Relation Network for Omniglot few-shot learning.

Paper: Learning to Compare: Relation Network for Few-Shot Learning, Sung et al. 2018.

The model embeds support and query images, concatenates each support-query pair,
and learns a nonlinear relation module instead of using a fixed metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEmbedding(nn.Module):
    """Four-layer convolutional embedding module used by Relation Networks."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize the embedding network."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed images."""

        return self.net(x)


class RelationNetwork(nn.Module):
    """Few-shot relation network with learned pairwise comparator."""

    def __init__(self, ways: int = 3, channels: int = 16) -> None:
        """Initialize embedding and relation modules."""

        super().__init__()
        self.ways = ways
        self.embed = ConvEmbedding(channels)
        self.relation = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Score support-query class relations.

        Parameters
        ----------
        data:
            Support images ``(ways, 1, H, W)`` and query images ``(batch, 1, H, W)``.
        """

        support, query = data
        support_embed = self.embed(support)
        query_embed = self.embed(query)
        batch = query_embed.shape[0]
        support_expand = support_embed.unsqueeze(0).expand(batch, self.ways, -1, -1, -1)
        query_expand = query_embed.unsqueeze(1).expand(batch, self.ways, -1, -1, -1)
        pairs = torch.cat([support_expand, query_expand], dim=2).flatten(0, 1)
        return self.relation(pairs).view(batch, self.ways)


def build() -> nn.Module:
    """Build compact Relation Network."""

    return RelationNetwork()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return support and query Omniglot-like images."""

    return torch.randn(3, 1, 16, 16), torch.randn(2, 1, 16, 16)


MENAGERIE_ENTRIES = [
    ("relation_network_omniglot", "build", "example_input", "2018", "few-shot/relation")
]
