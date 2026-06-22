"""UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation.

Mao et al., CIKM 2021.
Paper: https://arxiv.org/abs/2110.15114
Source: https://github.com/xue-pai/UltraGCN

UltraGCN dispenses with explicit message passing entirely. It keeps only user
and item embedding tables; the "graph convolution" is replaced by a constraint
loss (approximating the infinite-layer GCN limit) plus an item-item
co-occurrence regularizer. The forward signal for a (user, item) pair is just
the dot product of the corresponding embeddings. This module reimplements the
scoring path: user/item embedding lookup followed by an inner product, which is
the entire learned architecture (the rest lives in the loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UltraGCN(nn.Module):
    """UltraGCN scoring model: embedding lookup + dot product.

    Parameters
    ----------
    n_users, n_items:
        Sizes of the user and item embedding tables.
    embed_dim:
        Embedding dimension (64 in the original).
    """

    def __init__(self, n_users: int = 1000, n_items: int = 2000, embed_dim: int = 64) -> None:
        super().__init__()
        self.user_embeds = nn.Embedding(n_users, embed_dim)
        self.item_embeds = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embeds.weight, std=1e-4)
        nn.init.normal_(self.item_embeds.weight, std=1e-4)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        # pairs: (B, 2) long tensor of [user_id, item_id]
        users = pairs[:, 0]
        items = pairs[:, 1]
        ue = self.user_embeds(users)
        ie = self.item_embeds(items)
        return (ue * ie).sum(dim=-1)


class _UltraGCNWrapper(nn.Module):
    """Wrapper accepting a float tensor of indices (cast to long) for tracing."""

    def __init__(self, model: UltraGCN) -> None:
        super().__init__()
        self.model = model

    def forward(self, pairs_float: torch.Tensor) -> torch.Tensor:
        return self.model(pairs_float.long())


def build() -> nn.Module:
    """Build the UltraGCN recommender scoring model."""
    return _UltraGCNWrapper(UltraGCN(n_users=1000, n_items=2000, embed_dim=64))


def example_input() -> torch.Tensor:
    """Batch of ``(user_id, item_id)`` index pairs as a float tensor ``(8, 2)``."""
    users = torch.randint(0, 1000, (8, 1)).float()
    items = torch.randint(0, 2000, (8, 1)).float()
    return torch.cat([users, items], dim=1)


MENAGERIE_ENTRIES = [
    (
        "UltraGCN (message-passing-free GCN recommender)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
