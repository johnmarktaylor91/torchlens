"""NODE-tabular Neural Oblivious Decision Ensemble.

Paper: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data,
Popov, Morozov, and Babenko 2019.

NODE uses differentiable oblivious decision trees: every tree level applies one
feature selector shared across all nodes at that depth, producing soft routing
probabilities to all leaves and a learned leaf-response table.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObliviousTreeLayer(nn.Module):
    """Differentiable ensemble of oblivious decision trees."""

    def __init__(self, in_features: int, trees: int = 4, depth: int = 3, tree_dim: int = 2) -> None:
        """Initialize selectors, thresholds, temperatures, and leaves."""

        super().__init__()
        self.trees = trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.feature_logits = nn.Parameter(torch.randn(trees, depth, in_features) * 0.1)
        self.thresholds = nn.Parameter(torch.zeros(trees, depth))
        self.log_temperatures = nn.Parameter(torch.zeros(trees, depth))
        self.leaves = nn.Parameter(torch.randn(trees, 2**depth, tree_dim) * 0.1)
        bits = torch.arange(2**depth).unsqueeze(1).bitwise_and(2 ** torch.arange(depth))
        self.register_buffer("leaf_bits", (bits > 0).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate soft oblivious trees and concatenate their responses."""

        selectors = F.softmax(self.feature_logits, dim=-1)
        selected = torch.einsum("bf,tdf->btd", x, selectors)
        logits = (selected - self.thresholds) * torch.exp(-self.log_temperatures)
        prob_right = torch.sigmoid(logits)
        bits = self.leaf_bits.unsqueeze(0).unsqueeze(0)
        route = prob_right.unsqueeze(2) * bits + (1.0 - prob_right.unsqueeze(2)) * (1.0 - bits)
        weights = route.prod(dim=-1)
        return torch.einsum("btl,tlo->bto", weights, self.leaves).flatten(1)


class NODETabular(nn.Module):
    """Compact NODE stack with dense hierarchical representation learning."""

    def __init__(self, in_features: int = 8) -> None:
        """Initialize the NODE model."""

        super().__init__()
        self.layer1 = ObliviousTreeLayer(in_features)
        self.layer2 = ObliviousTreeLayer(8)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits from tabular features."""

        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return self.head(h2)


def build() -> nn.Module:
    """Build compact NODE-tabular."""

    return NODETabular()


def example_input() -> torch.Tensor:
    """Return tabular features."""

    return torch.randn(2, 8)


MENAGERIE_ENTRIES = [("NODE-tabular", "build", "example_input", "2019", "tabular/oblivious-trees")]
