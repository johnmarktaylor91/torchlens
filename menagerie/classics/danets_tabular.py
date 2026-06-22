"""DANets compact Deep Abstract Network for tabular data.

Chen et al., "DANets: Deep Abstract Networks for Tabular Data Classification
and Regression", AAAI 2022.  DANets introduce the Abstract Layer (AbstLay),
which explicitly groups correlated raw features into higher-level abstract
features, and stack basic blocks with shortcuts from raw tabular inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractLayer(nn.Module):
    """DANet Abstract Layer with learned feature grouping."""

    def __init__(self, in_features: int, groups: int, group_width: int) -> None:
        """Initialize grouping attention and grouped projections.

        Parameters
        ----------
        in_features:
            Number of tabular input features.
        groups:
            Number of abstract feature groups.
        group_width:
            Width of each abstract group.
        """

        super().__init__()
        self.groups = groups
        self.selector = nn.Linear(in_features, groups * in_features)
        self.project = nn.Linear(in_features, groups * group_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Group correlated features and emit abstract features.

        Parameters
        ----------
        x:
            Tabular features.

        Returns
        -------
        torch.Tensor
            Abstracted feature vector.
        """

        gate = torch.softmax(self.selector(x).view(x.shape[0], self.groups, -1), dim=-1)
        grouped = gate * x.unsqueeze(1)
        projected = self.project(grouped.reshape(x.shape[0] * self.groups, -1))
        projected = projected.view(x.shape[0], self.groups, self.groups, -1).diagonal(
            dim1=1, dim2=2
        )
        return projected.flatten(1)


class DANetBlock(nn.Module):
    """DANet block with raw-feature shortcut."""

    def __init__(self, raw_features: int, hidden: int) -> None:
        """Initialize abstract layer and shortcut projection.

        Parameters
        ----------
        raw_features:
            Raw tabular feature count.
        hidden:
            Hidden representation width.
        """

        super().__init__()
        self.abstract = AbstractLayer(raw_features, groups=4, group_width=hidden // 4)
        self.shortcut = nn.Linear(raw_features, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, raw: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Update hidden state from raw feature abstractions.

        Parameters
        ----------
        raw:
            Raw tabular features.
        state:
            Current hidden state.

        Returns
        -------
        torch.Tensor
            Updated hidden state.
        """

        return self.norm(state + F.gelu(self.abstract(raw)) + self.shortcut(raw))


class DANetsCompact(nn.Module):
    """Stacked Deep Abstract Network for tabular classification."""

    def __init__(self, in_features: int = 12, hidden: int = 32, classes: int = 3) -> None:
        """Initialize DANets blocks and head.

        Parameters
        ----------
        in_features:
            Raw tabular feature count.
        hidden:
            Hidden width.
        classes:
            Output class count.
        """

        super().__init__()
        self.input = nn.Linear(in_features, hidden)
        self.blocks = nn.ModuleList([DANetBlock(in_features, hidden) for _ in range(3)])
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify tabular features.

        Parameters
        ----------
        x:
            Raw tabular tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        state = F.gelu(self.input(x))
        for block in self.blocks:
            state = block(x, state)
        return self.head(state)


def build() -> nn.Module:
    """Build a compact DANets model.

    Returns
    -------
    nn.Module
        DANets classifier.
    """

    return DANetsCompact()


def example_input() -> torch.Tensor:
    """Create tabular input.

    Returns
    -------
    torch.Tensor
        Example tensor ``(4, 12)``.
    """

    return torch.randn(4, 12)


MENAGERIE_ENTRIES = [
    ("DANets-tabular (Abstract-Layer tabular network)", "build", "example_input", "2022", "DC"),
]
