"""Sibling-order toy models for the collapse segment tests.

Interleaved ``TinyBlock`` / ``Transition`` siblings and parent-op connectors: a
segment box may only group siblings that are adjacent in execution order, and these
models give the rank-segment and collapse-plan tests non-adjacent siblings to refuse.
"""

from __future__ import annotations

import torch


class TinyBlock(torch.nn.Module):
    """Three-op block; every instance has the same role."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the block."""

        super().__init__()
        self.fc1 = torch.nn.Linear(width, width)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.fc2(self.relu(self.fc1(x)))


class Transition(torch.nn.Module):
    """Two-op block of a second role, interleaved between ``TinyBlock`` s."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the transition."""

        super().__init__()
        self.fc = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transition."""

        return self.act(self.fc(x))


class AlternatingSiblings(torch.nn.Module):
    """DenseNet-shaped sibling order: block, transition, block, ..., block."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the alternating siblings."""

        super().__init__()
        self.stem = torch.nn.Linear(width, width)
        self.b1 = TinyBlock(width)
        self.t1 = Transition(width)
        self.b2 = TinyBlock(width)
        self.t2 = Transition(width)
        self.b3 = TinyBlock(width)
        self.t3 = Transition(width)
        self.b4 = TinyBlock(width)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the siblings in alternating order."""

        x = self.stem(x)
        for name in ("b1", "t1", "b2", "t2", "b3", "t3", "b4"):
            x = getattr(self, name)(x)
        return self.head(x)


class ParentOpConnectors(torch.nn.Module):
    """Same-role siblings joined through parent-owned ops, not directly."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the connector chain."""

        super().__init__()
        self.b0 = TinyBlock(width)
        self.b1 = TinyBlock(width)
        self.b2 = TinyBlock(width)
        self.b3 = TinyBlock(width)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks with a parent-owned op between each pair."""

        x = self.b0(x)
        x = torch.sigmoid(x)
        x = self.b1(x)
        x = torch.sigmoid(x)
        x = self.b2(x)
        x = torch.sigmoid(x)
        x = self.b3(x)
        return self.head(x)
