"""Progressive Neural Network with lateral transfer columns.

Rusu et al. (2016), "Progressive Neural Networks."  Progressive nets add a new
task-specific column for each task while freezing earlier columns and feeding
their hidden activations through learned lateral connections into the new
column.  This compact reconstruction has two frozen source columns and one
active target column with layer-wise lateral adapters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveColumn(nn.Module):
    """Single multilayer column that returns hidden activations."""

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        """Initialize the column.

        Parameters
        ----------
        in_features:
            Input feature count.
        hidden:
            Hidden feature width.
        out_features:
            Output feature count.
        """

        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features, hidden), nn.Linear(hidden, hidden)])
        self.head = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the column and return hidden activations plus logits.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[list[torch.Tensor], torch.Tensor]
            Hidden activations and output logits.
        """

        activations = []
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
            activations.append(h)
        return activations, self.head(h)


class CompactProgressiveNeuralNetwork(nn.Module):
    """Three-column progressive network with lateral adapters."""

    def __init__(self, in_features: int = 12, hidden: int = 24, classes: int = 4) -> None:
        """Initialize source and target columns.

        Parameters
        ----------
        in_features:
            Input feature count.
        hidden:
            Hidden feature width.
        classes:
            Number of output classes.
        """

        super().__init__()
        self.sources = nn.ModuleList(
            [
                ProgressiveColumn(in_features, hidden, classes),
                ProgressiveColumn(in_features, hidden, classes),
            ]
        )
        for source in self.sources:
            for param in source.parameters():
                param.requires_grad_(False)
        self.target_layers = nn.ModuleList(
            [nn.Linear(in_features, hidden), nn.Linear(hidden, hidden)]
        )
        self.lateral = nn.ModuleList(
            [nn.ModuleList([nn.Linear(hidden, hidden) for _ in self.sources]) for _ in range(2)]
        )
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen source columns and target column with lateral transfer.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Target-task logits.
        """

        source_acts = [source(x)[0] for source in self.sources]
        h = x
        for idx, layer in enumerate(self.target_layers):
            lateral_sum = 0.0
            for src_idx, adapter in enumerate(self.lateral[idx]):
                lateral_sum = lateral_sum + adapter(source_acts[src_idx][idx])
            h = F.relu(layer(h) + lateral_sum)
        return self.head(h)


def build() -> nn.Module:
    """Build the compact Progressive Neural Network.

    Returns
    -------
    nn.Module
        Random-init progressive network in evaluation mode.
    """

    return CompactProgressiveNeuralNetwork().eval()


def example_input() -> torch.Tensor:
    """Return compact vector observations.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, 12)``.
    """

    return torch.randn(2, 12)


MENAGERIE_ENTRIES = [
    ("ProgressiveNeuralNetwork", "build", "example_input", "2016", "E3"),
]
