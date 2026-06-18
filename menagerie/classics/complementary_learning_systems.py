"""Complementary Learning Systems, 1995, McClelland, McNaughton, and O'Reilly.

Paper: "Why there are complementary learning systems in the hippocampus and neocortex."
A fast hippocampal autoassociator runs beside a slow neocortical MLP; replay and
consolidation training are omitted from this minimal forward substrate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Complementary Learning Systems (CLS)", "build", "example_input", "1995", "DB")
]


class ComplementaryLearningSystems(nn.Module):
    """Slow cortical mapper plus fast associative hippocampal recall."""

    def __init__(self, n_input: int = 100, n_hidden: int = 64, n_patterns: int = 8) -> None:
        """Initialize cortical network and fixed hippocampal memory.

        Parameters
        ----------
        n_input
            Input and recalled-pattern dimensionality.
        n_hidden
            Cortical hidden-layer size.
        n_patterns
            Number of random hippocampal patterns.
        """
        super().__init__()
        self.cortex = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_input)
        )
        patterns = torch.sign(torch.randn(n_patterns, n_input))
        weights = patterns.T @ patterns / float(n_input)
        self.register_buffer("hippo_weights", weights * (1.0 - torch.eye(n_input)))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute slow cortical output and fast hippocampal recall.

        Parameters
        ----------
        x
            Episodic input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Cortical output and hippocampal pattern completion.
        """
        cortical = self.cortex(x)
        hippo = torch.tanh(torch.tanh(x) @ self.hippo_weights)
        return cortical, hippo


def build() -> nn.Module:
    """Build a small complementary-learning-systems module.

    Returns
    -------
    nn.Module
        Configured ``ComplementaryLearningSystems`` instance.
    """
    return ComplementaryLearningSystems()


def example_input() -> Tensor:
    """Return an episodic vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 100)``.
    """
    return torch.randn(1, 100)
