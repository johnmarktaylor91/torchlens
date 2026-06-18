"""Hinton-Shallice deep-dyslexia attractor network, 1991, in miniature.

Paper: Hinton and Shallice 1991, "Lesioning an Attractor Network: Investigations of Acquired Dyslexia."
Orthographic input drives semantic units that settle through recurrent cleanup
connections; this minimal forward pass omits lesion experiments and training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Hinton-Shallice deep-dyslexia attractor network", "build", "example_input", "1991", "CB")
]


class HintonShalliceAttractor(nn.Module):
    """Orthography-to-semantics attractor with recurrent cleanup."""

    def __init__(
        self, n_grapheme: int = 12, n_hidden: int = 10, n_semantic: int = 8, steps: int = 5
    ) -> None:
        """Initialize feedforward and cleanup pathways.

        Parameters
        ----------
        n_grapheme
            Orthographic input width.
        n_hidden
            Intermediate hidden width.
        n_semantic
            Semantic attractor width.
        steps
            Number of settling steps.
        """
        super().__init__()
        self.orth_to_hidden = nn.Linear(n_grapheme, n_hidden)
        self.hidden_to_semantic = nn.Linear(n_hidden, n_semantic)
        cleanup = torch.randn(n_semantic, n_semantic) * 0.25
        cleanup = (cleanup + cleanup.T) / 2
        cleanup.fill_diagonal_(0.4)
        self.cleanup = nn.Parameter(cleanup)
        self.steps = steps

    def forward(self, orthography: Tensor) -> Tensor:
        """Settle semantic activations from an orthographic code.

        Parameters
        ----------
        orthography
            Orthographic code with shape ``(batch, n_grapheme)``.

        Returns
        -------
        Tensor
            Settled semantic activation.
        """
        hidden = torch.sigmoid(self.orth_to_hidden(orthography))
        semantic = torch.sigmoid(self.hidden_to_semantic(hidden))
        for _ in range(self.steps):
            semantic = torch.sigmoid(self.hidden_to_semantic(hidden) + semantic @ self.cleanup)
        return semantic


def build() -> nn.Module:
    """Build a small Hinton-Shallice attractor.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return HintonShalliceAttractor()


def example_input() -> Tensor:
    """Create an orthographic code example.

    Returns
    -------
    Tensor
        Example code with shape ``(2, 12)``.
    """
    return torch.rand(2, 12)
