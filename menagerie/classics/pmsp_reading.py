"""PMSP reading model, 1996, as a simplified triangle attractor.

Paper: Plaut, McClelland, Seidenberg, and Patterson 1996, "Understanding Normal and Impaired Word Reading."
This simplified forward pass sends orthography through hidden units into a
phoneme layer with recurrent cleanup; it omits training, semantics, and lesion modes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("PMSP reading model (triangle model)", "build", "example_input", "1996", "CB")
]


class PMSPModel(nn.Module):
    """Simplified orthography-to-phonology attractor model."""

    def __init__(
        self, n_grapheme: int = 14, n_hidden: int = 12, n_phoneme: int = 10, steps: int = 5
    ) -> None:
        """Initialize feedforward and phonological cleanup layers.

        Parameters
        ----------
        n_grapheme
            Orthographic input width.
        n_hidden
            Hidden layer width.
        n_phoneme
            Phonological output width.
        steps
            Number of phonological settling steps.
        """
        super().__init__()
        self.ortho_to_hidden = nn.Linear(n_grapheme, n_hidden)
        self.hidden_to_phoneme = nn.Linear(n_hidden, n_phoneme)
        cleanup = torch.randn(n_phoneme, n_phoneme) * 0.2
        cleanup = (cleanup + cleanup.T) / 2
        cleanup.fill_diagonal_(0.3)
        self.cleanup = nn.Parameter(cleanup)
        self.steps = steps

    def forward(self, orthography: Tensor) -> Tensor:
        """Settle phonological activations from orthography.

        Parameters
        ----------
        orthography
            Orthographic code with shape ``(batch, n_grapheme)``.

        Returns
        -------
        Tensor
            Phoneme activation probabilities.
        """
        hidden = torch.sigmoid(self.ortho_to_hidden(orthography))
        feedforward = self.hidden_to_phoneme(hidden)
        phoneme = torch.sigmoid(feedforward)
        for _ in range(self.steps):
            phoneme = torch.sigmoid(feedforward + phoneme @ self.cleanup)
        return phoneme


def build() -> nn.Module:
    """Build a small simplified PMSP reading model.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return PMSPModel()


def example_input() -> Tensor:
    """Create an orthographic input example.

    Returns
    -------
    Tensor
        Example code with shape ``(2, 14)``.
    """
    return torch.rand(2, 14)
