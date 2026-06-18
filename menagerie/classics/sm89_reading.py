"""Seidenberg-McClelland reading model, 1989, as a Wickelgraph MLP.

Paper: Seidenberg and McClelland 1989, "A Distributed, Developmental Model of Word Recognition and Naming."
Distributed orthographic Wickelgraph input passes through a hidden layer to
phonological output and orthographic reconstruction heads.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Seidenberg-McClelland reading model (SM89)", "build", "example_input", "1989", "CB")
]


class SM89Reading(nn.Module):
    """Compact SM89-style orthography-to-phonology MLP."""

    def __init__(self, n_orth: int = 64, n_hidden: int = 24, n_phon: int = 48) -> None:
        """Initialize hidden, phonology, and reconstruction layers.

        Parameters
        ----------
        n_orth
            Orthographic input width.
        n_hidden
            Hidden layer width.
        n_phon
            Phonological output width.
        """
        super().__init__()
        self.hidden = nn.Linear(n_orth, n_hidden)
        self.phonology = nn.Linear(n_hidden, n_phon)
        self.orth_recon = nn.Linear(n_hidden, n_orth)

    def forward(self, orthography: Tensor) -> tuple[Tensor, Tensor]:
        """Compute phonology and orthographic reconstruction.

        Parameters
        ----------
        orthography
            Distributed orthographic code with shape ``(batch, n_orth)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Phonological probabilities and orthographic reconstruction probabilities.
        """
        hidden = torch.sigmoid(self.hidden(orthography))
        phon = torch.sigmoid(self.phonology(hidden))
        recon = torch.sigmoid(self.orth_recon(hidden))
        return phon, recon


def build() -> nn.Module:
    """Build a small SM89 reading model.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return SM89Reading()


def example_input() -> Tensor:
    """Create an orthographic code example.

    Returns
    -------
    Tensor
        Example code with shape ``(2, 64)``.
    """
    return (torch.rand(2, 64) > 0.75).float()
