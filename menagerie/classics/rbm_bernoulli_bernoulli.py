"""Bernoulli-Bernoulli Restricted Boltzmann Machine.

Smolensky, 1986; Hinton, 2002 contrastive-divergence training.
Paper: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

A Bernoulli-Bernoulli RBM is an undirected bipartite energy model with binary
visible and hidden units.  The faithful forward here exposes the distinctive
primitive used by RBM libraries: positive hidden probabilities, one-step Gibbs
reconstruction, and free-energy computation under random weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BernoulliBernoulliRBM(nn.Module):
    """Compact Bernoulli visible / Bernoulli hidden RBM."""

    def __init__(self, n_visible: int = 12, n_hidden: int = 6) -> None:
        """Initialize RBM parameters.

        Parameters
        ----------
        n_visible:
            Number of visible binary units.
        n_hidden:
            Number of hidden binary units.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.05)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))

    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """Compute RBM free energy.

        Parameters
        ----------
        visible:
            Visible probabilities or binary states with shape ``(batch, n_visible)``.

        Returns
        -------
        torch.Tensor
            Free-energy values with shape ``(batch,)``.
        """

        vbias_term = visible @ self.visible_bias
        hidden_term = F.softplus(visible @ self.weight + self.hidden_bias).sum(dim=-1)
        return -vbias_term - hidden_term

    def forward(self, visible: torch.Tensor) -> torch.Tensor:
        """Run one deterministic Gibbs reconstruction and append free energy.

        Parameters
        ----------
        visible:
            Visible binary/probability tensor with shape ``(batch, n_visible)``.

        Returns
        -------
        torch.Tensor
            Concatenated reconstruction and free energy.
        """

        hidden_prob = torch.sigmoid(visible @ self.weight + self.hidden_bias)
        recon = torch.sigmoid(hidden_prob @ self.weight.t() + self.visible_bias)
        energy = self.free_energy(visible).unsqueeze(-1)
        return torch.cat([recon, energy], dim=-1)


def build() -> nn.Module:
    """Build a compact Bernoulli-Bernoulli RBM.

    Returns
    -------
    nn.Module
        Random-init RBM.
    """

    return BernoulliBernoulliRBM()


def example_input() -> torch.Tensor:
    """Create a small binary visible vector batch.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(2, 12)``.
    """

    return torch.bernoulli(torch.full((2, 12), 0.5))


MENAGERIE_ENTRIES = [
    ("RBM_BernoulliBernoulli", "build", "example_input", "1986", "CB"),
]
