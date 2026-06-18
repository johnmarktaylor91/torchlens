"""Neural Theorem Prover, 2017, Rocktaschel and Riedel, "End-to-end Differentiable Proving".

Paper: Rocktaschel 2017, "End-to-end Differentiable Proving."
This simplified prover computes soft unification between a goal triple and fact
triples with an RBF kernel, then max-pools proof scores over available facts.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuralTheoremProver(nn.Module):
    """Soft-unification prover over embedded triples."""

    def __init__(self, sigma: float = 1.0) -> None:
        """Initialize prover bandwidth.

        Parameters
        ----------
        sigma
            RBF unification bandwidth.
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, packed: Tensor) -> Tensor:
        """Score goals against embedded facts by differentiable unification.

        Parameters
        ----------
        packed
            Tensor of shape ``(batch, n_facts + 1, 3, dim)`` with the first triple as
            the goal and the remaining triples as available facts.

        Returns
        -------
        Tensor
            Maximum proof score for each goal.
        """
        goal = packed[:, 0]
        facts = packed[:, 1:]
        diff = goal[:, None, :, :] - facts
        atom_scores = torch.exp(-torch.sum(diff * diff, dim=-1) / (2.0 * self.sigma * self.sigma))
        proof_scores = torch.amin(atom_scores, dim=-1)
        return torch.amax(proof_scores, dim=-1, keepdim=True)


MENAGERIE_ENTRIES = [("Neural Theorem Prover (NTP)", "build", "example_input", "2017", "CD")]


def build() -> nn.Module:
    """Build a simplified neural theorem prover.

    Returns
    -------
    nn.Module
        Configured NTP module.
    """
    return NeuralTheoremProver()


def example_input() -> Tensor:
    """Create packed fact and goal embedding examples.

    Returns
    -------
    Tensor
        Packed goals and facts with shape ``(2, 6, 3, 6)``.
    """
    return torch.randn(2, 6, 3, 6)
