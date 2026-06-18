"""Resonator Network, 2020, Frady et al., "Resonator Networks".

Paper: Frady 2020, "Resonator Networks for factoring distributed representations of data structures."
The module iteratively unbinds a composite hypervector by the current estimates of
other factors and projects each estimate onto fixed VSA codebooks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResonatorNetwork(nn.Module):
    """Traceable VSA resonator factorizer."""

    def __init__(
        self, dim: int = 16, n_factors: int = 3, atoms_per_factor: int = 6, n_iters: int = 4
    ) -> None:
        """Initialize bipolar codebooks and iteration count.

        Parameters
        ----------
        dim
            Hypervector dimensionality.
        n_factors
            Number of factors in the composite.
        atoms_per_factor
            Number of candidate atoms per factor.
        n_iters
            Number of resonator updates.
        """
        super().__init__()
        codebooks = torch.sign(torch.randn(n_factors, atoms_per_factor, dim))
        self.register_buffer("codebooks", codebooks)
        self.n_iters = n_iters

    def forward(self, composite: Tensor) -> Tensor:
        """Factor a composite hypervector into soft codebook estimates.

        Parameters
        ----------
        composite
            Bipolar composite hypervector of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Factor logits of shape ``(batch, n_factors, atoms_per_factor)``.
        """
        batch = composite.shape[0]
        estimates = self.codebooks.mean(dim=1).unsqueeze(0).expand(batch, -1, -1)
        logits = composite.new_zeros(batch, self.codebooks.shape[0], self.codebooks.shape[1])
        for _ in range(self.n_iters):
            next_estimates: list[Tensor] = []
            next_logits: list[Tensor] = []
            for factor in range(self.codebooks.shape[0]):
                others = torch.ones_like(composite)
                for other in range(self.codebooks.shape[0]):
                    if other != factor:
                        others = others * estimates[:, other]
                candidate = composite * others
                factor_logits = candidate @ self.codebooks[factor].T
                weights = torch.softmax(factor_logits, dim=-1)
                next_estimates.append(weights @ self.codebooks[factor])
                next_logits.append(factor_logits)
            estimates = torch.stack(next_estimates, dim=1)
            logits = torch.stack(next_logits, dim=1)
        return logits


MENAGERIE_ENTRIES = [
    ("Resonator Network (VSA factorization)", "build", "example_input", "2020", "CE")
]


def build() -> nn.Module:
    """Build a compact resonator network.

    Returns
    -------
    nn.Module
        Configured resonator module.
    """
    return ResonatorNetwork()


def example_input() -> Tensor:
    """Create a bipolar composite hypervector.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 16)``.
    """
    return torch.sign(torch.randn(2, 16))
