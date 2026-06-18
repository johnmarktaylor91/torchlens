"""Neural Gas, 1991, Martinetz and Schulten.

Paper: Martinetz and Schulten 1991, "A neural-gas network learns topologies."
Prototype adaptation is weighted by rank-ordered distance neighborhoods instead
of a single winner-take-all update.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Neural Gas", "build", "example_input", "1991", "CF")]


class NeuralGas(nn.Module):
    """Prototype layer with rank-neighborhood responses."""

    def __init__(self, dim: int = 4, n_prototypes: int = 7, lambda_rank: float = 2.0) -> None:
        """Initialize prototypes and neighborhood scale.

        Parameters
        ----------
        dim
            Input dimensionality.
        n_prototypes
            Number of prototype vectors.
        lambda_rank
            Rank-neighborhood decay scale.
        """
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim))
        self.lambda_rank = lambda_rank

    def forward(self, x: Tensor) -> Tensor:
        """Return rank-neighborhood prototype activations.

        Parameters
        ----------
        x
            Input samples of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Neighborhood activations ordered by prototype index.
        """
        dist = torch.cdist(x, self.prototypes)
        order = dist.argsort(dim=-1)
        ranks = torch.zeros_like(dist)
        rank_values = torch.arange(dist.shape[-1], device=x.device, dtype=x.dtype).expand_as(dist)
        ranks = ranks.scatter(dim=-1, index=order, src=rank_values)
        return torch.exp(-ranks / self.lambda_rank)


def build() -> nn.Module:
    """Build a compact neural gas module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return NeuralGas()


def example_input() -> Tensor:
    """Return vector samples.

    Returns
    -------
    Tensor
        Example tensor of shape ``(5, 4)``.
    """
    return torch.randn(5, 4)
