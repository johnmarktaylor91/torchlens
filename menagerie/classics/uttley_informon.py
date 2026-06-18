"""Uttley conditional-probability machine, 1956/1970, as an Informon.

Paper: Uttley 1956, "Conditional Probability Machines"; Uttley 1970, "The Informon."
Running co-occurrence counters estimate conditional outputs from binary features,
implemented here as fixed counters read by a trace-clean forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    (
        "Uttley conditional-probability machine / Informon",
        "build",
        "example_input",
        "1956/1970",
        "CA",
    )
]


class Informon(nn.Module):
    """Conditional-probability readout from co-occurrence counters."""

    def __init__(self, n_in: int = 8, n_out: int = 4) -> None:
        """Initialize count tables.

        Parameters
        ----------
        n_in
            Number of binary input features.
        n_out
            Number of output informons.
        """
        super().__init__()
        joint = torch.rand(n_out, n_in) * 3.0 + 0.5
        counts = torch.rand(n_in) * 4.0 + 2.0
        self.register_buffer("count_joint", joint)
        self.register_buffer("count_in", counts)
        self.register_buffer("threshold", torch.linspace(0.2, 0.5, n_out))

    def forward(self, features: Tensor) -> Tensor:
        """Estimate conditional outputs for active input features.

        Parameters
        ----------
        features
            Binary features with shape ``(batch, n_in)``.

        Returns
        -------
        Tensor
            Thresholded conditional-probability responses.
        """
        weighted_counts = features @ self.count_joint.T
        active_counts = features @ self.count_in.unsqueeze(-1)
        probability = weighted_counts / (active_counts + 1.0e-6)
        return torch.sigmoid(14.0 * (probability - self.threshold))


def build() -> nn.Module:
    """Build a small Informon module.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return Informon()


def example_input() -> Tensor:
    """Create a binary feature example.

    Returns
    -------
    Tensor
        Example features with shape ``(2, 8)``.
    """
    return torch.tensor(
        [[1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]]
    )
