"""PBWM prefrontal-basal-ganglia working memory, 2006, O'Reilly and Frank.

Paper: "Making working memory work: A computational model of learning in the
prefrontal cortex and basal ganglia." This simplified model uses soft BG gates to
update PFC stripes while ungated stripes maintain prior working-memory state.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("PBWM Prefrontal-Basal-Ganglia Working Memory", "build", "example_input", "2006", "DB")
]


class PBWM(nn.Module):
    """Soft-gated PFC working-memory stripe module."""

    def __init__(self, n_input: int = 64, n_stripes: int = 4, stripe_dim: int = 16) -> None:
        """Initialize input encoder, BG gate, and output head.

        Parameters
        ----------
        n_input
            Input-vector dimensionality.
        n_stripes
            Number of independent PFC stripes.
        stripe_dim
            Units per working-memory stripe.
        """
        super().__init__()
        self.n_stripes = n_stripes
        self.stripe_dim = stripe_dim
        self.encoder = nn.Linear(n_input, n_stripes * stripe_dim)
        self.gate = nn.Linear(n_input, n_stripes)
        self.output = nn.Linear(n_stripes * stripe_dim, n_input)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Update working memory through differentiable BG gates.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output reconstruction and stripe gate values.
        """
        proposal = torch.tanh(self.encoder(x)).view(x.shape[0], self.n_stripes, self.stripe_dim)
        prior = torch.zeros_like(proposal)
        gate = torch.sigmoid(self.gate(x)).unsqueeze(-1)
        wm = gate * proposal + (1.0 - gate) * prior
        return self.output(wm.reshape(x.shape[0], -1)), gate.squeeze(-1)


def build() -> nn.Module:
    """Build a small simplified PBWM module.

    Returns
    -------
    nn.Module
        Configured ``PBWM`` instance.
    """
    return PBWM()


def example_input() -> Tensor:
    """Return a working-memory input example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)
