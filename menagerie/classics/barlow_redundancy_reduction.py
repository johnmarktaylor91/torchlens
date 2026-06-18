"""Barlow redundancy-reduction coding net, 1961, Horace Barlow.

Paper: "Possible principles underlying the transformation of sensory messages."
A linear whitening-like code represents the redundancy-reduction principle; batch
decorrelation losses are deliberately outside this minimal forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Barlow Redundancy-Reduction Coding Net", "build", "example_input", "1961", "DA")
]


class BarlowRedundancyReduction(nn.Module):
    """Linear decorrelating sensory code."""

    def __init__(self, n_input: int = 256, n_code: int = 64) -> None:
        """Initialize the linear coding matrix.

        Parameters
        ----------
        n_input
            Input feature count.
        n_code
            Output code dimensionality.
        """
        super().__init__()
        self.encoder = nn.Linear(n_input, n_code, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Encode inputs as normalized redundancy-reduced features.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        Tensor
            Centered and variance-normalized code.
        """
        z = self.encoder(x)
        return (z - z.mean(dim=-1, keepdim=True)) / (z.std(dim=-1, keepdim=True) + 1.0e-4)


def build() -> nn.Module:
    """Build a small Barlow coding module.

    Returns
    -------
    nn.Module
        Configured ``BarlowRedundancyReduction`` instance.
    """
    return BarlowRedundancyReduction()


def example_input() -> Tensor:
    """Return a sensory vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 256)``.
    """
    return torch.randn(1, 256)
