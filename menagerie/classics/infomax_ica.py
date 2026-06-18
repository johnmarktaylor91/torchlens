"""Bell-Sejnowski InfoMax ICA, 1995, Bell and Sejnowski.

Paper: "An information-maximization approach to blind separation and blind deconvolution."
A linear unmixing matrix followed by a logistic nonlinearity represents the core
entropy-maximizing independent-component analysis transform.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Bell-Sejnowski InfoMax ICA", "build", "example_input", "1995", "DA")]


class InfoMaxICA(nn.Module):
    """Single-layer logistic InfoMax unmixing network."""

    def __init__(self, n_features: int = 128) -> None:
        """Initialize the square unmixing transform.

        Parameters
        ----------
        n_features
            Number of observed mixture features.
        """
        super().__init__()
        self.unmix = nn.Linear(n_features, n_features)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply linear unmixing and logistic source coding.

        Parameters
        ----------
        x
            Mixture tensor of shape ``(batch, n_features)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Logistic source estimates and pre-nonlinearity activations.
        """
        u = self.unmix(x)
        y = torch.sigmoid(u)
        return y, u


def build() -> nn.Module:
    """Build a small random-init InfoMax ICA module.

    Returns
    -------
    nn.Module
        Configured ``InfoMaxICA`` instance.
    """
    return InfoMaxICA()


def example_input() -> Tensor:
    """Return a float mixture example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 128)``.
    """
    return torch.randn(1, 128)
