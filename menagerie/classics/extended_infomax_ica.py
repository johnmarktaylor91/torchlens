"""Extended InfoMax ICA, 1999, Lee, Girolami, and Sejnowski.

Paper: "Independent component analysis using an extended infomax algorithm for mixed
sub-Gaussian and super-Gaussian sources." Fixed component signs select score
functions for mixed kurtosis source families while preserving the InfoMax unmixing core.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Extended InfoMax ICA", "build", "example_input", "1999", "DA")]


class ExtendedInfoMaxICA(nn.Module):
    """InfoMax unmixing with sign-adapted score functions."""

    def __init__(self, n_features: int = 128) -> None:
        """Initialize the unmixing transform and kurtosis sign buffer.

        Parameters
        ----------
        n_features
            Number of observed mixture features.
        """
        super().__init__()
        self.unmix = nn.Linear(n_features, n_features)
        signs = torch.ones(n_features)
        signs[1::2] = -1.0
        self.register_buffer("kurtosis_sign", signs)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute source activations and extended InfoMax scores.

        Parameters
        ----------
        x
            Mixture tensor of shape ``(batch, n_features)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Linear source estimates and sign-adapted score values.
        """
        y = self.unmix(x)
        super_gauss = torch.tanh(y)
        sub_gauss = y - torch.tanh(y)
        positive = (self.kurtosis_sign > 0.0).to(y.dtype)
        score = positive * super_gauss + (1.0 - positive) * sub_gauss
        return y, score


def build() -> nn.Module:
    """Build a small extended InfoMax ICA module.

    Returns
    -------
    nn.Module
        Configured ``ExtendedInfoMaxICA`` instance.
    """
    return ExtendedInfoMaxICA()


def example_input() -> Tensor:
    """Return a float mixture example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 128)``.
    """
    return torch.randn(1, 128)
