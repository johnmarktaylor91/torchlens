"""Wavelet Networks, 1992, Zhang and Benveniste.

Paper: Zhang and Benveniste 1992, "Wavelet networks." Translated and dilated
wavelet basis functions act as hidden units for nonlinear function
approximation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Wavelet Networks (Zhang-Benveniste)", "build", "example_input", "1992", "CF")
]


class WaveletNet(nn.Module):
    """Feedforward network with Mexican-hat wavelet hidden units."""

    def __init__(self, dim: int = 3, n_wavelets: int = 6, output_size: int = 2) -> None:
        """Initialize wavelet centers, scales, and readout.

        Parameters
        ----------
        dim
            Input dimensionality.
        n_wavelets
            Number of wavelet basis functions.
        output_size
            Number of output features.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_wavelets, dim))
        self.log_scales = nn.Parameter(torch.zeros(n_wavelets, dim))
        self.readout = nn.Linear(n_wavelets, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate wavelet basis functions and readout.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Readout tensor.
        """
        z = (x.unsqueeze(1) - self.centers) / self.log_scales.exp().clamp_min(1.0e-3)
        psi = (1.0 - z.square()) * torch.exp(-0.5 * z.square())
        hidden = psi.prod(dim=-1)
        return self.readout(hidden)


def build() -> nn.Module:
    """Build a compact wavelet network.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return WaveletNet()


def example_input() -> Tensor:
    """Return vector inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(3, 3)``.
    """
    return torch.randn(3, 3)
