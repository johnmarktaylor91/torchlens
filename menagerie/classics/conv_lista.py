"""Convolutional LISTA, 2010, Karol Gregor and Yann LeCun.

Paper: Learning Fast Approximations of Sparse Coding.
The convolutional variant unrolls a fixed number of ISTA updates with learned
encoder dictionaries, decoder dictionaries, and shrinkage thresholds.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [("Convolutional LISTA (ConvLISTA)", "build", "example_input", "2010", "DC")]


class ConvolutionalLISTA(nn.Module):
    """Small unrolled convolutional sparse-coding module."""

    def __init__(self, codes: int = 10, steps: int = 4) -> None:
        """Initialize encoder, decoder, thresholds, and iteration count.

        Parameters
        ----------
        codes
            Number of sparse code channels.
        steps
            Number of unrolled ISTA iterations.
        """
        super().__init__()
        self.steps = steps
        self.encoder = nn.Conv2d(1, codes, kernel_size=5, padding=2, bias=False)
        self.decoder = nn.ConvTranspose2d(codes, 1, kernel_size=5, padding=2, bias=False)
        self.threshold = nn.Parameter(torch.full((1, codes, 1, 1), 0.1))

    def _shrink(self, x: Tensor) -> Tensor:
        """Apply learned soft-threshold shrinkage.

        Parameters
        ----------
        x
            Dense code tensor.

        Returns
        -------
        Tensor
            Thresholded sparse code tensor.
        """
        return torch.sign(x) * torch.relu(torch.abs(x) - torch.relu(self.threshold))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run unrolled convolutional ISTA and reconstruct the input.

        Parameters
        ----------
        x
            Grayscale image tensor with shape ``(B, 1, 32, 32)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Reconstruction and final sparse code tensor.
        """
        z = self.encoder(x).mul(0.0)
        residual = x
        for _ in range(self.steps):
            z = self._shrink(z + self.encoder(residual))
            reconstruction = self.decoder(z)
            residual = x - reconstruction
        return self.decoder(z), z


def build() -> nn.Module:
    """Build a compact ConvLISTA module.

    Returns
    -------
    nn.Module
        Random-initialized unrolled sparse coder.
    """
    return ConvolutionalLISTA()


def example_input() -> Tensor:
    """Return a traceable grayscale image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
