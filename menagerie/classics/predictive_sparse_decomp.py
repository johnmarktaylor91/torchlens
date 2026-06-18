"""Predictive Sparse Decomposition, 2008, Kavukcuoglu, Ranzato, and LeCun.

Paper: "Fast inference in sparse coding algorithms with applications to object
recognition." A convolutional encoder predicts sparse codes and a transposed
convolutional dictionary reconstructs the input; ISTA-like refinements are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Predictive Sparse Decomposition (PSD, DA variant)", "build", "example_input", "2008", "DA")
]


class PredictiveSparseDecomp(nn.Module):
    """Convolutional sparse-code predictor and decoder."""

    def __init__(self, n_code: int = 12, sparsity: float = 0.1) -> None:
        """Initialize the predictive encoder and decoder dictionary.

        Parameters
        ----------
        n_code
            Number of sparse feature maps.
        sparsity
            Soft-threshold level.
        """
        super().__init__()
        self.encoder = nn.Conv2d(1, n_code, 5, padding=2)
        self.decoder = nn.ConvTranspose2d(n_code, 1, 5, padding=2)
        self.sparsity = sparsity

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict sparse feature maps and reconstruct the image.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, 28, 28)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Sparse code and reconstruction.
        """
        activations = self.encoder(x)
        code = torch.sign(activations) * F.relu(torch.abs(activations) - self.sparsity)
        recon = self.decoder(code)
        return code, recon


def build() -> nn.Module:
    """Build a small predictive sparse decomposition module.

    Returns
    -------
    nn.Module
        Configured ``PredictiveSparseDecomp`` instance.
    """
    return PredictiveSparseDecomp()


def example_input() -> Tensor:
    """Return an image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 28, 28)``.
    """
    return torch.randn(1, 1, 28, 28)
