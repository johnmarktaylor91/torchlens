"""Deconvolutional sparse coding network, 2010, Matthew Zeiler et al.

Paper: Deconvolutional Networks.
The model encodes images into rectified sparse feature maps and reconstructs
them with learned convolutional dictionaries, a minimal differentiable version
of the original generative sparse-coding hierarchy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Deconvolutional Network (Zeiler sparse coding)", "build", "example_input", "2010", "DC")
]


class DeconvolutionalSparseCoder(nn.Module):
    """Single-layer convolutional sparse coder with transposed-conv decoder."""

    def __init__(self, codes: int = 12) -> None:
        """Initialize encoder, dictionary decoder, and sparse threshold.

        Parameters
        ----------
        codes
            Number of sparse code channels.
        """
        super().__init__()
        self.encoder = nn.Conv2d(1, codes, kernel_size=5, padding=2)
        self.decoder = nn.ConvTranspose2d(codes, 1, kernel_size=5, padding=2)
        self.threshold = nn.Parameter(torch.full((1, codes, 1, 1), 0.15))

    def _soft_threshold(self, x: Tensor) -> Tensor:
        """Apply signed soft-thresholding for sparse codes.

        Parameters
        ----------
        x
            Dense code tensor.

        Returns
        -------
        Tensor
            Sparse code tensor.
        """
        return torch.sign(x) * torch.relu(torch.abs(x) - torch.relu(self.threshold))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and reconstruct an image.

        Parameters
        ----------
        x
            Grayscale image tensor with shape ``(B, 1, 32, 32)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Reconstruction and sparse codes.
        """
        pooled = F.max_pool2d(x, kernel_size=2)
        codes = self._soft_threshold(self.encoder(pooled))
        reconstruction = self.decoder(codes)
        reconstruction = F.interpolate(
            reconstruction, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return reconstruction, codes


def build() -> nn.Module:
    """Build a compact deconvolutional sparse coder.

    Returns
    -------
    nn.Module
        Random-initialized sparse-coding module.
    """
    return DeconvolutionalSparseCoder()


def example_input() -> Tensor:
    """Return a traceable grayscale image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
