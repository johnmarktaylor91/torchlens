"""CryptoNets, 2016, Gilad-Bachrach et al.

Paper: "CryptoNets: Applying neural networks to encrypted data with high throughput
and accuracy." This plaintext PyTorch module mirrors the FHE-friendly arithmetic:
linear layers, average pooling, and square activations, omitting encryption.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CryptoNets(nn.Module):
    """Small polynomial-activation CNN compatible with FHE-style arithmetic."""

    def __init__(self) -> None:
        """Initialize convolutional and linear plaintext layers."""
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(4 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, image: Tensor) -> Tensor:
        """Compute class logits with square activations and average pooling.

        Parameters
        ----------
        image
            Image tensor with shape ``(batch, 1, 28, 28)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = self.conv(image).pow(2)
        x = F.avg_pool2d(x, kernel_size=2)
        x = self.fc1(x.flatten(1)).pow(2)
        return self.fc2(x)


MENAGERIE_ENTRIES = [("CryptoNets", "build", "example_input", "2016", "DA")]


def build() -> nn.Module:
    """Build a small CryptoNets model.

    Returns
    -------
    nn.Module
        Configured CryptoNets module.
    """
    return CryptoNets()


def example_input() -> Tensor:
    """Create an image example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 28, 28)``.
    """
    return torch.randn(1, 1, 28, 28)
