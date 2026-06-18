"""Neural cryptography, 2016, Abadi and Andersen, "Learning to Protect".

Paper: Abadi 2016, "Learning to Protect Communications with Adversarial Neural
Cryptography." The joint forward module contains Alice, Bob, and Eve networks;
training losses and alternating adversarial optimization are intentionally
omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CryptoNet(nn.Module):
    """Small one-dimensional convolutional network used by Alice, Bob, or Eve."""

    def __init__(self, in_features: int, n_bits: int = 16) -> None:
        """Initialize dense projection followed by convolutional mixing.

        Parameters
        ----------
        in_features
            Number of concatenated input features.
        n_bits
            Number of plaintext or ciphertext bits.
        """
        super().__init__()
        self.n_bits = n_bits
        self.project = nn.Linear(in_features, n_bits)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform bit vectors through dense and convolutional layers.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Output bit-like tensor with shape ``(batch, n_bits)``.
        """
        projected = torch.tanh(self.project(x)).unsqueeze(1)
        return self.conv(projected).squeeze(1)


class NeuralCryptoTriplet(nn.Module):
    """Joint Alice/Bob/Eve neural cryptography forward graph."""

    def __init__(self, n_bits: int = 16) -> None:
        """Initialize Alice, Bob, and Eve subnetworks.

        Parameters
        ----------
        n_bits
            Number of plaintext and key bits.
        """
        super().__init__()
        self.alice = CryptoNet(n_bits * 2, n_bits)
        self.bob = CryptoNet(n_bits * 2, n_bits)
        self.eve = CryptoNet(n_bits, n_bits)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Encrypt with Alice and decode with Bob and Eve.

        Parameters
        ----------
        inputs
            Tuple ``(plaintext, key)`` where both tensors have shape
            ``(batch, n_bits)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Ciphertext, Bob reconstruction, and Eve reconstruction.
        """
        plaintext, key = inputs
        ciphertext = self.alice(torch.cat((plaintext, key), dim=-1))
        bob_plaintext = self.bob(torch.cat((ciphertext, key), dim=-1))
        eve_plaintext = self.eve(ciphertext)
        return ciphertext, bob_plaintext, eve_plaintext


def build() -> nn.Module:
    """Build the joint neural cryptography triplet.

    Returns
    -------
    nn.Module
        Configured ``NeuralCryptoTriplet`` instance.
    """
    return NeuralCryptoTriplet()


def example_input() -> tuple[Tensor, Tensor]:
    """Create plaintext and key examples.

    Returns
    -------
    tuple[Tensor, Tensor]
        Plaintext and key tensors, each with shape ``(1, 16)``.
    """
    plaintext = torch.sign(torch.randn(1, 16))
    key = torch.sign(torch.randn(1, 16))
    return plaintext, key


MENAGERIE_ENTRIES = [
    ("Neural cryptography (Alice/Bob/Eve)", "build", "example_input", "2016", "CH-D")
]
