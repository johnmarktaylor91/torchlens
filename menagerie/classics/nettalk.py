"""NETtalk, 1987, Sejnowski and Rosenberg.

Paper: "Parallel Networks that Learn to Pronounce English Text." NETtalk is a
windowed letter-to-phoneme multilayer perceptron over seven one-hot character
positions with separate phoneme and stress outputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NETtalk(nn.Module):
    """Seven-letter window MLP for phoneme and stress prediction."""

    def __init__(
        self,
        window: int = 7,
        alphabet: int = 29,
        hidden: int = 16,
        phonemes: int = 12,
        stresses: int = 4,
    ) -> None:
        """Initialize the NETtalk-style MLP.

        Parameters
        ----------
        window:
            Number of letter slots in the input window.
        alphabet:
            One-hot character alphabet size.
        hidden:
            Hidden unit count.
        phonemes:
            Number of phoneme output classes.
        stresses:
            Number of stress output classes.
        """
        super().__init__()
        self.window = window
        self.alphabet = alphabet
        self.hidden = nn.Linear(window * alphabet, hidden)
        self.phoneme_head = nn.Linear(hidden, phonemes)
        self.stress_head = nn.Linear(hidden, stresses)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict phoneme and stress logits from a one-hot letter window.

        Parameters
        ----------
        x:
            One-hot tensor with shape ``(B, 7, 29)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Phoneme logits and stress logits.
        """
        flat = x.reshape(x.shape[0], self.window * self.alphabet)
        hidden = torch.tanh(self.hidden(flat))
        return self.phoneme_head(hidden), self.stress_head(hidden)


def build() -> nn.Module:
    """Build a small random-init NETtalk module.

    Returns
    -------
    nn.Module
        A traceable ``NETtalk`` instance.
    """
    return NETtalk()


def example_input() -> Tensor:
    """Return one-hot seven-letter windows.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 7, 29)``.
    """
    indices = torch.tensor([[0, 5, 12, 12, 15, 28, 28], [19, 15, 18, 3, 8, 28, 28]])
    return torch.nn.functional.one_hot(indices, num_classes=29).to(torch.float32)
