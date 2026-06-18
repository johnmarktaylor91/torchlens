"""Willshaw correlograph, 1969, as a 2D optical associative memory.

Paper: Willshaw, Buneman, and Longuet-Higgins 1969, "Non-Holographic Associative Memory."
Binary point patterns are recalled by spatial cross-correlation against a learned
screen of displacements, a geometric cousin of Willshaw's correlation matrix memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Willshaw correlograph (optical associative net)", "build", "example_input", "1969", "CA")
]


class Correlograph(nn.Module):
    """Binary correlograph recall by spatial correlation."""

    def __init__(self, size: int = 7) -> None:
        """Initialize a small displacement screen.

        Parameters
        ----------
        size
            Height and width of the correlograph screen.
        """
        super().__init__()
        screen = torch.zeros(1, 1, size, size)
        screen[:, :, size // 2, size // 2] = 1.0
        screen[:, :, size // 2 - 1, size // 2 + 1] = 1.0
        screen[:, :, size // 2 + 1, size // 2 - 2] = 1.0
        self.register_buffer("screen", screen)

    def forward(self, key_map: Tensor) -> Tensor:
        """Recall a binary point pattern by cross-correlation.

        Parameters
        ----------
        key_map
            Binary key map with shape ``(batch, height, width)``.

        Returns
        -------
        Tensor
            Recalled point-likelihood map.
        """
        key = key_map.unsqueeze(1)
        padding = self.screen.shape[-1] // 2
        recall = F.conv2d(key, self.screen, padding=padding)
        return torch.sigmoid(8.0 * (recall.squeeze(1) - 0.5))


def build() -> nn.Module:
    """Build a small Willshaw correlograph.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return Correlograph()


def example_input() -> Tensor:
    """Create a binary point-map example.

    Returns
    -------
    Tensor
        Example map with shape ``(1, 7, 7)``.
    """
    key = torch.zeros(1, 7, 7)
    key[:, 3, 3] = 1.0
    key[:, 2, 4] = 1.0
    return key
