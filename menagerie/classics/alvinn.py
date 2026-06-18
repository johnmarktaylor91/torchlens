"""ALVINN autonomous driving network, 1989, Pomerleau.

Paper: "ALVINN: An autonomous land vehicle in a neural network."
A low-resolution road image is flattened into a feedforward steering distribution;
camera preprocessing and vehicle-control loops are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ALVINNAutonomousDrivingNetwork(nn.Module):
    """Small image-to-steering MLP inspired by ALVINN."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 30, 32),
        hidden_size: int = 29,
        n_bins: int = 30,
    ) -> None:
        """Initialize the steering network.

        Parameters
        ----------
        image_shape
            Channel, height, and width of the low-resolution road image.
        hidden_size
            Number of hidden units.
        n_bins
            Number of discrete steering bins.
        """
        super().__init__()
        n_features = image_shape[0] * image_shape[1] * image_shape[2]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_bins),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Map a road image to steering probabilities.

        Parameters
        ----------
        image
            Road image tensor of shape ``(batch, 1, 30, 32)``.

        Returns
        -------
        Tensor
            Steering-bin probability distribution.
        """
        return torch.softmax(self.net(image), dim=-1)


def build() -> nn.Module:
    """Build a small ALVINN-style driving network.

    Returns
    -------
    nn.Module
        Configured ``ALVINNAutonomousDrivingNetwork`` instance.
    """
    return ALVINNAutonomousDrivingNetwork()


def example_input() -> Tensor:
    """Create an example low-resolution road image.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 30, 32)``.
    """
    return torch.randn(1, 1, 30, 32)


MENAGERIE_ENTRIES = [("ALVINN Autonomous Driving Network", "build", "example_input", "1989", "DD")]
