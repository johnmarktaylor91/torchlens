"""Cellular Neural Network, 1988, Chua and Yang.

Paper: Chua and Yang 1988, "Cellular neural networks: Theory."
Locally coupled analog cells settle by Euler integration with 3x3 feedback and
control templates plus a clipped piecewise-linear output nonlinearity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Cellular Neural Network (Chua-Yang CNN)", "build", "example_input", "1988", "CF")
]


class ChuaYangCNN(nn.Module):
    """Small Chua-Yang cellular neural network with fixed edge template."""

    def __init__(self, steps: int = 6, dt: float = 0.2) -> None:
        """Initialize feedback and control templates.

        Parameters
        ----------
        steps
            Number of Euler settling steps.
        dt
            Integration step size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        a = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]) / 4.0
        b = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]) / 8.0
        self.feedback = nn.Parameter(a.view(1, 1, 3, 3))
        self.control = nn.Parameter(b.view(1, 1, 3, 3))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, u: Tensor) -> Tensor:
        """Settle the analog cell state for an input image.

        Parameters
        ----------
        u
            Input image tensor of shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Settled cell outputs with the same shape as ``u``.
        """
        x = u.clone()
        y = 0.5 * (F.relu(x + 1.0) - F.relu(x - 1.0))
        for _ in range(self.steps):
            fb = F.conv2d(y, self.feedback, padding=1)
            ctrl = F.conv2d(u, self.control, padding=1)
            x = x + self.dt * (-x + fb + ctrl + self.bias)
            y = 0.5 * (F.relu(x + 1.0) - F.relu(x - 1.0))
        return y


def build() -> nn.Module:
    """Build a compact Chua-Yang cellular neural network.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return ChuaYangCNN()


def example_input() -> Tensor:
    """Return a small grayscale image input.

    Returns
    -------
    Tensor
        Example image of shape ``(2, 1, 8, 8)``.
    """
    return torch.randn(2, 1, 8, 8)
