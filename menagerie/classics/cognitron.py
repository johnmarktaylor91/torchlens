"""Cognitron (1975), Kunihiko Fukushima.

Paper: "Cognitron: A self-organizing multilayered neural network."
Local excitatory receptive fields are opposed by inhibitory competition, forming
a multilayer feature hierarchy without the Neocognitron's shift-invariant pooling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Cognitron(nn.Module):
    """Compact local-excitation/lateral-inhibition Cognitron hierarchy."""

    def __init__(self, n_classes: int = 4) -> None:
        """Initialize local receptive-field filters and classifier.

        Parameters
        ----------
        n_classes
            Number of output classes.
        """
        super().__init__()
        self.excite1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.excite2 = nn.Conv2d(4, 6, kernel_size=3, padding=1)
        self.readout = nn.Linear(6 * 8 * 8, n_classes)
        self.register_buffer("inhibit_kernel", torch.ones(1, 1, 3, 3) / 9.0)

    def _compete(self, x: Tensor) -> Tensor:
        """Apply channelwise local lateral inhibition.

        Parameters
        ----------
        x
            Excitatory feature map.

        Returns
        -------
        Tensor
            Rectified inhibited feature map.
        """
        bsz, channels, height, width = x.shape
        kernel = self.inhibit_kernel.expand(channels, 1, 3, 3)
        local_mean = F.conv2d(x, kernel, padding=1, groups=channels)
        inhibited = x - 0.6 * local_mean
        return torch.relu(inhibited).view(bsz, channels, height, width)

    def forward(self, x: Tensor) -> Tensor:
        """Run the Cognitron feature hierarchy.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, 8, 8)``.

        Returns
        -------
        Tensor
            Class scores.
        """
        h1 = self._compete(torch.relu(self.excite1(x)))
        h2 = self._compete(torch.relu(self.excite2(h1)))
        return self.readout(h2.flatten(1))


def build() -> nn.Module:
    """Build a small Cognitron module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return Cognitron()


def example_input() -> Tensor:
    """Return an example grayscale image batch.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.randn(1, 1, 8, 8)
