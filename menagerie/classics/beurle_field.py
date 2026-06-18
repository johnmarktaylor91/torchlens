"""Beurle regenerating-cell field, 1956, as a small neural-field module.

Paper: Beurle 1956, "Properties of a Mass of Cells Capable of Regenerating Pulses."
The model evolves a 2D activity-density field with Gaussian local excitation,
threshold regeneration, and refractory decay that supports wave-like propagation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Beurle regenerating-cell field", "build", "example_input", "1956", "CA")]


class BeurleField(nn.Module):
    """Discrete Beurle activity field with refractory recovery."""

    def __init__(self, size: int = 9, steps: int = 4, threshold: float = 0.18) -> None:
        """Initialize Gaussian connectivity and refractory parameters.

        Parameters
        ----------
        size
            Spatial side length used by the example field.
        steps
            Number of Euler-style field updates.
        threshold
            Drive threshold for regenerating activity.
        """
        super().__init__()
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / 4.0)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, size, size))
        self.steps = steps
        self.threshold = threshold
        self.refractory_decay = 0.65

    def forward(self, field: Tensor) -> Tensor:
        """Evolve the activity-density field for a few regeneration steps.

        Parameters
        ----------
        field
            Seed activity with shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Final activity field.
        """
        activity = field.clamp(0.0, 1.0)
        refractory = torch.zeros_like(activity)
        padding = self.kernel.shape[-1] // 2
        for _ in range(self.steps):
            drive = F.conv2d(activity, self.kernel, padding=padding)
            new_activity = torch.sigmoid(20.0 * (drive - self.threshold)) * (1.0 - refractory)
            refractory = (self.refractory_decay * refractory + new_activity).clamp(0.0, 1.0)
            activity = new_activity
        return activity


def build() -> nn.Module:
    """Build a small Beurle field.

    Returns
    -------
    nn.Module
        Configured field module.
    """
    return BeurleField()


def example_input() -> Tensor:
    """Create a pulse-seeded 2D field.

    Returns
    -------
    Tensor
        Example field with shape ``(1, 1, 9, 9)``.
    """
    field = torch.zeros(1, 1, 9, 9)
    field[:, :, 4, 4] = 1.0
    field[:, :, 4, 5] = 0.6
    return field
