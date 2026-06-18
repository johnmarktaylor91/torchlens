"""HMAX visual hierarchy, 1999, Riesenhuber and Poggio.

Paper: Hierarchical models of object recognition in cortex.
Alternating S and C stages use fixed Gabor-like simple cells, local max-pooling
complex cells, template RBF matching, and a global C2 readout.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("HMAX Standard Model of Visual Cortex", "build", "example_input", "1999", "DC")
]


class HMAXStandardModel(nn.Module):
    """Small HMAX-style S1-C1-S2-C2 image classifier."""

    def __init__(self, orientations: int = 4, templates: int = 6, num_classes: int = 5) -> None:
        """Initialize fixed S1 filters, S2 templates, and readout.

        Parameters
        ----------
        orientations
            Number of oriented S1 filters.
        templates
            Number of S2 radial-basis templates.
        num_classes
            Number of classifier outputs.
        """
        super().__init__()
        self.register_buffer("s1_weight", self._make_gabor_bank(orientations))
        self.templates = nn.Parameter(torch.randn(templates, orientations, 4, 4) * 0.2)
        self.beta = nn.Parameter(torch.tensor(0.4))
        self.classifier = nn.Linear(templates, num_classes)

    def _make_gabor_bank(self, orientations: int) -> Tensor:
        """Create a small bank of fixed oriented Gabor filters.

        Parameters
        ----------
        orientations
            Number of orientations to sample.

        Returns
        -------
        Tensor
            Filter tensor with shape ``(orientations, 1, 9, 9)``.
        """
        coords = torch.linspace(-1.0, 1.0, 9)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernels = []
        for idx in range(orientations):
            theta = math.pi * float(idx) / float(orientations)
            xr = xx * math.cos(theta) + yy * math.sin(theta)
            yr = -xx * math.sin(theta) + yy * math.cos(theta)
            envelope = torch.exp(-(xr.square() + 0.5 * yr.square()) / 0.4)
            carrier = torch.cos(2.5 * math.pi * xr)
            kernel = envelope * carrier
            kernels.append(kernel - kernel.mean())
        return torch.stack(kernels).unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        """Compute HMAX C2 logits for a grayscale image.

        Parameters
        ----------
        x
            Input image tensor with shape ``(B, 1, 128, 128)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        s1 = torch.relu(F.conv2d(x, self.s1_weight, padding=4))
        c1 = F.max_pool2d(s1, kernel_size=4, stride=4)
        patches = F.unfold(c1, kernel_size=4, stride=2).transpose(1, 2)
        templates = self.templates.flatten(1)
        distance = torch.cdist(patches, templates.unsqueeze(0).expand(x.shape[0], -1, -1))
        s2 = torch.exp(-torch.relu(self.beta) * distance.square())
        c2 = s2.amax(dim=1)
        return self.classifier(c2)


def build() -> nn.Module:
    """Build a compact HMAX model.

    Returns
    -------
    nn.Module
        Random-initialized HMAX-style classifier.
    """
    return HMAXStandardModel()


def example_input() -> Tensor:
    """Return a traceable grayscale image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 128, 128)``.
    """
    return torch.randn(1, 1, 128, 128)
