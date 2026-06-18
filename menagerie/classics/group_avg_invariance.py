"""Pitts-McCulloch group-averaging invariance net, 1947, in miniature.

Paper: Pitts and McCulloch 1947, "How We Know Universals: The Perception of Auditory and Visual Forms."
Feature responses are averaged over a small discrete transformation group,
capturing the classical scan-and-average route to invariant recognition.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    (
        "Pitts-McCulloch group-averaging invariance net",
        "build",
        "example_input",
        "1947",
        "CA",
    )
]


class GroupAvgNet(nn.Module):
    """Tiny convolutional recognizer with group-averaged features."""

    def __init__(self, in_channels: int = 1, n_classes: int = 3) -> None:
        """Initialize feature extractor and readout.

        Parameters
        ----------
        in_channels
            Number of image channels.
        n_classes
            Number of output classes.
        """
        super().__init__()
        self.features = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.readout = nn.Linear(4, n_classes)

    def _feature_pool(self, image: Tensor) -> Tensor:
        """Extract pooled features for one transformed image batch.

        Parameters
        ----------
        image
            Image batch.

        Returns
        -------
        Tensor
            Pooled feature vector.
        """
        feature_map = torch.relu(self.features(image))
        return feature_map.mean(dim=(-2, -1))

    def forward(self, image: Tensor) -> Tensor:
        """Average features over identity, shifts, and reflection transforms.

        Parameters
        ----------
        image
            Input image with shape ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Class logits from invariant features.
        """
        transformed = [
            image,
            torch.roll(image, shifts=1, dims=-1),
            torch.roll(image, shifts=1, dims=-2),
            torch.flip(image, dims=(-1,)),
        ]
        pooled = torch.stack([self._feature_pool(view) for view in transformed], dim=0).mean(dim=0)
        return self.readout(pooled)


def build() -> nn.Module:
    """Build a small group-averaging net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return GroupAvgNet()


def example_input() -> Tensor:
    """Create a small image example.

    Returns
    -------
    Tensor
        Example image with shape ``(2, 1, 8, 8)``.
    """
    return torch.randn(2, 1, 8, 8)
