"""EEGNet, 2017, Lawhern et al., "EEGNet: A Compact Convolutional Network".

Paper: Lawhern 2017, "EEGNet: A Compact Convolutional Network for EEG-based
Brain-Computer Interfaces." The module keeps the temporal convolution,
depthwise spatial filtering, and separable convolution blocks used by EEGNet.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EEGNet(nn.Module):
    """Compact EEG decoder with temporal, depthwise spatial, and separable convs."""

    def __init__(self, n_chans: int = 22, n_outputs: int = 4, n_times: int = 1000) -> None:
        """Initialize a small EEGNet-style classifier.

        Parameters
        ----------
        n_chans
            Number of EEG channels.
        n_outputs
            Number of decoded classes.
        n_times
            Number of time samples in the example window.
        """
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 31), padding=(0, 15), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )
        self.separable_depth = nn.Conv2d(
            16, 16, kernel_size=(1, 15), padding=(0, 7), groups=16, bias=False
        )
        self.separable_point = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )
        self.classifier = nn.Linear(16 * (n_times // 16), n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        """Decode EEG windows into class logits.

        Parameters
        ----------
        x
            EEG tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(batch, n_outputs)``.
        """
        features = x.unsqueeze(1)
        features = self.temporal(features)
        features = self.spatial(features)
        features = self.separable_depth(features)
        features = self.separable_point(features)
        return self.classifier(torch.flatten(features, start_dim=1))


def build() -> nn.Module:
    """Build a compact EEGNet decoder.

    Returns
    -------
    nn.Module
        Configured ``EEGNet`` instance.
    """
    return EEGNet()


def example_input() -> Tensor:
    """Create an EEG window example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 22, 1000)``.
    """
    return torch.randn(1, 22, 1000)


MENAGERIE_ENTRIES = [("EEGNet (braindecode)", "build", "example_input", "2017", "CH-D")]
