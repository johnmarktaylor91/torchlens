"""Convolutional Pose Machines (CPM).

Wei et al., CVPR 2016, arXiv:1602.00134.  CPM refines pose belief maps through a
sequence of convolutional stages; later stages consume image features plus the
previous stage's belief maps and are trained with intermediate supervision.  This
compact module emits all stage belief maps concatenated along channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CPMStage(nn.Module):
    """One CPM refinement stage."""

    def __init__(self, in_channels: int, joints: int) -> None:
        """Initialize a CPM stage.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        joints:
            Number of pose belief maps.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, joints, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict belief maps.

        Parameters
        ----------
        x:
            Stage input features.

        Returns
        -------
        torch.Tensor
            Joint belief maps.
        """

        return self.net(x)


class CompactCPM(nn.Module):
    """Compact sequential pose machine."""

    def __init__(self, joints: int = 6, stages: int = 3) -> None:
        """Initialize compact CPM.

        Parameters
        ----------
        joints:
            Number of joints.
        stages:
            Number of refinement stages.
        """

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
        )
        self.first = CPMStage(24, joints)
        self.refine = nn.ModuleList([CPMStage(24 + joints, joints) for _ in range(stages - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run staged pose refinement.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        torch.Tensor
            Concatenated stage belief maps.
        """

        feat = self.features(x)
        belief = self.first(feat)
        outputs = [belief]
        for stage in self.refine:
            belief = stage(torch.cat([feat, belief], dim=1))
            outputs.append(belief)
        return torch.cat(outputs, dim=1)


def build() -> nn.Module:
    """Build compact CPM.

    Returns
    -------
    nn.Module
        Random-init CPM reconstruction.
    """

    return CompactCPM()


def example_input() -> torch.Tensor:
    """Create image input.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("ConvolutionalPoseMachines-CPM", "build", "example_input", "2016", "CV")]
