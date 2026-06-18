"""Broad Learning System, 2017, Chen and Liu.

Paper: "Broad Learning System: An Effective and Efficient Incremental Learning System..."
Flat feature-node windows and enhancement nodes are concatenated and mapped to
outputs by a ridge-style readout; incremental pseudoinverse fitting is omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BroadLearningSystem(nn.Module):
    """Flat feature-node plus enhancement-node Broad Learning System."""

    def __init__(
        self,
        n_inputs: int = 64,
        n_feature_nodes: int = 24,
        n_enhancement_nodes: int = 16,
        n_outputs: int = 8,
    ) -> None:
        """Initialize broad feature, enhancement, and readout blocks.

        Parameters
        ----------
        n_inputs:
            Number of input features.
        n_feature_nodes:
            Number of lateral feature nodes.
        n_enhancement_nodes:
            Number of enhancement nodes.
        n_outputs:
            Number of output units.
        """
        super().__init__()
        self.feature = nn.Linear(n_inputs, n_feature_nodes)
        self.enhancement = nn.Linear(n_feature_nodes, n_enhancement_nodes)
        self.output = nn.Linear(n_feature_nodes + n_enhancement_nodes, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        """Compute broad feature and enhancement-node readout.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Output tensor.
        """
        features = torch.tanh(self.feature(x))
        enhancements = torch.tanh(self.enhancement(features))
        broad_state = torch.cat((features, enhancements), dim=-1)
        return self.output(broad_state)


def build() -> nn.Module:
    """Build a small Broad Learning System.

    Returns
    -------
    nn.Module
        Configured ``BroadLearningSystem`` instance.
    """
    return BroadLearningSystem()


def example_input() -> Tensor:
    """Create a Broad Learning System input.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)


MENAGERIE_ENTRIES = [("Broad Learning System (BLS)", "build", "example_input", "2017", "MB1")]
