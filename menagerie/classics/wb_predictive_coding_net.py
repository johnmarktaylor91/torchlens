"""Whittington-Bogacz predictive-coding network, 2017, Whittington and Bogacz.

Paper: "An approximation of the error backpropagation algorithm in a predictive
coding network with local Hebbian synaptic plasticity." Iterative hidden-state
updates combine bottom-up evidence with top-down prediction errors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Whittington-Bogacz Predictive Coding Network", "build", "example_input", "2017", "DA")
]


class WBPredictiveCodingNet(nn.Module):
    """Layered predictive-coding inference loop."""

    def __init__(
        self, n_input: int = 256, n_hidden: int = 64, n_output: int = 10, steps: int = 4
    ) -> None:
        """Initialize recognition, generative, and classifier weights.

        Parameters
        ----------
        n_input
            Input feature count.
        n_hidden
            Hidden representation size.
        n_output
            Number of output logits.
        steps
            Number of inference iterations.
        """
        super().__init__()
        self.recognition = nn.Linear(n_input, n_hidden)
        self.generative = nn.Linear(n_hidden, n_input)
        self.classifier = nn.Linear(n_hidden, n_output)
        self.steps = steps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run predictive-coding hidden-state inference.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output logits and final sensory prediction error.
        """
        h = torch.tanh(self.recognition(x))
        error = x - self.generative(h)
        for _ in range(self.steps):
            h = torch.tanh(self.recognition(x) + error @ self.generative.weight)
            error = x - self.generative(h)
        return self.classifier(h), error


def build() -> nn.Module:
    """Build a small Whittington-Bogacz predictive-coding module.

    Returns
    -------
    nn.Module
        Configured ``WBPredictiveCodingNet`` instance.
    """
    return WBPredictiveCodingNet()


def example_input() -> Tensor:
    """Return a sensory vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 256)``.
    """
    return torch.randn(1, 256)
