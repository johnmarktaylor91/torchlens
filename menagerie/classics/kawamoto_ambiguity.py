"""Kawamoto distributed lexical-ambiguity net, 1988, as recurrent cleanup.

Paper: Kawamoto 1988, "Distributed Representations of Ambiguous Words and Their Resolution in a Connectionist Network."
A concatenated orthography, phonology, and meaning state settles through
Hebbian-like recurrent weights so partial cues resolve toward competing attractors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Kawamoto distributed lexical-ambiguity net", "build", "example_input", "1988", "CB")
]


class KawamotoNet(nn.Module):
    """Distributed lexical ambiguity attractor."""

    def __init__(self, n_total: int = 18, steps: int = 6) -> None:
        """Initialize recurrent attractor weights.

        Parameters
        ----------
        n_total
            Total concatenated unit count.
        steps
            Number of settling steps.
        """
        super().__init__()
        prototypes = torch.sign(torch.randn(4, n_total))
        weights = prototypes.T @ prototypes / n_total
        weights.fill_diagonal_(0.0)
        self.register_buffer("weights", weights)
        self.steps = steps

    def forward(self, code: Tensor) -> Tensor:
        """Settle a partial lexical code to a bipolar attractor.

        Parameters
        ----------
        code
            Concatenated orthographic, phonological, and meaning code.

        Returns
        -------
        Tensor
            Settled bipolar-like code.
        """
        state = code
        clamp = code
        for _ in range(self.steps):
            free = torch.tanh(state @ self.weights)
            state = torch.where(clamp.abs() > 0.5, clamp, free)
        return state


def build() -> nn.Module:
    """Build a small Kawamoto ambiguity net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return KawamotoNet()


def example_input() -> Tensor:
    """Create a partial lexical code example.

    Returns
    -------
    Tensor
        Example code with shape ``(2, 18)``.
    """
    x = torch.zeros(2, 18)
    x[:, :4] = torch.sign(torch.randn(2, 4))
    return x
