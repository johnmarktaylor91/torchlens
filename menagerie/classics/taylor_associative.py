"""W.K. Taylor associative learning net, 1956, as a three-layer associator.

Paper: Taylor 1956, "Electrical Simulation of Some Nervous System Functional Activities."
A fixed input-to-association map activates hidden units whose Hebbian output
weights retrieve associated responses.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("W.K. Taylor associative learning net", "build", "example_input", "1956", "CA")
]


class TaylorNet(nn.Module):
    """Three-layer analogue Hebbian associative net."""

    def __init__(self, n_in: int = 7, n_assoc: int = 9, n_out: int = 4) -> None:
        """Initialize fixed association and Hebbian readout weights.

        Parameters
        ----------
        n_in
            Number of stimulus units.
        n_assoc
            Number of association units.
        n_out
            Number of output units.
        """
        super().__init__()
        self.input_to_assoc = nn.Linear(n_in, n_assoc)
        prototypes = torch.rand(n_out, n_assoc)
        weights = prototypes / prototypes.sum(dim=-1, keepdim=True)
        self.register_buffer("assoc_to_out", weights)
        self.register_buffer("assoc_threshold", torch.full((n_assoc,), 0.45))

    def forward(self, stimulus: Tensor) -> Tensor:
        """Retrieve an associated output from a stimulus pattern.

        Parameters
        ----------
        stimulus
            Binary or analogue stimulus with shape ``(batch, n_in)``.

        Returns
        -------
        Tensor
            Output association activations.
        """
        assoc_drive = torch.sigmoid(self.input_to_assoc(stimulus))
        assoc = torch.sigmoid(18.0 * (assoc_drive - self.assoc_threshold))
        return torch.sigmoid(assoc @ self.assoc_to_out.T)


def build() -> nn.Module:
    """Build a small Taylor associative net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return TaylorNet()


def example_input() -> Tensor:
    """Create a stimulus example.

    Returns
    -------
    Tensor
        Example stimulus with shape ``(2, 7)``.
    """
    return torch.rand(2, 7)
