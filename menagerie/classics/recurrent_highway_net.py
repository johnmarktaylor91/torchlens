"""Recurrent Highway Network, 2016, Zilly et al.

Paper: "Recurrent Highway Networks." Each recurrent transition contains a deep
stack of highway microsteps, increasing transition depth without adding time
steps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RecurrentHighwayNetwork(nn.Module):
    """RHN with gated highway microsteps inside each recurrent transition."""

    def __init__(self, input_size: int = 128, hidden_size: int = 48, depth: int = 3) -> None:
        """Initialize recurrent highway projections.

        Parameters
        ----------
        input_size:
            Per-step feature size.
        hidden_size:
            Hidden state size.
        depth:
            Number of highway microsteps per time step.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.input_h = nn.Linear(input_size, hidden_size)
        self.input_t = nn.Linear(input_size, hidden_size)
        self.state_h = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.state_t = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        """Run RHN recurrence over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Hidden sequence with shape ``(batch, time, hidden_size)``.
        """
        state = x.new_zeros(x.shape[0], self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            base_h = self.input_h(x[:, step])
            base_t = self.input_t(x[:, step])
            for layer_index in range(self.depth):
                proposal = torch.tanh(base_h + self.state_h[layer_index](state))
                transform = torch.sigmoid(base_t + self.state_t[layer_index](state))
                state = proposal * transform + state * (1.0 - transform)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact RHN.

    Returns
    -------
    nn.Module
        Random-initialized RHN.
    """
    return RecurrentHighwayNetwork()


def example_input() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 64, 128)``.
    """
    return torch.randn(1, 64, 128)


MENAGERIE_ENTRIES = [("Recurrent Highway Network (RHN)", "build", "example_input", "2016", "DE")]
