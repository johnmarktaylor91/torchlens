"""Recurrent Cascade-Correlation, 1991, Fahlman, "The Recurrent Cascade-Correlation Architecture".

Each frozen cascade feature unit is given a self-recurrent connection, so installed
feature detectors become sequence-state detectors over an unrolled forward pass.
"""

import torch
from torch import Tensor, nn


class RecurrentCascadeCorrelation(nn.Module):
    """Static recurrent cascade-correlation sequence module."""

    def __init__(
        self, input_size: int = 4, hidden_sizes: tuple[int, ...] = (3, 3), out_size: int = 2
    ) -> None:
        """Initialize frozen cascade units, recurrent diagonals, and readout.

        Parameters
        ----------
        input_size:
            Number of input features per time step.
        hidden_sizes:
            Width of each installed recurrent cascade stage.
        out_size:
            Number of output features per time step.
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.units = nn.ModuleList()
        self.self_weights = nn.ParameterList()
        running_size = input_size
        for hidden_size in hidden_sizes:
            unit = nn.Linear(running_size, hidden_size)
            for parameter in unit.parameters():
                parameter.requires_grad_(False)
            self.units.append(unit)
            self.self_weights.append(
                nn.Parameter(torch.randn(hidden_size) * 0.1, requires_grad=False)
            )
            running_size += hidden_size
        self.readout = nn.Linear(running_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        """Unroll recurrent cascade features over a sequence.

        Parameters
        ----------
        x:
            Input sequence of shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Output sequence of shape ``(batch, time, out_size)``.
        """
        batch = x.shape[0]
        states = [x.new_zeros(batch, hidden_size) for hidden_size in self.hidden_sizes]
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            features = [x[:, step]]
            joined = x[:, step]
            new_states: list[Tensor] = []
            for index, unit in enumerate(self.units):
                rec = states[index] * self.self_weights[index]
                h = torch.tanh(unit(joined) + rec)
                new_states.append(h)
                features.append(h)
                joined = torch.cat(features, dim=-1)
            states = new_states
            outputs.append(self.readout(joined))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small recurrent cascade-correlation module.

    Returns
    -------
    nn.Module
        Configured ``RecurrentCascadeCorrelation`` instance.
    """
    return RecurrentCascadeCorrelation()


def example_input() -> Tensor:
    """Create a batch-first sequence example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 3, 4)``.
    """
    return torch.randn(2, 3, 4)
