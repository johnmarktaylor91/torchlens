"""Grossberg recurrent competitive shunting field, 1973.

Grossberg's on-center/off-surround shunting dynamics bound activity with excitatory
growth terms and inhibitory suppression, foundational for ART and visual fields.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GrossbergShuntingField(nn.Module):
    """One-dimensional on-center/off-surround shunting field."""

    def __init__(self, n_units: int = 24, steps: int = 8, dt: float = 0.1) -> None:
        """Initialize fixed shunting kernels.

        Parameters
        ----------
        n_units
            Number of field units.
        steps
            Number of integration steps.
        dt
            Euler update size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.a = 0.2
        self.b = 1.0
        self.d = 0.5
        pos = torch.arange(n_units, dtype=torch.float32)
        dist = torch.minimum(pos, n_units - pos)
        self.register_buffer("exc_weight", torch.exp(-dist.square() / 6.0))
        self.register_buffer("inh_weight", 0.4 * torch.exp(-dist.square() / 60.0))

    def _circular_mix(self, x: Tensor, weight: Tensor) -> Tensor:
        """Apply a dense circular interaction matrix.

        Parameters
        ----------
        x
            Activity tensor.
        weight
            Circular kernel values.

        Returns
        -------
        Tensor
            Mixed activity.
        """
        rows = [torch.roll(weight, shifts=i) for i in range(weight.numel())]
        matrix = torch.stack(rows, dim=0)
        return x @ matrix.t() / weight.numel()

    def forward(self, x: Tensor) -> Tensor:
        """Integrate shunting competitive dynamics.

        Parameters
        ----------
        x
            Input field of shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Bounded field activity.
        """
        state = torch.relu(x)
        for _ in range(self.steps):
            firing = torch.sigmoid(3.0 * state)
            exc = self._circular_mix(firing, self.exc_weight) + torch.relu(x)
            inh = self._circular_mix(firing, self.inh_weight)
            dx = -self.a * state + (self.b - state) * exc - (state + self.d) * inh
            state = torch.clamp(state + self.dt * dx, min=0.0, max=self.b)
        return state


def build() -> nn.Module:
    """Build a Grossberg shunting field.

    Returns
    -------
    nn.Module
        Shunting competitive field module.
    """
    return GrossbergShuntingField()


def example_input() -> Tensor:
    """Return a float32 field input.

    Returns
    -------
    Tensor
        Input of shape ``(2, 24)``.
    """
    x = torch.zeros(2, 24, dtype=torch.float32)
    x[:, 10:14] = 1.0
    return x
