"""Dynamic Field Theory architecture, 1995.

Schoner, Spencer, and colleagues composed Amari-style neural fields into embodied
cognition architectures with local excitation, global inhibition, memory, and decisions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DynamicFieldTheory(nn.Module):
    """Coupled input, memory, and decision dynamic fields."""

    def __init__(self, field_size: int = 24, steps: int = 6, dt: float = 0.1) -> None:
        """Initialize fixed Mexican-hat field kernels.

        Parameters
        ----------
        field_size
            Number of units per field.
        steps
            Number of field updates.
        dt
            Euler update size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        pos = torch.arange(field_size, dtype=torch.float32)
        dist = torch.minimum(pos, field_size - pos)
        local_exc = torch.exp(-dist.square() / 8.0)
        broad_inh = 0.35 * torch.exp(-dist.square() / 80.0)
        kernel = (local_exc - broad_inh).view(1, 1, field_size)
        self.register_buffer("kernel", kernel)
        self.memory_gate = nn.Linear(field_size, field_size)

    def _field_input(self, state: Tensor) -> Tensor:
        """Compute circular recurrent field input.

        Parameters
        ----------
        state
            Field activity of shape ``(batch, field_size)``.

        Returns
        -------
        Tensor
            Recurrent input.
        """
        padded = torch.cat(
            (
                state[:, -self.kernel.shape[-1] // 2 :],
                state,
                state[:, : self.kernel.shape[-1] // 2],
            ),
            dim=-1,
        )
        conv = F.conv1d(padded.unsqueeze(1), self.kernel).squeeze(1)
        return conv[:, : state.shape[-1]]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run coupled dynamic fields from an input pattern.

        Parameters
        ----------
        x
            Input field of shape ``(batch, field_size)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Decision field and memory field.
        """
        memory = torch.tanh(self.memory_gate(x))
        decision = x.new_zeros(x.shape)
        for _ in range(self.steps):
            memory_drive = self._field_input(torch.relu(memory)) + x - 0.2
            decision_drive = (
                self._field_input(torch.relu(decision)) + 0.6 * torch.relu(memory) - 0.3
            )
            memory = memory + self.dt * (-memory + memory_drive)
            decision = decision + self.dt * (-decision + decision_drive)
        return decision, memory


def build() -> nn.Module:
    """Build a small dynamic-field architecture.

    Returns
    -------
    nn.Module
        Dynamic Field Theory module.
    """
    return DynamicFieldTheory()


def example_input() -> Tensor:
    """Return a float32 field input.

    Returns
    -------
    Tensor
        Input field of shape ``(2, 24)``.
    """
    base = torch.zeros(2, 24, dtype=torch.float32)
    base[:, 8:12] = 1.0
    return base
