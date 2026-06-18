"""FORCE-trained chaotic reservoir, 2009, Sussillo and Abbott.

Paper: "Generating coherent patterns of activity from chaotic neural networks."
This is the fixed recurrent reservoir and linear readout substrate; RLS/FORCE weight
updates are not part of the forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FORCETrainedChaoticReservoir(nn.Module):
    """Chaotic reservoir with fixed recurrent and feedback weights."""

    def __init__(
        self, n_reservoir: int = 32, n_out: int = 3, dt: float = 0.1, gain: float = 1.5
    ) -> None:
        """Initialize reservoir buffers and trainable readout.

        Parameters
        ----------
        n_reservoir
            Number of recurrent reservoir units.
        n_out
            Number of output channels.
        dt
            Euler integration step size.
        gain
            Recurrent gain controlling chaotic tendency.
        """
        super().__init__()
        recurrent = torch.randn(n_reservoir, n_reservoir) * (gain / n_reservoir**0.5)
        feedback = torch.randn(n_reservoir, n_out) * 0.2
        self.register_buffer("recurrent", recurrent)
        self.register_buffer("feedback", feedback)
        self.readout = nn.Linear(n_reservoir, n_out, bias=False)
        self.dt = dt

    def forward(self, initial_state: Tensor) -> Tensor:
        """Unroll a few reservoir steps from an initial reservoir state.

        Parameters
        ----------
        initial_state
            Initial state tensor whose first 32 features seed the reservoir.

        Returns
        -------
        Tensor
            Readout sequence with shape ``(batch, time, n_out)``.
        """
        state = initial_state[:, : self.recurrent.shape[0]]
        z = self.readout(torch.tanh(state))
        outputs: list[Tensor] = []
        for _step in range(4):
            rates = torch.tanh(state)
            drive = rates @ self.recurrent.T + z @ self.feedback.T
            state = (1.0 - self.dt) * state + self.dt * drive
            z = self.readout(torch.tanh(state))
            outputs.append(z)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small FORCE reservoir.

    Returns
    -------
    nn.Module
        Configured ``FORCETrainedChaoticReservoir`` instance.
    """
    return FORCETrainedChaoticReservoir()


def example_input() -> Tensor:
    """Create an example reservoir seed vector.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 200)``.
    """
    return torch.randn(1, 200)


MENAGERIE_ENTRIES = [("FORCE-Trained Chaotic Reservoir", "build", "example_input", "2009", "DD")]
