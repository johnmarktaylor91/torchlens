"""Polychronization network, 2006, Izhikevich.

Paper: Izhikevich 2006, "Polychronization: Computation with spikes." Izhikevich
neurons with heterogeneous axonal delays create reproducible groups from
delayed co-arrivals. This simplified traceable module uses dense delay buffers
and smooth surrogate spikes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Polychronization network (delay-coupled Izhikevich)", "build", "example_input", "2006", "CF")
]


class PolychronizationNet(nn.Module):
    """Delay-coupled Izhikevich network with surrogate spikes."""

    def __init__(self, n_neurons: int = 5, max_delay: int = 3, dt: float = 0.2) -> None:
        """Initialize coupling weights and heterogeneous delays.

        Parameters
        ----------
        n_neurons
            Number of neurons.
        max_delay
            Maximum delay bucket.
        dt
            Euler integration step size.
        """
        super().__init__()
        self.max_delay = max_delay
        self.dt = dt
        self.weight = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.2)
        delays = torch.arange(n_neurons).view(-1, 1) + torch.arange(n_neurons).view(1, -1)
        self.register_buffer("delays", delays.remainder(max_delay) + 1)

    def forward(self, input_spikes: Tensor) -> Tensor:
        """Roll out delayed Izhikevich dynamics.

        Parameters
        ----------
        input_spikes
            External spike/current sequence of shape ``(batch, time, n_neurons)``.

        Returns
        -------
        Tensor
            Smooth spike trajectory.
        """
        batch, time, n_neurons = input_spikes.shape
        v = input_spikes.new_full((batch, n_neurons), -0.65)
        u = input_spikes.new_full((batch, n_neurons), -0.13)
        delay_lines = [input_spikes.new_zeros(batch, n_neurons) for _ in range(self.max_delay + 1)]
        outputs: list[Tensor] = []
        for step in range(time):
            delayed_drive = input_spikes.new_zeros(batch, n_neurons)
            for delay in range(1, self.max_delay + 1):
                mask = (self.delays == delay).to(input_spikes.dtype)
                delayed_drive = delayed_drive + delay_lines[delay] @ (self.weight * mask)
            drive = input_spikes[:, step] + delayed_drive
            v = v + self.dt * (0.04 * v.square() + 5.0 * v + 1.4 - u + drive)
            u = u + self.dt * (0.02 * (0.2 * v - u))
            spike = torch.sigmoid(30.0 * (v - 0.3))
            v = v * (1.0 - spike) + (-0.65) * spike
            u = u + 0.08 * spike
            delay_lines = [spike] + delay_lines[:-1]
            outputs.append(spike)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact polychronization network.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return PolychronizationNet()


def example_input() -> Tensor:
    """Return external spike inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 8, 5)``.
    """
    x = torch.zeros(2, 8, 5)
    x[:, 0, :2] = 1.0
    x[:, 3, 2:] = 0.5
    return x
