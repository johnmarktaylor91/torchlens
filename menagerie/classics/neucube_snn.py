"""NeuCube, 2014, Kasabov.

Paper: "NeuCube: A spiking neural network architecture for mapping, learning and understanding..."
A simplified 3D reservoir of LIF neurons receives spike trains and projects to an
eSNN-style readout. Atlas placement, STDP, and evolving classifier growth are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuCubeSNN(nn.Module):
    """Simplified NeuCube 3D spiking reservoir."""

    def __init__(
        self, n_inputs: int = 1471, reservoir_size: int = 64, time_steps: int = 100
    ) -> None:
        """Initialize input projection, reservoir, and readout.

        Parameters
        ----------
        n_inputs:
            Number of input spike-train channels.
        reservoir_size:
            Number of simplified 3D reservoir neurons.
        time_steps:
            Expected number of input time steps.
        """
        super().__init__()
        self.time_steps = time_steps
        self.input_projection = nn.Linear(n_inputs, reservoir_size, bias=False)
        raw = torch.rand(reservoir_size, reservoir_size) * 0.05
        mask = (torch.rand(reservoir_size, reservoir_size) < 0.12).to(raw.dtype)
        recurrent = raw * mask
        recurrent.fill_diagonal_(0.0)
        self.register_buffer("recurrent", recurrent)
        self.readout = nn.Linear(reservoir_size, 8)

    def _surrogate_spike(self, voltage: Tensor) -> Tensor:
        """Compute smooth LIF spike activations.

        Parameters
        ----------
        voltage:
            Reservoir membrane voltage.

        Returns
        -------
        Tensor
            Smooth spike activation.
        """
        return torch.sigmoid(10.0 * (voltage - 0.5))

    def forward(self, spikes: Tensor) -> Tensor:
        """Run the simplified 3D reservoir over spike trains.

        Parameters
        ----------
        spikes:
            Input spike trains with shape ``(batch, n_inputs, time)``.

        Returns
        -------
        Tensor
            Readout logits from the mean reservoir spike state.
        """
        batch = spikes.shape[0]
        voltage = spikes.new_zeros(batch, self.recurrent.shape[0])
        reservoir_spikes = spikes.new_zeros(batch, self.recurrent.shape[0])
        spike_sum = spikes.new_zeros(batch, self.recurrent.shape[0])
        for step in range(spikes.shape[-1]):
            incoming = self.input_projection(spikes[:, :, step])
            recurrent_drive = reservoir_spikes @ self.recurrent
            voltage = 0.88 * voltage + incoming + recurrent_drive
            reservoir_spikes = self._surrogate_spike(voltage)
            spike_sum = spike_sum + reservoir_spikes
        return self.readout(spike_sum / self.time_steps)


def build() -> nn.Module:
    """Build a small simplified NeuCube SNN.

    Returns
    -------
    nn.Module
        Configured ``NeuCubeSNN`` instance.
    """
    return NeuCubeSNN()


def example_input() -> Tensor:
    """Create NeuCube spike-train input.

    Returns
    -------
    Tensor
        Example spike train with shape ``(1, 1471, 100)``.
    """
    return (torch.rand(1, 1471, 100) > 0.97).to(torch.float32)


MENAGERIE_ENTRIES = [("NeuCube (3D Evolving SNN)", "build", "example_input", "2014", "MB1")]
