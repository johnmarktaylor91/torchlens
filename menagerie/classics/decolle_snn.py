"""DECOLLE, 2020, Kaiser and Neftci.

Paper: "Synaptic Plasticity Dynamics for Deep Continuous Local Learning."
This forward-only module uses LIF layers, surrogate spikes, and frozen random
readouts per layer; layer-local loss construction and online updates are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DECOLLESNN(nn.Module):
    """Small LIF stack with DECOLLE-style layer-local readouts."""

    def __init__(self, n_inputs: int = 28 * 28, hidden_sizes: tuple[int, ...] = (48, 32)) -> None:
        """Initialize feedforward LIF layers and random local readouts.

        Parameters
        ----------
        n_inputs:
            Flattened input dimensionality per time step.
        hidden_sizes:
            Hidden spiking layer sizes.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        prev = n_inputs
        for hidden in hidden_sizes:
            self.layers.append(nn.Linear(prev, hidden))
            prev = hidden
        for idx, hidden in enumerate(hidden_sizes):
            self.register_buffer(f"readout_{idx}", torch.randn(hidden, 10) * 0.1)

    def _surrogate_spike(self, voltage: Tensor) -> Tensor:
        """Compute differentiable surrogate spikes.

        Parameters
        ----------
        voltage:
            Membrane voltage.

        Returns
        -------
        Tensor
            Smooth spike activations.
        """
        return torch.sigmoid(12.0 * (voltage - 0.6))

    def forward(self, x: Tensor) -> Tensor:
        """Run LIF dynamics and return stacked local readouts.

        Parameters
        ----------
        x:
            Spike-frame input with shape ``(batch, time, 1, 28, 28)``.

        Returns
        -------
        Tensor
            Layer-local logits with shape ``(batch, n_layers, 10)``.
        """
        batch, time = x.shape[0], x.shape[1]
        voltages = [x.new_zeros(batch, layer.out_features) for layer in self.layers]
        readouts = [x.new_zeros(batch, 10) for _ in self.layers]
        for step in range(time):
            activity = x[:, step].flatten(1)
            for idx, layer in enumerate(self.layers):
                voltages[idx] = 0.82 * voltages[idx] + layer(activity)
                spikes = self._surrogate_spike(voltages[idx])
                readout = getattr(self, f"readout_{idx}")
                readouts[idx] = readouts[idx] + spikes @ readout
                activity = spikes
        return torch.stack([readout / time for readout in readouts], dim=1)


def build() -> nn.Module:
    """Build a small DECOLLE-style spiking network.

    Returns
    -------
    nn.Module
        Configured ``DECOLLESNN`` instance.
    """
    return DECOLLESNN()


def example_input() -> Tensor:
    """Create a sparse spike-frame input.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 100, 1, 28, 28)``.
    """
    return (torch.rand(1, 100, 1, 28, 28) > 0.93).to(torch.float32)


MENAGERIE_ENTRIES = [
    ("DECOLLE (Deep Continuous Local Learning)", "build", "example_input", "2020", "MB1")
]
