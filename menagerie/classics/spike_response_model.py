"""Spike Response Model, 1995, Gerstner.

Paper: Gerstner 1995, "Time structure of the activity in neural network models."
Causal postsynaptic kernels and refractory response kernels determine membrane
potential; this traceable version uses sigmoid surrogate spikes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Spike Response Model (SRM)", "build", "example_input", "1995", "CF")]


class SpikeResponseModel(nn.Module):
    """Convolutional SRM neuron layer with surrogate spikes."""

    def __init__(self, n_in: int = 4, n_out: int = 3, kernel_size: int = 5) -> None:
        """Initialize synaptic and refractory kernels.

        Parameters
        ----------
        n_in
            Number of input channels.
        n_out
            Number of output neurons.
        kernel_size
            Length of causal response kernels.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out) * 0.25)
        t = torch.arange(kernel_size, dtype=torch.float32)
        self.register_buffer("kappa", (t + 1.0) * torch.exp(-t / 2.0))
        self.register_buffer("eta", -torch.exp(-t / 1.5))

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute SRM membrane and smooth spike trajectories.

        Parameters
        ----------
        inputs
            Input current/spike tensor of shape ``(batch, time, n_in)``.

        Returns
        -------
        Tensor
            Smooth output spike trajectory of shape ``(batch, time, n_out)``.
        """
        syn = inputs @ self.weight
        bsz, time, n_out = syn.shape
        kernel = self.kappa.view(1, 1, -1).repeat(n_out, 1, 1)
        flat = syn.permute(0, 2, 1)
        psp = F.conv1d(F.pad(flat, (self.kappa.shape[0] - 1, 0)), kernel, groups=n_out)
        refractory = syn.new_zeros(bsz, n_out, time)
        spikes: list[Tensor] = []
        for step in range(time):
            voltage = psp[:, :, step] + refractory[:, :, step]
            spike = torch.sigmoid(10.0 * (voltage - 0.5))
            spikes.append(spike)
            if step + 1 < time:
                remaining = min(self.eta.shape[0], time - step - 1)
                add = spike.unsqueeze(-1) * self.eta[:remaining].view(1, 1, -1)
                updated = refractory[:, :, step + 1 : step + 1 + remaining] + add
                refractory = torch.cat(
                    (
                        refractory[:, :, : step + 1],
                        updated,
                        refractory[:, :, step + 1 + remaining :],
                    ),
                    dim=-1,
                )
        return torch.stack(spikes, dim=1)


def build() -> nn.Module:
    """Build a compact SRM layer.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return SpikeResponseModel()


def example_input() -> Tensor:
    """Return input current/spike trains.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 7, 4)``.
    """
    return torch.rand(2, 7, 4)
