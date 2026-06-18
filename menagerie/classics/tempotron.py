"""Tempotron, 2006.

Gutig and Sompolinsky's Tempotron is a single spiking-neuron classifier whose decision
is whether a postsynaptic-potential trace crosses threshold.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Tempotron(nn.Module):
    """Single-neuron spike-timing classifier."""

    def __init__(
        self, n_synapses: int = 8, time: int = 25, tau: float = 5.0, threshold: float = 1.0
    ) -> None:
        """Initialize synaptic weights and PSP kernel.

        Parameters
        ----------
        n_synapses
            Number of input spike trains.
        time
            Number of time bins.
        tau
            PSP decay constant.
        threshold
            Spike decision threshold.
        """
        super().__init__()
        self.threshold = threshold
        self.weights = nn.Parameter(torch.rand(n_synapses) * 0.3)
        t = torch.arange(time, dtype=torch.float32)
        kernel = (t / tau) * torch.exp(1.0 - t / tau)
        self.register_buffer("kernel", kernel.view(1, 1, time))

    def forward(self, spikes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute membrane trace and spike/no-spike decision.

        Parameters
        ----------
        spikes
            Binary spike trains of shape ``(batch, n_synapses, time)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Membrane trace, peak voltage, and binary spike decision.
        """
        weighted = spikes * self.weights.view(1, -1, 1)
        summed = weighted.sum(dim=1, keepdim=True)
        padded = F.pad(summed, (self.kernel.shape[-1] - 1, 0))
        membrane = F.conv1d(padded, self.kernel.flip(-1)).squeeze(1)
        peak = membrane.max(dim=-1).values
        decision = (peak > self.threshold).to(spikes.dtype)
        return membrane, peak, decision


def build() -> nn.Module:
    """Build a Tempotron classifier.

    Returns
    -------
    nn.Module
        Random Tempotron module.
    """
    return Tempotron()


def example_input() -> Tensor:
    """Return float32 binary spike trains.

    Returns
    -------
    Tensor
        Spike tensor of shape ``(2, 8, 25)``.
    """
    spikes = torch.zeros(2, 8, 25, dtype=torch.float32)
    spikes[:, ::2, 4::7] = 1.0
    spikes[:, 1::2, 9::8] = 1.0
    return spikes
