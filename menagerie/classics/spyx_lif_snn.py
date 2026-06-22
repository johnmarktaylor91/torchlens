"""Spyx-style JAX spiking neural network as a PyTorch classic.

Paper: Heckel et al. 2024, "Spyx: A Library for Just-In-Time Compiled
Optimization of Spiking Neural Networks." The compact model keeps feed-forward
linear layers with leaky integrate-and-fire membrane dynamics and surrogate
spike generation over time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LIFLayer(nn.Module):
    """Leaky integrate-and-fire layer with reset-on-spike dynamics."""

    def __init__(self, in_dim: int, out_dim: int, decay: float = 0.85) -> None:
        """Initialize LIF layer.

        Parameters
        ----------
        in_dim:
            Input width.
        out_dim:
            Output neurons.
        decay:
            Membrane decay.
        """

        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.decay = decay
        self.threshold = nn.Parameter(torch.ones(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate LIF dynamics over time.

        Parameters
        ----------
        x:
            Input spike/features with shape ``(batch, time, features)``.

        Returns
        -------
        torch.Tensor
            Output spikes over time.
        """

        mem = x.new_zeros(x.shape[0], self.threshold.numel())
        spikes = []
        for step in range(x.shape[1]):
            mem = self.decay * mem + self.linear(x[:, step])
            spike = torch.sigmoid(10.0 * (mem - self.threshold))
            mem = mem * (1.0 - (spike > 0.5).float())
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class SpyxLIFSNN(nn.Module):
    """Compact two-layer Spyx LIF SNN classifier."""

    def __init__(self) -> None:
        """Initialize SNN layers."""

        super().__init__()
        self.lif1 = LIFLayer(12, 24)
        self.lif2 = LIFLayer(24, 16)
        self.readout = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a spike train.

        Parameters
        ----------
        x:
            Spike train tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        spikes = self.lif2(self.lif1(x))
        return self.readout(spikes.mean(dim=1))


def build() -> nn.Module:
    """Build compact Spyx LIF SNN.

    Returns
    -------
    nn.Module
        SNN model.
    """

    return SpyxLIFSNN()


def example_input() -> torch.Tensor:
    """Create a small spike train.

    Returns
    -------
    torch.Tensor
        Spike/features tensor.
    """

    return (torch.rand(1, 6, 12) > 0.6).float()


MENAGERIE_ENTRIES = [("spyx_lif_snn", "build", "example_input", "2024", "E7")]
