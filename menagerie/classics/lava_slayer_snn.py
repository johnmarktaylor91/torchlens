"""Lava-DL SLAYER compact spiking blocks.

Paper/Library: Lava-DL SLAYER, Intel Neuromorphic Computing Lab, 2022+.

SLAYER traces deep event-based networks with synapse filters, axon delays, and
learnable leaky/CUBA neuron dynamics.  This file registers the dense SLAYER,
CUBA block, and CUBA dense-v2 targets with compact unrolled spike dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CUBALIFLayer(nn.Module):
    """Current-based leaky integrate-and-fire layer with surrogate spikes."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize dense synapse and neuron constants."""

        super().__init__()
        self.synapse = nn.Linear(in_features, out_features)
        self.current_decay = nn.Parameter(torch.tensor(0.75))
        self.voltage_decay = nn.Parameter(torch.tensor(0.65))
        self.threshold = nn.Parameter(torch.ones(out_features))

    def forward(self, spikes: Tensor) -> Tensor:
        """Unroll CUBA dynamics over time."""

        current = torch.zeros(spikes.shape[0], self.threshold.numel(), device=spikes.device)
        voltage = torch.zeros_like(current)
        outs = []
        for step in range(spikes.shape[1]):
            current = torch.sigmoid(self.current_decay) * current + self.synapse(spikes[:, step])
            voltage = torch.sigmoid(self.voltage_decay) * voltage + current
            out = torch.sigmoid(10.0 * (voltage - self.threshold))
            voltage = voltage * (1 - out.detach())
            outs.append(out)
        return torch.stack(outs, dim=1)


class SlayerDenseSNN(nn.Module):
    """Compact SLAYER dense SNN with delayed spike readout."""

    def __init__(self, cuba: bool = False) -> None:
        """Initialize two spiking layers and a rate head."""

        super().__init__()
        self.cuba = cuba
        self.layer1 = CUBALIFLayer(10, 24)
        self.layer2 = CUBALIFLayer(24, 16)
        self.delay = nn.Parameter(torch.randn(1, 1, 16) * 0.02)
        self.head = nn.Linear(16, 4)

    def forward(self, spikes: Tensor) -> Tensor:
        """Classify event sequences with SLAYER-style temporal filtering."""

        x = self.layer1(spikes)
        x = self.layer2(x)
        if self.cuba:
            x = x + torch.tanh(self.delay)
        return self.head(x.mean(dim=1))


def build_dense() -> nn.Module:
    """Build compact dense SLAYER SNN."""

    return SlayerDenseSNN(cuba=False).eval()


def build_cuba() -> nn.Module:
    """Build compact SLAYER CUBA block."""

    return SlayerDenseSNN(cuba=True).eval()


def example_input() -> Tensor:
    """Return a small binary event sequence."""

    return (torch.rand(1, 8, 10) > 0.6).float()


MENAGERIE_ENTRIES = [
    ("lava_slayer_dense_snn", "build_dense", "example_input", "2022", "SNN"),
    ("lava_dl_slayer_cuba_block", "build_cuba", "example_input", "2022", "SNN"),
    ("lava_slayer_cuba_dense_v2", "build_cuba", "example_input", "2022", "SNN"),
]
