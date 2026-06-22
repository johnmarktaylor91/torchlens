"""sinabs-exodus LIF compact faithful reconstruction.

EXODUS provides vectorized CUDA kernels for sinabs spiking neuron recurrences.
The traceable base-environment version here keeps the dependency-gated model's
semantics: a leaky integrate-and-fire state unrolled over time with reset by
subtraction and a surrogate spike nonlinearity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpikeFn(torch.autograd.Function):
    """Surrogate threshold spike."""

    @staticmethod
    def forward(ctx: object, voltage: Tensor) -> Tensor:
        """Emit spikes for positive threshold crossings.

        Parameters
        ----------
        ctx:
            Autograd context.
        voltage:
            Voltage above threshold.

        Returns
        -------
        Tensor
            Binary spikes.
        """
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> Tensor:
        """Pass gradients straight through.

        Parameters
        ----------
        ctx:
            Autograd context.
        grad_output:
            Upstream gradient.

        Returns
        -------
        Tensor
            Surrogate gradient.
        """
        return grad_output


class ExodusLIFLayer(nn.Module):
    """Vectorized EXODUS-style LIF layer."""

    def __init__(self, features: int = 16, decay: float = 0.8, threshold: float = 1.0) -> None:
        """Initialize neuron constants and input projection.

        Parameters
        ----------
        features:
            Feature count.
        decay:
            Membrane decay factor.
        threshold:
            Spike threshold.
        """
        super().__init__()
        self.proj = nn.Linear(features, features)
        self.decay = decay
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        """Run the LIF recurrence over time.

        Parameters
        ----------
        x:
            Input sequence with shape ``(batch, time, features)``.

        Returns
        -------
        Tensor
            Spike sequence.
        """
        batch, time, features = x.shape
        voltage = x.new_zeros(batch, features)
        outs: list[Tensor] = []
        current = self.proj(x)
        for step in range(time):
            voltage = self.decay * voltage + current[:, step]
            spike = SpikeFn.apply(voltage - self.threshold)
            voltage = voltage - spike * self.threshold
            outs.append(spike)
        return torch.stack(outs, dim=1)


class ExodusLIFNet(nn.Module):
    """Small sinabs-exodus LIF classifier."""

    def __init__(self) -> None:
        """Initialize LIF layer and readout."""
        super().__init__()
        self.lif = ExodusLIFLayer()
        self.readout = nn.Linear(16, 4)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a short temporal feature sequence.

        Parameters
        ----------
        x:
            Input sequence.

        Returns
        -------
        Tensor
            Class logits.
        """
        spikes = self.lif(x)
        return self.readout(spikes.mean(dim=1))


def build() -> nn.Module:
    """Build compact random-init sinabs-exodus LIF model.

    Returns
    -------
    nn.Module
        Compact EXODUS LIF model.
    """
    return ExodusLIFNet()


def example_input() -> Tensor:
    """Return a short feature sequence.

    Returns
    -------
    Tensor
        Input tensor.
    """
    return torch.randn(1, 8, 16)


MENAGERIE_ENTRIES = [("sinabs_exodus_lif", "build", "example_input", "2023", "E7")]
