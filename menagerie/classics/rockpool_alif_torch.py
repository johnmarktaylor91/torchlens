"""Rockpool aLIFTorch: adaptive leaky integrate-and-fire neurons.

Source: Rockpool ``rockpool.nn.modules.torch.ahp_lif_torch.aLIFTorch`` docs
describe a leaky integrate-and-fire spiking neuron with adaptive
hyperpolarisation feedback using a Torch backend.

This compact reconstruction keeps the dependency-gated model's characteristic
state update: input current is filtered into membrane voltage, emitted spikes
reset the membrane, and an after-hyperpolarisation/adaptation state raises the
effective threshold after recent spikes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SurrogateSpike(torch.autograd.Function):
    """Straight-through threshold spike for traceable random-init inference."""

    @staticmethod
    def forward(ctx: object, voltage: Tensor, threshold: Tensor) -> Tensor:
        """Emit binary spikes from threshold crossings.

        Parameters
        ----------
        ctx:
            Autograd context, unused for inference-only tracing.
        voltage:
            Membrane voltage tensor.
        threshold:
            Adaptive threshold tensor.

        Returns
        -------
        Tensor
            Float spike tensor with ones where ``voltage >= threshold``.
        """
        return (voltage >= threshold).to(voltage.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        """Pass gradients straight through the discontinuity.

        Parameters
        ----------
        ctx:
            Autograd context, unused.
        grad_output:
            Upstream gradient.

        Returns
        -------
        tuple[Tensor, None]
            Gradient for voltage and no threshold gradient.
        """
        return grad_output, None


class AdaptiveLIFLayer(nn.Module):
    """Adaptive LIF recurrent layer with after-hyperpolarisation feedback."""

    def __init__(self, input_size: int = 6, hidden_size: int = 8) -> None:
        """Initialize input projection and learnable neuron constants.

        Parameters
        ----------
        input_size:
            Number of input channels per time step.
        hidden_size:
            Number of adaptive spiking neurons.
        """
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_mem = nn.Parameter(torch.full((hidden_size,), 0.72))
        self.tau_ahp = nn.Parameter(torch.full((hidden_size,), 0.90))
        self.base_threshold = nn.Parameter(torch.ones(hidden_size))
        self.ahp_gain = nn.Parameter(torch.full((hidden_size,), 0.35))
        self.reset = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """Run the adaptive spiking recurrence over time.

        Parameters
        ----------
        x:
            Input sequence with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Spike sequence with shape ``(batch, time, hidden_size)``.
        """
        batch, time, _ = x.shape
        voltage = x.new_zeros(batch, self.base_threshold.numel())
        ahp = x.new_zeros(batch, self.base_threshold.numel())
        spike = x.new_zeros(batch, self.base_threshold.numel())
        outputs: list[Tensor] = []
        mem_decay = self.tau_mem.sigmoid()
        ahp_decay = self.tau_ahp.sigmoid()
        for step in range(time):
            current = self.input(x[:, step]) + self.recurrent(spike)
            threshold = self.base_threshold + self.ahp_gain.abs() * ahp
            voltage = mem_decay * voltage + (1.0 - mem_decay) * current
            spike = SurrogateSpike.apply(voltage, threshold)
            voltage = voltage * (1.0 - spike) + self.reset * spike
            ahp = ahp_decay * ahp + spike
            outputs.append(spike)
        return torch.stack(outputs, dim=1)


class RockpoolALIFNet(nn.Module):
    """Small aLIFTorch-style spiking classifier."""

    def __init__(self) -> None:
        """Initialize adaptive spiking layer and readout."""
        super().__init__()
        self.alif = AdaptiveLIFLayer()
        self.readout = nn.Linear(8, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a short event sequence.

        Parameters
        ----------
        x:
            Event feature sequence with shape ``(batch, time, 6)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(batch, 3)``.
        """
        spikes = self.alif(x)
        return self.readout(spikes.mean(dim=1))


def build() -> nn.Module:
    """Build the compact Rockpool aLIFTorch reconstruction.

    Returns
    -------
    nn.Module
        Random-initialized adaptive LIF network.
    """
    return RockpoolALIFNet()


def example_input() -> Tensor:
    """Return a short event-feature sequence.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 10, 6)``.
    """
    return torch.randn(1, 10, 6)


MENAGERIE_ENTRIES = [
    ("rockpool_alif_torch", "build", "example_input", "2019", "E6"),
]
