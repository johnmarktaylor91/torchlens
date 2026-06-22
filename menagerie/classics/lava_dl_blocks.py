"""Lava-DL SLAYER ALIF and RF block compact reconstructions.

Lava-DL SLAYER provides trainable spiking neuron blocks including adaptive LIF
and resonate-and-fire variants. These random-init torch modules preserve the
time-unrolled recurrent state updates without requiring the Lava dependency.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpikeFn(torch.autograd.Function):
    """Straight-through threshold spike."""

    @staticmethod
    def forward(ctx: object, x: Tensor) -> Tensor:
        """Emit binary spikes.

        Parameters
        ----------
        ctx:
            Autograd context.
        x:
            Threshold-centered voltage.

        Returns
        -------
        Tensor
            Binary spike tensor.
        """
        return (x >= 0).to(x.dtype)

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


class LavaALIFBlock(nn.Module):
    """Adaptive LIF block with decaying threshold adaptation."""

    def __init__(self, features: int = 12) -> None:
        """Initialize projection and readout.

        Parameters
        ----------
        features:
            Feature count.
        """
        super().__init__()
        self.input = nn.Linear(features, features)
        self.readout = nn.Linear(features, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Run adaptive LIF dynamics.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, time, features)``.

        Returns
        -------
        Tensor
            Readout logits.
        """
        batch, time, features = x.shape
        v = x.new_zeros(batch, features)
        th = x.new_ones(batch, features)
        outs: list[Tensor] = []
        current = self.input(x)
        for step in range(time):
            v = 0.85 * v + current[:, step]
            spike = SpikeFn.apply(v - th)
            v = v - spike * th
            th = 0.9 * th + 0.25 * spike
            outs.append(spike)
        return self.readout(torch.stack(outs, dim=1).mean(dim=1))


class LavaRFBlock(nn.Module):
    """Resonate-and-fire block with damped oscillator state."""

    def __init__(self, features: int = 12) -> None:
        """Initialize projection and readout.

        Parameters
        ----------
        features:
            Feature count.
        """
        super().__init__()
        self.input = nn.Linear(features, features)
        self.readout = nn.Linear(features, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Run resonate-and-fire dynamics.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, time, features)``.

        Returns
        -------
        Tensor
            Readout logits.
        """
        batch, time, features = x.shape
        real = x.new_zeros(batch, features)
        imag = x.new_zeros(batch, features)
        outs: list[Tensor] = []
        current = self.input(x)
        for step in range(time):
            new_real = 0.88 * real - 0.35 * imag + current[:, step]
            imag = 0.35 * real + 0.88 * imag
            real = new_real
            spike = SpikeFn.apply(real - 1.0)
            real = real - spike
            outs.append(spike)
        return self.readout(torch.stack(outs, dim=1).mean(dim=1))


def build_alif() -> nn.Module:
    """Build compact Lava-DL ALIF block.

    Returns
    -------
    nn.Module
        ALIF block.
    """
    return LavaALIFBlock()


def build_rf() -> nn.Module:
    """Build compact Lava-DL resonate-and-fire block.

    Returns
    -------
    nn.Module
        RF block.
    """
    return LavaRFBlock()


def example_input() -> Tensor:
    """Return a short temporal feature sequence.

    Returns
    -------
    Tensor
        Input tensor.
    """
    return torch.randn(1, 8, 12)


MENAGERIE_ENTRIES = [
    ("lava_dl_alif_block", "build_alif", "example_input", "2021", "E7"),
    ("lava_dl_rf_block", "build_rf", "example_input", "2024", "E7"),
]
