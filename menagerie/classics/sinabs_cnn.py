"""sinabs ``from_model`` converted spiking CNN (IF-neuron SNN).

sinabs is SynSense's open-source PyTorch library for spiking convolutional
networks.  Its signature workflow is ANN-to-SNN *conversion*: take an ordinary
CNN trained (or initialised) with ``nn.ReLU`` activations and call
``sinabs.from_model(...)`` to swap every ``ReLU`` for an Integrate-and-Fire (IF)
spiking-neuron layer.  The resulting stateful network is then run over ``T``
discrete timesteps, with the (rate-coded) input replayed each step; the rectified
real-valued ANN activation is reproduced in expectation by the IF neuron's spike
rate.

Source: https://github.com/synsense/sinabs  (sinabs.from_model, sinabs.layers.IAF)
Tutorial: https://sinabs.readthedocs.io/en/v2.0.0/tutorials/weight_transfer_mnist.html

This faithful reimplementation reproduces the *converted* topology exactly:
each ``nn.ReLU`` of a small Conv-ReLU-Pool CNN is replaced by an ``IFSpike``
Integrate-and-Fire layer, the model is unrolled over ``T`` timesteps with the
membrane potentials carried across steps, and the per-step spike trains are
accumulated.  The summed output spikes of the final layer are the rate-coded
class logits.  ``T`` is kept small so the time-unrolled computation graph stays
compact for Graphviz rendering; the dynamics are identical at any horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside spike in the forward pass with a fast-sigmoid surrogate grad."""

    @staticmethod
    def forward(ctx, v: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(v)
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (v,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * v.abs()) ** 2
        return grad_output * sg


def _spike(v_over_thresh: torch.Tensor) -> torch.Tensor:
    return _SurrogateSpike.apply(v_over_thresh)


class IFSpike(nn.Module):
    """Integrate-and-Fire spiking neuron (sinabs ``IAF``-style, single step).

    A *non-leaky* integrate-and-fire neuron.  Holds a membrane potential ``v``
    as internal state; each call integrates the layer input, fires a spike
    where ``v >= thresh`` and subtracts the threshold (soft reset, the sinabs
    ``reset_fn`` default ``membrane_subtract``).  ``reset_state`` clears the
    potential at the start of each multi-timestep forward pass.
    """

    def __init__(self, thresh: float = 1.0) -> None:
        super().__init__()
        self.thresh = thresh
        self.v: torch.Tensor | None = None

    def reset_state(self) -> None:
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        v = self.v + x
        s = _spike(v - self.thresh)
        # Soft reset: subtract threshold for each emitted spike (membrane_subtract).
        v = v - s * self.thresh
        self.v = v
        return s


class SinabsConvSNN(nn.Module):
    """A small Conv-IF-Pool CNN with every ReLU replaced by an IF neuron.

    Mirrors the canonical sinabs ``from_model`` conversion of a standard CNN:
    the ANN graph (Conv2d / AvgPool2d / Linear / Flatten) is preserved verbatim
    and only the ``nn.ReLU`` activations become ``IFSpike`` Integrate-and-Fire
    layers.  The network is run for ``T`` timesteps with the rate-coded input
    replayed each step; per-step output spikes are summed into class logits.
    """

    def __init__(self, in_ch: int = 3, n_classes: int = 10, T: int = 3) -> None:
        super().__init__()
        self.T = T
        # Standard small CNN backbone (the "ANN" being converted).
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1, bias=False)
        self.if1 = IFSpike(thresh=1.0)
        self.pool1 = nn.AvgPool2d(2)  # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.if2 = IFSpike(thresh=1.0)
        self.pool2 = nn.AvgPool2d(2)  # 16 -> 8
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.if3 = IFSpike(thresh=1.0)
        self.pool3 = nn.AvgPool2d(2)  # 8 -> 4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128, bias=False)
        self.if4 = IFSpike(thresh=1.0)
        self.fc2 = nn.Linear(128, n_classes, bias=False)

    def _reset(self) -> None:
        for m in (self.if1, self.if2, self.if3, self.if4):
            m.reset_state()

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.if1(self.conv1(x)))
        x = self.pool2(self.if2(self.conv2(x)))
        x = self.pool3(self.if3(self.conv3(x)))
        x = self.flatten(x)
        x = self.if4(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the converted SNN over ``T`` timesteps.

        ``x`` is a static image ``(batch, C, H, W)``; it is rate-coded by being
        replayed at every timestep.  The summed output of the final linear layer
        across timesteps is returned as class logits ``(batch, n_classes)``.
        """
        self._reset()
        out = None
        for _ in range(self.T):
            step_out = self._step(x)
            out = step_out if out is None else out + step_out
        return out / self.T


def build() -> nn.Module:
    """Build a sinabs ``from_model`` converted spiking CNN (IF-neuron SNN)."""
    return SinabsConvSNN(in_ch=3, n_classes=10, T=3)


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 32, 32)`` (replayed over ``T`` steps internally)."""
    return torch.rand(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "sinabs from_model converted spiking CNN (IF-neuron SNN)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
