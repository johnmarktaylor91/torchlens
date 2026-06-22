"""Shared spiking-neural-network primitives for the menagerie SNN "classics".

This module is intentionally prefixed with an underscore so it is NOT discovered
as a menagerie entry (it declares no ``MENAGERIE_ENTRIES``).  It provides the
single LIF (Leaky-Integrate-and-Fire) neuron + surrogate-gradient spike function
that every spiking architecture in this batch reuses, so that each model file
captures the *distinctive* spiking mechanism faithfully without re-deriving the
neuron dynamics or depending on spikingjelly / snntorch / norse (none of which
are installed).

Standard LIF dynamics (discrete time, leading TIME axis):

    v[t] = beta * v[t-1] + input[t]
    s[t] = Heaviside(v[t] - threshold)          # surrogate gradient on backward
    v[t] = v[t] - threshold * s[t]              # soft reset (subtract threshold)

The Heaviside spike is exact in the forward pass; the backward pass substitutes a
fast-sigmoid / triangular surrogate so the unrolled graph stays differentiable.

Everything here is pure torch.  No external SNN library.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside spike in the forward pass, fast-sigmoid surrogate gradient.

    forward:  s = (v >= 0)              (v is the *over-threshold* membrane)
    backward: ds/dv ~= 1 / (1 + alpha*|v|)^2   (fast-sigmoid surrogate)
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * v.abs()) ** 2
        return grad_output * sg


def spike_fn(v_over_thresh: torch.Tensor) -> torch.Tensor:
    """Emit a spike where the (already threshold-subtracted) membrane is >= 0."""
    return _SurrogateSpike.apply(v_over_thresh)


class LIFNeuron(nn.Module):
    """Leaky-Integrate-and-Fire neuron unrolled over a leading TIME axis.

    Membrane decay ``beta``, firing ``threshold`` and a soft (subtractive) reset.
    Accepts an input current train shaped ``(T, ...)`` (the first dimension is
    time; any trailing shape is preserved) and returns the spike train of the
    same shape.  When ``T == 1`` it behaves as a single-step integrate-and-fire.

    Parameters
    ----------
    beta:
        Membrane leak / decay factor in ``[0, 1)``.
    threshold:
        Firing threshold on the membrane potential.
    reset:
        ``"sub"`` subtracts the threshold from the membrane on a spike (the
        standard "soft" reset); ``"zero"`` clamps the membrane back to 0.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset: str = "sub",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.reset = reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = x.shape[0]
        v = torch.zeros_like(x[0])
        spikes = []
        for t in range(time):
            v = self.beta * v + x[t]
            s = spike_fn(v - self.threshold)
            if self.reset == "zero":
                v = v * (1.0 - s)
            else:  # subtractive (soft) reset
                v = v - self.threshold * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class SpikingActivation(nn.Module):
    """Single-step IF spiking non-linearity that can stand in for ReLU.

    Operates element-wise on a tensor WITHOUT a leading time axis: it integrates
    the input once and fires.  This is the drop-in "replace ReLU with a spike"
    operator used by conversion-style SNNs (SpikeZIP-TF, Spiking-UNet,
    Spiking-YOLO) where the time dimension is handled outside the activation
    (e.g. by repeating the forward pass over T steps at the top level).

    With ``levels > 1`` it emits a *quantised / multi-level* spike (a staircase
    of unit steps up to ``levels``), matching the multi-level spiking neurons
    used by quantisation-equivalent SNNs.
    """

    def __init__(self, threshold: float = 1.0, levels: int = 1) -> None:
        super().__init__()
        self.threshold = threshold
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.levels <= 1:
            # binary IF spike scaled by threshold (rate-equivalent of ReLU)
            return spike_fn(x - self.threshold) * self.threshold
        # multi-level / quantised spike: sum of unit steps, capped at `levels`
        out = torch.zeros_like(x)
        for lvl in range(1, self.levels + 1):
            out = out + spike_fn(x - lvl * self.threshold)
        out = torch.clamp(out, max=float(self.levels)) * self.threshold
        return out


def lif_over_time(module: nn.Module, x: torch.Tensor, time: int) -> torch.Tensor:
    """Run a (static, non-temporal) ``module`` over ``time`` repeated steps.

    Helper for conversion-style SNNs whose backbone is an ordinary CNN with
    spiking activations: the same input is presented for ``time`` steps and the
    per-step outputs are stacked along a new leading time axis.  Keeps the
    distinctive "run the net for T spiking timesteps" structure visible to the
    tracer while keeping T small.
    """
    outs = [module(x) for _ in range(time)]
    return torch.stack(outs, dim=0)
