"""Intel Lava neuromorphic neuron processes: Loihi-style spiking blocks (traceable).

Source: https://github.com/lava-nc/lava  (``lava.proc.lif``, ``lava.proc.cuba``,
``lava.proc.sdn``, ``lava.proc.rf``) and Intel's Loihi neuromorphic chip dynamics.

Lava is Intel's neuromorphic framework; its ``Process`` library defines the spiking
neuron models that compile to Loihi.  Their defining trait (vs the BindsNET node
family already in the menagerie, which uses continuous exponential membrane decay) is
the **Loihi integer-decay update form**: state decays by a fixed fraction per step,
``u[t] = u[t-1] * (1 - du) + a_in`` (synaptic current) and
``v[t] = v[t-1] * (1 - dv) + u[t] + bias`` (membrane), with a threshold-and-reset
spike.  This module reimplements five distinctive Lava neuron blocks as faithful,
traceable ``nn.Module``s integrating the published Lava update equations over a
leading time dimension, with a surrogate-gradient spike so the graph is differentiable:

  - **CUBA-LIF**  : the Loihi default current-based LIF (two states u, v; du/dv decay).
  - **LIF**       : single-state leaky integrate-and-fire (Lava ``proc.lif.LIF``).
  - **TernaryLIF**: LIF emitting ternary spikes (+1 / 0 / -1) at an upper / lower
                    threshold pair (Lava ``proc.lif.TernaryLIF``).
  - **AdaptiveLIF**: LIF with a spike-incremented, decaying adaptive threshold (ALIF).
  - **Sigma-Delta**: graded sigma-delta neuron -- accumulate (sigma), emit the delta
                    that exceeds a threshold (Lava ``proc.sdn`` activation), the
                    graded-spike sparsification used by Loihi sigma-delta layers.

The CEILING in the menagerie is the ``lava`` / ``lava-loihi`` stack (git build, Loihi
runtime); on a conventional box the kernels are not the bottleneck -- the dynamics are
elementary recurrences.  These pure-torch reimplementations trace and render.

Inputs are current/spike trains of shape ``(time, batch, n)``; outputs are the
emitted (graded or binary) spike train of the same shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside spike forward, fast-sigmoid surrogate gradient (slayer-style)."""

    @staticmethod
    def forward(ctx, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * v.abs()) ** 2
        return grad_output * sg


_spike = _SurrogateSpike.apply


class CUBALIF(nn.Module):
    """Loihi CUBA-LIF: two states (synaptic current u, membrane voltage v)."""

    def __init__(self, n: int, du: float = 0.25, dv: float = 0.1, vth: float = 1.0) -> None:
        super().__init__()
        self.n = n
        self.du = du
        self.dv = dv
        self.vth = vth
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tt, B, _ = x.shape
        u = x.new_zeros(B, self.n)
        v = x.new_zeros(B, self.n)
        outs = []
        for t in range(Tt):
            u = u * (1.0 - self.du) + x[t]
            v = v * (1.0 - self.dv) + u + self.bias
            s = _spike(v - self.vth)
            v = v - s * self.vth  # reset by subtraction
            outs.append(s)
        return torch.stack(outs, dim=0)


class LavaLIF(nn.Module):
    """Lava single-state leaky integrate-and-fire (membrane-only)."""

    def __init__(self, n: int, dv: float = 0.1, vth: float = 1.0) -> None:
        super().__init__()
        self.n = n
        self.dv = dv
        self.vth = vth
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tt, B, _ = x.shape
        v = x.new_zeros(B, self.n)
        outs = []
        for t in range(Tt):
            v = v * (1.0 - self.dv) + x[t] + self.bias
            s = _spike(v - self.vth)
            v = v - s * self.vth
            outs.append(s)
        return torch.stack(outs, dim=0)


class TernaryLIF(nn.Module):
    """Lava TernaryLIF: emit +1 above upper threshold, -1 below lower threshold."""

    def __init__(self, n: int, dv: float = 0.1, vth_hi: float = 1.0, vth_lo: float = -1.0) -> None:
        super().__init__()
        self.n = n
        self.dv = dv
        self.vth_hi = vth_hi
        self.vth_lo = vth_lo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tt, B, _ = x.shape
        v = x.new_zeros(B, self.n)
        outs = []
        for t in range(Tt):
            v = v * (1.0 - self.dv) + x[t]
            s_pos = _spike(v - self.vth_hi)
            s_neg = _spike(self.vth_lo - v)
            s = s_pos - s_neg  # ternary {+1, 0, -1}
            v = v - s_pos * self.vth_hi + s_neg * (-self.vth_lo)
            outs.append(s)
        return torch.stack(outs, dim=0)


class AdaptiveLIF(nn.Module):
    """Lava ALIF: LIF with a spike-incremented, decaying adaptive threshold."""

    def __init__(
        self, n: int, dv: float = 0.1, dth: float = 0.05, vth0: float = 1.0, beta: float = 0.3
    ) -> None:
        super().__init__()
        self.n = n
        self.dv = dv
        self.dth = dth
        self.vth0 = vth0
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tt, B, _ = x.shape
        v = x.new_zeros(B, self.n)
        th = x.new_full((B, self.n), self.vth0)
        outs = []
        for t in range(Tt):
            v = v * (1.0 - self.dv) + x[t]
            s = _spike(v - th)
            v = v - s * th
            th = th * (1.0 - self.dth) + self.beta * s  # adaptation
            outs.append(s)
        return torch.stack(outs, dim=0)


class SigmaDelta(nn.Module):
    """Lava sigma-delta graded neuron: accumulate (sigma), emit delta over threshold."""

    def __init__(self, n: int, vth: float = 0.1) -> None:
        super().__init__()
        self.n = n
        self.vth = vth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tt, B, _ = x.shape
        acc = x.new_zeros(B, self.n)  # residual (already-sent) accumulator
        sigma = x.new_zeros(B, self.n)  # running integral of the input
        outs = []
        for t in range(Tt):
            sigma = sigma + x[t]
            delta = sigma - acc
            fire = _spike(delta.abs() - self.vth)
            graded = delta * fire  # graded spike: send the residual that exceeds vth
            acc = acc + graded
            outs.append(graded)
        return torch.stack(outs, dim=0)


def _build(cls):
    def _b() -> nn.Module:
        return cls(n=16)

    return _b


build_cuba_lif = _build(CUBALIF)
build_lava_lif = _build(LavaLIF)
build_ternary_lif = _build(TernaryLIF)
build_adaptive_lif = _build(AdaptiveLIF)
build_sigma_delta = _build(SigmaDelta)


def example_input() -> torch.Tensor:
    """Input current/spike train ``(time=8, batch=1, n=16)``."""
    return torch.randn(8, 1, 16)


MENAGERIE_ENTRIES = [
    ("Lava CUBA-LIF (Loihi current-based LIF)", "build_cuba_lif", "example_input", "2021", "MB1"),
    ("Lava LIF (Loihi leaky integrate-and-fire)", "build_lava_lif", "example_input", "2021", "MB1"),
    (
        "Lava TernaryLIF (ternary +1/0/-1 spiking LIF)",
        "build_ternary_lif",
        "example_input",
        "2021",
        "MB1",
    ),
    (
        "Lava AdaptiveLIF (ALIF, adaptive-threshold LIF)",
        "build_adaptive_lif",
        "example_input",
        "2021",
        "MB1",
    ),
    (
        "Lava Sigma-Delta (graded sigma-delta neuron)",
        "build_sigma_delta",
        "example_input",
        "2021",
        "MB1",
    ),
]
