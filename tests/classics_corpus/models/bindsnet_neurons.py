"""BindsNET neuron groups: spiking-neuron node models as traceable modules.

Source: https://github.com/BindsNET/bindsnet  (``bindsnet/network/nodes.py``)

BindsNET defines a family of spiking-neuron "node" groups used as layers inside a
``Network``.  Each node type is a small, fully-specified dynamical system: a
membrane potential (and optional auxiliary state) integrated over discrete time,
with a threshold-and-reset spike rule.  In BindsNET these are *not* standalone
``nn.Module`` forward functions (they are stepped by the surrounding Network), so
this module wraps each as a faithful, traceable ``nn.Module`` that integrates the
published update equations over a leading time dimension, emitting a surrogate
gradient at the spike threshold so the computation graph is differentiable.

Reimplemented node types (random init), faithful to the BindsNET equations:
  - McCullochPitts : instantaneous threshold unit (s = x >= thresh)
  - IFNodes        : non-leaky integrate-and-fire
  - LIFNodes       : leaky integrate-and-fire (exponential membrane decay)
  - CurrentLIFNodes: LIF with a synaptic-current state variable
  - AdaptiveLIFNodes: LIF with an adaptive (spike-incremented, decaying) threshold
  - DiehlAndCookNodes: adaptive-threshold LIF from Diehl & Cook (2015)
  - IzhikevichNodes: Izhikevich 2-variable (v, u) spiking model
  - SRM0Nodes      : simplified Spike Response Model (SRM0)

Inputs are spike/current trains of shape ``(time, batch, n)``; outputs are the
emitted spike train of the same shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside spike in the forward pass, fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * v.abs()) ** 2
        return grad_output * sg


def spike_fn(v_over_thresh: torch.Tensor) -> torch.Tensor:
    """Emit a spike where membrane potential reaches threshold (surrogate grad)."""
    return _SurrogateSpike.apply(v_over_thresh)


class McCullochPitts(nn.Module):
    """Instantaneous McCulloch-Pitts threshold unit."""

    def __init__(self, n: int = 100, thresh: float = 1.0) -> None:
        super().__init__()
        self.n = n
        self.thresh = thresh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return spike_fn(x - self.thresh)


class IFNodes(nn.Module):
    """Non-leaky integrate-and-fire neurons."""

    def __init__(self, n: int = 100, thresh: float = 1.0, reset: float = 0.0) -> None:
        super().__init__()
        self.n = n
        self.thresh = thresh
        self.reset = reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, _ = x.shape
        v = torch.full((batch, self.n), self.reset, device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(time):
            v = v + x[t]
            s = spike_fn(v - self.thresh)
            v = v * (1 - s) + self.reset * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class LIFNodes(nn.Module):
    """Leaky integrate-and-fire neurons (exponential membrane decay)."""

    def __init__(
        self,
        n: int = 100,
        thresh: float = 1.0,
        rest: float = 0.0,
        reset: float = 0.0,
        decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.n = n
        self.thresh = thresh
        self.rest = rest
        self.reset = reset
        self.decay = decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, _ = x.shape
        v = torch.full((batch, self.n), self.rest, device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(time):
            v = self.rest + self.decay * (v - self.rest) + x[t]
            s = spike_fn(v - self.thresh)
            v = v * (1 - s) + self.reset * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class CurrentLIFNodes(nn.Module):
    """LIF neurons with an explicit synaptic-current state variable."""

    def __init__(
        self,
        n: int = 100,
        thresh: float = 1.0,
        rest: float = 0.0,
        reset: float = 0.0,
        decay: float = 0.95,
        i_decay: float = 0.9,
    ) -> None:
        super().__init__()
        self.n = n
        self.thresh = thresh
        self.rest = rest
        self.reset = reset
        self.decay = decay
        self.i_decay = i_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, _ = x.shape
        v = torch.full((batch, self.n), self.rest, device=x.device, dtype=x.dtype)
        i = torch.zeros((batch, self.n), device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(time):
            i = self.i_decay * i + x[t]
            v = self.rest + self.decay * (v - self.rest) + i
            s = spike_fn(v - self.thresh)
            v = v * (1 - s) + self.reset * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class AdaptiveLIFNodes(nn.Module):
    """LIF neurons with an adaptive threshold (spike-incremented, decaying)."""

    def __init__(
        self,
        n: int = 100,
        thresh: float = 1.0,
        rest: float = 0.0,
        reset: float = 0.0,
        decay: float = 0.95,
        theta_plus: float = 0.05,
        theta_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.n = n
        self.thresh = thresh
        self.rest = rest
        self.reset = reset
        self.decay = decay
        self.theta_plus = theta_plus
        self.theta_decay = theta_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, _ = x.shape
        v = torch.full((batch, self.n), self.rest, device=x.device, dtype=x.dtype)
        theta = torch.zeros((batch, self.n), device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(time):
            v = self.rest + self.decay * (v - self.rest) + x[t]
            s = spike_fn(v - (self.thresh + theta))
            v = v * (1 - s) + self.reset * s
            theta = self.theta_decay * theta + self.theta_plus * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class DiehlAndCookNodes(nn.Module):
    """Adaptive-threshold LIF neurons from Diehl & Cook (2015)."""

    def __init__(
        self,
        n: int = 100,
        thresh: float = -52.0,
        rest: float = -65.0,
        reset: float = -65.0,
        decay: float = 0.95,
        theta_plus: float = 0.05,
        theta_decay: float = 0.9999,
    ) -> None:
        super().__init__()
        self.n = n
        self.thresh = thresh
        self.rest = rest
        self.reset = reset
        self.decay = decay
        self.theta_plus = theta_plus
        self.theta_decay = theta_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, _ = x.shape
        v = torch.full((batch, self.n), self.rest, device=x.device, dtype=x.dtype)
        theta = torch.zeros((batch, self.n), device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(time):
            v = self.rest + self.decay * (v - self.rest) + x[t]
            s = spike_fn(v - (self.thresh + theta))
            v = v * (1 - s) + self.reset * s
            theta = self.theta_decay * theta + self.theta_plus * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class IzhikevichNodes(nn.Module):
    """Izhikevich two-variable (v, u) spiking neurons."""

    def __init__(
        self,
        n: int = 100,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        thresh: float = 30.0,
    ) -> None:
        super().__init__()
        self.n = n
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.thresh = thresh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, _ = x.shape
        v = torch.full((batch, self.n), self.c, device=x.device, dtype=x.dtype)
        u = self.b * v
        spikes = []
        for t in range(time):
            # Two 0.5 ms sub-steps for numerical stability (BindsNET convention).
            v = v + 0.5 * (0.04 * v * v + 5 * v + 140 - u + x[t])
            v = v + 0.5 * (0.04 * v * v + 5 * v + 140 - u + x[t])
            u = u + self.a * (self.b * v - u)
            s = spike_fn(v - self.thresh)
            v = v * (1 - s) + self.c * s
            u = u + self.d * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class SRM0Nodes(nn.Module):
    """Simplified Spike Response Model (SRM0) neurons."""

    def __init__(
        self,
        n: int = 100,
        thresh: float = -50.0,
        rest: float = -65.0,
        reset: float = -65.0,
        decay: float = 0.9,
        eps_0: float = 1.0,
    ) -> None:
        super().__init__()
        self.n = n
        self.thresh = thresh
        self.rest = rest
        self.reset = reset
        self.decay = decay
        self.eps_0 = eps_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, _ = x.shape
        v = torch.full((batch, self.n), self.rest, device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(time):
            # SRM0: membrane = rest + filtered input kernel response.
            v = self.rest + self.decay * (v - self.rest) + self.eps_0 * x[t]
            s = spike_fn(v - self.thresh)
            v = v * (1 - s) + self.reset * s
            spikes.append(s)
        return torch.stack(spikes, dim=0)


def _build(cls: type) -> nn.Module:
    return cls(n=100)


def build_mccullochpitts() -> nn.Module:
    """Build a McCulloch-Pitts threshold-unit group (n=100)."""
    return _build(McCullochPitts)


def build_ifnodes() -> nn.Module:
    """Build an integrate-and-fire neuron group (n=100)."""
    return _build(IFNodes)


def build_lifnodes() -> nn.Module:
    """Build a leaky integrate-and-fire neuron group (n=100)."""
    return _build(LIFNodes)


def build_currentlifnodes() -> nn.Module:
    """Build a current-based LIF neuron group (n=100)."""
    return _build(CurrentLIFNodes)


def build_adaptivelifnodes() -> nn.Module:
    """Build an adaptive-threshold LIF neuron group (n=100)."""
    return _build(AdaptiveLIFNodes)


def build_diehlandcooknodes() -> nn.Module:
    """Build a Diehl & Cook adaptive-LIF neuron group (n=100)."""
    return _build(DiehlAndCookNodes)


def build_izhikevichnodes() -> nn.Module:
    """Build an Izhikevich neuron group (n=100)."""
    return _build(IzhikevichNodes)


def build_srm0nodes() -> nn.Module:
    """Build an SRM0 neuron group (n=100)."""
    return _build(SRM0Nodes)


def build() -> nn.Module:
    """Default build (LIF neuron group) for the generic classics contract."""
    return build_lifnodes()


def example_input() -> torch.Tensor:
    """Example spike/current train ``(time=8, batch=4, n=100)``.

    A short time horizon keeps the unrolled-time computation graph compact so
    the Graphviz sibling-ordering layout renders quickly; the neuron dynamics
    are identical at any sequence length.
    """
    return torch.rand(8, 4, 100)


MENAGERIE_ENTRIES = [
    (
        "BindsNET McCullochPitts (threshold node)",
        "build_mccullochpitts",
        "example_input",
        "2018",
        "E1",
    ),
    ("BindsNET IFNodes (integrate-and-fire)", "build_ifnodes", "example_input", "2018", "E1"),
    (
        "BindsNET LIFNodes (leaky integrate-and-fire)",
        "build_lifnodes",
        "example_input",
        "2018",
        "E1",
    ),
    (
        "BindsNET CurrentLIFNodes (current-based LIF)",
        "build_currentlifnodes",
        "example_input",
        "2018",
        "E1",
    ),
    (
        "BindsNET AdaptiveLIFNodes (adaptive-threshold LIF)",
        "build_adaptivelifnodes",
        "example_input",
        "2018",
        "E1",
    ),
    (
        "BindsNET DiehlAndCookNodes (Diehl & Cook 2015 LIF)",
        "build_diehlandcooknodes",
        "example_input",
        "2015",
        "E1",
    ),
    (
        "BindsNET IzhikevichNodes (Izhikevich model)",
        "build_izhikevichnodes",
        "example_input",
        "2003",
        "E6",
    ),
    ("BindsNET SRM0Nodes (Spike Response Model)", "build_srm0nodes", "example_input", "2018", "E6"),
]
