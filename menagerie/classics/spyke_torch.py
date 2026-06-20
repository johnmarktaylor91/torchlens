"""SpykeTorch STDP convolutional spiking network (time-to-first-spike, WTA).

Mozafari, Ganjtabesh, Nowzari-Dalini, Masquelier (2019),
"SpykeTorch: Efficient Simulation of Convolutional Spiking Neural Networks
with at most one Spike per Neuron."
Paper: https://arxiv.org/abs/1903.02440
Source: https://github.com/miladmozafari/SpykeTorch

SpykeTorch simulates deep convolutional SNNs that fire *at most one spike per
neuron* under a time-to-first-spike (rank-order) coding scheme and are trained
with STDP / reward-modulated STDP plus winner-take-all lateral inhibition.  Its
canonical pipeline (the ``DeepConvSNN`` / Mozafari deep network) is:

    input -> DoG filter bank -> intensity-to-latency temporal coding
          -> [ S1 spiking conv -> threshold -> lateral inhibition -> C1 pool ]
          -> [ S2 spiking conv -> threshold -> lateral inhibition -> C2 pool ]
          -> [ S3 spiking conv -> threshold -> global pool / K-winners ]
          -> per-class spike counts

This faithful reimplementation reproduces the *forward* architecture (STDP/RSTDP
weight updates are training-time and omitted for the atlas):

  * a fixed DoG (Difference-of-Gaussians) edge-filter front end producing ON/OFF
    contrast channels,
  * ``Intensity2Latency`` temporal coding that bins filtered intensities into
    ``T`` ascending time steps (strongest features spike first; at most one
    spike per neuron via a cumulative-OR over time),
  * three spiking convolution stages, each ``conv -> integrate-over-time ->
    threshold spike -> local lateral inhibition (max-based) -> pooling``,
  * a final K-winners-take-all readout over cumulative spike counts giving
    per-class scores.

``T`` is kept small so the time-unrolled graph renders quickly; the dynamics are
identical at any horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside spike (forward) with a fast-sigmoid surrogate gradient."""

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


def _dog_kernel(size: int, s1: float, s2: float) -> torch.Tensor:
    """Difference-of-Gaussians kernel (center-surround), normalised zero-mean."""
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r2 = xx * xx + yy * yy
    g1 = torch.exp(-r2 / (2 * s1 * s1)) / (2 * 3.141592653589793 * s1 * s1)
    g2 = torch.exp(-r2 / (2 * s2 * s2)) / (2 * 3.141592653589793 * s2 * s2)
    dog = g1 - g2
    dog = dog - dog.mean()
    return dog


class DoGFilterBank(nn.Module):
    """Fixed ON/OFF Difference-of-Gaussians contrast front end.

    Produces 2 channels per input channel: an ON-center (DoG) and OFF-center
    (-DoG) response, faithfully reproducing SpykeTorch's center-surround
    retinal-style preprocessing that feeds the latency encoder.
    """

    def __init__(self, in_ch: int = 1, size: int = 7) -> None:
        super().__init__()
        k = _dog_kernel(size, s1=1.0, s2=2.0)
        # weight: (2*in_ch, in_ch, size, size) with ON = +DoG, OFF = -DoG per channel.
        w = torch.zeros(2 * in_ch, in_ch, size, size)
        for c in range(in_ch):
            w[2 * c, c] = k
            w[2 * c + 1, c] = -k
        self.register_buffer("weight", w)
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.weight, padding=self.size // 2)
        return F.relu(y)  # only positive contrast survives (ON/OFF rectified)


class Intensity2Latency(nn.Module):
    """Intensity-to-latency temporal coding (time-to-first-spike).

    Strong filter responses spike at early time bins, weaker responses later.
    Thresholds the (normalised) intensity at ``T`` ascending levels and forms a
    cumulative spike train so each neuron fires at most once (rank-order code).
    Returns a tensor of shape ``(T, batch, C, H, W)``.
    """

    def __init__(self, time_steps: int = 4) -> None:
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.flatten(1)
        maxv = flat.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        norm = (flat / maxv).view(b, c, h, w)
        spikes = []
        # Bin t fires neurons whose normalised intensity exceeds (T-1-t)/T,
        # so the strongest fire first.  Cumulative -> at most one onset.
        for t in range(self.time_steps):
            level = (self.time_steps - 1 - t) / self.time_steps
            spikes.append(_spike(norm - level))
        return torch.stack(spikes, dim=0)  # (T, b, c, h, w)


def _lateral_inhibition(spikes: torch.Tensor) -> torch.Tensor:
    """Feature-wise lateral inhibition: keep only the max-responding channel.

    At every spatial location only the strongest feature map is allowed to keep
    its spike (winner-take-all across channels), as in SpykeTorch's
    ``pointwise_inhibition``.
    """
    # spikes: (b, c, h, w)
    maxc = spikes.max(dim=1, keepdim=True).values
    winner = (spikes >= maxc).to(spikes.dtype)
    return spikes * winner


class SpikingConv(nn.Module):
    """One SpykeTorch spiking-convolution stage.

    ``conv -> integrate over time -> threshold spike (>=1 onset per neuron) ->
    lateral inhibition -> pooling``.  Operates on a ``(T, b, C, H, W)`` spike
    train and returns the post-pool spike train at the same time resolution.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 5,
        threshold: float = 15.0,
        pool: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.threshold = threshold
        self.pool = nn.MaxPool2d(pool, ceil_mode=True)

    def forward(self, spk: torch.Tensor) -> torch.Tensor:
        T = spk.shape[0]
        pot = None  # accumulated membrane potential
        fired = None  # neurons that have already fired (at-most-one-spike rule)
        outs = []
        for t in range(T):
            current = self.conv(spk[t])
            pot = current if pot is None else pot + current
            onset = _spike(pot - self.threshold)
            if fired is None:
                new = onset
                fired = onset
            else:
                # Only allow a neuron's first crossing to count as a spike.
                new = onset * (1.0 - fired)
                fired = torch.clamp(fired + new, max=1.0)
            new = _lateral_inhibition(new)
            outs.append(self.pool(new))
        return torch.stack(outs, dim=0)


class SpykeTorchDeepSNN(nn.Module):
    """Deep convolutional spiking network in the SpykeTorch style.

    DoG front end -> latency coding -> S1/C1 -> S2/C2 -> S3 + K-winners readout.
    Returns per-class scores derived from cumulative final-stage spike counts.
    """

    def __init__(self, in_ch: int = 2, n_classes: int = 10, time_steps: int = 4) -> None:
        super().__init__()
        # in_ch=2 means input already has 2 DoG/ON-OFF channels.
        self.latency = Intensity2Latency(time_steps=time_steps)
        self.s1 = SpikingConv(in_ch, 16, kernel=5, threshold=10.0, pool=2)
        self.s2 = SpikingConv(16, 32, kernel=3, threshold=12.0, pool=2)
        self.s3 = SpikingConv(32, 64, kernel=3, threshold=12.0, pool=2)
        self.readout = nn.Linear(64, n_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is a ``(batch, in_ch, H, W)`` DoG-filtered intensity map."""
        spk = self.latency(x)  # (T, b, in_ch, H, W)
        spk = self.s1(spk)
        spk = self.s2(spk)
        spk = self.s3(spk)  # (T, b, 64, h, w)
        # Cumulative spike counts over time, then global spatial pooling.
        counts = spk.sum(dim=0)  # (b, 64, h, w)
        pooled = F.adaptive_avg_pool2d(counts, 1).flatten(1)  # (b, 64)
        scores = self.readout(pooled)  # (b, n_classes)
        # K-winners-take-all (k=1) readout: emphasise the winning class.
        wta = _spike(scores - scores.max(dim=1, keepdim=True).values)
        return scores + wta


class _DoGFrontEndSNN(nn.Module):
    """Single-channel-input wrapper: applies the DoG front end then the SNN.

    Lets the example input be a raw 1-channel image while still exercising the
    fixed center-surround filter bank that defines SpykeTorch's preprocessing.
    """

    def __init__(self, n_classes: int = 10, time_steps: int = 4) -> None:
        super().__init__()
        self.dog = DoGFilterBank(in_ch=1, size=7)  # -> 2 ON/OFF channels
        self.snn = SpykeTorchDeepSNN(in_ch=2, n_classes=n_classes, time_steps=time_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.snn(self.dog(x))


def build() -> nn.Module:
    """Build the SpykeTorch deep convolutional spiking network (TTFS, WTA)."""
    return _DoGFrontEndSNN(n_classes=10, time_steps=4)


def example_input() -> torch.Tensor:
    """Example grayscale image ``(1, 1, 28, 28)`` (DoG front end expands to ON/OFF)."""
    return torch.rand(1, 1, 28, 28)


MENAGERIE_ENTRIES = [
    (
        "SpykeTorch STDP convolutional spiking network (time-to-first-spike, WTA)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
