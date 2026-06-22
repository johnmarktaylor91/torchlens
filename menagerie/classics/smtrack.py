"""SMTrack: end-to-end spiking neural network tracker.

"SMTrack: End-to-End Trained Spiking Neural Networks for Multi-Object Tracking
in RGB Videos", 2025.  Paper: https://arxiv.org/abs/2508.14607

SMTrack is the first directly-trained deep-SNN framework for end-to-end tracking
from standard RGB video.  Its spiking detector is built on SpikeYOLO, which
combines the YOLOv8 macro-architecture with Meta-SpikeFormer modules and
introduces the distinctive INTEGER LEAKY-INTEGRATE-AND-FIRE (I-LIF) neuron:
during training it emits an *integer* spike count (unifying integer training
with spike-based inference), giving graded activations while staying spike-based.

We read SMTrack in its faithful SNN-tracker form: a Siamese-style spiking
backbone (SpikeYOLO with I-LIF neurons) that embeds a template and a search
region, followed by a depthwise cross-correlation head that produces the
response/score map used for association.  This random-init reimplementation
keeps the distinctive I-LIF neuron + spiking Meta-SpikeFormer-style conv blocks
and the Siamese correlation tracking head.  A leading TIME axis (T spiking steps)
is unrolled by TorchLens; sizes are kept small so the graph renders quickly.
The forward returns the single correlation response map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics._snn_neurons import _SurrogateSpike


class _IntegerLIF(nn.Module):
    """Integer Leaky-Integrate-and-Fire (I-LIF) neuron.

    Emits an INTEGER spike count per step (clamped to ``levels``) rather than a
    binary spike, unifying integer training with spike inference -- the
    distinctive SpikeYOLO / SMTrack neuron.  Unrolled over the leading time axis.
    """

    def __init__(self, beta: float = 0.9, threshold: float = 1.0, levels: int = 4) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = x.shape[0]
        v = torch.zeros_like(x[0])
        outs = []
        for t in range(time):
            v = self.beta * v + x[t]
            # integer spike count: sum of unit steps up to `levels`
            count = torch.zeros_like(v)
            for lvl in range(1, self.levels + 1):
                count = count + _SurrogateSpike.apply(v - lvl * self.threshold)
            count = torch.clamp(count, max=float(self.levels))
            v = v - count * self.threshold  # subtractive reset by emitted count
            outs.append(count)
        return torch.stack(outs, dim=0)


class _SpikeConvBlock(nn.Module):
    """Conv -> BN -> I-LIF spiking block (Meta-SpikeFormer-style), over time."""

    def __init__(
        self, in_ch: int, out_ch: int, time: int, stride: int = 1, levels: int = 4
    ) -> None:
        super().__init__()
        self.time = time
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.spike = _IntegerLIF(beta=0.9, threshold=1.0, levels=levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        out = self.bn(self.conv(x.reshape(t * b, c, h, w)))
        _, oc, oh, ow = out.shape
        out = out.reshape(t, b, oc, oh, ow)
        return self.spike(out)


class _SpikeYOLOBackbone(nn.Module):
    """SpikeYOLO-style spiking backbone (shared Siamese feature extractor)."""

    def __init__(self, in_ch: int, embed_ch: int, time: int) -> None:
        super().__init__()
        self.time = time
        self.stem = _SpikeConvBlock(in_ch, 32, time, stride=2)
        self.b1 = _SpikeConvBlock(32, 64, time, stride=2)
        self.b2 = _SpikeConvBlock(64, embed_ch, time, stride=2)
        self.b3 = _SpikeConvBlock(embed_ch, embed_ch, time, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> repeat over T spiking steps
        x_t = x.unsqueeze(0).expand(self.time, -1, -1, -1, -1)  # (T,B,C,H,W)
        x_t = self.stem(x_t)
        x_t = self.b1(x_t)
        x_t = self.b2(x_t)
        x_t = self.b3(x_t)
        return x_t.mean(dim=0)  # temporal mean -> (B, embed_ch, h, w)


class SMTrack(nn.Module):
    """SMTrack Siamese spiking tracker (SpikeYOLO backbone + correlation head)."""

    def __init__(self, in_ch: int = 3, embed_ch: int = 128, time: int = 2) -> None:
        super().__init__()
        self.backbone = _SpikeYOLOBackbone(in_ch, embed_ch, time)
        # post-correlation head
        self.head = nn.Sequential(
            nn.Conv2d(embed_ch, embed_ch, 1),
            nn.BatchNorm2d(embed_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x packs template + search along the batch dim: x[0]=template, x[1]=search
        template = x[0:1]  # (1, C, Ht, Wt)
        search = x[1:2]  # (1, C, Hs, Ws)
        zf = self.backbone(template)  # (1, E, ht, wt)  -- correlation kernel
        xf = self.backbone(search)  # (1, E, hs, ws)
        # cross-correlation: zf (all E channels) is the conv kernel over xf,
        # collapsing to a single-channel Siamese response map.
        e = zf.shape[1]
        resp = F.conv2d(xf, zf)  # (1, 1, Hr, Wr)
        # broadcast response over channels for the post-correlation head
        resp = resp.repeat(1, e, 1, 1)
        return self.head(resp)  # (1, 1, Hr, Wr) score map


def build_smtrack() -> nn.Module:
    """Build the SMTrack Siamese spiking tracker (random init)."""
    return SMTrack(in_ch=3, embed_ch=128, time=2)


def example_input() -> torch.Tensor:
    """Example packed (template, search) pair ``(2, 3, 64, 64)``.

    Index 0 is the template crop, index 1 is the search region; the Siamese
    backbone embeds both and cross-correlates them into a response map.
    """
    return torch.randn(2, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "SMTrack (Siamese spiking tracker, SpikeYOLO backbone + correlation head)",
        "build_smtrack",
        "example_input",
        "2025",
        "DC",
    ),
]
