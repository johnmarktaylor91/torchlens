"""SpikeDet / SpikSSD: spiking object detector with better firing patterns.

Fan et al., "SpikeDet: Better Firing Patterns for Accurate and Energy-Efficient
Object Detection with Spiking Neural Networks" (a.k.a. SpikSSD), 2025.
Paper: https://arxiv.org/abs/2501.15151
Source: https://github.com/yimeng-fan/SpikSSD

A spiking single-shot detector built from three distinctive pieces:
  * MDSNet BACKBONE (Membrane-based Deformed Shortcut residual Network): spiking
    residual blocks whose shortcut adjusts the MEMBRANE synaptic-input
    distribution at each layer, producing better neuron firing patterns during
    spiking feature extraction.
  * SMFM NECK (Spiking Multi-direction Fusion Module): fuses multi-scale spiking
    features in multiple directions (top-down + bottom-up) to improve
    multi-scale detection.
  * Multi-scale spiking detection heads.

This faithful random-init reimplementation keeps that spine: an MDSNet spiking
backbone producing 3 pyramid levels, an SMFM that fuses them bidirectionally,
and per-scale detection heads.  A leading TIME axis (T spiking steps) is
unrolled by TorchLens; the membrane-adjusting shortcut is the load-bearing
feature.  Channels/spatial/T are kept small so the graph renders quickly.
The forward returns a single concatenated detection tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics._snn_neurons import LIFNeuron, spike_fn


class _MembraneShortcutBlock(nn.Module):
    """MDSNet block: spiking residual with a membrane-adjusting deformed shortcut.

    The shortcut path applies a learned per-channel membrane bias/scale that
    "deforms" the residual's membrane synaptic-input distribution before the
    spiking neuron, the distinctive MDSNet mechanism.  Operates over (T,B,C,H,W).
    """

    def __init__(self, in_ch: int, out_ch: int, time: int, stride: int = 1) -> None:
        super().__init__()
        self.time = time
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.spike1 = LIFNeuron(beta=0.9, threshold=1.0)
        self.spike2 = LIFNeuron(beta=0.9, threshold=1.0)
        # membrane-adjusting deformed shortcut
        self.short_conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.mem_scale = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        self.mem_bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def _apply_conv(self, conv: nn.Module, bn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        out = bn(conv(x.reshape(t * b, c, h, w)))
        _, oc, oh, ow = out.shape
        return out.reshape(t, b, oc, oh, ow)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spike1(self._apply_conv(self.conv1, self.bn1, x))
        out = self._apply_conv(self.conv2, self.bn2, out)
        # membrane-adjusting deformed shortcut
        t, b, c, h, w = x.shape
        short = self.short_conv(x.reshape(t * b, c, h, w))
        _, sc, sh, sw = short.shape
        short = short.reshape(t, b, sc, sh, sw)
        short = short * self.mem_scale + self.mem_bias
        return self.spike2(out + short)


class _MDSNetBackbone(nn.Module):
    """MDSNet spiking backbone producing 3 pyramid feature levels."""

    def __init__(self, in_ch: int, time: int) -> None:
        super().__init__()
        self.time = time
        self.stem = _MembraneShortcutBlock(in_ch, 32, time, stride=2)
        self.layer1 = _MembraneShortcutBlock(32, 64, time, stride=2)  # C3
        self.layer2 = _MembraneShortcutBlock(64, 128, time, stride=2)  # C4
        self.layer3 = _MembraneShortcutBlock(128, 256, time, stride=2)  # C5

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)
        return c3, c4, c5


class _SMFM(nn.Module):
    """Spiking Multi-direction Fusion Module: bidirectional multi-scale fusion."""

    def __init__(self, channels: list, fuse_ch: int, time: int) -> None:
        super().__init__()
        self.time = time
        self.lateral = nn.ModuleList([nn.Conv2d(c, fuse_ch, 1, bias=False) for c in channels])
        self.spike = LIFNeuron(beta=0.9, threshold=1.0)
        self.out_conv = nn.ModuleList(
            [nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1, bias=False) for _ in channels]
        )

    def _lat(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        out = conv(x.reshape(t * b, c, h, w))
        _, oc, oh, ow = out.shape
        return out.reshape(t, b, oc, oh, ow)

    def _resize(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        _, _, _, rh, rw = ref.shape
        out = F.interpolate(x.reshape(t * b, c, h, w), size=(rh, rw), mode="nearest")
        return out.reshape(t, b, c, rh, rw)

    def forward(self, feats):
        # feats: list of 3 (T,B,C,H,W) at decreasing resolution
        p = [self._lat(conv, f) for conv, f in zip(self.lateral, feats)]
        # top-down
        p[1] = p[1] + self._resize(p[2], p[1])
        p[0] = p[0] + self._resize(p[1], p[0])
        # bottom-up
        p[1] = p[1] + self._resize(p[0], p[1])
        p[2] = p[2] + self._resize(p[1], p[2])
        outs = []
        for conv, f in zip(self.out_conv, p):
            f = self.spike(f)
            t, b, c, h, w = f.shape
            o = conv(f.reshape(t * b, c, h, w))
            _, oc, oh, ow = o.shape
            outs.append(o.reshape(t, b, oc, oh, ow))
        return outs


class SpikeDet(nn.Module):
    """SpikeDet / SpikSSD spiking object detector (MDSNet + SMFM, random init)."""

    def __init__(
        self, in_ch: int = 3, num_classes: int = 20, num_anchors: int = 3, time: int = 2
    ) -> None:
        super().__init__()
        self.time = time
        fuse_ch = 64
        self.backbone = _MDSNetBackbone(in_ch, time)
        self.smfm = _SMFM([64, 128, 256], fuse_ch, time)
        out_ch = num_anchors * (5 + num_classes)
        self.heads = nn.ModuleList([nn.Conv2d(fuse_ch, out_ch, 1) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.unsqueeze(0).expand(self.time, -1, -1, -1, -1)  # (T,B,C,H,W)
        c3, c4, c5 = self.backbone(x_t)
        fused = self.smfm([c3, c4, c5])
        # temporal-mean each level, run detection head, flatten + concat
        dets = []
        for head, f in zip(self.heads, fused):
            f_mean = f.mean(dim=0)  # (B,C,H,W)
            d = head(f_mean)  # (B, out_ch, H, W)
            dets.append(d.flatten(2))  # (B, out_ch, H*W)
        return torch.cat(dets, dim=2)  # (B, out_ch, sum H*W)


def build_spikedet() -> nn.Module:
    """Build the SpikeDet / SpikSSD spiking object detector (random init)."""
    return SpikeDet(in_ch=3, num_classes=20, num_anchors=3, time=2)


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 64, 64)``; the net repeats it over T=2 steps."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "SpikeDet (spiking object detector, MDSNet backbone + SMFM neck)",
        "build_spikedet",
        "example_input",
        "2025",
        "DC",
    ),
]
