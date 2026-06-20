"""Spiking-YOLO: spiking neural network for energy-efficient object detection.

Kim et al., "Spiking-YOLO: Spiking Neural Network for Energy-Efficient Object
Detection", AAAI 2020.  Paper: https://ojs.aaai.org/index.php/AAAI/article/view/6787

The first spike-based object detector: a Tiny-YOLO backbone converted to an SNN.
Two distinctive contributions:
  * CHANNEL-WISE NORMALIZATION: weights are normalised per-channel by the maximum
    achievable activation in that channel, eliminating tiny activations and
    fixing under-firing so deep SNN layers transmit information accurately.
  * SIGNED NEURON WITH IMBALANCED THRESHOLD: an integrate-and-fire neuron that
    fires for BOTH positive and negative membrane potentials (two thresholds,
    one per sign) so it can represent leaky-ReLU's negative region.

This faithful random-init reimplementation keeps the Tiny-YOLO convolutional
backbone (conv/BN/pool stack -> detection conv head producing a YOLO grid)
with every leaky-ReLU replaced by a signed IF spiking neuron and a channel-wise
normalisation (per-channel scale) applied before spiking.  Single tensor in /
single YOLO-grid tensor out.  Spatial size and channels are kept small so the
graph renders quickly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import spike_fn


class _SignedIFNeuron(nn.Module):
    """Signed integrate-and-fire neuron with an imbalanced (two-sided) threshold.

    Fires +1 when the membrane exceeds the positive threshold and -1 when it
    falls below the negative threshold, representing both positive and negative
    (leaky-ReLU-like) activations -- the distinctive Spiking-YOLO neuron.
    """

    def __init__(self, pos_thresh: float = 1.0, neg_thresh: float = 1.0) -> None:
        super().__init__()
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = spike_fn(x - self.pos_thresh)
        neg = spike_fn(-x - self.neg_thresh)
        return pos * self.pos_thresh - neg * self.neg_thresh


class _ChannelWiseNorm(nn.Module):
    """Per-channel scale normalisation (channel-wise normalization)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # learned per-channel scale standing in for the max-activation normaliser
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _SpikeConvBlock(nn.Module):
    """Conv -> BN -> channel-wise norm -> signed IF spike (the Spiking-YOLO unit)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.cwn = _ChannelWiseNorm(out_ch)
        self.spike = _SignedIFNeuron(pos_thresh=1.0, neg_thresh=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spike(self.cwn(self.bn(self.conv(x))))


class SpikingYOLO(nn.Module):
    """Tiny-YOLO backbone converted to an SNN (signed IF + channel-wise norm)."""

    def __init__(self, in_ch: int = 3, num_classes: int = 20, num_anchors: int = 5) -> None:
        super().__init__()
        # Tiny-YOLO style conv/pool backbone (scaled down widths)
        self.block1 = _SpikeConvBlock(in_ch, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = _SpikeConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = _SpikeConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = _SpikeConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.block5 = _SpikeConvBlock(128, 256)
        self.block6 = _SpikeConvBlock(256, 256)
        # Detection head: 1x1 conv producing (anchors * (5 + classes)) channels
        out_ch = num_anchors * (5 + num_classes)
        self.det = nn.Conv2d(256, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.block5(x)
        x = self.block6(x)
        return self.det(x)  # (B, anchors*(5+classes), H', W')


def build_spiking_yolo() -> nn.Module:
    """Build the Spiking-YOLO detector (Tiny-YOLO SNN, random init)."""
    return SpikingYOLO(in_ch=3, num_classes=20, num_anchors=5)


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 64, 64)`` for Spiking-YOLO."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Spiking-YOLO (Tiny-YOLO converted to SNN, signed IF + channel-wise norm)",
        "build_spiking_yolo",
        "example_input",
        "2020",
        "DC",
    ),
]
