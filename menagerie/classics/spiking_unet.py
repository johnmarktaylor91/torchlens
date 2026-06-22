"""Spiking-UNet: multi-threshold spiking U-Net for image processing.

Li et al., "Deep Multi-Threshold Spiking-UNet for Image Processing",
NeuroComputing 2024.  Paper: https://arxiv.org/abs/2307.10974
Source: https://github.com/SNNresearch/Spiking-UNet

A U-Net whose ReLU activations are replaced by spiking neurons.  The
distinctive contribution is the MULTI-THRESHOLD spiking neuron: a neuron that
fires spikes at several thresholds within a short time window, raising the
information-carrying capacity of a spike (vs. a binary IF neuron) so that
high-fidelity feature maps can propagate through the encoder-decoder with skip
connections.  A connection-wise normalization keeps firing rates accurate.

This faithful random-init reimplementation keeps the canonical U-Net spine
(encoder downsampling path + bottleneck + decoder upsampling path with
skip-concatenations) and replaces every ReLU with a MULTI-LEVEL (multi-threshold)
spiking activation.  Channels and spatial size are kept small so the graph
renders quickly.  Run as a single-step conversion-style SNN (the multi-threshold
neuron already encodes graded magnitude), single tensor in / single tensor out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import SpikingActivation


class _MultiThreshSpikeConv(nn.Module):
    """Conv -> BN -> multi-threshold spiking neuron (the Spiking-UNet block unit)."""

    def __init__(self, in_ch: int, out_ch: int, levels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        # multi-threshold spiking neuron: fires at `levels` graded thresholds
        self.spike = SpikingActivation(threshold=0.5, levels=levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spike(self.bn(self.conv(x)))


class _DoubleSpikeConv(nn.Module):
    """Two multi-threshold spiking conv units (the standard U-Net double conv)."""

    def __init__(self, in_ch: int, out_ch: int, levels: int = 3) -> None:
        super().__init__()
        self.conv1 = _MultiThreshSpikeConv(in_ch, out_ch, levels)
        self.conv2 = _MultiThreshSpikeConv(out_ch, out_ch, levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class SpikingUNet(nn.Module):
    """Multi-threshold Spiking-UNet (encoder/decoder with skip connections)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 16, levels: int = 3) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        # Encoder
        self.enc1 = _DoubleSpikeConv(in_ch, c1, levels)
        self.enc2 = _DoubleSpikeConv(c1, c2, levels)
        self.enc3 = _DoubleSpikeConv(c2, c3, levels)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = _DoubleSpikeConv(c3, c4, levels)
        # Decoder (transpose-conv upsampling + skip concat)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleSpikeConv(c3 * 2, c3, levels)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleSpikeConv(c2 * 2, c2, levels)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleSpikeConv(c1 * 2, c1, levels)
        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def build_spiking_unet() -> nn.Module:
    """Build the multi-threshold Spiking-UNet (random init)."""
    return SpikingUNet(in_ch=3, out_ch=1, base=16, levels=3)


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 64, 64)`` for the Spiking-UNet."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Spiking-UNet (multi-threshold spiking U-Net for image processing)",
        "build_spiking_unet",
        "example_input",
        "2023",
        "DC",
    ),
]
