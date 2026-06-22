"""SU-YOLO: spiking neural network for efficient underwater object detection.

Li et al., "SU-YOLO: Spiking Neural Network for Efficient Underwater Object
Detection", 2025.  Paper: https://arxiv.org/abs/2503.24389
Source: https://github.com/lwxfight/snn-underwater

A spiking CSP-YOLO detector for underwater imagery.  Distinctive contributions:
  * SEPARATED BATCH NORMALIZATION (SeBN): instead of one shared BatchNorm across
    all SNN timesteps, SeBN keeps a SEPARATE BN per timestep, normalising each
    step's feature map independently to capture the SNN's temporal dynamics
    (optimised for residual structures).
  * SPIKING CSP RESIDUAL BLOCKS: the Cross-Stage-Partial design fused with YOLO
    and spiking residual blocks to mitigate spike degradation.
  * Spike-based integer-addition denoising of the input.

This faithful random-init reimplementation keeps the distinctive SU-YOLO spine:
an integer-addition spike denoiser, a CSP spiking backbone whose residual blocks
use per-timestep Separated BN, and a YOLO detection head.  A leading TIME axis
(T spiking steps) is unrolled by TorchLens; the per-step SeBN is the load-bearing
feature.  Channels/spatial size/T kept small so the graph renders quickly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import spike_fn


class _SpikeNeuron(nn.Module):
    """Binary IF spiking neuron applied per-timestep (no leak, hard threshold)."""

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return spike_fn(x - self.threshold)


class _SeparatedBN(nn.Module):
    """Separated Batch Normalization (SeBN): one BatchNorm per SNN timestep.

    Input is (T, B, C, H, W); each timestep t is normalised by its own BN_t.
    """

    def __init__(self, channels: int, time: int) -> None:
        super().__init__()
        self.time = time
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(time)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W)
        outs = [self.bns[t](x[t]) for t in range(self.time)]
        return torch.stack(outs, dim=0)


class _SpikeConvSeBN(nn.Module):
    """Conv (shared) -> Separated-BN (per timestep) -> spike, over the time axis."""

    def __init__(self, in_ch: int, out_ch: int, time: int, stride: int = 1) -> None:
        super().__init__()
        self.time = time
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.sebn = _SeparatedBN(out_ch, time)
        self.spike = _SpikeNeuron(threshold=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W) -> shared conv per step
        t, b, c, h, w = x.shape
        conv_out = self.conv(x.reshape(t * b, c, h, w))
        _, oc, oh, ow = conv_out.shape
        conv_out = conv_out.reshape(t, b, oc, oh, ow)
        return self.spike(self.sebn(conv_out))


class _SpikeCSPBlock(nn.Module):
    """Spiking Cross-Stage-Partial block: split -> spiking residual conv -> concat."""

    def __init__(self, channels: int, time: int) -> None:
        super().__init__()
        half = channels // 2
        self.split_main = _SpikeConvSeBN(half, half, time)
        self.res_conv = _SpikeConvSeBN(half, half, time)
        self.fuse = _SpikeConvSeBN(channels, channels, time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W); CSP split along channel dim
        c = x.shape[2]
        x1 = x[:, :, : c // 2]
        x2 = x[:, :, c // 2 :]
        main = self.split_main(x1)
        res = self.res_conv(main) + main  # spiking residual
        out = torch.cat([res, x2], dim=2)
        return self.fuse(out)


class _SpikeDenoiser(nn.Module):
    """Spike-based integer-addition denoiser (depthwise smoothing + spike)."""

    def __init__(self, channels: int, time: int) -> None:
        super().__init__()
        self.smooth = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.spike = _SpikeNeuron(threshold=0.5)
        self.time = time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        out = self.smooth(x.reshape(t * b, c, h, w)).reshape(t, b, c, h, w)
        return self.spike(out + x)  # integer-addition residual denoise


class SUYOLO(nn.Module):
    """SU-YOLO spiking CSP underwater detector (random init, SeBN per timestep)."""

    def __init__(
        self, in_ch: int = 3, num_classes: int = 4, num_anchors: int = 3, time: int = 2
    ) -> None:
        super().__init__()
        self.time = time
        self.denoise = _SpikeDenoiser(in_ch, time)
        self.stem = _SpikeConvSeBN(in_ch, 32, time, stride=2)
        self.csp1 = _SpikeCSPBlock(32, time)
        self.down1 = _SpikeConvSeBN(32, 64, time, stride=2)
        self.csp2 = _SpikeCSPBlock(64, time)
        self.down2 = _SpikeConvSeBN(64, 128, time, stride=2)
        self.csp3 = _SpikeCSPBlock(128, time)
        out_ch = num_anchors * (5 + num_classes)
        self.det = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> repeat over T spiking steps
        x_t = x.unsqueeze(0).expand(self.time, -1, -1, -1, -1)  # (T, B, C, H, W)
        x_t = self.denoise(x_t)
        x_t = self.stem(x_t)
        x_t = self.csp1(x_t)
        x_t = self.down1(x_t)
        x_t = self.csp2(x_t)
        x_t = self.down2(x_t)
        x_t = self.csp3(x_t)
        # temporal mean -> YOLO detection grid
        feat = x_t.mean(dim=0)  # (B, C, H', W')
        return self.det(feat)


def build_su_yolo() -> nn.Module:
    """Build the SU-YOLO spiking underwater detector (random init)."""
    return SUYOLO(in_ch=3, num_classes=4, num_anchors=3, time=2)


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 64, 64)``; the net repeats it over T=2 steps."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "SU-YOLO (spiking CSP-YOLO underwater detector with separated batch norm)",
        "build_su_yolo",
        "example_input",
        "2025",
        "DC",
    ),
]
