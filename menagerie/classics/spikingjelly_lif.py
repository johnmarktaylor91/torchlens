"""SpikingJelly-style LIF spiking convnet (multi-step surrogate-gradient SNN).

Based on the SpikingJelly tutorial canonical architecture:
  Fang et al., "SpikingJelly: An open-source machine learning infrastructure
  platform for spike-based intelligence", Science Advances, 2023.
  Paper: https://arxiv.org/abs/2310.16620 (SpikingJelly v2)
  Source: https://github.com/fangwei123456/spikingjelly
  Tutorial: https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based/tutorial_mnist_mutil_step.html

SpikingJelly is the canonical open-source SNN training framework in PyTorch.
Its tutorial network for image classification is a small spiking convolutional
network where every activation is replaced by a multi-step LIF neuron:

  Input (B, C, H, W) -> repeat over T timesteps
  -> Conv2d -> BN -> LIF -> MaxPool  (spiking conv block 1)
  -> Conv2d -> BN -> LIF -> MaxPool  (spiking conv block 2)
  -> Flatten -> Linear -> LIF        (spiking FC)
  -> Linear                          (output head, averaged over T)

The key distinguishing feature vs. SDT-v3 or Spikformer:
  - No transformer / attention: this is a pure spiking CNN.
  - Multi-step execution: the SAME static conv weights are applied at each
    of T spiking timesteps, with the LIF neuron updating its membrane across
    steps (membrane-potential unrolled over the time axis in the trace).
  - Surrogate gradient: the Heaviside spike threshold is replaced by a
    differentiable surrogate during backward (fast-sigmoid here).
  - This is the "spike-based ResNet" predecessor to the spiking transformer
    family, and shows clearly what T-step LIF unrolling looks like in a CNN.

The compact proxy uses tiny channels (16/32) and a short T=4 so the
unrolled graph renders quickly; the dynamics are identical at any T or width.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics._snn_neurons import LIFNeuron


class _ConvBNLIF(nn.Module):
    """Conv2d -> BN -> multi-step LIF, applied per timestep (shared weights over T).

    Input: (T, B, C_in, H, W).  Applies the conv+BN by folding T into B,
    then reshapes and passes through LIFNeuron (which unrolls over the T axis).
    Output: (T, B, C_out, H_out, W_out).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIFNeuron(beta=0.9, threshold=1.0)
        self.pool = nn.MaxPool2d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, H, W)
        time, b = x.shape[0], x.shape[1]
        y = self.bn(self.conv(x.flatten(0, 1)))  # (T*B, C_out, h, w)
        y = y.reshape(time, b, *y.shape[1:])  # (T, B, C_out, h, w)
        y = self.lif(y)  # surrogate-gradient LIF
        # pool per timestep
        yp = self.pool(y.flatten(0, 1))
        return yp.reshape(time, b, *yp.shape[1:])


class _SpikingFC(nn.Module):
    """Flatten -> Linear -> multi-step LIF.

    Input: (T, B, C).  Applies the linear over the feature axis, then LIF.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.lif = LIFNeuron(beta=0.9, threshold=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        y = self.linear(x)  # (T, B, out_dim)
        return self.lif(y)


class SpikingJellyLIFNet(nn.Module):
    """SpikingJelly tutorial LIF spiking convnet.

    Two spiking conv-BN-LIF-pool blocks -> flatten -> spiking FC-LIF ->
    linear output head averaged over T.  Captures the canonical SpikingJelly
    multi-step spiking-CNN graph with LIF membrane unrolled over T steps.
    """

    def __init__(
        self,
        in_ch: int = 3,
        ch1: int = 16,
        ch2: int = 32,
        fc_dim: int = 128,
        num_classes: int = 10,
        timesteps: int = 4,
        img_size: int = 32,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.block1 = _ConvBNLIF(in_ch, ch1, kernel=3, pool=2)
        self.block2 = _ConvBNLIF(ch1, ch2, kernel=3, pool=2)
        # After two 2x2 pools from img_size x img_size
        h_out = img_size // 4
        flat_dim = ch2 * h_out * h_out
        self.fc1 = _SpikingFC(flat_dim, fc_dim)
        self.head = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> repeat T times -> (T, B, C, H, W)
        x = x.unsqueeze(0).expand(self.timesteps, -1, -1, -1, -1)
        x = self.block1(x)  # (T, B, ch1, H/2, W/2)
        x = self.block2(x)  # (T, B, ch2, H/4, W/4)
        time, b = x.shape[0], x.shape[1]
        x = x.reshape(time, b, -1)  # (T, B, flat_dim)
        x = self.fc1(x)  # (T, B, fc_dim)
        logits = self.head(x)  # (T, B, num_classes)
        return logits.mean(dim=0)  # (B, num_classes)


def build_spikingjelly_lif() -> nn.Module:
    """Build SpikingJelly-style LIF spiking convnet (multi-step, surrogate gradient)."""
    return SpikingJellyLIFNet(
        in_ch=3, ch1=16, ch2=32, fc_dim=128, num_classes=10, timesteps=4, img_size=32
    )


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 32, 32)``; model repeats over T internally."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SpikingJelly LIF spiking convnet (multi-step surrogate-gradient SNN)",
        "build_spikingjelly_lif",
        "example_input",
        "2023",
        "DC",
    ),
]
