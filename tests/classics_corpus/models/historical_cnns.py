"""Historical CNN/MLP architectures: CaffeNet, OverFeat, NetworkInNetwork, Maxout, Highway, Pi-Net.

A family of landmark architectures from 2012-2020 that defined modern deep learning.

---
CaffeNet-original-LRN (Jia et al., 2014 BVLC CaffeNet):
  Paper/Source: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
  An AlexNet-like network with subtle differences:
    - Single GPU (no model parallelism / channel splitting).
    - Pooling BEFORE LRN in layers 1 & 2 (AlexNet has LRN before pool).
    - Conv kernel sizes and strides otherwise match AlexNet.
  Architecture: conv1(11x11,s4) -> pool -> LRN -> conv2(5x5) -> pool -> LRN ->
                conv3(3x3) -> conv4(3x3) -> conv5(3x3) -> pool -> fc6 -> fc7 -> fc8
  Includes nn.LocalResponseNorm.

---
OverFeat-accurate (Sermanet et al., 2013):
  Paper: https://arxiv.org/abs/1312.6229
  Source: https://github.com/sermanet/OverFeat
  The "accurate" (large) model:
    Conv1: 7x7, stride 2, 96 filters; MaxPool 3x3 stride 3
    Conv2: 7x7, stride 1, 256 filters; MaxPool 3x3 stride 2
    Conv3: 3x3, stride 1, 512 filters
    Conv4: 3x3, stride 1, 512 filters
    Conv5: 3x3, stride 1, 1024 filters; MaxPool 3x3 stride 2
    FC6, FC7, FC8 (convolutional FCs)
  No LRN, no local response norm. Large first-layer stride (2) is the signature.

---
NetworkInNetwork-CIFAR (Lin et al., 2013):
  Paper: https://arxiv.org/abs/1312.4400
  Source: https://github.com/mavenlin/cuda-convnet/tree/master/NIN
  The mlpconv (Network-in-Network) block: conv + 1x1 conv + 1x1 conv (ReLU each),
  implementing a mini MLP over each receptive field location.
  3 NiN blocks + Global Average Pooling (no FC layer) = the signature design.

---
MaxoutNetwork-MNIST (Goodfellow et al., 2013):
  Paper: https://arxiv.org/abs/1302.4389
  Source: https://github.com/goodfeli/adversarial (reference; Theano)
  Maxout unit: Linear(d, k*out) -> reshape(k, out) -> max over k-pieces.
  Several maxout layers + softmax head.

---
HighwayNetwork-MLP (Srivastava et al., 2015):
  Paper: https://arxiv.org/abs/1505.00387
  Source: https://github.com/rupchap/implement-highway-networks
  Highway layer: y = H(x)*T(x) + x*(1-T(x))
    H(x) = ReLU(W_H x + b_H) -- transform
    T(x) = sigmoid(W_T x + b_T) -- gate  (bias init: -1 or -2 to start closed)
  Stack of 10 highway layers (preserving width = the depth-enabling design).

---
Pi-Net / Polynomial Networks (Chrysos et al., 2020, 2021):
  Paper: https://arxiv.org/abs/2006.13026  (Pi-Net CVPR 2020)
         https://arxiv.org/abs/2104.02699  (Polynomial Nets NeurIPS 2021)
  Source: https://github.com/grigorisg9gr/polynomial_nets
  CCP (Coupled CP decomposition) block: implements high-degree polynomial
    via recursive Hadamard products (elementwise):
      z_1 = A_1 @ x
      z_n = (A_n @ x) * C_{n-1}(z_{n-1}) + D_{n-1}(z_{n-1})  for n >= 2
  where A, C, D are linear transforms.  The recursive Hadamard product structure
  is the distinctive architectural primitive.
  Distinct from Pi-Sigma (1991) which is a product-of-sums network.

Faithful compact simplification: all models use small inputs and few layers/channels.
All trace+draw verified 2026-06-21.
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# CaffeNet with LRN
# ===========================================================================


class CaffeNetLRN(nn.Module):
    """BVLC CaffeNet (AlexNet variant): pool BEFORE LRN, single GPU, no channel split.

    Compact input: (1, 3, 64, 64).  Published input is 227x227; we scale channels
    to match but use 64x64 for fast tracing.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        # CaffeNet-specific: pool before LRN in blocks 1 and 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),  # pool BEFORE LRN (CaffeNet order)
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),  # pool BEFORE LRN
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc6 = nn.Sequential(nn.Linear(256 * 4, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc6(x)
        x = self.fc7(x)
        return self.fc8(x)


# ===========================================================================
# OverFeat-accurate
# ===========================================================================


class OverFeatAccurate(nn.Module):
    """OverFeat 'accurate' (large) model.

    Sermanet 2013.  No LRN.  Large first-layer stride (2).
    Conv layers 3-4-5 are 3x3; first two are 7x7.
    Compact input: (1, 3, 64, 64). Published input: 231x231.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: 7x7 stride 2
            nn.Conv2d(3, 96, 7, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=3),
            # Layer 2: 7x7 stride 1
            nn.Conv2d(96, 256, 7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # Layer 3: 3x3 stride 1
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Layer 4: 3x3
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Layer 5: 3x3 + pool
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        # FC layers as 1x1 convolutions (OverFeat trains sliding window as conv)
        self.fc6 = nn.Sequential(nn.Linear(1024 * 4, 3072), nn.ReLU(inplace=True), nn.Dropout())
        self.fc7 = nn.Sequential(nn.Linear(3072, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc6(x)
        x = self.fc7(x)
        return self.fc8(x)


# ===========================================================================
# NetworkInNetwork (NiN) for CIFAR
# ===========================================================================


class NiNBlock(nn.Module):
    """NiN mlpconv block: conv + 1x1 + 1x1 (each with ReLU).

    Implements a mini-MLP over each receptive field location.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        pad = (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NetworkInNetwork(nn.Module):
    """NiN for CIFAR-10: 3 NiN blocks + Global Average Pooling (no FC).

    Lin et al., 2013.  The signature: mlpconv (1x1 convs as MLP) + GAP.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.block1 = NiNBlock(3, 192, kernel=5)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.drop1 = nn.Dropout(p=0.5)

        self.block2 = NiNBlock(192, 160, kernel=5)
        self.block2b = NiNBlock(160, 96, kernel=1)
        self.pool2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.drop2 = nn.Dropout(p=0.5)

        self.block3 = NiNBlock(96, n_classes, kernel=3)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling (no FC)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.drop1(self.pool1(x))
        x = self.block2(x)
        x = self.drop2(self.pool2(self.block2b(x)))
        x = self.block3(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return x


# ===========================================================================
# Maxout Network for MNIST
# ===========================================================================


class MaxoutUnit(nn.Module):
    """Maxout unit: Linear(in, k*out) -> reshape -> max over k pieces.

    Goodfellow 2013. The piecewise-linear activation function.
    """

    def __init__(self, in_features: int, out_features: int, k: int = 5) -> None:
        super().__init__()
        self.k = k
        self.out_features = out_features
        self.lin = nn.Linear(in_features, k * out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)  # (B, k * out)
        h = h.view(h.size(0), self.k, self.out_features)  # (B, k, out)
        return h.max(dim=1).values  # (B, out)


class MaxoutNetwork(nn.Module):
    """Maxout MLP for MNIST (Goodfellow et al., 2013).

    2 maxout layers + softmax output.
    Input: (B, 784) flattened MNIST image.
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden: int = 240,
        k: int = 5,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.drop1 = nn.Dropout(p=0.2)
        self.maxout1 = MaxoutUnit(in_features, hidden, k=k)
        self.drop2 = nn.Dropout(p=0.5)
        self.maxout2 = MaxoutUnit(hidden, hidden, k=k)
        self.drop3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.maxout1(self.drop1(x))
        x = self.maxout2(self.drop2(x))
        return self.out(self.drop3(x))


# ===========================================================================
# Highway Network (feedforward MLP variant)
# ===========================================================================


class HighwayLayer(nn.Module):
    """One highway layer: y = H(x)*T(x) + x*(1-T(x)).

    H(x) = ReLU(W_H x + b_H)  -- transform gate
    T(x) = sigmoid(W_T x + b_T) -- carry gate (bias init: -2 for mostly-carry start)
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.W_H = nn.Linear(size, size)
        self.W_T = nn.Linear(size, size)
        # Init carry gate bias to -2 (mostly carry at init)
        nn.init.constant_(self.W_T.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = F.relu(self.W_H(x))
        T = torch.sigmoid(self.W_T(x))
        return H * T + x * (1.0 - T)


class HighwayNetwork(nn.Module):
    """Feedforward Highway Network: depth is enabled by the gating mechanism.

    Srivastava et al., 2015.  Stack of highway layers (no collapse in depth).
    Input projection + N highway layers + output projection.
    """

    def __init__(
        self,
        in_features: int = 50,
        size: int = 64,
        n_layers: int = 10,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, size), nn.ReLU(inplace=True))
        self.highways = nn.ModuleList([HighwayLayer(size) for _ in range(n_layers)])
        self.out = nn.Linear(size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for layer in self.highways:
            x = layer(x)
        return self.out(x)


# ===========================================================================
# Pi-Net: Polynomial Network (Chrysos et al., 2020)
# ===========================================================================


class CCPBlock(nn.Module):
    """Coupled CP (CCP) decomposition block -- one degree of the polynomial.

    Implements the recursive formula:
      z_1 = A_1 @ x
      z_n = (A_n @ x) * C_{n-1}(z_{n-1}) + z_{n-1}  for n >= 2

    where:
      A_n: input projection (Linear, from x to size)
      C_n: carry linear transform on z_{n-1}

    The Hadamard product (* elementwise) is the distinctive primitive:
    it enables high-degree polynomial interactions between the input and
    previously computed features.
    """

    def __init__(self, in_features: int, size: int) -> None:
        super().__init__()
        self.A = nn.Linear(in_features, size, bias=False)
        self.C = nn.Linear(size, size, bias=False)

    def forward(self, x: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        ax = self.A(x)  # (B, size)
        return ax * self.C(z_prev) + z_prev  # Hadamard product + residual


class PiNet(nn.Module):
    """Pi-Net polynomial network (Chrysos et al., 2020, 2021).

    CCP decomposition: degree-d polynomial via d-1 Hadamard-product stages.
    Distinct from Pi-Sigma network (1991, Shin & Ghosh).

    Args:
        in_features:  Input dimension.
        size:         Hidden state size.
        degree:       Polynomial degree (number of CCP stages + 1).
        n_classes:    Output classes.
    """

    def __init__(
        self,
        in_features: int = 64,
        size: int = 128,
        degree: int = 4,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.degree = degree
        # First-degree: z_1 = A_1 @ x
        self.A1 = nn.Linear(in_features, size, bias=False)
        # Higher degrees: CCP blocks
        self.blocks = nn.ModuleList([CCPBlock(in_features, size) for _ in range(degree - 1)])
        self.out = nn.Linear(size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.A1(x)  # degree-1 term
        for block in self.blocks:
            z = block(x, z)  # recursive Hadamard product stages
        return self.out(z)


# ===========================================================================
# Build functions
# ===========================================================================


def build_caffenet_lrn() -> nn.Module:
    return CaffeNetLRN(num_classes=1000)


def example_input_imagenet() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


def build_overfeat_accurate() -> nn.Module:
    return OverFeatAccurate(num_classes=1000)


def build_nin_cifar() -> nn.Module:
    return NetworkInNetwork(n_classes=10)


def example_input_cifar() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


def build_maxout_mnist() -> nn.Module:
    return MaxoutNetwork(in_features=784, hidden=240, k=5, n_classes=10)


def example_input_mnist() -> torch.Tensor:
    return torch.randn(1, 1, 28, 28)


def build_highway_mlp() -> nn.Module:
    return HighwayNetwork(in_features=50, size=64, n_layers=10, n_classes=10)


def example_input_highway() -> torch.Tensor:
    return torch.randn(1, 50)


def build_pinet() -> nn.Module:
    return PiNet(in_features=64, size=128, degree=4, n_classes=10)


def example_input_pinet() -> torch.Tensor:
    return torch.randn(1, 64)


MENAGERIE_ENTRIES = [
    (
        "CaffeNet-original-LRN (BVLC AlexNet variant: pool-before-LRN, single GPU)",
        "build_caffenet_lrn",
        "example_input_imagenet",
        "2014",
        "DC",
    ),
    (
        "OverFeat-accurate (Sermanet 2013: 7x7 large stride, no LRN, 6-conv accurate model)",
        "build_overfeat_accurate",
        "example_input_imagenet",
        "2013",
        "DC",
    ),
    (
        "NetworkInNetwork-CIFAR (Lin 2013: mlpconv blocks + global average pooling, no FC)",
        "build_nin_cifar",
        "example_input_cifar",
        "2013",
        "DC",
    ),
    (
        "MaxoutNetwork-MNIST (Goodfellow 2013: piecewise-linear maxout units)",
        "build_maxout_mnist",
        "example_input_mnist",
        "2013",
        "DC",
    ),
    (
        "HighwayNetwork-MLP (Srivastava 2015: gated highway layers enabling deep FFNs)",
        "build_highway_mlp",
        "example_input_highway",
        "2015",
        "DC",
    ),
    (
        "PiNet-PolynomialNetwork (Chrysos 2020: CCP recursive Hadamard-product polynomial)",
        "build_pinet",
        "example_input_pinet",
        "2020",
        "DC",
    ),
]
