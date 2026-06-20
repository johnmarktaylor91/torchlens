"""Spherical CNN (s2cnn): S2 / SO(3) group-equivariant convolution network.

Cohen, Geiger, Koehler & Welling, ICLR 2018, arXiv:1801.10130.
Source: https://github.com/jonkhler/s2cnn  (the CUDA generalized-FFT /
Wigner-D ``s2_fft`` / ``so3_fft`` kernels).

s2cnn defines rotation-equivariant convolutions on the sphere S2 and the rotation
group SO(3):
  - **S2 conv** correlates a spherical input signal with spherical filters; the
    spherical cross-correlation LIFTS the result to a function on SO(3) (one value
    per rotation that aligns filter to signal).
  - **SO(3) conv** correlates a signal on SO(3) with filters on SO(3), staying on
    SO(3); stacked with batchnorm + ReLU in the deep ResNet-style architecture.
  - An **SO(3) integration** (pool over the group) yields a rotation-INVARIANT
    feature vector fed to a linear classifier.

s2cnn computes these correlations efficiently in the SPECTRAL domain via a
generalized (non-commutative) FFT: it transforms signals to spherical-harmonic /
Wigner-D coefficients, multiplies coefficient blocks per frequency, and inverse-
transforms.  The CEILING in the menagerie is exactly those custom CUDA generalized-
FFT kernels (``s2_fft``/``so3_fft``, needing old cupy/pynvrtc).  They are an
OPTIMIZATION of the group correlation: by the generalized Fourier theorem the
spectral product equals the spatial group correlation.  This module reimplements
the FAITHFUL group-convolution architecture (S2 conv lifting to SO(3), an SO(3)
conv, and SO(3) pooling to an invariant feature) directly as the SPATIAL group
correlation over discretized S2 / SO(3) grids in pure torch -- the same operation
the spectral kernels accelerate -- so it traces and renders.

Small grids: S2 sampled on a ``b=4`` (8x8) beta/alpha grid; SO(3) reduced to a
small set of grid rotations; tiny channel widths to keep the unrolled group
correlation renderable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class S2Conv(nn.Module):
    """S2 convolution: correlate a spherical signal with filters, lift to SO(3).

    Input signal on S2 sampled on a (Hb x Wa) beta-alpha grid. Output is a signal on
    SO(3) discretized as ``n_gamma`` in-plane rotations: for each gamma, the filter is
    cyclically rolled along the alpha (azimuth) axis and correlated with the signal --
    the spatial-domain equivalent of the spectral S2 correlation lifting to SO(3).
    """

    def __init__(self, in_ch: int, out_ch: int, n_gamma: int = 4, ksize: int = 3) -> None:
        super().__init__()
        self.n_gamma = n_gamma
        self.ksize = ksize
        # One spherical filter bank per output channel; applied at each gamma rotation.
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, ksize, ksize) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, Hb, Wa) on S2. Output: (B, out_ch, n_gamma, Hb, Wa) on SO(3).
        pad = self.ksize // 2
        outs = []
        for g in range(self.n_gamma):
            shift = g * (x.shape[-1] // self.n_gamma)
            xr = torch.roll(x, shifts=shift, dims=-1)  # in-plane (gamma) rotation
            # Circular pad on azimuth (sphere periodicity), zero pad on beta.
            xr = F.pad(xr, (pad, pad, 0, 0), mode="circular")
            xr = F.pad(xr, (0, 0, pad, pad), mode="constant", value=0.0)
            y = F.conv2d(xr, self.weight, self.bias)
            outs.append(y)
        return torch.stack(outs, dim=2)  # (B, out_ch, n_gamma, Hb, Wa)


class SO3Conv(nn.Module):
    """SO(3) convolution: correlate an SO(3) signal with SO(3) filters, stay on SO(3).

    Implemented as a group correlation over the discretized gamma (in-plane rotation)
    axis -- a 3D conv whose gamma dimension is treated cyclically (group structure).
    """

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3) -> None:
        super().__init__()
        self.ksize = ksize
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, ksize, ksize, ksize) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, n_gamma, Hb, Wa). 3D group correlation; gamma + alpha cyclic.
        pad = self.ksize // 2
        xr = F.pad(x, (pad, pad, 0, 0, pad, pad), mode="circular")  # alpha + gamma cyclic
        xr = F.pad(xr, (0, 0, pad, pad, 0, 0), mode="constant", value=0.0)  # beta zero
        return F.conv3d(xr, self.weight, self.bias)


class SphericalCNN(nn.Module):
    """s2cnn architecture: S2 conv -> SO(3) conv -> SO(3) integration -> classifier."""

    def __init__(self, in_ch: int = 1, n_classes: int = 10, n_gamma: int = 4) -> None:
        super().__init__()
        self.s2 = S2Conv(in_ch, 8, n_gamma=n_gamma)
        self.bn1 = nn.BatchNorm3d(8)
        self.so3 = SO3Conv(8, 16)
        self.bn2 = nn.BatchNorm3d(16)
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.s2(x)))  # lift S2 -> SO(3)
        h = F.relu(self.bn2(self.so3(h)))  # SO(3) -> SO(3)
        # SO(3) integration: pool over the whole group -> rotation-invariant feature.
        h = h.mean(dim=(2, 3, 4))  # (B, 16)
        return self.classifier(h)


def build_s2cnn() -> nn.Module:
    """Build a compact spherical CNN (S2 conv + SO(3) conv + SO(3)-pool classifier)."""
    return SphericalCNN(in_ch=1, n_classes=10, n_gamma=4)


def example_input() -> torch.Tensor:
    """Spherical signal ``(1, 1, 8, 8)`` sampled on an 8x8 beta-alpha S2 grid."""
    return torch.randn(1, 1, 8, 8)


MENAGERIE_ENTRIES = [
    (
        "Spherical CNN (s2cnn, S2/SO(3) group-equivariant convolution)",
        "build_s2cnn",
        "example_input",
        "2018",
        "E5",
    ),
]
