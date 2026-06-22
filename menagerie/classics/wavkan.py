"""Wav-KAN: Wavelet Kolmogorov-Arnold Network.

Bozorgasl & Chen, 2024.
Paper: https://arxiv.org/abs/2405.12832
Source: https://github.com/zavareh1/Wav-KAN

A Kolmogorov-Arnold Network replaces fixed node activations + linear weights with
learnable activation functions on the *edges*.  Wav-KAN parameterizes each edge
function as a scaled-and-translated mother wavelet (Mexican-hat / Morlet / DoG),
so each KAN layer computes, for every (in, out) edge, a per-edge wavelet response
``w * psi((x - b) / a)`` summed over inputs, plus a base linear path.  Stacking
such layers gives the network.

This is a faithful random-init reimplementation of the Wav-KAN layer
(``KANLinear`` / ``WavKAN`` in the source), supporting the Mexican-hat,
Morlet, and DoG mother wavelets, with per-edge learnable scale/translation,
a wavelet-weight matrix, and a base (SiLU) linear path with BatchNorm — exactly
as in the published implementation.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """One Wav-KAN layer: per-edge wavelet activation + base linear path."""

    def __init__(
        self, in_features: int, out_features: int, wavelet_type: str = "mexican_hat"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Per-edge learnable scale (a) and translation (b).
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        # Wavelet-weight (combines per-edge wavelet responses) and base weight.
        self.wavelet_weights = nn.Parameter(torch.empty(out_features, in_features))
        self.weight1 = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=5**0.5)
        nn.init.kaiming_uniform_(self.weight1, a=5**0.5)

        self.base_activation = nn.SiLU()
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)  # (B, 1, in)
        else:
            x_expanded = x

        translation = self.translation.unsqueeze(0)  # (1, out, in)
        scale = self.scale.unsqueeze(0)  # (1, out, in)
        x_scaled = (x_expanded - translation) / scale  # (B, out, in)

        if self.wavelet_type == "mexican_hat":
            term1 = (x_scaled**2) - 1
            term2 = torch.exp(-0.5 * x_scaled**2)
            wavelet = (2 / (3**0.5 * torch.pi**0.25)) * term1 * term2
        elif self.wavelet_type == "morlet":
            omega0 = 5.0
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled**2)
            wavelet = envelope * real
        elif self.wavelet_type == "dog":
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled**2)
        else:
            raise ValueError(f"Unsupported wavelet_type={self.wavelet_type!r}")

        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0)
        return wavelet_weighted.sum(dim=2)  # (B, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wavelet_output = self.wavelet_transform(x)
        base_output = F.linear(self.base_activation(x), self.weight1)
        return self.bn(wavelet_output + base_output)


class WavKAN(nn.Module):
    """A stack of Wav-KAN layers."""

    def __init__(self, layers_hidden: List[int], wavelet_type: str = "mexican_hat") -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLinear(layers_hidden[i], layers_hidden[i + 1], wavelet_type)
                for i in range(len(layers_hidden) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def build() -> nn.Module:
    """Build a 3-layer Wav-KAN ``[16, 32, 8]`` with Mexican-hat wavelets."""
    return WavKAN([16, 32, 8], wavelet_type="mexican_hat")


def example_input() -> torch.Tensor:
    """Example feature batch ``(8, 16)`` for the Wav-KAN."""
    return torch.randn(8, 16)


MENAGERIE_ENTRIES = [
    (
        "Wav-KAN (wavelet Kolmogorov-Arnold Network)",
        "build",
        "example_input",
        "2024",
        "DC",
    ),
]
