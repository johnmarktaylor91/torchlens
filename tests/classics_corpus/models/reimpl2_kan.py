"""Compact Kolmogorov-Arnold Network classics.

This module implements dependency-free KAN targets using learnable univariate
edge functions represented by triangular spline bases.  Variants include pykan
aliases, MultKAN multiplication nodes, convolutional KAN, and KAN-Mixer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """KAN layer with learnable spline functions on edges."""

    def __init__(self, in_features: int, out_features: int, grid: int = 7) -> None:
        """Initialize spline coefficients and residual base weights.

        Parameters
        ----------
        in_features:
            Input feature count.
        out_features:
            Output feature count.
        grid:
            Number of spline control points.
        """

        super().__init__()
        self.grid = grid
        self.coeff = nn.Parameter(torch.randn(out_features, in_features, grid) * 0.05)
        self.base = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        knots = torch.linspace(-1.0, 1.0, grid)
        self.register_buffer("knots", knots)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate triangular spline basis functions.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Basis tensor with an appended grid dimension.
        """

        spacing = 2.0 / max(1, self.grid - 1)
        return (1.0 - (x.unsqueeze(-1).clamp(-1.2, 1.2) - self.knots).abs() / spacing).clamp_min(
            0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply summed learnable edge functions.

        Parameters
        ----------
        x:
            Input tensor ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor ``(..., out_features)``.
        """

        basis = self._basis(torch.tanh(x))
        spline = torch.einsum("...ig,oig->...o", basis, self.coeff)
        base = F.silu(x) @ self.base.t()
        return spline + base


class KANCompact(nn.Module):
    """Compact feed-forward Kolmogorov-Arnold Network."""

    def __init__(self, in_features: int = 8, hidden: int = 16, out_features: int = 4) -> None:
        """Initialize two KAN layers.

        Parameters
        ----------
        in_features:
            Input feature count.
        hidden:
            Hidden feature count.
        out_features:
            Output feature count.
        """

        super().__init__()
        self.net = nn.Sequential(KANLayer(in_features, hidden), KANLayer(hidden, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the KAN.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Output features.
        """

        return self.net(x)


class MultKANCompact(nn.Module):
    """KAN with explicit multiplication nodes between KAN layers."""

    def __init__(self, in_features: int = 8, hidden: int = 12, out_features: int = 4) -> None:
        """Initialize additive KAN paths and multiplicative nodes.

        Parameters
        ----------
        in_features:
            Input feature count.
        hidden:
            Hidden additive units.
        out_features:
            Output feature count.
        """

        super().__init__()
        self.first = KANLayer(in_features, hidden)
        self.second = KANLayer(hidden + hidden // 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate additive and multiplicative hidden features.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Output features.
        """

        h = self.first(x)
        products = h[:, 0::2] * h[:, 1::2]
        return self.second(torch.cat([h, products], dim=-1))


class ConvKANCompact(nn.Module):
    """Convolutional KAN using KAN edge functions on image patches."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize patch KAN and classifier.

        Parameters
        ----------
        classes:
            Output class count.
        """

        super().__init__()
        self.patch_kan = KANLayer(3 * 3, 12)
        self.point = nn.Conv2d(12, 16, 1)
        self.head = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images with Conv-KAN patch activations.

        Parameters
        ----------
        x:
            Grayscale image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        patches = F.unfold(x, kernel_size=3, padding=1).transpose(1, 2)
        feats = self.patch_kan(patches).transpose(1, 2).view(x.shape[0], 12, x.shape[2], x.shape[3])
        pooled = F.avg_pool2d(F.relu(self.point(feats)), 4).mean(dim=(2, 3))
        return self.head(pooled)


class KANMixerCompact(nn.Module):
    """Vision KAN-Mixer with KAN token and channel mixing."""

    def __init__(self, patches: int = 16, channels: int = 16, classes: int = 5) -> None:
        """Initialize patch embedding and mixer layers.

        Parameters
        ----------
        patches:
            Number of image patches.
        channels:
            Per-patch channel width.
        classes:
            Output class count.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, channels, 7, stride=7)
        self.token_mix = KANLayer(patches, patches)
        self.channel_mix = KANLayer(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images with KAN token/channel mixers.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.token_mix(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + self.channel_mix(tokens)
        return self.head(self.norm(tokens).mean(dim=1))


def build_kan() -> nn.Module:
    """Build a compact KAN.

    Returns
    -------
    nn.Module
        KAN model.
    """

    return KANCompact()


def build_multkan() -> nn.Module:
    """Build a compact MultKAN.

    Returns
    -------
    nn.Module
        MultKAN model.
    """

    return MultKANCompact()


def build_conv_kan() -> nn.Module:
    """Build a compact convolutional KAN.

    Returns
    -------
    nn.Module
        Conv-KAN model.
    """

    return ConvKANCompact()


def build_kan_mixer() -> nn.Module:
    """Build a compact KAN-Mixer.

    Returns
    -------
    nn.Module
        KAN-Mixer model.
    """

    return KANMixerCompact()


def example_features() -> torch.Tensor:
    """Create tabular feature input.

    Returns
    -------
    torch.Tensor
        Feature tensor ``(2, 8)``.
    """

    return torch.randn(2, 8)


def example_image() -> torch.Tensor:
    """Create grayscale image input.

    Returns
    -------
    torch.Tensor
        Image tensor ``(2, 1, 28, 28)``.
    """

    return torch.randn(2, 1, 28, 28)


MENAGERIE_ENTRIES = [
    ("KAN", "build_kan", "example_features", "2024", "E5"),
    ("kan_pykan", "build_kan", "example_features", "2024", "E5"),
    ("pykan_kan", "build_kan", "example_features", "2024", "E5"),
    ("pykan:KAN", "build_kan", "example_features", "2024", "E5"),
    ("pykan_multkan", "build_multkan", "example_features", "2024", "E5"),
    ("conv_kan", "build_conv_kan", "example_image", "2024", "E5"),
    ("kan_mixer", "build_kan_mixer", "example_image", "2024", "E5"),
]
