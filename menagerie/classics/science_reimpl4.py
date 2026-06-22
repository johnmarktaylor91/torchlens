"""Compact faithful scientific neural architecture classics.

Paper: Deep Ritz Method, Deep Galerkin Method, Behler-Parrinello HDNNP, HIP-NN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Residual multilayer perceptron used as a trial function."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 1) -> None:
        """Initialize the residual MLP.

        Parameters
        ----------
        in_dim:
            Input dimension.
        hidden:
            Hidden width.
        out_dim:
            Output dimension.
        """

        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual MLP.

        Parameters
        ----------
        x:
            Input coordinates.

        Returns
        -------
        torch.Tensor
            Network output.
        """

        h = torch.tanh(self.inp(x))
        for layer in self.layers:
            h = h + torch.tanh(layer(h))
        return self.out(h)


class DeepRitzNet(nn.Module):
    """Deep Ritz trial function returning solution and variational energy density."""

    def __init__(self, dim: int = 3) -> None:
        """Initialize Deep Ritz model.

        Parameters
        ----------
        dim:
            Spatial dimension.
        """

        super().__init__()
        self.trial = ResidualMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute trial solution and Ritz energy-density surrogate.

        Parameters
        ----------
        x:
            Spatial coordinates.

        Returns
        -------
        torch.Tensor
            Concatenated solution and energy density.
        """

        u = self.trial(x)
        eps = 1e-2
        grads = []
        eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        for axis in range(x.shape[-1]):
            delta = eps * eye[axis]
            grads.append((self.trial(x + delta) - self.trial(x - delta)) / (2 * eps))
        grad = torch.cat(grads, dim=-1)
        forcing = torch.sin(x).sum(dim=-1, keepdim=True)
        energy = 0.5 * grad.pow(2).sum(dim=-1, keepdim=True) - forcing * u
        return torch.cat([u, energy], dim=-1)


class DGMLayer(nn.Module):
    """Sirignano-Spiliopoulos DGM gated layer."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        """Initialize the DGM layer.

        Parameters
        ----------
        in_dim:
            Coordinate dimension.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.z = nn.Linear(in_dim + hidden, hidden)
        self.g = nn.Linear(in_dim + hidden, hidden)
        self.r = nn.Linear(in_dim + hidden, hidden)
        self.h = nn.Linear(in_dim + hidden, hidden)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Apply gated DGM state update.

        Parameters
        ----------
        x:
            Coordinates.
        s:
            Previous DGM state.

        Returns
        -------
        torch.Tensor
            Updated state.
        """

        xs = torch.cat([x, s], dim=-1)
        z = torch.sigmoid(self.z(xs))
        g = torch.sigmoid(self.g(xs))
        r = torch.sigmoid(self.r(xs))
        h = torch.tanh(self.h(torch.cat([x, s * r], dim=-1)))
        return (1 - g) * h + z * s


class DeepGalerkinNet(nn.Module):
    """Deep Galerkin Method network with gated DGM layers."""

    def __init__(self, dim: int = 4, hidden: int = 32) -> None:
        """Initialize DGM model.

        Parameters
        ----------
        dim:
            Time-plus-space coordinate dimension.
        hidden:
            Hidden width.
        """

        super().__init__()
        self.input = nn.Linear(dim, hidden)
        self.layers = nn.ModuleList([DGMLayer(dim, hidden) for _ in range(3)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate solution and a mesh-free PDE residual surrogate.

        Parameters
        ----------
        x:
            Time/space coordinates.

        Returns
        -------
        torch.Tensor
            Solution and residual.
        """

        s = torch.tanh(self.input(x))
        for layer in self.layers:
            s = layer(x, s)
        u = self.out(s)
        eps = 1e-2
        grads = []
        eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        for axis in range(x.shape[-1]):
            delta = eps * eye[axis]
            sp = torch.tanh(self.input(x + delta))
            sm = torch.tanh(self.input(x - delta))
            for layer in self.layers:
                sp = layer(x + delta, sp)
                sm = layer(x - delta, sm)
            grads.append((self.out(sp) - self.out(sm)) / (2 * eps))
        grad = torch.cat(grads, dim=-1)
        residual = grad[:, :1] + grad[:, 1:].pow(2).sum(dim=-1, keepdim=True)
        return torch.cat([u, residual], dim=-1)


class BehlerParrinelloNet(nn.Module):
    """High-dimensional neural network potential with atom-centered symmetry functions."""

    def __init__(self, species: int = 3, hidden: int = 24) -> None:
        """Initialize BP-HDNNP.

        Parameters
        ----------
        species:
            Number of atom species.
        hidden:
            Atomic network hidden width.
        """

        super().__init__()
        self.embedding = nn.Embedding(species, 4)
        self.atomic_net = nn.Sequential(
            nn.Linear(4 + 4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict molecular energy from species and coordinates.

        Parameters
        ----------
        data:
            Tuple of species ids ``(batch, atoms)`` and coordinates ``(batch, atoms, 3)``.

        Returns
        -------
        torch.Tensor
            Total energy.
        """

        species, pos = data
        dist = torch.cdist(pos, pos).clamp_min(1e-6)
        cutoff = 0.5 * (torch.cos(dist.clamp_max(4.0) * torch.pi / 4.0) + 1.0)
        radial = torch.stack(
            [
                torch.exp(-eta * (dist - shift).pow(2)) * cutoff
                for eta, shift in [(0.5, 0.0), (1.0, 1.0), (2.0, 2.0), (4.0, 3.0)]
            ],
            dim=-1,
        ).sum(dim=2)
        features = torch.cat([self.embedding(species), radial], dim=-1)
        return self.atomic_net(features).sum(dim=1)


class HIPNNNet(nn.Module):
    """Hierarchically interacting particle neural network."""

    def __init__(self, species: int = 3, hidden: int = 24, orders: int = 3) -> None:
        """Initialize HIP-NN.

        Parameters
        ----------
        species:
            Number of atom species.
        hidden:
            Hidden width.
        orders:
            Number of many-body hierarchy orders.
        """

        super().__init__()
        self.embed = nn.Embedding(species, hidden)
        self.interactions = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(orders)])
        self.energy_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(orders)])

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict energy as sum over hierarchical atomic contributions.

        Parameters
        ----------
        data:
            Tuple of species ids and coordinates.

        Returns
        -------
        torch.Tensor
            Total molecular energy.
        """

        species, pos = data
        h = self.embed(species)
        dist = torch.cdist(pos, pos).clamp_min(1e-6)
        weights = torch.exp(-dist)
        total = torch.zeros(species.shape[0], 1, device=pos.device)
        for interaction, head in zip(self.interactions, self.energy_heads, strict=True):
            msg = torch.matmul(weights, h) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            h = F.softplus(interaction(h + msg))
            total = total + head(h).sum(dim=1)
        return total


def build_deep_ritz() -> nn.Module:
    """Build Deep Ritz Method model.

    Returns
    -------
    nn.Module
        Compact Deep Ritz model.
    """

    return DeepRitzNet()


def build_deep_galerkin() -> nn.Module:
    """Build Deep Galerkin Method model.

    Returns
    -------
    nn.Module
        Compact DGM model.
    """

    return DeepGalerkinNet()


def build_hdnnp() -> nn.Module:
    """Build Behler-Parrinello HDNNP.

    Returns
    -------
    nn.Module
        Compact HDNNP model.
    """

    return BehlerParrinelloNet()


def build_hipnn() -> nn.Module:
    """Build HIP-NN.

    Returns
    -------
    nn.Module
        Compact HIP-NN model.
    """

    return HIPNNNet()


def example_points() -> torch.Tensor:
    """Return PDE collocation points.

    Returns
    -------
    torch.Tensor
        Coordinate tensor.
    """

    return torch.randn(8, 3)


def example_time_points() -> torch.Tensor:
    """Return time-space collocation points.

    Returns
    -------
    torch.Tensor
        Coordinate tensor.
    """

    return torch.randn(8, 4)


def example_molecule() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a compact molecule.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Species ids and coordinates.
    """

    return torch.randint(0, 3, (1, 5)), torch.randn(1, 5, 3)


MENAGERIE_ENTRIES = [
    ("Deep Ritz Method", "build_deep_ritz", "example_points", "2018", "scientific/pde"),
    (
        "Deep Galerkin Method",
        "build_deep_galerkin",
        "example_time_points",
        "2017",
        "scientific/pde",
    ),
    ("HDNNP/Behler-Parrinello", "build_hdnnp", "example_molecule", "2007", "molecular"),
    ("HIP-NN", "build_hipnn", "example_molecule", "2017", "molecular"),
]
