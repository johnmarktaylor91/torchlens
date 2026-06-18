"""Lenia and Flow-Lenia continuous cellular automata.

Paper: Chan 2019, "Lenia: Biology of Artificial Life."
Paper: Plantec et al. 2023, "Flow Lenia: Towards Open-Ended Evolution in Cellular Automata..."
Both modules use smooth convolutional fields and differentiable updates; Flow-Lenia
adds a simple mass-renormalized flow step in place of full reintegration tracking.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _ring_kernel(size: int = 15, radius: float = 0.45, width: float = 0.12) -> Tensor:
    """Create a normalized smooth radial Lenia kernel.

    Parameters
    ----------
    size:
        Odd kernel width and height.
    radius:
        Preferred ring radius in normalized coordinates.
    width:
        Gaussian shell width.

    Returns
    -------
    Tensor
        Kernel tensor with shape ``(1, 1, size, size)``.
    """
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist = torch.sqrt(xx.pow(2) + yy.pow(2))
    kernel = torch.exp(-0.5 * ((dist - radius) / width).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1.0e-6)
    return kernel[None, None, :, :]


class Lenia(nn.Module):
    """Smooth continuous-state cellular automaton."""

    def __init__(self, steps: int = 4, dt: float = 0.12) -> None:
        """Initialize Lenia update parameters.

        Parameters
        ----------
        steps:
            Number of CA update steps.
        dt:
            Integration step size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.mu = 0.28
        self.sigma = 0.055
        self.register_buffer("kernel", _ring_kernel())

    def _growth(self, field: Tensor) -> Tensor:
        """Compute Lenia's smooth growth response.

        Parameters
        ----------
        field:
            Convolved potential field.

        Returns
        -------
        Tensor
            Growth field in approximately ``[-1, 1]``.
        """
        return 2.0 * torch.exp(-0.5 * ((field - self.mu) / self.sigma).pow(2)) - 1.0

    def forward(self, state: Tensor) -> Tensor:
        """Run several Lenia updates.

        Parameters
        ----------
        state:
            Continuous CA state with shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Updated continuous CA state.
        """
        cells = state.clamp(0.0, 1.0)
        pad = self.kernel.shape[-1] // 2
        for _ in range(self.steps):
            field = F.conv2d(F.pad(cells, (pad, pad, pad, pad), mode="circular"), self.kernel)
            cells = (cells + self.dt * self._growth(field)).clamp(0.0, 1.0)
        return cells


class FlowLenia(nn.Module):
    """Mass-conserving Lenia-style flow automaton."""

    def __init__(self, steps: int = 4, dt: float = 0.10, flow_rate: float = 0.18) -> None:
        """Initialize Flow-Lenia update parameters.

        Parameters
        ----------
        steps:
            Number of CA update steps.
        dt:
            Integration step size.
        flow_rate:
            Strength of local mass redistribution.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.flow_rate = flow_rate
        self.mu = 0.30
        self.sigma = 0.07
        self.register_buffer("kernel", _ring_kernel())

    def _growth(self, field: Tensor) -> Tensor:
        """Compute smooth Flow-Lenia affinity growth.

        Parameters
        ----------
        field:
            Convolved affinity field.

        Returns
        -------
        Tensor
            Growth field.
        """
        return 2.0 * torch.exp(-0.5 * ((field - self.mu) / self.sigma).pow(2)) - 1.0

    def _flow(self, cells: Tensor, affinity: Tensor) -> Tensor:
        """Redistribute mass toward nearby high-affinity cells.

        Parameters
        ----------
        cells:
            Current mass field.
        affinity:
            Smooth affinity field.

        Returns
        -------
        Tensor
            Locally redistributed mass field.
        """
        north = torch.roll(cells, shifts=-1, dims=-2) * torch.roll(affinity, shifts=-1, dims=-2)
        south = torch.roll(cells, shifts=1, dims=-2) * torch.roll(affinity, shifts=1, dims=-2)
        east = torch.roll(cells, shifts=-1, dims=-1) * torch.roll(affinity, shifts=-1, dims=-1)
        west = torch.roll(cells, shifts=1, dims=-1) * torch.roll(affinity, shifts=1, dims=-1)
        incoming = 0.25 * (north + south + east + west)
        return (1.0 - self.flow_rate) * cells + self.flow_rate * incoming

    def forward(self, state: Tensor) -> Tensor:
        """Run mass-normalized Flow-Lenia updates.

        Parameters
        ----------
        state:
            Continuous CA state with shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Updated mass-normalized CA state.
        """
        cells = state.clamp(0.0, 1.0)
        initial_mass = cells.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        pad = self.kernel.shape[-1] // 2
        for _ in range(self.steps):
            field = F.conv2d(F.pad(cells, (pad, pad, pad, pad), mode="circular"), self.kernel)
            affinity = torch.sigmoid(4.0 * self._growth(field))
            cells = (cells + self.dt * (affinity - 0.5)).clamp_min(0.0)
            cells = self._flow(cells, affinity).clamp_min(0.0)
            mass = cells.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
            cells = cells * (initial_mass / mass)
        return cells.clamp(0.0, 1.0)


def build_lenia() -> nn.Module:
    """Build a small Lenia cellular automaton.

    Returns
    -------
    nn.Module
        Configured ``Lenia`` instance.
    """
    return Lenia()


def example_input_lenia() -> Tensor:
    """Create a continuous Lenia seed state.

    Returns
    -------
    Tensor
        Example state with shape ``(1, 1, 128, 128)``.
    """
    state = torch.zeros(1, 1, 128, 128)
    state[:, :, 56:72, 56:72] = torch.rand(1, 1, 16, 16) * 0.8
    return state


def build_flow_lenia() -> nn.Module:
    """Build a small Flow-Lenia cellular automaton.

    Returns
    -------
    nn.Module
        Configured ``FlowLenia`` instance.
    """
    return FlowLenia()


def example_input_flow_lenia() -> Tensor:
    """Create a continuous Flow-Lenia seed state.

    Returns
    -------
    Tensor
        Example state with shape ``(1, 1, 128, 128)``.
    """
    state = torch.zeros(1, 1, 128, 128)
    state[:, :, 52:76, 52:76] = torch.rand(1, 1, 24, 24) * 0.6
    return state


MENAGERIE_ENTRIES = [
    ("Lenia (Continuous Cellular Automaton)", "build_lenia", "example_input_lenia", "2019", "MB1"),
    (
        "Flow-Lenia (Mass-Conserving Lenia)",
        "build_flow_lenia",
        "example_input_flow_lenia",
        "2023",
        "MB1",
    ),
]
