"""Ising-machine-inspired neural nets, 2016-2018, optical, p-bit, and quantum variants.

Paper: Yamamoto 2016, "Coherent Ising machines"; Camsari 2017, "p-bits";
Benedetti 2018, "Quantum-assisted learning of hardware-embedded probabilistic graphical
models." These are classical differentiable approximations of hardware or quantum
substrates, omitting real optical, stochastic-bit, and quantum sampling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PbitNetwork(nn.Module):
    """Differentiable probabilistic-bit relaxation with sigmoid magnetizations."""

    def __init__(self, n_spins: int = 32, steps: int = 5) -> None:
        """Initialize symmetric Ising couplings.

        Parameters
        ----------
        n_spins
            Number of p-bit spins.
        steps
            Number of analog relaxation steps.
        """
        super().__init__()
        raw = torch.randn(n_spins, n_spins) * 0.12
        coupling = (raw + raw.T) * 0.5
        coupling.fill_diagonal_(0.0)
        self.register_buffer("coupling", coupling)
        self.bias = nn.Parameter(torch.zeros(n_spins))
        self.steps = steps

    def forward(self, spins: Tensor) -> Tensor:
        """Relax p-bit magnetizations with differentiable sigmoid updates.

        Parameters
        ----------
        spins
            Initial spin-like tensor of shape ``(batch, 32)``.

        Returns
        -------
        Tensor
            Relaxed magnetizations in ``[-1, 1]``.
        """
        state = spins.clamp(-1.0, 1.0)
        for _ in range(self.steps):
            current = state @ self.coupling + self.bias
            state = 2.0 * torch.sigmoid(2.0 * current) - 1.0
        return state


class CoherentIsingMachine(nn.Module):
    """Continuous OPO-amplitude Ising optimizer surrogate."""

    def __init__(self, n_spins: int = 32, steps: int = 6, dt: float = 0.15) -> None:
        """Initialize optical-parametric-oscillator coupling dynamics.

        Parameters
        ----------
        n_spins
            Number of oscillator amplitudes.
        steps
            Number of Euler integration steps.
        dt
            Integration step size.
        """
        super().__init__()
        raw = torch.randn(n_spins, n_spins) * 0.08
        coupling = (raw + raw.T) * 0.5
        coupling.fill_diagonal_(0.0)
        self.register_buffer("coupling", coupling)
        self.register_buffer("pump", torch.linspace(0.6, 1.25, steps))
        self.steps = steps
        self.dt = dt

    def forward(self, amplitudes: Tensor) -> Tensor:
        """Integrate gain-saturated coherent Ising machine dynamics.

        Parameters
        ----------
        amplitudes
            Initial oscillator amplitudes with shape ``(batch, 32)``.

        Returns
        -------
        Tensor
            Final soft spin amplitudes.
        """
        state = amplitudes
        for step in range(self.steps):
            pump = self.pump[step]
            delta = (pump - 1.0) * state - state.pow(3) + state @ self.coupling
            state = state + self.dt * delta
        return torch.tanh(state)


class QuantumBoltzmannMachine(nn.Module):
    """Classical energy-net surrogate for a quantum Boltzmann machine."""

    def __init__(self, n_visible: int = 16, n_hidden: int = 12) -> None:
        """Initialize visible-hidden energy parameters and transverse surrogate.

        Parameters
        ----------
        n_visible
            Number of visible units.
        n_hidden
            Number of hidden units.
        """
        super().__init__()
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))
        self.weight = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.15)
        self.transverse = nn.Parameter(torch.full((n_hidden,), 0.2))

    def forward(self, visible: Tensor) -> Tensor:
        """Return free-energy and hidden mean-field features.

        Parameters
        ----------
        visible
            Visible state tensor with shape ``(batch, 16)``.

        Returns
        -------
        Tensor
            Concatenated free energy and hidden probabilities.
        """
        v = torch.sigmoid(visible)
        field = v @ self.weight + self.hidden_bias
        hidden = torch.sigmoid(field)
        classical = -(v * self.visible_bias).sum(dim=-1, keepdim=True)
        coupling = -torch.sum(hidden * field, dim=-1, keepdim=True)
        transverse = -torch.sum(
            torch.sqrt(hidden * (1.0 - hidden) + 1.0e-6) * self.transverse, dim=-1, keepdim=True
        )
        return torch.cat((classical + coupling + transverse, hidden), dim=-1)


MENAGERIE_ENTRIES = [
    ("P-bit Network (Probabilistic Spin Logic)", "build_pbit", "example_input_pbit", "2017", "DA"),
    ("Coherent Ising Machine (CIM)", "build_cim", "example_input_cim", "2016", "DA"),
    ("Quantum Boltzmann Machine (QBM)", "build_qbm", "example_input_qbm", "2018", "DA"),
]


def build_pbit() -> nn.Module:
    """Build a p-bit network.

    Returns
    -------
    nn.Module
        Configured p-bit module.
    """
    return PbitNetwork()


def example_input_pbit() -> Tensor:
    """Create example p-bit spin inputs.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 32)``.
    """
    return torch.randn(1, 32)


def build_cim() -> nn.Module:
    """Build a coherent Ising machine surrogate.

    Returns
    -------
    nn.Module
        Configured CIM module.
    """
    return CoherentIsingMachine()


def example_input_cim() -> Tensor:
    """Create example oscillator amplitudes.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 32)``.
    """
    return torch.randn(1, 32) * 0.1


def build_qbm() -> nn.Module:
    """Build a quantum Boltzmann machine surrogate.

    Returns
    -------
    nn.Module
        Configured QBM module.
    """
    return QuantumBoltzmannMachine()


def example_input_qbm() -> Tensor:
    """Create example visible states.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 16)``.
    """
    return torch.randn(1, 16)
