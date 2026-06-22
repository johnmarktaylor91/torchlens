"""Liquid Time-Constant and Closed-form Continuous-time recurrent cells.

Hasani et al. introduced Liquid Time-Constant (LTC) networks as neural ODE-style
recurrent cells with input-dependent conductances and time constants.  Lechner
et al. / Hasani et al. later derived Closed-form Continuous-time (CfC) cells
that approximate the LTC state flow with solver-free gated closed-form updates.

This module provides compact random-init versions of the inference cells for the
dependency-gated NCPS/CfC/LTC target names.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CfCCell(nn.Module):
    """Closed-form Continuous-time recurrent cell."""

    def __init__(self, input_size: int = 8, hidden_size: int = 16) -> None:
        """Initialize CfC gates.

        Parameters
        ----------
        input_size:
            Input feature dimension.
        hidden_size:
            Hidden-state dimension.
        """

        super().__init__()
        merged = input_size + hidden_size
        self.ff1 = nn.Linear(merged, hidden_size)
        self.ff2 = nn.Linear(merged, hidden_size)
        self.time_a = nn.Linear(merged, hidden_size)
        self.time_b = nn.Linear(merged, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Apply one solver-free CfC update.

        Parameters
        ----------
        x:
            Input features for one step.
        h:
            Previous hidden state.
        dt:
            Positive elapsed time for the step, shape ``(B, 1)``.

        Returns
        -------
        torch.Tensor
            Updated hidden state.
        """

        z = torch.cat([x, h], dim=-1)
        ff1 = torch.tanh(self.ff1(z))
        ff2 = torch.tanh(self.ff2(z))
        gate = torch.sigmoid(self.time_a(z) * dt + self.time_b(z))
        return ff1 * (1.0 - gate) + ff2 * gate


class LTCCell(nn.Module):
    """Liquid Time-Constant recurrent cell with conductance dynamics."""

    def __init__(self, input_size: int = 8, hidden_size: int = 16) -> None:
        """Initialize LTC conductance projections.

        Parameters
        ----------
        input_size:
            Input feature dimension.
        hidden_size:
            Hidden-state dimension.
        """

        super().__init__()
        merged = input_size + hidden_size
        self.g = nn.Linear(merged, hidden_size)
        self.e = nn.Linear(merged, hidden_size)
        self.leak = nn.Parameter(torch.ones(hidden_size))
        self.cm = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Apply one explicit compact LTC integration step.

        Parameters
        ----------
        x:
            Input features for one step.
        h:
            Previous hidden state.
        dt:
            Positive elapsed time for the step, shape ``(B, 1)``.

        Returns
        -------
        torch.Tensor
            Updated hidden state.
        """

        z = torch.cat([x, h], dim=-1)
        conductance = torch.sigmoid(self.g(z))
        reversal = torch.tanh(self.e(z))
        numerator = self.cm * h + dt * (conductance * reversal)
        denominator = self.cm + dt * (self.leak.abs() + conductance)
        return numerator / denominator.clamp_min(1e-6)


class LiquidSequence(nn.Module):
    """Sequence wrapper around a liquid recurrent cell."""

    def __init__(self, cell: nn.Module, hidden_size: int = 16, classes: int = 4) -> None:
        """Initialize recurrent readout model.

        Parameters
        ----------
        cell:
            Recurrent cell.
        hidden_size:
            Hidden-state dimension.
        classes:
            Output class count.
        """

        super().__init__()
        self.cell = cell
        self.hidden_size = hidden_size
        self.readout = nn.Linear(hidden_size, classes)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run the recurrent liquid model over a sequence.

        Parameters
        ----------
        inputs:
            Tuple ``(x, dt)`` with ``x`` shaped ``(B, T, F)`` and ``dt`` shaped
            ``(B, T, 1)``.

        Returns
        -------
        torch.Tensor
            Final-step logits.
        """

        x, dt = inputs
        h = x.new_zeros(x.shape[0], self.hidden_size)
        for step in range(x.shape[1]):
            h = self.cell(x[:, step], h, dt[:, step])
        return self.readout(h)


def build_cfc() -> nn.Module:
    """Build a compact CfC sequence classifier.

    Returns
    -------
    nn.Module
        CfC sequence model.
    """

    return LiquidSequence(CfCCell())


def build_ltc() -> nn.Module:
    """Build a compact LTC sequence classifier.

    Returns
    -------
    nn.Module
        LTC sequence model.
    """

    return LiquidSequence(LTCCell())


def build_liquid_time_constant_cell() -> nn.Module:
    """Build the liquid time-constant cell target.

    Returns
    -------
    nn.Module
        LTC sequence model.
    """

    return build_ltc()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sequence features and elapsed times.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Feature and ``dt`` tensors.
    """

    x = torch.randn(1, 6, 8)
    dt = torch.rand(1, 6, 1).clamp_min(0.05)
    return x, dt


MENAGERIE_ENTRIES = [
    ("CfC liquid closed-form continuous-time cell", "build_cfc", "example_input", "2022", "E6"),
    ("LTC liquid time-constant recurrent cell", "build_ltc", "example_input", "2020", "E6"),
    (
        "Liquid Time-Constant cell (NCPS-style)",
        "build_liquid_time_constant_cell",
        "example_input",
        "2020",
        "E6",
    ),
]
