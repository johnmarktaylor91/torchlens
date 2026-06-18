"""Jansen-Rit cortical column neural mass, 1995, Jansen and Rit.

Paper: "Electroencephalogram and visual evoked potential generation in a mathematical
model of coupled cortical columns." Three coupled populations with second-order
synaptic filters generate a pyramidal-potential trace under external drive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Jansen-Rit Cortical Column Neural Mass", "build", "example_input", "1995", "DB")
]


class JansenRitNeuralMass(nn.Module):
    """Trace-clean Euler simulation of a Jansen-Rit cortical column."""

    def __init__(self, steps: int = 12, dt: float = 0.002) -> None:
        """Initialize neural-mass integration settings.

        Parameters
        ----------
        steps
            Number of Euler integration steps.
        dt
            Integration step size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.gain_e = 3.25
        self.gain_i = 22.0
        self.rate_e = 100.0
        self.rate_i = 50.0

    def forward(self, ext_drive: Tensor) -> Tensor:
        """Integrate the cortical-column state under external drive.

        Parameters
        ----------
        ext_drive
            Drive tensor of shape ``(batch, 6)``.

        Returns
        -------
        Tensor
            Pyramidal trace over integration steps.
        """
        y0, y1, y2, v0, v1, v2 = torch.unbind(ext_drive, dim=-1)
        trace: list[Tensor] = []
        for _ in range(self.steps):
            firing = torch.sigmoid(y1 - y2)
            dy0 = v0
            dv0 = self.gain_e * self.rate_e * firing - 2.0 * self.rate_e * v0 - self.rate_e**2 * y0
            dy1 = v1
            dv1 = self.gain_e * self.rate_e * (ext_drive[:, 0] + torch.sigmoid(y0))
            dv1 = dv1 - 2.0 * self.rate_e * v1 - self.rate_e**2 * y1
            dy2 = v2
            dv2 = self.gain_i * self.rate_i * torch.sigmoid(y0) - 2.0 * self.rate_i * v2
            dv2 = dv2 - self.rate_i**2 * y2
            y0 = y0 + self.dt * dy0
            v0 = v0 + self.dt * dv0
            y1 = y1 + self.dt * dy1
            v1 = v1 + self.dt * dv1
            y2 = y2 + self.dt * dy2
            v2 = v2 + self.dt * dv2
            trace.append((y1 - y2).unsqueeze(-1))
        return torch.cat(trace, dim=-1)


def build() -> nn.Module:
    """Build a small Jansen-Rit neural-mass module.

    Returns
    -------
    nn.Module
        Configured ``JansenRitNeuralMass`` instance.
    """
    return JansenRitNeuralMass()


def example_input() -> Tensor:
    """Return an initial neural-mass drive/state example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 6)``.
    """
    return torch.randn(1, 6) * 0.01
