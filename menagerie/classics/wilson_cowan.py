"""Wilson-Cowan population model, 1972.

Wilson and Cowan described coupled excitatory and inhibitory neural-mass rate equations.
This module unrolls the E/I dynamics under an external drive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WilsonCowanModel(nn.Module):
    """Coupled excitatory/inhibitory population dynamics."""

    def __init__(self, dt: float = 0.1, tau_e: float = 1.0, tau_i: float = 2.0) -> None:
        """Initialize Wilson-Cowan constants.

        Parameters
        ----------
        dt
            Euler step size.
        tau_e
            Excitatory time constant.
        tau_i
            Inhibitory time constant.
        """
        super().__init__()
        self.dt = dt
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.w_ee = nn.Parameter(torch.tensor(10.0))
        self.w_ei = nn.Parameter(torch.tensor(8.0))
        self.w_ie = nn.Parameter(torch.tensor(10.0))
        self.w_ii = nn.Parameter(torch.tensor(2.0))

    def _gain(self, x: Tensor) -> Tensor:
        """Apply the sigmoidal population gain.

        Parameters
        ----------
        x
            Synaptic input.

        Returns
        -------
        Tensor
            Firing-rate response.
        """
        return torch.sigmoid(x - 2.0)

    def forward(self, drive: Tensor) -> Tensor:
        """Integrate E/I rates for an external-drive sequence.

        Parameters
        ----------
        drive
            External drive of shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Rate trace of shape ``(batch, time, 2)``.
        """
        e = drive.new_full((drive.shape[0],), 0.1)
        i = drive.new_full((drive.shape[0],), 0.1)
        states: list[Tensor] = []
        for t in range(drive.shape[1]):
            e_input = self.w_ee * e - self.w_ei * i + drive[:, t]
            i_input = self.w_ie * e - self.w_ii * i
            de = (-e + (1.0 - e) * self._gain(e_input)) / self.tau_e
            di = (-i + (1.0 - i) * self._gain(i_input)) / self.tau_i
            e = e + self.dt * de
            i = i + self.dt * di
            states.append(torch.stack((e, i), dim=-1))
        return torch.stack(states, dim=1)


def build() -> nn.Module:
    """Build a Wilson-Cowan model.

    Returns
    -------
    nn.Module
        Wilson-Cowan neural-mass module.
    """
    return WilsonCowanModel()


def example_input() -> Tensor:
    """Return a float32 drive sequence.

    Returns
    -------
    Tensor
        External drive of shape ``(2, 20)``.
    """
    return torch.full((2, 20), 0.8, dtype=torch.float32)
