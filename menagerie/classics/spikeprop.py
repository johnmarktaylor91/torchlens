"""SpikeProp, 2000, Bohte, Kok, and La Poutre.

Paper: Bohte et al. 2000, "SpikeProp: Backpropagation for networks of spiking
neurons." Feedforward synapses with delays form alpha-kernel postsynaptic
potentials; first output spike times are approximated differentiably here by a
soft minimum over threshold-crossing evidence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("SpikeProp (backprop through spike times)", "build", "example_input", "2000", "CF")
]


class SpikeProp(nn.Module):
    """Differentiable approximation of SpikeProp first-spike timing."""

    def __init__(self, n_in: int = 4, n_out: int = 3, n_grid: int = 16) -> None:
        """Initialize weights, delays, and time grid.

        Parameters
        ----------
        n_in
            Number of input spike times.
        n_out
            Number of output neurons.
        n_grid
            Number of candidate output time samples.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out) * 0.4)
        self.delay = nn.Parameter(torch.rand(n_in, n_out) * 0.5)
        self.register_buffer("time_grid", torch.linspace(0.0, 3.0, n_grid))

    def forward(self, spike_times: Tensor) -> Tensor:
        """Estimate output first-spike times from input spike times.

        Parameters
        ----------
        spike_times
            Input spike times of shape ``(batch, n_in)``.

        Returns
        -------
        Tensor
            Soft first-spike times of shape ``(batch, n_out)``.
        """
        t = self.time_grid.view(1, -1, 1, 1)
        arrival = spike_times.unsqueeze(1).unsqueeze(-1) + self.delay.unsqueeze(0).unsqueeze(1)
        tau = (t - arrival).clamp_min(0.0)
        alpha = tau * torch.exp(1.0 - tau)
        voltage = (alpha * self.weight.unsqueeze(0).unsqueeze(1)).sum(dim=2)
        crossing = torch.sigmoid(8.0 * (voltage - 0.5))
        weights = crossing / crossing.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return (weights * self.time_grid.view(1, -1, 1)).sum(dim=1)


def build() -> nn.Module:
    """Build a compact SpikeProp timing layer.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return SpikeProp()


def example_input() -> Tensor:
    """Return input spike times.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 4)``.
    """
    return torch.rand(2, 4) * 1.5
