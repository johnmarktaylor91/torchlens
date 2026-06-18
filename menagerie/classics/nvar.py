"""NVAR / Next-Generation Reservoir Computing, 2021, Gauthier et al.

Paper: Gauthier et al. 2021, "Next generation reservoir computing." Time-delay
taps and quadratic products form a deterministic reservoir feature map followed
by a linear readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("NVAR / Next-Generation Reservoir Computing", "build", "example_input", "2021", "CF")
]


class NVAR(nn.Module):
    """Time-delay and quadratic-feature reservoir readout."""

    def __init__(self, input_size: int = 3, delays: int = 3, output_size: int = 2) -> None:
        """Initialize the deterministic feature readout.

        Parameters
        ----------
        input_size
            Number of features per time step.
        delays
            Number of delay taps.
        output_size
            Number of readout features.
        """
        super().__init__()
        self.input_size = input_size
        self.delays = delays
        linear_size = input_size * delays
        quad_size = linear_size * (linear_size + 1) // 2
        self.readout = nn.Linear(1 + linear_size + quad_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Compute NVAR features and readout for each time step.

        Parameters
        ----------
        x
            Batch-first sequence of shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Readout sequence of shape ``(batch, time, output_size)``.
        """
        padded = torch.cat((x.new_zeros(x.shape[0], self.delays - 1, x.shape[2]), x), dim=1)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            lags = [padded[:, step + self.delays - 1 - lag] for lag in range(self.delays)]
            linear = torch.cat(lags, dim=-1)
            products = []
            for i in range(linear.shape[-1]):
                for j in range(i, linear.shape[-1]):
                    products.append(linear[:, i] * linear[:, j])
            quad = torch.stack(products, dim=-1)
            bias = x.new_ones(x.shape[0], 1)
            outputs.append(self.readout(torch.cat((bias, linear, quad), dim=-1)))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small NVAR readout.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return NVAR()


def example_input() -> Tensor:
    """Return a batch-first time series.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 6, 3)``.
    """
    return torch.randn(2, 6, 3)
