"""Cascade-Correlation, 1990, Fahlman and Lebiere, "The Cascade-Correlation Learning Architecture".

Hidden units are installed as frozen feature detectors connected to inputs and all
previous hidden units; the output layer reads the growing cascade of features.
"""

import torch
from torch import Tensor, nn


class CascadeCorrelation(nn.Module):
    """Static random-init cascade-correlation forward architecture."""

    def __init__(
        self, input_size: int = 6, hidden_sizes: tuple[int, ...] = (4, 3), out_size: int = 2
    ) -> None:
        """Initialize frozen cascade units and a trainable readout.

        Parameters
        ----------
        input_size:
            Number of input features.
        hidden_sizes:
            Width of each installed cascade stage.
        out_size:
            Number of readout outputs.
        """
        super().__init__()
        self.units = nn.ModuleList()
        running_size = input_size
        for hidden_size in hidden_sizes:
            unit = nn.Linear(running_size, hidden_size)
            for parameter in unit.parameters():
                parameter.requires_grad_(False)
            self.units.append(unit)
            running_size += hidden_size
        self.readout = nn.Linear(running_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        """Compute cascade features and output logits.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        Tensor
            Readout tensor of shape ``(batch, out_size)``.
        """
        features = [x]
        joined = x
        for unit in self.units:
            h = torch.tanh(unit(joined))
            features.append(h)
            joined = torch.cat(features, dim=-1)
        return self.readout(joined)


def build() -> nn.Module:
    """Build a small cascade-correlation module.

    Returns
    -------
    nn.Module
        Configured ``CascadeCorrelation`` instance.
    """
    return CascadeCorrelation()


def example_input() -> Tensor:
    """Create a tabular float example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 6)``.
    """
    return torch.randn(2, 6)
