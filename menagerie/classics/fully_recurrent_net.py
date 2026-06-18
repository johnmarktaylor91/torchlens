"""Williams-Zipser Fully Recurrent Net, 1989, "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks".

All units, including output-designated units, feed back into the next recurrent
state; RTRL is the training rule and is intentionally excluded from forward().
"""

import torch
from torch import Tensor, nn


class FullyRecurrentNet(nn.Module):
    """All-to-all recurrent sequence module."""

    def __init__(self, input_size: int = 4, n_units: int = 6, out_size: int = 2) -> None:
        """Initialize the fully recurrent state transition.

        Parameters
        ----------
        input_size:
            Number of input features per time step.
        n_units:
            Number of recurrent units.
        out_size:
            Number of leading units exposed as outputs.
        """
        super().__init__()
        self.n_units = n_units
        self.out_size = out_size
        self.transition = nn.Linear(input_size + n_units, n_units)

    def forward(self, x: Tensor) -> Tensor:
        """Run the fully recurrent net over a batch-first sequence.

        Parameters
        ----------
        x:
            Input sequence of shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Output-unit sequence of shape ``(batch, time, out_size)``.
        """
        state = x.new_zeros(x.shape[0], self.n_units)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            state = torch.tanh(self.transition(torch.cat((x[:, step], state), dim=-1)))
            outputs.append(state[:, : self.out_size])
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small fully recurrent net.

    Returns
    -------
    nn.Module
        Configured ``FullyRecurrentNet`` instance.
    """
    return FullyRecurrentNet()


def example_input() -> Tensor:
    """Create a batch-first sequence example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 4, 4)``.
    """
    return torch.randn(2, 4, 4)
