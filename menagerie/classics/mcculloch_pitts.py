"""McCulloch-Pitts threshold logic net (1943), Warren McCulloch and Walter Pitts.

Paper: "A logical calculus of the ideas immanent in nervous activity."
Binary threshold neurons implement propositional logic with fixed excitatory and
inhibitory connections; any active inhibitory input vetoes the unit.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class McCullochPittsNet(nn.Module):
    """Fixed binary threshold-logic network with inhibitory veto connections."""

    def __init__(self, n_in: int = 6, n_hidden: int = 4, n_out: int = 3) -> None:
        """Initialize fixed threshold-logic weights.

        Parameters
        ----------
        n_in
            Number of binary input units.
        n_hidden
            Number of hidden threshold units.
        n_out
            Number of output threshold units.
        """
        super().__init__()
        hidden_exc = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            ]
        )[:n_hidden, :n_in]
        hidden_inh = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )[:n_hidden, :n_in]
        out_exc = torch.ones(n_hidden, n_out)
        out_inh = torch.zeros(n_hidden, n_out)
        out_inh[0, 1 % n_out] = 1.0
        self.register_buffer("hidden_exc", hidden_exc)
        self.register_buffer("hidden_inh", hidden_inh)
        self.register_buffer("hidden_theta", torch.full((n_hidden,), 2.0))
        self.register_buffer("out_exc", out_exc)
        self.register_buffer("out_inh", out_inh)
        self.register_buffer("out_theta", torch.tensor([2.0, 2.0, 3.0])[:n_out])

    def _threshold_layer(
        self,
        x: Tensor,
        exc: Tensor,
        inh: Tensor,
        theta: Tensor,
    ) -> Tensor:
        """Apply a McCulloch-Pitts threshold layer.

        Parameters
        ----------
        x
            Binary input activations of shape ``(batch, features)``.
        exc
            Excitatory connection matrix.
        inh
            Inhibitory connection matrix.
        theta
            Threshold vector.

        Returns
        -------
        Tensor
            Binary output activations.
        """
        excitation = x @ exc.T
        inhibited = (x @ inh.T) > 0.0
        active = (excitation >= theta) & (~inhibited)
        return active.to(x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Compute binary threshold-logic outputs.

        Parameters
        ----------
        x
            Binary input tensor of shape ``(batch, n_in)``.

        Returns
        -------
        Tensor
            Binary output tensor.
        """
        hidden = self._threshold_layer(x, self.hidden_exc, self.hidden_inh, self.hidden_theta)
        return self._threshold_layer(hidden, self.out_exc.T, self.out_inh.T, self.out_theta)


def build() -> nn.Module:
    """Build a small random-free McCulloch-Pitts threshold network.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return McCullochPittsNet()


def example_input() -> Tensor:
    """Return a binary example input.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
