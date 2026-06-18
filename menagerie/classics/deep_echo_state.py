"""Deep Echo State Network, 2017, Gallicchio, Micheli, and Pedrelli.

Paper: Gallicchio et al. 2017, "Deep reservoir computing: A critical
experimental analysis." Fixed stacked recurrent reservoirs create hierarchical
temporal dynamics that feed a linear readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Deep Echo State Network", "build", "example_input", "2017", "CF")]


class DeepEchoStateNetwork(nn.Module):
    """Stack of fixed random tanh reservoirs with a trainable readout."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 6,
        n_layers: int = 2,
        output_size: int = 3,
    ) -> None:
        """Initialize fixed reservoir weights and trainable readout.

        Parameters
        ----------
        input_size
            Number of input features.
        hidden_size
            Reservoir width per layer.
        n_layers
            Number of stacked reservoirs.
        output_size
            Number of readout features.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        in_weights = []
        rec_weights = []
        for layer in range(n_layers):
            layer_in = input_size if layer == 0 else hidden_size
            in_weights.append(torch.randn(layer_in, hidden_size) * 0.4)
            rec = torch.randn(hidden_size, hidden_size)
            rec = rec / rec.abs().sum(dim=0, keepdim=True).clamp_min(1.0) * 0.8
            rec_weights.append(rec)
        self.input_weights = nn.ParameterList(
            [nn.Parameter(w, requires_grad=False) for w in in_weights]
        )
        self.recurrent_weights = nn.ParameterList(
            [nn.Parameter(w, requires_grad=False) for w in rec_weights]
        )
        self.readout = nn.Linear(hidden_size * n_layers, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Run stacked reservoirs over a time-major sequence.

        Parameters
        ----------
        x
            Input sequence of shape ``(time, batch, input_size)``.

        Returns
        -------
        Tensor
            Readout sequence of shape ``(time, batch, output_size)``.
        """
        batch = x.shape[1]
        states = [x.new_zeros(batch, self.hidden_size) for _ in range(self.n_layers)]
        outputs: list[Tensor] = []
        for step in range(x.shape[0]):
            layer_input = x[step]
            new_states = []
            for layer in range(self.n_layers):
                state = torch.tanh(
                    layer_input @ self.input_weights[layer]
                    + states[layer] @ self.recurrent_weights[layer]
                )
                new_states.append(state)
                layer_input = state
            states = new_states
            outputs.append(self.readout(torch.cat(states, dim=-1)))
        return torch.stack(outputs, dim=0)


def build() -> nn.Module:
    """Build a small deep echo state network.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return DeepEchoStateNetwork()


def example_input() -> Tensor:
    """Return a time-major sequence input.

    Returns
    -------
    Tensor
        Example tensor of shape ``(5, 2, 4)``.
    """
    return torch.randn(5, 2, 4)
