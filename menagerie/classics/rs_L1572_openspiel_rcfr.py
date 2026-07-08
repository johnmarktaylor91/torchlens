# SOURCE: vendored from google-deepmind/open_spiel @ master
# (open_spiel/python/pytorch/rcfr.py)
#
# DeepRcfrModel + ResidualMLPBlock are the real PyTorch regret-approximation network
# used by OpenSpiel's Deep RCFR (Regression Counterfactual Regret Minimization)
# solver -- a flexible feedforward MLP with an optional low-rank "gate" factorization
# (`num_hidden_factors`) per hidden layer and optional residual/skip connections
# across hidden layers (`use_skip_connections`), which is the architecturally
# distinctive bit relative to a plain MLP. Vendored verbatim, with one constructor
# change: the original `DeepRcfrModel.__init__` takes an OpenSpiel `game` object and
# calls `open_spiel.python.pytorch.rcfr.num_features(game)` to compute the input
# size; since `open_spiel` is not an installed base lib here, this staging build
# helper passes `input_size` directly as an int instead of deriving it from a game.
# No layer/module inside `ResidualMLPBlock`/`DeepRcfrModel` was changed. The original
# class also defines `__call__` (not `forward`) -- kept as-is since that is the real
# API surface OpenSpiel calls it through.

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """A residual MLP block."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_hidden_factors: int = 0,
        hidden_activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self._activation = hidden_activation
        self._gate_layer = (
            (nn.Linear(num_hidden_factors, output_size)) if num_hidden_factors > 0 else None
        )
        self._layer = nn.Sequential(
            nn.Linear(
                input_size,
                output_size if self._gate_layer is None else num_hidden_factors,
            ),
            self._activation if self._gate_layer else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        x = self._layer(x)
        if self._gate_layer:
            x = self._gate_layer(x)
        x += residual
        return self._activation(x)


class DeepRcfrModel(nn.Module):
    """A flexible deep feedforward RCFR model class.

    Properties:
      layers: The `torch.nn.Sequential` layers describing this  model.
    """

    def __init__(
        self,
        input_size: int,
        num_hidden_units: int,
        num_hidden_layers: int = 1,
        num_hidden_factors: int = 0,
        hidden_activation: nn.Module = nn.ReLU(),
        use_skip_connections: bool = False,
    ) -> None:
        """Creates a new `DeepRcfrModel`.

        Args:
          input_size: dimensionality of the (already-featurized) info-state input.
            [Staging note: the real OpenSpiel constructor instead takes a `game`
            object and derives this via `num_features(game)`; that game-coupled
            featurizer is not part of the network architecture, so it is not
            vendored here.]
          num_hidden_units: The number of units in each hidden layer.
          num_hidden_layers: The number of hidden layers. Defaults to 1.
          num_hidden_factors: The number of hidden factors or the matrix rank of the
            layer. If greater than zero, hidden layers will be split into two
            separate linear transformations. Defaults to 0
          hidden_activation: The activation function to apply over hidden layers.
            Defaults to `torch.nn.ReLU`.
          use_skip_connections: Whether or not to apply skip connections (layer
            output = layer(x) + x) on hidden layers.
        """
        super().__init__()

        layers_ = [nn.Sequential(nn.Linear(input_size, num_hidden_units), hidden_activation)]

        layers_.extend(
            [
                (
                    ResidualMLPBlock(
                        num_hidden_units,
                        num_hidden_units,
                        num_hidden_factors,
                        hidden_activation,
                    )
                    if use_skip_connections
                    else nn.Sequential(
                        nn.Linear(num_hidden_units, num_hidden_units), hidden_activation
                    )
                )
                for _ in range(num_hidden_layers)
            ]
        )

        layers_.append(nn.Linear(num_hidden_units, 1))

        self.layers = nn.Sequential(*layers_)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates this model on `x`."""
        return self.layers(x).squeeze(-1)


def build_deep_rcfr() -> nn.Module:
    """Build a small DeepRcfrModel with skip connections + low-rank gating enabled
    (the two architecturally distinctive knobs), at a toy info-state feature size."""

    return DeepRcfrModel(
        input_size=32,
        num_hidden_units=64,
        num_hidden_layers=3,
        num_hidden_factors=16,
        use_skip_connections=True,
    )


def example_input_deep_rcfr() -> torch.Tensor:
    return torch.randn(4, 32)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "OpenSpiel Deep RCFR model (low-rank gated residual MLP regret net)",
        "build_deep_rcfr",
        "example_input_deep_rcfr",
        "2019",
        "DC",
    ),
]
