"""NARX neural network, 1994, Narendra and Parthasarathy.

Nonlinear autoregressive models with exogenous inputs use tapped delay windows
of external inputs and prior outputs to predict the next output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NARXNet(nn.Module):
    """Feed-forward NARX predictor over tapped input and output delays."""

    def __init__(
        self, x_lags: int = 10, y_lags: int = 4, hidden_size: int = 32, output_size: int = 1
    ) -> None:
        """Initialize the tapped-delay MLP.

        Parameters
        ----------
        x_lags:
            Number of exogenous input lags.
        y_lags:
            Number of autoregressive output lags.
        hidden_size:
            Hidden layer size.
        output_size:
            Prediction size.
        """
        super().__init__()
        self.y_lags = y_lags
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(x_lags + y_lags * output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x_window: Tensor, y_window: Tensor | None = None) -> Tensor:
        """Predict the next output from tapped delay windows.

        Parameters
        ----------
        x_window:
            Exogenous lags with shape ``(batch, x_lags)``.
        y_window:
            Optional prior output lags with shape ``(batch, y_lags, output_size)``.
            If omitted, zero prior outputs are used.

        Returns
        -------
        Tensor
            Next output prediction with shape ``(batch, output_size)``.
        """
        if y_window is None:
            y_window = x_window.new_zeros(x_window.shape[0], self.y_lags, self.output_size)
        features = torch.cat((x_window, y_window.reshape(y_window.shape[0], -1)), dim=-1)
        return self.net(features)


def build() -> nn.Module:
    """Build a compact NARX network.

    Returns
    -------
    nn.Module
        Random-initialized NARX network.
    """
    return NARXNet()


def example_input() -> Tensor:
    """Return an example exogenous tapped-delay window.

    Returns
    -------
    Tensor
        Exogenous window with shape ``(1, 10)``.
    """
    return torch.randn(1, 10)


MENAGERIE_ENTRIES = [("NARX Neural Network", "build", "example_input", "1994", "DE")]
