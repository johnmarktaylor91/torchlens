"""NeuralForecast xLSTM: long-horizon forecasting with extended LSTM blocks.

Paper: xLSTMTime: Long-Term Time Series Forecasting With xLSTM, Alharthi et al. 2024.

The load-bearing primitive is Beck et al.'s xLSTM recurrence adapted to time-series
forecasting: exponential input/forget gates with a stabilizer state and a normalized
memory update.  This compact random-init reconstruction uses an xLSTM-style recurrent
encoder followed by NeuralForecast-style horizon projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactXLSTMCell(nn.Module):
    """xLSTM scalar-memory cell with exponential gates and normalization state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize recurrent projections.

        Parameters
        ----------
        input_size:
            Number of input variables at each step.
        hidden_size:
            Hidden state width.
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, 4 * hidden_size)
        self.h_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        n_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one normalized exponential-gated xLSTM update.

        Parameters
        ----------
        x_t:
            Input slice ``(batch, input_size)``.
        h_prev:
            Previous hidden state.
        c_prev:
            Previous scalar memory state.
        n_prev:
            Previous normalizer state.
        m_prev:
            Previous log-gate stabilizer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Updated hidden, memory, normalizer, and stabilizer states.
        """

        i_raw, f_raw, z_raw, o_raw = (self.x_proj(x_t) + self.h_proj(h_prev)).chunk(4, dim=-1)
        m_t = torch.maximum(f_raw + m_prev, i_raw)
        i_t = torch.exp(i_raw - m_t)
        f_t = torch.exp(f_raw + m_prev - m_t)
        z_t = torch.tanh(z_raw)
        o_t = torch.sigmoid(o_raw)
        c_t = f_t * c_prev + i_t * z_t
        n_t = f_t * n_prev + i_t
        h_t = o_t * c_t / n_t.clamp_min(1e-6)
        return h_t, c_t, n_t, m_t


class NeuralForecastXLSTM(nn.Module):
    """Compact xLSTM forecaster with direct multi-step horizon head."""

    def __init__(self, n_vars: int = 4, hidden_size: int = 24, horizon: int = 6) -> None:
        """Initialize the forecasting model.

        Parameters
        ----------
        n_vars:
            Number of observed time-series variables.
        hidden_size:
            xLSTM hidden width.
        horizon:
            Forecast horizon length.
        """

        super().__init__()
        self.cell = CompactXLSTMCell(n_vars, hidden_size)
        self.skip = nn.Linear(n_vars, hidden_size)
        self.head = nn.Linear(hidden_size, horizon * n_vars)
        self.horizon = horizon
        self.n_vars = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast future values from a history tensor.

        Parameters
        ----------
        x:
            Input history ``(batch, time, variables)``.

        Returns
        -------
        torch.Tensor
            Forecast tensor ``(batch, horizon, variables)``.
        """

        batch = x.shape[0]
        h = x.new_zeros(batch, self.cell.hidden_size)
        c = x.new_zeros(batch, self.cell.hidden_size)
        n = x.new_zeros(batch, self.cell.hidden_size)
        m = x.new_zeros(batch, self.cell.hidden_size)
        for t in range(x.shape[1]):
            h, c, n, m = self.cell(x[:, t], h, c, n, m)
        h = h + torch.tanh(self.skip(x.mean(dim=1)))
        return self.head(h).view(batch, self.horizon, self.n_vars)


def build() -> nn.Module:
    """Build the compact NeuralForecast xLSTM model.

    Returns
    -------
    nn.Module
        Random-init xLSTM forecaster.
    """

    return NeuralForecastXLSTM()


def example_input() -> torch.Tensor:
    """Create a small time-series history.

    Returns
    -------
    torch.Tensor
        Example input ``(2, 12, 4)``.
    """

    return torch.randn(2, 12, 4)


MENAGERIE_ENTRIES = [
    (
        "NeuralForecast xLSTM",
        "build",
        "example_input",
        "2024",
        "time-series/forecasting",
    ),
]
