"""Phased LSTM, 2016, Neil, Pfeiffer, and Liu.

Paper: "Phased LSTM: Accelerating Recurrent Network Training for Long or
Event-based Sequences." A learned oscillating time gate controls when each LSTM
unit updates; this version takes implicit evenly spaced timestamps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PhasedLSTM(nn.Module):
    """LSTM cell sequence with learned rhythmic time gates."""

    def __init__(self, input_size: int = 128, hidden_size: int = 48, leak: float = 0.001) -> None:
        """Initialize LSTM gates and time-gate parameters.

        Parameters
        ----------
        input_size:
            Per-step feature size.
        hidden_size:
            Hidden state size.
        leak:
            Small open value outside the active phase.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.leak = leak
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.shift = nn.Parameter(torch.rand(hidden_size))
        self.r_on = nn.Parameter(torch.full((hidden_size,), 0.1))

    def _time_gate(self, step: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Compute the phased time gate for one integer step.

        Parameters
        ----------
        step:
            Sequence step index.
        device:
            Target device.
        dtype:
            Target dtype.

        Returns
        -------
        Tensor
            Per-hidden-unit gate values.
        """
        time = torch.as_tensor(float(step), device=device, dtype=dtype)
        tau = torch.exp(self.log_tau) + 1.0
        r_on = torch.clamp(self.r_on, 0.05, 0.95)
        phase = torch.remainder(time - self.shift, tau) / tau
        rising = phase / (0.5 * r_on)
        falling = 2.0 - phase / (0.5 * r_on)
        closed = self.leak * phase
        return torch.where(phase < 0.5 * r_on, rising, torch.where(phase < r_on, falling, closed))

    def forward(self, x: Tensor) -> Tensor:
        """Run phased LSTM updates over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Hidden sequence with shape ``(batch, time, hidden_size)``.
        """
        hidden = x.new_zeros(x.shape[0], self.hidden_size)
        cell = x.new_zeros(x.shape[0], self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            gates = self.gates(torch.cat((x[:, step], hidden), dim=-1))
            i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=-1)
            new_cell = torch.sigmoid(f_gate) * cell + torch.sigmoid(i_gate) * torch.tanh(g_gate)
            new_hidden = torch.sigmoid(o_gate) * torch.tanh(new_cell)
            k_gate = self._time_gate(step, x.device, x.dtype).unsqueeze(0)
            cell = k_gate * new_cell + (1.0 - k_gate) * cell
            hidden = k_gate * new_hidden + (1.0 - k_gate) * hidden
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact Phased LSTM.

    Returns
    -------
    nn.Module
        Random-initialized Phased LSTM.
    """
    return PhasedLSTM()


def example_input() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 64, 128)``.
    """
    return torch.randn(1, 64, 128)


MENAGERIE_ENTRIES = [("Phased LSTM", "build", "example_input", "2016", "DE")]
