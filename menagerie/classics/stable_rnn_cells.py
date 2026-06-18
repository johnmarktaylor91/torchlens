"""Stable and continuous-time recurrent cells, 2019-2022.

Paper: Chang et al. 2019, "AntisymmetricRNN"; Rusch and Mishra 2021,
"Coupled Oscillatory Recurrent Neural Network (coRNN)" and LEM follow-ups.
The modules implement small trace-clean recurrent updates, omitting solvers and
training losses beyond forward-pass dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LipschitzRNN(nn.Module):
    """RNN with Lipschitz-bounded antisymmetric/symmetric dynamics."""

    def __init__(
        self, d_in: int = 4, d_hidden: int = 6, beta: float = 0.1, gamma: float = 0.05
    ) -> None:
        """Initialize Lipschitz RNN parameters.

        Parameters
        ----------
        d_in
            Input dimensionality.
        d_hidden
            Hidden state size.
        beta
            Symmetric component weight.
        gamma
            Diagonal damping.
        """
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(d_hidden, d_hidden) * 0.1)
        self.input = nn.Linear(d_in, d_hidden)
        self.beta = beta
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        """Run bounded recurrent dynamics over a batch-first sequence.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, time, d_in)``.

        Returns
        -------
        Tensor
            Hidden states for all time steps.
        """
        batch = x.shape[0]
        hidden = x.new_zeros(batch, self.matrix.shape[0])
        eye = torch.eye(self.matrix.shape[0], dtype=x.dtype, device=x.device)
        matrix = (1.0 - self.beta) * (self.matrix - self.matrix.T)
        matrix = matrix + self.beta * (self.matrix + self.matrix.T) - self.gamma * eye
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            hidden = hidden + torch.tanh(hidden @ matrix.T + self.input(x[:, step]))
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class AntisymmetricRNN(nn.Module):
    """Antisymmetric Euler RNN with damping."""

    def __init__(
        self, d_in: int = 4, d_hidden: int = 6, gamma: float = 0.05, dt: float = 0.1
    ) -> None:
        """Initialize antisymmetric RNN parameters.

        Parameters
        ----------
        d_in
            Input dimensionality.
        d_hidden
            Hidden state size.
        gamma
            Diagonal damping.
        dt
            Euler step size.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_hidden, d_hidden) * 0.1)
        self.input = nn.Linear(d_in, d_hidden)
        self.gamma = gamma
        self.dt = dt

    def forward(self, x: Tensor) -> Tensor:
        """Run antisymmetric recurrent dynamics.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, time, d_in)``.

        Returns
        -------
        Tensor
            Hidden states for all time steps.
        """
        batch = x.shape[0]
        hidden = x.new_zeros(batch, self.weight.shape[0])
        eye = torch.eye(self.weight.shape[0], dtype=x.dtype, device=x.device)
        matrix = self.weight - self.weight.T - self.gamma * eye
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            hidden = hidden + self.dt * torch.tanh(hidden @ matrix.T + self.input(x[:, step]))
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class LongExpressiveMemory(nn.Module):
    """Small LEM-style slow/fast gated recurrent cell."""

    def __init__(self, d_in: int = 4, d_hidden: int = 6) -> None:
        """Initialize LEM gates and candidate transforms.

        Parameters
        ----------
        d_in
            Input dimensionality.
        d_hidden
            Hidden state size.
        """
        super().__init__()
        self.slow_gate = nn.Linear(d_in + d_hidden, d_hidden)
        self.fast_gate = nn.Linear(d_in + d_hidden, d_hidden)
        self.candidate = nn.Linear(d_in + d_hidden, d_hidden)

    def forward(self, x: Tensor) -> Tensor:
        """Run slow/fast gated LEM dynamics.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, time, d_in)``.

        Returns
        -------
        Tensor
            Slow states for all time steps.
        """
        batch = x.shape[0]
        slow = x.new_zeros(batch, self.candidate.out_features)
        fast = x.new_zeros(batch, self.candidate.out_features)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            joined = torch.cat((x[:, step], fast), dim=-1)
            gate_slow = torch.sigmoid(self.slow_gate(joined))
            gate_fast = torch.sigmoid(self.fast_gate(joined))
            proposal = torch.tanh(self.candidate(joined))
            fast = gate_fast * fast + (1.0 - gate_fast) * proposal
            slow = gate_slow * slow + (1.0 - gate_slow) * fast
            outputs.append(slow)
        return torch.stack(outputs, dim=1)


class GRUODEBayes(nn.Module):
    """Traceable GRU-ODE-Bayes sketch with fixed-step ODE flow and jumps."""

    def __init__(self, d_in: int = 3, d_hidden: int = 6) -> None:
        """Initialize ODE drift and jump GRU cell.

        Parameters
        ----------
        d_in
            Observation dimensionality.
        d_hidden
            Hidden state size.
        """
        super().__init__()
        self.drift = nn.Linear(d_hidden, d_hidden)
        self.jump = nn.GRUCell(d_in, d_hidden)
        self.obs_head = nn.Linear(d_hidden, d_in)

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Run ODE flow between masked observations and Bayesian-like jumps.

        Parameters
        ----------
        inputs
            Tuple of observations ``(batch, time, d_in)``, mask with the same
            shape, and times with shape ``(time,)``.

        Returns
        -------
        Tensor
            Observation predictions for all time steps.
        """
        x, mask, times = inputs
        batch = x.shape[0]
        hidden = x.new_zeros(batch, self.drift.out_features)
        outputs: list[Tensor] = []
        previous = times[0]
        for step in range(x.shape[1]):
            dt = (times[step] - previous).clamp_min(0.0)
            hidden = hidden + dt * torch.tanh(self.drift(hidden))
            observed = x[:, step] * mask[:, step]
            jumped = self.jump(observed, hidden)
            keep = (mask[:, step].mean(dim=-1, keepdim=True) > 0.0).to(x.dtype)
            hidden = keep * jumped + (1.0 - keep) * hidden
            outputs.append(self.obs_head(hidden))
            previous = times[step]
        return torch.stack(outputs, dim=1)


def build_lipschitz_rnn() -> nn.Module:
    """Build a Lipschitz RNN.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return LipschitzRNN()


def example_input_lipschitz_rnn() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 4, 4)


def build_antisymmetric_rnn() -> nn.Module:
    """Build an antisymmetric RNN.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return AntisymmetricRNN()


def example_input_antisymmetric_rnn() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 4, 4)


def build_lem() -> nn.Module:
    """Build a Long Expressive Memory cell stack.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return LongExpressiveMemory()


def example_input_lem() -> Tensor:
    """Return an example LEM sequence.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 4, 4)


def build_gru_ode_bayes() -> nn.Module:
    """Build a GRU-ODE-Bayes sketch.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return GRUODEBayes()


def example_input_gru_ode_bayes() -> tuple[Tensor, Tensor, Tensor]:
    """Return example irregular observations.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Observations, mask, and times.
    """
    observations = torch.randn(2, 4, 3)
    mask = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ]
    )
    times = torch.tensor([0.0, 0.4, 1.0, 1.7])
    return observations, mask, times


MENAGERIE_ENTRIES = [
    ("Lipschitz RNN", "build_lipschitz_rnn", "example_input_lipschitz_rnn", "2020", "CH-C"),
    (
        "Antisymmetric RNN",
        "build_antisymmetric_rnn",
        "example_input_antisymmetric_rnn",
        "2019",
        "CH-C",
    ),
    ("Long Expressive Memory (LEM)", "build_lem", "example_input_lem", "2022", "CH-C"),
    ("GRU-ODE-Bayes", "build_gru_ode_bayes", "example_input_gru_ode_bayes", "2019", "CH-C"),
]
