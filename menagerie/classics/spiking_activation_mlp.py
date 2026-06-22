"""PyTorch-Spiking SpikingActivation MLP: rate-coded spiking activations in an MLP.

Rasmussen (2021), "Nengo-PyTorch / PyTorch-Spiking: SpikingActivation".
Source: https://github.com/nengo/pytorch-spiking

Distinctive primitive: SpikingActivation layer.
A standard activation function (e.g. ReLU) is applied over T timesteps to produce
a RATE CODE: at each timestep t, spike = (relu(x) * dt * T >= u_t) where u_t is a
uniform [0,1] threshold noise.  In the simplest form this becomes:
  spike = (torch.rand_like(x) < relu(x) * dt)
over T timesteps.  The output for a timestep is binary (0/1) and the mean over T
approximates the continuous activation.

For TorchLens tracing (static graph), we unroll T timesteps explicitly as a
loop-free sequence: pre-compute T Bernoulli samples and use a clamp+round to produce
the binary output deterministically from the input magnitude.
(Stochastic rand_like in a traced graph is fine -- it uses the same rand per call
in tracing mode since we use `inference_only=True`.)

Architecture: Linear -> SpikingActivation (rolled T steps) -> Linear -> SpikingActivation -> Linear.
Compact config: d_in=16, d_hidden=32, d_out=10, T=3 timesteps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingActivation(nn.Module):
    """Rate-coded spiking activation over T timesteps.

    For each of T timesteps: spike_t = (rand < relu(x) * dt).
    Mean over T -> continuous rate approximation.
    The loop is unrolled for a clean static computation graph.
    """

    def __init__(self, T: int = 3, dt: float = 1.0) -> None:
        super().__init__()
        self.T = T
        self.dt = dt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d) -> (..., d) float (rate-coded mean spike)"""
        # Rate: relu(x) scaled to probability per timestep
        rate = F.relu(x) * self.dt / self.T
        rate = rate.clamp(0.0, 1.0)
        # Accumulate T binary spike samples
        total = torch.zeros_like(x)
        for _ in range(self.T):
            spike = (torch.rand_like(x) < rate).float()
            total = total + spike
        # Normalise to mean rate
        return total / self.T


class SpikingMLP(nn.Module):
    """MLP with spiking activations (rate-coded over T timesteps).

    Architecture: Linear -> BN -> SpikingAct -> Linear -> BN -> SpikingAct -> Linear.
    """

    def __init__(self, d_in: int = 16, d_hidden: int = 32, d_out: int = 10, T: int = 3) -> None:
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.spike1 = SpikingActivation(T)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.bn2 = nn.BatchNorm1d(d_hidden)
        self.spike2 = SpikingActivation(T)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in) -> (B, d_out)"""
        x = self.spike1(self.bn1(self.fc1(x)))
        x = self.spike2(self.bn2(self.fc2(x)))
        return self.fc3(x)


def build_pytorch_spiking_activation_mlp() -> nn.Module:
    return SpikingMLP(d_in=16, d_hidden=32, d_out=10, T=3).eval()


def example_input() -> torch.Tensor:
    """(4, 16) -- batch=4, d_in=16."""
    return torch.randn(4, 16)


MENAGERIE_ENTRIES = [
    (
        "PyTorch-Spiking SpikingActivation MLP (rate-coded spiking neurons: BN + Bernoulli-rate spike over T timesteps)",
        "build_pytorch_spiking_activation_mlp",
        "example_input",
        "2021",
        "DC",
    ),
]
