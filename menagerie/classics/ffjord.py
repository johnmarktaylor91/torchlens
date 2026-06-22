"""FFJORD: Free-Form Jacobian of Reversible Dynamics (continuous normalizing flow).

Grathwohl et al., ICLR 2019.
Paper: https://arxiv.org/abs/1810.01367
Source: https://github.com/rtqichen/ffjord

FFJORD is a continuous normalizing flow (CNF) based on Neural ODEs:
  dz/dt = f_theta(z, t)

Log-density change via Hutchinson's trace estimator:
  d log p(z)/dt = -Tr(df/dz) ≈ -epsilon^T (df/dz) epsilon
where epsilon is a random vector (standard normal or Rademacher).

Key architectural primitives:
  1. ODE function net f_theta: a neural network that takes (z, t) as input
     and outputs dz/dt. Typically a ResNet-like or MADE-like structure.
     The time t is injected via concatenation or learned time embedding.
  2. Hutchinson divergence estimator: during training, augments the state
     with an estimated log-determinant accumulator. During inference (just
     the flow), we run the ODE forward.
  3. Unrolled integration: for the graph atlas, we unroll a fixed-step
     Euler integrator with a small number of steps so the computation graph
     shows the flow structure.

Architecture:
  - ODEFunc: MLP(z_dim + 1, hidden_dim, z_dim) with time input
  - UnrolledEulerCNF: 4 Euler steps, each computing ODE func + updating z
    and accumulating log-density change via autograd trace or Hutchinson

For the menagerie trace (inference, CPU), we show the unrolled graph:
  z_0 -> [f_theta(z_0, t_0)] -> z_1 -> ... -> z_T

Simplifications: z_dim=4, hidden_dim=32, 4 Euler steps, Hutchinson with
fixed epsilon (not resampled per step for graph clarity).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """Neural ODE function: f_theta(z, t) -> dz/dt.

    Time t is concatenated to the state z before the MLP.
    Uses SOFTPLUS activations (common in FFJORD) for smooth dynamics.
    """

    def __init__(self, z_dim: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim), t: scalar or (B, 1)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1).expand(z.size(0), 1)
        zt = torch.cat([z, t], dim=-1)
        return self.net(zt)


class FFJORDUnrolled(nn.Module):
    """FFJORD with unrolled fixed-step Euler integration.

    Shows the computation graph explicitly for TorchLens tracing.
    The Hutchinson divergence estimator uses a fixed epsilon for graph clarity.

    Forward: z_0 -> z_T with accumulated log|det J|
    Returns: (z_T, delta_log_p) where delta_log_p is the accumulated log-density change.
    """

    def __init__(
        self,
        z_dim: int = 4,
        hidden_dim: int = 32,
        n_steps: int = 4,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> None:
        super().__init__()
        self.ode_func = ODEFunc(z_dim, hidden_dim)
        self.n_steps = n_steps
        self.t0 = t0
        self.t1 = t1
        self.z_dim = z_dim
        self.dt = (t1 - t0) / n_steps

    def _hutchinson_div(self, z: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> tuple:
        """Approximate divergence via Hutchinson: e^T (df/dz) e using autograd.

        For the unrolled graph, we use a simple approximation:
        compute (f(z + eps*delta) - f(z - eps*delta)) / (2*delta) dot eps.
        This is the finite-difference Hutchinson estimator (no autograd needed).
        """
        delta = 1e-3
        dz_p = self.ode_func(z + delta * eps, t)
        dz_m = self.ode_func(z - delta * eps, t)
        dz = self.ode_func(z, t)
        # Approximate Jacobian-vector product via finite differences
        jvp = (dz_p - dz_m) / (2 * delta)  # approx J @ eps
        div_est = (eps * jvp).sum(dim=-1, keepdim=True)  # Hutchinson trace estimate
        return dz, -div_est

    def forward(self, z0: torch.Tensor) -> tuple:
        # z0: (B, z_dim)
        B = z0.size(0)

        # Fixed epsilon for trace (Rademacher: +/-1)
        eps = torch.sign(torch.randn_like(z0))

        z = z0
        log_p = torch.zeros(B, 1, device=z0.device)

        dt = torch.tensor(self.dt, device=z0.device, dtype=z0.dtype)

        for step in range(self.n_steps):
            t = torch.tensor(self.t0 + step * self.dt, device=z0.device, dtype=z0.dtype)
            dz, dlogp = self._hutchinson_div(z, t, eps)
            z = z + dt * dz
            log_p = log_p + dt * dlogp

        return z, log_p


def build_ffjord() -> nn.Module:
    return FFJORDUnrolled(z_dim=4, hidden_dim=32, n_steps=4)


def example_input_ffjord() -> torch.Tensor:
    # (B=2, z_dim=4): batch of latent vectors
    return torch.randn(2, 4)


MENAGERIE_ENTRIES = [
    ("FFJORD", "build_ffjord", "example_input_ffjord", "2019", "DC"),
]
