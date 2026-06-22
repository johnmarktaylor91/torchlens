"""FFJORD and Latent ODE: continuous-time neural ODE models.

--- FFJORD ---
Grathwohl et al. (2018), "FFJORD: Free-form Continuous Dynamics for
Scalable Reversible Generative Models".  ICLR 2019.  arXiv:1810.01367.
Source: https://github.com/rtqichen/ffjord

Distinctive primitive: a continuous normalising flow (CNF) where the
transformation is defined by an ODE:
    dz/dt = f(z, t; theta)
where f is a small MLP with t concatenated to z.  The transformation is
integrated by a fixed-step numerical solver (Euler steps here).  For the
atlas we reproduce the ODE dynamics network (f-MLP) and roll it over a few
Euler steps -- the trace estimator for log-density is dropped as it requires
stochastic trace estimation (Hutchinson), which is out of scope.

--- Latent ODE ---
Rubanova et al. (2019), "Latent ODEs for Irregularly-Sampled Time Series".
NeurIPS 2019.  arXiv:1907.03907.
Source: https://github.com/YuliaRubanova/latent_neural_odes

Distinctive primitive: RNN encoder -> initial latent state z0 -> ODE decoder.
  1. GRU encoder processes observed (x, t) pairs in reverse to produce z0.
  2. Neural ODE: z(t) = z0 integrated forward by dz/dt = f_theta(z, t).
  3. Decoder: linear projection of ODE latent at query times -> reconstruction.

For the atlas: GRU encoder -> z0 -> ODE function MLP -> a few Euler steps ->
linear decoder.  No irregularly-spaced times; we use uniform t in [0,1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Shared: ODE dynamics MLP (f(z, t) network)
# ==============================================================


class ODEFunc(nn.Module):
    """ODE dynamics network: f(z, t) -> dz/dt.

    Input: z (B, d_latent), t scalar -> concatenate t as extra feature.
    """

    def __init__(self, d_latent: int = 16, d_hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_latent),
        )

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        """z: (B, d_latent), t: scalar -> dz/dt: (B, d_latent)"""
        t_vec = torch.full((z.shape[0], 1), t, device=z.device, dtype=z.dtype)
        zt = torch.cat([z, t_vec], dim=-1)
        return self.net(zt)


def euler_integrate(func: ODEFunc, z0: torch.Tensor, n_steps: int = 4) -> torch.Tensor:
    """Euler integration of dz/dt = func(z, t) from t=0 to t=1.

    Returns z at final time (B, d_latent).
    """
    dt = 1.0 / n_steps
    z = z0
    for i in range(n_steps):
        t = i * dt
        z = z + dt * func(z, t)
    return z


def euler_trajectory(func: ODEFunc, z0: torch.Tensor, n_steps: int = 4) -> torch.Tensor:
    """Return trajectory of shape (B, n_steps+1, d_latent) including z0."""
    dt = 1.0 / n_steps
    z = z0
    traj = [z]
    for i in range(n_steps):
        t = i * dt
        z = z + dt * func(z, t)
        traj.append(z)
    return torch.stack(traj, dim=1)  # (B, n_steps+1, d_latent)


# ==============================================================
# 1.  FFJORD CNF (atlas version: dynamics net + Euler-unrolled transform)
# ==============================================================


class FFJORDCNFWrapper(nn.Module):
    """FFJORD-style CNF: data -> ODE-dynamics transform -> transformed data.

    For the atlas we omit the log-det trace estimator (Hutchinson requires
    random noise vectors and stochastic computation, which makes the static
    graph non-deterministic).  We reproduce the f-MLP + Euler integration.
    Input: (B, d_data); output: (B, d_data) transformed.
    """

    def __init__(self, d_data: int = 8, d_hidden: int = 32, n_steps: int = 4) -> None:
        super().__init__()
        self.ode_func = ODEFunc(d_data, d_hidden)
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_data) -> z: (B, d_data)"""
        return euler_integrate(self.ode_func, x, self.n_steps)


def build_ffjord_cnf() -> nn.Module:
    return FFJORDCNFWrapper(d_data=8, d_hidden=32, n_steps=4).eval()


def example_input_ffjord() -> torch.Tensor:
    """(2, 8) -- batch=2, d_data=8."""
    return torch.randn(2, 8)


# ==============================================================
# 2.  Latent ODE with GRU encoder
# ==============================================================


class LatentODEEncoder(nn.Module):
    """GRU encoder: processes (B, T, d_obs) in REVERSE to produce z0 (B, d_latent)."""

    def __init__(self, d_obs: int = 4, d_latent: int = 16, d_hidden: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(d_obs, d_hidden, batch_first=True)
        self.fc_z0 = nn.Linear(d_hidden, 2 * d_latent)  # mean + log-var
        self.d_latent = d_latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_obs) -> z0: (B, d_latent)"""
        # Reverse sequence (RNN encoder processes from last to first)
        x_rev = torch.flip(x, dims=[1])
        _, h = self.gru(x_rev)  # h: (1, B, d_hidden)
        h = h.squeeze(0)  # (B, d_hidden)
        z0_params = self.fc_z0(h)  # (B, 2*d_latent)
        # For the atlas: use mean (no reparameterisation -- keeps graph deterministic)
        z0 = z0_params[:, : self.d_latent]
        return z0


class LatentODEDecoder(nn.Module):
    """Linear decoder: z(t) -> x_hat(t) for each step in trajectory."""

    def __init__(self, d_latent: int = 16, d_obs: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(d_latent, d_obs)

    def forward(self, z_traj: torch.Tensor) -> torch.Tensor:
        """z_traj: (B, T, d_latent) -> (B, T, d_obs)"""
        return self.linear(z_traj)


class LatentODEModel(nn.Module):
    """Full Latent ODE: GRU encoder -> z0 -> ODE integration -> linear decoder."""

    def __init__(
        self,
        d_obs: int = 4,
        d_latent: int = 16,
        d_hidden: int = 32,
        n_ode_steps: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = LatentODEEncoder(d_obs, d_latent, d_hidden)
        self.ode_func = ODEFunc(d_latent, d_hidden)
        self.decoder = LatentODEDecoder(d_latent, d_obs)
        self.n_ode_steps = n_ode_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_in, d_obs) -> (B, T_out, d_obs) reconstruction"""
        z0 = self.encoder(x)  # (B, d_latent)
        z_traj = euler_trajectory(
            self.ode_func, z0, self.n_ode_steps
        )  # (B, n_ode_steps+1, d_latent)
        return self.decoder(z_traj)  # (B, n_ode_steps+1, d_obs)


def build_latent_ode_encoder() -> nn.Module:
    return LatentODEModel(d_obs=4, d_latent=16, d_hidden=32, n_ode_steps=4).eval()


def example_input_latent_ode() -> torch.Tensor:
    """(1, 8, 4) -- batch=1, T=8 obs steps, d_obs=4."""
    return torch.randn(1, 8, 4)


MENAGERIE_ENTRIES = [
    (
        "FFJORD CNF (continuous normalising flow: ODE-dynamics MLP + Euler integration)",
        "build_ffjord_cnf",
        "example_input_ffjord",
        "2019",
        "DC",
    ),
    (
        "Latent ODE (GRU encoder -> z0 -> neural ODE decoder trajectory)",
        "build_latent_ode_encoder",
        "example_input_latent_ode",
        "2019",
        "DC",
    ),
]
