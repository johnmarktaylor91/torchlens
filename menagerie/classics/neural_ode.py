"""Neural ODEs, Augmented Neural ODEs, and Multiple Shooting Layers (torchdyn-style).

Neural ODE: Chen et al., NeurIPS 2018. arXiv:1806.07366
Augmented Neural ODE: Dupont et al., NeurIPS 2019. arXiv:1904.01681
Multiple Shooting Layer: Massaroli et al., NeurIPS 2021. arXiv:2106.03712

Sources:
  https://github.com/DiffEqML/torchdyn
  https://github.com/rtqichen/torchdiffeq

This module reimplements all three in PURE PyTorch (NO torchdyn/torchdiffeq imports)
using a fixed-step RK4 solver UNROLLED in forward().  The unrolled steps ARE the
point: they show the ODE integration structure in the TorchLens graph.

All three share an autonomous vector-field MLP f(x) (time-invariant for simplicity;
the full Neural ODE uses f(t, x) but the architecture is identical — we pass t=0
for compact graphs).

Architectures:

1. NeuralODE:
   Input x0 (1, dim). RK4-integrate dx/dt = f(x) for n_steps steps.
   The graph shows n_steps calls to the same vector-field MLP (shared weights).
   Output: final state x_T.

2. AugmentedNeuralODE:
   Augment x0 with a_dim zeros -> (1, dim+a_dim), then integrate, then
   project back to (1, dim) via a linear layer.
   Graph shows: augmentation concat + unrolled integration + projection.

3. MultipleShootingLayer:
   Split the time interval [0, T] into n_segments sub-intervals.
   Each segment has its own learned initial shooting parameter s_i.
   Integrate segment i from s_i for a few steps.
   At junctions, apply a soft matching: add a learned correction toward
   the previous segment's endpoint (continuity encouragement).
   Output: concatenation of segment endpoints.
   Graph shows: parallel shooting segments + junction matching.

Compact:
  - dim=8, a_dim=4, n_steps=6, n_segments=3 (each 2 steps).
  - Input: (1, 8) for all entries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Vector field MLP  f(x) -> dx/dt
# ============================================================


class VectorField(nn.Module):
    """Autonomous vector field f(x) used as the ODE right-hand side."""

    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Fixed-step RK4 integrator (unrolled)
# ============================================================


def rk4_step(f: nn.Module, x: torch.Tensor, dt: float) -> torch.Tensor:
    """One RK4 step: x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# 1. Neural ODE
# ============================================================


class NeuralODE(nn.Module):
    """Neural ODE: dx/dt = f(x), integrated with fixed-step RK4 (unrolled).

    The graph shows n_steps calls to the same vector-field MLP.
    Input: (B, dim). Output: (B, dim).
    """

    def __init__(self, dim: int = 8, hidden: int = 32, n_steps: int = 6, T: float = 1.0) -> None:
        super().__init__()
        self.f = VectorField(dim, hidden)
        self.n_steps = n_steps
        self.dt = T / n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = rk4_step(self.f, x, self.dt)
        return x


# ============================================================
# 2. Augmented Neural ODE
# ============================================================


class AugmentedNeuralODE(nn.Module):
    """Augmented Neural ODE: augment state with extra zeros, integrate, project back.

    Augmented state: z = cat([x0, zeros(a_dim)]) in R^(dim+a_dim).
    Integrate dz/dt = f_aug(z).
    Project z_T -> dim via linear layer.
    Input: (B, dim). Output: (B, dim).
    """

    def __init__(
        self, dim: int = 8, a_dim: int = 4, hidden: int = 32, n_steps: int = 6, T: float = 1.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.a_dim = a_dim
        self.f = VectorField(dim + a_dim, hidden)
        self.proj = nn.Linear(dim + a_dim, dim)
        self.n_steps = n_steps
        self.dt = T / n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Augment: concat zeros along last dim
        aug = torch.zeros(*x.shape[:-1], self.a_dim, device=x.device, dtype=x.dtype)
        z = torch.cat([x, aug], dim=-1)  # (B, dim+a_dim)
        # Unrolled RK4 integration
        for _ in range(self.n_steps):
            z = rk4_step(self.f, z, self.dt)
        # Project back to original dim
        return self.proj(z)


# ============================================================
# 3. Multiple Shooting Layer
# ============================================================


class MultipleShootingLayer(nn.Module):
    """Multiple Shooting Layer: parallel sub-interval integration + junction matching.

    Split [0, T] into n_segments sub-intervals, each with a learnable shooting
    parameter s_i (the initial condition for segment i).
    Integrate each from s_i for steps_per_seg RK4 steps.
    At each interior junction, apply a soft matching correction:
      endpoint of segment i is nudged toward s_{i+1} via a learned matching MLP.
    Output: concatenated endpoints (B, n_segments * dim).

    This shows the multiple-shooting structure: parallel segments + junction matching.
    """

    def __init__(
        self,
        dim: int = 8,
        hidden: int = 32,
        n_segments: int = 3,
        steps_per_seg: int = 2,
        T: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_segments = n_segments
        self.steps_per_seg = steps_per_seg
        self.dt = T / (n_segments * steps_per_seg)

        # Shared vector field across all segments
        self.f = VectorField(dim, hidden)

        # Learnable initial shooting parameters s_i for segments 1..n_segments-1
        # Segment 0 uses the actual input; segments 1+ have learned parameters.
        self.shooting_params = nn.ParameterList(
            [nn.Parameter(torch.randn(1, dim)) for _ in range(n_segments - 1)]
        )

        # Junction matching MLPs: project endpoint + shooting param -> correction
        self.junction_mlps = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(n_segments - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim) -- initial condition (shooting param for segment 0)
        B = x.shape[0]
        endpoints = []

        # Segment 0: integrate from x
        state = x
        for _ in range(self.steps_per_seg):
            state = rk4_step(self.f, state, self.dt)
        endpoints.append(state)

        # Segments 1..n_segments-1: integrate from shooting params
        for i in range(self.n_segments - 1):
            s_i = self.shooting_params[i].expand(B, -1)  # (B, dim)

            # Integrate segment i+1 from its shooting parameter
            seg_state = s_i
            for _ in range(self.steps_per_seg):
                seg_state = rk4_step(self.f, seg_state, self.dt)

            # Junction matching: nudge toward continuity
            # Use the junction MLP to compute correction from [endpoint_prev, s_i]
            junction_input = torch.cat([endpoints[-1], s_i], dim=-1)
            correction = torch.tanh(self.junction_mlps[i](junction_input))
            seg_state = seg_state + correction

            endpoints.append(seg_state)

        return torch.cat(endpoints, dim=-1)  # (B, n_segments * dim)


# ============================================================
# Zero-arg builders and example inputs
# ============================================================


def build_neuralode() -> nn.Module:
    return NeuralODE(dim=8, hidden=32, n_steps=6)


def build_augmented_neuralode() -> nn.Module:
    return AugmentedNeuralODE(dim=8, a_dim=4, hidden=32, n_steps=6)


def build_multiple_shooting() -> nn.Module:
    return MultipleShootingLayer(dim=8, hidden=32, n_segments=3, steps_per_seg=2)


def example_input_ode() -> torch.Tensor:
    """Initial state (1, 8)."""
    return torch.randn(1, 8)


def example_input_ode_aug() -> torch.Tensor:
    """Initial state (1, 8)."""
    return torch.randn(1, 8)


def example_input_shooting() -> torch.Tensor:
    """Initial state (1, 8)."""
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [
    (
        "Neural ODE (torchdyn-style, RK4-unrolled vector-field MLP integration)",
        "build_neuralode",
        "example_input_ode",
        "2018",
        "DC",
    ),
    (
        "Augmented Neural ODE (state augmented with zeros before RK4-unrolled integration)",
        "build_augmented_neuralode",
        "example_input_ode_aug",
        "2019",
        "DC",
    ),
    (
        "Multiple Shooting Layer (parallel sub-interval ODE integration + junction matching)",
        "build_multiple_shooting",
        "example_input_shooting",
        "2021",
        "DC",
    ),
]
