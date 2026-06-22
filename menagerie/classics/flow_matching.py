"""Conditional Flow Matching and variants (torchcfm-style, pure PyTorch).

Conditional Flow Matching (CFM): Lipman et al., ICLR 2023. arXiv:2210.02747
Exact OT Flow Matching: Pooladian et al. / Tong et al. 2023. arXiv:2302.00482 / arXiv:2304.14772
Schrodinger Bridge Flow Matching (SB-CFM): De Bortoli et al. / Tong et al. 2023. arXiv:2303.01469
Variance-Preserving Flow Matching (VP-CFM): Albergo & Vanden-Eijnden 2023. arXiv:2209.15571

Sources:
  https://github.com/atong01/conditional-flow-matching
  https://github.com/facebookresearch/flow_matching

ARCHITECTURE NOTE:
  All four variants share the SAME velocity-field network architecture:
    v_theta(x, t) -- an MLP that takes a data point x + a sinusoidal t-embedding
    and predicts the velocity (dx/dt) at time t.

  The variants differ ONLY in the TRAINING-TIME conditional probability path used to
  generate (x0, x1, t) pairs (i.e., which coupling between source and target distribution):
    - CFM:         independent coupling, linear path x_t = (1-t)*x0 + t*x1
    - Exact OT:    minibatch OT coupling (Hungarian/Sinkhorn), same linear path
    - SB-CFM:      Schrodinger bridge coupling (diffusion-adjusted OT path)
    - VP-CFM:      variance-preserving (diffusion) interpolant, different t-schedule

  Since the NETWORK ARCHITECTURE is the same, we build one VelocityFieldNet module,
  then expose four separate entry points that document which path each corresponds to.
  For VP-CFM, the time embedding uses a variance-preserving schedule (sigma(t) scaling),
  which is reflected in the time_embed method below.

Distinctive primitive shown:
  - Sinusoidal time embedding (standard for diffusion/flow models).
  - Velocity field MLP v(x, t) -> dx/dt.
  - Few-step Euler ODE integration of the velocity field (sampling path).
  - For VP-CFM: different time embedding (sigma-scaled).

Compact:
  - Input: (1, dim=8) data point.
  - n_steps=6 Euler integration steps.
  - d_model=32, hidden=64.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal time embedding
# ============================================================


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding: scalar t -> (embed_dim,) vector."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        # Learnable projection after sinusoidal encoding
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1) scalar time values in [0, 1]
        t = t.reshape(-1)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class VPSinusoidalTimeEmbed(nn.Module):
    """Variance-preserving time embedding: t -> sigma(t)-scaled sinusoidal + proj.

    For VP-CFM the interpolant is x_t = alpha(t)*x1 + sigma(t)*eps where
    sigma(t) = sqrt(1 - alpha(t)^2) with alpha(t) = exp(-int beta(s) ds).
    Here we use a simple VP schedule: sigma(t) = sin(pi/2 * t).
    """

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        half = self.embed_dim // 2
        # VP schedule: sigma = sin(pi/2 * t), alpha = cos(pi/2 * t)
        sigma = torch.sin(math.pi / 2 * t)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # Append sigma as extra conditioning signal
        emb = torch.cat([emb, sigma.unsqueeze(-1)], dim=-1)
        return self.proj(emb)


# ============================================================
# Velocity field network
# ============================================================


class VelocityFieldNet(nn.Module):
    """Velocity field v_theta(x, t) -> dx/dt.

    Architecture: concatenate [x, t_emb] -> MLP -> velocity (same dim as x).
    The time embedding module is injected so VP-CFM can use a different one.
    """

    def __init__(
        self,
        data_dim: int = 8,
        t_embed_dim: int = 16,
        hidden_dim: int = 64,
        time_embed_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.t_embed = time_embed_module or SinusoidalTimeEmbed(t_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, data_dim), t: (B,) time in [0, 1]
        t_emb = self.t_embed(t)  # (B, t_embed_dim)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


# ============================================================
# Flow Matching model: velocity net + Euler ODE integrator
# ============================================================


class FlowMatchingModel(nn.Module):
    """Flow Matching model = velocity field net + Euler ODE integrator (sampling).

    At inference, integrates dx/dt = v_theta(x, t) from t=0 to t=1
    using n_steps Euler steps.  The velocity net is the architecture;
    the probability path used during training is documented per-entry.

    The Euler integration loop is UNROLLED in forward() so TorchLens captures
    the full integration graph.
    """

    def __init__(
        self,
        data_dim: int = 8,
        t_embed_dim: int = 16,
        hidden_dim: int = 64,
        n_steps: int = 6,
        time_embed_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.v = VelocityFieldNet(data_dim, t_embed_dim, hidden_dim, time_embed_module)
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Euler integration of the velocity field from t=0 to t=1.

        x: (B, data_dim) -- starting point (e.g. sampled from source distribution).
        Returns x at t=1 (sample from target distribution).
        """
        B = x.shape[0]
        for i in range(self.n_steps):
            t = torch.full((B,), i * self.dt, device=x.device, dtype=x.dtype)
            v = self.v(x, t)
            x = x + self.dt * v
        return x


# ============================================================
# Zero-arg builders and example inputs
# ============================================================


def build_cfm() -> nn.Module:
    """Conditional Flow Matching: linear path, independent coupling."""
    return FlowMatchingModel(
        data_dim=8,
        t_embed_dim=16,
        hidden_dim=64,
        n_steps=6,
        time_embed_module=SinusoidalTimeEmbed(16),
    )


def build_exact_ot_cfm() -> nn.Module:
    """Exact OT Flow Matching: linear path, minibatch OT coupling.
    Same velocity-field architecture as CFM; OT coupling is a training detail.
    """
    return FlowMatchingModel(
        data_dim=8,
        t_embed_dim=16,
        hidden_dim=64,
        n_steps=6,
        time_embed_module=SinusoidalTimeEmbed(16),
    )


def build_sb_cfm() -> nn.Module:
    """Schrodinger Bridge Flow Matching: diffusion-bridge conditional paths.
    Same velocity-field architecture; SB coupling is a training detail.
    """
    return FlowMatchingModel(
        data_dim=8,
        t_embed_dim=16,
        hidden_dim=64,
        n_steps=6,
        time_embed_module=SinusoidalTimeEmbed(16),
    )


def build_vp_cfm() -> nn.Module:
    """Variance-Preserving Flow Matching: VP interpolant with sigma(t)=sin(pi/2*t).
    Uses a sigma-augmented time embedding to reflect the VP schedule.
    """
    return FlowMatchingModel(
        data_dim=8,
        t_embed_dim=16,
        hidden_dim=64,
        n_steps=6,
        time_embed_module=VPSinusoidalTimeEmbed(16),
    )


def example_input() -> torch.Tensor:
    """Source distribution sample (1, 8) for flow ODE integration."""
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [
    (
        "Conditional Flow Matching (velocity-field MLP + Euler ODE, linear independent-coupling path)",
        "build_cfm",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "Exact OT Flow Matching (same velocity-field net; minibatch-OT coupling at train time)",
        "build_exact_ot_cfm",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "Schrodinger Bridge Flow Matching (same velocity-field net; SB diffusion-bridge coupling)",
        "build_sb_cfm",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "Variance-Preserving Flow Matching (VP interpolant, sigma-augmented time embedding)",
        "build_vp_cfm",
        "example_input",
        "2023",
        "DC",
    ),
]
