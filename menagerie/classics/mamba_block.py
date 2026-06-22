"""Mamba: a standalone selective state-space (S6) block (mamba-ssm reference block).

Gu & Dao, 2023, arXiv:2312.00752.  Source: https://github.com/state-spaces/mamba
(``mamba_ssm/modules/mamba_simple.py`` ``Mamba`` + ``mamba_ssm/modules/block.py``
``Block``; the CUDA ``selective_scan_cuda`` and ``causal_conv1d`` packages).

Mamba is the canonical selective-SSM sequence model.  The standalone reference
block (the unit ``mamba-ssm`` exports and that downstream LMs stack) is:
``Block = residual + Mamba(RMSNorm(x))`` where the ``Mamba`` mixer is:
  1. ``in_proj`` expands d_model -> 2 * d_inner, split into (x, z) gate branches;
  2. a depthwise **causal conv1d** over the x branch (short-range mixing), SiLU;
  3. ``x_proj`` produces the INPUT-DEPENDENT selective parameters
     (delta, B, C) -- the "S6" selection that makes the SSM matrices vary per token;
  4. a **selective scan**: per-step recurrence
     ``h_t = exp(delta_t * A) h_{t-1} + delta_t * B_t * x_t``,
     ``y_t = C_t . h_t + D * x_t`` with a learned negative-real diagonal ``A``;
  5. gate by ``SiLU(z)`` and ``out_proj`` back to d_model.

This module reimplements that reference block standalone.  It is distinct from the
``jamba`` classic (which captures Jamba's HYBRID Transformer-Mamba-MoE LM with
Jamba-specific inner RMSNorms on dt/B/C); here we capture the *plain* mamba-ssm
``Block`` (RMSNorm + Mamba mixer + residual) on its own.

The CEILINGs in the menagerie are ``mamba-ssm`` (``selective_scan_cuda``) and
``causal-conv1d`` -- both custom CUDA kernels needing nvcc.  Both are OPTIMIZATIONS:
``causal_conv1d`` is a depthwise causal ``nn.Conv1d``; ``selective_scan_cuda`` is the
sequential linear recurrence above, expressed here as an explicit Python time loop
(the kernel's parallel-scan is just a fast associative reduction of the same maths).
Pure-torch reimplementation traces and renders.

Small config: d_model=64, d_state=16, short sequence so the unrolled scan stays
renderable.  One standalone Mamba block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MambaMixer(nn.Module):
    """The ``Mamba`` selective-SSM mixer (mamba-ssm ``mamba_simple.Mamba``)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)
        # Causal depthwise conv1d (causal_conv1d kernel, pure torch).
        xs = self.conv1d(xs.transpose(1, 2))[..., :T].transpose(1, 2)
        xs = F.silu(xs)
        # Input-dependent selective parameters (the S6 selection).
        dbc = self.x_proj(xs)
        dt, Bp, Cp = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        # Selective scan (selective_scan_cuda kernel, explicit time loop).
        h = x.new_zeros(B, self.d_inner, self.d_state)
        outs = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
            dBx = dt[:, t].unsqueeze(-1) * Bp[:, t].unsqueeze(1) * xs[:, t].unsqueeze(-1)
            h = dA * h + dBx
            y = torch.einsum("bds,bs->bd", h, Cp[:, t]) + self.D * xs[:, t]
            outs.append(y)
        y = torch.stack(outs, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """mamba-ssm reference ``Block``: residual + Mamba(RMSNorm(x))."""

    def __init__(self, d_model: int = 64, d_state: int = 16) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


def build_mamba_block() -> nn.Module:
    """Build one standalone Mamba (S6) reference block, d_model=64, d_state=16."""
    return MambaBlock(d_model=64, d_state=16)


def example_input() -> torch.Tensor:
    """Feature sequence ``(1, 12, 64)`` (short, to keep the unrolled scan compact)."""
    return torch.randn(1, 12, 64)


MENAGERIE_ENTRIES = [
    (
        "Mamba block (standalone selective state-space / S6 mixer)",
        "build_mamba_block",
        "example_input",
        "2023",
        "DC",
    ),
]
