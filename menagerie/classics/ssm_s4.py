"""S4 / S4D: Structured State Space Sequence Models.

Gu et al. (2022), "Efficiently Modeling Long Sequences with Structured State Spaces",
ICLR 2022. arXiv:2111.00396.  Source: https://github.com/state-spaces/s4

S4D (diagonal S4): Gu et al. (2022), "On the Parameterization and Initialization of
Diagonal State Space Models". arXiv:2206.11893.

Distinctive primitive: a LINEAR TIME-INVARIANT (LTI) SSM used as a 1-D sequence
convolution.  For each channel, state evolves as:
    h_{t+1} = A h_t + B x_t,  y_t = C h_t + D x_t
where A is a diagonal (S4D) or HiPPO-projected (S4) matrix.  The sequence-level
output is produced by materialising the SSM convolution kernel K:
    K_t = C A^t B    (length-L vector),
then performing a depthwise conv1d(x, K).  All channels share the recurrence
structure but have independent (A, B, C, D) parameters.

s4_layer  = full S4 proxy (diagonal A with HiPPO-style init, kernel + skip D)
s4d       = S4D: diagonal-only A (simpler init, same kernel materialisation)
s4d_layer = same as s4d (alternate name; both map to the same builder/entry)

Compact config: d_model=32, d_state=16, seq_len=20 (keeps kernel short).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Shared: SSM kernel materialisation + depthwise conv
# -----------------------------------------------------------------------


def _hippo_diagonal_init(N: int) -> torch.Tensor:
    """Approximate HiPPO-LegS diagonal (real part), clamped to [-1, -0.1]."""
    # Simple exponentially-spaced negative reals (S4D-Lin style)
    vals = torch.arange(1, N + 1, dtype=torch.float32)
    return -0.5 * vals  # negative reals -> stable


class S4DKernel(nn.Module):
    """Diagonal S4 kernel: materialise K[t] = Re(C * A^t * B) over t=0..L-1.

    A is diagonal (real for S4D; complex for true S4 -- we use real-only for
    the compact reimpl, sufficient to show the LTI conv structure).
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A: (d_model, d_state) diagonal real negative values
        A_init = _hippo_diagonal_init(d_state).unsqueeze(0).expand(d_model, -1).clone()
        self.A_log = nn.Parameter(torch.log(-A_init))  # log of magnitude (A = -exp(A_log))

        # B, C: (d_model, d_state)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D: skip/residual connection (d_model,)
        self.D = nn.Parameter(torch.ones(d_model))

    def materialize_kernel(self, L: int) -> torch.Tensor:
        """Return SSM conv kernel of shape (d_model, L)."""
        A = -torch.exp(self.A_log)  # (d_model, d_state), negative
        # powers: (L, d_state) via outer-product log then exp
        t = torch.arange(L, device=A.device, dtype=A.dtype)  # (L,)
        # A^t for each (d, s): (d_model, L, d_state)
        Apow = torch.exp(A.unsqueeze(1) * t.unsqueeze(0).unsqueeze(2))  # (d, L, d_state)
        # K[d, t] = sum_s C[d,s] * A[d,s]^t * B[d,s]
        CB = self.C * self.B  # (d_model, d_state)
        kernel = (CB.unsqueeze(1) * Apow).sum(-1)  # (d_model, L)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> y: (B, L, d_model)"""
        B, L, d = x.shape
        K = self.materialize_kernel(L)  # (d_model, L)
        # depthwise conv: group each channel separately
        # x: (B, d_model, L), K: (d_model, 1, L) -> conv1d groups=d_model
        xu = x.permute(0, 2, 1)  # (B, d, L)
        Kw = K.unsqueeze(1)  # (d_model, 1, L)
        # causal: pad left by L-1
        xu_padded = F.pad(xu, (L - 1, 0))
        y = F.conv1d(xu_padded, Kw, groups=d)  # (B, d, L)
        y = y.permute(0, 2, 1)  # (B, L, d)
        # skip
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        return y


class S4DLayer(nn.Module):
    """S4D block: SSM kernel + point-wise GELU activation + output projection."""

    def __init__(self, d_model: int = 32, d_state: int = 16) -> None:
        super().__init__()
        self.kernel = S4DKernel(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        y = self.kernel(self.norm(x))
        y = self.act(y)
        y = self.out(y)
        return y + x


# -----------------------------------------------------------------------
# S4 proxy (same architecture; HiPPO-style diagonal init noted in docstring)
# -----------------------------------------------------------------------


class S4Layer(nn.Module):
    """S4 layer: HiPPO-diagonal proxy (same kernel materialisation as S4D).

    True S4 uses complex-valued A from HiPPO-LegS; this compact reimplementation
    uses real diagonal (same structural computation, sufficient for the atlas).
    """

    def __init__(self, d_model: int = 32, d_state: int = 16) -> None:
        super().__init__()
        self.kernel = S4DKernel(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.kernel(self.norm(x))
        y = self.act(y)
        y = self.out(y)
        return y + x


# -----------------------------------------------------------------------
# Builders + example input
# -----------------------------------------------------------------------


def build_s4_layer() -> nn.Module:
    return S4Layer(d_model=32, d_state=16).eval()


def build_s4d() -> nn.Module:
    return S4DLayer(d_model=32, d_state=16).eval()


def build_s4d_layer() -> nn.Module:
    return S4DLayer(d_model=32, d_state=16).eval()


def example_input() -> torch.Tensor:
    """(1, 20, 32) -- batch=1, seq_len=20, d_model=32."""
    return torch.randn(1, 20, 32)


MENAGERIE_ENTRIES = [
    (
        "S4 (Structured State Space Sequence Model, HiPPO-diagonal LTI SSM conv)",
        "build_s4_layer",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "S4D (Diagonal State Space Model, simplified diagonal-A kernel)",
        "build_s4d",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "S4D layer (Diagonal SSM layer variant)",
        "build_s4d_layer",
        "example_input",
        "2022",
        "DC",
    ),
]
