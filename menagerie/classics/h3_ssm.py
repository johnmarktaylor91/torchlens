"""H3: Hungry Hungry Hippos -- two-SSM gated sequence layer.

Fu et al. (2023), "Hungry Hungry Hippos: Towards Language Modeling with
State Space Models".  ICLR 2023.  arXiv:2212.14052.
Source: https://github.com/HazyResearch/H3

Distinctive primitive: H3 replaces attention with two SSMs and a
multiplicative interaction.  The block:
  1. Project x -> Q (d), K (d), V (d) via linear heads.
  2. Apply a SHIFT SSM (or simple shift-conv) to Q: Q' = shift_ssm(Q).
  3. Compute elementwise product: inter = Q' * K   (the "diagonal SSM of
     shifted-Q times K" gating).
  4. Apply a DIAGONAL SSM to `inter`: out_inner = diag_ssm(inter).
  5. Multiply by V: out = out_inner * V.
  6. Output projection.

Both SSMs use the diagonal LTI kernel (same as S4D): materialise K over seq_len,
then depthwise conv1d.  The shift SSM uses A = 0 (pure shift / unit delay),
while the diagonal SSM uses learned negative-real A.

Compact config: d_model=32, d_state=8, seq_len=16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagSSMKernel(nn.Module):
    """Diagonal-A SSM kernel (S4D style) for one H3 SSM head."""

    def __init__(self, d: int, d_state: int = 8) -> None:
        super().__init__()
        self.d = d
        self.d_state = d_state
        vals = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_init = -0.5 * vals  # negative reals
        self.A_log = nn.Parameter(torch.log(-A_init).unsqueeze(0).expand(d, -1).clone())
        self.B = nn.Parameter(torch.randn(d, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) -> (B, L, d)"""
        B, L, d = x.shape
        A = -torch.exp(self.A_log)  # (d, d_state)
        t = torch.arange(L, device=A.device, dtype=A.dtype)
        Apow = torch.exp(A.unsqueeze(1) * t.unsqueeze(0).unsqueeze(2))  # (d, L, d_state)
        CB = self.C * self.B  # (d, d_state)
        K = (CB.unsqueeze(1) * Apow).sum(-1)  # (d, L)
        xu = x.permute(0, 2, 1)  # (B, d, L)
        Kw = K.unsqueeze(1)  # (d, 1, L)
        xu_pad = F.pad(xu, (L - 1, 0))
        y = F.conv1d(xu_pad, Kw, groups=d).permute(0, 2, 1)  # (B, L, d)
        y = y + self.D * x
        return y


class ShiftSSM(nn.Module):
    """Shift SSM: unit-delay per channel (A=0 => K = [1, 0, 0, ...]).

    Equivalent to shifting the sequence by 1 position (causal).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.D = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) -> shifted (B, L, d)"""
        # shift right by 1 (causal: position t sees x[t-1])
        shifted = torch.roll(x, shifts=1, dims=1)
        shifted[:, 0, :] = 0.0  # zero out wrapped position
        return shifted + self.D * x


class H3Layer(nn.Module):
    """H3 gated two-SSM block.

    Q' = shift_ssm(Q),  inter = Q' * K,
    out = diag_ssm(inter) * V,  then output projection.
    """

    def __init__(self, d_model: int = 32, d_state: int = 8) -> None:
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.shift = ShiftSSM(d_model)
        self.diag = DiagSSMKernel(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        r = x
        x = self.norm(x)
        qkv = self.qkv(x)  # (B, L, 3d)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q_prime = self.shift(Q)  # shift-SSM on Q
        inter = Q_prime * K  # elementwise gate
        inner = self.diag(inter)  # diagonal SSM on inter
        out = inner * V  # multiply by V
        return self.out_proj(out) + r


def build_h3_layer() -> nn.Module:
    return H3Layer(d_model=32, d_state=8).eval()


def example_input() -> torch.Tensor:
    """(1, 16, 32) -- batch=1, seq_len=16, d_model=32."""
    return torch.randn(1, 16, 32)


MENAGERIE_ENTRIES = [
    (
        "H3 (Hungry Hungry Hippos: two-SSM gated block, shift x diagonal SSM)",
        "build_h3_layer",
        "example_input",
        "2023",
        "DC",
    ),
]
