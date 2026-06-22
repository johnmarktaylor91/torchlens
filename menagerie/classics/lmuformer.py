"""LMUFormer: Legendre Memory Unit + Transformer for speech (Google Speech Commands).

Liao et al. (2022), "LMUFormer: Low Complexity Yet Powerful Spiking Transformer
with Legendre Memory Units".  arXiv:2209.14491.
Source: https://github.com/zhhlee/LMUFormer

Distinctive primitive: LMU convolutional state-space layer as sequence mixer.
The LMU maps an input signal x(t) to a d_mem-dim vector of Legendre polynomial
projections via a 1st-order ODE approximated as a discrete recurrence:
  m(t) = A m(t-1) + B x(t)   (A: d_mem x d_mem, B: d_mem x 1)
  h(t) = tanh(H x(t) + M m(t))
where A and B are fixed (frozen) matrices derived from the Legendre delay
approximation and M, H are learned encoders.

In LMUFormer this LMU layer acts as a convolutional sequence encoder (the input
is the time axis), replacing multi-head self-attention in the first token-mixing
layer with a causal LMU, followed by standard transformer FFN blocks.

LMU matrices (Voelker et al. 2019): A_ij = (2i+1)(-1)^(i-j) if i>=j else 0,
  B_i = (2i+1)(-1)^i.  Scaled by 1/d_mem.

Compact config: d_model=32, d_mem=8, n_attn_layers=1, n_classes=35, seq_len=16.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lmu_matrices(d_mem: int, dtype=torch.float32):
    """Build the frozen LMU A (d_mem x d_mem) and B (d_mem,) matrices."""
    A = torch.zeros(d_mem, d_mem, dtype=dtype)
    B = torch.zeros(d_mem, dtype=dtype)
    for i in range(d_mem):
        B[i] = (2 * i + 1) * ((-1) ** i)
        for j in range(d_mem):
            if i >= j:
                A[i, j] = (2 * i + 1) * ((-1) ** (i - j))
    # Scale
    A = A / d_mem
    B = B / d_mem
    return A, B


class LMULayer(nn.Module):
    """LMU convolutional state-space layer.

    Processes (B, T, d_input) -> (B, T, d_model) sequentially.
    A and B are frozen Legendre matrices; M (d_mem x d_mem) and H (d_model x d_input)
    are learned.  The hidden state merges the LMU polynomial memory with the input.
    """

    def __init__(self, d_input: int = 32, d_model: int = 32, d_mem: int = 8) -> None:
        super().__init__()
        self.d_mem = d_mem
        self.d_model = d_model
        A, B = _lmu_matrices(d_mem)
        self.register_buffer("A", A)  # (d_mem, d_mem)
        self.register_buffer("B", B)  # (d_mem,)
        # Learned: H: d_model x d_input, M: d_model x d_mem
        self.H = nn.Linear(d_input, d_model, bias=False)
        self.M = nn.Linear(d_mem, d_model, bias=False)
        self.theta = nn.Parameter(torch.ones(1))  # scalar input projection for LMU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_input) -> (B, T, d_model)"""
        B, T, din = x.shape
        # LMU state (B, d_mem)
        m = torch.zeros(B, self.d_mem, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            xt = x[:, t, :]  # (B, din)
            # Project to scalar via mean (simplified input encoding for compactness)
            u = xt.mean(dim=-1, keepdim=True) * self.theta  # (B, 1)
            # Legendre recurrence: m = A @ m + B * u
            m = (m @ self.A.t()) + self.B.unsqueeze(0) * u  # (B, d_mem)
            # h = tanh(H x + M m)
            h = torch.tanh(self.H(xt) + self.M(m))  # (B, d_model)
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # (B, T, d_model)


class LMUTransformerLayer(nn.Module):
    """Standard pre-norm transformer layer (no LMU, used as stack after LMU)."""

    def __init__(self, d_model: int = 32, n_heads: int = 2, d_ff: int = 64) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


class LMUFormerGSC(nn.Module):
    """LMUFormer: LMU sequence encoder + transformer layers for speech commands.

    Input: (B, T, d_input) raw (or mel) features.
    LMU layer processes the sequence causally, then transformer layers refine.
    Final classification via mean pooling + linear head.
    """

    def __init__(
        self,
        d_input: int = 16,
        d_model: int = 32,
        d_mem: int = 8,
        n_attn_layers: int = 1,
        n_classes: int = 35,
    ) -> None:
        super().__init__()
        self.lmu = LMULayer(d_input, d_model, d_mem)
        self.attn_layers = nn.ModuleList(
            [LMUTransformerLayer(d_model, 2, 64) for _ in range(n_attn_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_input) -> (B, n_classes)"""
        x = self.lmu(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling over time
        return self.cls(x)


def build_lmuformer_gsc() -> nn.Module:
    return LMUFormerGSC(d_input=16, d_model=32, d_mem=8, n_attn_layers=1, n_classes=35).eval()


def example_input() -> torch.Tensor:
    """(1, 16, 16) -- batch=1, T=16 frames, d_input=16 (tiny mel features)."""
    return torch.randn(1, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "LMUFormer GSC (Legendre Memory Unit conv SSM + transformer for speech commands)",
        "build_lmuformer_gsc",
        "example_input",
        "2022",
        "DC",
    ),
    ("lmuformer", "build_lmuformer_gsc", "example_input", "2022", "DC"),
]
