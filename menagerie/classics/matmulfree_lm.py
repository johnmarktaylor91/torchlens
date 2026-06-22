"""MatMul-Free LM: a language model without matrix multiplications.

Zhu et al. 2024, arXiv:2406.02528.
Source: ridgerchu/matmulfreellm.

Replaces every dense matmul with a **BitLinear** (RMSNorm of the activations,
8-bit per-token activation quantization, ternary {-1, 0, +1} weights), and uses:
  - an **MLGRU / HGRN-bit** token mixer: an element-wise gated linear recurrence
    ``h_t = f_t * h_{t-1} + (1 - f_t) * silu(candidate_t)`` with a swish-gated
    RMSNorm output (no matmul in the state transition),
  - a **GLU channel mixer** (BitLinear gate+up fused, SiLU-gated, BitLinear down).

This faithful reimplementation keeps the BitLinear quantization (straight-through
ternary weights), the MLGRU recurrence, and the GLU, in pre-norm residual blocks.
Random init, CPU, forward-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token 8-bit absmax quantization with straight-through estimator."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-5)
    q = (x * scale).round().clamp(-128, 127) / scale
    return x + (q - x).detach()


def _weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor ternary {-1,0,+1} quantization with straight-through estimator."""
    scale = 1.0 / w.abs().mean().clamp_min(1e-5)
    q = (w * scale).round().clamp(-1, 1) / scale
    return w + (q - w).detach()


class BitLinear(nn.Linear):
    """nn.Linear with input RMSNorm + 8-bit activations + ternary weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = _activation_quant(x)
        w = _weight_quant(self.weight)
        return F.linear(x, w, self.bias)


class FusedRMSNormSwishGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)

    def forward(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h) * F.silu(g)


class MLGRU(nn.Module):
    """MatMul-free Linear GRU (HGRN-bit) token mixer."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.i_proj = BitLinear(d_model, d_model)
        self.f_proj = BitLinear(d_model, d_model)
        self.g_proj = BitLinear(d_model, d_model)
        self.g_norm = FusedRMSNormSwishGate(d_model)
        self.o_proj = BitLinear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        f = torch.sigmoid(self.f_proj(x))
        cand = F.silu(self.i_proj(x)) * (1.0 - f)
        h = x.new_zeros(B, D)
        outs = []
        for t in range(T):
            h = f[:, t] * h + cand[:, t]
            outs.append(h)
        h_seq = torch.stack(outs, dim=1)
        o = self.g_norm(self.g_proj(x), h_seq)
        return self.o_proj(o)


class BitGLU(nn.Module):
    """GLU channel mixer built from BitLinear (fused gate+up)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_up = BitLinear(d_model, d_ff * 2)
        self.down = BitLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class MatMulFreeBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mixer = MLGRU(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = BitGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MatMulFreeLM(nn.Module):
    def __init__(
        self, vocab: int = 256, d_model: int = 128, d_ff: int = 320, n_layers: int = 4
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([MatMulFreeBlock(d_model, d_ff) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.final_norm(x))


def build_matmulfree_lm() -> nn.Module:
    return MatMulFreeLM()


def example_input() -> torch.Tensor:
    """Token-id sequence ``(1, 16)`` (short, to keep the unrolled MLGRU compact)."""
    return torch.randint(0, 256, (1, 16))


MENAGERIE_ENTRIES = [
    (
        "MatMul-Free LM (ternary BitLinear, MLGRU + GLU)",
        "build_matmulfree_lm",
        "example_input",
        "2024",
        "DC",
    ),
]
