"""Flash-Linear-Attention token-mixing layers (faithful recurrent reference forms).

Source: fla-org/flash-linear-attention (``fla/layers/*.py``, ``fla/ops/*``).
These are linear-attention / gated-RNN sequence mixers that replace softmax
attention with an explicit recurrent state update.  Each block here reproduces
the *fused_recurrent* (plain per-timestep loop) form of the published kernel,
which is the exact math -- the Triton "chunk" path is only a parallelization of
the same recurrence.  Random init, CPU, forward-only (no Triton needed).

Families (one block each), with paper:
  - DeltaNet            (Yang et al. 2024, arXiv:2406.06484) -- ungated delta rule
  - GatedDeltaNet       (Yang et al. 2024, arXiv:2412.06464) -- delta rule + scalar decay
  - GatedDeltaProduct   (Siems et al. 2025, arXiv:2502.10297) -- n_h delta steps/token
  - GLA                 (Yang et al. 2024, arXiv:2312.06635) -- per-key-dim gated linear attn
  - GSA                 (Zhang et al. 2024, arXiv:2409.07146) -- gated slot attention (bounded memory)
  - HGRN                (Qin et al. 2023, arXiv:2311.04823) -- hierarchically gated RNN

Shared submodules: depthwise causal short conv (kernel 4), RMSNorm, and a gated
RMSNorm output (``RMSNorm(o) * swish(g)``).  Each block consumes a token sequence
``(B, T, D)`` and returns ``(B, T, D)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared submodules
# ============================================================


class ShortConvolution(nn.Module):
    """Depthwise causal Conv1d over the time axis (kernel ``conv_size``)."""

    def __init__(self, dim: int, conv_size: int = 4, activation: str | None = "silu") -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=conv_size, groups=dim, padding=conv_size - 1, bias=False
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.shape[1]
        y = self.conv(x.transpose(1, 2))[..., :t].transpose(1, 2)
        if self.activation == "silu":
            y = F.silu(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FusedRMSNormGated(nn.Module):
    """``RMSNorm(o) * swish(g)`` -- the gate multiplies AFTER normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps)

    def forward(self, o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.norm(o) * F.silu(g)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt())


# ============================================================
# DeltaNet -- ungated delta rule
# ============================================================


class DeltaNet(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, conv_size: int = 4) -> None:
        super().__init__()
        self.h = num_heads
        self.dk = hidden_size // num_heads
        self.dv = self.dk
        key_dim = hidden_size
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.q_conv = ShortConvolution(key_dim, conv_size, "silu")
        self.k_conv = ShortConvolution(key_dim, conv_size, "silu")
        self.v_conv = ShortConvolution(key_dim, conv_size, "silu")
        self.o_norm = RMSNorm(self.dv)
        self.o_proj = nn.Linear(key_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_conv(self.q_proj(x)).view(B, T, self.h, self.dk)
        k = self.k_conv(self.k_proj(x)).view(B, T, self.h, self.dk)
        v = self.v_conv(self.v_proj(x)).view(B, T, self.h, self.dv)
        q, k = _l2norm(q), _l2norm(k)
        beta = torch.sigmoid(self.b_proj(x))  # (B,T,H)
        S = x.new_zeros(B, self.h, self.dk, self.dv)
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            old = torch.einsum("bhkd,bhk->bhd", S, kt)
            delta = beta[:, t].unsqueeze(-1) * (vt - old)
            S = S + torch.einsum("bhk,bhd->bhkd", kt, delta)
            outs.append(torch.einsum("bhkd,bhk->bhd", S, qt))
        o = torch.stack(outs, dim=1)  # (B,T,H,dv)
        o = self.o_norm(o).reshape(B, T, -1)
        return self.o_proj(o)


# ============================================================
# GatedDeltaNet -- delta rule + scalar (Mamba2-style) decay
# ============================================================


class GatedDeltaNet(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, conv_size: int = 4) -> None:
        super().__init__()
        self.h = num_heads
        self.dk = hidden_size // num_heads
        self.dv = self.dk
        key_dim = hidden_size
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.q_conv = ShortConvolution(key_dim, conv_size, "silu")
        self.k_conv = ShortConvolution(key_dim, conv_size, "silu")
        self.v_conv = ShortConvolution(key_dim, conv_size, "silu")
        self.g_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.dv)
        self.o_proj = nn.Linear(key_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_conv(self.q_proj(x)).view(B, T, self.h, self.dk)
        k = self.k_conv(self.k_proj(x)).view(B, T, self.h, self.dk)
        v = self.v_conv(self.v_proj(x)).view(B, T, self.h, self.dv)
        q, k = _l2norm(q), _l2norm(k)
        beta = torch.sigmoid(self.b_proj(x))
        a = self.a_proj(x)
        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)  # log-decay (B,T,H)
        alpha = torch.exp(g)
        S = x.new_zeros(B, self.h, self.dk, self.dv)
        outs = []
        for t in range(T):
            S = alpha[:, t].view(B, self.h, 1, 1) * S
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            old = torch.einsum("bhkd,bhk->bhd", S, kt)
            delta = beta[:, t].unsqueeze(-1) * (vt - old)
            S = S + torch.einsum("bhk,bhd->bhkd", kt, delta)
            outs.append(torch.einsum("bhkd,bhk->bhd", S, qt))
        o = torch.stack(outs, dim=1)
        gate = self.g_proj(x).view(B, T, self.h, self.dv)
        o = self.o_norm(o, gate).reshape(B, T, -1)
        return self.o_proj(o)


# ============================================================
# GatedDeltaProduct -- n_h delta sub-steps per token (Householder product)
# ============================================================


class GatedDeltaProduct(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_householder: int = 2,
        conv_size: int = 4,
    ) -> None:
        super().__init__()
        self.h = num_heads
        self.nh = num_householder
        self.dk = hidden_size // num_heads
        self.dv = self.dk
        key_dim = hidden_size
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim * num_householder, bias=False)
        self.v_proj = nn.Linear(hidden_size, key_dim * num_householder, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads * num_householder, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.q_conv = ShortConvolution(key_dim, conv_size, "silu")
        self.g_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.dv)
        self.o_proj = nn.Linear(key_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_conv(self.q_proj(x)).view(B, T, self.h, self.dk)
        q = _l2norm(q)
        k = self.k_proj(x).view(B, T, self.nh, self.h, self.dk)
        v = self.v_proj(x).view(B, T, self.nh, self.h, self.dv)
        k = _l2norm(k)
        beta = torch.sigmoid(self.b_proj(x)).view(B, T, self.nh, self.h)
        a = self.a_proj(x)
        alpha = torch.exp(-torch.exp(self.A_log) * F.softplus(a + self.dt_bias))  # (B,T,H)
        S = x.new_zeros(B, self.h, self.dk, self.dv)
        outs = []
        for t in range(T):
            S = alpha[:, t].view(B, self.h, 1, 1) * S  # decay once per real token
            for j in range(self.nh):  # n_h delta sub-steps
                kj, vj = k[:, t, j], v[:, t, j]
                old = torch.einsum("bhkd,bhk->bhd", S, kj)
                delta = beta[:, t, j].unsqueeze(-1) * (vj - old)
                S = S + torch.einsum("bhk,bhd->bhkd", kj, delta)
            outs.append(torch.einsum("bhkd,bhk->bhd", S, q[:, t]))  # read once per token
        o = torch.stack(outs, dim=1)
        gate = self.g_proj(x).view(B, T, self.h, self.dv)
        o = self.o_norm(o, gate).reshape(B, T, -1)
        return self.o_proj(o)


# ============================================================
# GLA -- per-key-dim data-dependent gated linear attention
# ============================================================


class GLA(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        gate_low_rank_dim: int = 16,
        conv_size: int = 4,
    ) -> None:
        super().__init__()
        self.h = num_heads
        self.dk = (hidden_size // 2) // num_heads  # expand_k=0.5
        self.dv = hidden_size // num_heads  # expand_v=1.0
        key_dim = hidden_size // 2
        value_dim = hidden_size
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, value_dim, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, key_dim, bias=True),
        )
        self.g_proj = nn.Linear(hidden_size, value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.dv)
        self.o_proj = nn.Linear(value_dim, hidden_size, bias=False)
        self.gate_logit_normalizer = 16.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dk)
        k = self.k_proj(x).view(B, T, self.h, self.dk)
        v = self.v_proj(x).view(B, T, self.h, self.dv)
        gk = F.logsigmoid(self.gk_proj(x)) / self.gate_logit_normalizer
        gk = gk.view(B, T, self.h, self.dk)
        alpha = torch.exp(gk)  # per-key-dim decay (B,T,H,dk)
        S = x.new_zeros(B, self.h, self.dk, self.dv)
        outs = []
        for t in range(T):
            S = alpha[:, t].unsqueeze(-1) * S + torch.einsum("bhk,bhd->bhkd", k[:, t], v[:, t])
            outs.append(torch.einsum("bhkd,bhk->bhd", S, q[:, t]))
        o = torch.stack(outs, dim=1)
        gate = self.g_proj(x).view(B, T, self.h, self.dv)
        o = self.o_norm(o, gate).reshape(B, T, -1)
        return self.o_proj(o)


# ============================================================
# GSA -- gated slot attention (bounded slot memory, two-pass softmax)
# ============================================================


class GSA(nn.Module):
    def __init__(
        self, hidden_size: int = 128, num_heads: int = 4, num_slots: int = 16, conv_size: int = 4
    ) -> None:
        super().__init__()
        self.h = num_heads
        self.dk = hidden_size // num_heads
        self.dv = self.dk
        self.m = num_slots
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.f_proj = nn.Linear(hidden_size, num_heads * num_slots, bias=False)
        self.g_norm = RMSNorm(hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_logit_normalizer = 8.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = F.silu(self.q_proj(x)).view(B, T, self.h, self.dk)
        k = F.silu(self.k_proj(x)).view(B, T, self.h, self.dk)
        v = F.silu(self.v_proj(x)).view(B, T, self.h, self.dv)
        f = F.logsigmoid(self.f_proj(x)) / self.gate_logit_normalizer
        f = f.view(B, T, self.h, self.m)
        decay = torch.exp(f)  # (B,T,H,m)
        s = 1.0 - decay  # write coefficient per slot
        hk = x.new_zeros(B, self.h, self.m, self.dk)  # slot key memory
        hv = x.new_zeros(B, self.h, self.m, self.dv)  # slot value memory
        outs = []
        for t in range(T):
            d = decay[:, t].unsqueeze(-1)  # (B,H,m,1)
            sw = s[:, t].unsqueeze(-1)
            hk = d * hk + sw * k[:, t].unsqueeze(2)
            hv = d * hv + sw * v[:, t].unsqueeze(2)
            # pass 1: query reads slot logits from hk; softmax over slots
            logits = torch.einsum("bhmk,bhk->bhm", hk, q[:, t])
            p = torch.softmax(logits, dim=-1)
            # pass 2: read values out of hv weighted by slot probs
            outs.append(torch.einsum("bhm,bhmd->bhd", p, hv))
        o = torch.stack(outs, dim=1).reshape(B, T, -1)
        o = self.g_norm(F.silu(o))
        return self.o_proj(o)


# ============================================================
# HGRN -- hierarchically gated (elementwise) recurrent network
# ============================================================


class HGRN(nn.Module):
    def __init__(self, hidden_size: int = 128, conv_size: int = 4) -> None:
        super().__init__()
        self.i_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.f_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_norm = FusedRMSNormGated(hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        f = F.logsigmoid(self.f_proj(x))  # log forget gate
        i = F.silu(self.i_proj(x)) * (1.0 - torch.exp(f))  # gated input (swiglu)
        decay = torch.exp(f)
        h = x.new_zeros(B, D)
        outs = []
        for t in range(T):
            h = decay[:, t] * h + i[:, t]
            outs.append(h)
        o = torch.stack(outs, dim=1)
        o = self.g_norm(o, self.g_proj(x))
        return self.o_proj(o)


# ============================================================
# Builders + example input + menagerie entries
# ============================================================


def _seq_input(t: int = 24) -> torch.Tensor:
    """Token sequence ``(1, t, 128)`` (B, T, D)."""
    return torch.randn(1, t, 128)


def build_deltanet() -> nn.Module:
    return DeltaNet(hidden_size=128, num_heads=4)


def build_gated_deltanet() -> nn.Module:
    return GatedDeltaNet(hidden_size=128, num_heads=4)


def build_gated_deltaproduct() -> nn.Module:
    return GatedDeltaProduct(hidden_size=128, num_heads=4, num_householder=2)


def build_gla() -> nn.Module:
    return GLA(hidden_size=128, num_heads=4)


def build_gsa() -> nn.Module:
    return GSA(hidden_size=128, num_heads=4, num_slots=16)


def build_hgrn() -> nn.Module:
    return HGRN(hidden_size=128)


def example_input() -> torch.Tensor:
    return _seq_input(24)


def example_input_short() -> torch.Tensor:
    """Shorter ``(1, 10, 128)`` sequence for the heavier gated-decay variants.

    The per-timestep decay+delta graph is wider; a shorter unroll keeps the
    Graphviz sibling-ordering layout under the render time budget.  Dynamics are
    identical at any sequence length.
    """
    return _seq_input(10)


MENAGERIE_ENTRIES = [
    ("DeltaNet (delta-rule linear attention)", "build_deltanet", "example_input", "2024", "DC"),
    (
        "GatedDeltaNet (delta rule + scalar decay, Mamba2-style)",
        "build_gated_deltanet",
        "example_input_short",
        "2024",
        "DC",
    ),
    (
        "GatedDeltaProduct (Householder-product delta steps per token)",
        "build_gated_deltaproduct",
        "example_input_short",
        "2025",
        "DC",
    ),
    ("GLA (Gated Linear Attention, per-key-dim decay)", "build_gla", "example_input", "2024", "DC"),
    ("GSA (Gated Slot Attention, bounded slot memory)", "build_gsa", "example_input", "2024", "DC"),
    ("HGRN (Hierarchically Gated Recurrent Network)", "build_hgrn", "example_input", "2023", "DC"),
]
