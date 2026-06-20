"""Jamba: a hybrid Transformer-Mamba mixture-of-experts language model.

Lieber et al. (AI21 Labs) 2024, arXiv:2403.19887.
Source: HuggingFace ``transformers`` ``JambaForCausalLM``.

Jamba interleaves three block types in a fixed period:
  - **Mamba (Mamba-1) mixer** layers (the majority),
  - **Transformer self-attention** layers (1 of every 8, at i % 8 == 4),
  - **MoE feed-forward** every 2 layers (i % 2 == 1), top-2 over 16 experts;
    the other layers use a single dense SwiGLU MLP.

One representative 8-layer block is built here (Mamba x7 + Attention x1; MoE on
odd layers), reproducing Jamba's hybrid interleaving and inner-RMSNorm Mamba
mixer.  Random init, CPU, forward-only; small hidden size for a compact graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class JambaMambaMixer(nn.Module):
    """Mamba-1 selective-state-space mixer with Jamba inner RMSNorms on dt/B/C."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_ln = RMSNorm(self.dt_rank)
        self.b_ln = RMSNorm(d_state)
        self.c_ln = RMSNorm(d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)
        xs = self.conv1d(xs.transpose(1, 2))[..., :T].transpose(1, 2)
        xs = F.silu(xs)
        dbc = self.x_proj(xs)
        dt, Bp, Cp = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(self.dt_ln(dt)))  # (B,T,d_inner)
        Bp = self.b_ln(Bp)
        Cp = self.c_ln(Cp)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        h = x.new_zeros(B, self.d_inner, self.d_state)
        outs = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))  # (B,d_inner,d_state)
            dBx = dt[:, t].unsqueeze(-1) * Bp[:, t].unsqueeze(1) * xs[:, t].unsqueeze(-1)
            h = dA * h + dBx
            y = torch.einsum("bds,bs->bd", h, Cp[:, t]) + self.D * xs[:, t]
            outs.append(y)
        y = torch.stack(outs, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class JambaAttention(nn.Module):
    """GQA self-attention with NO positional embedding (Jamba relies on Mamba)."""

    def __init__(self, d_model: int, n_heads: int = 8, n_kv_heads: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.hd = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(n_heads * self.hd, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5)
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        scores = scores + mask
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class JambaMoE(nn.Module):
    """Top-2 sparse MoE over ``num_experts`` SwiGLU experts.

    The full model uses 16 experts; we keep the same top-2 router structure with
    a reduced expert count so the unrolled atlas graph stays renderable.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense-compute then top-2 gate (graph-compact form of the sparse MoE):
        # the router still selects top-2 experts; non-selected weights are zeroed.
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        gate = torch.zeros_like(probs).scatter(-1, topi, topw)
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            out = out + gate[..., e : e + 1] * expert(x)
        return out


class JambaBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, idx: int) -> None:
        super().__init__()
        self.in_ln = RMSNorm(d_model)
        self.is_attn = (idx % 8) == 4
        self.mixer = JambaAttention(d_model) if self.is_attn else JambaMambaMixer(d_model)
        self.ff_ln = RMSNorm(d_model)
        self.is_moe = (idx % 2) == 1
        self.ff = JambaMoE(d_model, d_ff) if self.is_moe else SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.in_ln(x))
        x = x + self.ff(self.ff_ln(x))
        return x


class JambaLM(nn.Module):
    def __init__(
        self, vocab: int = 256, d_model: int = 128, d_ff: int = 256, n_layers: int = 8
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([JambaBlock(d_model, d_ff, i) for i in range(n_layers)])
        self.final_ln = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.final_ln(x))


def build_jamba() -> nn.Module:
    return JambaLM()


def example_input() -> torch.Tensor:
    """Token-id sequence ``(1, 12)`` (short, to keep the unrolled SSM scan compact)."""
    return torch.randint(0, 256, (1, 12))


MENAGERIE_ENTRIES = [
    (
        "Jamba (hybrid Transformer-Mamba-MoE language model)",
        "build_jamba",
        "example_input",
        "2024",
        "DC",
    ),
]
