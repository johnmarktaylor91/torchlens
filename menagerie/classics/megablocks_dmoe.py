"""MegaBlocks dropless-MoE (dMoE): token-routed sparse mixture-of-experts block.

Gale, Narayanan, Young & Zaharia (Stanford / Microsoft), MLSys 2023,
arXiv:2211.15841.  Source: https://github.com/databricks/megablocks
(``megablocks/layers/dmoe.py``, ``megablocks/layers/moe.py``).

MegaBlocks' contribution is *dropless* MoE: unlike token-dropping MoE (which caps
each expert at a fixed capacity and discards overflow tokens), dMoE routes EVERY
token to its top-k experts with NO capacity limit and NO padding.  To do this
efficiently on GPU it reformulates the expert feed-forward as a single
**block-sparse grouped GEMM**: tokens are sorted by their assigned expert, the
per-expert MLPs are applied as one block-diagonal (blocked-CSR) matmul, then the
outputs are scattered back to the original token order and scaled by the router
gate weights.

The CEILING in the menagerie is the custom block-sparse / grouped-GEMM CUDA kernels
(the ``stk`` / grouped_gemm package, compiled with nvcc).  Those kernels are purely
an OPTIMIZATION of an operation expressible in plain torch: routing is a softmax +
top-k over a linear router; the grouped GEMM is just "apply expert e's MLP to the
tokens routed to e"; the sort/scatter is an index permutation.  This module
reimplements the FULL dropless-MoE architecture -- router, top-k gating,
per-expert token grouping via a sort-by-expert permutation, per-expert SwiGLU MLPs,
ungroup/scatter, and gate-weighted combine -- in pure torch.  The block-sparse
grouped matmul is realised as a loop over experts applied to that expert's grouped
token slice (the mathematically identical dense-equivalent of the kernel), so the
graph traces and renders.

The block here is a standard Transformer MoE FFN block (RMSNorm -> dMoE -> residual).
Small config: d_model=64, 8 experts, top-2, so the unrolled per-expert graph stays
renderable.  Single dropless top-k MoE block.
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


class SwiGLUExpert(nn.Module):
    """One expert: a SwiGLU (GLU) feed-forward, matching MegaBlocks' GLU variant."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate proj
        self.v1 = nn.Linear(d_model, d_ff, bias=False)  # up proj
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.v1(x))


class DroplessMoE(nn.Module):
    """Dropless top-k MoE: route every token, grouped per-expert GEMM, scatter back.

    Faithful to MegaBlocks dMoE except the block-sparse CUDA grouped GEMM is replaced
    by the mathematically identical per-expert apply on the sorted token grouping.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        tokens = x.reshape(B * T, D)  # flatten to (n_tokens, d_model)

        # --- Router: softmax over experts, then top-k gate (no capacity / dropless) ---
        logits = self.router(tokens)
        probs = torch.softmax(logits, dim=-1)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)  # (n_tokens, top_k)
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # --- Expand each token into its top_k (token, expert) assignments ---
        n_tok = tokens.shape[0]
        flat_expert = topi.reshape(-1)  # (n_tok * top_k,)
        flat_weight = topw.reshape(-1, 1)  # (n_tok * top_k, 1)
        repeat_tokens = tokens.repeat_interleave(self.top_k, dim=0)  # gather inputs

        # --- Sort assignments by expert id: this is the "group tokens by expert" step
        #     that the block-sparse grouped GEMM consumes (here an index permutation). ---
        order = torch.argsort(flat_expert)
        sorted_expert = flat_expert[order]
        sorted_tokens = repeat_tokens[order]
        sorted_weight = flat_weight[order]

        # --- Grouped per-expert GEMM: apply expert e to its contiguous token block.
        #     (The CUDA kernel fuses these into one block-diagonal matmul.) ---
        out_sorted = torch.zeros_like(sorted_tokens)
        for e, expert in enumerate(self.experts):
            mask = sorted_expert == e
            sel = sorted_tokens[mask]
            out_sorted = out_sorted.masked_scatter(mask.unsqueeze(-1), expert(sel))

        out_sorted = out_sorted * sorted_weight  # gate-weight the expert outputs

        # --- Ungroup / scatter back to original (token, k) order, then combine k. ---
        unsort = torch.empty_like(order)
        unsort[order] = torch.arange(order.shape[0], device=order.device)
        out_flat = out_sorted[unsort].reshape(n_tok, self.top_k, D).sum(dim=1)
        return out_flat.reshape(B, T, D)


class DMoEBlock(nn.Module):
    """Transformer MoE FFN block: RMSNorm -> dropless-MoE -> residual."""

    def __init__(self, d_model: int = 64, d_ff: int = 128, num_experts: int = 8) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.moe = DroplessMoE(d_model, d_ff, num_experts=num_experts, top_k=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.moe(self.norm(x))


def build_megablocks_dmoe() -> nn.Module:
    """Build one MegaBlocks dropless top-2 MoE block (8 experts, d_model=64)."""
    return DMoEBlock(d_model=64, d_ff=128, num_experts=8)


def example_input() -> torch.Tensor:
    """Token feature sequence ``(1, 8, 64)`` (short, to keep the grouped GEMM compact)."""
    return torch.randn(1, 8, 64)


MENAGERIE_ENTRIES = [
    (
        "MegaBlocks dropless-MoE (dMoE block, grouped-GEMM token routing)",
        "build_megablocks_dmoe",
        "example_input",
        "2023",
        "DC",
    ),
]
