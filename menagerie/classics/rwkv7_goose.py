"""RWKV-7 "Goose": generalized delta-rule linear-attention RNN.

Peng et al. (2025), "RWKV-7 Goose with Expressive Dynamic State Evolution".
arXiv:2503.14456.  Source: https://github.com/BlinkDL/RWKV-LM (v7 branch)

Distinctive primitive: WKV7 token-mixing recurrence.
RWKV-7 introduces a DATA-DEPENDENT STATE MATRIX update (generalised delta rule):

  For each token t, per-head:
    v = r * state          -- "read" from current state (matrix * vector)
    q = w * (a * v + b)    -- decay + in-context learning rate gate
    state_new = state * diag(decay_vec) + outer(a * (w + v), k)
                           -- data-dependent decay + rank-1 update
    output = sigmoid(g) * (v + q)   -- sigmoid-gated mix of old/new read

  where r, w, k, v_emb, a, b, g are all data-dependent (linear projections of x).
  The "a" vector acts as an in-context learning rate (ICRL), making the state
  evolution fully data-dependent -- the key advance over RWKV-4/5/6.

Channel-mix (FFN equivalent): x -> squared-ReLU(x @ W1) @ W2 * sigmoid(gate(x)).

Compact config: d_model=64, n_heads=4, head_size=16, seq_len=8, n_layers=2.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, [x.shape[-1]], self.g, self.b, self.eps)


class WKV7Mixing(nn.Module):
    """RWKV-7 WKV7 token-mixing: generalised delta-rule recurrent state update.

    Processes a (B, T, d_model) sequence with a recurrent state per head.
    Implemented as a sequential Python loop for transparency (the parallel
    chunk kernel in the actual code is a speedup of the same recurrence).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads

        # Data-dependent projections (7 vectors per token)
        self.proj_r = nn.Linear(d_model, d_model, bias=False)  # receptance
        self.proj_w = nn.Linear(d_model, d_model, bias=False)  # decay
        self.proj_k = nn.Linear(d_model, d_model, bias=False)  # key
        self.proj_v = nn.Linear(d_model, d_model, bias=False)  # value
        self.proj_a = nn.Linear(d_model, d_model, bias=False)  # in-context lr
        self.proj_b = nn.Linear(d_model, d_model, bias=False)  # bonus
        self.proj_g = nn.Linear(d_model, d_model, bias=False)  # gate
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ln_x = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, d = x.shape
        H = self.n_heads
        S = self.head_size  # head_size

        r = torch.sigmoid(self.proj_r(x))  # (B, T, d)
        w = torch.sigmoid(self.proj_w(x))  # decay in (0,1)
        k = self.proj_k(x)  # (B, T, d)
        v_emb = self.proj_v(x)  # (B, T, d)
        a = torch.sigmoid(self.proj_a(x))  # in-context lr
        b = self.proj_b(x)  # bonus
        g = torch.sigmoid(self.proj_g(x))  # output gate

        # Normalise k for numerics
        k = F.normalize(k, dim=-1)

        # Reshape for heads: (B, T, H, S)
        r = r.view(B, T, H, S)
        w = w.view(B, T, H, S)
        k = k.view(B, T, H, S)
        v_emb = v_emb.view(B, T, H, S)
        a = a.view(B, T, H, S)
        b = b.view(B, T, H, S)

        # Initial state: (B, H, S, S) zero matrix per head
        state = torch.zeros(B, H, S, S, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            r_t = r[:, t]  # (B, H, S)
            w_t = w[:, t]  # (B, H, S)  decay
            k_t = k[:, t]  # (B, H, S)
            v_t = v_emb[:, t]  # (B, H, S)
            a_t = a[:, t]  # (B, H, S)  in-context lr
            b_t = b[:, t]  # (B, H, S)  bonus

            # Read from state: v = r @ state  -> (B, H, S)
            v_read = (r_t.unsqueeze(-2) @ state).squeeze(-2)  # (B, H, S)

            # State update: state = state * diag(w) + outer(a*(w+v_read), k)
            # diag(w): scale each column of state
            state = state * w_t.unsqueeze(-1)  # broadcast (B,H,S,S)
            # Rank-1 update: outer product of (a*(v_t + v_read)) and k
            update_vec = a_t * (v_t + v_read)  # (B, H, S)
            # outer: (B,H,S,1) x (B,H,1,S) -> (B,H,S,S)
            state = state + update_vec.unsqueeze(-1) * k_t.unsqueeze(-2)

            # Output at t: re-read with w
            q_t = w_t * (a_t * v_read + b_t)  # (B, H, S)
            out_t = v_read + q_t  # (B, H, S)
            outputs.append(out_t)

        # Stack: (B, T, H, S) -> (B, T, d)
        y = torch.stack(outputs, dim=1).view(B, T, d)
        y = self.ln_x(y)
        y = y * g
        return self.out(y)


class RWKV7ChannelMix(nn.Module):
    """RWKV-7 channel-mix (FFN): squared-ReLU + sigmoid gate."""

    def __init__(self, d_model: int = 64, d_ff: int = 128) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, d_ff, bias=False)
        self.val = nn.Linear(d_ff, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        k = F.relu(self.key(x)) ** 2  # squared ReLU
        v = self.val(k)
        g = torch.sigmoid(self.gate(x))
        return v * g


class RWKV7Block(nn.Module):
    """One RWKV-7 block: LN + token-mix + LN + channel-mix, both residual."""

    def __init__(self, d_model: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.wkv = WKV7Mixing(d_model, n_heads)
        self.cmix = RWKV7ChannelMix(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.wkv(self.ln1(x))
        x = x + self.cmix(self.ln2(x))
        return x


class RWKV7Goose(nn.Module):
    """Compact RWKV-7 Goose LM block stack."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([RWKV7Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_out = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        for blk in self.blocks:
            x = blk(x)
        return self.ln_out(x)


def build_rwkv_v7_goose() -> nn.Module:
    # Keep seq_len tiny (T=6) to avoid O(T*H*S^2) cost blowing up the graph
    return RWKV7Goose(d_model=64, n_heads=4, n_layers=2).eval()


def example_input() -> torch.Tensor:
    """(1, 6, 64) -- batch=1, T=6 tokens, d_model=64."""
    return torch.randn(1, 6, 64)


MENAGERIE_ENTRIES = [
    (
        "RWKV-7 Goose (generalized delta-rule WKV7 recurrence with data-dependent state decay + ICRL)",
        "build_rwkv_v7_goose",
        "example_input",
        "2025",
        "DC",
    ),
]
