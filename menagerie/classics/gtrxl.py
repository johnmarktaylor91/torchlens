"""GTrXL: gated Transformer-XL for reinforcement learning.

Paper: "Stabilizing Transformers for Reinforcement Learning", Parisotto et
al., ICML 2020.

This small model preserves the two architectural changes highlighted by GTrXL:
identity-map reordering through pre-layer normalization, and GRU-style gated
residual connections around the attention and feed-forward sublayers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUGate(nn.Module):
    """GRU-style residual gate used by GTrXL blocks."""

    def __init__(self, dim: int) -> None:
        """Initialize the residual gate.

        Parameters
        ----------
        dim:
            Hidden dimension of both residual streams.
        """

        super().__init__()
        self.update = nn.Linear(dim * 2, dim)
        self.reset = nn.Linear(dim * 2, dim)
        self.candidate = nn.Linear(dim * 2, dim)

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Blend a residual stream with a sublayer update.

        Parameters
        ----------
        residual:
            Identity stream entering the sublayer.
        update:
            Sublayer output to gate into the identity stream.

        Returns
        -------
        torch.Tensor
            Gated residual output.
        """

        pair = torch.cat([update, residual], dim=-1)
        z = torch.sigmoid(self.update(pair))
        r = torch.sigmoid(self.reset(pair))
        candidate = torch.tanh(self.candidate(torch.cat([update, r * residual], dim=-1)))
        return (1.0 - z) * residual + z * candidate


class GTrXLBlock(nn.Module):
    """One compact GTrXL recurrent-attention block."""

    def __init__(self, dim: int, num_heads: int = 2, memory_len: int = 3) -> None:
        """Initialize a GTrXL block.

        Parameters
        ----------
        dim:
            Hidden dimension.
        num_heads:
            Number of self-attention heads.
        memory_len:
            Number of previous-memory slots prepended to keys and values.
        """

        super().__init__()
        if dim % num_heads != 0:
            msg = "dim must be divisible by num_heads"
            raise ValueError(msg)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.memory_len = memory_len
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_ff = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim)
        self.attn_gate = GRUGate(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.ff_gate = GRUGate(dim)

    def _causal_memory_mask(self, steps: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask over memory plus current tokens.

        Parameters
        ----------
        steps:
            Number of current sequence steps.
        device:
            Device for the returned mask.

        Returns
        -------
        torch.Tensor
            Boolean mask with shape ``(steps, memory_len + steps)``.
        """

        current = torch.tril(torch.ones(steps, steps, dtype=torch.bool, device=device))
        memory = torch.ones(steps, self.memory_len, dtype=torch.bool, device=device)
        return torch.cat([memory, current], dim=-1)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Run identity-reordered memory attention followed by gated MLP.

        Parameters
        ----------
        x:
            Current hidden sequence, shape ``(batch, steps, dim)``.
        memory:
            Previous hidden memory, shape ``(batch, memory_len, dim)``.

        Returns
        -------
        torch.Tensor
            Updated hidden sequence.
        """

        batch, steps, dim = x.shape
        attn_in = self.ln_attn(x)
        kv_in = torch.cat([memory, attn_in], dim=1)
        q = self.q_proj(attn_in).view(batch, steps, self.num_heads, self.head_dim)
        k = self.k_proj(kv_in).view(batch, self.memory_len + steps, self.num_heads, self.head_dim)
        v = self.v_proj(kv_in).view(batch, self.memory_len + steps, self.num_heads, self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)
        mask = self._causal_memory_mask(steps, x.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, -1.0e4)
        attended = torch.einsum("bhts,bshd->bthd", torch.softmax(scores, dim=-1), v)
        attended = self.o_proj(attended.reshape(batch, steps, dim))
        x = self.attn_gate(x, attended)
        feedforward = self.ff(self.ln_ff(x))
        return self.ff_gate(x, feedforward)


class GTrXLActorCritic(nn.Module):
    """Small actor-critic network with GTrXL memory blocks."""

    def __init__(self, obs_dim: int = 7, dim: int = 16, actions: int = 5) -> None:
        """Initialize the actor-critic model.

        Parameters
        ----------
        obs_dim:
            Observation feature dimension.
        dim:
            Hidden model dimension.
        actions:
            Number of discrete actions.
        """

        super().__init__()
        self.embed = nn.Linear(obs_dim, dim)
        self.memory = nn.Parameter(torch.zeros(1, 3, dim))
        self.block1 = GTrXLBlock(dim)
        self.block2 = GTrXLBlock(dim)
        self.policy = nn.Linear(dim, actions)
        self.value = nn.Linear(dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy logits and values for an observation sequence.

        Parameters
        ----------
        observations:
            Observation sequence with shape ``(batch, steps, obs_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Policy logits and scalar values for each time step.
        """

        batch = observations.shape[0]
        hidden = F.relu(self.embed(observations))
        memory = self.memory.expand(batch, -1, -1)
        hidden = self.block1(hidden, memory)
        hidden = self.block2(hidden, memory)
        return self.policy(hidden), self.value(hidden)


def build() -> nn.Module:
    """Build a compact GTrXL actor-critic.

    Returns
    -------
    nn.Module
        Randomly initialized GTrXL actor-critic.
    """

    return GTrXLActorCritic()


def example_input() -> torch.Tensor:
    """Create a short RL observation sequence.

    Returns
    -------
    torch.Tensor
        Random input with shape ``(1, 5, 7)``.
    """

    return torch.randn(1, 5, 7)


MENAGERIE_ENTRIES = [
    (
        "GTrXL (Gated Transformer-XL actor-critic)",
        "build",
        "example_input",
        "2020",
        "rl/control",
    ),
]
