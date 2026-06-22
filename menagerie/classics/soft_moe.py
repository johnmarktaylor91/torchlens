"""SoftMoE compact faithful reconstruction.

Puigcerver et al. 2024, "From Sparse to Soft Mixtures of Experts".

Soft MoE uses learned dispatch weights to send soft mixtures of all input tokens
to expert slots, applies experts to those slots, then combines expert outputs
back to tokens with learned combine weights. This keeps the fully
differentiable dispatch/combine mechanism in a small transformer block.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SoftMoELayer(nn.Module):
    """Soft mixture-of-experts layer with slots."""

    def __init__(self, dim: int = 48, experts: int = 4, slots: int = 2) -> None:
        """Initialize dispatch/combine parameters and experts.

        Parameters
        ----------
        dim:
            Token dimension.
        experts:
            Number of experts.
        slots:
            Slots per expert.
        """
        super().__init__()
        self.experts = experts
        self.slots = slots
        self.dispatch = nn.Parameter(torch.randn(dim, experts, slots) * 0.02)
        self.combine = nn.Parameter(torch.randn(dim, experts, slots) * 0.02)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
                )
                for _ in range(experts)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply soft expert dispatch and combine.

        Parameters
        ----------
        x:
            Token tensor with shape ``(batch, time, dim)``.

        Returns
        -------
        Tensor
            Expert-mixed token tensor.
        """
        dispatch_logits = torch.einsum("btd,des->btes", x, self.dispatch)
        combine_logits = torch.einsum("btd,des->btes", x, self.combine)
        dispatch = torch.softmax(dispatch_logits, dim=1)
        combine = torch.softmax(combine_logits.flatten(2), dim=-1).reshape_as(combine_logits)
        slots = torch.einsum("btd,btes->besd", x, dispatch)
        expert_outs = []
        for idx, expert in enumerate(self.mlps):
            expert_outs.append(expert(slots[:, idx]))
        expert_tokens = torch.stack(expert_outs, dim=1)
        return torch.einsum("besd,btes->btd", expert_tokens, combine)


class SoftMoETransformer(nn.Module):
    """Compact transformer block using SoftMoE in place of a dense MLP."""

    def __init__(self, vocab: int = 64, dim: int = 48, heads: int = 4) -> None:
        """Initialize embeddings, attention, SoftMoE, and head.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token dimension.
        heads:
            Attention head count.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, 12, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.moe = SoftMoELayer(dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run the compact SoftMoE transformer.

        Parameters
        ----------
        tokens:
            Token ids.

        Returns
        -------
        Tensor
            Vocabulary logits.
        """
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        y, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)
        x = x + y
        x = x + self.moe(x)
        return self.out(x)


def build() -> nn.Module:
    """Build compact random-init SoftMoE.

    Returns
    -------
    nn.Module
        Compact SoftMoE transformer.
    """
    return SoftMoETransformer()


def example_input() -> Tensor:
    """Return token ids.

    Returns
    -------
    Tensor
        Token tensor.
    """
    return torch.randint(0, 64, (1, 12))


MENAGERIE_ENTRIES = [("SoftMoE", "build", "example_input", "2024", "E7")]
