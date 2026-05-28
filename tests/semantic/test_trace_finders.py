"""Tests for Trace-level semantic facet finders."""

from __future__ import annotations

import torch
from torch import nn

from torchlens import trace as trace_fn


class DistilBertSdpaAttention(nn.Module):
    """Tiny DistilBERT-like attention block for recipe tests."""

    def __init__(self) -> None:
        """Initialize the tiny attention block."""

        super().__init__()
        self.n_heads = 2
        self.dim = 8
        self.q_lin = nn.Linear(8, 8)
        self.k_lin = nn.Linear(8, 8)
        self.v_lin = nn.Linear(8, 8)
        self.out_lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention-shaped projections."""

        return self.out_lin(self.q_lin(x) + self.k_lin(x) + self.v_lin(x))


class _FinderModel(nn.Module):
    """Tiny model with attention and normalization modules."""

    def __init__(self) -> None:
        """Initialize finder-test modules."""

        super().__init__()
        self.attn = DistilBertSdpaAttention()
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention followed by normalization."""

        return self.norm(self.attn(x))


def test_trace_finders_return_matching_modules() -> None:
    """Trace finders locate attention and normalization facet providers."""

    log = trace_fn(_FinderModel(), torch.randn(2, 3, 8), layers_to_save="all")
    attention_blocks = list(log.attention_blocks())
    q_modules = list(log.modules_with_facet("q"))
    norm_modules = list(log.modules_with_facet("normalized"))

    assert [module.address for module in attention_blocks] == ["attn"]
    assert q_modules == attention_blocks
    assert [module.address for module in norm_modules] == ["norm"]
