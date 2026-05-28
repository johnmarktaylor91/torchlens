"""Tests for built-in attention recipe behavior."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens import trace as trace_fn


pytest.importorskip("transformers")


class DistilBertSdpaAttention(nn.Module):
    """Tiny DistilBERT-like attention block for shape tests."""

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


class GPT2Attention(nn.Module):
    """Tiny GPT-2-like fused-QKV attention block."""

    def __init__(self) -> None:
        """Initialize the tiny GPT-2-like block."""

        super().__init__()
        self.num_heads = 2
        self.embed_dim = 8
        self.c_attn = nn.Linear(8, 24)
        self.c_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fused projection and output projection."""

        q, _k, _v = self.c_attn(x).split(8, dim=-1)
        return self.c_proj(q)


class _AttentionModel(nn.Module):
    """Wrapper exposing a named attention module."""

    def __init__(self, attention: nn.Module) -> None:
        """Initialize with an attention-like child."""

        super().__init__()
        self.attn = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention child."""

        return self.attn(x)


@pytest.mark.slow
def test_distilbert_attention_q_shape_and_head_view() -> None:
    """DistilBERT recipe reshapes Q and slices heads correctly."""

    log = trace_fn(
        _AttentionModel(DistilBertSdpaAttention()), torch.randn(2, 3, 8), layers_to_save="all"
    )
    view = log.modules["attn"].facets
    assert view.q.shape == (2, 3, 2, 4)
    assert view.head(1).q.shape == (2, 3, 4)
    assert torch.equal(view.head(1).q, view.q[:, :, 1, :])


@pytest.mark.slow
def test_gpt2_fused_qkv_split_matches_manual_reference() -> None:
    """GPT-2 recipe splits fused QKV output into equal thirds."""

    log = trace_fn(_AttentionModel(GPT2Attention()), torch.randn(2, 3, 8), layers_to_save="all")
    view = log.modules["attn"].facets
    c_attn = log.modules["attn.c_attn"].out
    q_ref, _k_ref, _v_ref = c_attn.split(c_attn.shape[-1] // 3, dim=-1)
    assert torch.equal(view.q, q_ref.view(2, 3, 2, 4))
