"""Tests for built-in attention recipe behavior."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens import trace as trace_fn
from torchlens.semantic import MissingFacetError


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


class MultiHeadSelfAttention(nn.Module):
    """Tiny DistilBERT eager attention block (q_lin/k_lin/v_lin children)."""

    def __init__(self) -> None:
        """Initialize the tiny eager attention block."""

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


class DistilBertSelfAttention(nn.Module):
    """Tiny transformers-5.x DistilBERT unified attention block."""

    def __init__(self) -> None:
        """Initialize the tiny unified attention block."""

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


class LlamaAttention(nn.Module):
    """Tiny GQA-style attention block."""

    def __init__(self) -> None:
        """Initialize projections with fewer KV heads than Q heads."""

        super().__init__()
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 2
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 4)
        self.v_proj = nn.Linear(8, 4)
        self.o_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run enough projections for recipe extraction."""

        q = self.q_proj(x)
        k = self.k_proj(x).repeat_interleave(2, dim=-1)
        v = self.v_proj(x).repeat_interleave(2, dim=-1)
        return self.o_proj(q + k + v)


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
def test_distilbert_eager_attention_q_shape_and_head_view() -> None:
    """DistilBERT eager class (MultiHeadSelfAttention) exposes q/k/v facets.

    Regression: only ``DistilBertSdpaAttention`` was registered, so switching to
    ``_attn_implementation='eager'`` (the very fix the ``pattern`` missing-capture
    recommends) landed on the unregistered eager class and raised ``KeyError('q')``.
    """

    log = trace_fn(
        _AttentionModel(MultiHeadSelfAttention()), torch.randn(2, 3, 8), layers_to_save="all"
    )
    view = log.modules["attn"].facets
    assert view.recipe_source == "distilbert_attention"
    assert view.q.shape == (2, 3, 2, 4)
    assert view.head(1).q.shape == (2, 3, 4)
    assert torch.equal(view.head(1).q, view.q[:, :, 1, :])
    # Eager omits ``pattern`` (consistent with other eager recipes).
    assert "pattern" not in view.keys()


@pytest.mark.slow
def test_distilbert_unified_attention_q_shape_and_pattern_missing() -> None:
    """transformers-5.x unified class (DistilBertSelfAttention) exposes q/k/v.

    Regression: transformers 5.x collapsed the per-backend subclasses into one
    ``DistilBertSelfAttention`` class; it was unregistered, so facets.q raised
    ``KeyError('q')``. Its default backend is fused SDPA, so ``pattern`` is the
    informative MissingFacetError on access, not a silent AttributeError.
    """

    log = trace_fn(
        _AttentionModel(DistilBertSelfAttention()), torch.randn(2, 3, 8), layers_to_save="all"
    )
    view = log.modules["attn"].facets
    assert view.recipe_source == "distilbert_attention"
    assert view.q.shape == (2, 3, 2, 4)
    assert view.head(1).q.shape == (2, 3, 4)
    # Fused-by-default: pattern is declared in the menu but unavailable until recapture.
    assert "pattern" not in view.keys()
    assert view.menu()["pattern"].status == "needs_capture"
    with pytest.raises(MissingFacetError, match="attention pattern not captured"):
        view.pattern


@pytest.mark.slow
def test_gpt2_fused_qkv_split_matches_manual_reference() -> None:
    """GPT-2 recipe splits fused QKV output into equal thirds."""

    log = trace_fn(_AttentionModel(GPT2Attention()), torch.randn(2, 3, 8), layers_to_save="all")
    view = log.modules["attn"].facets
    c_attn = log.modules["attn.c_attn"].out
    q_ref, _k_ref, _v_ref = c_attn.split(c_attn.shape[-1] // 3, dim=-1)
    assert torch.equal(view.q, q_ref.view(2, 3, 2, 4))
    assert view.q.spec.home_label == view.k.spec.home_label == view.v.spec.home_label


@pytest.mark.slow
def test_gqa_kv_head_selection_is_aliasing_read_grad_only() -> None:
    """GQA K/V query-head selection records aliasing capability."""

    log = trace_fn(_AttentionModel(LlamaAttention()), torch.randn(2, 3, 8), layers_to_save="all")
    view = log.modules["attn"].facets
    k_head = view.head(3).k

    assert torch.equal(k_head, view.k[:, :, 1, :])
    assert k_head.spec.capability_class == "aliasing_selection"
    assert k_head.spec.capability_flags.read is True
    assert k_head.spec.capability_flags.grad is True
    assert k_head.spec.capability_flags.write is False
