"""Tests for Trace-level semantic facet finders."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
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


def test_trace_finders_use_available_now_semantics() -> None:
    """Trace finders skip declared facets whose payloads were not captured.

    The model has no relu, which is the point: the capture deliberately saves
    nothing so every declared facet is unavailable. That also trips the
    zero-match save-selector warning, which is asserted here rather than
    silenced.
    """

    with pytest.warns(UserWarning, match="matched zero sites"):
        log = trace_fn(_FinderModel(), torch.randn(2, 3, 8), save=tl.func("relu"))

    assert list(log.attention_blocks()) == []
    assert list(log.modules_with_facet("normalized")) == []
    assert log.modules["attn"].facets.declared()["q"].status == "needs_capture"
    assert log.modules["norm"].facets.declared()["normalized"].status == "needs_capture"


class _VitStyleModel(nn.Module):
    """One attention block called once and one dropout reused around it (ViT layout)."""

    def __init__(self) -> None:
        """Initialize the single-call attention and the shared dropout."""

        super().__init__()
        self.attn = DistilBertSdpaAttention()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dropout, attention, dropout."""

        return self.dropout(self.attn(self.dropout(x)))


class _AlbertStyleModel(nn.Module):
    """One attention block whose weights are shared across two layer calls (ALBERT layout)."""

    def __init__(self) -> None:
        """Initialize the shared attention block."""

        super().__init__()
        self.shared = DistilBertSdpaAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared block twice."""

        return self.shared(self.shared(x))


def test_modules_with_facet_skips_reused_modules_without_the_facet() -> None:
    """A reused dropout no longer aborts the sweep; the single-call attention block is found."""

    log = trace_fn(_VitStyleModel(), torch.randn(2, 3, 8))

    assert log.modules["dropout"].num_calls == 2
    assert [module.address for module in log.modules_with_facet("q")] == ["attn"]
    assert [module.address for module in log.attention_blocks()] == ["attn"]


def test_modules_with_facet_refuses_a_shared_layer_that_declares_the_facet() -> None:
    """A shared attention layer is refused by name, never silently dropped from the sweep."""

    log = trace_fn(_AlbertStyleModel(), torch.randn(2, 3, 8))

    with pytest.raises(tl.errors.InvalidArgumentError, match="'shared' has 2 calls") as info:
        list(log.modules_with_facet("q"))
    assert info.value.fields["code"] == "module_call_ambiguous"
    assert "trace.modules['shared'].calls[N].facets.q" in str(info.value)
    assert list(log.modules_with_facet("normalized")) == []


def test_module_call_facets_read_each_pass_of_a_shared_layer() -> None:
    """Each call of a shared attention layer has its own q facet, read from that pass."""

    log = trace_fn(_AlbertStyleModel(), torch.randn(2, 3, 8))
    shared, q_lin = log.modules["shared"], log.modules["shared.q_lin"]

    assert shared.num_calls == 2
    q_values = []
    for index in range(shared.num_calls):
        call = shared.calls[index]
        assert call.facets.has("q")
        assert call.facets.declared()["q"].status == "available_now"
        q_value = call.facets.q.value
        expected = q_lin.calls[index].out
        assert q_value.numel() == expected.numel()
        assert torch.allclose(q_value.sum(), expected.sum())
        q_values.append(q_value)
    assert not torch.equal(q_values[0], q_values[1])
    by_label = log.module_calls["shared:2"].facets.q.value
    assert torch.equal(by_label, q_values[1])


def test_module_call_facets_match_the_module_view_for_a_single_call() -> None:
    """A module called once reads the same facet through its Module and its one call."""

    log = trace_fn(_VitStyleModel(), torch.randn(2, 3, 8))
    attn = log.modules["attn"]

    assert torch.equal(attn.calls[0].facets.q.value, attn.facets.q.value)
    assert torch.equal(attn.calls[0].facets.v.value, attn.facets.v.value)
    for call in log.modules["dropout"].calls.values():
        assert not call.facets.has("q")
