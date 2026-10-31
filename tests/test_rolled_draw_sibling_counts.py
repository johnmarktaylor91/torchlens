"""Regression tests for per-draw atomic-module sibling-count reuse."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization import _render_nodes


class _SplitReuse(nn.Module):
    """Model whose atomic modules are reused at split rolled call sites."""

    def __init__(self) -> None:
        """Initialize the test model."""

        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.act = nn.Tanh()
        self.head = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the projection twice in split loops around a breaker op."""

        hidden = inputs
        for _ in range(2):
            hidden = self.act(self.proj(hidden))
        hidden = hidden + 1.0
        for _ in range(2):
            hidden = self.act(self.proj(hidden))
        return self.head(hidden)


def test_rolled_draw_reuses_atomic_module_sibling_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One sibling-count map should be built per draw and shared across nodes."""

    trace = tl.trace(_SplitReuse(), torch.randn(1, 4))

    built_counts: list[Mapping[str, int]] = []
    received_counts: list[Mapping[str, int] | None] = []
    original_build = _render_nodes._atomic_module_sibling_counts
    original_split = _render_nodes._atomic_module_split_range

    def spy_build(owning_trace: Any) -> dict[str, int]:
        """Record every sibling-count map construction."""

        counts = original_build(owning_trace)
        built_counts.append(counts)
        return counts

    def spy_split(
        owning_trace: Any,
        layer_log: Any,
        address: str,
        sibling_counts: Mapping[str, int] | None = None,
    ) -> str:
        """Record the sibling counts threaded into each split-range lookup."""

        received_counts.append(sibling_counts)
        return original_split(owning_trace, layer_log, address, sibling_counts)

    monkeypatch.setattr(_render_nodes, "_atomic_module_sibling_counts", spy_build)
    monkeypatch.setattr(_render_nodes, "_atomic_module_split_range", spy_split)

    trace.draw(vis_mode="rolled", return_graph=True, vis_save_only=True)

    assert len(built_counts) == 1
    assert len(received_counts) > 1
    assert all(counts is built_counts[0] for counts in received_counts)
    assert built_counts[0] == _render_nodes._atomic_module_sibling_counts(trace)


def test_sibling_counts_match_per_layer_scan() -> None:
    """The one-pass counter must reproduce the historical per-node scan."""

    trace = tl.trace(_SplitReuse(), torch.randn(1, 4))
    counts = _render_nodes._atomic_module_sibling_counts(trace)

    addresses = {
        layer.modules[-1].rsplit(":", 1)[0]
        for layer in trace.layer_logs.values()
        if getattr(layer, "modules", None)
    }
    assert addresses
    for address in addresses | {"not_a_module"}:
        expected = sum(
            1
            for other in trace.layer_logs.values()
            if isinstance(other, _render_nodes.Layer)
            and getattr(other, "is_atomic_module", False)
            and other.modules
            and other.modules[-1].rsplit(":", 1)[0] == address
        )
        assert counts.get(address, 0) == expected


def test_split_range_fallback_matches_threaded_counts() -> None:
    """Calling the split-range helper without counts must not change output."""

    trace = tl.trace(_SplitReuse(), torch.randn(1, 4))
    counts = _render_nodes._atomic_module_sibling_counts(trace)
    for layer in trace.layer_logs.values():
        if not getattr(layer, "modules", None):
            continue
        address = layer.modules[-1].rsplit(":", 1)[0]
        assert _render_nodes._atomic_module_split_range(
            trace, layer, address
        ) == _render_nodes._atomic_module_split_range(trace, layer, address, counts)
