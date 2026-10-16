"""Attribution-graph assembly runs chunk by chunk of targets.

The builder used to hold every target's gradient at every source at once,
``(targets, sources, positions, width)``; at Gemma-2 2B scale that is about
20 GB in float32. It now reads and contracts one chunk of targets at a time.
These tests pin two things on the AG-1 fixtures (``tests/_ag1_fixtures.py``):
the graph does not depend on the chunk size (bit for bit), and the largest
gradient buffer the builder holds is one chunk's, counted exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

import torchlens as tl
from tests import _ag1_fixtures as fx
from torchlens.attribution.graph import _build as build_module

#: F1 and F3 shapes: (targets, sources, positions, width, element bytes).
F1_TARGETS = 12 + 2
F1_SOURCES = 2 + 1
F3_TARGETS = 18 + 2
F3_SOURCES = 3 + 1
FLOAT64_BYTES = 8


def _trace(model: torch.nn.Module) -> tl.Trace:
    return tl.trace(model, torch.tensor([fx.TOKENS]))


def _graph_and_counters(model, transcoders, n_layers, chunk, monkeypatch):
    monkeypatch.setattr(build_module, "_TARGETS_PER_READ", chunk)
    counters = build_module._BuildCounters()
    graph = build_module._build(
        _trace(model),
        transcoders,
        embedding="embed",
        target=None,
        max_n_logits=2,
        desired_logit_prob=0.99,
        demeaned_logits=True,
        logits_at=None,
        frozen=fx.frozen_names(n_layers),
        counters=counters,
    )
    return graph, counters


def _assert_same_graph(left, right) -> None:
    assert left.nodes == right.nodes
    assert torch.equal(left.adjacency_matrix, right.adjacency_matrix)
    pairs = [(left.node_values, right.node_values)]
    pairs += [(left.constants[name], right.constants[name]) for name in left.constants]
    for one, other in pairs:
        assert torch.allclose(one, other, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(left.error_vectors, right.error_vectors)
    assert torch.equal(left.activation_values, right.activation_values)


class TestChunkedAssembly:
    """The graph is independent of the chunk size; the buffer is one chunk's."""

    @pytest.mark.parametrize("chunk", [1, 3, 5])
    def test_per_layer_graph_does_not_depend_on_the_chunk_size(self, chunk, monkeypatch) -> None:
        model = fx.build_model(2)
        transcoders = fx.lossy_transcoders(model)
        whole, _ = _graph_and_counters(model, transcoders, 2, 10_000, monkeypatch)
        chunked, _ = _graph_and_counters(model, transcoders, 2, chunk, monkeypatch)
        _assert_same_graph(whole, chunked)

    def test_cross_layer_graph_does_not_depend_on_the_chunk_size(self, monkeypatch) -> None:
        model = fx.build_model(3)
        transcoders = fx.cross_layer_transcoders(model)
        whole, _ = _graph_and_counters(model, transcoders, 3, 10_000, monkeypatch)
        chunked, _ = _graph_and_counters(model, transcoders, 3, 4, monkeypatch)
        _assert_same_graph(whole, chunked)

    def test_gradient_buffer_peak_is_one_chunk(self, monkeypatch) -> None:
        model = fx.build_model(2)
        transcoders = fx.exact_transcoders(model)
        per_target = F1_SOURCES * fx.N_POS * fx.D_MODEL * FLOAT64_BYTES
        _graph, counters = _graph_and_counters(model, transcoders, 2, 3, monkeypatch)
        assert counters.reads == math.ceil(F1_TARGETS / 3)
        assert counters.gradient_buffer_peak_bytes == 3 * per_target
        _graph, whole = _graph_and_counters(model, transcoders, 2, 10_000, monkeypatch)
        assert whole.reads == 1
        assert whole.gradient_buffer_peak_bytes == F1_TARGETS * per_target

    def test_cross_layer_buffer_counts_every_written_output(self, monkeypatch) -> None:
        model = fx.build_model(3)
        transcoders = fx.cross_layer_transcoders(model)
        per_target = F3_SOURCES * fx.N_POS * fx.D_MODEL * FLOAT64_BYTES
        _graph, counters = _graph_and_counters(model, transcoders, 3, 7, monkeypatch)
        assert counters.reads == math.ceil(F3_TARGETS / 7)
        assert counters.gradient_buffer_peak_bytes == 7 * per_target
