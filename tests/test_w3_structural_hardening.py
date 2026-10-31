"""Round-26 W3 structural-signature hardening regression tests.

HIGH-1: ``compute_graph_shape_hash`` silently dropped every parent edge whose
source is a multi-pass (recurrent) layer: the ordering map was keyed by the
non-pass-qualified ``layer_label`` while ``layer.parents`` references
multi-pass parents by their pass-qualified ``label`` (``linear_1_1:2``), so the
membership guard never matched. Structurally different recurrent graphs hashed
identically (false match), defeating the operand-order guarantee and the public
``tl.fingerprint.assert_unchanged`` structural pin.

MED-2: the RF staleness fingerprint ``_graph_revision`` omitted the geometry
arguments RF rules consume (kernel/stride/padding/dilation read via
``func_config`` with non-tensor-arg fallbacks), so a shape-preserving in-place
geometry change did not bump the revision: ``log.receptive_fields()`` kept
serving a stale frozen descriptor while ``op.receptive_field.at()`` recomputed
fresh from live values -- the two public RF paths disagreed.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.hashing import compute_graph_shape_hash
from torchlens.receptive_field import _engine


class CrossPassReaderA(nn.Module):
    """Recurrent cell applied three times; relu reads pass 2, sigmoid pass 3."""

    def __init__(self) -> None:
        """Initialize the shared recurrent cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``relu(cell**2 x) + sigmoid(cell**3 x)``."""

        h1 = self.cell(x)
        h2 = self.cell(h1)
        h3 = self.cell(h2)
        return torch.relu(h2) + torch.sigmoid(h3)


class CrossPassReaderB(nn.Module):
    """Recurrent cell applied three times; relu reads pass 3, sigmoid pass 2."""

    def __init__(self) -> None:
        """Initialize the shared recurrent cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``relu(cell**3 x) + sigmoid(cell**2 x)``."""

        h1 = self.cell(x)
        h2 = self.cell(h1)
        h3 = self.cell(h2)
        return torch.relu(h3) + torch.sigmoid(h2)


class SwappedOperandA(nn.Module):
    """Noncommutative op whose operands are two passes of one recurrent cell."""

    def __init__(self) -> None:
        """Initialize the shared recurrent cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``cell**2 x - cell**3 x``."""

        h1 = self.cell(x)
        h2 = self.cell(h1)
        h3 = self.cell(h2)
        return h2 - h3


class SwappedOperandB(nn.Module):
    """Operand-order swap of :class:`SwappedOperandA`."""

    def __init__(self) -> None:
        """Initialize the shared recurrent cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``cell**3 x - cell**2 x``."""

        h1 = self.cell(x)
        h2 = self.cell(h1)
        h3 = self.cell(h2)
        return h3 - h2


class DilatedConvModel(nn.Module):
    """Single-conv model with configurable dilation/padding (16x16-preserving)."""

    def __init__(self, dilation: int = 1, padding: int = 1) -> None:
        """Initialize a 3x3 conv whose output shape stays 16x16."""

        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=padding, dilation=dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution."""

        return self.conv(x)


def _parent_reference_order(trace: Any) -> dict[str, int]:
    """Rebuild the injective parent-reference ordering map the hash must use."""

    return {
        (layer.layer_label if layer.num_passes == 1 else layer.label): index
        for index, layer in enumerate(trace._layer_list)
    }


def _trace_pair(model_a: nn.Module, model_b: nn.Module, x: torch.Tensor) -> tuple[Any, Any]:
    """Capture both models on one input with tied weights."""

    model_b.load_state_dict(model_a.state_dict())
    return tl.trace(model_a, x), tl.trace(model_b, x)


def test_recurrent_cross_pass_reader_pair_hashes_distinct() -> None:
    """Two recurrent models differing only in which pass each consumer reads must not collide."""

    torch.manual_seed(0)
    x = torch.randn(1, 4)
    model_a, model_b = CrossPassReaderA(), CrossPassReaderB()
    trace_a, trace_b = _trace_pair(model_a, model_b, x)

    # The pair computes genuinely different functions.
    assert not torch.allclose(model_a(x), model_b(x))

    assert compute_graph_shape_hash(trace_a) != compute_graph_shape_hash(trace_b)
    # Address-free public spelling must distinguish them too.
    assert tl.fingerprint.trace(trace_a) != tl.fingerprint.trace(trace_b)


def test_recurrent_swapped_operand_pair_hashes_distinct() -> None:
    """Operand-order sensitivity must hold when both operands are multi-pass parents."""

    torch.manual_seed(0)
    x = torch.randn(1, 4)
    trace_a, trace_b = _trace_pair(SwappedOperandA(), SwappedOperandB(), x)

    assert compute_graph_shape_hash(trace_a) != compute_graph_shape_hash(trace_b)
    assert tl.fingerprint.trace(trace_a) != tl.fingerprint.trace(trace_b)


def test_every_parent_edge_contributes_to_hash_input() -> None:
    """No parent edge may be silently dropped: each edge must be hash-sensitive."""

    torch.manual_seed(0)
    x = torch.randn(1, 4)
    trace = tl.trace(CrossPassReaderA(), x)

    # 1) Resolution completeness: every recorded parent reference resolves in
    #    the injective reference-label space the hash keys by.
    order_by_reference = _parent_reference_order(trace)
    edges = [
        (layer, position, parent)
        for layer in trace._layer_list
        for position, parent in enumerate(layer.parents)
    ]
    assert edges, "expected a non-trivial recurrent graph"
    unresolved = [parent for _, _, parent in edges if parent not in order_by_reference]
    assert unresolved == []

    # 2) Edge sensitivity: rewiring ANY single parent edge changes the digest,
    #    proving the edge is present in the hash input (the pre-fix code
    #    silently dropped every multi-pass parent edge, making rewires of
    #    those edges invisible). The replacement stays within the parent's
    #    qualification class (pass-qualified -> another pass-qualified label,
    #    plain -> another plain label): a cross-class rewire would become
    #    visible to the broken hash merely by entering its keyed label space,
    #    masking the dropped-edge blindness this test exists to catch.
    baseline = compute_graph_shape_hash(trace)
    reference_labels = list(order_by_reference)
    for layer, position, parent in edges:
        assert isinstance(layer.parents, tuple)
        replacement = next(
            label
            for label in reference_labels
            if label != parent and (":" in label) == (":" in parent)
        )
        original = tuple(layer.parents)
        try:
            layer.parents = original[:position] + (replacement,) + original[position + 1 :]
            assert compute_graph_shape_hash(trace) != baseline, (
                f"rewiring parent edge {layer.label!r}[{position}] "
                f"({parent!r} -> {replacement!r}) did not change the graph-shape hash"
            )
        finally:
            layer.parents = original
    assert compute_graph_shape_hash(trace) == baseline


def test_unchanged_recurrent_model_still_hashes_same() -> None:
    """``assert_unchanged`` stays a true negative for genuinely unchanged models."""

    torch.manual_seed(0)
    x = torch.randn(1, 4)

    # Structural hash is weight-blind: two fresh instances of one architecture pin
    # identically, so an existing CI pin keeps passing after the fix lands.
    pinned = tl.fingerprint.model(CrossPassReaderA(), x)
    assert tl.fingerprint.assert_unchanged(CrossPassReaderA(), x, pinned) == pinned

    trace_1 = tl.trace(CrossPassReaderA(), x)
    trace_2 = tl.trace(CrossPassReaderA(), x)
    assert compute_graph_shape_hash(trace_1) == compute_graph_shape_hash(trace_2)


def _conv_table_sizes(trace: Any, op_label: str) -> list[tuple[int, ...]]:
    """Return the RF table 'size' entries for one op label."""

    pytest.importorskip("pandas")
    frame = trace.receptive_fields().to_pandas()
    return [tuple(size) for size in frame[frame["output_op"] == op_label]["size"]]


def _windowed_bounds(box: Any) -> list[tuple[int, int]]:
    """Return (start, stop) index bounds for a box's windowed axes."""

    return [(axis.index_start, axis.index_stop) for axis in box.axes if axis.kind == "windowed"]


def test_rf_table_matches_at_after_geometry_mutation() -> None:
    """A shape-preserving geometry-arg change must invalidate the RF table cache."""

    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16)
    log = tl.trace(DilatedConvModel(dilation=1, padding=1), x)
    conv_op = next(op for op in log._layer_list if op.label.startswith("conv2d"))
    assert tuple(conv_op.shape) == (1, 1, 16, 16)

    # Prime the trace-level solution cache through the public table path.
    assert _conv_table_sizes(log, conv_op.label) == [(3, 3)]
    revision_before = _engine._graph_revision(log)
    solution_before = _engine.solve(log)

    # In-place, shape-preserving geometry mutation: dil 1->2 / pad 1->2 keeps
    # the 16x16 output while changing the true RF span from 3 to 5.
    conv_op.func_config["dilation"] = (2, 2)
    conv_op.func_config["padding"] = (2, 2)

    # The staleness signature must bump and the cached solution must be rebuilt.
    assert _engine._graph_revision(log) != revision_before
    assert _engine.solve(log) is not solution_before

    # Ground truth: a fresh capture with the mutated geometry.
    fresh = tl.trace(DilatedConvModel(dilation=2, padding=2), x)
    fresh_op = next(op for op in fresh._layer_list if op.label.startswith("conv2d"))

    table_sizes = _conv_table_sizes(log, conv_op.label)
    assert table_sizes == _conv_table_sizes(fresh, fresh_op.label) == [(5, 5)]

    # Both public paths agree: the table extent equals the .at() extent.
    box = conv_op.receptive_field.at("center")
    fresh_box = fresh_op.receptive_field.at("center")
    assert _windowed_bounds(box) == _windowed_bounds(fresh_box)
    at_widths = tuple(stop - start for start, stop in _windowed_bounds(box))
    assert at_widths == table_sizes[0]


def test_rf_solution_cache_stable_when_nothing_changes() -> None:
    """The by-value geometry snapshot must not spuriously invalidate the cache."""

    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16)
    log = tl.trace(DilatedConvModel(), x)
    first = _engine.solve(log)
    assert _engine.solve(log) is first
    assert _engine._graph_revision(log) == _engine._graph_revision(log)
