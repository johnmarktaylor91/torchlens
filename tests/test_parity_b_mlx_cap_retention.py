"""Capture-time retention on MLX: sparse ``save=`` and structure-only captures.

A capture inside ``retention_scope`` keeps public payloads only for selected
ops and spills every other value as a replay witness. These tests hold it to
the full capture filtered by the same selector: identical public payloads
(bit-equal), identical saved flags and the same validation verdict, with
validation still re-executing every op against real witnesses.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

if sys.platform != "darwin" and os.environ.get("TORCHLENS_RUN_UNSTABLE_MLX_TESTS") != "1":
    pytest.skip(
        "MLX exec-level tests are skipped on non-Darwin platforms by default; "
        "set TORCHLENS_RUN_UNSTABLE_MLX_TESTS=1 to opt in.",
        allow_module_level=True,
    )

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens._capture.structure_only import (  # noqa: E402
    StructureOnlyCapabilityError,
    require_structure_only_capability,
)
from torchlens._postprocess._selective_save import apply_static_label_save_policy  # noqa: E402
from torchlens.backends.mlx._retention import (  # noqa: E402
    RetentionRequest,
    SpillStore,
    retention_scope,
)
from torchlens.backends.mlx.backend import MLXBackend  # noqa: E402
from torchlens.errors import TorchLensWarning  # noqa: E402

pytestmark = [pytest.mark.rare, pytest.mark.backend_mlx]


class ResidualMLP(nn.Module):
    """Three linear layers with a skip connection, so witnesses span chunks."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.l1 = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, 4)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = nn.relu(self.l1(x))
        hidden = nn.relu(self.l2(hidden)) + x
        return self.l3(hidden)


class NoisyMLP(nn.Module):
    """A forward that draws from the MLX random generator."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(8, 8)
        self.l2 = nn.Linear(8, 2)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = nn.relu(self.l1(x))
        hidden = hidden + mx.random.normal(hidden.shape)
        return self.l2(hidden)


def _full(model: Any, x: mx.array, selector: Any) -> Any:
    """Full capture filtered after the fact: the reference the sparse path must equal."""

    trace = MLXBackend().capture_trace(model, x)
    apply_static_label_save_policy(trace, selector, backend_name="MLX")
    return trace


def _sparse(model: Any, x: mx.array, selector: Any, chunk_bytes: int = 1) -> Any:
    """Capture under a retention request (what the dispatch hunk does)."""

    with retention_scope(RetentionRequest(selector=selector, chunk_bytes=chunk_bytes)):
        return MLXBackend().capture_trace(model, x)


def _status_key(status: Any) -> tuple[Any, ...]:
    """Comparable summary of a validation status."""

    return (
        bool(status.passed),
        str(getattr(status, "state", "")),
        getattr(status, "num_replayed_ops", None),
        getattr(status, "num_unverified_ops", None),
        getattr(status, "num_failed_ops", None),
    )


def _assert_same_public_surface(sparse: Any, full: Any) -> None:
    """Public payloads and saved flags of ``sparse`` equal those of ``full``."""

    assert list(sparse.saved_ops.keys()) == list(full.saved_ops.keys())
    assert [op.label for op in sparse._layer_list] == [op.label for op in full._layer_list]
    for sparse_op, full_op in zip(sparse._layer_list, full._layer_list, strict=True):
        assert sparse_op.has_saved_activation == full_op.has_saved_activation, sparse_op.label
        if full_op.out is None:
            assert sparse_op.out is None, sparse_op.label
        else:
            assert sparse_op.out is not None, sparse_op.label
            assert sparse_op.out.dtype == full_op.out.dtype, sparse_op.label
            assert bool(mx.array_equal(sparse_op.out, full_op.out, equal_nan=True)), sparse_op.label
    assert sparse._num_saved_ops == full._num_saved_ops


@pytest.mark.parametrize(
    "selector",
    [
        tl.func("relu"),
        tl.label_contains("linear"),
        tl.func("linear") & ~tl.label_contains("linear_2"),
        tl.label("relu_1_2:1"),
    ],
    ids=["func", "contains", "composite", "final-label"],
)
def test_sparse_equals_full_on_selected_ops_and_validates(selector: Any) -> None:
    """Sparse public payloads equal the full capture's; both validate alike."""

    model = ResidualMLP()
    x = mx.random.normal((3, 16))
    full = _full(model, x, selector)
    sparse = _sparse(model, x, selector)

    _assert_same_public_surface(sparse, full)
    sparse_status = MLXBackend().validate_trace(sparse)
    full_status = MLXBackend().validate_trace(full)
    assert sparse_status.passed
    assert _status_key(sparse_status) == _status_key(full_status)


def test_witnesses_leave_memory_and_back_validation() -> None:
    """Non-selected values live in the spill store, not on the trace."""

    model = ResidualMLP()
    x = mx.random.normal((3, 16))
    sparse = _sparse(model, x, tl.func("relu"))
    hidden = sparse._selective_save_hidden_payloads
    store = hidden.maps[-1]
    assert isinstance(store, SpillStore)
    assert store.in_memory_count == 0
    unsaved = [op for op in sparse._layer_list if not op.has_saved_activation]
    assert unsaved
    for op in unsaved:
        if op.label.startswith(("input", "output")):
            continue
        assert op.out is None
        assert op._label_raw in store and op.label in store


def test_tampered_witness_fails_validation() -> None:
    """A changed spilled witness is caught: validation re-executes against it."""

    model = ResidualMLP()
    x = mx.random.normal((3, 16))
    sparse = _sparse(model, x, tl.func("relu"))
    store = sparse._selective_save_hidden_payloads.maps[-1]
    first_linear = next(op for op in sparse._layer_list if op.label.startswith("linear_1"))
    path = store._files[first_linear._label_raw]
    original = store[first_linear._label_raw]
    mx.save_safetensors(path, {"v": original + 1.0})

    status = MLXBackend().validate_trace(sparse)
    assert not status.passed


def test_random_forward_gives_identical_values() -> None:
    """Chunked realization keeps random draws and payloads bit-equal."""

    model = NoisyMLP()
    x = mx.random.normal((2, 8))
    # Random draws inside a forward are not wrapped ops today; the capture says so.
    unwrapped = "not produced by a wrapped operation"
    mx.random.seed(7)
    with pytest.warns(TorchLensWarning, match=unwrapped):
        full = _full(model, x, tl.func("linear"))
    mx.random.seed(7)
    with pytest.warns(TorchLensWarning, match=unwrapped):
        sparse = _sparse(model, x, tl.func("linear"))

    _assert_same_public_surface(sparse, full)
    assert _status_key(MLXBackend().validate_trace(sparse)) == _status_key(
        MLXBackend().validate_trace(full)
    )


def _bfloat16_linear_runs() -> bool:
    """Whether this MLX build evaluates a bfloat16 linear layer (Linux CPU builds do not)."""

    layer = nn.Linear(2, 2)
    layer.set_dtype(mx.bfloat16)
    try:
        mx.eval(layer(mx.ones((1, 2), dtype=mx.bfloat16)))
    except RuntimeError:
        return False
    return True


@pytest.mark.skipif(
    not _bfloat16_linear_runs(), reason="this MLX build has no bfloat16 matmul kernel"
)
def test_bfloat16_witnesses_spill() -> None:
    """Half-precision witnesses spill to disk too."""

    model = ResidualMLP()
    model.set_dtype(mx.bfloat16)
    x = mx.random.normal((3, 16)).astype(mx.bfloat16)
    full = _full(model, x, tl.func("relu"))
    sparse = _sparse(model, x, tl.func("relu"), chunk_bytes=64)

    _assert_same_public_surface(sparse, full)
    assert sparse._selective_save_hidden_payloads.maps[-1].in_memory_count == 0
    assert _status_key(MLXBackend().validate_trace(sparse)) == _status_key(
        MLXBackend().validate_trace(full)
    )


def test_structure_only_keeps_no_payloads_and_refuses_validation() -> None:
    """Structure-only keeps the graph, no values, and refuses validation entry."""

    model = ResidualMLP()
    x = mx.random.normal((3, 16))
    full = tl.trace(model, x, backend="mlx")
    with retention_scope(RetentionRequest(structure_only=True)):
        structure = MLXBackend().capture_trace(model, x)

    assert [op.label for op in structure._layer_list] == [op.label for op in full._layer_list]
    assert all(op.out is None for op in structure._layer_list)
    assert not any(op.has_saved_activation for op in structure._layer_list)
    assert structure._options_structure_only is True
    with pytest.raises(StructureOnlyCapabilityError):
        require_structure_only_capability(structure, "validation_entry")


def test_plain_capture_is_unchanged() -> None:
    """Without a request the capture keeps every payload and no witness store."""

    model = ResidualMLP()
    x = mx.random.normal((3, 16))
    plain = MLXBackend().capture_trace(model, x)

    ops = [op for op in plain._layer_list if not op.label.startswith("output")]
    assert all(op.has_saved_activation and op.out is not None for op in ops)
    assert not getattr(plain, "_selective_save_hidden_payloads", None)
    assert plain._options_structure_only is not True
    assert MLXBackend().validate_trace(plain).passed


def test_scope_does_not_leak_between_captures() -> None:
    """A capture after the scope closes is a plain capture again."""

    model = ResidualMLP()
    x = mx.random.normal((3, 16))
    _sparse(model, x, tl.func("relu"))
    after = MLXBackend().capture_trace(model, x)
    assert all(op.out is not None for op in after._layer_list if not op.label.startswith("output"))
