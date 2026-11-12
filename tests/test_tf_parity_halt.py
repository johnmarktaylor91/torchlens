"""TensorFlow ``trace(stop_after=...)``: the halted prefix is the full run's prefix.

Each roster row (raw eager functions, Keras functional and subclassed models) is
captured in full, then halted at the first occurrence of a mid-trace op type. The
halted trace must equal the same-length prefix of the full trace op for op (type,
label and value), carry torch's stopped-outcome fields, and pass replay validation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from tests.backends.conftest import tensorflow_backend_modules
from torchlens._capture.outcome import CaptureStatus
from torchlens.backends.tf.backend import TFBackend
from torchlens.errors import BackendUnsupportedError, InvalidArgumentError, TorchLensWarning
from torchlens.recording._exceptions import PredicateError

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
    pytest.mark.rare,
]

#: Roster rows covering raw eager code, Keras functional and Keras subclassed models.
_ROWS = ("chain", "multi", "loop3", "mlp", "shared", "cnn", "nested_residual")
#: Parameter reads and constants are not useful frontiers (they precede the forward).
_NOT_FRONTIERS = {"ReadVariableOp", "VarHandleOp", "Const", "Identity", "NoOp"}


def _roster_row(family: str) -> tuple[Any, Any]:
    """Build one TF roster row's model and input."""

    from tests.conformance._rosters import tf as tf_roster

    spec = next(spec for spec in tf_roster.ROW_SPECS if spec["family"] == family)
    return spec["builder"]()


def _ops(trace: Any) -> list[tuple[str, str, np.ndarray]]:
    """Return ``(func_name, label, value)`` per recorded layer, in capture order."""

    return [
        (str(layer.func_name), str(layer.label), np.asarray(layer.out))
        for layer in trace._layer_list
    ]


def _frontier_op_type(trace: Any) -> str:
    """Return the op type of a compute op near the middle of a full trace."""

    candidates = [
        str(layer.func_name)
        for layer in trace._layer_list
        if layer.func_name and str(layer.func_name) not in _NOT_FRONTIERS and layer.parents
    ]
    assert candidates, "roster row recorded no compute ops"
    return candidates[len(candidates) // 2]


@pytest.mark.parametrize("family", _ROWS)
def test_tf_halted_prefix_equals_full_run_prefix(family: str) -> None:
    """The halted capture is op-for-op the full capture's prefix and validates."""

    model, x = _roster_row(family)
    full = tl.trace(model, x, backend="tf")
    op_type = _frontier_op_type(full)
    halted = tl.trace(model, x, backend="tf", stop_after=tl.func(op_type.lower()))

    assert halted.is_stopped is True
    assert halted.outcome is not None
    assert halted.outcome.status is CaptureStatus.STOPPED
    assert halted.graph.stop_frontier == halted.stop_reason
    halted_ops, full_ops = _ops(halted), _ops(full)
    assert 0 < len(halted_ops) < len(full_ops)
    assert halted_ops[-1][0] == op_type
    first = next(index for index, op in enumerate(full_ops) if op[0] == op_type)
    assert len(halted_ops) == first + 1
    for (h_type, h_label, h_value), (f_type, f_label, f_value) in zip(halted_ops, full_ops):
        assert (h_type, h_label) == (f_type, f_label)
        np.testing.assert_array_equal(h_value, f_value)
    assert TFBackend().validate_trace(halted) is True


def test_tf_halt_stops_execution_at_frontier() -> None:
    """Post-frontier user code never runs; the frontier op is the last record."""

    executed: list[str] = []

    def forward(x: Any) -> Any:
        hidden = tf.nn.relu(x * 2.0)
        executed.append("post-frontier")
        return tf.reduce_sum(hidden + 1.0)

    trace = tl.trace(
        forward, tf.constant([1.0, -1.0, 2.0]), backend="tf", stop_after=tl.func("relu")
    )
    assert executed == []
    assert trace.is_stopped is True
    assert "relu" in str(trace.stop_reason).lower()
    assert str(trace._layer_list[-1].func_name) == "Relu"
    assert not any(str(layer.func_name) == "Sum" for layer in trace._layer_list)
    assert TFBackend().validate_trace(trace) is True


def test_tf_halt_callable_predicate_reads_context() -> None:
    """A callable stop predicate sees the op context and halts like a selector."""

    model, x = _roster_row("mlp")
    trace = tl.trace(
        model, x, backend="tf", stop_after=lambda ctx: str(ctx.func_name).lower() == "relu"
    )
    assert trace.is_stopped is True
    assert str(trace._layer_list[-1].func_name) == "Relu"


def test_tf_unfired_stop_after_selector_refuses_like_torch() -> None:
    """A stop_after selector that never fires refuses typed, with torch's code."""

    model, x = _roster_row("mlp")
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model, x, backend="tf", stop_after=tl.func("does_not_exist"))
    assert excinfo.value.fields["code"] == "stop_after_never_fired"


def test_tf_unfired_stop_after_callable_warns_like_torch() -> None:
    """An exploratory stop_after callable that never fires warns coded, torch-style."""

    model, x = _roster_row("mlp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, x, backend="tf", stop_after=lambda ctx: False)
    assert trace.is_stopped is False
    codes = [
        getattr(entry.message, "fields", {}).get("code")
        for entry in caught
        if isinstance(entry.message, TorchLensWarning)
    ]
    assert "stop_after_never_fired_callable" in codes


def test_tf_halt_non_bool_predicate_refuses_typed() -> None:
    """halt predicates must return bool, mirroring the torch contract."""

    model, x = _roster_row("mlp")
    with pytest.raises(PredicateError, match="must return bool"):
        tl.trace(model, x, backend="tf", stop_after=lambda ctx: 1)


def test_tf_halt_refuses_static_graph_entry_typed() -> None:
    """A graph-only entry has no forward to stop; stop_after refuses typed."""

    compiled = tf.function(lambda value: tf.reduce_sum(tf.nn.relu(value)))
    with pytest.raises(BackendUnsupportedError, match="eager live capture") as excinfo:
        tl.trace(
            compiled,
            tf.constant([1.0, -1.0, 2.0]),
            backend="tf",
            stop_after=tl.func("relu"),
        )
    assert excinfo.value.code == "trace_option_unsupported"


def test_tf_halt_after_edit_on_the_same_op() -> None:
    """With an edit and a halt on one op, the edit lands before the halt (torch order)."""

    x = tf.constant([1.0, -1.0, 2.0])

    def forward(v: Any) -> Any:
        return tf.reduce_sum(tf.nn.relu(v) + 1.0)

    trace = tl.trace(
        forward,
        x,
        backend="tf",
        intervene=tl.when(tl.func("relu"), tl.scale(3.0)),
        stop_after=tl.func("relu"),
    )
    assert trace.is_stopped is True
    relu = next(layer for layer in trace._layer_list if str(layer.func_name) == "Relu")
    assert relu.intervention_fired is True
    final = np.asarray(trace._layer_list[-1].out)
    np.testing.assert_allclose(final, [3.0, 0.0, 6.0])
    assert not any(str(layer.func_name) == "Sum" for layer in trace._layer_list)
