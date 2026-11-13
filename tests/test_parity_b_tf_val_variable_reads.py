"""TensorFlow variable reads verified as parameter sources (ruling X4).

An eager ``ReadVariableOp`` whose handle resolves to a variable of the discovered
module tree, with no other op consuming that handle during the forward and a saved
output equal bit for bit to the variable's pre-forward snapshot, validates as a
parameter source; downstream ops keep full replay and perturbation. Every
neighbouring case keeps the old ``effectful_region`` verdict or fails: a write to
the handle, a training BatchNorm moving-statistics write, a variable outside the
module tree, a graph-mode function call, and a saved read that differs from the
snapshot (fails, naming the read).

The TF capture enters :func:`variable_read_scope` through its forward hooks; on a
tree without those hooks the ``wired`` fixture wraps ``run_wired_session`` the same
way.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.backends.conftest import tensorflow_backend_modules
from torchlens.backends.tf import _variable_reads

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

_needs_tf = pytest.mark.skipif(
    _TF_BACKEND_SKIP_REASON is not None,
    reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
)

#: The four Keras module rows that were ``unverified`` only because of their reads.
_KERAS_ROWS = ("mlp_module", "shared_weight_module", "cnn_module", "nested_residual_module")
_READ = "ReadVariableOp"


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enter the variable-read scope around the eager forward, as the backend hunk does."""

    from torchlens import _state
    from torchlens.backends.tf import backend as tf_backend

    if _backend_wires_the_scope():
        yield
        return
    original = tf_backend.run_wired_session

    def run_wired_session(tf_module: Any, session: Any, wiring: Any) -> Any:
        """Run the forward inside the scope and attach its ledger to the active trace."""

        trace = _state._active_trace
        with _variable_reads.variable_read_scope(tf_module, session.module_tree) as ledger:
            result = original(tf_module, session, wiring)
        _variable_reads.attach_variable_reads(trace, ledger)
        return result

    monkeypatch.setattr(tf_backend, "run_wired_session", run_wired_session)
    yield


def _backend_wires_the_scope() -> bool:
    """Whether the TF capture's forward hooks enter the variable-read scope themselves."""

    try:
        from torchlens.backends.tf import _capture_hooks
    except ImportError:
        return False
    return getattr(_capture_hooks, "_VARIABLE_READS_MODULE", None) == _variable_reads.__name__


def _row(family: str) -> tuple[Any, Any]:
    """Build one TF roster row's model and input."""

    from tests.conformance._rosters import tf as tf_roster

    spec = next(spec for spec in tf_roster.ROW_SPECS if spec["family"] == family)
    return spec["builder"]()


def _capture(model: Any, x: Any) -> Any:
    """Trace ``model`` on ``x`` with the TF backend."""

    import torchlens as tl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, x, backend="tf")


def _validate(trace: Any) -> Any:
    """Run TF replay validation and return the status."""

    from torchlens.backends.tf.validation import validate_tf_trace_status

    return validate_tf_trace_status(trace)


def _read_decisions(status: Any) -> list[dict[str, Any]]:
    """The validation decisions of every variable read."""

    return [dict(item) for item in status.decisions if item.get("func_name") == _READ]


def _verified_addresses(status: Any) -> list[str]:
    """The variable address each validated read names, in decision order."""

    out = []
    for item in _read_decisions(status):
        if item["decision"] == "validated":
            out.append(str(item["justification"]).split("variable ", 1)[1].split(" and ", 1)[0])
    return out


@_needs_tf
@pytest.mark.tf_backend
@pytest.mark.parametrize("family", _KERAS_ROWS)
def test_keras_row_reads_verify_and_row_passes(family: str, wired: None) -> None:
    """Every read of an untouched module variable validates; the row passes."""

    trace = _capture(*_row(family))
    status = _validate(trace)
    reads = _read_decisions(status)
    assert reads, "the row reads its variables"
    assert all(item["decision"] == "validated" for item in reads), reads
    assert all(item["reason"] == "replay_match" for item in reads)
    assert status.state == "passed", status.message
    assert status.num_replayed_ops > 0, "downstream ops are still replayed"


@_needs_tf
@pytest.mark.tf_backend
def test_repeated_reads_bind_to_one_snapshot(wired: None) -> None:
    """The shared layer's second call reads the same two variables again."""

    status = _validate(_capture(*_row("shared_weight_module")))
    addresses = _verified_addresses(status)
    assert len(addresses) == 4 and len(set(addresses)) == 2, addresses


@_needs_tf
@pytest.mark.tf_backend
def test_without_a_ledger_reads_stay_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that leaves no ledger keeps the effectful_region verdict."""

    @contextmanager
    def no_ledger(tf_module: Any, module_tree: Any) -> Iterator[None]:
        """Record nothing, as a capture without a module tree does."""

        del tf_module, module_tree
        yield None

    monkeypatch.setattr(_variable_reads, "variable_read_scope", no_ledger)
    status = _validate(_capture(*_row("mlp_module")))
    assert {item["reason"] for item in _read_decisions(status)} == {"effectful_region"}
    assert status.state == "unverified"


def _writer_model() -> tuple[Any, Any]:
    """A layer that reads its kernel, then writes it in the same forward."""

    keras.utils.set_random_seed(0)

    class Writer(keras.layers.Layer):
        """Matmul with ``w``, then ``w.assign(w)``; ``b`` is never written."""

        def build(self, shape: Any) -> None:
            """Create the written kernel and the untouched bias."""

            self.w = self.add_weight(shape=(4, 4), initializer="ones", name="w")
            self.b = self.add_weight(shape=(4,), initializer="zeros", name="b")

        def call(self, v: Any) -> Any:
            """Use both variables, then write ``w`` with its own value."""

            out = tf.matmul(v, self.w) + self.b
            self.w.assign(self.w * 1.0)
            return out

    model = keras.Sequential([keras.Input((4,)), Writer()])
    x = tf.constant(np.ones((2, 4), np.float32))
    return model, x


@_needs_tf
@pytest.mark.tf_backend
def test_write_to_the_handle_keeps_its_reads_unverified(wired: None) -> None:
    """Reads of a written variable stay unverified; the untouched bias still verifies."""

    status = _validate(_capture(*_writer_model()))
    reasons = [item["reason"] for item in _read_decisions(status)]
    assert "effectful_region" in reasons, reasons
    assert any(address.endswith("b") for address in _verified_addresses(status))
    assert not any(address.endswith("w") for address in _verified_addresses(status))
    assert status.state == "unverified"


@_needs_tf
@pytest.mark.tf_backend
def test_training_batchnorm_moving_statistics_stay_unverified(wired: None) -> None:
    """A BatchNorm called with ``training=True`` writes its moving statistics."""

    keras.utils.set_random_seed(0)
    inputs = keras.Input((4,))
    outputs = keras.layers.BatchNormalization()(keras.layers.Dense(4)(inputs), training=True)
    model = keras.Model(inputs, outputs)
    status = _validate(_capture(model, tf.constant(np.arange(8, dtype=np.float32).reshape(2, 4))))
    verified = _verified_addresses(status)
    assert not any("moving" in address for address in verified), verified
    assert status.state != "passed"


@_needs_tf
@pytest.mark.tf_backend
def test_variable_outside_the_module_tree_stays_unverified(wired: None) -> None:
    """A closure variable no module tracks is an unknown handle."""

    keras.utils.set_random_seed(0)
    outside = tf.Variable(np.full((4,), 0.5, np.float32), name="outside")

    class Uses(keras.Model):
        """Dense plus an untracked global variable."""

        def __init__(self) -> None:
            super().__init__()
            self.d = keras.layers.Dense(4)

        def call(self, v: Any) -> Any:
            """Add the untracked variable to the dense output."""

            return self.d(v) + outside

    model = Uses()
    x = tf.constant(np.ones((2, 4), np.float32))
    model(x)
    status = _validate(_capture(model, x))
    reasons = [item["reason"] for item in _read_decisions(status)]
    assert reasons.count("effectful_region") == 1, reasons
    assert len(_verified_addresses(status)) == 2
    assert status.state == "unverified"


@_needs_tf
@pytest.mark.tf_backend
def test_read_that_differs_from_the_snapshot_fails_naming_it(wired: None) -> None:
    """A saved read changed after capture no longer equals the snapshot: failed."""

    trace = _capture(*_row("mlp_module"))
    read = next(op for op in trace._layer_list if op.func_name == _READ)
    read.out = read.out + 1
    status = _validate(trace)
    assert status.state == "failed"
    failed = [item for item in _read_decisions(status) if item["decision"] == "failed"]
    assert [item["op_label"] for item in failed] == [str(read.label)]
    assert failed[0]["reason"] == "replay_mismatch"


@_needs_tf
@pytest.mark.tf_backend
def test_graph_mode_function_call_verifies_no_read(wired: None) -> None:
    """A ``tf.function`` inside the forward is an opaque region: no read verifies."""

    keras.utils.set_random_seed(0)

    @tf.function
    def inner(v: Any) -> Any:
        """A compiled helper the eager callback cannot see into."""

        return v * 2.0

    class Compiled(keras.Model):
        """Dense followed by a compiled helper."""

        def __init__(self) -> None:
            super().__init__()
            self.d = keras.layers.Dense(4)

        def call(self, v: Any) -> Any:
            """Run the dense layer, then the compiled helper."""

            return inner(self.d(v))

    model = Compiled()
    x = tf.constant(np.ones((2, 4), np.float32))
    model(x)
    status = _validate(_capture(model, x))
    assert _verified_addresses(status) == []
    assert status.state != "passed"


# Framework-free checks of the read gate (fake captures and ledgers).


class _FakeTrace:
    """A weakly referenceable stand-in for a trace."""


def _fake_capture(key: object, op_type: str = _READ) -> Any:
    """A capture record whose handle input carries ``key``."""

    handle = SimpleNamespace(input_index=0, ref_key=key)
    return SimpleNamespace(op_type=op_type, inputs=(handle,), label_raw="readvariableop_1")


def _snapshot() -> _variable_reads.VariableSnapshot:
    """A snapshot of a 2-element float32 zero vector."""

    value = np.zeros((2,), np.float32)
    return _variable_reads.VariableSnapshot(
        address="dense.kernel",
        dtype="float32",
        shape=(2,),
        digest=_variable_reads._digest(value),
    )


@pytest.mark.parametrize(
    ("ledger_kwargs", "key", "op_type", "verifiable"),
    [
        pytest.param({}, "h", _READ, True, id="untouched-known-handle"),
        pytest.param({"touched": {"h": "AssignVariableOp"}}, "h", _READ, False, id="written"),
        pytest.param({}, "other", _READ, False, id="unknown-handle"),
        pytest.param({"opaque": "StatefulPartitionedCall"}, "h", _READ, False, id="opaque"),
        pytest.param({}, "h", "ResourceGather", False, id="not-a-read"),
    ],
)
def test_read_gate(
    ledger_kwargs: dict[str, Any], key: object, op_type: str, verifiable: bool
) -> None:
    """Only an untouched read of a snapshotted handle in a transparent forward binds."""

    ledger = _variable_reads.VariableReadLedger(snapshots={"h": _snapshot()}, **ledger_kwargs)
    found = _variable_reads._verifiable_snapshot(ledger, _fake_capture(key, op_type))
    assert (found is not None) is verifiable


def test_no_ledger_verifies_nothing() -> None:
    """A trace without a ledger (static, loaded, unwired) keeps every read unverified."""

    assert _variable_reads._verifiable_snapshot(None, _fake_capture("h")) is None
    assert _variable_reads.parameter_read_decision(_FakeTrace(), SimpleNamespace(), None) is None
    assert (
        _variable_reads.parameter_read_decision(SimpleNamespace(), None, _fake_capture("h")) is None
    )


@pytest.mark.parametrize(
    ("payload", "decision"),
    [
        pytest.param(np.zeros((2,), np.float32), "validated", id="equal"),
        pytest.param(np.array([0.0, 1e-30], np.float32), "failed", id="one-bit-off"),
        pytest.param(np.zeros((2,), np.float64), "failed", id="dtype"),
        pytest.param(np.zeros((1, 2), np.float32), "failed", id="shape"),
    ],
)
def test_bit_for_bit_comparison(payload: np.ndarray, decision: str) -> None:
    """The saved read must equal the snapshot in dtype, shape and every bit."""

    trace = _FakeTrace()
    _variable_reads.attach_variable_reads(
        trace, _variable_reads.VariableReadLedger(snapshots={"h": _snapshot()})
    )
    op = SimpleNamespace(label="readvariableop_1", has_saved_activation=True, out=payload)
    result = _variable_reads.parameter_read_decision(trace, op, _fake_capture("h"))
    assert result is not None and result["decision"] == decision
    expected = [] if decision == "validated" else ["replay_or_perturbation_failed"]
    assert [item.split(":", 1)[0] for item in _variable_reads.read_failures("x", result)] == (
        expected
    )
