"""TensorFlow preview: protocol object, named validation verdicts, constants rule."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import tensorflow_backend_modules

import torchlens as tl
from torchlens.backends._backend_protocol import (
    REFUSAL_CODES,
    VALIDATION_REASONS,
    BackendProtocol,
    protocol_problems,
)
from torchlens.backends.registry import get_backend_spec
from torchlens.validation.status import ValidationStatus

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
    pytest.mark.rare,
]


def _protocol() -> Any:
    """The registered tf protocol object."""

    spec = get_backend_spec("tf")
    assert spec.protocol is not None
    return spec.protocol()


def _x() -> Any:
    """Deterministic float32 example input."""

    return tf.constant(np.arange(8, dtype=np.float32).reshape(2, 4) / 8.0 - 0.4)


def test_tf_protocol_object_is_complete_and_declares_registered_codes() -> None:
    """The shell satisfies the protocol; its pins agree; its codes are registered."""

    proto = _protocol()
    assert isinstance(proto, BackendProtocol)
    assert protocol_problems(proto, expected_name="tf") == ()
    assert proto.maturity == "preview"
    assert proto.materialization == "eager"
    assert proto.framework._consistency_problems() == ()
    assert proto.framework.pinned is not None
    assert proto.framework._admits(proto.framework.pinned)
    assert set(proto.refusal_codes()) <= REFUSAL_CODES
    assert proto.runtime_available() is None
    assert set(proto.trace_options()) == set(proto.capabilities().trace_options)


def test_tf_host_copy_keeps_scalar_shape_and_dtype() -> None:
    """``host_copy`` keeps a 0-d tensor 0-d and refuses a resource handle typed."""

    proto = _protocol()
    scalar = proto.host_copy(tf.constant(2.5))
    assert scalar.shape == ()
    assert scalar.dtype == np.float32
    matrix = proto.host_copy(tf.Variable(np.ones((2, 3), dtype=np.float32)))
    assert matrix.shape == (2, 3)


def test_tf_validate_returns_status_never_bool() -> None:
    """``protocol.validate`` returns the status object for passed and failed traces."""

    proto = _protocol()
    weights = tf.constant(np.ones((4, 3), dtype=np.float32))

    def fn(v: Any) -> Any:
        return tf.nn.relu(tf.matmul(v, weights))

    trace = tl.trace(fn, _x(), backend="tf")
    status = proto.validate(trace, proto.reference_output(fn, _x(), None))
    assert isinstance(status, ValidationStatus)
    assert status.state == "passed", status.message
    assert status.decisions
    assert all(record["decision"] in {"validated", "exempted"} for record in status.decisions)
    assert any(record["decision"] == "validated" for record in status.decisions)
    assert set(status.unverified_reason_counts) <= VALIDATION_REASONS
    assert set(status.exempted_reason_counts) <= VALIDATION_REASONS


def _assert_constant_leaf_decisions(status: ValidationStatus, constants: list[Any]) -> None:
    """A9: the decisions name every constant op under the two leaf reasons."""

    by_label = {
        (record["op_label"], record["phase"]): record
        for record in status.decisions
        if record["func_name"] == "constant"
    }
    for op in constants:
        replay = by_label[(op.label, "replay")]
        perturb = by_label[(op.label, "perturbation")]
        assert (replay["decision"], replay["reason"]) == ("exempted", "constant_producer_leaf")
        assert (perturb["decision"], perturb["reason"]) == ("exempted", "constant_parent_payload")
    assert status.exempted_reason_counts["constant_producer_leaf"] == len(constants)
    assert status.exempted_reason_counts["constant_parent_payload"] == len(constants)


def test_tf_closure_constant_is_a_constant_source_op_with_payload() -> None:
    """SPEC 2 rule 1: a closure array is a ``constant`` op, parent of its consumer."""

    weights = tf.constant(np.full((4, 3), 0.5, dtype=np.float32))
    bias = tf.constant(np.array([1.0, -1.0, 0.0], dtype=np.float32))

    def fn(v: Any) -> Any:
        return tf.matmul(v, weights) + bias

    trace = tl.trace(fn, _x(), backend="tf")
    constants = [op for op in trace._layer_list if op.func_name == "constant"]
    assert len(constants) == 2
    for op in constants:
        assert tuple(op.parents or ()) == ()
        assert op.out is not None
        assert op.children
    matmul = next(op for op in trace._layer_list if op.func_name == "MatMul")
    assert any(parent in {c.label, c.layer_label} for c in constants for parent in matmul.parents)
    np.testing.assert_array_equal(np.asarray(constants[0].out), np.full((4, 3), 0.5))
    status = _protocol().validate(trace, None)
    assert status.state == "passed", status.message
    _assert_constant_leaf_decisions(status, constants)


def test_tf_scalar_literal_stays_replayable_without_an_op() -> None:
    """A rank-0 literal is a replayable captured value, not a parentless op."""

    def fn(v: Any) -> Any:
        return v * 2.0 + 1.0

    trace = tl.trace(fn, _x(), backend="tf")
    assert [op.func_name for op in trace._layer_list] == ["input", "Mul", "AddV2"]
    assert _protocol().validate(trace, None).state == "passed"


def test_tf_forward_fill_is_a_literal_producer_that_validates() -> None:
    """F5: ``tf.ones`` inside the forward replays from its literal arguments."""

    def fn(v: Any) -> Any:
        return tf.matmul(tf.transpose(tf.reshape(v, [4, 2])), tf.ones([4, 1]))

    trace = tl.trace(fn, _x(), backend="tf")
    fill = next(op for op in trace._layer_list if op.func_name == "Fill")
    assert tuple(fill.parents or ()) == ()
    assert fill.out is not None and tuple(np.asarray(fill.out).shape) == (4, 1)
    status = _protocol().validate(trace, None)
    assert status.state == "passed", status.message
    fill_decisions = [r for r in status.decisions if r["func_name"] == "Fill"]
    assert fill_decisions and fill_decisions[0]["decision"] == "validated"


def test_tf_keras_gelu_erfc_is_classified_and_replayed() -> None:
    """Keras exact gelu runs ``Erfc``; validation classifies and replays it."""

    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [keras.Input((4,)), keras.layers.Dense(4), keras.layers.Activation("gelu")]
    )
    trace = tl.trace(model, _x(), backend="tf")
    assert any(op.func_name == "Erfc" for op in trace._layer_list)
    status = _protocol().validate(trace, None)
    assert status.state != "failed", status.message
    erfc = [r for r in status.decisions if r["func_name"] == "Erfc"]
    assert erfc and erfc[0]["decision"] == "validated"
    assert set(status.unverified_reason_counts) <= VALIDATION_REASONS


def test_tf_tampered_payload_fails_naming_the_op_and_reason() -> None:
    """A11: a tampered payload yields ``failed`` naming the op in message and decisions."""

    weights = tf.constant(np.ones((4, 3), dtype=np.float32))

    def fn(v: Any) -> Any:
        return tf.nn.relu(tf.matmul(v, weights))

    trace = tl.trace(fn, _x(), backend="tf")
    relu = next(op for op in trace._layer_list if op.func_name == "Relu")
    relu.out = tf.constant(np.asarray(relu.out) + 7.0)
    status = _protocol().validate(trace, None)
    assert isinstance(status, ValidationStatus)
    assert status.state == "failed"
    assert relu.label in status.message or relu.layer_label in status.message
    failed = [r for r in status.decisions if r["decision"] == "failed"]
    assert failed
    assert failed[0]["op_label"] in {relu.label, relu.layer_label}
    assert failed[0]["reason"] in VALIDATION_REASONS


def test_tf_initializer_fill_without_consumers_still_fails_closed() -> None:
    """The literal-producer carve-out admits only a consumed ``Fill``."""

    from torchlens.backends.tf.validation import _is_literal_producer

    class _Op:
        func_name = "Fill"
        parents: tuple[str, ...] = ()
        children: tuple[str, ...] = ()

    assert _is_literal_producer(_Op()) is False
    consumed = _Op()
    consumed.children = ("matmul_1_2",)
    assert _is_literal_producer(consumed) is True
    with_parent = _Op()
    with_parent.children = ("matmul_1_2",)
    with_parent.parents = ("input_1_1",)
    assert _is_literal_producer(with_parent) is False
