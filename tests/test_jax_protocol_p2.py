"""JAX preview: protocol object, named validation verdicts, constants rule."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import torchlens as tl
from torchlens._validation_status import ValidationStatus
from torchlens.backends._backend_protocol import (
    REFUSAL_CODES,
    VALIDATION_REASONS,
    BackendProtocol,
    protocol_problems,
)
from torchlens.backends.registry import get_backend_spec

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]


def _protocol() -> Any:
    """The registered jax protocol object."""

    spec = get_backend_spec("jax")
    assert spec.protocol is not None
    return spec.protocol()


def _x() -> Any:
    """Deterministic float32 example input."""

    return jnp.asarray(np.arange(8, dtype=np.float32).reshape(2, 4) / 8.0 - 0.4)


def test_jax_protocol_object_is_complete_and_declares_registered_codes() -> None:
    """The shell satisfies the protocol; its pins agree; its codes are registered."""

    proto = _protocol()
    assert isinstance(proto, BackendProtocol)
    assert protocol_problems(proto, expected_name="jax") == ()
    assert not hasattr(proto, "maturity")
    assert proto.materialization == "eager"
    assert proto.framework._consistency_problems() == ()
    assert proto.framework.pinned is not None
    assert proto.framework._admits(proto.framework.pinned)
    assert set(proto.refusal_codes()) <= REFUSAL_CODES
    assert proto.runtime_available() is None
    assert set(proto.trace_options()) == set(proto.capabilities().trace_options)


def test_jax_host_copy_keeps_scalar_shape_and_dtype() -> None:
    """``host_copy`` keeps a 0-d array 0-d with its dtype."""

    proto = _protocol()
    scalar = proto.host_copy(jnp.asarray(np.float32(2.5)))
    assert scalar.shape == ()
    assert scalar.dtype == np.float32
    assert proto.host_copy(jnp.ones((2, 3))).shape == (2, 3)


def test_jax_validate_returns_status_never_bool() -> None:
    """``protocol.validate`` returns the status object with per-op decisions."""

    proto = _protocol()

    def fn(v: Any) -> Any:
        return jnp.transpose(v.reshape(4, 2)) @ jnp.ones((4, 1))

    trace = tl.trace(cast(Any, fn), _x(), backend="jax")
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


def test_jax_closure_constant_is_a_constant_source_op_with_payload() -> None:
    """SPEC 2 rule 1 and F7: closure arrays are ``constant`` ops, parents of their readers."""

    weights = jnp.asarray(np.full((4, 3), 0.5, dtype=np.float32))
    bias = jnp.asarray(np.array([1.0, -1.0, 0.0], dtype=np.float32))

    def fn(v: Any) -> Any:
        return jnp.matmul(v, weights) + bias

    trace = tl.trace(cast(Any, fn), _x(), backend="jax")
    constants = [op for op in trace._layer_list if op.func_name == "constant"]
    assert len(constants) == 2
    for op in constants:
        assert tuple(op.parents or ()) == ()
        assert op.out is not None
        assert op.children
    dot = next(op for op in trace._layer_list if op.func_name == "dot_general")
    assert any(parent in {c.label, c.layer_label} for c in constants for parent in dot.parents)
    np.testing.assert_array_equal(np.asarray(constants[0].out), np.full((4, 3), 0.5))
    status = _protocol().validate(trace, None)
    assert status.state == "passed", status.message
    _assert_constant_leaf_decisions(status, constants)


def test_jax_norm_softmax_validation_is_a_named_verdict() -> None:
    """F6: literal producers and the softmax chain either pass or name the op that did not."""

    def fn(v: Any) -> Any:
        mu = jnp.mean(v, axis=-1, keepdims=True)
        var = jnp.var(v, axis=-1, keepdims=True)
        n = (v - mu) / jnp.sqrt(var + 1e-5)
        return jax.nn.softmax(n, axis=-1)

    trace = tl.trace(cast(Any, fn), _x(), backend="jax")
    literal = [
        op for op in trace._layer_list if op.func_name == "convert_element_type" and not op.parents
    ]
    assert literal, "literal convert_element_type producers are recorded with zero parents"
    assert all(op.out is not None for op in literal)
    status = _protocol().validate(trace, None)
    assert isinstance(status, ValidationStatus)
    if status.state == "failed":
        failed = [r for r in status.decisions if r["decision"] == "failed"]
        assert failed and failed[0]["op_label"] in status.message
        pytest.fail(f"norm_softmax validation failed at {failed[0]}: {status.message}")
    assert status.state == "passed", status.message


def test_jax_tampered_payload_fails_naming_the_op_and_reason() -> None:
    """A11: a tampered payload yields ``failed`` naming the op in message and decisions."""

    def fn(v: Any) -> Any:
        return jax.nn.relu(v * 2.0)

    trace = tl.trace(cast(Any, fn), _x(), backend="jax")
    mul = next(op for op in trace._layer_list if op.func_name == "mul")
    mul.out = jnp.asarray(np.asarray(mul.out) + 7.0)
    status = _protocol().validate(trace, None)
    assert isinstance(status, ValidationStatus)
    assert status.state == "failed"
    assert mul.label in status.message or mul.layer_label in status.message
    failed = [r for r in status.decisions if r["decision"] == "failed"]
    assert failed
    assert failed[0]["op_label"] in {mul.label, mul.layer_label}
    assert failed[0]["reason"] in VALIDATION_REASONS
