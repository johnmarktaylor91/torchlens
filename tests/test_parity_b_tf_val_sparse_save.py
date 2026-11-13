"""TensorFlow validation of a trace whose ``save=`` selector declined some payloads.

An op the validator would replay, whose own payload or a producer parent's payload
is absent because the user's ``save=`` selector declined it, is ``exempted`` with
reason ``not_saved_by_user`` (the PyTorch rule) instead of failing as a replay
mismatch. The proof is the capture's ``payload_declined`` flag. A payload absent
for any other reason (no flag) keeps the replay check and fails, and a fully saved
trace replays every op as before.

"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from tests.backends.conftest import tensorflow_backend_modules
from torchlens.backends._validation_shared import replay_payload
from torchlens.backends.tf import _replay_exemptions

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

_needs_tf = pytest.mark.skipif(
    _TF_BACKEND_SKIP_REASON is not None,
    reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
)
_NOT_SAVED = "not_saved_by_user"


def _chain(x: Any) -> Any:
    """relu((h + 1) * 0.5), four times."""

    h = x
    for _ in range(4):
        h = tf.nn.relu(tf.multiply(tf.add(h, 1.0), 0.5))
    return h


def _input() -> Any:
    return tf.constant(np.random.default_rng(0).standard_normal(64).astype("float32"))


def _set_flags(trace: Any, declined: Any) -> None:
    """Replace the trace's captures with copies whose ``payload_declined`` is ``declined(op)``."""

    ops = {str(op._label_raw): op for op in trace._layer_list}
    trace._tf_op_captures = tuple(
        replace(capture, payload_declined=declined(ops.get(capture.label_raw)))
        for capture in trace._tf_op_captures
    )


def _sparse(selector: Any) -> Any:
    """Trace the chain with ``save=selector``; the capture records which payloads it declined."""

    return tl.trace(_chain, _input(), backend="tf", save=selector)


def _status(trace: Any) -> Any:
    from torchlens.backends.tf.validation import validate_tf_trace_status

    return validate_tf_trace_status(trace, validate_metadata=False)


def _by_decision(status: Any, decision: str) -> list[Any]:
    return [item for item in status.decisions if item["decision"] == decision]


@_needs_tf
@pytest.mark.tf_backend
@pytest.mark.parametrize(
    "selector",
    [
        pytest.param(lambda: tl.func("Relu") | tl.func("Mul"), id="relu-mul"),
        pytest.param(lambda: tl.func("Relu") | tl.func("AddV2"), id="relu-add"),
        pytest.param(lambda: tl.func("Mul"), id="mul"),
    ],
)
def test_selector_declined_payloads_are_exempted_not_failed(selector: Any) -> None:
    trace = _sparse(selector())
    status = _status(trace)

    assert not _by_decision(status, "failed"), status.decisions
    assert status.state != "failed"
    exempted = [item for item in _by_decision(status, "exempted") if item["reason"] == _NOT_SAVED]
    assert exempted
    declined = {str(op.label) for op in trace._layer_list if replay_payload(op) is None}
    for item in exempted:
        assert "save selector declined" in item["justification"]
    assert {item["op_label"] for item in exempted} >= declined & {
        str(op.label) for op in trace._layer_list if op.func_name in {"Relu", "Mul", "AddV2"}
    }


@_needs_tf
@pytest.mark.tf_backend
def test_an_op_with_every_payload_saved_still_replays() -> None:
    """save=Relu|Mul: a Relu whose Mul parent was saved replays and validates."""

    status = _status(_sparse(tl.func("Relu") | tl.func("Mul")))

    validated = {item["func_name"] for item in _by_decision(status, "validated")}
    assert "Relu" in validated


@_needs_tf
@pytest.mark.tf_backend
def test_absent_payload_without_selector_proof_still_fails() -> None:
    """With no declined flag and no eager constant proof, a missing payload still fails.

    The flags are cleared and the trace is marked as a FuncGraph capture, whose
    ``has_saved_activation`` records payload presence rather than the selector.
    """

    trace = tl.trace(_chain, _input(), backend="tf", save=tl.func("Mul"))
    _set_flags(trace, lambda op: False)
    trace._tf_static_region_labels = ()
    status = _status(trace)

    failed = _by_decision(status, "failed")
    assert failed
    assert {item["reason"] for item in failed} == {"replay_mismatch"}
    assert not [item for item in status.decisions if item["reason"] == _NOT_SAVED]


@_needs_tf
@pytest.mark.tf_backend
def test_fully_saved_trace_has_no_exemption() -> None:
    trace = tl.trace(_chain, _input(), backend="tf")
    status = _status(trace)

    assert status.state == "passed"
    assert not [item for item in status.decisions if item["reason"] == _NOT_SAVED]


# Framework-free checks of the rule on hand-built records.


class _EagerTrace:
    """A trace stand-in without the FuncGraph marker."""


class _StaticTrace:
    _tf_static_region_labels = ()


class _Op:
    def __init__(self, label: str, out: Any, func_name: str = "Mul") -> None:
        self.label = label
        self._label_raw = label
        self.func_name = func_name
        self.out = out
        self.has_saved_activation = out is not None


def _capture(label: str, parents: tuple[str, ...], declined: bool) -> Any:
    inputs = tuple(
        SimpleNamespace(input_index=index, producer_label_raw=parent)
        for index, parent in enumerate(parents)
    )
    return SimpleNamespace(label_raw=label, op_type="Mul", inputs=inputs, payload_declined=declined)


@pytest.mark.parametrize(
    ("own", "parent", "expected"),
    [
        pytest.param((None, True), (np.ones(2), False), ["child"], id="own-declined"),
        pytest.param((np.ones(2), False), (None, True), ["parent"], id="parent-declined"),
        pytest.param((None, True), (None, True), ["child", "parent"], id="both-declined"),
        pytest.param((None, False), (np.ones(2), False), None, id="own-absent-not-declined"),
        pytest.param((np.ones(2), False), (None, False), None, id="parent-absent-not-declined"),
        pytest.param((np.ones(2), True), (np.ones(2), True), None, id="flag-but-saved"),
    ],
)
def test_not_saved_rule(own: tuple[Any, bool], parent: tuple[Any, bool], expected: Any) -> None:
    child_op, parent_op = _Op("child", own[0]), _Op("parent", parent[0])
    child = _capture("child", ("parent",), own[1])
    captures = {"child": child, "parent": _capture("parent", (), parent[1])}
    ops = {"child": child_op, "parent": parent_op}

    decision = _replay_exemptions.not_saved_decision(
        _EagerTrace(), child_op, child, (ops, captures)
    )

    if expected is None:
        assert decision is None
        return
    assert decision is not None
    assert (decision["decision"], decision["reason"]) == ("exempted", _NOT_SAVED)
    assert decision["justification"].startswith(
        f"save selector declined the replay payload of {', '.join(expected)} "
    )


def test_unreplayable_op_is_never_exempted() -> None:
    op = _Op("child", None)
    capture = _capture("child", (), True)

    decision = _replay_exemptions.early_decision(
        SimpleNamespace(), op, "value", frozenset({"Relu"}), {"child": capture}, {"child": op}
    )

    assert decision is None


def _constant_case(trace: Any, saved_flag: bool) -> Any:
    """A child with a saved producer and a payload-less constant source parent."""

    child_op, producer, constant = (
        _Op("child", np.ones(2)),
        _Op("parent", np.ones(2)),
        _Op("const", None, func_name="constant"),
    )
    constant.has_saved_activation = saved_flag
    child = _capture("child", ("parent",), False)
    child.inputs += (
        SimpleNamespace(input_index=1, producer_label_raw=None, source_label_raw="const"),
    )
    captures = {"child": child, "parent": _capture("parent", (), False)}
    ops = {"child": child_op, "parent": producer, "const": constant}
    return _replay_exemptions.not_saved_decision(trace, child_op, child, (ops, captures))


def test_eager_declined_constant_parent_is_exempted() -> None:
    decision = _constant_case(_EagerTrace(), saved_flag=False)

    assert decision is not None
    assert "declined the replay payload of const " in decision["justification"]


@pytest.mark.parametrize(
    ("trace", "saved_flag"),
    [
        pytest.param(_StaticTrace(), False, id="static-trace"),
        pytest.param(_EagerTrace(), True, id="selected-but-absent"),
    ],
)
def test_constant_parent_without_selector_proof_is_not_exempted(
    trace: Any, saved_flag: bool
) -> None:
    assert _constant_case(trace, saved_flag) is None
