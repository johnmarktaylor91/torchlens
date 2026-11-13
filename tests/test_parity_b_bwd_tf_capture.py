"""Native backward capture on TensorFlow, checked against TensorFlow itself.

The fixture stands in for the two pieces other lanes own: the TensorFlow
capture's calls into ``backward_ready_scope`` / ``retain_backward_state`` and
the registry's ``backward_capture`` entry for ``tf``. Every recorded pass is
checked by the native backward validator (node, sum and leaf checks), its leaf
gradients against a plain ``tape.gradient``, and the forward against a run with
no tape.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends._backward_native import (  # noqa: E402
    native_backward_surface,
    tf as native_tf,
)
from torchlens.backends._replay_adapters import get_adapter  # noqa: E402
from torchlens.backends._reverse_engine import graph_from_recipes  # noqa: E402
from torchlens.backends.tf.op_callback_capture import TFEagerCaptureSession  # noqa: E402
from torchlens.errors import RunCapabilityUnavailableError  # noqa: E402
from torchlens.validation._backward_neutral import (  # noqa: E402
    BackwardCheckResult,
    RecordedBackward,
    check_backward_records,
    recorded_backward_from_trace,
)

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

ADAPTER = get_adapter("tf")
keras = tf.keras


class RawModule(tf.Module):
    """Raw TensorFlow module: two variables, matmul, add, relu."""

    def __init__(self) -> None:
        super().__init__(name="raw_module")
        self.weight = tf.Variable([[2.0, -1.0], [3.0, 0.5]], name="weight")
        self.bias = tf.Variable([1.0, -0.25], name="bias")

    def __call__(self, x: Any) -> Any:
        return tf.nn.relu(tf.matmul(x, self.weight) + self.bias)


def diamond(x: Any) -> Any:
    """One value used three times; the tape sums its gradients outside every rule."""

    a = tf.tanh(x)
    return tf.reduce_sum(a * a + tf.sin(a))


def _mlp() -> Any:
    tf.keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [keras.layers.Dense(4, activation="relu"), keras.layers.Dense(1)], name="mlp"
    )
    model(tf.zeros((3, 5)))
    return model


def _square_sum(output: Any) -> Any:
    return tf.reduce_sum(output * output)


@pytest.fixture
def backward_ready(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tl.Trace]:
    """Capture backward-ready the way the TensorFlow capture will once wired."""

    sessions: list[Any] = []
    original = TFEagerCaptureSession.run

    def run(self: Any) -> Any:
        with native_tf.backward_ready_scope(self):
            result = original(self)
        sessions.append(self)
        return result

    monkeypatch.setattr(TFEagerCaptureSession, "run", run)
    spec = registry.get_backend_spec("tf")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, backward_capture=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "backward_capture": lambda: native_backward_surface("tf"),
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "tf", replaced)

    def capture(model: Any, example: Any) -> tl.Trace:
        trace = tl.trace(model, example, backend="tf")
        native_tf.retain_backward_state(trace, sessions[-1])
        return trace

    return capture


def _captures_by_raw(trace: Any) -> dict[str, Any]:
    return {c.label_raw: c for c in trace._tf_op_captures}


def _leaf_tensor(trace: Any, op: Any) -> Any | None:
    """The live tensor of a graph leaf: a source, or a variable read (X4)."""

    state = trace.__dict__[native_tf._STATE_KEY]
    raw = getattr(op, "_label_raw", None) or getattr(op, "label_raw", None)
    if raw in state.sources:
        return state.sources[raw]
    capture = _captures_by_raw(trace).get(raw)
    if capture is not None and capture.op_type == "ReadVariableOp":
        return capture.output_tensor
    return None


def _is_leaf_op(trace: Any, op: Any) -> bool:
    raw = getattr(op, "_label_raw", None) or getattr(op, "label_raw", None)
    capture = _captures_by_raw(trace).get(raw)
    return capture is None or capture.op_type == "ReadVariableOp"


def check_pass(trace: Any, pass_number: int) -> BackwardCheckResult:
    """Run the native backward validator on one recorded pass."""

    state = trace.__dict__[native_tf._STATE_KEY]
    ops = list(trace._layer_list)
    by_label = {op.label: op for op in ops}
    graph = graph_from_recipes(
        trace,
        ADAPTER,
        [op for op in ops if not _is_leaf_op(trace, op)],
        lambda label: by_label[label].out,
    )
    produced = {slot for node in graph.nodes for slot in node.outputs}
    leaves = sorted({p for node in graph.nodes for p in node.parents if p not in produced})
    tensors = [_leaf_tensor(trace, by_label[label]) for label in leaves]
    known = [(label, t) for label, t in zip(leaves, tensors, strict=True) if t is not None]
    grads = state.tape.gradient(state.losses[pass_number], [t for _, t in known])
    leaf_grads = {
        label: grad.numpy()
        for (label, _), grad in zip(known, grads, strict=True)
        if grad is not None
    }
    return check_backward_records(trace, ADAPTER, graph, _recorded(trace, pass_number), leaf_grads)


def _recorded(trace: Any, pass_number: int) -> RecordedBackward:
    """The pass's records keyed by the labels the replay adapter's graph uses.

    ``recorded_backward_from_trace`` keys gradients by ``Op.layer_label``; the
    TensorFlow adapter's recipes name ops by ``Op.label`` (``"relu_1_5:1"``),
    so the gradients are re-read here under ``Op.label``. Same values, same
    checks: only the key differs.
    """

    calls = recorded_backward_from_trace(trace, pass_number).calls
    grads = {}
    for op in trace._layer_list:
        for record in op._slot("_grad_records") or ():
            if record.backward_pass_number == pass_number and record.grad is not None:
                grads[op.label] = record.grad
    return RecordedBackward(calls=calls, grads=grads)


def _labels(trace: Any, pass_number: int) -> dict[str, Any]:
    """What the validator keyed on, for a failure message."""

    recorded = _recorded(trace, pass_number)
    return {
        "calls": sorted(recorded.calls),
        "grads": sorted(recorded.grads),
        "grad_fn_labels": sorted(str(fn.op_label) for fn in trace._grad_fn_logs.values()),
        "ops": [
            (op.label, op.layer_label, getattr(op, "_label_raw", None)) for op in trace._layer_list
        ],
    }


def _assert_valid(trace: Any, pass_number: int) -> BackwardCheckResult:
    result = check_pass(trace, pass_number)
    findings = [dataclasses.astuple(f) for f in result.findings]
    assert result.passed, (findings, result.nodes_checked, _labels(trace, pass_number))
    assert result.leaves_checked > 0
    return result


def _param_grads(trace: Any, pass_number: int) -> dict[str, np.ndarray]:
    found = {}
    for address, param in trace._param_logs.items():
        for record in param._slot("_grad_records") or ():
            if record.backward_pass_number == pass_number and record.grad is not None:
                found[str(address)] = np.asarray(record.grad)
    return found


def test_raw_module_pass_validates_and_matches_a_plain_tape(backward_ready: Any) -> None:
    model = RawModule()
    x = tf.constant([[1.0, -2.0], [0.5, 1.5], [2.0, 0.25]])
    plain_out = model(x).numpy()
    trace = backward_ready(model, x)
    np.testing.assert_array_equal(trace.output.numpy(), plain_out)
    assert trace.backward(_square_sum, save_grads=True) is trace
    _assert_valid(trace, 1)
    with tf.GradientTape() as tape:
        loss = _square_sum(model(x))
    want = tape.gradient(loss, [model.weight, model.bias])
    got = _param_grads(trace, 1)
    assert len(got) == 2
    for expected, value in zip(want, sorted(got.items()), strict=True):
        np.testing.assert_allclose(value[1], expected.numpy(), rtol=0, atol=0)
    assert len(trace.autograd.backward_passes) == 1
    names = {grad_fn.class_name for grad_fn in trace.autograd.grad_fns}
    assert {"MatMul", "Relu"} <= names


def test_diamond_accumulates_and_validates(backward_ready: Any) -> None:
    x = tf.constant([0.3, -1.2, 2.0])
    trace = backward_ready(diamond, x)
    trace.backward(trace.output, save_grads=True)
    result = _assert_valid(trace, 1)
    assert result.sums_checked > 0
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = diamond(x)
    want = tape.gradient(loss, x).numpy()
    inputs = [op for op in trace._layer_list if getattr(op, "is_input", False)]
    assert len(inputs) == 1
    np.testing.assert_allclose(np.asarray(inputs[0].grad), want, rtol=0, atol=0)


def test_keras_mlp_two_passes_each_validate(backward_ready: Any) -> None:
    model = _mlp()
    x = tf.reshape(tf.range(15, dtype=tf.float32) / 7.0, (3, 5))
    plain_out = model(x).numpy()
    trace = backward_ready(model, x)
    np.testing.assert_array_equal(trace.output.numpy(), plain_out)
    trace.backward(_square_sum, save_grads=True)
    trace.backward(lambda out: tf.reduce_mean(out), save_grads=True)
    assert len(trace.autograd.backward_passes) == 2
    for pass_number in (1, 2):
        _assert_valid(trace, pass_number)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    want = tape.gradient(loss, model.trainable_variables)
    got = _param_grads(trace, 2)
    assert len(got) == len(want)
    by_shape = sorted(got.values(), key=lambda a: (a.shape, a.sum()))
    expected = sorted((w.numpy() for w in want), key=lambda a: (a.shape, a.sum()))
    for value, ref in zip(by_shape, expected, strict=True):
        np.testing.assert_allclose(value, ref, rtol=0, atol=0)


def test_metadata_only_pass_keeps_no_values(backward_ready: Any) -> None:
    trace = backward_ready(RawModule(), tf.constant([[1.0, 2.0]]))
    trace.backward(_square_sum, save_grads=False)
    calls = [call for fn in trace.autograd.grad_fns for call in fn.calls.values()]
    assert calls
    assert all(call.grad_inputs is None and call.grad_outputs is None for call in calls)


def test_untaped_trace_refuses_typed() -> None:
    trace = tl.trace(RawModule(), tf.constant([[1.0, 2.0]]), backend="tf")
    with pytest.raises(RunCapabilityUnavailableError, match="kept no gradient tape"):
        native_tf.record_backward(trace, _square_sum)


@pytest.mark.parametrize(
    ("loss", "kwargs", "match"),
    [
        (lambda out: out, {}, "must be a scalar"),
        (lambda out: 3.0, {}, "must be a TensorFlow tensor"),
        (_square_sum, {"retain_graph": True}, "no TensorFlow meaning"),
        (_square_sum, {"save_grads": ["relu"]}, "selects ops"),
    ],
)
def test_bad_backward_calls_refuse_typed(
    backward_ready: Any, loss: Any, kwargs: dict[str, Any], match: str
) -> None:
    trace = backward_ready(RawModule(), tf.constant([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(RunCapabilityUnavailableError, match=match):
        native_tf.record_backward(trace, loss, **kwargs)


def test_disconnected_loss_refuses_and_closes_the_pass(backward_ready: Any) -> None:
    trace = backward_ready(RawModule(), tf.constant([[1.0, 2.0]]))
    detached = tf.reduce_sum(tf.constant(trace.output.numpy()))
    with pytest.raises(RunCapabilityUnavailableError, match="not connected"):
        native_tf.record_backward(trace, detached)


def test_hooks_are_restored_after_a_pass(backward_ready: Any) -> None:
    from tensorflow.python.eager import backprop, record

    before = (backprop._gradient_function, record.record_operation)
    trace = backward_ready(RawModule(), tf.constant([[1.0, 2.0]]))
    trace.backward(_square_sum)
    assert (backprop._gradient_function, record.record_operation) == before
    with tf.GradientTape() as tape:
        v = tf.Variable(2.0)
        y = v * v
    assert float(tape.gradient(y, v)) == 4.0


def test_missing_binding_refuses_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorflow.python.eager import backprop

    monkeypatch.delattr(backprop, "_gradient_function")
    with pytest.raises(RunCapabilityUnavailableError, match="backprop._gradient_function"):
        native_tf._require_bindings()
