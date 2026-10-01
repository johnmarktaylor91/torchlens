"""TensorFlow backend derived-gradient preview tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import tensorflow_backend_modules

import torchlens as tl
from torchlens.backends import BackendUnsupportedError
from torchlens.backends.tf import GradOptions
from torchlens.backends.tf.backend import TFBackend
from torchlens.backends.tf.derived_grads import (
    TFIntermediateSignature,
    _tf_trace_intermediate_signatures,
)

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()


pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
]


def _assert_close(actual: Any, expected: Any) -> None:
    """Assert two TensorFlow tensors are numerically close.

    Parameters
    ----------
    actual
        Actual TensorFlow tensor.
    expected
        Expected TensorFlow tensor or array.
    """

    assert np.allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)


def _loss(output: Any) -> Any:
    """Return scalar sum loss.

    Parameters
    ----------
    output
        Model output.

    Returns
    -------
    Any
        Scalar TensorFlow loss.
    """

    return tf.reduce_sum(output)


class DenseRelu(keras.Model):
    """Deterministic dense-relu fixture."""

    def __init__(self) -> None:
        """Initialize deterministic parameters."""

        super().__init__(name="m")
        self.dense = keras.layers.Dense(2, activation="relu", name="dense")
        self.dense.build((None, 3))
        self.dense.kernel.assign(
            tf.constant([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype="float32")
        )
        self.dense.bias.assign(tf.constant([0.05, -0.1], dtype="float32"))

    def call(self, x: Any) -> Any:
        """Run ``dense -> relu``.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Any
            Model output.
        """

        return self.dense(x)


def _fixture_input() -> Any:
    """Return the deterministic capture input.

    Returns
    -------
    Any
        Float32 input tensor.
    """

    return tf.constant([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype="float32")


def _native_variable(variable: Any) -> Any:
    """Return the TensorFlow resource variable behind a Keras variable.

    Parameters
    ----------
    variable
        Keras 3 variable or raw ``tf.Variable``.

    Returns
    -------
    Any
        Watchable TensorFlow variable.
    """

    inner = getattr(variable, "value", None)
    return inner if inner is not None and not callable(inner) else variable


def test_tf_leaf_input_and_param_grads_match_direct_reference() -> None:
    """Leaf input and parameter derived grads should match a direct tape oracle."""

    model = DenseRelu()
    x = _fixture_input()
    model(x)
    trace = tl.trace(model, x, backend="tf", grad_options=GradOptions(loss_fn=_loss))

    kernel = _native_variable(model.dense.kernel)
    bias = _native_variable(model.dense.bias)
    with tf.GradientTape() as tape:
        tape.watch(x)
        expected = tape.gradient(_loss(model(x)), [x, kernel, bias])

    kernel_address = next(key for key in trace.params.keys() if key.endswith("kernel"))
    bias_address = next(key for key in trace.params.keys() if key.endswith("bias"))
    assert set(trace.derived_grads.keys()) == {
        "inputs.0",
        f"params.{kernel_address}",
        f"params.{bias_address}",
    }
    _assert_close(trace.derived_grads["inputs.0"].grad, expected[0])
    _assert_close(trace.derived_grads[f"params.{kernel_address}"].grad, expected[1])
    _assert_close(trace.derived_grads[f"params.{bias_address}"].grad, expected[2])
    assert trace.params[kernel_address].grad is trace.derived_grads[f"params.{kernel_address}"].grad
    record = trace.derived_grads["inputs.0"]
    assert record.provenance["backend"] == "tf"
    assert record.provenance["mechanism"] == "tf.GradientTape"


def test_tf_derived_grads_do_not_create_backward_logs() -> None:
    """Derived gradients should not masquerade as captured backward-pass logs."""

    model = DenseRelu()
    x = _fixture_input()
    model(x)
    trace = tl.trace(model, x, backend="tf", grad_options=GradOptions(loss_fn=_loss))

    assert trace.has_backward_pass is False
    assert not trace.backward_pass_logs
    assert not trace.grad_fn_logs
    with pytest.raises(ValueError, match="derived_grads"):
        trace.backward_passes


def test_tf_intermediate_grads_match_direct_oracle_and_skip_disconnected() -> None:
    """Intermediate grads should match direct AD values and skip disconnected ops."""

    def model(x: Any) -> Any:
        """Run a reachable two-op path plus one gradient-disconnected op."""

        hidden = x * 2.0
        blocked = tf.stop_gradient(hidden) * 3.0
        return tf.reduce_sum(tf.nn.relu(hidden)) + 0.0 * tf.reduce_sum(blocked)

    x = tf.ones((2, 2), dtype="float32")
    trace = tl.trace(
        model,
        x,
        backend="tf",
        grad_options=GradOptions(intermediate_grads=True, max_intermediate_grads=16),
    )

    records = trace.intermediate_derived_grads
    relu_op = next(op for op in trace.layer_list if op.func_name == "Relu")
    _assert_close(records[relu_op.label].grad, np.ones((2, 2), dtype="float32"))
    stop_gradient_op = next(op for op in trace.layer_list if op.func_name == "StopGradient")
    assert stop_gradient_op.label not in records
    assert records[relu_op.label].provenance["status"] == "exact"


def test_tf_same_shape_relus_are_disambiguated_by_stream_position() -> None:
    """Two same-shape relus should receive their own cotangents."""

    def model(x: Any) -> Any:
        """Run two same-shape relus with different downstream weights."""

        first = tf.nn.relu(x)
        second = tf.nn.relu(x + 1.0)
        return tf.reduce_sum(first * 2.0 + second * 3.0)

    trace = tl.trace(
        model,
        tf.ones((2, 2), dtype="float32"),
        backend="tf",
        grad_options=GradOptions(intermediate_grads=True, max_intermediate_grads=32),
    )

    relu_ops = [op for op in trace.layer_list if op.func_name == "Relu"]
    assert len(relu_ops) == 2
    _assert_close(
        trace.intermediate_derived_grads[relu_ops[0].label].grad,
        np.full((2, 2), 2.0, dtype="float32"),
    )
    _assert_close(
        trace.intermediate_derived_grads[relu_ops[1].label].grad,
        np.full((2, 2), 3.0, dtype="float32"),
    )


def test_tf_duplicate_trace_signature_group_is_ambiguous() -> None:
    """A duplicate signature group should be detectable and skipped by attach logic."""

    def model(x: Any) -> Any:
        """Run two relus."""

        return tf.nn.relu(x) + tf.nn.relu(x)

    trace = tl.trace(model, tf.ones((2, 2), dtype="float32"), backend="tf")
    relu_ops = [op for op in trace.layer_list if op.func_name == "Relu"]
    relu_ops[1].func_call_id = relu_ops[0].func_call_id
    relu_ops[1].parents = tuple(relu_ops[0].parents)
    groups = _tf_trace_intermediate_signatures(trace)
    # Torch parity: ``op.parents`` holds FINAL labels (every op's edges are
    # relabeled, not only multi-pass group members) -- the pass-qualified
    # ``op.label`` for a multi-pass referenced layer, but the BARE
    # ``op.layer_label`` for a single-pass one. ``_tf_trace_intermediate_signatures``
    # builds its signature in RAW label space (replay-side signatures match
    # capture input records) and resolves parents back to raw via a
    # label->raw map first, keyed by BOTH spellings (see its own
    # ``final_to_raw`` construction). The expected signature here must go
    # through the identical dual-key resolution, or it never matches the
    # production grouping.
    label_to_raw: dict[str, str] = {}
    for op in trace.layer_list:
        label_to_raw[op.label] = op._label_raw
        if op.num_passes == 1:
            label_to_raw[op.layer_label] = op._label_raw
    signature = TFIntermediateSignature(
        func_call_id=relu_ops[0].func_call_id,
        op_name=relu_ops[0].func_name,
        parent_labels=tuple(
            dict.fromkeys(label_to_raw.get(parent, parent) for parent in relu_ops[0].parents)
        ),
        module_stack=tuple(relu_ops[0].modules),
    )

    assert len(groups[signature]) == 2


def test_tf_max_intermediate_grads_cap_raises() -> None:
    """Intermediate cap should raise when exact attached records exceed it."""

    with pytest.raises(BackendUnsupportedError, match="capped"):
        tl.trace(
            lambda x: tf.reduce_sum(tf.nn.relu(x + 1.0)),
            tf.ones((2, 2), dtype="float32"),
            backend="tf",
            grad_options=GradOptions(intermediate_grads=True, max_intermediate_grads=1),
        )


def test_tf_non_scalar_output_without_loss_fn_raises() -> None:
    """Non-scalar raw output should require ``loss_fn``."""

    with pytest.raises(ValueError, match="scalar"):
        tl.trace(
            lambda x: x + 1.0,
            tf.ones((2, 2), dtype="float32"),
            backend="tf",
            grad_options=GradOptions(),
        )


def test_tf_replay_output_divergence_refuses_grads() -> None:
    """Divergent AD replay output should refuse derived gradients."""

    class Diverges:
        """Callable that changes output between capture and replay."""

        def __init__(self) -> None:
            """Initialize call counter."""

            self.calls = 0

        def __call__(self, x: Any) -> Any:
            """Return a call-count-dependent scalar."""

            self.calls += 1
            return tf.reduce_sum(x + float(self.calls))

    with pytest.raises(ValueError, match="diverged"):
        tl.trace(
            Diverges(),
            tf.ones((2, 2), dtype="float32"),
            backend="tf",
            grad_options=GradOptions(),
        )


def test_tf_graph_only_capture_refuses_grad_options() -> None:
    """Static FuncGraph capture cannot run the GradientTape replay."""

    compiled = tf.function(lambda x: tf.reduce_sum(x * 2.0))

    with pytest.raises(BackendUnsupportedError, match="eager live capture"):
        tl.trace(
            compiled,
            tf.ones((2, 2), dtype="float32"),
            backend="tf",
            grad_options=GradOptions(),
        )


def test_tf_grad_options_type_is_validated() -> None:
    """Foreign grad_options values should refuse typed."""

    with pytest.raises(BackendUnsupportedError, match="GradOptions"):
        tl.trace(
            lambda x: tf.reduce_sum(x),
            tf.ones((2, 2), dtype="float32"),
            backend="tf",
            grad_options={"loss_fn": None},
        )


def test_tf_validation_outcome_unchanged_by_grad_options() -> None:
    """The derived-gradient replay must not change the validation verdict."""

    def model(x: Any) -> Any:
        """Run a pure-op forward."""

        return tf.reduce_sum(tf.nn.relu(x * 2.0))

    x = tf.constant([[1.0, -2.0], [0.5, 0.3]], dtype="float32")
    baseline = tl.trace(model, x, backend="tf")
    baseline_result = TFBackend().validate_trace(baseline, validate_metadata=False)
    augmented = tl.trace(
        model,
        x,
        backend="tf",
        grad_options=GradOptions(intermediate_grads=True),
    )
    augmented_result = TFBackend().validate_trace(augmented, validate_metadata=False)

    assert type(augmented_result) is type(baseline_result)
    if hasattr(baseline_result, "state"):
        assert augmented_result.state == baseline_result.state
    else:
        assert augmented_result == baseline_result


def test_tf_module_tree_call_counts_restored_after_replay() -> None:
    """The replay must not leak call counts into the captured module tree."""

    model = DenseRelu()
    x = _fixture_input()
    model(x)
    baseline = tl.trace(model, x, backend="tf")
    augmented = tl.trace(model, x, backend="tf", grad_options=GradOptions(loss_fn=_loss))

    baseline_calls = {call.call_label for call in baseline.module_calls}
    augmented_calls = {call.call_label for call in augmented.module_calls}
    assert baseline_calls == augmented_calls
