"""TensorFlow backend validation tripwire tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from conftest import tensorflow_backend_modules

import torchlens as tl
from torchlens.backends.tf import TFBackend
from torchlens.backends.tf.validation import replay_allowlist
from torchlens.validation.invariants import check_metadata_invariants
from torchlens.validation.status import ValidationReplayStatus

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()


pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
]


class SmallCnn(keras.Model):
    """Small deterministic Keras CNN for validation coverage."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__(name="small_cnn")
        self.conv = keras.layers.Conv2D(2, 3, padding="same", activation="relu", name="conv")
        self.pool = keras.layers.MaxPool2D(name="pool")
        self.flat = keras.layers.Flatten(name="flat")
        self.dense = keras.layers.Dense(3, name="dense")

    def call(self, x: Any) -> Any:
        """Run the CNN forward."""

        x = self.conv(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.dense(x)


class SmallTransformer(keras.Model):
    """Small deterministic Transformer-style encoder block."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__(name="small_transformer")
        self.mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, name="mha")
        self.norm = keras.layers.LayerNormalization(name="norm")
        self.ffn = keras.layers.Dense(8, activation="relu", name="ffn")

    def call(self, x: Any) -> Any:
        """Run a compact encoder-style block."""

        y = self.mha(x, x)
        return self.ffn(self.norm(x + y))


class PollutingModule(tf.Module):
    """Module that creates a variable during the captured forward."""

    def __init__(self) -> None:
        """Initialize call counter."""

        super().__init__(name="polluting_module")
        self.calls = 0

    def __call__(self, x: Any) -> Any:
        """Create late state on the second call and use it."""

        self.calls += 1
        if self.calls >= 2 and not hasattr(self, "late"):
            self.late = tf.Variable(tf.ones_like(x), name="late")
        if hasattr(self, "late"):
            return x + self.late
        return x + 1.0


def _validate(trace: Any) -> bool | ValidationReplayStatus:
    """Validate a trace through the TensorFlow backend.

    Parameters
    ----------
    trace
        TensorFlow trace.

    Returns
    -------
    bool | ValidationReplayStatus
        Backend validation result.
    """

    return TFBackend().validate_trace(trace, validate_metadata=False)


def _failures(trace: Any) -> tuple[str, ...]:
    """Return TensorFlow validation failure reasons.

    Parameters
    ----------
    trace
        TensorFlow trace already validated.

    Returns
    -------
    tuple[str, ...]
        Failure reason strings.
    """

    return tuple(getattr(trace, "_tf_validation_result").failures)


def test_tf_validation_fails_when_interior_callback_record_is_dropped() -> None:
    """Dropping an interior op capture leaves a consumed producer unproven."""

    def chain(x: Any) -> Any:
        """Return a small op chain."""

        return (x + tf.constant([1.0, 2.0])) * tf.constant([3.0, 4.0])

    trace = tl.trace(chain, tf.constant([2.0, 3.0]), backend="tf")
    dropped = next(capture for capture in trace._tf_op_captures if capture.op_type == "AddV2")
    trace._tf_op_captures = tuple(
        capture for capture in trace._tf_op_captures if capture is not dropped
    )

    assert _validate(trace) is False
    assert any("missing_capture" in failure for failure in _failures(trace))


def test_tf_validation_fails_on_initializer_contamination() -> None:
    """Late variable creation during captured forward trips validation."""

    trace = tl.trace(PollutingModule(), tf.ones((2,), dtype=tf.float32), backend="tf")

    assert getattr(trace, "_tf_init_op_labels", ())
    assert _validate(trace) is False
    assert any("initializer_contamination" in failure for failure in _failures(trace))


def test_tf_validation_identity_annotation_passes_legitimate_passthrough() -> None:
    """Identity is classified as a label-preserving annotation.

    The graph needs at least one genuinely replayable (non-exempt) node:
    a bare ``tf.identity`` call is the ONLY op, so with Identity exempted as
    an annotation ``replayed_node_count`` would be 0 and validation would
    legitimately refuse to call that "passed" (exemptions alone cannot
    produce a passing result -- see ``validation/status.py``'s
    ``no_nodes_replay_validated`` guard). Add a real op so the identity
    passthrough is checked alongside something replay actually verifies.
    """

    def identity(x: Any) -> Any:
        """Return an eager TensorFlow identity composed with a real op."""

        return tf.identity(x) + 0.0

    trace = tl.trace(identity, tf.constant([1.0, 2.0]), backend="tf")

    assert _validate(trace) is True
    assert trace.validation_replay_status.state == "passed"
    assert (
        getattr(trace, "_tf_validation_result").classes[
            next(op._label_raw for op in trace.layer_list if op.func_name == "Identity")
        ]
        == "annotation"
    )


def test_tf_validation_fails_on_mislabeled_equal_valued_parent() -> None:
    """Mislabeling an asymmetric op parent is caught by edge conservation."""

    def asymmetric(x: Any) -> Any:
        """Create two equal-valued parents and consume them asymmetrically."""

        left = x + tf.constant([0.0, 0.0])
        right = x + tf.constant([0.0, 0.0])
        return left - right

    trace = tl.trace(asymmetric, tf.constant([2.0, 5.0]), backend="tf")
    sub_capture = next(capture for capture in trace._tf_op_captures if capture.op_type == "Sub")
    first_parent = sub_capture.inputs[0].producer_label_raw
    mutated_inputs = (
        sub_capture.inputs[0],
        replace(sub_capture.inputs[1], producer_label_raw=first_parent),
    )
    trace._tf_op_captures = tuple(
        replace(capture, inputs=mutated_inputs) if capture is sub_capture else capture
        for capture in trace._tf_op_captures
    )

    assert _validate(trace) is False
    assert any("graph_parent_edges_not_conserved" in failure for failure in _failures(trace))


def test_tf_validation_reports_unverified_for_pure_allowlist_gap() -> None:
    """A classified pure op outside the replay allowlist is not a green pass."""

    def mostly_unverified(x: Any) -> Any:
        """Use a classified but non-allowlisted pure op."""

        return tf.math.exp(x)

    trace = tl.trace(mostly_unverified, tf.constant([1.0, 2.0]), backend="tf")
    result = _validate(trace)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    assert result.pure_unverified_node_count == 1
    assert result.replayed_node_count == 0


def test_tf_validation_fails_on_unclassified_op_type() -> None:
    """Unknown op types fail closed."""

    def chain(x: Any) -> Any:
        """Return one known TensorFlow value op."""

        return x + tf.constant([1.0, 2.0])

    trace = tl.trace(chain, tf.constant([1.0, 2.0]), backend="tf")
    add = next(op for op in trace.layer_list if op.func_name == "AddV2")
    add.func_name = "DefinitelyUnknownTfOp"

    assert _validate(trace) is False
    assert any("unclassified_op" in failure for failure in _failures(trace))


def test_tf_validation_positive_cnn_and_transformer_report_replay_coverage() -> None:
    """Clean warmed CNN and Transformer validate with honest replay counts."""

    assert "BatchMatMulV2" in replay_allowlist()
    assert "Einsum" in replay_allowlist()
    cases = (
        (SmallCnn, tf.ones((1, 8, 8, 1), dtype=tf.float32)),
        (SmallTransformer, tf.ones((1, 4, 8), dtype=tf.float32)),
    )
    for model_type, x in cases:
        tf.random.set_seed(11)
        trace = tl.trace(model_type(), x, backend="tf")
        result = _validate(trace)
        status = trace.validation_replay_status

        assert result is True or (
            isinstance(result, ValidationReplayStatus) and result.state == "unverified"
        )
        assert status.failed_node_count == 0
        assert status.replayed_node_count > 0
        assert getattr(trace, "_tf_validation_result").replayed_histogram


def test_tf_num_layers_with_params_populated_in_object_module_mode() -> None:
    """``_finish_trace`` must populate ``trace.num_layers_with_params``.

    Sibling-gap regression test for the Paddle backend's identical bug: TF's
    ``TFBackend._finish_trace`` (``torchlens/backends/tf/backend.py``) calls
    the shared ``finalize_single_pass_trace`` helper with
    ``attach_op_params=_attach_tf_op_params_for_finalize`` (so per-op
    ``_param_logs`` are attached correctly) but -- before this fix -- without
    ``count_layers_with_attached_params=True`` and without
    ``update_param_totals_from_layers=True``, so the trace-level summary
    counter ``trace.num_layers_with_params`` stayed at its dataclass default
    of ``0`` for every parameterized Keras model. That trips the
    ``[param_xrefs]`` metadata invariant
    (``_check_layer_param_aggregate_dedup``) for any object-module-mode
    trace with attached params, exactly like the Paddle backend's MAJOR
    finding. MLX already passes the flag at the identical call site; TF and
    Paddle must both match it.
    """

    trace = tl.trace(SmallCnn(), tf.ones((1, 8, 8, 1), dtype=tf.float32), backend="tf")

    assert trace.module_identity_mode == "object_module"
    assert trace.num_layers_with_params > 0
    assert check_metadata_invariants(trace) is True


def test_tf_public_validate_forward_scope_does_not_raise() -> None:
    """``tl.validate(scope="forward", backend="tf")`` validates instead of refusing.

    ``validate_metadata`` is a validation switch: forwarding it into the TF
    capture made every public forward-scope call raise
    ``BackendUnsupportedError``.
    """

    def chain(x: Any) -> Any:
        """Return a small replayable op chain."""

        return tf.nn.relu(x * tf.constant([2.0, -1.0]) + tf.constant([1.0, 1.0]))

    x = tf.constant([1.0, 3.0])
    assert tl.validate(chain, x, scope="forward", backend="tf") is True
    assert tl.validate(chain, x, scope="forward", backend="tf", validate_metadata=False) is True


def test_tf_public_validate_forward_scope_still_fails_closed() -> None:
    """The public entry still returns False for an unclassified op type."""

    def cumulative(x: Any) -> Any:
        """Use an op type the TF classifier does not know."""

        return tf.math.cumsum(x) + tf.constant([1.0, 1.0])

    with pytest.warns(Warning, match="tl.validate FAILED"):
        assert (
            tl.validate(cumulative, tf.constant([1.0, 3.0]), scope="forward", backend="tf") is False
        )


def _depthwise_relu6(x: Any, kernel: Any) -> Any:
    """Run pad, depthwise convolution, ``relu6`` (the MobileNet block shape).

    Parameters
    ----------
    x
        NHWC input.
    kernel
        Depthwise kernel.

    Returns
    -------
    Any
        Block output.
    """

    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    y = tf.nn.depthwise_conv2d(padded, kernel, strides=[1, 1, 1, 1], padding="VALID")
    return tf.nn.relu6(y * 4.0)


def _depthwise_inputs() -> tuple[Any, Any]:
    """Return deterministic depthwise-block inputs."""

    x = tf.reshape(tf.range(32, dtype=tf.float32) / 8.0 - 1.5, (1, 4, 4, 2))
    kernel = tf.reshape(tf.range(18, dtype=tf.float32) / 9.0 - 0.5, (3, 3, 2, 1))
    return x, kernel


def test_tf_validation_replays_depthwise_conv_and_relu6() -> None:
    """MobileNet's pad, depthwise conv and ``relu6`` replay instead of failing or skipping."""

    trace = tl.trace(_depthwise_relu6, _depthwise_inputs(), backend="tf")

    assert _validate(trace) is True
    replayed = getattr(trace, "_tf_validation_result").replayed_histogram
    assert replayed["DepthwiseConv2dNative"] == 1
    assert replayed["Relu6"] == 1
    assert replayed["Pad"] == 1


def test_tf_validation_depthwise_and_relu6_corruption_fails() -> None:
    """A corrupted pad, depthwise or ``relu6`` payload fails replay, never passes."""

    for op_type in ("Pad", "DepthwiseConv2dNative", "Relu6"):
        trace = tl.trace(_depthwise_relu6, _depthwise_inputs(), backend="tf")
        target = next(op for op in trace.layer_list if op.func_name == op_type)
        target.out = target.out + 0.5

        assert _validate(trace) is False
        assert any(op_type in failure for failure in _failures(trace))


def test_tf_function_root_trace_has_buffers_and_graph_shape_hash() -> None:
    """A function-root preview trace carries ``buffers`` and a graph hash."""

    def chain(x: Any) -> Any:
        """Return a small op chain."""

        return tf.nn.relu(x + tf.constant([1.0, 2.0]))

    def other(x: Any) -> Any:
        """Return a different op chain."""

        return tf.math.tanh(x + tf.constant([1.0, 2.0]))

    x = tf.constant([1.0, 3.0])
    first = tl.trace(chain, x, backend="tf")
    second = tl.trace(chain, x, backend="tf")
    different = tl.trace(other, x, backend="tf")

    assert first.module_identity_mode == "function_root"
    assert first.buffers is not None and len(first.buffers) == 0
    assert isinstance(first.graph_shape_hash, str)
    assert first.graph_shape_hash == second.graph_shape_hash
    assert different.graph_shape_hash != first.graph_shape_hash


class _DenseRelu(keras.Model):
    """Keras model whose forward reads two variables (kernel and bias)."""

    def __init__(self) -> None:
        """Build one dense layer."""

        super().__init__(name="dense_relu")
        self.dense = keras.layers.Dense(3, name="dense")

    def call(self, x: Any) -> Any:
        """Run dense then relu."""

        return tf.nn.relu(self.dense(x))


class _CountingModule(tf.Module):
    """Module that WRITES a variable during its forward."""

    def __init__(self) -> None:
        """Create the weight and the counter variables."""

        super().__init__(name="counting_module")
        self.weight = tf.Variable([2.0, 3.0], name="weight")
        self.count = tf.Variable(0.0, name="count")

    def __call__(self, x: Any) -> Any:
        """Read the weight, then bump the counter."""

        out = x * self.weight
        self.count.assign_add(1.0)
        return out


def _warm_dense() -> tuple[Any, Any]:
    """Return a built dense model and its input."""

    tf.random.set_seed(3)
    model = _DenseRelu()
    x = tf.reshape(tf.range(8, dtype=tf.float32) / 8.0, (2, 4))
    model(x)
    return model, x


def test_tf_validation_rereads_variables_when_no_op_can_write_them() -> None:
    """A Keras model with no variable writes validates: its reads are re-read and checked."""

    model, x = _warm_dense()
    trace = tl.trace(model, x, backend="tf")

    assert _validate(trace) is True
    status = trace.validation_replay_status
    assert status.effect_region_node_count == 0
    assert getattr(trace, "_tf_validation_result").replayed_histogram["ReadVariableOp"] == 2


def test_tf_validation_variable_changed_after_capture_fails() -> None:
    """A variable changed after capture makes the re-read disagree: it fails."""

    model, x = _warm_dense()
    trace = tl.trace(model, x, backend="tf")
    model.dense.bias.assign([7.0, 7.0, 7.0])

    assert _validate(trace) is False
    assert any("resource_read_replay_failed" in failure for failure in _failures(trace))


def test_tf_validation_corrupted_variable_read_fails() -> None:
    """A corrupted saved read payload fails the re-read check."""

    model, x = _warm_dense()
    trace = tl.trace(model, x, backend="tf")
    read = next(op for op in trace.layer_list if op.func_name == "ReadVariableOp")
    read.out = read.out + 1.0

    assert _validate(trace) is False


def test_tf_validation_reads_stay_unverified_when_the_forward_writes_a_variable() -> None:
    """A forward that writes a variable keeps every read an unverified effect region."""

    trace = tl.trace(_CountingModule(), tf.constant([1.0, 1.0]), backend="tf")
    result = _validate(trace)

    assert isinstance(result, ValidationReplayStatus)
    assert result.state == "unverified"
    histogram = getattr(trace, "_tf_validation_result").effect_region_histogram
    assert histogram["ReadVariableOp"] >= 1
    assert getattr(trace, "_tf_validation_result").replayed_histogram["ReadVariableOp"] == 0
