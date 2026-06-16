"""TensorFlow backend validation tripwire tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

import torchlens as tl
from conftest import tensorflow_backend_modules
from torchlens.backends.tf import TFBackend
from torchlens.backends.tf.validation import replay_allowlist
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
    """Identity is classified as a label-preserving annotation."""

    def identity(x: Any) -> Any:
        """Return an eager TensorFlow identity."""

        return tf.identity(x)

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
