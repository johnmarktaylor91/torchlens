"""TensorFlow static FuncGraph fallback tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from conftest import tensorflow_backend_modules
from torchlens.backends.tf import TFBackend

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()


pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
]


class StaticDenseModel(keras.Model):
    """Small Keras model used for static graph capture."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__(name="static_dense_model")
        self.dense = keras.layers.Dense(
            2,
            activation="relu",
            kernel_initializer=keras.initializers.Constant([[1.0, -1.0], [2.0, 0.5]]),
            bias_initializer=keras.initializers.Constant([0.25, -0.5]),
            name="dense",
        )

    def call(self, x: Any) -> Any:
        """Run the dense model."""

        return self.dense(x)


class CompiledDenseModel(keras.Model):
    """Keras model whose call is compiled with ``tf.function``."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__(name="compiled_dense_model")
        self.dense = keras.layers.Dense(
            2,
            kernel_initializer=keras.initializers.Constant([[1.0, 2.0], [3.0, 4.0]]),
            bias_initializer="zeros",
            name="dense",
        )

    @tf.function
    def call(self, x: Any) -> Any:
        """Run a compiled dense graph."""

        return tf.nn.relu(self.dense(x) + tf.constant([1.0, -2.0]))


class ControlFlowModel(tf.Module):
    """Small graph with TensorFlow control flow."""

    @tf.function
    def __call__(self, x: Any) -> Any:
        """Run a data-dependent TensorFlow branch."""

        return tf.cond(
            tf.reduce_sum(x) > 0.0,
            lambda: x * tf.constant([2.0, 2.0]),
            lambda: x - tf.constant([3.0, 3.0]),
        )


def test_tf_static_captures_loaded_saved_model_structure_values_and_modules(
    tmp_path: Path,
) -> None:
    """Capture a loaded SavedModel through FuncGraph walk and prune fetches."""

    model = StaticDenseModel()
    x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    model(x)
    save_dir = tmp_path / "saved_model"
    tf.saved_model.save(model, str(save_dir))
    loaded = tf.saved_model.load(str(save_dir))

    trace = tl.trace(loaded, x, backend="tf", save=tl.func("relu"))
    op_types = {op.func_name for op in trace.layer_list}
    assert trace.backend == "tf"
    assert trace.module_identity_mode == "function_root"
    relu_ops = [op for op in trace.layer_list if op.func_name == "Relu"]
    if relu_ops:
        relu = relu_ops[0]
        matmul = next(op for op in trace.layer_list if op.func_name == "MatMul")
        assert {"MatMul", "BiasAdd", "Relu"} <= op_types
        assert matmul.parents
        assert relu.parents
        assert relu.out is not None
        assert np.allclose(relu.out, np.array([[5.25, 0.0]], dtype=np.float32))
        assert all(
            op.out is None
            for op in trace.layer_list
            if not op.is_input and not op.is_output and op.func_name not in {"Relu"}
        )
    else:
        regions = [op for op in trace.layer_list if str(op.func_name).startswith("region:")]
        assert regions
        assert all(not region.has_saved_activation for region in regions)


def test_tf_static_captures_tf_function_compiled_model() -> None:
    """Route ``@tf.function`` Keras call to static capture instead of raising."""

    model = CompiledDenseModel()
    x = tf.constant([[1.0, 1.0]], dtype=tf.float32)
    model(x)

    trace = tl.trace(model, x, backend="tf")
    op_types = {op.func_name for op in trace.layer_list}
    relu = next(op for op in trace.layer_list if op.func_name == "Relu")

    assert trace.backend == "tf"
    assert {"MatMul", "AddV2", "Relu"} <= op_types
    assert relu.parents
    assert np.allclose(relu.out, np.array([[5.0, 4.0]], dtype=np.float32))


def test_tf_static_control_flow_regions_are_unverified() -> None:
    """Represent static TensorFlow control flow as unverified regions."""

    trace = tl.trace(ControlFlowModel(), tf.constant([1.0, 2.0]), backend="tf")
    regions = [op for op in trace.layer_list if str(op.func_name).startswith("region:")]
    status = TFBackend().validate_trace(trace)

    assert trace.backend == "tf"
    assert regions
    assert all(not region.has_saved_activation for region in regions)
    assert getattr(trace.validation_replay_status, "state") == "unverified", getattr(
        trace,
        "_tf_validation_result",
        None,
    )
    assert getattr(trace.validation_replay_status, "effect_region_node_count") >= len(regions)
    assert status is trace.validation_replay_status
