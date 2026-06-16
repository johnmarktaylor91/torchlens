"""TensorFlow eager capture tests."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

import torchlens as tl

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")


pytestmark = pytest.mark.tf_backend


class SmallCnn(keras.Model):
    """Small deterministic Keras CNN."""

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


class RawModule(tf.Module):
    """Raw TensorFlow module with a concrete subclass ``__call__``."""

    def __init__(self) -> None:
        """Initialize raw variables."""

        super().__init__(name="raw_module")
        self.weight = tf.Variable([[2.0], [3.0]], name="weight")
        self.bias = tf.Variable([1.0], name="bias")

    def __call__(self, x: Any) -> Any:
        """Run a raw module forward."""

        return tf.nn.relu(tf.matmul(x, self.weight) + self.bias)


def test_tf_capture_hand_built_op_chain_edges_and_saved_values() -> None:
    """Capture a bare eager op chain with parent edges and saved outputs."""

    def chain(x: Any) -> Any:
        """Return a deterministic TensorFlow op chain."""

        y = x + tf.constant([1.0, 2.0])
        return y * tf.constant([3.0, 4.0])

    trace = tl.trace(chain, tf.constant([2.0, 3.0]), backend="tf")
    by_func = {op.func_name: op for op in trace.layer_list}

    assert trace.backend == "tf"
    assert {"AddV2", "Mul"} <= {op.func_name for op in trace.layer_list}
    assert by_func["Mul"].parents == [by_func["AddV2"]._label_raw]
    assert np.allclose(trace[by_func["Mul"].label].out, np.array([9.0, 20.0], dtype=np.float32))
    assert np.isfinite(trace[by_func["AddV2"].label].out).all()


def test_tf_capture_small_keras_cnn_modules_params_and_warm_boundary() -> None:
    """Capture a Keras CNN with module frames, params, and no init contamination."""

    tf.random.set_seed(1)
    model = SmallCnn()
    trace = tl.trace(model, tf.ones((1, 8, 8, 1)), backend="tf")
    op_types = {op.func_name for op in trace.layer_list}
    conv = next(op for op in trace.layer_list if op.func_name == "Conv2D")
    dense = next(op for op in trace.layer_list if op.func_name == "MatMul")

    assert trace.backend == "tf"
    assert {"Conv2D", "MatMul", "MaxPool", "Relu"} <= op_types
    assert conv.modules[-1].startswith("conv:")
    assert dense.modules[-1].startswith("dense:")
    assert len(trace.params) == 4
    assert getattr(trace, "_tf_init_op_labels", ()) == ()
    assert trace[conv.label].out is not None
    assert np.isfinite(trace[dense.label].out).all()


def test_tf_capture_small_transformer_nested_mha_frames() -> None:
    """Capture a Keras attention block with nested MHA projection frames."""

    tf.random.set_seed(2)
    model = SmallTransformer()
    trace = tl.trace(model, tf.ones((1, 4, 8)), backend="tf")
    op_types = {op.func_name for op in trace.layer_list}
    projection_ops = [
        op for op in trace.layer_list if any("mha.query:" in module for module in op.modules)
    ]
    attention_ops = [op for op in trace.layer_list if op.func_name == "BatchMatMulV2"]

    assert {"Einsum", "BatchMatMulV2", "Softmax", "Relu"} <= op_types
    assert projection_ops
    assert attention_ops
    assert any("mha:" in module for module in attention_ops[0].modules)
    assert getattr(trace, "_tf_init_op_labels", ()) == ()
    assert np.isfinite(projection_ops[0].out).all()


def test_tf_capture_raw_tf_module_class_level_stack() -> None:
    """Capture a raw tf.Module through concrete subclass class-level patching."""

    module = RawModule()
    trace = tl.trace(module, tf.ones((1, 2)), backend="tf")
    op_types = {op.func_name for op in trace.layer_list}
    matmul = next(op for op in trace.layer_list if op.func_name == "MatMul")

    assert {"ReadVariableOp", "MatMul", "AddV2", "Relu"} <= op_types
    assert trace.module_identity_mode == "object_module"
    assert matmul.modules == ["self:1"]
    assert getattr(trace, "_tf_init_op_labels", ()) == ()
    assert np.allclose(trace[matmul.label].out, np.array([[5.0]], dtype=np.float32))
