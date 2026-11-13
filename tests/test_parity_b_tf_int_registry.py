"""TensorFlow live-edit registry widening: variable reads and unary minus are editable.

Before this widening a live ``intervene=`` site matching a ``ReadVariableOp``
(a Keras layer's weight read) or a ``Neg`` from unary minus refused with the
typed unreachable error, because no writable wrapper presented those outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

import torchlens as tl  # noqa: E402
from torchlens.backends.tf import TFBackend  # noqa: E402

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

_KERNEL = np.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=np.float32)
_X = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=np.float32)


class _NoBias(keras.Model):
    """One bias-free dense layer: the kernel read is the only variable read."""

    def __init__(self) -> None:
        super().__init__(name="m")
        self.dense = keras.layers.Dense(2, use_bias=False, name="dense")
        self.dense.build((None, 3))
        self.dense.kernel.assign(_KERNEL)

    def call(self, x: Any) -> Any:
        return self.dense(x)


def _negate(x: Any) -> Any:
    return -x * 2.0


def _output(trace: Any) -> np.ndarray:
    outputs = [op for op in trace._layer_list if getattr(op, "is_output", False)]
    return np.asarray((outputs or list(trace._layer_list))[-1].out)


def _ops(trace: Any, name: str) -> list[Any]:
    return [op for op in trace._layer_list if str(op.func_name).lower() == name]


def test_a_keras_weight_read_is_editable_live() -> None:
    model, x = _NoBias(), tf.constant(_X)
    clean = tl.trace(model, x, backend="tf")
    assert _ops(clean, "readvariableop"), "the capture records the kernel read"
    edited = tl.trace(
        model, x, backend="tf", intervene=tl.when(tl.func("ReadVariableOp"), tl.scale(2.0))
    )
    np.testing.assert_allclose(_output(edited), _X @ (2.0 * _KERNEL), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_output(clean), _X @ _KERNEL, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(model.dense.kernel), _KERNEL)


def test_unary_minus_is_editable_live_and_validates() -> None:
    x, before = tf.constant(_X), tf.Tensor.__neg__
    clean = tl.trace(_negate, x, backend="tf")
    assert _ops(clean, "neg"), "the capture records unary minus as Neg"
    edited = tl.trace(_negate, x, backend="tf", intervene=tl.when(tl.func("Neg"), tl.scale(3.0)))
    np.testing.assert_allclose(_output(edited), -3.0 * _X * 2.0, rtol=1e-6, atol=1e-6)
    assert TFBackend().validate_trace(edited) is True
    assert tf.Tensor.__neg__ is before, "unary minus is restored after the capture"
