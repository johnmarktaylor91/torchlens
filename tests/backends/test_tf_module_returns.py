"""TensorFlow module returns on a raw ``tf.Module`` tree (the Keras cases live in the cell).

``tests/test_module_return_cell.py`` checks every recorded-return case on Keras layers. A raw
``tf.Module`` child reaches the module stack by a different path (its concrete class
``__call__`` is patched rather than Keras ``Layer.__call__``), so it is checked here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import tensorflow_backend_modules

import torchlens as tl

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
    pytest.mark.rare,
]


class Affine(tf.Module):
    """Raw module computing ``x @ w + b``."""

    def __init__(self) -> None:
        """Create the variables."""

        super().__init__(name="affine")
        self.w = tf.Variable([[2.0], [3.0]], name="w")
        self.b = tf.Variable([1.0], name="b")

    def __call__(self, x: Any) -> Any:
        """Return ``x @ w + b``."""

        return tf.matmul(x, self.w) + self.b


class Root(tf.Module):
    """Raw root calling one child."""

    def __init__(self) -> None:
        """Create the child."""

        super().__init__(name="root")
        self.affine = Affine()

    def __call__(self, x: Any) -> Any:
        """Return ``relu(affine(x))``."""

        return tf.nn.relu(self.affine(x))


def test_raw_tf_module_records_returns() -> None:
    """A raw ``tf.Module`` child with its own ``__call__`` records its return."""

    root = Root()
    x = tf.ones((1, 2))
    trace = tl.trace(root, x, backend="tf")
    np.testing.assert_allclose(np.asarray(trace.modules["affine"].out), root.affine(x).numpy())
    assert trace.module_calls["affine:1"].output_paths == ((),)
