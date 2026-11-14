"""Backward capture reached through the registry on TensorFlow and Paddle.

``backward_ready=True`` dispatches on the real capture path, with no test-side spec
replacement or hook harness, and ``trace.backward`` records a pass whose parameter
gradients equal the framework's own gradients for the same loss. The per-backend
validator suites (``test_parity_b_bwd_tf_capture.py``, ``test_parity_b_bwd_paddle_capture.py``)
check each recorded pass node by node.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import torchlens as tl

pytestmark = pytest.mark.rare

_W = [[2.0, -1.0], [3.0, 0.5]]
_X = [[1.0, -2.0], [0.5, 1.5], [2.0, 0.25]]


def _param_grads(trace: Any) -> list[np.ndarray]:
    """Gradients recorded on the trace's parameters for backward pass 1."""

    return [
        np.asarray(record.grad)
        for param in trace._param_logs.values()
        for record in param._grad_records
        if record.backward_pass_number == 1 and record.grad is not None
    ]


@pytest.mark.tf_backend
def test_tf_backward_ready_records_a_pass_through_the_registry() -> None:
    """A TensorFlow capture with ``backward_ready=True`` records ``trace.backward``."""

    tf = pytest.importorskip("tensorflow")

    class Raw(tf.Module):
        """One variable, matmul and relu."""

        def __init__(self) -> None:
            super().__init__(name="raw")
            self.weight = tf.Variable(_W, name="weight")

        def __call__(self, x: Any) -> Any:
            return tf.nn.relu(tf.matmul(x, self.weight))

    model, x = Raw(), tf.constant(_X)
    assert tl.backends.spec("tf").capabilities.backward_capture
    trace = tl.trace(model, x, backend="tf", capture=tl.options.CaptureOptions(backward_ready=True))
    assert trace.backward(lambda out: tf.reduce_sum(out * out), save_grads=True) is trace
    assert len(trace.autograd.backward_passes) == 1
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model(x) * model(x))
    want = tape.gradient(loss, model.weight).numpy()
    got = _param_grads(trace)
    assert len(got) == 1
    np.testing.assert_allclose(got[0], want, rtol=1e-6, atol=1e-6)


@pytest.mark.backend_paddle
def test_paddle_backward_ready_records_a_pass_through_the_registry() -> None:
    """A Paddle capture with ``backward_ready=True`` records ``trace.backward``."""

    paddle = pytest.importorskip("paddle")

    class Raw(paddle.nn.Layer):
        """One parameter, matmul and relu."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = self.create_parameter(
                [2, 2], default_initializer=paddle.nn.initializer.Assign(np.asarray(_W))
            )

        def forward(self, x: Any) -> Any:
            return paddle.nn.functional.relu(paddle.matmul(x, self.weight))

    model, x = Raw(), paddle.to_tensor(np.asarray(_X, dtype=np.float32))
    reference = model(x)
    paddle.sum(reference * reference).backward()
    want = model.weight.grad.numpy().copy()
    model.clear_gradients()

    assert tl.backends.spec("paddle").capabilities.backward_capture
    trace = tl.trace(
        model, x, backend="paddle", capture=tl.options.CaptureOptions(backward_ready=True)
    )
    trace.backward(lambda out: paddle.sum(out * out))
    assert len(trace.autograd.backward_passes) == 1
    assert trace.autograd.grad_fns
    np.testing.assert_allclose(model.weight.grad.numpy(), want, rtol=1e-6, atol=1e-6)
