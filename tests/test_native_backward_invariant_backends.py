"""A native backward on TensorFlow, Paddle and JAX passes the metadata invariants.

After ``trace.backward`` on a backend that declares backward capture, the trace
reports ``has_backward_pass``, satisfies ``check_metadata_invariants`` and draws
its backward and combined graphs. Corrupting the real records afterwards (the
flag, the pass count, a grad fn's ownership) makes the invariant fire, so the
declared contract is a live tripwire on real traces, not only on stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.errors import MetadataInvariantError
from torchlens.validation import check_metadata_invariants

_W = [[2.0, -1.0], [3.0, 0.5]]
_X = [[1.0, -2.0], [0.5, 1.5], [2.0, 0.25]]


def _tf_trace() -> Any:
    """A TensorFlow trace after one native backward pass."""

    tf = pytest.importorskip("tensorflow")

    class Raw(tf.Module):
        """One variable, matmul and relu."""

        def __init__(self) -> None:
            super().__init__(name="raw")
            self.weight = tf.Variable(_W, name="weight")

        def __call__(self, x: Any) -> Any:
            return tf.nn.relu(tf.matmul(x, self.weight))

    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(Raw(), tf.constant(_X), backend="tf", capture=capture)
    return trace.backward(lambda out: tf.reduce_sum(out * out))


def _paddle_trace() -> Any:
    """A Paddle trace after one native backward pass."""

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

    x = paddle.to_tensor(np.asarray(_X, dtype=np.float32))
    capture = tl.options.CaptureOptions(backward_ready=True)
    trace = tl.trace(Raw(), x, backend="paddle", capture=capture)
    return trace.backward(lambda out: paddle.sum(out * out))


def _jax_trace() -> Any:
    """A JAX trace after one native backward pass."""

    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    weight = jnp.asarray(_W)

    def net(x: Any) -> Any:
        return jax.nn.relu(x @ weight)

    trace = tl.trace(net, jnp.asarray(_X), backend="jax")
    return trace.backward(lambda out: jnp.sum(out * out))


_BUILDERS: list[Any] = [
    pytest.param("tf", _tf_trace, marks=pytest.mark.tf_backend, id="tf"),
    pytest.param("paddle", _paddle_trace, marks=pytest.mark.backend_paddle, id="paddle"),
    pytest.param("jax", _jax_trace, marks=pytest.mark.backend_jax, id="jax"),
]


def _orphan_one_grad_fn(trace: Any) -> None:
    """Strip the calls of one grad fn that is no pass's root."""

    roots = {
        root for record in trace._backward_pass_logs.values() for root in record.root_grad_fn_ids
    }
    for object_id, record in trace._grad_fn_logs.items():
        if object_id not in roots:
            record.calls = {}
            return
    pytest.fail("every grad fn is a root; the model leaves nothing to orphan")


_CORRUPTIONS: list[Any] = [
    pytest.param(
        lambda trace: setattr(trace, "_autograd_has_backward_pass", False),
        "has_backward_pass is false",
        id="flag-cleared",
    ),
    pytest.param(
        lambda trace: setattr(trace, "_num_backward_passes", trace._num_backward_passes + 1),
        "disagrees with the recorded backward pass",
        id="pass-count",
    ),
    pytest.param(_orphan_one_grad_fn, "not owned by a recorded backward pass", id="orphan"),
]


@pytest.mark.parametrize(("backend", "build"), _BUILDERS)
def test_native_backward_satisfies_the_declared_contract(
    backend: str, build: Callable[[], Any], tmp_path: Path
) -> None:
    """The backend declares backward capture and its recorded pass is consistent and drawable."""

    trace = build()
    assert tl.backends.spec(backend).capabilities.backward_capture
    assert trace.autograd.has_backward_pass is True
    assert len(trace.autograd.backward_passes) == 1
    assert trace.autograd.grad_fns
    assert check_metadata_invariants(trace) is True
    for graph in ("backward", "combined"):
        drawn = trace.draw(graph=graph, path=str(tmp_path / graph), file_format="svg", rolled=False)
        assert "digraph" in drawn.dot


@pytest.mark.parametrize(("backend", "build"), _BUILDERS)
@pytest.mark.parametrize(("corrupt", "match"), _CORRUPTIONS)
def test_corrupting_native_backward_records_fires_the_invariant(
    backend: str, build: Callable[[], Any], corrupt: Callable[[Any], None], match: str
) -> None:
    """Each declared arm fires on the real records of a native backward trace."""

    trace = build()
    assert check_metadata_invariants(trace) is True
    corrupt(trace)
    with pytest.raises(MetadataInvariantError, match=match):
        check_metadata_invariants(trace)
