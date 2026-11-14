"""A backward-ready non-PyTorch trace saves and loads, with or without a backward pass.

The capture keeps live native-backward state on the trace (a TF tape, a Paddle
grad-node scope). That state is never portable: a save drops it, and a loaded
trace's ``backward()`` refuses with the typed capability error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError

pytestmark = pytest.mark.heavy

_W = [[2.0, -1.0], [3.0, 0.5]]
_X = [[1.0, -2.0], [0.5, 1.5], [2.0, 0.25]]


def _held_until_flip(backend: str) -> pytest.MarkDecorator:
    """Skip while ``backend`` declares no backward capture (the flip is held)."""

    return pytest.mark.skipif(
        not tl.backends.spec(backend).capabilities.backward_capture,
        reason=f"{backend} native backward capture is built but held undeclared "
        "until the non_torch_backward_inert contract allows recorded passes",
    )


def _tf_trace(*, backward_ready: bool = True) -> tuple[Any, Any]:
    tf = pytest.importorskip("tensorflow")

    class Raw(tf.Module):
        def __init__(self) -> None:
            super().__init__(name="raw")
            self.weight = tf.Variable(_W, name="weight")

        def __call__(self, x: Any) -> Any:
            return tf.nn.relu(tf.matmul(x, self.weight))

    trace = tl.trace(
        Raw(),
        tf.constant(_X),
        backend="tf",
        capture=tl.options.CaptureOptions(backward_ready=backward_ready),
    )
    return trace, lambda out: tf.reduce_sum(out * out)


def _paddle_trace() -> tuple[Any, Any]:
    paddle = pytest.importorskip("paddle")

    class Raw(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.weight = self.create_parameter(
                [2, 2], default_initializer=paddle.nn.initializer.Assign(np.asarray(_W))
            )

        def forward(self, x: Any) -> Any:
            return paddle.nn.functional.relu(paddle.matmul(x, self.weight))

    trace = tl.trace(
        Raw(),
        paddle.to_tensor(np.asarray(_X, dtype=np.float32)),
        backend="paddle",
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    return trace, lambda out: paddle.sum(out * out)


_BUILDERS = {"tf": _tf_trace, "paddle": _paddle_trace}


@pytest.mark.parametrize(
    "backend",
    [pytest.param(name, marks=_held_until_flip(name)) for name in sorted(_BUILDERS)],
)
@pytest.mark.parametrize("run_backward", [False, True], ids=["forward_only", "after_backward"])
def test_backward_ready_trace_saves_and_loaded_backward_refuses(
    backend: str, run_backward: bool, tmp_path: Path
) -> None:
    trace, loss = _BUILDERS[backend]()
    if run_backward:
        trace.backward(loss)
    path = tmp_path / f"{backend}.tlspec"
    trace.save(str(path))
    loaded = tl.load(str(path))
    assert len(loaded.autograd.backward_passes) == len(trace.autograd.backward_passes)
    with pytest.raises(RunCapabilityUnavailableError):
        loaded.backward(loss)


def test_plain_tf_trace_saves_without_grad_site_bookkeeping(tmp_path: Path) -> None:
    """A plain TF capture arms its (empty) tl.grad sites; the bookkeeping never saves."""

    trace, _ = _tf_trace(backward_ready=False)
    assert "_tf_grad_sites_armed" in trace.__dict__
    path = tmp_path / "plain.tlspec"
    trace.save(str(path))
    loaded = tl.load(str(path))
    assert "_tf_grad_sites_armed" not in loaded.__dict__
    assert len(loaded.layers) == len(trace.layers)
