"""Paddle and MLX traces stamp ``Trace.root_entry_point`` and reload.

From tlspec 10 the loader refuses an artifact whose root entry-point identity
fact is absent (``_forgery_identity_facts._ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC``).
The Paddle and MLX writers built their Trace directly and never stamped the
fact, so every trace they saved refused to reload with
``artifact_root_entry_point_invalid``. Each case here saves a fresh capture,
reloads it, and asserts the exact stamped spelling: tf's ``module_call:`` form
for a framework module root, and the ``function_call:`` form of
``backends/_finalize.py`` ``new_preview_function_trace`` for a function root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import torchlens as tl

pytestmark = pytest.mark.heavy


def _roundtrip(trace: Any, tmp_path: Path) -> Any:
    """Save one trace, reload it, and check the reload is a distinct equal trace."""

    path = tmp_path / "trace.tlspec"
    tl.save(trace, path)
    loaded = tl.load(str(path))
    assert loaded is not trace
    assert list(loaded.layer_labels) == list(trace.layer_labels)
    assert loaded.root_entry_point == trace.root_entry_point
    return loaded


def _module_call(model: Any) -> str:
    """The module-root spelling tf stamps."""

    return f"module_call:{type(model).__module__}.{type(model).__qualname__}.__call__"


@pytest.mark.backend_paddle
def test_paddle_function_root_stamps_and_reloads(tmp_path: Path) -> None:
    """A Paddle function root stamps function_call:<qualname> and reloads."""

    paddle = pytest.importorskip("paddle")

    def paddle_relu_root(x: Any) -> Any:
        """Two-op Paddle function root."""

        return paddle.nn.functional.relu(paddle.matmul(x, x.T))

    trace = tl.trace(paddle_relu_root, paddle.ones([2, 3], dtype="float32"), backend="paddle")
    assert trace.root_entry_point == f"function_call:{paddle_relu_root.__qualname__}"
    _roundtrip(trace, tmp_path)


@pytest.mark.backend_paddle
def test_paddle_module_root_stamps_and_reloads(tmp_path: Path) -> None:
    """A Paddle Layer root stamps tf's module_call spelling and reloads."""

    paddle = pytest.importorskip("paddle")
    paddle.seed(0)
    model = paddle.nn.Sequential(paddle.nn.Linear(3, 4), paddle.nn.ReLU())
    trace = tl.trace(model, paddle.ones([2, 3], dtype="float32"), backend="paddle")
    assert trace.root_entry_point == _module_call(model)
    _roundtrip(trace, tmp_path)


@pytest.mark.backend_mlx
def test_mlx_function_root_stamps_and_reloads(tmp_path: Path) -> None:
    """An MLX function root stamps function_call:<qualname> and reloads."""

    mx = pytest.importorskip("mlx.core")

    def mlx_relu_root(x: Any) -> Any:
        """Two-op MLX function root."""

        return mx.maximum(mx.matmul(x, x.T), 0.0)

    trace = tl.trace(mlx_relu_root, mx.ones((2, 3)), backend="mlx")
    assert trace.root_entry_point == f"function_call:{mlx_relu_root.__qualname__}"
    _roundtrip(trace, tmp_path)


@pytest.mark.backend_mlx
def test_mlx_module_root_stamps_and_reloads(tmp_path: Path) -> None:
    """An MLX Module root stamps tf's module_call spelling and reloads."""

    pytest.importorskip("mlx.core")
    import mlx.nn as mlx_nn

    model = mlx_nn.Sequential(mlx_nn.Linear(3, 4), mlx_nn.ReLU())
    import mlx.core as mx

    trace = tl.trace(model, mx.ones((2, 3)), backend="mlx")
    assert trace.root_entry_point == _module_call(model)
    _roundtrip(trace, tmp_path)
