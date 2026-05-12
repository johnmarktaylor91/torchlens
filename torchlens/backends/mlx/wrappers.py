"""MLX function-call interception for technical-preview capture."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from ... import _state

_ORIGINALS: dict[tuple[object, str], object] = {}
_WRAPPED = False


def wrap_mlx(backend: object) -> None:
    """Install MLX wrappers if MLX is available.

    Parameters
    ----------
    backend:
        Active :class:`MLXBackend` instance receiving wrapper events.
    """

    global _WRAPPED
    if _WRAPPED:
        return
    mx, nn = _import_mlx()
    for name in ("add", "matmul", "multiply", "subtract", "divide", "maximum", "minimum"):
        _wrap_attr(mx, name, backend, name)
    for name in ("relu", "gelu", "sigmoid", "tanh", "softmax"):
        _wrap_attr(nn, name, backend, name)
    linear_cls = getattr(nn, "Linear", None)
    if linear_cls is not None:
        _wrap_attr(linear_cls, "__call__", backend, "linear")
    _WRAPPED = True


def unwrap_mlx() -> None:
    """Restore original MLX callables."""

    global _WRAPPED
    for (owner, name), original in list(_ORIGINALS.items()):
        setattr(owner, name, original)
    _ORIGINALS.clear()
    _WRAPPED = False


def is_mlx_wrapped() -> bool:
    """Return whether MLX wrappers are currently installed.

    Returns
    -------
    bool
        ``True`` when wrappers are installed.
    """

    return _WRAPPED


def _import_mlx() -> tuple[object, object]:
    """Import MLX lazily.

    Returns
    -------
    tuple[object, object]
        ``mlx.core`` and ``mlx.nn`` modules.
    """

    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise ImportError("MLX backend requires the optional 'mlx' package.") from exc
    return mx, nn


def _wrap_attr(owner: object, name: str, backend: object, op_name: str) -> None:
    """Replace one MLX attribute with a logging wrapper.

    Parameters
    ----------
    owner:
        Module or class containing the callable.
    name:
        Attribute name to wrap.
    backend:
        MLX backend receiving wrapper events.
    op_name:
        TorchLens operation type to emit.
    """

    original = getattr(owner, name, None)
    if original is None or not callable(original):
        return
    key = (owner, name)
    if key in _ORIGINALS:
        return
    _ORIGINALS[key] = original

    @functools.wraps(
        original,
        assigned=("__module__", "__name__", "__doc__", "__annotations__"),
    )
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Call an MLX function and emit an op event when logging is active."""

        if not _state._logging_enabled:
            return original(*args, **kwargs)
        trace = _state._active_trace
        if trace is None or getattr(trace, "_mlx_capture_depth", 0) > 0:
            return original(*args, **kwargs)
        trace._mlx_capture_depth = getattr(trace, "_mlx_capture_depth", 0) + 1
        try:
            output = original(*args, **kwargs)
        finally:
            trace._mlx_capture_depth -= 1
        emit = getattr(backend, "emit_mlx_operation")
        emit(trace, op_name, original, args, kwargs, output)
        return output

    setattr(owner, name, wrapper)


__all__ = ["is_mlx_wrapped", "unwrap_mlx", "wrap_mlx"]
