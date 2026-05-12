"""MLX function-call interception for technical-preview capture."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from ... import _state


class _MLXWrapperRegistry:
    """Per-process MLX wrapper registry with clean backend rebinding."""

    def __init__(self) -> None:
        """Initialize an empty wrapper registry."""

        self._originals: dict[tuple[object, str], object] = {}
        self._wrapped = False

    def wrap(self, backend: object) -> None:
        """Install wrappers bound to ``backend``.

        Parameters
        ----------
        backend:
            Active MLX backend that receives wrapper events.
        """

        if self._wrapped:
            self.unwrap()
        mx, nn = _import_mlx()
        for name in (
            "add",
            "matmul",
            "multiply",
            "subtract",
            "divide",
            "maximum",
            "minimum",
            "power",
            "sum",
            "mean",
            "max",
            "min",
            "argmax",
            "argmin",
            "reshape",
            "transpose",
            "concatenate",
            "stack",
            "split",
        ):
            self.wrap_attr(mx, name, backend, name)
        for name in ("relu", "gelu", "sigmoid", "tanh", "softmax", "silu"):
            self.wrap_attr(nn, name, backend, name)
        for cls_name, op_name in (
            ("Linear", "linear"),
            ("Conv2d", "conv2d"),
            ("LayerNorm", "layernorm"),
            ("RMSNorm", "rmsnorm"),
            ("BatchNorm", "batchnorm"),
            ("GroupNorm", "groupnorm"),
            ("Dropout", "dropout"),
            ("Embedding", "embedding"),
            ("MultiHeadAttention", "multiheadattention"),
        ):
            cls = getattr(nn, cls_name, None)
            if cls is not None:
                self.wrap_attr(cls, "__call__", backend, op_name)
        self._wrapped = True

    def unwrap(self) -> None:
        """Restore all original MLX callables."""

        for (owner, name), original in list(self._originals.items()):
            setattr(owner, name, original)
        self._originals.clear()
        self._wrapped = False

    def is_wrapped(self) -> bool:
        """Return whether this registry currently has installed wrappers."""

        return self._wrapped

    def wrap_attr(self, owner: object, name: str, backend: object, op_name: str) -> None:
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
        if key in self._originals:
            return
        self._originals[key] = original

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


_REGISTRY = _MLXWrapperRegistry()


def wrap_mlx(backend: object) -> None:
    """Install MLX wrappers if MLX is available.

    Parameters
    ----------
    backend:
        Active :class:`MLXBackend` instance receiving wrapper events.
    """

    _REGISTRY.wrap(backend)


def unwrap_mlx() -> None:
    """Restore original MLX callables."""

    _REGISTRY.unwrap()


def is_mlx_wrapped() -> bool:
    """Return whether MLX wrappers are currently installed.

    Returns
    -------
    bool
        ``True`` when wrappers are installed.
    """

    return _REGISTRY.is_wrapped()


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


__all__ = ["is_mlx_wrapped", "unwrap_mlx", "wrap_mlx"]
