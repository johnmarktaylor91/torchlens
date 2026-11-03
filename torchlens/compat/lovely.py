"""Thin optional adapter for ``lovely-tensors``."""

from __future__ import annotations

import builtins
from typing import Any

import torch


def _tensor_from(obj: Any) -> torch.Tensor:
    """Extract a tensor from a TorchLens object or tensor.

    Parameters
    ----------
    obj:
        Tensor, ``Layer``, or ``Op``.

    Returns
    -------
    torch.Tensor
        Tensor to forward to lovely-tensors.
    """

    if isinstance(obj, torch.Tensor):
        return obj
    out = getattr(obj, "transformed_out", None)
    if isinstance(out, torch.Tensor):
        return out
    out = getattr(obj, "out", None)
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError("Expected a torch.Tensor or TorchLens layer log with a saved out.")


def lovely(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Forward a TorchLens tensor payload to ``lovely_tensors.lovely``.

    Parameters
    ----------
    obj:
        Tensor or TorchLens layer log.
    *args, **kwargs:
        Forwarded to ``lovely_tensors.lovely`` when available.

    Returns
    -------
    Any
        Downstream lovely-tensors result or patched tensor repr.

    Examples
    --------
    >>> # requires: lovely_tensors
    >>> text = repr(lovely(torch.zeros(2, 3)))
    >>> "[2, 3]" in text
    True
    """

    try:
        import lovely_tensors
    except ImportError as exc:
        raise ImportError("Install torchlens[viz] to use torchlens.compat.lovely.") from exc
    tensor = _tensor_from(obj)
    formatter = getattr(lovely_tensors, "lovely", None)
    if formatter is None:
        # The old fallback called lovely_tensors.monkey_patch(), a process-wide
        # mutation of torch.Tensor.__repr__, as a side effect of a value-formatting
        # call, and silently dropped the caller's args/kwargs. Refuse to mutate
        # global state from a formatter; fail loud instead.
        raise RuntimeError(
            "The installed lovely_tensors does not expose a lovely() formatter. "
            "torchlens.compat.lovely will not call the global monkey_patch() to "
            "work around this. Upgrade lovely-tensors to a version exposing "
            "lovely(), or call lovely_tensors directly if you want global patching."
        )
    return formatter(tensor, *args, **kwargs)


def str(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the lovely-tensors string for a TorchLens tensor payload.

    Parameters
    ----------
    obj:
        Tensor or TorchLens layer log.
    *args, **kwargs:
        Forwarded to :func:`lovely`.

    Returns
    -------
    str
        Lovely representation.

    Examples
    --------
    >>> # requires: lovely_tensors
    >>> text = tl.compat.lovely.str(torch.zeros(2, 3))
    >>> "[2, 3]" in text
    True
    """

    return builtins.str(lovely(obj, *args, **kwargs))


__all__ = ["lovely", "str"]
