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
        Tensor, ``LayerLog``, or ``LayerPassLog``.

    Returns
    -------
    torch.Tensor
        Tensor to forward to lovely-tensors.
    """

    if isinstance(obj, torch.Tensor):
        return obj
    activation = getattr(obj, "transformed_activation", None)
    if isinstance(activation, torch.Tensor):
        return activation
    activation = getattr(obj, "activation", None)
    if isinstance(activation, torch.Tensor):
        return activation
    raise TypeError("Expected a torch.Tensor or TorchLens layer log with a saved activation.")


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
    """

    try:
        import lovely_tensors
    except ImportError as exc:
        raise ImportError("Install torchlens[viz] to use torchlens.compat.lovely.") from exc
    tensor = _tensor_from(obj)
    formatter = getattr(lovely_tensors, "lovely", None)
    if formatter is not None:
        return formatter(tensor, *args, **kwargs)
    lovely_tensors.monkey_patch()
    return repr(tensor)


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
    """

    return builtins.str(lovely(obj, *args, **kwargs))


__all__ = ["lovely", "str"]
