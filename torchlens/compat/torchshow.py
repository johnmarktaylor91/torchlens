"""Thin optional adapter for the ``torchshow`` visualization package."""

from __future__ import annotations

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
        Tensor to forward to torchshow.
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


def show(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Forward a TorchLens tensor payload to ``torchshow.show``.

    Parameters
    ----------
    obj:
        Tensor or TorchLens layer log.
    *args, **kwargs:
        Forwarded to ``torchshow.show``.

    Returns
    -------
    Any
        Downstream torchshow result.
    """

    try:
        import torchshow
    except ImportError as exc:
        raise ImportError("Install torchlens[viz] to use torchlens.compat.torchshow.") from exc
    return torchshow.show(_tensor_from(obj), *args, **kwargs)


__all__ = ["show"]
