"""Thin optional adapter for the ``torchshow`` visualization package."""

from __future__ import annotations

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
        Tensor to forward to torchshow.
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
