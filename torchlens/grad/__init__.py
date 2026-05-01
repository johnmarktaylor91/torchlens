"""Autograd memory estimators."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _tensor_memory(tensor: torch.Tensor) -> int:
    """Return tensor memory in bytes.

    Parameters
    ----------
    tensor:
        Tensor to inspect.

    Returns
    -------
    int
        Number of bytes.
    """

    return int(tensor.nelement() * tensor.element_size())


def autograd_saved_bytes(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Estimate autograd-saved bytes from parameters and one input shape.

    Parameters
    ----------
    model:
        Model to estimate.
    input_shape:
        Input tensor shape used for static sizing.

    Returns
    -------
    int
        Conservative byte estimate.
    """

    param_bytes = sum(_tensor_memory(param) for param in model.parameters() if param.requires_grad)
    input_elements = 1
    for dim in input_shape:
        input_elements *= int(dim)
    return int(param_bytes + input_elements * torch.empty((), dtype=torch.float32).element_size())


def grad_fn_memory_cost(grad_fn: Any) -> int:
    """Estimate memory retained by one autograd grad_fn.

    Parameters
    ----------
    grad_fn:
        Autograd function object.

    Returns
    -------
    int
        Sum of saved tensor sizes visible through ``_saved_*`` attributes.
    """

    if grad_fn is None:
        return 0
    total = 0
    for attr_name in dir(grad_fn):
        if not attr_name.startswith("_saved_"):
            continue
        try:
            value = getattr(grad_fn, attr_name)
        except RuntimeError:
            continue
        if isinstance(value, torch.Tensor):
            total += _tensor_memory(value)
        elif isinstance(value, (tuple, list)):
            total += sum(_tensor_memory(item) for item in value if isinstance(item, torch.Tensor))
    return total


__all__ = ["autograd_saved_bytes", "grad_fn_memory_cost"]
