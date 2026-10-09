"""FLOPs estimation for PyTorch quantized modules logged as internal sources.

A quantized ``Linear``/``Conv`` runs a fused kernel whose output reaches
TorchLens unwrapped, so ``model_prep._ensure_module_output_tensor_logged`` logs
it as an internal source and estimates its forward FLOPs here from the module's
geometry and the first input shape (split out of ``model_prep.py`` under the R43
file-size ratchet; behaviour unchanged).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ...utils.introspection import get_vars_of_type_from_obj
from ._module_arg_stubs import first_stub_shape

_QUANTIZED_MODULE_PREFIXES = (
    "torch.ao.nn.quantized",
    "torch.nn.quantized",
    "torch.ao.nn.intrinsic.quantized",
)


def _is_quantized_module(module: nn.Module) -> bool:
    """Return whether ``module`` is a PyTorch quantized module.

    Parameters
    ----------
    module:
        Module to inspect.

    Returns
    -------
    bool
        Whether the module class is from a known PyTorch quantized namespace.
    """

    module_name = type(module).__module__
    return module_name.startswith(_QUANTIZED_MODULE_PREFIXES)


def _first_tensor_shape(value: Any) -> tuple[int, ...] | None:
    """Return the shape of the first tensor found in ``value``.

    Parameters
    ----------
    value:
        Object tree to search.

    Returns
    -------
    tuple[int, ...] | None
        First tensor shape, or ``None`` when no tensor is present.
    """

    tensors = get_vars_of_type_from_obj(value, torch.Tensor, search_depth=5)
    if tensors:
        return tuple(tensors[0].shape)
    # F20 W1a: module-arg stashes carry payload-free stubs; their recorded
    # shape serves the same estimation read.
    return first_stub_shape(value)


def _quantized_module_bias_present(module: nn.Module) -> bool:
    """Return whether a quantized module appears to have a bias term.

    Parameters
    ----------
    module:
        Quantized module to inspect.

    Returns
    -------
    bool
        Whether the module exposes a non-``None`` bias.
    """

    bias = getattr(module, "bias", None)
    if callable(bias):
        try:
            return bias() is not None
        except Exception:
            return False
    return bias is not None


def estimate_quantized_module_forward_flops(
    module: nn.Module,
    output_shape: tuple[int, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> int | None:
    """Estimate FLOPs for common quantized modules logged as internal sources.

    Parameters
    ----------
    module:
        Module that produced the unwrapped quantized output.
    output_shape:
        Shape of the module output tensor.
    args:
        Positional module-forward arguments.
    kwargs:
        Keyword module-forward arguments.

    Returns
    -------
    int | None
        Estimated forward FLOPs for recognized quantized Linear/Conv modules,
        otherwise ``None``.
    """

    if not _is_quantized_module(module):
        return None
    input_shape = _first_tensor_shape((args, kwargs))
    if input_shape is None:
        return None
    out_numel = int(math.prod(output_shape)) if output_shape else 1
    # Lazy: model_prep imports this module at load time.
    from .model_prep import _module_type

    module_kind = _module_type(module).lower()
    bias_flops = out_numel if _quantized_module_bias_present(module) else 0
    if "linear" in module_kind:
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            return None
        batch = out_numel // out_features if out_features > 0 else 0
        return 2 * batch * in_features * out_features + bias_flops
    if "conv" in module_kind:
        in_channels = getattr(module, "in_channels", None)
        groups = getattr(module, "groups", 1)
        kernel_size = getattr(module, "kernel_size", None)
        if not isinstance(in_channels, int) or not isinstance(groups, int):
            return None
        if isinstance(kernel_size, int):
            kernel_numel = kernel_size
        elif isinstance(kernel_size, tuple) and all(isinstance(v, int) for v in kernel_size):
            kernel_numel = int(math.prod(kernel_size))
        else:
            return None
        channels_per_group = in_channels // groups if groups > 0 else in_channels
        return 2 * out_numel * channels_per_group * kernel_numel + bias_flops
    return None
