"""Shared helpers for built-in semantic recipes."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..facets import MissingFacet


def child_module(module: Any, child_name: str) -> Any | None:
    """Return a direct child Module record by address.

    Parameters
    ----------
    module:
        Parent TorchLens Module record.
    child_name:
        Child attribute name.
    """

    trace = getattr(module, "_source_trace", None)
    if trace is None:
        return None
    child_address = child_name if module.address == "self" else f"{module.address}.{child_name}"
    try:
        return trace.modules[child_address]
    except (KeyError, ValueError):
        return None


def child_out(module: Any, child_name: str) -> Any | None:
    """Return a child module's first-call output when available."""

    child = child_module(module, child_name)
    if child is None:
        return None
    try:
        return child.calls[0].out
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None


def first_input(module: Any) -> Any | None:
    """Return a module's first captured forward input when available."""

    try:
        args = module.calls[0].forward_args
    except (AttributeError, KeyError, IndexError):
        return None
    if args:
        return args[0]
    return None


def module_output(module: Any) -> Any | None:
    """Return a module's first-call single output when available."""

    try:
        return module.calls[0].out
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None


def add_if_present(result: dict[str, Any], name: str, value: Any) -> None:
    """Set a result item when the value is available."""

    if value is not None:
        result[name] = value


def config_value(obj: Any, *names: str) -> Any | None:
    """Return the first available attribute from an object."""

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    custom_attributes = getattr(obj, "custom_attributes", None)
    if isinstance(custom_attributes, dict):
        for name in names:
            if name in custom_attributes:
                return custom_attributes[name]
    return None


def reshape_heads(value: Any, n_heads: int | None, d_head: int | None = None) -> Any | None:
    """Reshape a projection output to ``(B, S, n_heads, d_head)``."""

    if not isinstance(value, torch.Tensor) or n_heads is None:
        return None
    if value.ndim < 3:
        return None
    inferred_d_head = d_head if d_head is not None else value.shape[-1] // n_heads
    if inferred_d_head <= 0 or value.shape[-1] != n_heads * inferred_d_head:
        return None
    return value.view(*value.shape[:-1], n_heads, inferred_d_head)


def activation_gelu(value: Any) -> Any | None:
    """Return GELU activation when the value is tensor-like."""

    if isinstance(value, torch.Tensor):
        return F.gelu(value)
    return None


def activation_silu(value: Any) -> Any | None:
    """Return SiLU activation when the value is tensor-like."""

    if isinstance(value, torch.Tensor):
        return F.silu(value)
    return None


def fused_sdpa_pattern(module: Any) -> MissingFacet:
    """Return a MissingFacet explaining fused-SDPA pattern unavailability."""

    label = getattr(module.calls[0], "call_label", getattr(module, "address", "<unknown>"))
    return MissingFacet(
        f"attention pattern not captured: model uses fused SDPA at {label}. "
        "Re-run with model.config._attn_implementation='eager' to expose the pattern."
    )
