"""Shared helpers for built-in semantic recipes."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..facets import Facet, FacetSpec, MissingFacet
from ..reconstruction import ReconstructionFacet, sdpa_reconstruction_spec


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


def child_output_spec(module: Any, child_name: str, recipe_id: str) -> FacetSpec | None:
    """Return an op-anchored spec for a child module's single output."""

    child = child_module(module, child_name)
    if child is None:
        return None
    try:
        call = child._single_call_or_error()
        if len(call.output_ops) != 1:
            return None
        op = child.trace.ops[call.output_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None
    return FacetSpec.from_home(op, home_kind="op", recipe_id=recipe_id)


def first_input(module: Any) -> Any | None:
    """Return a module's first captured forward input when available."""

    try:
        args = module.calls[0].forward_args
    except (AttributeError, KeyError, IndexError):
        return None
    if args:
        return args[0]
    return None


def first_input_spec(module: Any, recipe_id: str) -> FacetSpec | None:
    """Return a read-only spec for a module's first captured input."""

    value = first_input(module)
    if value is None:
        return None
    return FacetSpec.from_home(value, home_kind="module_input", recipe_id=recipe_id)


def module_input_op_spec(module: Any, recipe_id: str) -> FacetSpec | None:
    """Return an op-anchored spec for a module's first input op.

    Parameters
    ----------
    module:
        TorchLens module record.
    recipe_id:
        Recipe identifier.

    Returns
    -------
    FacetSpec | None
        Op-anchored input spec when the module input is a captured op.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    try:
        call = module._single_call_or_error()
        input_ops = list(getattr(call, "input_ops", ()) or ())
        if not input_ops:
            return None
        op = trace.ops[input_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None
    return FacetSpec.from_home(op, home_kind="op", recipe_id=recipe_id)


def module_output(module: Any) -> Any | None:
    """Return a module's first-call single output when available."""

    try:
        return module.calls[0].out
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None


def module_output_spec(module: Any, recipe_id: str) -> FacetSpec | None:
    """Return an op-anchored spec for a module's single output."""

    try:
        call = module._single_call_or_error()
        if len(call.output_ops) != 1:
            return None
        op = module.trace.ops[call.output_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None
    return FacetSpec.from_home(op, home_kind="op", recipe_id=recipe_id)


def parameter_spec(module: Any, name: str, recipe_id: str) -> FacetSpec | None:
    """Return a read-only spec for a named module parameter."""

    params = getattr(module, "params", None)
    if params is None:
        return None
    try:
        for param in params:
            if getattr(param, "name", None) == name:
                return FacetSpec.from_home(param, home_kind="parameter", recipe_id=recipe_id)
    except TypeError:
        return None
    cls = getattr(module, "cls", None)
    parameter = getattr(cls, name, None)
    if parameter is None:
        return None
    return FacetSpec.from_home(parameter, home_kind="parameter", recipe_id=recipe_id)


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

    if isinstance(value, FacetSpec):
        if n_heads is None:
            return None
        if d_head is None:
            try:
                last_dim = value.read().shape[-1]
            except (AttributeError, RuntimeError, ValueError):
                return None
            d_head = last_dim // n_heads
        return value.heads(n_heads, d_head)
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

    if isinstance(value, FacetSpec):
        return FacetSpec.computed(lambda: F.gelu(Facet(value).value), recipe_id=value.recipe_id)
    if isinstance(value, torch.Tensor):
        return F.gelu(value)
    return None


def activation_silu(value: Any) -> Any | None:
    """Return SiLU activation when the value is tensor-like."""

    if isinstance(value, FacetSpec):
        return FacetSpec.computed(lambda: F.silu(Facet(value).value), recipe_id=value.recipe_id)
    if isinstance(value, torch.Tensor):
        return F.silu(value)
    return None


def computed_product(left: Any, right: Any, recipe_id: str) -> FacetSpec | None:
    """Return a read-only computed product spec for two spec-backed values."""

    if isinstance(left, FacetSpec) and isinstance(right, FacetSpec):
        return FacetSpec.computed(
            lambda: Facet(left).value * Facet(right).value,
            recipe_id=recipe_id,
        )
    return None


def fused_sdpa_facet(module: Any, facet: ReconstructionFacet, recipe_id: str) -> Any:
    """Return a reconstructed SDPA facet or a MissingFacet.

    Parameters
    ----------
    module:
        Attention module record.
    facet:
        Reconstructed facet name.
    recipe_id:
        Recipe identifier for provenance.

    Returns
    -------
    Any
        Computed read-only ``FacetSpec`` or ``MissingFacet``.
    """

    return sdpa_reconstruction_spec(module, facet, recipe_id=recipe_id)


def fused_sdpa_pattern(module: Any) -> Any:
    """Return a reconstructed fused attention pattern or a MissingFacet."""

    label = getattr(module.calls[0], "call_label", getattr(module, "address", "<unknown>"))
    value = fused_sdpa_facet(module, "pattern", "attention_reconstruction")
    if not isinstance(value, MissingFacet):
        return value
    return MissingFacet(
        f"attention pattern not captured: model uses a fused attention kernel "
        f"(SDPA/FlashAttention) at {label}. "
        f"{value.reason} To READ pattern/scores/z, re-run with reconstruction_ready=True "
        "(or save_arg_values=True). To EDIT the pattern, capture a model that runs eager / "
        "unfused attention so it is a real, editable op -- e.g. load HF models with "
        "attn_implementation='eager' (AutoModel.from_pretrained(..., attn_implementation='eager')). "
        "The reconstructed pattern is read-only by design: editing it would silently change the "
        "attention kernel, so capture eager for a consistent baseline."
    )
