"""Built-in semantic recipes for normalization modules."""

from __future__ import annotations

from typing import Any

from ..facets import register
from ._helpers import add_if_present, first_input, module_output

_NORM_FACETS = ("normalized", "gamma", "beta", "input")


@register(class_name="LayerNorm", target_scope="module", facets=_NORM_FACETS)
def layer_norm(module: Any) -> dict[str, Any]:
    """Return facets for LayerNorm modules."""

    result: dict[str, Any] = {}
    add_if_present(result, "normalized", module_output(module))
    add_if_present(result, "input", first_input(module))
    cls = getattr(module, "cls", None)
    add_if_present(result, "gamma", getattr(cls, "weight", None))
    if hasattr(cls, "bias"):
        result["beta"] = getattr(cls, "bias")
    return result


@register(
    class_name=("RMSNorm", "LlamaRMSNorm", "MistralRMSNorm"),
    target_scope="module",
    facets=_NORM_FACETS,
)
def rms_norm(module: Any) -> dict[str, Any]:
    """Return facets for RMSNorm-family modules."""

    result: dict[str, Any] = {"beta": None}
    add_if_present(result, "normalized", module_output(module))
    add_if_present(result, "input", first_input(module))
    cls = getattr(module, "cls", None)
    add_if_present(result, "gamma", getattr(cls, "weight", None))
    return result
