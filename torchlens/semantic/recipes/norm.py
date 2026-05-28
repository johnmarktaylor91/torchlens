"""Built-in semantic recipes for normalization modules."""

from __future__ import annotations

from typing import Any

from ..facets import register
from ._helpers import add_if_present, first_input, module_output


def _param_value(module: Any, name: str) -> Any:
    """Resolve a parameter tensor by bare name; None if not present.

    Module.params is keyed by full address (e.g., `embeddings.LayerNorm.weight`),
    but recipes want to look up by the bare parameter name (`weight`). Iterate
    params and match on `.name`. Falls back to None when the param is absent or
    its tensor value isn't currently in memory.
    """
    params = getattr(module, "params", None)
    if params is None:
        return None
    try:
        for param in params:
            if getattr(param, "name", None) == name:
                return getattr(param, "value", None)
    except TypeError:
        return None
    return None


_NORM_FACETS = ("normalized", "gamma", "beta", "input")


@register(class_name="LayerNorm", target_scope="module", facets=_NORM_FACETS)
def layer_norm(module: Any) -> dict[str, Any]:
    """Return facets for LayerNorm modules."""

    result: dict[str, Any] = {}
    add_if_present(result, "normalized", module_output(module))
    add_if_present(result, "input", first_input(module))
    add_if_present(result, "gamma", _param_value(module, "weight"))
    add_if_present(result, "beta", _param_value(module, "bias"))
    return result


@register(
    class_name=("RMSNorm", "LlamaRMSNorm", "MistralRMSNorm"),
    target_scope="module",
    facets=_NORM_FACETS,
)
def rms_norm(module: Any) -> dict[str, Any]:
    """Return facets for RMSNorm-family modules. RMSNorm has no bias term."""

    result: dict[str, Any] = {"beta": None}
    add_if_present(result, "normalized", module_output(module))
    add_if_present(result, "input", first_input(module))
    add_if_present(result, "gamma", _param_value(module, "weight"))
    return result
