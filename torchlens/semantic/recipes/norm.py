"""Built-in semantic recipes for normalization modules."""

from __future__ import annotations

from typing import Any

from ..facets import MissingFacet, register
from ._helpers import add_if_present, first_input_spec, module_output_spec, parameter_spec


_NORM_FACETS = ("normalized", "gamma", "beta", "input")


@register(class_name="LayerNorm", target_scope="module", facets=_NORM_FACETS)
def layer_norm(module: Any) -> dict[str, Any]:
    """Return facets for LayerNorm modules."""

    result: dict[str, Any] = {}
    add_if_present(result, "normalized", module_output_spec(module, "layer_norm"))
    add_if_present(result, "input", first_input_spec(module, "layer_norm"))
    add_if_present(result, "gamma", parameter_spec(module, "weight", "layer_norm"))
    add_if_present(result, "beta", parameter_spec(module, "bias", "layer_norm"))
    return result


@register(
    class_name=("RMSNorm", "LlamaRMSNorm", "MistralRMSNorm"),
    target_scope="module",
    facets=_NORM_FACETS,
)
def rms_norm(module: Any) -> dict[str, Any]:
    """Return facets for RMSNorm-family modules. RMSNorm has no bias term."""

    result: dict[str, Any] = {"beta": MissingFacet("RMSNorm has no beta/bias parameter.")}
    add_if_present(result, "normalized", module_output_spec(module, "rms_norm"))
    add_if_present(result, "input", first_input_spec(module, "rms_norm"))
    add_if_present(result, "gamma", parameter_spec(module, "weight", "rms_norm"))
    return result
