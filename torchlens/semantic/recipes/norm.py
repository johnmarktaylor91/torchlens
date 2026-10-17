"""Built-in semantic recipes for normalization modules."""

from __future__ import annotations

from typing import Any

from ..facets import AbsenceReason, FacetSpec, register
from ._helpers import (
    add_if_present,
    first_input_spec,
    module_op_records,
    module_output_spec,
    needs_capture,
    op_output_readable,
    parameter_spec,
    structural,
)

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
    class_name=("RMSNorm", "LlamaRMSNorm", "MistralRMSNorm", "Qwen2RMSNorm"),
    target_scope="module",
    facets=_NORM_FACETS,
)
def rms_norm(module: Any) -> dict[str, Any]:
    """Return facets for RMSNorm-family modules. RMSNorm has no bias term."""

    result: dict[str, Any] = {}
    add_if_present(result, "beta", structural("RMSNorm has no beta/bias parameter"))
    add_if_present(result, "normalized", module_output_spec(module, "rms_norm"))
    add_if_present(result, "input", first_input_spec(module, "rms_norm"))
    add_if_present(result, "gamma", parameter_spec(module, "weight", "rms_norm"))
    return result


def _inv_rms_spec(module: Any, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return the module's single ``rsqrt`` op as the ``inv_rms`` facet, or an absence."""

    label = getattr(module, "address", "<unknown>")
    ops = [op for op in module_op_records(module) if getattr(op, "func_name", None) == "rsqrt"]
    if len(ops) != 1:
        return structural(
            f"{label} ran {len(ops)} rsqrt ops in this capture; the inv_rms facet needs "
            "exactly one (one call of the norm module)"
        )
    if not op_output_readable(ops[0]):
        return needs_capture(
            f"norm scale op {ops[0].label!r} was not saved", f"save=... including {ops[0].label!r}"
        )
    return FacetSpec.from_home(ops[0], home_kind="op", recipe_id=recipe_id)


@register(class_name="Gemma2RMSNorm", target_scope="module", facets=("inv_rms",))
def gemma2_rms_norm(module: Any) -> dict[str, Any]:
    """Return the ``inv_rms`` facet of a Hugging Face ``Gemma2RMSNorm``.

    ``inv_rms`` is the ``rsqrt`` op's output, ``1 / sqrt(mean(x**2) + eps)``,
    the per-position factor that multiplies the input before the
    ``(1 + weight)`` gain. Holding it at its captured value is the
    normalisation-scale freeze of an attribution graph
    (``tl.attribution.build_graph(..., frozen=[tl.facet("inv_rms"), ...])``).
    Documented-unstable.
    """

    result: dict[str, Any] = {}
    add_if_present(result, "inv_rms", _inv_rms_spec(module, "gemma2_rms_norm"))
    return result
