"""Built-in semantic recipes for MLP and FFN modules."""

from __future__ import annotations

from typing import Any

from ..facets import register
from ._helpers import (
    activation_gelu,
    activation_silu,
    add_if_present,
    child_output_spec,
    computed_product,
    first_input_spec,
    module_output_spec,
)

_MLP_FACETS = ("gated_out", "up_out", "down_out", "intermediate", "input", "output")


@register(
    class_name=("LlamaMLP", "MistralMLP", "MixtralMLP"),
    target_scope="module",
    facets=_MLP_FACETS,
)
def gated_mlp(module: Any) -> dict[str, Any]:
    """Return facets for gated Llama/Mistral/Mixtral MLP modules."""

    result: dict[str, Any] = {}
    gate_raw = child_output_spec(module, "gate_proj", "gated_mlp")
    up_out = child_output_spec(module, "up_proj", "gated_mlp")
    gated_out = activation_silu(gate_raw)
    if gated_out is not None and up_out is not None:
        add_if_present(result, "intermediate", computed_product(gated_out, up_out, "gated_mlp"))
    add_if_present(result, "gated_out", gated_out)
    add_if_present(result, "up_out", up_out)
    add_if_present(result, "down_out", child_output_spec(module, "down_proj", "gated_mlp"))
    add_if_present(result, "input", first_input_spec(module, "gated_mlp"))
    add_if_present(result, "output", module_output_spec(module, "gated_mlp"))
    return result


@register(class_name="GPT2MLP", target_scope="module", facets=_MLP_FACETS)
def gpt2_mlp(module: Any) -> dict[str, Any]:
    """Return facets for GPT-2 MLP modules."""

    result: dict[str, Any] = {}
    up_out = child_output_spec(module, "c_fc", "gpt2_mlp")
    add_if_present(result, "up_out", up_out)
    add_if_present(result, "intermediate", activation_gelu(up_out))
    add_if_present(result, "down_out", child_output_spec(module, "c_proj", "gpt2_mlp"))
    add_if_present(result, "input", first_input_spec(module, "gpt2_mlp"))
    add_if_present(result, "output", module_output_spec(module, "gpt2_mlp"))
    return result


@register(class_name=("FFN", "DistilBertFFN"), target_scope="module", facets=_MLP_FACETS)
def distilbert_ffn(module: Any) -> dict[str, Any]:
    """Return facets for DistilBERT FFN modules.

    transformers >= 4.57 renamed the module class from ``DistilBertFFN`` to ``FFN``;
    both exact names are matched so facets populate across transformers versions.
    """

    result: dict[str, Any] = {}
    up_out = child_output_spec(module, "lin1", "distilbert_ffn")
    add_if_present(result, "up_out", up_out)
    add_if_present(result, "intermediate", activation_gelu(up_out))
    add_if_present(result, "down_out", child_output_spec(module, "lin2", "distilbert_ffn"))
    add_if_present(result, "input", first_input_spec(module, "distilbert_ffn"))
    add_if_present(result, "output", module_output_spec(module, "distilbert_ffn"))
    return result
