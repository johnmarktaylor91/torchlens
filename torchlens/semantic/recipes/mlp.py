"""Built-in semantic recipes for MLP and FFN modules."""

from __future__ import annotations

from typing import Any

import torch

from ..facets import register
from ._helpers import (
    activation_gelu,
    activation_silu,
    add_if_present,
    child_out,
    first_input,
    module_output,
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
    gate_raw = child_out(module, "gate_proj")
    up_out = child_out(module, "up_proj")
    gated_out = activation_silu(gate_raw)
    if isinstance(gated_out, torch.Tensor) and isinstance(up_out, torch.Tensor):
        result["intermediate"] = gated_out * up_out
    add_if_present(result, "gated_out", gated_out)
    add_if_present(result, "up_out", up_out)
    add_if_present(result, "down_out", child_out(module, "down_proj"))
    add_if_present(result, "input", first_input(module))
    add_if_present(result, "output", module_output(module))
    return result


@register(class_name="GPT2MLP", target_scope="module", facets=_MLP_FACETS)
def gpt2_mlp(module: Any) -> dict[str, Any]:
    """Return facets for GPT-2 MLP modules."""

    result: dict[str, Any] = {}
    up_out = child_out(module, "c_fc")
    add_if_present(result, "up_out", up_out)
    add_if_present(result, "intermediate", activation_gelu(up_out))
    add_if_present(result, "down_out", child_out(module, "c_proj"))
    add_if_present(result, "input", first_input(module))
    add_if_present(result, "output", module_output(module))
    return result


@register(class_name="DistilBertFFN", target_scope="module", facets=_MLP_FACETS)
def distilbert_ffn(module: Any) -> dict[str, Any]:
    """Return facets for DistilBERT FFN modules."""

    result: dict[str, Any] = {}
    up_out = child_out(module, "lin1")
    add_if_present(result, "up_out", up_out)
    add_if_present(result, "intermediate", activation_gelu(up_out))
    add_if_present(result, "down_out", child_out(module, "lin2"))
    add_if_present(result, "input", first_input(module))
    add_if_present(result, "output", module_output(module))
    return result
