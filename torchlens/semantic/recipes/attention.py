"""Built-in semantic recipes for attention modules."""

from __future__ import annotations

from typing import Any

import torch

from ..facets import register
from ._helpers import add_if_present, child_out, config_value, first_input, fused_sdpa_pattern
from ._helpers import module_output, reshape_heads

_ATTENTION_FACETS_BASE = (
    "q",
    "k",
    "v",
    "attn_out",
    "input",
    "n_heads",
    "n_q_heads",
    "n_kv_heads",
    "d_head",
    "head",
)
_ATTENTION_FACETS_SDPA = (
    *_ATTENTION_FACETS_BASE,
    "pattern",
)


def _with_attention_common(
    result: dict[str, Any],
    module: Any,
    n_q_heads: int | None,
    n_kv_heads: int | None,
    d_head: int | None,
) -> dict[str, Any]:
    """Attach common attention facets to a recipe result."""

    add_if_present(result, "attn_out", module_output(module))
    add_if_present(result, "input", first_input(module))
    if n_q_heads is not None:
        result["n_q_heads"] = n_q_heads
        result["n_heads"] = n_q_heads
    if n_kv_heads is not None:
        result["n_kv_heads"] = n_kv_heads
    if d_head is not None:
        result["d_head"] = d_head
    result["head"] = module.facets.head
    class_name = getattr(module, "class_name", "")
    if "Sdpa" in class_name:
        result["pattern"] = fused_sdpa_pattern(module)
    return result


def _attention_config(module: Any) -> tuple[int | None, int | None, int | None]:
    """Return ``(n_q_heads, n_kv_heads, d_head)`` from common HF attention configs."""

    cls = getattr(module, "cls", None)
    n_q_heads = config_value(module, "n_heads", "num_heads", "num_attention_heads")
    if n_q_heads is None:
        n_q_heads = config_value(cls, "n_heads", "num_heads", "num_attention_heads")
    n_kv_heads = config_value(module, "num_key_value_heads", "n_kv_heads")
    if n_kv_heads is None:
        n_kv_heads = config_value(cls, "num_key_value_heads", "n_kv_heads")
    if n_kv_heads is None:
        n_kv_heads = n_q_heads
    d_head = config_value(module, "head_dim", "attention_head_size", "d_kv")
    if d_head is None:
        d_head = config_value(cls, "head_dim", "attention_head_size", "d_kv")
    if d_head is None:
        hidden_size = config_value(module, "dim", "embed_dim", "hidden_size", "all_head_size")
        if hidden_size is None:
            hidden_size = config_value(cls, "dim", "embed_dim", "hidden_size", "all_head_size")
        if isinstance(hidden_size, int) and isinstance(n_q_heads, int) and n_q_heads:
            d_head = hidden_size // n_q_heads
    return (
        n_q_heads if isinstance(n_q_heads, int) else None,
        n_kv_heads if isinstance(n_kv_heads, int) else None,
        d_head if isinstance(d_head, int) else None,
    )


@register(
    class_name="DistilBertSdpaAttention",
    target_scope="module",
    facets=_ATTENTION_FACETS_SDPA,
)
def distilbert_sdpa_attention(module: Any) -> dict[str, Any]:
    """Return facets for DistilBERT SDPA attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(result, "q", reshape_heads(child_out(module, "q_lin"), n_q_heads, d_head))
    add_if_present(result, "k", reshape_heads(child_out(module, "k_lin"), n_kv_heads, d_head))
    add_if_present(result, "v", reshape_heads(child_out(module, "v_lin"), n_kv_heads, d_head))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(class_name="GPT2Attention", target_scope="module", facets=_ATTENTION_FACETS_BASE)
def gpt2_attention(module: Any) -> dict[str, Any]:
    """Return facets for GPT-2 fused-QKV attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    c_attn_out = child_out(module, "c_attn")
    result: dict[str, Any] = {}
    if isinstance(c_attn_out, torch.Tensor) and n_q_heads is not None:
        q_raw, k_raw, v_raw = c_attn_out.split(c_attn_out.shape[-1] // 3, dim=-1)
        add_if_present(result, "q", reshape_heads(q_raw, n_q_heads, d_head))
        add_if_present(result, "k", reshape_heads(k_raw, n_kv_heads, d_head))
        add_if_present(result, "v", reshape_heads(v_raw, n_kv_heads, d_head))
    add_if_present(result, "attn_out", child_out(module, "c_proj"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(class_name="BertSelfAttention", target_scope="module", facets=_ATTENTION_FACETS_BASE)
def bert_self_attention(module: Any) -> dict[str, Any]:
    """Return facets for BERT self-attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(result, "q", reshape_heads(child_out(module, "query"), n_q_heads, d_head))
    add_if_present(result, "k", reshape_heads(child_out(module, "key"), n_kv_heads, d_head))
    add_if_present(result, "v", reshape_heads(child_out(module, "value"), n_kv_heads, d_head))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(
    class_name=("LlamaAttention", "MistralAttention"),
    target_scope="module",
    facets=_ATTENTION_FACETS_BASE,
)
def gqa_attention(module: Any) -> dict[str, Any]:
    """Return facets for GQA-style Llama and Mistral attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(result, "q", reshape_heads(child_out(module, "q_proj"), n_q_heads, d_head))
    add_if_present(result, "k", reshape_heads(child_out(module, "k_proj"), n_kv_heads, d_head))
    add_if_present(result, "v", reshape_heads(child_out(module, "v_proj"), n_kv_heads, d_head))
    add_if_present(result, "attn_out", child_out(module, "o_proj"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(
    class_name=("LlamaSdpaAttention", "MistralSdpaAttention"),
    target_scope="module",
    facets=_ATTENTION_FACETS_SDPA,
)
def gqa_sdpa_attention(module: Any) -> dict[str, Any]:
    """Return facets for GQA-style SDPA attention modules."""

    return gqa_attention(module)


@register(class_name="T5Attention", target_scope="module", facets=_ATTENTION_FACETS_BASE)
def t5_attention(module: Any) -> dict[str, Any]:
    """Return facets for T5 attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(result, "q", reshape_heads(child_out(module, "q"), n_q_heads, d_head))
    add_if_present(result, "k", reshape_heads(child_out(module, "k"), n_kv_heads, d_head))
    add_if_present(result, "v", reshape_heads(child_out(module, "v"), n_kv_heads, d_head))
    add_if_present(result, "attn_out", child_out(module, "o"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(class_name="ViTSelfAttention", target_scope="module", facets=_ATTENTION_FACETS_BASE)
def vit_self_attention(module: Any) -> dict[str, Any]:
    """Return facets for ViT self-attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(result, "q", reshape_heads(child_out(module, "query"), n_q_heads, d_head))
    add_if_present(result, "k", reshape_heads(child_out(module, "key"), n_kv_heads, d_head))
    add_if_present(result, "v", reshape_heads(child_out(module, "value"), n_kv_heads, d_head))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)
