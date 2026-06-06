"""Built-in semantic recipes for attention modules."""

from __future__ import annotations

from typing import Any

from ..facets import register
from ._helpers import (
    add_if_present,
    child_output_spec,
    config_value,
    fused_sdpa_facet,
    first_input_spec,
    fused_sdpa_pattern,
    module_output_spec,
    reshape_heads,
)

# Candidate attribute/config names for head counts + head dim. These are looked up
# via config_value (which checks record attrs AND the captured custom_attributes
# snapshot), so ANY module that stores a conventionally-named head count -- HF
# configs, torch.nn.MultiheadAttention (.num_heads), or a custom module using one
# of these names -- infers heads automatically without the recipe hardcoding them.
# (num_heads cannot be recovered from tensor shapes alone: a q-projection's output
# dim is num_heads * d_head, an unfactorable product, so it MUST come from a field.)
_HEAD_COUNT_NAMES = ("n_heads", "num_heads", "num_attention_heads", "n_head", "nhead", "nheads")
_KV_HEAD_COUNT_NAMES = ("num_key_value_heads", "n_kv_heads", "num_kv_heads", "n_kv_head")
_HEAD_DIM_NAMES = ("head_dim", "attention_head_size", "d_kv", "d_head", "head_size")

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
    "scores",
    "pattern",
    "z",
    "result",
)


def _with_attention_common(
    result: dict[str, Any],
    module: Any,
    n_q_heads: int | None,
    n_kv_heads: int | None,
    d_head: int | None,
) -> dict[str, Any]:
    """Attach common attention facets to a recipe result."""

    add_if_present(result, "attn_out", module_output_spec(module, "attention"))
    add_if_present(result, "input", first_input_spec(module, "attention"))
    if n_q_heads is not None:
        result["n_q_heads"] = n_q_heads
        result["n_heads"] = n_q_heads
    if n_kv_heads is not None:
        result["n_kv_heads"] = n_kv_heads
    if d_head is not None:
        result["d_head"] = d_head
    result["head"] = module.facets.head
    if _attention_is_fused(getattr(module, "class_name", "")):
        result["scores"] = fused_sdpa_facet(module, "scores", "attention_reconstruction")
        result["pattern"] = fused_sdpa_pattern(module)
        result["z"] = fused_sdpa_facet(module, "z", "attention_reconstruction")
        result["result"] = fused_sdpa_facet(module, "result", "attention_reconstruction")
    return result


def _attention_is_fused(class_name: str) -> bool:
    """Return whether an attention class hides its pattern behind a fused kernel.

    Covers the explicitly-fused transformers 4.x subclasses (``...SdpaAttention`` /
    ``...FlashAttention2``) and the transformers 5.x unified ``DistilBertSelfAttention``
    -- the 5.x class name no longer encodes the backend, but its default is fused SDPA,
    and the recipe never extracts the pattern in any case, so ``pattern`` is surfaced as
    an informative MissingFacet rather than silently absent (AttributeError).
    """

    return (
        "Sdpa" in class_name
        or "FlashAttention" in class_name
        or class_name == "DistilBertSelfAttention"
    )


def _attention_config(module: Any) -> tuple[int | None, int | None, int | None]:
    """Return ``(n_q_heads, n_kv_heads, d_head)`` from common HF attention configs."""

    cls = getattr(module, "cls", None)
    n_q_heads = config_value(module, *_HEAD_COUNT_NAMES)
    if n_q_heads is None:
        n_q_heads = config_value(cls, *_HEAD_COUNT_NAMES)
    n_kv_heads = config_value(module, *_KV_HEAD_COUNT_NAMES)
    if n_kv_heads is None:
        n_kv_heads = config_value(cls, *_KV_HEAD_COUNT_NAMES)
    if n_kv_heads is None:
        n_kv_heads = n_q_heads
    d_head = config_value(module, *_HEAD_DIM_NAMES)
    if d_head is None:
        d_head = config_value(cls, *_HEAD_DIM_NAMES)
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
    class_name=("DistilBertSdpaAttention", "DistilBertFlashAttention2", "DistilBertSelfAttention"),
    target_scope="module",
    facets=_ATTENTION_FACETS_SDPA,
)
@register(
    class_name="MultiHeadSelfAttention",
    target_scope="module",
    facets=_ATTENTION_FACETS_BASE,
)
def distilbert_attention(module: Any) -> dict[str, Any]:
    """Return facets for DistilBERT attention modules.

    The class name has changed across transformers releases, so match all forms:
    ``MultiHeadSelfAttention`` (eager base, transformers 4.x), the fused
    ``DistilBertSdpaAttention`` / ``DistilBertFlashAttention2`` subclasses (4.x),
    and ``DistilBertSelfAttention`` (transformers 5.x, which unified the per-backend
    subclasses into one class and selects eager/sdpa via a runtime function). All
    forms project q/k/v through the same ``q_lin``/``k_lin``/``v_lin`` children, so
    one extraction body serves every implementation. The explicitly-fused 4.x
    subclasses expose a ``pattern`` MissingFacet (their scores are never
    materialized); the unified/eager classes omit ``pattern`` (the class name no
    longer reveals the backend), matching the other eager recipes (e.g. BERT).
    """

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(
            child_output_spec(module, "q_lin", "distilbert_attention"), n_q_heads, d_head
        ),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(
            child_output_spec(module, "k_lin", "distilbert_attention"), n_kv_heads, d_head
        ),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(
            child_output_spec(module, "v_lin", "distilbert_attention"), n_kv_heads, d_head
        ),
    )
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(class_name="GPT2Attention", target_scope="module", facets=_ATTENTION_FACETS_BASE)
def gpt2_attention(module: Any) -> dict[str, Any]:
    """Return facets for GPT-2 fused-QKV attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    c_attn_out = child_output_spec(module, "c_attn", "gpt2_attention")
    result: dict[str, Any] = {}
    if c_attn_out is not None and n_q_heads is not None:
        q_raw, k_raw, v_raw = c_attn_out.split(3, dim=-1)
        add_if_present(result, "q", reshape_heads(q_raw, n_q_heads, d_head))
        add_if_present(result, "k", reshape_heads(k_raw, n_kv_heads, d_head))
        add_if_present(result, "v", reshape_heads(v_raw, n_kv_heads, d_head))
    add_if_present(result, "attn_out", child_output_spec(module, "c_proj", "gpt2_attention"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(class_name="BertSelfAttention", target_scope="module", facets=_ATTENTION_FACETS_BASE)
def bert_self_attention(module: Any) -> dict[str, Any]:
    """Return facets for BERT self-attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "query", "bert_self_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "key", "bert_self_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(
            child_output_spec(module, "value", "bert_self_attention"), n_kv_heads, d_head
        ),
    )
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
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "q_proj", "gqa_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "k_proj", "gqa_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(child_output_spec(module, "v_proj", "gqa_attention"), n_kv_heads, d_head),
    )
    add_if_present(result, "attn_out", child_output_spec(module, "o_proj", "gqa_attention"))
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
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "q", "t5_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "k", "t5_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(child_output_spec(module, "v", "t5_attention"), n_kv_heads, d_head),
    )
    add_if_present(result, "attn_out", child_output_spec(module, "o", "t5_attention"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(class_name="ViTSelfAttention", target_scope="module", facets=_ATTENTION_FACETS_BASE)
def vit_self_attention(module: Any) -> dict[str, Any]:
    """Return facets for ViT self-attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "query", "vit_self_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "key", "vit_self_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(child_output_spec(module, "value", "vit_self_attention"), n_kv_heads, d_head),
    )
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)
