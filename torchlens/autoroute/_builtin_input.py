"""Built-in input auto-route detectors."""

from __future__ import annotations

from typing import Any, cast

from torchlens.autoroute.input import register
from torchlens.bridge import hf as _hf
from torchlens.data_classes.model_log import Trace


@register(name="hf_text", priority=10)
def hf_text(model: Any, payload: Any, **kwargs: Any) -> Trace | None:
    """Route HuggingFace model + text input through bridge.hf.trace_text.

    Triggers when payload is a string, list of strings, or list of chat-message dicts,
    AND the model has a resolvable name_or_path resolving an AutoTokenizer.
    """

    if not _hf._is_hf_text_input(payload):
        return None
    if not _hf._can_resolve_hf_tokenizer(model):
        return None
    chat_template = isinstance(payload, list) and bool(payload) and isinstance(payload[0], dict)
    return _hf.trace_text(
        model,
        cast("str | list[str] | list[dict[str, Any]]", payload),
        chat_template=chat_template,
        **kwargs,
    )


@register(name="hf_multimodal", priority=20)
def hf_multimodal(model: Any, payload: Any, **kwargs: Any) -> Trace | None:
    """Route HuggingFace model + modality-key dict input through bridge.hf.trace_multimodal."""

    if not _hf._is_hf_multimodal_input(payload):
        return None
    if not _hf._can_resolve_hf_processor(model):
        return None
    return _hf.trace_multimodal(model, cast("dict[str, Any]", payload), **kwargs)


@register(name="hf_image", priority=30)
def hf_image(model: Any, payload: Any, **kwargs: Any) -> Trace | None:
    """Route PIL image input through bridge.hf.trace_image (four-tier preprocessing cascade)."""

    if not _hf._is_hf_image_input(payload):
        return None
    if _hf._has_attached_image_processor(model):
        return None
    return _hf.trace_image(model, payload, **kwargs)
