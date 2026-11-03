"""Built-in input auto-route detectors."""

from __future__ import annotations

from typing import Any, cast

from torchlens._autoroute.input import register
from torchlens._data_classes.trace import Trace
from torchlens._input_route import _install_route
from torchlens.bridge import transformers as _hf


@register(name="hf_text", priority=10)
def hf_text(model: Any, payload: Any, **kwargs: Any) -> Trace | None:
    """Route HuggingFace model + text input through bridge.transformers.trace_text.

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
    """Route HuggingFace model + modality-key dict input through bridge.transformers.trace_multimodal."""

    if not _hf._is_hf_multimodal_input(payload):
        return None
    if not _hf._can_resolve_hf_processor(model):
        return None
    return _hf.trace_multimodal(model, cast("dict[str, Any]", payload), **kwargs)


@register(name="hf_image", priority=30)
def hf_image(model: Any, payload: Any, **kwargs: Any) -> Trace | None:
    """Route PIL image input through bridge.transformers.trace_image (four-tier preprocessing cascade)."""

    if not _hf._is_hf_image_input(payload):
        return None
    if _hf._has_attached_image_processor(model):
        return None
    return _hf.trace_image(model, payload, **kwargs)


def hf_input_claims(model: Any, payload: Any) -> bool:
    """Return whether the HF auto-route would claim ``payload`` for ``model``.

    The same claims as ``hf_text``, ``hf_multimodal`` and ``hf_image`` (input
    shape plus what the model can resolve), without resolving anything; the
    input coercion door asks this so ``Trace.run``, ``tl.record`` and the
    ``tl.debug`` entries see the inputs a ``tl.trace`` of the same payload sees.

    Parameters
    ----------
    model:
        Model or model-like object.
    payload:
        Whole user input.

    Returns
    -------
    bool
        ``True`` when :func:`hf_input_transform` would transform ``payload``.
    """

    if _hf._is_hf_text_input(payload):
        return _hf._can_resolve_hf_tokenizer(model)
    if _hf._is_hf_multimodal_input(payload):
        return _hf._can_resolve_hf_processor(model)
    if _hf._is_hf_image_input(payload):
        return not _hf._has_attached_image_processor(model)
    return False


def hf_input_transform(model: Any, payload: Any) -> Any:
    """Build the model inputs the HF auto-route's ``input_transform`` builds.

    Parameters
    ----------
    model:
        Model :func:`hf_input_claims` claimed.
    payload:
        Text (a string, a batch of strings or a chat-message list), a modality
        dict, or PIL images.

    Returns
    -------
    Any
        The tokenizer's mapping, the processor's mapping, or the resolved image
        transform's batched tensor (or mapping), as ``trace_image`` builds it.

    Raises
    ------
    InvalidArgumentError
        ``preprocessing_unresolved`` when the processor or image recipe cannot
        be resolved, as ``trace_multimodal`` and ``trace_image`` refuse.
    """

    if _hf._is_hf_text_input(payload):
        chat_template = isinstance(payload, list) and bool(payload) and isinstance(payload[0], dict)
        transform, _state = _hf._make_text_transform(
            _hf._resolve_tokenizer(model), chat_template=chat_template
        )
        return transform(payload)
    if _hf._is_hf_multimodal_input(payload):
        from torchlens._input_route import _apply_hf_processor
        from torchlens.bridge._hf_processor import _resolve_hf_processor

        processor = _resolve_hf_processor(model, str(_hf._model_name_or_path(model)))
        return _apply_hf_processor(
            processor, payload, sampling_rate=getattr(model, "sampling_rate", None)
        )
    transform, _record = _hf._resolve_image_preprocessing(model)
    return _hf._make_image_transform(transform)(payload)


_install_route(hf_input_claims, hf_input_transform)
