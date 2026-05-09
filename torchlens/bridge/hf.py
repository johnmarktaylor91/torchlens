"""Hugging Face text-input bridge helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torchlens.data_classes.model_log import Trace


def trace_text(
    model: Any,
    text: str | list[str] | list[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    chat_template: bool = False,
    **kwargs: Any,
) -> Trace:
    """Trace a Hugging Face language model with raw text input.

    Convenience wrapper around ``torchlens.trace`` that auto-resolves a
    tokenizer for the model and applies it as the input transform.

    Parameters
    ----------
    model:
        Hugging Face model, or any PyTorch model with a resolvable tokenizer.
    text:
        String, list of strings, or list of message dictionaries when
        ``chat_template=True``.
    tokenizer:
        Optional explicit tokenizer. Defaults to auto-resolving from
        ``model.name_or_path`` or ``model.config.name_or_path``.
    chat_template:
        When True and text is a chat-message list, apply the tokenizer's chat
        template before tokenization.
    **kwargs:
        Additional keyword arguments forwarded to ``torchlens.trace``.

    Returns
    -------
    Trace
        TorchLens trace with ``raw_input`` set to the original text.
    """

    import torchlens as tl

    tok = tokenizer or _resolve_tokenizer(model)
    transform = _make_text_transform(tok, chat_template=chat_template)
    return tl.trace(model, cast(Any, text), transform=transform, **kwargs)


def _resolve_tokenizer(model: Any) -> Any:
    """Resolve a tokenizer for a Hugging Face model.

    Parameters
    ----------
    model:
        Hugging Face model, or a PyTorch model exposing ``name_or_path`` on
        itself or on its ``config`` object.

    Returns
    -------
    Any
        Tokenizer returned by ``transformers.AutoTokenizer.from_pretrained``.

    Raises
    ------
    ImportError
        If ``transformers`` is unavailable.
    ValueError
        If no model name or path can be found.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "tl.bridge.hf.trace_text requires the `transformers` package. "
            "Install with `pip install transformers`."
        ) from exc

    name_or_path = getattr(model, "name_or_path", None) or getattr(
        getattr(model, "config", None), "name_or_path", None
    )
    if name_or_path is None:
        raise ValueError(
            "Could not auto-resolve tokenizer for this model. "
            "Pass `tokenizer=` explicitly to tl.bridge.hf.trace_text."
        )
    return AutoTokenizer.from_pretrained(name_or_path)


def _make_text_transform(tokenizer: Any, *, chat_template: bool = False) -> Callable[[Any], Any]:
    """Build a tokenizer transform for ``torchlens.trace``.

    Parameters
    ----------
    tokenizer:
        Hugging Face tokenizer-like object.
    chat_template:
        Whether to apply a chat template to message-list inputs before
        tokenization.

    Returns
    -------
    Callable[[Any], Any]
        Transform that maps raw text inputs to model-ready tokenized inputs.
    """

    def transform(text: Any) -> Any:
        """Tokenize one raw text payload.

        Parameters
        ----------
        text:
            String, batch of strings, or chat-message list.

        Returns
        -------
        Any
            Tokenizer output, usually a Hugging Face ``BatchEncoding``.
        """

        original_text = text
        if chat_template and isinstance(text, list) and text and isinstance(text[0], dict):
            text = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        try:
            return tokenizer(text, return_tensors="pt", padding=True)
        except ValueError as exc:
            if isinstance(original_text, str) and "padding token" in str(exc):
                return tokenizer(text, return_tensors="pt", padding=False)
            raise

    return transform


__all__ = ["trace_text"]
