"""Duck-typed ergonomic input coercion for TorchLens entry points."""

from __future__ import annotations

from typing import Any


def _coerce_input(model: Any, x: Any) -> Any:
    """Convert common ergonomic inputs to model-ready tensors.

    Parameters
    ----------
    model:
        Model or model-like object that may expose tokenizer, processor, or
        feature-extractor methods.
    x:
        User-provided input.

    Returns
    -------
    Any
        Coerced input when a supported path is available; otherwise ``x``.
    """

    import torch

    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, str):
        return _tokenize_text(model, x)

    if isinstance(x, list) and x and all(isinstance(item, str) for item in x):
        return _tokenize_text(model, x)

    if _is_pil_image(x) or (isinstance(x, list) and x and all(_is_pil_image(item) for item in x)):
        return _process_image(model, x)

    if _has_explicit_audio_processor(model) and _is_audio_like_input(x):
        return _process_audio(model, x)

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
    except ImportError:
        pass

    return x


def _coerce_input_args(model: Any, input_args: Any) -> Any:
    """Coerce positional model inputs while preserving batched ergonomic inputs.

    Parameters
    ----------
    model:
        Model or model-like object used for duck-typed dispatch.
    input_args:
        Single input or positional input container.

    Returns
    -------
    Any
        Coerced input payload.
    """

    if _is_batched_text_input(input_args) or _is_batched_image_input(input_args):
        return _coerce_input(model, input_args)
    if isinstance(input_args, tuple):
        return tuple(_coerce_input(model, item) for item in input_args)
    if isinstance(input_args, list):
        return [_coerce_input(model, item) for item in input_args]
    return _coerce_input(model, input_args)


def _tokenize_text(model: Any, x: str | list[str]) -> Any:
    """Tokenize text input through duck-typed model methods.

    Parameters
    ----------
    model:
        Model exposing ``to_tokens`` or a callable ``tokenizer``.
    x:
        Text string or batch of strings.

    Returns
    -------
    Any
        Tokenized model input.

    Raises
    ------
    TypeError
        If no supported tokenizer interface is attached to ``model``.
    """

    if hasattr(model, "to_tokens"):
        return model.to_tokens(x)
    if hasattr(model, "tokenizer") and callable(model.tokenizer):
        result = model.tokenizer(x, return_tensors="pt")
        return result.input_ids if hasattr(result, "input_ids") else result
    raise TypeError(
        "String input requires either model.to_tokens(...) (TransformerLens) "
        "or model.tokenizer(...) (HuggingFace with attached tokenizer). "
        "For HuggingFace models without an attached tokenizer, run:\n"
        "    from transformers import AutoTokenizer\n"
        "    model.tokenizer = AutoTokenizer.from_pretrained(<model_name>)\n"
        "or pass tokens directly: model.tokenizer(text, return_tensors='pt').input_ids"
    )


def _process_image(model: Any, x: Any) -> Any:
    """Process PIL image input through a duck-typed model processor.

    Parameters
    ----------
    model:
        Model exposing ``image_processor`` or ``processor``.
    x:
        PIL image or batch of PIL images.

    Returns
    -------
    Any
        Processed image tensor payload.

    Raises
    ------
    TypeError
        If no supported image processor is attached to ``model``.
    """

    proc = getattr(model, "image_processor", None) or getattr(model, "processor", None)
    if proc is None:
        raise TypeError(
            "PIL Image input requires model.image_processor or model.processor. "
            "Attach one: model.image_processor = AutoImageProcessor.from_pretrained(<model_name>)"
        )
    result = proc(x, return_tensors="pt")
    return result.pixel_values if hasattr(result, "pixel_values") else result


def _process_audio(model: Any, x: Any) -> Any:
    """Process raw audio input through a duck-typed model processor.

    Parameters
    ----------
    model:
        Model exposing ``feature_extractor`` or ``processor``.
    x:
        Raw audio input.

    Returns
    -------
    Any
        Processed audio tensor payload.

    Raises
    ------
    TypeError
        If no supported audio processor is attached to ``model``.
    """

    proc = getattr(model, "feature_extractor", None) or getattr(model, "processor", None)
    if proc is None:
        raise TypeError("Audio input requires model.feature_extractor or model.processor.")
    sampling_rate = getattr(model, "sampling_rate", None)
    if sampling_rate is not None:
        result = proc(x, sampling_rate=sampling_rate, return_tensors="pt")
    else:
        result = proc(x, return_tensors="pt")
    return result.input_values if hasattr(result, "input_values") else result


def _is_pil_image(x: Any) -> bool:
    """Check for PIL image input without requiring PIL at import time.

    Parameters
    ----------
    x:
        Candidate object.

    Returns
    -------
    bool
        ``True`` when ``x`` is a PIL image instance.
    """

    try:
        from PIL.Image import Image

        return isinstance(x, Image)
    except ImportError:
        return False


def _is_batched_text_input(x: Any) -> bool:
    """Return whether ``x`` is a non-empty batch of text strings.

    Parameters
    ----------
    x:
        Candidate object.

    Returns
    -------
    bool
        ``True`` for non-empty ``list[str]`` values.
    """

    return isinstance(x, list) and bool(x) and all(isinstance(item, str) for item in x)


def _is_batched_image_input(x: Any) -> bool:
    """Return whether ``x`` is a non-empty batch of PIL images.

    Parameters
    ----------
    x:
        Candidate object.

    Returns
    -------
    bool
        ``True`` for non-empty lists of PIL images.
    """

    return isinstance(x, list) and bool(x) and all(_is_pil_image(item) for item in x)


def _has_explicit_audio_processor(model: Any) -> bool:
    """Return whether ``model`` exposes an audio-specific processor.

    Parameters
    ----------
    model:
        Candidate model object.

    Returns
    -------
    bool
        ``True`` only when a feature extractor is present and no image
        processor is attached.
    """

    return hasattr(model, "feature_extractor") and not hasattr(model, "image_processor")


def _is_audio_like_input(x: Any) -> bool:
    """Return whether ``x`` has a raw-audio container shape.

    Parameters
    ----------
    x:
        Candidate raw audio input.

    Returns
    -------
    bool
        ``True`` for Python sequences and array-like objects.
    """

    return isinstance(x, (list, tuple)) or hasattr(x, "shape")
