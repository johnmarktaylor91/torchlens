"""Hugging Face auto-route bridge helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import torch

from torchlens.data_classes.trace import ResolvedPreprocessing, Trace


_MODALITY_KEYS = frozenset({"text", "image", "images", "audio", "videos"})


def _is_hf_text_input(value: Any) -> bool:
    """Return whether ``value`` is a supported Hugging Face text payload.

    Parameters
    ----------
    value:
        Candidate user input passed to ``torchlens.trace``.

    Returns
    -------
    bool
        True for a string, a non-empty list of strings, or a non-empty
        chat-message list with ``role`` and ``content`` keys.
    """

    if isinstance(value, str):
        return True
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return all(isinstance(item, str) for item in value)
        if isinstance(first, dict):
            return all(
                isinstance(item, dict) and "role" in item and "content" in item for item in value
            )
    return False


def _can_resolve_hf_tokenizer(model: Any) -> bool:
    """Return whether ``model`` exposes a non-empty Hugging Face model path.

    Parameters
    ----------
    model:
        Candidate model passed to ``torchlens.trace``.

    Returns
    -------
    bool
        True when ``model.name_or_path`` or ``model.config.name_or_path`` is a
        non-empty string.
    """

    try:
        direct = getattr(model, "name_or_path", None)
    except Exception:
        return False
    if isinstance(direct, str) and direct:
        return True
    try:
        config = getattr(model, "config", None)
        via_config = getattr(config, "name_or_path", None) if config is not None else None
    except Exception:
        return False
    return isinstance(via_config, str) and bool(via_config)


def _is_hf_image_input(value: Any) -> bool:
    """Return whether ``value`` is a conservative PIL image payload.

    Parameters
    ----------
    value:
        Candidate user input passed to ``torchlens.trace``.

    Returns
    -------
    bool
        True for one PIL image or a non-empty list of PIL images.
    """

    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        return False
    if isinstance(value, PILImage):
        return True
    if isinstance(value, list) and value:
        return all(isinstance(item, PILImage) for item in value)
    return False


def _is_hf_multimodal_input(value: Any) -> bool:
    """Return whether ``value`` looks like a Hugging Face multimodal payload.

    Parameters
    ----------
    value:
        Candidate user input passed to ``torchlens.trace``.

    Returns
    -------
    bool
        True for a non-empty dict with modality keys and at least one plausible
        modality value.
    """

    if not isinstance(value, dict) or not value:
        return False
    keys = set(value.keys())
    if not (keys & _MODALITY_KEYS):
        return False
    for key, item in value.items():
        if key in {"image", "images"}:
            try:
                from PIL.Image import Image as PILImage
            except ImportError:
                continue
            if isinstance(item, PILImage):
                return True
            if isinstance(item, list) and item and isinstance(item[0], PILImage):
                return True
        if key == "text" and isinstance(item, str):
            return True
        if key in {"audio", "videos"}:
            return True
    return False


def _can_resolve_hf_processor(model: Any) -> bool:
    """Return whether a Hugging Face ``AutoProcessor`` resolves for ``model``.

    Parameters
    ----------
    model:
        Candidate model passed to ``torchlens.trace``.

    Returns
    -------
    bool
        True when ``transformers.AutoProcessor`` can load a processor for the
        model name/path.
    """

    name_or_path = _model_name_or_path(model)
    if name_or_path is None:
        return False
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(name_or_path)
    except Exception:
        return False
    return True


def _has_attached_image_processor(model: Any) -> bool:
    """Return whether ``model`` exposes the legacy duck-typed image processor.

    Parameters
    ----------
    model:
        Candidate model passed to ``torchlens.trace``.

    Returns
    -------
    bool
        True when normal input coercion should handle PIL images through an
        attached ``image_processor`` or ``processor``.
    """

    return (
        getattr(model, "image_processor", None) is not None
        or getattr(model, "processor", None) is not None
    )


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
    log = tl.trace(model, cast(Any, text), transform=transform, **kwargs)
    log.input_preprocessor = _tokenizer_preprocessing_record(tok, model)
    return log


def trace_image(model: Any, image: Any, **kwargs: Any) -> Trace:
    """Trace an image model with PIL input and resolved preprocessing.

    Parameters
    ----------
    model:
        PyTorch model to trace.
    image:
        PIL image or non-empty list of PIL images.
    **kwargs:
        Additional keyword arguments forwarded to ``torchlens.trace``.

    Returns
    -------
    Trace
        TorchLens trace with ``input_preprocessor`` populated.
    """

    import torchlens as tl

    transform, record = _resolve_image_preprocessing(model)
    log = tl.trace(model, image, transform=_make_image_transform(transform), **kwargs)
    log.input_preprocessor = record
    return log


def trace_multimodal(model: Any, input_dict: dict[str, Any], **kwargs: Any) -> Trace:
    """Trace a Hugging Face multimodal model with dict-keyed input.

    Parameters
    ----------
    model:
        Hugging Face multimodal model with resolvable ``AutoProcessor``.
    input_dict:
        Dict containing modality keys such as ``text`` and ``images``.
    **kwargs:
        Additional keyword arguments forwarded to ``torchlens.trace``.

    Returns
    -------
    Trace
        TorchLens trace with ``input_preprocessor`` populated.
    """

    import torchlens as tl
    from transformers import AutoProcessor

    name_or_path = _model_name_or_path(model)
    if name_or_path is None:
        raise ValueError("Could not auto-resolve AutoProcessor for this model.")
    processor = AutoProcessor.from_pretrained(name_or_path)

    def transform(raw_dict: dict[str, Any]) -> Any:
        """Apply the resolved multimodal processor.

        Parameters
        ----------
        raw_dict:
            User-provided modality dict.

        Returns
        -------
        Any
            Processor output, usually a mapping of tensors.
        """

        return processor(**raw_dict, return_tensors="pt")

    record = ResolvedPreprocessing(
        source="hf_auto_processor",
        identifier=name_or_path,
        verified=True,
        config=_extract_hf_processor_config(processor),
        description=f"AutoProcessor: {name_or_path}",
    )
    log = tl.trace(model, cast(Any, input_dict), transform=transform, **kwargs)
    log.input_preprocessor = record
    return log


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

    name_or_path = _model_name_or_path(model)
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


def _model_name_or_path(model: Any) -> str | None:
    """Return a Hugging Face model name/path if one is exposed.

    Parameters
    ----------
    model:
        Candidate Hugging Face model.

    Returns
    -------
    str | None
        Non-empty model name/path, otherwise None.
    """

    name_or_path = getattr(model, "name_or_path", None) or getattr(
        getattr(model, "config", None), "name_or_path", None
    )
    if isinstance(name_or_path, str) and name_or_path:
        return name_or_path
    return None


def _tokenizer_preprocessing_record(tokenizer: Any, model: Any) -> ResolvedPreprocessing:
    """Build a preprocessing provenance record for a tokenizer.

    Parameters
    ----------
    tokenizer:
        Hugging Face tokenizer-like object.
    model:
        Model used to resolve fallback identifier metadata.

    Returns
    -------
    ResolvedPreprocessing
        Structured tokenizer provenance.
    """

    identifier = getattr(tokenizer, "name_or_path", None) or _model_name_or_path(model) or "unknown"
    config = {
        "tokenizer_name": identifier,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding": True,
        "truncation": False,
    }
    return ResolvedPreprocessing(
        source="hf_auto_tokenizer",
        identifier=str(identifier),
        verified=True,
        config=config,
        description=f"AutoTokenizer: {identifier}",
    )


def _try_hf_image_processor(
    model: Any,
) -> tuple[Callable[[Any], Any], ResolvedPreprocessing] | None:
    """Resolve a Hugging Face image processor for ``model`` if possible.

    Parameters
    ----------
    model:
        Candidate Hugging Face image model.

    Returns
    -------
    tuple[Callable[[Any], Any], ResolvedPreprocessing] | None
        Processor transform and provenance, or None when unavailable.
    """

    name_or_path = _model_name_or_path(model)
    if name_or_path is None:
        return None
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(name_or_path)
    except Exception:
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(name_or_path)
            if not hasattr(processor, "image_processor"):
                return None
        except Exception:
            return None

    def transform(image: Any) -> Any:
        """Apply the resolved image processor.

        Parameters
        ----------
        image:
            PIL image or batch of PIL images.

        Returns
        -------
        Any
            Processor output, usually a mapping with ``pixel_values``.
        """

        return processor(images=image, return_tensors="pt")

    return (
        transform,
        ResolvedPreprocessing(
            source="hf_auto_image_processor",
            identifier=name_or_path,
            verified=True,
            config=_extract_hf_processor_config(processor),
            description=f"AutoImageProcessor: {name_or_path}",
        ),
    )


def _try_torchvision_transforms(
    model: Any,
) -> tuple[Callable[[Any], Any], ResolvedPreprocessing] | None:
    """Resolve torchvision weights preprocessing if attached to ``model``.

    Parameters
    ----------
    model:
        Candidate torchvision model.

    Returns
    -------
    tuple[Callable[[Any], Any], ResolvedPreprocessing] | None
        Transform and provenance, or None when unavailable.
    """

    if not model.__class__.__module__.startswith("torchvision.models"):
        return None
    weights = getattr(model, "_torchlens_weights", None)
    if weights is None:
        return None
    try:
        transform = weights.transforms()
    except Exception:
        return None
    return (
        transform,
        ResolvedPreprocessing(
            source="torchvision_weights",
            identifier=str(weights),
            verified=True,
            config=_extract_torchvision_transform_config(transform),
            description=f"torchvision: {weights}",
        ),
    )


def _try_timm_transforms(model: Any) -> tuple[Callable[[Any], Any], ResolvedPreprocessing] | None:
    """Resolve timm preprocessing for models exposing ``default_cfg``.

    Parameters
    ----------
    model:
        Candidate timm model.

    Returns
    -------
    tuple[Callable[[Any], Any], ResolvedPreprocessing] | None
        Transform and provenance, or None when unavailable.
    """

    default_cfg = getattr(model, "default_cfg", None)
    if not isinstance(default_cfg, dict):
        return None
    try:
        import timm

        data_config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**data_config)
    except Exception:
        return None
    return (
        transform,
        ResolvedPreprocessing(
            source="timm",
            identifier=str(default_cfg.get("architecture", model.__class__.__name__)),
            verified=True,
            config=dict(data_config),
            description=(
                f"timm: input_size={data_config.get('input_size')} "
                f"crop_pct={data_config.get('crop_pct')}"
            ),
        ),
    )


def _imagenet_default_transform() -> tuple[Callable[[Any], Any], ResolvedPreprocessing]:
    """Return the ImageNet default preprocessing fallback.

    Returns
    -------
    tuple[Callable[[Any], Any], ResolvedPreprocessing]
        Default transform and unverified provenance record.
    """

    warnings.warn(
        "torchlens.trace: applying ImageNet default preprocessing (resize 256, "
        "center-crop 224, ImageNet normalization). Model not registered for known "
        "preprocessing; verify this matches your training pipeline. To suppress "
        "this warning, pass transform=... explicitly to tl.trace.",
        UserWarning,
        stacklevel=3,
    )
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return (
        transform,
        ResolvedPreprocessing(
            source="imagenet_default",
            identifier="ImageNet-default-resize256-crop224",
            verified=False,
            config={
                "do_resize": True,
                "size": 256,
                "do_center_crop": True,
                "center_crop_size": 224,
                "do_normalize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            },
            description=(
                "ImageNet default (UNVERIFIED): resize 256 -> center_crop 224 -> "
                "normalize mu=(0.485, 0.456, 0.406) sigma=(0.229, 0.224, 0.225). "
                "Model not registered for known preprocessing."
            ),
        ),
    )


def _resolve_image_preprocessing(model: Any) -> tuple[Callable[[Any], Any], ResolvedPreprocessing]:
    """Resolve image preprocessing through the configured cascade.

    Parameters
    ----------
    model:
        Candidate image model.

    Returns
    -------
    tuple[Callable[[Any], Any], ResolvedPreprocessing]
        Transform and provenance record.
    """

    for tier in (_try_hf_image_processor, _try_torchvision_transforms, _try_timm_transforms):
        result = tier(model)
        if result is not None:
            return result
    return _imagenet_default_transform()


def _make_image_transform(transform: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap image transforms so tensor outputs are batched for model input.

    Parameters
    ----------
    transform:
        PIL-to-tensor or PIL-to-mapping transform.

    Returns
    -------
    Callable[[Any], Any]
        Transform suitable for ``torchlens.trace``.
    """

    def wrapped(image: Any) -> Any:
        """Apply image preprocessing and batch plain tensor outputs.

        Parameters
        ----------
        image:
            PIL image or non-empty list of PIL images.

        Returns
        -------
        Any
            Mapping processor output or batched tensor.
        """

        if isinstance(image, list):
            transformed_items = [transform(item) for item in image]
            if transformed_items and all(
                isinstance(item, torch.Tensor) for item in transformed_items
            ):
                return torch.stack(cast(list[torch.Tensor], transformed_items), dim=0)
            return transform(image)
        transformed = transform(image)
        if isinstance(transformed, torch.Tensor) and transformed.ndim == 3:
            return transformed.unsqueeze(0)
        return transformed

    return wrapped


def _extract_hf_processor_config(processor: Any) -> dict[str, Any]:
    """Extract a best-effort serializable Hugging Face processor config.

    Parameters
    ----------
    processor:
        Hugging Face tokenizer, image processor, or multimodal processor.

    Returns
    -------
    dict[str, Any]
        Public scalar/list/dict config values where available.
    """

    config: dict[str, Any] = {}
    for attr in (
        "name_or_path",
        "model_max_length",
        "do_resize",
        "size",
        "do_center_crop",
        "crop_size",
        "do_normalize",
        "image_mean",
        "image_std",
    ):
        value = getattr(processor, attr, None)
        if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            config[attr] = value
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and image_processor is not processor:
        nested = _extract_hf_processor_config(image_processor)
        config.update({f"image_processor.{key}": value for key, value in nested.items()})
    return config


def _extract_torchvision_transform_config(transform: Any) -> dict[str, Any]:
    """Extract a best-effort torchvision transform config.

    Parameters
    ----------
    transform:
        Torchvision weights transform object.

    Returns
    -------
    dict[str, Any]
        Public scalar/list/tuple/dict fields where available.
    """

    config: dict[str, Any] = {}
    for attr in ("crop_size", "resize_size", "mean", "std", "interpolation", "antialias"):
        value = getattr(transform, attr, None)
        if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            config[attr] = value
    return config


__all__ = [
    "_can_resolve_hf_processor",
    "_can_resolve_hf_tokenizer",
    "_has_attached_image_processor",
    "_is_hf_image_input",
    "_is_hf_multimodal_input",
    "_is_hf_text_input",
    "trace_image",
    "trace_multimodal",
    "trace_text",
]
