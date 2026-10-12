"""Hugging Face auto-route bridge helpers and the TorchLens attention backend.

The attention backend (``register_attention``, ``record_attention``) lives in
``torchlens.bridge.hf_attention`` and is re-exported here.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, cast

import torch

from torchlens._errors import InvalidArgumentError
from torchlens.bridge.hf_attention import (
    AttentionCall,
    AttentionRecording,
    record_attention,
    register_attention,
)
from torchlens.data_classes.trace import ResolvedPreprocessing, Trace

_MODALITY_KEYS = frozenset({"text", "image", "images", "audio", "videos"})


def _is_hf_chat_message(item: Any) -> bool:
    """Return whether ``item`` is a plausible Hugging Face chat-message dict.

    A conservative shape check: a chat message must be a dict whose ``role`` is a
    string and whose ``content`` is a string or a list (the two shapes real chat
    templates accept). Presence of the keys alone is not sufficient -- values
    such as ``None`` or arbitrary objects must decline the text auto-route.

    Parameters
    ----------
    item:
        Candidate chat-message record.

    Returns
    -------
    bool
        True only for a dict with a string ``role`` and a str/list ``content``.
    """

    return (
        isinstance(item, dict)
        and isinstance(item.get("role"), str)
        and isinstance(item.get("content"), (str, list))
    )


def _is_plausible_media_value(value: Any) -> bool:
    """Return whether ``value`` is a plausible audio/video modality payload.

    Conservative acceptance so ``None`` and arbitrary objects decline the
    multimodal auto-route. Accepts tensors, raw bytes, non-empty sequences (raw
    waveforms, frame lists), and array-like objects (e.g. NumPy arrays exposing
    ``__array__``); everything else -- ``None``, bare objects, empty containers,
    scalars -- is rejected.

    Parameters
    ----------
    value:
        Candidate value for an ``audio`` or ``videos`` modality key.

    Returns
    -------
    bool
        True only for a plausibly media-bearing value.
    """

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, (list, tuple)):
        return bool(value)
    return hasattr(value, "__array__")


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
        chat-message list whose entries have a string ``role`` and str/list
        ``content``.
    """

    if isinstance(value, str):
        return True
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return all(isinstance(item, str) for item in value)
        if isinstance(first, dict):
            return all(_is_hf_chat_message(item) for item in value)
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

    # hostile __getattr__ hooks on foreign model objects must never crash the
    # probe; any attribute-read failure simply declines the auto-route
    direct = None
    with contextlib.suppress(Exception):
        direct = getattr(model, "name_or_path", None)
    if isinstance(direct, str) and direct:
        return True
    via_config = None
    with contextlib.suppress(Exception):
        config = getattr(model, "config", None)
        via_config = getattr(config, "name_or_path", None) if config is not None else None
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
        if key in {"audio", "videos"} and _is_plausible_media_value(item):
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

        # Detection must never trigger live Hub network I/O (a plain offline
        # trace would otherwise block through a multi-retry download loop inside
        # routing). Probe the local cache only; an uncached model simply declines
        # the auto-route, matching compat's offline-first stance.
        AutoProcessor.from_pretrained(name_or_path, local_files_only=True)
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

    tokenizer_was_explicit = tokenizer is not None
    tok = tokenizer or _resolve_tokenizer(model)
    structure_only = bool(getattr(kwargs.get("capture"), "structure_only", False))
    transform, transform_state = _make_text_transform(
        tok, chat_template=chat_template, structure_only=structure_only
    )
    capture = tl.options.merge_capture_options(
        capture=kwargs.pop("capture", None),
        transform=transform,
        output_style="hf_text",
    )
    had_tokenizer = hasattr(model, "_torchlens_output_tokenizer")
    previous_tokenizer = getattr(model, "_torchlens_output_tokenizer", None)
    model._torchlens_output_tokenizer = tok
    try:
        log = _trace_text_capture(model, text, capture, structure_only, kwargs)
    finally:
        if had_tokenizer:
            model._torchlens_output_tokenizer = previous_tokenizer
        else:
            delattr(model, "_torchlens_output_tokenizer")
    log.input_preprocessor = _tokenizer_preprocessing_record(
        tok,
        model,
        explicit=tokenizer_was_explicit,
        padding=bool(transform_state["padding"]),
    )
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
    capture = tl.options.merge_capture_options(
        capture=kwargs.pop("capture", None),
        transform=_make_image_transform(transform),
    )
    log = tl.trace(model, image, capture=capture, **kwargs)
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

    from transformers import AutoProcessor

    import torchlens as tl

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
    capture = tl.options.merge_capture_options(
        capture=kwargs.pop("capture", None),
        transform=transform,
    )
    log = tl.trace(model, cast(Any, input_dict), capture=capture, **kwargs)
    log.input_preprocessor = record
    return log


def _trace_text_capture(
    model: Any, text: Any, capture: Any, structure_only: bool, kwargs: dict[str, Any]
) -> Trace:
    """Run the text capture; name the attention remedy on a mask value read.

    Under ``structure_only`` the synthesized all-ones mask reaches transformers'
    mask builder; the sdpa builder reads mask values to decide whether to skip
    the mask, which refuses typed. The refusal is re-raised with the same type
    and code plus the one-line remedy (never an automatic backend switch).
    """

    import torchlens as tl

    if not structure_only:
        return tl.trace(model, cast(Any, text), capture=capture, **kwargs)
    from torchlens.capture.structure_only import ValueDependentBranchError

    try:
        return tl.trace(model, cast(Any, text), capture=capture, **kwargs)
    except ValueDependentBranchError as exc:
        offenses = exc.fields.get("offenses") or ()
        if not any(str(o.get("file", "")).endswith("masking_utils.py") for o in offenses):
            raise
        remedy = (
            "structure-only text capture needs a mask path that reads no values: load the "
            'model with attn_implementation="eager", or call '
            "tl.bridge.hf.register_attention(model)"
        )
        fields = {**exc.fields, "remedy": remedy}
        raise ValueDependentBranchError(
            f"{exc}\nRemedy for text input: {remedy}.", **fields
        ) from exc


def _meta_text_inputs(encoded: Any) -> dict[str, Any]:
    """Return tokenized text on the meta device with an all-ones mask (U38.1).

    The structure-only text policy: the token ids keep only their shape on
    meta and the mask is a meta all-ones mask. A tokenizer mask that is not
    all ones (padded prompts) refuses typed: its pattern would be lost.
    """

    mask = encoded.get("attention_mask")
    if isinstance(mask, torch.Tensor) and not bool(torch.all(mask == 1)):
        raise InvalidArgumentError(
            "structure-only text capture builds an all-ones attention mask, but the "
            "tokenized prompts are padded (their mask is not all ones)",
            code="structure_only_text_mask_not_all_ones",
            remedy="trace one prompt (or prompts of equal token length) at a time",
        )
    meta = {
        key: value.to("meta") if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }
    if isinstance(mask, torch.Tensor):
        meta["attention_mask"] = torch.ones(mask.shape, dtype=mask.dtype, device="meta")
    return meta


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


def _make_text_transform(
    tokenizer: Any, *, chat_template: bool = False, structure_only: bool = False
) -> tuple[Callable[[Any], Any], dict[str, Any]]:
    """Build a tokenizer transform for ``torchlens.trace``.

    Parameters
    ----------
    tokenizer:
        Hugging Face tokenizer-like object.
    chat_template:
        Whether to apply a chat template to message-list inputs before
        tokenization.
    structure_only:
        Whether the capture is structure-only: the tokenized inputs then move
        to the meta device with an all-ones mask (``_meta_text_inputs``).

    Returns
    -------
    tuple[Callable[[Any], Any], dict[str, Any]]
        The transform that maps raw text inputs to model-ready tokenized inputs,
        and a mutable ``state`` dict the transform updates with the ``padding``
        actually applied (``True`` normally, ``False`` on the no-pad-token
        fallback) so provenance can be reported honestly.
    """

    state: dict[str, Any] = {"padding": True}

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
            result = tokenizer(text, return_tensors="pt", padding=True)
            state["padding"] = True
        except ValueError as exc:
            if not (isinstance(original_text, str) and "padding token" in str(exc)):
                raise
            state["padding"] = False
            result = tokenizer(text, return_tensors="pt", padding=False)
        return _meta_text_inputs(result) if structure_only else result

    return transform, state


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


def _tokenizer_preprocessing_record(
    tokenizer: Any, model: Any, *, explicit: bool = False, padding: bool = True
) -> ResolvedPreprocessing:
    """Build a preprocessing provenance record for a tokenizer.

    Parameters
    ----------
    tokenizer:
        Hugging Face tokenizer-like object.
    model:
        Model used to resolve fallback identifier metadata.
    explicit:
        Whether the tokenizer was supplied explicitly by the caller rather than
        auto-resolved from the model's own metadata.
    padding:
        The padding actually applied by the transform (``False`` when the
        no-pad-token fallback fired).

    Returns
    -------
    ResolvedPreprocessing
        Structured tokenizer provenance. ``verified`` is True only when the
        tokenizer was auto-resolved from model metadata AND an identifier was
        actually recovered; ``config['padding']`` reflects the padding used.
    """

    resolved = getattr(tokenizer, "name_or_path", None) or _model_name_or_path(model)
    identifier = resolved or "unknown"
    # ``verified`` means the preprocessing came from model-specific metadata: it
    # cannot be true for an explicit user tokenizer or when no identifier could
    # be resolved at all.
    verified = (not explicit) and resolved is not None
    config = {
        "tokenizer_name": identifier,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding": padding,
        "truncation": False,
    }
    return ResolvedPreprocessing(
        source="hf_auto_tokenizer",
        identifier=str(identifier),
        verified=verified,
        config=config,
        description=f"AutoTokenizer: {identifier}",
    )


def _resolve_image_preprocessing(model: Any) -> tuple[Callable[[Any], Any], ResolvedPreprocessing]:
    """Resolve image preprocessing through the neutral authority resolver.

    Delegates to :func:`torchlens.preprocessing.resolve` (tvscope B2), which
    tries the model-attached no-network tiers first (a planted torchvision
    weights handle, a timm ``default_cfg``) and the network-gated HF
    processor fetch last, with the attempt disclosed on the record; a fetch
    failure yields ``unknown``, never a silently substituted lower tier
    (memo D8). Whatever the registry resolves is applied. When NOTHING
    resolves the capture is refused typed (walk U02.4): TorchLens never
    applies a recipe the model was not shown to use. The refusal names the
    explicit ImageNet-default opt-in.

    Parameters
    ----------
    model:
        Candidate image model.

    Returns
    -------
    tuple[Callable[[Any], Any], ResolvedPreprocessing]
        Transform and provenance record.

    Raises
    ------
    InvalidArgumentError
        ``preprocessing_unresolved`` when no authority resolves.
    """

    from torchlens import preprocessing as _preprocessing

    resolution = _preprocessing.resolve(model=model)
    record = resolution.record
    if resolution.transform is not None and record.source != "unknown":
        if record.source == "hf_image_processor":
            # Legacy source token for the model-discovered HF tier (pinned by
            # the autoroute contract tests since before the neutral resolver).
            record.source = "hf_auto_image_processor"
            record.description = f"AutoImageProcessor: {record.identifier}"
        return resolution.transform, record
    reason = record.config.get("reason") if isinstance(record.config, dict) else None
    raise InvalidArgumentError(
        f"torchlens.trace: no preprocessing authority resolved for the PIL image "
        f"input to {type(model).__name__} ({reason or 'nothing resolved'}). "
        "TorchLens refuses to guess a recipe the model may not have been "
        "trained with.",
        code="preprocessing_unresolved",
        remedy=(
            "pass your loader's own transform, e.g. "
            "capture=tl.options.CaptureOptions(transform=weights.transforms()); "
            "or opt in to the ImageNet default (resize 256, center-crop 224, "
            "ImageNet mean/std) with "
            "capture=tl.options.CaptureOptions(transform=tl.preprocessing.imagenet_default()); "
            "or pass an already-preprocessed tensor"
        ),
        model_type=type(model).__name__,
        resolution_attempt=dict(record.config) if isinstance(record.config, dict) else {},
    )


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
            if not image:
                return transform(image)
            # Batch-native processors (HF image processors) accept the whole list
            # in one call and return a mapping; invoke them exactly once.
            if getattr(transform, "_tl_batch_input", False):
                return transform(image)
            # Otherwise discriminate with a SINGLE probe on the first item rather
            # than transforming every item speculatively and discarding the
            # results. A tensor result means a per-item transform (reuse the probe
            # and transform the rest); a non-tensor result means a batch-native
            # processor, so call it once on the whole list.
            first = transform(image[0])
            if isinstance(first, torch.Tensor):
                rest = [transform(item) for item in image[1:]]
                return torch.stack(cast(list[torch.Tensor], [first, *rest]), dim=0)
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


__all__ = [
    "AttentionCall",
    "AttentionRecording",
    "_can_resolve_hf_processor",
    "_can_resolve_hf_tokenizer",
    "_has_attached_image_processor",
    "_is_hf_image_input",
    "_is_hf_multimodal_input",
    "_is_hf_text_input",
    "record_attention",
    "register_attention",
    "trace_image",
    "trace_multimodal",
    "trace_text",
]
