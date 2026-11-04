"""Duck-typed ergonomic input coercion for TorchLens entry points."""

from __future__ import annotations

import collections.abc
import inspect
import os
from typing import Any

from ._errors import ArgumentTypeError, InvalidArgumentError
from ._input_route import _apply_hf_processor, _route_claims, _route_input, _RoutedInputs
from ._options_removed import REMOVED_CAPTURE_MEMBERS

# Hugging Face processor keyword spellings a modality dict is keyed by
# (``processor(text=..., images=..., audio=..., videos=...)``).
_MODALITY_KEYS = frozenset({"text", "image", "images", "audio", "videos"})

#: File suffixes TorchLens opens as images when a trace input is a path.
_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def _coerce_input(model: Any, x: Any, *, whole_input: bool = True) -> Any:
    """Convert common ergonomic inputs to model-ready tensors.

    Parameters
    ----------
    model:
        Model or model-like object that may expose tokenizer, processor, or
        feature-extractor methods.
    x:
        User-provided input.
    whole_input:
        Whether ``x`` is the entire model input rather than one positional
        item. A text item among other positional inputs (``mode="soft"``) is
        tokenized only when the model exposes a tokenizer; otherwise it is an
        ordinary string argument and passes through unchanged. Text that is
        the whole input still raises the tokenizer guidance when none exists.

    Returns
    -------
    Any
        Coerced input when a supported path is available; otherwise ``x``. A
        modality dict processed by an attached ``processor`` comes back as the
        processor's mapping (keyword inputs), see :func:`_split_coerced_mapping`.
    """

    import torch

    if isinstance(x, torch.Tensor):
        return x

    if whole_input and _route_claims(model, x):
        # The ``tl.trace`` auto-route's resolution (Hugging Face tokenizer,
        # AutoProcessor, image recipe) at every entry, so run, record, debug
        # and validation see the same model inputs as a trace does.
        return _route_input(model, x)

    is_text = isinstance(x, str) or (
        isinstance(x, list) and bool(x) and all(isinstance(item, str) for item in x)
    )
    if is_text:
        if whole_input or _has_text_tokenizer(model):
            return _tokenize_text(model, x)
        return x

    if _is_pil_image(x) or (isinstance(x, list) and x and all(_is_pil_image(item) for item in x)):
        return _process_image(model, x)

    if whole_input and _is_modality_input(x) and _has_attached_processor(model):
        # A paired-modality dict (CLIP/BLIP/LLaVA text+images, Whisper audio)
        # processed by the attached processor; the mapping it returns becomes
        # the forward's keyword inputs (``_split_coerced_mapping``). A dict on
        # a model without a processor is an ordinary positional input.
        return _process_modalities(model, x)

    return _coerce_array_like(model, x)


def _coerce_array_like(model: Any, x: Any) -> Any:
    """Coerce audio-like and NumPy inputs; anything else passes through.

    Parameters
    ----------
    model:
        Model or model-like object.
    x:
        Input no earlier door rule took.

    Returns
    -------
    Any
        Processed audio, a tensor from a NumPy array, or ``x`` unchanged.
    """

    if _has_explicit_audio_processor(model) and _is_audio_like_input(x):
        return _process_audio(model, x)

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            import torch

            return torch.from_numpy(x)
    except ImportError:
        pass

    return x


def _reject_unrouted_forward_kwargs(forward_kwargs: dict[str, Any]) -> None:
    """Refuse unknown ``trace()`` keywords with a typed routing remedy.

    Typed recovery on the natural HF spelling (agent memo 3.10 item 5):
    ``tl.trace(model, input_ids=...)`` used to die with a bare CPython
    TypeError. Unknown keywords are NEVER silently absorbed as forward
    kwargs (a typo'd option must not become a model input); the refusal
    names both working spellings instead.

    Parameters
    ----------
    forward_kwargs:
        The unknown keyword arguments ``trace()`` collected. Empty means
        nothing to refuse and the function returns silently.

    Raises
    ------
    ArgumentTypeError
        With code ``trace_forward_kwargs_unrouted`` when any unknown
        keyword is present.
    """

    if not forward_kwargs:
        return

    unknown = ", ".join(sorted(forward_kwargs))
    removed = [key for key in sorted(forward_kwargs) if key in REMOVED_CAPTURE_MEMBERS]
    if removed:
        # A removed capture option is never a model input: name its replacement.
        message = f"trace() got removed capture option keyword(s): {', '.join(removed)}" + (
            f" (and unrouted keyword(s): {unknown})" if len(removed) < len(forward_kwargs) else ""
        )
        remedy = "; ".join(f"{key}: {REMOVED_CAPTURE_MEMBERS[key]}" for key in removed)
    else:
        message = (
            f"trace() got model-input keyword(s) it does not route: {unknown}. "
            "Forward keyword arguments travel through input_kwargs="
        )
        remedy = (
            "pass them as tl.trace(model, (), input_kwargs={"
            + ", ".join(f"{key!r}: ..." for key in sorted(forward_kwargs))
            + "}); for HuggingFace text models, "
            "torchlens.bridge.transformers.trace_text(model, 'prompt') builds the "
            "inputs for you"
        )
    raise ArgumentTypeError(
        message,
        code="trace_forward_kwargs_unrouted",
        remedy=remedy,
        unrouted_keywords=sorted(forward_kwargs),
    )


def _reject_extra_positional_input(extra_positional: tuple[Any, ...]) -> None:
    """Refuse anything past ``trace()``'s two positional slots.

    Parameters
    ----------
    extra_positional:
        Positional values ``trace()`` received after ``inputs``. Only
        ``model`` and the inputs are positional; a value here is almost always
        an extra positional model input that overflowed past ``inputs``, or
        an ``input_kwargs`` mapping passed by position.

    Raises
    ------
    ArgumentTypeError
        With code ``extra_positional_input_invalid`` when the tuple is not empty.
    """

    if not extra_positional:
        return

    kinds = ", ".join(type(value).__name__ for value in extra_positional)
    raise ArgumentTypeError(
        f"trace() takes only model and inputs by position; received {len(extra_positional)} "
        f"extra positional value(s) ({kinds}). Bundle model inputs as one tuple, e.g. "
        "tl.trace(model, (input_a, input_b, input_c)), and pass keyword inputs as "
        "input_kwargs={...}.",
        code="extra_positional_input_invalid",
        remedy="bundle positional inputs into one tuple; every other argument is a keyword",
    )


def _reject_non_mapping_input_kwargs(input_kwargs: Any) -> None:
    """Refuse a non-Mapping ``input_kwargs`` before any backend dispatch.

    The natural multi-input spelling ``tl.trace(model, a, b)`` lands ``b`` in
    ``input_kwargs``. The torch path refused it typed, but other backends (the
    JAX two-positional spelling) crashed deep inside their capture instead, so
    the check runs at ``trace()`` entry for every backend.

    Parameters
    ----------
    input_kwargs:
        The value ``trace()`` received in its ``input_kwargs`` slot.

    Raises
    ------
    ArgumentTypeError
        With code ``input_kwargs_type_invalid`` when the slot is neither
        ``None`` nor a ``Mapping``.
    """

    import collections.abc

    if input_kwargs is None or isinstance(input_kwargs, collections.abc.Mapping):
        return

    raise ArgumentTypeError(
        "input_kwargs must be a Mapping of keyword arguments for "
        f"model.forward(), got {type(input_kwargs).__name__}. Passing "
        "multiple positional inputs as separate arguments is not "
        "supported: bundle them as one tuple, e.g. "
        "tl.trace(model, (input_a, input_b, input_c)).",
        code="input_kwargs_type_invalid",
        remedy="pass keyword args as a dict, or bundle positional inputs into one tuple",
    )


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

    if (
        _is_batched_text_input(input_args)
        or _is_batched_image_input(input_args)
        or _route_claims(model, input_args)
    ):
        return _coerce_input(model, input_args)
    if isinstance(input_args, tuple):
        # NamedTuple subclasses (e.g. a GNN model's batch container) are also
        # ``tuple`` instances. Reconstruct through the original type so field
        # access (``batch.some_field``) keeps working downstream instead of
        # silently degrading to a plain ``tuple`` (issue: interatomic-potential
        # GNN inputs like orb-models' ``AtomGraphs``). Reconstruction routes
        # through the verified constructor ladder (T11.7): ``_fields`` presence
        # is NOT proof of an *args constructor, and the historical inference
        # crashed plain ``tl.trace`` untyped on a tuple subclass carrying a
        # malformed ``_fields``.
        whole = len(input_args) == 1
        coerced_items = [_coerce_input(model, item, whole_input=whole) for item in input_args]
        if whole and _is_new_mapping(coerced_items[0], input_args[0]):
            return coerced_items[0]
        if all(new is old for new, old in zip(coerced_items, input_args, strict=True)):
            # Nothing coerced: keep the exact original instance (type-faithful,
            # and a fragile subclass constructor is never run needlessly).
            return input_args
        arg_type = type(input_args)
        if arg_type is tuple:
            return tuple(coerced_items)
        from ._utils.arg_handling import rebuild_tuple_like

        rebuilt = rebuild_tuple_like(arg_type, coerced_items)
        if rebuilt is not None:
            return rebuilt
        # Unreconstructable subclass: keep the original rather than crash or
        # silently substitute a different container class; downstream coercion
        # of its items is skipped exactly like other custom wrappers.
        return input_args
    if isinstance(input_args, list):
        whole = len(input_args) == 1
        coerced_list = [_coerce_input(model, item, whole_input=whole) for item in input_args]
        if whole and _is_new_mapping(coerced_list[0], input_args[0]):
            return coerced_list[0]
        return coerced_list
    return _coerce_input(model, input_args)


def _is_new_mapping(coerced: Any, original: Any) -> bool:
    """Return whether coercion turned one whole input into a fresh mapping.

    Parameters
    ----------
    coerced:
        The coerced value of a one-item input container's only item.
    original:
        That item before coercion.

    Returns
    -------
    bool
        ``True`` when the item became a mapping it was not before (a tokenizer's
        or processor's output), which the door routes into keyword inputs.
    """

    return coerced is not original and isinstance(coerced, collections.abc.Mapping)


def _has_text_tokenizer(model: Any) -> bool:
    """Return whether ``model`` exposes a tokenizer :func:`_tokenize_text` can use.

    Parameters
    ----------
    model:
        Model or model-like object.

    Returns
    -------
    bool
        ``True`` for a ``to_tokens`` method or a callable ``tokenizer``.
    """

    return hasattr(model, "to_tokens") or (
        hasattr(model, "tokenizer") and callable(model.tokenizer)
    )


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
        "    model.tokenizer = AutoTokenizer.from_pretrained(<model_class_name>)\n"
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
            "Attach one: model.image_processor = AutoImageProcessor.from_pretrained(<model_class_name>)"
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


def _is_modality_input(x: Any) -> bool:
    """Return whether ``x`` is a non-empty modality dict with a plausible payload.

    The shape test shared by the Hugging Face multimodal auto-route and the
    attached-processor coercion path: a mapping keyed by processor spellings
    (``text``, ``image``/``images``, ``audio``, ``videos``) holding a string,
    a PIL image (or list of them), or a media payload (tensor, bytes, non-empty
    sequence, array-like). ``None`` and arbitrary objects decline.

    Parameters
    ----------
    x:
        Candidate user input.

    Returns
    -------
    bool
        ``True`` only when at least one modality key carries a plausible value.
    """

    if not isinstance(x, collections.abc.Mapping) or not (set(x.keys()) & _MODALITY_KEYS):
        return False
    for key, item in x.items():
        if key in {"image", "images"} and (
            _is_pil_image(item)
            or (isinstance(item, list) and bool(item) and _is_pil_image(item[0]))
        ):
            return True
        if key == "text" and isinstance(item, str):
            return True
        if key in {"audio", "videos"} and _is_plausible_media_value(item):
            return True
    return False


def _is_plausible_media_value(value: Any) -> bool:
    """Return whether ``value`` is a plausible audio/video modality payload.

    Parameters
    ----------
    value:
        Candidate value for an ``audio`` or ``videos`` modality key.

    Returns
    -------
    bool
        ``True`` for tensors, raw bytes, non-empty sequences and array-likes;
        ``False`` for ``None``, scalars, bare objects and empty containers.
    """

    import torch

    if isinstance(value, (torch.Tensor, bytes, bytearray)):
        return True
    if isinstance(value, (list, tuple)):
        return bool(value)
    return hasattr(value, "__array__")


def _has_attached_processor(model: Any) -> bool:
    """Return whether ``model`` carries a callable multimodal ``processor``.

    Parameters
    ----------
    model:
        Model or model-like object.

    Returns
    -------
    bool
        ``True`` for a callable ``model.processor`` attribute.
    """

    return callable(getattr(model, "processor", None))


def _process_modalities(model: Any, x: collections.abc.Mapping[str, Any]) -> Any:
    """Run a modality dict through the attached ``model.processor``.

    Parameters
    ----------
    model:
        Model exposing a callable ``processor`` (``AutoProcessor``-like).
    x:
        Modality dict (``text``, ``images``, ``audio``, ``videos``, ...).

    Returns
    -------
    Any
        The processor's mapping of model-ready tensors (``input_ids``,
        ``pixel_values``, ``input_features``, ...): the forward's keyword inputs.
    """

    return _apply_hf_processor(
        model.processor, x, sampling_rate=getattr(model, "sampling_rate", None)
    )


def _split_coerced_mapping(
    original: Any, coerced: Any, input_kwargs: dict[Any, Any] | None
) -> tuple[Any, dict[Any, Any] | None]:
    """Route a mapping produced by coercion into the forward's keyword inputs.

    Parameters
    ----------
    original:
        The positional input before :func:`_coerce_input_args`.
    coerced:
        Its return value.
    input_kwargs:
        The caller's keyword inputs (``None`` when none were passed).

    Returns
    -------
    tuple[Any, dict[Any, Any] | None]
        ``(input_args, input_kwargs)``. When coercion produced a new mapping
        (an attached processor's output) the positional slot empties and the
        mapping merges under the caller's keywords, which win on a clash;
        otherwise both pass through unchanged.
    """

    if coerced is original or not isinstance(coerced, collections.abc.Mapping):
        return coerced, input_kwargs
    merged: dict[Any, Any] = dict(coerced)
    merged.update(input_kwargs or {})
    return [], merged


def _input_owner(model: Any) -> Any:
    """Return the object whose tokenizer and processors resolve ``model``'s inputs.

    A bound method of a module (``tl.trace(model.generate, "text")``) and the
    synthetic root TorchLens wraps it in both resolve through their owner
    (``method.__self__``), so text, images and modality dicts reach
    ``model.generate`` exactly as they reach ``model``: an attached
    ``tokenizer``, a ``to_tokens`` method or the owner's ``name_or_path``.

    Parameters
    ----------
    model:
        Model, bound method of a module, or TorchLens's bound-method root.

    Returns
    -------
    Any
        The owning module for a bound method or its root; ``model`` otherwise.
    """

    # Duck-typed so this module adds no import edge to the backend layer: the
    # root (``TLBoundMethodRoot``) exposes its owner as the ``tl_owner``
    # property, and a bound method carries it as ``__self__``.
    if isinstance(getattr(type(model), "tl_owner", None), property):
        return model.tl_owner
    if inspect.ismethod(model) and hasattr(model.__self__, "forward"):
        return model.__self__
    return model


def _coerce_inputs(
    model: Any, input_args: Any, input_kwargs: dict[Any, Any] | None
) -> tuple[Any, dict[Any, Any] | None]:
    """Coerce positional inputs and route a processor mapping into the keywords.

    Parameters
    ----------
    model:
        Model or model-like object used for duck-typed dispatch.
    input_args:
        Single input or positional input container.
    input_kwargs:
        The caller's keyword inputs (``None`` when none were passed).

    Returns
    -------
    tuple[Any, dict[Any, Any] | None]
        :func:`_coerce_input_args` applied, then :func:`_split_coerced_mapping`.
    """

    if isinstance(input_args, _RoutedInputs):
        merged: dict[Any, Any] = dict(input_args.input_kwargs or {})
        merged.update(input_kwargs or {})
        return input_args.input_args, merged
    owner = _input_owner(model)
    return _split_coerced_mapping(input_args, _coerce_input_args(owner, input_args), input_kwargs)


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


def _chunk_rerun_plan(model: Any, x: Any, chunk_size: int, chunk_paths: Any) -> tuple[Any, Any]:
    """Coerce a chunked rerun's input through the door and plan its chunks.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Forward input supplied by the caller.
    chunk_size:
        Maximum leading-dimension chunk size.
    chunk_paths:
        Optional explicit tensor leaf paths to split.

    Returns
    -------
    tuple[Any, Any]
        The model-ready positional input and its ``ChunkPlan``.

    Raises
    ------
    ChunkedForwardConfigError
        When the door routed the input into keyword inputs: chunked forwards
        are positional-input only, as at the capture door.
    """

    from ._capture.prepare.chunked import _refuse_chunk_option_conflict, plan_chunks

    input_args, input_kwargs = _coerce_inputs(model, x, None)
    if input_kwargs:
        _refuse_chunk_option_conflict(
            "chunk_size capture is positional-input only in v1; keyword inputs are unsupported.",
            "pass the inputs positionally",
        )
    return input_args, plan_chunks(input_args, chunk_size=chunk_size, chunk_paths=chunk_paths)


def _image_path_candidate(x: Any) -> str | None:
    """Return ``x`` as a path string when it names an image file, else ``None``.

    A ``str`` qualifies only when it carries an image suffix and no
    whitespace (prose never does); any ``os.PathLike`` with an image suffix
    qualifies.
    """

    if isinstance(x, os.PathLike):
        candidate = os.fspath(x)
    elif isinstance(x, str) and not any(char.isspace() for char in x):
        candidate = x
    else:
        return None
    if not isinstance(candidate, str):
        return None
    suffix = os.path.splitext(candidate)[1].lower()
    return candidate if suffix in _IMAGE_SUFFIXES else None


def _open_image_path(path: str) -> Any:
    """Open one image file as an RGB PIL image, refusing typed when it cannot be."""

    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB")
    except (ImportError, OSError) as exc:
        detail = (
            "Pillow is not installed"
            if isinstance(exc, ImportError)
            else f"it could not be decoded ({exc})"
        )
        raise InvalidArgumentError(
            f"torchlens.trace: the input {path!r} names an image file but {detail}.",
            code="image_path_unreadable",
            remedy=(
                "install Pillow (pip install pillow), check the file is a readable "
                "image, or pass an already-loaded image"
            ),
            path=path,
        ) from exc


def _materialize_image_paths(model: Any, x: Any) -> Any:
    """Open image files named by a trace input so they take the image path.

    Parameters
    ----------
    model:
        Model passed to ``trace``; a model with a text tokenizer keeps a
        missing-file string as text.
    x:
        Raw trace input: a path (``str`` or ``os.PathLike``) to an image file,
        a list of them, or anything else.

    Returns
    -------
    Any
        The opened RGB PIL image (or list of them) when ``x`` names existing
        image files; otherwise ``x`` unchanged.

    Raises
    ------
    InvalidArgumentError
        ``image_path_missing`` when ``x`` names an image file that does not
        exist and the model has no tokenizer to read it as text;
        ``image_path_unreadable`` when the file cannot be opened as an image.
    """

    items = x if isinstance(x, list) else [x]
    candidates = [_image_path_candidate(item) for item in items]
    paths = [candidate for candidate in candidates if candidate is not None]
    if not paths or len(paths) != len(candidates):
        return x
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        if isinstance(x, str) and _has_text_tokenizer(model):
            return x
        raise InvalidArgumentError(
            f"torchlens.trace: the input names an image file that does not exist: {missing[0]!r}.",
            code="image_path_missing",
            remedy="pass the path of an existing image file, or an already-loaded image",
            path=missing[0],
        )
    images = [_open_image_path(candidate) for candidate in candidates if candidate is not None]
    return images if isinstance(x, list) else images[0]
