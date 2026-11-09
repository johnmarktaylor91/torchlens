"""Built-in output auto-route detectors and decoders."""

from __future__ import annotations

from dataclasses import replace
from importlib import resources
from typing import Any, NoReturn

import torch

from torchlens._autoroute._decode_heads import (
    _decode_hf_text,
    _normalized_id2label,
    _normalized_mapping,
    _normalized_sequence,
    _safe_int,
    _safe_str,
    _select_tensor_by_head,
    _softmax_topk,
    _tensor_candidates,
)
from torchlens._autoroute._decode_spellings import (
    HEAD_KW,
    STYLE_KW,
    UNDETECTED_REMEDY,
)
from torchlens._autoroute.output import list as _registered_output_detectors, register
from torchlens._capture._episode_derivation import _render_container_path
from torchlens._capture_fingerprint import _never_matching_fragment
from torchlens._data_classes.trace import ResolvedPostprocessing, Trace
from torchlens._errors import InvalidArgumentError, PayloadUnavailableError
from torchlens._ir.container_registry import _is_hf_model_output
from torchlens._utils._display_errors import lookup_message
from torchlens._utils.torchvision_weights import match_torchvision_weights

DETECTOR_VERSION = "torchlens-output-a2-v1"
#: Styles ``Trace.decode_output(style=...)`` resolves itself; a registered
#: custom detector may resolve others (docs/semantic_io.md, Custom Output Detectors).
BUILTIN_OUTPUT_STYLES: tuple[str, ...] = ("classification", "imagenet", "imagenet1k", "hf_text")
#: The built-in detector names (everything else in the registry is custom).
_BUILTIN_DETECTOR_NAMES = frozenset({"hf_config_classifier", "imagenet_verified", "hf_text"})
IMAGENET_LABEL_SOURCE = "torchvision ResNet50_Weights.DEFAULT.meta['categories']"
IMAGENET_LABEL_SOURCE_VERSION = "torchvision-imagenet1k-categories-v1"
_TOP_N_CAPTURED = 5


def load_imagenet1k_labels() -> list[str]:
    """Load the shipped ImageNet-1k label bank.

    Returns
    -------
    list[str]
        Ordered ImageNet-1k display labels.
    """

    label_text = (
        resources.files("torchlens._autoroute.data")
        .joinpath("imagenet1k_labels.json")
        .read_text(encoding="utf-8")
    )
    import json

    labels = json.loads(label_text)
    return [str(label) for label in labels]


def decode_outputs_for_trace(trace: Trace, outputs: Any) -> None:
    """Decode live model outputs into ``trace._decoded_output`` when safely resolved.

    This is the automatic, option-free pass that runs once at the end of every
    capture: the registered output detectors inspect the live output object
    and the trace's output metadata, and the first match decodes. A user who
    wants a different style or head decodes again at read time with
    ``trace.decode_output(style=..., head=...)``.

    Parameters
    ----------
    trace:
        Trace receiving decoded JSON-primitive rows and provenance.
    outputs:
        Live model forward return value, before TorchLens output extraction.
    """

    try:
        _decode_outputs_for_trace_unguarded(trace, outputs)
    except Exception as exc:
        # The output decode is an opportunistic post-capture nicety; a
        # heuristic sniffer must never abort an otherwise-successful capture
        # (R65: nested-tensor outputs raised raw RuntimeError from shape
        # reads). Degrade to "no decode" -- but leave a DURABLE record, not
        # just a warning: the warning is the most losable disclosure kind,
        # and without the annotation a failed decode was byte-
        # indistinguishable from "decode not applicable" on the returned AND
        # saved Trace (sibling of the zero-match ledger).
        # Roll back the partial-write window first: the unguarded body
        # assigns decoded_output BEFORE output_postprocessor, so a raise
        # between the two would otherwise leave a decoded output with no
        # provenance record while the message claims none was attached.
        import warnings

        from torchlens.errors import TorchLensWarning

        trace._decoded_output = None
        trace._provenance_output_postprocessor = None
        annotations = getattr(trace, "annotations", None)
        if isinstance(annotations, dict):
            annotations.setdefault("decode_skipped", []).append(
                {"error": f"{type(exc).__name__}: {exc}"}
            )
        warnings.warn(
            "Semantic output decode skipped: the output detector raised "
            f"{type(exc).__name__}: {exc}. The capture itself is unaffected; "
            "no decoded_output/output_postprocessor was attached "
            "(durable record: trace.annotations['decode_skipped']).",
            TorchLensWarning,
            stacklevel=2,
        )


def _decode_outputs_for_trace_unguarded(trace: Trace, outputs: Any) -> None:
    """Run the automatic output decode without the capture-protecting belt."""

    meta = _build_output_meta(trace, style=None, head=None)
    resolved, _detector_name = _resolve_postprocessor_named(outputs, meta)
    if resolved is None:
        return

    logits_info = _select_logits(outputs, resolved, meta)
    if logits_info is None:
        if resolved.verified:
            trace._provenance_output_postprocessor = replace(
                resolved,
                ambiguous=True,
                description=f"decode: undetected; {UNDETECTED_REMEDY}.",
            )
        return

    logits, selected_head = logits_info
    decoded = _decode_selected(logits, resolved, meta, top_n=_TOP_N_CAPTURED)
    if decoded is None:
        return
    trace._decoded_output = decoded
    trace._provenance_output_postprocessor = replace(
        resolved,
        selected_output_head=selected_head,
        top_n_captured=_TOP_N_CAPTURED,
    )


def _decode_selected(
    logits: torch.Tensor, resolved: ResolvedPostprocessing, meta: dict[str, Any], *, top_n: int
) -> Any | None:
    """Decode one selected tensor under a resolved style into JSON rows."""

    if resolved.style == "hf_text":
        return _decode_hf_text(logits, meta, top_n=top_n)
    labels = _labels_for_resolved(resolved)
    if labels is None:
        return None
    rows = _decode_classification(logits, labels, top_n=top_n)
    if rows is None:
        return None
    return {"kind": "batch_topk", "rows": rows}


def decode_trace_output(
    trace: Trace, *, style: str | None, head: str | None, tokenizer: Any | None, top_n: int
) -> Any:
    """Decode a finished trace's retained outputs on demand; never mutates the trace.

    The read-time door behind ``Trace.decode_output(style=, head=, tokenizer=)``,
    which documents the arguments; ``top_n`` caps the rows per batch item for
    the classification styles.

    Raises
    ------
    PayloadUnavailableError
        ``decoded_output_unavailable``: nothing retained, or nothing decodable.
    InvalidArgumentError
        ``decode_head_unknown``, ``decode_head_ambiguous``, ``decode_style_unknown``.
    """

    view, heads = _retained_output_view(trace)
    heads_text = "the single output tensor (no named heads)" if heads == ("",) else ", ".join(heads)
    if view is None:
        raise PayloadUnavailableError(
            "no output tensor retained; re-decode unavailable",
            code="decoded_output_unavailable",
            remedy="keep the output activations retained (the default save policy) to re-decode",
        )
    if head is not None and _select_tensor_by_head(view, head) is None:
        raise InvalidArgumentError(
            f"{HEAD_KW}={head!r} is not an output head of this trace; "
            f"available heads: {heads_text}",
            code="decode_head_unknown",
            remedy=f"pass one of the available heads as {HEAD_KW}=",
            **_head_choice(heads),
        )
    meta, resolved = _resolve_read_time_style(trace, view, style, head, tokenizer)
    logits_info = _select_logits(view, resolved, meta)
    if logits_info is None:
        raise InvalidArgumentError(
            f"several output heads qualify for {STYLE_KW}={resolved.style!r} and none was "
            f"named; available heads: {heads_text}",
            code="decode_head_ambiguous",
            remedy=f"pass {HEAD_KW}= naming one of the available heads",
            **_head_choice(heads),
        )
    decoded = _decode_selected(logits_info[0], resolved, meta, top_n=top_n)
    if decoded is None and resolved.style == "hf_text":
        raise DecodeStyleUnknownError(
            f"{STYLE_KW}='hf_text' needs rank-2 [seq, vocab] or rank-3 [batch, seq, vocab] logits",
            code="decode_style_unknown",
            remedy=f"pass {HEAD_KW}= the logits head of this trace",
            argument=HEAD_KW,
        )
    if decoded is None:
        raise PayloadUnavailableError(
            f"the retained output does not decode as {resolved.style!r} "
            "(no labels for its width, or not a rank-1 or rank-2 logits tensor)",
            code="decoded_output_unavailable",
            remedy=f"pass {HEAD_KW}= the logits head, or a {STYLE_KW} whose labels match",
        )
    return decoded


def _head_choice(heads: tuple[str, ...]) -> dict[str, Any]:
    """Return the ``argument``/``available_heads`` fields of a head refusal."""

    return {"argument": HEAD_KW, "available_heads": list(heads)}


def _resolve_read_time_style(
    trace: Trace, view: Any, style: str | None, head: str | None, tokenizer: Any | None
) -> tuple[dict[str, Any], ResolvedPostprocessing]:
    """Build the read-time metadata and resolve the decode style for ``view``.

    A requested built-in style resolves directly; otherwise the registered
    detectors run over the retained view.

    Raises
    ------
    DecodeStyleUnknownError
        When neither the requested style nor any detector resolves one.
    """

    meta = _build_output_meta(trace, style=style, head=head, tokenizer=tokenizer)
    resolved = _resolve_requested_style(meta)
    if resolved is None:
        skip = _BUILTIN_DETECTOR_NAMES if style is not None else frozenset()
        resolved, _detector_name = _resolve_postprocessor_named(view, meta, skip=skip)
    if resolved is None:
        refuse_unknown_decode_style(style)
    return meta, resolved


def _retained_output_view(trace: Trace) -> tuple[Any | None, tuple[str, ...]]:
    """Rebuild a head-addressable view of the retained output tensors.

    Returns the single tensor when the model returned one bare tensor, else a
    dict keyed by the rendered output container path (``"logits"``, ``"0"``,
    ``"0.logits"``), plus the tuple of available head names. ``(None, ())``
    when no output tensor was retained.
    """

    tensors: dict[str, torch.Tensor] = {}
    for op in getattr(trace, "output_ops", []) or []:
        out = getattr(op, "out", None)
        if not isinstance(out, torch.Tensor):
            continue
        path = _render_container_path(getattr(op, "container_path", ()))
        tensors.setdefault(path, out)
    if not tensors:
        return None, ()
    if len(tensors) == 1 and "" in tensors:
        return tensors[""], ("",)
    return tensors, tuple(tensors)


def _resolve_requested_style(meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Resolve a user-requested built-in style from the read-time metadata."""

    style = meta.get(STYLE_KW)
    if style is None:
        return None
    if style in {"classification", "imagenet", "imagenet1k"}:
        id2label = meta.get("id2label")
        return ResolvedPostprocessing(
            source="user",
            identifier=str(style),
            verified=True,
            config={"id2label": id2label} if isinstance(id2label, dict) else {},
            description=f"user-requested output decode: {style}",
            style="imagenet" if style in {"imagenet", "imagenet1k"} else "classification",
            selected_output_head=meta.get(HEAD_KW),
            label_source="config.id2label" if style == "classification" else IMAGENET_LABEL_SOURCE,
            label_source_version=DETECTOR_VERSION
            if style == "classification"
            else IMAGENET_LABEL_SOURCE_VERSION,
            confidence=1.0,
        )
    if style == "hf_text":
        return ResolvedPostprocessing(
            source="user",
            identifier="hf_text",
            verified=True,
            config={},
            description="user-requested Hugging Face text decode",
            style="hf_text",
            selected_output_head=meta.get(HEAD_KW),
            label_source="tokenizer",
            label_source_version=DETECTOR_VERSION,
            confidence=1.0,
        )
    return None


def output_decoders_cache_key() -> tuple[tuple[str, int, int], ...]:
    """Return the registered output detectors as a capture-cache fragment.

    Registered detectors label the capture-time decode, so a cached trace made
    before a register or unregister must not serve the next capture.

    Returns
    -------
    tuple[tuple[str, int, int], ...]
        ``(name, priority, registration_order)`` per registered detector, in
        dispatch order; re-registering a name changes its order and the key.
    """

    return tuple(
        (detector.name, detector.priority, detector.registration_order)
        for detector in _registered_output_detectors()
    )


def semantic_output_cache_key(model: Any) -> dict[str, Any]:
    """Return a serializable semantic-output cache fingerprint.

    Parameters
    ----------
    model:
        Model passed to ``trace``.

    Returns
    -------
    dict[str, Any]
        Stable-ish metadata that affects output auto-detection and labels.
    """

    # This fingerprint participates in the capture-cache
    # EQUALITY key (user_funcs folds it under cache=True), so a raising
    # ``config``/``default_cfg`` getter must NOT silently drop the whole
    # config axis -- two models differing only in config (id2label,
    # num_labels: the axis this key exists for) would collide and the second
    # capture would serve the FIRST's cached Trace with the first's decoded
    # labels. A raising getter mints the never-matching token instead
    # (false hits never; the only cost is a cache miss).
    config_unreadable: object | None = None
    try:
        config = getattr(model, "config", None)
    except Exception:
        config = None
        config_unreadable = _never_matching_fragment("model-config-getter-raised")
    try:
        default_cfg = getattr(model, "default_cfg", None)
    except Exception:
        default_cfg = None
        config_unreadable = _never_matching_fragment("model-default-cfg-getter-raised")
    weights = getattr(model, "_torchlens_weights", None)
    if weights is None:
        # No handle planted: a torchvision model still names its preset when
        # its loaded state equals exactly one cached preset checkpoint.
        weights = match_torchvision_weights(model)
    return {
        "version": DETECTOR_VERSION,
        "config_unreadable": config_unreadable,
        "config": {
            "id2label": _normalized_id2label(getattr(config, "id2label", None)),
            "label2id": _normalized_mapping(getattr(config, "label2id", None)),
            "num_labels": _safe_int(getattr(config, "num_labels", None)),
            "model_type": _safe_str(getattr(config, "model_type", None)),
            "architectures": _normalized_sequence(getattr(config, "architectures", None)),
        },
        "torchvision": _torchvision_weights_cache_fragment(model, weights),
        "timm": {
            "architecture": _safe_str(default_cfg.get("architecture"))
            if isinstance(default_cfg, dict)
            else None,
            "num_classes": _safe_int(default_cfg.get("num_classes"))
            if isinstance(default_cfg, dict)
            else None,
        },
    }


def _build_output_meta(
    trace: Trace,
    *,
    style: str | None,
    head: str | None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Build detector metadata from captured trace fields.

    Parameters
    ----------
    trace:
        Trace with in-band output metadata.
    style:
        Requested style (read time) or ``None`` (the automatic capture pass).
    head:
        Requested output head (read time) or ``None``.
    tokenizer:
        Tokenizer passed at read time; at capture time the bridge-attached
        session tokenizer is read from the trace instead.

    Returns
    -------
    dict[str, Any]
        Detector metadata.
    """

    return {
        STYLE_KW: style,
        HEAD_KW: head,
        "id2label": getattr(trace, "output_id2label", None),
        "num_classes": getattr(trace, "output_num_classes", None),
        "tokenizer": tokenizer
        if tokenizer is not None
        else getattr(trace, "_output_tokenizer", None),
        "model_metadata": getattr(trace, "_semantic_output_metadata", None),
        "model_class_name": trace._source_class_name,
    }


class DecodeStyleUnknownError(InvalidArgumentError):
    """``decode_output(style=...)`` names a style no built-in or registered detector resolves.

    Examples
    --------
    >>> issubclass(DecodeStyleUnknownError, tl.errors.InvalidArgumentError)
    True
    >>> issubclass(DecodeStyleUnknownError, tl.errors.TorchLensError)
    True
    """


def refuse_unknown_decode_style(style: Any) -> NoReturn:
    """Raise the typed did-you-mean refusal for an unresolvable decode style.

    Parameters
    ----------
    style:
        The requested style, or ``None`` when no style was given and no
        registered detector resolved one from the retained outputs.

    Raises
    ------
    DecodeStyleUnknownError
        Always, with code ``decode_style_unknown``.
    """

    if style is None:
        raise DecodeStyleUnknownError(
            "no output detector resolved a decode style from the retained outputs. "
            f"Built-in styles: {', '.join(BUILTIN_OUTPUT_STYLES)}.",
            code="decode_style_unknown",
            remedy=f"pass {STYLE_KW}= a built-in style or a registered custom style",
            argument=STYLE_KW,
        )
    raise DecodeStyleUnknownError(
        lookup_message(
            f"unknown {STYLE_KW}={style!r}; no built-in or registered custom output detector resolved it.",
            list(BUILTIN_OUTPUT_STYLES),
            "pass a built-in style, or register a custom output detector.",
            query=str(style),
        ),
        code="decode_style_unknown",
        remedy=(
            "pass a built-in style, or register a custom detector that resolves it "
            "with tl.postprocessing.register(...)"
        ),
        argument=STYLE_KW,
    )


def _resolve_postprocessor_named(
    outputs: Any, meta: dict[str, Any], *, skip: frozenset[str] = frozenset()
) -> tuple[ResolvedPostprocessing | None, str | None]:
    """Resolve the first matching detector not named in ``skip`` and name the one that matched."""

    from . import output as autoroute_output

    for detector in autoroute_output.list():
        if detector.name in skip:
            continue
        resolved = detector.func(outputs, meta)
        if resolved is not None:
            return resolved, detector.name
    return None, None


@register(name="hf_config_classifier", priority=10)
def hf_config_classifier(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Detect classifiers with explicit ``config.id2label`` metadata."""

    id2label = meta.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        return None
    resolved = ResolvedPostprocessing(
        source="hf_config",
        identifier="config.id2label",
        verified=True,
        config={"num_labels": meta.get("num_classes"), "id2label": id2label},
        description="HF config labels",
        style="classification",
        selected_output_head=meta.get(HEAD_KW),
        label_source="config.id2label",
        label_source_version=DETECTOR_VERSION,
        confidence=0.99,
    )
    if not _labels_cover_selected_logits(outputs, resolved, meta, id2label):
        return None
    return resolved


def _labels_cover_selected_logits(
    outputs: Any, resolved: ResolvedPostprocessing, meta: dict[str, Any], id2label: dict[Any, Any]
) -> bool:
    """Return whether ``id2label`` can label the logits this record would select.

    Every Hugging Face config carries a placeholder ``id2label`` (two
    ``LABEL_n`` entries) even on language models, whose ``logits`` are a
    rank-3 vocabulary-width tensor no label table covers. Claiming those
    outputs would block the tokenizer-backed text decode behind them, so the
    classifier detector declines when the selected logits are not a rank-1 or
    rank-2 tensor at most ``len(id2label)`` wide. An ambiguous head choice
    (no single logits tensor) keeps the claim: the ambiguity is disclosed on
    the record instead.
    """

    logits_info = _select_logits(outputs, resolved, meta)
    if logits_info is None:
        return True
    logits = logits_info[0]
    width = _safe_last_dim(logits)
    if width is None:
        return True
    try:
        rank = int(logits.ndim)
    except Exception:
        return True
    return rank in (1, 2) and width <= len(id2label)


@register(name="imagenet_verified", priority=20)
def imagenet_verified(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Detect verified ImageNet-width logits from live outputs only."""

    if meta.get(STYLE_KW) not in {None, "imagenet", "imagenet1k"}:
        return None
    logits = _select_tensor_by_head(outputs, meta.get(HEAD_KW))
    if logits is None and isinstance(outputs, torch.Tensor):
        logits = outputs
    if logits is None or _safe_last_dim(logits) != 1000:
        return None
    metadata = meta.get("model_metadata")
    if not isinstance(metadata, dict):
        return None
    torchvision_meta = metadata.get("torchvision")
    timm_meta = metadata.get("timm")
    verified_sources: list[str] = []
    identifier = "imagenet1k"
    if isinstance(torchvision_meta, dict):
        model_module = torchvision_meta.get("model_module")
        if (
            isinstance(model_module, str)
            and model_module.startswith("torchvision.models")
            and torchvision_meta.get("weights_id") is not None
            and torchvision_meta.get("categories_len") == 1000
        ):
            verified_sources.append("torchvision_weights")
            identifier = str(torchvision_meta.get("weights_id"))
    if isinstance(timm_meta, dict) and timm_meta.get("num_classes") == 1000:
        verified_sources.append("timm_default_cfg")
        identifier = str(timm_meta.get("architecture") or identifier)
    if len(set(verified_sources)) != 1:
        return None
    return ResolvedPostprocessing(
        source=verified_sources[0],
        identifier=identifier,
        verified=True,
        config={},
        description="verified ImageNet-1k output labels",
        style="imagenet",
        selected_output_head=meta.get(HEAD_KW),
        label_source=IMAGENET_LABEL_SOURCE,
        label_source_version=IMAGENET_LABEL_SOURCE_VERSION,
        confidence=0.95,
    )


@register(name="hf_text", priority=30)
def hf_text(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Detect text logits when a tokenizer is attached and a ``logits`` head exists.

    The transformers bridge attaches the tokenizer for the span of its own
    capture, so an attached tokenizer means "this is a text model traced as
    text". Only a head named ``logits`` (or the head the caller named) is
    decoded (plus one bare output tensor): an encoder's ``last_hidden_state``
    must never be argmax-decoded as token ids.
    """

    if meta.get(STYLE_KW) not in {None, "hf_text"}:
        return None
    tokenizer = meta.get("tokenizer")
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return None
    head = meta.get(HEAD_KW)
    if head is None and isinstance(outputs, torch.Tensor):
        head = None  # one bare tensor: the whole output is the logits
    elif _select_tensor_by_head(outputs, head or "logits") is None:
        return None
    else:
        head = head or "logits"
    return ResolvedPostprocessing(
        source="attached_tokenizer",
        identifier="hf_text",
        verified=True,
        config={},
        description="Hugging Face text decode with the attached tokenizer",
        style="hf_text",
        selected_output_head=head,
        label_source="tokenizer",
        label_source_version=DETECTOR_VERSION,
        confidence=0.9,
    )


def _select_logits(
    outputs: Any, resolved: ResolvedPostprocessing, meta: dict[str, Any]
) -> tuple[torch.Tensor, str] | None:
    """Select one logits tensor from live outputs, failing closed on ambiguity.

    Parameters
    ----------
    outputs:
        Live model outputs.
    resolved:
        Resolved postprocessor.
    meta:
        Detector metadata.

    Returns
    -------
    tuple[torch.Tensor, str] | None
        Selected tensor and head label, or None when ambiguous.
    """

    requested = resolved.selected_output_head or meta.get(HEAD_KW)
    by_head = _select_tensor_by_head(outputs, requested)
    if by_head is not None:
        return by_head, str(requested or "output")
    logits = _named_logits(outputs)
    if logits is not None:
        return logits, "logits"
    if isinstance(outputs, torch.Tensor):
        return outputs, "output"
    expected = resolved.config.get("num_labels") or meta.get("num_classes")
    candidates = _tensor_candidates(outputs, expected_width=_safe_int(expected))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _named_logits(outputs: Any) -> torch.Tensor | None:
    """Return the output's ``logits`` tensor: attribute, mapping key or HF output item.

    Parameters
    ----------
    outputs:
        Live model outputs or the read-time retained view.

    Returns
    -------
    torch.Tensor | None
        The ``logits`` tensor, or ``None`` when the output names none.
    """

    if hasattr(outputs, "logits"):
        logits = getattr(outputs, "logits")
        if isinstance(logits, torch.Tensor):
            return logits
    if isinstance(outputs, dict) and isinstance(outputs.get("logits"), torch.Tensor):
        # The read-time view keys retained output tensors by container path.
        return outputs["logits"]
    if _is_hf_model_output(outputs):
        try:
            logits = outputs["logits"]
        except Exception:
            logits = None
        if isinstance(logits, torch.Tensor):
            return logits
    return None


def _labels_for_resolved(resolved: ResolvedPostprocessing) -> list[str] | None:
    """Return labels for a resolved postprocessor.

    Parameters
    ----------
    resolved:
        Resolved output postprocessor.

    Returns
    -------
    list[str] | None
        Ordered labels, if available.
    """

    if resolved.style == "imagenet":
        return load_imagenet1k_labels()
    id2label = resolved.config.get("id2label")
    if isinstance(id2label, dict):
        return [str(id2label[index]) for index in sorted(id2label)]
    return None


def _safe_last_dim(tensor: torch.Tensor) -> int | None:
    """Return the last-dimension size, or ``None`` when unreadable.

    Nested tensors raise ``RuntimeError`` from ``.shape`` and other exotic
    subclasses may refuse metadata reads entirely; an opportunistic output
    sniffer treats those as "not logits" rather than aborting capture.
    """

    try:
        if tensor.is_nested or tensor.ndim < 1:
            return None
        return int(tensor.shape[-1])
    except Exception:
        return None


def _decode_classification(
    logits: torch.Tensor, labels: list[str], *, top_n: int
) -> list[dict[str, Any]] | None:
    """Decode classification logits into bounded JSON rows.

    Parameters
    ----------
    logits:
        Live logits tensor.
    labels:
        Label display strings indexed by class id.
    top_n:
        Maximum rows per batch item.

    Returns
    -------
    list[dict[str, Any]] | None
        JSON-primitive rows.
    """

    if logits.ndim == 1:
        logits_copy = logits.detach().clone().unsqueeze(0)
    elif logits.ndim == 2:
        logits_copy = logits.detach().clone()
    else:
        return None
    if logits_copy.shape[-1] > len(labels):
        return None
    values, indices, k = _softmax_topk(logits_copy, top_n)
    rows: list[dict[str, Any]] = []
    for batch_index in range(indices.shape[0]):
        for rank in range(k):
            class_index = int(indices[batch_index, rank].item())
            rows.append(
                {
                    "batch_item": int(batch_index),
                    "rank": int(rank + 1),
                    "class_index": class_index,
                    "label": labels[class_index],
                    "prob": float(values[batch_index, rank].item()),
                }
            )
    return rows


def _torchvision_weights_cache_fragment(model: Any, weights: Any) -> dict[str, Any]:
    """Return torchvision weights metadata without importing torchvision."""

    if weights is None:
        return {
            "model_module": getattr(type(model), "__module__", None),
            "weights_id": None,
            "categories_len": None,
        }
    meta = getattr(weights, "meta", None)
    categories = meta.get("categories") if isinstance(meta, dict) else None
    return {
        "model_module": getattr(type(model), "__module__", None),
        "weights_id": str(weights),
        "categories_len": len(categories) if isinstance(categories, list) else None,
    }
