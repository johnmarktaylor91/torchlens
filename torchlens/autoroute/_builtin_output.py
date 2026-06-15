"""Built-in output auto-route detectors and decoders."""

from __future__ import annotations

from dataclasses import replace
from importlib import resources
from typing import Any

import torch

from torchlens.autoroute.output import register
from torchlens.data_classes.trace import ResolvedPostprocessing, Trace
from torchlens.ir.container_registry import _is_hf_model_output

DETECTOR_VERSION = "torchlens-output-a2-v1"
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
        resources.files("torchlens.autoroute.data")
        .joinpath("imagenet1k_labels.json")
        .read_text(encoding="utf-8")
    )
    import json

    labels = json.loads(label_text)
    return [str(label) for label in labels]


def decode_outputs_for_trace(
    trace: Trace,
    outputs: Any,
    *,
    output_style: str | None,
    output_head: str | None,
) -> None:
    """Decode live model outputs into ``trace.decoded_output`` when safely resolved.

    Parameters
    ----------
    trace:
        Trace receiving decoded JSON-primitive rows and provenance.
    outputs:
        Live model forward return value, before TorchLens output extraction.
    output_style:
        Optional user-requested decode style.
    output_head:
        Optional user-requested output head name/path.
    """

    meta = _build_output_meta(trace, output_style=output_style, output_head=output_head)
    resolved = _resolve_postprocessor(outputs, meta)
    if resolved is None:
        return

    logits_info = _select_logits(outputs, resolved, meta)
    if logits_info is None:
        if output_style is not None or resolved.verified:
            trace.output_postprocessor = replace(
                resolved,
                ambiguous=True,
                description="decode: undetected; pass output_style=/output_head= to decode.",
            )
        return

    logits, selected_head = logits_info
    labels = _labels_for_resolved(resolved)
    if labels is None and resolved.style != "hf_text":
        return
    if resolved.style == "hf_text":
        decoded = _decode_hf_text(logits, meta)
    else:
        decoded = _decode_classification(logits, labels or [], top_n=_TOP_N_CAPTURED)
    if decoded is None:
        return

    trace.decoded_output = decoded
    trace.output_postprocessor = replace(
        resolved,
        selected_output_head=selected_head,
        top_n_captured=_TOP_N_CAPTURED,
    )


def semantic_output_cache_key(
    model: Any,
    *,
    output_style: str | None,
    output_head: str | None,
) -> dict[str, Any]:
    """Return a serializable semantic-output cache fingerprint.

    Parameters
    ----------
    model:
        Model passed to ``trace``.
    output_style:
        Requested output decoding style.
    output_head:
        Requested output head.

    Returns
    -------
    dict[str, Any]
        Stable-ish metadata that affects output auto-detection and labels.
    """

    config = getattr(model, "config", None)
    default_cfg = getattr(model, "default_cfg", None)
    weights = getattr(model, "_torchlens_weights", None)
    return {
        "version": DETECTOR_VERSION,
        "output_style": output_style,
        "output_head": output_head,
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
    trace: Trace, *, output_style: str | None, output_head: str | None
) -> dict[str, Any]:
    """Build detector metadata from captured trace fields.

    Parameters
    ----------
    trace:
        Trace with in-band output metadata.
    output_style:
        Optional user-requested style.
    output_head:
        Optional user-requested output head.

    Returns
    -------
    dict[str, Any]
        Detector metadata.
    """

    return {
        "output_style": output_style,
        "output_head": output_head,
        "id2label": getattr(trace, "output_id2label", None),
        "num_classes": getattr(trace, "output_num_classes", None),
        "tokenizer": getattr(trace, "_output_tokenizer", None),
        "model_metadata": getattr(trace, "_semantic_output_metadata", None),
    }


def _resolve_postprocessor(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Resolve the first matching registered output postprocessor.

    Parameters
    ----------
    outputs:
        Live model outputs.
    meta:
        Detector metadata.

    Returns
    -------
    ResolvedPostprocessing | None
        Matching postprocessor, if any.
    """

    from torchlens import autoroute

    for detector in autoroute.output.iter_by_priority():
        resolved = detector(outputs, meta)
        if resolved is not None:
            return resolved
    return None


@register(name="explicit_style", priority=0)
def explicit_style(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Resolve user-requested built-in output styles."""

    style = meta.get("output_style")
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
            selected_output_head=meta.get("output_head"),
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
            selected_output_head=meta.get("output_head"),
            label_source="tokenizer",
            label_source_version=DETECTOR_VERSION,
            confidence=1.0,
        )
    return None


@register(name="hf_config_classifier", priority=10)
def hf_config_classifier(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Detect classifiers with explicit ``config.id2label`` metadata."""

    id2label = meta.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        return None
    return ResolvedPostprocessing(
        source="hf_config",
        identifier="config.id2label",
        verified=True,
        config={"num_labels": meta.get("num_classes"), "id2label": id2label},
        description="HF config labels",
        style="classification",
        selected_output_head=meta.get("output_head"),
        label_source="config.id2label",
        label_source_version=DETECTOR_VERSION,
        confidence=0.99,
    )


@register(name="imagenet_verified", priority=20)
def imagenet_verified(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Detect verified ImageNet-width logits from live outputs only."""

    if meta.get("output_style") not in {None, "imagenet", "imagenet1k"}:
        return None
    logits = _select_tensor_by_head(outputs, meta.get("output_head"))
    if logits is None and isinstance(outputs, torch.Tensor):
        logits = outputs
    if logits is None or logits.ndim < 1 or logits.shape[-1] != 1000:
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
        selected_output_head=meta.get("output_head"),
        label_source=IMAGENET_LABEL_SOURCE,
        label_source_version=IMAGENET_LABEL_SOURCE_VERSION,
        confidence=0.95,
    )


@register(name="hf_text", priority=30)
def hf_text(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Detect Hugging Face text logits only when explicitly requested."""

    if meta.get("output_style") != "hf_text":
        return None
    return explicit_style(outputs, meta)


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

    requested = resolved.selected_output_head or meta.get("output_head")
    by_head = _select_tensor_by_head(outputs, requested)
    if by_head is not None:
        return by_head, str(requested or "output")
    if hasattr(outputs, "logits"):
        logits = getattr(outputs, "logits")
        if isinstance(logits, torch.Tensor):
            return logits, "logits"
    if _is_hf_model_output(outputs):
        try:
            logits = outputs["logits"]
        except Exception:
            logits = None
        if isinstance(logits, torch.Tensor):
            return logits, "logits"
    if isinstance(outputs, torch.Tensor):
        return outputs, "output"
    expected = resolved.config.get("num_labels") or meta.get("num_classes")
    candidates = _tensor_candidates(outputs, expected_width=_safe_int(expected))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _select_tensor_by_head(outputs: Any, output_head: Any) -> torch.Tensor | None:
    """Select a tensor by explicit live-output head name.

    Parameters
    ----------
    outputs:
        Live model outputs.
    output_head:
        Requested head.

    Returns
    -------
    torch.Tensor | None
        Selected tensor, if found.
    """

    if output_head is None:
        return None
    if isinstance(output_head, str):
        if hasattr(outputs, output_head):
            value = getattr(outputs, output_head)
            if isinstance(value, torch.Tensor):
                return value
        if isinstance(outputs, dict):
            value = outputs.get(output_head)
            if isinstance(value, torch.Tensor):
                return value
        if output_head.isdigit() and isinstance(outputs, (tuple, list)):
            index = int(output_head)
            if 0 <= index < len(outputs) and isinstance(outputs[index], torch.Tensor):
                return outputs[index]
    return None


def _tensor_candidates(
    outputs: Any, *, expected_width: int | None
) -> list[tuple[torch.Tensor, str]]:
    """Collect plausible logits candidates from a live output object.

    Parameters
    ----------
    outputs:
        Live model outputs.
    expected_width:
        Optional expected last dimension.

    Returns
    -------
    list[tuple[torch.Tensor, str]]
        Candidate tensors and head labels.
    """

    candidates: list[tuple[torch.Tensor, str]] = []
    if isinstance(outputs, dict):
        iterable = [(str(key), value) for key, value in outputs.items()]
    elif isinstance(outputs, (tuple, list)):
        iterable = [(str(index), value) for index, value in enumerate(outputs)]
    else:
        iterable = []
    for name, value in iterable:
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            if expected_width is None or value.shape[-1] == expected_width:
                candidates.append((value, name))
    return candidates


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
    elif logits.ndim >= 2:
        logits_copy = logits.detach().clone().reshape(-1, logits.shape[-1])
    else:
        return None
    if logits_copy.shape[-1] > len(labels):
        return None
    probs = torch.softmax(logits_copy.float(), dim=-1)
    k = min(top_n, int(probs.shape[-1]))
    values, indices = torch.topk(probs, k=k, dim=-1)
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


def _decode_hf_text(logits: torch.Tensor, meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Decode Hugging Face text logits with an attached tokenizer when present.

    Parameters
    ----------
    logits:
        Live logits tensor.
    meta:
        Detector metadata.

    Returns
    -------
    list[dict[str, Any]] | None
        JSON-primitive rows.
    """

    tokenizer = meta.get("tokenizer")
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return None
    logits_copy = logits.detach().clone()
    token_ids = torch.argmax(logits_copy, dim=-1).reshape(-1).tolist()
    text = tokenizer.decode(token_ids)
    return [
        {"batch_item": 0, "rank": 1, "text": str(text), "token_ids": [int(i) for i in token_ids]}
    ]


def _normalized_id2label(value: Any) -> dict[int, str] | None:
    """Normalize an id-to-label mapping.

    Parameters
    ----------
    value:
        Candidate mapping.

    Returns
    -------
    dict[int, str] | None
        Normalized mapping.
    """

    if not isinstance(value, dict):
        return None
    normalized: dict[int, str] = {}
    for key, label in value.items():
        try:
            normalized[int(key)] = str(label)
        except (TypeError, ValueError):
            continue
    return normalized or None


def _normalized_mapping(value: Any) -> dict[str, str] | None:
    """Normalize a small string mapping for cache keys."""

    if not isinstance(value, dict):
        return None
    return {str(key): str(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}


def _normalized_sequence(value: Any) -> tuple[str, ...] | None:
    """Normalize a string sequence for cache keys."""

    if not isinstance(value, (list, tuple)):
        return None
    return tuple(str(item) for item in value)


def _safe_int(value: Any) -> int | None:
    """Return ``value`` as int when possible."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    """Return ``value`` as str when present."""

    if value is None:
        return None
    return str(value)


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
