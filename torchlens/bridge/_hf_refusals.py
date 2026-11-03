"""Typed preprocessing refusals shared by the Hugging Face auto-routes."""

from __future__ import annotations

import sys
from typing import Any, NoReturn

from torchlens._errors import InvalidArgumentError


def refuse_uncached_multimodal_processor(
    model: Any, name_or_path: str | None, payload: Any
) -> None:
    """Refuse a modality-key dict whose model processor is not cached locally.

    Input detection reads the local Hugging Face cache only and never
    downloads. Without this refusal an uncached processor would let the raw
    modality dict (strings and PIL images) reach the model, which fails deep
    in its forward with no hint of the cause. Only a ``transformers``
    ``PreTrainedModel`` with a name or path is refused; any other model keeps
    the plain path, since its own forward may accept such a dict.

    Parameters
    ----------
    model:
        Candidate model passed to ``torchlens.trace``.
    name_or_path:
        The model's Hugging Face name or path, or None when it has none.
    payload:
        The modality-key dict the multimodal detector matched.

    Raises
    ------
    InvalidArgumentError
        ``preprocessing_unresolved`` for a Hugging Face model whose
        ``AutoProcessor`` is not in the local cache.
    """

    if name_or_path is None or not _is_hf_pretrained_model(model):
        return
    keys = sorted(str(key) for key in payload) if isinstance(payload, dict) else []
    refuse_preprocessing_unresolved(
        model,
        f"the modality-key input {keys} to {type(model).__name__} needs the "
        f"AutoProcessor for {name_or_path!r}, which is not in the local Hugging Face "
        "cache. Input detection never downloads, and the raw dict is not a valid "
        "model input.",
        remedy=(
            "load the processor once so it is cached, e.g. "
            f"transformers.AutoProcessor.from_pretrained({name_or_path!r}); or call "
            "torchlens.bridge.transformers.trace_multimodal(model, inputs), which "
            "fetches it; or run the processor yourself and pass its output as "
            "keyword inputs, e.g. tl.trace(model, input_kwargs=dict(processor(text=..., "
            "images=..., return_tensors='pt')))"
        ),
        resolution_attempt={
            "source": "hf_auto_processor",
            "identifier": name_or_path,
            "local_files_only": True,
            "reason": "processor not in the local cache",
        },
    )


def _is_hf_pretrained_model(model: Any) -> bool:
    """Return whether ``model`` is a ``transformers.PreTrainedModel``.

    Reads ``transformers`` from ``sys.modules`` only: a model built by
    ``transformers`` has already imported it, so no import happens here.
    """

    transformers = sys.modules.get("transformers")
    base = getattr(transformers, "PreTrainedModel", None) if transformers is not None else None
    return isinstance(base, type) and isinstance(model, base)


def refuse_preprocessing_unresolved(
    model: Any, detail: str, *, remedy: str, resolution_attempt: dict[str, Any]
) -> NoReturn:
    """Raise the typed ``preprocessing_unresolved`` refusal.

    Parameters
    ----------
    model:
        Model whose input preprocessing did not resolve.
    detail:
        What did not resolve and why, after the ``torchlens.trace:`` prefix.
    remedy:
        The explicit ways forward.
    resolution_attempt:
        Structured record of the attempt.

    Raises
    ------
    InvalidArgumentError
        Always, with code ``preprocessing_unresolved``.
    """

    raise InvalidArgumentError(
        f"torchlens.trace: {detail}",
        code="preprocessing_unresolved",
        remedy=remedy,
        model_type=type(model).__name__,
        resolution_attempt=resolution_attempt,
    )
