"""Typed preprocessing refusals shared by the Hugging Face auto-routes."""

from __future__ import annotations

import sys
from typing import Any, NoReturn

from torchlens._errors import InvalidArgumentError


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
