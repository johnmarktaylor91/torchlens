"""``AutoProcessor`` resolution for the multimodal bridge (the one network seam).

The multimodal auto-route claims a Hugging Face model on shape alone; this
module loads the processor for it (cache first, then the Hub unless offline)
and refuses typed when it cannot, so a modality dict never reaches an HF
forward as a positional ``input_ids``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torchlens._errors import InvalidArgumentError
from torchlens._input_route import _apply_hf_processor


def _load_hf_processor(name_or_path: str) -> Any:
    """Load the ``AutoProcessor`` for ``name_or_path``.

    Parameters
    ----------
    name_or_path:
        Hugging Face model id or local path.

    Returns
    -------
    Any
        The processor ``transformers.AutoProcessor.from_pretrained`` returns.
    """

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(name_or_path)


def _resolve_hf_processor(model: Any, name_or_path: str) -> Any:
    """Resolve the ``AutoProcessor`` for a multimodal capture or refuse typed.

    Parameters
    ----------
    model:
        Hugging Face model the modality dict is for.
    name_or_path:
        Its model id or path.

    Returns
    -------
    Any
        The resolved processor.

    Raises
    ------
    InvalidArgumentError
        ``preprocessing_unresolved`` when the processor cannot be loaded (an
        uncached model offline, a missing processor config): TorchLens never
        hands the raw dict to the forward.
    """

    try:
        return _load_hf_processor(name_or_path)
    except Exception as exc:  # the fetch failure is the finding, never a guess
        raise InvalidArgumentError(
            f"torchlens.trace: no AutoProcessor resolved for the modality dict "
            f"input to {type(model).__name__} ({name_or_path!r}): "
            f"{type(exc).__name__}: {exc}",
            code="preprocessing_unresolved",
            remedy=(
                "run once with network access so the processor files join the "
                "model's cache; or attach the processor you built the inputs with, "
                "model.processor = AutoProcessor.from_pretrained(<name_or_path>); or "
                "pass the processor's output yourself, tl.trace(model, (), "
                "input_kwargs=dict(processor(text=..., images=..., return_tensors='pt')))"
            ),
            model_type=type(model).__name__,
            resolution_attempt={
                "name_or_path": name_or_path,
                "error": f"{type(exc).__name__}: {exc}",
            },
        ) from exc


def _make_multimodal_transform(processor: Any) -> Callable[[dict[str, Any]], Any]:
    """Build the ``input_transform`` that applies a resolved multimodal processor.

    Parameters
    ----------
    processor:
        Resolved ``AutoProcessor``-like callable.

    Returns
    -------
    Callable[[dict[str, Any]], Any]
        Maps the user's modality dict to the processor's mapping of tensors.
    """

    def transform(raw_dict: dict[str, Any]) -> Any:
        """Apply the resolved multimodal processor to one modality dict."""

        return _apply_hf_processor(processor, raw_dict)

    return transform
