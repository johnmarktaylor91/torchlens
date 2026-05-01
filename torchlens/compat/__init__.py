"""Compatibility adapter namespace reserved for TorchLens 2.0."""

from __future__ import annotations

from typing import Any

from . import lovely, torchshow
from .torchextractor import Extractor


def from_huggingface(
    model_id: str,
    *,
    local_files_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a Hugging Face Transformers model with offline-first defaults.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier or local model path.
    local_files_only:
        Whether to restrict loading to the local cache.
    **kwargs:
        Additional keyword arguments forwarded to ``AutoModel.from_pretrained``.

    Returns
    -------
    Any
        Loaded Transformers model.

    Raises
    ------
    ImportError
        If Transformers is unavailable.
    """

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "Hugging Face loading requires the `hf` extra: install torchlens[hf]."
        ) from exc

    return AutoModel.from_pretrained(model_id, local_files_only=local_files_only, **kwargs)


def from_timm(model_name: str, *, pretrained: bool = False, **kwargs: Any) -> Any:
    """Load a timm model through TorchLens' HF compatibility extra.

    Parameters
    ----------
    model_name:
        timm model name.
    pretrained:
        Whether timm should load pretrained weights.
    **kwargs:
        Additional keyword arguments forwarded to ``timm.create_model``.

    Returns
    -------
    Any
        Loaded timm model.

    Raises
    ------
    ImportError
        If timm is unavailable.
    """

    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm loading requires the `hf` extra: install torchlens[hf].") from exc

    return timm.create_model(model_name, pretrained=pretrained, **kwargs)


__all__ = ["Extractor", "from_huggingface", "from_timm", "lovely", "torchshow"]
