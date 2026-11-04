"""Recover the torchvision weights preset a model was built with, from its state.

A torchvision model built with ``weights=`` keeps no handle to the preset,
and the class alone never implies one (``resnet50`` has several). What the
model does keep is the loaded state, and the preset's checkpoint is already
on disk (``torch.hub`` downloaded it to build the model). This module compares
the two: every preset of the matching architecture whose checkpoint is cached
is loaded lazily (memory-mapped, never downloaded) and a sample of its tensors
is compared for exact equality with the model's own. One exact match names the
preset; none, or more than one, names nothing. The match is a statement about
tensor values, not about the class, so it stays inside the resolution law that
forbids inferring weights from the model type.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import torch

__tl_layer__ = "L1"

#: Number of state tensors compared per candidate (first, last and evenly
#: spaced ones in between); exact equality on all of them is the match.
_SAMPLE_TENSORS = 6


def is_torchvision_model(model: Any) -> bool:
    """Return whether ``model`` is an instance of a ``torchvision.models`` class."""

    module = getattr(type(model), "__module__", None)
    return isinstance(module, str) and module.startswith("torchvision.models")


def cached_checkpoint_path(weights: Any) -> str | None:
    """Return the cached checkpoint file of a weights member, or ``None`` when absent.

    Mirrors the ``torch.hub`` download layout (``<hub_dir>/checkpoints/<url
    basename>``) without downloading anything.
    """

    url = getattr(weights, "url", None)
    if not isinstance(url, str) or not url:
        return None
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        return None
    path = os.path.join(torch.hub.get_dir(), "checkpoints", filename)
    return path if os.path.isfile(path) else None


def _parameter_count(model: Any) -> int | None:
    """Return the model's parameter count, or ``None`` when it cannot be read."""

    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        return None


def _candidate_weights(model: Any) -> list[Any]:
    """Return the weights members whose declared parameter count fits ``model``.

    Members that declare no ``num_params`` stay candidates; the tensor
    comparison decides for them.
    """

    try:
        from torchvision import models as torchvision_models
    except ImportError:
        return []
    count = _parameter_count(model)
    candidates: list[Any] = []
    seen: set[int] = set()
    for name in torchvision_models.list_models():
        try:
            enum = torchvision_models.get_model_weights(name)
        except Exception:
            continue
        for member in enum:
            if id(member) in seen:
                continue
            seen.add(id(member))
            meta = getattr(member, "meta", None)
            declared = meta.get("num_params") if isinstance(meta, dict) else None
            if count is not None and isinstance(declared, int) and declared != count:
                continue
            candidates.append(member)
    return candidates


def _sample_keys(keys: list[str]) -> list[str]:
    """Return up to ``_SAMPLE_TENSORS`` keys spread over the state order."""

    if len(keys) <= _SAMPLE_TENSORS:
        return list(keys)
    step = (len(keys) - 1) / (_SAMPLE_TENSORS - 1)
    return [keys[round(index * step)] for index in range(_SAMPLE_TENSORS)]


def _load_checkpoint(path: str) -> Any:
    """Load a cached checkpoint for reading only (memory-mapped when supported)."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # torch without the mmap keyword
        return torch.load(path, map_location="cpu", weights_only=True)


def _state_matches(model_state: dict[str, torch.Tensor], checkpoint: Any) -> bool:
    """Return whether the sampled model tensors equal the checkpoint's exactly."""

    if not isinstance(checkpoint, dict):
        return False
    for key in _sample_keys(list(model_state)):
        expected = checkpoint.get(key)
        actual = model_state[key]
        if not isinstance(expected, torch.Tensor) or tuple(expected.shape) != tuple(actual.shape):
            return False
        try:
            if not torch.equal(actual.detach().cpu(), expected.to(actual.dtype)):
                return False
        except Exception:
            return False
    return True


def match_torchvision_weights(model: Any) -> Any | None:
    """Return the one torchvision weights member whose cached checkpoint equals ``model``'s state.

    Parameters
    ----------
    model:
        A model built from a ``torchvision.models`` class.

    Returns
    -------
    Any | None
        The matching ``WeightsEnum`` member, or ``None`` when the model is not
        a torchvision model, no candidate checkpoint is cached, no candidate's
        tensors equal the model's, or more than one does.
    """

    if not is_torchvision_model(model):
        return None
    try:
        model_state = {
            key: value
            for key, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
    except Exception:
        return None
    if not model_state:
        return None
    matches: list[Any] = []
    for member in _candidate_weights(model):
        path = cached_checkpoint_path(member)
        if path is None:
            continue
        try:
            checkpoint = _load_checkpoint(path)
        except Exception:
            continue
        if _state_matches(model_state, checkpoint):
            matches.append(member)
    return matches[0] if len(matches) == 1 else None


__all__ = ["cached_checkpoint_path", "is_torchvision_model", "match_torchvision_weights"]
