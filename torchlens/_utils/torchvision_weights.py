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
import warnings
from typing import Any
from urllib.parse import urlparse

import torch

__tl_layer__ = "L1"

#: Number of state tensors compared per candidate (first, last and evenly
#: spaced ones in between); exact equality on all of them is the match.
_SAMPLE_TENSORS = 6

#: Total bytes of sampled checkpoint tensors kept between matches. A sample
#: that does not fit is read from the checkpoint file on every match instead.
_SAMPLE_CACHE_BYTES = 32 * 1024 * 1024

#: Stand-in kept for a checkpoint that is not a state mapping (never a match).
_NOT_A_STATE: Any = object()

#: ``(path, mtime_ns, ctime_ns, size, inode, sampled keys)`` -> ``(sampled tensors, bytes)``.
#: Only checkpoint tensors copied out of the file are held, never a model. The
#: file identity in the key makes a replaced or rewritten checkpoint a miss.
_sample_cache: dict[tuple[Any, ...], tuple[Any, int]] = {}


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
    """Load a cached checkpoint for reading only (memory-mapped when supported).

    Older torch's own loader reads ``dtype`` off its legacy typed storage
    classes for every stored tensor and so emits torch's TypedStorage
    deprecation warning from inside ``torch.load``; no caller code touched a
    typed storage, so that one torch-internal warning is not shown.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="TypedStorage is deprecated", category=UserWarning
        )
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # torch without the mmap keyword
            return torch.load(path, map_location="cpu", weights_only=True)


def _sample_nbytes(sample: dict[str, Any]) -> int:
    """Return the bytes of the tensors in ``sample`` (dense element storage)."""

    return sum(
        int(value.numel()) * int(value.element_size())
        for value in sample.values()
        if isinstance(value, torch.Tensor)
    )


def _remember_sample(cache_key: tuple[Any, ...], sample: Any, nbytes: int) -> None:
    """Keep ``sample`` under ``cache_key``, dropping the oldest entries to fit the budget."""

    held = sum(entry[1] for entry in _sample_cache.values())
    for stale_key in list(_sample_cache):
        if held + nbytes <= _SAMPLE_CACHE_BYTES:
            break
        held -= _sample_cache.pop(stale_key)[1]
    _sample_cache[cache_key] = (sample, nbytes)


def _checkpoint_sample(path: str, keys: list[str]) -> Any:
    """Return what ``_state_matches`` reads from the checkpoint at ``path``.

    The answer is the same mapping lookups the loaded checkpoint would give
    for ``keys``. The first match of a checkpoint file loads it and copies the
    sampled tensors out of the memory map; later matches of the same file
    (same path, modification and change times, size and inode) and the same keys reuse the
    copies, so a model traced many times does not reload its checkpoint on
    every trace. A sample over :data:`_SAMPLE_CACHE_BYTES`, a file that cannot
    be stat'ed and a sample that cannot be copied are never kept: the loaded
    checkpoint is returned as before.

    Raises
    ------
    Exception
        Whatever the checkpoint load raises; nothing is kept for that file.
    """

    try:
        stat = os.stat(path)
    except OSError:
        return _load_checkpoint(path)
    file_identity = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
    cache_key = (path, *file_identity, tuple(keys))
    held = _sample_cache.get(cache_key)
    if held is not None:
        return held[0]
    checkpoint = _load_checkpoint(path)
    if not isinstance(checkpoint, dict):
        _remember_sample(cache_key, _NOT_A_STATE, 0)
        return checkpoint
    try:
        sample = {key: checkpoint.get(key) for key in keys}
        nbytes = _sample_nbytes(sample)
        if nbytes > _SAMPLE_CACHE_BYTES:
            return checkpoint
        copies = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in sample.items()
        }
    except Exception:
        return checkpoint
    _remember_sample(cache_key, copies, nbytes)
    return copies


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
    sampled_keys = _sample_keys(list(model_state))
    for member in _candidate_weights(model):
        path = cached_checkpoint_path(member)
        if path is None:
            continue
        try:
            checkpoint = _checkpoint_sample(path, sampled_keys)
        except Exception:
            continue
        if _state_matches(model_state, checkpoint):
            matches.append(member)
    return matches[0] if len(matches) == 1 else None


__all__ = ["cached_checkpoint_path", "is_torchvision_model", "match_torchvision_weights"]
