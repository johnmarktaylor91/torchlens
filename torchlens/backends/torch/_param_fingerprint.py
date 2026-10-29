"""Capture-time parameter fingerprints for the replay changed-parameter check.

A replay template holds the live ``nn.Parameter`` a call consumed, so when a template
classifies a parameter argument it records what that parameter looked like then: its
version counter, storage pointer and shape. Replay compares it before using the value
(:mod:`torchlens.intervention._param_fingerprint`). This module is the capture-side half,
kept in the backend layer that records templates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...utils._torch_compat import tensor_version_or_none
from .completeness_witness import (
    _ORIG_TENSORBASE_UNTYPED_STORAGE,
    _ORIG_UNTYPED_STORAGE_DATA_PTR,
    internal_scalar_read,
)


@dataclass(frozen=True)
class ParamFingerprint:
    """What a parameter looked like when a template recorded it.

    A dataclass, not a tuple, so the template rewriters (labeling's parent-ref rename,
    cleanup's removed-ref sweep) rebuild it with ``dataclasses.replace`` and keep its type.
    """

    address: str | None
    version: int | None
    storage_ptr: int | None
    shape: tuple[int, ...]


def _storage_ptr(param: torch.Tensor) -> int | None:
    """Return the storage pointer through the unwrapped C accessors, or ``None``."""

    try:
        return int(_ORIG_UNTYPED_STORAGE_DATA_PTR(_ORIG_TENSORBASE_UNTYPED_STORAGE(param)))
    except (RuntimeError, TypeError, NotImplementedError, AttributeError):
        return None


def param_fingerprint(param: torch.Tensor, address: str | None = None) -> ParamFingerprint:
    """Return the current fingerprint of a parameter.

    Parameters
    ----------
    param:
        Parameter to fingerprint.
    address:
        Parameter address for refusal messages, when known.

    Returns
    -------
    ParamFingerprint
        Version counter, storage pointer and shape (no tensor op runs).
    """

    # TorchLens bookkeeping read: unmarked, a capture-time ``_version`` read on registered
    # state would record a forward read-kind and refuse the runnable save.
    with internal_scalar_read():
        version = tensor_version_or_none(param)
    return ParamFingerprint(
        address=address,
        version=version,
        storage_ptr=_storage_ptr(param),
        shape=tuple(param.shape),
    )
