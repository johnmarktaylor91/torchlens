"""Portable I/O primitives for TorchLens model logs.

The ``torchlens._io`` package implements TorchLens' portable save/load path:
it scrubs a ``Trace`` into metadata plus tensor blobs, writes directory
bundles backed by ``safetensors``, and rehydrates those bundles into eager or
lazy model logs. Portable bundles are for archival and analysis, not replay:
``validate_forward_pass()`` is unsupported after ``torchlens.load()`` (Fork L),
expert ``lazy=True, materialize_nested=False`` loads must call
``torchlens.rehydrate_nested()`` before re-save (Fork M), and lazy refs open,
verify, and close blob files per materialization instead of sharing handles
(Fork K).
"""

from __future__ import annotations

import copy
import warnings
from enum import Enum
from typing import Any, NamedTuple

import torch

from ..errors._base import CompatibilityError

# v5 adds backend-neutral object-state fields for the backend substrate
# (`module_identity_mode`, `param_source`, and neutral dtype/device/address refs).
TLSPEC_VERSION = 5
_LEGACY_THREAD_WARNING_EMITTED: dict[str, bool] = {"flag": False}


def _warn_legacy_thread_fields_dropped() -> None:
    """Emit one deprecation warning for legacy thread-replay fields.

    Older TorchLens portable bundles with ``tlspec_version <= 2`` carried
    private fields removed by the module-containment-refactor sprint. Current
    load code drops those fields and uses the stored ``modules`` field
    directly.
    """

    if not _LEGACY_THREAD_WARNING_EMITTED["flag"]:
        warnings.warn(
            "Loaded a TorchLens bundle from tlspec_version<=2; "
            "legacy thread-replay fields were dropped. "
            "Module containment is reconstructed from hook-stack "
            "snapshots in current capture; this load uses the stored "
            "modules field directly.",
            DeprecationWarning,
            stacklevel=3,
        )
        _LEGACY_THREAD_WARNING_EMITTED["flag"] = True


def reset_legacy_thread_warning() -> None:
    """Reset the once-per-process legacy-thread warning flag for tests."""

    _LEGACY_THREAD_WARNING_EMITTED["flag"] = False


class TorchLensIOError(CompatibilityError, RuntimeError):
    """Raised when TorchLens portable bundle state is invalid or unsupported."""


class BlobRef(NamedTuple):
    """Reference to a persisted tensor blob in a portable bundle."""

    blob_id: str
    kind: str


class FieldPolicy(str, Enum):
    """Portable scrub policy for one serialized field."""

    KEEP = "keep"
    BLOB = "blob"
    BLOB_RECURSIVE = "blob_recursive"
    DROP = "drop"
    STRINGIFY = "stringify"
    WEAKREF_STRIP = "weakref_strip"


def read_tlspec_version(state: dict[str, Any], *, cls_name: str) -> int:
    """Validate the serialized I/O format version for one object state.

    Parameters
    ----------
    state:
        Serialized state dict for the object being restored.
    cls_name:
        Human-readable class name used in warnings and errors.

    Returns
    -------
    int
        The decoded version. Pre-sprint states return ``0``.

    Raises
    ------
    TorchLensIOError
        If the serialized version is newer than this runtime understands or
        is not an integer.
    """

    version = state.pop("tlspec_version", None)
    if version is None:
        warnings.warn(
            f"{cls_name} pickle state predates TorchLens portable I/O versioning; "
            "compat mode is deprecated.",
            DeprecationWarning,
            stacklevel=3,
        )
        return 0
    if not isinstance(version, int):
        raise TorchLensIOError(f"{cls_name} pickle state has invalid tlspec_version={version!r}.")
    if version > TLSPEC_VERSION:
        raise TorchLensIOError(
            f"{cls_name} pickle state uses tlspec_version={version}, "
            f"but this runtime only supports up to {TLSPEC_VERSION}."
        )
    return version


def default_fill_state(state: dict[str, Any], *, defaults: dict[str, Any]) -> None:
    """Populate missing state keys with deep-copied default values.

    Parameters
    ----------
    state:
        Mutable serialized state dict being restored.
    defaults:
        Mapping from field name to the default value that should be injected
        when that field is absent.
    """

    for field_name, default_value in defaults.items():
        if field_name not in state:
            state[field_name] = copy.deepcopy(default_value)


def rehydrate_nested(trace: Any, *, map_location: str | torch.device = "cpu") -> None:
    """Replace any remaining nested portable blob refs on a loaded model log.

    This function is a no-op unless the model log was loaded with
    ``lazy=True, materialize_nested=False``. In the default load mode, nested
    tensors are already materialized.

    Parameters
    ----------
    trace:
        Model log loaded from a portable bundle.
    map_location:
        Target device for the materialized nested tensors.

    Examples
    --------
    >>> import torchlens as tl
    >>> log = tl.load("demo_bundle", lazy=True, materialize_nested=False)
    >>> tl.rehydrate_nested(log)
    >>> log.save("demo_bundle_copy")
    """

    from .rehydrate import rehydrate_nested as _rehydrate_nested

    _rehydrate_nested(trace, map_location=map_location)
