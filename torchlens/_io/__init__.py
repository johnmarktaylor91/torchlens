"""Portable I/O primitives for TorchLens model logs.

The ``torchlens._io`` package implements TorchLens' portable save/load path:
it scrubs a ``Trace`` into metadata plus tensor blobs, writes directory
bundles backed by ``safetensors``, and rehydrates those bundles into eager or
lazy model logs. Portable bundles are for archival and analysis, not replay:
``validate_forward_pass()`` is unsupported after ``torchlens.load()``,
expert ``lazy=True, materialize_nested=False`` loads must call
``torchlens.rehydrate_nested()`` before re-save, and lazy refs open,
verify, and close blob files per materialization instead of sharing handles.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import torch

from ..errors._base import CompatibilityError

# v6 adds persisted ModuleCall forward-pre-hook provenance value objects.
TLSPEC_VERSION = 6
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


@dataclass(frozen=True)
class JaxPayloadLoadHint:
    """JAX-specific payload materialization hints.

    Parameters
    ----------
    sharding:
        Explicit JAX ``Sharding`` target. When supplied, materialized JAX
        payloads are placed with this sharding and reconstruction metadata is
        not consulted.
    reconstruct_sharding:
        Whether to reconstruct a saved ``NamedSharding`` from portable
        ``codec_metadata``.
    mesh_axis_names:
        Optional expected mesh axis names for metadata reconstruction. When
        provided, the saved contract must match exactly.
    platform:
        Optional JAX device platform to use for metadata reconstruction, for
        example ``"cpu"``.
    devices:
        Optional explicit local devices to use for metadata reconstruction.
    """

    sharding: Any | None = None
    reconstruct_sharding: bool = False
    mesh_axis_names: tuple[str, ...] | None = None
    platform: str | None = None
    devices: Sequence[Any] | None = None


@dataclass(frozen=True)
class PayloadLoadHints:
    """Backend-specific payload materialization hints.

    Parameters
    ----------
    jax:
        Optional JAX payload hint.
    """

    jax: JaxPayloadLoadHint | None = None


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


_COERCIBLE_CONTAINER_TYPES = (list, dict, tuple, set)


def coerce_container_typed_state(
    state: dict[str, Any],
    defaults: dict[str, Any],
    *,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Coerce present-but-wrong-typed container fields to their declared type.

    ``default_fill_state`` only fills keys that are *absent* from ``state``; by
    design it never inspects the type of a key that is already present, so a
    field restored from an older serialization that used a different
    container type for the same name (e.g. a field that used to default to
    ``{}`` and now defaults to ``[]``) survives unchanged with the wrong type.
    That is silent today and crashes or misbehaves later, whenever code calls
    a list-only method (``.append``) on the restored dict, or vice versa.

    This closes that gap for fields whose declared type is unambiguous: a
    field is only touched here if its entry in ``defaults`` is exactly a
    ``list``, ``dict``, ``tuple``, or ``set`` literal. Fields defaulting to
    ``None``, a scalar, or a non-builtin container (e.g. ``OrderedDict``,
    a custom accessor class) are left untouched, since for those fields the
    "declared type" is not a single unambiguous builtin and a naive coercion
    could silently corrupt a legitimately-varying value (see ``exclude`` for
    the one known case of this: a field typed ``List[int] | str``).

    ``None`` is treated as the "no data was ever recorded" case (e.g. an old
    ``Optional``-typed field that is now a required container): converting it
    to an empty typed container is lossless, since there was never any
    element data to preserve, and mirrors what an absent key would have
    received from ``default_fill_state``.

    A present value that is genuinely incompatible with the declared type
    (``declared_type(current_value)`` raises) is a DIFFERENT case: real
    corruption, not a legacy container-type migration. Per the
    validation-is-a-tripwire principle, this function must never silently
    swallow that into an empty default -- doing so previously turned a loud
    crash into silent, undetectable data loss (e.g. a corrupted
    ``layer_dict_main_keys`` quietly coming back as an empty dict instead of
    surfacing the corruption). Such a field is left in its original,
    detectably-wrong state and this function raises ``TorchLensIOError`` so
    the caller learns about the corruption immediately, instead of shipping a
    Trace that looks valid but silently lost data.

    Parameters
    ----------
    state:
        Mutable serialized state dict being restored. Expected to have
        already been passed through ``default_fill_state`` with the same
        ``defaults`` mapping (so every field in ``defaults`` is present).
    defaults:
        The same defaults mapping passed to ``default_fill_state``.
    exclude:
        Field names to skip even though their default value is a plain
        container, for fields whose real declared type is a union with a
        non-container alternative that must not be coerced away (e.g. a
        legacy sentinel string like ``"all"``).

    Raises
    ------
    TorchLensIOError
        If a field is present with a value that cannot be converted to its
        declared container type (e.g. an ``int`` where a ``dict`` is
        declared). This is corruption, not a legacy-format field -- the
        function refuses to silently discard it.
    """

    for field_name, default_value in defaults.items():
        if field_name in exclude or field_name not in state:
            continue
        declared_type = type(default_value)
        if declared_type not in _COERCIBLE_CONTAINER_TYPES:
            continue
        current_value = state[field_name]
        if isinstance(current_value, declared_type):
            continue
        if current_value is None:
            # Absent-equivalent: field existed in the old schema but was never
            # populated. Lossless to convert to an empty typed container.
            state[field_name] = declared_type()
            continue
        try:
            state[field_name] = declared_type(current_value)
        except (TypeError, ValueError) as exc:
            raise TorchLensIOError(
                f"State field {field_name!r} is present with value of type "
                f"{type(current_value).__name__!r} that cannot be converted to the "
                f"declared container type {declared_type.__name__!r}. This is not a "
                "recognized legacy container-type migration (e.g. a set stored where "
                "a list is now declared) -- it looks like corrupted or tampered "
                "pickle state. Refusing to silently discard the field's contents; "
                "fix the source of the corruption rather than suppressing this error."
            ) from exc


def rehydrate_nested(
    trace: Any,
    *,
    map_location: str | torch.device = "cpu",
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
) -> None:
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
    payload_hints:
        Optional backend payload hints used during materialization.

    Examples
    --------
    >>> import torchlens as tl
    >>> log = tl.load("demo_bundle", lazy=True, materialize_nested=False)
    >>> tl.rehydrate_nested(log)
    >>> log.save("demo_bundle_copy")
    """

    from .rehydrate import rehydrate_nested as _rehydrate_nested

    _rehydrate_nested(trace, map_location=map_location, payload_hints=payload_hints)
