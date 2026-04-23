"""Portable I/O primitives for TorchLens model logs.

This module defines the shared versioning, field-policy, and compatibility
helpers used by the portable scrub/rehydrate path introduced in the I/O
sprint.
"""

from __future__ import annotations

import copy
import warnings
from enum import Enum
from typing import Any, NamedTuple

IO_FORMAT_VERSION = 1


class TorchLensIOError(RuntimeError):
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


def read_io_format_version(state: dict[str, Any], *, cls_name: str) -> int:
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

    version = state.get("io_format_version")
    if version is None:
        warnings.warn(
            f"{cls_name} pickle state predates TorchLens portable I/O versioning; "
            "compat mode is deprecated.",
            DeprecationWarning,
            stacklevel=3,
        )
        return 0
    if not isinstance(version, int):
        raise TorchLensIOError(
            f"{cls_name} pickle state has invalid io_format_version={version!r}."
        )
    if version > IO_FORMAT_VERSION:
        raise TorchLensIOError(
            f"{cls_name} pickle state uses io_format_version={version}, "
            f"but this runtime only supports up to {IO_FORMAT_VERSION}."
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
