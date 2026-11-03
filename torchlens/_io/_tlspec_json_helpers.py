"""Pure value helpers for the unified ``.tlspec`` manifest writer."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import torch


def _hash_named_tensor_meta(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash ``(name, shape, dtype)`` metadata for named tensors.

    Parameters
    ----------
    named_tensors:
        Iterable of named tensor pairs.

    Returns
    -------
    str
        SHA-256 hex digest.
    """

    entries = [(name, tuple(tensor.shape), str(tensor.dtype)) for name, tensor in named_tensors]
    return sha256(repr(entries).encode("utf-8")).hexdigest()


def _json_str_or_none(value: Any) -> str | None:
    """Return a string for JSON output, preserving ``None``.

    Parameters
    ----------
    value:
        Raw value.

    Returns
    -------
    str | None
        JSON-safe string or ``None``.
    """

    if value is None:
        return None
    return str(value)


def _json_int_or_none(value: Any) -> int | None:
    """Return an int for JSON output when possible.

    Parameters
    ----------
    value:
        Raw value.

    Returns
    -------
    int | None
        JSON-safe int or ``None``.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _python_version() -> str:
    """Return the runtime Python version string.

    Returns
    -------
    str
        ``major.minor.micro`` version string.
    """

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _utc_timestamp() -> str:
    """Return a UTC ISO 8601 timestamp.

    Returns
    -------
    str
        UTC timestamp with ``Z`` suffix.
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
