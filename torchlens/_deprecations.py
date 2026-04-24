"""Shared helpers for additive public-API deprecations."""

from __future__ import annotations

import warnings
from typing import Final, TypeVar, cast

T = TypeVar("T")


class MissingType:
    """Sentinel type used to detect explicitly supplied public kwargs.

    Notes
    -----
    Public APIs in this sprint must distinguish ``caller omitted this kwarg``
    from ``caller explicitly passed the public default``. A dedicated sentinel
    keeps those cases separate without relying on value comparisons.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a stable debugging representation."""

        return "MISSING"


MISSING: Final[MissingType] = MissingType()
_WARNED_DEPRECATIONS: set[str] = set()


def warn_deprecated_alias(old: str, new: str) -> None:
    """Emit a once-per-process deprecation warning for an old public name.

    Parameters
    ----------
    old:
        Deprecated public name.
    new:
        Canonical replacement name.
    """

    key = f"{old}->{new}"
    if key in _WARNED_DEPRECATIONS:
        return
    _WARNED_DEPRECATIONS.add(key)
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead. "
        "The old name continues to work but will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )


def resolve_renamed_kwarg(
    *,
    old_name: str,
    new_name: str,
    old_value: T | MissingType,
    new_value: T | MissingType,
    default: T,
) -> T:
    """Resolve a deprecated kwarg alias to its canonical replacement.

    Parameters
    ----------
    old_name:
        Deprecated kwarg name.
    new_name:
        Canonical replacement kwarg name.
    old_value:
        Value supplied via the deprecated kwarg, or ``MISSING``.
    new_value:
        Value supplied via the canonical kwarg, or ``MISSING``.
    default:
        Public default used when neither spelling is supplied.

    Returns
    -------
    T
        Resolved kwarg value.

    Raises
    ------
    TypeError
        If both the deprecated and canonical spellings were supplied.
    """

    if old_value is not MISSING and new_value is not MISSING:
        raise TypeError(f"kwarg {old_name} deprecated, use {new_name}; do not pass both")
    if old_value is not MISSING:
        warn_deprecated_alias(old_name, new_name)
        return cast(T, old_value)
    if new_value is not MISSING:
        return cast(T, new_value)
    return default
