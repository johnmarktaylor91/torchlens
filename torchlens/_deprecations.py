"""Shared helpers for additive public-API deprecations."""

from __future__ import annotations

import warnings
from typing import Final


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
        stacklevel=6,
    )
