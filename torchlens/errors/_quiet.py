"""``tl.errors.quiet``: the one switch that silences TorchLens warnings."""

from __future__ import annotations

import warnings
from types import TracebackType

from . import _base

_QUIET_ALL = False
"""Whether a ``quiet()`` with no codes is in force."""


class _QuietScope:
    """What ``quiet`` returns: already in force; ``with`` restores the prior state on exit."""

    def __init__(self, codes: frozenset[object]) -> None:
        """Put the quiet state in force, remembering what to restore."""

        global _QUIET_ALL
        self._prior_all = _QUIET_ALL
        self._prior_codes = _base._QUIET_CODES
        self._filters = warnings.catch_warnings()
        self._filters.__enter__()
        if codes:
            _base._QUIET_CODES = self._prior_codes | codes
            warnings.filterwarnings("ignore", category=_base._QuietedWarning)
        else:
            _QUIET_ALL = True
            warnings.filterwarnings("ignore", category=_base.TorchLensWarning)

    def __enter__(self) -> _QuietScope:
        """Return this scope; the state is already in force."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the quiet state and warning filters in force before the call."""

        global _QUIET_ALL
        _QUIET_ALL = self._prior_all
        _base._QUIET_CODES = self._prior_codes
        self._filters.__exit__(exc_type, exc, traceback)


def quiet(*codes: str) -> _QuietScope:
    """Silence TorchLens warnings: every one, or only those with the given codes.

    ``quiet()`` silences every TorchLens warning; ``quiet("code", ...)`` silences
    only the coded warnings whose ``fields["code"]`` is one of ``codes``. Called
    bare it lasts for the session; as ``with tl.errors.quiet(...):`` the state and
    the warning filters in force before the call come back at the end of the block.
    It works through the standard warning filters, so an explicit
    ``warnings.simplefilter(...)`` made later takes precedence.

    Parameters
    ----------
    *codes:
        Warning codes to silence. None silences every TorchLens warning.

    Returns
    -------
    _QuietScope
        A context manager; the quiet state is already in force.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     with tl.errors.quiet("trace_zero_fire"):
    ...         warnings.warn(tl.errors.TorchLensWarning("hidden", code="trace_zero_fire"))
    ...         warnings.warn(tl.errors.TorchLensWarning("shown", code="bind_zero_fire"))
    >>> [str(item.message) for item in caught]
    ['shown']
    """

    return _QuietScope(frozenset(codes))


def quiet_all_active() -> bool:
    """Whether a ``quiet()`` with no codes is in force (guards that skip work check it)."""

    return _QUIET_ALL


__all__ = ["quiet"]
