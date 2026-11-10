"""Mute log records at WARNING and below for one tensor-search crawl, without cache clears.

``logging.disable`` clears the level cache of every logger, and a library with
many loggers (transformers registers about 380) makes each clear cost hundreds
of calls. The crawl's mute window needs the effect of ``logging.disable(WARNING)``
(every record at WARNING or below is dropped before it is built, on any logger
and any thread) and, when it ends, the user's logging state exactly as found.

This module gets that effect without touching any logger: for the window,
``logging.Logger.isEnabledFor`` answers ``False`` for levels at or below WARNING
and defers to the method it replaced for higher levels, and
``logging.root.manager.disable`` reads WARNING, as under ``logging.disable``.
No level-cache entry is written for a muted level and every entry written for
a higher level is the one the unmuted logger would write, so the caches stay
valid and nothing needs clearing. :func:`unmute` puts the method and the
manager's level back; if something inside the window called
``logging.disable`` itself, it restores through ``logging.disable`` so that
call's cache entries are cleared, as base's restore does.
"""

from __future__ import annotations

import logging
from typing import Any

#: The level ``logging.disable`` is called with in the base mute: records at or below it drop.
MUTED_LEVEL = logging.WARNING

#: ``[method]``: the ``isEnabledFor`` the mute replaced, called for levels above the mute.
_replaced: list[Any] = [logging.Logger.isEnabledFor]


def _muted_is_enabled_for(self: logging.Logger, level: int) -> bool:
    """``Logger.isEnabledFor`` while muted: ``False`` at or below WARNING, else the replaced method."""

    if level <= MUTED_LEVEL:
        return False
    return _replaced[0](self, level)  # type: ignore[no-any-return]


def mute() -> None:
    """Open the mute window (the caller checked ``manager.disable`` is below WARNING)."""

    logger_class = logging.Logger
    current = logger_class.__dict__.get("isEnabledFor")
    if current is not _muted_is_enabled_for:  # another thread's window may hold it already
        _replaced[0] = current
        logger_class.isEnabledFor = _muted_is_enabled_for  # type: ignore[method-assign]
    logging.root.manager.disable = MUTED_LEVEL


def unmute(previous: int) -> None:
    """Close the mute window, restoring ``manager.disable`` to ``previous`` and the method."""

    logger_class = logging.Logger
    if logger_class.__dict__.get("isEnabledFor") is _muted_is_enabled_for:
        logger_class.isEnabledFor = _replaced[0]  # type: ignore[method-assign]
    manager = logging.root.manager
    if manager.disable == MUTED_LEVEL:
        manager.disable = previous
    else:  # ``logging.disable`` ran inside the window: clear its cache entries too
        logging.disable(previous)
