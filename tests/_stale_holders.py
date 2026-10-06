"""Shared holder for stale pre-wrap torch references in rescue tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class OpaqueCallable:
    """A custom callable object: a holder the per-capture rebind never scans.

    Capture preparation rebinds pristine torch functions held in closures,
    attributes, partials and builtin containers, so a bare stale reference no
    longer escapes. Routing the stale call through a custom object keeps a
    genuine escape that only the rescue re-run can recover.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)
