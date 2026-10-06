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


def count_root_forwards(model: Any) -> list[int]:
    """Count the root module's forward calls with a pre-hook (one entry per call)."""

    calls: list[int] = []
    model.register_forward_pre_hook(lambda _module, _args: calls.append(1))
    return calls


def provenance_warnings(caught: list[Any]) -> list[str]:
    """Return the stale-reference provenance and capture-gap warning texts."""

    markers = ("no graph/source provenance", "re-ran the forward pass", "adopted at module exit")
    return [str(w.message) for w in caught if any(m in str(w.message) for m in markers)]
