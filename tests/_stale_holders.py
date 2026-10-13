"""Shared holder for stale pre-wrap torch references in rescue tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn


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


class CountedRoot(nn.Module):
    """Root wrapper counting how many times a capture ran the forward.

    The count is a forward-time write: a default (pure-observer) capture puts
    ``calls`` back to ``[0]``, and the count is read from the trace's
    attribute-write record with :func:`forward_runs`.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = [0]

    def forward(self, *args: Any) -> Any:
        """Count, then delegate."""

        self.calls[0] += 1
        return self.inner(*args)


def forward_runs(trace: Any, model: nn.Module) -> int:
    """Forward runs a capture made, read from the trace's attribute-write record.

    Also checks the observer contract: the root's own counter is back at 0.
    """

    assert model.calls == [0]
    write = trace.modules["self"].attribute_writes["calls"]
    assert write.kind == "replaced"
    return int(write.value[0])


def provenance_warnings(caught: list[Any]) -> list[str]:
    """Return the stale-reference provenance and capture-gap warning texts."""

    markers = ("no graph/source provenance", "re-ran the forward pass", "adopted at module exit")
    return [str(w.message) for w in caught if any(m in str(w.message) for m in markers)]
