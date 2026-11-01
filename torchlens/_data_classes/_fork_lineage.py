"""Fork lineage: whether two traces are related through the fork parent chain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._trace_missing_attrs import is_live_trace

if TYPE_CHECKING:
    from .trace import Trace


def _is_fork_relative(trace: Trace, other: Trace) -> bool:
    """Return whether two traces are related through the fork parent chain.

    A fork keeps a ``parent_run`` weakref to its source; two traces are
    relatives when either appears on the other's (bounded) parent chain.
    Structural corollary relied on by callers: fork relatives share one op
    label space, because a fork is a structural copy of its parent.

    Parameters
    ----------
    trace:
        First trace.
    other:
        Second trace.

    Returns
    -------
    bool
        ``True`` when one trace is a fork ancestor of the other.
    """

    for start, target in ((trace, other), (other, trace)):
        current: Any = start
        for _ in range(64):
            parent_ref = current._parent_run if is_live_trace(current) else None
            parent = parent_ref() if callable(parent_ref) else None
            if parent is None:
                break
            if parent is target:
                return True
            current = parent
    return False
