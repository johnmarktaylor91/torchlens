"""Fork lineage: the ``fork`` verb's record-side helpers and the parent-chain test.

``fork_trace`` is the ``Trace.fork`` body (build the copy-on-write fork, record
the operation); ``next_fork_name`` and ``rebind_fork_owner_refs`` are the
record-side helpers the fork builder and the rerun-refresh paths call;
``_is_fork_relative`` answers whether two traces share a fork parent chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._trace_intervention_records import record_operation
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


def fork_trace(trace: Trace, name: str | None = None) -> Trace:
    """Create a copy-on-write intervention fork of this log. (the ``Trace.fork`` body)."""

    fork = trace._fork_trace(name=name)
    record_operation(trace, "fork", source_id=id(trace), name=fork.name)
    return fork


def next_fork_name(trace: Trace) -> str:
    """Return a deterministic default fork name for this parent log."""

    base_name = trace.name or "trace"
    fork_count = sum(
        1
        for record in trace.state_history
        if isinstance(record, dict) and record.get("op") == "fork"
    )
    return f"{base_name}_fork_{fork_count + 1}"


def rebind_fork_owner_refs(trace: Trace) -> None:
    """Rebind weak owner references on child objects to this trace.

    Used by the fork builder and by the rerun-refresh paths, which
    replace record state wholesale and must re-point every child's
    owner reference (and drop stale facet caches) afterwards.
    """

    for layer_pass in trace._layer_list:
        layer_pass.source_trace = trace
        del layer_pass.facets
    for layer_log in trace._layer_logs.values():
        layer_log.source_trace = trace
    for module_log in trace.modules:
        module_log._source_trace = trace
        module_log.__dict__.pop("_facets_cache", None)
        for module_call in module_log.calls.values():
            module_call._source_trace = trace
