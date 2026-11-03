"""``Site.record``: the record a selection row names, through a weak trace link.

``Selection`` binds every row to its trace at construction (``bind_entries``);
``site_record`` resolves a row back to the ``Op``, ``Param`` or
``EdgeUseRecord`` it names, refusing typed once the trace is released or when
the live trace has no such record. Kept beside ``selection.py`` so the
selection module stays within its size cap.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from .._utils.fail_open import fail_open
from .errors import SelectionError

if TYPE_CHECKING:
    from .selection import Site


def bind_entries(trace: Any, entries: Iterable[Site]) -> None:
    """Give every row a weak link to ``trace`` so ``Site.record`` can find it."""

    try:
        ref = weakref.ref(trace)
    except TypeError:
        return
    for entry in entries:
        object.__setattr__(entry, "_trace_ref", ref)


def site_trace(site: Site) -> Any:
    """Return the live trace behind one row, or refuse typed when it is gone."""

    ref = site._trace_ref
    trace = ref() if ref is not None else None
    if trace is None:
        raise SelectionError(
            f"Site {site.key!r} cannot reach its trace: the trace was released (or the "
            "row was never bound to one), so its record cannot be read.",
            code="selection_trace_released",
            remedy="keep the trace alive while reading rows, or resolve the selector again on a live trace",
            site_key=site.key,
        )
    return trace


def site_record(
    site: Site,
    forward_ops: Callable[[Any], Iterable[Any]],
    edge_records: Callable[[Any], Iterable[Any]],
) -> Any:
    """Resolve one row to the ``Op``, ``Param`` or ``EdgeUseRecord`` it names.

    ``forward_ops`` and ``edge_records`` are the selection module's walkers,
    passed in so this module stays a leaf below it.
    """

    trace = site_trace(site)
    found: Any = None
    if site.kind == "ACT" and len(site.key) == 2:
        layer_label, pass_index = site.key
        for op in forward_ops(trace):
            if (getattr(op, "layer_label", None) or op.label) == layer_label and (
                getattr(op, "pass_number", 1) or 1
            ) == pass_index:
                found = op
                break
    elif site.kind == "PARAM" and len(site.key) == 1:
        found = fail_open(lambda: trace.params[site.key[0]], lambda _error: None)
    elif site.kind == "EDGE":
        for record in edge_records(trace):
            if edge_address_of(record) == site.key:
                found = record
                break
    if found is None:
        raise SelectionError(
            f"Site {site.key!r} ({site.kind}) names no record on its trace.",
            code="selection_unresolvable",
            remedy="resolve the selector again on this trace; rows carried over from another trace do not transfer",
            reason="site_not_in_trace",
            site_key=site.key,
        )
    return found


def edge_address_of(record: Any) -> tuple[Any, ...]:
    """Return one EdgeUseRecord's canonical occurrence address."""

    return (record.child_func_call_id, record.arg_kind, tuple(record.arg_path))


def _require_edge_provenance(trace: Any) -> None:
    """Refuse typed when edge provenance was not captured."""

    if not bool(getattr(trace, "is_replayable", False)):
        raise SelectionError(
            "edge provenance requires an replayable capture "
            "(EdgeUseRecords exist only under it). Re-capture with "
            "tl.trace(model, x, capture=tl.options.CaptureOptions("
            "replayable=True)).",
            code="edge_provenance_unavailable",
        )
