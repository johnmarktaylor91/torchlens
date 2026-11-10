"""Step-0 ingest of the exhaustive capture rows (capture C2c).

The exhaustive commit writes each op's record-sourced cells into a capture row
(C2b, ``backends/torch/_ops_row_journal.py``). Ingest used to rebuild those
cells from the op's record (``scatter_record_to_cells`` into a fresh fields
dict) and construct the ``Op`` from that dict. Here the row itself becomes the
op's store row:

1. :func:`hand_off_capture_rows` takes each unamended row handle's row and
   record from the sealed journal (the record is built once, from the row as
   committed) instead of cloning the entry, and builds any other handle that
   still reads a taken row.
2. :func:`adopt_capture_row` writes ingest's join cells, the neutral ``None``
   seeds and ``Op`` construction's defaults and conversions
   (``_ir/_op_row_seed.py``, shared with ``Op.__init__``) into the row,
   adopts it into the trace's op store and returns one ``Op`` shell over it.

Amended handles and every other journal entry keep the record path. This
module imports no module of the package's largest import cycle.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .._constants import LAYER_PASS_LOG_FIELD_ORDER
from .._ir._op_row_seed import layout_keys, seed_op_values
from .._ir._row_journal import _ROW_TAKE_PATHS, _RowOpRecord
from .._ir.op_record_scatter import _write_overlay_cells
from .._ir.refs import journal_record
from .._trace_core.op_store import _MISSING

__all__: tuple[str, ...] = ()

#: The four streaming-only blob ids: kept when the row or the join set them, else ``None``.
_PENDING_FIELDS = (
    "_pending_blob_id",
    "_pending_transformed_out_blob_id",
    "_pending_grad_blob_id",
    "_pending_transformed_grad_blob_id",
)
_NO_OVERRIDE = object()


def hand_off_capture_rows(
    session: Any, source: Any, clone: Callable[[Any], Any]
) -> tuple[list[Any], dict[int, list[Any]]]:
    """Return the replay projection of the sealed op lane and the capture rows ingest adopts.

    Parameters
    ----------
    session
        The capture session (its seal is the amendment-folded op lane), or
        ``None`` for a session-detached journal, whose fold is projected as
        built records and adopts no rows.
    source
        The capture's ``CaptureEvents``; its raw lane and live index may hold
        other handles over the same rows.
    clone
        The projection clone for entries that keep the record path.

    Returns
    -------
    tuple[list[Any], dict[int, list[Any]]]
        The projected entries (a taken handle's built record, else
        ``clone(entry)``) and ``id(record) -> capture row`` for each taken row.
        Each record is fresh (its mutable containers are copies of the row's),
        so the projection shares no mutable state with the row ingest writes.
    """

    if session is None:
        return list(map(journal_record, source.amended_op_records())), {}
    projected: list[Any] = []
    capture_rows: dict[int, list[Any]] = {}
    overlaid: list[tuple[list[Any], _RowOpRecord, Any]] = []
    for event in session.seal().events:
        taken = event.take_row() if type(event) is _RowOpRecord else None
        if taken is None:
            projected.append(clone(event))
            continue
        row, record = taken
        capture_rows[id(record)] = row
        projected.append(record)
        if event._patch:
            overlaid.append((row, event, record))
    if capture_rows:
        _settle_aliases(source, capture_rows, overlaid)
    return projected, capture_rows


def _settle_aliases(
    source: Any,
    capture_rows: dict[int, list[Any]],
    overlaid: list[tuple[list[Any], _RowOpRecord, Any]],
) -> None:
    """Build every other handle over a taken row, then fold each overlay into its row.

    Ingest writes into the taken rows, so a raw-lane or live-index entry over
    one is built first. An alias of an amended row is built from the taken
    handle's record (no row build); any other alias builds from its row.
    """

    taken_ids = {id(row) for row in capture_rows.values()}
    amended = {id(row): taken for row, taken, _ in overlaid}
    for event in (*source.op_events, *source.live_index.by_raw_label.values()):
        if type(event) is not _RowOpRecord or event._row is None:
            continue
        row_id = id(event._row)
        taken = amended.get(row_id)
        if taken is not None and _ROW_TAKE_PATHS.issuperset(event._patch):
            event.settle_alias(taken)
        elif row_id in taken_ids:
            event.materialize()
    for row, taken, record in overlaid:
        _write_overlay_cells(row, taken._journal.fids, record)


class RowIngestPlan:
    """The field ids row ingest writes, resolved once per ingest from the store layout."""

    __slots__ = (
        "construction_done",
        "fid_by_name",
        "grad_records",
        "is_in_conditional_body",
        "keys",
        "none_fids",
        "order_fids",
        "pending_fids",
        "source_trace_ref",
        "unset_fids",
    )

    def __init__(self, layout: Any) -> None:
        """Resolve the field ids of ``layout`` (an ``OpStoreLayout``)."""

        fid_by_name: Mapping[str, int] = layout.fid_by_name
        self.fid_by_name = fid_by_name
        self.keys = layout_keys(fid_by_name)
        self.pending_fids = tuple(fid_by_name[name] for name in _PENDING_FIELDS)
        self.order_fids = tuple(
            fid_by_name[name]
            for name in LAYER_PASS_LOG_FIELD_ORDER
            if name in fid_by_name and name not in _PENDING_FIELDS
        )
        self.none_fids = (fid_by_name["out_ref"], fid_by_name["grad_ref"])
        self.grad_records = fid_by_name["_grad_records"]
        self.construction_done = fid_by_name["_construction_done"]
        self.source_trace_ref = fid_by_name["_source_trace_ref"]
        self.is_in_conditional_body = fid_by_name["_is_in_conditional_body"]
        self.unset_fids: tuple[int, ...] | None = None

    def seed_unset(self, row: list[Any]) -> None:
        """Seed ``None`` into the op fields the capture row leaves unset.

        The commit writes the same cell names for every exhaustive op (only
        the pending blob ids vary, and those are handled apart), so the unset
        field ids are read off the first row and reused for the rest.
        """

        unset = self.unset_fids
        if unset is None:
            unset = tuple(fid for fid in self.order_fids if row[fid] is _MISSING)
            self.unset_fids = unset
        for fid in unset:
            row[fid] = None


def adopt_capture_row(
    row: list[Any],
    join_cells: Mapping[str, object],
    store: Any,
    op_class: type,
    plan: RowIngestPlan,
) -> Any:
    """Finish one capture row as ``Op`` construction would and return one ``Op`` over it.

    Parameters
    ----------
    row
        The op's capture row (layout-ordered; the commit's cells).
    join_cells
        Ingest's join cells for the op (``_apply_join_cells`` output), plus the
        ``_materialized_backend_address`` override when the join sets it.
    store
        The trace's building ``OpRowStore``; it adopts ``row`` without a copy.
    op_class
        The ``Op`` class (allocated without ``__init__``).
    plan
        The field ids for ``store``'s layout.

    Returns
    -------
    Op
        The op whose store row is ``row``.
    """

    plan.seed_unset(row)
    override = _write_join_cells(row, join_cells, plan)
    for fid in plan.pending_fids:
        if row[fid] is _MISSING:
            row[fid] = None
    for fid in plan.none_fids:
        row[fid] = None
    seed_op_values(row, plan.keys)
    if override is not _NO_OVERRIDE:
        row[plan.keys.backend_address] = override
    if row[plan.grad_records] is _MISSING:
        row[plan.grad_records] = []
    row[plan.construction_done] = True
    op: Any = object.__new__(op_class)
    object.__setattr__(op, "_core", store)
    object.__setattr__(op, "_row", store.adopt_row(row))
    return op


def _write_join_cells(row: list[Any], join_cells: Mapping[str, object], plan: RowIngestPlan) -> Any:
    """Write the join cells into ``row`` as ``Op.__init__`` stores them; return the override.

    Names that are not store fields are ``Op`` properties and are dropped, as
    construction drops them; ``source_trace`` and ``is_in_conditional_body``
    go to their backing cells.
    """

    fid_by_name = plan.fid_by_name
    override = _NO_OVERRIDE
    for name, value in join_cells.items():
        fid = fid_by_name.get(name)
        if fid is not None:
            row[fid] = value
        elif name == "_materialized_backend_address":
            override = value
        elif name == "source_trace":
            row[plan.source_trace_ref] = weakref.ref(value) if value is not None else None
        elif name == "is_in_conditional_body":
            row[plan.is_in_conditional_body] = value
    return override


def input_layer_labels(op_events: Iterable[Any], initial: tuple[str, ...]) -> list[str]:
    """Return the raw labels of input ops not already in ``initial``, first-seen order."""

    labels: list[str] = []
    for event in op_events:
        if (
            event.layer_type == "input"
            and event.label_raw not in initial
            and event.label_raw not in labels
        ):
            labels.append(event.label_raw)
    return labels


def _op_events_in_raw_order(op_events: list[Any]) -> list[Any]:
    """Return operation events sorted by their reserved raw index.

    Parameters
    ----------
    op_events
        Operation events in backend append order.

    Returns
    -------
    list[Any]
        Events in graph raw-index order.
    """

    return sorted(op_events, key=lambda event: event.raw_index)


def _drop_missing_buffer_sources(local_registrations: dict[str, Any]) -> None:
    """Clear buffer source labels that are absent from raw materialized logs.

    X2 resolution: consumes ingest's LOCAL registration map, never a
    workspace read-back of state ingest itself created mid-loop.

    Parameters
    ----------
    local_registrations
        Ingest's local (label_raw -> op_log) registration map.

    Returns
    -------
    None
        Mutates buffer logs in place.
    """

    raw_labels = set(local_registrations)
    for op_log in local_registrations.values():
        buffer_source = getattr(op_log, "_graph_buffer_source", None)
        if buffer_source is not None and buffer_source not in raw_labels:
            op_log._graph_buffer_source = None


def _output_versions_by_parent(
    output_version_events: tuple[Any, ...],
) -> dict[str, dict[str, object]]:
    """Return output-version payloads grouped by parent raw label.

    Parameters
    ----------
    output_version_events
        The journal's output-version lane.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping from parent raw label to child raw label to pre-child payload.
    """

    grouped: dict[str, dict[str, object]] = defaultdict(dict)
    for event in output_version_events:
        grouped[event.parent_raw_label][event.child_raw_label] = event.payload
    return grouped
